// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#include <x86intrin.h>
#include <time.h>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/util/type_fwd.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/column_reader.h"
#include "parquet/file_reader.h"
#include "parquet/file_writer.h"
#include "parquet/metadata.h"

#include "arrow/dataset/api.h"
#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

#include <iostream>
#include <cstdlib>

namespace ds = arrow::dataset;
namespace cp = arrow::compute;

constexpr int BATCH_SIZE = 1024;                     // the batch size

void set_bit(uint64_t* bitmap, uint64_t curr) {
    bitmap[curr >> 6] |= (1ULL << (curr & 0x3f));
}

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::shared_ptr<arrow::Table>& l_table,
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key, const std::string& project_name) {
  auto l_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(l_table));
  auto r_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(r_table));

  arrow::dataset::internal::Initialize();
  auto l_options = std::make_shared<arrow::dataset::ScanOptions>();
  // create empty projection: "default" projection where each field is mapped to a
  // field_ref
  l_options->projection = arrow::compute::project({}, {});

  auto r_options = std::make_shared<arrow::dataset::ScanOptions>();
  // create empty projection: "default" projection where each field is mapped to a
  // field_ref
  r_options->projection = arrow::compute::project({}, {});

  auto r_schema = r_dataset->schema();
  std::vector<arrow::FieldRef> r_output_fields({project_name});

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{l_key},
                                              /*in_right_keys=*/{r_key},
                                              {},
                                              r_output_fields,
                                              /*filter*/ arrow::compute::literal(true),
                                              /*output_suffix_for_left*/ "_l",
                                              /*output_suffix_for_right*/ "_r"};
  arrow::acero::Declaration hashjoin{
      "hashjoin", {std::move(left), std::move(right)}, join_opts};

  // expected columns l_a, l_b
  std::shared_ptr<arrow::Table> response_table = arrow::acero::DeclarationToTable(std::move(hashjoin)).ValueOrDie();
  return response_table;
}


void getOffset(const std::string& path_to_offset_file, const int64_t& vertex_id, int64_t& offset, int64_t& length) {
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_offset_file);
  auto file_metadata = reader_->metadata();
  int row_group_index = 0;
  int64_t id_offset = vertex_id;
  // std::cout << "num_row_groups: " << file_metadata->num_row_groups() << std::endl;
  while (row_group_index < file_metadata->num_row_groups()) {
    auto row_group_metadata = file_metadata->RowGroup(row_group_index);
    if (id_offset > row_group_metadata->num_rows()) {
      id_offset -= row_group_metadata->num_rows();
      row_group_index++;
      continue;
    } else {
      break;
    }
  }
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(row_group_index++)->Column(0));
  col_reader->Skip(id_offset);
  int64_t value_to_read = 2;
  int64_t values_read = 0;
  std::vector<int64_t> values(value_to_read);
  while (col_reader->HasNext() && value_to_read > 0) {
    col_reader->ReadBatch(value_to_read, nullptr, nullptr, values.data() + (2 - value_to_read), &values_read);
    value_to_read -= values_read;
  }
  while (value_to_read > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(row_group_index++)->Column(0));
    while (col_reader->HasNext() && value_to_read > 0) {
      col_reader->ReadBatch(value_to_read, nullptr, nullptr, values.data() + (2 - value_to_read), &values_read);
      value_to_read -= values_read;
    }
  }
  offset = values[0];
  length = values[1] - values[0];
}

void ReadBitMap(const std::string& path_to_file, const int64_t& offset, const int64_t& length, uint64_t* bit_map) {
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  auto file_metadata = reader_->metadata();
  int i = 0;
  int64_t remain_offset = offset;
  while (i < file_metadata->num_row_groups()) {
    auto row_group_metadata = file_metadata->RowGroup(i);
    if (remain_offset > row_group_metadata->num_rows()) {
      remain_offset -= row_group_metadata->num_rows();
      i++;
      continue;
    } else {
      break;
    }
  }
  int64_t total_values_remaining = length;
  int64_t values_read = 0;
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
  col_reader->Skip(remain_offset);
  while (col_reader->HasNext() && total_values_remaining > 0) {
    col_reader->ReadBatch(total_values_remaining, bit_map, &values_read);
    total_values_remaining -= values_read;
  }
  while (total_values_remaining > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
    while (col_reader->HasNext() && total_values_remaining > 0) {
      col_reader->ReadBatch(total_values_remaining, bit_map, &values_read);
      total_values_remaining -= values_read;
    }
  }
 return;
}

void ReadBitMapBaseLine(const std::string& path_to_file, const int64_t& offset, const int64_t& length, uint64_t* bit_map) {
  // #include "arrow/io/api.h"
  // #include "arrow/parquet/arrow/reader.h"
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  //int64_t remain_offset = 1316469;
  //int64_t delta_length = 22846;
  // int64_t remain_offset = 1888612;
  // int64_t delta_length = 278490;
  int64_t remain_offset = offset;
  auto file_metadata = reader_->metadata();
  int i = 0;
  // std::cout << "num_row_groups: " << file_metadata->num_row_groups() << std::endl;
  while (i < file_metadata->num_row_groups()) {
    auto row_group_metadata = file_metadata->RowGroup(i);
    if (remain_offset > row_group_metadata->num_rows()) {
      remain_offset -= row_group_metadata->num_rows();
      i++;
      continue;
    } else {
      break;
    }
  }
  // uint64_t* bit_map = new uint64_t[BITMAP_SIZE];
  int64_t total_values_remaining = length;
  std::vector<int64_t> values(length);
  int64_t values_read = 0;
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
  col_reader->Skip(remain_offset);
  while (col_reader->HasNext() && total_values_remaining > 0) {
    col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values.data() + (length - total_values_remaining), &values_read);
    total_values_remaining -= values_read;
  }
  while (total_values_remaining > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
    while (col_reader->HasNext() && total_values_remaining > 0) {
      col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values.data() + (length - total_values_remaining), &values_read);
      total_values_remaining -= values_read;
    }
  }
  for (int64_t i = 0; i < length; ++i) {
    set_bit(bit_map, values[i]);
    /*
    auto index = static_cast<uint64_t>(values[i]);
    if (!(bit_map[index >> 6] & (1UL << (index & 63)))) {
      throw std::runtime_error("Bit map is not correct");
    }
    */
  }

  return;
}

std::shared_ptr<arrow::Table> ReadBitMapBaseLineNoOffset(const std::string& path_to_file, const int64_t& vertex_id) {
  std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
  auto fs = arrow::fs::FileSystemFromUriOrPath(path_to_file).ValueOrDie();
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                        fs, {path_to_file}, format,
                        arrow::dataset::FileSystemFactoryOptions()).ValueOrDie();
  auto dataset = factory->Finish().ValueOrDie();
  auto options = std::make_shared<arrow::dataset::ScanOptions>();

  cp::Expression filter_expr = cp::equal(cp::field_ref("src"), cp::literal(vertex_id));
  options->filter = filter_expr;
  auto scan_builder = dataset->NewScan().ValueOrDie();
  scan_builder->Project({"dst"});
  scan_builder->Filter(std::move(filter_expr));
  scan_builder->UseThreads(false);
  auto scanner = scan_builder->Finish().ValueOrDie();
  return scanner->ToTable().ValueOrDie();
}

int RunIntersection(uint64_t* bit_map, uint64_t* bit_map_2, int64_t length) {
  int count = 0;
  for (int64_t i = 0; i < length; ++i) {
    count += _mm_popcnt_u64(bit_map[i] & bit_map_2[i]);
  }
  return count;
}

int Intersection(std::shared_ptr<arrow::Table>& table1, std::shared_ptr<arrow::Table>& table2) {
  std::unordered_set<int64_t> ids;
  auto chunked_array = table1->column(0);
  auto chunk_num = chunked_array->num_chunks();
  for (int i = 0; i < chunk_num; ++i) {
    auto array = static_cast<arrow::Int64Array*>(chunked_array->chunk(i).get());
    for (int j = 0; j < array->length(); ++j) {
      ids.insert(array->GetView(j));
    }
  }
  int count = 0;
  chunked_array = table2->column(0);
  chunk_num = chunked_array->num_chunks();
  for (int i = 0; i < chunk_num; ++i) {
    auto array = static_cast<arrow::Int64Array*>(chunked_array->chunk(i).get());
    for (int j = 0; j < array->length(); ++j) {
      if (ids.find(array->GetView(j)) != ids.end()) {
        count++;
      }
    }
  }
  return count;
}

void RunExamples(const std::string& path_to_file, int64_t vertex_num, int64_t vertex_id, int64_t vertex_id2) {
  std::string path = path_to_file + "-delta";
  int64_t bit_map_length = vertex_num / 64 + 1;
  uint64_t* bit_map = new uint64_t[bit_map_length];
  uint64_t* bit_map2 = new uint64_t[bit_map_length];
  memset(bit_map, 0, sizeof(uint64_t) * bit_map_length);
  memset(bit_map2, 0, sizeof(uint64_t) * bit_map_length);
  int64_t offset = 0, length = 0, offset2 = 0, length2 = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  getOffset(path_to_file + "-offset", vertex_id2, offset2, length2);
  std::cout << "offset: " << offset << ", length: " << length << std::endl;
  std::cout << "offset2: " << offset2 << ", length2: " << length2 << std::endl;
  auto run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  ReadBitMap(path, offset2, length2, bit_map2);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  ReadBitMap(path, offset2, length2, bit_map2);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  ReadBitMap(path, offset2, length2, bit_map2);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  ReadBitMap(path, offset2, length2, bit_map2);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;

  int inter_count = 0;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, bit_map_length);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First intersection time: " << run_time << " ms" << std::endl;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, bit_map_length);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, bit_map_length);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, bit_map_length);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average intersection time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::cout << "inter_count: " << inter_count << std::endl;
  delete[] bit_map; 
  delete[] bit_map2;
  // return;
}

void RunExamplesBaseLine(const std::string& path_to_file, int64_t vertex_num, int64_t vertex_id) {
  std::string path = path_to_file + "-base";
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  uint64_t* bit_map = new uint64_t[vertex_num / 64 + 1];
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num / 64 + 1));
  int64_t offset = 0, length = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  std::cout << "offset: " << offset << ", length: " << length << std::endl;
  auto run_start = clock();
  ReadBitMapBaseLine(path, offset, length, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMapBaseLine(path, offset, length, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLine(path, offset, length, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLine(path, offset, length, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  delete[] bit_map;
  return;
}

void RunExamplesBaseLineNoOffset(const std::string& path_to_file, int64_t vertex_num, int64_t vertex_id, int64_t vertex_id2 ) {
  std::string path = path_to_file + "-origin-base";
  std::shared_ptr<arrow::Table> table, table2;
  auto run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  table2 = ReadBitMapBaseLineNoOffset(path, vertex_id2);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run get bit map time: " << run_time << " ms" << std::endl;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  table2 = ReadBitMapBaseLineNoOffset(path, vertex_id2);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  table2 = ReadBitMapBaseLineNoOffset(path, vertex_id2);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  table2 = ReadBitMapBaseLineNoOffset(path, vertex_id2);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run get bit map time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  int count = 0;
  run_start = clock();
  count = Intersection(table, table2);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run project time: " << run_time << " ms" << std::endl;
  run_start = clock();
  count = Intersection(table, table2);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  count = Intersection(table, table2);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  count = Intersection(table, table2);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run project time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::cout << "count: " << count << std::endl;
  return;
} 

int main(int argc, char** argv) {
  if (argc < 2) {
    // Fake success for CI purposes.
    return EXIT_SUCCESS;
  }

  std::string path_to_file = argv[1];
  int64_t vertex_num = std::stol(argv[2]);
  int64_t vertex_id = std::stol(argv[3]);
  int64_t vertex_id2 = std::stol(argv[4]);
  std::cout << "path_to_file: " << path_to_file << " vertex_num: " << vertex_num << " vertex_id: " << vertex_id  << " vertex_id2: " << vertex_id2 << std::endl;
  // CheckCorretness(path_to_file, vertex_num, vertex_id);
  // return 0;
  std::string type = argv[5];  
  if (type == "delta") {
    RunExamples(path_to_file, vertex_num, vertex_id, vertex_id2);
  } else {
    RunExamplesBaseLineNoOffset(path_to_file, vertex_num, vertex_id, vertex_id2);
  }

  // if (!status.ok()) {
  //  std::cerr << "Error occurred: " << status.message() << std::endl;
  //  return EXIT_FAILURE;
  // }
  return EXIT_SUCCESS;
}
