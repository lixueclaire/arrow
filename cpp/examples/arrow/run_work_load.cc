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

#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

#include <iostream>
#include <cstdlib>

// #define BITMAP_SIZE (4847571 / 64) + 1
#define BITMAP_SIZE (58655849 / 64) + 1

int random_num(int last) {
  std::srand(last);
  return rand() % 15 + 1;
}

void set_bit(uint64_t* bitmap, uint64_t curr) {
    bitmap[curr >> 6] |= (1ULL << (curr & 0x3f));
}

arrow::Result<std::shared_ptr<arrow::Table>> GetTable() {
  auto builder = arrow::Int64Builder();
  auto builder2 = arrow::Int64Builder();

  std::shared_ptr<arrow::Array> arr_src, arr_dst;
  int last = 0;
  for (int i = 0; i < 10000000; ++i) {
    // last += random_num(last);
    ARROW_RETURN_NOT_OK(builder.Append(last));
    ARROW_RETURN_NOT_OK(builder2.Append(last));
    last += 1;
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&arr_src));
  ARROW_RETURN_NOT_OK(builder.Finish(&arr_dst));

  // std::shared_ptr<arrow::Array> arr_y;
  // ARROW_RETURN_NOT_OK(builder.AppendValues({2, 4, 6, 8, 10}));
  // ARROW_RETURN_NOT_OK(builder.Finish(&arr_y));

  auto schema = arrow::schema(
      // {arrow::field("x", arrow::int64()), arrow::field("y", arrow::int32())});
      {arrow::field("x", arrow::int64())});

  return arrow::Table::Make(schema, {arr_src});
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

void Project(std::shared_ptr<arrow::Table> left_table, const std::string& vertex_file, const std::string& project_name, std::shared_ptr<arrow::Table>& result_table) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = arrow::io::ReadableFile::Open(vertex_file).ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
  if (!status.ok()) {
    std::cerr << "Parquet read error: " << status.message() << std::endl;
    return;
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> r_table;
  status = arrow_reader->ReadTable(&r_table);
  if (!status.ok()) {
    std::cerr << "Table read error: " << status.message() << std::endl;
    return;
  }
  result_table = DoHashJoin(left_table, r_table, "index", "_graphArSrcIndex", project_name);
  return;
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

arrow::Result<std::shared_ptr<arrow::Table>> ReadBitMapBaseLineNoOffset(const std::string& path_to_file, const int64_t& vertex_id) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = arrow::io::ReadableFile::Open(path_to_file).ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

  // flatten the arrow table
  auto flatten_table = table->CombineChunks(pool).ValueOrDie();
  auto src_array = std::dynamic_pointer_cast<arrow::Int64Array>(flatten_table->column(0)->chunk(0));
  auto dst_array = std::dynamic_pointer_cast<arrow::Int64Array>(flatten_table->column(1)->chunk(0));
  auto builder = arrow::Int64Builder();
  std::shared_ptr<arrow::Array> array;
  for (int64_t i = 0; i < src_array->length(); ++i) {
    if (src_array->Value(i) == vertex_id) {
      ARROW_RETURN_NOT_OK(builder.Append(dst_array->Value(i)));
    }
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  auto schema = arrow::schema(
    // {arrow::field("x", arrow::int64()), arrow::field("y", arrow::int32())});
    {arrow::field("index", arrow::int64())});
  return arrow::Table::Make(schema, {array});
}

void RealWoldWorkLoad(const std::string& path_to_file, uint64_t* bit_map) {
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  auto file_metadata = reader_->metadata();
  std::cout << "num_row: " << file_metadata->num_rows() << std::endl;
  int64_t index = 0;
  int64_t count = 0;
  for (int64_t rg_i = 0; rg_i < file_metadata->num_row_groups(); ++rg_i) {
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(rg_i)->Column(1));
    // auto content_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(rg_i)->Column(5));
    auto row_group_metadata = file_metadata->RowGroup(rg_i);
    int64_t last_row_i = 0;
    for (int64_t row_i = 0; row_i < row_group_metadata->num_rows(); row_i++) {
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        id_reader->Skip(row_i - last_row_i);
        int64_t value = 0;
        int64_t value_read = 0;
        id_reader->ReadBatch(1, nullptr, nullptr, &value, &value_read);
        last_row_i = row_i + 1;
        count++;
      }
      index++;
    }
  }
  std::cout << "count: " << count << " index=" << index << std::endl;
}

void RunExamples(const std::string& path_to_file, const std::string& vertex_path_to_file, int64_t vertex_num, int64_t vertex_id) {
  std::string path = path_to_file + "-delta";
  uint64_t* bit_map = new uint64_t[vertex_num / 64 + 1];
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num / 64 + 1));
  int64_t offset = 0, length = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  std::cout << "offset: " << offset << ", length: " << length << std::endl;
  auto run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;

  run_start = clock();
  RealWoldWorkLoad(vertex_path_to_file, bit_map);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First project time: " << run_time << " ms" << std::endl;
  run_start = clock();
  RealWoldWorkLoad(vertex_path_to_file, bit_map);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  RealWoldWorkLoad(vertex_path_to_file, bit_map);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  RealWoldWorkLoad(vertex_path_to_file, bit_map);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average project time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  delete[] bit_map;
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

void RunExamplesBaseLineNoOffset(const std::string& path_to_file, const std::string& vertex_path_to_file, int64_t vertex_num, int64_t vertex_id) {
  std::string path = path_to_file + "-base";
  std::shared_ptr<arrow::Table> table;
  auto run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id).ValueOrDie();
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run get bit map time: " << run_time << " ms" << std::endl;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id).ValueOrDie();
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id).ValueOrDie();
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id).ValueOrDie();
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run get bit map time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::shared_ptr<arrow::Table> result_table;
  run_start = clock();
  Project(table, vertex_path_to_file, "id", result_table);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run project time: " << run_time << " ms" << std::endl;
  run_start = clock();
  Project(table, vertex_path_to_file, "id", result_table);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  Project(table, vertex_path_to_file, "id", result_table);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  Project(table, vertex_path_to_file, "id", result_table);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run project time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::cout << "result_table->num_rows(): " << result_table->num_rows() << std::endl;
  return;
} 

int main(int argc, char** argv) {
  if (argc < 2) {
    // Fake success for CI purposes.
    return EXIT_SUCCESS;
  }

  std::string path_to_file = argv[1];
  std::string vertex_path_file = argv[2];
  int64_t vertex_num = std::stol(argv[3]);
  int64_t vertex_id = std::stol(argv[4]);
  std::cout << "path_to_file: " << path_to_file << " vertex_num: " << vertex_num << " vertex_id: " << vertex_id << std::endl;
  // CheckCorretness(path_to_file, vertex_num, vertex_id);
  // return 0;
  std::string type = argv[5];  
  if (type == "delta") {
    RunExamples(path_to_file, vertex_path_file, vertex_num, vertex_id);
  } else {
    RunExamplesBaseLineNoOffset(path_to_file, vertex_path_file, vertex_num, vertex_id);
  }

  // if (!status.ok()) {
  //  std::cerr << "Error occurred: " << status.message() << std::endl;
  //  return EXIT_FAILURE;
  // }
  return EXIT_SUCCESS;
}
