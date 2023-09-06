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

int RealWoldWorkLoad(const std::string& path_to_file, uint64_t* bit_map) {
  std::vector<int64_t> _id;
  std::vector<std::string> _first_name;
  std::vector<std::string> _last_name;
  std::vector<int64_t> _creation_date;

  int index = 0, count = 0;  
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("id");
  int col_id2 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id3 = file_metadata->schema()->ColumnIndex("lastName");
  int col_id4 = file_metadata->schema()->ColumnIndex("creationDate");
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;

    // Read the label column
    // Get the Column Reader for the ByteArray column
    column_reader = row_group_reader->Column(col_id);
    // parquet::ByteArrayReader* string_reader =
    //     static_cast<parquet::ByteArrayReader*>(column_reader.get());
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    auto first_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id2));
    auto last_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id3));
    auto creation_date_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id4));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    int64_t last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        id_reader->Skip(i - last_i);
        first_name_reader->Skip(i - last_i);
        last_name_reader->Skip(i - last_i);
        int64_t value = 0;
        id_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _id.push_back(std::move(value));
        first_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _first_name.push_back(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _last_name.push_back(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        int64_t date = 0;
        creation_date_reader->ReadBatch(1, nullptr, nullptr, &date, &values_read);
        _creation_date.push_back(std::move(date));
        last_i = i + 1;
        count++;
      }
      index++;
    }
  }
  for (int i = 0; i < count; ++i) {
    std::cout << _id[i] << ", " << _first_name[i] << ", " << _last_name[i] << ", " << _creation_date[i] << std::endl;
  }

  delete[] byte_value;
  return count;
}

int Project(const std::unordered_set<int64_t>& ids, const std::string& path_to_table) {
  std::vector<int64_t> _id;
  std::vector<std::string> _first_name;
  std::vector<std::string> _last_name;

  std::unordered_set<int64_t> indices;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_table, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();
  std::unordered_map<int64_t, int64_t> index_to_id;

  // char* char_buffer = new char[total_length];
  int64_t *values = new int64_t[BATCH_SIZE];
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  memset(values, 0, sizeof(int64_t) * BATCH_SIZE);
  int index = 0, count = 0;

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("id");
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    int64_t rows_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;

    ARROW_UNUSED(rows_read);  // prevent warning in release build

    // Read the label column
    // Get the Column Reader for the ByteArray column
    column_reader = row_group_reader->Column(col_id);
    // parquet::ByteArrayReader* string_reader =
    //     static_cast<parquet::ByteArrayReader*>(column_reader.get());
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    // Read all the rows in the column
    while (id_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          id_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if (ids.find(values[i]) != ids.end()) {
          indices.insert(index);
          index_to_id[index] = values[i];
        }
        index++;
      }
    }
  }

  int col_id2 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id3 = file_metadata->schema()->ColumnIndex("lastName");
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;

    auto first_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id2));
    auto last_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id3));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    int64_t last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if (indices.find(index) != indices.end()) {
        first_name_reader->Skip(i - last_i);
        last_name_reader->Skip(i - last_i);
        first_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _first_name.push_back(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _last_name.push_back(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_i = i + 1;
        count++;
      }
      index++;
    }
  }
  return count;
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

  int count = 0;
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
  count =  RealWoldWorkLoad(vertex_path_to_file, bit_map);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average project time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::cout << "count: " << count << std::endl;
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
  std::string path = path_to_file + "-origin-base";
  std::shared_ptr<arrow::Table> table;
  auto run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run get bit map time: " << run_time << " ms" << std::endl;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  table = ReadBitMapBaseLineNoOffset(path, vertex_id);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run get bit map time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;

  std::unordered_set<int64_t> ids;
  auto chunked_array = table->column(0);
  auto chunk_num = chunked_array->num_chunks();
  for (int i = 0; i < chunk_num; ++i) {
    auto array = static_cast<arrow::Int64Array*>(chunked_array->chunk(i).get());
    for (int j = 0; j < array->length(); ++j) {
      ids.insert(array->GetView(j));
    }
  }
  std::cout << "size: " << ids.size() << std::endl;

  int count = 0;
  run_start = clock();
  Project(ids, vertex_path_to_file);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run project time: " << run_time << " ms" << std::endl;
  run_start = clock();
  Project(ids, vertex_path_to_file);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  Project(ids, vertex_path_to_file);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  count = Project(ids, vertex_path_to_file);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run project time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
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
  // CheckCorrectness(path_to_file, vertex_num, vertex_id);
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
