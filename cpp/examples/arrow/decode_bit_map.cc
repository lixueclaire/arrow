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

  std::shared_ptr<arrow::Array> arr_x;
  int last = 0;
  for (int i = 0; i < 140; ++i) {
    // last += random_num(last);
    ARROW_RETURN_NOT_OK(builder.Append(last));
    last += 2;
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&arr_x));

  // std::shared_ptr<arrow::Array> arr_y;
  // ARROW_RETURN_NOT_OK(builder.AppendValues({2, 4, 6, 8, 10}));
  // ARROW_RETURN_NOT_OK(builder.Finish(&arr_y));

  auto schema = arrow::schema(
      // {arrow::field("x", arrow::int64()), arrow::field("y", arrow::int32())});
      {arrow::field("x", arrow::int64())});

  return arrow::Table::Make(schema, {arr_x});
}

arrow::Status WriteToFile(std::string path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Table> table, GetTable());

  // Choose compression
  std::shared_ptr<WriterProperties> props =
       WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding(parquet::Encoding::DELTA_BINARY_PACKED_FOR_BIT_MAP)->build();
  // std::shared_ptr<WriterProperties> props =
  //     WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<ArrowWriterProperties> arrow_props =
      ArrowWriterProperties::Builder().build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path_to_file));

  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table.get(),
                                                 arrow::default_memory_pool(), outfile,
                                                 // /*chunk_size=*/64 * 1024 * 1024, props, arrow_props));
                                                 /*chunk_size=*/128, props, arrow_props));
  return arrow::Status::OK();
}

void ReadBitMap(const std::string& path_to_file, uint64_t* bit_map) {

  // #include "arrow/io/api.h"
  // #include "arrow/parquet/arrow/reader.h"
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  // int64_t remain_offset = 1316469;
  // int64_t delta_length = 22846;
  int64_t remain_offset = 1888612;
  int64_t delta_length = 278490;
  auto file_metadata = reader_->metadata();
  int i = 0;
  std::cout << "num_row_groups: " << file_metadata->num_row_groups() << std::endl;
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
  int64_t total_values_remaining = delta_length;
  int64_t values_read = 0;
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
  std::cout << "remain_offset: " << remain_offset << " i: " << i << std::endl;
  col_reader->Skip(remain_offset);
  while (col_reader->HasNext() && total_values_remaining > 0) {
    auto levels = col_reader->ReadBatch(total_values_remaining, bit_map, &values_read);
    std::cout << "values_read: " << values_read << std::endl;
    total_values_remaining -= values_read;
  }
  while (total_values_remaining > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
    while (col_reader->HasNext() && total_values_remaining > 0) {
      auto levels = col_reader->ReadBatch(total_values_remaining, bit_map, &values_read);
      total_values_remaining -= values_read;
    }
  }
  /*
  for (int i = 0; i < 10; i++) {
    std::cout << bit_map[i] << std::endl;
  }
  */
 return;
}

void ReadBitMapBaseLine(const std::string& path_to_file, uint64_t* bit_map) {

  // #include "arrow/io/api.h"
  // #include "arrow/parquet/arrow/reader.h"
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  //int64_t remain_offset = 1316469;
  //int64_t delta_length = 22846;
  int64_t remain_offset = 1888612;
  int64_t delta_length = 278490;
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
  int64_t total_values_remaining = delta_length;
  std::vector<int64_t> values(delta_length);
  int64_t values_read = 0;
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
  col_reader->Skip(remain_offset);
  while (col_reader->HasNext() && total_values_remaining > 0) {
    auto levels = col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values.data() + (delta_length - total_values_remaining), &values_read);
    // std::cout << "values_read: " << values_read << std::endl;
    total_values_remaining -= values_read;
  }
  while (total_values_remaining > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(1));
    while (col_reader->HasNext() && total_values_remaining > 0) {
      auto levels = col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values.data() + (delta_length - total_values_remaining), &values_read);
      // std::cout << "values_read: " << values_read << " total_values_remaining: " << total_values_remaining << std::endl;
      total_values_remaining -= values_read;
    }
  }
  for (int64_t i = 0; i < delta_length; ++i) {
    set_bit(bit_map, values[i]);
  }
  // std::cout << "total_values_remaining: " << total_values_remaining << std::endl;

  // for (int i = 0; i < 10; i++) {
  //   std::cout << bit_map[i] << std::endl;
  // }

  return;
}

void ReadBitMapBaseLineNoOffset(const std::string& path_to_file, uint64_t* bit_map) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = arrow::io::ReadableFile::Open(path_to_file).ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  parquet::arrow::OpenFile(input, pool, &arrow_reader);

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  arrow_reader->ReadTable(&table);

  // flatten the arrow table
  auto flatten_table = table->CombineChunks(pool).ValueOrDie();
  auto src_array = std::dynamic_pointer_cast<arrow::Int64Array>(flatten_table->column(0)->chunk(0));
  auto dst_array = std::dynamic_pointer_cast<arrow::Int64Array>(flatten_table->column(1)->chunk(0));
  for (int64_t i = 0; i < src_array->length(); ++i) {
    // if (src_array->Value(i) == 10010) {
    if (src_array->Value(i) == 1029) {
      set_bit(bit_map, dst_array->Value(i));
    }
  }
}

arrow::Status RunExamples(std::string path_to_file) {
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  uint64_t* bit_map = new uint64_t[BITMAP_SIZE];
  auto run_start = clock();
  ReadBitMap(path_to_file, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMap(path_to_file, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path_to_file, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path_to_file, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  return arrow::Status::OK();
}

arrow::Status RunExamplesBaseLine(std::string path_to_file) {
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  uint64_t* bit_map = new uint64_t[BITMAP_SIZE];
  auto run_start = clock();
  ReadBitMapBaseLine(path_to_file, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMapBaseLine(path_to_file, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLine(path_to_file, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLine(path_to_file, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  return arrow::Status::OK();
}

arrow::Status RunExamplesBaseLineNoOffset(std::string path_to_file) {
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  uint64_t* bit_map = new uint64_t[BITMAP_SIZE];
  auto run_start = clock();
  ReadBitMapBaseLineNoOffset(path_to_file, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path_to_file, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path_to_file, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path_to_file, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  return arrow::Status::OK();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    // Fake success for CI purposes.
    return EXIT_SUCCESS;
  }

  std::string path_to_file = argv[1];
  arrow::Status status = RunExamples(path_to_file);

  if (!status.ok()) {
    std::cerr << "Error occurred: " << status.message() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
