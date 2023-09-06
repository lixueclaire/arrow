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

#include "arrow/dataset/api.h"
#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

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

namespace ds = arrow::dataset;
namespace cp = arrow::compute;

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

void ReadBitMapBaseLineNoOffset(const std::string& path_to_file, const int64_t& vertex_id, uint64_t* bit_map) {
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
  auto table = scanner->ToTable().ValueOrDie();

  auto chunked_array = table->column(0);
  auto chunk_num = chunked_array->num_chunks();
  for (int i = 0; i < chunk_num; ++i) {
    auto array = static_cast<arrow::Int64Array*>(chunked_array->chunk(i).get());
    for (int j = 0; j < array->length(); ++j) {
      set_bit(bit_map, array->Value(j));
      // auto index = static_cast<uint64_t>(array->Value(j));
      // if (!(bit_map[index >> 6] & (1UL << (index & 63)))) {
      //   throw std::runtime_error("Bit map is not correct");
      // }
    }
  }
}

void RealWoldWorkLoad(const std::string& path_to_file, uint64_t* bit_map) {
  std::unique_ptr<parquet::ParquetFileReader> reader_ =
      parquet::ParquetFileReader::OpenFile(path_to_file);
  auto file_metadata = reader_->metadata();
  int64_t index = 0;
  for (int64_t rg_i = 0; rg_i < file_metadata->num_row_groups(); ++rg_i) {
    auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(rg_i)->Column(0));
    auto row_group_metadata = file_metadata->RowGroup(rg_i);
    int64_t last_row_i = 0;
    for (int64_t row_i = 0; row_i < row_group_metadata->num_rows(); row_i++) {
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        std::cout << "index: " << index << std::endl;
        col_reader->Skip(row_i - last_row_i);
        std::cout << "skip done" << std::endl;
        int64_t value = 0;
        int64_t value_read = 0;
        col_reader->ReadBatch(1, nullptr, nullptr, &value, &value_read);
        std::cout << "value: " << value << std::endl;
        last_row_i = row_i + 1;
      }
      index++;
    }
  }
}

void RunExamples(const std::string& path_to_file, int64_t vertex_num, int64_t vertex_id) {
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  std::string path = path_to_file + "-delta";
  uint64_t* bit_map = new uint64_t[vertex_num / 64 + 1];
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num / 64 + 1));
  int64_t offset = 0, length = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  std::cout << "offset: " << offset << ", length: " << length << std::endl;
  auto run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  std::cout << "ReadBitMap done" << std::endl;
  RealWoldWorkLoad("/mnt/ldbc/ldbc-sf10/person_0_0.csv.parquet", bit_map);

  return;
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

void RunExamplesBaseLineNoOffset(const std::string& path_to_file, int64_t vertex_num, int64_t vertex_id) {
  // ARROW_RETURN_NOT_OK(WriteToFile(path_to_file));
  std::string path = path_to_file + "-origin-base";
  uint64_t* bit_map = new uint64_t[vertex_num / 64 + 1];
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num / 64 + 1));
  auto run_start = clock();
  ReadBitMapBaseLineNoOffset(path, vertex_id, bit_map);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path, vertex_id, bit_map);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path, vertex_id, bit_map);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMapBaseLineNoOffset(path, vertex_id, bit_map);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  delete[] bit_map;
  return;
} 

void CheckCorretness(const std::string& path_to_file, int32_t vertex_num, int64_t vertex_id) {
  uint64_t* bit_map = new uint64_t[vertex_num / 64 + 1];
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num / 64 + 1));
  int64_t offset = 0, length = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  std::cout << "offset: " << offset << " length: " << length << std::endl;
  ReadBitMap(path_to_file + "-2-delta", offset, length, bit_map);
  // ReadBitMapBaseLine(path_to_file + "-base", offset, length, bit_map);
  ReadBitMapBaseLineNoOffset(path_to_file + "-origin-base", vertex_id, bit_map);
  delete[] bit_map;
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
  std::cout << "path_to_file: " << path_to_file << " vertex_num: " << vertex_num << " vertex_id: " << vertex_id << std::endl;
  // CheckCorretness(path_to_file, vertex_num, vertex_id);
  // return 0;
  if (argc > 4) {
    std::string type = argv[4];  
    if (type == "delta") {
      RunExamples(path_to_file, vertex_num, vertex_id);
    } else {
      RunExamplesBaseLine(path_to_file, vertex_num, vertex_id);
    }
  } else {
    RunExamplesBaseLineNoOffset(path_to_file, vertex_num, vertex_id);
  }

  // if (!status.ok()) {
  //  std::cerr << "Error occurred: " << status.message() << std::endl;
  //  return EXIT_FAILURE;
  // }
  return EXIT_SUCCESS;
}
