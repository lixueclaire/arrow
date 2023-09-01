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

#define VERTEX_NUM 1128069
constexpr int BATCH_SIZE = 1024;                     // the batch size

/// Set bit in a range
static inline void SetBitmap(uint64_t* bitmap, const int start, const int end) {
  int pos1 = start >> 6, pos2 = end >> 6;
  if (pos1 == pos2) {
    bitmap[pos1] |= (1ULL << (end & 63)) - (1ULL << (start & 63));
  } else {
    bitmap[pos1] |= ~((1ULL << (start & 63)) - 1);
    bitmap[pos2] |= (1ULL << (end & 63)) - 1;
    for (int i = pos1 + 1; i < pos2; ++i) {
      bitmap[i] = ~0ULL;
    }
  }
}

/// The user-defined function to check if the state is valid.
/// A default implementation is provided here, which checks if all labels are contained.
static inline bool IsValid(bool* state, int column_number) {
  for (int i = 0; i < column_number; ++i) {
    // AND case
    if (!state[i]) return false;
    // OR case
    // if (state[i]) return true;
  }
  // AND case
  return true;
  // OR case
  // return false;
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

/// Get the valid intervals of the labels, "column_number" is the number of columns
static inline void GetValidIntervals(const int column_number, int64_t row_number,
                                    int32_t repeated_nums[][VERTEX_NUM],
                                    bool repeated_values[][VERTEX_NUM],
                                    int32_t* length,
                                    const std::function<bool(bool*, int)>& IsValid,
                                    uint64_t* bitmap = nullptr) {
  // initialization
  int current_pos = 0, previous_pos = 0, min_pos;
  int *pos = new int[column_number], *index = new int[column_number];
  bool *state = new bool[column_number];
  memset(pos, 0, sizeof(int) * column_number);
  memset(index, 0, sizeof(int) * column_number);
  memset(state, 0, sizeof(bool) * column_number);
  for (int i = 0; i < column_number; ++i) {
    state[i] = repeated_values[i][index[i]];
  }
  bool state_change = true, last_res = false;

  // K-path merging
  while (true) {
    // find the minimum position of change
    min_pos = INT32_MAX;
    for (int i = 0; i < column_number; ++i) {
      if (index[i] < length[i]) {
        min_pos = std::min(min_pos, pos[i] + repeated_nums[i][index[i]]);
      }
    }
    // check the last interval and add it to the result if it is valid
    previous_pos = current_pos;
    current_pos = min_pos;
    if (state_change) {
      last_res = IsValid(state, column_number);
      state_change = false;
    }
    if (last_res) {
      SetBitmap(bitmap, previous_pos, current_pos);
    }
    // if current position is N, break the loop
    if (min_pos == row_number) {
      break;
    }
    // update the states
    for (int i = 0; i < column_number; ++i) {
      if (index[i] < length[i] && pos[i] + repeated_nums[i][index[i]] == min_pos) {
        pos[i] = min_pos;
        index[i]++;
        if (index[i] < length[i]) {
          state_change |= (state[i] != repeated_values[i][index[i]]);
          state[i] = repeated_values[i][index[i]];
        }
      }
    }
  }
  delete[] pos;
  delete[] index;
  delete[] state;
  return;
}

void GetLabelBitMap(
    const std::string& parquet_filename,
    const std::vector<std::string>& label_names,
    int32_t repeated_nums[][VERTEX_NUM], bool repeated_values[][VERTEX_NUM],
    const std::function<bool(bool*, int)>& IsValid,
    uint64_t* bitmap) {

  std::vector<int> tested_label_ids;
  int tested_label_num = label_names.size();
  // Initialize the global variables for save labels
  int *length = new int32_t[tested_label_num];
  memset(length, 0, tested_label_num * sizeof(int32_t));

  // Create a ParquetReader instance
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(parquet_filename, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();
  int64_t row_num = file_metadata->num_rows();
  auto schema = file_metadata->schema();
  for (auto& label : label_names) {
    tested_label_ids.push_back(schema->ColumnIndex(label));
  }

  // Iterate over all the RowGroups in the file
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    int64_t rows_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;

    ARROW_UNUSED(rows_read);  // prevent warning in release build

    // Read the label columns
    for (int k = 0; k < tested_label_num; k++) {
      // Get the Column Reader for the Bool column
      int col_id = tested_label_ids[k];
      column_reader = row_group_reader->Column(col_id);
      parquet::BoolReader* bool_reader =
          static_cast<parquet::BoolReader*>(column_reader.get());
      // Read all the rows in the column
      while (bool_reader->HasNext()) {
        // Read BATCH_SIZE values at a time. The number of rows read is returned.
        // values_read contains the number of non-null rows

        // !!! This is the default implementation of ReadBatch, which is not used in our
        // case. bool* value = new bool[BATCH_SIZE]; rows_read =
        // bool_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, value, &values_read);

        // !!! This is the optimized implementation of ReadBatch for decoding GraphAr
        // labels.
        rows_read = bool_reader->ReadBatch(BATCH_SIZE, repeated_nums[k] + length[k],
                                           repeated_values[k] + length[k], length[k],
                                           &values_read);
      }
    }

  }

  // std::cout << "The parquet file is read successfully!" << std::endl << std::endl;
  // return the valid count
  GetValidIntervals(tested_label_num, row_num, repeated_nums, repeated_values,
                    length, IsValid, bitmap);
  delete[] length;
  return;
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

int RunIntersection(uint64_t* bit_map, uint64_t* bit_map_2, const std::string& path_to_table) {
  std::vector<int64_t> _id;
  std::vector<std::string> _first_name;
  std::vector<std::string> _last_name;

  int index = 0, index2 = 0, index3 = 0, count = 0;  
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_table, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // char* char_buffer = new char[total_length];
  int64_t *values = new int64_t[BATCH_SIZE];
  parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("id");
  int col_id2 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id3 = file_metadata->schema()->ColumnIndex("lastName");
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
        if ((bit_map[index >> 6] & (1UL << (index & 63))) && (bit_map_2[index >> 6] & (1UL << (index & 63)))) {
          _id.push_back(values[i]);
          count++;
        }
        index++;
      }
    }
    column_reader = row_group_reader->Column(col_id2);
    parquet::ByteArrayReader* string_reader =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, byte_values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if ((bit_map[index2 >> 6] & (1UL << (index2 & 63))) && (bit_map_2[index2 >> 6] & (1UL << (index2 & 63)))) {
          _first_name.push_back(std::string((char*)byte_values[i].ptr, byte_values[i].len));
        }
        index2++;
      }
    }
    column_reader = row_group_reader->Column(col_id3);
    string_reader =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, byte_values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if ((bit_map[index3 >> 6] & (1UL << (index3 & 63))) && (bit_map_2[index3 >> 6] & (1UL << (index3 & 63)))) {
          _last_name.push_back(std::string((char*)byte_values[i].ptr, byte_values[i].len));
        }
        index3++;
      }
    }
  }
  return count;
}

int Intersection(const std::unordered_set<int64_t>& ids, const std::string& path_to_table, const std::string& label) {
  std::vector<int64_t> _id;
  std::vector<std::string> _first_name;
  std::vector<std::string> _last_name;

  std::unordered_set<int64_t> indices, indices2;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_table, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();
  std::unordered_map<int64_t, int64_t> index_to_id;

  // char* char_buffer = new char[total_length];
  int64_t *values = new int64_t[BATCH_SIZE];
  parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
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

  // Iterate over all the RowGroups in the file
  index = 0;
  col_id = file_metadata->schema()->ColumnIndex("labels");
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
    parquet::ByteArrayReader* string_reader =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, byte_values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if (byte_values[i].len > 0 && indices.find(index) != indices.end()) {
          auto find_ptr =
              strstr((char*)byte_values[i].ptr, label.c_str());
          if (find_ptr != nullptr && find_ptr +
                                             label.size() -
                                             (char*)byte_values[i].ptr <=
                                         byte_values[i].len) {
            indices2.insert(index);
            _id.push_back(index_to_id[index]);
            count++;
          }
        }
        index++;
      }
    }
  }

  // process first name
  index = 0;
  int index2 = 0; 
  col_id = file_metadata->schema()->ColumnIndex("firstName");
  int col_id2 = file_metadata->schema()->ColumnIndex("lastName");
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
    parquet::ByteArrayReader* string_reader =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, byte_values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if (byte_values[i].len > 0 && indices2.find(index) != indices2.end()) {
          _first_name.push_back(std::string((char*)byte_values[i].ptr, byte_values[i].len));
        }
        index++;
      }
    }

    column_reader = row_group_reader->Column(col_id2);
    parquet::ByteArrayReader* string_reader2 =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader2->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader2->ReadBatch(BATCH_SIZE, nullptr, nullptr, byte_values, &values_read);

      // check and update results
      for (int i = 0; i < rows_read; i++) {
        if (byte_values[i].len > 0 && indices2.find(index2) != indices2.end()) {
          _last_name.push_back(std::string((char*)byte_values[i].ptr, byte_values[i].len));
        }
        index2++;
      }
    }
  }
  delete[] values;
  delete[] byte_values;
  return count;
}

void RunExamples(const std::string& path_to_file, const std::string& path_to_label, const std::string& path_to_vertex, int64_t vertex_num, int64_t vertex_id, const std::string& label) {
  std::string path = path_to_file + "-delta";
  int64_t bit_map_length = vertex_num / 64 + 1;
  uint64_t* bit_map = new uint64_t[bit_map_length];
  uint64_t* bit_map2 = new uint64_t[bit_map_length];
  int32_t repeated_nums[1][VERTEX_NUM];
  bool repeated_values[1][VERTEX_NUM];
  std::vector<std::string> labels = {label};  
  memset(bit_map, 0, sizeof(uint64_t) * bit_map_length);
  memset(bit_map2, 0, sizeof(uint64_t) * bit_map_length);
  int64_t offset = 0, length = 0, offset2 = 0, length2 = 0;
  getOffset(path_to_file + "-offset", vertex_id, offset, length);
  std::cout << "offset: " << offset << ", length: " << length << std::endl;
  auto run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  GetLabelBitMap(path_to_label, labels, repeated_nums, repeated_values, IsValid, bit_map2);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run time: " << run_time << " ms" << std::endl;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  GetLabelBitMap(path_to_label, labels, repeated_nums, repeated_values, IsValid, bit_map2);
  auto run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  GetLabelBitMap(path_to_label, labels, repeated_nums, repeated_values, IsValid, bit_map2);
  auto run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  ReadBitMap(path, offset, length, bit_map);
  GetLabelBitMap(path_to_label, labels, repeated_nums, repeated_values, IsValid, bit_map2);
  auto run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average run time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;

  int inter_count = 0;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, path_to_vertex);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First intersection time: " << run_time << " ms" << std::endl;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, path_to_vertex);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, path_to_vertex);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  inter_count = RunIntersection(bit_map, bit_map2, path_to_vertex);
  run_time_3 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "Average intersection time: " << (run_time_1 + run_time_2 + run_time_3) / 3 << " ms" << std::endl;
  std::cout << "inter_count: " << inter_count << std::endl;
  delete[] bit_map; 
  delete[] bit_map2;
  // return;
}

void RunExamplesBaseLineNoOffset(const std::string& path_to_file, const std::string& path_to_label, int64_t vertex_num, int64_t vertex_id, const std::string& label) {
  std::string path = path_to_file + "-origin-base";
  std::shared_ptr<arrow::Table> table, table2;
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
  count = Intersection(ids, path_to_label, label);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "First run project time: " << run_time << " ms" << std::endl;
  run_start = clock();
  count = Intersection(ids, path_to_label, label);
  run_time_1 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  count = Intersection(ids, path_to_label, label);
  run_time_2 = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  run_start = clock();
  count = Intersection(ids, path_to_label, label);
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
  std::string path_to_label = argv[2];
  std::string path_to_vertex = argv[3];
  int64_t vertex_num = std::stol(argv[4]);
  int64_t vertex_id = std::stol(argv[5]);
  std::string label = argv[6];
  std::cout << "path_to_file: " << path_to_file << " vertex_num: " << vertex_num << " vertex_id: " << vertex_id  << " label: " << label << std::endl;
  // CheckCorretness(path_to_file, vertex_num, vertex_id);
  // return 0;
  std::string type = argv[7];  
  if (type == "delta") {
    RunExamples(path_to_file, path_to_label, path_to_vertex, vertex_num, vertex_id, label);
  } else {
    RunExamplesBaseLineNoOffset(path_to_file, path_to_vertex, vertex_num, vertex_id, label);
  }

  // if (!status.ok()) {
  //  std::cerr << "Error occurred: " << status.message() << std::endl;
  //  return EXIT_FAILURE;
  // }
  return EXIT_SUCCESS;
}
