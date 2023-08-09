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

#include <label.h>

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>

/// constants related to the test
constexpr int TOT_LABEL_NUM = 8;       // the number of total labels
constexpr int TESTED_LABEL_NUM = 4;    // the number of tested labels
constexpr int TOT_ROWS_NUM = 1000000;  // the number of total vertices
const char PARQUET_FILENAME_RLE[] =
    "parquet_graphar_label_RLE.parquet";  // the RLE filename
const char PARQUET_FILENAME_PLAIN[] =
    "parquet_graphar_label_plain.parquet";  // the PALIN filename
const char PARQUET_FILENAME_STRING[] =
    "parquet_graphar_label_string.parquet";  // the STRING filename
const char PARQUET_FILENAME_STRING_DICT[] =
    "parquet_graphar_label_string_dict.parquet";  // the STRING filename

/// The label names
const std::string label_names[TOT_LABEL_NUM] = {"label0", "label1", "label2", "label3",
                                                "label4", "label5", "label6", "label7"};
/// Tested label ids
const int tested_label_ids[TESTED_LABEL_NUM] = {0, 2, 3, 7};

/// global variables
int32_t true_num[TOT_LABEL_NUM];
int32_t false_num[TOT_LABEL_NUM];
int32_t length[TOT_LABEL_NUM];
int32_t repeated_nums[TOT_LABEL_NUM][MAX_DECODED_NUM];
bool repeated_values[TOT_LABEL_NUM][MAX_DECODED_NUM];
bool label_column_data[TOT_ROWS_NUM][MAX_LABEL_NUM];

/// The user-defined function to check if the state is valid.
/// A default implementation is provided here, which checks if all labels are contained.
static inline bool IsValid(bool* state, int column_number) {
  for (int i = 0; i < column_number; ++i) {
    if (!state[i]) {
      return false;
    }
  }
  return true;
}

/// Generate data of label columns for the parquet file (using bool datatype).
void generage_label_column_data_bool(const int num_rows, const int num_columns);

/// A hard-coded test for the function GetValidIntervals.
void hard_coded_test();

/// The test using string encoding/decoding for GraphAr labels.
void string_test(bool validate = false, bool enable_dictionary = false) {
  const char* filename;
  std::cout << "----------------------------------------\n";
  std::cout << "Running string test ";
  if (enable_dictionary) {
    std::cout << "with dictionary...\n";
    filename = PARQUET_FILENAME_STRING_DICT;
  } else {
    std::cout << "without dictionary...\n";
    filename = PARQUET_FILENAME_STRING;
  }

  // gnerate parquet file by ParquetWriter
  generate_parquet_file_string(filename, TOT_ROWS_NUM, TOT_LABEL_NUM, label_column_data,
                               label_names, false, enable_dictionary);

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));

  // test getting count
  auto run_start = clock();
  auto count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
      label_names, IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting indices
  run_start = clock();
  count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
      label_names, IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting bitmap
  run_start = clock();
  count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
      label_names, IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
            << " ms.\n"
            << std::endl;

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (int i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

/// The test using bool plain encoding/decoding for GraphAr labels.
void bool_plain_test(bool validate = false,
                     const char* filename = PARQUET_FILENAME_PLAIN) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running baseline bool test ";
  if (filename == PARQUET_FILENAME_PLAIN) {
    std::cout << "(plain)..." << std::endl;
    // gnerate parquet file by ParquetWriter
    generate_parquet_file_bool_plain(PARQUET_FILENAME_PLAIN, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                     label_column_data, label_names, false);
  } else {
    std::cout << "(RLE)..." << std::endl;
    // gnerate parquet file by ParquetWriter
    generate_parquet_file_bool_RLE(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                   label_column_data, label_names, false);
  }

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));

  // test getting count
  auto run_start = clock();
  auto count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids, IsValid,
      &indices, bitmap, QUERY_TYPE::COUNT);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting indices
  run_start = clock();
  count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids, IsValid,
      &indices, bitmap, QUERY_TYPE::INDEX);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting bitmap
  run_start = clock();
  count = read_parquet_file_and_get_valid_indices(
      filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids, IsValid,
      &indices, bitmap, QUERY_TYPE::BITMAP);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
            << " ms.\n"
            << std::endl;

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (int i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

/// The test using bool RLE encoding/decoding for GraphAr labels.
void RLE_test(bool validate = false) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running optimized RLE test..." << std::endl;

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));

  // test getting count
  auto run_start = clock();
  auto count = read_parquet_file_and_get_valid_intervals(
      PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
      tested_label_ids, repeated_nums, repeated_values, true_num, false_num, length,
      IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting indices
  run_start = clock();
  count = read_parquet_file_and_get_valid_intervals(
      PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
      tested_label_ids, repeated_nums, repeated_values, true_num, false_num, length,
      IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
            << " ms.\n"
            << std::endl;

  // test getting bitmap
  run_start = clock();
  count = read_parquet_file_and_get_valid_intervals(
      PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
      tested_label_ids, repeated_nums, repeated_values, true_num, false_num, length,
      IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
            << " ms.\n"
            << std::endl;

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (int i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

int main(int argc, char** argv) {
  // hard-coded test
  hard_coded_test();

  // generate label column data
  generage_label_column_data_bool(TOT_ROWS_NUM, TOT_LABEL_NUM);

  // string test: disable dictionary
  string_test();

  // string test: enable dictionary
  string_test(false, true);

  // bool plain test
  bool_plain_test();

  // bool test use RLE but not optimized
  bool_plain_test(false, PARQUET_FILENAME_RLE);

  // bool test use RLE and optimized
  RLE_test();

  return 0;
}

void generage_label_column_data_bool(const int num_rows, const int num_columns) {
  for (int index = 0; index < num_rows; index++) {
    for (int k = 0; k < num_columns; k++) {
      bool value;
      if (index < 50) {
        value = true;
      } else if (k == tested_label_ids[0]) {
        value = true;
      } else if (k == tested_label_ids[1]) {
        value = (index % 3333 < 999) ? true : false;
      } else if (k == tested_label_ids[2]) {
        value = ((index / 10) % 10 == 0) ? true : false;
      } else if (k == tested_label_ids[3]) {
        value = ((index / 100) % 2 == 1) ? true : false;
      } else {
        value = true;
      }
      label_column_data[index][k] = value;
    }
  }
}

void hard_coded_test() {
  std::cout << "----------------------------------------\n";
  std::cout << "Running hard-coded test..." << std::endl;

  // set length, repeated_nums, repeated_values
  // num_rows = 13, num_columns = 3
  length[0] = 3;
  length[1] = 4;
  length[2] = 6;
  repeated_nums[0][0] = 6;
  repeated_values[0][0] = 1;
  repeated_nums[0][1] = 6;
  repeated_values[0][1] = 1;
  repeated_nums[0][2] = 1;
  repeated_values[0][2] = 1;
  repeated_nums[1][0] = 4;
  repeated_values[1][0] = 0;
  repeated_nums[1][1] = 5;
  repeated_values[1][1] = 1;
  repeated_nums[1][2] = 3;
  repeated_values[1][2] = 0;
  repeated_nums[1][3] = 1;
  repeated_values[1][3] = 1;
  repeated_nums[2][0] = 2;
  repeated_values[2][0] = 0;
  repeated_nums[2][1] = 2;
  repeated_values[2][1] = 1;
  repeated_nums[2][2] = 1;
  repeated_values[2][2] = 0;
  repeated_nums[2][3] = 3;
  repeated_values[2][3] = 1;
  repeated_nums[2][4] = 4;
  repeated_values[2][4] = 0;
  repeated_nums[2][5] = 1;
  repeated_values[2][5] = 1;

  auto count = GetValidIntervals(3, 13, repeated_nums, repeated_values, length, IsValid);

  std::cout << "The valid count is: " << count << std::endl;
  std::cout << "----------------------------------------\n";
}
