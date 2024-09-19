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
#include <set>

/// constants related to the test, person
// sf30 165430 2 1460
// sf100 448626 2 1460
// sf300 1128069 2 1460

/// constants related to the test, comment, post, tagclasses
// sf1 1739438 1121226 71
// sf30 67126524 19968658 71
// sf100 220096052 57987023 71
// sf300 650086949 155783470 71

const int TOT_ROWS_NUM = 100;                    // the number of total vertices

const int TEST_ROUNDS = 1;                             // the number of test rounds
const int TOT_LABEL_NUM = 71;                          // the number of total labels
const int TESTED_LABEL_NUM = 1;                        // the number of tested labels
int tested_label_ids[TESTED_LABEL_NUM] = {17};         // the ids of tested labels
std::string tested_label_name = "SoccerPlayer";       // the name of tested label
const QUERY_TYPE fix_query_type = QUERY_TYPE::BITMAP;  // the query type

const char PARQUET_FILENAME_RLE[] =  //"parquet_graphar_label_RLE.parquet";
    "/mnt/ldbc/lixue/backup/opt-2/ldbc_bi_label/sf1/post/"
    "parquet_graphar_label_RLE.parquet";  // the RLE filename
const char PARQUET_FILENAME_STRING[] =
    "/mnt/ldbc/lixue/backup/opt-2/ldbc_bi_label/sf1/post/"
    "parquet_graphar_label_string.parquet";  // the STRING filename

const char PARQUET_FILENAME_STRING_DICT[] =
    "parquet_graphar_label_string_dict.parquet";  // the STRING filename
 const char PARQUET_FILENAME_PLAIN[] =
     "parquet_graphar_label_plain.parquet";  // the PLAIN filename */

std::string label_names[MAX_LABEL_NUM];

/// global variables
int32_t length[TOT_LABEL_NUM];
// int32_t repeated_nums[TESTED_LABEL_NUM][MAX_DECODED_NUM];
// bool repeated_values[TESTED_LABEL_NUM][MAX_DECODED_NUM];
int32_t* repeated_nums1;
bool* repeated_values1;
// bool label_column_data[TOT_ROWS_NUM][MAX_LABEL_NUM];
std::set<int32_t> has_label[TOT_LABEL_NUM];

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

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

/// The test using bool plain encoding/decoding for GraphAr labels.
/* void bool_plain_test(bool validate = false,
                     const char* filename = PARQUET_FILENAME_PLAIN) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running baseline bool test ";
  if (filename == PARQUET_FILENAME_PLAIN) {
    std::cout << "(plain)..." << std::endl;
  } else {
    std::cout << "(RLE)..." << std::endl;
  }

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
} */

/// The test using bool RLE encoding/decoding for GraphAr labels.
/* void RLE_test(bool validate = false) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running optimized RLE test..." << std::endl;

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test adaptive mode
  if (fix_query_type == QUERY_TYPE::ADAPTIVE) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::ADAPTIVE);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (ADAPTIVE) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
} */

void RLE_2_test(bool validate = false) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running optimized RLE test (version 2)..." << std::endl;

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals_2(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums1, repeated_values1, length, IsValid, &indices,
          bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

// read csv and generate label column data
void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns);

void generate_parquet_file();

int main(int argc, char** argv) {
  // read csv and generate label column data
  // read_csv_file_and_generate_label_column_data_bool(TOT_ROWS_NUM, TOT_LABEL_NUM);

  // generate parquet file by ParquetWriter
  // generate_parquet_file();

  // string test: disable dictionary
  label_names[tested_label_ids[0]] = tested_label_name;
  string_test();

  // string test: enable dictionary
  // string_test(false, true);

  // bool plain test
  // bool_plain_test();

  // bool test use RLE but not optimized
  // bool_plain_test(false, PARQUET_FILENAME_RLE);

  // bool test use RLE and optimized
  repeated_nums1 = new int32_t[TOT_ROWS_NUM / 2];
  repeated_values1 = new bool[TOT_ROWS_NUM / 2];
  RLE_2_test();
  delete[] repeated_nums1;
  delete[] repeated_values1;

  return 0;
}

void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns) {
  // the first line is the header
  for (int k = 0; k < num_columns; k++) {
    std::cin >> label_names[k];
  }
  // read the data
  int index, label_id;
  while (scanf("%d %d", &index, &label_id) == 2) {
    // label_column_data[index][label_id] = true;
    has_label[label_id].insert(index);
  }
}

void generate_parquet_file() {
  generate_parquet_file_bool_RLE_2(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                   has_label, label_names, false);
  generate_parquet_file_string_2(PARQUET_FILENAME_STRING, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                 has_label, label_names, false, false);
}
