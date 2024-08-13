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
// 1. cyber-security-ad-44-nodes.csv - too small
// row number: 954, label number: 7, test id: 0, 1, AND
// 2. graph-data-science-43-nodes.csv - poor
// row number: 2687, label number: 12, test id: 1, 6, AND
// 3. twitter-v2-43-nodes.csv
// row number: 43337, label number: 6, test id: 0, 1, AND
// 4. network-management-43-nodes.csv
// row number: 83847, label number: 17, test id: 4, 5, AND
// 5. fraud-detection-43-nodes.csv
// row number: 333022, label number: 13, test id: 3, 5, AND

// new datasets
// 1. legis-graph-43-nodes.csv, 11825, 8, {0, 1}, OR
// 2. recommendations-43-nodes.csv, 33880, 6, {3, 4}, AND - poor
// 3. bloom-43-nodes.csv, 32960, 18, {8, 9}, OR
// 4. pole-43-nodes.csv, 61534, 11, {1, 5}, OR
// 5. openstreetmap-43-nodes.csv, 71566, 10, {2, 5}, AND
// 6. icij-paradise-papers-43-nodes.csv, 163414, 5, {0, 4}, OR
// 7. citations-43-nodes.csv, 263902, 3, {0, 2}, OR
// 8. twitter-trolls-43-nodes.csv, 281177, 6, {0, 1}, AND
// 9. icij-offshoreleaks-44-nodes.csv, 1969309, 5, {1, 3}, OR

// ldbc datasets
// 1. place_0_0.csv, 1460, 4, {0, 2}, AND
// 2. organisation_0_0.csv, 7955, 3, {0, 1}, AND

// ogb datasets
// 1. ogbn-arxiv.csv, 169343, 40, {0, 1}, OR
// 2. ogbn-proteins.csv, 132534, 112, {0, 1}, AND
// 3. ogbn-mag.csv, 736389, 349, {0, 1}, OR
// 4. ogbn-products.csv, 2449029, 47, {0, 1}, OR

const int TEST_ROUNDS = 1;                            // the number of test rounds
const int TOT_ROWS_NUM = 100000;                    // the number of total vertices
const int TOT_LABEL_NUM = 20;                         // the number of total labels
const QUERY_TYPE fix_query_type = QUERY_TYPE::COUNT;  // the query type
int TESTED_LABEL_NUM = 1;                             // the number of tested labels
int tested_label_ids[TOT_LABEL_NUM] = {0};            // the ids of tested labels
double res[TOT_LABEL_NUM][5];                         // the results
size_t res_size[TOT_LABEL_NUM][4];                    // the results for size

const char PARQUET_FILENAME_RLE[] =
    "parquet_graphar_label_RLE.parquet";  // the RLE filename
const char PARQUET_FILENAME_PLAIN[] =
    "parquet_graphar_label_plain.parquet";  // the PLAIN filename
const char PARQUET_FILENAME_STRING[] =
    "parquet_graphar_label_string.parquet";  // the STRING filename
const char PARQUET_FILENAME_STRING_DICT[] =
    "parquet_graphar_label_string_dict.parquet";  // the STRING filename

std::string label_names[MAX_LABEL_NUM];

/// global variables
int32_t length[TOT_LABEL_NUM];
int32_t repeated_nums[TOT_LABEL_NUM][MAX_DECODED_NUM];
bool repeated_values[TOT_LABEL_NUM][MAX_DECODED_NUM];
bool label_column_data[TOT_ROWS_NUM][MAX_LABEL_NUM];

/// The user-defined function to check if the state is valid.
/// A default implementation is provided here, which checks if all labels are contained.
static inline bool IsValid(bool* state, int column_number) {
  for (int i = 0; i < column_number; ++i) {
    // if (!state[i]) return false; // AND case
    if (state[i]) return true;  // OR case
  }
  // return true; // AND case
  return false;  // OR case
}

/// The test using string encoding/decoding for GraphAr labels.
double string_test(bool validate = false, bool enable_dictionary = false) {
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

  if (TESTED_LABEL_NUM == 1 && tested_label_ids[0] == 0) {
    // generate parquet file by ParquetWriter
    generate_parquet_file_string(filename, TOT_ROWS_NUM, TOT_LABEL_NUM, label_column_data,
                                 label_names, false, enable_dictionary);
  }
  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  int count;
  double res;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::ADAPTIVE || fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      indices.clear();
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;

    res = run_time;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";

  return res;
}

/// The test using bool plain encoding/decoding for GraphAr labels.
double bool_plain_test(bool validate = false,
                       const char* filename = PARQUET_FILENAME_PLAIN) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running baseline bool test ";
  if (filename == PARQUET_FILENAME_PLAIN) {
    std::cout << "(plain)..." << std::endl;
    if (TESTED_LABEL_NUM == 1 && tested_label_ids[0] == 0) {
      // generate parquet file by ParquetWriter
      generate_parquet_file_bool_plain(PARQUET_FILENAME_PLAIN, TOT_ROWS_NUM,
                                       TOT_LABEL_NUM, label_column_data, label_names,
                                       false);
    }
  } else {
    std::cout << "(RLE)..." << std::endl;
    // generate parquet file by ParquetWriter
    if (TESTED_LABEL_NUM == 1 && tested_label_ids[0] == 0) {
      generate_parquet_file_bool_RLE(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                     label_column_data, label_names, false);
    }
  }

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;
  double res;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::ADAPTIVE || fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      indices.clear();
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
  return res;
}

/// The test using bool RLE encoding/decoding for GraphAr labels.
double RLE_test(bool validate = false) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running optimized RLE test..." << std::endl;

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;
  double res;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::COUNT);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test adaptive mode
  if (fix_query_type == QUERY_TYPE::ADAPTIVE) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      indices.clear();
      memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::ADAPTIVE);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (ADAPTIVE) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
    if (TESTED_LABEL_NUM == 1) {
      res_size[tested_label_ids[0]][0] = count;
      res_size[tested_label_ids[0]][1] = sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1);
      res_size[tested_label_ids[0]][2] = sizeof(int) * count;
      res_size[tested_label_ids[0]][3] =
          std::min(res_size[tested_label_ids[0]][1], res_size[tested_label_ids[0]][2]);
    }
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      indices.clear();
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
    res = run_time;
  }

  std::cout << "The valid count is: " << count << std::endl;

  delete[] bitmap;
  std::cout << "----------------------------------------\n";

  return res;
}

// read csv and generate label column data
void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns);

// the scale test
void scale_test() {
  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    tested_label_ids[i] = i;
  }

  for (int i = 0; i <= TOT_LABEL_NUM; i++) {
    // the first round is omitted (preload)
    TESTED_LABEL_NUM = std::max(1, i);
    // string test: disable dictionary
    res[TESTED_LABEL_NUM - 1][0] = string_test();
  }

  /* for (int i = 0; i <= TOT_LABEL_NUM; i++) {
    // the first round is omitted (preload)
    TESTED_LABEL_NUM = std::max(1, i);
    // string test: enable dictionary
    res[TESTED_LABEL_NUM - 1][1] = string_test(false, true);
  } */

  for (int i = 0; i <= TOT_LABEL_NUM; i++) {
    // the first round is omitted (preload)
    TESTED_LABEL_NUM = std::max(1, i);
    // bool plain test
    res[TESTED_LABEL_NUM - 1][2] = bool_plain_test();
  }

  for (int i = 0; i <= TOT_LABEL_NUM; i++) {
    // the first round is omitted (preload)
    TESTED_LABEL_NUM = std::max(1, i);
    // bool test use RLE but not optimized
    res[TESTED_LABEL_NUM - 1][3] = bool_plain_test(false, PARQUET_FILENAME_RLE);
  }

  for (int i = 0; i <= TOT_LABEL_NUM; i++) {
    // the first round is omitted (preload)
    TESTED_LABEL_NUM = std::max(1, i);
    // bool test use RLE and optimized
    res[TESTED_LABEL_NUM - 1][4] = RLE_test();
  }

  std::cout << "----------------------------------------\n";
  std::cout << "Test rounds = " << TEST_ROUNDS << std::endl;
  std::cout << "The results for scale are: \n";
  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%.6lf", res[i][j]);
      if (j != 4) {
        printf(", ");
      }
    }
    std::cout << std::endl;
  }
  std::cout << "----------------------------------------\n";
}

void one_column_test() {
  TESTED_LABEL_NUM = 1;
  std::vector<std::vector<double> > res;
  for (int i = 0; i < 5; i++) {
    std::vector<double> tmp;
    res.push_back(tmp);
  }

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = string_test();
    }
    // string test: disable dictionary
    res[0].push_back(string_test());
  }

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = string_test(false, true);
    }
    // string test: enable dictionary
    res[1].push_back(string_test(false, true));
  }

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = bool_plain_test();
    }
    // bool plain test
    res[2].push_back(bool_plain_test());
  }

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = bool_plain_test(false, PARQUET_FILENAME_RLE);
    }
    // bool test use RLE but not optimized
    res[3].push_back(bool_plain_test(false, PARQUET_FILENAME_RLE));
  }

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = RLE_test();
    }
    // bool test use RLE and optimized
    res[4].push_back(RLE_test());
  }

  std::cout << "----------------------------------------\n";
  std::cout << "Test rounds = " << TEST_ROUNDS << std::endl;
  std::cout << "The results for one_column are: \n";
  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%.6lf", res[j][i]);
      if (j != 4) {
        printf(", ");
      }
    }
    std::cout << std::endl;
  }
  std::cout << "----------------------------------------\n";
  for (int j = 0; j < 5; j++) {
    std::sort(res[j].begin(), res[j].end());
  }
  std::cout << "The sorted results for one_column are: \n";
  double sum[5] = {0};
  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%.6lf", res[j][i]);
      if (j != 4) {
        printf(", ");
      }
      sum[j] += res[j][i];
    }
    std::cout << std::endl;
  }
  std::cout << "----------------------------------------\n";
  std::cout << "The average results for one_column are: \n";
  for (int j = 0; j < 5; j++) {
    printf("%.6lf", sum[j] / TOT_LABEL_NUM);
    if (j != 4) {
      printf(", ");
    }
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------\n";
}

void adaptive_test() {
  TESTED_LABEL_NUM = 1;
  std::vector<double> res;

  generate_parquet_file_bool_RLE(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
                                 label_column_data, label_names, false);

  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    // set the tested label id
    tested_label_ids[0] = i;
    // the first round is omitted (preload)
    if (i == 0) {
      double tmp = RLE_test();
    }

    // bool test use RLE and optimized
    res.push_back(RLE_test());
  }

  std::cout << "----------------------------------------\n";
  std::cout << "Test rounds = " << TEST_ROUNDS << ", ROWS_NUM = " << TOT_ROWS_NUM
            << std::endl;
  std::cout << "The size for results are: \n";
  std::cout << "Time, Count, Density, Bitmap size, Index size, Adaptive size\n";
  for (int i = 0; i < TOT_LABEL_NUM; i++) {
    printf("%.6lf, ", res[i]);
    printf("%zu, ", res_size[i][0]);
    printf("%.6lf, ", res_size[i][0] * 1.0 / TOT_ROWS_NUM);
    for (int j = 1; j < 4; j++) {
      printf("%zu", res_size[i][j]);
      if (j != 3) {
        printf(", ");
      }
    }
    std::cout << std::endl;
  }
  std::cout << "----------------------------------------\n";
}

int main(int argc, char** argv) {
  // read csv and generate label column data
  read_csv_file_and_generate_label_column_data_bool(TOT_ROWS_NUM, TOT_LABEL_NUM);

  // scale test
  // scale_test();

  // one column test
  one_column_test();

  // adaptive test
  // adaptive_test();

  return 0;
}

void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns) {
  // the first line is the header
  for (int k = 0; k < num_columns; k++) {
    std::cin >> label_names[k];
  }
  // read the data
  int cnt[TOT_LABEL_NUM] = {0};
  for (int index = 0; index < num_rows; index++) {
    for (int k = 0; k < num_columns; k++) {
      std::cin >> label_column_data[index][k];
      if (label_column_data[index][k] == 1) {
        cnt[k]++;
      }
    }
  }
}
