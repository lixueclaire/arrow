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

#ifndef PARQUET_EXAMPLES_GRAPHAR_LABEL_H
#define PARQUET_EXAMPLES_GRAPHAR_LABEL_H

#include <arrow/io/file.h>
#include <arrow/util/logging.h>
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>
#include <parquet/properties.h>

#include <iostream>
#include <vector>

using parquet::ConvertedType;
using parquet::Encoding;
using parquet::Repetition;
using parquet::Type;
using parquet::schema::GroupNode;
using parquet::schema::PrimitiveNode;

/// constants related to encoding and decoding of the labels
constexpr int MAX_LABEL_NUM = 20;         // the maximum number of labels
constexpr int MAX_DECODED_NUM = 5000000;  // the maximum number of decoded values
/// constants related to the parquet file
constexpr int NUM_ROWS_PER_ROW_GROUP = 1024 * 1024;  // the number of rows per row group
constexpr int BATCH_SIZE = 1024;                   // the batch size
// kDefaultDataPageSize = 1024 * 1024
// DEFAULT_WRITE_BATCH_SIZE = 1024
// DEFAULT_MAX_ROW_GROUP_LENGTH = 1024 * 1024

/// The query type
enum QUERY_TYPE {
  COUNT,  // return the number of valid vertices
  INDEX,  // return the indices of valid vertices
  BITMAP  // return the bitmap of valid vertices
};

/// Set bit
static inline void SetBitmap(uint64_t* bitmap, const int index) {
  bitmap[index >> 6] |= (1ULL << (index & 63));
}

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

/// Get bit
static inline bool GetBit(const uint64_t* bitmap, const int index) {
  return (bitmap[index >> 6]) & (1ULL << (index & 63));
}

/// Get the valid intervals of the labels, "column_number" is the number of columns
static int GetValidIntervals(const int column_number, const int row_number,
                             int32_t repeated_nums[][MAX_DECODED_NUM],
                             bool repeated_values[][MAX_DECODED_NUM], int32_t* length,
                             const std::function<bool(bool*, int)>& IsValid,
                             std::vector<int>* indices = nullptr,
                             uint64_t* bitmap = nullptr,
                             const QUERY_TYPE query_type = COUNT) {
  // initialization
  std::vector<std::pair<int, int> > intervals;
  int current_pos = 0, previous_pos = 0, count = 0;
  int pos[MAX_LABEL_NUM] = {0};
  int index[MAX_LABEL_NUM] = {0};
  bool state[MAX_LABEL_NUM];
  for (int i = 0; i < column_number; ++i) {
    state[i] = repeated_values[i][index[i]];
  }

  // K-path merging
  while (true) {
    // find the minimum position of change
    int min_pos = INT32_MAX;
    for (int i = 0; i < column_number; ++i) {
      if (index[i] < length[i] && pos[i] + repeated_nums[i][index[i]] < min_pos) {
        min_pos = pos[i] + repeated_nums[i][index[i]];
      }
    }
    // check the last interval and add it to the result if it is valid
    previous_pos = current_pos;
    current_pos = min_pos;
    if (IsValid(state, column_number)) {
      count += current_pos - previous_pos;
      if (query_type == INDEX) {
        for (int i = previous_pos; i < current_pos; ++i) {
          indices->push_back(i);
        }
      } else if (query_type == BITMAP) {
        SetBitmap(bitmap, previous_pos, current_pos);
      }
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
          state[i] = repeated_values[i][index[i]];
        }
      }
    }
  }

  return count;
}

int read_parquet_file_and_get_valid_indices(
    const char* parquet_filename, const int row_num, const int tot_label_num,
    const int tested_label_num, const int* tested_label_ids,
    const std::string* label_names, const std::function<bool(bool*, int)>& IsValid,
    std::vector<int>* indices = nullptr, uint64_t* bitmap = nullptr,
    const QUERY_TYPE query_type = COUNT);

int read_parquet_file_and_get_valid_indices(
    const char* parquet_filename, const int row_num, const int tot_label_num,
    const int tested_label_num, const int* tested_label_ids,
    const std::function<bool(bool*, int)>& IsValid, std::vector<int>* indices = nullptr,
    uint64_t* bitmap = nullptr, const QUERY_TYPE query_type = COUNT);

int read_parquet_file_and_get_valid_intervals(
    const char* parquet_filename, const int row_num, const int tot_label_num,
    const int tested_label_num, const int* tested_label_ids,
    int32_t repeated_nums[][MAX_DECODED_NUM], bool repeated_values[][MAX_DECODED_NUM],
    int32_t* true_num, int32_t* false_num, int32_t* length,
    const std::function<bool(bool*, int)>& IsValid, std::vector<int>* indices = nullptr,
    uint64_t* bitmap = nullptr, const QUERY_TYPE query_type = COUNT);

void generate_parquet_file_string(const char* parquet_filename, const int row_num,
                                  const int label_num,
                                  bool label_column_data[][MAX_LABEL_NUM],
                                  const std::string* label_names = nullptr,
                                  const bool contain_id_column = true,
                                  const bool enable_dictionary = false);

void generate_parquet_file_bool_plain(const char* parquet_filename, const int row_num,
                                      const int label_num,
                                      bool label_column_data[][MAX_LABEL_NUM],
                                      const std::string* label_names = nullptr,
                                      const bool contain_id_column = true);

void generate_parquet_file_bool_RLE(const char* parquet_filename, const int row_num,
                                    const int label_num,
                                    bool label_column_data[][MAX_LABEL_NUM],
                                    const std::string* label_names = nullptr,
                                    const bool contain_id_column = true);

/// !!! This is only used for debug
/* static inline void validate_column(const int col_id, const int row_num,
                                   int32_t repeated_nums[][MAX_DECODED_NUM],
                                   bool repeated_values[][MAX_DECODED_NUM],
                                   int32_t* true_num, int32_t* false_num,
                                   int32_t* length); */

#endif  // PARQUET_EXAMPLES_GRAPHAR_LABEL_H