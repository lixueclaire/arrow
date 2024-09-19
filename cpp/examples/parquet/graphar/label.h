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
#include <set>
#include <vector>

#include "config.h"

using parquet::ConvertedType;
using parquet::Encoding;
using parquet::Repetition;
using parquet::Type;
using parquet::schema::GroupNode;
using parquet::schema::PrimitiveNode;

/// The query type
enum QUERY_TYPE {
  COUNT,    // return the number of valid vertices
  INDEX,    // return the indices of valid vertices
  BITMAP,   // return the bitmap of valid vertices
  ADAPTIVE  // adaptively return indices or bitmap
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
static inline int GetValidIntervals(const int column_number, const int row_number,
                                    int32_t repeated_nums[][MAX_DECODED_NUM],
                                    bool repeated_values[][MAX_DECODED_NUM],
                                    int32_t* length,
                                    const std::function<bool(bool*, int)>& IsValid,
                                    std::vector<int>* indices = nullptr,
                                    uint64_t* bitmap = nullptr,
                                    const QUERY_TYPE query_type = COUNT) {
  // initialization
  int current_pos = 0, previous_pos = 0, count = 0, min_pos;
  int pos[MAX_LABEL_NUM] = {0};
  int index[MAX_LABEL_NUM] = {0};
  bool state[MAX_LABEL_NUM];
  for (int i = 0; i < column_number; ++i) {
    state[i] = repeated_values[i][index[i]];
  }
  std::vector<int> interval;
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
    if (min_pos == INT32_MAX) {
      // std::cout << length[0] << " " << pos[0] << " " << index[0] << std::endl;
    }
    // check the last interval and add it to the result if it is valid
    previous_pos = current_pos;
    current_pos = min_pos;
    if (state_change) {
      last_res = IsValid(state, column_number);
      state_change = false;
    }
    if (last_res) {
      count += current_pos - previous_pos;
      if (query_type == ADAPTIVE) {
        if (interval.size() > 0 && interval.back() == previous_pos) {
          interval[interval.size() - 1] = current_pos;  // merge the intervals
        } else {
          interval.push_back(previous_pos);
          interval.push_back(current_pos);
        }
      } else if (query_type == INDEX) {
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
          state_change |= (state[i] != repeated_values[i][index[i]]);
          state[i] = repeated_values[i][index[i]];
        }
      }
    }
  }

  if (query_type == ADAPTIVE) {
    size_t m = interval.size();
    if (count < row_number / THRESHOLD) {  // sparse mode
      for (size_t i = 0; i < m; i += 2) {
        for (int j = interval[i]; j < interval[i + 1]; j++) {
          indices->push_back(j);
        }
      }
    } else {  // dense mode
      for (size_t i = 0; i < m; i += 2) {
        SetBitmap(bitmap, interval[i], interval[i + 1]);
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
    int32_t* length, const std::function<bool(bool*, int)>& IsValid,
    std::vector<int>* indices = nullptr, uint64_t* bitmap = nullptr,
    const QUERY_TYPE query_type = COUNT);

int read_parquet_file_and_get_valid_intervals_2(
    const char* parquet_filename, const int row_num, const int tot_label_num,
    const int tested_label_num, const int* tested_label_ids, int32_t* repeated_nums,
    bool* repeated_values, int32_t* length,
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

void generate_parquet_file_string_2(const char* parquet_filename, const int row_num,
                                    const int label_num, std::set<int32_t> has_labels[],
                                    const std::string* label_names = nullptr,
                                    const bool contain_id_column = true,
                                    const bool enable_dictionary = false);

void generate_parquet_file_bool_RLE_2(const char* parquet_filename, const int row_num,
                                      const int label_num, std::set<int32_t> has_labels[],
                                      const std::string* label_names = nullptr,
                                      const bool contain_id_column = true);

/// !!! This is only used for debug
/* static inline void validate_column(const int col_id, const int row_num,
                                   int32_t repeated_nums[][MAX_DECODED_NUM],
                                   bool repeated_values[][MAX_DECODED_NUM],
                                   int32_t* length); */

/// Get the valid intervals of the labels, "column_number" is the number of columns
static inline int GetValidIntervals2(const int column_number, const int row_number,
                                     int32_t* repeated_nums, bool* repeated_values,
                                     int32_t* length,
                                     const std::function<bool(bool*, int)>& IsValid,
                                     std::vector<int>* indices = nullptr,
                                     uint64_t* bitmap = nullptr,
                                     const QUERY_TYPE query_type = COUNT) {
  if (column_number != 1) {
    std::cout << "Error: column_number != 1" << std::endl;
    return -1;
  }
  // initialization
  int current_pos = 0, previous_pos = 0, count = 0, min_pos;
  int pos[MAX_LABEL_NUM] = {0};
  int index[MAX_LABEL_NUM] = {0};
  bool state[MAX_LABEL_NUM];
  for (int i = 0; i < column_number; ++i) {
    state[i] = repeated_values[index[i]];
  }
  std::vector<int> interval;
  bool state_change = true, last_res = false;

  // K-path merging
  while (true) {
    // find the minimum position of change
    min_pos = INT32_MAX;
    for (int i = 0; i < column_number; ++i) {
      if (index[i] < length[i]) {
        min_pos = std::min(min_pos, pos[i] + repeated_nums[index[i]]);
      }
    }
    if (min_pos == INT32_MAX) {
      // std::cout << length[0] << " " << pos[0] << " " << index[0] << std::endl;
    }
    // check the last interval and add it to the result if it is valid
    previous_pos = current_pos;
    current_pos = min_pos;
    if (state_change) {
      last_res = IsValid(state, column_number);
      state_change = false;
    }
    if (last_res) {
      count += current_pos - previous_pos;
      if (query_type == ADAPTIVE) {
        if (interval.size() > 0 && interval.back() == previous_pos) {
          interval[interval.size() - 1] = current_pos;  // merge the intervals
        } else {
          interval.push_back(previous_pos);
          interval.push_back(current_pos);
        }
      } else if (query_type == INDEX) {
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
      if (index[i] < length[i] && pos[i] + repeated_nums[index[i]] == min_pos) {
        pos[i] = min_pos;
        index[i]++;
        if (index[i] < length[i]) {
          state_change |= (state[i] != repeated_values[index[i]]);
          state[i] = repeated_values[index[i]];
        }
      }
    }
  }

  if (query_type == ADAPTIVE) {
    size_t m = interval.size();
    if (count < row_number / THRESHOLD) {  // sparse mode
      for (size_t i = 0; i < m; i += 2) {
        for (int j = interval[i]; j < interval[i + 1]; j++) {
          indices->push_back(j);
        }
      }
    } else {  // dense mode
      for (size_t i = 0; i < m; i += 2) {
        SetBitmap(bitmap, interval[i], interval[i + 1]);
      }
    }
  }

  return count;
}

#endif  // PARQUET_EXAMPLES_GRAPHAR_LABEL_H