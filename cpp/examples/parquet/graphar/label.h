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
constexpr int TOT_LABEL_NUM = 8;         // the number of total labels
constexpr int MAX_DECODED_NUM = 100000;  // the maximum number of decoded values
/// constants related to the parquet file
constexpr int NUM_ROWS_PER_ROW_GROUP = 2000;  // the number of rows per row group
constexpr int BATCH_SIZE = 1024;              // the batch size
// kDefaultDataPageSize = 1024 * 1024
// DEFAULT_WRITE_BATCH_SIZE = 1024
// DEFAULT_MAX_ROW_GROUP_LENGTH = 1024 * 1024

/// Setup the schema of the parquet file
static std::shared_ptr<GroupNode> SetupSchema(const std::string* label_names) {
  parquet::schema::NodeVector fields;

  // Add TOT_LABEL_NUM primitive nodes with specific names to the group node
  for (int i = 0; i < TOT_LABEL_NUM; ++i) {
    // Create a primitive node with type:BOOLEAN, repetition:REQUIRED
    if (label_names == nullptr) {
      fields.push_back(PrimitiveNode::Make("label_" + std::to_string(i),
                                           Repetition::REQUIRED, Type::BOOLEAN,
                                           ConvertedType::NONE));
    } else {
      fields.push_back(PrimitiveNode::Make(label_names[i], Repetition::REQUIRED,
                                           Type::BOOLEAN, ConvertedType::NONE));
    }
  }

  // Create a primitive node named 'id' with type:INT64, repetition:REQUIRED,
  fields.push_back(
      PrimitiveNode::Make("id", Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));

  // Create a GroupNode named 'schema' using the primitive nodes defined above
  // This GroupNode is the root node of the schema tree
  return std::static_pointer_cast<GroupNode>(
      GroupNode::Make("schema", Repetition::REQUIRED, fields));
}

/// Add an interval to the result
static inline void AddToResult(std::vector<std::pair<int, int> >& intervals,
                               const int previous_pos, const int current_pos) {
  if (intervals.empty() || intervals.back().second != previous_pos) {
    intervals.push_back(std::make_pair(previous_pos, current_pos));
  } else {
    intervals.back().second = current_pos;
  }
}

/// Get the valid intervals of the labels, "column_number" is the number of columns
static std::vector<std::pair<int, int> > GetValidIntervals(
    const int column_number, const int row_number,
    int32_t repeated_nums[][MAX_DECODED_NUM], bool repeated_values[][MAX_DECODED_NUM],
    int32_t* length, const std::function<bool(bool*, int)>& IsValid) {
  // initialization
  std::vector<std::pair<int, int> > intervals;
  int current_pos = 0, previous_pos = 0;
  int pos[TOT_LABEL_NUM] = {0};
  int index[TOT_LABEL_NUM] = {0};
  bool state[TOT_LABEL_NUM];
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
      AddToResult(intervals, previous_pos, current_pos);
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

  return intervals;
}

std::vector<std::pair<int, int> > read_parquet_file_and_get_valid_intervals(
    const char* parquet_filename, const int row_num, const int column_num,
    int32_t repeated_nums[][MAX_DECODED_NUM], bool repeated_values[][MAX_DECODED_NUM],
    int32_t* true_num, int32_t* false_num, int32_t* length,
    const std::function<bool(bool*, int)>& IsValid);

void generate_parquet_file(const char* parquet_filename, const int row_num,
                           const std::string* label_names = nullptr);
