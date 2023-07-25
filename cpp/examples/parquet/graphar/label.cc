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
#include <fstream>
#include <iostream>
#include <memory>

/// Generate a parquet file by ParquetWriter, containing row_num rows,
/// the label names are given by label_names.
/// The function SetupSchema() is used to setup the parquet schema.
void generate_parquet_file(const char* parquet_filename, const int row_num,
                           const int label_num, const std::string* label_names,
                           const bool contain_id_column) {
  std::cout << "Generating a parquet file: " << parquet_filename << std::endl
            << "row_num = " << row_num << ", label_num = " << label_num
            << ", contain_id_column = " << contain_id_column << std::endl;

  // Create a local file output stream instance
  using FileClass = ::arrow::io::FileOutputStream;
  std::shared_ptr<FileClass> out_file;
  PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(parquet_filename));

  // Setup the parquet schema: label_num label columns and one id column
  std::shared_ptr<GroupNode> schema =
      SetupSchema(label_num, label_names, contain_id_column);
  // Add writer properties: no compression
  parquet::WriterProperties::Builder builder;
  builder.compression(parquet::Compression::UNCOMPRESSED);
  // Add writer properties: use RLE encoding for the label columns
  for (int i = 0; i < label_num; ++i) {
    builder.encoding(schema->field(i)->name(), parquet::Encoding::RLE);
  }
  // Add writer properties: use PLAIN encoding for the id column
  if (contain_id_column) {
    builder.encoding(schema->field(label_num)->name(), parquet::Encoding::PLAIN);
  }
  std::shared_ptr<parquet::WriterProperties> props = builder.build();

  // Create a ParquetFileWriter instance
  std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

  // Calulate the number of RowGroups
  int row_group_count = (row_num + NUM_ROWS_PER_ROW_GROUP - 1) / NUM_ROWS_PER_ROW_GROUP;

  // Allocate buffers
  bool* bool_buffer = new bool[BATCH_SIZE];
  int64_t* int_buffer = new int64_t[BATCH_SIZE];
  int buffer_size = 0;

  // Append a RowGroup to the ParquetFileWriter instance
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Append a RowGroup with a specific number of rows
    parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();

    // Write the label columns
    for (int k = 0; k < label_num; k++) {
      parquet::BoolWriter* bool_writer =
          static_cast<parquet::BoolWriter*>(rg_writer->NextColumn());

      buffer_size = 0;
      for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
        int64_t index = i + rg * NUM_ROWS_PER_ROW_GROUP;
        if (index >= row_num) {
          break;  // all rows are written
        }
        bool value;
        if (k == 0) {
          value = (i % 2 == 1) ? true : false;  // label_0 is true for odd rows
        } else if (k == 1) {
          value = (i % 10 == 1) ? true
                                : false;  // label_1 is true for rows with index % 10 == 1
        } else if (k == 2) {
          value = ((i / 10) % 10 == 0)
                      ? true
                      : false;  // label_2 is true for rows with index / 10 % 10 == 0
        } else {
          value = ((i / 100) % 2 == 1)
                      ? true
                      : false;  // label_3 is true for rows with index / 100 % 2 == 1
        }
        bool_buffer[buffer_size++] = value;
        if (buffer_size == BATCH_SIZE) {
          bool_writer->WriteBatch(buffer_size, nullptr, nullptr, bool_buffer);
          buffer_size = 0;
        }
      }

      if (buffer_size > 0) {
        bool_writer->WriteBatch(buffer_size, nullptr, nullptr, bool_buffer);
      }
    }

    if (!contain_id_column) {
      continue;  // skip generating the id column
    }

    buffer_size = 0;
    // Write the Int32 column (id)
    parquet::Int64Writer* int64_writer =
        static_cast<parquet::Int64Writer*>(rg_writer->NextColumn());
    for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
      int64_t index = i + rg * NUM_ROWS_PER_ROW_GROUP;
      if (index >= row_num) {
        break;  // all rows are written
      }
      int_buffer[buffer_size++] = index;
      if (buffer_size == BATCH_SIZE) {
        int64_writer->WriteBatch(buffer_size, nullptr, nullptr, int_buffer);
        buffer_size = 0;
      }
    }
    if (buffer_size > 0) {
      int64_writer->WriteBatch(buffer_size, nullptr, nullptr, int_buffer);
    }
  }
  file_writer->Close();

  // Write the bytes to file
  DCHECK(out_file->Close().ok());
  std::cout << "The parquet file is generated successfully!" << std::endl << std::endl;

  // delete the allocated space
  delete[] bool_buffer;
  delete[] int_buffer;
}

/// Read a parquet file by ParquetReader & get valid intervals
/// The first column_num labels are concerned.
std::vector<std::pair<int, int> > read_parquet_file_and_get_valid_intervals(
    const char* parquet_filename, const int row_num, const int tot_label_num,
    const int tested_label_num, int32_t repeated_nums[][MAX_DECODED_NUM],
    bool repeated_values[][MAX_DECODED_NUM], int32_t* true_num, int32_t* false_num,
    int32_t* length, const std::function<bool(bool*, int)>& IsValid) {
  std::cout << "Reading a parquet file: " << parquet_filename << std::endl
            << "row_num = " << row_num << ", tot_label_num = " << tot_label_num
            << ", tested_label_num = " << tested_label_num << std::endl;

  // !!! The id column is only used for debug
  // Allocate space to save the index
  // int64_t* index_value = new int64_t[row_num];

  // Initialize the global variables for save labels
  memset(true_num, 0, tested_label_num * sizeof(int32_t));
  memset(false_num, 0, tested_label_num * sizeof(int32_t));
  memset(length, 0, tested_label_num * sizeof(int32_t));

  // Create a ParquetReader instance
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(parquet_filename, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // Initialize the column row counts
  std::vector<int> col_row_counts(num_columns, 0);

  // Iterate over all the RowGroups in the file
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    int64_t rows_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;
    int col_id = 0;

    ARROW_UNUSED(rows_read);  // prevent warning in release build

    // Read the label columns
    for (int k = 0; k < tested_label_num; k++) {
      // Get the Column Reader for the Bool column
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
        rows_read = bool_reader->ReadBatch(
            BATCH_SIZE, repeated_nums[col_id] + length[col_id],
            repeated_values[col_id] + length[col_id], true_num[col_id], false_num[col_id],
            length[col_id], &values_read);

        // There are no NULL values in the rows written
        col_row_counts[col_id] += rows_read;
      }
      col_id++;
    }

    /* !!! The id column is only used for debug
    // Get the Column Reader for the Int64 column, which is the id column
    col_id = tot_label_num;  // skip some labels
    column_reader = row_group_reader->Column(col_id);
    parquet::Int64Reader* int64_reader =
        static_cast<parquet::Int64Reader*>(column_reader.get());
    // Read all the rows in the column
    while (int64_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      // values_read contains the number of non-null rows
      rows_read =
          int64_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr,
                                  index_value + col_row_counts[col_id], &values_read);
      // There are no NULL values in the rows written
      col_row_counts[col_id] += rows_read;
    } */
  }

  // !!! The id column is only used for debug
  // delete the allocated space
  // delete[] index_value;

  for (int i = 0; i < num_columns; i++) {
    std::cout << "col_row_counts[" << i << "] = " << col_row_counts[i] << std::endl;
    if (i < tested_label_num) {
      std::cout << "--- true_num[" << i << "] = " << true_num[i] << std::endl;
      std::cout << "--- false_num[" << i << "] = " << false_num[i] << std::endl;
      std::cout << "--- length[" << i << "] = " << length[i] << std::endl;
      // !!! This is only used for debug
      // validate_column(i, row_num, repeated_nums, repeated_values, true_num, false_num,
      //                 length);
    }
  }

  std::cout << "The parquet file is read successfully!" << std::endl << std::endl;
  // return the valid intervals
  return GetValidIntervals(tested_label_num, row_num, repeated_nums, repeated_values,
                           length, IsValid);
}

/// Validate the data read from a column is correct
static inline void validate_column(const int col_id, const int row_num,
                                   int32_t repeated_nums[][MAX_DECODED_NUM],
                                   bool repeated_values[][MAX_DECODED_NUM],
                                   int32_t* true_num, int32_t* false_num,
                                   int32_t* length) {
  int curr = 0, expect_true = 0, expect_false = 0;
  for (int i = 0; i < length[col_id]; i++) {
    for (int j = 0; j < repeated_nums[col_id][i]; j++) {
      bool value = repeated_values[col_id][i];
      bool expect;
      if (col_id == 0) {
        expect = (curr % 2 == 1) ? true : false;
      } else if (col_id == 1) {
        expect = (curr % 10 == 1) ? true : false;
      } else if (col_id == 2) {
        expect = ((curr / 10) % 10 == 0) ? true : false;
      } else {
        expect = ((curr / 100) % 2 == 1) ? true : false;
      }
      if (expect) {
        expect_true++;
      } else {
        expect_false++;
      }
      curr++;
      if (value != expect) {
        std::cout << "Error: value = " << value << ", expect = " << expect << std::endl;
      }
    }
  }
  if (expect_true != true_num[col_id]) {
    std::cout << "Error: expect_true = " << expect_true
              << ", true_num[col_id] = " << true_num[col_id] << std::endl;
  }
  if (expect_false != false_num[col_id]) {
    std::cout << "Error: expect_false = " << expect_false
              << ", false_num[col_id] = " << false_num[col_id] << std::endl;
  }
  if (curr != row_num) {
    std::cout << "Error: curr = " << curr << ", NUM_TOT_ROWS = " << row_num << std::endl;
  }
}
