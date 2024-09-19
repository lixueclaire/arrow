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

#include "arrow/csv/api.h"
#include "arrow/dataset/api.h"
#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

#include <iostream>
#include <cstdlib>

using AsyncGeneratorType =
    arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>;

namespace ds = arrow::dataset;
namespace cp = arrow::compute;

constexpr int BATCH_SIZE = 1024;                     // the batch size

void set_bit(uint64_t* bitmap, uint64_t curr) {
    bitmap[curr >> 6] |= (1ULL << (curr & 0x3f));
}

std::shared_ptr<arrow::Table> ExecutePlanAndCollectAsTable(
    const arrow::compute::ExecContext& exec_context,
    std::shared_ptr<arrow::acero::ExecPlan> plan,
    std::shared_ptr<arrow::Schema> schema, AsyncGeneratorType sink_gen) {
  // translate sink_gen (async) to sink_reader (sync)
  std::shared_ptr<arrow::RecordBatchReader> sink_reader =
      arrow::acero::MakeGeneratorReader(schema, std::move(sink_gen),
                                                 exec_context.memory_pool());

  // validate the ExecPlan
  // RETURN_NOT_ARROW_OK(plan->Validate());
  //  start the ExecPlan
  plan->StartProducing();  // arrow 12.0.0 or later return void, not Status

  // collect sink_reader into a Table
  std::shared_ptr<arrow::Table> response_table;
  response_table = arrow::Table::FromRecordBatchReader(sink_reader.get()).ValueOrDie();

  // stop producing
  plan->StopProducing();
  // plan mark finished
  plan->finished().status();
  return response_table;
}

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::shared_ptr<arrow::Table>& l_table,
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key, const std::vector<std::string>& l_project_names,
    const std::vector<std::string>& r_project_names) {
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
  std::vector<arrow::FieldRef> l_output_fields, r_output_fields;
  for (auto& name : l_project_names) {
    l_output_fields.emplace_back(name.c_str());
  }
  for (auto& name : r_project_names) {
    r_output_fields.emplace_back(name.c_str());
  }

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{l_key},
                                              /*in_right_keys=*/{r_key},
                                              l_output_fields,
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

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::shared_ptr<arrow::Table>& l_table,
    const std::string& r_path_to_file,
    const std::string& l_key, const std::string& r_key, const std::vector<std::string>& l_project_names,
    const std::vector<std::string>& r_project_names) {
  auto l_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(l_table));
  std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
  auto fs = arrow::fs::FileSystemFromUriOrPath(r_path_to_file).ValueOrDie();
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                        fs, {r_path_to_file}, format,
                        arrow::dataset::FileSystemFactoryOptions()).ValueOrDie();
  auto r_dataset = factory->Finish().ValueOrDie();
  // auto r_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(r_table));

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
  std::vector<arrow::FieldRef> l_output_fields, r_output_fields;
  for (auto& name : l_project_names) {
    l_output_fields.emplace_back(name.c_str());
  }
  for (auto& name : r_project_names) {
    r_output_fields.emplace_back(name.c_str());
  }

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{l_key},
                                              /*in_right_keys=*/{r_key},
                                              l_output_fields,
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

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::string& l_path_to_file,
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key, const std::vector<std::string>& l_project_names,
    const std::vector<std::string>& r_project_names) {
  auto r_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(r_table));
  std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
  auto fs = arrow::fs::FileSystemFromUriOrPath(l_path_to_file).ValueOrDie();
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                        fs, {l_path_to_file}, format,
                        arrow::dataset::FileSystemFactoryOptions()).ValueOrDie();
  auto l_dataset = factory->Finish().ValueOrDie();
  // auto r_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(r_table));

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
  std::vector<arrow::FieldRef> l_output_fields, r_output_fields;
  for (auto& name : l_project_names) {
    l_output_fields.emplace_back(name.c_str());
  }
  for (auto& name : r_project_names) {
    r_output_fields.emplace_back(name.c_str());
  }

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{l_key},
                                              /*in_right_keys=*/{r_key},
                                              l_output_fields,
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

void getOffset(std::unique_ptr<parquet::ParquetFileReader>& reader_, const int64_t& vertex_id, int64_t& offset, int64_t& length) {
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

void ReadBitMap(std::unique_ptr<parquet::ParquetFileReader>& reader_, const int64_t& offset, const int64_t& length, uint64_t* bit_map) {
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

std::shared_ptr<arrow::Table> GetMessages(const std::string& path_to_file, const int64_t& vertex_id, 
    const std::string& target_col, const std::string& project_col) {
  std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
  auto fs = arrow::fs::FileSystemFromUriOrPath(path_to_file).ValueOrDie();
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                        fs, {path_to_file}, format,
                        arrow::dataset::FileSystemFactoryOptions()).ValueOrDie();
  auto dataset = factory->Finish().ValueOrDie();
  auto options = std::make_shared<arrow::dataset::ScanOptions>();

  cp::Expression filter_expr = cp::equal(cp::field_ref(target_col), cp::literal(vertex_id));
  options->filter = filter_expr;
  auto scan_builder = dataset->NewScan().ValueOrDie();
  scan_builder->Project({project_col});
  scan_builder->Filter(std::move(filter_expr));
  scan_builder->UseThreads(false);
  auto scanner = scan_builder->Finish().ValueOrDie();
  return scanner->ToTable().ValueOrDie();
}

std::shared_ptr<arrow::Table> SortOperation(const std::shared_ptr<arrow::Table>& table,
    const std::string& first_key, const std::string& second_key) {
  std::vector<arrow::compute::SortKey> sort_keys;
  sort_keys.emplace_back(first_key, arrow::compute::SortOrder::Descending);
  sort_keys.emplace_back(second_key, arrow::compute::SortOrder::Ascending);
  arrow::compute::SortOptions sort_options(sort_keys);
  auto exec_context = arrow::compute::default_exec_context();
  auto plan = arrow::acero::ExecPlan::Make(exec_context).ValueOrDie();
  auto table_source_options =
      arrow::acero::TableSourceNodeOptions{table};
  auto source = arrow::acero::MakeExecNode("table_source", plan.get(),
                                                    {}, table_source_options)
                    .ValueOrDie();
  AsyncGeneratorType sink_gen;
  arrow::acero::MakeExecNode(
        "order_by_sink", plan.get(), {source},
        arrow::acero::OrderBySinkNodeOptions{
            sort_options,
            &sink_gen}).status();
  return ExecutePlanAndCollectAsTable(*exec_context, plan,
                                      table->schema(), sink_gen);
}

/*
std::shared_ptr<arrow::Table> SelectOptimize(const std::string& path_to_file, uint64_t* bit_map) {
  arrow::Int64Builder _id_builder;
  arrow::StringBuilder _first_name_builder;
  arrow::StringBuilder _last_name_builder;
  arrow::Int16Builder _index_builder;

  int index = 0, count = 0;  
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("id");
  int col_id2 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id3 = file_metadata->schema()->ColumnIndex("lastName");
  int64_t value;
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;

    // Read the label column
    // Get the Column Reader for the ByteArray column
    column_reader = row_group_reader->Column(col_id);
    // parquet::ByteArrayReader* string_reader =
    //     static_cast<parquet::ByteArrayReader*>(column_reader.get());
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    auto first_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id2));
    auto last_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id3));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    int64_t last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        _index_builder.Append(index);
        id_reader->Skip(i - last_i);
        first_name_reader->Skip(i - last_i);
        last_name_reader->Skip(i - last_i);
        id_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _id_builder.Append(value);
        first_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _first_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _last_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_i = i + 1;
        count++;
      }
      index++;
    }
  }

  delete[] byte_value;

  std::shared_ptr<arrow::Array> _id_array, _first_name_array, _last_name_array, _index_array;
  _id_builder.Finish(&_id_array);
  _first_name_builder.Finish(&_first_name_array);
  _last_name_builder.Finish(&_last_name_array);
  _index_builder.Finish(&_index_array);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("personId", arrow::int64()),
      arrow::field("firstName", arrow::utf8()),
      arrow::field("lastName", arrow::utf8()),
      arrow::field("index", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  return arrow::Table::Make(schema, {_id_array, _first_name_array, _last_name_array, _index_array});
}
*/

std::shared_ptr<arrow::Table> SelectOptimize(const std::string& path_to_file, uint64_t* bit_map) {
  arrow::Int64Builder _id_builder, _creation_date_builder, _person_id_builder;
  arrow::StringBuilder _content_builder, _first_name_builder, _last_name_builder;

  int index = 0, count = 0;  
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("id");
  int col_id2 = file_metadata->schema()->ColumnIndex("content");
  int col_id3 = file_metadata->schema()->ColumnIndex("creationDate");
  int col_id4 = file_metadata->schema()->ColumnIndex("personId");
  int col_id5 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id6 = file_metadata->schema()->ColumnIndex("secondName");
  int64_t value;
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;
    std::shared_ptr<parquet::ColumnReader> column_reader;

    // Read the label column
    // Get the Column Reader for the ByteArray column
    column_reader = row_group_reader->Column(col_id);
    // parquet::ByteArrayReader* string_reader =
    //     static_cast<parquet::ByteArrayReader*>(column_reader.get());
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    auto content_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id2));
    auto creation_date_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id3));
    auto person_id_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id4));
    auto first_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id5));
    auto last_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id6));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    int64_t last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        id_reader->Skip(i - last_i);
        content_reader->Skip(i - last_i);
        person_id_reader->Skip(i - last_i);
        creation_date_reader->Skip(i - last_i);
        first_name_reader->Skip(i - last_i);
        last_name_reader->Skip(i - last_i);
        id_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _id_builder.Append(value);
        content_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _content_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        person_id_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _person_id_builder.Append(value);
        creation_date_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _creation_date_builder.Append(value);
        first_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _first_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _last_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_i = i + 1;
        count++;
      }
      index++;
    }
  }

  delete[] byte_value;

  std::shared_ptr<arrow::Array> _id_array, _content_array, _person_id_array, _creation_date_array, _first_name_array, _last_name_array;
  _id_builder.Finish(&_id_array);
  _content_builder.Finish(&_content_array);
  _person_id_builder.Finish(&_person_id_array);
  _creation_date_builder.Finish(&_creation_date_array);
  _first_name_builder.Finish(&_first_name_array);
  _last_name_builder.Finish(&_last_name_array);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("Comment.id", arrow::int64()),
      arrow::field("content", arrow::utf8()),
      arrow::field("creationDate", arrow::int64()),
      arrow::field("personId", arrow::int64()),
      arrow::field("firstName", arrow::utf8()),
      arrow::field("lastName", arrow::utf8())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  return arrow::Table::Make(schema, {_id_array, _content_array, _creation_date_array, _person_id_array, _first_name_array, _last_name_array});
}

std::shared_ptr<arrow::Table> Select(const std::unordered_set<int64_t>& ids, const std::string& path_to_table) {
  arrow::Int64Builder _id_builder;
  arrow::StringBuilder _first_name_builder;
  arrow::StringBuilder _last_name_builder;
  arrow::Int64Builder _creation_date_builder;

  std::unordered_set<int64_t> indices;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_table, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();
  std::unordered_map<int64_t, int64_t> index_to_id;

  // char* char_buffer = new char[total_length];
  int64_t *values = new int64_t[BATCH_SIZE];
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
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
          _id_builder.Append(values[i]);
        }
        index++;
      }
    }
  }


  int col_id2 = file_metadata->schema()->ColumnIndex("firstName");
  int col_id3 = file_metadata->schema()->ColumnIndex("lastName");
  int col_id4 = file_metadata->schema()->ColumnIndex("creationDate");
  int64_t value;
  index = 0;
  for (int rg = 0; rg < row_group_count; ++rg) {
    // Get the RowGroup Reader
    std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(rg);

    int64_t values_read = 0;

    auto first_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id2));
    auto last_name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(row_group_reader->Column(col_id3));
    auto creation_date_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id4));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    int64_t last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if (indices.find(index) != indices.end()) {
        first_name_reader->Skip(i - last_i);
        last_name_reader->Skip(i - last_i);
        creation_date_reader->Skip(i - last_i);
        first_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _first_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        last_name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        _last_name_builder.Append(std::string((char*)byte_value[0].ptr, byte_value[0].len));
        creation_date_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        _creation_date_builder.Append(value);
        last_i = i + 1;
        count++;
      }
      index++;
    }
  }

  delete[] byte_value;

  std::shared_ptr<arrow::Array> _id_array, _first_name_array, _last_name_array, _creation_date_array;
  _id_builder.Finish(&_id_array);
  _first_name_builder.Finish(&_first_name_array);
  _last_name_builder.Finish(&_last_name_array);
  _creation_date_builder.Finish(&_creation_date_array);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("personId", arrow::int64()),
      arrow::field("firstName", arrow::utf8()),
      arrow::field("lastName", arrow::utf8()),
      arrow::field("friendshipCreationDate", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  return arrow::Table::Make(schema, {_id_array, _first_name_array, _last_name_array, _creation_date_array});
}

void IC8_GRAPHAR(
  int64_t vertex_id,
  int64_t vertex_num_post,
  int64_t vertex_num_comment,
  int64_t vertex_num_person, 
  const std::string& post_has_creator_person_path_file,
  const std::string& comment_has_creator_person_path_file,
  const std::string& comment_replyof_post_path_file,
  const std::string& comment_replyof_comment_path_file,
  const std::string& comment_path_file,
  const std::string& person_path_file) {
  int64_t post_length = vertex_num_post / 64 + 1;
  int64_t comment_length = vertex_num_comment / 64 + 1;
  uint64_t* bit_map_post = new uint64_t[vertex_num_post / 64 + 1];
  uint64_t* bit_map_comment = new uint64_t[vertex_num_comment / 64 + 1];
  uint64_t* bit_map = new uint64_t[vertex_num_comment / 64 + 1];
  memset(bit_map_post, 0, sizeof(uint64_t) * (vertex_num_post / 64 + 1));
  memset(bit_map_comment, 0, sizeof(uint64_t) * (vertex_num_comment / 64 + 1));
  memset(bit_map, 0, sizeof(uint64_t) * (vertex_num_comment / 64 + 1));
  int64_t offset = 0, length = 0;
  auto run_start = clock();
  getOffset(post_has_creator_person_path_file + "-offset", vertex_id, offset, length);
  ReadBitMap(post_has_creator_person_path_file + "-delta", offset, length, bit_map_post);
  getOffset(comment_has_creator_person_path_file + "-offset", vertex_id, offset, length);
  ReadBitMap(comment_has_creator_person_path_file + "-delta", offset, length, bit_map_comment);
  // auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  // std::cout << "read time: " << run_time << " ms" << std::endl;

  // run_start = clock();
  uint64_t bit_map_post_i;
  int64_t index;
  int bit_offset;
  std::unique_ptr<parquet::ParquetFileReader> post_offset_reader =
      parquet::ParquetFileReader::OpenFile(comment_replyof_post_path_file + "-offset");
  std::unique_ptr<parquet::ParquetFileReader> post_delta_reader =
      parquet::ParquetFileReader::OpenFile(comment_replyof_post_path_file + "-delta");
  for (int64_t i = 0; i < post_length; ++i) {
    bit_map_post_i = bit_map_post[i];
    index = i * 64;
    while (bit_map_post_i) {
      bit_offset = __builtin_ctzll(bit_map_post_i);
      index += bit_offset;
      getOffset(post_offset_reader, index, offset, length);
      if (length > 0) {
        ReadBitMap(post_delta_reader, offset, length, bit_map);
      }
      bit_map_post_i = (bit_map_post_i >> bit_offset) >> 1;
      ++index;
    }
  }
  // run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  // std::cout << "post time: " << run_time << " ms" << std::endl;

  // run_start = clock();
  std::unique_ptr<parquet::ParquetFileReader> comment_offset_reader =
      parquet::ParquetFileReader::OpenFile(comment_replyof_comment_path_file + "-offset");
  std::unique_ptr<parquet::ParquetFileReader> comment_delta_reader =
      parquet::ParquetFileReader::OpenFile(comment_replyof_comment_path_file + "-delta");
  uint64_t bit_map_comment_i;
  for (int64_t i = 0; i < comment_length; ++i) {
    bit_map_comment_i = bit_map_comment[i];
    index = i * 64;
    while (bit_map_comment_i) {
      bit_offset = __builtin_ctzll(bit_map_comment_i);
      index += bit_offset;
      getOffset(comment_offset_reader, index, offset, length);
      if (length > 0) {
        ReadBitMap(comment_delta_reader, offset, length, bit_map);
      }
      bit_map_comment_i = (bit_map_comment_i >> bit_offset) >> 1;
      ++index;
    }
  }
  // run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  // std::cout << "comment time: " << run_time << " ms" << std::endl;
  // run_start = clock();
  auto comment_property_table = SelectOptimize(comment_path_file, bit_map);
  // std::cout << "comment row num: " << comment_property_table->num_rows() << std::endl; 
  // run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  // std::cout << "select time: " << run_time << " ms" << std::endl;
  // run_start = clock();
  auto sorted_table = SortOperation(comment_property_table, "creationDate", "Comment.id")->Slice(0, 20);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "run time: " << run_time << " ms" << std::endl;
  // auto person_property_table = SelectOptimize(person_path_file, bit_map_person);
  // std::cout << "person row num: " << person_property_table->num_rows() << std::endl;
  
  // auto result = DoHashJoin(person_property_table, comment_property_table, "index", "Person.Index", {"personId", "firstName", "lastName"}, {"Comment.id", "creationDate", "content"});
  // std::cout << "row num: " << result->num_rows() << std::endl;
  // std::cout << "table: " << result->ToString() << std::endl;
  // Can optimize as select
  // std::unordered_map<int64_t, int64_t> person_to_comment;
  // for (int64_t i = 0; i < vertex_num_comment; ++i) {
  //   if ((bit_map[i >> 6] & (1UL << (i & 63)))) {
  //     ReadBitMap(comment_has_creator_person_path_file + "-delta", i, 1, bit_map_person);
  //   }
  // }

  // auto comment_property_table = SelectOptimize(comment_path_file, bit_map);
  // auto person_property_table = SelectOptimize(person_path_file, person_to_comment, bit_map_person);
}

void writeToCsv(const std::shared_ptr<arrow::Table>& table, const std::string& path_to_file) {
  std::shared_ptr<arrow::io::OutputStream> output = arrow::io::FileOutputStream::Open(path_to_file).ValueOrDie();
  auto write_options = arrow::csv::WriteOptions::Defaults();
  write_options.include_header = false;
  arrow::csv::WriteCSV(*table, write_options, output.get());
  return;
}

void IC8_ACERO(
  int64_t vertex_id,
  const std::string& post_has_creator_person_path_file,
  const std::string& comment_has_creator_person_path_file,
  const std::string& comment_replyof_post_path_file,
  const std::string& comment_replyof_comment_path_file,
  const std::string& comment_path_file,
  const std::string& person_path_file) {
  auto post_table = GetMessages(post_has_creator_person_path_file + ".parquet", vertex_id, "Person.id", "Post.id");
  auto comment_table = GetMessages(comment_has_creator_person_path_file + ".parquet", vertex_id, "Person.id", "Comment.id");

  auto comment_table_1 = DoHashJoin(post_table, comment_replyof_post_path_file + ".parquet", "Post.id", "Post.id", {}, {"Comment.id"});
  // std::cout << "comment_table_1 row num: " << comment_table_1->num_rows() << std::endl;
  auto comment_table_2 = DoHashJoin(comment_table, comment_replyof_comment_path_file + ".parquet", "Comment.id", "Comment.id2", {}, {"Comment.id"});
  // std::cout << "comment_table_2 row num: " << comment_table_2->num_rows() << std::endl;

  auto all_comment_table = arrow::ConcatenateTables({comment_table_1, comment_table_2}).ValueOrDie();

  auto person_table = DoHashJoin(all_comment_table, comment_has_creator_person_path_file + ".parquet", "Comment.id", "Comment.id", {}, {"Comment.id", "Person.id"});
  // std::cout << "person_table row num: " << person_table->num_rows() << std::endl;

  // Or Select
  // auto comment_property_table = DoHashJoin(all_comment_table, comment_path_file, "Comment.id", "id", {"Comment.id"}, {"creationDate", "content"});
  auto comment_property_table = DoHashJoin(comment_path_file, all_comment_table, "id", "Comment.id", {"creationDate", "content"}, {"Comment.id"});
  // std::cout << "comment_property_table row num: " << comment_property_table->num_rows() << std::endl;

  // Or Select
  auto person_property_table = DoHashJoin(person_table, person_path_file, "Person.id", "id", {"Comment.id"}, {"id", "firstName", "lastName"});
  // std::cout << "person_property_table row num: " << person_property_table->num_rows() << std::endl;

  auto result = DoHashJoin(person_property_table, comment_property_table, "Comment.id", "Comment.id", {"id", "firstName", "lastName"}, {"Comment.id", "creationDate", "content"});
  // std::cout << "row num: " << result->num_rows() << std::endl;

  auto sorted_table = SortOperation(result, "creationDate", "Comment.id")->Slice(0, 20);
  // writeToCsv(sorted_table, "result.csv");

  return;
} 

int main(int argc, char** argv) {
  if (argc < 2) {
    // Fake success for CI purposes.
    return EXIT_SUCCESS;
  }

  std::string post_has_creator_person_path_file = argv[1];
  std::string comment_has_creator_person_path_file = argv[2];
  std::string comment_replyof_post_path_file = argv[3];
  std::string comment_replyof_comment_path_file = argv[4];
  std::string comment_path_file = argv[5];
  std::string person_path_file = argv[6];
  int64_t vertex_id = std::stol(argv[7]);
  int64_t vertex_num_post = std::stol(argv[8]);
  int64_t vertex_num_comment = std::stol(argv[9]);
  int64_t vertex_num_person = std::stol(argv[10]);
  std::string type = argv[11];
  if (type == "acero") {
    auto run_start = clock();
    IC8_ACERO(vertex_id, post_has_creator_person_path_file, comment_has_creator_person_path_file, comment_replyof_post_path_file, comment_replyof_comment_path_file, comment_path_file, person_path_file);
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "run time: " << run_time << "ms" << std::endl;
  } else {
    IC8_GRAPHAR(vertex_id, vertex_num_post, vertex_num_comment, vertex_num_person, post_has_creator_person_path_file, comment_has_creator_person_path_file, comment_replyof_post_path_file, comment_replyof_comment_path_file, comment_path_file, person_path_file);
  }
  // CheckCorrectness(path_to_file, vertex_num, vertex_id);
  // return 0;
  // if (!status.ok()) {
  //  std::cerr << "Error occurred: " << status.message() << std::endl;
  //  return EXIT_FAILURE;
  // }
  return EXIT_SUCCESS;
}
