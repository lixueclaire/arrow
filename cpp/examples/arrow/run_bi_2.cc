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
#include "arrow/util/logging.h"
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
namespace ac = ::arrow::acero;

#define DAY100 8640000000
#define DAY200 17280000000
#define VERTEX_NUM 200000000
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

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::string& l_path_to_file,
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key, const std::vector<std::string>& l_project_names,
    const std::vector<std::string>& r_project_names) {
  std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
  auto fs = arrow::fs::FileSystemFromUriOrPath(l_path_to_file).ValueOrDie();
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                        fs, {l_path_to_file}, format,
                        arrow::dataset::FileSystemFactoryOptions()).ValueOrDie();
  auto l_dataset = factory->Finish().ValueOrDie();
  auto r_dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(r_table));
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

std::shared_ptr<arrow::Table> FilterDateOperation(std::shared_ptr<arrow::Table>& table,
    const std::string& target_col, const std::string& project_col, int64_t date_begin, int64_t date_end) {
  auto dataset = std::dynamic_pointer_cast<arrow::dataset::Dataset>(std::make_shared<arrow::dataset::InMemoryDataset>(table));
  auto options = std::make_shared<arrow::dataset::ScanOptions>();

  std::shared_ptr<arrow::TimestampScalar> lowerBound = std::make_shared<arrow::TimestampScalar>(date_begin * 1000000, arrow::timestamp(arrow::TimeUnit::NANO, "UTC"));
  std::shared_ptr<arrow::TimestampScalar> upperBound = std::make_shared<arrow::TimestampScalar>(date_end * 1000000, arrow::timestamp(arrow::TimeUnit::NANO, "UTC"));
  cp::Expression expr_1 = cp::greater_equal(cp::field_ref(target_col), cp::literal(lowerBound));
  cp::Expression expr_2 = cp::less(cp::field_ref(target_col), cp::literal(upperBound));
  cp::Expression filter_expr = cp::and_(expr_1, expr_2);
  options->filter = filter_expr;
  auto scan_builder = dataset->NewScan().ValueOrDie();
  scan_builder->Project({project_col});
  scan_builder->Filter(std::move(filter_expr));
  scan_builder->UseThreads(false);
  auto scanner = scan_builder->Finish().ValueOrDie();
  return scanner->ToTable().ValueOrDie();
  /*
  options->projection = cp::project({}, {});

  auto scan_node_options = arrow::dataset::ScanNodeOptions{dataset, options};
  std::cout << "Scan node options created" << std::endl;

  ac::Declaration scan{"scan", std::move(scan_node_options)};

  // pipe the scan node into the filter node
  // Need to set the filter in scan node options and filter node options.
  // At scan node it is used for on-disk / push-down filtering.
  // At filter node it is used for in-memory filtering.
  ac::Declaration filter{
      "filter", {std::move(scan)}, ac::FilterNodeOptions(std::move(filter_expr))};

  return ac::DeclarationToTable(std::move(filter)).ValueOrDie();
  */
}

std::shared_ptr<arrow::Table> CountOperation(std::shared_ptr<arrow::Table>& table,
  const std::string& key_col, const std::string& value_col) {
    std::unordered_map<int64_t, int64_t> count_map;
  auto chunked_array = table->GetColumnByName(key_col);
  for (int i = 0; i < chunked_array->num_chunks(); ++i) {
    auto chunk = chunked_array->chunk(i);
    auto chunk_length = chunk->length();
    auto chunk_data = chunk->data();
    auto chunk_data_ptr = chunk_data->GetValues<int64_t>(1);
    for (int j = 0; j < chunk_length; ++j) {
      count_map[chunk_data_ptr[j]]++;
    }
  }
  arrow::Int64Builder key_builder, count_builder;
  for (auto& item : count_map) {
    key_builder.Append(item.first);
    count_builder.Append(item.second);
  }
  std::shared_ptr<arrow::Array> key_array, count_array;
  key_builder.Finish(&key_array);
  count_builder.Finish(&count_array);
  std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field(key_col, arrow::int64()),
                                                        arrow::field(value_col, arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, {key_array, count_array});
}

std::shared_ptr<arrow::Table> CountOperation(std::shared_ptr<arrow::Table>& tag_table,
  std::shared_ptr<arrow::Table>& message_table_1, std::shared_ptr<arrow::Table>& message_table_2) {
  std::unordered_map<int64_t, int64_t> count1_map, count2_map;
  // initialize the count map
  auto chunked_array = tag_table->column(0);
  for (int i = 0; i < chunked_array->num_chunks(); ++i) {
    auto chunk = std::dynamic_pointer_cast<arrow::Int64Array>(chunked_array->chunk(i));
    auto chunk_length = chunk->length();
    for (int j = 0; j < chunk_length; ++j) {
      count1_map[chunk->GetView(j)] = 0;
      count2_map[chunk->GetView(j)] = 0;
    }
  }

  chunked_array = message_table_1->GetColumnByName("Tag.id");
  for (int i = 0; i < chunked_array->num_chunks(); ++i) {
    auto chunk = std::dynamic_pointer_cast<arrow::Int64Array>(chunked_array->chunk(i));
    auto chunk_length = chunk->length();
    for (int j = 0; j < chunk_length; ++j) {
      count1_map[chunk->GetView(j)]++;
    }
  }

  chunked_array = message_table_2->GetColumnByName("Tag.id");
  for (int i = 0; i < chunked_array->num_chunks(); ++i) {
    auto chunk = std::dynamic_pointer_cast<arrow::Int64Array>(chunked_array->chunk(i));
    auto chunk_length = chunk->length();
    for (int j = 0; j < chunk_length; ++j) {
      count2_map[chunk->GetView(j)]++;
    }
  }


  arrow::Int64Builder key_builder, count1_builder, count2_builder, diff_builder;
  for (auto& item : count1_map) {
    key_builder.Append(item.first);
    count1_builder.Append(item.second);
    count2_builder.Append(count2_map.at(item.first));
    diff_builder.Append(std::abs(count2_map.at(item.first) - item.second));
  }
  std::shared_ptr<arrow::Array> key_array, count1_array, count2_array, diff_array;
  key_builder.Finish(&key_array);
  count1_builder.Finish(&count1_array);
  count2_builder.Finish(&count2_array);
  diff_builder.Finish(&diff_array);
  std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("Tag.id", arrow::int64()),
                                                        arrow::field("countWindow1", arrow::int64()),
                                                        arrow::field("countWindow2", arrow::int64()),
                                                        arrow::field("diff", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, {key_array, count1_array, count2_array, diff_array});
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

void FilterDateTime(const std::string& path_to_file, uint64_t* bit_map, int64_t date, uint64_t* bit_map_count_1, uint64_t* bit_map_count_2) {
  int index = 0;
  int64_t begin1 = date * 1000000, end1 = (date + DAY100) * 1000000;
  int64_t begin2 = (date + DAY100) * 1000000, end2 = (date + DAY200) * 1000000;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  // parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("creationDate");
  int64_t value;
  int64_t last_i = 0;
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
    auto date_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        date_reader->Skip(i - last_i);
        date_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        if (value >= begin1 && value < end1) {
          set_bit(bit_map_count_1, index);
        } else if (value >= begin2 && value < end2) {
          set_bit(bit_map_count_2, index);
        }
        last_i = i + 1;
      }
      index++;
    }
  }
  return;
}

void FilterTagAndCount(const std::string& path_to_file, uint64_t* bit_map_tag, uint64_t* bit_map_count_1, uint64_t* bit_map_count_2,
               std::unordered_map<int64_t, int>& tag2count_1, std::unordered_map<int64_t, int>& tag2count_2) {
  int index = 0;
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  // parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("_graphArSrcIndex");
  int col_id2 = file_metadata->schema()->ColumnIndex("_graphArDstIndex");
  int64_t value, value2;
  int64_t last_i = 0;
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
    auto src_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    auto dst_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id2));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      src_reader->ReadBatch(1, nullptr, nullptr, &value2, &values_read);
      if ((bit_map_count_1[value2 >> 6] & (1UL << (value2 & 63)))) {
        dst_reader->Skip(i - last_i);
        dst_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        if ((bit_map_tag[value >> 6] & (1UL << (value & 63)))) {
          tag2count_1[value]++;
        }
        last_i = i + 1;
      } else if ((bit_map_count_2[value2 >> 6] & (1UL << (value2 & 63)))) {
        dst_reader->Skip(i - last_i);
        dst_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
        if ((bit_map_tag[value >> 6] & (1UL << (value & 63)))) {
          tag2count_2[value]++;
        }
        last_i = i + 1;
      }
      index++;
    }
  }
  return;
}

int64_t CountBitMap(uint64_t* bit_map, int64_t num) {
  int64_t count = 0;
  for (int64_t i = 0; i < num; ++i) {
    if ((bit_map[i >> 6] & (1UL << (i & 63)))) {
      count++;
    }
  }
  return count;
}

void GetTagId2Name(const std::string& path_to_file, uint64_t* bit_map, std::unordered_map<int64_t, std::string>& tag_id2_name,
  std::unordered_map<int64_t, int>& tag2count_1, std::unordered_map<int64_t, int>& tag2count_2) {
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  int col_id = file_metadata->schema()->ColumnIndex("name");
  int64_t index = 0, last_i = 0;
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
    auto name_reader = std::static_pointer_cast<parquet::ByteArrayReader>(column_reader);
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    last_i = 0;
    for (int64_t i = 0; i < num_row_rg; i++) {
      if ((bit_map[index >> 6] & (1UL << (index & 63)))) {
        name_reader->Skip(i - last_i);
        name_reader->ReadBatch(1, nullptr, nullptr, byte_value, &values_read);
        tag_id2_name[index] = std::string((char*)byte_value[0].ptr, byte_value[0].len);
        tag2count_1[index] = 0;
        tag2count_2[index] = 0;
        last_i = i + 1;
      }
      ++index;
    }
  }
  delete[] byte_value;
}

void ReadValues(std::unique_ptr<parquet::ParquetFileReader>& reader_, const int64_t& offset, const int64_t& length, int64_t* values) {
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
  auto col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(0));
  col_reader->Skip(remain_offset);
  while (col_reader->HasNext() && total_values_remaining > 0) {
    col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values + (length - total_values_remaining), &values_read);
    total_values_remaining -= values_read;
  }
  while (total_values_remaining > 0) {
    col_reader = std::static_pointer_cast<parquet::Int64Reader>(reader_->RowGroup(i++)->Column(0));
    while (col_reader->HasNext() && total_values_remaining > 0) {
      col_reader->ReadBatch(total_values_remaining, nullptr, nullptr, values + (length - total_values_remaining), &values_read);
      total_values_remaining -= values_read;
    }
  }
 return;
}

/// Set bit in a range
void SetBitmap(uint64_t* bitmap, const int start, const int end) {
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
bool IsValid(bool* state, int column_number) {
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

/// Get the valid intervals of the labels, "column_number" is the number of columns
void GetValidIntervals(const int column_number, int64_t row_number,
                                    int32_t* repeated_nums,
                                    bool* repeated_values,
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
    state[i] = repeated_values[index[i]];
  }
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
  delete[] pos;
  delete[] index;
  delete[] state;
  return;
}

void GetLabelBitMap(
    const std::string& parquet_filename,
    const std::string& label_name,
    int32_t* repeated_nums, bool* repeated_values,
    const std::function<bool(bool*, int)>& IsValid,
    uint64_t* bitmap) {
  // int tested_label_num = label_names.size();
  // Initialize the global variables for save labels
  // int32_t* length = new int32_t[1];
  int32_t length = 0;
  // memset(length, 0, 1 * sizeof(int32_t));

  // Create a ParquetReader instance
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(parquet_filename, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int64_t row_num = file_metadata->num_rows();
  auto schema = file_metadata->schema();
  // std::cout << "schema: " << schema->ToString() << std::endl;

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
    // Get the Column Reader for the Bool column
    int col_id = schema->ColumnIndex(label_name);
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
      rows_read = bool_reader->ReadBatch(BATCH_SIZE, repeated_nums + length,
                                         repeated_values + length, length,
                                         &values_read);
    }
  }

  // std::cout << "The parquet file is read successfully!" << std::endl << std::endl;
  // return the valid count
  GetValidIntervals(1, row_num, repeated_nums, repeated_values,
                    &length, IsValid, bitmap);
  return;
}

void RunIntersection(uint64_t* bit_map_1, uint64_t* bit_map_2, uint64_t* bit_map, int64_t length) {
  for (int64_t i = 0; i < length; ++i) {
    bit_map[i] = bit_map_1[i] & bit_map_2[i];
  } 
}

int InterSectionCount(uint64_t* bit_map_1, uint64_t* bit_map_2, int64_t length, bool clear = false) {
  int count = 0;
  for (int64_t i = 0; i < length; ++i) {
    count += __builtin_popcountll(bit_map_1[i] & bit_map_2[i]);
    if (clear) {
      bit_map_1[i] = 0;
    }
  }
  return count;
}

void writeToCsv(const std::shared_ptr<arrow::Table>& table, const std::string& path_to_file) {
  std::shared_ptr<arrow::io::OutputStream> output = arrow::io::FileOutputStream::Open(path_to_file).ValueOrDie();
  auto write_options = arrow::csv::WriteOptions::Defaults();
  write_options.include_header = true;
  arrow::csv::WriteCSV(*table, write_options, output.get());
  return;
}

arrow::Status writeToParquet(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  std::cout << "WriteToFileBaseLine, num rows: " << table->num_rows() << std::endl;
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  // Choose compression
  // std::shared_ptr<WriterProperties> props =
  //       WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding("src", parquet::Encoding::PLAIN)->encoding("dst", parquet::Encoding::PLAIN)->build();
  std::shared_ptr<WriterProperties> props =
      WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<ArrowWriterProperties> arrow_props =
      ArrowWriterProperties::Builder().build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path_to_file));

  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table.get(),
                                                 arrow::default_memory_pool(), outfile,
                                                 /*chunk_size=*/1024 * 1024, props, arrow_props));
  return arrow::Status::OK();
}

void readParquet(const std::string& path_to_file) {
  int index = 0;  
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path_to_file, false);
  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();

  // char* char_buffer = new char[total_length];
  // int64_t *values = new int64_t[BATCH_SIZE];
  // parquet::ByteArray* byte_values = new parquet::ByteArray[BATCH_SIZE];
  // parquet::ByteArray* byte_value = new parquet::ByteArray[1];
  // memset(values, 0, sizeof(int64_t) * BATCH_SIZE);

  // Iterate over all the RowGroups in the file
  int col_id = file_metadata->schema()->ColumnIndex("creationDate");
  int col_id2 = file_metadata->schema()->ColumnIndex("id");
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
    auto date_reader = std::static_pointer_cast<parquet::Int64Reader>(column_reader);
    auto id_reader = std::static_pointer_cast<parquet::Int64Reader>(row_group_reader->Column(col_id2));
    // Read BATCH_SIZE values at a time. The number of rows read is returned.
    auto num_row_rg = file_metadata->RowGroup(rg)->num_rows();
    for (int64_t i = 0; i < num_row_rg; i++) {
      // check and update results
      int64_t id;
      date_reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
      id_reader->ReadBatch(1, nullptr, nullptr, &id, &values_read);
      std::cout << "id: " << id << ", value: " << value << std::endl;
      index++;
    }
  }
  return;
}

void test_label_filtering(
  const std::string& tag_class,
  int64_t vertex_num_comment,
  const std::string& comment_tagClass_label_path_file) {
  int64_t comment_length = vertex_num_comment / 64 + 1;
  uint64_t* bit_map_comment = new uint64_t[comment_length];
  memset(bit_map_comment, 0, sizeof(uint64_t) * comment_length);
  int32_t* repeated_nums_comment = new int32_t[vertex_num_comment / 2];
  bool* repeated_values_comment = new bool[vertex_num_comment / 2];
  auto start = std::chrono::high_resolution_clock::now();
  GetLabelBitMap(comment_tagClass_label_path_file, tag_class, repeated_nums_comment, repeated_values_comment, IsValid, bit_map_comment);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "run_time: " << duration << std::endl;
  delete[] bit_map_comment;
  delete[] repeated_nums_comment;
  delete[] repeated_values_comment;
  return;
}

static inline void SetBitmap(uint64_t* bitmap, const int64_t index) {
  bitmap[index >> 6] |= (1ULL << (index & 63));
}

int read_parquet_file_and_get_valid_indices(
    const char* parquet_filename, const std::string& label_name,
    std::vector<int>* indices, uint64_t* bitmap) {
  /* std::cout << "Reading a string plain encoded parquet file: " << parquet_filename
            << std::endl
            << "row_num = " << row_num << ", tot_label_num = " << tot_label_num
            << ", tested_label_num = " << tested_label_num << std::endl; */

  // Create a ParquetReader instance
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(parquet_filename, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  int row_group_count = file_metadata->num_row_groups();
  int num_columns = file_metadata->num_columns();

  // Initialize the column row counts
  std::vector<int> col_row_counts(num_columns, 0);

  // char* char_buffer = new char[total_length];
  parquet::ByteArray* value = new parquet::ByteArray[BATCH_SIZE];
  int64_t index = 0;

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

    // Read the label column
    // Get the Column Reader for the ByteArray column
    column_reader = row_group_reader->Column(col_id);
    parquet::ByteArrayReader* string_reader =
        static_cast<parquet::ByteArrayReader*>(column_reader.get());
    // Read all the rows in the column
    while (string_reader->HasNext()) {
      // Read BATCH_SIZE values at a time. The number of rows read is returned.
      rows_read =
          string_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr, value, &values_read);

      // There are no NULL values in the rows written
      col_row_counts[col_id] += rows_read;

      // check and update results
      for (int i = 0; i < rows_read; i++) {
          if (value[i].len > 0) {
            auto find_ptr =
                strstr((char*)value[i].ptr, label_name.c_str());
            if (find_ptr != nullptr && find_ptr +
                                               label_name.size() -
                                               (char*)value[i].ptr <=
                                           value[i].len) {
              SetBitmap(bitmap, index);
            }
        }
        index++;
      }
    }
  }

  // destroy the allocated space
  // delete[] char_buffer;
  delete[] value;

  // std::cout << "The parquet file is read successfully!" << std::endl << std::endl;
  // return the valid count
  return 0;
}

/// The test using string encoding/decoding for GraphAr labels.
void test_string_filtering(
  const std::string& tag_class,
  int64_t vertex_num_comment,
  const std::string& comment_tagClass_label_path_file) {
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[vertex_num_comment / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (vertex_num_comment / 64 + 1));
  int count;

  // test getting bitmap
    auto start = std::chrono::high_resolution_clock::now();
    count = read_parquet_file_and_get_valid_indices(
      comment_tagClass_label_path_file.c_str(), tag_class, &indices, bitmap);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << duration
              << " ms.\n"
              << std::endl;
  delete[] bitmap;
}

void test_graphar(
  int64_t tag_class_id,
  int64_t date,
  const std::string& tag_class,
  int64_t vertex_num_tag,
  int64_t vertex_num_post,
  int64_t vertex_num_comment,
  const std::string& tag_hasType_tagClass_path_file,
  const std::string& post_hasTag_tag_path_file,
  const std::string& comment_hasTag_tag_path_file,
  const std::string& post_tagClass_label_path_file,
  const std::string& comment_tagClass_label_path_file,
  const std::string& comment_path_file,
  const std::string& post_path_file,
  const std::string& tag_path_file) {
  int64_t post_length = vertex_num_post / 64 + 1;
  int64_t comment_length = vertex_num_comment / 64 + 1;
  int64_t tag_length = vertex_num_tag / 64 + 1;
  uint64_t* bit_map_post_1 = new uint64_t[post_length];
  uint64_t* bit_map_post_2 = new uint64_t[post_length];
  uint64_t* bit_map_post = new uint64_t[post_length];
  uint64_t* bit_map_comment_1 = new uint64_t[comment_length];
  uint64_t* bit_map_comment_2 = new uint64_t[comment_length];
  uint64_t* bit_map_comment = new uint64_t[comment_length];
  uint64_t* bit_map_tag = new uint64_t[tag_length];
  int64_t* tag_values = new int64_t[vertex_num_tag];
  memset(bit_map_post, 0, sizeof(uint64_t) * post_length);
  memset(bit_map_post_1, 0, sizeof(uint64_t) * post_length);
  memset(bit_map_post_2, 0, sizeof(uint64_t) * post_length);
  memset(bit_map_comment, 0, sizeof(uint64_t) * comment_length);
  memset(bit_map_comment_1, 0, sizeof(uint64_t) * comment_length);
  memset(bit_map_comment_2, 0, sizeof(uint64_t) * comment_length);
  memset(bit_map_tag, 0, sizeof(uint64_t) * tag_length);
  memset(tag_values, 0, sizeof(int64_t) * vertex_num_tag);
  std::unordered_map<int64_t, std::string> tag_id_to_name;
  std::unordered_map<int64_t, int> tag2count_1, tag2count_2;
  int32_t* repeated_nums_post = new int32_t[vertex_num_post / 2];
  bool* repeated_values_post = new bool[vertex_num_post / 2];
  int32_t* repeated_nums_comment = new int32_t[vertex_num_comment / 2];
  bool* repeated_values_comment = new bool[vertex_num_comment / 2];

  auto run_start = clock();
  int64_t offset = 0, length = 0;
  getOffset(tag_hasType_tagClass_path_file + "-offset", tag_class_id, offset, length);
  ReadBitMap(tag_hasType_tagClass_path_file + "-delta", offset, length, bit_map_tag);
  GetTagId2Name(tag_path_file, bit_map_tag, tag_id_to_name, tag2count_1, tag2count_2);

  GetLabelBitMap(post_tagClass_label_path_file, tag_class, repeated_nums_post, repeated_values_post, IsValid, bit_map_post);
  GetLabelBitMap(comment_tagClass_label_path_file, tag_class, repeated_nums_comment, repeated_values_comment, IsValid, bit_map_comment);

  // filter date time
  FilterDateTime(post_path_file, bit_map_post, date, bit_map_post_1, bit_map_post_2);
  FilterDateTime(comment_path_file, bit_map_comment, date, bit_map_comment_1, bit_map_comment_2);
  /*
  {
    run_start = clock();
    FilterTagAndCount(post_hasTag_tag_path_file + "-alter-delta", bit_map_tag, bit_map_post_1, bit_map_post_2, tag2count_1, tag2count_2);
    run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "post: " << run_time << std::endl;
    run_start = clock();
    FilterTagAndCount(comment_hasTag_tag_path_file + "-alter-delta", bit_map_tag, bit_map_comment_1, bit_map_comment_2, tag2count_1, tag2count_2);
    run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "comment: " << run_time << std::endl;
  }
  */
  /*
  {
  std::unique_ptr<parquet::ParquetFileReader> post_offset_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "-alter-offset");
  std::unique_ptr<parquet::ParquetFileReader> post_delta_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "-alter-delta");
  for (int64_t i = 0; i < vertex_num_post; ++i) {
    if (bit_map_post_1[i >> 6] & (1UL << (i & 63))) {
      getOffset(post_offset_reader, i, offset, length); 
      ReadValues(post_delta_reader, offset, length, tag_values);
      for (int i = 0; i < length; ++i) {
        int64_t index = tag_values[i]; 
        if ((bit_map_tag[index >> 6] & (1UL << (index & 63)))) {
          tag2count_1[tag_values[i]]++;
        }
      }
    } else if (bit_map_post_2[i >> 6] & (1UL << (i & 63))) {
      getOffset(post_offset_reader, i, offset, length); 
      ReadValues(post_delta_reader, offset, length, tag_values);
      for (int i = 0; i < length; ++i) {
        int64_t index = tag_values[i]; 
        if ((bit_map_tag[index >> 6] & (1UL << (index & 63)))) {
          tag2count_2[tag_values[i]]++;
        }
      }
    }
  }
  }
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "post: " << run_time << std::endl;

  run_start = clock();
  {
  std::unique_ptr<parquet::ParquetFileReader> comment_offset_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "-alter-offset");
  std::unique_ptr<parquet::ParquetFileReader> comment_delta_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "-alter-delta");
  for (int64_t i = 0; i < vertex_num_comment; ++i) {
    if (bit_map_comment_1[i >> 6] & (1UL << (i & 63))) {
      getOffset(comment_offset_reader, i, offset, length); 
      ReadValues(comment_delta_reader, offset, length, tag_values);
      for (int i = 0; i < length; ++i) {
        int64_t index = tag_values[i]; 
        if ((bit_map_tag[index >> 6] & (1UL << (index & 63)))) {
          tag2count_1[tag_values[i]]++;
        }
      }
    } else if (bit_map_comment_2[i >> 6] & (1UL << (i & 63))) {
      getOffset(comment_offset_reader, i, offset, length); 
      ReadValues(comment_delta_reader, offset, length, tag_values);
      for (int i = 0; i < length; ++i) {
        int64_t index = tag_values[i]; 
        if ((bit_map_tag[index >> 6] & (1UL << (index & 63)))) {
          tag2count_2[tag_values[i]]++;
        }
      }
    }
  }
  }
  run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "comment: " << run_time << std::endl;
  */

  {
  memset(bit_map_post, 0, sizeof(uint64_t) * post_length);
  memset(bit_map_comment, 0, sizeof(uint64_t) * comment_length);
  std::unique_ptr<parquet::ParquetFileReader> post_offset_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "_reverse-alter-offset");
  std::unique_ptr<parquet::ParquetFileReader> post_delta_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "_reverse-alter-delta");
  for (int64_t i = 0; i < vertex_num_tag; ++i) {
    if (bit_map_tag[i >> 6] & (1UL << (i & 63))) {
      // memset(bit_map_post, 0, sizeof(uint64_t) * post_length);
      getOffset(post_offset_reader, i, offset, length); 
      ReadBitMap(post_delta_reader, offset, length, bit_map_post);
      tag2count_1[i] += InterSectionCount(bit_map_post, bit_map_post_1, post_length);
      tag2count_2[i] += InterSectionCount(bit_map_post, bit_map_post_2, post_length, true);
    }
  }

  std::unique_ptr<parquet::ParquetFileReader> comment_offset_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "_reverse-alter-offset");
  std::unique_ptr<parquet::ParquetFileReader> comment_delta_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "_reverse-alter-delta");
  for (int64_t i = 0; i < vertex_num_tag; ++i) {
    if (bit_map_tag[i >> 6] & (1UL << (i & 63))) {
      // memset(bit_map_comment, 0, sizeof(uint64_t) * comment_length);
      getOffset(comment_offset_reader, i, offset, length); 
      ReadBitMap(comment_delta_reader, offset, length, bit_map_comment);
      tag2count_1[i] += InterSectionCount(bit_map_comment, bit_map_comment_1, comment_length);
      tag2count_2[i] += InterSectionCount(bit_map_comment, bit_map_comment_2, comment_length, true);
    }
  }
  }
  
  arrow::StringBuilder _tag_name_builder;
  arrow::Int32Builder _count_window_1_builder, _count_window_2_builder, _diff_builder;
  for (auto& tag_id : tag2count_2) {
    DCHECK_OK(_tag_name_builder.Append(tag_id_to_name.at(tag_id.first)));
    DCHECK_OK(_count_window_1_builder.Append(tag2count_1.at(tag_id.first)));
    DCHECK_OK(_count_window_2_builder.Append(tag_id.second));
    DCHECK_OK(_diff_builder.Append(std::abs(tag_id.second - tag2count_1.at(tag_id.first))));
  }
  /*
  std::unique_ptr<parquet::ParquetFileReader> post_offset_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "-offset");
  std::unique_ptr<parquet::ParquetFileReader> post_delta_reader =
      parquet::ParquetFileReader::OpenFile(post_hasTag_tag_path_file + "-delta");
  std::unique_ptr<parquet::ParquetFileReader> comment_offset_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "-offset");
  std::unique_ptr<parquet::ParquetFileReader> comment_delta_reader =
      parquet::ParquetFileReader::OpenFile(comment_hasTag_tag_path_file + "-delta");
  uint64_t bit_map_tag_i;
  int64_t index;
  int bit_offset;
  
  for (int64_t i = 0; i < tag_length; ++i) {
    bit_map_tag_i = bit_map_tag[i];
    index = i * 64;
    while (bit_map_tag_i) {
      bit_offset = __builtin_ctzll(bit_map_tag_i);
      index += bit_offset;
      getOffset(post_offset_reader, index, offset, length);
      int count_1 = 0, count_2 = 0;
      if (length > 0) {
        ReadBitMap(post_delta_reader, offset, length, bit_map_post_2);
        count_post += length;
        // RunIntersection(bit_map_post_1, bit_map_post_2, bit_map_post, post_length);
        // FilterDateTime(post_path_file, bit_map_post, date, count_1, count_2);
        FilterDateTime(post_path_file, bit_map_post_2, date, count_1, count_2);
      }
      getOffset(comment_offset_reader, index, offset, length);
      if (length > 0) {
        ReadBitMap(comment_delta_reader, offset, length, bit_map_comment_2);
        count_comment += length;
        // RunIntersection(bit_map_comment_1, bit_map_comment_2, bit_map_comment, comment_length);
        // FilterDateTime(comment_path_file, bit_map_comment, date, count_1, count_2);
        FilterDateTime(comment_path_file, bit_map_comment_2, date, count_1, count_2);
      }
      if (count_1 > 0)
        filter_count_1 += count_1;
      if (count_2 > 0)
        filter_count_2 += count_2;
      DCHECK_OK(_tag_name_builder.Append(tag_id_to_name[index]));
      DCHECK_OK(_count_window_1_builder.Append(count_1));
      DCHECK_OK(_count_window_2_builder.Append(count_2));
      DCHECK_OK(_diff_builder.Append(std::abs(count_2 - count_1)));
      bit_map_tag_i = (bit_map_tag_i >> bit_offset) >> 1;
      ++index;
    }
  }
  */
  std::shared_ptr<arrow::Array> _tag_name_array, _count_window_1_array, _count_window_2_array, _diff_array;
  DCHECK_OK(_tag_name_builder.Finish(&_tag_name_array));
  DCHECK_OK(_count_window_1_builder.Finish(&_count_window_1_array));
  DCHECK_OK(_count_window_2_builder.Finish(&_count_window_2_array));
  DCHECK_OK(_diff_builder.Finish(&_diff_array));
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("tagName", arrow::utf8()),
      arrow::field("countWindow1", arrow::int32()),
      arrow::field("countWindow2", arrow::int32()),
      arrow::field("diff", arrow::int32())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto table = arrow::Table::Make(schema, {_tag_name_array, _count_window_1_array, _count_window_2_array, _diff_array});
  table = SortOperation(table, "diff", "tagName");
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "run time: " << run_time << "ms" << std::endl;
  // writeToCsv(table, "result.csv");

  delete[] bit_map_post;
  delete[] bit_map_post_1;
  delete[] bit_map_post_2;
  delete[] bit_map_comment;
  delete[] bit_map_comment_1;
  delete[] bit_map_comment_2;
  delete[] bit_map_tag;
  delete[] tag_values;
  delete[] repeated_nums_post;
  delete[] repeated_values_post;
  delete[] repeated_nums_comment;
  delete[] repeated_values_comment;
  return;
}

void test_acero(
  int64_t tag_class_id,
  int64_t date,
  const std::string& tag_hasType_tagClass_path_file,
  const std::string& post_hasTag_tag_path_file,
  const std::string& comment_hasTag_tag_path_file,
  const std::string& comment_path_file,
  const std::string& post_path_file,
  const std::string& tag_path_file) {
 auto tag_table = GetMessages(tag_hasType_tagClass_path_file + ".parquet", tag_class_id, "TagClass.id", "Tag.id");
 auto post_table = DoHashJoin(post_hasTag_tag_path_file + ".parquet", tag_table, "Tag.id", "Tag.id", {"Post.id"}, {"Tag.id"}); 
 auto comment_table = DoHashJoin(comment_hasTag_tag_path_file + ".parquet", tag_table, "Tag.id", "Tag.id", {"Comment.id"}, {"Tag.id"});
 // auto post_property_table = DoHashJoin(post_path_file, post_table, "id", "Post.id", {"creationDate"}, {"Tag.id"});
 // auto comment_property_table = DoHashJoin(comment_path_file, comment_table, "id", "Comment.id", {"creationDate"}, {"Tag.id"});
 auto post_property_table = DoHashJoin(post_table, post_path_file, "Post.id", "id", {"Tag.id"}, {"creationDate"});
 // writeToParquet(post_property_table, "post_property_table.parquet");
 // writeToCsv(post_property_table, "post_property_table.csv");
 // readParquet("/root/dataset/sf1/post_property_table.parquet");
 auto comment_property_table = DoHashJoin(comment_table, comment_path_file, "Comment.id", "id", {"Tag.id"}, {"creationDate"});
 // count windows 1
 auto post_filter_table_1 = FilterDateOperation(post_property_table, "creationDate", "Tag.id", date, date + DAY100);
 auto comment_filter_table_1 = FilterDateOperation(comment_property_table, "creationDate", "Tag.id", date, date + DAY100);
 auto merged_table_1 = arrow::ConcatenateTables({post_filter_table_1, comment_filter_table_1}).ValueOrDie();
 auto post_filter_table_2 = FilterDateOperation(post_property_table, "creationDate", "Tag.id", date + DAY100, date + DAY200);
 auto comment_filter_table_2 = FilterDateOperation(comment_property_table, "creationDate", "Tag.id", date + DAY100, date + DAY200);
 auto merged_table_2 = arrow::ConcatenateTables({post_filter_table_2, comment_filter_table_2}).ValueOrDie();
 auto count_window_table = CountOperation(tag_table, merged_table_1, merged_table_2);
 // auto count_window_1_table = CountOperation(merged_table_1, "Tag.id", "countWindow1");
 // auto count_window_2_table = CountOperation(merged_table_2, "Tag.id", "countWindow2");
 // auto merge_table = DoHashJoin(count_window_1_table, count_window_2_table, "Tag.id", "Tag.id", {"Tag.id", "countWindow1"}, {"countWindow2"});
 auto result = DoHashJoin(count_window_table, tag_path_file, "Tag.id", "id", {"diff", "countWindow1", "countWindow2"}, {"name"});
 auto sorted_table = SortOperation(result, "diff", "name");
 writeToCsv(sorted_table, "result_acero.csv");
 return;
}

void check_schema(const std::string& path) {
  // Create a ParquetReader instance
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(path, false);

  // Get the File MetaData
  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  std::cout << "schema: " << file_metadata->schema()->ToString() << std::endl;
}

int main(int argc, char** argv) {
  std::string tag_hasType_tagClass_path_file = argv[1];
  std::string post_hasTag_tag_path_file = argv[2];
  std::string comment_hasTag_tag_path_file = argv[3];
  std::string post_tagClass_label_path_file = argv[4];
  std::string comment_tagClass_label_path_file = argv[5];
  std::string comment_path_file = argv[6];
  std::string post_path_file = argv[7];
  std::string tag_path_file = argv[8];
  std::string tag_class = argv[9];
  int64_t tag_class_id = std::stol(argv[10]);
  int64_t date = std::stol(argv[11]);
  int64_t vertex_num_tag = std::stol(argv[12]);
  int64_t vertex_num_post = std::stol(argv[13]);
  int64_t vertex_num_comment = std::stol(argv[14]);
  auto run_start = clock();
  test_acero(tag_class_id, date, tag_hasType_tagClass_path_file, post_hasTag_tag_path_file, comment_hasTag_tag_path_file, comment_path_file, post_path_file, tag_path_file);
  auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  std::cout << "run time: " << run_time << "ms" << std::endl;
  test_graphar(tag_class_id, date, tag_class, vertex_num_tag, vertex_num_post, vertex_num_comment, tag_hasType_tagClass_path_file, post_hasTag_tag_path_file, comment_hasTag_tag_path_file, post_tagClass_label_path_file, comment_tagClass_label_path_file, comment_path_file, post_path_file, tag_path_file);
  // IC8_GRAPHAR();
  return 0;
}

/*
int main(int argc, char** argv) {
  std::string comment_tagClass_label_path_file = argv[1];
  std::string tag_class = argv[2];
  int64_t vertex_num_comment = std::stol(argv[3]);
  // auto run_start = clock();
  // test_acero(tag_class_id, date, tag_hasType_tagClass_path_file, post_hasTag_tag_path_file, comment_hasTag_tag_path_file, comment_path_file, post_path_file, tag_path_file);
  // auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
  // std::cout << "run time: " << run_time << "ms" << std::endl;
 test_label_filtering(tag_class, vertex_num_comment, comment_tagClass_label_path_file);
 // test_string_filtering(tag_class, vertex_num_comment, comment_tagClass_label_path_file);
  // IC8_GRAPHAR();
  return 0;
}
*/
