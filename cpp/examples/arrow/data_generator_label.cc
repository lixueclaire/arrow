/** Copyright 2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <optional>

#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/stl.h"
#include "arrow/util/uri.h"
#include "arrow/util/logging.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"

#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

using AsyncGeneratorType =
    arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>;

static constexpr const char* kVertexIndexCol = "_graphArVertexIndex";
static constexpr const char* kSrcIndexCol = "_graphArSrcIndex";
static constexpr const char* kDstIndexCol = "_graphArDstIndex";
static constexpr const char* kOffsetCol = "_graphArOffset";

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

arrow::Status WriteToParquet(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  // Choose compression
  std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->build();
  // std::shared_ptr<WriterProperties> props =
  //     WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->build();

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

std::shared_ptr<arrow::Table> read_csv_to_arrow_table(
    const std::string& csv_file, bool is_weighted, std::string& delemiter, int ignore_rows = 0) {
  arrow::io::IOContext io_context = arrow::io::default_io_context();

  auto fs = arrow::fs::FileSystemFromUriOrPath(csv_file).ValueOrDie();
  std::shared_ptr<arrow::io::InputStream> input =
      fs->OpenInputStream(csv_file).ValueOrDie();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  read_options.skip_rows = ignore_rows;
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  if (delemiter == "tab") {
    parse_options.delimiter = '\t';
  } else if (delemiter == "comma") {
    parse_options.delimiter = ',';
  } else if  (delemiter == "space") {
    parse_options.delimiter = ' ';
  } else {
    parse_options.delimiter = '|'; 
  }
  auto convert_options = arrow::csv::ConvertOptions::Defaults();
  /*
  if (is_weighted) {
    read_options.column_names = {"src", "dst", "weight"};
  } else {
    read_options.column_names = {"src", "dst"};
  }
  */

  // Instantiate TableReader from input stream and options
  auto maybe_reader = arrow::csv::TableReader::Make(
      io_context, input, read_options, parse_options, convert_options);
  std::shared_ptr<arrow::csv::TableReader> reader = *maybe_reader;

  // Read table from CSV file
  auto maybe_table = reader->Read();
  std::shared_ptr<arrow::Table> table = *maybe_table;
  return table;
}

std::shared_ptr<arrow::Table> read_vertex_csv_to_arrow_table(
    const std::string& csv_file, std::string& delemiter, int ignore_rows = 0) {
  arrow::io::IOContext io_context = arrow::io::default_io_context();

  auto fs = arrow::fs::FileSystemFromUriOrPath(csv_file).ValueOrDie();
  std::shared_ptr<arrow::io::InputStream> input =
      fs->OpenInputStream(csv_file).ValueOrDie();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  read_options.skip_rows = ignore_rows;
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  if (delemiter == "tab") {
    parse_options.delimiter = '\t';
  } else if (delemiter == "comma") {
    parse_options.delimiter = ',';
  } else if  (delemiter == "space") {
    parse_options.delimiter = ' ';
  } else {
    parse_options.delimiter = '|'; 
  }
  auto convert_options = arrow::csv::ConvertOptions::Defaults();
  // read_options.column_names = {"id"};
  // read_options.skip_rows = 2;

  // Instantiate TableReader from input stream and options
  auto maybe_reader = arrow::csv::TableReader::Make(
      io_context, input, read_options, parse_options, convert_options);
  std::shared_ptr<arrow::csv::TableReader> reader = *maybe_reader;

  // Read table from CSV file
  auto maybe_table = reader->Read();
  std::shared_ptr<arrow::Table> table = *maybe_table;
  // DCHECK_OK(WriteToParquet(table, csv_file + ".parquet"));
  return table;
}

std::shared_ptr<arrow::Table> read_parquet_to_table(
    const std::string& path_to_file) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = arrow::io::ReadableFile::Open(path_to_file).ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  arrow_reader->ReadTable(&table);
  return table;
}

std::shared_ptr<arrow::Table> convert_to_undirected(
    const std::shared_ptr<arrow::Table>& table) {
  auto reverse_table = table->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();
  auto new_table = arrow::ConcatenateTables({table, reverse_table}).ValueOrDie();
  return new_table;
}

void writeToCsv(const std::shared_ptr<arrow::Table>& table, const std::string& path_to_file) {
  std::shared_ptr<arrow::io::OutputStream> output = arrow::io::FileOutputStream::Open(path_to_file).ValueOrDie();
  auto write_options = arrow::csv::WriteOptions::Defaults();
  write_options.include_header = true;
  write_options.delimiter = ' ';
  DCHECK_OK(arrow::csv::WriteCSV(*table, write_options, output.get()));
  return;
}

arrow::Status WriteToFile(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  std::cout << "WriteToFile, num rows: " << table->num_rows() << std::endl;
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  // Choose compression
  std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding("_graphArSrcIndex", parquet::Encoding::DELTA_BINARY_PACKED)->encoding("_graphArDstIndex", parquet::Encoding::DELTA_BINARY_PACKED_FOR_BIT_MAP)->build();
  // std::shared_ptr<WriterProperties> props =
  //      WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding(parquet::Encoding::DELTA_BINARY_PACKED_FOR_BIT_MAP)->build();
  // std::shared_ptr<WriterProperties> props =
  //     WriterProperties::Builder().build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<ArrowWriterProperties> arrow_props =
      ArrowWriterProperties::Builder().build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path_to_file));

  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table.get(),
                                                 arrow::default_memory_pool(), outfile,
                                                 /*chunk_size=*/16 * 1024, props, arrow_props));
  return arrow::Status::OK();
}

arrow::Status WriteToFileBaseLine(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  std::cout << "WriteToFileBaseLine, num rows: " << table->num_rows() << std::endl;
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  // Choose compression
  // std::shared_ptr<WriterProperties> props =
  //       WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding("_graphArSrcIndex", parquet::Encoding::PLAIN)->encoding("_graphArDstIndex", parquet::Encoding::PLAIN)->build();
  std::shared_ptr<WriterProperties> props =
      WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<ArrowWriterProperties> arrow_props =
      ArrowWriterProperties::Builder().build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path_to_file));

  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table.get(),
                                                 arrow::default_memory_pool(), outfile,
                                                 /*chunk_size=*/1024, props, arrow_props));
  return arrow::Status::OK();
}

arrow::Status WriteOffsetToFile(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;
  std::cout << "WriteOffsetToFile, num rows: " << table->num_rows() << std::endl;

  // Choose compression
  // std::shared_ptr<WriterProperties> props =
  //     WriterProperties::Builder().encoding(parquet::Encoding::DELTA_BINARY_PACKED)->build();
  std::shared_ptr<WriterProperties> props =
      WriterProperties::Builder().build();

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

std::shared_ptr<arrow::Table> add_index_column(
    const std::shared_ptr<arrow::Table>& table, const std::string& col_name) {
  // arrow::Int64Builder index_builder;
  // Get the number of rows in the table
  int64_t num_rows = table->num_rows();
  // Create an array containing the row numbers
  arrow::Int64Builder builder;
  for (int64_t i = 0; i < num_rows; i++) {
    DCHECK_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> row_numbers;
  DCHECK_OK(builder.Finish(&row_numbers));
  // std::shard_ptr<arrow::Field> field = std::make_shared<arrow::Field>(
  //     "index", arrow::int64(), /*nullable=*/false);
  // std::string col_name = is_src ? kSrcIndexCol : kDstIndexCol ;
  auto new_field = arrow::field(col_name, arrow::int64());
  auto chunked_array = arrow::ChunkedArray::Make({row_numbers}).ValueOrDie();
  // Create a new table with the row numbers column
  auto new_table = table->AddColumn(0, new_field, chunked_array).ValueOrDie();
  return new_table;
}

std::shared_ptr<arrow::Table> DoHashJoin(
    const std::shared_ptr<arrow::Table>& l_table,
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key) {
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
  std::vector<arrow::FieldRef> r_output_fields;
  for (auto& field : r_schema->fields()) {
    if (field->name() != r_key) {
      r_output_fields.push_back(field->name());
    }
  }

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{"id"},
                                              /*in_right_keys=*/{r_key},
                                              {l_key},
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
    const std::shared_ptr<arrow::Table>& r_table,
    const std::string& l_key, const std::string& r_key,
    const std::vector<std::string>& l_project_names, 
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
  std::vector<arrow::FieldRef> l_output_fileds, r_output_fields;
  for (auto& field : l_project_names) {
    l_output_fileds.push_back(field);
  }
  for (auto& field : r_project_names) {
    r_output_fields.push_back(field);
  }

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  arrow::acero::Declaration left{"scan", std::move(l_scan_node_options)};
  arrow::acero::Declaration right{"scan", std::move(r_scan_node_options)};

  arrow::acero::HashJoinNodeOptions join_opts{arrow::acero::JoinType::INNER,
                                              /*in_left_keys=*/{l_key},
                                              /*in_right_keys=*/{r_key},
                                              l_output_fileds,
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

std::shared_ptr<arrow::Table> SortKeys(
  const std::shared_ptr<arrow::Table>& input_table) {
  std::vector<arrow::compute::SortKey> sort_keys;
  sort_keys.emplace_back(kSrcIndexCol, arrow::compute::SortOrder::Ascending);
  sort_keys.emplace_back(kDstIndexCol, arrow::compute::SortOrder::Ascending);
  arrow::compute::SortOptions options(sort_keys);

  auto exec_context = arrow::compute::default_exec_context();
  auto plan = arrow::acero::ExecPlan::Make(exec_context).ValueOrDie();
  auto table_source_options =
      arrow::acero::TableSourceNodeOptions{input_table};
  auto source = arrow::acero::MakeExecNode("table_source", plan.get(),
                                                    {}, table_source_options)
                    .ValueOrDie();
  AsyncGeneratorType sink_gen;
  DCHECK_OK(arrow::acero::MakeExecNode(
        "order_by_sink", plan.get(), {source},
        arrow::acero::OrderBySinkNodeOptions{
            options,
            &sink_gen}).status());
  return ExecutePlanAndCollectAsTable(*exec_context, plan,
                                      input_table->schema(), sink_gen);
}

std::shared_ptr<arrow::Table> SortKeys2(
  const std::shared_ptr<arrow::Table>& input_table) {
  std::vector<arrow::compute::SortKey> sort_keys;
  sort_keys.emplace_back(kDstIndexCol, arrow::compute::SortOrder::Ascending);
  arrow::compute::SortOptions options(sort_keys);

  auto exec_context = arrow::compute::default_exec_context();
  auto plan = arrow::acero::ExecPlan::Make(exec_context).ValueOrDie();
  auto table_source_options =
      arrow::acero::TableSourceNodeOptions{input_table};
  auto source = arrow::acero::MakeExecNode("table_source", plan.get(),
                                                    {}, table_source_options)
                    .ValueOrDie();
  AsyncGeneratorType sink_gen;
  DCHECK_OK(arrow::acero::MakeExecNode(
        "order_by_sink", plan.get(), {source},
        arrow::acero::OrderBySinkNodeOptions{
            options,
            &sink_gen}).status());
  return ExecutePlanAndCollectAsTable(*exec_context, plan,
                                      input_table->schema(), sink_gen);
}

std::shared_ptr<arrow::Table> getOffsetTable(
    const std::shared_ptr<arrow::Table>& input_table,
    const std::string& column_name, int64_t vertex_num) {
  std::shared_ptr<arrow::ChunkedArray> column =
      input_table->GetColumnByName(column_name);
  int64_t array_index = 0, index = 0;
  auto ids =
      std::static_pointer_cast<arrow::Int64Array>(column->chunk(array_index));

  arrow::Int64Builder builder;
  DCHECK_OK(builder.Append(0));
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  auto schema = arrow::schema(
      // {arrow::field("x", arrow::int64()), arrow::field("y", arrow::int32())});
      {arrow::field(kOffsetCol, arrow::int64())});

  int64_t global_index = 0, pre_global_index = 0;
  int64_t max = 0, max_i = 0, max_pre_global_index = 0;
  for (int64_t i = 0; i < vertex_num; i++) {
    while (true) {
      if (array_index >= column->num_chunks())
        break;
      if (index >= ids->length()) {
        array_index++;
        if (array_index == column->num_chunks())
          break;
        ids = std::static_pointer_cast<arrow::Int64Array>(
            column->chunk(array_index));
        index = 0;
      }
      if (ids->IsNull(index) || !ids->IsValid(index)) {
        index++;
        global_index++;
        continue;
      }
      int64_t x = ids->Value(index);
      if (x <= i) {
        index++;
        global_index++;
      } else {
        break;
      }
    }
    if (global_index - pre_global_index > max) {
      max_pre_global_index = pre_global_index;
      max = global_index - pre_global_index;
      max_i = i;
    } 
    DCHECK_OK(builder.Append(global_index));
    pre_global_index = global_index;
  }
  std::cout << "max: " << max << ", max_i: " << max_i << ", max_pre_global_index: " << max_pre_global_index << std::endl;

  auto array = builder.Finish().ValueOrDie();
  arrays.push_back(array);
  return arrow::Table::Make(schema, arrays);
}

void write_to_graphar(
    const std::string& path_to_file,
    const std::string& src_vertex_source_file,
    const std::string& edge_source_file,
    const std::string& dst_vertex_source_file,
    const std::string& src_label,
    const std::string& dst_label,
    const std::string& src_file_type,
    const std::string& dst_file_type,
    const std::string& reversed,
    const std::string& to_undirected
    ) {
    // read vertex source to arrow table
    std::string delemiter = "|";
    // read vertex source to arrow table
    std::shared_ptr<arrow::Table> dst_vertex_table;
    if (dst_file_type == "csv") {
      dst_vertex_table = read_vertex_csv_to_arrow_table(dst_vertex_source_file, delemiter)->SelectColumns({0}).ValueOrDie();
    } else {
      dst_vertex_table = read_parquet_to_table(dst_vertex_source_file)->SelectColumns({0}).ValueOrDie();
    }
    // auto dst_vertex_table = read_vertex_csv_to_arrow_table(dst_vertex_source_file, delemiter);
    // std::cout << "schema: " << dst_vertex_table->schema()->ToString() << std::endl;
    auto edge_table = read_csv_to_arrow_table(edge_source_file, false, delemiter, 1)->SelectColumns({0, 1}).ValueOrDie()->RenameColumns({"src", "dst"}).ValueOrDie();
    // auto src_vertex_table = read_vertex_csv_to_arrow_table(src_vertex_source_file, delemiter)->SelectColumns({0}).ValueOrDie();
    auto dst_vertex_table_with_index = add_index_column(dst_vertex_table, kDstIndexCol);
    int64_t vertex_num = dst_vertex_table_with_index->num_rows();
    dst_vertex_table.reset();
    std::cout << "schema: " << dst_vertex_table_with_index->schema()->ToString() << std::endl;
    // auto& vertex_info = graph_info.GetVertexInfo(vertex_label).value();
    auto edge_table_with_dst_index = DoHashJoin(edge_table, dst_vertex_table_with_index, "dst", "id", {"src"}, {kDstIndexCol});
    edge_table.reset();
    std::shared_ptr<arrow::Table> src_vertex_table_with_index;
    if (src_label == dst_label) {
      // rename the column name
      auto old_schema = dst_vertex_table_with_index->schema();
      std::vector<std::string> new_names;
      for (int i = 0; i < old_schema->num_fields(); i++) {
        if (old_schema->field(i)->name() == kDstIndexCol) {
          // Create a new field with the new name and the same data type as the old field
          new_names.push_back(kSrcIndexCol);
        } else {
          // Copy over the old field
          new_names.push_back(old_schema->field(i)->name());
        }
      }
      src_vertex_table_with_index = dst_vertex_table_with_index->RenameColumns(new_names).ValueOrDie();
    } else {
      std::shared_ptr<arrow::Table> src_vertex_table;
      if (src_file_type == "csv") {
        src_vertex_table = read_vertex_csv_to_arrow_table(src_vertex_source_file, delemiter)->SelectColumns({0}).ValueOrDie();
      } else {
        src_vertex_table = read_parquet_to_table(src_vertex_source_file)->SelectColumns({0}).ValueOrDie();
      }
      src_vertex_table_with_index = add_index_column(src_vertex_table, kSrcIndexCol);
      if (reversed != "reverse") {
        vertex_num = src_vertex_table_with_index->num_rows();
      }
    }
    // DCHECK_OK(WriteToParquet(src_vertex_table_with_index, path_to_file + "-vertex"));
    dst_vertex_table_with_index.reset();
    std::shared_ptr<arrow::Table> edge_table_with_src_dst_index;
    if (reversed == "reverse") {
      edge_table_with_src_dst_index = DoHashJoin(edge_table_with_dst_index, src_vertex_table_with_index, "src", "id", {kDstIndexCol}, {kSrcIndexCol})->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();
    } else {
      std::cout << "not reverse" << std::endl;
      edge_table_with_src_dst_index = DoHashJoin(edge_table_with_dst_index, src_vertex_table_with_index, "src", "id", {kDstIndexCol}, {kSrcIndexCol});
    }
    auto first_column_name = edge_table_with_src_dst_index->schema()->field(0)->name();
    if (first_column_name != kSrcIndexCol) {
      edge_table_with_src_dst_index = edge_table_with_src_dst_index->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();
    }
    //->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();
    // auto edge_table_with_src_dst_index = DoHashJoin(src_vertex_table_with_index, edge_table_with_dst_index, kSrcIndexCol, "src");
    if (to_undirected == "to_undirected") {
      edge_table_with_src_dst_index = convert_to_undirected(edge_table_with_src_dst_index);
    }

    auto table = SortKeys(edge_table_with_src_dst_index);
    edge_table_with_src_dst_index.reset();

    DCHECK_OK(WriteToFile(table, path_to_file + "-delta"));
    // DCHECK_OK(WriteToFileBaseLine(table, path_to_file + "-alter-delta"));
    DCHECK_OK(WriteToFileBaseLine(table, path_to_file+"-base"));
    // writeToCsv(table, path_to_file + ".csv");
    auto offset = getOffsetTable(table, kSrcIndexCol, vertex_num);
    table.reset();
    DCHECK_OK(WriteOffsetToFile(offset, path_to_file + "-offset"));

    return;
}

void write_vertex_file(
  const std::string& path_to_file,
  const std::string& vertex_source_file,
  const std::string& person_vertex_file,
  const std::string& edge_source_file
) {
  std::string delemiter = "|";
  auto vertex_table = read_vertex_csv_to_arrow_table(vertex_source_file, delemiter);
  // std::cout << "num rows: " << vertex_table->num_rows() << std::endl;
  auto edge_table = read_csv_to_arrow_table(edge_source_file, false, delemiter, 1)->SelectColumns({0, 1}).ValueOrDie();
  // std::cout << "num rows: " << edge_table->num_rows() << std::endl;
  if (path_to_file.find("person") != std::string::npos) {
    auto person_table = read_vertex_csv_to_arrow_table(person_vertex_file, delemiter)->SelectColumns({0, 1, 2}).ValueOrDie()->RenameColumns({"personId", "firstName", "lastName"}).ValueOrDie();
    auto table = DoHashJoin(person_table, edge_table, "personId", "dst", {"personId", "firstName", "secondName"}, {"src"});
    DCHECK_OK(WriteToFileBaseLine(table, path_to_file));
  } else {
    auto person_table = read_vertex_csv_to_arrow_table(person_vertex_file, delemiter)->SelectColumns({0}).ValueOrDie();
    auto person_table_index = add_index_column(person_table, "personIndex");
    // DCHECK_OK(WriteToFileBaseLine(person_table_index, path_to_file));
    // return;
    auto edge_table_with_index = DoHashJoin(edge_table, person_table_index, "dst", "id", {"src"}, {"personIndex"});
    // std::cout << "num rows: " << edge_table_with_index->num_rows() << std::endl;
    // table = DoHashJoin(vertex_table, table, "id", "src", {"id", "creationDate", "content"}, {"personId", "firstName", "secondName"});
    auto table = DoHashJoin(vertex_table, edge_table_with_index, "id", "src", {"id", "creationDate", "content"}, {"personIndex"});
    // auto table = DoHashJoin(edge_table_with_index, vertex_table, "src", "id", {"personIndex"}, {"id", "creationDate", "content"});
  
    // std::cout << "table: " << table->ToString() << std::endl;
    DCHECK_OK(WriteToFileBaseLine(table, path_to_file));
  }
  return;
}

int main(int argc, char* argv[]) {
   std::string edge_source_file = std::string(argv[1]);
   std::string src_vertex_source_file = std::string(argv[2]);
   std::string dst_vertex_source_file = std::string(argv[3]);
   std::string src_label = std::string(argv[4]); 
   std::string dst_label = std::string(argv[5]);
   std::string src_file_type = std::string(argv[6]);
   std::string dst_file_type = std::string(argv[7]);
   std::string path_to_file = std::string(argv[8]);
   std::string reverse = std::string(argv[9]);
   std::string to_undirected = std::string(argv[10]);
   bool just_vertex = false;
   if (argc > 11) {
     just_vertex = std::string(argv[11]) == "just_vertex";
   }
   if (just_vertex) {
     write_vertex_file(path_to_file, src_vertex_source_file, dst_vertex_source_file, edge_source_file);
   } else {
      write_to_graphar(path_to_file, src_vertex_source_file, edge_source_file, dst_vertex_source_file, src_label, dst_label, src_file_type, dst_file_type, reverse, to_undirected);
   }
  return 0;
}