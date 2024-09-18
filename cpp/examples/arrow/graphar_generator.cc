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
#include <filesystem>
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

#include "arrow/dataset/api.h"
#include "arrow/acero/exec_plan.h"
#include "arrow/compute/api.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/plan.h"
#include "arrow/dataset/scanner.h"

namespace ds = arrow::dataset;
namespace cp = arrow::compute;


std::shared_ptr<arrow::Table> CreateVertexTable(int64_t vertex_num) {
  arrow::Int64Builder builder;
  for (int64_t i = 0; i < vertex_num; i++) {
    DCHECK_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> array;
  DCHECK_OK(builder.Finish(&array));
  auto schema = arrow::schema(
      // {arrow::field("x", arrow::int64()), arrow::field("y", arrow::int32())});
      {arrow::field("id", arrow::int64())});
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.push_back(array);
  return arrow::Table::Make(schema, arrays);
}

void TypeLoosen(const std::shared_ptr<arrow::Schema>& schema1,
                std::shared_ptr<arrow::Schema>& schema2) {
  int field_num = schema1->num_fields();
  std::shared_ptr<arrow::KeyValueMetadata> metadata(
      new arrow::KeyValueMetadata());
  if (schema1 != nullptr) {
    if (schema1->metadata() != nullptr) {
      std::unordered_map<std::string, std::string> metakv;
      schema1->metadata()->ToUnorderedMap(&metakv);
      for (auto const& kv : metakv) {
        metadata->Append(kv.first, kv.second);
      }
    } 
  } else {
    throw std::runtime_error("schema1 is null");
  }

  // Perform type lossen.
  // Date32 -> int32
  // Timestamp -> int64 -> double -> utf8   binary (not supported)

  // Timestamp value are stored as as number of seconds, milliseconds,
  // microseconds or nanoseconds since UNIX epoch.
  // CSV reader can only produce timestamp in seconds.
  std::cout << "field_num: " << field_num << std::endl;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int i = 0; i < field_num; ++i) {
    fields.push_back(schema1->field(i));
  }
  std::vector<std::shared_ptr<arrow::Field>> lossen_fields(field_num);

  for (int i = 0; i < field_num; ++i) {
    lossen_fields[i] = fields[i];
    auto res = fields[i]->type();
    if (res == arrow::null()) {
      continue;
    }
    if (res->Equals(arrow::date32())) {
      res = arrow::int32();
    }
    if (res->Equals(arrow::date64())) {
      res = arrow::int64();
    }
    if (res->id() == arrow::Type::TIMESTAMP) {
      res = arrow::int64();
    }
    lossen_fields[i] = lossen_fields[i]->WithType(res);
  }
  schema2 = std::make_shared<arrow::Schema>(lossen_fields, metadata);
  return;
}

void GeneralCast(const std::shared_ptr<arrow::Array>& in,
                   const std::shared_ptr<arrow::DataType>& to_type,
                   std::shared_ptr<arrow::Array>& out) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 1000000
  arrow::compute::FunctionContext ctx;
  arrow::compute::Cast(&ctx, *in, to_type, {}, &out);
#else
  out = arrow::compute::Cast(*in, to_type).ValueOrDie();
#endif
  return;
}

void CastTableToSchema(const std::shared_ptr<arrow::Table>& table,
                         const std::shared_ptr<arrow::Schema>& schema,
                         std::shared_ptr<arrow::Table>& out) {
  if (table->schema()->Equals(schema)) {
    out = table;
    return;
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
  for (int64_t i = 0; i < table->num_columns(); ++i) {
    auto col = table->column(i);
    if (table->field(i)->type()->Equals(schema->field(i)->type())) {
      new_columns.push_back(col);
      continue;
    }
    auto from_type = table->field(i)->type();
    auto to_type = schema->field(i)->type();
    std::vector<std::shared_ptr<arrow::Array>> chunks;
    for (int64_t j = 0; j < col->num_chunks(); ++j) {
      auto array = col->chunk(j);
      std::shared_ptr<arrow::Array> out;
      if (arrow::compute::CanCast(*from_type, *to_type)) {
        GeneralCast(array, to_type, out);
        chunks.push_back(out);
      }
    }
    new_columns.push_back(
        std::make_shared<arrow::ChunkedArray>(chunks, to_type));
  }
  out = arrow::Table::Make(schema, new_columns);
  return;
}

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
        WriterProperties::Builder().build();
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
  // read_options.autogenerate_column_names = true;
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
  read_options.column_names = {"src", "dst", "weight"};
  */

  // Instantiate TableReader from input stream and options
  auto maybe_reader = arrow::csv::TableReader::Make(
      io_context, input, read_options, parse_options, convert_options);
  std::shared_ptr<arrow::csv::TableReader> reader = *maybe_reader;

  // Read table from CSV file
  auto maybe_table = reader->Read();
  std::shared_ptr<arrow::Table> table = *maybe_table;
  std::shared_ptr<arrow::Schema> normalized_schema;
  TypeLoosen(table->schema(), normalized_schema);

  std::shared_ptr<arrow::Table> table_out;
  CastTableToSchema(table, normalized_schema, table_out);
  return table_out;
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
  std::cout << "Read vertex table, num rows: " << table->num_rows() << std::endl;
  std::shared_ptr<arrow::Schema> normalized_schema;
  TypeLoosen(table->schema(), normalized_schema);

  std::shared_ptr<arrow::Table> table_out;
  CastTableToSchema(table, normalized_schema, table_out);
  std::cout << "Loose type: " << table_out->schema()->ToString() << std::endl;
  return table_out;
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
                                                 /*chunk_size=*/1024 * 1024, props, arrow_props));
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
  std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding("_graphArSrcIndex", parquet::Encoding::PLAIN)->encoding("_graphArDstIndex", parquet::Encoding::PLAIN)->build();
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

arrow::Status WriteOffsetToFile(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;
  std::cout << "WriteOffsetToFile, num rows: " << table->num_rows() << std::endl;

  // Choose compression
  // std::shared_ptr<WriterProperties> props =
  //      WriterProperties::Builder().encoding(parquet::Encoding::DELTA_BINARY_PACKED)->build();
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
    const std::shared_ptr<arrow::Table>& table, bool is_src=true) {
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
  std::string col_name = is_src ? kSrcIndexCol : kDstIndexCol ;
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
  sort_keys.emplace_back(kSrcIndexCol, arrow::compute::SortOrder::Ascending);
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
    const std::string& column_name, int64_t vertex_num, int64_t begin) {
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
      if (x <= (i + begin)) {
        index++;
        global_index++;
      } else {
        break;
      }
    }
    DCHECK_OK(builder.Append(global_index));
    pre_global_index = global_index;
  }
  // std::cout << "max: " << max << ", max_i: " << max_i << ", max_pre_global_index: " << max_pre_global_index << std::endl;

  auto array = builder.Finish().ValueOrDie();
  arrays.push_back(array);
  return arrow::Table::Make(schema, arrays);
}

void ProcessVertexTable(const std::string& label,  std::shared_ptr<arrow::Table> table, int64_t vertex_chunk_size) {
  auto path_prefix = "/mnt/ldbc/weibin/ldbc-gar/ur/vertex/" + label;
  if (std::filesystem::exists(path_prefix)) {
    return;
  } else {
    std::cout << "path not exists: " << path_prefix << std::endl;
  }

  auto fs = arrow::fs::FileSystemFromUriOrPath("/mnt/ldbc/weibin/ldbc-gar/ur").ValueOrDie();

  auto schema = table->schema();
  std::string col_names = ""; 
  for (int i = 0; i < schema->num_fields(); i++) {
    col_names += schema->field(i)->name() + "_";
  }
  col_names.pop_back();
  auto num_row = table->num_rows();

  fs->CreateDir(path_prefix + "/" + col_names);
  for (int chunk_i = 0; chunk_i * vertex_chunk_size < num_row; chunk_i++) {
    auto sub_table = table->Slice(chunk_i * vertex_chunk_size, vertex_chunk_size);
    auto path_to_file = path_prefix + "/" + col_names + "/chunk" + std::to_string(chunk_i);
    WriteToParquet(sub_table, path_to_file);
    // writeToCsv(sub_table, path_to_file);
  }
  auto ofstream = fs->OpenOutputStream(path_prefix + "/vertex_count").ValueOrDie();
  ofstream->Write(&num_row, sizeof(int64_t));
  ofstream->Close();
}

std::shared_ptr<arrow::Table> FilterPushDown(std::shared_ptr<arrow::Table> table, int64_t begin, int64_t end) {
  auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);
  auto options = std::make_shared<arrow::dataset::ScanOptions>();

  cp::Expression filter_expr_1 = cp::greater_equal(cp::field_ref("_graphArSrcIndex"), cp::literal(begin));
  cp::Expression filter_expr_2 = cp::less(cp::field_ref("_graphArSrcIndex"), cp::literal(end));
  cp::Expression filter_expr = cp::and_(filter_expr_1, filter_expr_2);
  options->filter = filter_expr;
  auto scan_builder = dataset->NewScan().ValueOrDie();
  scan_builder->Project({"_graphArSrcIndex", "_graphArDstIndex"});
  scan_builder->Filter(std::move(filter_expr));
  scan_builder->UseThreads(false);
  auto scanner = scan_builder->Finish().ValueOrDie();
  return scanner->ToTable().ValueOrDie();
}

std::shared_ptr<arrow::Table> FilterPushDown2(std::shared_ptr<arrow::Table> table, int64_t begin, int64_t end) {
  auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(table);
  auto options = std::make_shared<arrow::dataset::ScanOptions>();

  cp::Expression filter_expr_1 = cp::greater_equal(cp::field_ref("_graphArDstIndex"), cp::literal(begin));
  cp::Expression filter_expr_2 = cp::less(cp::field_ref("_graphArDstIndex"), cp::literal(end));
  cp::Expression filter_expr = cp::and_(filter_expr_1, filter_expr_2);
  options->filter = filter_expr;
  auto scan_builder = dataset->NewScan().ValueOrDie();
  scan_builder->Project({"_graphArSrcIndex", "_graphArDstIndex"});
  scan_builder->Filter(std::move(filter_expr));
  scan_builder->UseThreads(false);
  auto scanner = scan_builder->Finish().ValueOrDie();
  return scanner->ToTable().ValueOrDie();
}

void ProcessAdjTable(std::shared_ptr<arrow::Table> table, int64_t vertex_num, int64_t vertex_chunk_size, int64_t edge_chunk_size, const std::string& src_label, const std::string& dst_label, const std::string& edge_label) {
  auto fs = arrow::fs::FileSystemFromUriOrPath("/mnt/ldbc/weibin/ldbc-gar/ur").ValueOrDie();

  table = table->SelectColumns({0, 1}).ValueOrDie();
  // std::cout << "schema: " << table->schema()->ToString() << "\n" << table->ToString() << std::endl;
  auto path_prefix = "/mnt/ldbc/weibin/ldbc-gar/ur/edge/" + src_label + "_" + edge_label + "_" + dst_label;
  fs->CreateDir(path_prefix + "/ordered_by_source/adj_list");
  for (int chunk_i = 0; chunk_i * vertex_chunk_size < vertex_num; chunk_i++) {
    int64_t begin = chunk_i * vertex_chunk_size;
    int64_t end = chunk_i * vertex_chunk_size + vertex_chunk_size;
    auto sub_table = FilterPushDown(table, begin, end);
    // std::cout << "sub_table : " << sub_table->ToString() << std::endl;
    auto sub_num_row = sub_table->num_rows();
    fs->CreateDir(path_prefix + "/ordered_by_source/adj_list/part" + std::to_string(chunk_i));
    for (int j = 0; j * edge_chunk_size < sub_num_row; j++) {
      auto sub_sub_table = sub_table->Slice(j * edge_chunk_size, edge_chunk_size);
      auto path_to_file = path_prefix + "/ordered_by_source/adj_list/part" + std::to_string(chunk_i) + "/chunk" + std::to_string(j);
      WriteToParquet(sub_sub_table, path_to_file);
      // writeToCsv(sub_sub_table, path_to_file);
    }
    fs->CreateDir(path_prefix + "/ordered_by_source/offset");
    auto offset = getOffsetTable(sub_table, "_graphArSrcIndex", vertex_chunk_size, begin);
    WriteToParquet(offset, path_prefix + "/ordered_by_source/offset/chunk" + std::to_string(chunk_i));
    // writeToCsv(offset, path_prefix + "/ordered_by_source/offset/chunk" + std::to_string(chunk_i));
    auto ofstream = fs->OpenOutputStream(path_prefix + "/ordered_by_source/edge_count" + std::to_string(chunk_i)).ValueOrDie();
    ofstream->Write(&sub_num_row, sizeof(int64_t));
    ofstream->Close();
  }
  auto ofstream = fs->OpenOutputStream(path_prefix + "/ordered_by_source/vertex_count").ValueOrDie();
  ofstream->Write(&vertex_num, sizeof(int64_t));
  ofstream->Close();
}

void ProcessAdjTable2(std::shared_ptr<arrow::Table> table, int64_t vertex_num, int64_t vertex_chunk_size, int64_t edge_chunk_size, const std::string& src_label, const std::string& dst_label, const std::string& edge_label) {
  auto fs = arrow::fs::FileSystemFromUriOrPath("/mnt/ldbc/weibin/ldbc-gar/ur").ValueOrDie();

  table = table->SelectColumns({0, 1}).ValueOrDie();
  // std::cout << "schema: " << table->schema()->ToString() << "\n" << table->ToString() << std::endl;
  auto path_prefix = "/mnt/ldbc/weibin/ldbc-gar/ur/edge/" + src_label + "_" + edge_label + "_" + dst_label;
  fs->CreateDir(path_prefix + "/ordered_by_dest/adj_list");
  for (int chunk_i = 0; chunk_i * vertex_chunk_size < vertex_num; chunk_i++) {
    int64_t begin = chunk_i * vertex_chunk_size;
    int64_t end = chunk_i * vertex_chunk_size + vertex_chunk_size;
    auto sub_table = FilterPushDown2(table, begin, end);
    // std::cout << "sub_table : " << sub_table->num_rows() << sub_table->ToString() << std::endl;
    auto sub_num_row = sub_table->num_rows();
    fs->CreateDir(path_prefix + "/ordered_by_dest/adj_list/part" + std::to_string(chunk_i));
    for (int j = 0; j * edge_chunk_size < sub_num_row; j++) {
      auto sub_sub_table = sub_table->Slice(j * edge_chunk_size, edge_chunk_size);
      auto path_to_file = path_prefix + "/ordered_by_dest/adj_list/part" + std::to_string(chunk_i) + "/chunk" + std::to_string(j);
      WriteToParquet(sub_sub_table, path_to_file);
      // writeToCsv(sub_sub_table, path_to_file);
    }
    fs->CreateDir(path_prefix + "/ordered_by_dest/offset");
    auto offset = getOffsetTable(sub_table, "_graphArDstIndex", vertex_chunk_size, begin);
    // std::cout << "offset : " << offset->num_rows() << offset->ToString() << std::endl;
    WriteToParquet(offset, path_prefix + "/ordered_by_dest/offset/chunk" + std::to_string(chunk_i));
    // writeToCsv(offset, path_prefix + "/ordered_by_dest/offset/chunk" + std::to_string(chunk_i));
    auto ofstream = fs->OpenOutputStream(path_prefix + "/ordered_by_dest/edge_count" + std::to_string(chunk_i)).ValueOrDie();
    ofstream->Write(&sub_num_row, sizeof(int64_t));
    ofstream->Close();
  }
  auto ofstream = fs->OpenOutputStream(path_prefix + "/ordered_by_dest/vertex_count").ValueOrDie();
  ofstream->Write(&vertex_num, sizeof(int64_t));
  ofstream->Close();
}

void write_to_graphar(
    const std::string& src_vertex_source_file,
    const std::string& edge_source_file,
    const std::string& dst_vertex_source_file,
    const std::string& src_label,
    const std::string& dst_label,
    const std::string& edge_label,
    int64_t vertex_chunk_size,
    int64_t edge_chunk_size) {
    std::string delemiter = "tab";
    // read vertex source to arrow table
    auto dst_vertex_table = read_vertex_csv_to_arrow_table(dst_vertex_source_file, delemiter);
    ProcessVertexTable(dst_label, dst_vertex_table, vertex_chunk_size);
    dst_vertex_table = dst_vertex_table->SelectColumns({0}).ValueOrDie()->RenameColumns({"id"}).ValueOrDie();
    auto edge_table = read_csv_to_arrow_table(edge_source_file, false, delemiter, 0)->SelectColumns({0, 1}).ValueOrDie()->RenameColumns({"src", "dst"}).ValueOrDie();
    auto dst_vertex_table_with_index = add_index_column(dst_vertex_table, false);
    // int64_t vertex_num = dst_vertex_table_with_index->num_rows();
    int64_t vertex_num = dst_vertex_table->num_rows(); 
    dst_vertex_table.reset();
    // auto& vertex_info = graph_info.GetVertexInfo(vertex_label).value();
    auto edge_table_with_dst_index = DoHashJoin(dst_vertex_table_with_index, edge_table, kDstIndexCol, "dst");
    // int64_t vertex_num = 65608366;
    // auto edge_table = read_csv_to_arrow_table(edge_source_file, false, delemiter, 0);
    std::cout << "schema: " << edge_table->schema()->ToString() << std::endl;
    // auto edge_table_with_src_dst_index = edge_table->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();
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
      auto src_vertex_table = read_vertex_csv_to_arrow_table(src_vertex_source_file, delemiter);
      ProcessVertexTable(src_label, src_vertex_table, vertex_chunk_size);
      src_vertex_table = src_vertex_table->SelectColumns({0}).ValueOrDie();
      src_vertex_table_with_index = add_index_column(src_vertex_table, true);
    }
    // DCHECK_OK(WriteToParquet(src_vertex_table_with_index, path_to_file + "-vertex"));
    dst_vertex_table_with_index.reset();
    // auto edge_table_with_src_dst_index = DoHashJoin(src_vertex_table_with_index, edge_table_with_dst_index, kSrcIndexCol, "src")->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({kSrcIndexCol, kDstIndexCol}).ValueOrDie();;
    auto edge_table_with_src_dst_index = DoHashJoin(src_vertex_table_with_index, edge_table_with_dst_index, kSrcIndexCol, "src");
    // edge_table_with_src_dst_index = convert_to_undirected(edge_table_with_src_dst_index);

    auto table = SortKeys(edge_table_with_src_dst_index);
    auto table_2 = SortKeys2(edge_table_with_src_dst_index);
    edge_table_with_src_dst_index.reset();
    // auto table = edge_table_with_src_dst_index;

    ProcessAdjTable(table, vertex_num, vertex_chunk_size, edge_chunk_size, src_label, dst_label, edge_label);
    ProcessAdjTable2(table_2, vertex_num, vertex_chunk_size, edge_chunk_size, src_label, dst_label, edge_label);

    // auto offset = getOffsetTable(table, kSrcIndexCol, vertex_num);
    // table.reset();
    // DCHECK_OK(WriteOffsetToFile(offset, path_to_file + "-offset"));

    return;
}

int main(int argc, char* argv[]) {
   std::string edge_source_file = std::string(argv[1]);
   std::string src_vertex_source_file = std::string(argv[2]);
   std::string dst_vertex_source_file = std::string(argv[3]);
   std::string src_label = std::string(argv[4]);
   std::string dst_label = std::string(argv[5]);
   std::string edge_label = std::string(argv[6]);
   int64_t vertex_chunk_size = std::stol(argv[7]);
   int64_t edge_chunk_size = std::stol(argv[8]);
   std::cout << "edge file: " << edge_source_file << std::endl;
   write_to_graphar(src_vertex_source_file, edge_source_file, dst_vertex_source_file, src_label, dst_label, edge_label, vertex_chunk_size, edge_chunk_size);
}
