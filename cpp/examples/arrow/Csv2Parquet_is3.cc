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

static constexpr const char* kVertexIndexCol = "_graphArVertexIndex";
static constexpr const char* kSrcIndexCol = "_graphArSrcIndex";
static constexpr const char* kDstIndexCol = "_graphArDstIndex";
static constexpr const char* kOffsetCol = "_graphArOffset";

arrow::Status WriteToParquet(std::shared_ptr<arrow::Table> table,
                          const std::string& path_to_file) {
  // #include "parquet/arrow/writer.h"
  // #include "arrow/util/type_fwd.h"
  using parquet::ArrowWriterProperties;
  using parquet::WriterProperties;

  // Choose compression
  std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding("src", parquet::Encoding::PLAIN)->encoding("dst", parquet::Encoding::PLAIN)->build();
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

arrow::Result<std::shared_ptr<arrow::Table>> read_label_parquet_to_arrow_table(
    const std::string& path_to_file) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = arrow::io::ReadableFile::Open(path_to_file).ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));
  return table;
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

  // Choose compressio
  // std::shared_ptr<WriterProperties> props =
  //       WriterProperties::Builder().disable_dictionary()->compression(parquet::Compression::UNCOMPRESSED)->encoding(parquet::Encoding::PLAIN)->encoding("dst", parquet::Encoding::PLAIN)->build();
  std::shared_ptr<WriterProperties> props =
      WriterProperties::Builder().compression(parquet::Compression::UNCOMPRESSED)->encoding(parquet::Encoding::PLAIN)->build();

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

std::shared_ptr<arrow::Table> convert_to_undirected(
    const std::shared_ptr<arrow::Table>& table) {
  auto reverse_table = table->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({"src", "dst"}).ValueOrDie();
  auto new_table = arrow::ConcatenateTables({table, reverse_table}).ValueOrDie();
  return new_table;
}

void writeToCsv(const std::shared_ptr<arrow::Table>& table, const std::string& path_to_file) {
  std::shared_ptr<arrow::io::OutputStream> output = arrow::io::FileOutputStream::Open(path_to_file).ValueOrDie();
  auto write_options = arrow::csv::WriteOptions::Defaults();
  write_options.include_header = true;
  write_options.delimiter = '|';
  arrow::csv::WriteCSV(*table, write_options, output.get());
  return;
}

void Csv2Csv(
    const std::string& path_to_file,
    // const std::string& vertex_source_file,
    const std::string& edge_source_file,
    int ignore_rows) {
    // const std::string& label_file) {
    // read vertex source to arrow table
    std::string delemiter = "|";
    // read vertex source to arrow table
    auto table = read_csv_to_arrow_table(edge_source_file, delemiter);
    std::cout << "num rows: " << table->num_rows() << std::endl;
    table = table->SelectColumns({1, 2}).ValueOrDie();
    // std::cout <<  "schema: " << table->schema()->ToString() << std::endl;
    // table = table->RenameColumns({"Comment.id", "Comment.id2"}).ValueOrDie();
    // auto label_table = read_label_parquet_to_arrow_table(label_file).ValueOrDie();
    // auto new_vertex_table = vertex_table->AddColumn(vertex_table->num_columns(), label_table->field(0), label_table->column(0)).ValueOrDie();
    // std::cout << "schema: " << new_vertex_table->schema()->ToString() << std::endl;
    // auto edge_table = read_csv_to_arrow_table(edge_source_file, true, delemiter, ignore_rows)->SelectColumns({0, 1}).ValueOrDie();
    // edge_table = edge_table->SelectColumns({1, 0}).ValueOrDie()->RenameColumns({"src", "dst"}).ValueOrDie();
    // std::cout << "reverse schema: " << edge_table->schema()->ToString() << std::endl;
    // edge_table = convert_to_undirected(edge_table);

    // DCHECK_OK(WriteToFileBaseLine(table, path_to_file+".parquet"));
    // DCHECK_OK(WriteToFileBaseLine(new_vertex_table, path_to_file+"-vertex-base"));
    writeToCsv(table, path_to_file+".csv");

    return;
}


void VertexCsv2Parquet(
    const std::string& path_to_file,
    const std::string& source_file,
    int ignore_rows) {
    // const std::string& label_file) {
    std::string delemiter = "|";
    // read vertex source to arrow table
    auto table = read_csv_to_arrow_table(source_file, delemiter);
    std::cout << "num rows: " << table->num_rows() << std::endl;
    std::cout <<  "schema: " << table->schema()->ToString() << std::endl;

    DCHECK_OK(WriteToFileBaseLine(table, path_to_file+"-vertex-base"));

    return;
}

void EdgeCsv2Parquet(
    const std::string& path_to_file,
    const std::string& source_file,
    int ignore_rows,
    bool to_undirected) {
    std::string delemiter = "|";
    // read source to arrow table
    auto edge_table = read_csv_to_arrow_table(source_file, delemiter)->SelectColumns({0, 1}).ValueOrDie()->RenameColumns({"src", "dst"}).ValueOrDie();
    if (to_undirected) {
      edge_table = convert_to_undirected(edge_table);
    }
    std::cout << "schema: " << edge_table->schema()->ToString() << std::endl;

    DCHECK_OK(WriteToFileBaseLine(edge_table, path_to_file+"origin-base"));

    return;
}

int main(int argc, char* argv[]) {
  std::string source_file = std::string(argv[1]);
  std::string path_to_file = std::string(argv[2]);
  int ignore_rows = std::stoi(std::string(argv[3])); 
  std::string type = std::string(argv[4]);
  bool to_undirected = false;
  if (argc > 5) {
    to_undirected = std::string(argv[5]) == "to_undirected";
  }

  if (type == "edge") {
    EdgeCsv2Parquet(path_to_file, source_file, ignore_rows, to_undirected);
  } else {
    // vertex
    VertexCsv2Parquet(path_to_file, source_file, ignore_rows);
  }
}
