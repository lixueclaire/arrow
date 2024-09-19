#ifndef ARQUET_EXAMPLES_GRAPHAR_CONFIG_H
#define ARQUET_EXAMPLES_GRAPHAR_CONFIG_H

/// constants related to encoding and decoding of the labels
constexpr int MAX_LABEL_NUM = 100;         // the maximum number of labels
constexpr int MAX_DECODED_NUM = 100000;  // the maximum number of decoded values
/// constants related to the parquet file
constexpr int NUM_ROWS_PER_ROW_GROUP = 1024 * 1024;  // the number of rows per row group
constexpr int BATCH_SIZE = 1024;                     // the batch size
constexpr int THRESHOLD = 10;                        // the threshold to num_vertices/10
// kDefaultDataPageSize = 1024 * 1024
// DEFAULT_WRITE_BATCH_SIZE = 1024
// DEFAULT_MAX_ROW_GROUP_LENGTH = 1024 * 1024

#endif