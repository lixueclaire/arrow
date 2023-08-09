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

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>
#include <vector>
#include <algorithm>

// define constants
#define BIT_WIDTH_MUST_BE_POWER_OF_TWO 1  // the bit width must be power of two
#define MINI_BLOCK_SIZE 32                // the size of mini block
#define MAX_BIT_WIDTH 32                  // the maximum bit width of delta
#define MAX_VERTEX_NUM 50000000           // the maximum number of vertices
#define MAX_DEGREE 10000000               // the maximum degree of vertices
static constexpr uint32_t VALID_BIT_WIDTH[] = {1, 2, 4, 8, 16, 32};  // valid bit widths

// define global variables
uint32_t max_out_degree, max_in_degree;
uint32_t vertex_id_of_max_out_degree, vertex_id_of_max_in_degree;
uint32_t out_degree[MAX_VERTEX_NUM], in_degree[MAX_VERTEX_NUM];
uint32_t vertex_num_of_out_degree[MAX_DEGREE], vertex_num_of_in_degree[MAX_DEGREE];
int32_t out_delta[MAX_DEGREE], in_delta[MAX_DEGREE];
uint32_t mini_block_num_of_bit_width[MAX_BIT_WIDTH + 1];

// conduct analysis, N is the number of vertices
void conduct_analysis(const int32_t N, bool is_directed, bool has_weight,
                      bool is_ordered) {
  // initialize
  int32_t src, dst;
  double weight;
  uint64_t tot_out_degree = 0, tot_in_degree = 0, edge_num = 0;
  max_out_degree = 0;
  max_in_degree = 0;
  memset(out_degree, 0, sizeof(out_degree));
  memset(in_degree, 0, sizeof(in_degree));
  memset(vertex_num_of_out_degree, 0, sizeof(vertex_num_of_out_degree));
  memset(vertex_num_of_in_degree, 0, sizeof(vertex_num_of_in_degree));

  // read the graph
  while (scanf("%d %d", &src, &dst) != EOF) {
    if (has_weight) scanf("%lf", &weight);
    assert(src < N);
    assert(dst < N);
    edge_num++;
    out_degree[src]++;
    in_degree[dst]++;
    if (!is_directed) {
      edge_num++;
      out_degree[dst]++;
      in_degree[src]++;
    }
  }

  // find the vertex of max degree
  for (int i = 0; i < N; ++i) {
    tot_out_degree += out_degree[i];
    tot_in_degree += in_degree[i];
    vertex_num_of_out_degree[out_degree[i]]++;
    vertex_num_of_in_degree[in_degree[i]]++;
    if (out_degree[i] > max_out_degree) {
      max_out_degree = out_degree[i];
      vertex_id_of_max_out_degree = i;
    }
    if (in_degree[i] > max_in_degree) {
      max_in_degree = in_degree[i];
      vertex_id_of_max_in_degree = i;
    }
  }

  // output
  std::cout << "number of vertices: " << N << std::endl;
  std::cout << "number of edges: " << edge_num << std::endl;
  std::cout << "vertex id of max out degree: " << vertex_id_of_max_out_degree
            << std::endl;
  std::cout << "max out degree: " << max_out_degree << std::endl;
  std::cout << "vertex id of max in degree: " << vertex_id_of_max_in_degree << std::endl;
  std::cout << "max in degree: " << max_in_degree << std::endl;
  std::cout << "average out degree: " << tot_out_degree / (double)N << std::endl;
  std::cout << "average in degree: " << tot_in_degree / (double)N << std::endl;
}

// get the minimal bit width required to store a number
inline uint32_t get_bit_width(uint32_t num) {
  uint32_t bit_width = 0;
  while (num) {
    num >>= 1;
    bit_width++;
  }
  if (BIT_WIDTH_MUST_BE_POWER_OF_TWO) {
    for (int i = 0;; i++) {
      if (bit_width <= VALID_BIT_WIDTH[i]) {
        bit_width = VALID_BIT_WIDTH[i];
        break;
      }
    }
  } else {
    bit_width = std::max(bit_width, 1U);
  }
  return bit_width;
}

// find neighbors of vertex with max degree
void find_neighbors(const char* graph_name, const char* output_path, const int32_t N,
                    bool is_directed, bool has_weight, bool is_ordered) {
  // read the edges
  int32_t src, dst, prev_src = -1, prev_dst = -1, edge_num_of_src = 0;
  double weight;
  std::vector<int32_t> out_neighbors, in_neighbors;

  // mini block
  uint32_t current_mini_block_size = 0, max_delta_in_mini_block = 0, bit_width;
  memset(mini_block_num_of_bit_width, 0, sizeof(mini_block_num_of_bit_width));

  while (scanf("%d %d", &src, &dst) != EOF) {
    if (has_weight) scanf("%lf", &weight);
    if (!is_directed) {
      if (src == vertex_id_of_max_in_degree) {
        out_neighbors.push_back(dst);
      }
      if (dst == vertex_id_of_max_out_degree) {
        out_neighbors.push_back(src);
      }
    } else {
      if (src == vertex_id_of_max_out_degree) {
        out_neighbors.push_back(dst);
      }
      if (dst == vertex_id_of_max_in_degree) {
        in_neighbors.push_back(src);
      }
    }
    if (is_directed && is_ordered) {
      if (src != prev_src) {
        if (edge_num_of_src >= MINI_BLOCK_SIZE && current_mini_block_size > 0) {
          bit_width = get_bit_width(max_delta_in_mini_block);
          mini_block_num_of_bit_width[bit_width]++;
        }
        prev_src = src;
        prev_dst = dst;
        edge_num_of_src = 1;
        current_mini_block_size = 0;
        max_delta_in_mini_block = 0;
      } else {
        edge_num_of_src++;
        current_mini_block_size++;
        if (dst - prev_dst > max_delta_in_mini_block) {
          max_delta_in_mini_block = dst - prev_dst;
        }
        if (current_mini_block_size == MINI_BLOCK_SIZE) {
          bit_width = get_bit_width(max_delta_in_mini_block);
          mini_block_num_of_bit_width[bit_width]++;
          current_mini_block_size = 0;
          max_delta_in_mini_block = 0;
        }
        prev_dst = dst;
      }
    }
  }

  // the last mini block
  if (is_directed && is_ordered) {
    if (edge_num_of_src >= MINI_BLOCK_SIZE && current_mini_block_size > 0) {
      bit_width = get_bit_width(max_delta_in_mini_block);
      mini_block_num_of_bit_width[bit_width]++;
    }
    // output
    std::string delta_distribution_file =
        std::string(output_path) + "/" + graph_name + "_all_delta_distribution.csv";
    std::ofstream delta_distribution_out(delta_distribution_file);
    if (BIT_WIDTH_MUST_BE_POWER_OF_TWO) {
      for (int i = 0;; i++) {
        delta_distribution_out << VALID_BIT_WIDTH[i] << ","
                               << mini_block_num_of_bit_width[VALID_BIT_WIDTH[i]]
                               << std::endl;
        if (VALID_BIT_WIDTH[i] == MAX_BIT_WIDTH) {
          break;
        }
      }
    } else {
      for (int i = 1; i <= MAX_BIT_WIDTH; ++i) {
        delta_distribution_out << i << "," << mini_block_num_of_bit_width[i] << std::endl;
      }
    }
    delta_distribution_out.close();
  }

  // calculate the delta
  if (!is_ordered) {
    std::sort(out_neighbors.begin(), out_neighbors.end());
  }
  for (auto i = 1; i < out_neighbors.size(); ++i) {
    out_delta[i - 1] = out_neighbors[i] - out_neighbors[i - 1];
  }
  if (is_directed) {
    if (!is_ordered) {
      std::sort(in_neighbors.begin(), in_neighbors.end());
    }
    for (auto i = 1; i < in_neighbors.size(); ++i) {
      in_delta[i - 1] = in_neighbors[i] - in_neighbors[i - 1];
    }
  }
}

// output the result
void output_result(const char* graph_name, const char* output_path, const int32_t* delta,
                   const uint32_t* vertex_num_of_degree, uint32_t max_degree,
                   std::string dir) {
  // output the degree distribution
  std::string degree_distribution_file = std::string(output_path) + "/" + graph_name +
                                         "_" + dir + "_degree_distribution.csv";
  std::ofstream degree_distribution_out(degree_distribution_file);
  for (int i = 0; i <= max_degree; ++i) {
    degree_distribution_out << i << "," << vertex_num_of_degree[i] << std::endl;
  }
  degree_distribution_out.close();

  // output the delta distribution of the vertex of max degree
  std::string delta_distribution_file =
      std::string(output_path) + "/" + graph_name + "_" + dir + "_delta_distribution.csv";
  std::ofstream delta_distribution_out(delta_distribution_file);
  uint32_t length = max_degree - 1;
  uint32_t current_mini_block_size = 0, max_delta_in_mini_block = 0;
  memset(mini_block_num_of_bit_width, 0, sizeof(mini_block_num_of_bit_width));
  for (int i = 0; i < length; ++i) {
    if (delta[i] > max_delta_in_mini_block) {
      max_delta_in_mini_block = delta[i];
    }
    current_mini_block_size++;
    if (current_mini_block_size == MINI_BLOCK_SIZE || i == length - 1) {
      uint32_t bit_width = get_bit_width(max_delta_in_mini_block);
      assert(bit_width <= MAX_BIT_WIDTH);
      mini_block_num_of_bit_width[bit_width]++;
      current_mini_block_size = 0;
      max_delta_in_mini_block = 0;
    }
  }
  if (BIT_WIDTH_MUST_BE_POWER_OF_TWO) {
    for (int i = 0;; i++) {
      delta_distribution_out << VALID_BIT_WIDTH[i] << ","
                             << mini_block_num_of_bit_width[VALID_BIT_WIDTH[i]]
                             << std::endl;
      if (VALID_BIT_WIDTH[i] == MAX_BIT_WIDTH) {
        break;
      }
    }
  } else {
    for (int i = 1; i <= MAX_BIT_WIDTH; ++i) {
      delta_distribution_out << i << "," << mini_block_num_of_bit_width[i] << std::endl;
    }
  }
  delta_distribution_out.close();
}

int main(int argc, char** argv) {
  // argv[1] is the graph name
  char* graph_name = argv[1];
  // argv[2] is the path of the data file
  char* data_file = argv[2];
  // argv[3] is the vertex number of the graph
  int32_t vertex_num = atoi(argv[3]);
  // argv[4] is the root path of the output
  char* output_path = argv[4];
  // argv[5] means if it is a directed graph
  bool is_directed = false;
  if (strcmp(argv[5], "true") == 0) {
    is_directed = true;
  }
  // argv[6] means if there are weights on edge
  bool has_weight = false;
  if (strcmp(argv[6], "true") == 0) {
    has_weight = true;
  }
  // argv[7] means is it is ordered
  bool is_ordered = false;
  if (strcmp(argv[7], "true") == 0) {
    is_ordered = true;
  }

  // open the data file
  freopen(data_file, "r", stdin);

  // conduct analysis
  conduct_analysis(vertex_num, is_directed, has_weight, is_ordered);

  // close the input file
  fclose(stdin);

  // open the data file again
  freopen(data_file, "r", stdin);

  // find neighbors of vertex with max degree
  find_neighbors(graph_name, output_path, vertex_num, is_directed, has_weight,
                 is_ordered);

  // close the input file
  fclose(stdin);

  // output the result
  output_result(graph_name, output_path, out_delta, vertex_num_of_out_degree,
                max_out_degree, "out");
  if (is_directed) {
    output_result(graph_name, output_path, in_delta, vertex_num_of_in_degree,
                  max_in_degree, "in");
  }

  return 0;
}
