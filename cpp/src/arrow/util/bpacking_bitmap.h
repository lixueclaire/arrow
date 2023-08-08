// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <stdint.h>
#include <x86intrin.h>

namespace arrow {
namespace plaint {

inline void set_bit(uint64_t* bitmap, uint64_t curr) {
    bitmap[curr >> 6] |= (1ULL << (curr & 0x3f));
}

inline void unpack1_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  curr += ((*in)) & 1; // d0
  set_bit(bitmap, curr);
  curr += ((*in) >> 1) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 2) & 1; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 3) & 1; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 4) & 1; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 5) & 1; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 6) & 1; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 7) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 8) & 1; // d8
  set_bit(bitmap, curr);
  curr += ((*in) >> 9) & 1; // d9
  set_bit(bitmap, curr);
  curr += ((*in) >> 10) & 1; // d10
  set_bit(bitmap, curr);
  curr += ((*in) >> 11) & 1; // d11
  set_bit(bitmap, curr);
  curr += ((*in) >> 12) & 1; // d12
  set_bit(bitmap, curr);
  curr += ((*in) >> 13) & 1; // d13
  set_bit(bitmap, curr);
  curr += ((*in) >> 14) & 1; // d14
  set_bit(bitmap, curr);
  curr += ((*in) >> 15) & 1; // d15
  set_bit(bitmap, curr);
  curr += ((*in) >> 16) & 1; // d16
  set_bit(bitmap, curr);
  curr += ((*in) >> 17) & 1; // d17
  set_bit(bitmap, curr);
  curr += ((*in) >> 18) & 1; // d18
  set_bit(bitmap, curr);
  curr += ((*in) >> 19) & 1; // d19
  set_bit(bitmap, curr);
  curr += ((*in) >> 20) & 1; // d20
  set_bit(bitmap, curr);
  curr += ((*in) >> 21) & 1; // d21
  set_bit(bitmap, curr);
  curr += ((*in) >> 22) & 1; // d22
  set_bit(bitmap, curr);
  curr += ((*in) >> 23) & 1; // d23
  set_bit(bitmap, curr);
  curr += ((*in) >> 24) & 1; // d24
  set_bit(bitmap, curr);
  curr += ((*in) >> 25) & 1; // d25
  set_bit(bitmap, curr);
  curr += ((*in) >> 26) & 1; // d26
  set_bit(bitmap, curr);
  curr += ((*in) >> 27) & 1; // d27
  set_bit(bitmap, curr);
  curr += ((*in) >> 28) & 1; // d28
  set_bit(bitmap, curr);
  curr += ((*in) >> 29) & 1; // d29
  set_bit(bitmap, curr);
  curr += ((*in) >> 30) & 1; // d30
  set_bit(bitmap, curr);
  curr += ((*in) >> 31); // d31
  set_bit(bitmap, curr);
}

inline void unpack2_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
    curr += (*in) & 3; // d0
    set_bit(bitmap, curr);
    curr += (*in >> 2) & 3; // d1
    set_bit(bitmap, curr);
    curr += (*in >> 4) & 3; // d2
    set_bit(bitmap, curr);
    curr += (*in >> 6) & 3; // d3
    set_bit(bitmap, curr);
    curr += (*in >> 8) & 3; // d4
    set_bit(bitmap, curr);
    curr += (*in >> 10) & 3; // d5
    set_bit(bitmap, curr);
    curr += (*in >> 12) & 3; // d6
    set_bit(bitmap, curr);
    curr += (*in >> 14) & 3; // d7
    set_bit(bitmap, curr);
    curr += (*in >> 16) & 3; // d8
    set_bit(bitmap, curr);
    curr += (*in >> 18) & 3; // d9
    set_bit(bitmap, curr);
    curr += (*in >> 20) & 3; // d10
    set_bit(bitmap, curr);
    curr += (*in >> 22) & 3; // d11
    set_bit(bitmap, curr);
    curr += (*in >> 24) & 3; // d12
    set_bit(bitmap, curr);
    curr += (*in >> 26) & 3; // d13
    set_bit(bitmap, curr);
    curr += (*in >> 28) & 3; // d14
    set_bit(bitmap, curr);
    curr += (*in >> 30); // d15
    set_bit(bitmap, curr);
}

inline void unpack4_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
    curr += (*in) & 15; // d0
    set_bit(bitmap, curr);
    curr += (*in >> 4) & 15; // d1
    set_bit(bitmap, curr);
    curr += (*in >> 8) & 15; // d2
    set_bit(bitmap, curr);
    curr += (*in >> 12) & 15; // d3
    set_bit(bitmap, curr);
    curr += (*in >> 16) & 15; // d4
    set_bit(bitmap, curr);
    curr += (*in >> 20) & 15; // d5
    set_bit(bitmap, curr);
    curr += (*in >> 24) & 15; // d6
    set_bit(bitmap, curr);
    curr += (*in >> 28); // d7
    set_bit(bitmap, curr);
}

inline void unpack8_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
    curr += (*in) & 255; // d0
    set_bit(bitmap, curr);
    curr += (*in >> 8) & 255; // d1
    set_bit(bitmap, curr);
    curr += (*in >> 16) & 255; // d2
    set_bit(bitmap, curr);
    curr += (*in >> 24); // d3
    set_bit(bitmap, curr);
}

}  // namespace plaint

namespace simd {

// optimize by SIMD instructions
// const static __m128i m15 = _mm_set1_epi32(15U); // m15 = 4 * (32-bit 15)
const static __m128i m1 = _mm_set1_epi16(1U); // m1 = 8 * (16-bit 1)
const static __m128i m1_32 = _mm_set1_epi32(1U); // m1_32 = 2 * (32-bit 1)
// const static uint64_t extract_mask = 0x000F000F000F000FULL; // extract_mask = 4 * (16-bit 15)

/// @brief apply a packed bitmap to the final result bitmap
/// @param bitmap the final result bitmap
/// @param curr the highest 1 in result bitmap
/// @param res the packed bitmap
inline void apply_packed_bitmap(uint64_t* bitmap, uint64_t& curr, uint64_t& res) {
  uint64_t index = curr >> 6; // index = curr / 64
  uint64_t offset = curr & 63; // offset = curr % 64
  bitmap[index] |= (res << (offset + 1)); // apply the low bits to bitmap
  bitmap[index + 1] |= (res >> (63 - offset)); // apply the high bits to bitmap
  curr += 64 - __builtin_clzll(res); // update the highest 1 in bitmap
}

inline void unpack1_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  // apply packed bitmap to bitmap
  uint64_t res = static_cast<uint64_t>(*in);
  apply_packed_bitmap(bitmap, curr, res);
}

inline void unpack2_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  uint16_t low_i = static_cast<uint16_t>(*in);
  __m128i b = _mm_set_epi16(1 << ((low_i >> 14) - 1), 1 << (((low_i >> 12) & 3) - 1), 1 << (((low_i >> 10) & 3) - 1),
      1 << (((low_i >> 8) & 3) - 1), 1 << (((low_i >> 6) & 3) - 1), 1 << (((low_i >> 4) & 3) - 1), 1 << (((low_i >> 2) & 3) - 1),
      1 << ((low_i & 3) - 1));

  // generate mask = [m7, m6, m5, m4, m3, m2, m1, m0]
  __m128i m = _mm_sub_epi16(_mm_slli_epi16(b, 1), m1);
  // use low 64 bits of delta bitmap & mask to genearte packed bitmap
  uint64_t res = _pext_u64(_mm_cvtsi128_si64(b), _mm_cvtsi128_si64(m));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
  // use high 64 bits of delta bitmap & mask to genearte packed bitmap
  res = _pext_u64(_mm_cvtsi128_si64(_mm_srli_si128(b, 8)), _mm_cvtsi128_si64(_mm_srli_si128(m, 8)));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);

  uint16_t high_i = static_cast<uint16_t>(*in >> 16);
  b = _mm_set_epi16(1 << ((high_i >> 14) - 1), 1 << (((high_i >> 12) & 3) - 1), 1 << (((high_i >> 10) & 3) - 1),
      1 << (((high_i >> 8) & 3) - 1), 1 << (((high_i >> 6) & 3) - 1), 1 << (((high_i >> 4) & 3) - 1), 1 << (((high_i >> 2) & 3) - 1),
      1 << ((high_i & 3) - 1)); 
  // generate mask = [m7, m6, m5, m4, m3, m2, m1, m0]
  m = _mm_sub_epi16(_mm_slli_epi16(b, 1), m1);
  // use low 64 bits of delta bitmap & mask to genearte packed bitmap
  res = _pext_u64(_mm_cvtsi128_si64(b), _mm_cvtsi128_si64(m));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
  // use high 64 bits of delta bitmap & mask to genearte packed bitmap
  res = _pext_u64(_mm_cvtsi128_si64(_mm_srli_si128(b, 8)), _mm_cvtsi128_si64(_mm_srli_si128(m, 8)));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
}

/// @brief unpack 8 * 4-bit = 32-bit int
/// @param in a 8 * 4-bit delta value
/// @param bitmap the result bitmap
/// @param curr the highest 1 in bitmap
inline void unpack4_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  uint32_t i = *in;
  __m128i b = _mm_set_epi16(1 << ((i >> 28) - 1), 1 << (((i >> 24) & 15) - 1), 1 << (((i >> 20) & 15) - 1),
      1 << (((i >> 16) & 15) - 1), 1 << (((i >> 12) & 15) - 1), 1 << (((i >> 8) & 15) - 1), 1 << (((i >> 4) & 15) - 1),
      1 << ((i & 15) - 1));

  // generate mask = [m7, m6, m5, m4, m3, m2, m1, m0]
  __m128i m = _mm_sub_epi16(_mm_slli_epi16(b, 1), m1);
  // use low 64 bits of delta bitmap & mask to genearte packed bitmap
  uint64_t res = _pext_u64(_mm_cvtsi128_si64(b), _mm_cvtsi128_si64(m));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
  // use high 64 bits of delta bitmap & mask to genearte packed bitmap
  res = _pext_u64(_mm_cvtsi128_si64(_mm_srli_si128(b, 8)), _mm_cvtsi128_si64(_mm_srli_si128(m, 8)));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
}

/// @brief unpack 8 * 4-bit = 32-bit int
/// @param in a 8 * 4-bit delta value
/// @param bitmap the result bitmap
/// @param curr the highest 1 in bitmap
inline void unpack8_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  uint32_t i = *in;
  __m128i b = _mm_set_epi32(1 << ((i >> 24) - 1), 1 << (((i >> 16) & 255) - 1), 1 << (((i >> 8) & 255) - 1),
      1 << ((i & 255) - 1));

  // generate mask = [m7, m6, m5, m4, m3, m2, m1, m0]
  __m128i m = _mm_sub_epi32(_mm_slli_epi32(b, 1), m1_32);
  // use low 64 bits of delta bitmap & mask to genearte packed bitmap
  uint64_t res = _pext_u64(_mm_cvtsi128_si64(b), _mm_cvtsi128_si64(m));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
  // use high 64 bits of delta bitmap & mask to genearte packed bitmap
  res = _pext_u64(_mm_cvtsi128_si64(_mm_srli_si128(b, 8)), _mm_cvtsi128_si64(_mm_srli_si128(m, 8)));
  // apply packed bitmap to bitmap
  apply_packed_bitmap(bitmap, curr, res);
}

}  // namespace simd
}  // namespace arrow
