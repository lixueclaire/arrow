
namespace parquet {

inline void set_bit(uint64_t* bitmap, uint64_t curr) {
    bitmap[curr >> 6] |= (1 << (curr & 63));
}

#define DELTA_PLAIT_DECODING

#ifdef DELTA_PLAIT_DECODING

inline void unpack1_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  curr += ((*in)) & 1; // d0
  set_bit(bitmap, curr);
  curr += ((*in) >> 1) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 2) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 3) & 1; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 4) & 1; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 5) & 1; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 6) & 1; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 7) & 1; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 8) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 9) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 10) & 1; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 11) & 1; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 12) & 1; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 13) & 1; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 14) & 1; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 15) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 16) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 17) & 1; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 18) & 1; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 19) & 1; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 20) & 1; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 21) & 1; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 22) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 23) & 1; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 24) & 1; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 25) & 1; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 26) & 1; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 27) & 1; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 28) & 1; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 29) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 30) & 1; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 31); // d7
  set_bit(bitmap, curr);
}

inline void unpack2_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  curr += ((*in)) & 3; // d0
  set_bit(bitmap, curr);
  curr += ((*in) >> 2) & 3; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 4) & 3; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 6) & 3; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 8) & 3; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 10) & 3; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 12) & 3; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 14) & 3; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 16) & 3; // d7
  set_bit(bitmap, curr);
  curr += ((*in) >> 18) & 3; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 20) & 3; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 22) & 3; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 24) & 3; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 26) & 3; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 28) & 3; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 30); // d7
  set_bit(bitmap, curr);
}

inline void unpack4_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
  curr += ((*in)) & 15; // d0
  set_bit(bitmap, curr);
  curr += ((*in) >> 4) & 15; // d1
  set_bit(bitmap, curr);
  curr += ((*in) >> 8) & 15; // d2
  set_bit(bitmap, curr);
  curr += ((*in) >> 12) & 15; // d3
  set_bit(bitmap, curr);
  curr += ((*in) >> 16) & 15; // d4
  set_bit(bitmap, curr);
  curr += ((*in) >> 20) & 15; // d5
  set_bit(bitmap, curr);
  curr += ((*in) >> 24) & 15; // d6
  set_bit(bitmap, curr);
  curr += ((*in) >> 28); // d7
  set_bit(bitmap, curr);
}
#endif

#ifdef DELTA_SIMD_DECODING

inline void unpack1_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
}

inline void unpack4_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
}

inline void unpack8_8(const uint32_t* in, uint64_t* bitmap, uint64_t& curr) {
}
#endif

} // namespace parquet


