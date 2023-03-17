#ifndef RANGE_CODING_H_
#define RANGE_CODING_H_
#include <CL/sycl.hpp>
#include <array>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>

#include "onchip_memory_with_cache.hpp"
#include "pipe_array.hpp"
#include "shifting_array.hpp"

using namespace sycl;
using std::array;

static constexpr uint kTotalFreq = 4096;
constexpr uint kRangeOutSize = 4;
struct RangeOutput {
  using IdxType = ac_int<Log2(kRangeOutSize) + 1, false>;
  IdxType size;
  ShiftingArray<uchar, kRangeOutSize> buffer;
};

using UintRCVecx2 = ac_int<kRangeOutSize * 8 * 2, false>;
using UintRCVec = ac_int<kRangeOutSize * 8, false>;

struct RCInputStream {
  UintRCVecx2 bits;
  uchar size;
};

using SymInPipe = PipeArray<class FreqPP, FlagBundle<uchar>, 1, 22>;
template <uint kNSymbol>
using FreqPipe =
    ext::intel::pipe<class FPP, ShiftingArray<ushort, kNSymbol>, 1>;
template <uint num_coder>
using RangePipe =
    ext::intel::pipe<class ROutP, FlagBundle<array<RangeOutput, num_coder>>,
                     4>;
template <uint num_coder>
using RangeCarryPipe =
    ext::intel::pipe<class ROutP, FlagBundle<array<uint, num_coder>>, 4>;

using SymbolOutPipe = ext::intel::pipe<class SYmOP, uchar, 4>;
using RCDataInPipe = ext::intel::pipe<class RCInnP, UintRCVec, 8>;
using RCInitPipe = ext::intel::pipe<class RCIP, uint3, 1>;
template <uint kNSymbol>
using DecodingFreqPipe =
    ext::intel::pipe<class FRDPP, ShiftingArray<ushort, kNSymbol>, 1>;


uint ExtractMantissa(uint fakeval) {
  constexpr uint kManBits = 24;
  uint tail = fakeval << 9;
  int expo = (fakeval >> 23) - 127 + 1;
  int tailLen = kManBits + expo - 1;
  uint res = 1;
  res <<= tailLen;
  res |= (tail >> (32 - tailLen));
  res <<= (31 - kManBits);
  return res;
}

uint ShiftDivide(uint a, uint b) {
  // return a*b;
  uint B = ExtractMantissa(b);
  uint res = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    bool abit = (a >> (31 - i)) & 0x1;
    res += abit * (B >> i);
  }
  return res;
}

uint ShiftMultiply(uint a, ushort b) {
  uint sum = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if ((b >> i) & 1) {
      sum += (a << i) & 0xffffffff;
    }
  }
  return sum;
}

void UpdateRange(uint &range, uint &code, RCInputStream &input_stream) {
  ac_int<32, false> range_bits(range);

  bool b_large_24bits = range_bits[24] | range_bits[25] | range_bits[26] |
                        range_bits[27] | range_bits[28] | range_bits[29] |
                        range_bits[30] | range_bits[31];
  bool b_large_16bits = range_bits[16] | range_bits[17] | range_bits[18] |
                        range_bits[19] | range_bits[20] | range_bits[21] |
                        range_bits[22] | range_bits[23];
  bool b_large_8bits = range_bits[8] | range_bits[9] | range_bits[10] |
                       range_bits[11] | range_bits[12] | range_bits[13] |
                       range_bits[14] | range_bits[15];

  bool b3 = !(b_large_24bits | b_large_16bits | b_large_8bits);
  bool b2 = !(b_large_24bits | b_large_16bits);
  bool b1 = !(b_large_24bits);

  uchar *bits_ptr = (uchar *)&input_stream.bits;
  if (b3) {
    range <<= 24;
    code = (code << 8) | bits_ptr[0];
    code = (code << 8) | bits_ptr[1];
    code = (code << 8) | bits_ptr[2];
    input_stream.bits >>= 24;
    input_stream.size -= 3;
  } else if (b2) {
    range <<= 16;
    code = (code << 8) | bits_ptr[0];
    code = (code << 8) | bits_ptr[1];
    input_stream.bits >>= 16;
    input_stream.size -= 2;
  } else if (b1) {
    range <<= 8;
    code = (code << 8) | bits_ptr[0];
    input_stream.bits >>= 8;
    input_stream.size -= 1;
  }
}

uint Multiply(uint a, ushort b) { return a * b; }

#endif  // RANGE_CODING_H_