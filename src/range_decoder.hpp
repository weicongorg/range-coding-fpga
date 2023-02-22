#ifndef RANGE_DECODING_HPP
#define RANGE_DECODING_HPP
#include <vector>
#include "range_coding.h"
#include "range_encoder.hpp"
#include "store.hpp"
#include "test_utils.h"


using UintRCVecx2 = ac_int<kRangeOutSize * 8 * 2, false>;
using UintRCVec = ac_int<kRangeOutSize * 8, false>;

struct RCInputStream {
  UintRCVecx2 bits;
  uchar size;
};

constexpr uint kStep = 8;
constexpr uint kMaxTotalFreq = (1 << 16) - 32;

void updateRange(uint &range, uint &code, RCInputStream &input_stream) {
  bool range_bits[32];
#pragma unroll
  for (char j = 0; j < 32; j++) {
    range_bits[j] = (range >> j) & 0x1;
  }

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

  if (b3) {
    range <<= 24;
    code = (code << 8) | input_stream.bits.to_uint() & 0xff;
    input_stream.bits >>= 8;
    code = (code << 8) | input_stream.bits.to_uint() & 0xff;
    input_stream.bits >>= 8;
    code = (code << 8) | input_stream.bits.to_uint() & 0xff;
    input_stream.bits >>= 8;
    input_stream.size -= 3;
  } else if (b2) {
    range <<= 16;
    code = (code << 8) | input_stream.bits.to_uint() & 0xff;
    input_stream.bits >>= 8;
    code = (code << 8) | input_stream.bits.to_uint() & 0xff;
    input_stream.bits >>= 8;
    input_stream.size -= 2;
  } else if (b1) {
    range <<= 8;
    code = (code << 8) | (input_stream.bits.to_uint()) & 0xff;
    input_stream.bits >>= 8;
    input_stream.size -= 1;
  }
}

struct FreqStat {
  uint totalFreq;
  bool needNorm;
};

using SymbolOutPipe = ext::intel::pipe<class SYmOP, uchar, 235>;
using RCDataInPipe = ext::intel::pipe<class RCInnP, UintRCVec, 235>;
using RCInitPipe = ext::intel::pipe<class RCIP, uint2, 235>;
using FreqInPipe = ext::intel::pipe<class FreqStatP, FreqStat, 235>;

template < uint kNSymbol>
struct FreqUpdater {
  void operator()(uint sym_count) const {
    uint totalFreq = kNSymbol;
    for (uint i = 0; i < sym_count; ++i) {
      FreqStat fs;
      float f = 1.0f / totalFreq;
      uint val = *(uint *)&f;
      fs.totalFreq = val;
      fs.needNorm = totalFreq >= (kMaxTotalFreq);
      FreqInPipe::write(fs);
      if (fs.needNorm) {
        totalFreq = (totalFreq + kStep * 2 + kNSymbol * 2) >> 1;
      } else {
        totalFreq = totalFreq + kStep;
      }
    }
  }
};
constexpr uint MULTIPLY_SHIFT_BITS = 24;

uint shift_base_uint(uint fakeval) {
  uint tail = fakeval << 9;
  int expo = (fakeval >> 23) - 127 + 1;
  int tailLen = MULTIPLY_SHIFT_BITS + expo - 1;
  uint res = 1;
  res <<= tailLen;
  res |= (tail >> (32 - tailLen));
  res <<= (31 - MULTIPLY_SHIFT_BITS);
  return res;
}

uint shift_divide_uint(uint a, uint b) {
  // return a*b;
  uint B = shift_base_uint(b);
  uint res = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    bool abit = (a >> (31 - i)) & 0x1;
    res += abit * (B >> i);
  }
  return res;
}

uint shift_multiply(uint a, ushort b) {
  ac_int<33, false> a33 = a;
  uint sum = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    if ((b >> i) & 1) {
      sum += (a << i) & 0xffffffff;
    }
  }
  return sum;
}

template < uint kNSymbol>
struct RangeDecoderKernel {
  void operator()() const {
    [[intel::fpga_register]] ushort freqs[kNSymbol];
#pragma unroll
    for (int i = 0; i < kNSymbol; i++) {
      freqs[i] = 1;
    }
    uint range = (uint)-1;
    uint2 init = RCInitPipe::read();
    uint num_symbol = init[0];
    uint code = init[1];
    RCInputStream input_stream;
    input_stream.bits = RCDataInPipe::read();
    input_stream.size = kRangeOutSize;

    for (uint s = 0; s < num_symbol; ++s) {
      auto fs = FreqInPipe::read();
      uint range_unit = shift_divide_uint(range, fs.totalFreq);

      auto initial_code = code;
      uchar symbol = 0;
      bool no_symbol_appeared = true;

      ushort acc_freq = 0;
#pragma unroll
      for (uint i = 0; i < kNSymbol; ++i) {
        // uint cmuc = shift_multiply(range_unit, acc_freq + freqs[i]);
        // uint acc_range = shift_multiply(range_unit, acc_freq);
        uint cmuc = range_unit * (acc_freq + freqs[i]);
        uint acc_range = range_unit * acc_freq;
        acc_freq += freqs[i];

        bool is_symbol = cmuc > initial_code && no_symbol_appeared;
        no_symbol_appeared = cmuc <= initial_code;

        if (is_symbol) {
          symbol = i;
          range = shift_multiply(range_unit, freqs[i]);
          code = initial_code - acc_range;
        }
      }

#pragma unroll
      for (int i = 0; i < kNSymbol; ++i) {
        if (fs.needNorm) {
          freqs[i] = (freqs[i] >> 1) | 1;
        }
        if (symbol == i) {
          freqs[i] += kStep;
        }
      }

      updateRange(range, code, input_stream);

      if (input_stream.size <= kRangeOutSize) {
        UintRCVecx2 in = RCDataInPipe::read();
        input_stream.bits |= in << (input_stream.size * 8);
        input_stream.size += kRangeOutSize;
      }

      SymbolOutPipe::write(symbol);
    }
  }
};


#endif  // RANGE_DECODING_HPP