#ifndef RANGE_DECODING_HPP
#define RANGE_DECODING_HPP
#include <vector>

#include "range_coding.h"
#include "range_encoder.hpp"

constexpr uint kStep = SimpleModel<0>::kStep;
constexpr uint kMaxTotalFreq = SimpleModel<0>::kBound;

void UpdateRange(uint &range, uint &code, RCInputStream &input_stream) {
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

template <uint kNSymbol>
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

uint ShiftMultiply(uint a, ushort b) {
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

template <uint kNSymbol>
struct RangeDecoderKernel {
  void operator()() const {
    [[intel::fpga_register]] ushort freqs[kNSymbol];
#pragma unroll
    for (int i = 0; i < kNSymbol; i++) {
      freqs[i] = 1;
    }
    uint range = (uint)-1;
    uint3 init = RCInitPipe::read();
    uint num_symbol = init[0];
    uint code = init[1];
    RCInputStream input_stream{init[2], kRangeOutSize};

    for (uint s = 0; s < num_symbol; ++s) {
      auto fs = FreqInPipe::read();
      uint range_unit = ShiftDivide(range, fs.totalFreq);

      auto initial_code = code;
      uchar symbol = 0;
      bool no_symbol_appeared = true;

      ushort acc_freq[kNSymbol + 1] = {0};
#pragma unroll
      for (uint i = 1; i <= kNSymbol; ++i) {
        acc_freq[i] = acc_freq[i - 1] + freqs[i - 1];
      }

#pragma unroll
      for (uint i = 0; i < kNSymbol; ++i) {
        uint cmuc = ShiftMultiply(range_unit, acc_freq[i + 1]);
        uint acc_range = ShiftMultiply(range_unit, acc_freq[i]);

        bool is_symbol = cmuc > initial_code && no_symbol_appeared;
        no_symbol_appeared = cmuc <= initial_code;

        if (is_symbol) {
          symbol = i;
          range = ShiftMultiply(range_unit, freqs[i]);
          code = initial_code - acc_range;
        }
        if (fs.needNorm) {
          freqs[i] = (freqs[i] >> 1) | 1;
        }
        if (is_symbol) {
          freqs[i] += kStep;
        }
      }

      UpdateRange(range, code, input_stream);

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