#ifndef RANGE_DECODING_HPP
#define RANGE_DECODING_HPP
#include <vector>

#include "range_coding.h"

template <uint kNSymbol>
struct RangeDecoder {
  void operator()() const {
    [[intel::fpga_register]] auto freqs = DecodingFreqPipe<kNSymbol>::read();
    [[intel::fpga_register]] auto cumulated_freq =
        DecodingFreqPipe<kNSymbol>::read();
    [[intel::fpga_register]] auto acc_freq = DecodingFreqPipe<kNSymbol>::read();

    uint range = (uint)-1;
    uint3 init = RCInitPipe::read();
    uint num_symbol = init[0];
    uint code = init[1];
    RCInputStream input_stream{init[2], kRangeOutSize};

    for (uint s = 0; s < num_symbol; ++s) {
      auto initial_code = code;
      uint range_unit = range / kTotalFreq;
      bool no_symbol_appeared = true;
      uchar symbol = 0;

      fpga_tools::UnrolledLoop<kNSymbol>([&](auto i) {
        uint cmuc = range_unit * acc_freq[i];
        bool cmp = cmuc > initial_code;
        bool is_symbol = cmp && no_symbol_appeared;
        no_symbol_appeared = !cmp;
        if (is_symbol) {
          symbol = i;
        }
      });

      auto freq_shift = freqs;
      freq_shift.AcInt() >>= symbol * 16;
      auto acc_shift = cumulated_freq;
      acc_shift.AcInt() >>= symbol * 16;

      range = ShiftMultiply(range_unit, freq_shift[0]);
      code = code - ShiftMultiply(range_unit, acc_shift[0]);
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