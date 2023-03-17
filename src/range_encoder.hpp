#ifndef RANGE_CODER_HPP_
#define RANGE_CODER_HPP_
#include <CL/sycl.hpp>

#include "range_coding.h"
#include "unrolled_loop.hpp"
using namespace sycl;

template <uint kNCoders, uint kNSymbol>
struct RangeCoder {
  static uint UpdateRange(uint freq, uint range, uint total_freq_reciprocal) {
    // return freq*range*total_freq_reciprocal;
    uint B = ExtractMantissa(total_freq_reciprocal);
    uint res = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      uint t = ((B >> i) * freq);
      if ((range >> (31 - i)) & 0x1) {
        res += t;
      }
    }
    return res;
  }

  void operator()() const {
    ulong low[kNCoders];
    uint range[kNCoders];
    uint range_sizes[kNCoders];
    bool can_continue[kNCoders];
#pragma unroll
    for (int i = 0; i < kNCoders; i++) {
      can_continue[i] = true;
      range_sizes[i] = 0;
      low[i] = 0;
      range[i] = (uint)-1;
    }
    [[intel::fpga_register]] auto freqs = FreqPipe<kNSymbol>::read();
    [[intel::fpga_register]] auto cumulated_freq = FreqPipe<kNSymbol>::read();

    // extend 2 loops for sending low out after all done
    bool do_ouput_low[2] = {false, false};
    bool alive = true;

    while (alive) {
      bool alive_exists = false;
      std::array<uint, kNCoders> carry_locations;
      fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
        carry_locations[i] = 0xffffffff;
        bool read_success = false;
        auto [symbol, done] = SymInPipe::read<i>(read_success);
        if (read_success) {
          can_continue[i] = !done;
        }
        if (read_success && !done) {
          auto freq_shift = freqs;
          freq_shift.AcInt() >>= symbol * 16;
          auto acc_shift = cumulated_freq;
          acc_shift.AcInt() >>= symbol * 16;

          uint range_unit = range[i] / kTotalFreq;
          auto temp = ShiftMultiply(range_unit, acc_shift[0]);
          ulong low_LS32b = low[i] & 0xffffffff;
          ulong low_MS32b = low[i] >> 32;
          if (low_MS32b == 0xffffffff && low_LS32b + temp > 0xffffffff) {
            carry_locations[i] = range_sizes[i];
          }
          low[i] += temp;
          range[i] = ShiftMultiply(range_unit, freq_shift[0]);
        }
      });
      fpga_tools::UnrolledLoop<0, kNCoders>(
          [&](auto i) { alive_exists = alive_exists || can_continue[i]; });

      do_ouput_low[1] = do_ouput_low[0];
      alive = !do_ouput_low[1];
      do_ouput_low[0] = !alive_exists;

      std::array<RangeOutput, kNCoders> out_buffers;
      fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
        ac_int<32, false> range_bits(range[i]);

        bool is_first_byte_zero = !(
            range_bits[24] | range_bits[25] | range_bits[26] | range_bits[27] |
            range_bits[28] | range_bits[29] | range_bits[30] | range_bits[31]);
        bool is_second_byte_zero = !(
            range_bits[16] | range_bits[17] | range_bits[18] | range_bits[19] |
            range_bits[20] | range_bits[21] | range_bits[22] | range_bits[23]);
        bool is_third_byte_zero = !(
            range_bits[8] | range_bits[9] | range_bits[10] | range_bits[11] |
            range_bits[12] | range_bits[13] | range_bits[14] | range_bits[15]);

        bool range_8bits =
            is_first_byte_zero & is_second_byte_zero & is_third_byte_zero;
        bool range_16bits = is_first_byte_zero & is_second_byte_zero;
        bool range_24bits = is_first_byte_zero;

        RangeOutput out{0, {0, 0, 0, 0}};
        if (range_8bits) {
          out = {3,
                 {(uchar)(low[i] >> 56), (uchar)(low[i] >> 48),
                  (uchar)(low[i] >> 40), 0}};
          range[i] <<= 24;
          low[i] <<= 24;
        } else if (range_16bits) {
          out = {2, {(uchar)(low[i] >> 56), (uchar)(low[i] >> 48), 0, 0}};
          range[i] <<= 16;
          low[i] <<= 16;
        } else if (range_24bits) {
          out = {1, {(uchar)(low[i] >> 56), 0, 0, 0}};
          range[i] <<= 8;
          low[i] <<= 8;
        }

        if (do_ouput_low[0]) {
          out.size = 4;
          ulong curr_low = low[i];
          if (do_ouput_low[1]) {
            curr_low <<= 32;
          }
#pragma unroll
          for (int k = 0; k < 4; k++) {
            out.buffer[k] = curr_low >> 56;
            curr_low <<= 8;
          }
        }
        range_sizes[i] += out.size;
        out_buffers[i] = out;
      });

      RangePipe<kNCoders>::write({out_buffers, do_ouput_low[1]});
      RangeCarryPipe<kNCoders>::write({carry_locations, do_ouput_low[1]});
    }
  }
};
#endif
