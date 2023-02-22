#ifndef RANGE_CODING_H_
#define RANGE_CODING_H_
#include <CL/sycl.hpp>
#include <type_traits>
#include <array>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "onchip_memory_with_cache.hpp"
#include "shifting_array.hpp"
#include "pipe_array.hpp"

using namespace sycl;
using std::array;


struct SymbolFrequence {
  ushort freq;
  ushort cumulative_freq;
  uint total_freq;
};

template <uint kNSymbol, typename TFreq = ushort>
struct SimpleModel {
  static constexpr uint kStep = std::is_same<ushort, TFreq>::value ? 8 : 1;
  static constexpr uint kBound =
      std::is_same<ushort, TFreq>::value ? (1 << 16) - 32 : (1 << 8) - 2;

  uint total_freq;
  ShiftingArray<TFreq, kNSymbol> freqs;

  void Init() {
    total_freq = kNSymbol;
#pragma unroll
    for (uint i = 0; i < kNSymbol; i++) {
      freqs[i] = 1;
    }
  }

  SymbolFrequence Update(uchar symbol) {
    auto sf = ExtractFreq(symbol);
    UpdateFreqs(symbol);
    return sf;
  }

  SymbolFrequence ExtractFreq(uchar symbol) {
    SymbolFrequence sf{0, 0, total_freq};
#pragma unroll
    for (uint j = 0; j < kNSymbol; ++j) {
      sf.cumulative_freq += freqs[j] * (j < symbol);
      if (symbol == j) {
        sf.freq = freqs[j];
      }
    }
    return sf;
  }

  void UpdateFreqs(uchar symbol) {
    bool need_norm = total_freq >= kBound;
#pragma unroll
    for (uint i = 0; i < kNSymbol; ++i) {
      if (need_norm) {
        freqs[i] = (freqs[i] >> 1) | 1;
      }
      if (symbol == i) {
        freqs[i] += kStep;
      }
    }
    if (need_norm) {
      total_freq = ((total_freq + kStep * 2 + kNSymbol * 2) >> 1);
    } else {
      total_freq = total_freq + kStep;
    }
  }
};

template <typename InPipe, typename FreqOutPipe, uint kNSymbol>
struct SimpleModelKernel {
  void operator()() const {
    bool done = false;
    SimpleModel<kNSymbol> model;
    model.Init();
    while (!done) {
      auto in = InPipe::read();
      done = in.done;
      auto f = model.Update(in.data);
      FreqOutPipe::write({f, done});
    }
  }
};

constexpr uint kRangeOutSize = 4;
struct RangeOutput {
  using IdxType = ac_int<Log2(kRangeOutSize) + 1, false>;
  IdxType size;
  ShiftingArray<uchar, kRangeOutSize> buffer;
};


template <uint num_coder>
using RangePipe =
    ext::intel::pipe<class ROutP, FlagBundle<array<RangeOutput, num_coder>>,
                     128>;

template <uint num_coder>
using RangeCarryPipe =
    ext::intel::pipe<class ROutP, FlagBundle<array<uint, num_coder>>, 128>;

using FrequncePipes =
    PipeArray<class FreqPP, FlagBundle<SymbolFrequence>, 128, 22>;


#endif  // RANGE_CODING_H_