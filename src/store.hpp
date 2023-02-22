#ifndef STORE_HPP_
#define STORE_HPP_
#include "range_coding.h"
#include "unrolled_loop.hpp"

template <uint num_enabled_coders, typename UintAccessorList,
          typename OutSizeAccessor>
void StoreCarry(UintAccessorList &out_accessors,
                OutSizeAccessor &size_accessor) {
  constexpr uint kNCoders = num_enabled_coders;
  uint num_locations[kNCoders];
#pragma unroll
  for (int i = 0; i < kNCoders; i++) {
    num_locations[i] = 0;
  }
  bool done = false;
  while (!done) {
    auto bundle = RangeCarryPipe<kNCoders>::read();
    done = bundle.done;
    fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
      if (bundle.data[i] != 0xffffffff) {
        out_accessors[i][num_locations[i]++] = bundle.data[i];
      }
    });
  }
  using PipelinedLSU = ext::intel::lsu<>;
  fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
    PipelinedLSU::store(size_accessor.get_pointer() + i, num_locations[i]);
  });
}
using RangeVector = decltype(RangeOutput::buffer);
using RangeVectorx2 = ShiftingArray<uchar, kRangeOutSize * 2>;

template <uint kNCoders, typename DataAccessor, typename SizeAccessor>
void Store(DataAccessor &out_accessors, SizeAccessor &size_accessor) {
  uint accessor_indices[kNCoders];
  std::array<RangeVectorx2, kNCoders> out_streams;
  ac_int<Log2(kRangeOutSize * 2) + 1, false> stream_sizes[kNCoders];

#pragma unroll
  for (int i = 0; i < kNCoders; i++) {
    stream_sizes[i] = 0;
    accessor_indices[i] = 0;
    out_streams[i].AcInt() = 0;
  }

  bool done = false;
  while (!done) {
    auto bundle = RangePipe<kNCoders>::read();
    done = bundle.done;
    fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
      RangeVectorx2 buffer;
      buffer.AcInt() = bundle.data[i].buffer.AcInt();
      out_streams[i].AcInt() |=
          buffer.ElementShift<false>(stream_sizes[i]).AcInt();
      if (stream_sizes[i] >= kRangeOutSize - bundle.data[i].size) {
        out_accessors[i][accessor_indices[i]++] = out_streams[i].AcInt();
        out_streams[i].ElementShift(kRangeOutSize);
      }
      stream_sizes[i] = (stream_sizes[i] + bundle.data[i].size) % kRangeOutSize;
    });
  }
  using PipelinedLSU = ext::intel::lsu<>;
  fpga_tools::UnrolledLoop<0, kNCoders>([&](auto i) {
    PipelinedLSU::store(
        out_accessors[i].get_pointer() + accessor_indices[i],
        static_cast<RangeVector::AcIntType>(out_streams[i].AcInt()));
    PipelinedLSU::store(
        size_accessor.get_pointer() + i,
        accessor_indices[i] * kRangeOutSize + stream_sizes[i].to_uint());
  });
}

static const property_list buffer_props{property::buffer::mem_channel{1}};
template <uchar kNCoders>
struct DoubleBufferingStore {
  template <typename T>
  struct BufferList {
    std::array<buffer<T, 1>, kNCoders> list;
    BufferList(size_t buffer_size) : list(CreateBufferArray(buffer_size)) {}
    static auto CreateBufferArray(size_t size) {
      return CreateArray<kNCoders>(
          [=](size_t i) { return buffer<T, 1>(range<1>(size), buffer_props); });
    }
    buffer<T, 1> &operator[](size_t idx) { return list[idx]; }
  };

  BufferList<RangeVector::AcIntType> rc_buffer[2];
  BufferList<uint> carry_buffer[2];
  buffer<uint, 1> rc_size_buffer[2];
  buffer<uint, 1> carry_size_buffer[2];
  event rc_event[2];
  event carry_event[2];

  DoubleBufferingStore(size_t fq_size)
      : rc_buffer{
        BufferList<RangeVector::AcIntType>(fq_size/kRangeOutSize),
        BufferList<RangeVector::AcIntType>(fq_size/kRangeOutSize),
      },
      carry_buffer{
        BufferList<uint>(fq_size/10),
        BufferList<uint>(fq_size/10),
      },
      rc_size_buffer{
        buffer<uint,1>{range<1>(kNCoders),buffer_props},
        buffer<uint,1>{range<1>(kNCoders),buffer_props},
      },
        carry_size_buffer{
        buffer<uint,1>{range<1>(kNCoders),buffer_props},
        buffer<uint,1>{range<1>(kNCoders),buffer_props},
        }
    {}

  void Launch(queue &q, bool id) {
    rc_event[id] = q.submit([&](handler &h) {
      auto acc_list = CreateArray<kNCoders>(
          [&](size_t idx) { return rc_buffer[id][idx].get_access(h); });
      auto size_acc = rc_size_buffer[id].get_access(h);
      h.single_task([=]() [[intel::kernel_args_restrict]] {
        Store<kNCoders>(acc_list, size_acc);
      });
    });
    carry_event[id] = q.submit([&](handler &h) {
      auto acc_list = CreateArray<kNCoders>(
          [&](size_t idx) { return carry_buffer[id][idx].get_access(h); });
      auto size_acc = carry_size_buffer[id].get_access(h);
      h.single_task([=]() [[intel::kernel_args_restrict]] {
        StoreCarry<kNCoders>(acc_list, size_acc);
      });
    });
  }

  void ApplyCarry(bool id, uint coder_idx) {
    uint i = coder_idx;
    auto data_buffer = rc_buffer[id][i];
    auto carrys = carry_buffer[id][i];
    uint carry_size = carry_size_buffer[id].get_host_access()[i];
    if (carry_size > 0) {
      auto carry_acc = carrys.get_host_access();
      auto data_acc = data_buffer.get_host_access();
      for (int i = 0; i < carry_size; ++i) {
        auto loc = carry_acc[i] - 1;
        while (data_acc[loc] == 0xff) {
          data_acc[loc] = 0x00;
          loc--;
        }
        data_acc[loc]++;
      }
    }
  }
};

#endif  // STORE_HPP_