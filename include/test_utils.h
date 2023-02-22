#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_
#include <CL/sycl.hpp>
#include <array>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

using namespace cl::sycl;

template <typename TPipe>
event AutoWriter(queue &q) {
  using T = decltype(TPipe::read());
  return q.single_task([]() { TPipe::write(T()); });
}

template <typename TPipe>
void AutoReader(queue &q) {
  q.single_task([] {
    while (true) {
      TPipe::read();
    }
  });
}

template <typename TPipe, typename TArray>
struct FlagBundleFeeder {
  TArray data;
  void operator()() const {
    for (int i = 0; i < data.size(); ++i) {
      TPipe::write({data[i], i == data.size() - 1});
    }
  }
};

template <typename TPipe>
struct FlagBundleReadOut {
  buffer<decltype(decltype(TPipe::read())::data), 1> output_buffer;
  buffer<uint, 1> n_output;
  FlagBundleReadOut(uint buffer_size)
      : output_buffer(range<1>(buffer_size)), n_output(range<1>(1)) {}

  void operator()(handler &h) {
    auto acc = output_buffer.get_access(h);
    auto size_acc = n_output.get_access(h);
    h.single_task([=] {
      bool done = false;
      uint i = 0;
      while (!done) {
        auto [data, done_in] = TPipe::read();
        done = done_in;
        acc[i++] = data;
      }
      size_acc[0] = i;
    });
  }

  auto Access() { return output_buffer.get_host_access(); }
  auto Size() { return n_output.get_host_access()[0]; }
};

queue CreateQueue() {
#if FPGA_EMULATOR
  ext::intel::fpga_emulator_selector device;
#else
  ext::intel::fpga_selector device;
#endif
  auto prop_list = property_list{property::queue::enable_profiling()};
  return queue{device, fpga_tools::exception_handler, prop_list};
}

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif
#ifdef FPGA_EMULATOR
#define KERNEL_PRINTF(format, ...)                                   \
  {                                                                  \
    static const CL_CONSTANT char _format[] = format;                \
    sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }
#else
#define KERNEL_PRINTF(format, ...)
#endif

template <typename p>
void PrintHex(const p *mem, int len) {
  uchar *data = (uchar *)mem;
  for (int i = 0; i < len; ++i) {
    KERNEL_PRINTF("%2X ", data[i]);
  }
  KERNEL_PRINTF("\n");
}

template <typename vint>
void PrintMessage(const void *msg, vint len) {
  KERNEL_PRINTF("%.*s", len, (const char *)msg);
}

template <typename p>
void PrintMessage(const p *msg) {
  KERNEL_PRINTF("%s", (const char *)msg);
}

#endif