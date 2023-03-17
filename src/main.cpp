

#include <chrono>
#include <random>
#include <vector>

#include "range_decoder.hpp"
#include "range_encoder.hpp"
#include "store.hpp"
#include "test_utils.h"

#ifdef FPGA_REPORT
constexpr uint kNSymbols = 128;
#else
constexpr uint kNSymbols = 256;
#endif

std::vector<ushort> GenerateFreqs(int n_mix = 1000) {
  ushort base = kTotalFreq / kNSymbols;
  std::vector<ushort> res(kNSymbols, base);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, kNSymbols - 1);
  for (int i = 0; i < n_mix; ++i) {
    auto from = dis(gen);
    auto to = dis(gen);
    while (res[from] < 2) {
      from = dis(gen);
    }
    std::uniform_int_distribution<> dis_moved(1, res[from] - 1);
    auto moved = dis_moved(gen);
    while (int(res[to]) + moved >= 65535) {
      to = dis(gen);
    }
    res[from] -= moved;
    res[to] += moved;
  }
  printf("freqs: ");
  int tot = 0;
  for (int i = 0; i < kNSymbols; ++i) {
    printf("%u ", res[i]);
    tot += res[i];
  }
  printf("\n");
  printf("total: %d\n", tot);
  return res;
}

using FreqT = decltype(FreqPipe<kNSymbols>::read());

int main(int argc, char** argv) {
  auto q = CreateQueue();

  struct stat fstat;
  stat(argv[1], &fstat);
  uint file_size = fstat.st_size;
  if (file_size > 3L * 1024 * 1024 * 1024) {
    throw std::runtime_error("file too large");
  }
  std::ifstream input_file(argv[1]);
  auto fq_host_buffer = std::make_unique<uchar[]>(file_size);
  input_file.read((char*)fq_host_buffer.get(), file_size);
  input_file.close();

  auto freqs = GenerateFreqs();
  std::vector<ushort> acc_freq(kNSymbols, 0);
  std::vector<ushort> cumulated_freq(kNSymbols, 0);
  ushort cumulator = 0;
  for (uint i = 0; i < kNSymbols; ++i) {
    cumulated_freq[i] = cumulator;
    acc_freq[i] = cumulator + freqs[i];
    cumulator += freqs[i];
  }
  std::vector<FreqT> freq_host(3);
  freq_host[0] = *(FreqT*)freqs.data();
  freq_host[1] = *(FreqT*)cumulated_freq.data();
  freq_host[2] = *(FreqT*)acc_freq.data();

  buffer<FreqT, 1> freq_buffer = {range<1>(3)};
  buffer<uchar, 1> fq_buffer = {range<1>(file_size)};
  q.submit([&](handler& h) {
     auto acc = fq_buffer.get_access<access::mode::write>(h);
     h.copy(fq_host_buffer.get(), acc);
   }).wait();
  q.submit([&](handler& h) {
     auto acc = freq_buffer.get_access<access::mode::write>(h);
     h.copy(freq_host.data(), acc);
   }).wait();
  // launch------------------
  printf("--launch encoding kernel--\n");
  DoubleBufferingStore<1> store(file_size);
  store.Launch(q, 0);

  q.submit([&](handler& h) {
    auto acc = freq_buffer.get_access<access::mode::read>(h);
    h.single_task<class ReadEncFreq>([=] {
      for (uint i = 0; i < 2; ++i) {
        FreqPipe<kNSymbols>::write(acc[i]);
      }
    });
  });

  q.submit([&](handler& h) {
    auto acc = fq_buffer.get_access<access::mode::read>(h);
    h.single_task<class ReadSymbols>([=] {
      for (uint i = 0; i < acc.get_size(); ++i) {
        SymInPipe::write<0>({acc[i], i == acc.get_size() - 1});
        if (i % 12800 == 0) {
          if (i != 0) {
            KERNEL_PRINTF("\033[A");
            KERNEL_PRINTF("\033[K");
          }
          KERNEL_PRINTF("encoding %.2f%%\n", i * 1.0f / acc.get_size() * 100);
        }
      }
    });
  });
  auto e_encoding = q.single_task(RangeCoder<1, kNSymbols>{});

  store.ApplyCarry(0, 0);

  auto start_enc =
      e_encoding.get_profiling_info<info::event_profiling::command_start>() *
      1.0 / 1e9;
  auto end_enc =
      e_encoding.get_profiling_info<info::event_profiling::command_end>() *
      1.0 / 1e9;
  auto thpt_enc = file_size * 1.0 / (end_enc - start_enc);

  printf("encoding thpt: %.4f M/s\n", thpt_enc / 1024 / 1024);

  printf("--launch decoding kernel--\n");

  q.submit([&](handler& h) {
    auto rc_ptr = store.rc_buffer[0][0].get_access(h);
    auto rc_size = store.rc_size_buffer[0].get_host_access()[0];
    h.single_task<class ReadRC>([=]() {
      auto code_data = rc_ptr[1];
      auto stream_init = rc_ptr[2];
      uint code_init = 0;
      uchar* p = (uchar*)&code_data;
#pragma unroll
      for (uint i = 0; i < 4; ++i) {
        code_init = code_init << 8 | p[i];
      }
      RCInitPipe::write({file_size, code_init, stream_init});
      for (uint i = 3; i < CountVecs<kRangeOutSize>(rc_size) + 2; ++i) {
        UintRCVecx2 v = 0;
        v |= rc_ptr[i];
        RCDataInPipe::write(v);
      }
    });
  });

  q.submit([&](handler& h) {
    auto acc = freq_buffer.get_access<access::mode::read>(h);
    h.single_task<class ReadDecFreq>([=] {
      for (uint i = 0; i < 3; ++i) {
        DecodingFreqPipe<kNSymbols>::write(acc[i]);
      }
    });
  });

  buffer<uchar, 1> sym_buffer = {range<1>(file_size)};
  q.submit([&](handler& h) {
    auto sym_ptr = sym_buffer.get_access(h);
    h.single_task<class StoreDecoded>([=]() {
      for (uint i = 0; i < file_size; ++i) {
        sym_ptr[i] = SymbolOutPipe::read();
        if (i % 12800 == 0) {
          if (i != 0) {
            KERNEL_PRINTF("\033[A");
            KERNEL_PRINTF("\033[K");
          }
          KERNEL_PRINTF("decoding %.2f%%\n", i * 1.0f / file_size * 100);
        }
      }
    });
  });

  auto e_decoding = q.single_task(RangeDecoder<kNSymbols>());
  auto start =
      e_decoding.get_profiling_info<info::event_profiling::command_start>() *
      1.0 / 1e9;
  auto end =
      e_decoding.get_profiling_info<info::event_profiling::command_end>() *
      1.0 / 1e9;
  auto thpt = file_size * 1.0 / (end - start);

  printf("decoding elapsed: %.4f s\n", end - start);
  printf("decoding thpt: %.4f M/s\n", thpt / 1024 / 1024);

  auto dec_ptr = sym_buffer.get_host_access().get_pointer();
  if (memcmp(dec_ptr, fq_host_buffer.get(), file_size - 1) != 0) {
    printf("decode failed\n");
    std::ofstream dec_dump(std::string(argv[1]) + ".decode-dump");
    dec_dump.write((char*)dec_ptr, file_size - 1);
    dec_dump.close();
  } else {
    printf("decode successfully\n");
  }
}