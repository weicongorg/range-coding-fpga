

#include <vector>

#include "host_decoder.hpp"
#include "range_encoder.hpp"
#include "range_decoder.hpp"
#include "store.hpp"
#include "test_utils.h"

using SymbPipe = ext::intel::pipe<class SxxxqP, FlagBundle<uchar>, 256>;
constexpr uint kNSymbols = 256;

int main(int argc, char** argv) {
  auto q = CreateQueue();

  struct stat fstat;
  stat(argv[1], &fstat);
  auto file_size = fstat.st_size;
  if (file_size > 3L * 1024 * 1024 * 1024) {
    throw std::runtime_error("file too large");
  }
  std::ifstream input_file(argv[1]);
  auto fq_host_buffer = std::make_unique<uchar[]>(file_size);
  input_file.read((char*)fq_host_buffer.get(), file_size);
  input_file.close();

  buffer<uchar, 1> fq_buffer = {range<1>(file_size)};

  q.submit([&](handler& h) {
     auto acc = fq_buffer.get_access<access::mode::write>(h);
     h.copy(fq_host_buffer.get(), acc);
   }).wait();

  // launch------------------

  q.single_task(
      SimpleModelKernel<SymbPipe, FrequncePipes::PipeAt<0>, kNSymbols>{});
  q.single_task(RangeCoder<1>{});
  DoubleBufferingStore<1> store(file_size);
  store.Launch(q, 0);

  q.submit([&](handler& h) {
    auto acc = fq_buffer.get_access<access::mode::read>(h);
    h.single_task([=] {
      for (uint i = 0; i < acc.get_size(); ++i) {
        SymbPipe::write({acc[i], i == acc.get_size() - 1});
      }
    });
  });
  store.ApplyCarry(0, 0);

  printf("-----------host deocoding\n");

  auto rc_ptr = store.rc_buffer[0][0].get_host_access().get_pointer();

  HostDecoder decoder(store.rc_buffer[0][0].get_host_access().get_pointer());
  SIMPLE_MODEL<kNSymbols> model;
  vector<uchar> decoded(file_size, 0);
  for (uint i = 0; i < file_size - 1; ++i) {
    decoded[i] = model.decodeSymbol(decoder);
  }
  if (memcmp(decoded.data(), fq_host_buffer.get(), file_size - 1) != 0) {
    printf("decode failed\n");
    printf("truth: {%.*s}\n", file_size, (char*)fq_host_buffer.get());
    printf("decode: {%.*s}\n", file_size, (char*)decoded.data());
  } else {
    printf("host decode successufully\n");
  }

  printf("-----------kernel deocoding\n");

  FreqUpdater<kNSymbols> fu;
  q.single_task([=] { fu(file_size); });
  q.single_task(RangeDecoderKernel<kNSymbols>());

  q.submit([&](handler& h) {
    auto rc_ptr = store.rc_buffer[0][0].get_access(h);
    auto rc_size = store.rc_size_buffer[0].get_host_access()[0];
    h.single_task([=]() {
      auto code_data = rc_ptr[1];
      uint code_init = 0;
      uchar* p = (uchar*)&code_data;
#pragma unroll
      for (uint i = 0; i < 4; ++i) {
        code_init = code_init << 8 | p[i];
      }
      RCInitPipe::write({file_size, code_init});
      for (uint i = 2; i < CountVecs<kRangeOutSize>(rc_size) + 2; ++i) {
        UintRCVecx2 v = 0;
        v |= rc_ptr[i];
        RCDataInPipe::write(v);
      }
    });
  });

  buffer<uchar, 1> sym_buffer = {range<1>(file_size)};
  q.submit([&](handler& h) {
     auto sym_ptr = sym_buffer.get_access(h);
     h.single_task([=]() {
       for (uint i = 0; i < file_size; ++i) {
         sym_ptr[i] = SymbolOutPipe::read();
         if (i % 12800 == 0) {
           KERNEL_PRINTF("decoding %u\n", i);
         }
       }
     });
   }).wait();

  auto dec_ptr = sym_buffer.get_host_access().get_pointer();
  if (memcmp(dec_ptr, fq_host_buffer.get(), file_size - 1) != 0) {
    printf("kernel decode failed\n");
    std::ofstream dec_dump(std::string(argv[1]) + ".decode-dump");
    dec_dump.write((char*)dec_ptr, file_size - 1);
    dec_dump.close();
    // printf("truth: {%.*s}\n", file_size, (char*)fq_host_buffer.get());
    // printf("decode: {%.*s}\n", file_size, (char*)dec_ptr);
  } else {
    printf("kernel decode successufully\n");
  }
}