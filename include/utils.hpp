#ifndef __UTILS_HPP
#define __UTILS_HPP
#include <functional>
#include <CL/sycl.hpp>
using namespace sycl;

template <typename T>
constexpr T Log2(T n) {
  if (n < 2) {
    return T(0);
  } else {
    T ret = 0;
    while (n >= 2) {
      ret++;
      n /= 2;
    }
    return ret;
  }
}

template <uint vec_size>
inline uint CountVecs(uint size) {
  return (size + vec_size - 1) / vec_size;
}

template <int bits>
uchar NBitRepresent(uint val) {
  uchar highbit = 0;
#pragma unroll
  for (uchar i = 0; i < bits; i++) {
    if (val & (1 << i)) {
      highbit = i + 1;
    }
  }
  return highbit;
}

namespace utils_detail {

template <typename TLambda, std::size_t... index_seq>
constexpr auto CreateArray(TLambda constructor,
                           std::index_sequence<index_seq...>) {
  using TElement = decltype(constructor(0));
  return std::array<TElement, sizeof...(index_seq)>{
      {(static_cast<void>(index_seq), constructor(index_seq))...}};
}
}  // namespace utils_detail

template <size_t n_element, typename TLambda>
constexpr auto CreateArray(TLambda lambda) {
  static_assert(std::is_invocable<TLambda, size_t>::value);
  auto indices = std::make_index_sequence<n_element>();
  return utils_detail::CreateArray(lambda, indices);
}

template <typename T>
struct FlagBundle {
  T data;
  bool done;
  FlagBundle() : done(false), data() {}
  FlagBundle(T d_in, bool f_in) : data(d_in), done(f_in) {}
  FlagBundle(bool f_in) : data(), done(f_in) {}
};

template <typename Pipe, typename ExpectDataT>
struct PipeInterface {
  static_assert(std::is_same<ExpectDataT, decltype(Pipe::read())>());

  static const auto& read = Pipe::read;
  static void write(const ExpectDataT& t) { Pipe::write(t); }
  static void write(const ExpectDataT& t, bool& success) {
    Pipe::write(t, success);
  }
};

template <typename Action>
struct ForeverRun {
  void operator()() const {
    Action t;
    [[intel::disable_loop_pipelining]]  //
    while (true) {
      t();
    }
  }
};

#endif
