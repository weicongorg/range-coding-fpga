#ifndef SHIFTING_ARRAY_HPP_
#define SHIFTING_ARRAY_HPP_

#define AC_NOT_USING_INTN

#include <array>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "utils.hpp"
template <typename ElementType, size_t size>
struct ShiftingArray {
  using IdxType = ac_int<Log2<>(size) + 1, false>;
  using AcIntType = ac_int<size * 8 * sizeof(ElementType), false>;
  using BitIdxType = ac_int<Log2(sizeof(ElementType) * size * 8) + 1, false>;

  AcIntType& AcInt() const { return *(AcIntType*)data; }

  template <bool towards_left = true>
  ShiftingArray& ElementShift(uint n_elements) {
#if FPGA_EMULATOR
    if constexpr (towards_left) {
      for (uint i = 0; i < size; ++i) {
        if (i < size - n_elements) {
          data[i] = data[i + n_elements];
        } else {
          data[i] = ElementType{0};
        }
      }
    } else {
      for (uint i = 0; i < size; ++i) {
        if (i < size - n_elements) {
          data[size - 1 - i] = data[size - n_elements - 1 - i];
        } else {
          data[size - 1 - i] = ElementType{0};
        }
      }
    }
#else
    if constexpr (towards_left) {
      AcInt() >>= static_cast<BitIdxType>(sizeof(ElementType) * 8 * n_elements);
    } else {
      AcInt() <<= static_cast<BitIdxType>(sizeof(ElementType) * 8 * n_elements);
    }
#endif
    return *this;
  }

  template <typename Idx>
  ElementType& operator[](Idx i) {
    return data[i];
  }
  template <typename Idx>
  const ElementType& operator[](Idx i) const {
    return data[i];
  }
  ElementType& back() { return data[size - 1]; }
  ElementType& front() { return data[0]; }
  ElementType data[size];
};

#endif  // SHIFTING_ARRAY_HPP_