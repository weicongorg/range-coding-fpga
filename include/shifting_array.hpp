#ifndef SHIFTING_ARRAY_HPP_
#define SHIFTING_ARRAY_HPP_

#define AC_NOT_USING_INTN

#include <array>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "utils.hpp"
template <typename ElementType, size_t size>
struct ShiftingArray : public std::array<ElementType, size> {
  using IdxType = ac_int<Log2<>(size) + 1, false>;
  using AcIntType = ac_int<size * 8 * sizeof(ElementType), false>;
  using BitIdxType = ac_int<Log2(sizeof(ElementType) * size * 8) + 1, false>;

  AcIntType& AcInt() const { return *(AcIntType*)this->data(); }

  template <bool towards_left = true>
  ShiftingArray& ElementShift(uint n_elements) {
    if constexpr (towards_left) {
      AcInt() >>= static_cast<BitIdxType>(sizeof(ElementType) * 8 * n_elements);
    } else {
      AcInt() <<= static_cast<BitIdxType>(sizeof(ElementType) * 8 * n_elements);
    }
    return *this;
  }

  ElementType& back() { return this->data()[size - 1]; }
  ElementType& front() { return this->data()[0]; }
  template <typename TArray>
  ShiftingArray& Copy(const TArray& rhs) {
    AcInt() = *(AcIntType*)rhs.data();
    return *this;
  }
};

#endif  // SHIFTING_ARRAY_HPP_