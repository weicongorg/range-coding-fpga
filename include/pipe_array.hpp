#ifndef __PIPE_ARRAY_H__
#define __PIPE_ARRAY_H__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace cl::sycl;

template <size_t dim1, size_t... dims>
struct VerifierDimLayer {
  template <size_t idx1, size_t... idxs>
  struct VerifierIdxLayer {
    static constexpr bool IsValid() {
      return idx1 < dim1 &&
             (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                 idxs...>::IsValid());
    }
  };
};

template <size_t dim>
struct VerifierDimLayer<dim> {
  template <size_t idx>
  struct VerifierIdxLayer {
    static constexpr bool IsValid() { return idx < dim; }
  };
};

template <class Id, typename BaseTy, size_t depth, size_t... dims>
struct PipeArray {
 private:
  PipeArray() = delete;

  template <size_t... idxs>
  struct StructId;

  template <size_t... idxs>
  struct VerifyIndices {
    static_assert(sizeof...(idxs) == sizeof...(dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<dims...>::template VerifierIdxLayer<
                      idxs...>::IsValid(),
                  "Index out of bounds");
    using VerifiedPipe =
        cl::sycl::ext::intel::pipe<StructId<idxs...>, BaseTy, depth>;
  };

  template <uint current>
  static BaseTy readAt(uint i) {
    BaseTy ret;
    if (i == current) {
      ret = PipeAt<current>::read();
    } else {
      if constexpr (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                        current + 1>::IsValid()) {
        ret = readAt<current + 1>(i);
      }
    }
    return ret;
  }

  template <size_t firstIdx, uint current>
  static BaseTy readAt(uint i) {
    BaseTy ret;
    if (i == current) {
      ret = PipeAt<firstIdx, current>::read();
    } else {
      if constexpr (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                        firstIdx, current + 1>::IsValid()) {
        ret = readAt<firstIdx, current + 1>(i);
      }
    }
    return ret;
  }

  template <size_t firstIdx, uint current>
  static void writeAt(uint i, const BaseTy& t) {
    if (i == current) {
      PipeAt<firstIdx, current>::write(t);
      return;
    } else {
      if constexpr (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                        firstIdx, current + 1>::IsValid()) {
        writeAt<firstIdx, current + 1>(i, t);
      }
    }
  }

 public:
  using DataType = BaseTy;

  template <size_t... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;

  template <size_t... idxs>
  SYCL_EXTERNAL static BaseTy read() {
    return PipeAt<idxs...>::read();
  }
  //?
  template <size_t... idxs>
  SYCL_EXTERNAL static BaseTy read(bool &success) {
    return PipeAt<idxs...>::read(success);
  }
  template <size_t... idxs>
  SYCL_EXTERNAL static void write(const BaseTy& val) {
    PipeAt<idxs...>::write(val);
  }

  // read with compile-time-unknown index for 1D array
  SYCL_EXTERNAL static BaseTy read(uint idx) { return readAt<0>(idx); }
  // read with compile-time-unknown index for 2D array and with first index
  // confirmed
  template <size_t firstIdx>
  SYCL_EXTERNAL static BaseTy readDynamic2ndDim(uint secondIdx) {
    return readAt<firstIdx, 0>(secondIdx);
  }

  template <size_t firstIdx>
  SYCL_EXTERNAL static void writeDynamic2ndDim(uint secondIdx, const BaseTy& t) {
    writeAt<firstIdx, 0>(secondIdx, t);
  }
};

#endif  // __PIPE_ARRAY_H__
