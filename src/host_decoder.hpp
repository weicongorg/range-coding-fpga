#ifndef HOST_DECODER_HPP
#define HOST_DECODER_HPP
#include <vector>

#include "range_encoder.hpp"
using std::vector;

struct HostDecoder {
  uint code;
  uint range;
  uchar *in_buf;

  HostDecoder(void *rc_ptr) {
    in_buf = (uchar *)rc_ptr;
    range = (uint)-1;
    for (uint i = 0; i < 8; ++i) {
      uchar c = *in_buf++;
      code = (code << 8) | c;
    }
  }

  uint GetFreq(uint totFreq) {
    auto t1 = 1.0f / totFreq;
    uint t2 = *(uint *)&t1;
    range = ShiftDivide(range, *(uint *)&t1);
    return code / range;
  }

  void Decode(uint cumFreq, uint freq, uint totFreq) {
    code -= cumFreq * range;
    range *= freq;

    bool b_range_8bits = (range & 0xffffff00) == 0;
    bool b_range_16bits = (range & 0xffff0000) == 0;
    bool b_range_24bits = (range & 0xff000000) == 0;

    if (b_range_8bits) {
      code = (code << 8) | *in_buf++;
      code = (code << 8) | *in_buf++;
      code = (code << 8) | *in_buf++;
      range <<= 24;
    } else if (b_range_16bits) {
      code = (code << 8) | *in_buf++;
      code = (code << 8) | *in_buf++;
      range <<= 16;
    } else if (b_range_24bits) {
      code = (code << 8) | *in_buf++, range <<= 8;
    }
  }
};
uint MAX_FREQ = SimpleModel<1>::kBound;
uchar STEP = SimpleModel<1>::kStep;

template <int NSYM>
struct SIMPLE_MODEL {
  struct SymFreqs {
    uint16_t Symbol;
    uint16_t Freq;
  } F[NSYM];
  uint32_t TotFreq;

  SIMPLE_MODEL() {
    for (int i = 0; i < NSYM; i++) {
      F[i].Symbol = i;
      F[i].Freq = 1;
    }
    TotFreq = NSYM;
  }
  uchar decodeSymbol(HostDecoder &rc, bool use_legacy_norm = false) {
    SymFreqs *s = F;
    uint32_t freq = rc.GetFreq(TotFreq);
    uint32_t AccFreq;
    for (AccFreq = 0; (AccFreq += s->Freq) <= freq; s++)
      ;
    AccFreq -= s->Freq;
    auto symb = s->Symbol;
    rc.Decode(AccFreq, s->Freq, TotFreq);

    if (use_legacy_norm) {
      bool needNorm = (TotFreq) > (MAX_FREQ - STEP);
      for (int i = 0; i < NSYM; ++i) {
        ushort curr_freq = F[i].Freq;
        ushort updated_freq;
        if (needNorm) {
          if (symb == i) {
            updated_freq = ((curr_freq + STEP) >> 1) | 1;
          } else {
            updated_freq = ((curr_freq) >> 1) | 1;
          }
        } else {
          if (symb == i) {
            updated_freq = curr_freq + STEP;
          } else {
            updated_freq = curr_freq;
          }
        }
        F[i].Freq = updated_freq;
      }
      if (needNorm) {
        TotFreq = ((TotFreq + STEP + NSYM * 2) >> 1);
      } else {
        TotFreq = TotFreq + STEP;
      }
    } else {
      bool need_norm = TotFreq >= MAX_FREQ;
      for (uint i = 0; i < NSYM; ++i) {
        if (need_norm) {
          F[i].Freq = (F[i].Freq >> 1) | 1;
        }
        if (symb == i) {
          F[i].Freq += STEP;
        }
      }
      if (need_norm) {
        TotFreq = ((TotFreq + STEP * 2 + NSYM * 2) >> 1);
      } else {
        TotFreq = TotFreq + STEP;
      }
    }
    return symb;
  }
};

#endif  // RANGE_DECODING_HPP