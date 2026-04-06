#ifndef __NUMPY_FFT_H
#define __NUMPY_FFT_H

#include "fft_internal.h"

namespace np::fft
{
    template <np::dtype T>
    class Fft : public np::Interface::FftInternal<T>
    {
      public:
        Fft();
    };
}
#endif