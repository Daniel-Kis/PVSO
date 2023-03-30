#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "convolution_global.h"

class CONVOLUTION_EXPORT Convolution
{
public:
    Convolution();
    double** convolute(double** kernel, double** image_padded, int hi, int wi, int hk, int wk);
};

#endif // CONVOLUTION_H
