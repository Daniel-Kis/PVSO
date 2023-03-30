#include "convolution.h"
#include <stdio.h>

Convolution::Convolution()
{
}

extern "C"
{
    __declspec(dllexport) double** convolute(double** kernel, double** image_padded, int hi, int wi, int hk, int wk)
    {
        double **out = (double**)malloc(hi * sizeof(double*));
        static double sum = 0;
        for (int row = 0; row < hi; row++)  {
            out[row] = (double*)malloc(wi * sizeof(double));
            for(int col = 0; col < wi; col++) {
                for(int i = 0; i < hk; i++) {
                    for(int j = 0; j < wk; j++) {
                        sum += image_padded[row+i][col+j] * kernel[i][j];
                    }
                }
                out[row][col] = sum;
                sum = 0;
            }
        }
        return out;
    }
}
