#ifndef CONVOLUTION_GLOBAL_H
#define CONVOLUTION_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(CONVOLUTION_LIBRARY)
#  define CONVOLUTION_EXPORT Q_DECL_EXPORT
#else
#  define CONVOLUTION_EXPORT Q_DECL_IMPORT
#endif

#endif // CONVOLUTION_GLOBAL_H
