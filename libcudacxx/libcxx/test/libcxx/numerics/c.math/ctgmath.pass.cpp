//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ctgmath>

#include <ctgmath>

#include "test_macros.h"

#ifndef _LIBCUDACXX_VERSION
#error _LIBCUDACXX_VERSION not defined
#endif

int main(int, char**)
{
    std::complex<double> cd;
    ((void)cd);
    double x = std::sin(0);
    ((void)x);

  return 0;
}