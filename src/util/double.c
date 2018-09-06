/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2018 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "rounding.h"
#include "double.h"
#include "softfloat.h"

typedef union { double f; int64_t i; uint64_t u; } fi_type;

/**
 * Convert a 8-byte double to a 4-byte float.
 *
 * Not all float64 values can be represented exactly as a float32 value. We
 * round such intermediate float64 values to the nearest float32. When the
 * float64 lies exactly between two float32 values, we round to the one with
 * an even mantissa.
 */

float
_mesa_double_to_float(double val)
{
   const fi_type fi = {val};
   const int64_t flt_m = fi.i & 0x0fffffffffffff;
   const int64_t flt_e = (fi.i >> 52) & 0x7ff;
   const int64_t flt_s = (fi.i >> 63) & 0x1;
   int s, e, m = 0;
   float result;

   /* sign bit */
   s = flt_s;

   /* handle special cases */
   if ((flt_e == 0) && (flt_m == 0)) {
      /* zero */
      /* m = 0; - already set */
      e = 0;
   }
   else if ((flt_e == 0) && (flt_m != 0)) {
      /* denorm -- denorm float64 maps to 0 */
      /* m = 0; - already set */
      e = 0;
   }
   else if ((flt_e == 0x7ff) && (flt_m == 0)) {
      /* infinity */
      /* m = 0; - already set */
      e = 255;
   }
   else if ((flt_e == 0x7ff) && (flt_m != 0)) {
      /* NaN */
      m = 1;
      e = 255;
   }
   else {
      /* regular number */
      const int new_exp = flt_e - 1023;
      if (new_exp < -126) {
         /* The float64 lies in the range (0.0, min_normal32) and is rounded
          * to a nearby float32 value. The result will be either zero, subnormal,
          * or normal.
          */
         e = 0;
         m = _mesa_lroundeven(((double)((uint64_t)1 << 54)) * fabs(fi.f));
      }
      else if (new_exp > 127) {
         /* map this value to infinity */
         /* m = 0; - already set */
         e = 255;
      }
      else {
         /* The float64 lies in the range
          *   [min_normal32, max_normal32 + max_step32)
          * and is rounded to a nearby float32 value. The result will be
          * either normal or infinite.
          */
         e = new_exp + 127;
         m = _mesa_lroundeven((double)flt_m / (double) (1 << 29));
      }
   }

   assert(0 <= m && m <= (1 << 23));
   if (m == (1 << 23)) {
      /* The float64 was rounded upwards into the range of the next exponent,
       * so bump the exponent. This correctly handles the case where f64
       * should be rounded up to float32 infinity.
       */
      ++e;
      m = 0;
   }

   unsigned result_int = (s << 31) | (e << 23) | m;
   memcpy(&result, &result_int, sizeof(float));
   return result;
}

float
_mesa_double_to_float_rtz(double val)
{
   const fi_type fi = {val};
   const int64_t flt_m = fi.i & 0x0fffffffffffff;
   const int64_t flt_e = (fi.i >> 52) & 0x7ff;
   const int64_t flt_s = (fi.i >> 63) & 0x1;
   int s, e, m = 0;
   float result;

   /* sign bit */
   s = flt_s;

   /* handle special cases */
   if ((flt_e == 0) && (flt_m == 0)) {
      /* zero */
      /* m = 0; - already set */
      e = 0;
   }
   else if ((flt_e == 0) && (flt_m != 0)) {
      /* denorm -- denorm float64 maps to 0 */
      /* m = 0; - already set */
      e = 0;
   }
   else if ((flt_e == 0x7ff) && (flt_m == 0)) {
      /* infinity */
      /* m = 0; - already set */
      e = 255;
   }
   else if ((flt_e == 0x7ff) && (flt_m != 0)) {
      /* NaN */
      m = 1;
      e = 255;
   }
   else {
      /* regular number */
      const int new_exp = flt_e - 1023;
      if (new_exp < -126) {
         /* The float64 lies in the range (0.0, min_normal32) and is rounded
          * to a nearby float32 value. The result will be either zero, subnormal,
          * or normal.
          */
         e = 0;
         m = _mesa_lroundtozero((double)((uint64_t)1 << 54) * fabs(fi.f));
      }
      else if (new_exp > 127) {
         /* map this value to infinity */
         /* m = 0; - already set */
         e = 255;
      }
      else {
         /* The float64 lies in the range
          *   [min_normal32, max_normal32 + max_step32)
          * and is rounded to a nearby float32 value. The result will be
          * either normal or infinite.
          */
         e = new_exp + 127;
         m = _mesa_lroundtozero((double)flt_m / (double) (1 << 29));
      }
   }

   assert(0 <= m && m <= (1 << 23));
   if (m == (1 << 23)) {
      /* The float64 was rounded upwards into the range of the next exponent,
       * so bump the exponent. This correctly handles the case where f64
       * should be rounded up to float32 infinity.
       */
      ++e;
      m = 0;
   }

   unsigned result_int = (s << 31) | (e << 23) | m;
   memcpy(&result, &result_int, sizeof(float));
   return result;
}
