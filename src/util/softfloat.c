/*
 * License for Berkeley SoftFloat Release 3e
 *
 * John R. Hauser
 * 2018 January 20
 *
 * The following applies to the whole of SoftFloat Release 3e as well as to
 * each source file individually.
 *
 * Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the
 * University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions, and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions, and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the University nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * The functions listed in this file are modified versions of the ones
 * from the Berkeley SoftFloat 3e Library.
 */

#include "rounding.h"
#include "softfloat.h"

#if defined(BIG_ENDIAN)
#define indexWord( total, n ) ((total) - 1 - (n))
#else
#define indexWord( total, n ) (n)
#endif

typedef union { double f; int64_t i; uint64_t u; } fi_type;

const uint8_t count_leading_zeros8[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*
 * Shifts 'a' right by the number of bits given in 'dist', which must not
 * be zero.  If any nonzero bits are shifted off, they are "jammed" into the
 * least-significant bit of the shifted value by setting the least-significant
 * bit to 1.  This shifted-and-jammed value is returned.
 * The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
 * greater than 64, the result will be either 0 or 1, depending on whether 'a'
 * is zero or nonzero.
 */
static inline
uint64_t _mesa_shift_right_jam64( uint64_t a, uint32_t dist )
{
    return
        (dist < 63) ? a>>dist | ((uint64_t) (a<<(-dist & 63)) != 0) : (a != 0);
}

static inline
double _mesa_roundtozero_f64(int64_t s, int64_t e, int64_t m)
{
   fi_type result;

   if ((uint64_t)e >= 0x7FD) {
      if (e < 0) {
         m = _mesa_shift_right_jam64(m, -e);
         e = 0;
      } else if ((e > 0x7FD) || (0x8000000000000000 <= m)) {
         e = 0x7FF;
         m = 0;
         result.u = (s << 63) + (e << 52) + m ;
         result.u -= 1;
         return result.f;
      }
   }

   m >>= 10;
   if (m == 0)
      e = 0;

   result.u = (s << 63) + (e << 52) + m;
   return result.f;
}

/* Calculate a + b but rounding to zero.
 *
 * From f64_add()
 */
double
_mesa_double_add_rtz(double a, double b)
{
   const fi_type a_fi = {a};
   uint64_t a_flt_m = a_fi.u & 0x0fffffffffffff;
   uint64_t a_flt_e = (a_fi.u >> 52) & 0x7ff;
   uint64_t a_flt_s = (a_fi.u >> 63) & 0x1;
   const fi_type b_fi = {b};
   uint64_t b_flt_m = b_fi.u & 0x0fffffffffffff;
   uint64_t b_flt_e = (b_fi.u >> 52) & 0x7ff;
   uint64_t b_flt_s = (b_fi.u >> 63) & 0x1;
   int64_t s, e, m = 0;

   s = a_flt_s;

   const int64_t exp_diff = a_flt_e - b_flt_e;

   /* Handle special cases */

   if ((a_flt_e == 0) && (a_flt_m == 0)) {
      /* 'a' is zero, return 'b' */
      return b;
   } else if ((b_flt_e == 0) && (b_flt_m == 0)) {
      /* 'b' is zero, return 'a' */
      return a;
   } else if (a_flt_e == 0x7ff && a_flt_m != 0) {
      /* 'a' is a NaN, return NaN */
      return a;
   } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
      /* 'b' is a NaN, return NaN */
      return b;
   } else if (a_flt_e == 0x7ff && a_flt_m == 0) {
      /* 'a' is infinity, return infinity */
      return a;
   } else if (b_flt_e == 0x7ff && b_flt_m == 0) {
      /* 'b' is infinity, return infinity */
      return b;
   } else if (a_flt_s != b_flt_s) {
      return _mesa_double_sub_rtz(a, -b);
   } else if (exp_diff == 0 && a_flt_e == 0) {
      uint64_t tmp = a_fi.u + b_flt_m;
      m = tmp & 0x0fffffffffffff;
      e = (tmp >> 52) & 0x7ff;
   } else if (exp_diff == 0) {
      e = a_flt_e;
      m = 0x0020000000000000 + a_flt_m + b_flt_m;
      m <<= 9;
   } else if (exp_diff < 0) {
      a_flt_m <<= 9;
      b_flt_m <<= 9;
      e = b_flt_e;

      if (a_flt_e != 0)
         a_flt_m += 0x0020000000000000;
      else
         a_flt_m <<= 1;

      a_flt_m = _mesa_shift_right_jam64(a_flt_m, -exp_diff);
      m = 0x2000000000000000 + a_flt_m + b_flt_m;
      if (m < 0x4000000000000000) {
         --e;
         m <<= 1;
      }
   } else {
      a_flt_m <<= 9;
      b_flt_m <<= 9;
      e = a_flt_e;

      if (a_flt_e != 0)
         b_flt_m += 0x0020000000000000;
      else
         b_flt_m <<= 1;

      b_flt_m = _mesa_shift_right_jam64(b_flt_m, -exp_diff);
      m = 0x2000000000000000 + a_flt_m + b_flt_m;
      if (m < 0x4000000000000000) {
         --e;
         m <<= 1;
      }
   }

   return _mesa_roundtozero_f64(s, e, m);
}

static inline unsigned
_mesa_count_leading_zeros64( uint64_t a )
{
    unsigned count;
    uint32_t a32;

    count = 0;
    a32 = a >> 32;
    if (!a32) {
        count = 32;
        a32 = a;
    }
    /*------------------------------------------------------------------------
    | From here, result is current count + count leading zeros of `a32'.
    *------------------------------------------------------------------------*/
    if ( a32 < 0x10000 ) {
        count += 16;
        a32 <<= 16;
    }
    if ( a32 < 0x1000000 ) {
        count += 8;
        a32 <<= 8;
    }
    count += count_leading_zeros8[a32>>24];
    return count;
}

static inline double
_mesa_norm_round_pack_f64( int64_t s, int64_t e, int64_t m )
{
    int8_t shift_dist;

    shift_dist = _mesa_count_leading_zeros64( m ) - 1;
    e -= shift_dist;
    if ( (10 <= shift_dist) && ((unsigned int) e < 0x7FD) ) {
        fi_type result;
        result.u = (s << 63) + ((m ? e : 0) << 52) + (m << (shift_dist - 10));
        return result.f;
    } else {
       return _mesa_roundtozero_f64(s, e, m);
    }
}

/* Calculate a - b but rounding to zero.
 *
 * From f64_sub()
 */
double
_mesa_double_sub_rtz(double a, double b)
{
   const fi_type a_fi = {a};
   uint64_t a_flt_m = a_fi.u & 0x0fffffffffffff;
   uint64_t a_flt_e = (a_fi.u >> 52) & 0x7ff;
   uint64_t a_flt_s = (a_fi.u >> 63) & 0x1;
   const fi_type b_fi = {b};
   uint64_t b_flt_m = b_fi.u & 0x0fffffffffffff;
   uint64_t b_flt_e = (b_fi.u >> 52) & 0x7ff;
   uint64_t b_flt_s = (b_fi.u >> 63) & 0x1;
   int64_t s, e, m = 0;
   int64_t m_diff = 0;
   unsigned shift_dist = 0;

   s = a_flt_s;

   const int64_t exp_diff = a_flt_e - b_flt_e;

   /* Handle special cases */

   if ((a_flt_e == 0) && (a_flt_m == 0)) {
      /* 'a' is zero, return '-b' */
      return -b;
   } else if ((b_flt_e == 0) && (b_flt_m == 0)) {
      /* 'b' is zero, return 'a' */
      return a;
   } else if (a_flt_e == 0x7ff && a_flt_m != 0) {
      /* 'a' is a NaN, return NaN */
      return a;
   } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
      /* 'b' is a NaN, return NaN */
      return b;
   } else if (a_flt_e == 0x7ff && a_flt_m == 0) {
      /* 'a' is infinity, return infinity */
      return a;
   } else if (b_flt_e == 0x7ff && b_flt_m == 0) {
      /* 'b' is infinity, return -infinity */
      return -b;
   } else if (a_flt_s != b_flt_s) {
      return _mesa_double_add_rtz(a, -b);
   } else if (exp_diff == 0 && a_flt_e == 0) {
      uint64_t tmp = a_fi.u + b_flt_m;
      m = tmp & 0x0fffffffffffff;
      e = (tmp >> 52) & 0x7ff;
   } else if (exp_diff == 0) {
      m_diff = a_flt_m - b_flt_m;

      if (m_diff == 0)
         return 0;
      if (a_flt_e)
         a_flt_e--;
      if (m_diff < 0) {
         s = !s;
         m_diff = -m_diff;
      }

      shift_dist = _mesa_count_leading_zeros64(m_diff) - 11;
      e = a_flt_e - shift_dist;
      return _mesa_roundtozero_f64(s, e, m_diff << shift_dist);
   } else if (exp_diff < 0) {
      a_flt_m <<= 10;
      b_flt_m <<= 10;
      s = !s;

      a_flt_m += (a_flt_e) ? 0x4000000000000000 : a_flt_m;
      a_flt_m = _mesa_shift_right_jam64(a_flt_m, -exp_diff);
      b_flt_m |= 0x4000000000000000;
      e = b_flt_e;
      m = b_flt_m - a_flt_m;
   } else {
      a_flt_m <<= 10;
      b_flt_m <<= 10;

      b_flt_m += (b_flt_e) ? 0x4000000000000000 : b_flt_m;
      b_flt_m = _mesa_shift_right_jam64(b_flt_m, -exp_diff);
      a_flt_m |= 0x4000000000000000;
      e = a_flt_e;
      m = a_flt_m - b_flt_m;
   }

   return _mesa_norm_round_pack_f64(s, e - 1, m);
}

static inline void
_mesa_norm_subnormal_mantissa_f64(uint64_t m, uint64_t *exp, uint64_t *m_out)
{
    unsigned shift_dist;

    shift_dist = _mesa_count_leading_zeros64( m ) - 11;
    *exp = 1 - shift_dist;
    *m_out = m << shift_dist;
}

static inline void
mesa_softfloat_mul64To128M( uint64_t a, uint64_t b, uint32_t *zPtr )
{
    uint32_t a32, a0, b32, b0;
    uint64_t z0, mid1, z64, mid;

    a32 = a >> 32;
    a0 = a;
    b32 = b >> 32;
    b0 = b;
    z0 = (uint64_t) a0 * b0;
    mid1 = (uint64_t) a32 * b0;
    mid = mid1 + (uint64_t) a0 * b32;
    z64 = (uint64_t) a32 * b32;
    z64 += (uint64_t) (mid < mid1) <<32 | mid >>32;
    mid <<= 32;
    z0 += mid;
    zPtr[indexWord( 4, 1 )] = z0 >> 32;
    zPtr[indexWord( 4, 0 )] = z0;
    z64 += (z0 < mid);
    zPtr[indexWord( 4, 3 )] = z64 >> 32;
    zPtr[indexWord( 4, 2 )] = z64;
}

/* Calculate a * b but rounding to zero.
 *
 * From f64_mul()
 */
double
_mesa_double_mul_rtz(double a, double b)
{
   const fi_type a_fi = {a};
   uint64_t a_flt_m = a_fi.u & 0x0fffffffffffff;
   uint64_t a_flt_e = (a_fi.u >> 52) & 0x7ff;
   uint64_t a_flt_s = (a_fi.u >> 63) & 0x1;
   const fi_type b_fi = {b};
   uint64_t b_flt_m = b_fi.u & 0x0fffffffffffff;
   uint64_t b_flt_e = (b_fi.u >> 52) & 0x7ff;
   uint64_t b_flt_s = (b_fi.u >> 63) & 0x1;
   int64_t s, e, m = 0;

   s = a_flt_s ^ b_flt_s;

   if (a_flt_e == 0x7ff) {
      if (a_flt_m != 0) {
         /* 'a' is a NaN, return NaN */
         return a;
      } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
         /* 'b' is a NaN, return NaN */
         return b;
      }

      if (!(b_flt_e | b_flt_m)) {
         /* Inf * 0 = NaN */
         fi_type result;
         e = 0x7ff;
         result.u = (s << 63) + (e << 52) + 0x1;
         return result.f;
      }
      /* Inf * x = Inf */
      fi_type result;
      e = 0x7ff;
      result.u = (s << 63) + (e << 52) + 0;
      return result.f;
   }

   if (b_flt_e == 0x7ff) {
      if (b_flt_m != 0) {
         /* 'b' is a NaN, return NaN */
         return b;
      }
      if (!(a_flt_e | a_flt_m)) {
         /* 0 * Inf = NaN */
         fi_type result;
         e = 0x7ff;
         result.u = (s << 63) + (e << 52) + 0x1;
         return result.f;
      }
      /* x * Inf = Inf */
      fi_type result;
      e = 0x7ff;
      result.u = (s << 63) + (e << 52) + 0;
      return result.f;
   }

   if (a_flt_e == 0) {
      if (a_flt_m == 0) {
         /* 'a' is zero. Return zero */
         fi_type result;
         result.u = (s << 63) + 0;
         return result.f;
      }
      _mesa_norm_subnormal_mantissa_f64( a_flt_m , &a_flt_e, &a_flt_m);
   }
   if (b_flt_e == 0) {
      if (b_flt_m == 0) {
         /* 'a' is zero. Return zero */
         fi_type result;
         result.u = (s << 63) + 0;
         return result.f;
      }
      _mesa_norm_subnormal_mantissa_f64( b_flt_m , &b_flt_e, &b_flt_m);
   }

   e = a_flt_e + b_flt_e - 0x3FF;
   a_flt_m = (a_flt_m | 0x0010000000000000) << 10;
   b_flt_m = (b_flt_m | 0x0010000000000000) << 11;

   uint32_t m_128[4];
   mesa_softfloat_mul64To128M(a_flt_m, b_flt_m, m_128);

   m = (uint64_t) m_128[indexWord( 4, 3 )] <<32 | m_128[indexWord( 4, 2 )];
   if (m_128[indexWord( 4, 1 )] || m_128[indexWord( 4, 0 )])
      m |= 1;

   if (m < 0x4000000000000000) {
      e--;
      m <<= 1;
   }
   return _mesa_roundtozero_f64(s, e, m);
}

double
_mesa_double_fma_rtz(double a, double b, double c)
{
   return _mesa_double_add_rtz(_mesa_double_mul_rtz(a, b), c);
}

/**
 * \brief Rounds \c x to zero, and returns the value as a long int.
 */
long
_mesa_lroundtozerof(float x)
{
   return _mesa_lroundtozero((double) x);
}

/**
 * \brief Rounds \c x to zero, and returns the value as a long int.
 *
 * From f64_to_i64_r_minMag()
 */
long
_mesa_lroundtozero(double x)
{
   const fi_type x_fi = {x};
   uint64_t m = x_fi.u & 0x0fffffffffffff;
   uint64_t e = (x_fi.u >> 52) & 0x7ff;
   uint64_t s = (x_fi.u >> 63) & 0x1;
   int shift_dist = 0x433 - e;
   int64_t abs_out;

   if (shift_dist <= 0) {
      if (shift_dist < -10) {
         /* NaN or overflow, return NaN */
         return -((int64_t) 0x7FFFFFFFFFFFFFFF) - 1;
      }
      m |= 0x0010000000000000;
      abs_out = m << -shift_dist;
   } else {
      if (shift_dist >= 53)
         return 0;

      m |= 0x0010000000000000;
      abs_out = m >> shift_dist;
   }
   return s ? -abs_out : abs_out;
}
