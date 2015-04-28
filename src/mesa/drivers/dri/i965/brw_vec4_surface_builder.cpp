/*
 * Copyright © 2013-2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "brw_vec4_surface_builder.h"

using namespace brw;

namespace {
   namespace array_utils {
      /**
       * A plain contiguous region of memory in your register file,
       * with well-defined size and no fancy addressing modes,
       * swizzling or striding.
       */
      struct array_reg : public backend_reg {
         array_reg() : backend_reg(), size(0)
         {
         }

         explicit
         array_reg(const backend_reg &reg, unsigned size = 1) :
            backend_reg(reg), size(size)
         {
         }

         /** Size of the region in 32B registers. */
         unsigned size;
      };

      /**
       * Increase the register base offset by the specified amount
       * given in 32B registers.
       */
      array_reg
      offset(array_reg reg, unsigned delta)
      {
         assert(delta == 0 || (reg.file != HW_REG && reg.file != IMM));
         reg.reg_offset += delta;
         return reg;
      }

      /**
       * Create a register of natural vector size and SIMD width
       * using array \p reg as storage.
       */
      dst_reg
      natural_reg(const vec4_builder &bld, const array_reg &reg)
      {
         return dst_reg(reg, WRITEMASK_XYZW);
      }

      /**
       * Allocate a raw chunk of memory from the virtual GRF file
       * with no special vector size or SIMD width.  \p n is given
       * in units of 32B registers.
       */
      array_reg
      alloc_array_reg(const vec4_builder &bld, enum brw_reg_type type,
                      unsigned n)
      {
         return array_reg(
            bld.vgrf(type,
                     DIV_ROUND_UP(n * REG_SIZE,
                                  type_sz(type) * bld.dispatch_width())),
            n);
      }

      /**
       * Fetch the i-th logical component of an array of registers and return
       * it as a natural-width register according to the current SIMD mode.
       *
       * Each logical component may be in fact a vector with a number of
       * per-channel values depending on the dispatch width and SIMD mode.
       * E.g. a single physical 32B register contains 4, 1, or 0.5 logical
       * 32-bit components depending on whether we're building SIMD4x2, SIMD8
       * or SIMD16 code respectively.
       */
      dst_reg
      index(const vec4_builder &bld, const array_reg &reg, unsigned i)
      {
         return writemask(offset(natural_reg(bld, reg), i / 4),
                          1 << (i % 4));
      }

      /**
       * "Flatten" a vector of \p size components into a simple array of
       * registers, getting rid of swizzles, strides and funky regioning
       * modes.
       */
      array_reg
      emit_flatten(const vec4_builder &bld, const src_reg &src, unsigned size)
      {
         if (src.file == BAD_FILE || size == 0) {
            return array_reg();

         } else {
            const unsigned mask = (1 << size) - 1;
            const array_reg dst = alloc_array_reg(bld, src.type, 1);

            bld.MOV(writemask(natural_reg(bld, dst), mask), src);
            if (size < 4)
               bld.MOV(writemask(natural_reg(bld, dst), ~mask), 0);

            return dst;
         }
      }

      /**
       * Copy one every \p src_stride logical components of the argument into
       * one every \p dst_stride logical components of the result.
       */
      array_reg
      emit_stride(const vec4_builder &bld, const array_reg &src, unsigned size,
                  unsigned dst_stride, unsigned src_stride)
      {
         if (src.file == BAD_FILE || size == 0) {
            return array_reg();

         } else if (dst_stride == 1 && src_stride == 1) {
            return src;

         } else {
            const unsigned n = DIV_ROUND_UP(size * dst_stride, 4);
            const array_reg dst = alloc_array_reg(bld, src.type, n);

            for (unsigned i = 0; i < size; ++i)
               bld.MOV(index(bld, dst, i * dst_stride),
                       index(bld, src, i * src_stride));

            return dst;
         }
      }

      /**
       * Interleave logical components from the given arguments.  If two
       * arguments are provided \p size components will be copied from each to
       * the even and odd components of the result respectively.
       *
       * It may be safely used to merge the two halves of a value calculated
       * separately.
       */
      array_reg
      emit_zip(const vec4_builder &bld,
               const array_reg &src0, const array_reg &src1,
               unsigned size)
      {
         const vec4_builder ubld = bld.exec_all();
         const unsigned n = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const array_reg srcs[] = { src0, src1 };
         const array_reg dst = size * n == 0 ? array_reg() :
            alloc_array_reg(bld, src0.type, DIV_ROUND_UP(size * n, 4));

         for (unsigned i = 0; i < size; ++i) {
            for (unsigned j = 0; j < n; ++j)
               ubld.MOV(index(ubld, dst, j + i * n),
                        index(ubld, srcs[j], i));
         }

         return dst;
      }

      /**
       * Concatenate a number of register arrays passed in as arguments.
       */
      array_reg
      emit_collect(const vec4_builder &bld,
                   const array_reg &header = array_reg(),
                   const array_reg &src0 = array_reg(),
                   const array_reg &src1 = array_reg(),
                   const array_reg &src2 = array_reg())
      {
         const vec4_builder ubld = bld.exec_all();
         const array_reg srcs[] = { header, src0, src1, src2 };
         const unsigned size = header.size + src0.size + src1.size + src2.size;
         const array_reg dst = size == 0 ? array_reg() :
            alloc_array_reg(bld, BRW_REGISTER_TYPE_UD, size);
         unsigned n = 0;

         for (unsigned i = 0; i < ARRAY_SIZE(srcs); ++i) {
            for (unsigned j = 0; j < srcs[i].size; ++j)
               ubld.MOV(offset(natural_reg(ubld, dst), n++),
                        retype(offset(natural_reg(ubld, srcs[i]), j),
                               BRW_REGISTER_TYPE_UD));
         }

         return dst;
      }
   }
}
