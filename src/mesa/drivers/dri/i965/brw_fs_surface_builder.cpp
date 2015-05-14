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

#include "brw_fs_surface_builder.h"

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
      fs_reg
      natural_reg(const fs_builder &bld, const array_reg &reg)
      {
         return fs_reg(reg, bld.dispatch_width());
      }

      /**
       * Allocate a raw chunk of memory from the virtual GRF file
       * with no special vector size or SIMD width.  \p n is given
       * in units of 32B registers.
       */
      array_reg
      alloc_array_reg(const fs_builder &bld, enum brw_reg_type type, unsigned n)
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
      fs_reg
      index(const fs_builder &bld, const array_reg &reg, unsigned i)
      {
         return offset(natural_reg(bld, reg), i);
      }

      /**
       * "Flatten" a vector of \p size components into a simple array of
       * registers, getting rid of funky regioning modes.
       */
      array_reg
      emit_flatten(const fs_builder &bld, const fs_reg &src, unsigned size)
      {
         if (src.file == BAD_FILE || size == 0) {
            return array_reg();

         } else {
            const array_reg dst =
               alloc_array_reg(bld, src.type, size * bld.dispatch_width() / 8);

            for (unsigned c = 0; c < size; ++c)
               bld.MOV(index(bld, dst, c), offset(src, c));

            return dst;
         }
      }

      /**
       * Copy one every \p src_stride logical components of the argument into
       * one every \p dst_stride logical components of the result.
       */
      array_reg
      emit_stride(const fs_builder &bld, const array_reg &src, unsigned size,
                  unsigned dst_stride, unsigned src_stride)
      {
         if (src.file == BAD_FILE || size == 0) {
            return array_reg();

         } else if (dst_stride == 1 && src_stride == 1) {
            return src;

         } else {
            const array_reg dst = alloc_array_reg(
               bld, src.type,
               size * dst_stride * bld.dispatch_width() / 8);

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
      emit_zip(const fs_builder &bld,
               const array_reg &src0, const array_reg &src1,
               unsigned size)
      {
         const unsigned n = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const array_reg srcs[] = { src0, src1 };
         const array_reg dst = size * n == 0 ? array_reg() :
            alloc_array_reg(bld, src0.type,
                             size * n * bld.dispatch_width() / 8);

         for (unsigned i = 0; i < size; ++i) {
            for (unsigned j = 0; j < n; ++j)
               exec_all(bld.MOV(index(bld, dst, j + i * n),
                                index(bld, srcs[j], i)));
         }

         return dst;
      }

      /**
       * Concatenate a number of register arrays passed in as arguments.
       */
      array_reg
      emit_collect(const fs_builder &bld,
                   const array_reg &header = array_reg(),
                   const array_reg &src0 = array_reg(),
                   const array_reg &src1 = array_reg(),
                   const array_reg &src2 = array_reg())
      {
         const array_reg srcs[] = { header, src0, src1, src2 };
         const unsigned size = header.size + src0.size + src1.size + src2.size;
         const array_reg dst = size == 0 ? array_reg() :
            alloc_array_reg(bld, BRW_REGISTER_TYPE_UD, size);
         fs_reg *const components = new fs_reg[size];
         unsigned n = 0;

         for (unsigned i = 0; i < ARRAY_SIZE(srcs); ++i) {
            /* Split the array in m elements of the correct width. */
            const unsigned width = (i == 0 ? 8 : bld.dispatch_width());
            const unsigned m = srcs[i].size * 8 / width;

            /* Get a builder of the same width. */
            const fs_builder ubld =
               (width == bld.dispatch_width() ? bld : bld.half(0));

            for (unsigned j = 0; j < m; ++j)
               components[n++] = retype(index(ubld, srcs[i], j),
                                        BRW_REGISTER_TYPE_UD);
         }

         bld.LOAD_PAYLOAD(natural_reg(bld, dst), components,
                          n, header.size);

         delete[] components;
         return dst;
      }

      /**
       * Description of the layout of a vector when stored in a message
       * payload in the form required by the recipient shared unit.
       */
      struct vector_layout {
         /**
          * Construct a vector_layout based on the current SIMD mode and
          * whether the target shared unit supports SIMD16 messages.
          */
         vector_layout(const fs_builder &bld, bool has_simd16) :
            halves(!has_simd16 && bld.dispatch_width() == 16 ? 2 : 1)
         {
         }

         /**
          * Number of reduced SIMD width vectors the original vector has to be
          * divided into.  It will be equal to one if the execution dispatch
          * width is natively supported by the shared unit.
          */
         unsigned halves;
      };

      /**
       * Convert a vector into an array of registers with the layout expected
       * by the recipient shared unit.  \p i selects the half of the payload
       * that will be returned.
       */
      array_reg
      emit_insert(const vector_layout &layout,
                  const fs_builder &bld,
                  const fs_reg &src,
                  unsigned size, unsigned i = 0)
      {
         assert(i < layout.halves);
         const array_reg tmp = emit_flatten(bld, src, size);

         if (layout.halves > 1 && tmp.file != BAD_FILE)
            return emit_stride(bld.half(i), offset(tmp, i),
                               size, 1, layout.halves);
         else
            return tmp;
      }

      /**
       * Convert an array of registers back into a vector according to the
       * layout expected from some shared unit.  The \p srcs array should
       * contain the halves of the payload as individual array registers.
       */
      fs_reg
      emit_extract(const vector_layout &layout,
                   const fs_builder &bld,
                   const array_reg srcs[],
                   unsigned size)
      {
         if (layout.halves > 1 &&
             srcs[0].file != BAD_FILE && srcs[1].file != BAD_FILE)
            return natural_reg(bld,
                               emit_zip(bld.half(0), srcs[0], srcs[1], size));
         else
            return natural_reg(bld, srcs[0]);
      }
   }
}

namespace brw {
   namespace surface_access {
      namespace {
         using namespace array_utils;

         /**
          * Generate a send opcode for a surface message and return the
          * result.
          */
         array_reg
         emit_send(const fs_builder &bld, enum opcode opcode,
                   const array_reg &payload,
                   const fs_reg &surface, const fs_reg &arg,
                   unsigned rlen, brw_predicate pred = BRW_PREDICATE_NONE)
         {
            const fs_reg usurface = bld.vgrf(BRW_REGISTER_TYPE_UD);
            const array_reg dst =
               rlen ? alloc_array_reg(bld, BRW_REGISTER_TYPE_UD, rlen) :
               array_reg(bld.null_reg_ud());

            /* Reduce the dynamically uniform surface index to a single
             * scalar.
             */
            bld.emit_uniformize(usurface, surface);

            fs_builder::instruction *inst =
               bld.emit(opcode, natural_reg(bld, dst),
                        natural_reg(bld, payload), usurface, arg);
            inst->mlen = payload.size;
            inst->regs_written = rlen;
            inst->predicate = pred;

            return dst;
         }

         /**
          * Initialize the header present in some typed and untyped surface
          * messages.
          */
         array_reg
         emit_header(const fs_builder &bld, const fs_reg &sample_mask)
         {
            fs_builder ubld = bld.half(0);
            const fs_reg dst = ubld.vgrf(BRW_REGISTER_TYPE_UD);
            exec_all(ubld.MOV(dst, fs_reg(0)));
            exec_all(ubld.MOV(component(dst, 7), sample_mask));
            return array_reg(dst);
         }
      }

      /**
       * Emit an untyped surface read opcode.  \p dims determines the number
       * of components of the address and \p size the number of components of
       * the returned value.
       */
      fs_reg
      emit_untyped_read(const fs_builder &bld,
                        const fs_reg &surface, const fs_reg &addr,
                        unsigned dims, unsigned size,
                        brw_predicate pred)
      {
         const vector_layout layout(bld, true);
         const array_reg payload = emit_insert(layout, bld, addr, dims);
         const unsigned rlen = size * bld.dispatch_width() / 8;
         const array_reg dst =
            emit_send(bld, SHADER_OPCODE_UNTYPED_SURFACE_READ,
                      payload, surface, fs_reg(size), rlen, pred);

         return emit_extract(layout, bld, &dst, size);
      }

      /**
       * Emit an untyped surface write opcode.  \p dims determines the number
       * of components of the address and \p size the number of components of
       * the argument.
       */
      void
      emit_untyped_write(const fs_builder &bld, const fs_reg &surface,
                         const fs_reg &addr, const fs_reg &src,
                         unsigned dims, unsigned size,
                         brw_predicate pred)
      {
         const vector_layout layout(bld, true);
         const array_reg payload =
            emit_collect(bld,
                         emit_header(bld, bld.sample_mask_reg()),
                         emit_insert(layout, bld, addr, dims),
                         emit_insert(layout, bld, src, size));

         emit_send(bld, SHADER_OPCODE_UNTYPED_SURFACE_WRITE,
                   payload, surface, fs_reg(size), 0, pred);
      }

      /**
       * Emit an untyped surface atomic opcode.  \p dims determines the number
       * of components of the address and \p rsize the number of components of
       * the returned value (either zero or one).
       */
      fs_reg
      emit_untyped_atomic(const fs_builder &bld,
                          const fs_reg &surface, const fs_reg &addr,
                          const fs_reg &src0, const fs_reg &src1,
                          unsigned dims, unsigned rsize, unsigned op,
                          brw_predicate pred)
      {
         const unsigned size = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const vector_layout layout(bld, true);
         /* Zip the components of both sources, they are represented as the X
          * and Y components of the same vector.
          */
         const fs_reg srcs = natural_reg(bld,
                                         emit_zip(bld, emit_flatten(bld, src0, 1),
                                                  emit_flatten(bld, src1, 1), 1));
         const array_reg payload =
            emit_collect(bld,
                         emit_header(bld, bld.sample_mask_reg()),
                         emit_insert(layout, bld, addr, dims),
                         emit_insert(layout, bld, srcs, size));
         const array_reg dst =
            emit_send(bld, SHADER_OPCODE_UNTYPED_ATOMIC,
                      payload, surface, fs_reg(op),
                      rsize * bld.dispatch_width() / 8, pred);

         return emit_extract(layout, bld, &dst, rsize);
      }

      /**
       * Emit a typed surface read opcode.  \p dims determines the number of
       * components of the address and \p size the number of components of the
       * returned value.
       */
      fs_reg
      emit_typed_read(const fs_builder &bld, const fs_reg &surface,
                      const fs_reg &addr, unsigned dims, unsigned size)
      {
         const vector_layout layout(bld, false);
         array_reg dsts[2];

         for (unsigned i = 0; i < layout.halves; ++i) {
            /* Get a half builder for this half if required. */
            const fs_builder ubld = (layout.halves > 1 ? bld.half(i) : bld);
            const array_reg payload =
               emit_collect(ubld,
                            emit_header(bld, fs_reg(0xffff)),
                            emit_insert(layout, bld, addr, dims, i));

            dsts[i] = emit_send(ubld, SHADER_OPCODE_TYPED_SURFACE_READ,
                                payload, surface, fs_reg(size), size);
         }

         return emit_extract(layout, bld, dsts, size);
      }

      /**
       * Emit a typed surface write opcode.  \p dims determines the number of
       * components of the address and \p size the number of components of the
       * argument.
       */
      void
      emit_typed_write(const fs_builder &bld, const fs_reg &surface,
                       const fs_reg &addr, const fs_reg &src,
                       unsigned dims, unsigned size)
      {
         const vector_layout layout(bld, false);

         for (unsigned i = 0; i < layout.halves; ++i) {
            /* Get a half builder for this half if required. */
            const fs_builder ubld = (layout.halves > 1 ? bld.half(i) : bld);
            const array_reg payload =
               emit_collect(ubld,
                            emit_header(bld, bld.sample_mask_reg()),
                            emit_insert(layout, bld, addr, dims, i),
                            emit_insert(layout, bld, src, size, i));

            emit_send(ubld, SHADER_OPCODE_TYPED_SURFACE_WRITE,
                      payload, surface, fs_reg(size), 0);
         }
      }

      /**
       * Emit a typed surface atomic opcode.  \p dims determines the number of
       * components of the address and \p rsize the number of components of
       * the returned value (either zero or one).
       */
      fs_reg
      emit_typed_atomic(const fs_builder &bld, const fs_reg &surface,
                        const fs_reg &addr,
                        const fs_reg &src0, const fs_reg &src1,
                        unsigned dims, unsigned rsize, unsigned op,
                        brw_predicate pred)
      {
         const unsigned size = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const vector_layout layout(bld, false);
         /* Zip the components of both sources, they are represented as the X
          * and Y components of the same vector.
          */
         const fs_reg srcs = natural_reg(
            bld, emit_zip(bld, emit_flatten(bld, src0, 1),
                          emit_flatten(bld, src1, 1), 1));
         array_reg dsts[2];

         for (unsigned i = 0; i < layout.halves; ++i) {
            /* Get a half builder for this half if required. */
            const fs_builder ubld = (layout.halves > 1 ? bld.half(i) : bld);
            const array_reg payload =
               emit_collect(ubld,
                            emit_header(bld, bld.sample_mask_reg()),
                            emit_insert(layout, bld, addr, dims, i),
                            emit_insert(layout, bld, srcs, size, i));

            dsts[i] = emit_send(ubld, SHADER_OPCODE_TYPED_ATOMIC,
                                payload, surface, fs_reg(op), rsize, pred);
         }

         return emit_extract(layout, bld, dsts, rsize);
      }
   }
}
