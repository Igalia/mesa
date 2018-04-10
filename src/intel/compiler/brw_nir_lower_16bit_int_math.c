/*
 * Copyright © 2018 Intel Corporation
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

#include "brw_nir.h"
#include "nir_builder.h"

/**
 * Intel hardware doesn't support 16-bit integer Math instructions so this
 * pass implements them in 32-bit and then converts the result back to 16-bit.
 */
static void
lower_math_instr(nir_builder *bld, nir_alu_instr *alu, bool is_signed)
{
   const nir_op op = alu->op;

   bld->cursor = nir_before_instr(&alu->instr);

   nir_ssa_def *srcs_32[4] = { NULL, NULL, NULL, NULL };
   const uint32_t num_inputs = nir_op_infos[op].num_inputs;
   for (uint32_t i = 0; i < num_inputs; i++) {
      nir_ssa_def *src = nir_ssa_for_alu_src(bld, alu, i);
      srcs_32[i] = is_signed ? nir_i2i32(bld, src) : nir_u2u32(bld, src);
   }

   nir_ssa_def *dst_32 =
      nir_build_alu(bld, op, srcs_32[0], srcs_32[1], srcs_32[2], srcs_32[3]);

   nir_ssa_def *dst_16 =
      is_signed ? nir_i2i16(bld, dst_32) : nir_u2u16(bld, dst_32);

   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, nir_src_for_ssa(dst_16));
}

static bool
lower_instr(nir_builder *bld, nir_alu_instr *alu)
{
   assert(alu->dest.dest.is_ssa);
   if (alu->dest.dest.ssa.bit_size != 16)
      return false;

   bool is_signed = false;
   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_imod:
      is_signed = true;
      /* Fallthrough */
   case nir_op_udiv:
   case nir_op_umod:
   case nir_op_irem:
      lower_math_instr(bld, alu, is_signed);
      return true;
   default:
      return false;
   }
}

static bool
lower_impl(nir_function_impl *impl)
{
   nir_builder b;
   nir_builder_init(&b, impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_alu)
            progress |= lower_instr(&b, nir_instr_as_alu(instr));
      }
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);

   return progress;
}

bool
brw_nir_lower_16bit_int_math(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= lower_impl(function->impl);
   }

   return progress;
}
