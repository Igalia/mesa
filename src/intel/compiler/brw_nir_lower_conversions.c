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
#include "compiler/nir/nir_builder.h"

static nir_op
get_conversion_op(nir_alu_type src_type,
                  unsigned src_bit_size,
                  nir_alu_type dst_type,
                  unsigned dst_bit_size)
{
   nir_alu_type src_full_type = (nir_alu_type) (src_type | src_bit_size);
   nir_alu_type dst_full_type = (nir_alu_type) (dst_type | dst_bit_size);

   return nir_type_conversion_op(src_full_type, dst_full_type,
                                 nir_rounding_mode_undef);
}

static void
split_conversion(nir_builder *b, nir_alu_instr *alu, nir_op op1, nir_op op2)
{
   b->cursor = nir_before_instr(&alu->instr);
   nir_ssa_def *tmp = nir_build_alu(b, op1, alu->src[0].src.ssa, NULL, NULL, NULL);
   nir_ssa_def *res = nir_build_alu(b, op2, tmp, NULL, NULL, NULL);
   nir_ssa_def_rewrite_uses(&alu->dest.dest.ssa, nir_src_for_ssa(res));
}

static bool
lower_instr(nir_builder *b, nir_alu_instr *alu)
{
   unsigned src_bit_size = nir_src_bit_size(alu->src[0].src);
   nir_alu_type src_type = nir_op_infos[alu->op].input_types[0];
   nir_alu_type src_full_type = (nir_alu_type) (src_type | src_bit_size);

   unsigned dst_bit_size = nir_dest_bit_size(alu->dest.dest);
   nir_alu_type dst_type = nir_op_infos[alu->op].output_type;
   nir_alu_type dst_full_type = (nir_alu_type) (dst_type | dst_bit_size);

   /* BDW PRM, vol02, Command Reference Instructions, mov - MOVE:
    *
    *   "There is no direct conversion from HF to DF or DF to HF.
    *    Use two instructions and F (Float) as an intermediate type.
    *
    *    There is no direct conversion from HF to Q/UQ or Q/UQ to HF.
    *    Use two instructions and F (Float) or a word integer type
    *    or a DWord integer type as an intermediate type."
    */
   if ((src_full_type == nir_type_float16 && dst_bit_size == 64) ||
       (src_bit_size == 64 && dst_full_type == nir_type_float16)) {
      nir_op op1 = get_conversion_op(src_type, src_bit_size, src_type, 32);
      nir_op op2 = get_conversion_op(src_type, 32, dst_type, dst_bit_size);
      split_conversion(b, alu, op1, op2);
      return true;
   }

   /* SKL PRM, vol 02a, Command Reference: Instructions, Move:
    *
    *   "There is no direct conversion from B/UB to DF or DF to B/UB. Use
    *    two instructions and a word or DWord intermediate type."
    *
    *   "There is no direct conversion from B/UB to Q/UQ or Q/UQ to B/UB.
    *    Use two instructions and a word or DWord intermediate integer
    *    type."
    */
   if ((src_bit_size == 8 && dst_bit_size == 64) ||
       (src_bit_size == 64 && dst_bit_size == 8)) {
      nir_op op1 = get_conversion_op(src_type, src_bit_size, src_type, 32);
      nir_op op2 = get_conversion_op(src_type, 32, dst_type, dst_bit_size);
      split_conversion(b, alu, op1, op2);
      return true;
   }

   return false;
}

static bool
lower_impl(nir_function_impl *impl)
{
   nir_builder b;
   nir_builder_init(&b, impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;

         nir_alu_instr *alu = nir_instr_as_alu(instr);
         assert(alu->dest.dest.is_ssa);

         if (nir_op_infos[alu->op].num_inputs > 1)
            continue;

         progress = lower_instr(&b, alu) || progress;
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   }

   return progress;
}

bool
brw_nir_lower_conversions(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= lower_impl(function->impl);
   }

   return progress;
}
