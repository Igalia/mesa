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

#include "nir_builder.h"

static bool
lower_b2f(nir_builder *b, nir_alu_instr *alu)
{
   nir_src *src = &alu->src[0].src;
   assert(src->is_ssa);

   if (src->ssa->parent_instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *parent_alu = nir_instr_as_alu(src->ssa->parent_instr);
   if (parent_alu->op != nir_op_i2i32)
      return false;

   const nir_src *parent_src = &parent_alu->src[0].src;
   assert(parent_src->is_ssa);

   if (parent_src->ssa->bit_size > 16)
      return false;

   /* Drop the conversion from 16-bit integer to 32-bit and convert
    * directly from 16-bit integer to boolean.
    */
   nir_instr_rewrite_src(&alu->instr, src, *parent_src);
   nir_src_copy(src, parent_src, &alu->instr);

   alu->dest.dest.ssa.bit_size = 16;

   return true;
}

static bool
lower_bool_dest(nir_builder *b, nir_alu_instr *alu)
{
   unsigned bit_size = 0;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      const unsigned bit_size_i = nir_src_bit_size(alu->src[i].src);

      /* All operands have to agree. */
      assert(bit_size == 0 || bit_size == bit_size_i);

      bit_size = bit_size_i;
   }

   if (bit_size != 16)
      return false;

   if (alu->dest.dest.is_ssa)
      alu->dest.dest.ssa.bit_size = 16;
   else
      alu->dest.dest.reg.reg->bit_size = 16;

   return true;
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

         if (alu->op == nir_op_b2f)
            progress |= lower_b2f(&b, alu);
         else if (nir_op_infos[alu->op].output_type == nir_type_bool32)
            progress |= lower_bool_dest(&b, alu);
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   }

   return progress;
}

bool
nir_lower_bool_size(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= lower_impl(function->impl);
   }

   return progress;
}
