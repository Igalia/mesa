/*
 * Copyright © 2015 Intel Corporation
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
 *
 * Authors:
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir.h"
#include "nir_builder.h"

/*
 * lowers:
 *
 * packDouble2x32(foo) -> packDouble2x32Split(foo.x, foo.y)
 * unpackDouble2x32(foo) -> vec2(unpackDouble2x32_x(foo), unpackDouble2x32_y(foo))
 */

static nir_ssa_def *
component(nir_builder *b, nir_ssa_def *src, unsigned component)
{
   return nir_swizzle(b, src, (unsigned[]) {component}, 1, true);
}

static nir_ssa_def *
lower_pack_double(nir_builder *b, nir_ssa_def *src)
{
   return nir_pack_double_2x32_split(b, component(b, src, 0),
                                        component(b, src, 1));
}

static nir_ssa_def *
lower_unpack_double(nir_builder *b, nir_ssa_def *src)
{
   return nir_vec2(b, nir_unpack_double_2x32_split_x(b, src),
                      nir_unpack_double_2x32_split_y(b, src));
}

static void
lower_double_pack_instr(nir_alu_instr *instr)
{
   if (instr->op != nir_op_pack_double_2x32 &&
       instr->op != nir_op_unpack_double_2x32)
      return;

   nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&instr->instr.block->cf_node));
   b.cursor = nir_before_instr(&instr->instr);

   nir_ssa_def *src = nir_fmov_alu(&b, instr->src[0],
                                   nir_op_infos[instr->op].input_sizes[0]);
   nir_ssa_def *dest =
      instr->op == nir_op_pack_double_2x32 ?
      lower_pack_double(&b, src) :
      lower_unpack_double(&b, src);

   nir_ssa_def_rewrite_uses(&instr->dest.dest.ssa, nir_src_for_ssa(dest));
   nir_instr_remove(&instr->instr);
}

static bool
lower_double_pack_block(nir_block *block, void *ctx)
{
   (void) ctx;

   nir_foreach_instr_safe(block, instr) {
      if (instr->type != nir_instr_type_alu)
         continue;

      lower_double_pack_instr(nir_instr_as_alu(instr));
   }

   return true;
}

static void
lower_double_pack_impl(nir_function_impl *impl)
{
   nir_foreach_block(impl, lower_double_pack_block, NULL);
}

void
nir_lower_double_pack(nir_shader *shader)
{
   nir_foreach_function(shader, function) {
      if (function->impl)
         lower_double_pack_impl(function->impl);
   }
}

