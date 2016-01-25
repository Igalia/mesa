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
 *    Connor Abbott <cwabbott0@gmail.com>
 */

#include "brw_nir.h"
#include "glsl/nir/nir_builder.h"

/*
 * In Align16 mode, swizzles are always applied to 32-bit chunks of a
 * register, and there are always 4 swizzle components that apply to each half
 * of a register, which means that 64-bit values with more than 2 components
 * can't represent all the possible swizzles. This pass splits up ALU
 * instructions with more than 2 components into 2 instructions, to avoid this
 * limitation.
 */

static void
lower_reduction(nir_alu_instr *instr, nir_op two_chan_op, nir_op chan_op,
                nir_op merge_op, nir_builder *b)
{
   unsigned num_components = nir_op_infos[instr->op].input_sizes[0];
   unsigned num_inputs = nir_op_infos[instr->op].num_inputs;
   unsigned dest_bitsize = nir_dest_bit_size(instr->dest.dest);

   /* reduce the first two channels */
   nir_alu_instr *first_half = nir_alu_instr_create(b->shader, two_chan_op);
   nir_ssa_dest_init(&first_half->instr, &first_half->dest.dest, 1,
                     dest_bitsize, NULL);
   first_half->dest.write_mask = 0x1;
   for (unsigned i = 0; i < num_inputs; i++)
      nir_alu_src_copy(&first_half->src[i], &instr->src[i], first_half);
   nir_builder_instr_insert(b, &first_half->instr);

   /* reduce the second two channels, or apply a channel operation to the lone
    * third channel
    */
   nir_alu_instr *second_half;
   if (num_components == 3) {
      second_half = nir_alu_instr_create(b->shader, chan_op);
   } else {
      assert(num_components == 4);
      second_half = nir_alu_instr_create(b->shader, two_chan_op);
   }

   nir_ssa_dest_init(&second_half->instr, &second_half->dest.dest,
                     1, dest_bitsize, NULL);
   second_half->dest.write_mask = 0x1;
   for (unsigned i = 0; i < num_inputs; i++) {
      nir_alu_src_copy(&second_half->src[i], &instr->src[i], second_half);
      second_half->src[i].swizzle[0] = 2;
      second_half->src[i].swizzle[1] = 3;
   }
   nir_builder_instr_insert(b, &second_half->instr);

   /* Finally, combine the two things into one */
   nir_alu_instr *merge = nir_alu_instr_create(b->shader, merge_op);
   nir_ssa_dest_init(&merge->instr, &merge->dest.dest, 1,
                     dest_bitsize, NULL);
   merge->dest.write_mask = 0x1;
   merge->src[0].src = nir_src_for_ssa(&first_half->dest.dest.ssa);
   merge->src[1].src = nir_src_for_ssa(&second_half->dest.dest.ssa);
   nir_builder_instr_insert(b, &merge->instr);

   nir_ssa_def_rewrite_uses(&instr->dest.dest.ssa,
                            nir_src_for_ssa(&merge->dest.dest.ssa));
   nir_instr_remove(&instr->instr);
}

static void
lower_alu_instr(nir_alu_instr *instr, nir_builder *b)
{
   unsigned num_src = nir_op_infos[instr->op].num_inputs;

   bool any_64 = false;
   if (nir_dest_bit_size(instr->dest.dest) == 64)
      any_64 = true;

   for (unsigned i = 0; i < num_src; i++) {
      if (nir_src_bit_size(instr->src[i].src) == 64)
         any_64 = true;
   }

   if (!any_64)
      return;

   bool any_vec3_or_4 = false;
   if (instr->dest.dest.ssa.num_components > 2)
      any_vec3_or_4 = true;

   for (unsigned i = 0; i < num_src; i++) {
      if (nir_ssa_alu_instr_src_components(instr, i) > 2)
         any_vec3_or_4 = true;
   }

   if (!any_vec3_or_4)
      return;

   b->cursor = nir_before_instr(&instr->instr);

#define LOWER_REDUCTION(name, chan, merge) \
   case name##3: \
   case name##4: \
      lower_reduction(instr, name##2, chan, merge, b); \
      return;

   switch(instr->op) {
   case nir_op_vec4:
   case nir_op_vec3:
      /* These ops are the ones that group up dvec2's and doubles into dvec3's
       * and dvec4's when necessary, so we don't lower them. If they're
       * unnecessary, copy propagation will clean them up.
       */
     return;

   LOWER_REDUCTION(nir_op_fdot, nir_op_fmul, nir_op_fadd);
   LOWER_REDUCTION(nir_op_ball_fequal, nir_op_feq, nir_op_iand);
   LOWER_REDUCTION(nir_op_bany_fnequal, nir_op_fne, nir_op_iand);

   default:
      break;
   }

   unsigned num_components = instr->dest.dest.ssa.num_components;
   assert(nir_op_infos[instr->op].output_size == 0);
   assert(num_components > 2);

   nir_alu_instr *vec_instr;
   if (num_components == 3)
      vec_instr = nir_alu_instr_create(b->shader, nir_op_vec3);
   else
      vec_instr = nir_alu_instr_create(b->shader, nir_op_vec4);
   nir_ssa_dest_init(&vec_instr->instr, &vec_instr->dest.dest, num_components,
                     instr->dest.dest.ssa.bit_size, NULL);
   vec_instr->dest.write_mask = (1 << num_components) - 1;

   for (unsigned half = 0; half < 2; half++) {
      unsigned chan = half * 2;
      nir_alu_instr *lower = nir_alu_instr_create(b->shader, instr->op);
      unsigned lower_components = 2;
      if (half == 1 && num_components == 3)
         lower_components = 1;

      nir_ssa_dest_init(&lower->instr, &lower->dest.dest, lower_components,
                        instr->dest.dest.ssa.bit_size, NULL);
      lower->dest.write_mask = (1 << lower_components) - 1;

      for (unsigned i = 0; i < num_src; i++) {
         assert(nir_op_infos[instr->op].input_sizes[i] < 2);
         nir_alu_src_copy(&lower->src[i], &instr->src[i], lower);
         for (unsigned j = 0; j < lower_components; j++) {
            lower->src[i].swizzle[j] = instr->src[i].swizzle[chan + j];
            vec_instr->src[j + chan].src = nir_src_for_ssa(&lower->dest.dest.ssa);
            vec_instr->src[j + chan].swizzle[0] = j;
         }
      }

      nir_builder_instr_insert(b, &lower->instr);
   }

   nir_builder_instr_insert(b, &vec_instr->instr);
   nir_ssa_def_rewrite_uses(&instr->dest.dest.ssa,
                            nir_src_for_ssa(&vec_instr->dest.dest.ssa));

   nir_instr_remove(&instr->instr);
}

static void
lower_load_const(nir_load_const_instr *instr, nir_builder *b)
{
   if (instr->def.bit_size != 64)
      return;

   if (instr->def.num_components < 3)
      return;

   b->cursor = nir_before_instr(&instr->instr);

   nir_load_const_instr *first_half = nir_load_const_instr_create(b->shader, 2);
   first_half->def.bit_size = 64;
   first_half->value.ul[0] = instr->value.ul[0];
   first_half->value.ul[1] = instr->value.ul[1];
   nir_builder_instr_insert(b, &first_half->instr);

   nir_load_const_instr *second_half =
      nir_load_const_instr_create(b->shader, instr->def.num_components - 2);
   second_half->def.bit_size = 64;
   for (unsigned i = 0; i < instr->def.num_components - 2; i++)
      second_half->value.ul[i] = instr->value.ul[i + 2];
   nir_builder_instr_insert(b, &second_half->instr);

   nir_op vec_op = (instr->def.num_components == 3) ? nir_op_vec3 : nir_op_vec4;
   nir_alu_instr *vec = nir_alu_instr_create(b->shader, vec_op);
   nir_ssa_dest_init(&vec->instr, &vec->dest.dest, instr->def.num_components,
                     64, NULL);
   vec->dest.write_mask = (1 << instr->def.num_components) - 1;
   vec->src[0].src = nir_src_for_ssa(&first_half->def);
   vec->src[1].src = nir_src_for_ssa(&first_half->def);
   vec->src[1].swizzle[0] = 1;
   vec->src[2].src = nir_src_for_ssa(&second_half->def);
   if (instr->def.num_components == 4) {
      vec->src[3].src = nir_src_for_ssa(&second_half->def);
      vec->src[3].swizzle[0] = 1;
   }
   nir_builder_instr_insert(b, &vec->instr);

   nir_ssa_def_rewrite_uses(&instr->def, nir_src_for_ssa(&vec->dest.dest.ssa));
}

static void
lower_phis(nir_block *block, void *mem_ctx)
{
   nir_phi_instr *last_phi = NULL;
   nir_foreach_instr(block, instr) {
      if (instr->type != nir_instr_type_phi)
         break;

      last_phi = nir_instr_as_phi(instr);
   }

   nir_foreach_instr_safe(block, instr) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);

      unsigned bit_size = phi->dest.ssa.bit_size;
      if (bit_size != 64)
         continue;

      unsigned num_components = phi->dest.ssa.num_components;
      if (num_components < 3)
         continue;

      nir_op vec_op;
      if (num_components == 3)
         vec_op = nir_op_vec3;
      else
         vec_op = nir_op_vec4;

      nir_alu_instr *vec = nir_alu_instr_create(mem_ctx, vec_op);
      nir_ssa_dest_init(&vec->instr, &vec->dest.dest,
                        num_components, bit_size, NULL);
      vec->dest.write_mask = (1 << num_components) - 1;

      for (unsigned half = 0; half < 2; half++) {
         unsigned lowered_components = 2;
         if (half == 1 && num_components == 3)
            lowered_components = 1;

         nir_phi_instr *new_phi = nir_phi_instr_create(mem_ctx);
         nir_ssa_dest_init(&new_phi->instr, &new_phi->dest,
                           lowered_components, bit_size, NULL);

         for (unsigned i = 0; i < lowered_components; i++) {
            unsigned component = 2 * half + i;
            vec->src[component].src = nir_src_for_ssa(&new_phi->dest.ssa);
            vec->src[component].swizzle[0] = i;
         }

         nir_foreach_phi_src(phi, src) {
            nir_alu_instr *mov = nir_alu_instr_create(mem_ctx,
                                                      nir_op_imov);
            nir_ssa_dest_init(&mov->instr, &mov->dest.dest, lowered_components,
                              bit_size, NULL);
            mov->dest.write_mask = (1 << lowered_components) - 1;
            nir_src_copy(&mov->src[0].src, &src->src, mem_ctx);
            for (unsigned i = 0; i < lowered_components; i++) {
               unsigned component = 2 * half + i;
               mov->src[0].swizzle[i] = component;
            }


            /* Insert at the end of the predecessor but before the jump */
            nir_instr *pred_last_instr = nir_block_last_instr(src->pred);
            if (pred_last_instr && pred_last_instr->type == nir_instr_type_jump)
               nir_instr_insert_before(pred_last_instr, &mov->instr);
            else
               nir_instr_insert_after_block(src->pred, &mov->instr);

            nir_phi_src *new_src = ralloc(new_phi, nir_phi_src);
            new_src->pred = src->pred;
            new_src->src = nir_src_for_ssa(&mov->dest.dest.ssa);

            exec_list_push_tail(&new_phi->srcs, &new_src->node);
         }

         nir_instr_insert_before(&phi->instr, &new_phi->instr);
      }

      nir_instr_insert_after(&last_phi->instr, &vec->instr);

      nir_ssa_def_rewrite_uses(&phi->dest.ssa,
                               nir_src_for_ssa(&vec->dest.dest.ssa));

      nir_instr_remove(&phi->instr);

      if (instr == &last_phi->instr)
         break;
   }
}

static bool
lower_block(nir_block *block, void *mem_ctx)
{
   nir_builder bld;
   nir_builder_init(&bld, nir_cf_node_get_function(&block->cf_node));

   lower_phis(block, mem_ctx);

   nir_foreach_instr_safe(block, instr) {
      if (instr->type == nir_instr_type_alu)
         lower_alu_instr(nir_instr_as_alu(instr), &bld);
      else if (instr->type == nir_instr_type_load_const)
         lower_load_const(nir_instr_as_load_const(instr), &bld);
   }

   return true;
}

static void
lower_impl(nir_function_impl *impl, void *mem_ctx)
{
   nir_foreach_block(impl, lower_block, mem_ctx);
}

void
brw_nir_split_doubles(nir_shader *shader)
{
   nir_foreach_function(shader, function) {
      if (function->impl)
         lower_impl(function->impl, shader);
   }
}

