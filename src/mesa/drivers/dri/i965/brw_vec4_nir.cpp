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
 */

#include "glsl/ir.h"
#include "glsl/ir_optimization.h"
#include "glsl/nir/glsl_to_nir.h"
#include "brw_vec4.h"
#include "brw_nir.h"
#include "brw_vs.h"
#include "brw_fs.h"

namespace brw {

void
vec4_visitor::emit_nir_code()
{
   nir_shader *nir = prog->nir;

   nir_num_inputs = 0;
   nir_loaded_inputs = 0;
   nir_inputs = reralloc(mem_ctx, nir_inputs, src_reg, 2);
   nir_setup_inputs(nir);

   nir_setup_outputs(nir);

   nir_emit_main(nir);
}

void
vec4_visitor::nir_setup_inputs(nir_shader *shader)
{
   /* @FIXME: We need to analyze why shader->num_inputs reports zero
    * here even if we have attributes. It looks like a bug.
    */
   foreach_list_typed(nir_variable, var, node, &shader->inputs) {
      src_reg src = src_reg(ATTR, var->data.location, var->type);
      nir_inputs[nir_num_inputs] = src;
      nir_num_inputs++;
   }
}

void
vec4_visitor::nir_setup_outputs(nir_shader *shader)
{
   foreach_list_typed(nir_variable, var, node, &shader->outputs) {
      /* @TODO */
   }
}

static bool
emit_block(nir_block *block, void *void_visitor)
{
   vec4_visitor *visitor = (vec4_visitor *) void_visitor;

   nir_foreach_instr(block, instr) {
      switch (instr->type) {

        // load const
      case nir_instr_type_load_const:
         visitor->nir_emit_load_const(nir_instr_as_load_const(instr));
         break;

         // load input
      case nir_instr_type_intrinsic:
         visitor->nir_emit_intrinsic(nir_instr_as_intrinsic(instr));
         break;

         // ALU
      case nir_instr_type_alu:
         visitor->nir_emit_alu(nir_instr_as_alu(instr));
         break;

      default:
         fprintf(stderr, "VS instruction not yet implemented by NIR -> vec4\n");
         break;
      }
   }

   return true;
}

dst_reg
vec4_visitor::get_nir_dest(nir_dest dest)
{
   dst_reg reg;

   reg = nir_locals[dest.reg.reg->index];
   reg = offset(reg, dest.reg.base_offset);

   return reg;
}

src_reg
vec4_visitor::get_nir_src(nir_src src)
{
   dst_reg reg;

   if (src.is_ssa) {
      assert(src.ssa->parent_instr->type == nir_instr_type_load_const);
      nir_load_const_instr *load = nir_instr_as_load_const(src.ssa->parent_instr);

      reg = dst_reg(GRF, alloc.allocate(src.ssa->num_components));
      reg.type = BRW_REGISTER_TYPE_D;

      for (unsigned i = 0; i < src.ssa->num_components; ++i) {
         reg.writemask = 1 << i;
         emit(MOV(reg, retype(src_reg(load->value.i[i]), reg.type)));
      }
   }
   else {
      reg = nir_locals[src.reg.reg->index];
      reg = offset(reg, src.reg.base_offset);
   }

   return src_reg(reg);
}

void
vec4_visitor::nir_emit_main(nir_shader *shader)
{
   nir_foreach_overload(shader, overload) {
      assert(strcmp(overload->function->name, "main") == 0);
      assert(overload->impl);

      /* setup local registers */
      /* @FIXME: this should be done for all func implentations, not only
         main(), so will have to be moved to a nir_emit_impl()
         function in the future, as fs_nir does. */
      nir_locals = reralloc(mem_ctx, nir_locals, dst_reg, overload->impl->reg_alloc);
      foreach_list_typed(nir_register, reg, node, &overload->impl->registers) {
         unsigned array_elems =
           reg->num_array_elems == 0 ? 1 : reg->num_array_elems;
         unsigned size = array_elems * reg->num_components;

         nir_locals[reg->index] =
           dst_reg(GRF, alloc.allocate(size));
      }

      nir_foreach_block(overload->impl, emit_block, this);
   }
}

void
vec4_visitor::nir_emit_load_const(nir_load_const_instr *instr)
{
   /* 'load_const' instructions are ignored. Instead, the get_nir_src() method
      indirectly implements loading constant values into volatile registers.
      This is how brw_fs_nir work and we should do the same. */
}

void
vec4_visitor::nir_emit_intrinsic(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {

     /* load input */
   case nir_intrinsic_load_input:
      nir_emit_intrinsic_load_input(instr);
      break;

      /* store output */
   case nir_intrinsic_store_output:
      nir_emit_intrinsic_store_output(instr);
      break;

   default:
      fprintf(stderr,
              "Non-implemented intrinsic instruction in NIR->vec4 (%d)\n",
              instr->intrinsic);
      break;
   }
}

void
vec4_visitor::nir_emit_intrinsic_load_input(nir_intrinsic_instr *instr)
{
   dst_reg dest = get_nir_dest(instr->dest);

   dest.writemask = 0;
   for (int i = 0; i < instr->num_components; i++)
      dest.writemask |= 1 << i;

   src_reg src = nir_inputs[nir_loaded_inputs];
   nir_loaded_inputs++;

   dest = retype(dest, src.type);

   emit(MOV(dest, src));
}

void
vec4_visitor::nir_emit_intrinsic_store_output(nir_intrinsic_instr *instr)
{
   src_reg reg = get_nir_src(instr->src[0]);
   dst_reg dst = dst_reg(reg);

   /* set the mask */
   dst.writemask = 0;
   for (unsigned i = 0; i < instr->num_components; i++)
      dst.writemask |= (1 << i);

   /* @FIXME: is it safe to hardcode the type to float here? */
   dst = retype(dst, BRW_REGISTER_TYPE_F);

   /* @FIXME: hardcode offset to zero to support only gl_Position by now,
      until const_index[0] is fixed to contain the right offset */
   output_reg[0] = dst;
}

/* @FIXME: C&P from brw_fs_nir. Candidate to be revamped to a common place. */
static brw_reg_type
brw_type_for_nir_type(nir_alu_type type)
{
   switch (type) {
   case nir_type_unsigned:
      return BRW_REGISTER_TYPE_UD;
   case nir_type_bool:
   case nir_type_int:
      return BRW_REGISTER_TYPE_D;
   case nir_type_float:
      return BRW_REGISTER_TYPE_F;
   default:
      unreachable("unknown type");
   }

   return BRW_REGISTER_TYPE_F;
}

static unsigned
brw_swizzle_for_nir_swizzle(uint8_t swizzle[4])
{
   return  BRW_SWIZZLE4(swizzle[0], swizzle[1], swizzle[2], swizzle[3]);
}

void
vec4_visitor::nir_emit_alu(nir_alu_instr *instr)
{
   vec4_instruction *inst;

   dst_reg dst = get_nir_dest(instr->dest.dest);
   dst.type = brw_type_for_nir_type(nir_op_infos[instr->op].output_type);
   dst.writemask = instr->dest.write_mask;

   src_reg op[4];
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      op[i] = get_nir_src(instr->src[i].src);
      op[i].swizzle = brw_swizzle_for_nir_swizzle(instr->src[i].swizzle);
      op[i].type = brw_type_for_nir_type(nir_op_infos[instr->op].input_types[i]);
      op[i].abs = instr->src[i].abs;
      op[i].negate = instr->src[i].negate;
   }

   switch(instr->op) {
   case nir_op_imov:
   case nir_op_fmov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   {
      /* @TODO: Improve the quality of the algorithm to avoid repeating code.
       */
      bool used[4] = {};
      for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
         if (!(instr->dest.write_mask & (1 << i)))
            continue;

         if (used[i])
            continue;

         dst.writemask = 1 << i;
         op[i].swizzle = instr->src[i].swizzle[0] << (i * 2);
         used[i] = true;

         for (unsigned j=i+1; j < nir_op_infos[instr->op].num_inputs; j++) {
            if (!(instr->dest.write_mask & (1 << j)))
               continue;

            if (used[j])
               continue;

            if (nir_srcs_equal(instr->src[i].src, instr->src[j].src)) {
               dst.writemask |= 1 << j;
               op[i].swizzle |= instr->src[j].swizzle[0] << (j* 2);
               used[j] = true;
            }
         }

         inst = emit(MOV(dst, retype(op[i], dst.type)));
         inst->saturate = instr->dest.saturate;
      }
      break;
   }
   case nir_op_f2i:
   case nir_op_f2u:
      inst = emit(MOV(dst, op[0]));
      break;

   case nir_op_fadd:
      inst = emit(ADD(dst, op[0], op[1]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmul:
      inst = emit(MUL(dst, op[0], op[1]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_inot:
      inst = emit(NOT(dst, op[0]));
      break;

   default:
      fprintf(stderr, "Non-implemented ALU operation (%d)\n", instr->op);
      break;
   }
}

}
