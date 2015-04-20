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

   nir_inputs = reralloc(mem_ctx, nir_inputs, src_reg, nir->num_inputs);
   nir_setup_inputs(nir);

   nir_outputs = reralloc(mem_ctx, nir_outputs, int, nir->num_outputs);
   nir_setup_outputs(nir);

   /* get the main function and emit it */
   nir_foreach_overload(nir, overload) {
      assert(strcmp(overload->function->name, "main") == 0);
      assert(overload->impl);
      nir_emit_impl(overload->impl);
   }
}

void
vec4_visitor::nir_setup_inputs(nir_shader *shader)
{
   foreach_list_typed(nir_variable, var, node, &shader->inputs) {
      src_reg src = src_reg(ATTR, var->data.location, var->type);

      int offset = var->data.driver_location;
      nir_inputs[offset] = src;
   }
}

void
vec4_visitor::nir_setup_outputs(nir_shader *shader)
{
   foreach_list_typed(nir_variable, var, node, &shader->outputs) {
      int offset = var->data.driver_location;
      nir_outputs[offset] = var->data.location;
   }
}

void
vec4_visitor::nir_emit_impl(nir_function_impl *impl)
{
   nir_locals = reralloc(mem_ctx, nir_locals, dst_reg, impl->reg_alloc);
   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      unsigned array_elems =
         reg->num_array_elems == 0 ? 1 : reg->num_array_elems;
      unsigned size = array_elems * reg->num_components;

      nir_locals[reg->index] = dst_reg(GRF, alloc.allocate(size));
   }

   nir_emit_cf_list(&impl->body);
}

void
vec4_visitor::nir_emit_cf_list(exec_list *list)
{
   exec_list_validate(list);
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if:
         /* @TODO */
         break;

      case nir_cf_node_loop:
         /* @TODO */
         break;

      case nir_cf_node_block:
         nir_emit_block(nir_cf_node_as_block(node));
         break;

      default:
         unreachable("Invalid CFG node block");
      }
   }
}

void
vec4_visitor::nir_emit_block(nir_block *block)
{
   nir_foreach_instr(block, instr) {
      nir_emit_instr(instr);
   }
}

void
vec4_visitor::nir_emit_instr(nir_instr *instr)
{
   this->base_ir = instr;

   switch (instr->type) {
   case nir_instr_type_load_const:
      /* We can hit these, but we do nothing now and use them as
       * immediates later in get_nir_src().
       * @FIXME: while this is what fs_nir does, we can do this better in the VS
       * stage because we can emit vector operations and save some MOVs in
       * cases where the constants are representable in 8 bits.
       */
      break;

   case nir_instr_type_intrinsic:
      nir_emit_intrinsic(nir_instr_as_intrinsic(instr));
      break;

   case nir_instr_type_alu:
      nir_emit_alu(nir_instr_as_alu(instr));
      break;

   case nir_instr_type_jump:
      /* @TODO */
      break;

   default:
      fprintf(stderr, "VS instruction not yet implemented by NIR->vec4\n");
      break;
   }
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
vec4_visitor::nir_emit_intrinsic(nir_intrinsic_instr *instr)
{
   dst_reg dest;
   src_reg src;

   switch (instr->intrinsic) {

   case nir_intrinsic_load_input: {
      dest = get_nir_dest(instr->dest);

      dest.writemask = 0;
      for (int i = 0; i < instr->num_components; i++)
         dest.writemask |= 1 << i;

      int offset = instr->const_index[0];
      src = nir_inputs[offset];

      dest = retype(dest, src.type);
      emit(MOV(dest, src));
      break;
   }

   case nir_intrinsic_store_output: {
      src = get_nir_src(instr->src[0]);
      dest = dst_reg(src);

      dest.writemask = 0;
      for (unsigned i = 0; i < instr->num_components; i++)
         dest.writemask |= (1 << i);

      dest = retype(dest, BRW_REGISTER_TYPE_F);

      int offset = instr->const_index[0];
      int output = nir_outputs[offset];
      output_reg[output] = dest;
      break;
   }

   default:
      fprintf(stderr,
              "Non-implemented intrinsic instruction in NIR->vec4 (%d)\n",
              instr->intrinsic);
      break;
   }
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

   case nir_op_i2f:
   case nir_op_u2f:
      inst = emit(MOV(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

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

   case nir_op_imul:
      if (brw->gen >= 8) {
	 emit(MUL(dst, op[0], op[1]));
      } else {
         nir_const_value *value0 = nir_src_as_const_value(instr->src[0].src);
         nir_const_value *value1 = nir_src_as_const_value(instr->src[1].src);

         if (value0 && value0->u[0] < (1 << 16)) {
            if (brw->gen < 7)
               emit(MUL(dst,  op[0], op[1]));
            else
               emit(MUL(dst, op[1], op[0]));
         } else if  (value1 && value1->u[0] < (1 << 16)) {
            if (brw->gen < 7)
               emit(MUL(dst, op[1], op[0]));
            else
               emit(MUL(dst, op[0], op[1]));
         } else {
            struct brw_reg acc = retype(brw_acc_reg(8), dst.type);

            emit(MUL(acc, op[0], op[1]));
            emit(MACH(dst_null_d(), op[0], op[1]));
            emit(MOV(dst, src_reg(acc)));
         }
      }
      break;

   case nir_op_imul_high:
   case nir_op_umul_high: {
      struct brw_reg acc = retype(brw_acc_reg(8), dst.type);

      emit(MUL(acc, op[0], op[1]));
      emit(MACH(dst, op[0], op[1]));
      break;
   }

   case nir_op_frcp:
      inst = emit_math(SHADER_OPCODE_RCP, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fexp2:
      inst = emit_math(SHADER_OPCODE_EXP2, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flog2:
      inst = emit_math(SHADER_OPCODE_LOG2, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fexp:
   case nir_op_flog:
      unreachable("not reached: should be handled by ir_explog_to_explog2");

   case nir_op_fsin:
      inst = emit_math(SHADER_OPCODE_SIN, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fcos:
      inst = emit_math(SHADER_OPCODE_COS, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_idiv:
   case nir_op_udiv:
      emit_math(SHADER_OPCODE_INT_QUOTIENT, dst, op[0], op[1]);
      break;

   case nir_op_umod:
      emit_math(SHADER_OPCODE_INT_REMAINDER, dst, op[0], op[1]);
      break;

   case nir_op_uadd_carry: {
      struct brw_reg acc = retype(brw_acc_reg(8), BRW_REGISTER_TYPE_UD);

      emit(ADDC(dst_null_ud(), op[0], op[1]));
      emit(MOV(dst, src_reg(acc)));
      break;
   }

   case nir_op_usub_borrow: {
      struct brw_reg acc = retype(brw_acc_reg(8), BRW_REGISTER_TYPE_UD);

      emit(SUBB(dst_null_ud(), op[0], op[1]));
      emit(MOV(dst, src_reg(acc)));
      break;
   }

   case nir_op_ldexp:
      unreachable("not reached: should be handled by ldexp_to_arith()");

   case nir_op_fsqrt:
      inst = emit_math(SHADER_OPCODE_SQRT, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_frsq:
      inst = emit_math(SHADER_OPCODE_RSQ, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fpow:
      inst = emit_math(SHADER_OPCODE_POW, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_ftrunc:
      inst = emit(RNDZ(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fceil: {
      src_reg tmp = src_reg(this, glsl_type::float_type);

      op[0].negate = !op[0].negate;
      emit(RNDD(dst_reg(tmp), op[0]));
      tmp.negate = true;
      inst = emit(MOV(dst, tmp));
      inst->saturate = instr->dest.saturate;
      break;
   }

   case nir_op_ffloor:
      inst = emit(RNDD(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_ffract:
      inst = emit(FRC(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fround_even:
      inst = emit(RNDE(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmin:
   case nir_op_imin:
   case nir_op_umin:
      /* @FIXME: I took the code for max and min from the vec4_visitor,
       * in fs_nir and fs_visitor  was a bit different. First, in the fs_visitor
       * before  performing the operation a call to resolve_ud_negate was
       * made for every operand. Second, the emit_minmax code of both
       * the fs_visitor and fs_nir was equivalent to the vec4 emit_minmax
       * code with the exception that (if gen < 6) they do:
       *    emit(CMP(reg_null_d, src0, src1, conditionalmod)); ...
       * instead of:
       *    emit(CMP(dst, src0, src1, conditionalmod)); ...
       * See the fs_visitor and vec4_visitor emit_minmax code for more details.
       */
      inst = emit_minmax(BRW_CONDITIONAL_L, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmax:
   case nir_op_imax:
   case nir_op_umax:
      inst = emit_minmax(BRW_CONDITIONAL_GE, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fddx:
   case nir_op_fddx_coarse:
   case nir_op_fddx_fine:
   case nir_op_fddy:
   case nir_op_fddy_coarse:
   case nir_op_fddy_fine:
      unreachable("derivatives not valid in vertex shader");

   case nir_op_flt:
   case nir_op_ilt:
   case nir_op_ult:
   case nir_op_fge:
   case nir_op_ige:
   case nir_op_uge:
   case nir_op_feq:
   case nir_op_ieq:
   case nir_op_fne:
   case nir_op_ine:
      /* @FIXME: if (gen <=5) both fs_visitor and vec4_visitor call the function
       * resolve_bool_comparison for every operand before doing the emit.
       * This check and function calls are not done in the brw_fs_nir.
       * Check if we need to add it, it could be the case that NIR is not available
       * for (gen <= 5) or this check and calls not needed for other reasons.
       */
      emit(CMP(dst, op[0], op[1],
               brw_conditional_for_nir_comparison(instr->op)));
      break;

      /* @FIXME: for the following logical operations: inot, ixor, ior, iand,
       * brw_fs_nir calls to resolve_source_modifiers for every operand
       * if (gen > =8). This call and check is not present in the fs and vec4
       * visitors. Decide if we want to add it.
       */
   case nir_op_inot:
      inst = emit(NOT(dst, op[0]));
      break;

  case nir_op_ixor:
      emit(XOR(dst, op[0], op[1]));
      break;

   case nir_op_ior:
      emit(OR(dst, op[0], op[1]));
      break;

   case nir_op_iand:
        emit(AND(dst, op[0], op[1]));
      break;

   case nir_op_b2i:
      emit(AND(dst, op[0], src_reg(1)));
      break;
   case nir_op_b2f:
      /* @FIXME: as for the relational operations,  both the fs_visitor and vec4_visitor
       * call to resolve_bool_comparison for the operand if (gen <= 5). Check if it is
       * needed.
      */

      /* @FIXME: fs_visitor and vec4_visitor, do:
       *      op[0].type = BRW_REGISTER_TYPE_D;
       *      result_dst.type = BRW_REGISTER_TYPE_D;
       *      emit(AND(result_dst, op[0], src_reg(0x3f800000u)));
       *      result_dst.type = BRW_REGISTER_TYPE_F;
       * instead of the following emit (C&P from brw_fs_nir):
       */
      emit(AND(retype(dst, BRW_REGISTER_TYPE_UD), op[0], src_reg(0x3f800000u)));
      break;

   case nir_op_f2b:
      emit(CMP(dst, op[0], src_reg(0.0f), BRW_CONDITIONAL_NZ));
      break;
   case nir_op_i2b:
      emit(CMP(dst, op[0],  src_reg(0), BRW_CONDITIONAL_NZ));
      break;

   case nir_op_fnoise1_1:
   case nir_op_fnoise1_2:
   case nir_op_fnoise1_3:
   case nir_op_fnoise1_4:
   case nir_op_fnoise2_1:
   case nir_op_fnoise2_2:
   case nir_op_fnoise2_3:
   case nir_op_fnoise2_4:
   case nir_op_fnoise3_1:
   case nir_op_fnoise3_2:
   case nir_op_fnoise3_3:
   case nir_op_fnoise3_4:
   case nir_op_fnoise4_1:
   case nir_op_fnoise4_2:
   case nir_op_fnoise4_3:
   case nir_op_fnoise4_4:
      unreachable("not reached: should be handled by lower_noise");

   case nir_op_unpack_half_2x16_split_x:
   case nir_op_unpack_half_2x16_split_y:
   case nir_op_pack_half_2x16_split:
      unreachable("not reached: should not occur in vertex shader");

   case nir_op_pack_snorm_2x16:
   case nir_op_pack_unorm_2x16:
   case nir_op_unpack_snorm_2x16:
   case nir_op_unpack_unorm_2x16:
      unreachable("not reached: should be handled by lower_packing_builtins");

   case nir_op_bitfield_reverse:
      emit(BFREV(dst, op[0]));
      break;

   case nir_op_bit_count:
      emit(CBIT(dst, op[0]));
      break;
   case nir_op_ufind_msb:
   case nir_op_ifind_msb: {
      /* @FIXME: FBH only supports UD type for dst, in the fs_visitor
       * and the vec4_visitor, instead of using a retype in the FBH call
       * like in fs_nir, they use a temporary UD register to make the call
       * and a MOV to convert back UD to D after the FBH. Something like this:
       *       src_reg temp = src_reg(this, glsl_type::uint_type);
       *       inst = emit(FBH(dst_reg(temp), op[0]));
       *       emit(MOV(dst, temp));
       * Apart from this, the vec4_visitor also set before the MOV:
       *       inst->dst.writemask = WRITEMASK_XYZW;
       *       temp.swizzle = BRW_SWIZZLE_NOOP;
       * Check if it is enough doing the retype (like in fs_nir) or we need to do the MOV.
       */
      emit(FBH(retype(dst, BRW_REGISTER_TYPE_UD), op[0]));

      /* FBH counts from the MSB side, while GLSL's findMSB() wants the count
       * from the LSB side. If FBH didn't return an error (0xFFFFFFFF), then
       * subtract the result from 31 to convert the MSB count into an LSB count.
       */
      src_reg tmp = src_reg(dst);

      emit(CMP(dst_null_d(), tmp, src_reg(-1), BRW_CONDITIONAL_NZ));
      tmp.negate = true;
      inst = emit(ADD(dst, tmp, src_reg(31)));
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;
   }

   case nir_op_find_lsb:
      emit(FBL(dst, op[0]));
      break;

   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      emit(BFE(dst, op[2], op[1], op[0]));
      break;

   case nir_op_bfm:
      emit(BFI1(dst, op[0], op[1]));
      break;

   case nir_op_bfi:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      emit(BFI2(dst, op[0], op[1], op[2]));
      break;

   case nir_op_bitfield_insert:
      unreachable("not reached: should be handled by "
                  "lower_instructions::bitfield_insert_to_bfm_bfi");

   case nir_op_fsign:
         /* AND(val, 0x80000000) gives the sign bit.
          *
          * Predicated OR ORs 1.0 (0x3f800000) with the sign bit if val is not
          * zero.
          */
         emit(CMP(dst_null_f(), op[0], src_reg(0.0f), BRW_CONDITIONAL_NZ));

         op[0].type = BRW_REGISTER_TYPE_UD;
         dst.type = BRW_REGISTER_TYPE_UD;
         emit(AND(dst, op[0], src_reg(0x80000000u)));

         inst = emit(OR(dst, src_reg(dst), src_reg(0x3f800000u)));
         inst->predicate = BRW_PREDICATE_NORMAL;
         /* @FIXME: brw_fs_nir adds the following code:
          *     if (instr->dest.saturate) {
          *         inst = emit(MOV(dst, dst));
          *         inst->saturate = true;
          *     }
          * and it also uses a temporary result_int register to emit
          * the previous operations.
          * The code related to the saturate and the int register are
          * not present in the fs and vec4 visitors.
          * However, both the fs and vec4 visitors add:
          *       this->result.type = BRW_REGISTER_TYPE_F;
          */
      break;
   case nir_op_isign:
         /*  ASR(val, 31) -> negative val generates 0xffffffff (signed -1).
          *               -> non-negative val generates 0x00000000.
          *  Predicated OR sets 1 if val is positive.
          */
         emit(CMP(dst_null_d(), op[0], src_reg(0), BRW_CONDITIONAL_G));
         emit(ASR(dst, op[0], src_reg(31)));
         inst = emit(OR(dst, src_reg(dst), src_reg(1)));
         inst->predicate = BRW_PREDICATE_NORMAL;
      break;

   case nir_op_ishl:
      emit(SHL(dst, op[0], op[1]));
      break;

   case nir_op_ishr:
      emit(ASR(dst, op[0], op[1]));
      break;

   case nir_op_ushr:
      emit(SHR(dst, op[0], op[1]));
      break;

   case nir_op_ffma:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      inst = emit(MAD(dst, op[2], op[1], op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flrp:
      /* @FIXME: emit_lrp is implicitly calling to fix_3src_operand.
       * Notice that we are not doing it for other operations.
       * Probably we want to do it.
       */
      emit_lrp(dst, op[0], op[1], op[2]);
      break;

   default:
      fprintf(stderr, "Non-implemented ALU operation (%d)\n", instr->op);
      break;
   }
}

}
