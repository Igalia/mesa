/*
 * Copyright © 2016 Bas Nieuwenhuizen
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

#include <llvm/Config/llvm-config.h>

#include "ac_nir_to_llvm.h"
#include "ac_llvm_build.h"
#include "ac_llvm_util.h"
#include "ac_binary.h"
#include "sid.h"
#include "nir/nir.h"
#include "nir/nir_deref.h"
#include "util/bitscan.h"
#include "util/u_math.h"
#include "ac_shader_abi.h"
#include "ac_shader_util.h"

struct ac_nir_context {
	struct ac_llvm_context ac;
	struct ac_shader_abi *abi;

	gl_shader_stage stage;
	shader_info *info;

	LLVMValueRef *ssa_defs;

	LLVMValueRef scratch;
	LLVMValueRef constant_data;

	struct hash_table *defs;
	struct hash_table *phis;
	struct hash_table *vars;

	LLVMValueRef main_function;
	LLVMBasicBlockRef continue_block;
	LLVMBasicBlockRef break_block;

	int num_locals;
	LLVMValueRef *locals;
};

static LLVMValueRef get_sampler_desc(struct ac_nir_context *ctx,
				     nir_deref_instr *deref_instr,
				     enum ac_descriptor_type desc_type,
				     const nir_instr *instr,
				     bool image, bool write);

static void
build_store_values_extended(struct ac_llvm_context *ac,
			     LLVMValueRef *values,
			     unsigned value_count,
			     unsigned value_stride,
			     LLVMValueRef vec)
{
	LLVMBuilderRef builder = ac->builder;
	unsigned i;

	for (i = 0; i < value_count; i++) {
		LLVMValueRef ptr = values[i * value_stride];
		LLVMValueRef index = LLVMConstInt(ac->i32, i, false);
		LLVMValueRef value = LLVMBuildExtractElement(builder, vec, index, "");
		LLVMBuildStore(builder, value, ptr);
	}
}

static enum ac_image_dim
get_ac_sampler_dim(const struct ac_llvm_context *ctx, enum glsl_sampler_dim dim,
		   bool is_array)
{
	switch (dim) {
	case GLSL_SAMPLER_DIM_1D:
		if (ctx->chip_class == GFX9)
			return is_array ? ac_image_2darray : ac_image_2d;
		return is_array ? ac_image_1darray : ac_image_1d;
	case GLSL_SAMPLER_DIM_2D:
	case GLSL_SAMPLER_DIM_RECT:
	case GLSL_SAMPLER_DIM_EXTERNAL:
		return is_array ? ac_image_2darray : ac_image_2d;
	case GLSL_SAMPLER_DIM_3D:
		return ac_image_3d;
	case GLSL_SAMPLER_DIM_CUBE:
		return ac_image_cube;
	case GLSL_SAMPLER_DIM_MS:
		return is_array ? ac_image_2darraymsaa : ac_image_2dmsaa;
	case GLSL_SAMPLER_DIM_SUBPASS:
		return ac_image_2darray;
	case GLSL_SAMPLER_DIM_SUBPASS_MS:
		return ac_image_2darraymsaa;
	default:
		unreachable("bad sampler dim");
	}
}

static enum ac_image_dim
get_ac_image_dim(const struct ac_llvm_context *ctx, enum glsl_sampler_dim sdim,
		 bool is_array)
{
	enum ac_image_dim dim = get_ac_sampler_dim(ctx, sdim, is_array);

	/* Match the resource type set in the descriptor. */
	if (dim == ac_image_cube ||
	    (ctx->chip_class <= GFX8 && dim == ac_image_3d))
		dim = ac_image_2darray;
	else if (sdim == GLSL_SAMPLER_DIM_2D && !is_array && ctx->chip_class == GFX9) {
		/* When a single layer of a 3D texture is bound, the shader
		 * will refer to a 2D target, but the descriptor has a 3D type.
		 * Since the HW ignores BASE_ARRAY in this case, we need to
		 * send 3 coordinates. This doesn't hurt when the underlying
		 * texture is non-3D.
		 */
		dim = ac_image_3d;
	}

	return dim;
}

static LLVMTypeRef get_def_type(struct ac_nir_context *ctx,
                                const nir_ssa_def *def)
{
	LLVMTypeRef type = LLVMIntTypeInContext(ctx->ac.context, def->bit_size);
	if (def->num_components > 1) {
		type = LLVMVectorType(type, def->num_components);
	}
	return type;
}

static LLVMValueRef get_src(struct ac_nir_context *nir, nir_src src)
{
	assert(src.is_ssa);
	return nir->ssa_defs[src.ssa->index];
}

static LLVMValueRef
get_memory_ptr(struct ac_nir_context *ctx, nir_src src)
{
	LLVMValueRef ptr = get_src(ctx, src);
	ptr = LLVMBuildGEP(ctx->ac.builder, ctx->ac.lds, &ptr, 1, "");
	int addr_space = LLVMGetPointerAddressSpace(LLVMTypeOf(ptr));

	return LLVMBuildBitCast(ctx->ac.builder, ptr,
				LLVMPointerType(ctx->ac.i32, addr_space), "");
}

static LLVMBasicBlockRef get_block(struct ac_nir_context *nir,
                                   const struct nir_block *b)
{
	struct hash_entry *entry = _mesa_hash_table_search(nir->defs, b);
	return (LLVMBasicBlockRef)entry->data;
}

static LLVMValueRef get_alu_src(struct ac_nir_context *ctx,
                                nir_alu_src src,
                                unsigned num_components)
{
	LLVMValueRef value = get_src(ctx, src.src);
	bool need_swizzle = false;

	assert(value);
	unsigned src_components = ac_get_llvm_num_components(value);
	for (unsigned i = 0; i < num_components; ++i) {
		assert(src.swizzle[i] < src_components);
		if (src.swizzle[i] != i)
			need_swizzle = true;
	}

	if (need_swizzle || num_components != src_components) {
		LLVMValueRef masks[] = {
		    LLVMConstInt(ctx->ac.i32, src.swizzle[0], false),
		    LLVMConstInt(ctx->ac.i32, src.swizzle[1], false),
		    LLVMConstInt(ctx->ac.i32, src.swizzle[2], false),
		    LLVMConstInt(ctx->ac.i32, src.swizzle[3], false)};

		if (src_components > 1 && num_components == 1) {
			value = LLVMBuildExtractElement(ctx->ac.builder, value,
			                                masks[0], "");
		} else if (src_components == 1 && num_components > 1) {
			LLVMValueRef values[] = {value, value, value, value};
			value = ac_build_gather_values(&ctx->ac, values, num_components);
		} else {
			LLVMValueRef swizzle = LLVMConstVector(masks, num_components);
			value = LLVMBuildShuffleVector(ctx->ac.builder, value, value,
		                                       swizzle, "");
		}
	}
	assert(!src.negate);
	assert(!src.abs);
	return value;
}

static LLVMValueRef emit_int_cmp(struct ac_llvm_context *ctx,
                                 LLVMIntPredicate pred, LLVMValueRef src0,
                                 LLVMValueRef src1)
{
	LLVMValueRef result = LLVMBuildICmp(ctx->builder, pred, src0, src1, "");
	return LLVMBuildSelect(ctx->builder, result,
	                       LLVMConstInt(ctx->i32, 0xFFFFFFFF, false),
	                       ctx->i32_0, "");
}

static LLVMValueRef emit_float_cmp(struct ac_llvm_context *ctx,
                                   LLVMRealPredicate pred, LLVMValueRef src0,
                                   LLVMValueRef src1)
{
	LLVMValueRef result;
	src0 = ac_to_float(ctx, src0);
	src1 = ac_to_float(ctx, src1);
	result = LLVMBuildFCmp(ctx->builder, pred, src0, src1, "");
	return LLVMBuildSelect(ctx->builder, result,
	                       LLVMConstInt(ctx->i32, 0xFFFFFFFF, false),
			       ctx->i32_0, "");
}

static LLVMValueRef emit_intrin_1f_param(struct ac_llvm_context *ctx,
					 const char *intrin,
					 LLVMTypeRef result_type,
					 LLVMValueRef src0)
{
	char name[64];
	LLVMValueRef params[] = {
		ac_to_float(ctx, src0),
	};

	ASSERTED const int length = snprintf(name, sizeof(name), "%s.f%d", intrin,
						 ac_get_elem_bits(ctx, result_type));
	assert(length < sizeof(name));
	return ac_build_intrinsic(ctx, name, result_type, params, 1, AC_FUNC_ATTR_READNONE);
}

static LLVMValueRef emit_intrin_2f_param(struct ac_llvm_context *ctx,
				       const char *intrin,
				       LLVMTypeRef result_type,
				       LLVMValueRef src0, LLVMValueRef src1)
{
	char name[64];
	LLVMValueRef params[] = {
		ac_to_float(ctx, src0),
		ac_to_float(ctx, src1),
	};

	ASSERTED const int length = snprintf(name, sizeof(name), "%s.f%d", intrin,
						 ac_get_elem_bits(ctx, result_type));
	assert(length < sizeof(name));
	return ac_build_intrinsic(ctx, name, result_type, params, 2, AC_FUNC_ATTR_READNONE);
}

static LLVMValueRef emit_intrin_3f_param(struct ac_llvm_context *ctx,
					 const char *intrin,
					 LLVMTypeRef result_type,
					 LLVMValueRef src0, LLVMValueRef src1, LLVMValueRef src2)
{
	char name[64];
	LLVMValueRef params[] = {
		ac_to_float(ctx, src0),
		ac_to_float(ctx, src1),
		ac_to_float(ctx, src2),
	};

	ASSERTED const int length = snprintf(name, sizeof(name), "%s.f%d", intrin,
						 ac_get_elem_bits(ctx, result_type));
	assert(length < sizeof(name));
	return ac_build_intrinsic(ctx, name, result_type, params, 3, AC_FUNC_ATTR_READNONE);
}

static LLVMValueRef emit_bcsel(struct ac_llvm_context *ctx,
			       LLVMValueRef src0, LLVMValueRef src1, LLVMValueRef src2)
{
	assert(LLVMGetTypeKind(LLVMTypeOf(src0)) != LLVMVectorTypeKind);

	LLVMValueRef v = LLVMBuildICmp(ctx->builder, LLVMIntNE, src0,
				       ctx->i32_0, "");
	return LLVMBuildSelect(ctx->builder, v,
			       ac_to_integer_or_pointer(ctx, src1),
			       ac_to_integer_or_pointer(ctx, src2), "");
}

static LLVMValueRef emit_iabs(struct ac_llvm_context *ctx,
			      LLVMValueRef src0)
{
	return ac_build_imax(ctx, src0, LLVMBuildNeg(ctx->builder, src0, ""));
}

static LLVMValueRef emit_uint_carry(struct ac_llvm_context *ctx,
				    const char *intrin,
				    LLVMValueRef src0, LLVMValueRef src1)
{
	LLVMTypeRef ret_type;
	LLVMTypeRef types[] = { ctx->i32, ctx->i1 };
	LLVMValueRef res;
	LLVMValueRef params[] = { src0, src1 };
	ret_type = LLVMStructTypeInContext(ctx->context, types,
					   2, true);

	res = ac_build_intrinsic(ctx, intrin, ret_type,
				 params, 2, AC_FUNC_ATTR_READNONE);

	res = LLVMBuildExtractValue(ctx->builder, res, 1, "");
	res = LLVMBuildZExt(ctx->builder, res, ctx->i32, "");
	return res;
}

static LLVMValueRef emit_b2f(struct ac_llvm_context *ctx,
			     LLVMValueRef src0,
			     unsigned bitsize)
{
	LLVMValueRef result = LLVMBuildAnd(ctx->builder, src0,
					   LLVMBuildBitCast(ctx->builder, LLVMConstReal(ctx->f32, 1.0), ctx->i32, ""),
					   "");
	result = LLVMBuildBitCast(ctx->builder, result, ctx->f32, "");

	switch (bitsize) {
	case 16:
		return LLVMBuildFPTrunc(ctx->builder, result, ctx->f16, "");
	case 32:
		return result;
	case 64:
		return LLVMBuildFPExt(ctx->builder, result, ctx->f64, "");
	default:
		unreachable("Unsupported bit size.");
	}
}

static LLVMValueRef emit_f2b(struct ac_llvm_context *ctx,
			     LLVMValueRef src0)
{
	src0 = ac_to_float(ctx, src0);
	LLVMValueRef zero = LLVMConstNull(LLVMTypeOf(src0));
	return LLVMBuildSExt(ctx->builder,
			     LLVMBuildFCmp(ctx->builder, LLVMRealUNE, src0, zero, ""),
			     ctx->i32, "");
}

static LLVMValueRef emit_b2i(struct ac_llvm_context *ctx,
			     LLVMValueRef src0,
			     unsigned bitsize)
{
	LLVMValueRef result = LLVMBuildAnd(ctx->builder, src0, ctx->i32_1, "");

	switch (bitsize) {
	case 8:
		return LLVMBuildTrunc(ctx->builder, result, ctx->i8, "");
	case 16:
		return LLVMBuildTrunc(ctx->builder, result, ctx->i16, "");
	case 32:
		return result;
	case 64:
		return LLVMBuildZExt(ctx->builder, result, ctx->i64, "");
	default:
		unreachable("Unsupported bit size.");
	}
}

static LLVMValueRef emit_i2b(struct ac_llvm_context *ctx,
			     LLVMValueRef src0)
{
	LLVMValueRef zero = LLVMConstNull(LLVMTypeOf(src0));
	return LLVMBuildSExt(ctx->builder,
			     LLVMBuildICmp(ctx->builder, LLVMIntNE, src0, zero, ""),
			     ctx->i32, "");
}

static LLVMValueRef emit_f2f16(struct ac_llvm_context *ctx,
			       LLVMValueRef src0)
{
	LLVMValueRef result;
	LLVMValueRef cond = NULL;

	src0 = ac_to_float(ctx, src0);
	result = LLVMBuildFPTrunc(ctx->builder, src0, ctx->f16, "");

	if (ctx->chip_class >= GFX8) {
		LLVMValueRef args[2];
		/* Check if the result is a denormal - and flush to 0 if so. */
		args[0] = result;
		args[1] = LLVMConstInt(ctx->i32, N_SUBNORMAL | P_SUBNORMAL, false);
		cond = ac_build_intrinsic(ctx, "llvm.amdgcn.class.f16", ctx->i1, args, 2, AC_FUNC_ATTR_READNONE);
	}

	/* need to convert back up to f32 */
	result = LLVMBuildFPExt(ctx->builder, result, ctx->f32, "");

	if (ctx->chip_class >= GFX8)
		result = LLVMBuildSelect(ctx->builder, cond, ctx->f32_0, result, "");
	else {
		/* for GFX6-GFX7 */
		/* 0x38800000 is smallest half float value (2^-14) in 32-bit float,
		 * so compare the result and flush to 0 if it's smaller.
		 */
		LLVMValueRef temp, cond2;
		temp = emit_intrin_1f_param(ctx, "llvm.fabs", ctx->f32, result);
		cond = LLVMBuildFCmp(ctx->builder, LLVMRealUGT,
				     LLVMBuildBitCast(ctx->builder, LLVMConstInt(ctx->i32, 0x38800000, false), ctx->f32, ""),
				     temp, "");
		cond2 = LLVMBuildFCmp(ctx->builder, LLVMRealUNE,
				      temp, ctx->f32_0, "");
		cond = LLVMBuildAnd(ctx->builder, cond, cond2, "");
		result = LLVMBuildSelect(ctx->builder, cond, ctx->f32_0, result, "");
	}
	return result;
}

static LLVMValueRef emit_umul_high(struct ac_llvm_context *ctx,
				   LLVMValueRef src0, LLVMValueRef src1)
{
	LLVMValueRef dst64, result;
	src0 = LLVMBuildZExt(ctx->builder, src0, ctx->i64, "");
	src1 = LLVMBuildZExt(ctx->builder, src1, ctx->i64, "");

	dst64 = LLVMBuildMul(ctx->builder, src0, src1, "");
	dst64 = LLVMBuildLShr(ctx->builder, dst64, LLVMConstInt(ctx->i64, 32, false), "");
	result = LLVMBuildTrunc(ctx->builder, dst64, ctx->i32, "");
	return result;
}

static LLVMValueRef emit_imul_high(struct ac_llvm_context *ctx,
				   LLVMValueRef src0, LLVMValueRef src1)
{
	LLVMValueRef dst64, result;
	src0 = LLVMBuildSExt(ctx->builder, src0, ctx->i64, "");
	src1 = LLVMBuildSExt(ctx->builder, src1, ctx->i64, "");

	dst64 = LLVMBuildMul(ctx->builder, src0, src1, "");
	dst64 = LLVMBuildAShr(ctx->builder, dst64, LLVMConstInt(ctx->i64, 32, false), "");
	result = LLVMBuildTrunc(ctx->builder, dst64, ctx->i32, "");
	return result;
}

static LLVMValueRef emit_bfm(struct ac_llvm_context *ctx,
			     LLVMValueRef bits, LLVMValueRef offset)
{
	/* mask = ((1 << bits) - 1) << offset */
	return LLVMBuildShl(ctx->builder,
			    LLVMBuildSub(ctx->builder,
					 LLVMBuildShl(ctx->builder,
						      ctx->i32_1,
						      bits, ""),
					 ctx->i32_1, ""),
			    offset, "");
}

static LLVMValueRef emit_bitfield_select(struct ac_llvm_context *ctx,
					 LLVMValueRef mask, LLVMValueRef insert,
					 LLVMValueRef base)
{
	/* Calculate:
	 *   (mask & insert) | (~mask & base) = base ^ (mask & (insert ^ base))
	 * Use the right-hand side, which the LLVM backend can convert to V_BFI.
	 */
	return LLVMBuildXor(ctx->builder, base,
			    LLVMBuildAnd(ctx->builder, mask,
					 LLVMBuildXor(ctx->builder, insert, base, ""), ""), "");
}

static LLVMValueRef emit_pack_2x16(struct ac_llvm_context *ctx,
				   LLVMValueRef src0,
				   LLVMValueRef (*pack)(struct ac_llvm_context *ctx,
							LLVMValueRef args[2]))
{
	LLVMValueRef comp[2];

	src0 = ac_to_float(ctx, src0);
	comp[0] = LLVMBuildExtractElement(ctx->builder, src0, ctx->i32_0, "");
	comp[1] = LLVMBuildExtractElement(ctx->builder, src0, ctx->i32_1, "");

	return LLVMBuildBitCast(ctx->builder, pack(ctx, comp), ctx->i32, "");
}

static LLVMValueRef emit_unpack_half_2x16(struct ac_llvm_context *ctx,
					  LLVMValueRef src0)
{
	LLVMValueRef const16 = LLVMConstInt(ctx->i32, 16, false);
	LLVMValueRef temps[2], val;
	int i;

	for (i = 0; i < 2; i++) {
		val = i == 1 ? LLVMBuildLShr(ctx->builder, src0, const16, "") : src0;
		val = LLVMBuildTrunc(ctx->builder, val, ctx->i16, "");
		val = LLVMBuildBitCast(ctx->builder, val, ctx->f16, "");
		temps[i] = LLVMBuildFPExt(ctx->builder, val, ctx->f32, "");
	}
	return ac_build_gather_values(ctx, temps, 2);
}

static LLVMValueRef emit_ddxy(struct ac_nir_context *ctx,
			      nir_op op,
			      LLVMValueRef src0)
{
	unsigned mask;
	int idx;
	LLVMValueRef result;

	if (op == nir_op_fddx_fine)
		mask = AC_TID_MASK_LEFT;
	else if (op == nir_op_fddy_fine)
		mask = AC_TID_MASK_TOP;
	else
		mask = AC_TID_MASK_TOP_LEFT;

	/* for DDX we want to next X pixel, DDY next Y pixel. */
	if (op == nir_op_fddx_fine ||
	    op == nir_op_fddx_coarse ||
	    op == nir_op_fddx)
		idx = 1;
	else
		idx = 2;

	result = ac_build_ddxy(&ctx->ac, mask, idx, src0);
	return result;
}

static void visit_alu(struct ac_nir_context *ctx, const nir_alu_instr *instr)
{
	LLVMValueRef src[4], result = NULL;
	unsigned num_components = instr->dest.dest.ssa.num_components;
	unsigned src_components;
	LLVMTypeRef def_type = get_def_type(ctx, &instr->dest.dest.ssa);

	assert(nir_op_infos[instr->op].num_inputs <= ARRAY_SIZE(src));
	switch (instr->op) {
	case nir_op_vec2:
	case nir_op_vec3:
	case nir_op_vec4:
		src_components = 1;
		break;
	case nir_op_pack_half_2x16:
	case nir_op_pack_snorm_2x16:
	case nir_op_pack_unorm_2x16:
		src_components = 2;
		break;
	case nir_op_unpack_half_2x16:
		src_components = 1;
		break;
	case nir_op_cube_face_coord:
	case nir_op_cube_face_index:
		src_components = 3;
		break;
	default:
		src_components = num_components;
		break;
	}
	for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++)
		src[i] = get_alu_src(ctx, instr->src[i], src_components);

	switch (instr->op) {
	case nir_op_mov:
		result = src[0];
		break;
	case nir_op_fneg:
	        src[0] = ac_to_float(&ctx->ac, src[0]);
		result = LLVMBuildFNeg(ctx->ac.builder, src[0], "");
		break;
	case nir_op_ineg:
		result = LLVMBuildNeg(ctx->ac.builder, src[0], "");
		break;
	case nir_op_inot:
		result = LLVMBuildNot(ctx->ac.builder, src[0], "");
		break;
	case nir_op_iadd:
		result = LLVMBuildAdd(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_fadd:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		result = LLVMBuildFAdd(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_fsub:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		result = LLVMBuildFSub(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_isub:
		result = LLVMBuildSub(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_imul:
		result = LLVMBuildMul(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_imod:
		result = LLVMBuildSRem(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_umod:
		result = LLVMBuildURem(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_fmod:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		result = ac_build_fdiv(&ctx->ac, src[0], src[1]);
		result = emit_intrin_1f_param(&ctx->ac, "llvm.floor",
		                              ac_to_float_type(&ctx->ac, def_type), result);
		result = LLVMBuildFMul(ctx->ac.builder, src[1] , result, "");
		result = LLVMBuildFSub(ctx->ac.builder, src[0], result, "");
		break;
	case nir_op_frem:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		result = LLVMBuildFRem(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_irem:
		result = LLVMBuildSRem(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_idiv:
		result = LLVMBuildSDiv(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_udiv:
		result = LLVMBuildUDiv(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_fmul:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		result = LLVMBuildFMul(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_frcp:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = ac_build_fdiv(&ctx->ac, LLVMConstReal(LLVMTypeOf(src[0]), 1.0), src[0]);
		break;
	case nir_op_iand:
		result = LLVMBuildAnd(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ior:
		result = LLVMBuildOr(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ixor:
		result = LLVMBuildXor(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ishl:
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) < ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildZExt(ctx->ac.builder, src[1],
					       LLVMTypeOf(src[0]), "");
		else if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) > ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildTrunc(ctx->ac.builder, src[1],
						LLVMTypeOf(src[0]), "");
		result = LLVMBuildShl(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ishr:
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) < ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildZExt(ctx->ac.builder, src[1],
					       LLVMTypeOf(src[0]), "");
		else if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) > ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildTrunc(ctx->ac.builder, src[1],
						LLVMTypeOf(src[0]), "");
		result = LLVMBuildAShr(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ushr:
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) < ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildZExt(ctx->ac.builder, src[1],
					       LLVMTypeOf(src[0]), "");
		else if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) > ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
			src[1] = LLVMBuildTrunc(ctx->ac.builder, src[1],
						LLVMTypeOf(src[0]), "");
		result = LLVMBuildLShr(ctx->ac.builder, src[0], src[1], "");
		break;
	case nir_op_ilt32:
		result = emit_int_cmp(&ctx->ac, LLVMIntSLT, src[0], src[1]);
		break;
	case nir_op_ine32:
		result = emit_int_cmp(&ctx->ac, LLVMIntNE, src[0], src[1]);
		break;
	case nir_op_ieq32:
		result = emit_int_cmp(&ctx->ac, LLVMIntEQ, src[0], src[1]);
		break;
	case nir_op_ige32:
		result = emit_int_cmp(&ctx->ac, LLVMIntSGE, src[0], src[1]);
		break;
	case nir_op_ult32:
		result = emit_int_cmp(&ctx->ac, LLVMIntULT, src[0], src[1]);
		break;
	case nir_op_uge32:
		result = emit_int_cmp(&ctx->ac, LLVMIntUGE, src[0], src[1]);
		break;
	case nir_op_feq32:
		result = emit_float_cmp(&ctx->ac, LLVMRealOEQ, src[0], src[1]);
		break;
	case nir_op_fne32:
		result = emit_float_cmp(&ctx->ac, LLVMRealUNE, src[0], src[1]);
		break;
	case nir_op_flt32:
		result = emit_float_cmp(&ctx->ac, LLVMRealOLT, src[0], src[1]);
		break;
	case nir_op_fge32:
		result = emit_float_cmp(&ctx->ac, LLVMRealOGE, src[0], src[1]);
		break;
	case nir_op_fabs:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.fabs",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_iabs:
		result = emit_iabs(&ctx->ac, src[0]);
		break;
	case nir_op_imax:
		result = ac_build_imax(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_imin:
		result = ac_build_imin(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_umax:
		result = ac_build_umax(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_umin:
		result = ac_build_umin(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_isign:
		result = ac_build_isign(&ctx->ac, src[0],
					instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_fsign:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = ac_build_fsign(&ctx->ac, src[0],
					instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_ffloor:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.floor",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_ftrunc:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.trunc",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_fceil:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.ceil",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_fround_even:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.rint",
		                              ac_to_float_type(&ctx->ac, def_type),src[0]);
		break;
	case nir_op_ffract:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = ac_build_fract(&ctx->ac, src[0],
					instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_fsin:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.sin",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_fcos:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.cos",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_fsqrt:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.sqrt",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_fexp2:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.exp2",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_flog2:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.log2",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		break;
	case nir_op_frsq:
		result = emit_intrin_1f_param(&ctx->ac, "llvm.sqrt",
		                              ac_to_float_type(&ctx->ac, def_type), src[0]);
		result = ac_build_fdiv(&ctx->ac, LLVMConstReal(LLVMTypeOf(result), 1.0), result);
		break;
	case nir_op_frexp_exp:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = ac_build_frexp_exp(&ctx->ac, src[0],
					    ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])));
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) == 16)
			result = LLVMBuildSExt(ctx->ac.builder, result,
					       ctx->ac.i32, "");
		break;
	case nir_op_frexp_sig:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = ac_build_frexp_mant(&ctx->ac, src[0],
					     instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_fpow:
		result = emit_intrin_2f_param(&ctx->ac, "llvm.pow",
		                              ac_to_float_type(&ctx->ac, def_type), src[0], src[1]);
		break;
	case nir_op_fmax:
		result = emit_intrin_2f_param(&ctx->ac, "llvm.maxnum",
		                              ac_to_float_type(&ctx->ac, def_type), src[0], src[1]);
		if (ctx->ac.chip_class < GFX9 &&
		    instr->dest.dest.ssa.bit_size == 32) {
			/* Only pre-GFX9 chips do not flush denorms. */
			result = emit_intrin_1f_param(&ctx->ac, "llvm.canonicalize",
						      ac_to_float_type(&ctx->ac, def_type),
						      result);
		}
		break;
	case nir_op_fmin:
		result = emit_intrin_2f_param(&ctx->ac, "llvm.minnum",
		                              ac_to_float_type(&ctx->ac, def_type), src[0], src[1]);
		if (ctx->ac.chip_class < GFX9 &&
		    instr->dest.dest.ssa.bit_size == 32) {
			/* Only pre-GFX9 chips do not flush denorms. */
			result = emit_intrin_1f_param(&ctx->ac, "llvm.canonicalize",
						      ac_to_float_type(&ctx->ac, def_type),
						      result);
		}
		break;
	case nir_op_ffma:
		/* FMA is better on GFX10, because it has FMA units instead of MUL-ADD units. */
		result = emit_intrin_3f_param(&ctx->ac, ctx->ac.chip_class >= GFX10 ? "llvm.fma" : "llvm.fmuladd",
		                              ac_to_float_type(&ctx->ac, def_type), src[0], src[1], src[2]);
		break;
	case nir_op_ldexp:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		if (ac_get_elem_bits(&ctx->ac, def_type) == 32)
			result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ldexp.f32", ctx->ac.f32, src, 2, AC_FUNC_ATTR_READNONE);
		else if (ac_get_elem_bits(&ctx->ac, def_type) == 16)
			result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ldexp.f16", ctx->ac.f16, src, 2, AC_FUNC_ATTR_READNONE);
		else
			result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ldexp.f64", ctx->ac.f64, src, 2, AC_FUNC_ATTR_READNONE);
		break;
	case nir_op_bfm:
		result = emit_bfm(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_bitfield_select:
		result = emit_bitfield_select(&ctx->ac, src[0], src[1], src[2]);
		break;
	case nir_op_ubfe:
		result = ac_build_bfe(&ctx->ac, src[0], src[1], src[2], false);
		break;
	case nir_op_ibfe:
		result = ac_build_bfe(&ctx->ac, src[0], src[1], src[2], true);
		break;
	case nir_op_bitfield_reverse:
		result = ac_build_bitfield_reverse(&ctx->ac, src[0]);
		break;
	case nir_op_bit_count:
		result = ac_build_bit_count(&ctx->ac, src[0]);
		break;
	case nir_op_vec2:
	case nir_op_vec3:
	case nir_op_vec4:
		for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++)
			src[i] = ac_to_integer(&ctx->ac, src[i]);
		result = ac_build_gather_values(&ctx->ac, src, num_components);
		break;
	case nir_op_f2i8:
	case nir_op_f2i16:
	case nir_op_f2i32:
	case nir_op_f2i64:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = LLVMBuildFPToSI(ctx->ac.builder, src[0], def_type, "");
		break;
	case nir_op_f2u8:
	case nir_op_f2u16:
	case nir_op_f2u32:
	case nir_op_f2u64:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		result = LLVMBuildFPToUI(ctx->ac.builder, src[0], def_type, "");
		break;
	case nir_op_i2f16:
	case nir_op_i2f32:
	case nir_op_i2f64:
		result = LLVMBuildSIToFP(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
		break;
	case nir_op_u2f16:
	case nir_op_u2f32:
	case nir_op_u2f64:
		result = LLVMBuildUIToFP(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
		break;
	case nir_op_f2f16_rtz:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		if (LLVMTypeOf(src[0]) == ctx->ac.f64)
			src[0] = LLVMBuildFPTrunc(ctx->ac.builder, src[0], ctx->ac.f32, "");
		LLVMValueRef param[2] = { src[0], ctx->ac.f32_0 };
		result = ac_build_cvt_pkrtz_f16(&ctx->ac, param);
		result = LLVMBuildExtractElement(ctx->ac.builder, result, ctx->ac.i32_0, "");
		break;
	case nir_op_f2f16_rtne:
	case nir_op_f2f16:
	case nir_op_f2f32:
	case nir_op_f2f64:
		src[0] = ac_to_float(&ctx->ac, src[0]);
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
			result = LLVMBuildFPExt(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
		else
			result = LLVMBuildFPTrunc(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
		break;
	case nir_op_u2u8:
	case nir_op_u2u16:
	case nir_op_u2u32:
	case nir_op_u2u64:
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
			result = LLVMBuildZExt(ctx->ac.builder, src[0], def_type, "");
		else
			result = LLVMBuildTrunc(ctx->ac.builder, src[0], def_type, "");
		break;
	case nir_op_i2i8:
	case nir_op_i2i16:
	case nir_op_i2i32:
	case nir_op_i2i64:
		if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
			result = LLVMBuildSExt(ctx->ac.builder, src[0], def_type, "");
		else
			result = LLVMBuildTrunc(ctx->ac.builder, src[0], def_type, "");
		break;
	case nir_op_b32csel:
		result = emit_bcsel(&ctx->ac, src[0], src[1], src[2]);
		break;
	case nir_op_find_lsb:
		result = ac_find_lsb(&ctx->ac, ctx->ac.i32, src[0]);
		break;
	case nir_op_ufind_msb:
		result = ac_build_umsb(&ctx->ac, src[0], ctx->ac.i32);
		break;
	case nir_op_ifind_msb:
		result = ac_build_imsb(&ctx->ac, src[0], ctx->ac.i32);
		break;
	case nir_op_uadd_carry:
		result = emit_uint_carry(&ctx->ac, "llvm.uadd.with.overflow.i32", src[0], src[1]);
		break;
	case nir_op_usub_borrow:
		result = emit_uint_carry(&ctx->ac, "llvm.usub.with.overflow.i32", src[0], src[1]);
		break;
	case nir_op_b2f16:
	case nir_op_b2f32:
	case nir_op_b2f64:
		result = emit_b2f(&ctx->ac, src[0], instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_f2b32:
		result = emit_f2b(&ctx->ac, src[0]);
		break;
	case nir_op_b2i8:
	case nir_op_b2i16:
	case nir_op_b2i32:
	case nir_op_b2i64:
		result = emit_b2i(&ctx->ac, src[0], instr->dest.dest.ssa.bit_size);
		break;
	case nir_op_i2b32:
		result = emit_i2b(&ctx->ac, src[0]);
		break;
	case nir_op_fquantize2f16:
		result = emit_f2f16(&ctx->ac, src[0]);
		break;
	case nir_op_umul_high:
		result = emit_umul_high(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_imul_high:
		result = emit_imul_high(&ctx->ac, src[0], src[1]);
		break;
	case nir_op_pack_half_2x16:
		result = emit_pack_2x16(&ctx->ac, src[0], ac_build_cvt_pkrtz_f16);
		break;
	case nir_op_pack_snorm_2x16:
		result = emit_pack_2x16(&ctx->ac, src[0], ac_build_cvt_pknorm_i16);
		break;
	case nir_op_pack_unorm_2x16:
		result = emit_pack_2x16(&ctx->ac, src[0], ac_build_cvt_pknorm_u16);
		break;
	case nir_op_unpack_half_2x16:
		result = emit_unpack_half_2x16(&ctx->ac, src[0]);
		break;
	case nir_op_fddx:
	case nir_op_fddy:
	case nir_op_fddx_fine:
	case nir_op_fddy_fine:
	case nir_op_fddx_coarse:
	case nir_op_fddy_coarse:
		result = emit_ddxy(ctx, instr->op, src[0]);
		break;

	case nir_op_unpack_64_2x32_split_x: {
		assert(ac_get_llvm_num_components(src[0]) == 1);
		LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0],
						    ctx->ac.v2i32,
						    "");
		result = LLVMBuildExtractElement(ctx->ac.builder, tmp,
						 ctx->ac.i32_0, "");
		break;
	}

	case nir_op_unpack_64_2x32_split_y: {
		assert(ac_get_llvm_num_components(src[0]) == 1);
		LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0],
						    ctx->ac.v2i32,
						    "");
		result = LLVMBuildExtractElement(ctx->ac.builder, tmp,
						 ctx->ac.i32_1, "");
		break;
	}

	case nir_op_pack_64_2x32_split: {
		LLVMValueRef tmp = ac_build_gather_values(&ctx->ac, src, 2);
		result = LLVMBuildBitCast(ctx->ac.builder, tmp, ctx->ac.i64, "");
		break;
	}

	case nir_op_pack_32_2x16_split: {
		LLVMValueRef tmp = ac_build_gather_values(&ctx->ac, src, 2);
		result = LLVMBuildBitCast(ctx->ac.builder, tmp, ctx->ac.i32, "");
		break;
	}

	case nir_op_unpack_32_2x16_split_x: {
		LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0],
						    ctx->ac.v2i16,
						    "");
		result = LLVMBuildExtractElement(ctx->ac.builder, tmp,
						 ctx->ac.i32_0, "");
		break;
	}

	case nir_op_unpack_32_2x16_split_y: {
		LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0],
						    ctx->ac.v2i16,
						    "");
		result = LLVMBuildExtractElement(ctx->ac.builder, tmp,
						 ctx->ac.i32_1, "");
		break;
	}

	case nir_op_cube_face_coord: {
		src[0] = ac_to_float(&ctx->ac, src[0]);
		LLVMValueRef results[2];
		LLVMValueRef in[3];
		for (unsigned chan = 0; chan < 3; chan++)
			in[chan] = ac_llvm_extract_elem(&ctx->ac, src[0], chan);
		results[0] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubesc",
						ctx->ac.f32, in, 3, AC_FUNC_ATTR_READNONE);
		results[1] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubetc",
						ctx->ac.f32, in, 3, AC_FUNC_ATTR_READNONE);
		LLVMValueRef ma = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubema",
						     ctx->ac.f32, in, 3, AC_FUNC_ATTR_READNONE);
		results[0] = ac_build_fdiv(&ctx->ac, results[0], ma);
		results[1] = ac_build_fdiv(&ctx->ac, results[1], ma);
		LLVMValueRef offset = LLVMConstReal(ctx->ac.f32, 0.5);
		results[0] = LLVMBuildFAdd(ctx->ac.builder, results[0], offset, "");
		results[1] = LLVMBuildFAdd(ctx->ac.builder, results[1], offset, "");
		result = ac_build_gather_values(&ctx->ac, results, 2);
		break;
	}

	case nir_op_cube_face_index: {
		src[0] = ac_to_float(&ctx->ac, src[0]);
		LLVMValueRef in[3];
		for (unsigned chan = 0; chan < 3; chan++)
			in[chan] = ac_llvm_extract_elem(&ctx->ac, src[0], chan);
		result = ac_build_intrinsic(&ctx->ac,  "llvm.amdgcn.cubeid",
						ctx->ac.f32, in, 3, AC_FUNC_ATTR_READNONE);
		break;
	}

	case nir_op_fmin3:
		result = emit_intrin_2f_param(&ctx->ac, "llvm.minnum",
						ac_to_float_type(&ctx->ac, def_type), src[0], src[1]);
		result = emit_intrin_2f_param(&ctx->ac, "llvm.minnum",
						ac_to_float_type(&ctx->ac, def_type), result, src[2]);
		break;
	case nir_op_umin3:
		result = ac_build_umin(&ctx->ac, src[0], src[1]);
		result = ac_build_umin(&ctx->ac, result, src[2]);
		break;
	case nir_op_imin3:
		result = ac_build_imin(&ctx->ac, src[0], src[1]);
		result = ac_build_imin(&ctx->ac, result, src[2]);
		break;
	case nir_op_fmax3:
		result = emit_intrin_2f_param(&ctx->ac, "llvm.maxnum",
						ac_to_float_type(&ctx->ac, def_type), src[0], src[1]);
		result = emit_intrin_2f_param(&ctx->ac, "llvm.maxnum",
						ac_to_float_type(&ctx->ac, def_type), result, src[2]);
		break;
	case nir_op_umax3:
		result = ac_build_umax(&ctx->ac, src[0], src[1]);
		result = ac_build_umax(&ctx->ac, result, src[2]);
		break;
	case nir_op_imax3:
		result = ac_build_imax(&ctx->ac, src[0], src[1]);
		result = ac_build_imax(&ctx->ac, result, src[2]);
		break;
	case nir_op_fmed3: {
		src[0] = ac_to_float(&ctx->ac, src[0]);
		src[1] = ac_to_float(&ctx->ac, src[1]);
		src[2] = ac_to_float(&ctx->ac, src[2]);
		result = ac_build_fmed3(&ctx->ac, src[0], src[1], src[2],
					instr->dest.dest.ssa.bit_size);
		break;
	}
	case nir_op_imed3: {
		LLVMValueRef tmp1 = ac_build_imin(&ctx->ac, src[0], src[1]);
		LLVMValueRef tmp2 = ac_build_imax(&ctx->ac, src[0], src[1]);
		tmp2 = ac_build_imin(&ctx->ac, tmp2, src[2]);
		result = ac_build_imax(&ctx->ac, tmp1, tmp2);
		break;
	}
	case nir_op_umed3: {
		LLVMValueRef tmp1 = ac_build_umin(&ctx->ac, src[0], src[1]);
		LLVMValueRef tmp2 = ac_build_umax(&ctx->ac, src[0], src[1]);
		tmp2 = ac_build_umin(&ctx->ac, tmp2, src[2]);
		result = ac_build_umax(&ctx->ac, tmp1, tmp2);
		break;
	}

	default:
		fprintf(stderr, "Unknown NIR alu instr: ");
		nir_print_instr(&instr->instr, stderr);
		fprintf(stderr, "\n");
		abort();
	}

	if (result) {
		assert(instr->dest.dest.is_ssa);
		result = ac_to_integer_or_pointer(&ctx->ac, result);
		ctx->ssa_defs[instr->dest.dest.ssa.index] = result;
	}
}

static void visit_load_const(struct ac_nir_context *ctx,
                             const nir_load_const_instr *instr)
{
	LLVMValueRef values[4], value = NULL;
	LLVMTypeRef element_type =
	    LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);

	for (unsigned i = 0; i < instr->def.num_components; ++i) {
		switch (instr->def.bit_size) {
		case 8:
			values[i] = LLVMConstInt(element_type,
			                         instr->value[i].u8, false);
			break;
		case 16:
			values[i] = LLVMConstInt(element_type,
			                         instr->value[i].u16, false);
			break;
		case 32:
			values[i] = LLVMConstInt(element_type,
			                         instr->value[i].u32, false);
			break;
		case 64:
			values[i] = LLVMConstInt(element_type,
			                         instr->value[i].u64, false);
			break;
		default:
			fprintf(stderr,
			        "unsupported nir load_const bit_size: %d\n",
			        instr->def.bit_size);
			abort();
		}
	}
	if (instr->def.num_components > 1) {
		value = LLVMConstVector(values, instr->def.num_components);
	} else
		value = values[0];

	ctx->ssa_defs[instr->def.index] = value;
}

static LLVMValueRef
get_buffer_size(struct ac_nir_context *ctx, LLVMValueRef descriptor, bool in_elements)
{
	LLVMValueRef size =
		LLVMBuildExtractElement(ctx->ac.builder, descriptor,
					LLVMConstInt(ctx->ac.i32, 2, false), "");

	/* GFX8 only */
	if (ctx->ac.chip_class == GFX8 && in_elements) {
		/* On GFX8, the descriptor contains the size in bytes,
		 * but TXQ must return the size in elements.
		 * The stride is always non-zero for resources using TXQ.
		 */
		LLVMValueRef stride =
			LLVMBuildExtractElement(ctx->ac.builder, descriptor,
						ctx->ac.i32_1, "");
		stride = LLVMBuildLShr(ctx->ac.builder, stride,
				       LLVMConstInt(ctx->ac.i32, 16, false), "");
		stride = LLVMBuildAnd(ctx->ac.builder, stride,
				      LLVMConstInt(ctx->ac.i32, 0x3fff, false), "");

		size = LLVMBuildUDiv(ctx->ac.builder, size, stride, "");
	}
	return size;
}

/* Gather4 should follow the same rules as bilinear filtering, but the hardware
 * incorrectly forces nearest filtering if the texture format is integer.
 * The only effect it has on Gather4, which always returns 4 texels for
 * bilinear filtering, is that the final coordinates are off by 0.5 of
 * the texel size.
 *
 * The workaround is to subtract 0.5 from the unnormalized coordinates,
 * or (0.5 / size) from the normalized coordinates.
 *
 * However, cube textures with 8_8_8_8 data formats require a different
 * workaround of overriding the num format to USCALED/SSCALED. This would lose
 * precision in 32-bit data formats, so it needs to be applied dynamically at
 * runtime. In this case, return an i1 value that indicates whether the
 * descriptor was overridden (and hence a fixup of the sampler result is needed).
 */
static LLVMValueRef lower_gather4_integer(struct ac_llvm_context *ctx,
					  nir_variable *var,
					  struct ac_image_args *args,
					  const nir_tex_instr *instr)
{
	const struct glsl_type *type = glsl_without_array(var->type);
	enum glsl_base_type stype = glsl_get_sampler_result_type(type);
	LLVMValueRef wa_8888 = NULL;
	LLVMValueRef half_texel[2];
	LLVMValueRef result;

	assert(stype == GLSL_TYPE_INT || stype == GLSL_TYPE_UINT);

	if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
		LLVMValueRef formats;
		LLVMValueRef data_format;
		LLVMValueRef wa_formats;

		formats = LLVMBuildExtractElement(ctx->builder, args->resource, ctx->i32_1, "");

		data_format = LLVMBuildLShr(ctx->builder, formats,
					    LLVMConstInt(ctx->i32, 20, false), "");
		data_format = LLVMBuildAnd(ctx->builder, data_format,
					   LLVMConstInt(ctx->i32, (1u << 6) - 1, false), "");
		wa_8888 = LLVMBuildICmp(
			ctx->builder, LLVMIntEQ, data_format,
			LLVMConstInt(ctx->i32, V_008F14_IMG_DATA_FORMAT_8_8_8_8, false),
			"");

		uint32_t wa_num_format =
			stype == GLSL_TYPE_UINT ?
			S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_USCALED) :
			S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_SSCALED);
		wa_formats = LLVMBuildAnd(ctx->builder, formats,
					  LLVMConstInt(ctx->i32, C_008F14_NUM_FORMAT, false),
					  "");
		wa_formats = LLVMBuildOr(ctx->builder, wa_formats,
					LLVMConstInt(ctx->i32, wa_num_format, false), "");

		formats = LLVMBuildSelect(ctx->builder, wa_8888, wa_formats, formats, "");
		args->resource = LLVMBuildInsertElement(
			ctx->builder, args->resource, formats, ctx->i32_1, "");
	}

	if (instr->sampler_dim == GLSL_SAMPLER_DIM_RECT) {
		assert(!wa_8888);
		half_texel[0] = half_texel[1] = LLVMConstReal(ctx->f32, -0.5);
	} else {
		struct ac_image_args resinfo = {};
		LLVMBasicBlockRef bbs[2];

		LLVMValueRef unnorm = NULL;
		LLVMValueRef default_offset = ctx->f32_0;
		if (instr->sampler_dim == GLSL_SAMPLER_DIM_2D &&
		    !instr->is_array) {
			/* In vulkan, whether the sampler uses unnormalized
			 * coordinates or not is a dynamic property of the
			 * sampler. Hence, to figure out whether or not we
			 * need to divide by the texture size, we need to test
			 * the sampler at runtime. This tests the bit set by
			 * radv_init_sampler().
			 */
			LLVMValueRef sampler0 =
				LLVMBuildExtractElement(ctx->builder, args->sampler, ctx->i32_0, "");
			sampler0 = LLVMBuildLShr(ctx->builder, sampler0,
						 LLVMConstInt(ctx->i32, 15, false), "");
			sampler0 = LLVMBuildAnd(ctx->builder, sampler0, ctx->i32_1, "");
			unnorm = LLVMBuildICmp(ctx->builder, LLVMIntEQ, sampler0, ctx->i32_1, "");
			default_offset = LLVMConstReal(ctx->f32, -0.5);
		}

		bbs[0] = LLVMGetInsertBlock(ctx->builder);
		if (wa_8888 || unnorm) {
			assert(!(wa_8888 && unnorm));
			LLVMValueRef not_needed = wa_8888 ? wa_8888 : unnorm;
			/* Skip the texture size query entirely if we don't need it. */
			ac_build_ifcc(ctx, LLVMBuildNot(ctx->builder, not_needed, ""), 2000);
			bbs[1] = LLVMGetInsertBlock(ctx->builder);
		}

		/* Query the texture size. */
		resinfo.dim = get_ac_sampler_dim(ctx, instr->sampler_dim, instr->is_array);
		resinfo.opcode = ac_image_get_resinfo;
		resinfo.dmask = 0xf;
		resinfo.lod = ctx->i32_0;
		resinfo.resource = args->resource;
		resinfo.attributes = AC_FUNC_ATTR_READNONE;
		LLVMValueRef size = ac_build_image_opcode(ctx, &resinfo);

		/* Compute -0.5 / size. */
		for (unsigned c = 0; c < 2; c++) {
			half_texel[c] =
				LLVMBuildExtractElement(ctx->builder, size,
							LLVMConstInt(ctx->i32, c, 0), "");
			half_texel[c] = LLVMBuildUIToFP(ctx->builder, half_texel[c], ctx->f32, "");
			half_texel[c] = ac_build_fdiv(ctx, ctx->f32_1, half_texel[c]);
			half_texel[c] = LLVMBuildFMul(ctx->builder, half_texel[c],
						      LLVMConstReal(ctx->f32, -0.5), "");
		}

		if (wa_8888 || unnorm) {
			ac_build_endif(ctx, 2000);

			for (unsigned c = 0; c < 2; c++) {
				LLVMValueRef values[2] = { default_offset, half_texel[c] };
				half_texel[c] = ac_build_phi(ctx, ctx->f32, 2,
							     values, bbs);
			}
		}
	}

	for (unsigned c = 0; c < 2; c++) {
		LLVMValueRef tmp;
		tmp = LLVMBuildBitCast(ctx->builder, args->coords[c], ctx->f32, "");
		args->coords[c] = LLVMBuildFAdd(ctx->builder, tmp, half_texel[c], "");
	}

	args->attributes = AC_FUNC_ATTR_READNONE;
	result = ac_build_image_opcode(ctx, args);

	if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
		LLVMValueRef tmp, tmp2;

		/* if the cube workaround is in place, f2i the result. */
		for (unsigned c = 0; c < 4; c++) {
			tmp = LLVMBuildExtractElement(ctx->builder, result, LLVMConstInt(ctx->i32, c, false), "");
			if (stype == GLSL_TYPE_UINT)
				tmp2 = LLVMBuildFPToUI(ctx->builder, tmp, ctx->i32, "");
			else
				tmp2 = LLVMBuildFPToSI(ctx->builder, tmp, ctx->i32, "");
			tmp = LLVMBuildBitCast(ctx->builder, tmp, ctx->i32, "");
			tmp2 = LLVMBuildBitCast(ctx->builder, tmp2, ctx->i32, "");
			tmp = LLVMBuildSelect(ctx->builder, wa_8888, tmp2, tmp, "");
			tmp = LLVMBuildBitCast(ctx->builder, tmp, ctx->f32, "");
			result = LLVMBuildInsertElement(ctx->builder, result, tmp, LLVMConstInt(ctx->i32, c, false), "");
		}
	}
	return result;
}

static nir_deref_instr *get_tex_texture_deref(const nir_tex_instr *instr)
{
	nir_deref_instr *texture_deref_instr = NULL;

	for (unsigned i = 0; i < instr->num_srcs; i++) {
		switch (instr->src[i].src_type) {
		case nir_tex_src_texture_deref:
			texture_deref_instr = nir_src_as_deref(instr->src[i].src);
			break;
		default:
			break;
		}
	}
	return texture_deref_instr;
}

static LLVMValueRef build_tex_intrinsic(struct ac_nir_context *ctx,
					const nir_tex_instr *instr,
					struct ac_image_args *args)
{
	if (instr->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
		unsigned mask = nir_ssa_def_components_read(&instr->dest.ssa);

		return ac_build_buffer_load_format(&ctx->ac,
			                           args->resource,
			                           args->coords[0],
			                           ctx->ac.i32_0,
			                           util_last_bit(mask),
			                           0, true);
	}

	args->opcode = ac_image_sample;

	switch (instr->op) {
	case nir_texop_txf:
	case nir_texop_txf_ms:
	case nir_texop_samples_identical:
		args->opcode = args->level_zero ||
			       instr->sampler_dim == GLSL_SAMPLER_DIM_MS ?
					ac_image_load : ac_image_load_mip;
		args->level_zero = false;
		break;
	case nir_texop_txs:
	case nir_texop_query_levels:
		args->opcode = ac_image_get_resinfo;
		if (!args->lod)
			args->lod = ctx->ac.i32_0;
		args->level_zero = false;
		break;
	case nir_texop_tex:
		if (ctx->stage != MESA_SHADER_FRAGMENT) {
			assert(!args->lod);
			args->level_zero = true;
		}
		break;
	case nir_texop_tg4:
		args->opcode = ac_image_gather4;
		args->level_zero = true;
		break;
	case nir_texop_lod:
		args->opcode = ac_image_get_lod;
		break;
	default:
		break;
	}

	if (instr->op == nir_texop_tg4 && ctx->ac.chip_class <= GFX8) {
		nir_deref_instr *texture_deref_instr = get_tex_texture_deref(instr);
		nir_variable *var = nir_deref_instr_get_variable(texture_deref_instr);
		const struct glsl_type *type = glsl_without_array(var->type);
		enum glsl_base_type stype = glsl_get_sampler_result_type(type);
		if (stype == GLSL_TYPE_UINT || stype == GLSL_TYPE_INT) {
			return lower_gather4_integer(&ctx->ac, var, args, instr);
		}
	}

	/* Fixup for GFX9 which allocates 1D textures as 2D. */
	if (instr->op == nir_texop_lod && ctx->ac.chip_class == GFX9) {
		if ((args->dim == ac_image_2darray ||
		     args->dim == ac_image_2d) && !args->coords[1]) {
			args->coords[1] = ctx->ac.i32_0;
		}
	}

	args->attributes = AC_FUNC_ATTR_READNONE;
	bool cs_derivs = ctx->stage == MESA_SHADER_COMPUTE &&
			 ctx->info->cs.derivative_group != DERIVATIVE_GROUP_NONE;
	if (ctx->stage == MESA_SHADER_FRAGMENT || cs_derivs) {
		/* Prevent texture instructions with implicit derivatives from being
		 * sinked into branches. */
		switch (instr->op) {
		case nir_texop_tex:
		case nir_texop_txb:
		case nir_texop_lod:
			args->attributes |= AC_FUNC_ATTR_CONVERGENT;
			break;
		default:
			break;
		}
	}

	return ac_build_image_opcode(&ctx->ac, args);
}

static LLVMValueRef visit_vulkan_resource_reindex(struct ac_nir_context *ctx,
                                                  nir_intrinsic_instr *instr)
{
	LLVMValueRef ptr = get_src(ctx, instr->src[0]);
	LLVMValueRef index = get_src(ctx, instr->src[1]);

	LLVMValueRef result = LLVMBuildGEP(ctx->ac.builder, ptr, &index, 1, "");
	LLVMSetMetadata(result, ctx->ac.uniform_md_kind, ctx->ac.empty_md);
	return result;
}

static LLVMValueRef visit_load_push_constant(struct ac_nir_context *ctx,
                                             nir_intrinsic_instr *instr)
{
	LLVMValueRef ptr, addr;
	LLVMValueRef src0 = get_src(ctx, instr->src[0]);
	unsigned index = nir_intrinsic_base(instr);

	addr = LLVMConstInt(ctx->ac.i32, index, 0);
	addr = LLVMBuildAdd(ctx->ac.builder, addr, src0, "");

	/* Load constant values from user SGPRS when possible, otherwise
	 * fallback to the default path that loads directly from memory.
	 */
	if (LLVMIsConstant(src0) &&
	    instr->dest.ssa.bit_size == 32) {
		unsigned count = instr->dest.ssa.num_components;
		unsigned offset = index;

		offset += LLVMConstIntGetZExtValue(src0);
		offset /= 4;

		offset -= ctx->abi->base_inline_push_consts;

		if (offset + count <= ctx->abi->num_inline_push_consts) {
			return ac_build_gather_values(&ctx->ac,
						      ctx->abi->inline_push_consts + offset,
						      count);
		}
	}

	ptr = LLVMBuildGEP(ctx->ac.builder, ctx->abi->push_constants, &addr, 1, "");

	if (instr->dest.ssa.bit_size == 8) {
		unsigned load_dwords = instr->dest.ssa.num_components > 1 ? 2 : 1;
		LLVMTypeRef vec_type = LLVMVectorType(LLVMInt8TypeInContext(ctx->ac.context), 4 * load_dwords);
		ptr = ac_cast_ptr(&ctx->ac, ptr, vec_type);
		LLVMValueRef res = LLVMBuildLoad(ctx->ac.builder, ptr, "");

		LLVMValueRef params[3];
		if (load_dwords > 1) {
			LLVMValueRef res_vec = LLVMBuildBitCast(ctx->ac.builder, res, LLVMVectorType(ctx->ac.i32, 2), "");
			params[0] = LLVMBuildExtractElement(ctx->ac.builder, res_vec, LLVMConstInt(ctx->ac.i32, 1, false), "");
			params[1] = LLVMBuildExtractElement(ctx->ac.builder, res_vec, LLVMConstInt(ctx->ac.i32, 0, false), "");
		} else {
			res = LLVMBuildBitCast(ctx->ac.builder, res, ctx->ac.i32, "");
			params[0] = ctx->ac.i32_0;
			params[1] = res;
		}
		params[2] = addr;
		res = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.alignbyte", ctx->ac.i32, params, 3, 0);

		res = LLVMBuildTrunc(ctx->ac.builder, res, LLVMIntTypeInContext(ctx->ac.context, instr->dest.ssa.num_components * 8), "");
		if (instr->dest.ssa.num_components > 1)
			res = LLVMBuildBitCast(ctx->ac.builder, res, LLVMVectorType(LLVMInt8TypeInContext(ctx->ac.context), instr->dest.ssa.num_components), "");
		return res;
	} else if (instr->dest.ssa.bit_size == 16) {
		unsigned load_dwords = instr->dest.ssa.num_components / 2 + 1;
		LLVMTypeRef vec_type = LLVMVectorType(LLVMInt16TypeInContext(ctx->ac.context), 2 * load_dwords);
		ptr = ac_cast_ptr(&ctx->ac, ptr, vec_type);
		LLVMValueRef res = LLVMBuildLoad(ctx->ac.builder, ptr, "");
		res = LLVMBuildBitCast(ctx->ac.builder, res, vec_type, "");
		LLVMValueRef cond = LLVMBuildLShr(ctx->ac.builder, addr, ctx->ac.i32_1, "");
		cond = LLVMBuildTrunc(ctx->ac.builder, cond, ctx->ac.i1, "");
		LLVMValueRef mask[] = { LLVMConstInt(ctx->ac.i32, 0, false), LLVMConstInt(ctx->ac.i32, 1, false),
					LLVMConstInt(ctx->ac.i32, 2, false), LLVMConstInt(ctx->ac.i32, 3, false),
					LLVMConstInt(ctx->ac.i32, 4, false)};
		LLVMValueRef swizzle_aligned = LLVMConstVector(&mask[0], instr->dest.ssa.num_components);
		LLVMValueRef swizzle_unaligned = LLVMConstVector(&mask[1], instr->dest.ssa.num_components);
		LLVMValueRef shuffle_aligned = LLVMBuildShuffleVector(ctx->ac.builder, res, res, swizzle_aligned, "");
		LLVMValueRef shuffle_unaligned = LLVMBuildShuffleVector(ctx->ac.builder, res, res, swizzle_unaligned, "");
		res = LLVMBuildSelect(ctx->ac.builder, cond, shuffle_unaligned, shuffle_aligned, "");
		return LLVMBuildBitCast(ctx->ac.builder, res, get_def_type(ctx, &instr->dest.ssa), "");
	}

	ptr = ac_cast_ptr(&ctx->ac, ptr, get_def_type(ctx, &instr->dest.ssa));

	return LLVMBuildLoad(ctx->ac.builder, ptr, "");
}

static LLVMValueRef visit_get_buffer_size(struct ac_nir_context *ctx,
                                          const nir_intrinsic_instr *instr)
{
	LLVMValueRef index = get_src(ctx, instr->src[0]);

	return get_buffer_size(ctx, ctx->abi->load_ssbo(ctx->abi, index, false), false);
}

static uint32_t widen_mask(uint32_t mask, unsigned multiplier)
{
	uint32_t new_mask = 0;
	for(unsigned i = 0; i < 32 && (1u << i) <= mask; ++i)
		if (mask & (1u << i))
			new_mask |= ((1u << multiplier) - 1u) << (i * multiplier);
	return new_mask;
}

static LLVMValueRef extract_vector_range(struct ac_llvm_context *ctx, LLVMValueRef src,
                                         unsigned start, unsigned count)
{
	LLVMValueRef mask[] = {
	ctx->i32_0, ctx->i32_1,
	LLVMConstInt(ctx->i32, 2, false), LLVMConstInt(ctx->i32, 3, false) };

	unsigned src_elements = ac_get_llvm_num_components(src);

	if (count == src_elements) {
		assert(start == 0);
		return src;
	} else if (count == 1) {
		assert(start < src_elements);
		return LLVMBuildExtractElement(ctx->builder, src, mask[start],  "");
	} else {
		assert(start + count <= src_elements);
		assert(count <= 4);
		LLVMValueRef swizzle = LLVMConstVector(&mask[start], count);
		return LLVMBuildShuffleVector(ctx->builder, src, src, swizzle, "");
	}
}

static unsigned get_cache_policy(struct ac_nir_context *ctx,
				 enum gl_access_qualifier access,
				 bool may_store_unaligned,
				 bool writeonly_memory)
{
	unsigned cache_policy = 0;

	/* GFX6 has a TC L1 bug causing corruption of 8bit/16bit stores.  All
	 * store opcodes not aligned to a dword are affected. The only way to
	 * get unaligned stores is through shader images.
	 */
	if (((may_store_unaligned && ctx->ac.chip_class == GFX6) ||
	     /* If this is write-only, don't keep data in L1 to prevent
	      * evicting L1 cache lines that may be needed by other
	      * instructions.
	      */
	     writeonly_memory ||
	     access & (ACCESS_COHERENT | ACCESS_VOLATILE))) {
		cache_policy |= ac_glc;
	}

	if (access & ACCESS_STREAM_CACHE_POLICY)
		cache_policy |= ac_slc;

	return cache_policy;
}

static void visit_store_ssbo(struct ac_nir_context *ctx,
                             nir_intrinsic_instr *instr)
{
	LLVMValueRef src_data = get_src(ctx, instr->src[0]);
	int elem_size_bytes = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src_data)) / 8;
	unsigned writemask = nir_intrinsic_write_mask(instr);
	enum gl_access_qualifier access = nir_intrinsic_access(instr);
	bool writeonly_memory = access & ACCESS_NON_READABLE;
	unsigned cache_policy = get_cache_policy(ctx, access, false, writeonly_memory);

	LLVMValueRef rsrc = ctx->abi->load_ssbo(ctx->abi,
				        get_src(ctx, instr->src[1]), true);
	LLVMValueRef base_data = src_data;
	base_data = ac_trim_vector(&ctx->ac, base_data, instr->num_components);
	LLVMValueRef base_offset = get_src(ctx, instr->src[2]);

	while (writemask) {
		int start, count;
		LLVMValueRef data, offset;
		LLVMTypeRef data_type;

		u_bit_scan_consecutive_range(&writemask, &start, &count);

		/* Due to an LLVM limitation with LLVM < 9, split 3-element
		 * writes into a 2-element and a 1-element write. */
		if (count == 3 &&
		    (elem_size_bytes != 4 || !ac_has_vec3_support(ctx->ac.chip_class, false))) {
			writemask |= 1 << (start + 2);
			count = 2;
		}
		int num_bytes = count * elem_size_bytes; /* count in bytes */

		/* we can only store 4 DWords at the same time.
		 * can only happen for 64 Bit vectors. */
		if (num_bytes > 16) {
			writemask |= ((1u << (count - 2)) - 1u) << (start + 2);
			count = 2;
			num_bytes = 16;
		}

		/* check alignment of 16 Bit stores */
		if (elem_size_bytes == 2 && num_bytes > 2 && (start % 2) == 1) {
			writemask |= ((1u << (count - 1)) - 1u) << (start + 1);
			count = 1;
			num_bytes = 2;
		}
		data = extract_vector_range(&ctx->ac, base_data, start, count);

		offset = LLVMBuildAdd(ctx->ac.builder, base_offset,
				      LLVMConstInt(ctx->ac.i32, start * elem_size_bytes, false), "");

		if (num_bytes == 1) {
			ac_build_tbuffer_store_byte(&ctx->ac, rsrc, data,
						    offset, ctx->ac.i32_0,
						    cache_policy);
		} else if (num_bytes == 2) {
			ac_build_tbuffer_store_short(&ctx->ac, rsrc, data,
						     offset, ctx->ac.i32_0,
						     cache_policy);
		} else {
			int num_channels = num_bytes / 4;

			switch (num_bytes) {
			case 16: /* v4f32 */
				data_type = ctx->ac.v4f32;
				break;
			case 12: /* v3f32 */
				data_type = ctx->ac.v3f32;
				break;
			case 8: /* v2f32 */
				data_type = ctx->ac.v2f32;
				break;
			case 4: /* f32 */
				data_type = ctx->ac.f32;
				break;
			default:
				unreachable("Malformed vector store.");
			}
			data = LLVMBuildBitCast(ctx->ac.builder, data, data_type, "");

			ac_build_buffer_store_dword(&ctx->ac, rsrc, data,
						    num_channels, offset,
						    ctx->ac.i32_0, 0,
						    cache_policy, false);
		}
	}
}

static LLVMValueRef emit_ssbo_comp_swap_64(struct ac_nir_context *ctx,
                                           LLVMValueRef descriptor,
					   LLVMValueRef offset,
					   LLVMValueRef compare,
					   LLVMValueRef exchange)
{
	LLVMBasicBlockRef start_block = NULL, then_block = NULL;
	if (ctx->abi->robust_buffer_access) {
		LLVMValueRef size = ac_llvm_extract_elem(&ctx->ac, descriptor, 2);

		LLVMValueRef cond = LLVMBuildICmp(ctx->ac.builder, LLVMIntULT, offset, size, "");
		start_block = LLVMGetInsertBlock(ctx->ac.builder);

		ac_build_ifcc(&ctx->ac, cond, -1);

		then_block = LLVMGetInsertBlock(ctx->ac.builder);
	}

	LLVMValueRef ptr_parts[2] = {
		ac_llvm_extract_elem(&ctx->ac, descriptor, 0),
		LLVMBuildAnd(ctx->ac.builder,
		             ac_llvm_extract_elem(&ctx->ac, descriptor, 1),
		             LLVMConstInt(ctx->ac.i32, 65535, 0), "")
	};

	ptr_parts[1] = LLVMBuildTrunc(ctx->ac.builder, ptr_parts[1], ctx->ac.i16, "");
	ptr_parts[1] = LLVMBuildSExt(ctx->ac.builder, ptr_parts[1], ctx->ac.i32, "");

	offset = LLVMBuildZExt(ctx->ac.builder, offset, ctx->ac.i64, "");

	LLVMValueRef ptr = ac_build_gather_values(&ctx->ac, ptr_parts, 2);
	ptr = LLVMBuildBitCast(ctx->ac.builder, ptr, ctx->ac.i64, "");
	ptr = LLVMBuildAdd(ctx->ac.builder, ptr, offset, "");
	ptr = LLVMBuildIntToPtr(ctx->ac.builder, ptr, LLVMPointerType(ctx->ac.i64, AC_ADDR_SPACE_GLOBAL), "");

	LLVMValueRef result = ac_build_atomic_cmp_xchg(&ctx->ac, ptr, compare, exchange, "singlethread-one-as");
	result = LLVMBuildExtractValue(ctx->ac.builder, result, 0, "");

	if (ctx->abi->robust_buffer_access) {
		ac_build_endif(&ctx->ac, -1);

		LLVMBasicBlockRef incoming_blocks[2] = {
			start_block,
			then_block,
		};

		LLVMValueRef incoming_values[2] = {
			LLVMConstInt(ctx->ac.i64, 0, 0),
			result,
		};
		LLVMValueRef ret = LLVMBuildPhi(ctx->ac.builder, ctx->ac.i64, "");
		LLVMAddIncoming(ret, incoming_values, incoming_blocks, 2);
		return ret;
	} else {
		return result;
	}
}

static LLVMValueRef visit_atomic_ssbo(struct ac_nir_context *ctx,
                                      const nir_intrinsic_instr *instr)
{
	LLVMTypeRef return_type = LLVMTypeOf(get_src(ctx, instr->src[2]));
	const char *op;
	char name[64], type[8];
	LLVMValueRef params[6], descriptor;
	int arg_count = 0;

	switch (instr->intrinsic) {
	case nir_intrinsic_ssbo_atomic_add:
		op = "add";
		break;
	case nir_intrinsic_ssbo_atomic_imin:
		op = "smin";
		break;
	case nir_intrinsic_ssbo_atomic_umin:
		op = "umin";
		break;
	case nir_intrinsic_ssbo_atomic_imax:
		op = "smax";
		break;
	case nir_intrinsic_ssbo_atomic_umax:
		op = "umax";
		break;
	case nir_intrinsic_ssbo_atomic_and:
		op = "and";
		break;
	case nir_intrinsic_ssbo_atomic_or:
		op = "or";
		break;
	case nir_intrinsic_ssbo_atomic_xor:
		op = "xor";
		break;
	case nir_intrinsic_ssbo_atomic_exchange:
		op = "swap";
		break;
	case nir_intrinsic_ssbo_atomic_comp_swap:
		op = "cmpswap";
		break;
	default:
		abort();
	}

	descriptor = ctx->abi->load_ssbo(ctx->abi,
	                                 get_src(ctx, instr->src[0]),
	                                 true);

	if (instr->intrinsic == nir_intrinsic_ssbo_atomic_comp_swap &&
	    return_type == ctx->ac.i64) {
		return emit_ssbo_comp_swap_64(ctx, descriptor,
					      get_src(ctx, instr->src[1]),
					      get_src(ctx, instr->src[2]),
					      get_src(ctx, instr->src[3]));
	}
	if (instr->intrinsic == nir_intrinsic_ssbo_atomic_comp_swap) {
		params[arg_count++] = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[3]), 0);
	}
	params[arg_count++] = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[2]), 0);
	params[arg_count++] = descriptor;

	if (LLVM_VERSION_MAJOR >= 9) {
		/* XXX: The new raw/struct atomic intrinsics are buggy with
		 * LLVM 8, see r358579.
		 */
		params[arg_count++] = get_src(ctx, instr->src[1]); /* voffset */
		params[arg_count++] = ctx->ac.i32_0; /* soffset */
		params[arg_count++] = ctx->ac.i32_0; /* slc */

		ac_build_type_name_for_intr(return_type, type, sizeof(type));
		snprintf(name, sizeof(name),
		         "llvm.amdgcn.raw.buffer.atomic.%s.%s", op, type);
	} else {
		params[arg_count++] = ctx->ac.i32_0; /* vindex */
		params[arg_count++] = get_src(ctx, instr->src[1]); /* voffset */
		params[arg_count++] = ctx->ac.i1false; /* slc */

		assert(return_type == ctx->ac.i32);
		snprintf(name, sizeof(name),
			 "llvm.amdgcn.buffer.atomic.%s", op);
	}

	return ac_build_intrinsic(&ctx->ac, name, return_type, params,
				  arg_count, 0);
}

static LLVMValueRef visit_load_buffer(struct ac_nir_context *ctx,
                                      const nir_intrinsic_instr *instr)
{
	int elem_size_bytes = instr->dest.ssa.bit_size / 8;
	int num_components = instr->num_components;
	enum gl_access_qualifier access = nir_intrinsic_access(instr);
	unsigned cache_policy = get_cache_policy(ctx, access, false, false);

	LLVMValueRef offset = get_src(ctx, instr->src[1]);
	LLVMValueRef rsrc = ctx->abi->load_ssbo(ctx->abi,
						get_src(ctx, instr->src[0]), false);
	LLVMValueRef vindex = ctx->ac.i32_0;

	LLVMTypeRef def_type = get_def_type(ctx, &instr->dest.ssa);
	LLVMTypeRef def_elem_type = num_components > 1 ? LLVMGetElementType(def_type) : def_type;

	LLVMValueRef results[4];
	for (int i = 0; i < num_components;) {
		int num_elems = num_components - i;
		if (elem_size_bytes < 4 && nir_intrinsic_align(instr) % 4 != 0)
			num_elems = 1;
		if (num_elems * elem_size_bytes > 16)
			num_elems = 16 / elem_size_bytes;
		int load_bytes = num_elems * elem_size_bytes;

		LLVMValueRef immoffset = LLVMConstInt(ctx->ac.i32, i * elem_size_bytes, false);

		LLVMValueRef ret;

		if (load_bytes == 1) {
			ret = ac_build_tbuffer_load_byte(&ctx->ac,
							  rsrc,
							  offset,
							  ctx->ac.i32_0,
							  immoffset,
							  cache_policy);
		} else if (load_bytes == 2) {
			ret = ac_build_tbuffer_load_short(&ctx->ac,
							 rsrc,
							 offset,
							 ctx->ac.i32_0,
							 immoffset,
							 cache_policy);
		} else {
			int num_channels = util_next_power_of_two(load_bytes) / 4;
			bool can_speculate = access & ACCESS_CAN_REORDER;

			ret = ac_build_buffer_load(&ctx->ac, rsrc, num_channels,
						   vindex, offset, immoffset, 0,
						   cache_policy, can_speculate, false);
		}

		LLVMTypeRef byte_vec = LLVMVectorType(ctx->ac.i8, ac_get_type_size(LLVMTypeOf(ret)));
		ret = LLVMBuildBitCast(ctx->ac.builder, ret, byte_vec, "");
		ret = ac_trim_vector(&ctx->ac, ret, load_bytes);

		LLVMTypeRef ret_type = LLVMVectorType(def_elem_type, num_elems);
		ret = LLVMBuildBitCast(ctx->ac.builder, ret, ret_type, "");

		for (unsigned j = 0; j < num_elems; j++) {
			results[i + j] = LLVMBuildExtractElement(ctx->ac.builder, ret, LLVMConstInt(ctx->ac.i32, j, false), "");
		}
		i += num_elems;
	}

	return ac_build_gather_values(&ctx->ac, results, num_components);
}

static LLVMValueRef visit_load_ubo_buffer(struct ac_nir_context *ctx,
                                          const nir_intrinsic_instr *instr)
{
	LLVMValueRef ret;
	LLVMValueRef rsrc = get_src(ctx, instr->src[0]);
	LLVMValueRef offset = get_src(ctx, instr->src[1]);
	int num_components = instr->num_components;

	if (ctx->abi->load_ubo)
		rsrc = ctx->abi->load_ubo(ctx->abi, rsrc);

	if (instr->dest.ssa.bit_size == 64)
		num_components *= 2;

	if (instr->dest.ssa.bit_size == 16 || instr->dest.ssa.bit_size == 8) {
		unsigned load_bytes = instr->dest.ssa.bit_size / 8;
		LLVMValueRef results[num_components];
		for (unsigned i = 0; i < num_components; ++i) {
			LLVMValueRef immoffset = LLVMConstInt(ctx->ac.i32,
							      load_bytes * i, 0);

			if (load_bytes == 1) {
				results[i] = ac_build_tbuffer_load_byte(&ctx->ac,
									rsrc,
									offset,
									ctx->ac.i32_0,
									immoffset,
									0);
			} else {
				assert(load_bytes == 2);
				results[i] = ac_build_tbuffer_load_short(&ctx->ac,
									 rsrc,
									 offset,
									 ctx->ac.i32_0,
									 immoffset,
									 0);
			}
		}
		ret = ac_build_gather_values(&ctx->ac, results, num_components);
	} else {
		ret = ac_build_buffer_load(&ctx->ac, rsrc, num_components, NULL, offset,
					   NULL, 0, 0, true, true);

		ret = ac_trim_vector(&ctx->ac, ret, num_components);
	}

	return LLVMBuildBitCast(ctx->ac.builder, ret,
	                        get_def_type(ctx, &instr->dest.ssa), "");
}

static void
get_deref_offset(struct ac_nir_context *ctx, nir_deref_instr *instr,
                 bool vs_in, unsigned *vertex_index_out,
                 LLVMValueRef *vertex_index_ref,
                 unsigned *const_out, LLVMValueRef *indir_out)
{
	nir_variable *var = nir_deref_instr_get_variable(instr);
	nir_deref_path path;
	unsigned idx_lvl = 1;

	nir_deref_path_init(&path, instr, NULL);

	if (vertex_index_out != NULL || vertex_index_ref != NULL) {
		if (vertex_index_ref) {
			*vertex_index_ref = get_src(ctx, path.path[idx_lvl]->arr.index);
			if (vertex_index_out)
				*vertex_index_out = 0;
		} else {
			*vertex_index_out = nir_src_as_uint(path.path[idx_lvl]->arr.index);
		}
		++idx_lvl;
	}

	uint32_t const_offset = 0;
	LLVMValueRef offset = NULL;

	if (var->data.compact) {
		assert(instr->deref_type == nir_deref_type_array);
		const_offset = nir_src_as_uint(instr->arr.index);
		goto out;
	}

	for (; path.path[idx_lvl]; ++idx_lvl) {
		const struct glsl_type *parent_type = path.path[idx_lvl - 1]->type;
		if (path.path[idx_lvl]->deref_type == nir_deref_type_struct) {
			unsigned index = path.path[idx_lvl]->strct.index;

			for (unsigned i = 0; i < index; i++) {
				const struct glsl_type *ft = glsl_get_struct_field(parent_type, i);
				const_offset += glsl_count_attribute_slots(ft, vs_in);
			}
		} else if(path.path[idx_lvl]->deref_type == nir_deref_type_array) {
			unsigned size = glsl_count_attribute_slots(path.path[idx_lvl]->type, vs_in);
			if (nir_src_is_const(path.path[idx_lvl]->arr.index)) {
				const_offset += size *
					nir_src_as_uint(path.path[idx_lvl]->arr.index);
			} else {
				LLVMValueRef array_off = LLVMBuildMul(ctx->ac.builder, LLVMConstInt(ctx->ac.i32, size, 0),
								      get_src(ctx, path.path[idx_lvl]->arr.index), "");
				if (offset)
					offset = LLVMBuildAdd(ctx->ac.builder, offset, array_off, "");
				else
					offset = array_off;
			}
		} else if(path.path[idx_lvl]->deref_type == nir_deref_type_cast) {
			/* continue */
		} else
			unreachable("Uhandled deref type in get_deref_instr_offset");
	}

out:
	nir_deref_path_finish(&path);

	if (const_offset && offset)
		offset = LLVMBuildAdd(ctx->ac.builder, offset,
				      LLVMConstInt(ctx->ac.i32, const_offset, 0),
				      "");

	*const_out = const_offset;
	*indir_out = offset;
}

static LLVMValueRef load_tess_varyings(struct ac_nir_context *ctx,
				       nir_intrinsic_instr *instr,
				       bool load_inputs)
{
	LLVMValueRef result;
	LLVMValueRef vertex_index = NULL;
	LLVMValueRef indir_index = NULL;
	unsigned const_index = 0;

	nir_variable *var = nir_deref_instr_get_variable(nir_instr_as_deref(instr->src[0].ssa->parent_instr));

	unsigned location = var->data.location;
	unsigned driver_location = var->data.driver_location;
	const bool is_patch =  var->data.patch;
	const bool is_compact = var->data.compact;

	get_deref_offset(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr),
	                 false, NULL, is_patch ? NULL : &vertex_index,
	                 &const_index, &indir_index);

	LLVMTypeRef dest_type = get_def_type(ctx, &instr->dest.ssa);

	LLVMTypeRef src_component_type;
	if (LLVMGetTypeKind(dest_type) == LLVMVectorTypeKind)
		src_component_type = LLVMGetElementType(dest_type);
	else
		src_component_type = dest_type;

	result = ctx->abi->load_tess_varyings(ctx->abi, src_component_type,
					      vertex_index, indir_index,
					      const_index, location, driver_location,
					      var->data.location_frac,
					      instr->num_components,
					      is_patch, is_compact, load_inputs);
	if (instr->dest.ssa.bit_size == 16) {
		result = ac_to_integer(&ctx->ac, result);
		result = LLVMBuildTrunc(ctx->ac.builder, result, dest_type, "");
	}
	return LLVMBuildBitCast(ctx->ac.builder, result, dest_type, "");
}

static unsigned
type_scalar_size_bytes(const struct glsl_type *type)
{
   assert(glsl_type_is_vector_or_scalar(type) ||
          glsl_type_is_matrix(type));
   return glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
}

static LLVMValueRef visit_load_var(struct ac_nir_context *ctx,
				   nir_intrinsic_instr *instr)
{
	nir_deref_instr *deref = nir_instr_as_deref(instr->src[0].ssa->parent_instr);
	nir_variable *var = nir_deref_instr_get_variable(deref);

	LLVMValueRef values[8];
	int idx = 0;
	int ve = instr->dest.ssa.num_components;
	unsigned comp = 0;
	LLVMValueRef indir_index;
	LLVMValueRef ret;
	unsigned const_index;
	unsigned stride = 4;
	int mode = deref->mode;
	
	if (var) {
		bool vs_in = ctx->stage == MESA_SHADER_VERTEX &&
			var->data.mode == nir_var_shader_in;
		idx = var->data.driver_location;
		comp = var->data.location_frac;
		mode = var->data.mode;

		get_deref_offset(ctx, deref, vs_in, NULL, NULL,
				 &const_index, &indir_index);

		if (var->data.compact) {
			stride = 1;
			const_index += comp;
			comp = 0;
		}
	}

	if (instr->dest.ssa.bit_size == 64 &&
	    (deref->mode == nir_var_shader_in ||
	     deref->mode == nir_var_shader_out ||
	     deref->mode == nir_var_function_temp))
		ve *= 2;

	switch (mode) {
	case nir_var_shader_in:
		if (ctx->stage == MESA_SHADER_TESS_CTRL ||
		    ctx->stage == MESA_SHADER_TESS_EVAL) {
			return load_tess_varyings(ctx, instr, true);
		}

		if (ctx->stage == MESA_SHADER_GEOMETRY) {
			LLVMTypeRef type = LLVMIntTypeInContext(ctx->ac.context, instr->dest.ssa.bit_size);
			LLVMValueRef indir_index;
			unsigned const_index, vertex_index;
			get_deref_offset(ctx, deref, false, &vertex_index, NULL,
			                 &const_index, &indir_index);
			assert(indir_index == NULL);

			return ctx->abi->load_inputs(ctx->abi, var->data.location,
						     var->data.driver_location,
						     var->data.location_frac,
						     instr->num_components, vertex_index, const_index, type);
		}

		for (unsigned chan = comp; chan < ve + comp; chan++) {
			if (indir_index) {
				unsigned count = glsl_count_attribute_slots(
						var->type,
						ctx->stage == MESA_SHADER_VERTEX);
				count -= chan / 4;
				LLVMValueRef tmp_vec = ac_build_gather_values_extended(
						&ctx->ac, ctx->abi->inputs + idx + chan, count,
						stride, false, true);

				values[chan] = LLVMBuildExtractElement(ctx->ac.builder,
								       tmp_vec,
								       indir_index, "");
			} else
				values[chan] = ctx->abi->inputs[idx + chan + const_index * stride];
		}
		break;
	case nir_var_function_temp:
		for (unsigned chan = 0; chan < ve; chan++) {
			if (indir_index) {
				unsigned count = glsl_count_attribute_slots(
					var->type, false);
				count -= chan / 4;
				LLVMValueRef tmp_vec = ac_build_gather_values_extended(
						&ctx->ac, ctx->locals + idx + chan, count,
						stride, true, true);

				values[chan] = LLVMBuildExtractElement(ctx->ac.builder,
								       tmp_vec,
								       indir_index, "");
			} else {
				values[chan] = LLVMBuildLoad(ctx->ac.builder, ctx->locals[idx + chan + const_index * stride], "");
			}
		}
		break;
	case nir_var_mem_shared: {
		LLVMValueRef address = get_src(ctx, instr->src[0]);
		LLVMValueRef val = LLVMBuildLoad(ctx->ac.builder, address, "");
		return LLVMBuildBitCast(ctx->ac.builder, val,
					get_def_type(ctx, &instr->dest.ssa),
					"");
	}
	case nir_var_shader_out:
		if (ctx->stage == MESA_SHADER_TESS_CTRL) {
			return load_tess_varyings(ctx, instr, false);
		}

		if (ctx->stage == MESA_SHADER_FRAGMENT &&
		    var->data.fb_fetch_output &&
		    ctx->abi->emit_fbfetch)
			return ctx->abi->emit_fbfetch(ctx->abi);

		for (unsigned chan = comp; chan < ve + comp; chan++) {
			if (indir_index) {
				unsigned count = glsl_count_attribute_slots(
						var->type, false);
				count -= chan / 4;
				LLVMValueRef tmp_vec = ac_build_gather_values_extended(
						&ctx->ac, ctx->abi->outputs + idx + chan, count,
						stride, true, true);

				values[chan] = LLVMBuildExtractElement(ctx->ac.builder,
								       tmp_vec,
								       indir_index, "");
			} else {
				values[chan] = LLVMBuildLoad(ctx->ac.builder,
						     ctx->abi->outputs[idx + chan + const_index * stride],
						     "");
			}
		}
		break;
	case nir_var_mem_global:  {
		LLVMValueRef address = get_src(ctx, instr->src[0]);
		unsigned explicit_stride = glsl_get_explicit_stride(deref->type);
		unsigned natural_stride = type_scalar_size_bytes(deref->type);
		unsigned stride = explicit_stride ? explicit_stride : natural_stride;

		LLVMTypeRef result_type = get_def_type(ctx, &instr->dest.ssa);
		if (stride != natural_stride) {
			LLVMTypeRef ptr_type =  LLVMPointerType(LLVMGetElementType(result_type),
			                                        LLVMGetPointerAddressSpace(LLVMTypeOf(address)));
			address = LLVMBuildBitCast(ctx->ac.builder, address, ptr_type , "");

			for (unsigned i = 0; i < instr->dest.ssa.num_components; ++i) {
				LLVMValueRef offset = LLVMConstInt(ctx->ac.i32, i * stride / natural_stride, 0);
				values[i] = LLVMBuildLoad(ctx->ac.builder,
				                          ac_build_gep_ptr(&ctx->ac, address, offset), "");
			}
			return ac_build_gather_values(&ctx->ac, values, instr->dest.ssa.num_components);
		} else {
			LLVMTypeRef ptr_type =  LLVMPointerType(result_type,
			                                        LLVMGetPointerAddressSpace(LLVMTypeOf(address)));
			address = LLVMBuildBitCast(ctx->ac.builder, address, ptr_type , "");
			LLVMValueRef val = LLVMBuildLoad(ctx->ac.builder, address, "");
			return val;
		}
	}
	default:
		unreachable("unhandle variable mode");
	}
	ret = ac_build_varying_gather_values(&ctx->ac, values, ve, comp);
	return LLVMBuildBitCast(ctx->ac.builder, ret, get_def_type(ctx, &instr->dest.ssa), "");
}

static void
visit_store_var(struct ac_nir_context *ctx,
		nir_intrinsic_instr *instr)
{
	nir_deref_instr *deref = nir_instr_as_deref(instr->src[0].ssa->parent_instr);
	nir_variable *var = nir_deref_instr_get_variable(deref);

	LLVMValueRef temp_ptr, value;
	int idx = 0;
	unsigned comp = 0;
	LLVMValueRef src = ac_to_float(&ctx->ac, get_src(ctx, instr->src[1]));
	int writemask = instr->const_index[0];
	LLVMValueRef indir_index;
	unsigned const_index;

	if (var) {
		get_deref_offset(ctx, deref, false,
		                 NULL, NULL, &const_index, &indir_index);
		idx = var->data.driver_location;
		comp = var->data.location_frac;

		if (var->data.compact) {
			const_index += comp;
			comp = 0;
		}
	}

	if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src)) == 64 &&
	    (deref->mode == nir_var_shader_out ||
	     deref->mode == nir_var_function_temp)) {

		src = LLVMBuildBitCast(ctx->ac.builder, src,
		                       LLVMVectorType(ctx->ac.f32, ac_get_llvm_num_components(src) * 2),
		                       "");

		writemask = widen_mask(writemask, 2);
	}

	writemask = writemask << comp;

	switch (deref->mode) {
	case nir_var_shader_out:

		if (ctx->stage == MESA_SHADER_TESS_CTRL) {
			LLVMValueRef vertex_index = NULL;
			LLVMValueRef indir_index = NULL;
			unsigned const_index = 0;
			const bool is_patch = var->data.patch;

			get_deref_offset(ctx, deref, false, NULL,
			                 is_patch ? NULL : &vertex_index,
			                 &const_index, &indir_index);

			ctx->abi->store_tcs_outputs(ctx->abi, var,
						    vertex_index, indir_index,
						    const_index, src, writemask);
			return;
		}

		for (unsigned chan = 0; chan < 8; chan++) {
			int stride = 4;
			if (!(writemask & (1 << chan)))
				continue;

			value = ac_llvm_extract_elem(&ctx->ac, src, chan - comp);

			if (var->data.compact)
				stride = 1;
			if (indir_index) {
				unsigned count = glsl_count_attribute_slots(
						var->type, false);
				count -= chan / 4;
				LLVMValueRef tmp_vec = ac_build_gather_values_extended(
						&ctx->ac, ctx->abi->outputs + idx + chan, count,
						stride, true, true);

				tmp_vec = LLVMBuildInsertElement(ctx->ac.builder, tmp_vec,
							         value, indir_index, "");
				build_store_values_extended(&ctx->ac, ctx->abi->outputs + idx + chan,
							    count, stride, tmp_vec);

			} else {
				temp_ptr = ctx->abi->outputs[idx + chan + const_index * stride];

				LLVMBuildStore(ctx->ac.builder, value, temp_ptr);
			}
		}
		break;
	case nir_var_function_temp:
		for (unsigned chan = 0; chan < 8; chan++) {
			if (!(writemask & (1 << chan)))
				continue;

			value = ac_llvm_extract_elem(&ctx->ac, src, chan);
			if (indir_index) {
				unsigned count = glsl_count_attribute_slots(
					var->type, false);
				count -= chan / 4;
				LLVMValueRef tmp_vec = ac_build_gather_values_extended(
					&ctx->ac, ctx->locals + idx + chan, count,
					4, true, true);

				tmp_vec = LLVMBuildInsertElement(ctx->ac.builder, tmp_vec,
								 value, indir_index, "");
				build_store_values_extended(&ctx->ac, ctx->locals + idx + chan,
							    count, 4, tmp_vec);
			} else {
				temp_ptr = ctx->locals[idx + chan + const_index * 4];

				LLVMBuildStore(ctx->ac.builder, value, temp_ptr);
			}
		}
		break;

	case nir_var_mem_global:
	case nir_var_mem_shared: {
		int writemask = instr->const_index[0];
		LLVMValueRef address = get_src(ctx, instr->src[0]);
		LLVMValueRef val = get_src(ctx, instr->src[1]);

		unsigned explicit_stride = glsl_get_explicit_stride(deref->type);
		unsigned natural_stride = type_scalar_size_bytes(deref->type);
		unsigned stride = explicit_stride ? explicit_stride : natural_stride;

		LLVMTypeRef ptr_type =  LLVMPointerType(LLVMTypeOf(val),
							LLVMGetPointerAddressSpace(LLVMTypeOf(address)));
		address = LLVMBuildBitCast(ctx->ac.builder, address, ptr_type , "");

		if (writemask == (1u << ac_get_llvm_num_components(val)) - 1 &&
		    stride == natural_stride) {
			LLVMTypeRef ptr_type =  LLVMPointerType(LLVMTypeOf(val),
			                                        LLVMGetPointerAddressSpace(LLVMTypeOf(address)));
			address = LLVMBuildBitCast(ctx->ac.builder, address, ptr_type , "");

			val = LLVMBuildBitCast(ctx->ac.builder, val,
			                       LLVMGetElementType(LLVMTypeOf(address)), "");
			LLVMBuildStore(ctx->ac.builder, val, address);
		} else {
			LLVMTypeRef ptr_type =  LLVMPointerType(LLVMGetElementType(LLVMTypeOf(val)),
			                                        LLVMGetPointerAddressSpace(LLVMTypeOf(address)));
			address = LLVMBuildBitCast(ctx->ac.builder, address, ptr_type , "");
			for (unsigned chan = 0; chan < 4; chan++) {
				if (!(writemask & (1 << chan)))
					continue;

				LLVMValueRef offset = LLVMConstInt(ctx->ac.i32, chan * stride / natural_stride, 0);

				LLVMValueRef ptr = ac_build_gep_ptr(&ctx->ac, address, offset);
				LLVMValueRef src = ac_llvm_extract_elem(&ctx->ac, val,
									chan);
				src = LLVMBuildBitCast(ctx->ac.builder, src,
				                       LLVMGetElementType(LLVMTypeOf(ptr)), "");
				LLVMBuildStore(ctx->ac.builder, src, ptr);
			}
		}
		break;
	}
	default:
		abort();
		break;
	}
}

static int image_type_to_components_count(enum glsl_sampler_dim dim, bool array)
{
	switch (dim) {
	case GLSL_SAMPLER_DIM_BUF:
		return 1;
	case GLSL_SAMPLER_DIM_1D:
		return array ? 2 : 1;
	case GLSL_SAMPLER_DIM_2D:
		return array ? 3 : 2;
	case GLSL_SAMPLER_DIM_MS:
		return array ? 4 : 3;
	case GLSL_SAMPLER_DIM_3D:
	case GLSL_SAMPLER_DIM_CUBE:
		return 3;
	case GLSL_SAMPLER_DIM_RECT:
	case GLSL_SAMPLER_DIM_SUBPASS:
		return 2;
	case GLSL_SAMPLER_DIM_SUBPASS_MS:
		return 3;
	default:
		break;
	}
	return 0;
}

static LLVMValueRef adjust_sample_index_using_fmask(struct ac_llvm_context *ctx,
						    LLVMValueRef coord_x, LLVMValueRef coord_y,
						    LLVMValueRef coord_z,
						    LLVMValueRef sample_index,
						    LLVMValueRef fmask_desc_ptr)
{
	unsigned sample_chan = coord_z ? 3 : 2;
	LLVMValueRef addr[4] = {coord_x, coord_y, coord_z};
	addr[sample_chan] = sample_index;

	ac_apply_fmask_to_sample(ctx, fmask_desc_ptr, addr, coord_z != NULL);
	return addr[sample_chan];
}

static nir_deref_instr *get_image_deref(const nir_intrinsic_instr *instr)
{
	assert(instr->src[0].is_ssa);
	return nir_instr_as_deref(instr->src[0].ssa->parent_instr);
}

static LLVMValueRef get_image_descriptor(struct ac_nir_context *ctx,
                                         const nir_intrinsic_instr *instr,
                                         enum ac_descriptor_type desc_type,
                                         bool write)
{
	nir_deref_instr *deref_instr =
		instr->src[0].ssa->parent_instr->type == nir_instr_type_deref ?
		nir_instr_as_deref(instr->src[0].ssa->parent_instr) : NULL;

	return get_sampler_desc(ctx, deref_instr, desc_type, &instr->instr, true, write);
}

static void get_image_coords(struct ac_nir_context *ctx,
			     const nir_intrinsic_instr *instr,
			     struct ac_image_args *args,
			     enum glsl_sampler_dim dim,
			     bool is_array)
{
	LLVMValueRef src0 = get_src(ctx, instr->src[1]);
	LLVMValueRef masks[] = {
		LLVMConstInt(ctx->ac.i32, 0, false), LLVMConstInt(ctx->ac.i32, 1, false),
		LLVMConstInt(ctx->ac.i32, 2, false), LLVMConstInt(ctx->ac.i32, 3, false),
	};
	LLVMValueRef sample_index = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[2]), 0);

	int count;
	ASSERTED bool add_frag_pos = (dim == GLSL_SAMPLER_DIM_SUBPASS ||
					  dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
	bool is_ms = (dim == GLSL_SAMPLER_DIM_MS ||
		      dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
	bool gfx9_1d = ctx->ac.chip_class == GFX9 && dim == GLSL_SAMPLER_DIM_1D;
	assert(!add_frag_pos && "Input attachments should be lowered by this point.");
	count = image_type_to_components_count(dim, is_array);

	if (is_ms && (instr->intrinsic == nir_intrinsic_image_deref_load ||
		      instr->intrinsic == nir_intrinsic_bindless_image_load)) {
		LLVMValueRef fmask_load_address[3];

		fmask_load_address[0] = LLVMBuildExtractElement(ctx->ac.builder, src0, masks[0], "");
		fmask_load_address[1] = LLVMBuildExtractElement(ctx->ac.builder, src0, masks[1], "");
		if (is_array)
			fmask_load_address[2] = LLVMBuildExtractElement(ctx->ac.builder, src0, masks[2], "");
		else
			fmask_load_address[2] = NULL;

		sample_index = adjust_sample_index_using_fmask(&ctx->ac,
							       fmask_load_address[0],
							       fmask_load_address[1],
							       fmask_load_address[2],
							       sample_index,
							       get_sampler_desc(ctx, nir_instr_as_deref(instr->src[0].ssa->parent_instr),
										AC_DESC_FMASK, &instr->instr, true, false));
	}
	if (count == 1 && !gfx9_1d) {
		if (instr->src[1].ssa->num_components)
			args->coords[0] = LLVMBuildExtractElement(ctx->ac.builder, src0, masks[0], "");
		else
			args->coords[0] = src0;
	} else {
		int chan;
		if (is_ms)
			count--;
		for (chan = 0; chan < count; ++chan) {
			args->coords[chan] = ac_llvm_extract_elem(&ctx->ac, src0, chan);
		}

		if (gfx9_1d) {
			if (is_array) {
				args->coords[2] = args->coords[1];
				args->coords[1] = ctx->ac.i32_0;
			} else
				args->coords[1] = ctx->ac.i32_0;
			count++;
		}
		if (ctx->ac.chip_class == GFX9 &&
		    dim == GLSL_SAMPLER_DIM_2D &&
		    !is_array) {
			/* The hw can't bind a slice of a 3D image as a 2D
			 * image, because it ignores BASE_ARRAY if the target
			 * is 3D. The workaround is to read BASE_ARRAY and set
			 * it as the 3rd address operand for all 2D images.
			 */
			LLVMValueRef first_layer, const5, mask;

			const5 = LLVMConstInt(ctx->ac.i32, 5, 0);
			mask = LLVMConstInt(ctx->ac.i32, S_008F24_BASE_ARRAY(~0), 0);
			first_layer = LLVMBuildExtractElement(ctx->ac.builder, args->resource, const5, "");
			first_layer = LLVMBuildAnd(ctx->ac.builder, first_layer, mask, "");

			args->coords[count] = first_layer;
			count++;
		}


		if (is_ms) {
			args->coords[count] = sample_index;
			count++;
		}
	}
}

static LLVMValueRef get_image_buffer_descriptor(struct ac_nir_context *ctx,
                                                const nir_intrinsic_instr *instr,
						bool write, bool atomic)
{
	LLVMValueRef rsrc = get_image_descriptor(ctx, instr, AC_DESC_BUFFER, write);
	if (ctx->ac.chip_class == GFX9 && LLVM_VERSION_MAJOR < 9 && atomic) {
		LLVMValueRef elem_count = LLVMBuildExtractElement(ctx->ac.builder, rsrc, LLVMConstInt(ctx->ac.i32, 2, 0), "");
		LLVMValueRef stride = LLVMBuildExtractElement(ctx->ac.builder, rsrc, LLVMConstInt(ctx->ac.i32, 1, 0), "");
		stride = LLVMBuildLShr(ctx->ac.builder, stride, LLVMConstInt(ctx->ac.i32, 16, 0), "");

		LLVMValueRef new_elem_count = LLVMBuildSelect(ctx->ac.builder,
		                                              LLVMBuildICmp(ctx->ac.builder, LLVMIntUGT, elem_count, stride, ""),
		                                              elem_count, stride, "");

		rsrc = LLVMBuildInsertElement(ctx->ac.builder, rsrc, new_elem_count,
		                              LLVMConstInt(ctx->ac.i32, 2, 0), "");
	}
	return rsrc;
}

static LLVMValueRef visit_image_load(struct ac_nir_context *ctx,
				     const nir_intrinsic_instr *instr,
				     bool bindless)
{
	LLVMValueRef res;

	enum glsl_sampler_dim dim;
	enum gl_access_qualifier access;
	bool is_array;
	if (bindless) {
		dim = nir_intrinsic_image_dim(instr);
		access = nir_intrinsic_access(instr);
		is_array = nir_intrinsic_image_array(instr);
	} else {
		const nir_deref_instr *image_deref = get_image_deref(instr);
		const struct glsl_type *type = image_deref->type;
		const nir_variable *var = nir_deref_instr_get_variable(image_deref);
		dim = glsl_get_sampler_dim(type);
		access = var->data.image.access;
		is_array = glsl_sampler_type_is_array(type);
	}

	struct ac_image_args args = {};

	args.cache_policy = get_cache_policy(ctx, access, false, false);

	if (dim == GLSL_SAMPLER_DIM_BUF) {
		unsigned mask = nir_ssa_def_components_read(&instr->dest.ssa);
		unsigned num_channels = util_last_bit(mask);
		LLVMValueRef rsrc, vindex;

		rsrc = get_image_buffer_descriptor(ctx, instr, false, false);
		vindex = LLVMBuildExtractElement(ctx->ac.builder, get_src(ctx, instr->src[1]),
						 ctx->ac.i32_0, "");

		bool can_speculate = access & ACCESS_CAN_REORDER;
		res = ac_build_buffer_load_format(&ctx->ac, rsrc, vindex,
						  ctx->ac.i32_0, num_channels,
						  args.cache_policy,
						  can_speculate);
		res = ac_build_expand_to_vec4(&ctx->ac, res, num_channels);

		res = ac_trim_vector(&ctx->ac, res, instr->dest.ssa.num_components);
		res = ac_to_integer(&ctx->ac, res);
	} else {
		args.opcode = ac_image_load;
		args.resource = get_image_descriptor(ctx, instr, AC_DESC_IMAGE, false);
		get_image_coords(ctx, instr, &args, dim, is_array);
		args.dim = get_ac_image_dim(&ctx->ac, dim, is_array);
		args.dmask = 15;
		args.attributes = AC_FUNC_ATTR_READONLY;

		res = ac_build_image_opcode(&ctx->ac, &args);
	}
	return res;
}

static void visit_image_store(struct ac_nir_context *ctx,
			      nir_intrinsic_instr *instr,
			      bool bindless)
{


	enum glsl_sampler_dim dim;
	enum gl_access_qualifier access;
	bool is_array;
	if (bindless) {
		dim = nir_intrinsic_image_dim(instr);
		access = nir_intrinsic_access(instr);
		is_array = nir_intrinsic_image_array(instr);
	} else {
		const nir_deref_instr *image_deref = get_image_deref(instr);
		const struct glsl_type *type = image_deref->type;
		const nir_variable *var = nir_deref_instr_get_variable(image_deref);
		dim = glsl_get_sampler_dim(type);
		access = var->data.image.access;
		is_array = glsl_sampler_type_is_array(type);
	}

	bool writeonly_memory = access & ACCESS_NON_READABLE;
	struct ac_image_args args = {};

	args.cache_policy = get_cache_policy(ctx, access, true, writeonly_memory);

	if (dim == GLSL_SAMPLER_DIM_BUF) {
		LLVMValueRef rsrc = get_image_buffer_descriptor(ctx, instr, true, false);
		LLVMValueRef src = ac_to_float(&ctx->ac, get_src(ctx, instr->src[3]));
		unsigned src_channels = ac_get_llvm_num_components(src);
		LLVMValueRef vindex;

		if (src_channels == 3)
			src = ac_build_expand_to_vec4(&ctx->ac, src, 3);

		vindex = LLVMBuildExtractElement(ctx->ac.builder,
						 get_src(ctx, instr->src[1]),
						 ctx->ac.i32_0, "");

		ac_build_buffer_store_format(&ctx->ac, rsrc, src, vindex,
					     ctx->ac.i32_0, src_channels,
					     args.cache_policy);
	} else {
		args.opcode = ac_image_store;
		args.data[0] = ac_to_float(&ctx->ac, get_src(ctx, instr->src[3]));
		args.resource = get_image_descriptor(ctx, instr, AC_DESC_IMAGE, true);
		get_image_coords(ctx, instr, &args, dim, is_array);
		args.dim = get_ac_image_dim(&ctx->ac, dim, is_array);
		args.dmask = 15;

		ac_build_image_opcode(&ctx->ac, &args);
	}

}

static LLVMValueRef visit_image_atomic(struct ac_nir_context *ctx,
                                       const nir_intrinsic_instr *instr,
                                       bool bindless)
{
	LLVMValueRef params[7];
	int param_count = 0;

	bool cmpswap = instr->intrinsic == nir_intrinsic_image_deref_atomic_comp_swap ||
		       instr->intrinsic == nir_intrinsic_bindless_image_atomic_comp_swap;
	const char *atomic_name;
	char intrinsic_name[64];
	enum ac_atomic_op atomic_subop;
	ASSERTED int length;

	enum glsl_sampler_dim dim;
	bool is_array;
	if (bindless) {
		if (instr->intrinsic == nir_intrinsic_bindless_image_atomic_imin ||
		    instr->intrinsic == nir_intrinsic_bindless_image_atomic_umin ||
		    instr->intrinsic == nir_intrinsic_bindless_image_atomic_imax ||
		    instr->intrinsic == nir_intrinsic_bindless_image_atomic_umax) {
			const GLenum format = nir_intrinsic_format(instr);
			assert(format == GL_R32UI || format == GL_R32I);
		}
		dim = nir_intrinsic_image_dim(instr);
		is_array = nir_intrinsic_image_array(instr);
	} else {
		const struct glsl_type *type = get_image_deref(instr)->type;
		dim = glsl_get_sampler_dim(type);
		is_array = glsl_sampler_type_is_array(type);
	}

	switch (instr->intrinsic) {
	case nir_intrinsic_bindless_image_atomic_add:
	case nir_intrinsic_image_deref_atomic_add:
		atomic_name = "add";
		atomic_subop = ac_atomic_add;
		break;
	case nir_intrinsic_bindless_image_atomic_imin:
	case nir_intrinsic_image_deref_atomic_imin:
		atomic_name = "smin";
		atomic_subop = ac_atomic_smin;
		break;
	case nir_intrinsic_bindless_image_atomic_umin:
	case nir_intrinsic_image_deref_atomic_umin:
		atomic_name = "umin";
		atomic_subop = ac_atomic_umin;
		break;
	case nir_intrinsic_bindless_image_atomic_imax:
	case nir_intrinsic_image_deref_atomic_imax:
		atomic_name = "smax";
		atomic_subop = ac_atomic_smax;
		break;
	case nir_intrinsic_bindless_image_atomic_umax:
	case nir_intrinsic_image_deref_atomic_umax:
		atomic_name = "umax";
		atomic_subop = ac_atomic_umax;
		break;
	case nir_intrinsic_bindless_image_atomic_and:
	case nir_intrinsic_image_deref_atomic_and:
		atomic_name = "and";
		atomic_subop = ac_atomic_and;
		break;
	case nir_intrinsic_bindless_image_atomic_or:
	case nir_intrinsic_image_deref_atomic_or:
		atomic_name = "or";
		atomic_subop = ac_atomic_or;
		break;
	case nir_intrinsic_bindless_image_atomic_xor:
	case nir_intrinsic_image_deref_atomic_xor:
		atomic_name = "xor";
		atomic_subop = ac_atomic_xor;
		break;
	case nir_intrinsic_bindless_image_atomic_exchange:
	case nir_intrinsic_image_deref_atomic_exchange:
		atomic_name = "swap";
		atomic_subop = ac_atomic_swap;
		break;
	case nir_intrinsic_bindless_image_atomic_comp_swap:
	case nir_intrinsic_image_deref_atomic_comp_swap:
		atomic_name = "cmpswap";
		atomic_subop = 0; /* not used */
		break;
	case nir_intrinsic_bindless_image_atomic_inc_wrap:
	case nir_intrinsic_image_deref_atomic_inc_wrap: {
		atomic_name = "inc";
		atomic_subop = ac_atomic_inc_wrap;
		/* ATOMIC_INC instruction does:
		 *      value = (value + 1) % (data + 1)
		 * but we want:
		 *      value = (value + 1) % data
		 * So replace 'data' by 'data - 1'.
		 */
		ctx->ssa_defs[instr->src[3].ssa->index] =
			LLVMBuildSub(ctx->ac.builder,
				     ctx->ssa_defs[instr->src[3].ssa->index],
				     ctx->ac.i32_1, "");
		break;
	}
	case nir_intrinsic_bindless_image_atomic_dec_wrap:
	case nir_intrinsic_image_deref_atomic_dec_wrap:
		atomic_name = "dec";
		atomic_subop = ac_atomic_dec_wrap;
		break;
	default:
		abort();
	}

	if (cmpswap)
		params[param_count++] = get_src(ctx, instr->src[4]);
	params[param_count++] = get_src(ctx, instr->src[3]);

	if (dim == GLSL_SAMPLER_DIM_BUF) {
		params[param_count++] = get_image_buffer_descriptor(ctx, instr, true, true);
		params[param_count++] = LLVMBuildExtractElement(ctx->ac.builder, get_src(ctx, instr->src[1]),
								ctx->ac.i32_0, ""); /* vindex */
		params[param_count++] = ctx->ac.i32_0; /* voffset */
		if (LLVM_VERSION_MAJOR >= 9) {
			/* XXX: The new raw/struct atomic intrinsics are buggy
			 * with LLVM 8, see r358579.
			 */
			params[param_count++] = ctx->ac.i32_0; /* soffset */
			params[param_count++] = ctx->ac.i32_0;  /* slc */

			length = snprintf(intrinsic_name, sizeof(intrinsic_name),
			                  "llvm.amdgcn.struct.buffer.atomic.%s.i32", atomic_name);
		} else {
			params[param_count++] = ctx->ac.i1false;  /* slc */

			length = snprintf(intrinsic_name, sizeof(intrinsic_name),
			                  "llvm.amdgcn.buffer.atomic.%s", atomic_name);
		}

		assert(length < sizeof(intrinsic_name));
		return ac_build_intrinsic(&ctx->ac, intrinsic_name, ctx->ac.i32,
					  params, param_count, 0);
	} else {
		struct ac_image_args args = {};
		args.opcode = cmpswap ? ac_image_atomic_cmpswap : ac_image_atomic;
		args.atomic = atomic_subop;
		args.data[0] = params[0];
		if (cmpswap)
			args.data[1] = params[1];
		args.resource = get_image_descriptor(ctx, instr, AC_DESC_IMAGE, true);
		get_image_coords(ctx, instr, &args, dim, is_array);
		args.dim = get_ac_image_dim(&ctx->ac, dim, is_array);

		return ac_build_image_opcode(&ctx->ac, &args);
	}
}

static LLVMValueRef visit_image_samples(struct ac_nir_context *ctx,
					const nir_intrinsic_instr *instr,
					bool bindless)
{
	enum glsl_sampler_dim dim;
	bool is_array;
	if (bindless) {
		dim = nir_intrinsic_image_dim(instr);
		is_array = nir_intrinsic_image_array(instr);
	} else {
		const struct glsl_type *type = get_image_deref(instr)->type;
		dim = glsl_get_sampler_dim(type);
		is_array = glsl_sampler_type_is_array(type);
	}

	struct ac_image_args args = { 0 };
	args.dim = get_ac_sampler_dim(&ctx->ac, dim, is_array);
	args.dmask = 0xf;
	args.resource = get_image_descriptor(ctx, instr, AC_DESC_IMAGE, false);
	args.opcode = ac_image_get_resinfo;
	args.lod = ctx->ac.i32_0;
	args.attributes = AC_FUNC_ATTR_READNONE;

	return ac_build_image_opcode(&ctx->ac, &args);
}

static LLVMValueRef visit_image_size(struct ac_nir_context *ctx,
				     const nir_intrinsic_instr *instr,
				     bool bindless)
{
	LLVMValueRef res;

	enum glsl_sampler_dim dim;
	bool is_array;
	if (bindless) {
		dim = nir_intrinsic_image_dim(instr);
		is_array = nir_intrinsic_image_array(instr);
	} else {
		const struct glsl_type *type = get_image_deref(instr)->type;
		dim = glsl_get_sampler_dim(type);
		is_array = glsl_sampler_type_is_array(type);
	}

	if (dim == GLSL_SAMPLER_DIM_BUF)
		return get_buffer_size(ctx, get_image_descriptor(ctx, instr, AC_DESC_BUFFER, false), true);

	struct ac_image_args args = { 0 };

	args.dim = get_ac_image_dim(&ctx->ac, dim, is_array);
	args.dmask = 0xf;
	args.resource = get_image_descriptor(ctx, instr, AC_DESC_IMAGE, false);
	args.opcode = ac_image_get_resinfo;
	args.lod = ctx->ac.i32_0;
	args.attributes = AC_FUNC_ATTR_READNONE;

	res = ac_build_image_opcode(&ctx->ac, &args);

	LLVMValueRef two = LLVMConstInt(ctx->ac.i32, 2, false);

	if (dim == GLSL_SAMPLER_DIM_CUBE && is_array) {
		LLVMValueRef six = LLVMConstInt(ctx->ac.i32, 6, false);
		LLVMValueRef z = LLVMBuildExtractElement(ctx->ac.builder, res, two, "");
		z = LLVMBuildSDiv(ctx->ac.builder, z, six, "");
		res = LLVMBuildInsertElement(ctx->ac.builder, res, z, two, "");
	}
	if (ctx->ac.chip_class == GFX9 && dim == GLSL_SAMPLER_DIM_1D && is_array) {
		LLVMValueRef layers = LLVMBuildExtractElement(ctx->ac.builder, res, two, "");
		res = LLVMBuildInsertElement(ctx->ac.builder, res, layers,
						ctx->ac.i32_1, "");

	}
	return res;
}

static void emit_membar(struct ac_llvm_context *ac,
			const nir_intrinsic_instr *instr)
{
	unsigned wait_flags = 0;

	switch (instr->intrinsic) {
	case nir_intrinsic_memory_barrier:
	case nir_intrinsic_group_memory_barrier:
		wait_flags = AC_WAIT_LGKM | AC_WAIT_VLOAD | AC_WAIT_VSTORE;
		break;
	case nir_intrinsic_memory_barrier_atomic_counter:
	case nir_intrinsic_memory_barrier_buffer:
	case nir_intrinsic_memory_barrier_image:
		wait_flags = AC_WAIT_VLOAD | AC_WAIT_VSTORE;
		break;
	case nir_intrinsic_memory_barrier_shared:
		wait_flags = AC_WAIT_LGKM;
		break;
	default:
		break;
	}

	ac_build_waitcnt(ac, wait_flags);
}

void ac_emit_barrier(struct ac_llvm_context *ac, gl_shader_stage stage)
{
	/* GFX6 only (thanks to a hw bug workaround):
	 * The real barrier instruction isn’t needed, because an entire patch
	 * always fits into a single wave.
	 */
	if (ac->chip_class == GFX6 && stage == MESA_SHADER_TESS_CTRL) {
		ac_build_waitcnt(ac, AC_WAIT_LGKM | AC_WAIT_VLOAD | AC_WAIT_VSTORE);
		return;
	}
	ac_build_s_barrier(ac);
}

static void emit_discard(struct ac_nir_context *ctx,
			 const nir_intrinsic_instr *instr)
{
	LLVMValueRef cond;

	if (instr->intrinsic == nir_intrinsic_discard_if) {
		cond = LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ,
				     get_src(ctx, instr->src[0]),
				     ctx->ac.i32_0, "");
	} else {
		assert(instr->intrinsic == nir_intrinsic_discard);
		cond = ctx->ac.i1false;
	}

	ctx->abi->emit_kill(ctx->abi, cond);
}

static LLVMValueRef
visit_load_local_invocation_index(struct ac_nir_context *ctx)
{
	LLVMValueRef result;
	LLVMValueRef thread_id = ac_get_thread_id(&ctx->ac);
	result = LLVMBuildAnd(ctx->ac.builder, ctx->abi->tg_size,
			      LLVMConstInt(ctx->ac.i32, 0xfc0, false), "");

	return LLVMBuildAdd(ctx->ac.builder, result, thread_id, "");
}

static LLVMValueRef
visit_load_subgroup_id(struct ac_nir_context *ctx)
{
	if (ctx->stage == MESA_SHADER_COMPUTE) {
		LLVMValueRef result;
		result = LLVMBuildAnd(ctx->ac.builder, ctx->abi->tg_size,
				LLVMConstInt(ctx->ac.i32, 0xfc0, false), "");
		return LLVMBuildLShr(ctx->ac.builder, result,  LLVMConstInt(ctx->ac.i32, 6, false), "");
	} else {
		return LLVMConstInt(ctx->ac.i32, 0, false);
	}
}

static LLVMValueRef
visit_load_num_subgroups(struct ac_nir_context *ctx)
{
	if (ctx->stage == MESA_SHADER_COMPUTE) {
		return LLVMBuildAnd(ctx->ac.builder, ctx->abi->tg_size,
		                    LLVMConstInt(ctx->ac.i32, 0x3f, false), "");
	} else {
		return LLVMConstInt(ctx->ac.i32, 1, false);
	}
}

static LLVMValueRef
visit_first_invocation(struct ac_nir_context *ctx)
{
	LLVMValueRef active_set = ac_build_ballot(&ctx->ac, ctx->ac.i32_1);
	const char *intr = ctx->ac.wave_size == 32 ? "llvm.cttz.i32" : "llvm.cttz.i64";

	/* The second argument is whether cttz(0) should be defined, but we do not care. */
	LLVMValueRef args[] = {active_set, ctx->ac.i1false};
	LLVMValueRef result =  ac_build_intrinsic(&ctx->ac, intr,
	                                          ctx->ac.iN_wavemask, args, 2,
	                                          AC_FUNC_ATTR_NOUNWIND |
	                                          AC_FUNC_ATTR_READNONE);

	return LLVMBuildTrunc(ctx->ac.builder, result, ctx->ac.i32, "");
}

static LLVMValueRef
visit_load_shared(struct ac_nir_context *ctx,
		   const nir_intrinsic_instr *instr)
{
	LLVMValueRef values[4], derived_ptr, index, ret;

	LLVMValueRef ptr = get_memory_ptr(ctx, instr->src[0]);

	for (int chan = 0; chan < instr->num_components; chan++) {
		index = LLVMConstInt(ctx->ac.i32, chan, 0);
		derived_ptr = LLVMBuildGEP(ctx->ac.builder, ptr, &index, 1, "");
		values[chan] = LLVMBuildLoad(ctx->ac.builder, derived_ptr, "");
	}

	ret = ac_build_gather_values(&ctx->ac, values, instr->num_components);
	return LLVMBuildBitCast(ctx->ac.builder, ret, get_def_type(ctx, &instr->dest.ssa), "");
}

static void
visit_store_shared(struct ac_nir_context *ctx,
		   const nir_intrinsic_instr *instr)
{
	LLVMValueRef derived_ptr, data,index;
	LLVMBuilderRef builder = ctx->ac.builder;

	LLVMValueRef ptr = get_memory_ptr(ctx, instr->src[1]);
	LLVMValueRef src = get_src(ctx, instr->src[0]);

	int writemask = nir_intrinsic_write_mask(instr);
	for (int chan = 0; chan < 4; chan++) {
		if (!(writemask & (1 << chan))) {
			continue;
		}
		data = ac_llvm_extract_elem(&ctx->ac, src, chan);
		index = LLVMConstInt(ctx->ac.i32, chan, 0);
		derived_ptr = LLVMBuildGEP(builder, ptr, &index, 1, "");
		LLVMBuildStore(builder, data, derived_ptr);
	}
}

static LLVMValueRef visit_var_atomic(struct ac_nir_context *ctx,
				     const nir_intrinsic_instr *instr,
				     LLVMValueRef ptr, int src_idx)
{
	LLVMValueRef result;
	LLVMValueRef src = get_src(ctx, instr->src[src_idx]);

	const char *sync_scope = LLVM_VERSION_MAJOR >= 9 ? "workgroup-one-as" : "workgroup";

	if (instr->intrinsic == nir_intrinsic_shared_atomic_comp_swap ||
	    instr->intrinsic == nir_intrinsic_deref_atomic_comp_swap) {
		LLVMValueRef src1 = get_src(ctx, instr->src[src_idx + 1]);
		result = ac_build_atomic_cmp_xchg(&ctx->ac, ptr, src, src1, sync_scope);
		result = LLVMBuildExtractValue(ctx->ac.builder, result, 0, "");
	} else {
		LLVMAtomicRMWBinOp op;
		switch (instr->intrinsic) {
		case nir_intrinsic_shared_atomic_add:
		case nir_intrinsic_deref_atomic_add:
			op = LLVMAtomicRMWBinOpAdd;
			break;
		case nir_intrinsic_shared_atomic_umin:
		case nir_intrinsic_deref_atomic_umin:
			op = LLVMAtomicRMWBinOpUMin;
			break;
		case nir_intrinsic_shared_atomic_umax:
		case nir_intrinsic_deref_atomic_umax:
			op = LLVMAtomicRMWBinOpUMax;
			break;
		case nir_intrinsic_shared_atomic_imin:
		case nir_intrinsic_deref_atomic_imin:
			op = LLVMAtomicRMWBinOpMin;
			break;
		case nir_intrinsic_shared_atomic_imax:
		case nir_intrinsic_deref_atomic_imax:
			op = LLVMAtomicRMWBinOpMax;
			break;
		case nir_intrinsic_shared_atomic_and:
		case nir_intrinsic_deref_atomic_and:
			op = LLVMAtomicRMWBinOpAnd;
			break;
		case nir_intrinsic_shared_atomic_or:
		case nir_intrinsic_deref_atomic_or:
			op = LLVMAtomicRMWBinOpOr;
			break;
		case nir_intrinsic_shared_atomic_xor:
		case nir_intrinsic_deref_atomic_xor:
			op = LLVMAtomicRMWBinOpXor;
			break;
		case nir_intrinsic_shared_atomic_exchange:
		case nir_intrinsic_deref_atomic_exchange:
			op = LLVMAtomicRMWBinOpXchg;
			break;
		default:
			return NULL;
		}

		result = ac_build_atomic_rmw(&ctx->ac, op, ptr, ac_to_integer(&ctx->ac, src), sync_scope);
	}
	return result;
}

static LLVMValueRef load_sample_pos(struct ac_nir_context *ctx)
{
	LLVMValueRef values[2];
	LLVMValueRef pos[2];

	pos[0] = ac_to_float(&ctx->ac, ctx->abi->frag_pos[0]);
	pos[1] = ac_to_float(&ctx->ac, ctx->abi->frag_pos[1]);

	values[0] = ac_build_fract(&ctx->ac, pos[0], 32);
	values[1] = ac_build_fract(&ctx->ac, pos[1], 32);
	return ac_build_gather_values(&ctx->ac, values, 2);
}

static LLVMValueRef lookup_interp_param(struct ac_nir_context *ctx,
					enum glsl_interp_mode interp, unsigned location)
{
	switch (interp) {
	case INTERP_MODE_FLAT:
	default:
		return NULL;
	case INTERP_MODE_SMOOTH:
	case INTERP_MODE_NONE:
		if (location == INTERP_CENTER)
			return ctx->abi->persp_center;
		else if (location == INTERP_CENTROID)
			return ctx->abi->persp_centroid;
		else if (location == INTERP_SAMPLE)
			return ctx->abi->persp_sample;
		break;
	case INTERP_MODE_NOPERSPECTIVE:
		if (location == INTERP_CENTER)
			return ctx->abi->linear_center;
		else if (location == INTERP_CENTROID)
			return ctx->abi->linear_centroid;
		else if (location == INTERP_SAMPLE)
			return ctx->abi->linear_sample;
		break;
	}
	return NULL;
}

static LLVMValueRef barycentric_center(struct ac_nir_context *ctx,
				       unsigned mode)
{
	LLVMValueRef interp_param = lookup_interp_param(ctx, mode, INTERP_CENTER);
	return LLVMBuildBitCast(ctx->ac.builder, interp_param, ctx->ac.v2i32, "");
}

static LLVMValueRef barycentric_offset(struct ac_nir_context *ctx,
				       unsigned mode,
				       LLVMValueRef offset)
{
	LLVMValueRef interp_param = lookup_interp_param(ctx, mode, INTERP_CENTER);
	LLVMValueRef src_c0 = ac_to_float(&ctx->ac, LLVMBuildExtractElement(ctx->ac.builder, offset, ctx->ac.i32_0, ""));
	LLVMValueRef src_c1 = ac_to_float(&ctx->ac, LLVMBuildExtractElement(ctx->ac.builder, offset, ctx->ac.i32_1, ""));

	LLVMValueRef ij_out[2];
	LLVMValueRef ddxy_out = ac_build_ddxy_interp(&ctx->ac, interp_param);

	/*
	 * take the I then J parameters, and the DDX/Y for it, and
	 * calculate the IJ inputs for the interpolator.
	 * temp1 = ddx * offset/sample.x + I;
	 * interp_param.I = ddy * offset/sample.y + temp1;
	 * temp1 = ddx * offset/sample.x + J;
	 * interp_param.J = ddy * offset/sample.y + temp1;
	 */
	for (unsigned i = 0; i < 2; i++) {
		LLVMValueRef ix_ll = LLVMConstInt(ctx->ac.i32, i, false);
		LLVMValueRef iy_ll = LLVMConstInt(ctx->ac.i32, i + 2, false);
		LLVMValueRef ddx_el = LLVMBuildExtractElement(ctx->ac.builder,
							      ddxy_out, ix_ll, "");
		LLVMValueRef ddy_el = LLVMBuildExtractElement(ctx->ac.builder,
							      ddxy_out, iy_ll, "");
		LLVMValueRef interp_el = LLVMBuildExtractElement(ctx->ac.builder,
								 interp_param, ix_ll, "");
		LLVMValueRef temp1, temp2;

		interp_el = LLVMBuildBitCast(ctx->ac.builder, interp_el,
					     ctx->ac.f32, "");

		temp1 = ac_build_fmad(&ctx->ac, ddx_el, src_c0, interp_el);
		temp2 = ac_build_fmad(&ctx->ac, ddy_el, src_c1, temp1);

		ij_out[i] = LLVMBuildBitCast(ctx->ac.builder,
					     temp2, ctx->ac.i32, "");
	}
	interp_param = ac_build_gather_values(&ctx->ac, ij_out, 2);
	return LLVMBuildBitCast(ctx->ac.builder, interp_param, ctx->ac.v2i32, "");
}

static LLVMValueRef barycentric_centroid(struct ac_nir_context *ctx,
					 unsigned mode)
{
	LLVMValueRef interp_param = lookup_interp_param(ctx, mode, INTERP_CENTROID);
	return LLVMBuildBitCast(ctx->ac.builder, interp_param, ctx->ac.v2i32, "");
}

static LLVMValueRef barycentric_at_sample(struct ac_nir_context *ctx,
					  unsigned mode,
					  LLVMValueRef sample_id)
{
	if (ctx->abi->interp_at_sample_force_center)
		return barycentric_center(ctx, mode);

	LLVMValueRef halfval = LLVMConstReal(ctx->ac.f32, 0.5f);

	/* fetch sample ID */
	LLVMValueRef sample_pos = ctx->abi->load_sample_position(ctx->abi, sample_id);

	LLVMValueRef src_c0 = LLVMBuildExtractElement(ctx->ac.builder, sample_pos, ctx->ac.i32_0, "");
	src_c0 = LLVMBuildFSub(ctx->ac.builder, src_c0, halfval, "");
	LLVMValueRef src_c1 = LLVMBuildExtractElement(ctx->ac.builder, sample_pos, ctx->ac.i32_1, "");
	src_c1 = LLVMBuildFSub(ctx->ac.builder, src_c1, halfval, "");
	LLVMValueRef coords[] = { src_c0, src_c1 };
	LLVMValueRef offset = ac_build_gather_values(&ctx->ac, coords, 2);

	return barycentric_offset(ctx, mode, offset);
}


static LLVMValueRef barycentric_sample(struct ac_nir_context *ctx,
				       unsigned mode)
{
	LLVMValueRef interp_param = lookup_interp_param(ctx, mode, INTERP_SAMPLE);
	return LLVMBuildBitCast(ctx->ac.builder, interp_param, ctx->ac.v2i32, "");
}

static LLVMValueRef load_interpolated_input(struct ac_nir_context *ctx,
					    LLVMValueRef interp_param,
					    unsigned index, unsigned comp_start,
					    unsigned num_components,
					    unsigned bitsize)
{
	LLVMValueRef attr_number = LLVMConstInt(ctx->ac.i32, index, false);

	interp_param = LLVMBuildBitCast(ctx->ac.builder,
				interp_param, ctx->ac.v2f32, "");
	LLVMValueRef i = LLVMBuildExtractElement(
		ctx->ac.builder, interp_param, ctx->ac.i32_0, "");
	LLVMValueRef j = LLVMBuildExtractElement(
		ctx->ac.builder, interp_param, ctx->ac.i32_1, "");

	LLVMValueRef values[4];
	assert(bitsize == 16 || bitsize == 32);
	for (unsigned comp = 0; comp < num_components; comp++) {
		LLVMValueRef llvm_chan = LLVMConstInt(ctx->ac.i32, comp_start + comp, false);
		if (bitsize == 16) {
			values[comp] = ac_build_fs_interp_f16(&ctx->ac, llvm_chan, attr_number,
							      ctx->abi->prim_mask, i, j);
		} else {
			values[comp] = ac_build_fs_interp(&ctx->ac, llvm_chan, attr_number,
							  ctx->abi->prim_mask, i, j);
		}
	}

	return ac_to_integer(&ctx->ac, ac_build_gather_values(&ctx->ac, values, num_components));
}

static LLVMValueRef load_flat_input(struct ac_nir_context *ctx,
				    unsigned index, unsigned comp_start,
				    unsigned num_components,
				    unsigned bit_size)
{
	LLVMValueRef attr_number = LLVMConstInt(ctx->ac.i32, index, false);

	LLVMValueRef values[8];

	/* Each component of a 64-bit value takes up two GL-level channels. */
	unsigned channels =
		bit_size == 64 ? num_components * 2 : num_components;

	for (unsigned chan = 0; chan < channels; chan++) {
		if (comp_start + chan > 4)
			attr_number = LLVMConstInt(ctx->ac.i32, index + 1, false);
		LLVMValueRef llvm_chan = LLVMConstInt(ctx->ac.i32, (comp_start + chan) % 4, false);
		values[chan] = ac_build_fs_interp_mov(&ctx->ac,
						      LLVMConstInt(ctx->ac.i32, 2, false),
						      llvm_chan,
						      attr_number,
						      ctx->abi->prim_mask);
		values[chan] = LLVMBuildBitCast(ctx->ac.builder, values[chan], ctx->ac.i32, "");
		values[chan] = LLVMBuildTruncOrBitCast(ctx->ac.builder, values[chan],
						       bit_size == 16 ? ctx->ac.i16 : ctx->ac.i32, "");
	}

	LLVMValueRef result = ac_build_gather_values(&ctx->ac, values, channels);
	if (bit_size == 64) {
		LLVMTypeRef type = num_components == 1 ? ctx->ac.i64 :
			LLVMVectorType(ctx->ac.i64, num_components);
		result = LLVMBuildBitCast(ctx->ac.builder, result, type, "");
	}
	return result;
}

static void visit_intrinsic(struct ac_nir_context *ctx,
                            nir_intrinsic_instr *instr)
{
	LLVMValueRef result = NULL;

	switch (instr->intrinsic) {
	case nir_intrinsic_ballot:
		result = ac_build_ballot(&ctx->ac, get_src(ctx, instr->src[0]));
		if (ctx->ac.ballot_mask_bits > ctx->ac.wave_size)
			result = LLVMBuildZExt(ctx->ac.builder, result, ctx->ac.iN_ballotmask, "");
		break;
	case nir_intrinsic_read_invocation:
		result = ac_build_readlane(&ctx->ac, get_src(ctx, instr->src[0]),
				get_src(ctx, instr->src[1]));
		break;
	case nir_intrinsic_read_first_invocation:
		result = ac_build_readlane(&ctx->ac, get_src(ctx, instr->src[0]), NULL);
		break;
	case nir_intrinsic_load_subgroup_invocation:
		result = ac_get_thread_id(&ctx->ac);
		break;
	case nir_intrinsic_load_work_group_id: {
		LLVMValueRef values[3];

		for (int i = 0; i < 3; i++) {
			values[i] = ctx->abi->workgroup_ids[i] ?
				    ctx->abi->workgroup_ids[i] : ctx->ac.i32_0;
		}

		result = ac_build_gather_values(&ctx->ac, values, 3);
		break;
	}
	case nir_intrinsic_load_base_vertex:
	case nir_intrinsic_load_first_vertex:
		result = ctx->abi->load_base_vertex(ctx->abi);
		break;
	case nir_intrinsic_load_local_group_size:
		result = ctx->abi->load_local_group_size(ctx->abi);
		break;
	case nir_intrinsic_load_vertex_id:
		result = LLVMBuildAdd(ctx->ac.builder, ctx->abi->vertex_id,
				      ctx->abi->base_vertex, "");
		break;
	case nir_intrinsic_load_vertex_id_zero_base: {
		result = ctx->abi->vertex_id;
		break;
	}
	case nir_intrinsic_load_local_invocation_id: {
		result = ctx->abi->local_invocation_ids;
		break;
	}
	case nir_intrinsic_load_base_instance:
		result = ctx->abi->start_instance;
		break;
	case nir_intrinsic_load_draw_id:
		result = ctx->abi->draw_id;
		break;
	case nir_intrinsic_load_view_index:
		result = ctx->abi->view_index;
		break;
	case nir_intrinsic_load_invocation_id:
		if (ctx->stage == MESA_SHADER_TESS_CTRL) {
			result = ac_unpack_param(&ctx->ac, ctx->abi->tcs_rel_ids, 8, 5);
		} else {
			if (ctx->ac.chip_class >= GFX10) {
				result = LLVMBuildAnd(ctx->ac.builder,
						      ctx->abi->gs_invocation_id,
						      LLVMConstInt(ctx->ac.i32, 127, 0), "");
			} else {
				result = ctx->abi->gs_invocation_id;
			}
		}
		break;
	case nir_intrinsic_load_primitive_id:
		if (ctx->stage == MESA_SHADER_GEOMETRY) {
			result = ctx->abi->gs_prim_id;
		} else if (ctx->stage == MESA_SHADER_TESS_CTRL) {
			result = ctx->abi->tcs_patch_id;
		} else if (ctx->stage == MESA_SHADER_TESS_EVAL) {
			result = ctx->abi->tes_patch_id;
		} else
			fprintf(stderr, "Unknown primitive id intrinsic: %d", ctx->stage);
		break;
	case nir_intrinsic_load_sample_id:
		result = ac_unpack_param(&ctx->ac, ctx->abi->ancillary, 8, 4);
		break;
	case nir_intrinsic_load_sample_pos:
		result = load_sample_pos(ctx);
		break;
	case nir_intrinsic_load_sample_mask_in:
		result = ctx->abi->load_sample_mask_in(ctx->abi);
		break;
	case nir_intrinsic_load_frag_coord: {
		LLVMValueRef values[4] = {
			ctx->abi->frag_pos[0],
			ctx->abi->frag_pos[1],
			ctx->abi->frag_pos[2],
			ac_build_fdiv(&ctx->ac, ctx->ac.f32_1, ctx->abi->frag_pos[3])
		};
		result = ac_to_integer(&ctx->ac,
		                       ac_build_gather_values(&ctx->ac, values, 4));
		break;
	}
	case nir_intrinsic_load_layer_id:
		result = ctx->abi->inputs[ac_llvm_reg_index_soa(VARYING_SLOT_LAYER, 0)];
		break;
	case nir_intrinsic_load_front_face:
		result = ctx->abi->front_face;
		break;
	case nir_intrinsic_load_helper_invocation:
		result = ac_build_load_helper_invocation(&ctx->ac);
		break;
	case nir_intrinsic_load_color0:
		result = ctx->abi->color0;
		break;
	case nir_intrinsic_load_color1:
		result = ctx->abi->color1;
		break;
	case nir_intrinsic_load_user_data_amd:
		assert(LLVMTypeOf(ctx->abi->user_data) == ctx->ac.v4i32);
		result = ctx->abi->user_data;
		break;
	case nir_intrinsic_load_instance_id:
		result = ctx->abi->instance_id;
		break;
	case nir_intrinsic_load_num_work_groups:
		result = ctx->abi->num_work_groups;
		break;
	case nir_intrinsic_load_local_invocation_index:
		result = visit_load_local_invocation_index(ctx);
		break;
	case nir_intrinsic_load_subgroup_id:
		result = visit_load_subgroup_id(ctx);
		break;
	case nir_intrinsic_load_num_subgroups:
		result = visit_load_num_subgroups(ctx);
		break;
	case nir_intrinsic_first_invocation:
		result = visit_first_invocation(ctx);
		break;
	case nir_intrinsic_load_push_constant:
		result = visit_load_push_constant(ctx, instr);
		break;
	case nir_intrinsic_vulkan_resource_index: {
		LLVMValueRef index = get_src(ctx, instr->src[0]);
		unsigned desc_set = nir_intrinsic_desc_set(instr);
		unsigned binding = nir_intrinsic_binding(instr);

		result = ctx->abi->load_resource(ctx->abi, index, desc_set,
						 binding);
		break;
	}
	case nir_intrinsic_vulkan_resource_reindex:
		result = visit_vulkan_resource_reindex(ctx, instr);
		break;
	case nir_intrinsic_store_ssbo:
		visit_store_ssbo(ctx, instr);
		break;
	case nir_intrinsic_load_ssbo:
		result = visit_load_buffer(ctx, instr);
		break;
	case nir_intrinsic_ssbo_atomic_add:
	case nir_intrinsic_ssbo_atomic_imin:
	case nir_intrinsic_ssbo_atomic_umin:
	case nir_intrinsic_ssbo_atomic_imax:
	case nir_intrinsic_ssbo_atomic_umax:
	case nir_intrinsic_ssbo_atomic_and:
	case nir_intrinsic_ssbo_atomic_or:
	case nir_intrinsic_ssbo_atomic_xor:
	case nir_intrinsic_ssbo_atomic_exchange:
	case nir_intrinsic_ssbo_atomic_comp_swap:
		result = visit_atomic_ssbo(ctx, instr);
		break;
	case nir_intrinsic_load_ubo:
		result = visit_load_ubo_buffer(ctx, instr);
		break;
	case nir_intrinsic_get_buffer_size:
		result = visit_get_buffer_size(ctx, instr);
		break;
	case nir_intrinsic_load_deref:
		result = visit_load_var(ctx, instr);
		break;
	case nir_intrinsic_store_deref:
		visit_store_var(ctx, instr);
		break;
	case nir_intrinsic_load_shared:
		result = visit_load_shared(ctx, instr);
		break;
	case nir_intrinsic_store_shared:
		visit_store_shared(ctx, instr);
		break;
	case nir_intrinsic_bindless_image_samples:
		result = visit_image_samples(ctx, instr, true);
		break;
	case nir_intrinsic_image_deref_samples:
		result = visit_image_samples(ctx, instr, false);
		break;
	case nir_intrinsic_bindless_image_load:
		result = visit_image_load(ctx, instr, true);
		break;
	case nir_intrinsic_image_deref_load:
		result = visit_image_load(ctx, instr, false);
		break;
	case nir_intrinsic_bindless_image_store:
		visit_image_store(ctx, instr, true);
		break;
	case nir_intrinsic_image_deref_store:
		visit_image_store(ctx, instr, false);
		break;
	case nir_intrinsic_bindless_image_atomic_add:
	case nir_intrinsic_bindless_image_atomic_imin:
	case nir_intrinsic_bindless_image_atomic_umin:
	case nir_intrinsic_bindless_image_atomic_imax:
	case nir_intrinsic_bindless_image_atomic_umax:
	case nir_intrinsic_bindless_image_atomic_and:
	case nir_intrinsic_bindless_image_atomic_or:
	case nir_intrinsic_bindless_image_atomic_xor:
	case nir_intrinsic_bindless_image_atomic_exchange:
	case nir_intrinsic_bindless_image_atomic_comp_swap:
	case nir_intrinsic_bindless_image_atomic_inc_wrap:
	case nir_intrinsic_bindless_image_atomic_dec_wrap:
		result = visit_image_atomic(ctx, instr, true);
		break;
	case nir_intrinsic_image_deref_atomic_add:
	case nir_intrinsic_image_deref_atomic_imin:
	case nir_intrinsic_image_deref_atomic_umin:
	case nir_intrinsic_image_deref_atomic_imax:
	case nir_intrinsic_image_deref_atomic_umax:
	case nir_intrinsic_image_deref_atomic_and:
	case nir_intrinsic_image_deref_atomic_or:
	case nir_intrinsic_image_deref_atomic_xor:
	case nir_intrinsic_image_deref_atomic_exchange:
	case nir_intrinsic_image_deref_atomic_comp_swap:
	case nir_intrinsic_image_deref_atomic_inc_wrap:
	case nir_intrinsic_image_deref_atomic_dec_wrap:
		result = visit_image_atomic(ctx, instr, false);
		break;
	case nir_intrinsic_bindless_image_size:
		result = visit_image_size(ctx, instr, true);
		break;
	case nir_intrinsic_image_deref_size:
		result = visit_image_size(ctx, instr, false);
		break;
	case nir_intrinsic_shader_clock:
		result = ac_build_shader_clock(&ctx->ac);
		break;
	case nir_intrinsic_discard:
	case nir_intrinsic_discard_if:
		emit_discard(ctx, instr);
		break;
	case nir_intrinsic_memory_barrier:
	case nir_intrinsic_group_memory_barrier:
	case nir_intrinsic_memory_barrier_atomic_counter:
	case nir_intrinsic_memory_barrier_buffer:
	case nir_intrinsic_memory_barrier_image:
	case nir_intrinsic_memory_barrier_shared:
		emit_membar(&ctx->ac, instr);
		break;
	case nir_intrinsic_barrier:
		ac_emit_barrier(&ctx->ac, ctx->stage);
		break;
	case nir_intrinsic_shared_atomic_add:
	case nir_intrinsic_shared_atomic_imin:
	case nir_intrinsic_shared_atomic_umin:
	case nir_intrinsic_shared_atomic_imax:
	case nir_intrinsic_shared_atomic_umax:
	case nir_intrinsic_shared_atomic_and:
	case nir_intrinsic_shared_atomic_or:
	case nir_intrinsic_shared_atomic_xor:
	case nir_intrinsic_shared_atomic_exchange:
	case nir_intrinsic_shared_atomic_comp_swap: {
		LLVMValueRef ptr = get_memory_ptr(ctx, instr->src[0]);
		result = visit_var_atomic(ctx, instr, ptr, 1);
		break;
	}
	case nir_intrinsic_deref_atomic_add:
	case nir_intrinsic_deref_atomic_imin:
	case nir_intrinsic_deref_atomic_umin:
	case nir_intrinsic_deref_atomic_imax:
	case nir_intrinsic_deref_atomic_umax:
	case nir_intrinsic_deref_atomic_and:
	case nir_intrinsic_deref_atomic_or:
	case nir_intrinsic_deref_atomic_xor:
	case nir_intrinsic_deref_atomic_exchange:
	case nir_intrinsic_deref_atomic_comp_swap: {
		LLVMValueRef ptr = get_src(ctx, instr->src[0]);
		result = visit_var_atomic(ctx, instr, ptr, 1);
		break;
	}
	case nir_intrinsic_load_barycentric_pixel:
		result = barycentric_center(ctx, nir_intrinsic_interp_mode(instr));
		break;
	case nir_intrinsic_load_barycentric_centroid:
		result = barycentric_centroid(ctx, nir_intrinsic_interp_mode(instr));
		break;
	case nir_intrinsic_load_barycentric_sample:
		result = barycentric_sample(ctx, nir_intrinsic_interp_mode(instr));
		break;
	case nir_intrinsic_load_barycentric_at_offset: {
		LLVMValueRef offset = ac_to_float(&ctx->ac, get_src(ctx, instr->src[0]));
		result = barycentric_offset(ctx, nir_intrinsic_interp_mode(instr), offset);
		break;
	}
	case nir_intrinsic_load_barycentric_at_sample: {
		LLVMValueRef sample_id = get_src(ctx, instr->src[0]);
		result = barycentric_at_sample(ctx, nir_intrinsic_interp_mode(instr), sample_id);
		break;
	}
	case nir_intrinsic_load_interpolated_input: {
		/* We assume any indirect loads have been lowered away */
		ASSERTED nir_const_value *offset = nir_src_as_const_value(instr->src[1]);
		assert(offset);
		assert(offset[0].i32 == 0);

		LLVMValueRef interp_param = get_src(ctx, instr->src[0]);
		unsigned index = nir_intrinsic_base(instr);
		unsigned component = nir_intrinsic_component(instr);
		result = load_interpolated_input(ctx, interp_param, index,
						 component,
						 instr->dest.ssa.num_components,
						 instr->dest.ssa.bit_size);
		break;
	}
	case nir_intrinsic_load_input: {
		/* We only lower inputs for fragment shaders ATM */
		ASSERTED nir_const_value *offset = nir_src_as_const_value(instr->src[0]);
		assert(offset);
		assert(offset[0].i32 == 0);

		unsigned index = nir_intrinsic_base(instr);
		unsigned component = nir_intrinsic_component(instr);
		result = load_flat_input(ctx, index, component,
					 instr->dest.ssa.num_components,
					 instr->dest.ssa.bit_size);
		break;
	}
	case nir_intrinsic_emit_vertex:
		ctx->abi->emit_vertex(ctx->abi, nir_intrinsic_stream_id(instr), ctx->abi->outputs);
		break;
	case nir_intrinsic_end_primitive:
		ctx->abi->emit_primitive(ctx->abi, nir_intrinsic_stream_id(instr));
		break;
	case nir_intrinsic_load_tess_coord:
		result = ctx->abi->load_tess_coord(ctx->abi);
		break;
	case nir_intrinsic_load_tess_level_outer:
		result = ctx->abi->load_tess_level(ctx->abi, VARYING_SLOT_TESS_LEVEL_OUTER, false);
		break;
	case nir_intrinsic_load_tess_level_inner:
		result = ctx->abi->load_tess_level(ctx->abi, VARYING_SLOT_TESS_LEVEL_INNER, false);
		break;
	case nir_intrinsic_load_tess_level_outer_default:
		result = ctx->abi->load_tess_level(ctx->abi, VARYING_SLOT_TESS_LEVEL_OUTER, true);
		break;
	case nir_intrinsic_load_tess_level_inner_default:
		result = ctx->abi->load_tess_level(ctx->abi, VARYING_SLOT_TESS_LEVEL_INNER, true);
		break;
	case nir_intrinsic_load_patch_vertices_in:
		result = ctx->abi->load_patch_vertices_in(ctx->abi);
		break;
	case nir_intrinsic_vote_all: {
		LLVMValueRef tmp = ac_build_vote_all(&ctx->ac, get_src(ctx, instr->src[0]));
		result = LLVMBuildSExt(ctx->ac.builder, tmp, ctx->ac.i32, "");
		break;
	}
	case nir_intrinsic_vote_any: {
		LLVMValueRef tmp = ac_build_vote_any(&ctx->ac, get_src(ctx, instr->src[0]));
		result = LLVMBuildSExt(ctx->ac.builder, tmp, ctx->ac.i32, "");
		break;
	}
	case nir_intrinsic_shuffle:
		result = ac_build_shuffle(&ctx->ac, get_src(ctx, instr->src[0]),
				get_src(ctx, instr->src[1]));
		break;
	case nir_intrinsic_reduce:
		result = ac_build_reduce(&ctx->ac,
				get_src(ctx, instr->src[0]),
				instr->const_index[0],
				instr->const_index[1]);
		break;
	case nir_intrinsic_inclusive_scan:
		result = ac_build_inclusive_scan(&ctx->ac,
				get_src(ctx, instr->src[0]),
				instr->const_index[0]);
		break;
	case nir_intrinsic_exclusive_scan:
		result = ac_build_exclusive_scan(&ctx->ac,
				get_src(ctx, instr->src[0]),
				instr->const_index[0]);
		break;
	case nir_intrinsic_quad_broadcast: {
		unsigned lane = nir_src_as_uint(instr->src[1]);
		result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]),
				lane, lane, lane, lane);
		break;
	}
	case nir_intrinsic_quad_swap_horizontal:
		result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 1, 0, 3 ,2);
		break;
	case nir_intrinsic_quad_swap_vertical:
		result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 2, 3, 0 ,1);
		break;
	case nir_intrinsic_quad_swap_diagonal:
		result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 3, 2, 1 ,0);
		break;
	case nir_intrinsic_quad_swizzle_amd: {
		uint32_t mask = nir_intrinsic_swizzle_mask(instr);
		result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]),
					       mask & 0x3, (mask >> 2) & 0x3,
					       (mask >> 4) & 0x3, (mask >> 6) & 0x3);
		break;
	}
	case nir_intrinsic_masked_swizzle_amd: {
		uint32_t mask = nir_intrinsic_swizzle_mask(instr);
		result = ac_build_ds_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), mask);
		break;
	}
	case nir_intrinsic_write_invocation_amd:
		result = ac_build_writelane(&ctx->ac, get_src(ctx, instr->src[0]),
					    get_src(ctx, instr->src[1]),
					    get_src(ctx, instr->src[2]));
		break;
	case nir_intrinsic_mbcnt_amd:
		result = ac_build_mbcnt(&ctx->ac, get_src(ctx, instr->src[0]));
		break;
	case nir_intrinsic_load_scratch: {
		LLVMValueRef offset = get_src(ctx, instr->src[0]);
		LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->scratch,
						 offset);
		LLVMTypeRef comp_type =
			LLVMIntTypeInContext(ctx->ac.context, instr->dest.ssa.bit_size);
		LLVMTypeRef vec_type =
			instr->dest.ssa.num_components == 1 ? comp_type :
			LLVMVectorType(comp_type, instr->dest.ssa.num_components);
		unsigned addr_space = LLVMGetPointerAddressSpace(LLVMTypeOf(ptr));
		ptr = LLVMBuildBitCast(ctx->ac.builder, ptr,
				       LLVMPointerType(vec_type, addr_space), "");
		result = LLVMBuildLoad(ctx->ac.builder, ptr, "");
		break;
	}
	case nir_intrinsic_store_scratch: {
		LLVMValueRef offset = get_src(ctx, instr->src[1]);
		LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->scratch,
						 offset);
		LLVMTypeRef comp_type =
			LLVMIntTypeInContext(ctx->ac.context, instr->src[0].ssa->bit_size);
		unsigned addr_space = LLVMGetPointerAddressSpace(LLVMTypeOf(ptr));
		ptr = LLVMBuildBitCast(ctx->ac.builder, ptr,
				       LLVMPointerType(comp_type, addr_space), "");
		LLVMValueRef src = get_src(ctx, instr->src[0]);
		unsigned wrmask = nir_intrinsic_write_mask(instr);
		while (wrmask) {
			int start, count;
			u_bit_scan_consecutive_range(&wrmask, &start, &count);
			
			LLVMValueRef offset = LLVMConstInt(ctx->ac.i32, start, false);
			LLVMValueRef offset_ptr = LLVMBuildGEP(ctx->ac.builder, ptr, &offset, 1, "");
			LLVMTypeRef vec_type =
				count == 1 ? comp_type : LLVMVectorType(comp_type, count);
			offset_ptr = LLVMBuildBitCast(ctx->ac.builder,
						      offset_ptr,
						      LLVMPointerType(vec_type, addr_space),
						      "");
			LLVMValueRef offset_src =
				ac_extract_components(&ctx->ac, src, start, count);
			LLVMBuildStore(ctx->ac.builder, offset_src, offset_ptr);
		}
		break;
	}
	case nir_intrinsic_load_constant: {
		LLVMValueRef offset = get_src(ctx, instr->src[0]);
		LLVMValueRef base = LLVMConstInt(ctx->ac.i32,
						 nir_intrinsic_base(instr),
						 false);
		offset = LLVMBuildAdd(ctx->ac.builder, offset, base, "");
		LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->constant_data,
						 offset);
		LLVMTypeRef comp_type =
			LLVMIntTypeInContext(ctx->ac.context, instr->dest.ssa.bit_size);
		LLVMTypeRef vec_type =
			instr->dest.ssa.num_components == 1 ? comp_type :
			LLVMVectorType(comp_type, instr->dest.ssa.num_components);
		unsigned addr_space = LLVMGetPointerAddressSpace(LLVMTypeOf(ptr));
		ptr = LLVMBuildBitCast(ctx->ac.builder, ptr,
				       LLVMPointerType(vec_type, addr_space), "");
		result = LLVMBuildLoad(ctx->ac.builder, ptr, "");
		break;
	}
	default:
		fprintf(stderr, "Unknown intrinsic: ");
		nir_print_instr(&instr->instr, stderr);
		fprintf(stderr, "\n");
		break;
	}
	if (result) {
		ctx->ssa_defs[instr->dest.ssa.index] = result;
	}
}

static LLVMValueRef get_bindless_index_from_uniform(struct ac_nir_context *ctx,
						    unsigned base_index,
						    unsigned constant_index,
						    LLVMValueRef dynamic_index)
{
	LLVMValueRef offset = LLVMConstInt(ctx->ac.i32, base_index * 4, 0);
	LLVMValueRef index = LLVMBuildAdd(ctx->ac.builder, dynamic_index,
					  LLVMConstInt(ctx->ac.i32, constant_index, 0), "");

	/* Bindless uniforms are 64bit so multiple index by 8 */
	index = LLVMBuildMul(ctx->ac.builder, index, LLVMConstInt(ctx->ac.i32, 8, 0), "");
	offset = LLVMBuildAdd(ctx->ac.builder, offset, index, "");

	LLVMValueRef ubo_index = ctx->abi->load_ubo(ctx->abi, ctx->ac.i32_0);

	LLVMValueRef ret = ac_build_buffer_load(&ctx->ac, ubo_index, 1, NULL, offset,
						NULL, 0, 0, true, true);

	return LLVMBuildBitCast(ctx->ac.builder, ret, ctx->ac.i32, "");
}

static LLVMValueRef get_sampler_desc(struct ac_nir_context *ctx,
				     nir_deref_instr *deref_instr,
				     enum ac_descriptor_type desc_type,
				     const nir_instr *instr,
				     bool image, bool write)
{
	LLVMValueRef index = NULL;
	unsigned constant_index = 0;
	unsigned descriptor_set;
	unsigned base_index;
	bool bindless = false;

	if (!deref_instr) {
		descriptor_set = 0;
		if (image) {
			nir_intrinsic_instr *img_instr = nir_instr_as_intrinsic(instr);
			base_index = 0;
			bindless = true;
			index = get_src(ctx, img_instr->src[0]);
		} else {
			nir_tex_instr *tex_instr = nir_instr_as_tex(instr);
			int sampSrcIdx = nir_tex_instr_src_index(tex_instr,
								 nir_tex_src_sampler_handle);
			if (sampSrcIdx != -1) {
				base_index = 0;
				bindless = true;
				index = get_src(ctx, tex_instr->src[sampSrcIdx].src);
			} else {
				assert(tex_instr && !image);
				base_index = tex_instr->sampler_index;
			}
		}
	} else {
		while(deref_instr->deref_type != nir_deref_type_var) {
			if (deref_instr->deref_type == nir_deref_type_array) {
				unsigned array_size = glsl_get_aoa_size(deref_instr->type);
				if (!array_size)
					array_size = 1;

				if (nir_src_is_const(deref_instr->arr.index)) {
					constant_index += array_size * nir_src_as_uint(deref_instr->arr.index);
				} else {
					LLVMValueRef indirect = get_src(ctx, deref_instr->arr.index);

					indirect = LLVMBuildMul(ctx->ac.builder, indirect,
						LLVMConstInt(ctx->ac.i32, array_size, false), "");

					if (!index)
						index = indirect;
					else
						index = LLVMBuildAdd(ctx->ac.builder, index, indirect, "");
				}

				deref_instr = nir_src_as_deref(deref_instr->parent);
			} else if (deref_instr->deref_type == nir_deref_type_struct) {
				unsigned sidx = deref_instr->strct.index;
				deref_instr = nir_src_as_deref(deref_instr->parent);
				constant_index += glsl_get_struct_location_offset(deref_instr->type, sidx);
			} else {
				unreachable("Unsupported deref type");
			}
		}
		descriptor_set = deref_instr->var->data.descriptor_set;

		if (deref_instr->var->data.bindless) {
			/* For now just assert on unhandled variable types */
			assert(deref_instr->var->data.mode == nir_var_uniform);

			base_index = deref_instr->var->data.driver_location;
			bindless = true;

			index = index ? index : ctx->ac.i32_0;
			index = get_bindless_index_from_uniform(ctx, base_index,
								constant_index, index);
		} else
			base_index = deref_instr->var->data.binding;
	}

	return ctx->abi->load_sampler_desc(ctx->abi,
					  descriptor_set,
					  base_index,
					  constant_index, index,
					  desc_type, image, write, bindless);
}

/* Disable anisotropic filtering if BASE_LEVEL == LAST_LEVEL.
 *
 * GFX6-GFX7:
 *   If BASE_LEVEL == LAST_LEVEL, the shader must disable anisotropic
 *   filtering manually. The driver sets img7 to a mask clearing
 *   MAX_ANISO_RATIO if BASE_LEVEL == LAST_LEVEL. The shader must do:
 *     s_and_b32 samp0, samp0, img7
 *
 * GFX8:
 *   The ANISO_OVERRIDE sampler field enables this fix in TA.
 */
static LLVMValueRef sici_fix_sampler_aniso(struct ac_nir_context *ctx,
                                           LLVMValueRef res, LLVMValueRef samp)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef img7, samp0;

	if (ctx->ac.chip_class >= GFX8)
		return samp;

	img7 = LLVMBuildExtractElement(builder, res,
	                               LLVMConstInt(ctx->ac.i32, 7, 0), "");
	samp0 = LLVMBuildExtractElement(builder, samp,
	                                LLVMConstInt(ctx->ac.i32, 0, 0), "");
	samp0 = LLVMBuildAnd(builder, samp0, img7, "");
	return LLVMBuildInsertElement(builder, samp, samp0,
	                              LLVMConstInt(ctx->ac.i32, 0, 0), "");
}

static void tex_fetch_ptrs(struct ac_nir_context *ctx,
			   nir_tex_instr *instr,
			   LLVMValueRef *res_ptr, LLVMValueRef *samp_ptr,
			   LLVMValueRef *fmask_ptr)
{
	nir_deref_instr *texture_deref_instr = NULL;
	nir_deref_instr *sampler_deref_instr = NULL;
	int plane = -1;

	for (unsigned i = 0; i < instr->num_srcs; i++) {
		switch (instr->src[i].src_type) {
		case nir_tex_src_texture_deref:
			texture_deref_instr = nir_src_as_deref(instr->src[i].src);
			break;
		case nir_tex_src_sampler_deref:
			sampler_deref_instr = nir_src_as_deref(instr->src[i].src);
			break;
		case nir_tex_src_plane:
			plane = nir_src_as_int(instr->src[i].src);
			break;
		default:
			break;
		}
	}

	if (!sampler_deref_instr)
		sampler_deref_instr = texture_deref_instr;

	enum ac_descriptor_type main_descriptor = instr->sampler_dim  == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE;

	if (plane >= 0) {
		assert(instr->op != nir_texop_txf_ms &&
		       instr->op != nir_texop_samples_identical);
		assert(instr->sampler_dim  != GLSL_SAMPLER_DIM_BUF);

		main_descriptor = AC_DESC_PLANE_0 + plane;
	}

	*res_ptr = get_sampler_desc(ctx, texture_deref_instr, main_descriptor, &instr->instr, false, false);

	if (samp_ptr) {
		*samp_ptr = get_sampler_desc(ctx, sampler_deref_instr, AC_DESC_SAMPLER, &instr->instr, false, false);
		if (instr->sampler_dim < GLSL_SAMPLER_DIM_RECT)
			*samp_ptr = sici_fix_sampler_aniso(ctx, *res_ptr, *samp_ptr);
	}
	if (fmask_ptr && (instr->op == nir_texop_txf_ms ||
	                  instr->op == nir_texop_samples_identical))
		*fmask_ptr = get_sampler_desc(ctx, texture_deref_instr, AC_DESC_FMASK, &instr->instr, false, false);
}

static LLVMValueRef apply_round_slice(struct ac_llvm_context *ctx,
				      LLVMValueRef coord)
{
	coord = ac_to_float(ctx, coord);
	coord = ac_build_round(ctx, coord);
	coord = ac_to_integer(ctx, coord);
	return coord;
}

static void visit_tex(struct ac_nir_context *ctx, nir_tex_instr *instr)
{
	LLVMValueRef result = NULL;
	struct ac_image_args args = { 0 };
	LLVMValueRef fmask_ptr = NULL, sample_index = NULL;
	LLVMValueRef ddx = NULL, ddy = NULL;
	unsigned offset_src = 0;

	tex_fetch_ptrs(ctx, instr, &args.resource, &args.sampler, &fmask_ptr);

	for (unsigned i = 0; i < instr->num_srcs; i++) {
		switch (instr->src[i].src_type) {
		case nir_tex_src_coord: {
			LLVMValueRef coord = get_src(ctx, instr->src[i].src);
			for (unsigned chan = 0; chan < instr->coord_components; ++chan)
				args.coords[chan] = ac_llvm_extract_elem(&ctx->ac, coord, chan);
			break;
		}
		case nir_tex_src_projector:
			break;
		case nir_tex_src_comparator:
			if (instr->is_shadow)
				args.compare = get_src(ctx, instr->src[i].src);
			break;
		case nir_tex_src_offset:
			args.offset = get_src(ctx, instr->src[i].src);
			offset_src = i;
			break;
		case nir_tex_src_bias:
			if (instr->op == nir_texop_txb)
				args.bias = get_src(ctx, instr->src[i].src);
			break;
		case nir_tex_src_lod: {
			if (nir_src_is_const(instr->src[i].src) && nir_src_as_uint(instr->src[i].src) == 0)
				args.level_zero = true;
			else
				args.lod = get_src(ctx, instr->src[i].src);
			break;
		}
		case nir_tex_src_ms_index:
			sample_index = get_src(ctx, instr->src[i].src);
			break;
		case nir_tex_src_ms_mcs:
			break;
		case nir_tex_src_ddx:
			ddx = get_src(ctx, instr->src[i].src);
			break;
		case nir_tex_src_ddy:
			ddy = get_src(ctx, instr->src[i].src);
			break;
		case nir_tex_src_texture_offset:
		case nir_tex_src_sampler_offset:
		case nir_tex_src_plane:
		default:
			break;
		}
	}

	if (instr->op == nir_texop_txs && instr->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
		result = get_buffer_size(ctx, args.resource, true);
		goto write_result;
	}

	if (instr->op == nir_texop_texture_samples) {
		LLVMValueRef res, samples, is_msaa;
		res = LLVMBuildBitCast(ctx->ac.builder, args.resource, ctx->ac.v8i32, "");
		samples = LLVMBuildExtractElement(ctx->ac.builder, res,
						  LLVMConstInt(ctx->ac.i32, 3, false), "");
		is_msaa = LLVMBuildLShr(ctx->ac.builder, samples,
					LLVMConstInt(ctx->ac.i32, 28, false), "");
		is_msaa = LLVMBuildAnd(ctx->ac.builder, is_msaa,
				       LLVMConstInt(ctx->ac.i32, 0xe, false), "");
		is_msaa = LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ, is_msaa,
					LLVMConstInt(ctx->ac.i32, 0xe, false), "");

		samples = LLVMBuildLShr(ctx->ac.builder, samples,
					LLVMConstInt(ctx->ac.i32, 16, false), "");
		samples = LLVMBuildAnd(ctx->ac.builder, samples,
				       LLVMConstInt(ctx->ac.i32, 0xf, false), "");
		samples = LLVMBuildShl(ctx->ac.builder, ctx->ac.i32_1,
				       samples, "");
		samples = LLVMBuildSelect(ctx->ac.builder, is_msaa, samples,
					  ctx->ac.i32_1, "");
		result = samples;
		goto write_result;
	}

	if (args.offset && instr->op != nir_texop_txf && instr->op != nir_texop_txf_ms) {
		LLVMValueRef offset[3], pack;
		for (unsigned chan = 0; chan < 3; ++chan)
			offset[chan] = ctx->ac.i32_0;

		unsigned num_components = ac_get_llvm_num_components(args.offset);
		for (unsigned chan = 0; chan < num_components; chan++) {
			offset[chan] = ac_llvm_extract_elem(&ctx->ac, args.offset, chan);
			offset[chan] = LLVMBuildAnd(ctx->ac.builder, offset[chan],
						    LLVMConstInt(ctx->ac.i32, 0x3f, false), "");
			if (chan)
				offset[chan] = LLVMBuildShl(ctx->ac.builder, offset[chan],
							    LLVMConstInt(ctx->ac.i32, chan * 8, false), "");
		}
		pack = LLVMBuildOr(ctx->ac.builder, offset[0], offset[1], "");
		pack = LLVMBuildOr(ctx->ac.builder, pack, offset[2], "");
		args.offset = pack;
	}

	/* TC-compatible HTILE on radeonsi promotes Z16 and Z24 to Z32_FLOAT,
	 * so the depth comparison value isn't clamped for Z16 and
	 * Z24 anymore. Do it manually here for GFX8-9; GFX10 has an explicitly
	 * clamped 32-bit float format.
	 *
	 * It's unnecessary if the original texture format was
	 * Z32_FLOAT, but we don't know that here.
	 */
	if (args.compare &&
	    ctx->ac.chip_class >= GFX8 &&
	    ctx->ac.chip_class <= GFX9 &&
	    ctx->abi->clamp_shadow_reference)
		args.compare = ac_build_clamp(&ctx->ac, ac_to_float(&ctx->ac, args.compare));

	/* pack derivatives */
	if (ddx || ddy) {
		int num_src_deriv_channels, num_dest_deriv_channels;
		switch (instr->sampler_dim) {
		case GLSL_SAMPLER_DIM_3D:
		case GLSL_SAMPLER_DIM_CUBE:
			num_src_deriv_channels = 3;
			num_dest_deriv_channels = 3;
			break;
		case GLSL_SAMPLER_DIM_2D:
		default:
			num_src_deriv_channels = 2;
			num_dest_deriv_channels = 2;
			break;
		case GLSL_SAMPLER_DIM_1D:
			num_src_deriv_channels = 1;
			if (ctx->ac.chip_class == GFX9) {
				num_dest_deriv_channels = 2;
			} else {
				num_dest_deriv_channels = 1;
			}
			break;
		}

		for (unsigned i = 0; i < num_src_deriv_channels; i++) {
			args.derivs[i] = ac_to_float(&ctx->ac,
				ac_llvm_extract_elem(&ctx->ac, ddx, i));
			args.derivs[num_dest_deriv_channels + i] = ac_to_float(&ctx->ac,
				ac_llvm_extract_elem(&ctx->ac, ddy, i));
		}
		for (unsigned i = num_src_deriv_channels; i < num_dest_deriv_channels; i++) {
			args.derivs[i] = ctx->ac.f32_0;
			args.derivs[num_dest_deriv_channels + i] = ctx->ac.f32_0;
		}
	}

	if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE && args.coords[0]) {
		for (unsigned chan = 0; chan < instr->coord_components; chan++)
			args.coords[chan] = ac_to_float(&ctx->ac, args.coords[chan]);
		if (instr->coord_components == 3)
			args.coords[3] = LLVMGetUndef(ctx->ac.f32);
		ac_prepare_cube_coords(&ctx->ac,
			instr->op == nir_texop_txd, instr->is_array,
			instr->op == nir_texop_lod, args.coords, args.derivs);
	}

	/* Texture coordinates fixups */
	if (instr->coord_components > 1 &&
	    instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
	    instr->is_array &&
	    instr->op != nir_texop_txf) {
		args.coords[1] = apply_round_slice(&ctx->ac, args.coords[1]);
	}

	if (instr->coord_components > 2 &&
	    (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
	     instr->sampler_dim == GLSL_SAMPLER_DIM_MS ||
	     instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS ||
	     instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) &&
	    instr->is_array &&
	    instr->op != nir_texop_txf && instr->op != nir_texop_txf_ms) {
		args.coords[2] = apply_round_slice(&ctx->ac, args.coords[2]);
	}

	if (ctx->ac.chip_class == GFX9 &&
	    instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
	    instr->op != nir_texop_lod) {
		LLVMValueRef filler;
		if (instr->op == nir_texop_txf)
			filler = ctx->ac.i32_0;
		else
			filler = LLVMConstReal(ctx->ac.f32, 0.5);

		if (instr->is_array)
			args.coords[2] = args.coords[1];
		args.coords[1] = filler;
	}

	/* Pack sample index */
	if (instr->op == nir_texop_txf_ms && sample_index)
		args.coords[instr->coord_components] = sample_index;

	if (instr->op == nir_texop_samples_identical) {
		struct ac_image_args txf_args = { 0 };
		memcpy(txf_args.coords, args.coords, sizeof(txf_args.coords));

		txf_args.dmask = 0xf;
		txf_args.resource = fmask_ptr;
		txf_args.dim = instr->is_array ? ac_image_2darray : ac_image_2d;
		result = build_tex_intrinsic(ctx, instr, &txf_args);

		result = LLVMBuildExtractElement(ctx->ac.builder, result, ctx->ac.i32_0, "");
		result = emit_int_cmp(&ctx->ac, LLVMIntEQ, result, ctx->ac.i32_0);
		goto write_result;
	}

	if ((instr->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS ||
	     instr->sampler_dim == GLSL_SAMPLER_DIM_MS) &&
	    instr->op != nir_texop_txs) {
		unsigned sample_chan = instr->is_array ? 3 : 2;
		args.coords[sample_chan] = adjust_sample_index_using_fmask(
			&ctx->ac, args.coords[0], args.coords[1],
			instr->is_array ? args.coords[2] : NULL,
			args.coords[sample_chan], fmask_ptr);
	}

	if (args.offset && (instr->op == nir_texop_txf || instr->op == nir_texop_txf_ms)) {
		int num_offsets = instr->src[offset_src].src.ssa->num_components;
		num_offsets = MIN2(num_offsets, instr->coord_components);
		for (unsigned i = 0; i < num_offsets; ++i) {
			args.coords[i] = LLVMBuildAdd(
				ctx->ac.builder, args.coords[i],
				LLVMConstInt(ctx->ac.i32, nir_src_comp_as_uint(instr->src[offset_src].src, i), false), "");
		}
		args.offset = NULL;
	}

	/* DMASK was repurposed for GATHER4. 4 components are always
	 * returned and DMASK works like a swizzle - it selects
	 * the component to fetch. The only valid DMASK values are
	 * 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
	 * (red,red,red,red) etc.) The ISA document doesn't mention
	 * this.
	 */
	args.dmask = 0xf;
	if (instr->op == nir_texop_tg4) {
		if (instr->is_shadow)
			args.dmask = 1;
		else
			args.dmask = 1 << instr->component;
	}

	if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF)
		args.dim = get_ac_sampler_dim(&ctx->ac, instr->sampler_dim, instr->is_array);
	result = build_tex_intrinsic(ctx, instr, &args);

	if (instr->op == nir_texop_query_levels)
		result = LLVMBuildExtractElement(ctx->ac.builder, result, LLVMConstInt(ctx->ac.i32, 3, false), "");
	else if (instr->is_shadow && instr->is_new_style_shadow &&
		 instr->op != nir_texop_txs && instr->op != nir_texop_lod &&
		 instr->op != nir_texop_tg4)
		result = LLVMBuildExtractElement(ctx->ac.builder, result, ctx->ac.i32_0, "");
	else if (instr->op == nir_texop_txs &&
		 instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
		 instr->is_array) {
		LLVMValueRef two = LLVMConstInt(ctx->ac.i32, 2, false);
		LLVMValueRef six = LLVMConstInt(ctx->ac.i32, 6, false);
		LLVMValueRef z = LLVMBuildExtractElement(ctx->ac.builder, result, two, "");
		z = LLVMBuildSDiv(ctx->ac.builder, z, six, "");
		result = LLVMBuildInsertElement(ctx->ac.builder, result, z, two, "");
	} else if (ctx->ac.chip_class == GFX9 &&
		   instr->op == nir_texop_txs &&
		   instr->sampler_dim == GLSL_SAMPLER_DIM_1D &&
		   instr->is_array) {
		LLVMValueRef two = LLVMConstInt(ctx->ac.i32, 2, false);
		LLVMValueRef layers = LLVMBuildExtractElement(ctx->ac.builder, result, two, "");
		result = LLVMBuildInsertElement(ctx->ac.builder, result, layers,
						ctx->ac.i32_1, "");
	} else if (instr->dest.ssa.num_components != 4)
		result = ac_trim_vector(&ctx->ac, result, instr->dest.ssa.num_components);

write_result:
	if (result) {
		assert(instr->dest.is_ssa);
		result = ac_to_integer(&ctx->ac, result);
		ctx->ssa_defs[instr->dest.ssa.index] = result;
	}
}


static void visit_phi(struct ac_nir_context *ctx, nir_phi_instr *instr)
{
	LLVMTypeRef type = get_def_type(ctx, &instr->dest.ssa);
	LLVMValueRef result = LLVMBuildPhi(ctx->ac.builder, type, "");

	ctx->ssa_defs[instr->dest.ssa.index] = result;
	_mesa_hash_table_insert(ctx->phis, instr, result);
}

static void visit_post_phi(struct ac_nir_context *ctx,
                           nir_phi_instr *instr,
                           LLVMValueRef llvm_phi)
{
	nir_foreach_phi_src(src, instr) {
		LLVMBasicBlockRef block = get_block(ctx, src->pred);
		LLVMValueRef llvm_src = get_src(ctx, src->src);

		LLVMAddIncoming(llvm_phi, &llvm_src, &block, 1);
	}
}

static void phi_post_pass(struct ac_nir_context *ctx)
{
	hash_table_foreach(ctx->phis, entry) {
		visit_post_phi(ctx, (nir_phi_instr*)entry->key,
		               (LLVMValueRef)entry->data);
	}
}


static void visit_ssa_undef(struct ac_nir_context *ctx,
			    const nir_ssa_undef_instr *instr)
{
	unsigned num_components = instr->def.num_components;
	LLVMTypeRef type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
	LLVMValueRef undef;

	if (num_components == 1)
		undef = LLVMGetUndef(type);
	else {
		undef = LLVMGetUndef(LLVMVectorType(type, num_components));
	}
	ctx->ssa_defs[instr->def.index] = undef;
}

static void visit_jump(struct ac_llvm_context *ctx,
		       const nir_jump_instr *instr)
{
	switch (instr->type) {
	case nir_jump_break:
		ac_build_break(ctx);
		break;
	case nir_jump_continue:
		ac_build_continue(ctx);
		break;
	default:
		fprintf(stderr, "Unknown NIR jump instr: ");
		nir_print_instr(&instr->instr, stderr);
		fprintf(stderr, "\n");
		abort();
	}
}

static LLVMTypeRef
glsl_base_to_llvm_type(struct ac_llvm_context *ac,
		       enum glsl_base_type type)
{
	switch (type) {
	case GLSL_TYPE_INT:
	case GLSL_TYPE_UINT:
	case GLSL_TYPE_BOOL:
	case GLSL_TYPE_SUBROUTINE:
		return ac->i32;
	case GLSL_TYPE_INT8:
	case GLSL_TYPE_UINT8:
		return ac->i8;
	case GLSL_TYPE_INT16:
	case GLSL_TYPE_UINT16:
		return ac->i16;
	case GLSL_TYPE_FLOAT:
		return ac->f32;
	case GLSL_TYPE_FLOAT16:
		return ac->f16;
	case GLSL_TYPE_INT64:
	case GLSL_TYPE_UINT64:
		return ac->i64;
	case GLSL_TYPE_DOUBLE:
		return ac->f64;
	default:
		unreachable("unknown GLSL type");
	}
}

static LLVMTypeRef
glsl_to_llvm_type(struct ac_llvm_context *ac,
		  const struct glsl_type *type)
{
	if (glsl_type_is_scalar(type)) {
		return glsl_base_to_llvm_type(ac, glsl_get_base_type(type));
	}

	if (glsl_type_is_vector(type)) {
		return LLVMVectorType(
		   glsl_base_to_llvm_type(ac, glsl_get_base_type(type)),
		   glsl_get_vector_elements(type));
	}

	if (glsl_type_is_matrix(type)) {
		return LLVMArrayType(
		   glsl_to_llvm_type(ac, glsl_get_column_type(type)),
		   glsl_get_matrix_columns(type));
	}

	if (glsl_type_is_array(type)) {
		return LLVMArrayType(
		   glsl_to_llvm_type(ac, glsl_get_array_element(type)),
		   glsl_get_length(type));
	}

	assert(glsl_type_is_struct_or_ifc(type));

	LLVMTypeRef member_types[glsl_get_length(type)];

	for (unsigned i = 0; i < glsl_get_length(type); i++) {
		member_types[i] =
			glsl_to_llvm_type(ac,
					  glsl_get_struct_field(type, i));
	}

	return LLVMStructTypeInContext(ac->context, member_types,
				       glsl_get_length(type), false);
}

static void visit_deref(struct ac_nir_context *ctx,
                        nir_deref_instr *instr)
{
	if (instr->mode != nir_var_mem_shared &&
	    instr->mode != nir_var_mem_global)
		return;

	LLVMValueRef result = NULL;
	switch(instr->deref_type) {
	case nir_deref_type_var: {
		struct hash_entry *entry = _mesa_hash_table_search(ctx->vars, instr->var);
		result = entry->data;
		break;
	}
	case nir_deref_type_struct:
		if (instr->mode == nir_var_mem_global) {
			nir_deref_instr *parent = nir_deref_instr_parent(instr);
			uint64_t offset = glsl_get_struct_field_offset(parent->type,
                                                                       instr->strct.index);
			result = ac_build_gep_ptr(&ctx->ac, get_src(ctx, instr->parent),
			                       LLVMConstInt(ctx->ac.i32, offset, 0));
		} else {
			result = ac_build_gep0(&ctx->ac, get_src(ctx, instr->parent),
			                       LLVMConstInt(ctx->ac.i32, instr->strct.index, 0));
		}
		break;
	case nir_deref_type_array:
		if (instr->mode == nir_var_mem_global) {
			nir_deref_instr *parent = nir_deref_instr_parent(instr);
			unsigned stride = glsl_get_explicit_stride(parent->type);

			if ((glsl_type_is_matrix(parent->type) &&
			     glsl_matrix_type_is_row_major(parent->type)) ||
			    (glsl_type_is_vector(parent->type) && stride == 0))
				stride = type_scalar_size_bytes(parent->type);

			assert(stride > 0);
			LLVMValueRef index = get_src(ctx, instr->arr.index);
			if (LLVMTypeOf(index) != ctx->ac.i64)
				index = LLVMBuildZExt(ctx->ac.builder, index, ctx->ac.i64, "");

			LLVMValueRef offset = LLVMBuildMul(ctx->ac.builder, index, LLVMConstInt(ctx->ac.i64, stride, 0), "");

			result = ac_build_gep_ptr(&ctx->ac, get_src(ctx, instr->parent), offset);
		} else {
			result = ac_build_gep0(&ctx->ac, get_src(ctx, instr->parent),
			                       get_src(ctx, instr->arr.index));
		}
		break;
	case nir_deref_type_ptr_as_array:
		if (instr->mode == nir_var_mem_global) {
			unsigned stride = nir_deref_instr_ptr_as_array_stride(instr);

			LLVMValueRef index = get_src(ctx, instr->arr.index);
			if (LLVMTypeOf(index) != ctx->ac.i64)
				index = LLVMBuildZExt(ctx->ac.builder, index, ctx->ac.i64, "");

			LLVMValueRef offset = LLVMBuildMul(ctx->ac.builder, index, LLVMConstInt(ctx->ac.i64, stride, 0), "");

			result = ac_build_gep_ptr(&ctx->ac, get_src(ctx, instr->parent), offset);
		} else {
			result = ac_build_gep_ptr(&ctx->ac, get_src(ctx, instr->parent),
			                       get_src(ctx, instr->arr.index));
		}
		break;
	case nir_deref_type_cast: {
		result = get_src(ctx, instr->parent);

		/* We can't use the structs from LLVM because the shader
		 * specifies its own offsets. */
		LLVMTypeRef pointee_type = ctx->ac.i8;
		if (instr->mode == nir_var_mem_shared)
			pointee_type = glsl_to_llvm_type(&ctx->ac, instr->type);

		unsigned address_space;

		switch(instr->mode) {
		case nir_var_mem_shared:
			address_space = AC_ADDR_SPACE_LDS;
			break;
		case nir_var_mem_global:
			address_space = AC_ADDR_SPACE_GLOBAL;
			break;
		default:
			unreachable("Unhandled address space");
		}

		LLVMTypeRef type = LLVMPointerType(pointee_type, address_space);

		if (LLVMTypeOf(result) != type) {
			if (LLVMGetTypeKind(LLVMTypeOf(result)) == LLVMVectorTypeKind) {
				result = LLVMBuildBitCast(ctx->ac.builder, result,
				                          type, "");
			} else {
				result = LLVMBuildIntToPtr(ctx->ac.builder, result,
				                           type, "");
			}
		}
		break;
	}
	default:
		unreachable("Unhandled deref_instr deref type");
	}

	ctx->ssa_defs[instr->dest.ssa.index] = result;
}

static void visit_cf_list(struct ac_nir_context *ctx,
                          struct exec_list *list);

static void visit_block(struct ac_nir_context *ctx, nir_block *block)
{
	nir_foreach_instr(instr, block)
	{
		switch (instr->type) {
		case nir_instr_type_alu:
			visit_alu(ctx, nir_instr_as_alu(instr));
			break;
		case nir_instr_type_load_const:
			visit_load_const(ctx, nir_instr_as_load_const(instr));
			break;
		case nir_instr_type_intrinsic:
			visit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
			break;
		case nir_instr_type_tex:
			visit_tex(ctx, nir_instr_as_tex(instr));
			break;
		case nir_instr_type_phi:
			visit_phi(ctx, nir_instr_as_phi(instr));
			break;
		case nir_instr_type_ssa_undef:
			visit_ssa_undef(ctx, nir_instr_as_ssa_undef(instr));
			break;
		case nir_instr_type_jump:
			visit_jump(&ctx->ac, nir_instr_as_jump(instr));
			break;
		case nir_instr_type_deref:
			visit_deref(ctx, nir_instr_as_deref(instr));
			break;
		default:
			fprintf(stderr, "Unknown NIR instr type: ");
			nir_print_instr(instr, stderr);
			fprintf(stderr, "\n");
			abort();
		}
	}

	_mesa_hash_table_insert(ctx->defs, block,
				LLVMGetInsertBlock(ctx->ac.builder));
}

static void visit_if(struct ac_nir_context *ctx, nir_if *if_stmt)
{
	LLVMValueRef value = get_src(ctx, if_stmt->condition);

	nir_block *then_block =
		(nir_block *) exec_list_get_head(&if_stmt->then_list);

	ac_build_uif(&ctx->ac, value, then_block->index);

	visit_cf_list(ctx, &if_stmt->then_list);

	if (!exec_list_is_empty(&if_stmt->else_list)) {
		nir_block *else_block =
			(nir_block *) exec_list_get_head(&if_stmt->else_list);

		ac_build_else(&ctx->ac, else_block->index);
		visit_cf_list(ctx, &if_stmt->else_list);
	}

	ac_build_endif(&ctx->ac, then_block->index);
}

static void visit_loop(struct ac_nir_context *ctx, nir_loop *loop)
{
	nir_block *first_loop_block =
		(nir_block *) exec_list_get_head(&loop->body);

	ac_build_bgnloop(&ctx->ac, first_loop_block->index);

	visit_cf_list(ctx, &loop->body);

	ac_build_endloop(&ctx->ac, first_loop_block->index);
}

static void visit_cf_list(struct ac_nir_context *ctx,
                          struct exec_list *list)
{
	foreach_list_typed(nir_cf_node, node, node, list)
	{
		switch (node->type) {
		case nir_cf_node_block:
			visit_block(ctx, nir_cf_node_as_block(node));
			break;

		case nir_cf_node_if:
			visit_if(ctx, nir_cf_node_as_if(node));
			break;

		case nir_cf_node_loop:
			visit_loop(ctx, nir_cf_node_as_loop(node));
			break;

		default:
			assert(0);
		}
	}
}

void
ac_handle_shader_output_decl(struct ac_llvm_context *ctx,
			     struct ac_shader_abi *abi,
			     struct nir_shader *nir,
			     struct nir_variable *variable,
			     gl_shader_stage stage)
{
	unsigned output_loc = variable->data.driver_location / 4;
	unsigned attrib_count = glsl_count_attribute_slots(variable->type, false);

	/* tess ctrl has it's own load/store paths for outputs */
	if (stage == MESA_SHADER_TESS_CTRL)
		return;

	if (stage == MESA_SHADER_VERTEX ||
	    stage == MESA_SHADER_TESS_EVAL ||
	    stage == MESA_SHADER_GEOMETRY) {
		int idx = variable->data.location + variable->data.index;
		if (idx == VARYING_SLOT_CLIP_DIST0) {
			int length = nir->info.clip_distance_array_size +
				     nir->info.cull_distance_array_size;

			if (length > 4)
				attrib_count = 2;
			else
				attrib_count = 1;
		}
	}

	bool is_16bit = glsl_type_is_16bit(glsl_without_array(variable->type));
	LLVMTypeRef type = is_16bit ? ctx->f16 : ctx->f32;
	for (unsigned i = 0; i < attrib_count; ++i) {
		for (unsigned chan = 0; chan < 4; chan++) {
			abi->outputs[ac_llvm_reg_index_soa(output_loc + i, chan)] =
		                       ac_build_alloca_undef(ctx, type, "");
		}
	}
}

static void
setup_locals(struct ac_nir_context *ctx,
	     struct nir_function *func)
{
	int i, j;
	ctx->num_locals = 0;
	nir_foreach_variable(variable, &func->impl->locals) {
		unsigned attrib_count = glsl_count_attribute_slots(variable->type, false);
		variable->data.driver_location = ctx->num_locals * 4;
		variable->data.location_frac = 0;
		ctx->num_locals += attrib_count;
	}
	ctx->locals = malloc(4 * ctx->num_locals * sizeof(LLVMValueRef));
	if (!ctx->locals)
	    return;

	for (i = 0; i < ctx->num_locals; i++) {
		for (j = 0; j < 4; j++) {
			ctx->locals[i * 4 + j] =
				ac_build_alloca_undef(&ctx->ac, ctx->ac.f32, "temp");
		}
	}
}

static void
setup_scratch(struct ac_nir_context *ctx,
	      struct nir_shader *shader)
{
	if (shader->scratch_size == 0)
		return;

	ctx->scratch = ac_build_alloca_undef(&ctx->ac,
					     LLVMArrayType(ctx->ac.i8, shader->scratch_size),
					     "scratch");
}

static void
setup_constant_data(struct ac_nir_context *ctx,
		    struct nir_shader *shader)
{
	if (!shader->constant_data)
		return;

	LLVMValueRef data =
		LLVMConstStringInContext(ctx->ac.context,
					 shader->constant_data,
					 shader->constant_data_size,
					 true);
	LLVMTypeRef type = LLVMArrayType(ctx->ac.i8, shader->constant_data_size);

	/* We want to put the constant data in the CONST address space so that
	 * we can use scalar loads. However, LLVM versions before 10 put these
	 * variables in the same section as the code, which is unacceptable
	 * for RadeonSI as it needs to relocate all the data sections after
	 * the code sections. See https://reviews.llvm.org/D65813.
	 */
	unsigned address_space =
		LLVM_VERSION_MAJOR < 10 ? AC_ADDR_SPACE_GLOBAL : AC_ADDR_SPACE_CONST;

	LLVMValueRef global =
		LLVMAddGlobalInAddressSpace(ctx->ac.module, type,
					    "const_data",
					    address_space);

	LLVMSetInitializer(global, data);
	LLVMSetGlobalConstant(global, true);
	LLVMSetVisibility(global, LLVMHiddenVisibility);
	ctx->constant_data = global;
}

static void
setup_shared(struct ac_nir_context *ctx,
	     struct nir_shader *nir)
{
	nir_foreach_variable(variable, &nir->shared) {
		LLVMValueRef shared =
			LLVMAddGlobalInAddressSpace(
			   ctx->ac.module, glsl_to_llvm_type(&ctx->ac, variable->type),
			   variable->name ? variable->name : "",
			   AC_ADDR_SPACE_LDS);
		_mesa_hash_table_insert(ctx->vars, variable, shared);
	}
}

void ac_nir_translate(struct ac_llvm_context *ac, struct ac_shader_abi *abi,
		      struct nir_shader *nir)
{
	struct ac_nir_context ctx = {};
	struct nir_function *func;

	ctx.ac = *ac;
	ctx.abi = abi;

	ctx.stage = nir->info.stage;
	ctx.info = &nir->info;

	ctx.main_function = LLVMGetBasicBlockParent(LLVMGetInsertBlock(ctx.ac.builder));

	nir_foreach_variable(variable, &nir->outputs)
		ac_handle_shader_output_decl(&ctx.ac, ctx.abi, nir, variable,
					     ctx.stage);

	ctx.defs = _mesa_hash_table_create(NULL, _mesa_hash_pointer,
	                                   _mesa_key_pointer_equal);
	ctx.phis = _mesa_hash_table_create(NULL, _mesa_hash_pointer,
	                                   _mesa_key_pointer_equal);
	ctx.vars = _mesa_hash_table_create(NULL, _mesa_hash_pointer,
	                                   _mesa_key_pointer_equal);

	func = (struct nir_function *)exec_list_get_head(&nir->functions);

	nir_index_ssa_defs(func->impl);
	ctx.ssa_defs = calloc(func->impl->ssa_alloc, sizeof(LLVMValueRef));

	setup_locals(&ctx, func);
	setup_scratch(&ctx, nir);
	setup_constant_data(&ctx, nir);

	if (gl_shader_stage_is_compute(nir->info.stage))
		setup_shared(&ctx, nir);

	visit_cf_list(&ctx, &func->impl->body);
	phi_post_pass(&ctx);

	if (!gl_shader_stage_is_compute(nir->info.stage))
		ctx.abi->emit_outputs(ctx.abi, AC_LLVM_MAX_OUTPUTS,
				      ctx.abi->outputs);

	free(ctx.locals);
	free(ctx.ssa_defs);
	ralloc_free(ctx.defs);
	ralloc_free(ctx.phis);
	ralloc_free(ctx.vars);
}

void
ac_lower_indirect_derefs(struct nir_shader *nir, enum chip_class chip_class)
{
	/* Lower large variables to scratch first so that we won't bloat the
	 * shader by generating large if ladders for them. We later lower
	 * scratch to alloca's, assuming LLVM won't generate VGPR indexing.
	 */
	NIR_PASS_V(nir, nir_lower_vars_to_scratch,
		   nir_var_function_temp,
		   256,
		   glsl_get_natural_size_align_bytes);

	/* While it would be nice not to have this flag, we are constrained
	 * by the reality that LLVM 9.0 has buggy VGPR indexing on GFX9.
	 */
	bool llvm_has_working_vgpr_indexing = chip_class != GFX9;

	/* TODO: Indirect indexing of GS inputs is unimplemented.
	 *
	 * TCS and TES load inputs directly from LDS or offchip memory, so
	 * indirect indexing is trivial.
	 */
	nir_variable_mode indirect_mask = 0;
	if (nir->info.stage == MESA_SHADER_GEOMETRY ||
	    (nir->info.stage != MESA_SHADER_TESS_CTRL &&
	     nir->info.stage != MESA_SHADER_TESS_EVAL &&
	     !llvm_has_working_vgpr_indexing)) {
		indirect_mask |= nir_var_shader_in;
	}
	if (!llvm_has_working_vgpr_indexing &&
	    nir->info.stage != MESA_SHADER_TESS_CTRL)
		indirect_mask |= nir_var_shader_out;

	/* TODO: We shouldn't need to do this, however LLVM isn't currently
	 * smart enough to handle indirects without causing excess spilling
	 * causing the gpu to hang.
	 *
	 * See the following thread for more details of the problem:
	 * https://lists.freedesktop.org/archives/mesa-dev/2017-July/162106.html
	 */
	indirect_mask |= nir_var_function_temp;

	nir_lower_indirect_derefs(nir, indirect_mask);
}

static unsigned
get_inst_tessfactor_writemask(nir_intrinsic_instr *intrin)
{
	if (intrin->intrinsic != nir_intrinsic_store_deref)
		return 0;

	nir_variable *var =
		nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[0]));

	if (var->data.mode != nir_var_shader_out)
		return 0;

	unsigned writemask = 0;
	const int location = var->data.location;
	unsigned first_component = var->data.location_frac;
	unsigned num_comps = intrin->dest.ssa.num_components;

	if (location == VARYING_SLOT_TESS_LEVEL_INNER)
		writemask = ((1 << (num_comps + 1)) - 1) << first_component;
	else if (location == VARYING_SLOT_TESS_LEVEL_OUTER)
		writemask = (((1 << (num_comps + 1)) - 1) << first_component) << 4;

	return writemask;
}

static void
scan_tess_ctrl(nir_cf_node *cf_node, unsigned *upper_block_tf_writemask,
	       unsigned *cond_block_tf_writemask,
	       bool *tessfactors_are_def_in_all_invocs, bool is_nested_cf)
{
	switch (cf_node->type) {
	case nir_cf_node_block: {
		nir_block *block = nir_cf_node_as_block(cf_node);
		nir_foreach_instr(instr, block) {
			if (instr->type != nir_instr_type_intrinsic)
				continue;

			nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
			if (intrin->intrinsic == nir_intrinsic_barrier) {

				/* If we find a barrier in nested control flow put this in the
				 * too hard basket. In GLSL this is not possible but it is in
				 * SPIR-V.
				 */
				if (is_nested_cf) {
					*tessfactors_are_def_in_all_invocs = false;
					return;
				}

				/* The following case must be prevented:
				 *    gl_TessLevelInner = ...;
				 *    barrier();
				 *    if (gl_InvocationID == 1)
				 *       gl_TessLevelInner = ...;
				 *
				 * If you consider disjoint code segments separated by barriers, each
				 * such segment that writes tess factor channels should write the same
				 * channels in all codepaths within that segment.
				 */
				if (upper_block_tf_writemask || cond_block_tf_writemask) {
					/* Accumulate the result: */
					*tessfactors_are_def_in_all_invocs &=
						!(*cond_block_tf_writemask & ~(*upper_block_tf_writemask));

					/* Analyze the next code segment from scratch. */
					*upper_block_tf_writemask = 0;
					*cond_block_tf_writemask = 0;
				}
			} else
				*upper_block_tf_writemask |= get_inst_tessfactor_writemask(intrin);
		}

		break;
	}
	case nir_cf_node_if: {
		unsigned then_tessfactor_writemask = 0;
		unsigned else_tessfactor_writemask = 0;

		nir_if *if_stmt = nir_cf_node_as_if(cf_node);
		foreach_list_typed(nir_cf_node, nested_node, node, &if_stmt->then_list) {
			scan_tess_ctrl(nested_node, &then_tessfactor_writemask,
				       cond_block_tf_writemask,
				       tessfactors_are_def_in_all_invocs, true);
		}

		foreach_list_typed(nir_cf_node, nested_node, node, &if_stmt->else_list) {
			scan_tess_ctrl(nested_node, &else_tessfactor_writemask,
				       cond_block_tf_writemask,
				       tessfactors_are_def_in_all_invocs, true);
		}

		if (then_tessfactor_writemask || else_tessfactor_writemask) {
			/* If both statements write the same tess factor channels,
			 * we can say that the upper block writes them too.
			 */
			*upper_block_tf_writemask |= then_tessfactor_writemask &
				else_tessfactor_writemask;
			*cond_block_tf_writemask |= then_tessfactor_writemask |
				else_tessfactor_writemask;
		}

		break;
	}
	case nir_cf_node_loop: {
		nir_loop *loop = nir_cf_node_as_loop(cf_node);
		foreach_list_typed(nir_cf_node, nested_node, node, &loop->body) {
			scan_tess_ctrl(nested_node, cond_block_tf_writemask,
				       cond_block_tf_writemask,
				       tessfactors_are_def_in_all_invocs, true);
		}

		break;
	}
	default:
		unreachable("unknown cf node type");
	}
}

bool
ac_are_tessfactors_def_in_all_invocs(const struct nir_shader *nir)
{
	assert(nir->info.stage == MESA_SHADER_TESS_CTRL);

	/* The pass works as follows:
	 * If all codepaths write tess factors, we can say that all
	 * invocations define tess factors.
	 *
	 * Each tess factor channel is tracked separately.
	 */
	unsigned main_block_tf_writemask = 0; /* if main block writes tess factors */
	unsigned cond_block_tf_writemask = 0; /* if cond block writes tess factors */

	/* Initial value = true. Here the pass will accumulate results from
	 * multiple segments surrounded by barriers. If tess factors aren't
	 * written at all, it's a shader bug and we don't care if this will be
	 * true.
	 */
	bool tessfactors_are_def_in_all_invocs = true;

	nir_foreach_function(function, nir) {
		if (function->impl) {
			foreach_list_typed(nir_cf_node, node, node, &function->impl->body) {
				scan_tess_ctrl(node, &main_block_tf_writemask,
					       &cond_block_tf_writemask,
					       &tessfactors_are_def_in_all_invocs,
					       false);
			}
		}
	}

	/* Accumulate the result for the last code segment separated by a
	 * barrier.
	 */
	if (main_block_tf_writemask || cond_block_tf_writemask) {
		tessfactors_are_def_in_all_invocs &=
			!(cond_block_tf_writemask & ~main_block_tf_writemask);
	}

	return tessfactors_are_def_in_all_invocs;
}
