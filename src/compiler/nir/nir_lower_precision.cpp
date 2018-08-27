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

/* TODO: Introduce helpers in C++ space for examining GLSL types and make
 *       this file just C.
 */

#include "nir.h"
#include "nir_builder.h"
#include "compiler/glsl_types.h"

typedef enum {
   op_src_bit_size_undef = 0,
   op_src_bit_size_8     = 8,
   op_src_bit_size_16    = 16,
   op_src_bit_size_32    = 32,
   op_src_bit_size_64    = 64,
} op_src_bit_size;

static const struct glsl_type *
get_lower_precision_type(const struct glsl_type *highp)
{
   if (highp->is_float())
      return glsl_type::get_instance(GLSL_TYPE_FLOAT16,
                                     highp->vector_elements, 
                                     highp->matrix_columns);

   if (highp->is_array() && highp->fields.array->is_float())
      return glsl_type::get_array_instance(
                glsl_type::get_instance(GLSL_TYPE_FLOAT16,
                                        highp->fields.array->vector_elements, 
                                        highp->fields.array->matrix_columns),
                highp->length);

   return highp;
}

/*
 * Check variables for floating point type and a marker allowing them to be
 * represented with lower precision than 32-bits (mediump or lowp). If the
 * marker is found, change the type of the variable to correspond to the
 * lower precision.
 */
static bool
lower_var_precision(nir_shader *shader,
                    nir_function_impl *impl,
                    struct exec_list *vars)
{
   bool progress = false;

   nir_foreach_variable_safe(var, vars) {
      if (!var->type->is_float())
         continue;

      if (var->data.precision != GLSL_PRECISION_MEDIUM &&
          var->data.precision != GLSL_PRECISION_LOW)
         continue;

      var->type = get_lower_precision_type(var->type);
      progress = true;
   }

   return progress;
}

bool
nir_lower_var_precision(nir_shader *shader,
                        const struct nir_lower_precision_options *options)
{
   bool progress = false;

   progress |= lower_var_precision(shader, NULL, &shader->uniforms);
   progress |= lower_var_precision(shader, NULL, &shader->globals);
   progress |= lower_var_precision(shader, NULL, &shader->outputs);

   if (options->has_16_bit_input_varyings)
      progress |= lower_var_precision(shader, NULL, &shader->inputs);

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      progress |= lower_var_precision(shader, function->impl,
                                      &function->impl->locals);
   }

   return progress;
}

static unsigned
get_alu_bit_sizes(const nir_alu_instr *alu)
{
   unsigned src_bit_sizes = op_src_bit_size_undef;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      const unsigned src_i_bit_size = nir_src_bit_size(alu->src[i].src);

      src_bit_sizes |= src_i_bit_size;
   }

   return src_bit_sizes;
}

/* Insert the conversion just after the instruction producing the given
 * source. This is needed when phi sources are converted - they can't have
 * non-phi predecessors in the block. Insert the conversion to the block
 * hosting the source instead.
 */
static nir_ssa_def *
insert_conversion_after_src(nir_shader *shader, nir_src *src, nir_op op)
{
   assert(src->is_ssa);

   const nir_op_info *op_info = &nir_op_infos[op];
   const unsigned bit_size = nir_alu_type_get_type_size(op_info->output_type);
   const unsigned num_components = src->ssa->num_components;
   nir_alu_instr *instr = nir_alu_instr_create(shader, op);

   instr->src[0].src = *src;

   nir_ssa_dest_init(&instr->instr, &instr->dest.dest, num_components,
                     bit_size, NULL);
   instr->dest.write_mask = (1 << num_components) - 1;

   nir_instr_insert_after(src->ssa->parent_instr, &instr->instr);

   return &instr->dest.dest.ssa;
}

static void
promote_src_to_high_precision(nir_builder *b, nir_instr *instr, nir_src *src)
{
   /* No need if source already has high precision. */
   if (nir_src_bit_size(*src) >= 32)
      return;

   const nir_src promoted = nir_src_for_ssa(
      insert_conversion_after_src(b->shader, src, nir_op_f2f32));
   nir_instr_rewrite_src(instr, src, promoted);
   nir_src_copy(src, &promoted, instr);
}

static void
lower_alu_precision(nir_builder *b, nir_alu_instr *alu)
{
   const unsigned dest_bit_size = nir_dest_bit_size(alu->dest.dest);
   unsigned src_bit_sizes = get_alu_bit_sizes(alu);
   const bool has_high_precision_srcs =
      src_bit_sizes & (op_src_bit_size_32 | op_src_bit_size_64);

   assert(dest_bit_size >= 32);

   /* First consider the case where the operation doesn't involve any medium
    * precision source operands.
    */
   if (!(src_bit_sizes & op_src_bit_size_16))
      return;

   /* In case of mixed mode operands one needs to consider case-by-case if
    * lower precision source operands need to be promoted to higher precision.
    *
    * Otherwise (in case of lower precision sources only), adjust the
    * destination size accordingly. As there isn't a bool16 type in NIR,
    * one leaves the bit size intact and relies on the compiler backend to
    * handle it.
    */
   if (has_high_precision_srcs) {
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         promote_src_to_high_precision(b, &alu->instr, &alu->src[i].src);
      }
   } else if (nir_op_infos[alu->op].output_type != nir_type_bool32) {
      assert(alu->dest.dest.is_ssa);
      alu->dest.dest.ssa.bit_size = 16;
   }
}

static bool
var_has_lower_precision(const nir_variable *var)
{
   return var->type->get_scalar_type()->base_type == GLSL_TYPE_FLOAT16;
}

static void
lower_deref_precision(nir_deref_instr *deref)
{
   /* Clean up any dead derefs we find lying around. They may refer
    * to variables whose precision got lowered.
    */
   if (nir_deref_instr_remove_if_unused(deref))
      return;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   assert(var);

   if (!var_has_lower_precision(var))
      return;

   deref->type = get_lower_precision_type(deref->type);

   assert(deref->dest.is_ssa);
   deref->dest.ssa.bit_size = 16;
}

/* TODO: Allow 16-bit coordinates if hardware supports them. */
static void
adjust_tex_src_precision(nir_builder *b, nir_tex_instr *tex,
                         bool has_16_bit_tex_coords)
{
   for (unsigned i = 0; i < tex->num_srcs; ++i) {
      assert(tex->src[i].src.is_ssa);
      if (tex->src[i].src.ssa->bit_size != 16)
         continue;

      /* TODO: Check individual tex parameters and skip coords if needed. */

      nir_src promoted = nir_src_for_ssa(nir_f2f32(b, tex->src[i].src.ssa));
      nir_instr_rewrite_src(&tex->instr, &tex->src[i].src, promoted);
      nir_src_copy(&tex->src[i].src, &promoted, &tex->instr);
   }
}

static void
convert_deref_store_src(nir_builder *b, nir_intrinsic_instr *store,
                        nir_op conversion_op)
{
   nir_src converted = nir_src_for_ssa(
      nir_build_alu(b, conversion_op, store->src[1].ssa, NULL, NULL, NULL));
   nir_instr_rewrite_src(&store->instr, &store->src[1], converted);
   nir_src_copy(&store->src[1], &converted, &store->instr);
}

static void
lower_intrinsic_precision(nir_builder *b, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_deref:
      if (intr->src[0].ssa->bit_size == 16) {
         assert(intr->dest.is_ssa);
         intr->dest.ssa.bit_size = 16;
      }
      break;

   /* If destination has lower precision but source doesn't, emit conversion
    * from higher to low. Otherwise consider if destination requires higher
    * precision but source is lower, and emit equivalent conversion. This can
    * happen with dereferences to temporaries. They don't have any precision
    * qualifiers and one can't tell if lower precision is allowed without
    * examining the expressions consuming their values.
    * Here one simply treats temporaries with full precision and emits
    * converting copies that preserves correctness. Later optimization passes
    * can remove the copies when all uses are known and it becomes clear if
    * lower/higher precision alone is sufficient.
    */
   case nir_intrinsic_store_deref:
      if (intr->src[0].ssa->bit_size == 16 &&
          intr->src[1].ssa->bit_size > 16)
         convert_deref_store_src(b, intr, nir_op_f2f16);
      else if (intr->src[1].ssa->bit_size == 16 &&
               intr->src[0].ssa->bit_size > 16)
         convert_deref_store_src(b, intr, nir_op_f2f32);
      break;

   case nir_intrinsic_copy_deref:
      unreachable("copy derefs should have been lowered to load-stores");

   default:
      break;
   }
}

static void
lower_instr_precision(nir_function_impl *impl,
                      const struct nir_lower_precision_options *options)
{
   nir_builder b;
   nir_builder_init(&b, impl);
       
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         b.cursor = nir_before_instr(instr);

         switch(instr->type) {
         case nir_instr_type_alu:
            lower_alu_precision(&b, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_deref:
            lower_deref_precision(nir_instr_as_deref(instr));
            break;
         case nir_instr_type_tex:
            adjust_tex_src_precision(&b, nir_instr_as_tex(instr),
                                     options->has_16_bit_tex_coords);
            break;
         case nir_instr_type_intrinsic:
            lower_intrinsic_precision(&b, nir_instr_as_intrinsic(instr));
            break;
         default:
            break;
         }
      }
   }
}

bool
nir_lower_precision(nir_shader *shader,
                    const struct nir_lower_precision_options *options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      lower_instr_precision(function->impl, options);
   }

   return progress;
}
