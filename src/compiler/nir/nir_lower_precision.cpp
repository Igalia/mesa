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

/*
 * This lowering pass seeks to change precision for float values and for the
 * expressions producing them. Decision making follows the rules described in
 * GLES 3.2 Specification and section 4.7.3 Precision Qualifiers. Shortly the
 * idea is that an arithmetic expression can be performed in 16-bit precision
 * if and only if all its operands are either already fixed to 16-bits or are
 * such that compiler is free to use either 32 or 16-bit precision.
 *
 * First step is to go over the variables as these are the only things that
 * are marked by the shader author with explicit instructions if high
 * precision is needed (or lower precision allowed respectively). This
 * implementation sets precision unconditionally to 16-bits whenever allowed
 * by the shader author. (This may not produce the most optimal end result
 * but is a design choice to keep the complexity at bay).
 *
 * This is followed by manipulation of instructions themselves. Variable
 * derefs and intrinsics dealing with derefs are straight-forward. For them
 * one only needs to consult the variables themselves and adjust the precision
 * of the instruction in question accordingly.
 *
 * At this point things get more complex as the rest are dependent on context.
 * Precision for texturing return values (sample values) and constant loads 
 * depend on the needs of consuming expressions. As there may be need for
 * both 16 and 32-bit precision, one cannot simply just set them as 16-bits.
 * There may be, for example, two separate multiplications of a sample value
 * S. One multiplying it with 32-bit value A and the other with 16-bit
 * value B. (Recall that the rules mandate that if one of the source operands
 * has full precision then the rest need to have full precision as well).
 * Hence A * S requires the sample value S with full precision. This in turn
 * means that texturing needs to return full precision and needs to convert
 * sample value as to 16-bits for the other multiplication (B * S).
 * NOTE: Hardware may have capability for mixed mode instructions and it is
 *       left for the backend to drop any unnecessary conversions.
 *
 * Here the implementation leaves all texturing and constant load operations
 * to 32-bit precision until all instructions are analysed. Instead it inserts
 * conversions from 32-bits to 16-bits for expressions that can operate with
 * lower precision. In the example above, the multiplication of the sample
 * with 16-bit value B would become C = B * f2f16(S). This is important for
 * the analysis of the rest of the instructions. Once the pass examines
 * expressions consuming C the pass can allow these expressions with lower
 * precision if all the operands are allowed in lower precision. If one had
 * left C with 32-bit precision it would have prevented the use of 16-bit
 * precision in the consuming expressions even though all other operands would
 * have allowed that.
 *
 * Once all instructions are examined there is separate pass that goes thru
 * all the uses of texturing return values. If all are happy with lower
 * precision, the pass removes the conversions (f2f16) and switches the
 * texturing itself to directly return 16-bit samples (given that hardware
 * support 16-bit sample values of course).
 *
 * For input varyings marked with lower precision there is an alternative to
 * uploading 16-bit values into the shader. One can load them with using full
 * precision but immediately convert them into 16-bits before they are used.
 * This allows one to perform all calculations based on them in 16-bit
 * precision but still keep the upload mechanism intact in the backend.
 *
 * TODO:
 *
 * 1) There is still major flaw: logic is against the rules as it considers
 *    arithmetic expressions without consider to their consuming expressions.
 *    As alus at nir level are just sub-expressions of larger expressions they
 *    are subject to the uses and shouldn't be examined just based on their
 *    own sources.
 *    One should recursively examaine uses until either a fixed search depth
 *    (heuristic to avoid runtime explosion) or it becomes clear which
 *    precision is needed. Naturally there may be both low and high precision
 *    uses. In order to keep things simple one could just force all lower
 *    precision uses to high in case even one high precision use is found or
 *    the search depth boundary is hit.
 *
 * 2) Validation (NIR) expects expression producing booleans to have
 *    destination size of 32-bits. This prevents one from marking expressions
 *    that operate with 16-bit sources and producing booleans as having
 *    16-bit destination. Some hardware (such as Intel) works this way.
 *    Implementation here inserts dummy conversions to 16-bits (namely i2i16
 *    beacuse NIR allows treating SSA values produced as booleans to be
 *    treated as integers without explicit conversions). This allows the
 *    aforementioned analysis to use 16-bit precision in consuming expressions
 *    such as logical and/or that have integer typed sources that could be
 *    produced using, for example, with less-than comparison.
 *
 *    Similar treatment is needed for boolean sources produced using integer
 *    expressions. For example, "bcsel" and "if" can operate with values
 *    produced with 16-bit iand. In order to keep validation happy that
 *    expects the bit size to be 32 one emits dummy i2i32.
 *
 *    At least Intel backend is capable enough to remove most of these
 *    redundant expressions and adjust the consumers to use the original
 *    value. In order to get rid of the rest, there is additional lowering
 *    pass that adjust boolean sized ssa definitions. This is hooked after
 *    all validation is done (see nir_lower_bool_size.c).
 *
 * 3) Allow interpolation of input varyings in 16-bit precision. Currently
 *    the conversion from 32-bits to 16-bits takes place after the conversion.
 *
 * 4) When only 32-bit tex coordinates are supported, see if backend supports
 *    conversion from 16-bits to 32-bits during the last alu producing the
 *    actual value. (Most likely belongs to backend such as described above).
 *
 * 5) Introduce helpers in C++ space for examining GLSL types and make this
 *    file just C.
 *
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
   op_src_bit_size_any   = 1 << 31,
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

static bool
is_instr_size_fixed(const nir_src *src)
{
   assert(src->is_ssa);

   switch(src->ssa->parent_instr->type) {
   case nir_instr_type_alu:
      return true;
   case nir_instr_type_tex:
   case nir_instr_type_load_const:
      return false;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr =
         nir_instr_as_intrinsic(src->ssa->parent_instr);

      if (intr->intrinsic != nir_intrinsic_load_deref)
         return true;

      assert(intr->src[0].is_ssa);
      const nir_deref_instr *deref =
         nir_instr_as_deref(intr->src[0].ssa->parent_instr);

      if (deref->var->data.mode != nir_var_shader_in)
         return true;

      return (deref->var->data.precision != GLSL_PRECISION_MEDIUM) &&
             (deref->var->data.precision != GLSL_PRECISION_LOW);
      }
   default:
      return true;
   }
}

/* Consider alu expressions according to the rules found in GLES 3.2
 * Specification, 4.7.3 Precision Qualifiers:
 *
 * In cases where operands do not have a precision qualifier, the precision
 * qualification will come from the other operands.  If no operands have a
 * precision qualifier, then the precision qualifications of the operands of
 * the next consuming operation in the expression will be used.  This rule can
 * be applied recursively until a precision qualified operand is found.  If
 * necessary, it will also include the precision qualification of l-values for
 * assignments, of the declared variable for initializers, of formal
 * parameters for function call arguments, or of function return types for
 * function return values.  If the precision cannot be determined by this
 * method e.g.  if an entire expression is composed only of operands with no
 * precision qualifier, and the result is not assigned or passed as an
 * argument,  then it is evaluated at the default precision of the type or
 * greater.  When this occurs in the fragment shader, the default precision
 * must be defined.
 * 
 * For example, consider the statements:
 * 
 *   uniform highp float h1;
 *   highp float h2 = 2.3 * 4.7; // operation and result are highp precision
 *   mediump float m;
 *   m = 3.7 * h1 * h2;          // all operations are highp precision
 *   h2 = m * h1;                // operation is highp precision
 *   m = h2 - h1;                // operation is highp precision
 *   h2 = m + m;                 // addition and result at mediump precision
 *   void f(highp float p);
 *   f(3.3);                     // 3.3 will be passed in at highp precisi
 */
static unsigned
get_src_bit_size(const nir_src *src, nir_alu_type type,
                 bool has_var_sized_bool)
{
   /* Booleans are special, at NIR level they are treated as 32-bits. There
    * are, however, hardware (such as Intel) where the presentation of
    * booleans is dependent on type of the expression producing them.
    * Logical operations evolving 16-bit sources produce 16-bit booleans.
    */
   if (has_var_sized_bool && type == nir_type_bool32) {
      assert(src->is_ssa);
       
      /* Only an ALU can produce booleans. */
      assert(src->ssa->parent_instr->type == nir_instr_type_alu);

      const nir_alu_instr *alu = nir_instr_as_alu(src->ssa->parent_instr);

      assert(nir_op_infos[alu->op].num_inputs);

      const unsigned src_bit_size = nir_src_bit_size(alu->src[0].src);

      /* All operands have to agree. */
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         assert(src_bit_size == nir_src_bit_size(alu->src[i].src));
      }

      return src_bit_size;
   }

   return nir_src_bit_size(*src);
}

static unsigned
get_alu_bit_sizes(const nir_alu_instr *alu, bool has_var_sized_bool)
{
   unsigned src_bit_sizes = op_src_bit_size_undef;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      const unsigned src_i_bit_size = get_src_bit_size(
         &alu->src[i].src, nir_op_infos[alu->op].input_types[i],
         has_var_sized_bool);

      if (src_i_bit_size < 32 || is_instr_size_fixed(&alu->src[i].src))
         src_bit_sizes |= src_i_bit_size;
      else
         src_bit_sizes |= op_src_bit_size_any;
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

static bool
demote_src_to_medium_precision(nir_builder *b, nir_instr *instr, nir_src *src)
{
   /* No need if source already has lower precision. */
   if (nir_src_bit_size(*src) < 32)
      return false;

   const nir_src demoted = nir_src_for_ssa(
      insert_conversion_after_src(b->shader, src, nir_op_f2f16));
   nir_instr_rewrite_src(instr, src, demoted);
   nir_src_copy(src, &demoted, instr);

   return true;
}

static void
adjust_int_precision(nir_builder *b, const nir_ssa_def *def,
                     nir_instr *instr, nir_src *src)
{
   assert(src->is_ssa);
   if (src->ssa->index != def->index)
      return;

   const nir_src demoted = nir_src_for_ssa(
      insert_conversion_after_src(b->shader, src, nir_op_i2i16));
   nir_instr_rewrite_src(instr, src, demoted);
   nir_src_copy(src, &demoted, instr);
}

/* Integer values may be produced with expressions producing booleans without
 * explicit conversions from boolean to the integer. Therefore one checks here
 * all the uses of the boolean and makes adjustments if needed.
 */
static void
adjust_bool_uses(nir_builder *b, nir_ssa_def *def)
{
   nir_foreach_if_use(use, def) {
   }

   nir_foreach_use_safe(use, def) {
      assert(use->is_ssa);

      switch (use->parent_instr->type) {
      case nir_instr_type_alu: {
         nir_alu_instr *alu = nir_instr_as_alu(use->parent_instr);

         /* Validation does not allow booleans to have 16-bit size. But
          * further analysis needs to know the real precisions in order to
          * allow later expressions with lower precision. Trick here is to
          * emit "false" conversions from 16-bits to 16-bits, in NIR terms
          * conversions from 32-bit booleans to 16-bit integers. Later passes
          * will notice them as no-op and remove them.
          */
         for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
            if (nir_op_infos[alu->op].input_types[i] == nir_type_int ||
                nir_op_infos[alu->op].input_types[i] == nir_type_uint) {
               adjust_int_precision(b, def, &alu->instr, &alu->src[i].src);
            }
         }
      }
      break;

      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr =
            nir_instr_as_intrinsic(use->parent_instr);

         switch (intr->intrinsic) {
         case nir_intrinsic_discard_if:
            adjust_int_precision(b, def, use->parent_instr, &intr->src[0]);
            break;

         default:
            unreachable("");
         }
      }
      break;
      
      default:
         unreachable("");
      }
   }
}

static void
lower_alu_precision(nir_builder *b, nir_alu_instr *alu,
                    const struct nir_lower_precision_options *options)
{
   const unsigned dest_bit_size = nir_dest_bit_size(alu->dest.dest);
   unsigned src_bit_sizes = get_alu_bit_sizes(
                               alu, options->has_var_sized_bool);
   const bool has_flexible_sized_srcs = src_bit_sizes & op_src_bit_size_any;
   const bool has_high_precision_srcs =
      src_bit_sizes & (op_src_bit_size_32 | op_src_bit_size_64);

   /* Ignore 1-bit bool results */
   if (dest_bit_size == 1)
      return;

   assert(dest_bit_size >= 32);

   /* First consider the case where there are sources for which precision
    * isn't fixed. If, in addition, there aren't any fixed high precision
    * sources involved then it is possible to treat the operation with lower
    * precision.
    * Since at this point it is not known if all the consumers of each
    * non-fixed source are allowed to use lower precision, insert conversions
    * for now. These can be removed later on if and when all uses are known
    * to cope with lower precision.
    */
   if (has_flexible_sized_srcs && !has_high_precision_srcs) {
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         /* Booleans are special, their presentation is dependent on hardware
          * and type of the expression producing them.
          */
         if (nir_alu_type_get_base_type(nir_op_infos[alu->op].input_types[i]) ==
             nir_type_bool)
            continue;

         if (demote_src_to_medium_precision(b, &alu->instr, &alu->src[i].src))
            src_bit_sizes |= op_src_bit_size_16;
      }
   }

   /* If there are no lower precision source operands there is nothing to
    * adjust.
    */
   if (!(src_bit_sizes & op_src_bit_size_16))
      return;

   /* Validation doesn't allow b2f from 16-bit value. Therefore convert
    * source first to 32-bits and only then to float.
    */
   if (alu->op == nir_op_b2f32 || alu->op == nir_op_bcsel) {
      const nir_src promoted = nir_src_for_ssa(
         insert_conversion_after_src(b->shader, &alu->src[0].src,
                                     nir_op_i2i32));
      nir_instr_rewrite_src(&alu->instr, &alu->src[0].src, promoted);
      nir_src_copy(&alu->src[0].src, &promoted, &alu->instr);
   }   

   /* In case of mixed mode operands one needs to consider case-by-case if
    * lower precision source operands need to be promoted to higher precision.
    *
    * Otherwise (in case of lower precision sources only), adjust the
    * destination size accordingly. As there isn't a bool16 type in NIR,
    * one must leave the bit size intact. There can be, however, uses of the
    * value as integer without explicit conversion and hence one needs to go
    * and adjust them.
    */
   if (has_high_precision_srcs) {
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         promote_src_to_high_precision(b, &alu->instr, &alu->src[i].src);
      }
   } else if (nir_op_infos[alu->op].output_type == nir_type_bool32) {
      if (options->has_var_sized_bool) {
         assert(alu->dest.dest.is_ssa);
         adjust_bool_uses(b, &alu->dest.dest.ssa);
      }
   } else {
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

/* This follows closely the logic in lower_alu_precision(), only the way of
 * iterating sources differs.
 */
static void
lower_phi_precision(nir_builder *b, nir_phi_instr *phi,
                    const struct nir_lower_precision_options *options)
{
   unsigned src_bit_sizes = op_src_bit_size_undef;

   nir_foreach_phi_src(src, phi) {
      const unsigned bit_size = nir_src_bit_size(src->src);

      if (bit_size < 32 || is_instr_size_fixed(&src->src))
         src_bit_sizes |= bit_size;
      else
         src_bit_sizes |= op_src_bit_size_any;
   }

   const bool has_high_precision_srcs =
      src_bit_sizes & (op_src_bit_size_32 | op_src_bit_size_64);
   const bool has_flexible_sized_srcs = src_bit_sizes & op_src_bit_size_any;

   if (has_flexible_sized_srcs && !has_high_precision_srcs) {
      nir_foreach_phi_src(src, phi) {
         if (demote_src_to_medium_precision(b, &phi->instr, &src->src))
            src_bit_sizes |= op_src_bit_size_16;
      }
   }

   if (!(src_bit_sizes & op_src_bit_size_16))
      return;

   if (has_high_precision_srcs) {
      nir_foreach_phi_src(src, phi) {
         promote_src_to_high_precision(b, &phi->instr, &src->src);
      }
   } else {
      assert(phi->dest.is_ssa);
      phi->dest.ssa.bit_size = 16;
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
            lower_alu_precision(&b, nir_instr_as_alu(instr), options);
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
         case nir_instr_type_phi:
            lower_phi_precision(&b, nir_instr_as_phi(instr), options);
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

/* Consider all uses of the samples returned by the given texturing operation
 * and determine if the samples are allowed with reduced precision. This can
 * deduced by checking if all uses are conversions from 32-bit to lower.
 *
 * If that holds than one can safely omit the conversions and just let the
 * sampler to return lower precision.
 */
static bool
lower_sample_precision(nir_builder *b, nir_tex_instr *tex)
{
   assert(tex->dest.is_ssa);

   /* Any direct 32-bit use in condition rules out lower precision. */
   if (!list_empty(&tex->dest.ssa.if_uses))
      return false;

   nir_foreach_use(use_src, &tex->dest.ssa) {
      if (use_src->parent_instr->type != nir_instr_type_alu ||
          nir_instr_as_alu(use_src->parent_instr)->op != nir_op_f2f16) {
         return false;
      }
   }

   /* First, modify the sampling operation to return 16-bit values. */
   tex->dest.ssa.bit_size = 16;
   tex->dest_type = nir_type_float16;

   /* Then consider each instruction consuming the samples. At this point
    * all of those are known to be conversions from 32-bit to 16-bits.
    */
   nir_foreach_use_safe(tex_use, &tex->dest.ssa) {
      assert(tex_use->parent_instr->type == nir_instr_type_alu);

      const nir_alu_instr *f2f16 = nir_instr_as_alu(tex_use->parent_instr);

      /* Texture uses get updated below. Skip the instructions already
       * modified to use the sample value directly.
       */
      if (f2f16->op != nir_op_f2f16)
         continue;
         
      assert(f2f16->dest.dest.is_ssa);

      /* Now consider each instruction that uses the converted 16-bit value.
       * Simply modify these instructions to use the original sample value
       * instead (which now has 16-bit precision itself).
       */
      nir_foreach_use_safe(f2f16_use, &f2f16->dest.dest.ssa) {
         nir_instr_rewrite_src(f2f16_use->parent_instr, f2f16_use,
                               f2f16->src[0].src);

         if (f2f16_use->parent_instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *store =
               nir_instr_as_intrinsic(f2f16_use->parent_instr);
            nir_src_copy(&store->src[1], &f2f16->src[0].src, &store->instr);
         }
      }
   }

   return true;
}

bool
nir_lower_sample_precision(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);
          
      nir_foreach_block(block, function->impl) {
         nir_foreach_instr_safe(instr, block) {
            b.cursor = nir_before_instr(instr);

            if (instr->type != nir_instr_type_tex)
               continue;

            progress |= lower_sample_precision(&b, nir_instr_as_tex(instr));
         }
      }

      foreach_list_typed_safe(nir_cf_node, node, node, &function->impl->body) {
         if (node->type != nir_cf_node_if)
            continue;

         nir_if *if_stmt = nir_cf_node_as_if(node);
         nir_src *cond = &if_stmt->condition;

         assert(cond->is_ssa);
         if (cond->ssa->bit_size > 16)
            continue;

         /* Validation wants source for nir_if to be 32-bit as it is boolean.
          * Just like before, emit a conversion that keeps validation happy
          * but such that can be removed later on.
          */ 
         const nir_src promoted = nir_src_for_ssa(
            insert_conversion_after_src(shader, cond, nir_op_i2i32));
         nir_if_rewrite_condition(if_stmt, promoted);
      }
   }

   return progress;
}
