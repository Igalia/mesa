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

static void
lower_instr_precision(nir_function_impl *impl)
{
   nir_builder b;
   nir_builder_init(&b, impl);
       
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         b.cursor = nir_before_instr(instr);

         switch(instr->type) {
         case nir_instr_type_deref:
            lower_deref_precision(nir_instr_as_deref(instr));
            break;
         default:
            break;
         }
      }
   }
}

bool
nir_lower_precision(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      lower_instr_precision(function->impl);
   }

   return progress;
}
