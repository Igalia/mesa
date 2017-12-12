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

#include "nir.h"
#include "gl_nir_linker.h"
#include "ir_uniform.h" /* for gl_uniform_storage */
#include "linker_util.h"
#include "main/context.h"

/* This file do the common link for GLSL transform feedback, using NIR,
 * instead of IR as the counter-part glsl/link_varyings.cpp
 *
 * Also note that this is tailored for ARB_gl_spirv needs and particularities
 */

struct active_xfb_varying {
   nir_variable *var;
};

struct active_xfb_buffer {
   GLuint stride;
   GLuint num_varyings;
};

struct active_xfb_varying_array {
   unsigned num_varyings;
   unsigned num_outputs;
   unsigned buffer_size;
   struct active_xfb_varying *varyings;
   struct active_xfb_buffer buffers[MAX_FEEDBACK_BUFFERS];
};

static unsigned
get_num_outputs(nir_variable *var)
{
   return glsl_count_attribute_slots(var->type,
                                     false /* is_vertex_input */);
}

static void
add_xfb_varying(struct active_xfb_varying_array *array,
                nir_variable *var)
{
   if (array->num_varyings >= array->buffer_size) {
      if (array->buffer_size == 0)
         array->buffer_size = 1;
      else
         array->buffer_size *= 2;

      array->varyings = realloc(array->varyings,
                                sizeof(*array->varyings) *
                                array->buffer_size);
   }

   array->varyings[array->num_varyings].var = var;
   array->num_varyings++;

   array->num_outputs += get_num_outputs(var);
}

static int
cmp_xfb_offset(const void *x_generic, const void *y_generic)
{
   const struct active_xfb_varying *x = x_generic;
   const struct active_xfb_varying *y = y_generic;

   if (x->var->data.xfb_buffer != y->var->data.xfb_buffer)
      return x->var->data.xfb_buffer - y->var->data.xfb_buffer;
   return x->var->data.offset - y->var->data.offset;
}

static void
get_active_xfb_varyings(struct gl_shader_program *prog,
                        struct active_xfb_varying_array *array)
{
   for (unsigned i = 0; i < MESA_SHADER_STAGES; ++i) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[i];
      if (sh == NULL)
         continue;

      nir_shader *nir = sh->Program->nir;

      nir_foreach_variable(var, &nir->outputs) {
         if (var->data.explicit_xfb_buffer &&
             var->data.explicit_xfb_stride &&
             var->data.xfb_buffer < MAX_FEEDBACK_BUFFERS) {
            array->buffers[var->data.xfb_buffer].stride =
               var->data.xfb_stride;
         }

         if (!var->data.explicit_xfb_buffer ||
             !var->data.explicit_offset)
            continue;

         array->buffers[var->data.xfb_buffer].num_varyings++;

         add_xfb_varying(array, var);
      }
   }

   qsort(array->varyings,
         array->num_varyings,
         sizeof(*array->varyings),
         cmp_xfb_offset);
}

static unsigned
add_varying_outputs(nir_variable *var,
                    const struct glsl_type *type,
                    unsigned location_offset,
                    unsigned dest_offset,
                    struct gl_transform_feedback_output *output)
{
   unsigned num_outputs = 0;

   if (glsl_type_is_array(type) || glsl_type_is_matrix(type)) {
      unsigned length = glsl_get_length(type);
      const struct glsl_type *child_type = glsl_get_array_element(type);
      unsigned component_slots = glsl_get_component_slots(child_type);

      for (unsigned i = 0; i < length; i++) {
         unsigned child_outputs = add_varying_outputs(var,
                                                      child_type,
                                                      location_offset,
                                                      dest_offset,
                                                      output + num_outputs);
         num_outputs += child_outputs;
         location_offset += child_outputs;
         dest_offset += component_slots;
      }
   } else if (glsl_type_is_struct(type)) {
      unsigned length = glsl_get_length(type);
      for (unsigned i = 0; i < length; i++) {
         const struct glsl_type *child_type = glsl_get_struct_field(type, i);
         unsigned child_outputs = add_varying_outputs(var,
                                                      child_type,
                                                      location_offset,
                                                      dest_offset,
                                                      output + num_outputs);
         num_outputs += child_outputs;
         location_offset += child_outputs;
         dest_offset += glsl_get_component_slots(child_type);
      }
   } else {
      unsigned location = var->data.location + location_offset;
      unsigned location_frac = var->data.location_frac;
      unsigned num_components = glsl_get_component_slots(type);

      while (num_components > 0) {
         unsigned output_size = MIN2(num_components, 4 - location_frac);

         output->OutputRegister = location;
         output->OutputBuffer = var->data.xfb_buffer;
         output->NumComponents = output_size;
         output->StreamId = var->data.stream;
         output->DstOffset = var->data.offset / 4 + dest_offset;
         output->ComponentOffset = location_frac;

         dest_offset += output_size;
         num_components -= output_size;
         num_outputs++;
         output++;
         location++;
         location_frac = 0;
      }
   }

   return num_outputs;
}

void
gl_nir_link_assign_xfb_resources(struct gl_context *ctx,
                                 struct gl_shader_program *prog)
{
   /* This is intended to work with SPIR-V shaders so it makes the following
    * assumptions provided by the GL spec:
    *
    * - All captured varyings have both an explicit buffer and offset. That
    *   means that no calculation of the offset is necessary.
    * - All buffers will have at least one captured varying with an explicit
    *   stride so there is no need to calculate it.
    */

   struct gl_program *xfb_prog = prog->last_vert_prog;

   if (xfb_prog == NULL)
      return;

   /* free existing varyings, if any */
   for (unsigned i = 0; i < prog->TransformFeedback.NumVarying; i++)
      free(prog->TransformFeedback.VaryingNames[i]);
   free(prog->TransformFeedback.VaryingNames);

   struct active_xfb_varying_array array = { 0 };

   get_active_xfb_varyings(prog, &array);

   for (unsigned buf = 0; buf < MAX_FEEDBACK_BUFFERS; buf++)
      prog->TransformFeedback.BufferStride[buf] = array.buffers[buf].stride;
   prog->TransformFeedback.BufferMode = GL_INTERLEAVED_ATTRIBS;

   prog->TransformFeedback.NumVarying = array.num_varyings;
   prog->TransformFeedback.VaryingNames =
      malloc(sizeof(GLchar *) * array.num_varyings);

   struct gl_transform_feedback_info *linked_xfb =
      rzalloc(xfb_prog, struct gl_transform_feedback_info);
   xfb_prog->sh.LinkedTransformFeedback = linked_xfb;

   linked_xfb->Outputs =
      rzalloc_array(xfb_prog,
                    struct gl_transform_feedback_output,
                    array.num_outputs);
   linked_xfb->NumOutputs = array.num_outputs;

   linked_xfb->Varyings =
      rzalloc_array(xfb_prog,
                    struct gl_transform_feedback_varying_info,
                    array.num_varyings);
   linked_xfb->NumVarying = array.num_varyings;

   for (unsigned i = 0, output_pos = 0; i < array.num_varyings; i++) {
      struct nir_variable *var = array.varyings[i].var;

      /* ARB_gl_spirv: names are considered optional debug info, so the linker
       * needs to work without them, and returning them is optional. For
       * simplicity we ignore names.
       */
      prog->TransformFeedback.VaryingNames[i] = NULL;

      struct gl_transform_feedback_output *output =
         linked_xfb->Outputs + output_pos;
      unsigned varying_outputs = add_varying_outputs(var,
                                                     var->type,
                                                     0, /* location_offset */
                                                     0, /* dest_offset */
                                                     output);
      assert(varying_outputs == get_num_outputs(var));
      output_pos += varying_outputs;

      struct gl_transform_feedback_varying_info *varying =
         linked_xfb->Varyings + i;

      /* ARB_gl_spirv: see above. */
      varying->Name = NULL;
      varying->Type = glsl_get_gl_type(var->type);
      varying->BufferIndex = var->data.xfb_buffer;
      varying->Size = glsl_get_length(var->type);
      varying->Offset = var->data.offset;
   }

   /* Make sure MaxTransformFeedbackBuffers is <= 32 so the bitmask for
    * tracking the number of buffers doesn't overflow.
    */
   unsigned buffers = 0;
   assert(ctx->Const.MaxTransformFeedbackBuffers <= sizeof(buffers) * 8);

   for (unsigned buf = 0; buf < MAX_FEEDBACK_BUFFERS; buf++) {
      if (array.buffers[buf].stride > 0) {
         linked_xfb->Buffers[buf].Stride = array.buffers[buf].stride / 4;
         linked_xfb->Buffers[buf].NumVaryings = array.buffers[buf].num_varyings;
         buffers |= 1 << buf;
      }
   }

   linked_xfb->ActiveBuffers = buffers;

   free(array.varyings);
}
