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
#include "linker_util.h"
#include "main/mtypes.h"
#include "ir_uniform.h" /* for gl_uniform_storage */

/* This file included general link methods, using NIR, instead of IR as
 * the counter-part glsl/linker.cpp
 *
 * Also note that this is tailored for ARB_gl_spirv needs and particularities
 */

bool
_glsl_type_is_leaf(const struct glsl_type *type)
{
   if (glsl_type_is_struct(type) ||
       (glsl_type_is_array(type) &&
        (glsl_type_is_array(glsl_get_array_element(type)) ||
         glsl_type_is_struct(glsl_get_array_element(type))))) {
      return false;
   } else {
      return true;
   }
}

unsigned
_get_type_size(const struct glsl_type *type,
               const struct glsl_type *parent_type,
               unsigned int index_in_parent)
{
   /* If the type is a struct then the members are supposed to presented in
    * increasing order of offset so we can just look at the last member.
    */
   if (glsl_type_is_struct(type)) {
      unsigned length = glsl_get_length(type);
      if (length > 0) {
         return (glsl_get_struct_field_offset(type, length - 1) +
                 _get_type_size(glsl_get_struct_field(type, length - 1),
                                type, length - 1));
      } else {
         return 0;
      }
   }

   /* Arrays must have an array stride */
   if (glsl_type_is_array(type)) {
      unsigned length =
         glsl_type_is_unsized_array(type) ? 1 : glsl_get_length(type);
      return glsl_get_explicit_array_stride(type) * length;
   }

   /* Matrices must have a matrix stride and either RowMajor or ColMajor */
   if (glsl_type_is_matrix(type)) {
      unsigned matrix_stride =
         glsl_get_struct_field_explicit_matrix_stride(parent_type, index_in_parent);

      bool row_major =
         glsl_get_struct_field_matrix_layout(parent_type, index_in_parent) ==
         GLSL_MATRIX_LAYOUT_ROW_MAJOR;

      unsigned length = row_major ? glsl_get_vector_elements(type)
         : glsl_get_length(type);

      /* We don't really need to compute the type_size of the matrix element
       * type. That should be already included as part of matrix_stride
       */
      return matrix_stride * length;
   }

   unsigned N = glsl_type_is_64bit(type) ? 8 : 4;

   return glsl_get_vector_elements(type) * N;
}

static bool
add_interface_variables(const struct gl_context *cts,
                        struct gl_shader_program *prog,
                        struct set *resource_set,
                        unsigned stage, GLenum programInterface)
{
   const struct exec_list *var_list = NULL;

   struct gl_linked_shader *sh = prog->_LinkedShaders[stage];
   if (!sh)
      return true;

   nir_shader *nir = sh->Program->nir;
   assert(nir);

   switch (programInterface) {
   case GL_PROGRAM_INPUT:
      var_list = &nir->inputs;
      break;
   case GL_PROGRAM_OUTPUT:
      var_list = &nir->outputs;
      break;
   default:
      assert("!Should not get here");
      break;
   }

   nir_foreach_variable(var, var_list) {
      if (var->data.how_declared == nir_var_hidden)
         continue;

      int loc_bias = 0;
      switch(var->data.mode) {
      case nir_var_system_value:
      case nir_var_shader_in:
         if (programInterface != GL_PROGRAM_INPUT)
            continue;
         loc_bias = (stage == MESA_SHADER_VERTEX) ? VERT_ATTRIB_GENERIC0
                                                  : VARYING_SLOT_VAR0;
         break;
      case nir_var_shader_out:
         if (programInterface != GL_PROGRAM_OUTPUT)
            continue;
         loc_bias = (stage == MESA_SHADER_FRAGMENT) ? FRAG_RESULT_DATA0
                                                    : VARYING_SLOT_VAR0;
         break;
      default:
         continue;
      }

      if (var->data.patch)
         loc_bias = VARYING_SLOT_PATCH0;

      /* @FIXME: in the GLSL linker, there is some code to skip packed varyings
       * and fragdata arrays identifying them by name.
       * For the moment, we are not adding them to the program resource list, so
       * there is nothing to do here. Check if we have to add them.
       */
      struct gl_shader_variable *sh_var =
         rzalloc(prog, struct gl_shader_variable);

      /* In the ARB_gl_spirv spec, names are considered optional debug info, so
       * the linker needs to work without them. Returning them is optional.
       * For simplicity, we ignore names.
       */
      sh_var->name = NULL;
      sh_var->type = var->type;
      sh_var->location = var->data.location - loc_bias;
      sh_var->index = var->data.index;

      /* @TODO: Fill in the rest of gl_shader_variable data. */
      /* @FIXME: manage arrays, structs, etc. in other words, implement the
       * equivalent to add_shader_variable.
       */
      if (!link_util_add_program_resource(prog, resource_set,
                                          programInterface,
                                          sh_var, 1 << stage)) {
         return false;
      }
   }

   return true;
}

/* From the OpenGL 4.6 specification, 7.3.1.1 Naming Active Resources:
 *
 * For an active shader storage block member declared as an array of an
 * aggregate type, an entry will be generated only for the first array
 * element, regardless of its type. Such block members are referred to
 * as top-level arrays. If the block member is an aggregate type, the
 * enumeration rules are then applied recursively.
 */
static bool
should_add_buffer_variable(struct gl_shader_program *prog,
                           struct gl_uniform_storage *uniform,
                           int *top_level_array_base_offset,
                           int *top_level_array_size_in_bytes,
                           int *block_index)
{
   /* FIXME: take into account stages ? will be the block index equal for different stages? */
   /* The uniform is not in an SSBO or it is not part of a top-level array. */
   if (!uniform->is_shader_storage || uniform->top_level_array_size <= 1)
      return true;

   /* New top-level array, initialize its base offset. */
   if (*top_level_array_base_offset == -1 ||
       *block_index != uniform->block_index ||
       uniform->offset >=
       (*top_level_array_base_offset + *top_level_array_size_in_bytes)) {
      *top_level_array_base_offset = uniform->offset;
      *top_level_array_size_in_bytes =
         uniform->top_level_array_size * uniform->top_level_array_stride;
      *block_index = uniform->block_index;
   }

   if (uniform->offset >= *top_level_array_base_offset &&
       uniform->offset < (*top_level_array_base_offset + uniform->top_level_array_stride)) {
      /* The uniform is a member of the first array element of the top-level array.
       * It should be added to the resource list.
       */
      return true;
   }

   return false;
}

void
nir_build_program_resource_list(struct gl_context *ctx,
                                struct gl_shader_program *prog)
{
   /* Rebuild resource list. */
   if (prog->data->ProgramResourceList) {
      ralloc_free(prog->data->ProgramResourceList);
      prog->data->ProgramResourceList = NULL;
      prog->data->NumProgramResourceList = 0;
   }

   int input_stage = MESA_SHADER_STAGES, output_stage = 0;

   /* Determine first input and final output stage. These are used to
    * detect which variables should be enumerated in the resource list
    * for GL_PROGRAM_INPUT and GL_PROGRAM_OUTPUT.
    */
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      if (!prog->_LinkedShaders[i])
         continue;
      if (input_stage == MESA_SHADER_STAGES)
         input_stage = i;
      output_stage = i;
   }

   /* Empty shader, no resources. */
   if (input_stage == MESA_SHADER_STAGES && output_stage == 0)
      return;

   struct set *resource_set = _mesa_set_create(NULL,
                                               _mesa_hash_pointer,
                                               _mesa_key_pointer_equal);

   /* Add inputs and outputs to the resource list. */
   if (!add_interface_variables(ctx, prog, resource_set, input_stage,
                                GL_PROGRAM_INPUT))
      return;

   if (!add_interface_variables(ctx, prog, resource_set, output_stage,
                                GL_PROGRAM_OUTPUT))
      return;

   /* Add transform feedback varyings and buffers. */
   if (prog->last_vert_prog) {
      struct gl_transform_feedback_info *linked_xfb =
         prog->last_vert_prog->sh.LinkedTransformFeedback;

      /* Add varyings. */
      if (linked_xfb->NumVarying > 0) {
         for (int i = 0; i < linked_xfb->NumVarying; i++) {
            if (!link_util_add_program_resource(prog, resource_set,
                                                GL_TRANSFORM_FEEDBACK_VARYING,
                                                &linked_xfb->Varyings[i], 0))
            return;
         }
      }

      /* Add buffers. */
      for (unsigned i = 0; i < ctx->Const.MaxTransformFeedbackBuffers; i++) {
         if ((linked_xfb->ActiveBuffers >> i) & 1) {
            linked_xfb->Buffers[i].Binding = i;
            if (!link_util_add_program_resource(prog, resource_set,
                                                GL_TRANSFORM_FEEDBACK_BUFFER,
                                                &linked_xfb->Buffers[i], 0))
            return;
         }
      }
   }

   /* Add uniforms
    *
    * Here, it is expected that nir_link_uniforms() has already been
    * called, so that UniformStorage table is already available.
    */
   int base_offset = -1;
   int block_index = -1;
   int size = -1;
   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      struct gl_uniform_storage *uniform = &prog->data->UniformStorage[i];

      /* Do not add uniforms internally used by Mesa. */
      if (uniform->hidden)
         continue;

      if (!should_add_buffer_variable(prog, uniform, &base_offset, &size, &block_index))
         continue;

      GLenum interface = uniform->is_shader_storage ? GL_BUFFER_VARIABLE : GL_UNIFORM;
      if (!link_util_add_program_resource(prog, resource_set, interface, uniform,
                                          uniform->active_shader_mask)) {
         return;
      }
   }


   /* Add program uniform blocks. */
   for (unsigned i = 0; i < prog->data->NumUniformBlocks; i++) {
      if (!link_util_add_program_resource(prog, resource_set, GL_UNIFORM_BLOCK,
                                          &prog->data->UniformBlocks[i],
                                          prog->data->UniformBlocks[i].stageref))
         return;
   }

   /* Add program shader storage blocks. */
   for (unsigned i = 0; i < prog->data->NumShaderStorageBlocks; i++) {
      if (!link_util_add_program_resource(prog, resource_set, GL_SHADER_STORAGE_BLOCK,
                                          &prog->data->ShaderStorageBlocks[i],
                                          prog->data->ShaderStorageBlocks[i].stageref))
         return;
   }

   /* Add atomic counter buffers. */
   for (unsigned i = 0; i < prog->data->NumAtomicBuffers; i++) {
      if (!link_util_add_program_resource(prog, resource_set, GL_ATOMIC_COUNTER_BUFFER,
                                          &prog->data->AtomicBuffers[i], 0))
         return;
   }

   _mesa_set_destroy(resource_set, NULL);
}
