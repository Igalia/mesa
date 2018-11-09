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

      struct gl_shader_variable *sh_var =
         rzalloc(prog, struct gl_shader_variable);

      /* In the ARB_gl_spirv spec, names are considered optional debug info, so
       * the linker needs to work without them. Returning them is optional.
       * For simplicity, we ignore names.
       */
      sh_var->name = NULL;
      sh_var->type = var->type;
      sh_var->location = var->data.location;

      if (!link_util_add_program_resource(prog, resource_set,
                                          programInterface,
                                          sh_var, 1 << stage)) {
         return false;
      }
   }

   return true;
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
   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      struct gl_uniform_storage *uniform = &prog->data->UniformStorage[i];

      /* Do not add uniforms internally used by Mesa. */
      if (uniform->hidden)
         continue;

      /* FIXME: ubo and ssbo resource count is different. Here is missing a
       * equivalent to should_add_buffer_variable (linker.cpp). For the
       * moment we are adding all of them.
       */
      bool is_shader_storage =
         prog->data->UniformStorage[i].is_shader_storage;

      GLenum interface = is_shader_storage ? GL_BUFFER_VARIABLE : GL_UNIFORM;
      if (!link_util_add_program_resource(prog, resource_set, interface, uniform,
                                          uniform->active_shader_mask)) {
         return;
      }
   }


   /* Add program uniform blocks. */
   for (unsigned i = 0; i < prog->data->NumUniformBlocks; i++) {
      if (!link_util_add_program_resource(prog, resource_set, GL_UNIFORM_BLOCK,
                                          &prog->data->UniformBlocks[i], 0))
         return;
   }

   /* Add program shader storage blocks. */
   for (unsigned i = 0; i < prog->data->NumShaderStorageBlocks; i++) {
      if (!link_util_add_program_resource(prog, resource_set, GL_SHADER_STORAGE_BLOCK,
                                          &prog->data->ShaderStorageBlocks[i], 0))
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
