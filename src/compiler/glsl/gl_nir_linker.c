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

   /* @FIXME: Program interface needs to expose varyings in case of SSO. */

   /* @FIXME: Do we have to do add_fragdata_arrays()? */

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

   _mesa_set_destroy(resource_set, NULL);
}
