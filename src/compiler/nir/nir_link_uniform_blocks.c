/*
 * Copyright © 2017 Intel Corporation
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
#include "nir_linker.h"
#include "compiler/glsl/ir_uniform.h" /* for gl_uniform_storage */
#include "main/shaderobj.h" /* _mesa_delete_linked_shader */

/* This file contains code to do a nir-based linking for uniform blocks. Note
 * that it is tailored to ARB_gl_spirv needs. As the uniform block name,
 * fields names, and other names could not be present, the linking would need
 * to ignore it.
 */


static bool
link_uniform_blocks_are_compatible(const struct gl_uniform_block *a,
                                   const struct gl_uniform_block *b)
{
   assert(strcmp(a->Name, b->Name) == 0);

   /* Page 35 (page 42 of the PDF) in section 4.3.7 of the GLSL 1.50 spec says:
    *
    *    Matched block names within an interface (as defined above) must match
    *    in terms of having the same number of declarations with the same
    *    sequence of types and the same sequence of member names, as well as
    *    having the same member-wise layout qualification....if a matching
    *    block is declared as an array, then the array sizes must also
    *    match... Any mismatch will generate a link error.
    *
    * Arrays are not yet supported, so there is no check for that.
    */
   if (a->NumUniforms != b->NumUniforms)
      return false;

   if (a->_Packing != b->_Packing)
      return false;

   if (a->_RowMajor != b->_RowMajor)
      return false;

   if (a->Binding != b->Binding)
      return false;

   for (unsigned i = 0; i < a->NumUniforms; i++) {
      if (strcmp(a->Uniforms[i].Name, b->Uniforms[i].Name) != 0)
         return false;

      if (a->Uniforms[i].Type != b->Uniforms[i].Type)
         return false;

      if (a->Uniforms[i].RowMajor != b->Uniforms[i].RowMajor)
         return false;
   }

   return true;
}

/* FIXME: this method is name-based. Need an alternative */
/**
 * Merges a uniform block into an array of uniform blocks that may or
 * may not already contain a copy of it.
 *
 * Returns the index of the new block in the array.
 */
static int
_link_cross_validate_uniform_block(void *mem_ctx,
                                   struct gl_uniform_block **linked_blocks,
                                   unsigned int *num_linked_blocks,
                                   struct gl_uniform_block *new_block)
{
   for (unsigned int i = 0; i < *num_linked_blocks; i++) {
      struct gl_uniform_block *old_block = &(*linked_blocks)[i];

      if (strcmp(old_block->Name, new_block->Name) == 0)
         return link_uniform_blocks_are_compatible(old_block, new_block)
            ? i : -1;
   }

   *linked_blocks = reralloc(mem_ctx, *linked_blocks,
                             struct gl_uniform_block,
                             *num_linked_blocks + 1);
   int linked_block_index = (*num_linked_blocks)++;
   struct gl_uniform_block *linked_block = &(*linked_blocks)[linked_block_index];

   memcpy(linked_block, new_block, sizeof(*new_block));
   linked_block->Uniforms = ralloc_array(*linked_blocks,
                                         struct gl_uniform_buffer_variable,
                                         linked_block->NumUniforms);

   memcpy(linked_block->Uniforms,
          new_block->Uniforms,
          sizeof(*linked_block->Uniforms) * linked_block->NumUniforms);

   linked_block->Name = ralloc_strdup(*linked_blocks, linked_block->Name);

   for (unsigned int i = 0; i < linked_block->NumUniforms; i++) {
      struct gl_uniform_buffer_variable *ubo_var =
         &linked_block->Uniforms[i];

      if (ubo_var->Name == ubo_var->IndexName) {
         ubo_var->Name = ralloc_strdup(*linked_blocks, ubo_var->Name);
         ubo_var->IndexName = ubo_var->Name;
      } else {
         ubo_var->Name = ralloc_strdup(*linked_blocks, ubo_var->Name);
         ubo_var->IndexName = ralloc_strdup(*linked_blocks, ubo_var->IndexName);
      }
   }

   return linked_block_index;
}


/**
 * Accumulates the array of buffer blocks and checks that all definitions of
 * blocks agree on their contents.
 *
 * FIXME: as it calls _link_cross_validate_uniform_block, it is inneherently
 * name-based. Needs to be ported.
 */
static bool
_nir_interstage_cross_validate_uniform_blocks(struct gl_shader_program *prog,
                                              bool validate_ssbo)
{
   int *InterfaceBlockStageIndex[MESA_SHADER_STAGES];
   struct gl_uniform_block *blks = NULL;
   unsigned *num_blks = validate_ssbo ? &prog->data->NumShaderStorageBlocks :
      &prog->data->NumUniformBlocks;

   unsigned max_num_buffer_blocks = 0;
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      if (prog->_LinkedShaders[i]) {
         if (validate_ssbo) {
            max_num_buffer_blocks +=
               prog->_LinkedShaders[i]->Program->info.num_ssbos;
         } else {
            max_num_buffer_blocks +=
               prog->_LinkedShaders[i]->Program->info.num_ubos;
         }
      }
   }

   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[i];

      InterfaceBlockStageIndex[i] = malloc(max_num_buffer_blocks * sizeof(int));
      for (unsigned int j = 0; j < max_num_buffer_blocks; j++)
         InterfaceBlockStageIndex[i][j] = -1;

      if (sh == NULL)
         continue;

      unsigned sh_num_blocks;
      struct gl_uniform_block **sh_blks;
      if (validate_ssbo) {
         sh_num_blocks = prog->_LinkedShaders[i]->Program->info.num_ssbos;
         sh_blks = sh->Program->sh.ShaderStorageBlocks;
      } else {
         sh_num_blocks = prog->_LinkedShaders[i]->Program->info.num_ubos;
         sh_blks = sh->Program->sh.UniformBlocks;
      }

      for (unsigned int j = 0; j < sh_num_blocks; j++) {
         int index = _link_cross_validate_uniform_block(prog->data, &blks,
                                                        num_blks, sh_blks[j]);

         if (index == -1) {
            nir_linker_error(prog, "buffer block `%s' has mismatching "
                             "definitions\n", sh_blks[j]->Name);

            for (unsigned k = 0; k <= i; k++) {
               free(InterfaceBlockStageIndex[k]);
            }

            /* Reset the block count. This will help avoid various segfaults
             * from api calls that assume the array exists due to the count
             * being non-zero.
             */
            *num_blks = 0;
            return false;
         }

         InterfaceBlockStageIndex[i][index] = j;
      }
   }

   /* Update per stage block pointers to point to the program list.
    * FIXME: We should be able to free the per stage blocks here.
    */
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      for (unsigned j = 0; j < *num_blks; j++) {
         int stage_index = InterfaceBlockStageIndex[i][j];

         if (stage_index != -1) {
            struct gl_linked_shader *sh = prog->_LinkedShaders[i];

            struct gl_uniform_block **sh_blks = validate_ssbo ?
               sh->Program->sh.ShaderStorageBlocks :
               sh->Program->sh.UniformBlocks;

            blks[j].stageref |= sh_blks[stage_index]->stageref;
            sh_blks[stage_index] = &blks[j];
         }
      }
   }

   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      free(InterfaceBlockStageIndex[i]);
   }

   if (validate_ssbo)
      prog->data->ShaderStorageBlocks = blks;
   else
      prog->data->UniformBlocks = blks;

   return true;
}

static bool
_var_is_uniform_block(nir_variable *var)
{
   return (var->data.mode == nir_var_uniform &&
           var->interface_type != NULL);
}

/*
 * Equivalent to glsl create_buffer_blocks
 * FIXME: block_hask?
 */
static void
_create_buffer_blocks(void *mem_ctx, struct gl_context *ctx,
                      struct gl_shader_program *prog,
                      struct gl_uniform_block **out_blks, unsigned num_blocks,
                      struct hash_table *block_hash, unsigned num_variables,
                      bool create_ubo_blocks)
{
   /* FIXME: fill me */
}

/*
 * Link uniform blocks for a given linked_shader/stage.
 */
static void
_link_linked_shader_uniform_blocks(void *mem_ctx,
                                   struct gl_context *ctx,
                                   struct gl_shader_program *prog,
                                   struct gl_linked_shader *shader,
                                   struct gl_uniform_block **ubo_blocks,
                                   unsigned *num_ubo_blocks)
{
   /* FIXME: note: one of the reasons we can't use directly the glsl method is
    * that it tracks the uniform blocks with a hash table that uses block-name
    * as id. With gl_spirv we could find shaders without name. If we want to
    * reuse it we would need to use a different id
    */

   /* FIXME: Determine which uniform blocks are active. For now we take them all */

   /* FIXME: Count the number of active uniform blocks. Count the total number
    * of active slots in those uniform blocks
    */
   unsigned num_ubo_variables = 0; /* FIXME: we are not counting the ubo variable yet */
   nir_foreach_variable(var, &shader->Program->nir->uniforms) {
      if (_var_is_uniform_block(var)) {
         /* FIXME: we don't take into account AOA */
         (*num_ubo_blocks)++;
      }
   }

   /* FIXME: create and fill buffer blocks. Pending filling it. */
   struct gl_uniform_block *blocks = rzalloc_array(mem_ctx, struct gl_uniform_block, *num_ubo_blocks);

   *ubo_blocks = blocks;

   /* FIXME: pending: create and fill buffer blocks variables */
   _create_buffer_blocks(mem_ctx, ctx, prog, ubo_blocks, *num_ubo_blocks,
                         NULL, num_ubo_variables, true);
}

bool
nir_link_uniform_blocks(struct gl_context *ctx,
                        struct gl_shader_program *prog)
{
   void *mem_ctx = ralloc_context(NULL); // temporary linker context

   for (int stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      struct gl_linked_shader *const linked = prog->_LinkedShaders[stage];
      struct gl_uniform_block *ubo_blocks = NULL;
      unsigned num_ubo_blocks = 0;

      if (!linked)
         continue;

      _link_linked_shader_uniform_blocks(mem_ctx, ctx, prog, linked,
                                         &ubo_blocks, &num_ubo_blocks);

      if (!prog->data->LinkStatus) {
         if (linked)
            _mesa_delete_linked_shader(ctx, linked);

         return false;
      }

      prog->_LinkedShaders[stage] = linked;
      prog->data->linked_stages |= 1 << stage;

      /* Copy ubo blocks to linked shader list */
      linked->Program->sh.UniformBlocks =
         ralloc_array(linked, struct gl_uniform_block *, num_ubo_blocks);
      ralloc_steal(linked, ubo_blocks);
      for (unsigned i = 0; i < num_ubo_blocks; i++) {
         linked->Program->sh.UniformBlocks[i] = &ubo_blocks[i];
      }
      linked->Program->info.num_ubos = num_ubo_blocks;
   }

   /* FIXME: check for cache_fallback ? */
   if (!_nir_interstage_cross_validate_uniform_blocks(prog, FALSE)) {
      return false;
   }

   return true;
}
