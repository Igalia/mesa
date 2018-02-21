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
#include "gl_nir_linker.h"
#include "ir_uniform.h" /* for gl_uniform_storage */
#include "linker_util.h"
#include "main/shaderobj.h" /* _mesa_delete_linked_shader */
#include "main/mtypes.h"

/* Summary: This file contains code to do a nir-based linking for uniform
 * blocks. This includes ubos and ssbos.
 *
 * More details:
 *
 * 1. Note that it is tailored to ARB_gl_spirv needs. Uniform block name,
 * fields names, and other names are considered optional debug infor so could
 * not be present. So the linking should work without it, and it is optional
 * to not handle them at all. From ARB_gl_spirv:
 *
 *    "19. How should the program interface query operations behave for program
 *         objects created from SPIR-V shaders?
 *
 *     DISCUSSION: we previously said we didn't need reflection to work for
 *     SPIR-V shaders (at least for the first version), however we are left
 *     with specifying how it should "not work". The primary issue is that
 *     SPIR-V binaries are not required to have names associated with
 *     variables. They can be associated in debug information, but there is no
 *     requirement for that to be present, and it should not be relied upon.
 *
 *     Options:
 *
 *     <skip>
 *
 *    C) Allow as much as possible to work "naturally". You can query for the
 *    number of active resources, and for details about them. Anything that
 *    doesn't query by name will work as expected. Queries for maximum length
 *    of names return one. Queries for anything "by name" return INVALID_INDEX
 *    (or -1). Querying the name property of a resource returns an empty
 *    string. This may allow many queries to work, but it's not clear how
 *    useful it would be if you can't actually know which specific variable
 *    you are retrieving information on. If everything is specified a-priori
 *    by location/binding/offset/index/component in the shader, this may be
 *    sufficient.
 *
 *  RESOLVED.  Pick (c), but also allow debug names to be returned if an
 *  implementation wants to."
 *
 * This implemention doesn't care for the names, as the main objective is
 * functional, and not support optional debug features.
 *
 * 2. As mentioned, the code on this file handles both ubo and ssbo. In some
 * terminology they are called "buffer-backed blocks", and don't consider ssbo
 * as "real uniforms". And for example, on nir, the mode for ubos are
 * nir_var_uniform but for ssbo are nir_var_shader_storage.
 *
 * But from ARB_gl_spirv spec:
 *   "Mapping of Storage Classes:
 *     <skip>
 *     uniform blockN { ... } ...;  -> Uniform, with Block decoration
 *     <skip>
 *     buffer  blockN { ... } ...;  -> Uniform, with BufferBlock decoration"
 *
 * Additionally, the GLSL (IR) path is already handling and calling them
 * uniform blocks (ie: struct gl_uniform_block can be a individual ubo or
 * ssbo), so for consistency we are doing the same here.
 */

/*
 * As we reuse some methods for ubos and ssbos, it is good to mark what we are
 * handling at each moment.
 */
enum block_type {
   BLOCK_UBO,
   BLOCK_SSBO
};

static bool
link_blocks_are_compatible(const struct gl_uniform_block *a,
                           const struct gl_uniform_block *b)
{
   /*
    * Names on ARB_gl_spirv are optional, so we are ignoring them. So
    * meanwhile on the equivalent GLSL method the matching is done using the
    * name, here we use the binding, that for SPIR-V binaries should be
    * explicit. FIXME: spec quote, still missing, see
    * https://gitlab.khronos.org/opengl/API/issues/55
    */
   if (a->Binding != b->Binding)
      return false;

   /* We are explicitly ignoring the names, so it would be good to check that
    * this is happening. TODO: But perhaps this is not the best place for the
    * assert */
   assert(a->Name == NULL);
   assert(b->Name == NULL);

   if (a->NumUniforms != b->NumUniforms)
      return false;

   if (a->_Packing != b->_Packing)
      return false;

   if (a->_RowMajor != b->_RowMajor)
      return false;

   for (unsigned i = 0; i < a->NumUniforms; i++) {
      if (a->Uniforms[i].Type != b->Uniforms[i].Type)
         return false;

      if (a->Uniforms[i].RowMajor != b->Uniforms[i].RowMajor)
         return false;

      /* See comment on previous assert */
      assert(a->Uniforms[i].Name == NULL);
      assert(b->Uniforms[i].Name == NULL);
   }

   return true;
}

/**
 * Merges a buffer block into an array of buffer blocks that may or may not
 * already contain a copy of it.
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

      if (old_block->Binding == new_block->Binding)
         return link_blocks_are_compatible(old_block, new_block) ? i : -1;
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

   return linked_block_index;
}


/**
 * Accumulates the array of buffer blocks and checks that all definitions of
 * blocks agree on their contents.
 *
 * TODO: right now this is really similar to its GLSL counter-part, but
 * calling our binding-based (instead of name-based)
 * _link_cross_validate_uniform_block and some C++ cleaning. Candidate for
 * refactoring.
 */
static bool
_nir_interstage_cross_validate_uniform_blocks(struct gl_shader_program *prog,
                                              enum block_type block_type)
{
   int *InterfaceBlockStageIndex[MESA_SHADER_STAGES];
   struct gl_uniform_block *blks = NULL;
   unsigned *num_blks = block_type == BLOCK_SSBO ? &prog->data->NumShaderStorageBlocks :
      &prog->data->NumUniformBlocks;

   unsigned max_num_buffer_blocks = 0;
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      if (prog->_LinkedShaders[i]) {
         if (block_type == BLOCK_SSBO) {
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
      if (block_type == BLOCK_SSBO) {
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
            /* We use the binding as we are ignoring the names */
            linker_error(prog, "buffer block with binding `%i' has mismatching "
                         "definitions\n", sh_blks[j]->Binding);

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
    */
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      for (unsigned j = 0; j < *num_blks; j++) {
         int stage_index = InterfaceBlockStageIndex[i][j];

         if (stage_index != -1) {
            struct gl_linked_shader *sh = prog->_LinkedShaders[i];

            struct gl_uniform_block **sh_blks = block_type == BLOCK_SSBO ?
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

   if (block_type == BLOCK_SSBO)
      prog->data->ShaderStorageBlocks = blks;
   else
      prog->data->UniformBlocks = blks;

   return true;
}

static bool
_var_is_ubo(nir_variable *var)
{
   return (var->data.mode == nir_var_uniform &&
           var->interface_type != NULL);
}

static bool
_var_is_ssbo(nir_variable *var)
{
   return (var->data.mode == nir_var_shader_storage);
}

/*
 * In opposite to the equivalent glsl one, this one only allocates the needed
 * space. We do a initial count here, just to avoid re-allocating for each one
 * we find.
 */
static void
_allocate_uniform_blocks(void *mem_ctx,
                         struct gl_linked_shader *shader,
                         struct gl_uniform_block **out_blks, unsigned *num_blocks,
                         struct gl_uniform_buffer_variable **out_variables, unsigned *num_variables,
                         enum block_type block_type)
{
   *num_variables = 0;
   *num_blocks = 0;

   nir_foreach_variable(var, &shader->Program->nir->uniforms) {
      if (block_type == BLOCK_UBO && !_var_is_ubo(var))
         continue;

      if (block_type == BLOCK_SSBO && !_var_is_ssbo(var))
         continue;

      const struct glsl_type *type = glsl_without_array(var->type);
      unsigned aoa_size = glsl_type_arrays_of_arrays_size(var->type);
      unsigned buffer_count = aoa_size == 0 ? 1 : aoa_size;

      *num_blocks += buffer_count;
      *num_variables += glsl_get_length(type) * buffer_count;
   }

   if (*num_blocks == 0) {
      assert(*num_variables == 0);
      return;
   }

   assert(*num_variables != 0);

   struct gl_uniform_block *blocks =
      rzalloc_array(mem_ctx, struct gl_uniform_block, *num_blocks);

   struct gl_uniform_buffer_variable *variables =
      ralloc_array(blocks, struct gl_uniform_buffer_variable, *num_variables);

   *out_blks = blocks;
   *out_variables = variables;
}

static unsigned
_get_type_size(const struct glsl_type *type,
               bool row_major,
               enum glsl_interface_packing packing)
{
   switch(packing) {
   case GLSL_INTERFACE_PACKING_STD140:
      return glsl_type_std140_size(type, row_major);
   case GLSL_INTERFACE_PACKING_STD430:
      return glsl_type_std430_size(type, row_major);
   default:
      /* gl_spirv doesn't support packed/shared */
      unreachable("Wrong interface packing");
   }
}

static void
_fill_block(struct gl_uniform_block *block,
            nir_variable *var,
            struct gl_uniform_buffer_variable *variables,
            unsigned *variable_index,
            unsigned array_index,
            struct gl_shader_program *prog)
{
   const struct glsl_type *type = glsl_without_array(var->type);

   block->Name = NULL; /* ARB_gl_spirv: allowed to ignore names */
   /* From ARB_gl_spirv spec:
    *    "Vulkan uses only one binding point for a resource array,
    *     while OpenGL still uses multiple binding points, so binding
    *     numbers are counted differently for SPIR-V used in Vulkan
    *     and OpenGL
    */
   block->Binding = var->data.binding + array_index;
   block->Uniforms = &variables[*variable_index];
   block->NumUniforms = glsl_get_length(type);
   /* FIXME: right now spirv_to_nir pass isn't filling row_major one
    * properly. It is not affecting us as RowMajor for each variable is
    * properly filled, but it would be good to get the proper value here
    */
   block->_RowMajor = glsl_get_row_major(type);
   block->_Packing = glsl_get_interface_packing(type);

   /* FIXME: default values pending to fill  */
   block->linearized_array_index = 0;

   block->UniformBufferSize = 0;
   for (unsigned i = 0; i < glsl_get_length(type); i++) {
      const struct glsl_type *field_type = glsl_get_struct_field(type, i);

      /* ARB_gl_spirv: allowed to ignore names */
      variables[*variable_index].Name = NULL;
      variables[*variable_index].IndexName = NULL;

      variables[*variable_index].Type = field_type;

      /**
       * From ARB_gl_spirv spec:
       *   "Mapping of layouts
       *    std140/std430 -> explicit *Offset*, *ArrayStride*, and
       *           *MatrixStride* Decoration on struct members"
       *
       *   "A variable in the *Uniform* Storage Class decorated as a
       *   *Block* must be explicitly laid out using the *Offset*,
       *   *ArrayStride*, and MatrixStride* decorations. If the variable
       *   is decorated as a BufferBlock*, its offsets and strides must
       *   not contradict std430 alignment and minimum offset
       *   requirements."
       *
       * So we use the offsets coming from the glsl_type, that should
       * have been filled during the spirv to nir pass.
       */
      variables[*variable_index].Offset = glsl_get_struct_field_offset(type, i);

      /* FIXME: untested. */
      /* FIXME: pending to manage INHERITED */
      variables[*variable_index].RowMajor =
         (glsl_get_struct_field_matrix_layout(type, i) ==
          GLSL_MATRIX_LAYOUT_ROW_MAJOR);

      (*variable_index)++;
   }

   /* At this point, to compute the size, we need to compute the size of
    * the last element of the block, add it to the last offset and
    * align
    */
   const struct glsl_type *last_field_type = glsl_get_struct_field(type, glsl_get_length(type) - 1);
   block->UniformBufferSize = variables[*variable_index - 1].Offset +
      _get_type_size(last_field_type, block->_RowMajor, block->_Packing);
   block->UniformBufferSize = glsl_align(block->UniformBufferSize, 16);
}

/*
 * Link ubos/ssbos for a given linked_shader/stage.
 */
static void
_link_linked_shader_uniform_blocks(void *mem_ctx,
                                   struct gl_context *ctx,
                                   struct gl_shader_program *prog,
                                   struct gl_linked_shader *shader,
                                   struct gl_uniform_block **blocks,
                                   unsigned *num_blocks,
                                   enum block_type block_type)
{
   struct gl_uniform_buffer_variable *variables = NULL;

   /* In opposite to GLSL IR linking we don't compute which uniform blocks are
    * inactive. From ARB_gl_spirv spec:
    *   " Removal of features from GLSL, as removed by GL_KHR_vulkan_glsl:
    *     <skip>
    *    - *shared* and *packed* block layouts"
    *
    * And as std430 was never allowed for ubos, only std140 remains as
    * allowed. From 4.6 spec (and before), section 7.6, "Uniform Variables":
    *   "All members of a named uniform block declared with a shared or std140
    *    layout qualifier are considered active, even if they are not
    *    referenced in any shader in the program. The uniform block itself is
    *    also considered active, even if no member of the block is referenced"
    *
    * Conclusion: alls ubos coming from a SPIR-V shader should be considered
    * as active, so we just count them.
    */
   unsigned num_variables = 0;

   _allocate_uniform_blocks(mem_ctx, shader,
                            blocks, num_blocks,
                            &variables, &num_variables,
                            block_type);

   /* Fill the content of uniforms and variables */
   unsigned block_index = 0;
   unsigned variable_index = 0;
   struct gl_uniform_block *blks = *blocks;

   nir_foreach_variable(var, &shader->Program->nir->uniforms) {
      if (block_type == BLOCK_UBO && !_var_is_ubo(var))
         continue;

      if (block_type == BLOCK_SSBO && !_var_is_ssbo(var))
         continue;

      unsigned aoa_size = glsl_type_arrays_of_arrays_size(var->type);
      unsigned buffer_count = aoa_size == 0 ? 1 : aoa_size;

      for (unsigned array_index = 0; array_index < buffer_count; array_index++) {
         _fill_block(&blks[block_index], var, variables, &variable_index,
                     array_index, prog);
         block_index++;
      }
   }

   assert(block_index == *num_blocks);
   assert(variable_index == num_variables);
}

bool
gl_nir_link_uniform_blocks(struct gl_context *ctx,
                           struct gl_shader_program *prog)
{
   void *mem_ctx = ralloc_context(NULL); // temporary linker context

   for (int stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      struct gl_linked_shader *const linked = prog->_LinkedShaders[stage];
      struct gl_uniform_block *ubo_blocks = NULL;
      unsigned num_ubo_blocks = 0;
      struct gl_uniform_block *ssbo_blocks = NULL;
      unsigned num_ssbo_blocks = 0;

      if (!linked)
         continue;

      _link_linked_shader_uniform_blocks(mem_ctx, ctx, prog, linked,
                                         &ubo_blocks, &num_ubo_blocks,
                                         BLOCK_UBO);

      _link_linked_shader_uniform_blocks(mem_ctx, ctx, prog, linked,
                                         &ssbo_blocks, &num_ssbo_blocks,
                                         BLOCK_SSBO);

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
      linked->Program->nir->info.num_ubos = num_ubo_blocks;
      /* This value will get overwritten by the one from nir in
       * brw_shader_gather_info
       */
      linked->Program->info.num_ubos = num_ubo_blocks;

      /* Copy ssbo blocks to linked shader list */
      linked->Program->sh.ShaderStorageBlocks =
         ralloc_array(linked, struct gl_uniform_block *, num_ssbo_blocks);
      ralloc_steal(linked, ssbo_blocks);
      for (unsigned i = 0; i < num_ssbo_blocks; i++) {
         linked->Program->sh.ShaderStorageBlocks[i] = &ssbo_blocks[i];
      }
      /* See previous comment on num_ubo_blocks
       *
       * FIXME: in general this is somewhat ugly. It would be better to try to
       * find a way to set the info once, and being able to properly gather
       * the info.
       */
      linked->Program->nir->info.num_ssbos = num_ssbo_blocks;
      linked->Program->info.num_ssbos = num_ssbo_blocks;
   }

   /* Process UBOs */
   if (!_nir_interstage_cross_validate_uniform_blocks(prog, BLOCK_UBO)) {
      return false;
   }

   /* Process SSBOs */
   if (!_nir_interstage_cross_validate_uniform_blocks(prog, BLOCK_SSBO)) {
      return false;
   }

   return true;
}
