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
 *
 * Authors:
 *    Neil Roberts (nroberts@igalia.com)
 *
 */

#include "nir.h"
#include "gl_nir.h"
#include "nir_builder.h"
#include "main/mtypes.h"

/*
 * This pass lowers the vulkan_resource_index intrinsic to a surface index. It
 * is intended to be used with GL_ARB_gl_spirv. Unlike Vulkan, in that case it
 * is not necessary to wait for the complete pipeline state to lower it.
 */
typedef bool (*nir_lower_vulkan_intrinsic_cb) (struct nir_builder *b,
                                               nir_intrinsic_instr *intrin,
                                               struct gl_linked_shader *linked_shader);

static unsigned
find_block_by_binding(unsigned num_blocks,
                      struct gl_uniform_block **blocks,
                      unsigned binding)
{
   for (unsigned i = 0; i < num_blocks; i++) {
      if (blocks[i]->Binding == binding)
         return i;
   }

   unreachable("No block found with the given binding");
}

static bool
find_intrinsic_usage(nir_ssa_def *def,
                     bool *is_ubo_usage)
{
   nir_foreach_use_safe(use_src, def) {
      if (use_src->parent_instr->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(use_src->parent_instr);

         if (find_intrinsic_usage(&alu->dest.dest.ssa, is_ubo_usage))
            return true;

         continue;
      }

      if (use_src->parent_instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(use_src->parent_instr);

      if (intr == NULL)
         continue;

      *is_ubo_usage = intr->intrinsic == nir_intrinsic_load_ubo;
      return true;
   }

   return false;
}

static bool
lower_vulkan_resource_index(struct nir_builder *b,
                            nir_intrinsic_instr *intrin,
                            struct gl_linked_shader *linked_shader)
{
   bool is_ubo_usage;
   if (!find_intrinsic_usage(&intrin->dest.ssa, &is_ubo_usage))
      return false;

   b->cursor = nir_after_instr(&intrin->instr);

   /* The descriptor set should always be zero for GL */
   assert(nir_intrinsic_desc_set(intrin) == 0);
   unsigned binding = nir_intrinsic_binding(intrin);

   unsigned num_blocks;
   struct gl_uniform_block **blocks;

   if (is_ubo_usage) {
      num_blocks = linked_shader->Program->info.num_ubos;
      blocks = linked_shader->Program->sh.UniformBlocks;
   } else {
      num_blocks = linked_shader->Program->info.num_ssbos;
      blocks = linked_shader->Program->sh.ShaderStorageBlocks;
   }

   unsigned block = find_block_by_binding(num_blocks, blocks, binding);

   nir_ssa_def *surface =
      nir_iadd(b,
               nir_imm_int(b, block),
               nir_ssa_for_src(b, intrin->src[0], 1));

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, nir_src_for_ssa(surface));
   nir_instr_remove(&intrin->instr);

   return true;
}

static bool
convert_impl(nir_function_impl *impl,
             nir_lower_vulkan_intrinsic_cb callback,
             nir_intrinsic_op intrin_op,
             struct gl_linked_shader *linked_shader)
{
   bool progress = false;
   nir_builder builder;
   nir_builder_init(&builder, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         if (intrin->intrinsic != intrin_op)
            continue;

         progress |= callback(&builder, intrin, linked_shader);
      }
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
   return progress;
}

static bool
lower_load_vulkan_descriptor(nir_builder *b,
                             nir_intrinsic_instr *intrin,
                             struct gl_linked_shader *linked_shader)
{
   b->cursor = nir_before_instr(&intrin->instr);

   /* We follow the nir_address_format_vk_index_offset model */
   assert(intrin->src[0].is_ssa);
   nir_ssa_def *vec2 = nir_vec2(b, intrin->src[0].ssa, nir_imm_int(b, 0));

   assert(intrin->dest.is_ssa);
   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, nir_src_for_ssa(vec2));
   nir_instr_remove(&intrin->instr);

   return true;
}

bool
gl_nir_lower_vulkan_descriptor(nir_shader *shader,
                               struct gl_linked_shader *linked_shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress = convert_impl(function->impl, lower_load_vulkan_descriptor,
                                 nir_intrinsic_load_vulkan_descriptor, linked_shader) || progress;
   }

   return progress;
}


bool
gl_nir_lower_vulkan_resource_index(nir_shader *shader,
                                   struct gl_linked_shader *linked_shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress = convert_impl(function->impl, lower_vulkan_resource_index,
                                 nir_intrinsic_vulkan_resource_index, linked_shader) || progress;
   }

   return progress;
}
