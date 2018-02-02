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
 * is intended to be used with GL_ARB_gl_spirv. Unlike Vulkan, it is not
 * necessary to wait for the complete pipeline state to lower it.
 */

static unsigned
find_block_by_binding(struct gl_linked_shader *linked_shader,
                      unsigned binding)
{
   unsigned num_blocks = linked_shader->Program->info.num_ubos;
   struct gl_uniform_block **blocks = linked_shader->Program->sh.UniformBlocks;

   for (unsigned i = 0; i < num_blocks; i++) {
      if (blocks[i]->Binding == binding)
         return i;
   }

   unreachable("No block found with the given binding");
}

static bool
convert_block(nir_block *block,
              struct gl_linked_shader *linked_shader,
              nir_builder *b)
{
   bool progress = false;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *res_index = nir_instr_as_intrinsic(instr);

      if (res_index->intrinsic != nir_intrinsic_vulkan_resource_index)
         continue;

      b->cursor = nir_after_instr(instr);

      /* The descriptor set should always be zero for GL */
      assert(nir_intrinsic_desc_set(res_index) == 0);

      unsigned binding = nir_intrinsic_binding(res_index);
      unsigned block = find_block_by_binding(linked_shader, binding);
      nir_ssa_def *surface =
         nir_iadd(b,
                  nir_imm_int(b, block),
                  nir_ssa_for_src(b, res_index->src[0], 1));

      nir_ssa_def_rewrite_uses(&res_index->dest.ssa, nir_src_for_ssa(surface));
      nir_instr_remove(instr);

      progress = true;
   }

   return progress;
}

static bool
convert_impl(nir_function_impl *impl,
             struct gl_linked_shader *linked_shader)
{
   bool progress = false;
   nir_builder builder;
   nir_builder_init(&builder, impl);

   nir_foreach_block(block, impl) {
      progress |= convert_block(block, linked_shader, &builder);
   }

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
   return progress;
}

bool
gl_nir_lower_vulkan_resource_index(nir_shader *shader,
                                   struct gl_linked_shader *linked_shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress = convert_impl(function->impl, linked_shader) || progress;
   }

   return progress;
}
