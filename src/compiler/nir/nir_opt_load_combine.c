/*
 * Copyright © 2016 Intel Corporation
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
 *    Iago Toral Quiroga <itoral@igalia.com>
 *
 */

/*
 * Implements a load-combine pass for load/store instructions. Similar to a
 * CSE pass, but needs to consider invalidation of cached loads by stores
 * or memory barriers. It only works on local blocks for now.
 */

#include "nir_instr_set.h"

/*
 * SSBO stores won't invalidate image loads for example, so we want to
 * classify load/store operations in groups and only invalidate / reuse
 * intrinsics in the same group.
 */
enum intrinsic_groups {
   INTRINSIC_GROUP_NONE = 0,
   INTRINSIC_GROUP_SSBO
};

/* SSBO load/store */
static bool
is_indirect_store_ssbo(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
      return true;
   default:
      return false;
   }
}

static bool
is_direct_store_ssbo(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_store_ssbo:
      return true;
   default:
      return false;
   }
}

static bool
is_direct_load_ssbo(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_load_ssbo:
      return true;
   default:
      return false;
   }
}

/*
 * General load/store functions: we'll add more groups to this as needed.
 * For now we only support SSBOs.
 */
static bool
is_indirect_store(nir_intrinsic_instr *intrinsic)
{
   return is_indirect_store_ssbo(intrinsic);
}

static bool
is_direct_store(nir_intrinsic_instr *intrinsic)
{
   return is_direct_store_ssbo(intrinsic);
}

static bool
is_store(nir_intrinsic_instr *intrinsic)
{
   return is_direct_store(intrinsic) || is_indirect_store(intrinsic);
}

static bool
is_indirect_load(nir_intrinsic_instr *intrinsic)
{
   return false;
}

static bool
is_direct_load(nir_intrinsic_instr *intrinsic)
{
   return is_direct_load_ssbo(intrinsic);
}

static bool
is_load(nir_intrinsic_instr *intrinsic)
{
  return is_direct_load(intrinsic) || is_indirect_load(intrinsic);
}

static bool
is_memory_barrier(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_memory_barrier;
}

static void
set_clear(struct nir_instr_set *instr_set)
{
   struct set_entry *entry;
   set_foreach(instr_set->set, entry)
      _mesa_set_remove(instr_set->set, entry);
}

static unsigned
intrinsic_group(nir_intrinsic_instr *intrinsic)
{
   if (is_direct_load_ssbo(intrinsic) ||
       is_direct_store_ssbo(intrinsic) ||
       is_indirect_store_ssbo(intrinsic))
      return INTRINSIC_GROUP_SSBO;
   return INTRINSIC_GROUP_NONE;
}

static bool
intrinsic_group_match(nir_intrinsic_instr *intrinsic1,
                      nir_intrinsic_instr *intrinsic2)
{
   return intrinsic_group(intrinsic1) == intrinsic_group(intrinsic2);
}

/*
 * Gets the block and offset of a load/store instruction.
 *
 * @instr: the intrinsic load/store operation
 * @block: the block index
 * @offset: the indirect offset (NULL for direct offset)
 * @const_offset: the direct offset (only if offset is not NULL)
 *
 * Each out parameter can be set to NULL if we are not interested in it.
 */
static void
get_load_store_address(nir_intrinsic_instr *instr,
                       nir_src **block,
                       nir_src **offset,
                       unsigned *const_offset)
{
   int block_index = -1;
   int offset_index = -1;
   int const_offset_index = -1;

   switch (instr->intrinsic) {
   case nir_intrinsic_load_ssbo:
      block_index = 0;
      const_offset_index = 0;
      break;
   case nir_intrinsic_store_ssbo:
      block_index = 1;
      const_offset_index = 0;
      break;
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
      block_index = 0;
      offset_index = 1;
      break;
   default:
      assert(!"not implemented");
   }

   assert(block_index >= 0 && (offset_index >= 0 || const_offset_index >= 0));

   if (block)
      *block = &instr->src[block_index];

   if (offset && offset_index >= 0)
      *offset = &instr->src[offset_index];

   if (const_offset && const_offset_index >= 0)
      *const_offset = instr->const_index[const_offset_index];
}

/*
 * Traverses the set of cached load/store intrinsics and invalidates all that
 * conflict with @store.
 */
static void
set_invalidate_for_store(struct nir_instr_set *instr_set,
                         nir_intrinsic_instr *store)
{
   assert(is_store(store));

   bool store_is_indirect = is_indirect_store(store);

   nir_src *store_block;
   unsigned store_offset;
   if (!store_is_indirect)
      get_load_store_address(store, &store_block, NULL, &store_offset);

   for (struct set_entry *entry = _mesa_set_next_entry(instr_set->set, NULL);
        entry != NULL; entry = _mesa_set_next_entry(instr_set->set, entry)) {

      /* Only invalidate instructions in the same load/store group */
      assert(((nir_instr *) entry->key)->type == nir_instr_type_intrinsic);
      nir_intrinsic_instr *cached =
         nir_instr_as_intrinsic((nir_instr *) entry->key);
      if (!intrinsic_group_match(store, cached))
         continue;

      bool cached_is_indirect = false;
      /* is_indirect_load(cached) || is_indirect_store(cached);*/
      if (store_is_indirect || cached_is_indirect) {
         nir_instr_set_remove(instr_set, (nir_instr *) entry->key);
      } else {
         /* direct store and cached */
         nir_src *cached_block;
         unsigned cached_offset;
         get_load_store_address(cached, &cached_block, NULL, &cached_offset);

         /* offset and block must match */
         if (store_offset != cached_offset)
            continue;

         if (!nir_srcs_equal(*store_block, *cached_block) &&
             store_block->ssa->parent_instr->type ==
             cached_block->ssa->parent_instr->type)
            continue;

         nir_instr_set_remove(instr_set, (nir_instr *) entry->key);
      }

   }
}

static unsigned
get_store_writemask(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_store_ssbo:
      return instr->const_index[1];
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
      return 0x1;
   default:
      assert(!"not implemented");
   }
}

/*
 * Traverses the set of load/store intrinsics trying to find a previous store
 * operation to the same block/offset which we can reuse to re-write a load
 * from the same block/offset.
 */
static bool
rewrite_load_with_store(struct nir_instr_set *instr_set,
                        nir_intrinsic_instr *load)
{
   nir_src *load_block = NULL;
   nir_src *load_offset = NULL;
   unsigned load_const_offset;
   get_load_store_address(load, &load_block, &load_offset, &load_const_offset);

   for (struct set_entry *entry = _mesa_set_next_entry(instr_set->set, NULL);
        entry != NULL; entry = _mesa_set_next_entry(instr_set->set, entry)) {

      nir_instr *instr = (nir_instr *) entry->key;
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
      if (!is_store(store))
         continue;

      nir_src *store_block = NULL;
      nir_src *store_offset = NULL;
      unsigned store_const_offset;
      get_load_store_address(store, &store_block, &store_offset,
                             &store_const_offset);

      /* The store must write to all the channels we are loading */
      unsigned store_writemask = get_store_writemask(store);
      bool writes_all_channels = true;
      for (int i = 0; i < load->num_components; i++) {
         if (!((1 << i) & store_writemask)) {
            writes_all_channels = false;
            break;
         }
      }
      if (!writes_all_channels)
         continue;

      /* blocks must match */
      if ((load_block->ssa->parent_instr->type !=
           store_block->ssa->parent_instr->type) ||
          !nir_srcs_equal(*store_block, *load_block))
         continue;

      /* address type (indirect/direct) must match */
      if (!load_offset != !store_offset)
         continue;

      /* indirect address mismatch */
      if (load_offset && store_offset &&
          !nir_srcs_equal(*store_offset, *load_offset))
         continue;

      /* direct address mismatch */
      if (!load_offset && !store_offset &&
          store_const_offset != load_const_offset)
         continue;

      /* rewrite the load with this store */
      nir_ssa_def *def = &load->dest.ssa;
      nir_ssa_def *new_def = store->src[0].ssa;
      nir_ssa_def_rewrite_uses(def, nir_src_for_ssa(new_def));
      return true;
   }

   return false;
}

static bool
load_combine_block(nir_block *block)
{
   bool progress = false;

   /* This pass only works on local blocks for now, so we create and destroy
    * the instruction set with each block.
    */
   struct nir_instr_set *instr_set = nir_instr_set_create(NULL, true);

   nir_foreach_instr_safe(block, instr) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
      if (is_load(intrinsic)) {
         /* Try to rewrite with a previous load, if that fails, try to
          * rewrite with a previous store
          */
         if (nir_instr_set_add_or_rewrite(instr_set, instr)) {
            progress = true;
            nir_instr_remove(instr);
         } else if(rewrite_load_with_store(instr_set, intrinsic)) {
            progress = true;
         }
      } else if (is_store(intrinsic)) {
         /* Invalidate conflicting load/stores and add the store to the set
          * so we can rewrite future loads with it
          */
         set_invalidate_for_store(instr_set, intrinsic);
         _mesa_set_add(instr_set->set, instr);
      } else if (is_memory_barrier(intrinsic)) {
         /* If we see a memory barrier we have to invalidate all cached
          * load/store operations
          */
         set_clear(instr_set);
      }
   }

   nir_instr_set_destroy(instr_set);

   for (unsigned i = 0; i < block->num_dom_children; i++) {
      nir_block *child = block->dom_children[i];
      progress |= load_combine_block(child);
   }

   return progress;
}

static bool
nir_opt_load_combine_impl(nir_function_impl *impl)
{
   nir_metadata_require(impl, nir_metadata_dominance);

   bool progress = load_combine_block(nir_start_block(impl));

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   return progress;
}

bool
nir_opt_load_combine(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(shader, function) {
      if (function->impl)
         progress |= nir_opt_load_combine_impl(function->impl);
   }

   return progress;
}
