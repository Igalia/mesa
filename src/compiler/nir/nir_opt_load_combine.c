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
 */

/**
 * Implements a load-combine pass for load/store instructions. Similar to a
 * CSE pass, but needs to consider invalidation of cached loads by stores
 * or memory barriers. It only works on local blocks for now.
 */

#include "nir.h"
#include "program/prog_instruction.h"

/*
 * SSBO stores won't invalidate image loads for example, so we want to
 * classify load/store operations in groups and only invalidate / reuse
 * intrinsics in the same group.
 */
enum intrinsic_groups {
   INTRINSIC_GROUP_NONE = 0,
   INTRINSIC_GROUP_ALL,
   INTRINSIC_GROUP_SSBO,
   INTRINSIC_GROUP_SHARED,
   INTRINSIC_GROUP_IMAGE
};

struct cache_node {
   struct list_head list;
   nir_instr *instr;
};

/* SSBO load/store */
static bool
is_atomic_ssbo(nir_intrinsic_instr *intrinsic)
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

static inline bool
is_store_ssbo(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_store_ssbo ||
      is_atomic_ssbo(intrinsic);
}

static inline bool
is_load_ssbo(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_load_ssbo;
}

static inline bool
is_memory_barrier_buffer(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_memory_barrier_buffer;
}

/* Shared variable load/store */
static bool
is_atomic_shared(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
      return true;
   default:
      return false;
   }
}

static inline bool
is_store_shared(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_store_shared ||
      is_atomic_shared(intrinsic);
}

static inline bool
is_load_shared(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_load_shared;
}

static inline bool
is_memory_barrier_shared(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_memory_barrier_shared;
}

/* Image load/store */
static bool
is_atomic_image(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_image_atomic_add:
   case nir_intrinsic_image_atomic_min:
   case nir_intrinsic_image_atomic_max:
   case nir_intrinsic_image_atomic_and:
   case nir_intrinsic_image_atomic_or:
   case nir_intrinsic_image_atomic_xor:
   case nir_intrinsic_image_atomic_exchange:
   case nir_intrinsic_image_atomic_comp_swap:
      return true;
   default:
      return false;
   }
}

static bool
is_store_image(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_image_store ||
      is_atomic_image(intrinsic);
}

static bool
is_load_image(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_image_load;
}

static inline bool
is_memory_barrier_image(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_memory_barrier_image;
}

/* General intrinsic classification functions */
static inline bool
is_store(nir_intrinsic_instr *intrinsic)
{
   return is_store_ssbo(intrinsic) || is_store_shared(intrinsic) ||
      is_store_image(intrinsic);
}

static inline bool
is_load(nir_intrinsic_instr *intrinsic)
{
   return is_load_ssbo(intrinsic) || is_load_shared(intrinsic) ||
      is_load_image(intrinsic);
}

static inline bool
is_atomic(nir_intrinsic_instr *intrinsic)
{
   return is_atomic_ssbo(intrinsic) || is_atomic_shared(intrinsic)
      || is_atomic_image(intrinsic);
}

static inline bool
is_memory_barrier(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_memory_barrier ||
      is_memory_barrier_buffer(intrinsic) ||
      is_memory_barrier_shared(intrinsic) ||
      is_memory_barrier_image(intrinsic);
}

static unsigned
intrinsic_group(nir_intrinsic_instr *intrinsic)
{
   if (intrinsic->intrinsic == nir_intrinsic_memory_barrier)
      return INTRINSIC_GROUP_ALL;
   else if (is_load_ssbo(intrinsic) || is_store_ssbo(intrinsic) ||
            is_memory_barrier_buffer(intrinsic))
      return INTRINSIC_GROUP_SSBO;
   else if (is_load_shared(intrinsic) || is_store_shared(intrinsic) ||
            is_memory_barrier_shared(intrinsic))
      return INTRINSIC_GROUP_SHARED;
   else if (is_load_image(intrinsic) || is_store_image(intrinsic) ||
            is_memory_barrier_image(intrinsic))
      return INTRINSIC_GROUP_IMAGE;
   else
      return INTRINSIC_GROUP_NONE;
}

static bool
intrinsic_group_match(nir_intrinsic_instr *intrinsic1,
                      nir_intrinsic_instr *intrinsic2)
{
   int group1 = intrinsic_group(intrinsic1);
   int group2 = intrinsic_group(intrinsic2);

   return group1 == INTRINSIC_GROUP_ALL || group2 == INTRINSIC_GROUP_ALL
      || group1 == group2;
}

static void
cache_add(struct cache_node *cache, nir_instr *instr)
{
   struct cache_node *node = ralloc(NULL, struct cache_node);
   node->instr = instr;
   list_addtail(&node->list, &cache->list);
}

static void
cache_clear(struct cache_node *cache)
{
   list_for_each_entry(struct cache_node, item, &cache->list, list) {
      ralloc_free(item);
   }

   list_empty(&cache->list);
   list_inithead(&cache->list);
}

/**
 * Returns true if a nir_src is direct, defined here as an SSA value
 * whose parent instruction is a load_const.
 */
static bool
nir_src_is_direct(nir_src *src)
{
   if (!src->is_ssa)
      return false;

   nir_instr *parent_instr = src->ssa->parent_instr;
   if (parent_instr->type != nir_instr_type_load_const)
      return false;

   return true;
}

/**
 * Returns true if a nir_src represents an undefined SSA register, false
 * otherwise.
 */
static inline bool
nir_src_is_undefined(nir_src src)
{
   return src.is_ssa &&
      src.ssa->parent_instr->type == nir_instr_type_ssa_undef;
}

/**
 * Gets the block and offset of a load/store instruction.
 *
 * @instr: the intrinsic load/store operation
 * @block: the output block, can be NULL if @base is non-NULL (shared-vars)
 * @offset: the output offset
 * @base: the output base, can be NULL if @block is non-NULL (SSBO)
 */
static void
get_load_store_address(nir_intrinsic_instr *instr,
                       nir_src **block,
                       nir_src **offset,
                       unsigned *base)
{
   int block_index = -1;
   int offset_index = -1;
   int base_index = -1;

   assert((block || base) && offset);

   switch (instr->intrinsic) {
      /* SSBO */
   case nir_intrinsic_store_ssbo:
      block_index = 1;
      offset_index = 2;
      break;

   case nir_intrinsic_load_ssbo:
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

      /* Shared-variables memory access is defined by a direct
       * value 'base' (const_index[0]) and an indirect SSA value 'offset'.
       */
   case nir_intrinsic_load_shared:
      base_index = 0;
      offset_index = 0;
      break;

   case nir_intrinsic_store_shared:
      base_index = 0;
      offset_index = 1;
      break;

   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
      base_index = 0;
      offset_index = 0;
      break;

   default:
      assert(!"not implemented");
   }

   assert((block_index >= 0 || base_index >= 0) && offset_index >= 0);

   if (block && block_index >= 0)
      *block = &instr->src[block_index];

   *offset = &instr->src[offset_index];

   if (base && base_index >= 0)
      *base = instr->const_index[base_index];
}

/**
 * Returns true if two coordinates sources of an image intrinsic match.
 * This means that both sources are defined by ALU vec4 ops, with all
 * 4 sources equivalent. For example, consider the following snippet:
 *
 *   vec1 ssa_1 = undefined
 *   vec4 ssa_2 = vec4 ssa_0, ssa_0, ssa_1, ssa_1
 *   vec4 ssa_3 = intrinsic image_load (ssa_2, ssa_1) (itex) ()
 *   ...
 *   vec1 ssa_6 = undefined
 *   vec4 ssa_7 = vec4 ssa_0, ssa_0, ssa_1, ssa_1
 *   vec4 ssa_8 = intrinsic image_load (ssa_7, ssa_6) (itex) ()
 *
 * Here, ssa_2 and ssa_7 are the coordinates inside the image, and
 * they are two different SSA definitions, so comparing them directly
 * won't work. This function is able to detect this and check that
 * ssa_2 and ssa_7 are indeed "equivalent"; so the pass can tell that
 * the two image_load instructions are a match.
 */
static bool
coordinates_match(nir_src *coord1, nir_src *coord2)
{
   assert(coord1->is_ssa);
   assert(coord2->is_ssa);

   nir_ssa_def *ssa1 = coord1->ssa;
   nir_ssa_def *ssa2 = coord2->ssa;

   nir_instr *parent1 = ssa1->parent_instr;
   nir_instr *parent2 = ssa2->parent_instr;

   /* @FIXME: currently, all coordinates into an image load/store
    * instruction are given by an ALU vec4 instruction. This may change in
    * the future, in which case we should detect it here.
    */
   assert(parent1->type == nir_instr_type_alu);
   assert(parent2->type == nir_instr_type_alu);

   nir_alu_instr *alu1 = nir_instr_as_alu(parent1);
   nir_alu_instr *alu2 = nir_instr_as_alu(parent2);

   assert(alu1->op == nir_op_vec4);
   assert(alu2->op == nir_op_vec4);

   for (unsigned i = 0; i < 4; i++) {
      if (! ((nir_src_is_undefined(alu1->src[i].src) &&
              nir_src_is_undefined(alu2->src[i].src)) ||
             nir_srcs_equal(alu1->src[i].src, alu2->src[i].src))) {
         return false;
      }
   }

   return true;
}

/**
 * Determines whether two instrinsic instructions conflict with each other,
 * meaning that a) they access the same memory area, or b) a non-conflict
 * cannot be determined (because at least one access is indirect).
 *
 * @full_match serves as an output flag to signal that
 * the conflict ocurred because both instructions access the exact same
 * memory region. This is used in the pass to know that two intructions
 * are safe to combine.
 *
 * Returns true upon conflict, false otherwise.
 */
static bool
detect_memory_access_conflict(nir_intrinsic_instr *instr1,
                              nir_intrinsic_instr *instr2,
                              bool *full_match)
{
   nir_src *instr1_block = NULL;
   nir_src *instr1_offset = NULL;
   unsigned instr1_base = 0;
   nir_src *instr2_block = NULL;
   nir_src *instr2_offset = NULL;
   unsigned instr2_base = 0;
   bool blocks_match = false;
   bool offsets_match = false;
   bool bases_match = false;

   if (full_match)
      *full_match = false;

   /* if intrinsic groups don't match, there can't be any conflict */
   if (!intrinsic_group_match(instr1, instr2))
      return false;

   /* Since image load/store are always indirect, we should always report
    * conflict because we are unable to determine it. However, we still want
    * to know if the texture objects and coordinates match, to report back
    * a full match.
    */
   if (intrinsic_group(instr1) == INTRINSIC_GROUP_IMAGE) {
      if (full_match) {
         assert(instr1->variables[0]);
         assert(instr2->variables[0]);

         if (instr1->variables[0]->var == instr2->variables[0]->var &&
             coordinates_match(&instr1->src[0], &instr2->src[0])) {
            *full_match = true;
         }
      }

      return true;
   }

   get_load_store_address(instr1, &instr1_block, &instr1_offset, &instr1_base);
   get_load_store_address(instr2, &instr2_block, &instr2_offset, &instr2_base);

   /* There is conflict if the blocks (or bases) and the offsets of each
    * instruction are not both direct or both indirect. If that's not the
    * case, then there is conflict if the blocks (or bases) and offsets
    * all match.
    */

   /* only SSBOs have a block source, so it can be NULL for shared-var */
   if (instr1_block) {
      if (! instr2_block)
         return true;

      /* For SSBOs the block is an SSA value, but it can still be direct,
       * if defined by a load_const instruction.
       */
      if (nir_src_is_direct(instr1_block) != nir_src_is_direct(instr2_block))
         return true;

      blocks_match = nir_srcs_equal(*instr1_block, *instr2_block);
   } else {
      bases_match = instr1_base == instr2_base;
   }

   /* For SSBOs, the offset is an SSA value, but it can still be direct,
    *if defined by a load_const instruction.
    */
   if (nir_src_is_direct(instr1_offset) != nir_src_is_direct(instr2_offset))
      return true;

   offsets_match = nir_srcs_equal(*instr1_offset, *instr2_offset);

   /* finally, if both blocks and offsets match, it means conflict */
   if (offsets_match && (blocks_match || bases_match)) {
      if (full_match)
         *full_match = true;

      return true;
   }

   return false;
}

static unsigned
get_store_writemask(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_store_ssbo:
      return instr->const_index[0];

   case nir_intrinsic_store_shared:
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
      /* fall-through to shared variable atomics */
   case nir_intrinsic_shared_atomic_add:
   case nir_intrinsic_shared_atomic_imin:
   case nir_intrinsic_shared_atomic_umin:
   case nir_intrinsic_shared_atomic_imax:
   case nir_intrinsic_shared_atomic_umax:
   case nir_intrinsic_shared_atomic_and:
   case nir_intrinsic_shared_atomic_or:
   case nir_intrinsic_shared_atomic_xor:
   case nir_intrinsic_shared_atomic_exchange:
   case nir_intrinsic_shared_atomic_comp_swap:
      return WRITEMASK_X;

   default:
      assert(!"not implemented");
   }
}

/**
 * Traverses the set of cached load/store intrinsics and invalidates all that
 * conflict with @store.
 */
static void
cache_invalidate_for_store(struct cache_node *cache,
                           nir_intrinsic_instr *store)
{
   assert(is_store(store));

   list_for_each_entry_safe(struct cache_node, item, &cache->list, list) {
      nir_instr *instr = item->instr;
      assert(instr->type == nir_instr_type_intrinsic);

      nir_intrinsic_instr *cached = nir_instr_as_intrinsic(instr);

      if (detect_memory_access_conflict(store, cached, NULL)) {
         /* remove the cached instruction from the list */
         list_del(&item->list);
         ralloc_free(item);
      }
   }
}

/**
 * Traverses the set of cached load/store intrinsics and tries to
 * rewrite the given load instruction with a previous compatible load.
 */
static bool
rewrite_load_with_load(struct cache_node *cache,
                       nir_intrinsic_instr *load)
{
   assert(is_load(load));

   list_for_each_entry(struct cache_node, item, &cache->list, list) {
      nir_instr *instr = item->instr;
      assert(instr->type == nir_instr_type_intrinsic);

      nir_intrinsic_instr *prev_load = nir_instr_as_intrinsic(instr);
      if (!is_load(prev_load))
         continue;

      /* Both intrinsics must access same memory area (block, offset, etc).
       *
       * Here. we reuse detect_memory_access_conflict(), which meets this
       * purpose semantically, except that we need to know if the conflict
       * happened because the blocks and offsets match.
       */
      bool blocks_and_offsets_match = false;
      if (!detect_memory_access_conflict(load, prev_load,
                                         &blocks_and_offsets_match)) {
         continue;
      }

      if (blocks_and_offsets_match) {
         /* rewrite the new load with the cached load instruction */
         nir_ssa_def *def = &load->dest.ssa;
         nir_ssa_def *new_def = &prev_load->dest.ssa;
         nir_ssa_def_rewrite_uses(def, nir_src_for_ssa(new_def));

         return true;
      }
   }

   cache_add(cache, &load->instr);

   return false;
}

/**
 * Traverses the set of load/store intrinsics trying to find a previous store
 * operation to the same block/offset which we can reuse to re-write a load
 * from the same block/offset.
 */
static bool
rewrite_load_with_store(struct cache_node *cache,
                        nir_intrinsic_instr *load)
{
   assert(is_load(load));

   list_for_each_entry(struct cache_node, item, &cache->list, list) {
      nir_instr *instr = item->instr;
      assert(instr->type == nir_instr_type_intrinsic);
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);

      if (!is_store(store))
         continue;

      /* We cannot rewrite with atomics, because the result of these is
       * the old value of the buffer before the atomic execution.
       */
      if (is_atomic(store))
         continue;

      /* Both intrinsics must access same memory area (block, offset, etc).
       *
       * Here. we reuse detect_memory_access_conflict(), which meets this
       * purpose semantically, except that we need to know if the conflict
       * happened because the blocks and offsets match.
       */
      bool blocks_and_offsets_match = false;
      if (!detect_memory_access_conflict(load, store,
                                         &blocks_and_offsets_match)) {
         continue;
      }
      if (!blocks_and_offsets_match)
         continue;

      if (intrinsic_group(store) != INTRINSIC_GROUP_IMAGE) {
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
      }

      /* rewrite the new load with the cached store instruction */
      nir_ssa_def *def = &load->dest.ssa;
      nir_ssa_def *new_def;
      /* while the value source for SSBO and shared-vars is src[0],
       * image stores use src[2].
       */
      if (intrinsic_group(store) != INTRINSIC_GROUP_IMAGE)
         new_def = store->src[0].ssa;
      else
         new_def = store->src[2].ssa;
      nir_ssa_def_rewrite_uses(def, nir_src_for_ssa(new_def));

      return true;
   }

   return false;
}

/**
 * Traverses the set of cached load/store intrinsics, and remove those
 * whose intrinsic group matches @group.
 */
static void
cache_invalidate_for_group(struct cache_node *cache, unsigned group)
{
   list_for_each_entry_safe(struct cache_node, item, &cache->list, list) {
      nir_instr *instr = item->instr;
      assert(instr->type == nir_instr_type_intrinsic);

      nir_intrinsic_instr *cached = nir_instr_as_intrinsic(instr);

      if (group == INTRINSIC_GROUP_ALL || intrinsic_group(cached) == group) {
         list_del(&item->list);
         ralloc_free(item);
      }
   }
}

static bool
load_combine_block(nir_block *block)
{
   bool progress = false;

   /* This pass only works on local blocks for now, so we create and destroy
    * the instruction cache with each block.
    */
   struct cache_node cache = {0};
   list_inithead(&cache.list);

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
      if (is_load(intrinsic)) {
         /* Try to rewrite with a previous load */
         if (rewrite_load_with_load(&cache, intrinsic)) {
            nir_instr_remove(instr);
            progress = true;
         } else if (rewrite_load_with_store(&cache, intrinsic)) {
            progress = true;
         }
      } else if (is_store(intrinsic)) {
         /* Invalidate conflicting load/stores and add the store to the cache
          * so we can rewrite future loads with it.
          */
         cache_invalidate_for_store(&cache, intrinsic);
         cache_add(&cache, instr);
      } else if (is_memory_barrier(intrinsic)) {
         /* If we see a memory barrier we have to invalidate all cached
          * load/store operations from the same intrinsic group.
          */
         cache_invalidate_for_group(&cache, intrinsic_group(intrinsic));
      }
   }

   cache_clear(&cache);

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

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_load_combine_impl(function->impl);
   }

   return progress;
}
