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
   INTRINSIC_GROUP_SSBO,
   INTRINSIC_GROUP_SHARED,
   INTRINSIC_GROUP_IMAGE
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
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_store_ssbo:
      return true;
   default:
      return is_atomic_ssbo(intrinsic);
   }
}

static inline bool
is_load_ssbo(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_load_ssbo;
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
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_store_shared:
      return true;
   default:
      return is_atomic_shared(intrinsic);
   }
}

static inline bool
is_load_shared(nir_intrinsic_instr *intrinsic)
{
   return intrinsic->intrinsic == nir_intrinsic_load_shared;
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
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_image_store:
      return true;
   default:
      return is_atomic_image(intrinsic);
   }
}

static bool
is_load_image(nir_intrinsic_instr *intrinsic)
{
   switch (intrinsic->intrinsic) {
   case nir_intrinsic_image_load:
      return true;
   default:
      return false;
   }
}

/*
 * General load/store functions: we'll add more groups to this as needed.
 * For now we only support SSBOs.
 */
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
      intrinsic->intrinsic == nir_intrinsic_memory_barrier_buffer ||
      intrinsic->intrinsic == nir_intrinsic_memory_barrier_shared ||
      intrinsic->intrinsic == nir_intrinsic_memory_barrier_image;
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
   if (is_load_ssbo(intrinsic) || is_store_ssbo(intrinsic))
      return INTRINSIC_GROUP_SSBO;
   else if (is_load_shared(intrinsic) || is_store_shared(intrinsic))
      return INTRINSIC_GROUP_SHARED;
   else if (is_load_image(intrinsic) || is_store_image(intrinsic))
      return INTRINSIC_GROUP_IMAGE;
   else
      return INTRINSIC_GROUP_NONE;
}

static inline bool
intrinsic_group_match(nir_intrinsic_instr *intrinsic1,
                      nir_intrinsic_instr *intrinsic2)
{
   return intrinsic_group(intrinsic1) == intrinsic_group(intrinsic2);
}

/*
 * Gets the block and offset of a load/store instruction.
 *
 * @instr: the intrinsic load/store operation
 * @block: the block index (NULL for direct block)
 * @const_block: the direct block (NULL for indirect block)
 * @offset: the indirect offset (NULL for direct offset)
 * @const_offset: the direct offset (NULL for indirect offset)
 *
 * Each out parameter can be set to NULL if we are not interested in it.
 */
static void
get_load_store_address(nir_intrinsic_instr *instr,
                       nir_src **block,
                       unsigned *const_block,
                       nir_src **offset,
                       unsigned *const_offset)
{
   int block_index = -1;
   int const_block_index = -1;
   int offset_index = -1;
   int const_offset_index = -1;

   switch (instr->intrinsic) {
      /* SSBO */
   case nir_intrinsic_load_ssbo:
      block_index = 0;
      offset_index = 1;
      break;
   case nir_intrinsic_store_ssbo:
      block_index = 1;
      offset_index = 2;
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

      /* shared variable */
   case nir_intrinsic_load_shared:
      const_block_index = 0;
      offset_index = 0;
      break;
   case nir_intrinsic_store_shared:
      const_block_index = 0;
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
      const_block_index = 0;
      offset_index = 0;
      break;

   default:
      assert(!"not implemented");
   }

   assert((block_index >= 0 || const_block_index >= 0) &&
          (offset_index >= 0 || const_offset_index >= 0));

   if (block && block_index >= 0)
      *block = &instr->src[block_index];

   if (const_block && const_block_index >= 0)
      *const_block = instr->const_index[const_block_index];

   if (offset && offset_index >= 0)
      *offset = &instr->src[offset_index];

   if (const_offset && const_offset_index >= 0)
      *const_offset = instr->const_index[const_offset_index];
}

static bool
intrinsic_block_and_offset_match(nir_src *instr1_block,
                                 unsigned instr1_const_block,
                                 nir_src *instr1_offset,
                                 unsigned instr1_const_offset,
                                 nir_intrinsic_instr *instr2)
{
   nir_src *instr2_block = NULL;
   unsigned instr2_const_block = 0;
   nir_src *instr2_offset = NULL;
   unsigned instr2_const_offset = 0;

   get_load_store_address(instr2, &instr2_block, &instr2_const_block,
                          &instr2_offset, &instr2_const_offset);

   if (instr1_const_block != instr2_const_block)
      return false;

   if (instr1_const_offset != instr2_const_offset)
      return false;

   if (instr1_block && instr2_block &&
       (instr1_block->ssa->parent_instr->type !=
        instr2_block->ssa->parent_instr->type ||
        !nir_srcs_equal(*instr1_block, *instr2_block)))
      return false;

   if (instr1_offset && instr2_offset &&
       (instr1_offset->ssa->parent_instr->type !=
        instr2_offset->ssa->parent_instr->type ||
        !nir_srcs_equal(*instr1_offset, *instr2_offset)))
      return false;

   return true;
}

static inline bool
nir_src_is_undefined(nir_src src)
{
   return src.is_ssa &&
      src.ssa->parent_instr->type == nir_instr_type_ssa_undef;
}

/**
 * Returns true if two coordinates sources of an image intrinsic match.
 * This means that both sources are defined by ALU vec4 ops, with all
 * 4 sources equivalent. For exmaple, consider the following snippet:
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
nir_image_coordinates_match(nir_ssa_def *coord1, nir_ssa_def *coord2)
{
   nir_instr *parent1 = coord1->parent_instr;
   nir_instr *parent2 = coord2->parent_instr;

   /* @FIXME: by now, we assume that all coordinates into an image load/store
    * instruction, is given by an ALU vec4 instruction. This might not be
    * the case.
    */
   assert(parent1->type == nir_instr_type_alu);
   assert(parent2->type == nir_instr_type_alu);

   nir_alu_instr *alu1 = nir_instr_as_alu(parent1);
   nir_alu_instr *alu2 = nir_instr_as_alu(parent2);

   assert(alu1->op = nir_op_vec4);
   assert(alu2->op = nir_op_vec4);

   for (unsigned i = 0; i < 4; i++) {
      if (nir_src_is_undefined(alu1->src[i].src)) {
         if (!nir_src_is_undefined(alu2->src[i].src))
            return false;
      } else if (nir_src_is_undefined(alu2->src[i].src)) {
         if (!nir_src_is_undefined(alu1->src[i].src))
            return false;
      } else if (!nir_srcs_equal(alu1->src[i].src, alu2->src[i].src)) {
         return false;
      }
   }

   return true;
}

/**
 * Returns true if the two image instructions match, meaning their targets
 * are the same texture object and coordinates. Otherwise returns false.
 */
static bool
intrinsic_image_and_coordinates_match(nir_intrinsic_instr *instr1,
                                      nir_intrinsic_instr *instr2)
{
   assert(instr1->variables[0]);
   assert(instr2->variables[0]);

   if (instr1->variables[0]->var != instr2->variables[0]->var)
      return false;

   assert(instr1->src[0].is_ssa);
   assert(instr2->src[0].is_ssa);

   nir_ssa_def *coord1 = instr1->src[0].ssa;
   nir_ssa_def *coord2 = instr2->src[0].ssa;

   if (!nir_image_coordinates_match(coord1, coord2))
      return false;

   return true;
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

   nir_src *store_block = NULL;
   unsigned store_const_block = 0;
   nir_src *store_offset = NULL;
   unsigned store_const_offset = 0;

   if (intrinsic_group(store) != INTRINSIC_GROUP_IMAGE) {
      get_load_store_address(store, &store_block, &store_const_block,
                             &store_offset, &store_const_offset);
   }

   for (struct set_entry *entry = _mesa_set_next_entry(instr_set->set, NULL);
        entry != NULL; entry = _mesa_set_next_entry(instr_set->set, entry)) {

      assert(((nir_instr *) entry->key)->type == nir_instr_type_intrinsic);
      nir_intrinsic_instr *cached =
         nir_instr_as_intrinsic((nir_instr *) entry->key);

      /* intrinsic groups must match */
      if (!intrinsic_group_match(store, cached))
         continue;

      if (intrinsic_group(store) == INTRINSIC_GROUP_IMAGE) {
         if (!intrinsic_image_and_coordinates_match(store, cached))
            continue;
      } else {
         /* block and offset must match */
         if (!intrinsic_block_and_offset_match(store_block,
                                               store_const_block,
                                               store_offset,
                                               store_const_offset,
                                               cached)) {
            continue;
         }
      }

      /* @DEBUG: traces for invalidations by a store instruction */
      printf("Store instruction:\n");
      nir_print_instr(&store->instr, stderr); printf("\n");
      printf("invalidates:\n");
      nir_print_instr(&cached->instr, stderr); printf("\n------\n");

      nir_instr_set_remove(instr_set, (nir_instr *) entry->key);
   }
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
   unsigned load_const_block = 0;
   nir_src *load_offset = NULL;
   unsigned load_const_offset = 0;

   if (intrinsic_group(load) != INTRINSIC_GROUP_IMAGE) {
      get_load_store_address(load, &load_block, &load_const_block,
                             &load_offset, &load_const_offset);
   }

   for (struct set_entry *entry = _mesa_set_next_entry(instr_set->set, NULL);
        entry != NULL; entry = _mesa_set_next_entry(instr_set->set, entry)) {

      nir_instr *instr = (nir_instr *) entry->key;
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);

      if (!is_store(store))
         continue;

      /* We cannot rewrite with atomics, because the result of those is
       * the old value.
       */
      if (is_atomic(store))
         continue;

      /* intrinsic groups must match */
      if (!intrinsic_group_match(load, store))
         continue;

      if (intrinsic_group(load) != INTRINSIC_GROUP_IMAGE) {
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

         /* block and offset must match */
         if (!intrinsic_block_and_offset_match(load_block, load_const_block,
                                               load_offset, load_const_offset,
                                               store)) {
            continue;
         }
      } else {
         if (!intrinsic_image_and_coordinates_match(load, store))
            continue;
      }

      /* @DEBUG: traces for an image load rewritten by a previous store */
      printf("Load instruction:\n");
      nir_print_instr(&load->instr, stderr); printf("\n");
      printf("rewritten by previous store:\n");
      nir_print_instr(&store->instr, stderr); printf("\n------\n");

      nir_ssa_def *def = &load->dest.ssa;
      nir_ssa_def *new_def = store->src[0].ssa;
      nir_ssa_def_rewrite_uses(def, nir_src_for_ssa(new_def));

      return true;
   }

   return false;
}

/**
 * Traverses the set of load/store intrinsics trying to find a previous image
 * load instruction on the same object and coordinates, that can be reused
 * to eliminate the given image load.
 */
static bool
rewrite_image_load_with_load(struct nir_instr_set *instr_set,
                             nir_intrinsic_instr *load)
{
   for (struct set_entry *entry = _mesa_set_next_entry(instr_set->set, NULL);
        entry != NULL; entry = _mesa_set_next_entry(instr_set->set, entry)) {

      nir_instr *instr = (nir_instr *) entry->key;
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *prev_load = nir_instr_as_intrinsic(instr);
      if (!is_load(prev_load))
         continue;

      /* intrinsic groups must match */
      if (!intrinsic_group_match(load, prev_load))
         continue;

      /* Texture object and coordinates must match */
      if (!intrinsic_image_and_coordinates_match(load, prev_load))
         continue;

      /* @DEBUG: traces for an image load combined with a previous load */
      printf("Image load instruction:\n");
      nir_print_instr(&load->instr, stderr); printf("\n");
      printf("combined with:\n");
      nir_print_instr(&prev_load->instr, stderr); printf("\n------\n");

      nir_ssa_def *def = &load->dest.ssa;
      nir_ssa_def *new_def = &prev_load->dest.ssa;
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
         /* Image load instructions are handled differently than
          * SSBO and shared-var loads, because instead of being defined
          * by block/offset, they are defined by tex-obj/coordinates.
          * So we handle those separately.
          */
         if (intrinsic_group(intrinsic) == INTRINSIC_GROUP_IMAGE &&
             rewrite_image_load_with_load(instr_set, intrinsic)) {
            progress = true;
         } else if (nir_instr_set_add_or_rewrite(instr_set, instr)) {
            /* Try to rewrite with a previous load */
            progress = true;
            nir_instr_remove(instr);

            /* @DEBUG: traces for a load combined with a previous load */
            nir_instr *match = nir_instr_set_get_match(instr_set, instr);
            assert(match);
            printf("Load instruction:\n");
            nir_print_instr(instr, stderr); printf("\n");
            printf("combined with:\n");
            nir_print_instr(match, stderr); printf("\n------\n");
         } else if (rewrite_load_with_store(instr_set, intrinsic)) {
            progress = true;
         }
      } else if (is_store(intrinsic)) {
         /* Invalidate conflicting load/stores and add the store to the set
          * so we can rewrite future loads with it.
          */
         set_invalidate_for_store(instr_set, intrinsic);
         _mesa_set_add(instr_set->set, instr);
      } else if (is_memory_barrier(intrinsic)) {
         /* If we see a memory barrier we have to invalidate all cached
          * load/store operations
          */
         set_clear(instr_set);

         /* @DEBUG: traces for a memory barrier clearing the set */
         printf("Set fully invalidated due to memory barrier\n------\n");
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
