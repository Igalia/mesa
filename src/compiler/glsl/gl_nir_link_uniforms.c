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
#include "compiler/glsl/ir_uniform.h" /* for gl_uniform_storage */
#include "linker_util.h"
#include "main/context.h"
#include "main/mtypes.h"

/* This file do the common link for GLSL uniforms, using NIR, instead of IR as
 * the counter-part glsl/link_uniforms.cpp
 *
 * Also note that this is tailored for ARB_gl_spirv needs and particularities
 * (like need to work/link without name available, explicit location for
 * normal uniforms as mandatory, and so on).
 */

#define UNMAPPED_UNIFORM_LOC ~0u

static void
nir_setup_uniform_remap_tables(struct gl_context *ctx,
                               struct gl_shader_program *prog)
{
   prog->UniformRemapTable = rzalloc_array(prog,
                                           struct gl_uniform_storage *,
                                           prog->NumUniformRemapTable);
   union gl_constant_value *data =
      rzalloc_array(prog->data,
                    union gl_constant_value, prog->data->NumUniformDataSlots);
   if (!prog->UniformRemapTable || !data) {
      linker_error(prog, "Out of memory during linking.\n");
      return;
   }
   prog->data->UniformDataSlots = data;

   unsigned data_pos = 0;

   /* Reserve all the explicit locations of the active uniforms. */
   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      struct gl_uniform_storage *uniform = &prog->data->UniformStorage[i];

      if (prog->data->UniformStorage[i].remap_location == UNMAPPED_UNIFORM_LOC)
         continue;

      /* How many new entries for this uniform? */
      const unsigned entries = MAX2(1, uniform->array_elements);
      unsigned num_slots = glsl_get_component_slots(uniform->type);

      uniform->storage = &data[data_pos];

      /* Set remap table entries point to correct gl_uniform_storage. */
      for (unsigned j = 0; j < entries; j++) {
         unsigned element_loc = uniform->remap_location + j;
         prog->UniformRemapTable[element_loc] = uniform;

         data_pos += num_slots;
      }
   }

   /* Reserve locations for rest of the uniforms. */
   link_util_update_empty_uniform_locations(prog);

   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      struct gl_uniform_storage *uniform = &prog->data->UniformStorage[i];

      if (uniform->is_shader_storage)
         continue;

      /* Built-in uniforms should not get any location. */
      if (uniform->builtin)
         continue;

      /* Explicit ones have been set already. */
      if (uniform->remap_location != UNMAPPED_UNIFORM_LOC)
         continue;

      /* How many entries for this uniform? */
      const unsigned entries = MAX2(1, uniform->array_elements);

      unsigned location =
         link_util_find_empty_block(prog, &prog->data->UniformStorage[i]);

      if (location == -1) {
         location = prog->NumUniformRemapTable;

         /* resize remap table to fit new entries */
         prog->UniformRemapTable =
            reralloc(prog,
                     prog->UniformRemapTable,
                     struct gl_uniform_storage *,
                     prog->NumUniformRemapTable + entries);
         prog->NumUniformRemapTable += entries;
      }

      /* set the base location in remap table for the uniform */
      uniform->remap_location = location;

      unsigned num_slots = glsl_get_component_slots(uniform->type);

      uniform->storage = &data[data_pos];

      /* Set remap table entries point to correct gl_uniform_storage. */
      for (unsigned j = 0; j < entries; j++) {
         unsigned element_loc = uniform->remap_location + j;
         prog->UniformRemapTable[element_loc] = uniform;

         data_pos += num_slots;
      }
   }
}

static void
update_uniform_storage(struct gl_uniform_storage *uniform,
                       unsigned stage)
{
   uniform->active_shader_mask |= 1 << stage;
}

/**
 * Finds, return, and update the stage infor for any uniform at the
 * UniformStorage any uniform defined by @var. In general this is done using
 * the explicit location, except:
 *
 * * UBOs/SSBOs: as they lack explicit location, binding is used to locate
 *   them. That means that more that one entry at the uniform storage can be
 *   found. In that case all of them are updated, and the first entry is
 *   returned, in order to update the location of nir variable.
 *
 * * Expecial uniforms: like atomic counters. They lack a explicit location,
 *   so they are skipped, handled in any case, and assign a location later.
 *
 */
static struct gl_uniform_storage *
find_and_update_previous_uniform_storage(struct gl_shader_program *prog,
                                         nir_variable *var,
                                         unsigned stage)
{
   if (nir_variable_is_in_block(var)) {
      struct gl_uniform_storage *uniform = NULL;

      unsigned num_blks = nir_variable_is_in_ubo(var) ?
         prog->data->NumUniformBlocks :
         prog->data->NumShaderStorageBlocks;

      struct gl_uniform_block *blks = nir_variable_is_in_ubo(var) ?
         prog->data->UniformBlocks : prog->data->ShaderStorageBlocks;

      for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
         /* UniformStorage contains both variables from ubos and ssbos */
         if ( prog->data->UniformStorage[i].is_shader_storage !=
              nir_variable_is_in_ssbo(var))
            continue;

         int block_index = prog->data->UniformStorage[i].block_index;
         if (block_index != -1) {
            assert(block_index < num_blks);

            if (var->data.binding == blks[block_index].Binding) {
               if (!uniform)
                  uniform = &prog->data->UniformStorage[i];
               update_uniform_storage(&prog->data->UniformStorage[i],
                                      stage);
            }
         }
      }

      return uniform;
   }

   /* Beyond blocks, there are still some corner cases of uniforms without
    * location (ie: atomic counters) that would have a initial location equal
    * to -1. We just return on that case. Those uniforms will be handled
    * later.
    */
   if (var->data.location == -1)
      return NULL;

   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      if (prog->data->UniformStorage[i].remap_location == var->data.location) {
         update_uniform_storage(&prog->data->UniformStorage[i], stage);

         return &prog->data->UniformStorage[i];
      }
   }

   return NULL;
}

/* Used to build a tree representing the glsl_type so that we can have a place
 * to store the next index for opaque types. Array types are expanded so that
 * they have a single child which is used for all elements of the array.
 * Struct types have a child for each member. The tree is walked while
 * processing a uniform so that we can recognise when an opaque type is
 * encountered a second time in order to reuse the same range of indices that
 * was reserved the first time. That way the sampler indices can be arranged
 * so that members of an array are placed sequentially even if the array is an
 * array of structs containing other opaque members.
 */
struct type_tree_entry {
   /* For opaque types, this will be the next index to use. If we haven’t
    * encountered this member yet, it will be UINT_MAX.
    */
   unsigned next_index;
   unsigned array_size;
   struct type_tree_entry *parent;
   struct type_tree_entry *next_sibling;
   struct type_tree_entry *children;
};

struct nir_link_uniforms_state {
   /* per-whole program */
   unsigned num_hidden_uniforms;
   unsigned num_values;
   unsigned max_uniform_location;
   unsigned next_sampler_index;
   unsigned next_image_index;

   /* per-shader stage */
   unsigned num_shader_samplers;
   unsigned num_shader_images;
   unsigned num_shader_uniform_components;
   unsigned shader_samplers_used;
   unsigned shader_shadow_samplers;

   nir_variable *current_var;

   struct type_tree_entry *current_type;
};

static struct type_tree_entry *
build_type_tree_for_type(const struct glsl_type *type)
{
   struct type_tree_entry *entry = malloc(sizeof *entry);

   entry->array_size = 1;
   entry->next_index = UINT_MAX;
   entry->children = NULL;
   entry->next_sibling = NULL;
   entry->parent = NULL;

   if (glsl_type_is_array(type)) {
      entry->array_size = glsl_get_length(type);
      entry->children = build_type_tree_for_type(glsl_get_array_element(type));
      entry->children->parent = entry;
   } else if (glsl_type_is_struct(type)) {
      struct type_tree_entry *last = NULL;

      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         const struct glsl_type *field_type = glsl_get_struct_field(type, i);
         struct type_tree_entry *field_entry =
            build_type_tree_for_type(field_type);

         if (last == NULL)
            entry->children = field_entry;
         else
            last->next_sibling = field_entry;

         field_entry->parent = entry;

         last = field_entry;
      }
   }

   return entry;
}

static void
free_type_tree(struct type_tree_entry *entry)
{
   struct type_tree_entry *p, *next;

   for (p = entry->children; p; p = next) {
      next = p->next_sibling;
      free_type_tree(p);
   }

   free(entry);
}

static unsigned
get_next_index(struct nir_link_uniforms_state *state,
               const struct gl_uniform_storage *uniform,
               unsigned *next_index)
{
   /* If we’ve already calculated an index for this member then we can just
    * offset from there.
    */
   if (state->current_type->next_index == UINT_MAX) {
      /* Otherwise we need to reserve enough indices for all of the arrays
       * enclosing this member.
       */

      unsigned array_size = 1;

      for (const struct type_tree_entry *p = state->current_type;
           p;
           p = p->parent) {
         array_size *= p->array_size;
      }

      state->current_type->next_index = *next_index;
      *next_index += array_size;
   }

   unsigned index = state->current_type->next_index;

   state->current_type->next_index += MAX2(1, uniform->array_elements);

   return index;
}


/**
 * Creates the neccessary entries in UniformStorage for the uniform. Returns
 * the number of locations used or -1 on failure.
 */
static int
nir_link_uniform(struct gl_context *ctx,
                 struct gl_shader_program *prog,
                 struct gl_program *stage_program,
                 gl_shader_stage stage,
                 const struct glsl_type *type,
                 const struct glsl_type *parent_type,
                 unsigned index_in_parent,
                 int location,
                 int *offset,
                 struct nir_link_uniforms_state *state)
{
   struct gl_uniform_storage *uniform = NULL;

   /* gl_uniform_storage can cope with one level of array, so if the type is a
    * composite type or an array where each element occupies more than one
    * location than we need to recursively process it.
    */
   if (!_glsl_type_is_leaf(type)) {
      int location_count = 0;
      struct type_tree_entry *old_type = state->current_type;
      unsigned int struct_base_offset;

      state->current_type = old_type->children;

      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         const struct glsl_type *field_type;

         if (glsl_type_is_struct(type)) {
            field_type = glsl_get_struct_field(type, i);
            /* Use the offset inside the struct only for variables backed by
             * a buffer object. For variables not backed by a buffer object,
             * offset is -1.
             */
            if (nir_variable_is_in_block(state->current_var)) {
               if (i == 0)
                  struct_base_offset = *offset;
               *offset =
                  struct_base_offset + glsl_get_struct_field_offset(type, i);
            }
         } else {
            field_type = glsl_get_array_element(type);
         }

         int entries = nir_link_uniform(ctx, prog, stage_program, stage,
                                        field_type, type, i, location, offset,
                                        state);
         if (entries == -1)
            return -1;

         if (location != -1)
            location += entries;
         location_count += entries;

         if (glsl_type_is_struct(type))
            state->current_type = state->current_type->next_sibling;
      }

      state->current_type = old_type;

      return location_count;
   } else {
      /* Create a new uniform storage entry */
      prog->data->UniformStorage =
         reralloc(prog->data,
                  prog->data->UniformStorage,
                  struct gl_uniform_storage,
                  prog->data->NumUniformStorage + 1);
      if (!prog->data->UniformStorage) {
         linker_error(prog, "Out of memory during linking.\n");
         return -1;
      }

      uniform = &prog->data->UniformStorage[prog->data->NumUniformStorage];
      prog->data->NumUniformStorage++;

      /* Initialize its members */
      memset(uniform, 0x00, sizeof(struct gl_uniform_storage));
      /* ARB_gl_spirv: names are considered optional debug info, so the linker
       * needs to work without them, and returning them is optional. For
       * simplicity we ignore names.
       */
      uniform->name = NULL;

      const struct glsl_type *type_no_array = glsl_without_array(type);
      if (glsl_type_is_array(type)) {
         uniform->type = type_no_array;
         uniform->array_elements = glsl_get_length(type);
         uniform->array_stride = glsl_get_explicit_array_stride(type);
      } else {
         uniform->type = type;
         uniform->array_elements = 0;
         uniform->array_stride = 0;
      }
      uniform->active_shader_mask |= 1 << stage;

      if (location >= 0) {
         /* Uniform has an explicit location */
         uniform->remap_location = location;
      } else {
         uniform->remap_location = UNMAPPED_UNIFORM_LOC;
      }

      uniform->hidden = state->current_var->data.how_declared == nir_var_hidden;
      if (uniform->hidden)
         state->num_hidden_uniforms++;

      uniform->is_shader_storage = nir_variable_is_in_ssbo(state->current_var);

      if (nir_variable_is_in_block(state->current_var) &&
          glsl_type_is_matrix(type)) {
         assert(parent_type);

         uniform->matrix_stride =
            glsl_get_struct_field_explicit_matrix_stride(parent_type, index_in_parent);

         uniform->row_major =
            glsl_get_struct_field_matrix_layout(parent_type, index_in_parent) ==
            GLSL_MATRIX_LAYOUT_ROW_MAJOR;
      } else {
         uniform->matrix_stride = 0;
         uniform->row_major = false;
      }

      uniform->offset = *offset;

      int buffer_block_index = -1;
      if (nir_variable_is_in_block(state->current_var)) {
         /* Determine the block index by comparing the bindings, we can not use
          * names.
          */
         struct gl_uniform_block *blocks = nir_variable_is_in_ssbo(state->current_var) ?
            prog->data->ShaderStorageBlocks : prog->data->UniformBlocks;

         int num_blocks = nir_variable_is_in_ssbo(state->current_var) ?
            prog->data->NumShaderStorageBlocks : prog->data->NumUniformBlocks;

         for (unsigned i = 0; i < num_blocks; i++) {
            if (state->current_var->data.binding == blocks[i].Binding) {
               buffer_block_index = i;
            }
         }
         assert(buffer_block_index >= 0);

         /* Compute the next offset. */
         (*offset) += _get_type_size(type, parent_type, index_in_parent);
      }

      uniform->block_index = buffer_block_index;

      /* @FIXME: the initialization of the following will be done as we
       * implement support for their specific features, like SSBO, atomics,
       * etc.
       */
      uniform->builtin = false;
      uniform->atomic_buffer_index = -1;
      uniform->top_level_array_size = 0;
      uniform->top_level_array_stride = 0;
      uniform->is_bindless = false;

      /* The following are not for features not supported by ARB_gl_spirv */
      uniform->num_compatible_subroutines = 0;

      unsigned entries = MAX2(1, uniform->array_elements);

      if (glsl_type_is_sampler(type_no_array)) {
         int sampler_index =
            get_next_index(state, uniform, &state->next_sampler_index);

         state->num_shader_samplers++;

         uniform->opaque[stage].active = true;
         uniform->opaque[stage].index = sampler_index;

         const unsigned shadow = glsl_sampler_type_is_shadow(type_no_array);

         for (unsigned i = sampler_index;
              i < MIN2(state->next_sampler_index, MAX_SAMPLERS);
              i++) {
            stage_program->sh.SamplerTargets[i] =
               glsl_get_sampler_target(type_no_array);
            state->shader_samplers_used |= 1U << i;
            state->shader_shadow_samplers |= shadow << i;
         }
      } else if (glsl_type_is_image(type_no_array)) {
         /* @FIXME: image_index should match that of the same image
          * uniform in other shaders. This means we need to match image
          * uniforms by location (GLSL does it by variable name, but we
          * want to avoid that).
          */
         int image_index = state->next_image_index;
         state->next_image_index += entries;

         state->num_shader_images++;

         uniform->opaque[stage].active = true;
         uniform->opaque[stage].index = image_index;

         /* Set image access qualifiers */
         enum gl_access_qualifier image_access =
            state->current_var->data.image.access;
         const GLenum access =
            (image_access & ACCESS_NON_WRITEABLE) ?
            ((image_access & ACCESS_NON_READABLE) ? GL_NONE :
                                                    GL_READ_ONLY) :
            ((image_access & ACCESS_NON_READABLE) ? GL_WRITE_ONLY :
                                                    GL_READ_WRITE);
         for (unsigned i = image_index;
              i < MIN2(state->next_image_index, MAX_IMAGE_UNIFORMS);
              i++) {
            stage_program->sh.ImageAccess[i] = access;
         }
      }

      unsigned values = glsl_get_component_slots(type);
      state->num_shader_uniform_components += values;
      state->num_values += values;

      if (state->max_uniform_location < uniform->remap_location + entries)
         state->max_uniform_location = uniform->remap_location + entries;

      return MAX2(uniform->array_elements, 1);
   }
}

/*
 * We need this wrapper in order to maintain a new exec_node for a given
 * nir_variable, as this only maintain one, so can't belong to two different
 * lists.
 */
typedef struct wrapped_variable {
   struct exec_node node;
   nir_variable *var;
} wrapped_variable;

static struct wrapped_variable*
wrapped_variable_create(void *mem_ctx,
                          nir_variable *var)
{
   wrapped_variable *variable = rzalloc(mem_ctx, wrapped_variable);

   variable->var = var;

   return variable;
}


static nir_variable*
nir_search_variable(struct exec_list *variables,
                    unsigned location)
{
   foreach_list_typed(wrapped_variable, wrapped_var, node, variables) {
      if (wrapped_var->var->data.location == location)
         return wrapped_var->var;
   }
   return NULL;
}

static bool
nir_uniforms_are_compatible(struct gl_shader_program *prog,
                            nir_variable *new,
                            nir_variable *existing)
{
   /* FIXME: As we didn't modify how the names are created, right now
    * spirv_to_nir is bringing the names from SPIR-V to the types. So the
    * following check would not be true if the names are different on one
    * stage to the other. That is technically possible as SPIR-V names are
    * debug. To prevent that we would need to do a type-compatible check that
    * ignored the name. Avoiding that corner case for now.
    */
   if (existing->type != new->type) {
      linker_error(prog, "uniform with location %i declared as "
                   "type `%s' and type `%s'\n", new->data.location,
                   glsl_get_type_name(new->type),
                   glsl_get_type_name(existing->type));

      return false;
   }

   return true;
}

/*
 * Validate if all the uniforms on different stages are defined in a
 * consistent way.
 *
 * GLSL linker has a more complete validation method, so probably this bit
 * will be moved later to a more general one.
 */
static bool
nir_cross_validate_uniforms(struct gl_context *ctx,
                            struct gl_shader_program *prog)
{
   /* Already processed variables. GLSL equivalent uses glsl_symbol_table, for
    * now we just use a list
    */
   struct exec_list variables;
   exec_list_make_empty(&variables);

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[stage];
      if (!sh)
         continue;

      nir_shader *nir = sh->Program->nir;
      assert(nir);

      nir_foreach_variable(var, &nir->uniforms) {
         /* Although in general uniforms should have a explicit location,
          * there are still cases without them, like UBOs, uniform atomic
          * counters, etc*/
         if (var->data.location < 0)
            continue;

         /* We are not interested on cross-stage uniforms that are implicitly
          * introduced by the compiler
          */
         if (var->data.how_declared == nir_var_hidden)
            continue;

         /* We search on the list of processed variables if we have a variable
          * with the same location. We can't use name as GLSL linker, as
          * SPIR-V names are optional. See issue 12 at ARB_gl_spirv spec.
          */
         nir_variable *existing = nir_search_variable(&variables, var->data.location);
         if (existing != NULL) {
            if (!nir_uniforms_are_compatible(prog, var, existing)) {
               return false;
            }
         } else {
            wrapped_variable *wrapper = wrapped_variable_create(ctx, var);
            exec_list_push_tail(&variables, &wrapper->node);
         }
      }
   }

   exec_list_make_empty(&variables);

   return true;
}

bool
gl_nir_link_uniforms(struct gl_context *ctx,
                     struct gl_shader_program *prog)
{
   /* We do an initial cross-validation based on the nir variables, to avoid
    * the work of creating gl_uniform_storage variables for individual
    * shaders */
   if (!nir_cross_validate_uniforms(ctx, prog))
      return false;

   /* First free up any previous UniformStorage items */
   ralloc_free(prog->data->UniformStorage);
   prog->data->UniformStorage = NULL;
   prog->data->NumUniformStorage = 0;

   /* Iterate through all linked shaders */
   struct nir_link_uniforms_state state = {0,};

   for (unsigned shader_type = 0; shader_type < MESA_SHADER_STAGES; shader_type++) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[shader_type];
      if (!sh)
         continue;

      nir_shader *nir = sh->Program->nir;
      assert(nir);

      state.num_shader_samplers = 0;
      state.num_shader_images = 0;
      state.num_shader_uniform_components = 0;
      state.shader_samplers_used = 0;
      state.shader_shadow_samplers = 0;

      nir_foreach_variable(var, &nir->uniforms) {
         struct gl_uniform_storage *uniform = NULL;

         /* Check if the uniform has been processed already for other
          * stage. We don't need to check if the previous uniform is
          * compatible with the current one. That was already done at
          * nir_cross_validate_uniforms
          */
         uniform = find_and_update_previous_uniform_storage(prog, var, shader_type);
         if (uniform) {
            var->data.location = uniform - prog->data->UniformStorage;

            continue;
         }

         int location = var->data.location;
         /* From now on the variable’s location will be its uniform index */
         var->data.location = prog->data->NumUniformStorage;

         state.current_var = var;

         /*
          * From ARB_program_interface spec, issue (16):
          *
          * "RESOLVED: We will follow the default rule for enumerating block
          *  members in the OpenGL API, which is:
          *
          *  * If a variable is a member of an interface block without an
          *    instance name, it is enumerated using just the variable name.
          *
          *  * If a variable is a member of an interface block with an
          *    instance name, it is enumerated as "BlockName.Member", where
          *    "BlockName" is the name of the interface block (not the
          *    instance name) and "Member" is the name of the variable.
          *
          * For example, in the following code:
          *
          * uniform Block1 {
          *   int member1;
          * };
          * uniform Block2 {
          *   int member2;
          * } instance2;
          * uniform Block3 {
          *  int member3;
          * } instance3[2];  // uses two separate buffer bindings
          *
          * the three uniforms (if active) are enumerated as "member1",
          * "Block2.member2", and "Block3.member3"."
          *
          * Note that in the last example, with an array of ubo, only one
          * uniform is generated. For that reason, while unrolling the
          * uniforms of a ubo, or the variables of a ssbo, we need to treat
          * arrays of instance as a single block.
          */
         const struct glsl_type *type = var->type;
         if (nir_variable_is_in_block(var) &&
             glsl_type_is_array(type)) {
            type = glsl_without_array(type);
         }

         struct type_tree_entry *type_tree =
            build_type_tree_for_type(type);
         state.current_type = type_tree;

         int offset = nir_variable_is_in_block(var) ? 0 : -1;
         int res = nir_link_uniform(ctx, prog, sh->Program, shader_type, type,
                                    NULL, 0,
                                    location, &offset, &state);

         free_type_tree(type_tree);

         if (res == -1)
            return false;
      }

      sh->Program->SamplersUsed = state.shader_samplers_used;
      sh->shadow_samplers = state.shader_shadow_samplers;
      sh->Program->info.num_textures = state.num_shader_samplers;
      sh->Program->info.num_images = state.num_shader_images;
      sh->num_uniform_components = state.num_shader_uniform_components;
      sh->num_combined_uniform_components = sh->num_uniform_components;
   }

   prog->data->NumHiddenUniforms = state.num_hidden_uniforms;
   prog->NumUniformRemapTable = state.max_uniform_location;
   prog->data->NumUniformDataSlots = state.num_values;

   nir_setup_uniform_remap_tables(ctx, prog);
   gl_nir_set_uniform_initializers(ctx, prog);

   return true;
}
