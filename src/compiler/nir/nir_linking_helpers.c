/*
 * Copyright © 2015 Intel Corporation
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
#include "util/set.h"
#include "util/hash_table.h"
#include "nir_linker.h"
#include "compiler/glsl/ir_uniform.h" /* for gl_uniform_storage */

/* This file contains various little helpers for doing simple linking in
 * NIR.  Eventually, we'll probably want a full-blown varying packing
 * implementation in here.  Right now, it just deletes unused things.
 */

/**
 * Returns the bits in the inputs_read, outputs_written, or
 * system_values_read bitfield corresponding to this variable.
 */
static uint64_t
get_variable_io_mask(nir_variable *var, gl_shader_stage stage)
{
   if (var->data.location < 0)
      return 0;

   unsigned location = var->data.patch ?
      var->data.location - VARYING_SLOT_PATCH0 : var->data.location;

   assert(var->data.mode == nir_var_shader_in ||
          var->data.mode == nir_var_shader_out ||
          var->data.mode == nir_var_system_value);
   assert(var->data.location >= 0);

   const struct glsl_type *type = var->type;
   if (nir_is_per_vertex_io(var, stage)) {
      assert(glsl_type_is_array(type));
      type = glsl_get_array_element(type);
   }

   unsigned slots = glsl_count_attribute_slots(type, false);
   return ((1ull << slots) - 1) << location;
}

static void
tcs_add_output_reads(nir_shader *shader, uint64_t *read, uint64_t *patches_read)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_foreach_block(block, function->impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intrin_instr =
                  nir_instr_as_intrinsic(instr);
               if (intrin_instr->intrinsic == nir_intrinsic_load_var &&
                   intrin_instr->variables[0]->var->data.mode ==
                   nir_var_shader_out) {

                  nir_variable *var = intrin_instr->variables[0]->var;
                  if (var->data.patch) {
                     patches_read[var->data.location_frac] |=
                        get_variable_io_mask(intrin_instr->variables[0]->var,
                                             shader->info.stage);
                  } else {
                     read[var->data.location_frac] |=
                        get_variable_io_mask(intrin_instr->variables[0]->var,
                                             shader->info.stage);
                  }
               }
            }
         }
      }
   }
}

static bool
remove_unused_io_vars(nir_shader *shader, struct exec_list *var_list,
                      uint64_t *used_by_other_stage,
                      uint64_t *used_by_other_stage_patches)
{
   bool progress = false;
   uint64_t *used;

   nir_foreach_variable_safe(var, var_list) {
      if (var->data.patch)
         used = used_by_other_stage_patches;
      else
         used = used_by_other_stage;

      if (var->data.location < VARYING_SLOT_VAR0 && var->data.location >= 0)
         continue;

      if (var->data.always_active_io)
         continue;

      uint64_t other_stage = used[var->data.location_frac];

      if (!(other_stage & get_variable_io_mask(var, shader->info.stage))) {
         /* This one is invalid, make it a global variable instead */
         var->data.location = 0;
         var->data.mode = nir_var_global;

         exec_node_remove(&var->node);
         exec_list_push_tail(&shader->globals, &var->node);

         progress = true;
      }
   }

   return progress;
}

bool
nir_remove_unused_varyings(nir_shader *producer, nir_shader *consumer)
{
   assert(producer->info.stage != MESA_SHADER_FRAGMENT);
   assert(consumer->info.stage != MESA_SHADER_VERTEX);

   uint64_t read[4] = { 0 }, written[4] = { 0 };
   uint64_t patches_read[4] = { 0 }, patches_written[4] = { 0 };

   nir_foreach_variable(var, &producer->outputs) {
      if (var->data.patch) {
         patches_written[var->data.location_frac] |=
            get_variable_io_mask(var, producer->info.stage);
      } else {
         written[var->data.location_frac] |=
            get_variable_io_mask(var, producer->info.stage);
      }
   }

   nir_foreach_variable(var, &consumer->inputs) {
      if (var->data.patch) {
         patches_read[var->data.location_frac] |=
            get_variable_io_mask(var, consumer->info.stage);
      } else {
         read[var->data.location_frac] |=
            get_variable_io_mask(var, consumer->info.stage);
      }
   }

   /* Each TCS invocation can read data written by other TCS invocations,
    * so even if the outputs are not used by the TES we must also make
    * sure they are not read by the TCS before demoting them to globals.
    */
   if (producer->info.stage == MESA_SHADER_TESS_CTRL)
      tcs_add_output_reads(producer, read, patches_read);

   bool progress = false;
   progress = remove_unused_io_vars(producer, &producer->outputs, read,
                                    patches_read);

   progress = remove_unused_io_vars(consumer, &consumer->inputs, written,
                                    patches_written) || progress;

   return progress;
}

static uint8_t
get_interp_type(nir_variable *var, bool default_to_smooth_interp)
{
   if (var->data.interpolation != INTERP_MODE_NONE)
      return var->data.interpolation;
   else if (default_to_smooth_interp)
      return INTERP_MODE_SMOOTH;
   else
      return INTERP_MODE_NONE;
}

#define INTERPOLATE_LOC_SAMPLE 0
#define INTERPOLATE_LOC_CENTROID 1
#define INTERPOLATE_LOC_CENTER 2

static uint8_t
get_interp_loc(nir_variable *var)
{
   if (var->data.sample)
      return INTERPOLATE_LOC_SAMPLE;
   else if (var->data.centroid)
      return INTERPOLATE_LOC_CENTROID;
   else
      return INTERPOLATE_LOC_CENTER;
}

static void
get_slot_component_masks_and_interp_types(struct exec_list *var_list,
                                          uint8_t *comps,
                                          uint8_t *interp_type,
                                          uint8_t *interp_loc,
                                          gl_shader_stage stage,
                                          bool default_to_smooth_interp)
{
   nir_foreach_variable_safe(var, var_list) {
      assert(var->data.location >= 0);

      /* Only remap things that aren't built-ins.
       * TODO: add TES patch support.
       */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < 32) {

         const struct glsl_type *type = var->type;
         if (nir_is_per_vertex_io(var, stage)) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         unsigned elements =
            glsl_get_vector_elements(glsl_without_array(type));

         bool dual_slot = glsl_type_is_dual_slot(glsl_without_array(type));
         unsigned slots = glsl_count_attribute_slots(type, false);
         unsigned comps_slot2 = 0;
         for (unsigned i = 0; i < slots; i++) {
            interp_type[location + i] =
               get_interp_type(var, default_to_smooth_interp);
            interp_loc[location + i] = get_interp_loc(var);

            if (dual_slot) {
               if (i & 1) {
                  comps[location + i] |= ((1 << comps_slot2) - 1);
               } else {
                  unsigned num_comps = 4 - var->data.location_frac;
                  comps_slot2 = (elements * 2) - num_comps;

                  /* Assume ARB_enhanced_layouts packing rules for doubles */
                  assert(var->data.location_frac == 0 ||
                         var->data.location_frac == 2);
                  assert(comps_slot2 <= 4);

                  comps[location + i] |=
                     ((1 << num_comps) - 1) << var->data.location_frac;
               }
            } else {
               comps[location + i] |=
                  ((1 << elements) - 1) << var->data.location_frac;
            }
         }
      }
   }
}

struct varying_loc
{
   uint8_t component;
   uint32_t location;
};

static void
remap_slots_and_components(struct exec_list *var_list, gl_shader_stage stage,
                           struct varying_loc (*remap)[4],
                           uint64_t *slots_used, uint64_t *out_slots_read)
 {
   uint64_t out_slots_read_tmp = 0;

   /* We don't touch builtins so just copy the bitmask */
   uint64_t slots_used_tmp =
      *slots_used & (((uint64_t)1 << (VARYING_SLOT_VAR0 - 1)) - 1);

   nir_foreach_variable(var, var_list) {
      assert(var->data.location >= 0);

      /* Only remap things that aren't built-ins */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < 32) {
         assert(var->data.location - VARYING_SLOT_VAR0 < 32);
         assert(remap[var->data.location - VARYING_SLOT_VAR0] >= 0);

         const struct glsl_type *type = var->type;
         if (nir_is_per_vertex_io(var, stage)) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         unsigned num_slots = glsl_count_attribute_slots(type, false);
         bool used_across_stages = false;
         bool outputs_read = false;

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         struct varying_loc *new_loc = &remap[location][var->data.location_frac];
         if (new_loc->location) {
            uint64_t slots = (((uint64_t)1 << num_slots) - 1) << var->data.location;
            if (slots & *slots_used)
               used_across_stages = true;

            if (slots & *out_slots_read)
               outputs_read = true;

            var->data.location = new_loc->location;
            var->data.location_frac = new_loc->component;
         }

         if (var->data.always_active_io) {
            /* We can't apply link time optimisations (specifically array
             * splitting) to these so we need to copy the existing mask
             * otherwise we will mess up the mask for things like partially
             * marked arrays.
             */
            if (used_across_stages) {
               slots_used_tmp |=
                  *slots_used & (((uint64_t)1 << num_slots) - 1) << var->data.location;
            }

            if (outputs_read) {
               out_slots_read_tmp |=
                  *out_slots_read & (((uint64_t)1 << num_slots) - 1) << var->data.location;
            }

         } else {
            for (unsigned i = 0; i < num_slots; i++) {
               if (used_across_stages)
                  slots_used_tmp |= (uint64_t)1 << (var->data.location + i);

               if (outputs_read)
                  out_slots_read_tmp |= (uint64_t)1 << (var->data.location + i);
            }
         }
      }
   }

   *slots_used = slots_used_tmp;
   *out_slots_read = out_slots_read_tmp;
}

/* If there are empty components in the slot compact the remaining components
 * as close to component 0 as possible. This will make it easier to fill the
 * empty components with components from a different slot in a following pass.
 */
static void
compact_components(nir_shader *producer, nir_shader *consumer, uint8_t *comps,
                   uint8_t *interp_type, uint8_t *interp_loc,
                   bool default_to_smooth_interp)
{
   struct exec_list *input_list = &consumer->inputs;
   struct exec_list *output_list = &producer->outputs;
   struct varying_loc remap[32][4] = {{{0}, {0}}};

   /* Create a cursor for each interpolation type */
   unsigned cursor[4] = {0};

   /* We only need to pass over one stage and we choose the consumer as it seems
    * to cause a larger reduction in instruction counts (tested on i965).
    */
   nir_foreach_variable(var, input_list) {

      /* Only remap things that aren't builtins.
       * TODO: add TES patch support.
       */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < 32) {

         /* We can't repack xfb varyings. */
         if (var->data.always_active_io)
            continue;

         const struct glsl_type *type = var->type;
         if (nir_is_per_vertex_io(var, consumer->info.stage)) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         /* Skip types that require more complex packing handling.
          * TODO: add support for these types.
          */
         if (glsl_type_is_array(type) ||
             glsl_type_is_dual_slot(type) ||
             glsl_type_is_matrix(type) ||
             glsl_type_is_struct(type) ||
             glsl_type_is_64bit(type))
            continue;

         /* We ignore complex types above and all other vector types should
          * have been split into scalar variables by the lower_io_to_scalar
          * pass. The only exeption should by OpenGL xfb varyings.
          */
         if (glsl_get_vector_elements(type) != 1)
            continue;

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         uint8_t used_comps = comps[location];

         /* If there are no empty components there is nothing more for us to do.
          */
         if (used_comps == 0xf)
            continue;

         bool found_new_offset = false;
         uint8_t interp = get_interp_type(var, default_to_smooth_interp);
         for (; cursor[interp] < 32; cursor[interp]++) {
            uint8_t cursor_used_comps = comps[cursor[interp]];

            /* We couldn't find anywhere to pack the varying continue on. */
            if (cursor[interp] == location &&
                (var->data.location_frac == 0 ||
                 cursor_used_comps & ((1 << (var->data.location_frac)) - 1)))
               break;

            /* We can only pack varyings with matching interpolation types */
            if (interp_type[cursor[interp]] != interp)
               continue;

            /* Interpolation loc must match also.
             * TODO: i965 can handle these if they don't match, but the
             * radeonsi nir backend handles everything as vec4s and so expects
             * this to be the same for all components. We could make this
             * check driver specfific or drop it if NIR ever become the only
             * radeonsi backend.
             */
            if (interp_loc[cursor[interp]] != get_interp_loc(var))
               continue;

            /* If the slot is empty just skip it for now, compact_var_list()
             * can be called after this function to remove empty slots for us.
             * TODO: finish implementing compact_var_list() requires array and
             * matrix splitting.
             */
            if (!cursor_used_comps)
               continue;

            uint8_t unused_comps = ~cursor_used_comps;

            for (unsigned i = 0; i < 4; i++) {
               uint8_t new_var_comps = 1 << i;
               if (unused_comps & new_var_comps) {
                  remap[location][var->data.location_frac].component = i;
                  remap[location][var->data.location_frac].location =
                     cursor[interp] + VARYING_SLOT_VAR0;

                  found_new_offset = true;

                  /* Turn off the mask for the component we are remapping */
                  if (comps[location] & 1 << var->data.location_frac) {
                     comps[location] ^= 1 << var->data.location_frac;
                     comps[cursor[interp]] |= new_var_comps;
                  }
                  break;
               }
            }

            if (found_new_offset)
               break;
         }
      }
   }

   uint64_t zero = 0;
   remap_slots_and_components(input_list, consumer->info.stage, remap,
                              &consumer->info.inputs_read, &zero);
   remap_slots_and_components(output_list, producer->info.stage, remap,
                              &producer->info.outputs_written,
                              &producer->info.outputs_read);
}

/* We assume that this has been called more-or-less directly after
 * remove_unused_varyings.  At this point, all of the varyings that we
 * aren't going to be using have been completely removed and the
 * inputs_read and outputs_written fields in nir_shader_info reflect
 * this.  Therefore, the total set of valid slots is the OR of the two
 * sets of varyings;  this accounts for varyings which one side may need
 * to read/write even if the other doesn't.  This can happen if, for
 * instance, an array is used indirectly from one side causing it to be
 * unsplittable but directly from the other.
 */
void
nir_compact_varyings(nir_shader *producer, nir_shader *consumer,
                     bool default_to_smooth_interp)
{
   assert(producer->info.stage != MESA_SHADER_FRAGMENT);
   assert(consumer->info.stage != MESA_SHADER_VERTEX);

   uint8_t comps[32] = {0};
   uint8_t interp_type[32] = {0};
   uint8_t interp_loc[32] = {0};

   get_slot_component_masks_and_interp_types(&producer->outputs, comps,
                                             interp_type, interp_loc,
                                             producer->info.stage,
                                             default_to_smooth_interp);
   get_slot_component_masks_and_interp_types(&consumer->inputs, comps,
                                             interp_type, interp_loc,
                                             consumer->info.stage,
                                             default_to_smooth_interp);

   compact_components(producer, consumer, comps, interp_type, interp_loc,
                      default_to_smooth_interp);
}

#define UNMAPPED_UNIFORM_LOC ~0u

void
nir_linker_error(struct gl_shader_program *prog, const char *fmt, ...)
{
   va_list ap;

   ralloc_strcat(&prog->data->InfoLog, "error: ");
   va_start(ap, fmt);
   ralloc_vasprintf_append(&prog->data->InfoLog, fmt, ap);
   va_end(ap);

   prog->data->LinkStatus = linking_failure;
}

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
      nir_linker_error(prog, "Out of memory during linking.\n");
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
      unsigned num_slots = glsl_get_components(uniform->type);

      uniform->storage = &data[data_pos];

      /* Set remap table entries point to correct gl_uniform_storage. */
      for (unsigned j = 0; j < entries; j++) {
         unsigned element_loc = uniform->remap_location + j;
         prog->UniformRemapTable[element_loc] = uniform;

         data_pos += num_slots;
      }
   }

   /* Reserve locations for rest of the uniforms. */
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

      /* How many new entries for this uniform? */
      const unsigned entries = MAX2(1, uniform->array_elements);

      /* @FIXME: By now, we add un-assigned unassigned uniform locations
       * to the end of the uniform file. We need to keep track of empty
       * locations and use them.
       */
      unsigned chosen_location = prog->NumUniformRemapTable;

      /* resize remap table to fit new entries */
      prog->UniformRemapTable =
         reralloc(prog,
                  prog->UniformRemapTable,
                  struct gl_uniform_storage *,
                  prog->NumUniformRemapTable + entries);
      prog->NumUniformRemapTable += entries;

      /* set the base location in remap table for the uniform */
      uniform->remap_location = chosen_location;

      unsigned num_slots = glsl_get_components(uniform->type);

      uniform->storage = &data[data_pos];

      /* Set remap table entries point to correct gl_uniform_storage. */
      for (unsigned j = 0; j < entries; j++) {
         unsigned element_loc = uniform->remap_location + j;
         prog->UniformRemapTable[element_loc] = uniform;

         data_pos += num_slots;
      }
   }
}

static struct gl_uniform_storage *
find_previous_uniform_storage(struct gl_shader_program *prog,
                              int location)
{
   /* This would only work for uniform with explicit location, as all the
    * uniforms without location (ie: atomic counters) would have a initial
    * location equal to -1. We early return in that case.
    */
   if (location == -1)
      return NULL;

   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++)
      if (prog->data->UniformStorage[i].remap_location == location)
         return &prog->data->UniformStorage[i];

   return NULL;
}

struct nir_link_uniforms_state {
   /* per-whole program */
   unsigned num_hidden_uniforms;
   unsigned num_values;
   unsigned max_uniform_location;
   unsigned shader_samplers_used;
   unsigned shader_shadow_samplers;
   unsigned next_sampler_index;
   unsigned next_image_index;

   /* per-shader stage */
   unsigned num_shader_samplers;
   unsigned num_shader_images;
   unsigned num_shader_uniform_components;

   nir_variable *current_var;
};

static bool
nir_link_uniform (struct gl_context *ctx,
                  struct gl_shader_program *prog,
                  struct gl_program *stage_program,
                  gl_shader_stage stage,
                  const struct glsl_type *type,
                  const char *name,
                  int location,
                  struct nir_link_uniforms_state *state)
{
   struct gl_uniform_storage *uniform = NULL;
   if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         const struct glsl_type *field_type = glsl_get_struct_field(type, i);

         const char *field_name = glsl_get_struct_elem_name(type, i);
         char *uniform_name = NULL;
         if (name)
            asprintf(&uniform_name, "%s.%s", name, field_name);
         else
            uniform_name = strdup(field_name);

         unsigned entries = MAX2(1, glsl_get_length(field_type));
         if (!nir_link_uniform(ctx, prog, stage_program, stage,
                               field_type, uniform_name, location,
                               state)) {
            return false;
         }

         location += entries;
         free(uniform_name);
      }
   } else {
      /* Create a new uniform storage entry */
      prog->data->UniformStorage =
         reralloc(prog->data,
                  prog->data->UniformStorage,
                  struct gl_uniform_storage,
                  prog->data->NumUniformStorage + 1);
      if (!prog->data->UniformStorage) {
         nir_linker_error(prog, "Out of memory during linking.\n");
         return false;
      }

      if (state->current_var->data.location == location)
         state->current_var->data.location = prog->data->NumUniformStorage;

      uniform = &prog->data->UniformStorage[prog->data->NumUniformStorage];
      prog->data->NumUniformStorage++;

      /* Initialize its members */
      memset(uniform, 0x00, sizeof(struct gl_uniform_storage));
      uniform->name = ralloc_strdup(prog, name ? name : "");

      const struct glsl_type *type_no_array = glsl_without_array(type);
      if (glsl_type_is_array(type))
         uniform->type = type_no_array;
      else
         uniform->type = type;
      uniform->array_elements = glsl_get_length(type);
      uniform->active_shader_mask |= 1 << stage;
      if (location >= 0) {
         /* Uniform has an explicit location */
         uniform->remap_location = location;
      } else {
         uniform->remap_location = UNMAPPED_UNIFORM_LOC;
      }

      /* @FIXME: Pending to initialize the following members */
      uniform->block_index = -1;
      uniform->offset = -1;
      uniform->matrix_stride = -1;
      uniform->array_stride = -1;
      uniform->row_major = false;
      uniform->hidden = false;
      uniform->builtin = false;
      uniform->is_shader_storage = false;
      uniform->atomic_buffer_index = -1;
      uniform->num_compatible_subroutines = 0;
      uniform->top_level_array_size = 0;
      uniform->top_level_array_stride = 0;
      uniform->is_bindless = false;

      unsigned entries = MAX2(1, uniform->array_elements);

      if (glsl_type_is_sampler(type_no_array)) {
         /* @FIXME: sampler_index should match that of the same sampler
          * uniform in other shaders. This means we need to match sampler
          * uniforms by location (GLSL does it by variable name, but we
          * want to avoid that).
          */
         int sampler_index = state->next_sampler_index;
         state->next_sampler_index += entries;

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
         const GLenum access =
            (state->current_var->data.image.read_only ? GL_READ_ONLY :
             state->current_var->data.image.write_only ? GL_WRITE_ONLY :
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
   }

   return true;
}

void
nir_link_uniforms(struct gl_context *ctx,
                  struct gl_shader_program *prog)
{
   /* First free up any previous UniformStorage items */
   ralloc_free(prog->data->UniformStorage);
   prog->data->UniformStorage = NULL;
   prog->data->NumUniformStorage = 0;

   /* Iterate through all linked shaders */
   struct nir_link_uniforms_state state = {0,};

   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[i];
      if (!sh)
         continue;

      nir_shader *nir = sh->Program->nir;
      assert(nir);

      state.num_shader_samplers = 0;
      state.num_shader_images = 0;
      state.num_shader_uniform_components = 0;

      nir_foreach_variable(var, &nir->uniforms) {
         struct gl_uniform_storage *uniform = NULL;

         /* Check if the uniform has been processed already for
          * other stage. If so, validate they are compatible and update
          * the active stage mask.
          */
         uniform = find_previous_uniform_storage(prog, var->data.location);
         if (uniform) {
            /* @FIXME: Perform compatibility checks between the uniforms of
             * this stage and the one processed earlier.
             */

            uniform->active_shader_mask |= 1 << i;

            continue;
         }

         state.current_var = var;

         if (!nir_link_uniform(ctx, prog, sh->Program, i, var->type,
                               var->name, var->data.location, &state)) {
            return;
         }
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
}

/* @FIXME: copied verbatim from linker.cpp, needs refactoring. */
static bool
add_program_resource(struct gl_shader_program *prog,
                     struct set *resource_set,
                     GLenum type, const void *data, uint8_t stages)
{
   assert(data);

   /* If resource already exists, do not add it again. */
   if (_mesa_set_search(resource_set, data))
      return true;

   prog->data->ProgramResourceList =
      reralloc(prog,
               prog->data->ProgramResourceList,
               struct gl_program_resource,
               prog->data->NumProgramResourceList + 1);

   if (!prog->data->ProgramResourceList) {
      nir_linker_error(prog, "Out of memory during linking.\n");
      return false;
   }

   struct gl_program_resource *res =
      &prog->data->ProgramResourceList[prog->data->NumProgramResourceList];

   res->Type = type;
   res->Data = data;
   res->StageReferences = stages;

   prog->data->NumProgramResourceList++;

   _mesa_set_add(resource_set, data);

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

   struct set *resource_set = _mesa_set_create(NULL,
                                               _mesa_hash_pointer,
                                               _mesa_key_pointer_equal);

   /* Add uniforms
    *
    * Here, it is expected that nir_link_uniforms() has already been
    * called, so that UniformStorage table is already available.
    */
   for (unsigned i = 0; i < prog->data->NumUniformStorage; i++) {
      struct gl_uniform_storage *uniform = &prog->data->UniformStorage[i];

      if (!add_program_resource(prog, resource_set, GL_UNIFORM, uniform,
                                uniform->active_shader_mask)) {
         return;
      }
   }

   /* Add inputs */
   struct gl_linked_shader *sh = prog->_LinkedShaders[MESA_SHADER_VERTEX];
   if (sh) {
      nir_shader *nir = sh->Program->nir;
      assert(nir);

      nir_foreach_variable(var, &nir->inputs) {
         struct gl_shader_variable *sh_var =
            rzalloc(prog, struct gl_shader_variable);

         sh_var->name = ralloc_strdup(sh_var, var->name ? var->name : "");
         sh_var->type = var->type;
         sh_var->location = var->data.location;

         /* @TODO: Fill in the rest of gl_shader_variable data. */

         if (!add_program_resource(prog, resource_set, GL_PROGRAM_INPUT,
                                   sh_var, 1 << MESA_SHADER_VERTEX)) {
            return;
         }
      }
   }

   /* Add program uniform blocks. */
   for (unsigned i = 0; i < prog->data->NumUniformBlocks; i++) {
      if (!add_program_resource(prog, resource_set, GL_UNIFORM_BLOCK,
                                &prog->data->UniformBlocks[i], 0))
         return;
   }

   _mesa_set_destroy(resource_set, NULL);
}

struct active_atomic_counter_uniform {
   unsigned loc;
   nir_variable *var;
};

struct active_atomic_buffer {
   struct active_atomic_counter_uniform *uniforms;
   unsigned num_uniforms;
   unsigned uniform_buffer_size;
   unsigned stage_counter_references[MESA_SHADER_STAGES];
   unsigned size;
};

static void
add_atomic_counter(const void *ctx,
                   struct active_atomic_buffer *buffer,
                   unsigned uniform_loc,
                   nir_variable *var)
{
   if (buffer->num_uniforms >= buffer->uniform_buffer_size) {
      if (buffer->uniform_buffer_size == 0)
         buffer->uniform_buffer_size = 1;
      else
         buffer->uniform_buffer_size *= 2;
      buffer->uniforms = reralloc(ctx,
                                  buffer->uniforms,
                                  struct active_atomic_counter_uniform,
                                  buffer->uniform_buffer_size);
   }

   struct active_atomic_counter_uniform *uniform =
      buffer->uniforms + buffer->num_uniforms;
   uniform->loc = uniform_loc;
   uniform->var = var;
   buffer->num_uniforms++;
}

static void
process_atomic_variable(const struct glsl_type *t,
                        struct gl_shader_program *prog,
                        unsigned *uniform_loc,
                        nir_variable *var,
                        struct active_atomic_buffer *buffers,
                        unsigned *num_buffers,
                        int *offset,
                        unsigned shader_stage)
{
   /* FIXME: Arrays of arrays get counted separately. For example:
    * x1[3][3][2] = 9 uniforms, 18 atomic counters
    * x2[3][2]    = 3 uniforms, 6 atomic counters
    * x3[2]       = 1 uniform, 2 atomic counters
    *
    * However this code marks all the counters as active even when they
    * might not be used.
    */
   if (glsl_type_is_array(t) &&
       glsl_type_is_array(glsl_get_array_element(t))) {
      for (unsigned i = 0; i < glsl_get_length(t); i++) {
         process_atomic_variable(glsl_get_array_element(t),
                                 prog,
                                 uniform_loc,
                                 var,
                                 buffers, num_buffers,
                                 offset,
                                 shader_stage);
      }
   } else {
      struct active_atomic_buffer *buf = buffers + var->data.binding;
      struct gl_uniform_storage *const storage =
         &prog->data->UniformStorage[*uniform_loc];

      /* If this is the first time the buffer is used, increment
       * the counter of buffers used.
       */
      if (buf->size == 0)
         (*num_buffers)++;

      add_atomic_counter(buffers, /* ctx */
                         buf,
                         *uniform_loc,
                         var);

      /* When checking for atomic counters we should count every member in
       * an array as an atomic counter reference.
       */
      if (glsl_type_is_array(t))
         buf->stage_counter_references[shader_stage] += glsl_get_length(t);
      else
         buf->stage_counter_references[shader_stage]++;
      buf->size = MAX2(buf->size, *offset + glsl_atomic_size(t));

      storage->offset = *offset;
      *offset += glsl_atomic_size(t);

      (*uniform_loc)++;
   }
}

static int
cmp_actives(const void *a, const void *b)
{
   const struct active_atomic_counter_uniform *const first =
      (struct active_atomic_counter_uniform *) a;
   const struct active_atomic_counter_uniform *const second =
      (struct active_atomic_counter_uniform *) b;

   return (int) first->var->data.offset - (int) second->var->data.offset;
}

static bool
check_atomic_counters_overlap(const nir_variable *x,
                              const nir_variable *y)
{
   return ((x->data.offset >= y->data.offset &&
            x->data.offset < y->data.offset + glsl_atomic_size(y->type)) ||
           (y->data.offset >= x->data.offset &&
            y->data.offset < x->data.offset + glsl_atomic_size(x->type)));
}

static struct active_atomic_buffer *
find_active_atomic_counters(struct gl_context *ctx,
                            struct gl_shader_program *prog,
                            unsigned *num_buffers)
{
   struct active_atomic_buffer *buffers =
      rzalloc_array(NULL, /* ctx */
                    struct active_atomic_buffer,
                    ctx->Const.MaxAtomicBufferBindings);
   *num_buffers = 0;

   for (unsigned i = 0; i < MESA_SHADER_STAGES; ++i) {
      struct gl_linked_shader *sh = prog->_LinkedShaders[i];
      if (sh == NULL)
         continue;

      nir_shader *nir = sh->Program->nir;

      nir_foreach_variable(var, &nir->uniforms) {
         if (!glsl_contains_atomic(var->type))
            continue;

         int offset = var->data.offset;
         unsigned uniform_loc = var->data.location;

         process_atomic_variable(var->type,
                                 prog,
                                 &uniform_loc,
                                 var,
                                 buffers,
                                 num_buffers,
                                 &offset,
                                 i);
      }
   }

   for (unsigned i = 0; i < ctx->Const.MaxAtomicBufferBindings; i++) {
      if (buffers[i].size == 0)
         continue;

      qsort(buffers[i].uniforms,
            buffers[i].num_uniforms,
            sizeof (struct active_atomic_counter_uniform),
            cmp_actives);

      for (unsigned j = 1; j < buffers[i].num_uniforms; j++) {
         /* If an overlapping counter found, it must be a reference to the
          * same counter from a different shader stage.
          *
          * TODO: What about uniforms with no name?
          */
         if (check_atomic_counters_overlap(buffers[i].uniforms[j - 1].var,
                                           buffers[i].uniforms[j].var) &&
             buffers[i].uniforms[j - 1].var->name &&
             buffers[i].uniforms[j].var->name &&
             strcmp(buffers[i].uniforms[j - 1].var->name,
                    buffers[i].uniforms[j].var->name) != 0) {
            nir_linker_error(prog,
                             "Atomic counter %s declared at offset %d which is "
                             "already in use.",
                             buffers[i].uniforms[j].var->name,
                             buffers[i].uniforms[j].var->data.offset);
         }
      }
   }

   return buffers;
}

void
nir_link_assign_atomic_counter_resources(struct gl_context *ctx,
                                         struct gl_shader_program *prog)
{
   unsigned num_buffers;
   unsigned num_atomic_buffers[MESA_SHADER_STAGES] = { };
   struct active_atomic_buffer *abs =
      find_active_atomic_counters(ctx, prog, &num_buffers);

   prog->data->AtomicBuffers =
      rzalloc_array(prog->data, struct gl_active_atomic_buffer, num_buffers);
   prog->data->NumAtomicBuffers = num_buffers;

   unsigned buffer_idx = 0;
   for (unsigned binding = 0;
        binding < ctx->Const.MaxAtomicBufferBindings;
        binding++) {

      /* If the binding was not used, skip.
       */
      if (abs[binding].size == 0)
         continue;

      struct active_atomic_buffer *ab = abs + binding;
      struct gl_active_atomic_buffer *mab =
         prog->data->AtomicBuffers + buffer_idx;

      /* Assign buffer-specific fields. */
      mab->Binding = binding;
      mab->MinimumSize = ab->size;
      mab->Uniforms = rzalloc_array(prog->data->AtomicBuffers, GLuint,
                                    ab->num_uniforms);
      mab->NumUniforms = ab->num_uniforms;

      /* Assign counter-specific fields. */
      for (unsigned j = 0; j < ab->num_uniforms; j++) {
         nir_variable *var = ab->uniforms[j].var;
         struct gl_uniform_storage *storage =
            &prog->data->UniformStorage[ab->uniforms[j].loc];

         mab->Uniforms[j] = ab->uniforms[j].loc;
         /* FIXME: this was in the previous GLSL IR linker, but I don’t think
          * it’s neccessary because if there was no explicit binding then
          * binding would be zero and it will just work.
          *
          * if (!var->data.explicit_binding)
          *    var->data.binding = buffer_idx;
          */

         storage->atomic_buffer_index = buffer_idx;
         storage->offset = var->data.offset;
         if (glsl_type_is_array(var->type)) {
            const struct glsl_type *without_array =
               glsl_without_array(var->type);
            storage->array_stride = glsl_atomic_size(without_array);
         } else {
            storage->array_stride = 0;
         }
         if (!glsl_type_is_matrix(var->type))
            storage->matrix_stride = 0;
      }

      /* Assign stage-specific fields. */
      for (unsigned stage = 0; stage < MESA_SHADER_STAGES; ++stage) {
         if (ab->stage_counter_references[stage]) {
            mab->StageReferences[stage] = GL_TRUE;
            num_atomic_buffers[stage]++;
         } else {
            mab->StageReferences[stage] = GL_FALSE;
         }
      }

      buffer_idx++;
   }

   /* Store a list pointers to atomic buffers per stage and store the index
    * to the intra-stage buffer list in uniform storage.
    */
   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; ++stage) {
      if (prog->_LinkedShaders[stage] == NULL ||
          num_atomic_buffers[stage] <= 0)
         continue;

      struct gl_program *gl_prog = prog->_LinkedShaders[stage]->Program;
      gl_prog->info.num_abos = num_atomic_buffers[stage];
      gl_prog->sh.AtomicBuffers =
         rzalloc_array(gl_prog,
                       struct gl_active_atomic_buffer *,
                       num_atomic_buffers[stage]);

      gl_prog->nir->info.num_abos = num_atomic_buffers[stage];

      unsigned intra_stage_idx = 0;
      for (unsigned i = 0; i < num_buffers; i++) {
         struct gl_active_atomic_buffer *atomic_buffer =
            &prog->data->AtomicBuffers[i];
         if (!atomic_buffer->StageReferences[stage])
            continue;

         gl_prog->sh.AtomicBuffers[intra_stage_idx] = atomic_buffer;

         for (unsigned u = 0; u < atomic_buffer->NumUniforms; u++) {
            GLuint uniform_loc = atomic_buffer->Uniforms[u];
            struct gl_opaque_uniform_index *opaque =
               prog->data->UniformStorage[uniform_loc].opaque + stage;
            opaque->index = intra_stage_idx;
            opaque->active = true;
         }

         intra_stage_idx++;
      }
   }

   assert(buffer_idx == num_buffers);

   ralloc_free(abs);
}
