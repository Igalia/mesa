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

static void
linker_error(struct gl_shader_program *prog, const char *fmt, ...)
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

   const nir_variable *current_var;
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
         linker_error(prog, "Out of memory during linking.\n");
         return false;
      }

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

         /* In this stage we only care for uniforms with explicit locations. */
         if (var->data.location == -1)
            continue;

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

   /* @TODO: Now process all the uniform with unspecified location (-1). */

   prog->data->NumHiddenUniforms = state.num_hidden_uniforms;
   prog->NumUniformRemapTable = state.max_uniform_location;
   prog->data->NumUniformDataSlots = state.num_values;

   nir_setup_uniform_remap_tables(ctx, prog);
}
