/*
 * Copyright © 2015 Red Hat
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "st_nir.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_context.h"

#include "program/program.h"
#include "program/prog_statevars.h"
#include "program/prog_parameter.h"
#include "program/ir_to_mesa.h"
#include "main/mtypes.h"
#include "main/errors.h"
#include "main/glspirv.h"
#include "main/shaderapi.h"
#include "main/uniforms.h"

#include "main/shaderobj.h"
#include "st_context.h"
#include "st_glsl_types.h"
#include "st_program.h"
#include "st_shader_cache.h"

#include "compiler/nir/nir.h"
#include "compiler/glsl_types.h"
#include "compiler/glsl/glsl_to_nir.h"
#include "compiler/glsl/gl_nir.h"
#include "compiler/glsl/gl_nir_linker.h"
#include "compiler/glsl/ir.h"
#include "compiler/glsl/ir_optimization.h"
#include "compiler/glsl/string_to_uint_map.h"

static int
type_size(const struct glsl_type *type)
{
   return type->count_attribute_slots(false);
}

/* Depending on PIPE_CAP_TGSI_TEXCOORD (st->needs_texcoord_semantic) we
 * may need to fix up varying slots so the glsl->nir path is aligned
 * with the anything->tgsi->nir path.
 */
static void
st_nir_fixup_varying_slots(struct st_context *st, struct exec_list *var_list)
{
   if (st->needs_texcoord_semantic)
      return;

   nir_foreach_variable(var, var_list) {
      if (var->data.location >= VARYING_SLOT_VAR0) {
         var->data.location += 9;
      } else if ((var->data.location >= VARYING_SLOT_TEX0) &&
               (var->data.location <= VARYING_SLOT_TEX7)) {
         var->data.location += VARYING_SLOT_VAR0 - VARYING_SLOT_TEX0;
      }
   }
}

/* input location assignment for VS inputs must be handled specially, so
 * that it is aligned w/ st's vbo state.
 * (This isn't the case with, for ex, FS inputs, which only need to agree
 * on varying-slot w/ the VS outputs)
 */
void
st_nir_assign_vs_in_locations(struct nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX)
      return;

   bool removed_inputs = false;

   nir->num_inputs = util_bitcount64(nir->info.inputs_read);
   nir_foreach_variable_safe(var, &nir->inputs) {
      /* NIR already assigns dual-slot inputs to two locations so all we have
       * to do is compact everything down.
       */
      if (nir->info.inputs_read & BITFIELD64_BIT(var->data.location)) {
         var->data.driver_location =
            util_bitcount64(nir->info.inputs_read &
                              BITFIELD64_MASK(var->data.location));
      } else {
         /* Move unused input variables to the globals list (with no
          * initialization), to avoid confusing drivers looking through the
          * inputs array and expecting to find inputs with a driver_location
          * set.
          */
         exec_node_remove(&var->node);
         var->data.mode = nir_var_shader_temp;
         exec_list_push_tail(&nir->globals, &var->node);
         removed_inputs = true;
      }
   }

   /* Re-lower global vars, to deal with any dead VS inputs. */
   if (removed_inputs)
      NIR_PASS_V(nir, nir_lower_global_vars_to_local);
}

static int
st_nir_lookup_parameter_index(struct gl_program *prog, nir_variable *var)
{
   struct gl_program_parameter_list *params = prog->Parameters;

   /* Lookup the first parameter that the uniform storage that match the
    * variable location.
    */
   for (unsigned i = 0; i < params->NumParameters; i++) {
      int index = params->Parameters[i].MainUniformStorageIndex;
      if (index == var->data.location)
         return i;
   }

   /* TODO: Handle this fallback for SPIR-V.  We need this for GLSL e.g. in
    * dEQP-GLES2.functional.uniform_api.random.3
    */

   /* is there a better way to do this?  If we have something like:
    *
    *    struct S {
    *           float f;
    *           vec4 v;
    *    };
    *    uniform S color;
    *
    * Then what we get in prog->Parameters looks like:
    *
    *    0: Name=color.f, Type=6, DataType=1406, Size=1
    *    1: Name=color.v, Type=6, DataType=8b52, Size=4
    *
    * So the name doesn't match up and _mesa_lookup_parameter_index()
    * fails.  In this case just find the first matching "color.*"..
    *
    * Note for arrays you could end up w/ color[n].f, for example.
    *
    * glsl_to_tgsi works slightly differently in this regard.  It is
    * emitting something more low level, so it just translates the
    * params list 1:1 to CONST[] regs.  Going from GLSL IR to TGSI,
    * it just calculates the additional offset of struct field members
    * in glsl_to_tgsi_visitor::visit(ir_dereference_record *ir) or
    * glsl_to_tgsi_visitor::visit(ir_dereference_array *ir).  It never
    * needs to work backwards to get base var loc from the param-list
    * which already has them separated out.
    */
   if (!prog->sh.data->spirv) {
      int namelen = strlen(var->name);
      for (unsigned i = 0; i < params->NumParameters; i++) {
         struct gl_program_parameter *p = &params->Parameters[i];
         if ((strncmp(p->Name, var->name, namelen) == 0) &&
             ((p->Name[namelen] == '.') || (p->Name[namelen] == '['))) {
            return i;
         }
      }
   }

   return -1;
}

static void
st_nir_assign_uniform_locations(struct gl_context *ctx,
                                struct gl_program *prog,
                                struct exec_list *uniform_list)
{
   int shaderidx = 0;
   int imageidx = 0;

   nir_foreach_variable(uniform, uniform_list) {
      int loc;

      /*
       * UBO's have their own address spaces, so don't count them towards the
       * number of global uniforms
       */
      if (uniform->data.mode == nir_var_mem_ubo || uniform->data.mode == nir_var_mem_ssbo)
         continue;

      const struct glsl_type *type = glsl_without_array(uniform->type);
      if (!uniform->data.bindless && (type->is_sampler() || type->is_image())) {
         if (type->is_sampler()) {
            loc = shaderidx;
            shaderidx += type_size(uniform->type);
         } else {
            loc = imageidx;
            imageidx += type_size(uniform->type);
         }
      } else if (uniform->state_slots) {
         const gl_state_index16 *const stateTokens = uniform->state_slots[0].tokens;
         /* This state reference has already been setup by ir_to_mesa, but we'll
          * get the same index back here.
          */

         unsigned comps;
         if (glsl_type_is_struct_or_ifc(type)) {
            comps = 4;
         } else {
            comps = glsl_get_vector_elements(type);
         }

         if (ctx->Const.PackedDriverUniformStorage) {
            loc = _mesa_add_sized_state_reference(prog->Parameters,
                                                  stateTokens, comps, false);
            loc = prog->Parameters->ParameterValueOffset[loc];
         } else {
            loc = _mesa_add_state_reference(prog->Parameters, stateTokens);
         }
      } else {
         loc = st_nir_lookup_parameter_index(prog, uniform);

         /* We need to check that loc is not -1 here before accessing the
          * array. It can be negative for example when we have a struct that
          * only contains opaque types.
          */
         if (loc >= 0 && ctx->Const.PackedDriverUniformStorage) {
            loc = prog->Parameters->ParameterValueOffset[loc];
         }
      }

      uniform->data.driver_location = loc;
   }
}

void
st_nir_opts(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS_V(nir, nir_lower_vars_to_ssa);
      
      /* Linking deals with unused inputs/outputs, but here we can remove
       * things local to the shader in the hopes that we can cleanup other
       * things. This pass will also remove variables with only stores, so we
       * might be able to make progress after it.
       */
      NIR_PASS(progress, nir, nir_remove_dead_variables,
               (nir_variable_mode)(nir_var_function_temp |
                                   nir_var_shader_temp |
                                   nir_var_mem_shared));

      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_dead_write_vars);

      if (nir->options->lower_to_scalar) {
         NIR_PASS_V(nir, nir_lower_alu_to_scalar, NULL, NULL);
         NIR_PASS_V(nir, nir_lower_phis_to_scalar);
      }

      NIR_PASS_V(nir, nir_lower_alu);
      NIR_PASS_V(nir, nir_lower_pack);
      NIR_PASS(progress, nir, nir_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      if (nir_opt_trivial_continues(nir)) {
         progress = true;
         NIR_PASS(progress, nir, nir_copy_prop);
         NIR_PASS(progress, nir, nir_opt_dce);
      }
      NIR_PASS(progress, nir, nir_opt_if, false);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_peephole_select, 8, true, true);

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      if (!nir->info.flrp_lowered) {
         unsigned lower_flrp =
            (nir->options->lower_flrp16 ? 16 : 0) |
            (nir->options->lower_flrp32 ? 32 : 0) |
            (nir->options->lower_flrp64 ? 64 : 0);

         if (lower_flrp) {
            bool lower_flrp_progress = false;

            NIR_PASS(lower_flrp_progress, nir, nir_lower_flrp,
                     lower_flrp,
                     false /* always_precise */,
                     nir->options->lower_ffma);
            if (lower_flrp_progress) {
               NIR_PASS(progress, nir,
                        nir_opt_constant_folding);
               progress = true;
            }
         }

         /* Nothing should rematerialize any flrps, so we only need to do this
          * lowering once.
          */
         nir->info.flrp_lowered = true;
      }

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_conditional_discard);
      if (nir->options->max_unroll_iterations) {
         NIR_PASS(progress, nir, nir_opt_loop_unroll, (nir_variable_mode)0);
      }
   } while (progress);
}

static void
shared_type_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type)
      ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length,
   *align = comp_size * (length == 3 ? 4 : length);
}

/* First third of converting glsl_to_nir.. this leaves things in a pre-
 * nir_lower_io state, so that shader variants can more easily insert/
 * replace variables, etc.
 */
static void
st_nir_preprocess(struct st_context *st, struct gl_program *prog,
                  struct gl_shader_program *shader_program,
                  gl_shader_stage stage)
{
   struct pipe_screen *screen = st->pipe->screen;
   const nir_shader_compiler_options *options =
      st->ctx->Const.ShaderCompilerOptions[prog->info.stage].NirOptions;
   assert(options);
   nir_shader *nir = prog->nir;

   /* Set the next shader stage hint for VS and TES. */
   if (!nir->info.separate_shader &&
       (nir->info.stage == MESA_SHADER_VERTEX ||
        nir->info.stage == MESA_SHADER_TESS_EVAL)) {

      unsigned prev_stages = (1 << (prog->info.stage + 1)) - 1;
      unsigned stages_mask =
         ~prev_stages & shader_program->data->linked_stages;

      nir->info.next_stage = stages_mask ?
         (gl_shader_stage) u_bit_scan(&stages_mask) : MESA_SHADER_FRAGMENT;
   } else {
      nir->info.next_stage = MESA_SHADER_FRAGMENT;
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (!st->ctx->SoftFP64 && nir->info.uses_64bit &&
       (options->lower_doubles_options & nir_lower_fp64_full_software) != 0) {
      st->ctx->SoftFP64 = glsl_float64_funcs_to_nir(st->ctx, options);
   }

   nir_variable_mode mask =
      (nir_variable_mode) (nir_var_shader_in | nir_var_shader_out);
   nir_remove_dead_variables(nir, mask);

   if (options->lower_all_io_to_temps ||
       nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_GEOMETRY) {
      NIR_PASS_V(nir, nir_lower_io_to_temporaries,
                 nir_shader_get_entrypoint(nir),
                 true, true);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT ||
              !screen->get_param(screen, PIPE_CAP_TGSI_CAN_READ_OUTPUTS)) {
      NIR_PASS_V(nir, nir_lower_io_to_temporaries,
                 nir_shader_get_entrypoint(nir),
                 true, false);
   }

   NIR_PASS_V(nir, nir_lower_global_vars_to_local);
   NIR_PASS_V(nir, nir_split_var_copies);
   NIR_PASS_V(nir, nir_lower_var_copies);

   if (options->lower_to_scalar) {
     NIR_PASS_V(nir, nir_lower_alu_to_scalar, NULL, NULL);
   }

   /* before buffers and vars_to_ssa */
   NIR_PASS_V(nir, gl_nir_lower_bindless_images);

   /* TODO: Change GLSL to not lower shared memory. */
   if (prog->nir->info.stage == MESA_SHADER_COMPUTE &&
       shader_program->data->spirv) {
      NIR_PASS_V(prog->nir, nir_lower_vars_to_explicit_types,
                 nir_var_mem_shared, shared_type_info);
      NIR_PASS_V(prog->nir, nir_lower_explicit_io,
                 nir_var_mem_shared, nir_address_format_32bit_offset);
   }

   NIR_PASS_V(nir, gl_nir_lower_buffers, shader_program);
   /* Do a round of constant folding to clean up address calculations */
   NIR_PASS_V(nir, nir_opt_constant_folding);
}

/* Second third of converting glsl_to_nir. This creates uniforms, gathers
 * info on varyings, etc after NIR link time opts have been applied.
 */
static void
st_glsl_to_nir_post_opts(struct st_context *st, struct gl_program *prog,
                         struct gl_shader_program *shader_program)
{
   nir_shader *nir = prog->nir;

   /* Make a pass over the IR to add state references for any built-in
    * uniforms that are used.  This has to be done now (during linking).
    * Code generation doesn't happen until the first time this shader is
    * used for rendering.  Waiting until then to generate the parameters is
    * too late.  At that point, the values for the built-in uniforms won't
    * get sent to the shader.
    */
   nir_foreach_variable(var, &nir->uniforms) {
      const nir_state_slot *const slots = var->state_slots;
      if (slots != NULL) {
         const struct glsl_type *type = glsl_without_array(var->type);
         for (unsigned int i = 0; i < var->num_state_slots; i++) {
            unsigned comps;
            if (glsl_type_is_struct_or_ifc(type)) {
               /* Builtin struct require specical handling for now we just
                * make all members vec4. See st_nir_lower_builtin.
                */
               comps = 4;
            } else {
               comps = glsl_get_vector_elements(type);
            }

            if (st->ctx->Const.PackedDriverUniformStorage) {
               _mesa_add_sized_state_reference(prog->Parameters,
                                               slots[i].tokens,
                                               comps, false);
            } else {
               _mesa_add_state_reference(prog->Parameters,
                                         slots[i].tokens);
            }
         }
      }
   }

   /* Avoid reallocation of the program parameter list, because the uniform
    * storage is only associated with the original parameter list.
    * This should be enough for Bitmap and DrawPixels constants.
    */
   _mesa_reserve_parameter_storage(prog->Parameters, 8);

   /* This has to be done last.  Any operation the can cause
    * prog->ParameterValues to get reallocated (e.g., anything that adds a
    * program constant) has to happen before creating this linkage.
    */
   _mesa_associate_uniform_storage(st->ctx, shader_program, prog);

   st_set_prog_affected_state_flags(prog);

   /* None of the builtins being lowered here can be produced by SPIR-V.  See
    * _mesa_builtin_uniform_desc.
    */
   if (!shader_program->data->spirv)
      NIR_PASS_V(nir, st_nir_lower_builtin);

   NIR_PASS_V(nir, gl_nir_lower_atomics, shader_program, true);
   NIR_PASS_V(nir, nir_opt_intrinsics);

   /* Lower 64-bit ops. */
   if (nir->options->lower_int64_options ||
       nir->options->lower_doubles_options) {
      bool lowered_64bit_ops = false;
      if (nir->options->lower_doubles_options) {
         NIR_PASS(lowered_64bit_ops, nir, nir_lower_doubles,
                  st->ctx->SoftFP64, nir->options->lower_doubles_options);
      }
      if (nir->options->lower_int64_options) {
         NIR_PASS(lowered_64bit_ops, nir, nir_lower_int64,
                  nir->options->lower_int64_options);
      }

      if (lowered_64bit_ops)
         st_nir_opts(nir);
   }

   nir_variable_mode mask = (nir_variable_mode)
      (nir_var_shader_in | nir_var_shader_out | nir_var_function_temp );
   nir_remove_dead_variables(nir, mask);

   NIR_PASS_V(nir, nir_lower_atomics_to_ssbo,
              st->ctx->Const.Program[nir->info.stage].MaxAtomicBuffers);

   st_finalize_nir_before_variants(nir);

   if (st->allow_st_finalize_nir_twice)
      st_finalize_nir(st, prog, shader_program, nir, true);

   if (st->ctx->_Shader->Flags & GLSL_DUMP) {
      _mesa_log("\n");
      _mesa_log("NIR IR for linked %s program %d:\n",
             _mesa_shader_stage_to_string(prog->info.stage),
             shader_program->Name);
      nir_print_shader(nir, _mesa_get_log_file());
      _mesa_log("\n\n");
   }
}

static void
set_st_program(struct gl_program *prog,
               struct gl_shader_program *shader_program,
               nir_shader *nir)
{
   struct st_vertex_program *stvp;
   struct st_common_program *stp;

   switch (prog->info.stage) {
   case MESA_SHADER_VERTEX:
      stvp = (struct st_vertex_program *)prog;
      stvp->shader_program = shader_program;
      stvp->state.type = PIPE_SHADER_IR_NIR;
      stvp->state.ir.nir = nir;
      break;
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_FRAGMENT:
      stp = (struct st_common_program *)prog;
      stp->shader_program = shader_program;
      stp->state.type = PIPE_SHADER_IR_NIR;
      stp->state.ir.nir = nir;
      break;
   default:
      unreachable("unknown shader stage");
   }
}

static void
st_nir_vectorize_io(nir_shader *producer, nir_shader *consumer)
{
   NIR_PASS_V(producer, nir_lower_io_to_vector, nir_var_shader_out);
   NIR_PASS_V(producer, nir_opt_combine_stores, nir_var_shader_out);
   NIR_PASS_V(consumer, nir_lower_io_to_vector, nir_var_shader_in);

   if ((producer)->info.stage != MESA_SHADER_TESS_CTRL) {
      /* Calling lower_io_to_vector creates output variable writes with
       * write-masks.  We only support these for TCS outputs, so for other
       * stages, we need to call nir_lower_io_to_temporaries to get rid of
       * them.  This, in turn, creates temporary variables and extra
       * copy_deref intrinsics that we need to clean up.
       */
      NIR_PASS_V(producer, nir_lower_io_to_temporaries,
                 nir_shader_get_entrypoint(producer), true, false);
      NIR_PASS_V(producer, nir_lower_global_vars_to_local);
      NIR_PASS_V(producer, nir_split_var_copies);
      NIR_PASS_V(producer, nir_lower_var_copies);
   }
}

static void
st_nir_link_shaders(nir_shader **producer, nir_shader **consumer)
{
   if ((*producer)->options->lower_to_scalar) {
      NIR_PASS_V(*producer, nir_lower_io_to_scalar_early, nir_var_shader_out);
      NIR_PASS_V(*consumer, nir_lower_io_to_scalar_early, nir_var_shader_in);
   }

   nir_lower_io_arrays_to_elements(*producer, *consumer);

   st_nir_opts(*producer);
   st_nir_opts(*consumer);

   if (nir_link_opt_varyings(*producer, *consumer))
      st_nir_opts(*consumer);

   NIR_PASS_V(*producer, nir_remove_dead_variables, nir_var_shader_out);
   NIR_PASS_V(*consumer, nir_remove_dead_variables, nir_var_shader_in);

   if (nir_remove_unused_varyings(*producer, *consumer)) {
      NIR_PASS_V(*producer, nir_lower_global_vars_to_local);
      NIR_PASS_V(*consumer, nir_lower_global_vars_to_local);

      st_nir_opts(*producer);
      st_nir_opts(*consumer);

      /* Optimizations can cause varyings to become unused.
       * nir_compact_varyings() depends on all dead varyings being removed so
       * we need to call nir_remove_dead_variables() again here.
       */
      NIR_PASS_V(*producer, nir_remove_dead_variables, nir_var_shader_out);
      NIR_PASS_V(*consumer, nir_remove_dead_variables, nir_var_shader_in);
   }
}

static void
st_lower_patch_vertices_in(struct gl_shader_program *shader_prog)
{
   struct gl_linked_shader *linked_tcs =
      shader_prog->_LinkedShaders[MESA_SHADER_TESS_CTRL];
   struct gl_linked_shader *linked_tes =
      shader_prog->_LinkedShaders[MESA_SHADER_TESS_EVAL];

   /* If we have a TCS and TES linked together, lower TES patch vertices. */
   if (linked_tcs && linked_tes) {
      nir_shader *tcs_nir = linked_tcs->Program->nir;
      nir_shader *tes_nir = linked_tes->Program->nir;

      /* The TES input vertex count is the TCS output vertex count,
       * lower TES gl_PatchVerticesIn to a constant.
       */
      uint32_t tes_patch_verts = tcs_nir->info.tess.tcs_vertices_out;
      NIR_PASS_V(tes_nir, nir_lower_patch_vertices, tes_patch_verts, NULL);
   }
}

extern "C" {

void
st_nir_lower_wpos_ytransform(struct nir_shader *nir,
                             struct gl_program *prog,
                             struct pipe_screen *pscreen)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return;

   static const gl_state_index16 wposTransformState[STATE_LENGTH] = {
      STATE_INTERNAL, STATE_FB_WPOS_Y_TRANSFORM
   };
   nir_lower_wpos_ytransform_options wpos_options = { { 0 } };

   memcpy(wpos_options.state_tokens, wposTransformState,
          sizeof(wpos_options.state_tokens));
   wpos_options.fs_coord_origin_upper_left =
      pscreen->get_param(pscreen,
                         PIPE_CAP_TGSI_FS_COORD_ORIGIN_UPPER_LEFT);
   wpos_options.fs_coord_origin_lower_left =
      pscreen->get_param(pscreen,
                         PIPE_CAP_TGSI_FS_COORD_ORIGIN_LOWER_LEFT);
   wpos_options.fs_coord_pixel_center_integer =
      pscreen->get_param(pscreen,
                         PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_INTEGER);
   wpos_options.fs_coord_pixel_center_half_integer =
      pscreen->get_param(pscreen,
                         PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER);

   if (nir_lower_wpos_ytransform(nir, &wpos_options)) {
      nir_validate_shader(nir, "after nir_lower_wpos_ytransform");
      _mesa_add_state_reference(prog->Parameters, wposTransformState);
   }
}

bool
st_link_nir(struct gl_context *ctx,
            struct gl_shader_program *shader_program)
{
   struct st_context *st = st_context(ctx);
   unsigned num_linked_shaders = 0;

   unsigned last_stage = 0;
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      struct gl_linked_shader *shader = shader_program->_LinkedShaders[i];
      if (shader == NULL)
         continue;

      num_linked_shaders++;

      const nir_shader_compiler_options *options =
         st->ctx->Const.ShaderCompilerOptions[shader->Stage].NirOptions;
      struct gl_program *prog = shader->Program;
      _mesa_copy_linked_program_data(shader_program, shader);

      assert(!prog->nir);

      if (shader_program->data->spirv) {
         prog->Parameters = _mesa_new_parameter_list();
         /* Parameters will be filled during NIR linking. */

         prog->nir = _mesa_spirv_to_nir(ctx, shader_program, shader->Stage, options);
         set_st_program(prog, shader_program, prog->nir);
      } else {
         validate_ir_tree(shader->ir);

         prog->Parameters = _mesa_new_parameter_list();
         _mesa_generate_parameters_list_for_uniforms(ctx, shader_program, shader,
                                                     prog->Parameters);

         if (ctx->_Shader->Flags & GLSL_DUMP) {
            _mesa_log("\n");
            _mesa_log("GLSL IR for linked %s program %d:\n",
                      _mesa_shader_stage_to_string(shader->Stage),
                      shader_program->Name);
            _mesa_print_ir(_mesa_get_log_file(), shader->ir, NULL);
            _mesa_log("\n\n");
         }

         prog->ExternalSamplersUsed = gl_external_samplers(prog);
         _mesa_update_shader_textures_used(shader_program, prog);

         prog->nir = glsl_to_nir(st->ctx, shader_program, shader->Stage, options);
         set_st_program(prog, shader_program, prog->nir);
         st_nir_preprocess(st, prog, shader_program, shader->Stage);
      }

      last_stage = i;

      if (options->lower_to_scalar) {
         NIR_PASS_V(shader->Program->nir, nir_lower_load_const_to_scalar);
      }
   }

   /* For SPIR-V, we have to perform the NIR linking before applying
    * st_nir_preprocess.
    */
   if (shader_program->data->spirv) {
      static const gl_nir_linker_options opts = {
         true /*fill_parameters */
      };
      if (!gl_nir_link(ctx, shader_program, &opts))
         return GL_FALSE;

      nir_build_program_resource_list(ctx, shader_program);

      for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
         struct gl_linked_shader *shader = shader_program->_LinkedShaders[i];
         if (shader == NULL)
            continue;

         struct gl_program *prog = shader->Program;
         prog->ExternalSamplersUsed = gl_external_samplers(prog);
         _mesa_update_shader_textures_used(shader_program, prog);

         st_nir_preprocess(st, prog, shader_program, shader->Stage);
      }
   }

   /* Linking the stages in the opposite order (from fragment to vertex)
    * ensures that inter-shader outputs written to in an earlier stage
    * are eliminated if they are (transitively) not used in a later
    * stage.
    */
   int next = last_stage;
   for (int i = next - 1; i >= 0; i--) {
      struct gl_linked_shader *shader = shader_program->_LinkedShaders[i];
      if (shader == NULL)
         continue;

      st_nir_link_shaders(&shader->Program->nir,
                          &shader_program->_LinkedShaders[next]->Program->nir);
      next = i;
   }

   int prev = -1;
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      struct gl_linked_shader *shader = shader_program->_LinkedShaders[i];
      if (shader == NULL)
         continue;

      nir_shader *nir = shader->Program->nir;

      /* Linked shaders are optimized in st_nir_link_shaders. Separate shaders
       * and shaders with a fixed-func VS or FS are optimized here.
       */
      if (num_linked_shaders == 1)
         st_nir_opts(nir);

      NIR_PASS_V(nir, st_nir_lower_wpos_ytransform, shader->Program,
                 st->pipe->screen);

      NIR_PASS_V(nir, nir_lower_system_values);
      NIR_PASS_V(nir, nir_lower_clip_cull_distance_arrays);

      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      shader->Program->info = nir->info;
      if (i == MESA_SHADER_VERTEX) {
         /* NIR expands dual-slot inputs out to two locations.  We need to
          * compact things back down GL-style single-slot inputs to avoid
          * confusing the state tracker.
          */
         shader->Program->info.inputs_read =
            nir_get_single_slot_attribs_mask(nir->info.inputs_read,
                                             shader->Program->DualSlotInputs);
      }

      if (prev != -1) {
         struct gl_program *prev_shader =
            shader_program->_LinkedShaders[prev]->Program;

         /* We can't use nir_compact_varyings with transform feedback, since
          * the pipe_stream_output->output_register field is based on the
          * pre-compacted driver_locations.
          */
         if (!(prev_shader->sh.LinkedTransformFeedback &&
               prev_shader->sh.LinkedTransformFeedback->NumVarying > 0))
            nir_compact_varyings(shader_program->_LinkedShaders[prev]->Program->nir,
                              nir, ctx->API != API_OPENGL_COMPAT);

         if (ctx->Const.ShaderCompilerOptions[i].NirOptions->vectorize_io)
            st_nir_vectorize_io(prev_shader->nir, nir);
      }
      prev = i;
   }

   st_lower_patch_vertices_in(shader_program);

   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++) {
      struct gl_linked_shader *shader = shader_program->_LinkedShaders[i];
      if (shader == NULL)
         continue;

      struct gl_program *prog = shader->Program;
      st_glsl_to_nir_post_opts(st, prog, shader_program);

      /* Initialize st_vertex_program members. */
      if (i == MESA_SHADER_VERTEX)
         st_prepare_vertex_program(st_vertex_program(prog));

      /* Get pipe_stream_output_info. */
      if (i == MESA_SHADER_VERTEX ||
          i == MESA_SHADER_TESS_EVAL ||
          i == MESA_SHADER_GEOMETRY)
         st_translate_stream_output_info(prog);

      st_store_ir_in_disk_cache(st, prog, true);

      if (!ctx->Driver.ProgramStringNotify(ctx,
                                           _mesa_shader_stage_to_program(i),
                                           prog)) {
         _mesa_reference_program(ctx, &shader->Program, NULL);
         return false;
      }

      nir_sweep(prog->nir);

      /* The GLSL IR won't be needed anymore. */
      ralloc_free(shader->ir);
      shader->ir = NULL;
   }

   return true;
}

void
st_nir_assign_varying_locations(struct st_context *st, nir_shader *nir)
{
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_assign_io_var_locations(&nir->outputs,
                                  &nir->num_outputs,
                                  nir->info.stage);
      st_nir_fixup_varying_slots(st, &nir->outputs);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY ||
              nir->info.stage == MESA_SHADER_TESS_CTRL ||
              nir->info.stage == MESA_SHADER_TESS_EVAL) {
      nir_assign_io_var_locations(&nir->inputs,
                                  &nir->num_inputs,
                                  nir->info.stage);
      st_nir_fixup_varying_slots(st, &nir->inputs);

      nir_assign_io_var_locations(&nir->outputs,
                                  &nir->num_outputs,
                                  nir->info.stage);
      st_nir_fixup_varying_slots(st, &nir->outputs);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_assign_io_var_locations(&nir->inputs,
                                  &nir->num_inputs,
                                  nir->info.stage);
      st_nir_fixup_varying_slots(st, &nir->inputs);
      nir_assign_io_var_locations(&nir->outputs,
                                  &nir->num_outputs,
                                  nir->info.stage);
   } else if (nir->info.stage == MESA_SHADER_COMPUTE) {
       /* TODO? */
   } else {
      unreachable("invalid shader type");
   }
}

void
st_nir_lower_samplers(struct pipe_screen *screen, nir_shader *nir,
                      struct gl_shader_program *shader_program,
                      struct gl_program *prog)
{
   if (screen->get_param(screen, PIPE_CAP_NIR_SAMPLERS_AS_DEREF))
      NIR_PASS_V(nir, gl_nir_lower_samplers_as_deref, shader_program);
   else
      NIR_PASS_V(nir, gl_nir_lower_samplers, shader_program);

   if (prog) {
      prog->info.textures_used = nir->info.textures_used;
      prog->info.textures_used_by_txf = nir->info.textures_used_by_txf;
   }
}

/* Last third of preparing nir from glsl, which happens after shader
 * variant lowering.
 */
void
st_finalize_nir(struct st_context *st, struct gl_program *prog,
                struct gl_shader_program *shader_program,
                nir_shader *nir, bool finalize_by_driver)
{
   struct pipe_screen *screen = st->pipe->screen;

   NIR_PASS_V(nir, nir_split_var_copies);
   NIR_PASS_V(nir, nir_lower_var_copies);

   st_nir_assign_varying_locations(st, nir);
   st_nir_assign_uniform_locations(st->ctx, prog,
                                   &nir->uniforms);

   /* Set num_uniforms in number of attribute slots (vec4s) */
   nir->num_uniforms = DIV_ROUND_UP(prog->Parameters->NumParameterValues, 4);

   if (st->ctx->Const.PackedDriverUniformStorage) {
      NIR_PASS_V(nir, nir_lower_io, nir_var_uniform, st_glsl_type_dword_size,
                 (nir_lower_io_options)0);
      NIR_PASS_V(nir, nir_lower_uniforms_to_ubo, 4);
   } else {
      NIR_PASS_V(nir, nir_lower_io, nir_var_uniform, st_glsl_uniforms_type_size,
                 (nir_lower_io_options)0);
   }

   st_nir_lower_samplers(screen, nir, shader_program, prog);

   if (finalize_by_driver && screen->finalize_nir)
      screen->finalize_nir(screen, nir, false);
}

} /* extern "C" */
