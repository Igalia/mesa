/*
 * Copyright 2003 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "main/version.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "intel_batchbuffer.h"

/**
 * Test if we can use MI_LOAD_REGISTER_MEM from an untrusted batchbuffer.
 *
 * Some combinations of hardware and kernel versions allow this feature,
 * while others don't.  Instead of trying to enumerate every case, just
 * try and write a register and see if works.
 */
static bool
can_do_pipelined_register_writes(struct brw_context *brw)
{
   /**
    * gen >= 8 specifically allows these writes. gen <= 6 also
    * doesn't block them.
    */
   if (brw->gen != 7)
      return true;

   static int result = -1;
   if (result != -1)
      return result;

   /* We use SO_WRITE_OFFSET0 since you're supposed to write it (unlike the
    * statistics registers), and we already reset it to zero before using it.
    */
   const int reg = GEN7_SO_WRITE_OFFSET(0);
   const int expected_value = 0x1337d0d0;
   const int offset = 100;

   /* The register we picked only exists on Gen7+. */
   assert(brw->gen == 7);

   uint32_t *data;
   /* Set a value in a BO to a known quantity.  The workaround BO already
    * exists and doesn't contain anything important, so we may as well use it.
    */
   drm_intel_bo_map(brw->workaround_bo, true);
   data = brw->workaround_bo->virtual;
   data[offset] = 0xffffffff;
   drm_intel_bo_unmap(brw->workaround_bo);

   /* Write the register. */
   BEGIN_BATCH(3);
   OUT_BATCH(MI_LOAD_REGISTER_IMM | (3 - 2));
   OUT_BATCH(reg);
   OUT_BATCH(expected_value);
   ADVANCE_BATCH();

   brw_emit_mi_flush(brw);

   /* Save the register's value back to the buffer. */
   BEGIN_BATCH(3);
   OUT_BATCH(MI_STORE_REGISTER_MEM | (3 - 2));
   OUT_BATCH(reg);
   OUT_RELOC(brw->workaround_bo,
             I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
             offset * sizeof(uint32_t));
   ADVANCE_BATCH();

   intel_batchbuffer_flush(brw);

   /* Check whether the value got written. */
   drm_intel_bo_map(brw->workaround_bo, false);
   data = brw->workaround_bo->virtual;
   bool success = data[offset] == expected_value;
   drm_intel_bo_unmap(brw->workaround_bo);

   result = success;

   return success;
}

static bool
can_write_oacontrol(struct brw_context *brw)
{
   if (brw->gen < 6 || brw->gen >= 8)
      return false;

   static int result = -1;
   if (result != -1)
      return result;

   /* Set "Select Context ID" to a particular address (which is likely not a
    * context), but leave all counting disabled.  This should be harmless.
    */
   const int expected_value = 0x31337000;
   const int offset = 110;

   uint32_t *data;
   /* Set a value in a BO to a known quantity.  The workaround BO already
    * exists and doesn't contain anything important, so we may as well use it.
    */
   drm_intel_bo_map(brw->workaround_bo, true);
   data = brw->workaround_bo->virtual;
   data[offset] = 0xffffffff;
   drm_intel_bo_unmap(brw->workaround_bo);

   /* Write OACONTROL. */
   BEGIN_BATCH(3);
   OUT_BATCH(MI_LOAD_REGISTER_IMM | (3 - 2));
   OUT_BATCH(OACONTROL);
   OUT_BATCH(expected_value);
   ADVANCE_BATCH();

   brw_emit_mi_flush(brw);

   /* Save the register's value back to the buffer. */
   BEGIN_BATCH(3);
   OUT_BATCH(MI_STORE_REGISTER_MEM | (3 - 2));
   OUT_BATCH(OACONTROL);
   OUT_RELOC(brw->workaround_bo,
             I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
             offset * sizeof(uint32_t));
   ADVANCE_BATCH();

   brw_emit_mi_flush(brw);

   /* Set OACONTROL back to zero (everything off). */
   BEGIN_BATCH(3);
   OUT_BATCH(MI_LOAD_REGISTER_IMM | (3 - 2));
   OUT_BATCH(OACONTROL);
   OUT_BATCH(0);
   ADVANCE_BATCH();

   intel_batchbuffer_flush(brw);

   /* Check whether the value got written. */
   drm_intel_bo_map(brw->workaround_bo, false);
   data = brw->workaround_bo->virtual;
   bool success = data[offset] == expected_value;
   drm_intel_bo_unmap(brw->workaround_bo);

   result = success;

   return success;
}

/**
 * Initializes potential list of extensions if ctx == NULL, or actually enables
 * extensions for a context.
 */
void
intelInitExtensions(struct gl_context *ctx)
{
   struct brw_context *brw = brw_context(ctx);

   assert(brw->gen >= 4);

   ctx->Extensions.ARB_arrays_of_arrays = true;
   ctx->Extensions.ARB_buffer_storage = true;
   ctx->Extensions.ARB_clear_texture = true;
   ctx->Extensions.ARB_clip_control = true;
   ctx->Extensions.ARB_copy_image = true;
   ctx->Extensions.ARB_depth_buffer_float = true;
   ctx->Extensions.ARB_depth_clamp = true;
   ctx->Extensions.ARB_depth_texture = true;
   ctx->Extensions.ARB_draw_elements_base_vertex = true;
   ctx->Extensions.ARB_draw_instanced = true;
   ctx->Extensions.ARB_ES2_compatibility = true;
   ctx->Extensions.ARB_explicit_attrib_location = true;
   ctx->Extensions.ARB_explicit_uniform_location = true;
   ctx->Extensions.ARB_fragment_coord_conventions = true;
   ctx->Extensions.ARB_fragment_program = true;
   ctx->Extensions.ARB_fragment_program_shadow = true;
   ctx->Extensions.ARB_fragment_shader = true;
   ctx->Extensions.ARB_framebuffer_object = true;
   ctx->Extensions.ARB_half_float_vertex = true;
   ctx->Extensions.ARB_instanced_arrays = true;
   ctx->Extensions.ARB_internalformat_query = true;
   ctx->Extensions.ARB_internalformat_query2 = true;
   ctx->Extensions.ARB_map_buffer_range = true;
   ctx->Extensions.ARB_occlusion_query = true;
   ctx->Extensions.ARB_occlusion_query2 = true;
   ctx->Extensions.ARB_pipeline_statistics_query = true;
   ctx->Extensions.ARB_point_sprite = true;
   ctx->Extensions.ARB_seamless_cube_map = true;
   ctx->Extensions.ARB_shader_bit_encoding = true;
   ctx->Extensions.ARB_shader_draw_parameters = true;
   ctx->Extensions.ARB_shader_texture_lod = true;
   ctx->Extensions.ARB_shading_language_packing = true;
   ctx->Extensions.ARB_shadow = true;
   ctx->Extensions.ARB_sync = true;
   ctx->Extensions.ARB_texture_border_clamp = true;
   ctx->Extensions.ARB_texture_compression_rgtc = true;
   ctx->Extensions.ARB_texture_cube_map = true;
   ctx->Extensions.ARB_texture_env_combine = true;
   ctx->Extensions.ARB_texture_env_crossbar = true;
   ctx->Extensions.ARB_texture_env_dot3 = true;
   ctx->Extensions.ARB_texture_float = true;
   ctx->Extensions.ARB_texture_mirror_clamp_to_edge = true;
   ctx->Extensions.ARB_texture_non_power_of_two = true;
   ctx->Extensions.ARB_texture_rg = true;
   ctx->Extensions.ARB_texture_rgb10_a2ui = true;
   ctx->Extensions.ARB_vertex_program = true;
   ctx->Extensions.ARB_vertex_shader = true;
   ctx->Extensions.ARB_vertex_type_2_10_10_10_rev = true;
   ctx->Extensions.ARB_vertex_type_10f_11f_11f_rev = true;
   ctx->Extensions.EXT_blend_color = true;
   ctx->Extensions.EXT_blend_equation_separate = true;
   ctx->Extensions.EXT_blend_func_separate = true;
   ctx->Extensions.EXT_blend_minmax = true;
   ctx->Extensions.EXT_draw_buffers2 = true;
   ctx->Extensions.EXT_framebuffer_sRGB = true;
   ctx->Extensions.EXT_gpu_program_parameters = true;
   ctx->Extensions.EXT_packed_float = true;
   ctx->Extensions.EXT_pixel_buffer_object = true;
   ctx->Extensions.EXT_point_parameters = true;
   ctx->Extensions.EXT_polygon_offset_clamp = true;
   ctx->Extensions.EXT_provoking_vertex = true;
   ctx->Extensions.EXT_stencil_two_side = true;
   ctx->Extensions.EXT_texture_array = true;
   ctx->Extensions.EXT_texture_env_dot3 = true;
   ctx->Extensions.EXT_texture_filter_anisotropic = true;
   ctx->Extensions.EXT_texture_integer = true;
   ctx->Extensions.EXT_texture_shared_exponent = true;
   ctx->Extensions.EXT_texture_snorm = true;
   ctx->Extensions.EXT_texture_sRGB = true;
   ctx->Extensions.EXT_texture_sRGB_decode = true;
   ctx->Extensions.EXT_texture_swizzle = true;
   ctx->Extensions.EXT_vertex_array_bgra = true;
   ctx->Extensions.KHR_robustness = true;
   ctx->Extensions.AMD_seamless_cubemap_per_texture = true;
   ctx->Extensions.APPLE_object_purgeable = true;
   ctx->Extensions.ATI_separate_stencil = true;
   ctx->Extensions.ATI_texture_env_combine3 = true;
   ctx->Extensions.MESA_pack_invert = true;
   ctx->Extensions.NV_conditional_render = true;
   ctx->Extensions.NV_primitive_restart = true;
   ctx->Extensions.NV_texture_barrier = true;
   ctx->Extensions.NV_texture_env_combine4 = true;
   ctx->Extensions.NV_texture_rectangle = true;
   ctx->Extensions.TDFX_texture_compression_FXT1 = true;
   ctx->Extensions.OES_compressed_ETC1_RGB8_texture = true;
   ctx->Extensions.OES_draw_texture = true;
   ctx->Extensions.OES_EGL_image = true;
   ctx->Extensions.OES_EGL_image_external = true;
   ctx->Extensions.OES_standard_derivatives = true;
   ctx->Extensions.OES_texture_float = true;
   ctx->Extensions.OES_texture_float_linear = true;
   ctx->Extensions.OES_texture_half_float = true;
   ctx->Extensions.OES_texture_half_float_linear = true;

   if (brw->gen >= 8)
      ctx->Const.GLSLVersion = 450;
   else if (brw->gen >= 6)
      ctx->Const.GLSLVersion = 330;
   else
      ctx->Const.GLSLVersion = 120;
   _mesa_override_glsl_version(&ctx->Const);

   ctx->Extensions.EXT_shader_integer_mix = ctx->Const.GLSLVersion >= 130;
   ctx->Extensions.MESA_shader_integer_functions = ctx->Const.GLSLVersion >= 130;

   if (brw->is_g4x || brw->gen >= 5) {
      ctx->Extensions.MESA_shader_framebuffer_fetch_non_coherent = true;
      ctx->Extensions.KHR_blend_equation_advanced = true;
   }

   if (brw->gen >= 5) {
      ctx->Extensions.ARB_texture_query_levels = ctx->Const.GLSLVersion >= 130;
      ctx->Extensions.ARB_texture_query_lod = true;
      ctx->Extensions.EXT_timer_query = true;

      if (brw->gen == 5 || can_write_oacontrol(brw)) {
         ctx->Extensions.AMD_performance_monitor = true;
         ctx->Extensions.INTEL_performance_query = true;
      }
   }

   if (brw->gen >= 6) {
      ctx->Extensions.ARB_blend_func_extended =
         !driQueryOptionb(&brw->optionCache, "disable_blend_func_extended");
      ctx->Extensions.ARB_conditional_render_inverted = true;
      ctx->Extensions.ARB_cull_distance = true;
      ctx->Extensions.ARB_draw_buffers_blend = true;
      ctx->Extensions.ARB_enhanced_layouts = true;
      ctx->Extensions.ARB_ES3_compatibility = true;
      ctx->Extensions.ARB_fragment_layer_viewport = true;
      ctx->Extensions.ARB_sample_shading = true;
      ctx->Extensions.ARB_shading_language_420pack = true;
      ctx->Extensions.ARB_texture_buffer_object = true;
      ctx->Extensions.ARB_texture_buffer_object_rgb32 = true;
      ctx->Extensions.ARB_texture_buffer_range = true;
      ctx->Extensions.ARB_texture_cube_map_array = true;
      ctx->Extensions.ARB_texture_gather = true;
      ctx->Extensions.ARB_texture_multisample = true;
      ctx->Extensions.ARB_uniform_buffer_object = true;

      ctx->Extensions.AMD_vertex_shader_layer = true;
      ctx->Extensions.EXT_framebuffer_multisample = true;
      ctx->Extensions.EXT_framebuffer_multisample_blit_scaled = true;
      ctx->Extensions.EXT_transform_feedback = true;
      ctx->Extensions.OES_depth_texture_cube_map = true;
      ctx->Extensions.OES_sample_variables = true;

      ctx->Extensions.ARB_timer_query = brw->screen->hw_has_timestamp;

      /* Only enable this in core profile because other parts of Mesa behave
       * slightly differently when the extension is enabled.
       */
      if (ctx->API == API_OPENGL_CORE) {
         ctx->Extensions.ARB_shader_subroutine = true;
         ctx->Extensions.ARB_shader_viewport_layer_array = true;
         ctx->Extensions.ARB_viewport_array = true;
         ctx->Extensions.AMD_vertex_shader_viewport_index = true;
      }
   }

   brw->predicate.supported = false;
   brw->can_do_pipelined_register_writes =
      can_do_pipelined_register_writes(brw);

   if (brw->gen >= 7) {
      ctx->Extensions.ARB_conservative_depth = true;
      ctx->Extensions.ARB_derivative_control = true;
      ctx->Extensions.ARB_framebuffer_no_attachments = true;
      ctx->Extensions.ARB_gpu_shader5 = true;
      ctx->Extensions.ARB_shader_atomic_counters = true;
      ctx->Extensions.ARB_shader_atomic_counter_ops = true;
      ctx->Extensions.ARB_shader_clock = true;
      ctx->Extensions.ARB_shader_image_load_store = true;
      ctx->Extensions.ARB_shader_image_size = true;
      ctx->Extensions.ARB_shader_texture_image_samples = true;
      ctx->Extensions.ARB_tessellation_shader = true;
      ctx->Extensions.ARB_texture_compression_bptc = true;
      ctx->Extensions.ARB_texture_view = true;
      ctx->Extensions.ARB_shader_storage_buffer_object = true;
      ctx->Extensions.EXT_shader_samples_identical = true;
      ctx->Extensions.OES_primitive_bounding_box = true;
      ctx->Extensions.OES_texture_buffer = true;

      if (brw->is_haswell)
         ctx->Extensions.ARB_gpu_shader_fp64 = true;

      if (brw->can_do_pipelined_register_writes) {
         ctx->Extensions.ARB_draw_indirect = true;
         ctx->Extensions.ARB_transform_feedback2 = true;
         ctx->Extensions.ARB_transform_feedback3 = true;
         ctx->Extensions.ARB_transform_feedback_instanced = true;

         if ((brw->gen >= 8 || brw->screen->cmd_parser_version >= 5) &&
             ctx->Const.MaxComputeWorkGroupSize[0] >= 1024) {
            ctx->Extensions.ARB_compute_shader = true;
            ctx->Extensions.ARB_ES3_1_compatibility =
               brw->gen >= 8 || brw->is_haswell;
         }

         if (brw->screen->cmd_parser_version >= 2)
            brw->predicate.supported = true;
      }
   }

   if (brw->gen >= 8 || brw->is_haswell) {
      ctx->Extensions.ARB_stencil_texturing = true;
      ctx->Extensions.ARB_texture_stencil8 = true;
   }

   if (brw->gen >= 8 || brw->is_haswell || brw->is_baytrail) {
      ctx->Extensions.ARB_robust_buffer_access_behavior = true;
   }

   if (brw->screen->has_mi_math_and_lrr) {
      ctx->Extensions.ARB_query_buffer_object = true;
   }

   if (brw->gen >= 8 || brw->is_baytrail) {
      /* For now, we only enable OES_copy_image on platforms that support
       * ETC2 natively in hardware.  We would need more hacks to support it
       * elsewhere.
       */
      ctx->Extensions.OES_copy_image = true;
   }

   if (brw->gen >= 8) {
      ctx->Extensions.ARB_gpu_shader_fp64 = true;
      ctx->Extensions.ARB_shader_precision = true;
      ctx->Extensions.ARB_vertex_attrib_64bit = true;
      ctx->Extensions.ARB_ES3_2_compatibility = true;
      ctx->Extensions.OES_geometry_shader = true;
      ctx->Extensions.OES_texture_cube_map_array = true;
      ctx->Extensions.OES_viewport_array = true;
   }

   if (brw->gen >= 9) {
      ctx->Extensions.ANDROID_extension_pack_es31a = true;
      ctx->Extensions.ARB_shader_stencil_export = true;
      ctx->Extensions.KHR_blend_equation_advanced_coherent = true;
      ctx->Extensions.KHR_texture_compression_astc_ldr = true;
      ctx->Extensions.KHR_texture_compression_astc_sliced_3d = true;
      ctx->Extensions.INTEL_conservative_rasterization = true;
      ctx->Extensions.MESA_shader_framebuffer_fetch = true;
      ctx->Extensions.ARB_post_depth_coverage = true;
   }

   if (ctx->API == API_OPENGL_CORE)
      ctx->Extensions.ARB_base_instance = true;
   if (ctx->API != API_OPENGL_CORE)
      ctx->Extensions.ARB_color_buffer_float = true;

   if (ctx->Mesa_DXTn || driQueryOptionb(&brw->optionCache, "force_s3tc_enable"))
      ctx->Extensions.EXT_texture_compression_s3tc = true;

   ctx->Extensions.ANGLE_texture_compression_dxt = true;
}
