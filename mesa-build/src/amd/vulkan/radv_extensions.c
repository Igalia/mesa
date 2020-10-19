/*
 * Copyright 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "radv_private.h"

#include "vk_util.h"

/* Convert the VK_USE_PLATFORM_* defines to booleans */
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
#   undef VK_USE_PLATFORM_DISPLAY_KHR
#   define VK_USE_PLATFORM_DISPLAY_KHR true
#else
#   define VK_USE_PLATFORM_DISPLAY_KHR false
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
#   undef VK_USE_PLATFORM_XLIB_KHR
#   define VK_USE_PLATFORM_XLIB_KHR true
#else
#   define VK_USE_PLATFORM_XLIB_KHR false
#endif
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
#   undef VK_USE_PLATFORM_XLIB_XRANDR_EXT
#   define VK_USE_PLATFORM_XLIB_XRANDR_EXT true
#else
#   define VK_USE_PLATFORM_XLIB_XRANDR_EXT false
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
#   undef VK_USE_PLATFORM_XCB_KHR
#   define VK_USE_PLATFORM_XCB_KHR true
#else
#   define VK_USE_PLATFORM_XCB_KHR false
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#   undef VK_USE_PLATFORM_WAYLAND_KHR
#   define VK_USE_PLATFORM_WAYLAND_KHR true
#else
#   define VK_USE_PLATFORM_WAYLAND_KHR false
#endif
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
#   undef VK_USE_PLATFORM_DIRECTFB_EXT
#   define VK_USE_PLATFORM_DIRECTFB_EXT true
#else
#   define VK_USE_PLATFORM_DIRECTFB_EXT false
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#   undef VK_USE_PLATFORM_ANDROID_KHR
#   define VK_USE_PLATFORM_ANDROID_KHR true
#else
#   define VK_USE_PLATFORM_ANDROID_KHR false
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
#   undef VK_USE_PLATFORM_WIN32_KHR
#   define VK_USE_PLATFORM_WIN32_KHR true
#else
#   define VK_USE_PLATFORM_WIN32_KHR false
#endif
#ifdef VK_USE_PLATFORM_VI_NN
#   undef VK_USE_PLATFORM_VI_NN
#   define VK_USE_PLATFORM_VI_NN true
#else
#   define VK_USE_PLATFORM_VI_NN false
#endif
#ifdef VK_USE_PLATFORM_IOS_MVK
#   undef VK_USE_PLATFORM_IOS_MVK
#   define VK_USE_PLATFORM_IOS_MVK true
#else
#   define VK_USE_PLATFORM_IOS_MVK false
#endif
#ifdef VK_USE_PLATFORM_MACOS_MVK
#   undef VK_USE_PLATFORM_MACOS_MVK
#   define VK_USE_PLATFORM_MACOS_MVK true
#else
#   define VK_USE_PLATFORM_MACOS_MVK false
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
#   undef VK_USE_PLATFORM_METAL_EXT
#   define VK_USE_PLATFORM_METAL_EXT true
#else
#   define VK_USE_PLATFORM_METAL_EXT false
#endif
#ifdef VK_USE_PLATFORM_FUCHSIA
#   undef VK_USE_PLATFORM_FUCHSIA
#   define VK_USE_PLATFORM_FUCHSIA true
#else
#   define VK_USE_PLATFORM_FUCHSIA false
#endif
#ifdef VK_USE_PLATFORM_GGP
#   undef VK_USE_PLATFORM_GGP
#   define VK_USE_PLATFORM_GGP true
#else
#   define VK_USE_PLATFORM_GGP false
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
#   undef VK_ENABLE_BETA_EXTENSIONS
#   define VK_ENABLE_BETA_EXTENSIONS true
#else
#   define VK_ENABLE_BETA_EXTENSIONS false
#endif

/* And ANDROID too */
#ifdef ANDROID
#   undef ANDROID
#   define ANDROID true
#else
#   define ANDROID false
#   define ANDROID_API_LEVEL 0
#endif

#define RADV_HAS_SURFACE (VK_USE_PLATFORM_WAYLAND_KHR ||                                        VK_USE_PLATFORM_XCB_KHR ||                                        VK_USE_PLATFORM_XLIB_KHR ||                                        VK_USE_PLATFORM_DISPLAY_KHR)

static const uint32_t MAX_API_VERSION = VK_MAKE_VERSION(1, 2, 145);

VkResult radv_EnumerateInstanceVersion(
    uint32_t*                                   pApiVersion)
{
    *pApiVersion = MAX_API_VERSION;
    return VK_SUCCESS;
}

const VkExtensionProperties radv_instance_extensions[RADV_INSTANCE_EXTENSION_COUNT] = {
   {"VK_KHR_device_group_creation", 1},
   {"VK_KHR_external_fence_capabilities", 1},
   {"VK_KHR_external_memory_capabilities", 1},
   {"VK_KHR_external_semaphore_capabilities", 1},
   {"VK_KHR_get_display_properties2", 1},
   {"VK_KHR_get_physical_device_properties2", 1},
   {"VK_KHR_get_surface_capabilities2", 1},
   {"VK_KHR_surface", 25},
   {"VK_KHR_surface_protected_capabilities", 1},
   {"VK_KHR_wayland_surface", 6},
   {"VK_KHR_xcb_surface", 6},
   {"VK_KHR_xlib_surface", 6},
   {"VK_KHR_display", 23},
   {"VK_EXT_direct_mode_display", 1},
   {"VK_EXT_acquire_xlib_display", 1},
   {"VK_EXT_display_surface_counter", 1},
   {"VK_EXT_debug_report", 9},
};

const struct radv_instance_extension_table radv_instance_extensions_supported = {
   .KHR_device_group_creation = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
   .KHR_external_fence_capabilities = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
   .KHR_external_memory_capabilities = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
   .KHR_external_semaphore_capabilities = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
   .KHR_get_display_properties2 = (!ANDROID || ANDROID_API_LEVEL >= 29) && (VK_USE_PLATFORM_DISPLAY_KHR),
   .KHR_get_physical_device_properties2 = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
   .KHR_get_surface_capabilities2 = (!ANDROID || ANDROID_API_LEVEL >= 26) && (RADV_HAS_SURFACE),
   .KHR_surface = (!ANDROID || ANDROID_API_LEVEL >= 26) && (RADV_HAS_SURFACE),
   .KHR_surface_protected_capabilities = (!ANDROID || ANDROID_API_LEVEL >= 29) && (RADV_HAS_SURFACE),
   .KHR_wayland_surface = (!ANDROID || ANDROID_API_LEVEL >= 26) && (VK_USE_PLATFORM_WAYLAND_KHR),
   .KHR_xcb_surface = (!ANDROID || ANDROID_API_LEVEL >= 26) && (VK_USE_PLATFORM_XCB_KHR),
   .KHR_xlib_surface = (!ANDROID || ANDROID_API_LEVEL >= 26) && (VK_USE_PLATFORM_XLIB_KHR),
   .KHR_display = (!ANDROID || ANDROID_API_LEVEL >= 26) && (VK_USE_PLATFORM_DISPLAY_KHR),
   .EXT_direct_mode_display = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_acquire_xlib_display = VK_USE_PLATFORM_XLIB_XRANDR_EXT,
   .EXT_display_surface_counter = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_debug_report = true,
};

uint32_t
radv_physical_device_api_version(struct radv_physical_device *device)
{
    uint32_t version = 0;

    uint32_t override = vk_get_version_override();
    if (override)
        return MIN2(override, MAX_API_VERSION);

    if (!(true))
        return version;
    version = VK_MAKE_VERSION(1, 0, 145);

    if (!(true))
        return version;
    version = VK_MAKE_VERSION(1, 1, 145);

    if (!(!ANDROID))
        return version;
    version = VK_MAKE_VERSION(1, 2, 145);

    return version;
}

const VkExtensionProperties radv_device_extensions[RADV_DEVICE_EXTENSION_COUNT] = {
   {"VK_ANDROID_external_memory_android_hardware_buffer", 3},
   {"VK_ANDROID_native_buffer", 5},
   {"VK_KHR_16bit_storage", 1},
   {"VK_KHR_bind_memory2", 1},
   {"VK_KHR_buffer_device_address", 1},
   {"VK_KHR_copy_commands2", 1},
   {"VK_KHR_create_renderpass2", 1},
   {"VK_KHR_dedicated_allocation", 3},
   {"VK_KHR_depth_stencil_resolve", 1},
   {"VK_KHR_descriptor_update_template", 1},
   {"VK_KHR_device_group", 4},
   {"VK_KHR_draw_indirect_count", 1},
   {"VK_KHR_driver_properties", 1},
   {"VK_KHR_external_fence", 1},
   {"VK_KHR_external_fence_fd", 1},
   {"VK_KHR_external_memory", 1},
   {"VK_KHR_external_memory_fd", 1},
   {"VK_KHR_external_semaphore", 1},
   {"VK_KHR_external_semaphore_fd", 1},
   {"VK_KHR_get_memory_requirements2", 1},
   {"VK_KHR_image_format_list", 1},
   {"VK_KHR_imageless_framebuffer", 1},
   {"VK_KHR_incremental_present", 1},
   {"VK_KHR_maintenance1", 2},
   {"VK_KHR_maintenance2", 1},
   {"VK_KHR_maintenance3", 1},
   {"VK_KHR_pipeline_executable_properties", 1},
   {"VK_KHR_push_descriptor", 2},
   {"VK_KHR_relaxed_block_layout", 1},
   {"VK_KHR_sampler_mirror_clamp_to_edge", 3},
   {"VK_KHR_sampler_ycbcr_conversion", 14},
   {"VK_KHR_separate_depth_stencil_layouts", 1},
   {"VK_KHR_shader_atomic_int64", 1},
   {"VK_KHR_shader_clock", 1},
   {"VK_KHR_shader_draw_parameters", 1},
   {"VK_KHR_shader_float_controls", 4},
   {"VK_KHR_shader_float16_int8", 1},
   {"VK_KHR_shader_non_semantic_info", 1},
   {"VK_KHR_shader_subgroup_extended_types", 1},
   {"VK_KHR_shader_terminate_invocation", 1},
   {"VK_KHR_spirv_1_4", 1},
   {"VK_KHR_storage_buffer_storage_class", 1},
   {"VK_KHR_swapchain", 70},
   {"VK_KHR_swapchain_mutable_format", 1},
   {"VK_KHR_timeline_semaphore", 2},
   {"VK_KHR_uniform_buffer_standard_layout", 1},
   {"VK_KHR_variable_pointers", 1},
   {"VK_KHR_vulkan_memory_model", 3},
   {"VK_KHR_multiview", 1},
   {"VK_KHR_8bit_storage", 1},
   {"VK_EXT_buffer_device_address", 2},
   {"VK_EXT_calibrated_timestamps", 1},
   {"VK_EXT_conditional_rendering", 2},
   {"VK_EXT_conservative_rasterization", 1},
   {"VK_EXT_custom_border_color", 12},
   {"VK_EXT_display_control", 1},
   {"VK_EXT_depth_clip_enable", 1},
   {"VK_EXT_depth_range_unrestricted", 1},
   {"VK_EXT_descriptor_indexing", 2},
   {"VK_EXT_discard_rectangles", 1},
   {"VK_EXT_extended_dynamic_state", 1},
   {"VK_EXT_external_memory_dma_buf", 1},
   {"VK_EXT_external_memory_host", 1},
   {"VK_EXT_global_priority", 2},
   {"VK_EXT_host_query_reset", 1},
   {"VK_EXT_image_robustness", 1},
   {"VK_EXT_index_type_uint8", 1},
   {"VK_EXT_inline_uniform_block", 1},
   {"VK_EXT_line_rasterization", 1},
   {"VK_EXT_memory_budget", 1},
   {"VK_EXT_memory_priority", 1},
   {"VK_EXT_pci_bus_info", 2},
   {"VK_EXT_pipeline_creation_cache_control", 3},
   {"VK_EXT_pipeline_creation_feedback", 1},
   {"VK_EXT_post_depth_coverage", 1},
   {"VK_EXT_private_data", 1},
   {"VK_EXT_queue_family_foreign", 1},
   {"VK_EXT_robustness2", 1},
   {"VK_EXT_sample_locations", 1},
   {"VK_EXT_sampler_filter_minmax", 2},
   {"VK_EXT_scalar_block_layout", 1},
   {"VK_EXT_shader_atomic_float", 1},
   {"VK_EXT_shader_demote_to_helper_invocation", 1},
   {"VK_EXT_shader_image_atomic_int64", 1},
   {"VK_EXT_shader_viewport_index_layer", 1},
   {"VK_EXT_shader_stencil_export", 1},
   {"VK_EXT_shader_subgroup_ballot", 1},
   {"VK_EXT_shader_subgroup_vote", 1},
   {"VK_EXT_subgroup_size_control", 2},
   {"VK_EXT_texel_buffer_alignment", 1},
   {"VK_EXT_transform_feedback", 1},
   {"VK_EXT_vertex_attribute_divisor", 3},
   {"VK_EXT_ycbcr_image_arrays", 1},
   {"VK_AMD_buffer_marker", 1},
   {"VK_AMD_device_coherent_memory", 1},
   {"VK_AMD_draw_indirect_count", 2},
   {"VK_AMD_gcn_shader", 1},
   {"VK_AMD_gpu_shader_half_float", 2},
   {"VK_AMD_gpu_shader_int16", 2},
   {"VK_AMD_memory_overallocation_behavior", 1},
   {"VK_AMD_mixed_attachment_samples", 1},
   {"VK_AMD_rasterization_order", 1},
   {"VK_AMD_shader_ballot", 1},
   {"VK_AMD_shader_core_properties", 2},
   {"VK_AMD_shader_core_properties2", 1},
   {"VK_AMD_shader_explicit_vertex_parameter", 1},
   {"VK_AMD_shader_image_load_store_lod", 1},
   {"VK_AMD_shader_fragment_mask", 1},
   {"VK_AMD_shader_info", 1},
   {"VK_AMD_shader_trinary_minmax", 1},
   {"VK_AMD_texture_gather_bias_lod", 1},
   {"VK_GOOGLE_decorate_string", 1},
   {"VK_GOOGLE_hlsl_functionality1", 1},
   {"VK_GOOGLE_user_type", 1},
   {"VK_NV_compute_shader_derivatives", 1},
   {"VK_EXT_4444_formats", 1},
};

void
radv_physical_device_get_supported_extensions(const struct radv_physical_device *device,
                                                   struct radv_device_extension_table *extensions)
{
   *extensions = (struct radv_device_extension_table) {
      .ANDROID_external_memory_android_hardware_buffer = (!ANDROID || ANDROID_API_LEVEL >= 28) && (RADV_SUPPORT_ANDROID_HARDWARE_BUFFER  && device->rad_info.has_syncobj_wait_for_submit),
      .ANDROID_native_buffer = (!ANDROID || ANDROID_API_LEVEL >= 26) && (ANDROID && device->rad_info.has_syncobj_wait_for_submit),
      .KHR_16bit_storage = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_bind_memory2 = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_buffer_device_address = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_copy_commands2 = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_create_renderpass2 = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_dedicated_allocation = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_depth_stencil_resolve = (!ANDROID || ANDROID_API_LEVEL >= 29) && (true),
      .KHR_descriptor_update_template = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
      .KHR_device_group = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_draw_indirect_count = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_driver_properties = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_external_fence = (!ANDROID || ANDROID_API_LEVEL >= 28) && (device->rad_info.has_syncobj_wait_for_submit),
      .KHR_external_fence_fd = (!ANDROID || ANDROID_API_LEVEL >= 28) && (device->rad_info.has_syncobj_wait_for_submit),
      .KHR_external_memory = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_external_memory_fd = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_external_semaphore = (!ANDROID || ANDROID_API_LEVEL >= 28) && (device->rad_info.has_syncobj),
      .KHR_external_semaphore_fd = (!ANDROID || ANDROID_API_LEVEL >= 28) && (device->rad_info.has_syncobj),
      .KHR_get_memory_requirements2 = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_image_format_list = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_imageless_framebuffer = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_incremental_present = (!ANDROID || ANDROID_API_LEVEL >= 26) && (RADV_HAS_SURFACE),
      .KHR_maintenance1 = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
      .KHR_maintenance2 = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_maintenance3 = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_pipeline_executable_properties = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_push_descriptor = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
      .KHR_relaxed_block_layout = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_sampler_mirror_clamp_to_edge = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
      .KHR_sampler_ycbcr_conversion = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_separate_depth_stencil_layouts = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_shader_atomic_int64 = (!ANDROID || ANDROID_API_LEVEL >= 29) && (LLVM_VERSION_MAJOR >= 9 || !device->use_llvm),
      .KHR_shader_clock = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_shader_draw_parameters = (!ANDROID || ANDROID_API_LEVEL >= 26) && (true),
      .KHR_shader_float_controls = (!ANDROID || ANDROID_API_LEVEL >= 29) && (true),
      .KHR_shader_float16_int8 = (!ANDROID || ANDROID_API_LEVEL >= 29) && (true),
      .KHR_shader_non_semantic_info = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_shader_subgroup_extended_types = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_shader_terminate_invocation = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_spirv_1_4 = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_storage_buffer_storage_class = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_swapchain = (!ANDROID || ANDROID_API_LEVEL >= 26) && (RADV_HAS_SURFACE),
      .KHR_swapchain_mutable_format = (!ANDROID || ANDROID_API_LEVEL >= 29) && (RADV_HAS_SURFACE),
      .KHR_timeline_semaphore = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (device->rad_info.has_syncobj_wait_for_submit),
      .KHR_uniform_buffer_standard_layout = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .KHR_variable_pointers = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_vulkan_memory_model = (!ANDROID || ANDROID_API_LEVEL >= 29) && (true),
      .KHR_multiview = (!ANDROID || ANDROID_API_LEVEL >= 28) && (true),
      .KHR_8bit_storage = (!ANDROID || ANDROID_API_LEVEL >= 29) && (true),
      .EXT_buffer_device_address = true,
      .EXT_calibrated_timestamps = true,
      .EXT_conditional_rendering = true,
      .EXT_conservative_rasterization = device->rad_info.chip_class >= GFX9,
      .EXT_custom_border_color = true,
      .EXT_display_control = VK_USE_PLATFORM_DISPLAY_KHR && device->rad_info.has_syncobj_wait_for_submit,
      .EXT_depth_clip_enable = true,
      .EXT_depth_range_unrestricted = true,
      .EXT_descriptor_indexing = true,
      .EXT_discard_rectangles = true,
      .EXT_extended_dynamic_state = true,
      .EXT_external_memory_dma_buf = true,
      .EXT_external_memory_host = device->rad_info.has_userptr,
      .EXT_global_priority = device->rad_info.has_ctx_priority,
      .EXT_host_query_reset = true,
      .EXT_image_robustness = true,
      .EXT_index_type_uint8 = device->rad_info.chip_class >= GFX8,
      .EXT_inline_uniform_block = true,
      .EXT_line_rasterization = device->rad_info.chip_class != GFX9,
      .EXT_memory_budget = true,
      .EXT_memory_priority = true,
      .EXT_pci_bus_info = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_post_depth_coverage = device->rad_info.chip_class >= GFX10,
      .EXT_private_data = true,
      .EXT_queue_family_foreign = true,
      .EXT_robustness2 = true,
      .EXT_sample_locations = device->rad_info.chip_class < GFX10,
      .EXT_sampler_filter_minmax = true,
      .EXT_scalar_block_layout = device->rad_info.chip_class >= GFX7,
      .EXT_shader_atomic_float = true,
      .EXT_shader_demote_to_helper_invocation = LLVM_VERSION_MAJOR >= 9 || !device->use_llvm,
      .EXT_shader_image_atomic_int64 = LLVM_VERSION_MAJOR >= 11 || !device->use_llvm,
      .EXT_shader_viewport_index_layer = true,
      .EXT_shader_stencil_export = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .EXT_subgroup_size_control = true,
      .EXT_texel_buffer_alignment = true,
      .EXT_transform_feedback = true,
      .EXT_vertex_attribute_divisor = true,
      .EXT_ycbcr_image_arrays = true,
      .AMD_buffer_marker = true,
      .AMD_device_coherent_memory = true,
      .AMD_draw_indirect_count = true,
      .AMD_gcn_shader = true,
      .AMD_gpu_shader_half_float = device->rad_info.has_packed_math_16bit,
      .AMD_gpu_shader_int16 = device->rad_info.has_packed_math_16bit,
      .AMD_memory_overallocation_behavior = true,
      .AMD_mixed_attachment_samples = true,
      .AMD_rasterization_order = device->rad_info.has_out_of_order_rast,
      .AMD_shader_ballot = true,
      .AMD_shader_core_properties = true,
      .AMD_shader_core_properties2 = true,
      .AMD_shader_explicit_vertex_parameter = true,
      .AMD_shader_image_load_store_lod = true,
      .AMD_shader_fragment_mask = true,
      .AMD_shader_info = true,
      .AMD_shader_trinary_minmax = true,
      .AMD_texture_gather_bias_lod = true,
      .GOOGLE_decorate_string = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .GOOGLE_hlsl_functionality1 = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .GOOGLE_user_type = (!ANDROID || ANDROID_API_LEVEL >= 9999) && (true),
      .NV_compute_shader_derivatives = true,
      .EXT_4444_formats = true,
   };
}
