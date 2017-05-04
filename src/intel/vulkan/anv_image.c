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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "anv_private.h"
#include "util/debug.h"

#include "vk_format_info.h"

/**
 * Exactly one bit must be set in \a aspect.
 */
static isl_surf_usage_flags_t
choose_isl_surf_usage(VkImageUsageFlags vk_usage,
                      VkImageAspectFlags aspect)
{
   isl_surf_usage_flags_t isl_usage = 0;

   if (vk_usage & VK_IMAGE_USAGE_SAMPLED_BIT)
      isl_usage |= ISL_SURF_USAGE_TEXTURE_BIT;

   if (vk_usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
      isl_usage |= ISL_SURF_USAGE_TEXTURE_BIT;

   if (vk_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      isl_usage |= ISL_SURF_USAGE_RENDER_TARGET_BIT;

   if (vk_usage & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
      isl_usage |= ISL_SURF_USAGE_CUBE_BIT;

   /* Even if we're only using it for transfer operations, clears to depth and
    * stencil images happen as depth and stencil so they need the right ISL
    * usage bits or else things will fall apart.
    */
   switch (aspect) {
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      isl_usage |= ISL_SURF_USAGE_DEPTH_BIT;
      break;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      isl_usage |= ISL_SURF_USAGE_STENCIL_BIT;
      break;
   case VK_IMAGE_ASPECT_COLOR_BIT:
      break;
   default:
      unreachable("bad VkImageAspect");
   }

   if (vk_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
      /* blorp implements transfers by sampling from the source image. */
      isl_usage |= ISL_SURF_USAGE_TEXTURE_BIT;
   }

   if (vk_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT &&
       aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
      /* blorp implements transfers by rendering into the destination image.
       * Only request this with color images, as we deal with depth/stencil
       * formats differently. */
      isl_usage |= ISL_SURF_USAGE_RENDER_TARGET_BIT;
   }

   return isl_usage;
}

/**
 * Exactly one bit must be set in \a aspect.
 */
static struct anv_surface *
get_surface(struct anv_image *image, VkImageAspectFlags aspect)
{
   switch (aspect) {
   default:
      unreachable("bad VkImageAspect");
   case VK_IMAGE_ASPECT_COLOR_BIT:
      return &image->color_surface;
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      return &image->depth_surface;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      return &image->stencil_surface;
   }
}

static void
add_surface(struct anv_image *image, struct anv_surface *surf)
{
   assert(surf->isl.size > 0); /* isl surface must be initialized */

   surf->offset = align_u32(image->size, surf->isl.alignment);
   image->size = surf->offset + surf->isl.size;
   image->alignment = MAX2(image->alignment, surf->isl.alignment);
}

/**
 * Initialize the anv_image::*_surface selected by \a aspect. Then update the
 * image's memory requirements (that is, the image's size and alignment).
 *
 * Exactly one bit must be set in \a aspect.
 */
static VkResult
make_surface(const struct anv_device *dev,
             struct anv_image *image,
             const struct anv_image_create_info *anv_info,
             VkImageAspectFlags aspect)
{
   const VkImageCreateInfo *vk_info = anv_info->vk_info;
   bool ok UNUSED;

   static const enum isl_surf_dim vk_to_isl_surf_dim[] = {
      [VK_IMAGE_TYPE_1D] = ISL_SURF_DIM_1D,
      [VK_IMAGE_TYPE_2D] = ISL_SURF_DIM_2D,
      [VK_IMAGE_TYPE_3D] = ISL_SURF_DIM_3D,
   };

   /* Translate the Vulkan tiling to an equivalent ISL tiling, then filter the
    * result with an optionally provided ISL tiling argument.
    */
   isl_tiling_flags_t tiling_flags =
      (vk_info->tiling == VK_IMAGE_TILING_LINEAR) ?
      ISL_TILING_LINEAR_BIT : ISL_TILING_ANY_MASK;

   if (anv_info->isl_tiling_flags)
      tiling_flags &= anv_info->isl_tiling_flags;

   assert(tiling_flags);

   struct anv_surface *anv_surf = get_surface(image, aspect);

   image->extent = anv_sanitize_image_extent(vk_info->imageType,
                                             vk_info->extent);

   enum isl_format format = anv_get_isl_format(&dev->info, vk_info->format,
                                               aspect, vk_info->tiling);
   assert(format != ISL_FORMAT_UNSUPPORTED);

   ok = isl_surf_init(&dev->isl_dev, &anv_surf->isl,
      .dim = vk_to_isl_surf_dim[vk_info->imageType],
      .format = format,
      .width = image->extent.width,
      .height = image->extent.height,
      .depth = image->extent.depth,
      .levels = vk_info->mipLevels,
      .array_len = vk_info->arrayLayers,
      .samples = vk_info->samples,
      .min_alignment = 0,
      .row_pitch = anv_info->stride,
      .usage = choose_isl_surf_usage(image->usage, aspect),
      .tiling_flags = tiling_flags);

   /* isl_surf_init() will fail only if provided invalid input. Invalid input
    * is illegal in Vulkan.
    */
   assert(ok);

   add_surface(image, anv_surf);

   /* Add a HiZ surface to a depth buffer that will be used for rendering.
    */
   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      /* We don't advertise that depth buffers could be used as storage
       * images.
       */
       assert(!(image->usage & VK_IMAGE_USAGE_STORAGE_BIT));

      /* Allow the user to control HiZ enabling. Disable by default on gen7
       * because resolves are not currently implemented pre-BDW.
       */
      if (!(image->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
         /* It will never be used as an attachment, HiZ is pointless. */
      } else if (dev->info.gen == 7) {
         anv_perf_warn("Implement gen7 HiZ");
      } else if (vk_info->mipLevels > 1) {
         anv_perf_warn("Enable multi-LOD HiZ");
      } else if (vk_info->arrayLayers > 1) {
         anv_perf_warn("Implement multi-arrayLayer HiZ clears and resolves");
      } else if (dev->info.gen == 8 && vk_info->samples > 1) {
         anv_perf_warn("Enable gen8 multisampled HiZ");
      } else if (!unlikely(INTEL_DEBUG & DEBUG_NO_HIZ)) {
         assert(image->aux_surface.isl.size == 0);
         ok = isl_surf_get_hiz_surf(&dev->isl_dev, &image->depth_surface.isl,
                                    &image->aux_surface.isl);
         assert(ok);
         add_surface(image, &image->aux_surface);
         image->aux_usage = ISL_AUX_USAGE_HIZ;
      }
   } else if (aspect == VK_IMAGE_ASPECT_COLOR_BIT && vk_info->samples == 1) {
      if (!unlikely(INTEL_DEBUG & DEBUG_NO_RBC)) {
         assert(image->aux_surface.isl.size == 0);
         ok = isl_surf_get_ccs_surf(&dev->isl_dev, &anv_surf->isl,
                                    &image->aux_surface.isl);
         if (ok) {
            add_surface(image, &image->aux_surface);

            /* For images created without MUTABLE_FORMAT_BIT set, we know that
             * they will always be used with the original format.  In
             * particular, they will always be used with a format that
             * supports color compression.  If it's never used as a storage
             * image, then it will only be used through the sampler or the as
             * a render target.  This means that it's safe to just leave
             * compression on at all times for these formats.
             */
            if (!(vk_info->usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
                !(vk_info->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
                isl_format_supports_ccs_e(&dev->info, format)) {
               image->aux_usage = ISL_AUX_USAGE_CCS_E;
            }
         }
      }
   } else if (aspect == VK_IMAGE_ASPECT_COLOR_BIT && vk_info->samples > 1) {
      assert(image->aux_surface.isl.size == 0);
      assert(!(vk_info->usage & VK_IMAGE_USAGE_STORAGE_BIT));
      ok = isl_surf_get_mcs_surf(&dev->isl_dev, &anv_surf->isl,
                                 &image->aux_surface.isl);
      if (ok) {
         add_surface(image, &image->aux_surface);
         image->aux_usage = ISL_AUX_USAGE_MCS;
      }
   }

   return VK_SUCCESS;
}

VkResult
anv_image_create(VkDevice _device,
                 const struct anv_image_create_info *create_info,
                 const VkAllocationCallbacks* alloc,
                 VkImage *pImage)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   const VkImageCreateInfo *pCreateInfo = create_info->vk_info;
   struct anv_image *image = NULL;
   VkResult r;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);

   anv_assert(pCreateInfo->mipLevels > 0);
   anv_assert(pCreateInfo->arrayLayers > 0);
   anv_assert(pCreateInfo->samples > 0);
   anv_assert(pCreateInfo->extent.width > 0);
   anv_assert(pCreateInfo->extent.height > 0);
   anv_assert(pCreateInfo->extent.depth > 0);

   image = vk_alloc2(&device->alloc, alloc, sizeof(*image), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   memset(image, 0, sizeof(*image));
   image->type = pCreateInfo->imageType;
   image->extent = pCreateInfo->extent;
   image->vk_format = pCreateInfo->format;
   image->aspects = vk_format_aspects(image->vk_format);
   image->levels = pCreateInfo->mipLevels;
   image->array_size = pCreateInfo->arrayLayers;
   image->samples = pCreateInfo->samples;
   image->usage = pCreateInfo->usage;
   image->tiling = pCreateInfo->tiling;
   image->aux_usage = ISL_AUX_USAGE_NONE;

   uint32_t b;
   for_each_bit(b, image->aspects) {
      r = make_surface(device, image, create_info, (1 << b));
      if (r != VK_SUCCESS)
         goto fail;
   }

   *pImage = anv_image_to_handle(image);

   return VK_SUCCESS;

fail:
   if (image)
      vk_free2(&device->alloc, alloc, image);

   return r;
}

VkResult
anv_CreateImage(VkDevice device,
                const VkImageCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator,
                VkImage *pImage)
{
   return anv_image_create(device,
      &(struct anv_image_create_info) {
         .vk_info = pCreateInfo,
      },
      pAllocator,
      pImage);
}

void
anv_DestroyImage(VkDevice _device, VkImage _image,
                 const VkAllocationCallbacks *pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image, image, _image);

   if (!image)
      return;

   vk_free2(&device->alloc, pAllocator, image);
}

VkResult anv_BindImageMemory(
    VkDevice                                    _device,
    VkImage                                     _image,
    VkDeviceMemory                              _memory,
    VkDeviceSize                                memoryOffset)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_device_memory, mem, _memory);
   ANV_FROM_HANDLE(anv_image, image, _image);

   if (mem == NULL) {
      image->bo = NULL;
      image->offset = 0;
      return VK_SUCCESS;
   }

   image->bo = mem->bo;
   image->offset = memoryOffset;

   if (image->aux_surface.isl.size > 0) {

      /* The offset and size must be a multiple of 4K or else the
       * anv_gem_mmap call below will return NULL.
       */
      assert((image->offset + image->aux_surface.offset) % 4096 == 0);
      assert(image->aux_surface.isl.size % 4096 == 0);

      /* Auxiliary surfaces need to have their memory cleared to 0 before they
       * can be used.  For CCS surfaces, this puts them in the "resolved"
       * state so they can be used with CCS enabled before we ever touch it
       * from the GPU.  For HiZ, we need something valid or else we may get
       * GPU hangs on some hardware and 0 works fine.
       */
      void *map = anv_gem_mmap(device, image->bo->gem_handle,
                               image->offset + image->aux_surface.offset,
                               image->aux_surface.isl.size,
                               device->info.has_llc ? 0 : I915_MMAP_WC);

      if (map == MAP_FAILED)
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

      memset(map, 0, image->aux_surface.isl.size);

      anv_gem_munmap(map, image->aux_surface.isl.size);
   }

   return VK_SUCCESS;
}

static void
anv_surface_get_subresource_layout(struct anv_image *image,
                                   struct anv_surface *surface,
                                   const VkImageSubresource *subresource,
                                   VkSubresourceLayout *layout)
{
   /* If we are on a non-zero mip level or array slice, we need to
    * calculate a real offset.
    */
   anv_assert(subresource->mipLevel == 0);
   anv_assert(subresource->arrayLayer == 0);

   layout->offset = surface->offset;
   layout->rowPitch = surface->isl.row_pitch;
   layout->depthPitch = isl_surf_get_array_pitch(&surface->isl);
   layout->arrayPitch = isl_surf_get_array_pitch(&surface->isl);
   layout->size = surface->isl.size;
}

void anv_GetImageSubresourceLayout(
    VkDevice                                    device,
    VkImage                                     _image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout)
{
   ANV_FROM_HANDLE(anv_image, image, _image);

   assert(__builtin_popcount(pSubresource->aspectMask) == 1);

   switch (pSubresource->aspectMask) {
   case VK_IMAGE_ASPECT_COLOR_BIT:
      anv_surface_get_subresource_layout(image, &image->color_surface,
                                         pSubresource, pLayout);
      break;
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      anv_surface_get_subresource_layout(image, &image->depth_surface,
                                         pSubresource, pLayout);
      break;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      anv_surface_get_subresource_layout(image, &image->stencil_surface,
                                         pSubresource, pLayout);
      break;
   default:
      assert(!"Invalid image aspect");
   }
}

/**
 * This function determines the optimal buffer to use for device
 * accesses given a VkImageLayout and other pieces of information needed to
 * make that determination. This does not determine the optimal buffer to
 * use during a resolve operation.
 *
 * NOTE: Some layouts do not support device access.
 *
 * @param devinfo The device information of the Intel GPU.
 * @param image The image that may contain a collection of buffers.
 * @param aspects The aspect(s) of the image to be accessed.
 * @param layout The current layout of the image aspect(s).
 *
 * @return The primary buffer that should be used for the given layout.
 */
enum isl_aux_usage
anv_layout_to_aux_usage(const struct gen_device_info * const devinfo,
                        const struct anv_image * const image,
                        const VkImageAspectFlags aspects,
                        const VkImageLayout layout)
{
   /* Validate the inputs. */

   /* The devinfo is needed as the optimal buffer varies across generations. */
   assert(devinfo != NULL);

   /* The layout of a NULL image is not properly defined. */
   assert(image != NULL);

   /* The aspects must be a subset of the image aspects. */
   assert(aspects & image->aspects && aspects <= image->aspects);

   /* Determine the optimal buffer. */

   /* If there is no auxiliary surface allocated, we must use the one and only
    * main buffer.
    */
   if (image->aux_surface.isl.size == 0)
      return ISL_AUX_USAGE_NONE;

   /* All images that use an auxiliary surface are required to be tiled. */
   assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);

   /* On BDW+, when clearing the stencil aspect of a depth stencil image,
    * the HiZ buffer allows us to record the clear with a relatively small
    * number of packets. Prior to BDW, the HiZ buffer provides no known benefit
    * to the stencil aspect.
    */
   if (devinfo->gen < 8 && aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      return ISL_AUX_USAGE_NONE;

   const bool color_aspect = aspects == VK_IMAGE_ASPECT_COLOR_BIT;

   /* The following switch currently only handles depth stencil aspects.
    * TODO: Handle the color aspect.
    */
   if (color_aspect)
      return image->aux_usage;

   switch (layout) {

   /* Invalid Layouts */

   /* According to the Vulkan Spec, the following layouts are valid only as
    * initial layouts in a layout transition and don't support device access.
    */
   case VK_IMAGE_LAYOUT_UNDEFINED:
   case VK_IMAGE_LAYOUT_PREINITIALIZED:
   case VK_IMAGE_LAYOUT_RANGE_SIZE:
   case VK_IMAGE_LAYOUT_MAX_ENUM:
      unreachable("Invalid image layout for device access.");


   /* Transfer Layouts
    *
    * This buffer could be a depth buffer used in a transfer operation. BLORP
    * currently doesn't use HiZ for transfer operations so we must use the main
    * buffer for this layout. TODO: Enable HiZ in BLORP.
    */
   case VK_IMAGE_LAYOUT_GENERAL:
   case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
   case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return ISL_AUX_USAGE_NONE;


   /* Sampling Layouts */
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      assert(!color_aspect);
      /* Fall-through */
   case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      if (anv_can_sample_with_hiz(devinfo, aspects, image->samples))
         return ISL_AUX_USAGE_HIZ;
      else
         return ISL_AUX_USAGE_NONE;

   case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
      assert(color_aspect);

      /* On SKL+, the render buffer can be decompressed by the presentation
       * engine. Support for this feature has not yet landed in the wider
       * ecosystem. TODO: Update this code when support lands.
       *
       * From the BDW PRM, Vol 7, Render Target Resolve:
       *
       *    If the MCS is enabled on a non-multisampled render target, the
       *    render target must be resolved before being used for other
       *    purposes (display, texture, CPU lock) The clear value from
       *    SURFACE_STATE is written into pixels in the render target
       *    indicated as clear in the MCS.
       *
       * Pre-SKL, the render buffer must be resolved before being used for
       * presentation. We can infer that the auxiliary buffer is not used.
       */
      return ISL_AUX_USAGE_NONE;


   /* Rendering Layouts */
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      assert(color_aspect);
      unreachable("Color images are not yet supported.");

   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      assert(!color_aspect);
      return ISL_AUX_USAGE_HIZ;
   }

   /* If the layout isn't recognized in the exhaustive switch above, the
    * VkImageLayout value is not defined in vulkan.h.
    */
   unreachable("layout is not a VkImageLayout enumeration member.");
}


static struct anv_state
alloc_surface_state(struct anv_device *device)
{
   return anv_state_pool_alloc(&device->surface_state_pool, 64, 64);
}

static enum isl_channel_select
remap_swizzle(VkComponentSwizzle swizzle, VkComponentSwizzle component,
              struct isl_swizzle format_swizzle)
{
   if (swizzle == VK_COMPONENT_SWIZZLE_IDENTITY)
      swizzle = component;

   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_ZERO:  return ISL_CHANNEL_SELECT_ZERO;
   case VK_COMPONENT_SWIZZLE_ONE:   return ISL_CHANNEL_SELECT_ONE;
   case VK_COMPONENT_SWIZZLE_R:     return format_swizzle.r;
   case VK_COMPONENT_SWIZZLE_G:     return format_swizzle.g;
   case VK_COMPONENT_SWIZZLE_B:     return format_swizzle.b;
   case VK_COMPONENT_SWIZZLE_A:     return format_swizzle.a;
   default:
      unreachable("Invalid swizzle");
   }
}


VkResult
anv_CreateImageView(VkDevice _device,
                    const VkImageViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkImageView *pView)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image, image, pCreateInfo->image);
   struct anv_image_view *iview;

   iview = vk_alloc2(&device->alloc, pAllocator, sizeof(*iview), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (iview == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   const VkImageSubresourceRange *range = &pCreateInfo->subresourceRange;

   assert(range->layerCount > 0);
   assert(range->baseMipLevel < image->levels);
   assert(image->usage & (VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_STORAGE_BIT |
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));

   switch (image->type) {
   default:
      unreachable("bad VkImageType");
   case VK_IMAGE_TYPE_1D:
   case VK_IMAGE_TYPE_2D:
      assert(range->baseArrayLayer + anv_get_layerCount(image, range) - 1 <= image->array_size);
      break;
   case VK_IMAGE_TYPE_3D:
      assert(range->baseArrayLayer + anv_get_layerCount(image, range) - 1
             <= anv_minify(image->extent.depth, range->baseMipLevel));
      break;
   }

   const struct anv_surface *surface =
      anv_image_get_surface_for_aspect_mask(image, range->aspectMask);

   iview->image = image;
   iview->bo = image->bo;
   iview->offset = image->offset + surface->offset;

   iview->aspect_mask = pCreateInfo->subresourceRange.aspectMask;
   iview->vk_format = pCreateInfo->format;

   struct anv_format format = anv_get_format(&device->info, pCreateInfo->format,
                                             range->aspectMask, image->tiling);

   iview->isl = (struct isl_view) {
      .format = format.isl_format,
      .base_level = range->baseMipLevel,
      .levels = anv_get_levelCount(image, range),
      .base_array_layer = range->baseArrayLayer,
      .array_len = anv_get_layerCount(image, range),
      .swizzle = {
         .r = remap_swizzle(pCreateInfo->components.r,
                            VK_COMPONENT_SWIZZLE_R, format.swizzle),
         .g = remap_swizzle(pCreateInfo->components.g,
                            VK_COMPONENT_SWIZZLE_G, format.swizzle),
         .b = remap_swizzle(pCreateInfo->components.b,
                            VK_COMPONENT_SWIZZLE_B, format.swizzle),
         .a = remap_swizzle(pCreateInfo->components.a,
                            VK_COMPONENT_SWIZZLE_A, format.swizzle),
      },
   };

   iview->extent = (VkExtent3D) {
      .width  = anv_minify(image->extent.width , range->baseMipLevel),
      .height = anv_minify(image->extent.height, range->baseMipLevel),
      .depth  = anv_minify(image->extent.depth , range->baseMipLevel),
   };

   if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_3D) {
      iview->isl.base_array_layer = 0;
      iview->isl.array_len = iview->extent.depth;
   }

   if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE ||
       pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
      iview->isl.usage = ISL_SURF_USAGE_CUBE_BIT;
   } else {
      iview->isl.usage = 0;
   }

   /* Input attachment surfaces for color are allocated and filled
    * out at BeginRenderPass time because they need compression information.
    * Compression is not yet enabled for depth textures and stencil doesn't
    * allow compression so we can just use the texture surface state from the
    * view.
    */
   if (image->usage & VK_IMAGE_USAGE_SAMPLED_BIT ||
       (image->usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT &&
        !(iview->aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT))) {
      iview->sampler_surface_state = alloc_surface_state(device);
      iview->no_aux_sampler_surface_state = alloc_surface_state(device);

      /* Sampling is performed in one of two buffer configurations in anv: with
       * an auxiliary buffer or without it. Sampler states aren't always needed
       * for both configurations, but are currently created unconditionally for
       * simplicity.
       *
       * TODO: Consider allocating each surface state only when necessary.
       */

      /* Create a sampler state with the optimal aux_usage for sampling. This
       * may use the aux_buffer.
       */
      const enum isl_aux_usage surf_usage =
         anv_layout_to_aux_usage(&device->info, image, iview->aspect_mask,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      /* If this is a HiZ buffer we can sample from with a programmable clear
       * value (SKL+), define the clear value to the optimal constant.
       */
      const float red_clear_color = surf_usage == ISL_AUX_USAGE_HIZ &&
                                    device->info.gen >= 9 ?
                                    ANV_HZ_FC_VAL : 0.0f;

      struct isl_view view = iview->isl;
      view.usage |= ISL_SURF_USAGE_TEXTURE_BIT;
      isl_surf_fill_state(&device->isl_dev,
                          iview->sampler_surface_state.map,
                          .surf = &surface->isl,
                          .view = &view,
                          .clear_color.f32 = { red_clear_color,},
                          .aux_surf = &image->aux_surface.isl,
                          .aux_usage = surf_usage,
                          .mocs = device->default_mocs);

      /* Create a sampler state that only uses the main buffer. */
      isl_surf_fill_state(&device->isl_dev,
                          iview->no_aux_sampler_surface_state.map,
                          .surf = &surface->isl,
                          .view = &view,
                          .mocs = device->default_mocs);

      anv_state_flush(device, iview->sampler_surface_state);
      anv_state_flush(device, iview->no_aux_sampler_surface_state);
   } else {
      iview->sampler_surface_state.alloc_size = 0;
      iview->no_aux_sampler_surface_state.alloc_size = 0;
   }

   /* NOTE: This one needs to go last since it may stomp isl_view.format */
   if (image->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
      iview->storage_surface_state = alloc_surface_state(device);
      iview->writeonly_storage_surface_state = alloc_surface_state(device);

      struct isl_view view = iview->isl;
      view.usage |= ISL_SURF_USAGE_STORAGE_BIT;

      /* Write-only accesses always used a typed write instruction and should
       * therefore use the real format.
       */
      isl_surf_fill_state(&device->isl_dev,
                          iview->writeonly_storage_surface_state.map,
                          .surf = &surface->isl,
                          .view = &view,
                          .aux_surf = &image->aux_surface.isl,
                          .aux_usage = image->aux_usage,
                          .mocs = device->default_mocs);

      if (isl_has_matching_typed_storage_image_format(&device->info,
                                                      format.isl_format)) {
         /* Typed surface reads support a very limited subset of the shader
          * image formats.  Translate it into the closest format the hardware
          * supports.
          */
         view.format = isl_lower_storage_image_format(&device->info,
                                                      format.isl_format);

         isl_surf_fill_state(&device->isl_dev,
                             iview->storage_surface_state.map,
                             .surf = &surface->isl,
                             .view = &view,
                             .aux_surf = &image->aux_surface.isl,
                             .aux_usage = image->aux_usage,
                             .mocs = device->default_mocs);
      } else {
         anv_fill_buffer_surface_state(device, iview->storage_surface_state,
                                       ISL_FORMAT_RAW,
                                       iview->offset,
                                       iview->bo->size - iview->offset, 1);
      }

      isl_surf_fill_image_param(&device->isl_dev,
                                &iview->storage_image_param,
                                &surface->isl, &iview->isl);

      anv_state_flush(device, iview->storage_surface_state);
      anv_state_flush(device, iview->writeonly_storage_surface_state);
   } else {
      iview->storage_surface_state.alloc_size = 0;
      iview->writeonly_storage_surface_state.alloc_size = 0;
   }

   *pView = anv_image_view_to_handle(iview);

   return VK_SUCCESS;
}

void
anv_DestroyImageView(VkDevice _device, VkImageView _iview,
                     const VkAllocationCallbacks *pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image_view, iview, _iview);

   if (!iview)
      return;

   if (iview->sampler_surface_state.alloc_size > 0) {
      anv_state_pool_free(&device->surface_state_pool,
                          iview->sampler_surface_state);
   }

   if (iview->storage_surface_state.alloc_size > 0) {
      anv_state_pool_free(&device->surface_state_pool,
                          iview->storage_surface_state);
   }

   if (iview->writeonly_storage_surface_state.alloc_size > 0) {
      anv_state_pool_free(&device->surface_state_pool,
                          iview->writeonly_storage_surface_state);
   }

   vk_free2(&device->alloc, pAllocator, iview);
}


VkResult
anv_CreateBufferView(VkDevice _device,
                     const VkBufferViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkBufferView *pView)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_buffer, buffer, pCreateInfo->buffer);
   struct anv_buffer_view *view;

   view = vk_alloc2(&device->alloc, pAllocator, sizeof(*view), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   /* TODO: Handle the format swizzle? */

   view->format = anv_get_isl_format(&device->info, pCreateInfo->format,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_IMAGE_TILING_LINEAR);
   const uint32_t format_bs = isl_format_get_layout(view->format)->bpb / 8;
   view->bo = buffer->bo;
   view->offset = buffer->offset + pCreateInfo->offset;
   view->range = anv_buffer_get_range(buffer, pCreateInfo->offset,
                                              pCreateInfo->range);
   view->range = align_down_npot_u32(view->range, format_bs);

   if (buffer->usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) {
      view->surface_state = alloc_surface_state(device);

      anv_fill_buffer_surface_state(device, view->surface_state,
                                    view->format,
                                    view->offset, view->range, format_bs);
   } else {
      view->surface_state = (struct anv_state){ 0 };
   }

   if (buffer->usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) {
      view->storage_surface_state = alloc_surface_state(device);
      view->writeonly_storage_surface_state = alloc_surface_state(device);

      enum isl_format storage_format =
         isl_has_matching_typed_storage_image_format(&device->info,
                                                     view->format) ?
         isl_lower_storage_image_format(&device->info, view->format) :
         ISL_FORMAT_RAW;

      anv_fill_buffer_surface_state(device, view->storage_surface_state,
                                    storage_format,
                                    view->offset, view->range,
                                    (storage_format == ISL_FORMAT_RAW ? 1 :
                                     isl_format_get_layout(storage_format)->bpb / 8));

      /* Write-only accesses should use the original format. */
      anv_fill_buffer_surface_state(device, view->writeonly_storage_surface_state,
                                    view->format,
                                    view->offset, view->range,
                                    isl_format_get_layout(view->format)->bpb / 8);

      isl_buffer_fill_image_param(&device->isl_dev,
                                  &view->storage_image_param,
                                  view->format, view->range);
   } else {
      view->storage_surface_state = (struct anv_state){ 0 };
      view->writeonly_storage_surface_state = (struct anv_state){ 0 };
   }

   *pView = anv_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

void
anv_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                      const VkAllocationCallbacks *pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_buffer_view, view, bufferView);

   if (!view)
      return;

   if (view->surface_state.alloc_size > 0)
      anv_state_pool_free(&device->surface_state_pool,
                          view->surface_state);

   if (view->storage_surface_state.alloc_size > 0)
      anv_state_pool_free(&device->surface_state_pool,
                          view->storage_surface_state);

   if (view->writeonly_storage_surface_state.alloc_size > 0)
      anv_state_pool_free(&device->surface_state_pool,
                          view->writeonly_storage_surface_state);

   vk_free2(&device->alloc, pAllocator, view);
}

const struct anv_surface *
anv_image_get_surface_for_aspect_mask(const struct anv_image *image,
                                      VkImageAspectFlags aspect_mask)
{
   switch (aspect_mask) {
   case VK_IMAGE_ASPECT_COLOR_BIT:
      assert(image->aspects == VK_IMAGE_ASPECT_COLOR_BIT);
      return &image->color_surface;
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      assert(image->aspects & VK_IMAGE_ASPECT_DEPTH_BIT);
      return &image->depth_surface;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      assert(image->aspects & VK_IMAGE_ASPECT_STENCIL_BIT);
      return &image->stencil_surface;
   case VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT:
      /* FINISHME: The Vulkan spec (git a511ba2) requires support for
       * combined depth stencil formats. Specifically, it states:
       *
       *    At least one of ename:VK_FORMAT_D24_UNORM_S8_UINT or
       *    ename:VK_FORMAT_D32_SFLOAT_S8_UINT must be supported.
       *
       * Image views with both depth and stencil aspects are only valid for
       * render target attachments, in which case
       * cmd_buffer_emit_depth_stencil() will pick out both the depth and
       * stencil surfaces from the underlying surface.
       */
      if (image->aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
         return &image->depth_surface;
      } else {
         assert(image->aspects == VK_IMAGE_ASPECT_STENCIL_BIT);
         return &image->stencil_surface;
      }
    default:
       unreachable("image does not have aspect");
       return NULL;
   }
}
