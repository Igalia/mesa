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

#include "brw_context.h"

size_t
brw_query_samples_for_format(struct gl_context *ctx, GLenum target,
                             GLenum internalFormat, int samples[16])
{
   struct brw_context *brw = brw_context(ctx);

   (void) target;
   (void) internalFormat;

   switch (brw->gen) {
   case 9:
      samples[0] = 16;
      samples[1] = 8;
      samples[2] = 4;
      samples[3] = 2;
      return 4;

   case 8:
      samples[0] = 8;
      samples[1] = 4;
      samples[2] = 2;
      return 3;

   case 7:
      samples[0] = 8;
      samples[1] = 4;
      return 2;

   case 6:
      samples[0] = 4;
      return 1;

   default:
      assert(brw->gen < 6);
      samples[0] = 1;
      return 1;
   }
}

void
brw_query_internal_format(struct gl_context *ctx, GLenum target,
                          GLenum internalFormat, GLenum pname, GLint *params)
{
   /* The API entry-point gives us a temporary params buffer that is non-NULL
    * and guaranteed to have at least 16 elements.
    */
   assert(params != NULL);

   switch (pname) {
   case GL_SAMPLES:
      /* @TODO */
      break;

   case GL_NUM_SAMPLE_COUNTS:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_SUPPORTED:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_PREFERRED:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_RED_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_GREEN_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_BLUE_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_ALPHA_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_DEPTH_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_STENCIL_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_SHARED_SIZE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_RED_TYPE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_GREEN_TYPE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_BLUE_TYPE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_ALPHA_TYPE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_DEPTH_TYPE:
      /* @TODO */
      break;

   case GL_INTERNALFORMAT_STENCIL_TYPE:
      /* @TODO */
      break;

   case GL_MAX_WIDTH:
      /* @TODO */
      break;

   case GL_MAX_HEIGHT:
      /* @TODO */
      break;

   case GL_MAX_DEPTH:
      /* @TODO */
      break;

   case GL_MAX_LAYERS:
      /* @TODO */
      break;

   case GL_MAX_COMBINED_DIMENSIONS:
      /* @TODO */
      break;

   case GL_COLOR_COMPONENTS:
      /* @TODO */
      break;

   case GL_DEPTH_COMPONENTS:
      /* @TODO */
      break;

   case GL_STENCIL_COMPONENTS:
      /* @TODO */
      break;

   case GL_COLOR_RENDERABLE:
      /* @TODO */
      break;

   case GL_DEPTH_RENDERABLE:
      /* @TODO */
      break;

   case GL_STENCIL_RENDERABLE:
      /* @TODO */
      break;

   case GL_FRAMEBUFFER_RENDERABLE:
      /* @TODO */
      break;

   case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
      /* @TODO */
      break;

   case GL_FRAMEBUFFER_BLEND:
      /* @TODO */
      break;

   case GL_READ_PIXELS:
      /* @TODO */
      break;

   case GL_READ_PIXELS_FORMAT:
      /* @TODO */
      break;

   case GL_READ_PIXELS_TYPE:
      /* @TODO */
      break;

   case GL_TEXTURE_IMAGE_FORMAT:
      /* @TODO */
      break;

   case GL_TEXTURE_IMAGE_TYPE:
      /* @TODO */
      break;

   case GL_GET_TEXTURE_IMAGE_FORMAT:
      /* @TODO */
      break;

   case GL_GET_TEXTURE_IMAGE_TYPE:
      /* @TODO */
      break;

   case GL_MIPMAP:
      /* @TODO */
      break;

   case GL_MANUAL_GENERATE_MIPMAP:
      /* @TODO */
      break;

   case GL_AUTO_GENERATE_MIPMAP:
      /* @TODO */
      break;

   case GL_COLOR_ENCODING:
      /* @TODO */
      break;

   case GL_SRGB_READ:
      /* @TODO */
      break;

   case GL_SRGB_WRITE:
      /* @TODO */
      break;

   case GL_SRGB_DECODE_ARB:
      /* @TODO */
      break;

   case GL_FILTER:
      /* @TODO */
      break;

   case GL_VERTEX_TEXTURE:
      /* @TODO */
      break;

   case GL_TESS_CONTROL_TEXTURE:
      /* @TODO */
      break;

   case GL_TESS_EVALUATION_TEXTURE:
      /* @TODO */
      break;

   case GL_GEOMETRY_TEXTURE:
      /* @TODO */
      break;

   case GL_FRAGMENT_TEXTURE:
      /* @TODO */
      break;

   case GL_COMPUTE_TEXTURE:
      /* @TODO */
      break;

   case GL_TEXTURE_SHADOW:
      /* @TODO */
      break;

   case GL_TEXTURE_GATHER:
      /* @TODO */
      break;

   case GL_TEXTURE_GATHER_SHADOW:
      /* @TODO */
      break;

   case GL_SHADER_IMAGE_LOAD:
      /* @TODO */
      break;

   case GL_SHADER_IMAGE_STORE:
      /* @TODO */
      break;

   case GL_SHADER_IMAGE_ATOMIC:
      /* @TODO */
      break;

   case GL_IMAGE_TEXEL_SIZE:
      /* @TODO */
      break;

   case GL_IMAGE_COMPATIBILITY_CLASS:
      /* @TODO */
      break;

   case GL_IMAGE_PIXEL_FORMAT:
      /* @TODO */
      break;

   case GL_IMAGE_PIXEL_TYPE:
      /* @TODO */
      break;

   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
      /* @TODO */
      break;

   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
      /* @TODO */
      break;

   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
      /* @TODO */
      break;

   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
      /* @TODO */
      break;

   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
      /* @TODO */
      break;

   case GL_TEXTURE_COMPRESSED:
      /* @TODO */
      break;

   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
      /* @TODO */
      break;

   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
      /* @TODO */
      break;

   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
      /* @TODO */
      break;

   case GL_CLEAR_BUFFER:
      /* @TODO */
      break;

   case GL_TEXTURE_VIEW:
      /* @TODO */
      break;

   case GL_VIEW_COMPATIBILITY_CLASS:
      /* @TODO */
      break;

   default:
      /* An invalid pname should have been filtered-out by the GL API
       * entry point.
       */
      unreachable("Invalid pname");
   }
}
