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

void
brw_query_internal_format(struct gl_context *ctx, GLenum target,
                          GLenum internalFormat, GLenum pname, GLint *params)
{
   /* The Mesa layer gives us a temporary params buffer that is guaranteed
    * to be non-NULL, and have at least 16 elements.
    */
   assert(params != NULL);

   switch (pname) {
   case GL_SAMPLES:
   case GL_NUM_SAMPLE_COUNTS:
   case GL_INTERNALFORMAT_SUPPORTED:
   case GL_INTERNALFORMAT_PREFERRED:
   case GL_INTERNALFORMAT_RED_SIZE:
   case GL_INTERNALFORMAT_GREEN_SIZE:
   case GL_INTERNALFORMAT_BLUE_SIZE:
   case GL_INTERNALFORMAT_ALPHA_SIZE:
   case GL_INTERNALFORMAT_DEPTH_SIZE:
   case GL_INTERNALFORMAT_STENCIL_SIZE:
   case GL_INTERNALFORMAT_SHARED_SIZE:
   case GL_INTERNALFORMAT_RED_TYPE:
   case GL_INTERNALFORMAT_GREEN_TYPE:
   case GL_INTERNALFORMAT_BLUE_TYPE:
   case GL_INTERNALFORMAT_ALPHA_TYPE:
   case GL_INTERNALFORMAT_DEPTH_TYPE:
   case GL_INTERNALFORMAT_STENCIL_TYPE:
   case GL_MAX_WIDTH:
   case GL_MAX_HEIGHT:
   case GL_MAX_DEPTH:
   case GL_MAX_LAYERS:
   case GL_MAX_COMBINED_DIMENSIONS:
   case GL_COLOR_COMPONENTS:
   case GL_DEPTH_COMPONENTS:
   case GL_STENCIL_COMPONENTS:
   case GL_COLOR_RENDERABLE:
   case GL_DEPTH_RENDERABLE:
   case GL_STENCIL_RENDERABLE:
   case GL_FRAMEBUFFER_RENDERABLE:
   case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
   case GL_FRAMEBUFFER_BLEND:
   case GL_READ_PIXELS:
   case GL_READ_PIXELS_FORMAT:
   case GL_READ_PIXELS_TYPE:
   case GL_TEXTURE_IMAGE_FORMAT:
   case GL_TEXTURE_IMAGE_TYPE:
   case GL_GET_TEXTURE_IMAGE_FORMAT:
   case GL_GET_TEXTURE_IMAGE_TYPE:
   case GL_MIPMAP:
   case GL_MANUAL_GENERATE_MIPMAP:
   case GL_AUTO_GENERATE_MIPMAP:
   case GL_COLOR_ENCODING:
   case GL_SRGB_READ:
   case GL_SRGB_WRITE:
   case GL_SRGB_DECODE_ARB:
   case GL_FILTER:
   case GL_VERTEX_TEXTURE:
   case GL_TESS_CONTROL_TEXTURE:
   case GL_TESS_EVALUATION_TEXTURE:
   case GL_GEOMETRY_TEXTURE:
   case GL_FRAGMENT_TEXTURE:
   case GL_COMPUTE_TEXTURE:
   case GL_TEXTURE_SHADOW:
   case GL_TEXTURE_GATHER:
   case GL_TEXTURE_GATHER_SHADOW:
   case GL_SHADER_IMAGE_LOAD:
   case GL_SHADER_IMAGE_STORE:
   case GL_SHADER_IMAGE_ATOMIC:
   case GL_IMAGE_TEXEL_SIZE:
   case GL_IMAGE_COMPATIBILITY_CLASS:
   case GL_IMAGE_PIXEL_FORMAT:
   case GL_IMAGE_PIXEL_TYPE:
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
   case GL_TEXTURE_COMPRESSED:
   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
   case GL_CLEAR_BUFFER:
   case GL_TEXTURE_VIEW:
   case GL_VIEW_COMPATIBILITY_CLASS:
      /* @TODO: Handle all the different queries */
      _mesa_warning(ctx, "Unimplemented query");
      break;

   default:
      /* An invalid pname should have been filtered-out by the GL API
       * entry point.
       */
      unreachable("Invalid pname");
   }
}
