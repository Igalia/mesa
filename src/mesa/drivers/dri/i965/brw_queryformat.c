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

static size_t
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
      brw_query_samples_for_format(ctx, target, internalFormat, params);
      break;

   case GL_NUM_SAMPLE_COUNTS: {
      size_t num_samples;
      num_samples = brw_query_samples_for_format(ctx, target, internalFormat,
                                                 params);
      params[0] = (GLint) num_samples;
      break;
   }

   /* Grouped queries that should be answered by Mesa frontend,
    * so are unreachable here.
    */
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
   case GL_COLOR_COMPONENTS:
   case GL_DEPTH_COMPONENTS:
   case GL_STENCIL_COMPONENTS:
   case GL_COLOR_RENDERABLE:
   case GL_DEPTH_RENDERABLE:
   case GL_STENCIL_RENDERABLE:
   case GL_TEXTURE_COMPRESSED:
   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:

      unreachable("Mesa should have answered these queries");
      break;

   /* Grouped queries that return NONE support */
   case GL_TESS_CONTROL_TEXTURE:
   case GL_TESS_EVALUATION_TEXTURE:
      params[0] = GL_NONE;
      break;

   case GL_INTERNALFORMAT_SUPPORTED:
      /* Mesa frontend should have answered this query based on the formats
       * known to it. Since we don't support any format on top of what Mesa
       * frontend knows about, we simply return NONE for whatever format
       * reaches this point.
       */
      params[0] = GL_NONE;
      break;

   case GL_INTERNALFORMAT_PREFERRED:
      /* @FIXME: I have doubts about this one */
      params[0] = GL_RGBA8UI;
      break;

   /* Grouped queries that return FULL_SUPPORT */
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
      params[0] = GL_FULL_SUPPORT;
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
