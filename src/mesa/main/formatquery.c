/*
 * Copyright © 2012 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "mtypes.h"
#include "context.h"
#include "glformats.h"
#include "macros.h"
#include "enums.h"
#include "fbobject.h"
#include "formatquery.h"

/* default implementation of QuerySamplesForFormat driverfunc, for
 * non-multisample-capable drivers. */
size_t
_mesa_query_samples_for_format(struct gl_context *ctx, GLenum target,
                               GLenum internalFormat, int samples[16])
{
   (void) target;
   (void) internalFormat;
   (void) ctx;

   samples[0] = 1;
   return 1;
}

/* Sets 'buffer' and 'count' to the appropriate "unsupported" response for each pname.
 *
 * From the ARB_internalformat_query2 specs, "Issues" section:
 * "3 a) What if the combination of <target> and <pname> is invalid/nonsense
 *         (e.g. any texture related query on RENDERBUFFER)?
 *      b) What if the <target>/<pname> make sense, but the <internalformat>
 *        does not for that <pname> (e.g. COLOR_ENCODING for non-color internal
 *        format)?
 * RESOLVED. If the combinations of parameters does not make sense the
 *   reponse best representing "not supported" or "not applicable" is returned
 *   as defined for each <pname>.
 *   In general:
 *     - size- or count-based queries will return zero,
 *    - support-, format- or type-based queries will return NONE,
 *    - boolean-based queries will return FALSE, and
 *    - list-based queries return no entries.
 */
static void
_set_unsupported(GLenum pname, GLint buffer[16], GLsizei *count)
{
   switch(pname) {
   case GL_SAMPLES:
      *count = 0;
      break;
   case GL_NUM_SAMPLE_COUNTS:
   case GL_INTERNALFORMAT_RED_SIZE:
   case GL_INTERNALFORMAT_GREEN_SIZE:
   case GL_INTERNALFORMAT_BLUE_SIZE:
   case GL_INTERNALFORMAT_ALPHA_SIZE:
   case GL_INTERNALFORMAT_DEPTH_SIZE:
   case GL_INTERNALFORMAT_STENCIL_SIZE:
   case GL_INTERNALFORMAT_SHARED_SIZE:
   case GL_MAX_WIDTH:
   case GL_MAX_HEIGHT:
   case GL_MAX_DEPTH:
   case GL_MAX_LAYERS:
   case GL_MAX_COMBINED_DIMENSIONS:
   case GL_IMAGE_TEXEL_SIZE:
   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
      buffer[0] = 0;
      *count = 1;
      break;
   case GL_INTERNALFORMAT_PREFERRED:
   case GL_INTERNALFORMAT_RED_TYPE:
   case GL_INTERNALFORMAT_GREEN_TYPE:
   case GL_INTERNALFORMAT_BLUE_TYPE:
   case GL_INTERNALFORMAT_ALPHA_TYPE:
   case GL_INTERNALFORMAT_DEPTH_TYPE:
   case GL_INTERNALFORMAT_STENCIL_TYPE:
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
   case GL_IMAGE_COMPATIBILITY_CLASS:
   case GL_IMAGE_PIXEL_FORMAT:
   case GL_IMAGE_PIXEL_TYPE:
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
   case GL_CLEAR_BUFFER:
   case GL_TEXTURE_VIEW:
   case GL_VIEW_COMPATIBILITY_CLASS:
      buffer[0] = GL_NONE;
      *count = 1;
      break;
   case GL_INTERNALFORMAT_SUPPORTED:
   case GL_COLOR_COMPONENTS:
   case GL_DEPTH_COMPONENTS:
   case GL_STENCIL_COMPONENTS:
   case GL_COLOR_RENDERABLE:
   case GL_DEPTH_RENDERABLE:
   case GL_STENCIL_RENDERABLE:
   case GL_MIPMAP:
   case GL_TEXTURE_COMPRESSED:
      buffer[0] = GL_FALSE;
      *count = 1;
      break;
   default:
      unreachable("invalid 'pname'");
   }
}

static bool
_legal_parameters(struct gl_context *ctx, GLenum target, GLenum internalformat,
                  GLenum pname, GLsizei bufSize, GLint *params)
{
   switch(target){
   case GL_TEXTURE_1D:
      break;
   case GL_TEXTURE_1D_ARRAY:
      break;
   case GL_TEXTURE_2D:
      break;
   case GL_TEXTURE_2D_ARRAY:
      break;
   case GL_TEXTURE_3D:
      break;
   case GL_TEXTURE_CUBE_MAP:
      break;
   case GL_TEXTURE_CUBE_MAP_ARRAY:
      break;
   case GL_TEXTURE_RECTANGLE:
      break;
   case GL_TEXTURE_BUFFER:
      break;
   case GL_RENDERBUFFER:
      break;
   case GL_TEXTURE_2D_MULTISAMPLE:
      break;
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      break;
   default:
      /* The ARB_internalformat_query spec says:
       *
       * "The INVALID_ENUM error is generated if the <target> parameter to
       * GetInternalformati*v is not one of the targets listed in Table 6.xx.
       *
       * Being Table 6.xxx in the same spec:
       *
       * Target                         Usage
       * -----------------              ------
       * TEXTURE_1D                     1D texture
       * TEXTURE_1D_ARRAY               1D array texture
       * TEXTURE_2D                     2D texture
       * TEXTURE_2D_ARRAY               2D array texture
       * TEXTURE_2D_MULTISAMPLE         2D multisample texture
       * TEXTURE_2D_MULTISAMPLE_ARRAY   2D multisample array texture
       * TEXTURE_3D                     3D texture
       * TEXTURE_BUFFER                 buffer texture
       * TEXTURE_CUBE_MAP               cube map texture
       * TEXTURE_CUBE_MAP_ARRAY         cube map array texture
       * TEXTURE_RECTANGLE              rectangle texture
       * RENDERBUFFER                   renderbuffer
       *
       * Table 6.xx: Possible targets that <internalformat> can be used with
       * and the corresponding usage meaning.
       */
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(target=%s)",
                  _mesa_enum_to_string(target));
      return false;
   }

   switch(pname){
   case GL_SAMPLES:
      break;
   case GL_NUM_SAMPLE_COUNTS:
      break;
   case GL_INTERNALFORMAT_SUPPORTED:
      break;
   case GL_INTERNALFORMAT_PREFERRED:
      break;
   case GL_INTERNALFORMAT_RED_SIZE:
      break;
   case GL_INTERNALFORMAT_GREEN_SIZE:
      break;
   case GL_INTERNALFORMAT_BLUE_SIZE:
      break;
   case GL_INTERNALFORMAT_ALPHA_SIZE:
      break;
   case GL_INTERNALFORMAT_DEPTH_SIZE:
      break;
   case GL_INTERNALFORMAT_STENCIL_SIZE:
      break;
   case GL_INTERNALFORMAT_SHARED_SIZE:
      break;
   case GL_INTERNALFORMAT_RED_TYPE:
      break;
   case GL_INTERNALFORMAT_GREEN_TYPE:
      break;
   case GL_INTERNALFORMAT_BLUE_TYPE:
      break;
   case GL_INTERNALFORMAT_ALPHA_TYPE:
      break;
   case GL_INTERNALFORMAT_DEPTH_TYPE:
      break;
   case GL_INTERNALFORMAT_STENCIL_TYPE:
      break;
   case GL_MAX_WIDTH:
      break;
   case GL_MAX_HEIGHT:
      break;
   case GL_MAX_DEPTH:
      break;
   case GL_MAX_LAYERS:
      break;
   case GL_MAX_COMBINED_DIMENSIONS:
      break;
   case GL_COLOR_COMPONENTS:
      break;
   case GL_DEPTH_COMPONENTS:
      break;
   case GL_STENCIL_COMPONENTS:
      break;
   case GL_COLOR_RENDERABLE:
      break;
   case GL_DEPTH_RENDERABLE:
      break;
   case GL_STENCIL_RENDERABLE:
      break;
   case GL_FRAMEBUFFER_RENDERABLE:
      break;
   case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
      break;
   case GL_FRAMEBUFFER_BLEND:
      break;
   case GL_READ_PIXELS:
      break;
   case GL_READ_PIXELS_FORMAT:
      break;
   case GL_READ_PIXELS_TYPE:
      break;
   case GL_TEXTURE_IMAGE_FORMAT:
      break;
   case GL_TEXTURE_IMAGE_TYPE:
      break;
   case GL_GET_TEXTURE_IMAGE_FORMAT:
      break;
   case GL_GET_TEXTURE_IMAGE_TYPE:
      break;
   case GL_MIPMAP:
      break;
   case GL_MANUAL_GENERATE_MIPMAP:
      break;
   case GL_AUTO_GENERATE_MIPMAP:
      break;
   case GL_COLOR_ENCODING:
      break;
   case GL_SRGB_READ:
      break;
   case GL_SRGB_WRITE:
      break;
   case GL_SRGB_DECODE_ARB:
      /* If ARB_texture_sRGB_decode or EXT_texture_sRGB_decode or
       * equivalent functionality is not supported, queries for the
       * SRGB_DECODE_ARB <pname> set the INVALID_ENUM error.
       */
      if (!ctx->Extensions.EXT_texture_sRGB_decode) {
         _mesa_error(ctx, GL_INVALID_ENUM,
                     "glGetInternalformativ(pname=%s)",
                     _mesa_enum_to_string(pname));
         return false;
      }

      break;
   case GL_FILTER:
      break;
   case GL_VERTEX_TEXTURE:
      break;
   case GL_TESS_CONTROL_TEXTURE:
      break;
   case GL_TESS_EVALUATION_TEXTURE:
      break;
   case GL_GEOMETRY_TEXTURE:
      break;
   case GL_FRAGMENT_TEXTURE:
      break;
   case GL_COMPUTE_TEXTURE:
      break;
   case GL_TEXTURE_SHADOW:
      break;
   case GL_TEXTURE_GATHER:
      break;
   case GL_TEXTURE_GATHER_SHADOW:
      break;
   case GL_SHADER_IMAGE_LOAD:
      break;
   case GL_SHADER_IMAGE_STORE:
      break;
   case GL_SHADER_IMAGE_ATOMIC:
      break;
   case GL_IMAGE_TEXEL_SIZE:
      break;
   case GL_IMAGE_COMPATIBILITY_CLASS:
      break;
   case GL_IMAGE_PIXEL_FORMAT:
      break;
   case GL_IMAGE_PIXEL_TYPE:
      break;
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
      break;
   case GL_TEXTURE_COMPRESSED:
      break;
   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
      break;
   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
      break;
   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
      break;
   case GL_CLEAR_BUFFER:
      break;
   case GL_TEXTURE_VIEW:
      break;
   case GL_VIEW_COMPATIBILITY_CLASS:
      break;
   default:
      /* The ARB_internalformat_query2 spec says:
       *
       * "The INVALID_ENUM error is generated if the <pname> parameter is
       * not one of the listed possibilities."
       *
       * Being the listed possibilities:
       *
       * SAMPLES, NUM_SAMPLE_COUNTS, INTERNALFORMAT_SUPPORTED,
       * INTERNALFORMAT_PREFERRED, INTERNALFORMAT_RED_SIZE,
       * INTERNALFORMAT_GREEN_SIZE, INTERNALFORMAT_BLUE_SIZE,
       * INTERNALFORMAT_ALPHA_SIZE, INTERNALFORMAT_DEPTH_SIZE,
       * INTERNALFORMAT_STENCIL_SIZE, INTERNALFORMAT_SHARED_SIZE,
       * INTERNALFORMAT_RED_TYPE, INTERNALFORMAT_GREEN_TYPE,
       * INTERNALFORMAT_BLUE_TYPE, INTERNALFORMAT_ALPHA_TYPE,
       * INTERNALFORMAT_DEPTH_TYPE, INTERNALFORMAT_STENCIL_TYPE,
       * MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH, MAX_LAYERS, MAX_COMBINED_DIMENSIONS,
       * COLOR_COMPONENTS, DEPTH_COMPONENTS, STENCIL_COMPONENTS,
       * COLOR_RENDERABLE, DEPTH_RENDERABLE, STENCIL_RENDERABLE,
       * FRAMEBUFFER_RENDERABLE, FRAMEBUFFER_RENDERABLE_LAYERED,
       * FRAMEBUFFER_BLEND,
       * READ_PIXELS, READ_PIXELS_FORMAT, READ_PIXELS_TYPE,
       * TEXTURE_IMAGE_FORMAT, TEXTURE_IMAGE_TYPE,
       * GET_TEXTURE_IMAGE_FORMAT, GET_TEXTURE_IMAGE_TYPE,
       * MIPMAP, MANUAL_GENERATE_MIPMAP, AUTO_GENERATE_MIPMAP,
       * COLOR_ENCODING, SRGB_READ, SRGB_WRITE, SRGB_DECODE_ARB, FILTER,
       * VERTEX_TEXTURE, TESS_CONTROL_TEXTURE, TESS_EVALUATION_TEXTURE,
       * GEOMETRY_TEXTURE, FRAGMENT_TEXTURE, COMPUTE_TEXTURE,
       * TEXTURE_SHADOW, TEXTURE_GATHER, TEXTURE_GATHER_SHADOW,
       * SHADER_IMAGE_LOAD, SHADER_IMAGE_STORE, SHADER_IMAGE_ATOMIC,
       * IMAGE_TEXEL_SIZE, IMAGE_COMPATIBILITY_CLASS, IMAGE_PIXEL_FORMAT,
       * IMAGE_PIXEL_TYPE, IMAGE_FORMAT_COMPATIBILITY_TYPE,
       * SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST,
       * SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST,
       * SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE,
       * SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE,
       * TEXTURE_COMPRESSED, TEXTURE_COMPRESSED_BLOCK_WIDTH,
       * TEXTURE_COMPRESSED_BLOCK_HEIGHT, TEXTURE_COMPRESSED_BLOCK_SIZE,
       * CLEAR_BUFFER, TEXTURE_VIEW, VIEW_COMPATIBILITY_CLASS
       */
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(pname=%s)",
                  _mesa_enum_to_string(pname));
      return false;
   }

   if (bufSize < 0) {
      _mesa_error(ctx, GL_INVALID_VALUE,
                  "glGetInternalformativ(target=%s)",
                  _mesa_enum_to_string(target));
      return false;
   }

   return true;
}

static void
_internalformat_query2(GLenum target, GLenum internalformat, GLenum pname,
                       GLsizei bufSize, GLint *params)
{
   GLint buffer[16];
   GLsizei count = 0;
   GET_CURRENT_CONTEXT(ctx);

   ASSERT_OUTSIDE_BEGIN_END(ctx);

   if (!_legal_parameters(ctx, target, internalformat, pname, bufSize, params))
      return;

   switch(pname){
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
      unreachable("bad param");
   }

   if (bufSize != 0 && params == NULL) {
      /* Emit a warning to aid application debugging, but go ahead and do the
       * memcpy (and probably crash) anyway.
       */
      _mesa_warning(ctx,
                    "glGetInternalformativ(bufSize = %d, but params = NULL)",
                    bufSize);
   }

   /* Copy the data from the temporary buffer to the buffer supplied by the
    * application.  Clamp the size of the copy to the size supplied by the
    * application.
    */
   memcpy(params, buffer, MIN2(count, bufSize) * sizeof(GLint));
}

void GLAPIENTRY
_mesa_GetInternalformativ(GLenum target, GLenum internalformat, GLenum pname,
                          GLsizei bufSize, GLint *params)
{
   GET_CURRENT_CONTEXT(ctx);

   /* FIXME: code-refactor */
   if (ctx->Extensions.ARB_internalformat_query2) {
      _internalformat_query2(target, internalformat, pname, bufSize, params);
      return;
   }

   GLint buffer[16];
   GLsizei count = 0;

   ASSERT_OUTSIDE_BEGIN_END(ctx);

   if (!ctx->Extensions.ARB_internalformat_query) {
      _mesa_error(ctx, GL_INVALID_OPERATION, "glGetInternalformativ");
      return;
   }

   assert(ctx->Driver.QuerySamplesForFormat != NULL);

   /* The ARB_internalformat_query spec says:
    *
    *     "If the <target> parameter to GetInternalformativ is not one of
    *     TEXTURE_2D_MULTISAMPLE, TEXTURE_2D_MULTISAMPLE_ARRAY or RENDERBUFFER
    *     then an INVALID_ENUM error is generated."
    */
   switch (target) {
   case GL_RENDERBUFFER:
      break;

   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      /* These enums are only valid if ARB_texture_multisample is supported */
      if ((_mesa_is_desktop_gl(ctx) &&
           ctx->Extensions.ARB_texture_multisample) ||
          _mesa_is_gles31(ctx))
         break;

   default:
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(target=%s)",
                  _mesa_enum_to_string(target));
      return;
   }

   /* The ARB_internalformat_query spec says:
    *
    *     "If the <internalformat> parameter to GetInternalformativ is not
    *     color-, depth- or stencil-renderable, then an INVALID_ENUM error is
    *     generated."
    *
    * Page 243 of the GLES 3.0.4 spec says this for GetInternalformativ:
    *
    *     "internalformat must be color-renderable, depth-renderable or
    *     stencilrenderable (as defined in section 4.4.4)."
    *
    * Section 4.4.4 on page 212 of the same spec says:
    *
    *     "An internal format is color-renderable if it is one of the
    *     formats from table 3.13 noted as color-renderable or if it
    *     is unsized format RGBA or RGB."
    *
    * Therefore, we must accept GL_RGB and GL_RGBA here.
    */
   if (internalformat != GL_RGB && internalformat != GL_RGBA &&
       _mesa_base_fbo_format(ctx, internalformat) == 0) {
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(internalformat=%s)",
                  _mesa_enum_to_string(internalformat));
      return;
   }

   /* The ARB_internalformat_query spec says:
    *
    *     "If the <bufSize> parameter to GetInternalformativ is negative, then
    *     an INVALID_VALUE error is generated."
    */
   if (bufSize < 0) {
      _mesa_error(ctx, GL_INVALID_VALUE,
                  "glGetInternalformativ(target=%s)",
                  _mesa_enum_to_string(target));
      return;
   }

   switch (pname) {
   case GL_SAMPLES:
      count = ctx->Driver.QuerySamplesForFormat(ctx, target,
            internalformat, buffer);
      break;
   case GL_NUM_SAMPLE_COUNTS: {
      if (_mesa_is_gles3(ctx) && _mesa_is_enum_format_integer(internalformat)) {
         /* From GL ES 3.0 specification, section 6.1.15 page 236: "Since
          * multisampling is not supported for signed and unsigned integer
          * internal formats, the value of NUM_SAMPLE_COUNTS will be zero
          * for such formats.
          */
         buffer[0] = 0;
         count = 1;
      } else {
         size_t num_samples;

         /* The driver can return 0, and we should pass that along to the
          * application.  The ARB decided that ARB_internalformat_query should
          * behave as ARB_internalformat_query2 in this situation.
          *
          * The ARB_internalformat_query2 spec says:
          *
          *     "- NUM_SAMPLE_COUNTS: The number of sample counts that would be
          *        returned by querying SAMPLES is returned in <params>.
          *        * If <internalformat> is not color-renderable,
          *          depth-renderable, or stencil-renderable (as defined in
          *          section 4.4.4), or if <target> does not support multiple
          *          samples (ie other than TEXTURE_2D_MULTISAMPLE,
          *          TEXTURE_2D_MULTISAMPLE_ARRAY, or RENDERBUFFER), 0 is
          *          returned."
          */
         num_samples =  ctx->Driver.QuerySamplesForFormat(ctx, target, internalformat, buffer);

         /* QuerySamplesForFormat writes some stuff to buffer, so we have to
          * separately over-write it with the requested value.
          */
         buffer[0] = (GLint) num_samples;
         count = 1;
      }
      break;
   }
   default:
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(pname=%s)",
                  _mesa_enum_to_string(pname));
      return;
   }

   if (bufSize != 0 && params == NULL) {
      /* Emit a warning to aid application debugging, but go ahead and do the
       * memcpy (and probably crash) anyway.
       */
      _mesa_warning(ctx,
                    "glGetInternalformativ(bufSize = %d, but params = NULL)",
                    bufSize);
   }

   /* Copy the data from the temporary buffer to the buffer supplied by the
    * application.  Clamp the size of the copy to the size supplied by the
    * application.
    */
   memcpy(params, buffer, MIN2(count, bufSize) * sizeof(GLint));

   return;
}
