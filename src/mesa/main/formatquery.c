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
#include "teximage.h"
#include "textureview.h"
#include "texcompress.h"
#include "shaderimage.h"
#include "texobj.h"
#include "genmipmap.h"
#include "texparam.h"

void
_mesa_query_internal_format_default(struct gl_context *ctx, GLenum target,
                                    GLenum internalFormat, GLenum pname,
                                    GLint *params)
{
   (void) ctx;
   (void) target;
   (void) internalFormat;
   (void) params;

   switch (pname) {
   case GL_SAMPLES:
   case GL_NUM_SAMPLE_COUNTS:
      params[0] = 1;
      break;

   case GL_TEXTURE_IMAGE_FORMAT:
   case GL_GET_TEXTURE_IMAGE_FORMAT:
      /* Return a generic preferred image format.
       * That's better than no answer at all.
       */
      params[0] = GL_RGBA;
      break;

   case GL_TEXTURE_IMAGE_TYPE:
   case GL_GET_TEXTURE_IMAGE_TYPE:
      /* Return a generic preferred image format.
       * That's better than no answer at all.
       */
      params[0] = GL_UNSIGNED_BYTE;
      break;

   default:
      /* @TODO: provide a default answer for the rest of the pnames */
      unreachable("Default value for query not yet implemented");
   }
}

static bool
_is_internalformat_supported(struct gl_context *ctx, GLenum target,
                             GLenum internalformat)
{
   switch(target){
   case GL_TEXTURE_1D:
   case GL_TEXTURE_1D_ARRAY:
   case GL_TEXTURE_2D:
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_3D:
   case GL_TEXTURE_CUBE_MAP:
   case GL_TEXTURE_CUBE_MAP_ARRAY:
   case GL_TEXTURE_RECTANGLE:
      /* Based on what is done in the "teximage" method  */
      if (_mesa_base_tex_format(ctx, internalformat) < 0)
         return false;

      /* additional checks for depth textures */
      if (!_mesa_legal_texture_base_format_for_target(ctx, target, internalformat))
         return false;

      /* additional checks for compressed textures */
      if (_mesa_is_compressed_format(ctx, internalformat) &&
          (!_mesa_target_can_be_compressed(ctx, target, internalformat, NULL) ||
           _mesa_format_no_online_compression(ctx, internalformat)))
         return false;

      break;
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      /* Based on what it is done in "texture_image_multisample" method */
      if (!_mesa_is_renderable_texture_format(ctx, internalformat))
         return false;

      break;
   case GL_TEXTURE_BUFFER:
      /* Based on what it is done in "_mesaTexBuffer" method */
      if (_mesa_validate_texbuffer_format(ctx, internalformat) == MESA_FORMAT_NONE)
         return false;

      break;
   case GL_RENDERBUFFER:
      /* Based on what it is done in "renderbuffer_storage" method */
      if (!_mesa_base_fbo_format(ctx, internalformat))
         return false;

      break;
   default:
      unreachable("bad target");
   }

   return true;
}

/* Sets 'buffer' to the appropriate "unsupported" response for each pname.
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
_set_unsupported(GLenum pname, GLint buffer[16])
{
   switch(pname) {
   case GL_SAMPLES:
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
      break;
   default:
      unreachable("invalid 'pname'");
   }
}

/* @FIXME: This method could also check if the target is supported for the
 * pname.
*/
static bool
_is_target_supported(struct gl_context *ctx, GLenum target)
{
   switch(target){
   case GL_TEXTURE_2D:
   case GL_TEXTURE_3D:
   case GL_TEXTURE_1D:
   case GL_TEXTURE_1D_ARRAY:
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_CUBE_MAP:
   case GL_TEXTURE_CUBE_MAP_ARRAY:
   case GL_TEXTURE_RECTANGLE:
      if (!(_mesa_legal_teximage_target(ctx, 1, target) ||
            _mesa_legal_teximage_target(ctx, 2, target) ||
            _mesa_legal_teximage_target(ctx, 3, target)))
          return false;

      break;
   case GL_TEXTURE_BUFFER:
      /* Taken from "_mesa_texture_buffer_range" method */

      /* NOTE: ARB_texture_buffer_object has interactions with
       * the compatibility profile that are not implemented.
       */
      if (!(ctx->API == API_OPENGL_CORE &&
            ctx->Extensions.ARB_texture_buffer_object))
         return false;

      break;
   case GL_RENDERBUFFER:
      /* @FIXME: Also check EXT_framebuffer_object */
      if (!ctx->Extensions.ARB_framebuffer_object)
         return false;

      break;
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      /* Taken from the "texture_image_multisample" method */
      if (!(ctx->Extensions.ARB_texture_multisample && _mesa_is_desktop_gl(ctx))
          && !_mesa_is_gles31(ctx))
         return false;

      break;
   default:
      unreachable("invalid target");
   }

   return true;
}

static bool
_legal_parameters(struct gl_context *ctx, GLenum target, GLenum internalformat,
                  GLenum pname, GLsizei bufSize, GLint *params)

{
   bool query2 = ctx->Extensions.ARB_internalformat_query2;

   switch(target){
   case GL_TEXTURE_1D:
   case GL_TEXTURE_1D_ARRAY:
   case GL_TEXTURE_2D:
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_3D:
   case GL_TEXTURE_CUBE_MAP:
   case GL_TEXTURE_CUBE_MAP_ARRAY:
   case GL_TEXTURE_RECTANGLE:
   case GL_TEXTURE_BUFFER:
      if (!query2) {
         /* The ARB_internalformat_query spec says:
          *
          *     "If the <target> parameter to GetInternalformativ is not one of
          *     TEXTURE_2D_MULTISAMPLE, TEXTURE_2D_MULTISAMPLE_ARRAY or RENDERBUFFER
          *     then an INVALID_ENUM error is generated."
          */
         _mesa_error(ctx, GL_INVALID_ENUM,
                     "glGetInternalformativ(target=%s)",
                     _mesa_enum_to_string(target));

         return false;
      }

      break;
   case GL_RENDERBUFFER:
      break;
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      /* The non-existence of ARB_texture_multisample is treated in
       * ARB_internalformat_query like an error, while ARB_internalformat_query2
       * sets an unsupported response (in _check_dependencies method).
       */
      if (!query2 &&
          !(ctx->Extensions.ARB_texture_multisample && _mesa_is_desktop_gl(ctx)) &&
          !_mesa_is_gles31(ctx)) {
         _mesa_error(ctx, GL_INVALID_ENUM,
                     "glGetInternalformativ(target=%s)",
                     _mesa_enum_to_string(target));

         return false;
      }

      break;
   default:
      /* The ARB_internalformat_query2 spec says:
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
   case GL_NUM_SAMPLE_COUNTS:
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

      /* No break */
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
      /* ARB_internalformat_query only accepts SAMPLES and NUM_SAMPLE_COUNTS
       * pnames.
       */
      if (!query2) {
         _mesa_error(ctx, GL_INVALID_ENUM,
                     "glGetInternalformativ(pname=%s)",
                     _mesa_enum_to_string(pname));

         return false;
      }

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

  /* The ARB_internalformat_query spec says:
    *
    *     "If the <bufSize> parameter to GetInternalformativ is negative, then
    *     an INVALID_VALUE error is generated."
    *
    * Nothing is said in ARB_internalformat_query2 but we assume the same.
    */
   if (bufSize < 0) {
      _mesa_error(ctx, GL_INVALID_VALUE,
                  "glGetInternalformativ(target=%s)",
                  _mesa_enum_to_string(target));
      return false;
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
   if (!query2 && internalformat != GL_RGB && internalformat != GL_RGBA &&
       _mesa_base_fbo_format(ctx, internalformat) == 0) {
      _mesa_error(ctx, GL_INVALID_ENUM,
                  "glGetInternalformativ(internalformat=%s)",
                  _mesa_enum_to_string(internalformat));
      return false;
   }

   return true;
}

void GLAPIENTRY
_mesa_GetInternalformativ(GLenum target, GLenum internalformat, GLenum pname,
                          GLsizei bufSize, GLint *params)
{
   GLint buffer[16];

   GET_CURRENT_CONTEXT(ctx);

   ASSERT_OUTSIDE_BEGIN_END(ctx);

   /* ARB_internalformat_query is also mandatory for ARB_internalformat_query2 */
   if (!ctx->Extensions.ARB_internalformat_query) {
      _mesa_error(ctx, GL_INVALID_OPERATION, "glGetInternalformativ");
      return;
   }

   /* @TODO: assertion checking the existence of the driver hook */

   if (!_legal_parameters(ctx, target, internalformat, pname, bufSize, params))
      return;

   /* The ARB_internalformat_query2 states that when querying for SAMPLES,
    * if no values are returned, then the given buffer is not modified. So,
    * we need to initialize the local buffer with the contents of the user's
    * buffer.
    *
    *     "If <internalformat> is not color-renderable, depth-renderable, or
    *      stencil-renderable (as defined in section 4.4.4), or if <target>
    *      does not support multiple samples (ie other than
    *      TEXTURE_2D_MULTISAMPLE, TEXTURE_2D_MULTISAMPLE_ARRAY, or
    *      RENDERBUFFER), <params> is not modified."
    *
    * This includes when it is not supported so we need to check the need of
    * the copy here.
    */
   if (pname == GL_SAMPLES)
      memcpy(buffer, params, MIN2(bufSize, 16) * sizeof(GLint));

   /* 'Unsupported' is the default response */
   _set_unsupported(pname, buffer);

   if (!_is_target_supported(ctx, target))
      goto end;

   if (!_is_internalformat_supported(ctx, target, internalformat))
      goto end;

   switch(pname){
   case GL_SAMPLES:
      /* fall-through */
   case GL_NUM_SAMPLE_COUNTS:
      /* Taken from Mesa's implementation of ARB_internalformat_query.
       *
       * From GL ES 3.0 specification, section 6.1.15 page 236: "Since
       * multisampling is not supported for signed and unsigned integer
       * internal formats, the value of NUM_SAMPLE_COUNTS will be zero
       * for such formats.
       */
      if (pname == GL_NUM_SAMPLE_COUNTS && _mesa_is_gles3(ctx) &&
          _mesa_is_enum_format_integer(internalformat))
         goto end;

      if (target != GL_RENDERBUFFER &&
          target != GL_TEXTURE_2D_MULTISAMPLE &&
          target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
         goto end;

      /* The renderability of the internalformat is already checked in
       * "_is_internalformat_supported" method.
       */

      /* ask the driver */
      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);
      break;

   case GL_INTERNALFORMAT_SUPPORTED:
      /* If we arrive here, the internalformat is supported */
      buffer[0] = GL_TRUE;

      break;
   case GL_INTERNALFORMAT_PREFERRED:
      /* let the driver answer */
      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);
      break;
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
   case GL_INTERNALFORMAT_STENCIL_TYPE: {
      GLint baseformat;
      mesa_format texformat;

      if (target != GL_RENDERBUFFER) {
         if (!_mesa_legal_get_tex_level_parameter_target(ctx, target, false))
            goto end;

         baseformat = _mesa_base_tex_format(ctx, internalformat);
      } else {
         baseformat = _mesa_base_fbo_format(ctx, internalformat);
      }

      /* Let the driver choose the texture format . */
      /* @FIXME: I am considering that drivers use for renderbuffers the same
       * format-choice logic as for textures. This is true at least for the i965
       * driver, see the 'intel_render_buffer_format' method.
       */
      texformat = ctx->Driver.ChooseTextureFormat(ctx, target, internalformat,
                                                  GL_NONE /*format */, GL_NONE /* type */);

      if (texformat == MESA_FORMAT_NONE || baseformat <= 0)
         goto end;

      /* Based on the 'get_tex_level_parameter_image' and
       * 'get_tex_level_parameter_buffer' implementations for texture
       * targets, and 'get_render_buffer_parameteriv' for the renderbuffer
       * target.
       */

      if (pname == GL_INTERNALFORMAT_SHARED_SIZE) {
         /* Version check taken from 'get_tex_level_parameter_image' */
         if ((ctx->Version >= 30 ||
              ctx->Extensions.EXT_texture_shared_exponent) &&
             target != GL_TEXTURE_BUFFER &&
             target != GL_RENDERBUFFER &&
             texformat == MESA_FORMAT_R9G9B9E5_FLOAT) {
            buffer[0] = 5;
         }
         goto end;
      }

      if (!_mesa_base_format_has_channel(baseformat, pname))
         goto end;

      switch (pname) {
      case GL_INTERNALFORMAT_DEPTH_SIZE:
         /* Extension check taken from 'get_tex_level_parameter_image' */
         if (target != GL_RENDERBUFFER &&
             target != GL_TEXTURE_BUFFER &&
             !ctx->Extensions.ARB_depth_texture)
            goto end;

         /* No break */
      case GL_INTERNALFORMAT_RED_SIZE:
      case GL_INTERNALFORMAT_GREEN_SIZE:
      case GL_INTERNALFORMAT_BLUE_SIZE:
      case GL_INTERNALFORMAT_ALPHA_SIZE:
      case GL_INTERNALFORMAT_STENCIL_SIZE:
         buffer[0] = _mesa_get_format_bits(texformat, pname);
         break;
      case GL_INTERNALFORMAT_DEPTH_TYPE:
         /* Extension check taken from 'get_tex_level_parameter_image' and
            'get_tex_level_parameter_buffer' */
         if (!ctx->Extensions.ARB_texture_float)
            goto end;

         /* No break */
      case GL_INTERNALFORMAT_RED_TYPE:
      case GL_INTERNALFORMAT_GREEN_TYPE:
      case GL_INTERNALFORMAT_BLUE_TYPE:
      case GL_INTERNALFORMAT_ALPHA_TYPE:
      case GL_INTERNALFORMAT_STENCIL_TYPE:
         buffer[0]  = _mesa_get_format_datatype(texformat);
         break;
      default:
         break;
      }
   }

      break;
  case GL_MAX_LAYERS:
     if (!ctx->Extensions.EXT_texture_array)
        goto end;

     /* No break */
   case GL_MAX_WIDTH:
   case GL_MAX_HEIGHT:
   case GL_MAX_DEPTH:
   case GL_MAX_COMBINED_DIMENSIONS:
      /* MAX_COMBINED_DIMENSIONS can be a 64-bit integer. It is packed on
       * buffer[0]-[1] 32-bit integers. As the default is the 32-bit query, we
       * don't do anything here. The frontend wrapper for the 64-bit query
       * will unpack the values.*/

      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);
      break;
   case GL_COLOR_COMPONENTS:
      /* @FIXME: _mesa_is_color_format, considers luminance and
       *  intensity colors, also GL_YCBCR_MESA, as color formats.
       * Some of them do not have and explicit R,G,B or A component,
       * they are calculated implicitly. Should we return FALSE for them?
      */
      if (_mesa_is_color_format(internalformat)) {
         buffer[0] = GL_TRUE;
      } else {
         buffer[0] = GL_FALSE;
      }

      break;
   case GL_DEPTH_COMPONENTS:
      if (_mesa_is_depth_format(internalformat) ||
          _mesa_is_depthstencil_format(internalformat)) {
         buffer[0] = GL_TRUE;
      } else {
         buffer[0] = GL_FALSE;
      }

      break;
   case GL_STENCIL_COMPONENTS:
      if (_mesa_is_stencil_format(internalformat) ||
          _mesa_is_depthstencil_format(internalformat)) {
         buffer[0] = GL_TRUE;
      } else {
         buffer[0] = GL_FALSE;
      }

      break;
   case GL_COLOR_RENDERABLE: {
      GLenum baseFormat =  _mesa_base_fbo_format(ctx, internalformat);
      switch (baseFormat) {
      case GL_ALPHA:
      case GL_LUMINANCE:
      case GL_LUMINANCE_ALPHA:
      case GL_INTENSITY:
      case GL_RGB8:
      case GL_RGB:
      case GL_RGBA:
      case GL_RED:
      case GL_RG:
         buffer[0] = GL_TRUE;
         break;
      default:
         buffer[0] = GL_FALSE;
         break;
      }
   }

      break;
   case GL_DEPTH_RENDERABLE:
   case GL_STENCIL_RENDERABLE: {
      GLenum baseFormat =  _mesa_base_fbo_format(ctx, internalformat);
      if (baseFormat ==  GL_DEPTH_STENCIL ||
          (pname == GL_DEPTH_RENDERABLE && baseFormat == GL_DEPTH_COMPONENT) ||
          (pname == GL_STENCIL_RENDERABLE && baseFormat ==  GL_STENCIL_INDEX)) {
         buffer[0] = GL_TRUE;
      } else {
         buffer[0] = GL_FALSE;
      }
   }

      break;
   case GL_FRAMEBUFFER_RENDERABLE:
      /* @TODO: Check dependencies for this pname */
      if (target != GL_RENDERBUFFER)
         goto end;

      /* If we arrived here, we already know that the internalformat is
       * framebuffer renderable (see_is_internalformat_supported)
       * method.
       */
      /* @FIXME: Is full support the correct answer? */
      buffer[0] = GL_FULL_SUPPORT;

      break;
   case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
      /* @TODO: Check dependencies for this pname */
      /* @TODO */
      break;
   case GL_FRAMEBUFFER_BLEND:
      /* @TODO: Check dependencies for this pname */
      /* @TODO */
      break;
   case GL_READ_PIXELS:
    /* @TODO */
      break;
   case GL_READ_PIXELS_FORMAT:
      /* @TODO: ask the driver */
      /* The specification says:
       * "The <format> to pass to ReadPixels to obtain the best performance and
       * image quality when reading from framebuffers with<internalformat> is
       * returned in <params>."
       * so it should probably be answered by the driver.
       */
      break;
   case GL_READ_PIXELS_TYPE:
      /* @TODO: ask the driver */
      /* The specification says:
       * "The <type> to pass to ReadPixels to obtain the best performance and
       * image quality when reading from framebuffers with<internalformat> is
       * returned in <params>."
       * so it should probably be answered by the driver.
       */

      break;
   case GL_TEXTURE_IMAGE_FORMAT:
   case GL_TEXTURE_IMAGE_TYPE:
   case GL_GET_TEXTURE_IMAGE_FORMAT:
   case GL_GET_TEXTURE_IMAGE_TYPE:
      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);
      break;
   case GL_MIPMAP:
      /* The following texture targets can not have mipmaps:
       * TEXTURE_RECTANGLE, TEXTURE_BUFFER, TEXTURE_2D_MULTISAMPLE and
       * TEXTURE_2D_MULTISAMPLE_ARRAY. @FIXME: For the moment, I am using
       * _mesa_is_valid_generate_texture_mipmap_target method to check this.
       */
      if (!_mesa_is_valid_generate_texture_mipmap_target(ctx, target))
         goto end;

      /*@FIXME: Are there restrictions about the internalformat like in the
       * mipmap generation?, i.e. should I call to
       * _mesa_is_valid_generate_texture_mipmap_internalformat too?
       */
      buffer[0] = GL_FULL_SUPPORT;
      break;

   case GL_AUTO_GENERATE_MIPMAP:
      /* @TODO: Check dependencies for this pname */
      /* Automatic mipmap generation refers to generate mipmaps
       * through glTexParameter (pname = GL_GENERATE_MIPMAP).
       * This is deprecated in some OpenGL versions,
       * see _check_dependencies method.
       */
      /* @FIXME: Is it correct to use the same implementation than for
       * MANUAL_GENERATE_MIPMAP?
       */

      /* No break */
   case GL_MANUAL_GENERATE_MIPMAP:
      /* @TODO: Check dependencies for this pname */
      /* Manual mipmap generation refers to generate mipmaps using the
       * 'glGenerateMipmap' method.
       */
      if (!(_mesa_is_valid_generate_texture_mipmap_target(ctx, target) &&
            _mesa_is_valid_generate_texture_mipmap_internalformat(ctx,
                                                                  internalformat))) {
         goto end;
      }

      /* @FIXME: Is full support the correct answer?. Review
       * meta_generate_mipmap.c, there is a fallback_required method.
       */
       buffer[0] = GL_FULL_SUPPORT;

      break;
   case GL_COLOR_ENCODING:
      if (!_mesa_is_color_format(internalformat))
         goto end;

      if (_mesa_get_linear_internalformat(internalformat) != internalformat)
         buffer[0] = GL_SRGB;
      else
         buffer[0] = GL_LINEAR;

      break;
   case GL_SRGB_READ:
      if (!ctx->Extensions.EXT_texture_sRGB)
         goto end;

      /* @TODO */
      break;
   case GL_SRGB_WRITE:
      if (!ctx->Extensions.EXT_framebuffer_sRGB)
         goto end;

      /* @TODO */
      break;
   case GL_SRGB_DECODE_ARB:
      /* ARB_texture_sRGB_decode is supported, we won't reach this point otherwise
      * (the check is done in _legal_parameters)
      */
      buffer[0] = GL_FULL_SUPPORT;
      /* @FIXME: Should we ask the driver to know if the support is: FULL, CAVEAT, etc? */
      break;
   case GL_FILTER:
      /* @TODO */
      break;
   case GL_VERTEX_TEXTURE:
      /* @TODO: ask the driver */
      break;
   case GL_TESS_CONTROL_TEXTURE:
   case GL_TESS_EVALUATION_TEXTURE:
     /* Mesa doesn't support tesselation stages, so we ask the backend
      * in case it supports it.
      */
      /* @FIXME: Should we add instead a call to:
       * if (!_mesa_has_tessellation(ctx))
       *      goto end;
       */
      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);

      break;
   case GL_GEOMETRY_TEXTURE:
      if (!_mesa_has_geometry_shaders(ctx))
         goto end;

      /* @TODO: ask the driver */
      break;
   case GL_FRAGMENT_TEXTURE:
      /* @TODO: ask the driver*/
      break;
   case GL_COMPUTE_TEXTURE:
      if (!_mesa_has_compute_shaders(ctx))
         goto end;

      /* @TODO: ask the driver*/
      break;
   case GL_TEXTURE_SHADOW:
      /* @TODO: Add dependencies check */
      /* @TODO */
      break;
   case GL_TEXTURE_GATHER:
      /* @TODO: Add dependencies check */
      /* @TODO */
      break;
   case GL_TEXTURE_GATHER_SHADOW:
      /* @TODO: Add dependencies check */
      /* @TODO */
      break;
   case GL_SHADER_IMAGE_LOAD:
   case GL_SHADER_IMAGE_STORE:
      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      /* The ARB_internalformat_query2 spec says:
       * "In this case the <internalformat> is the value of the <format>
       * parameter that is passed to BindImageTexture."
       *
       * We can call to _mesa_is_shader_image_format_supported
       * using "internalformat" as parameter.
       */
      if (target == GL_RENDERBUFFER ||
          !_mesa_is_image_format_supported(ctx, internalformat)) {
         goto end;
      }

      /* If we arrive here, ARB_shader_image_load_store is supported */
      /* @FIXME: Is FULL_SUPPORT the correct answer? */
      buffer[0] = GL_FULL_SUPPORT;

      break;
   case GL_SHADER_IMAGE_ATOMIC:
      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      /* @TODO: I have doubts about how to determine if the "resource" is
       * supported for being used with atomic memory operations. Mesa's
       * implementation of "glMemoryBarrier" does not make any checks,
       * only checks if the Driver hook exists.
       * Should be this answered by the Driver?.
       */
      break;
   case GL_IMAGE_TEXEL_SIZE: {
      mesa_format image_format;

      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      if (target == GL_RENDERBUFFER)
         goto end;

      /* @FIXME: I am considering that the resource is used for glBindTexture.
       * The ARB_internalformat_query2 extension spec says:
       * "The size of a texel when the resource when used as
       * an image texture is returned in <params>. This is the value from the
       *  /Size/ column in Table 3.22. If the resource is not supported for image
       * textures, or if image textures are not supported, zero is returned."
       * It is not clear to me what it means by "resource when used as an image
       * texture" but as the pname depends on the existance of the
       * ARB_image_load_store extension, I am assuming that means "when the
       * resource is bound to an image_unit, i.e., glBindTexture.
       */
      image_format = _mesa_get_shader_image_format(internalformat);
      if (image_format == MESA_FORMAT_NONE)
         goto end;

      /* We have to return bits */
      buffer[0] = (_mesa_get_format_bytes(image_format) * 8);
   }

      break;
   case GL_IMAGE_COMPATIBILITY_CLASS:
      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      if (target == GL_RENDERBUFFER)
         goto end;

      /* @FIXME: Verify if it is correct to call this function passing internalformat as param.
       * I am using internalformat as it were the format passed to glBindTexture, i.e.
       * the format in which formatted stores (writes from the shader) will be performed.
       * This format should match the format of the image uniform in the shaders that will
       * access the texture. However, it need not match the format of the actual
       * texture. For textures allocated by calling one of the glTexImage() or
       * glTexStorage() functions, any format that matches in size may be specified
       * for format.
       */
      /* In the extension spec:
       * The compatibility class of the resource when used as an image texture is returned in <params>.
       * This corresponds to the value from the /Class/ column in Table 3.22.
       * And Table 3.22 is: Texel sizes, compatibility classes, and pixel format/type combinations for
       * each image format.
       */
      /*
       * In Mesa's code (_mesa_is_image_unit_valid it uses both the internalformat of the textureObject
       * and the format passed to glTexImage to check if their compatibility class is the same, so I conclude
       * that is fine to use the internal format here.
       */
      buffer[0] = _mesa_get_image_format_class(internalformat);

      break;
   case GL_IMAGE_PIXEL_FORMAT:
      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      if (target == GL_RENDERBUFFER ||
          !_mesa_is_image_format_supported(ctx, internalformat)) {
         goto end;
      }

      GLint base_format = _mesa_base_tex_format(ctx, internalformat);
      if (base_format == -1)
         goto end;

      if (_mesa_is_enum_format_integer(internalformat))
         buffer[0] = _mesa_base_format_to_integer_format(base_format);
      else
         buffer[0] = base_format;

      break;
   case GL_IMAGE_PIXEL_TYPE: {
      mesa_format image_format;
      GLenum datatype;
      GLuint comps;

      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      if (target == GL_RENDERBUFFER)
         goto end;

      image_format = _mesa_get_shader_image_format(internalformat);
      if (image_format == MESA_FORMAT_NONE)
         goto end;

      _mesa_uncompressed_format_to_type_and_comps(image_format, &datatype,
                                                  &comps);
      if (!datatype)
         goto end;

      buffer[0] = datatype;
   }

      break;
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE: {
      /* Taken from "_mesa_BindImageTextures" method */
      if (!ctx->Extensions.ARB_shader_image_load_store)
         goto end;

      if (!_mesa_legal_get_tex_level_parameter_target(ctx, target, false))
         goto end;

      /* From spec: "Equivalent to calling GetTexParameter with <value> set
       * to IMAGE_FORMAT_COMPATIBILITY_TYPE."
       *
       * GetTexParameter just returns
       * tex_obj->ImageFormatCompatibilityType. We create a fake tex_obj
       * just with the purpose of getting the value.
       */
      struct gl_texture_object *tex_obj = _mesa_new_texture_object(ctx, 0, target);
      buffer[0] = tex_obj->ImageFormatCompatibilityType;
      _mesa_delete_texture_object(ctx, tex_obj);
   }

      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
      ctx->Driver.QueryInternalFormat(ctx, target, internalformat, pname,
                                      buffer);
      break;
   case GL_TEXTURE_COMPRESSED:
      buffer[0] = _mesa_is_compressed_format(ctx, internalformat);
      break;
   case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
   case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
   case GL_TEXTURE_COMPRESSED_BLOCK_SIZE: {
      mesa_format mesaformat = _mesa_glenum_to_compressed_format(internalformat);
      if (mesaformat == MESA_FORMAT_NONE)
         goto end;

      GLint block_size = _mesa_get_format_bytes(mesaformat);
      assert(block_size > 0);

      if (pname == GL_TEXTURE_COMPRESSED_BLOCK_SIZE) {
         buffer[0] = block_size;
      } else {
         GLuint bwidth, bheight;

         /* Returns the width and height in pixels. We have to return bytes */
         _mesa_get_format_block_size(mesaformat, &bwidth, &bheight);
         assert(bwidth > 0 && bheight > 0);

         if (pname == GL_TEXTURE_COMPRESSED_BLOCK_WIDTH)
            buffer[0] = block_size / bheight;
         else
            buffer[0] = block_size / bwidth;
      }
   }

      break;
   case GL_CLEAR_BUFFER:
      /* @TODO: Add dependencies check, if any */
      if (target != GL_TEXTURE_BUFFER)
         goto end;

      /* All drivers in Mesa support ARB_clear_buffer_object */
      /* @FIXME: is full support the correct answer ? */
      buffer[0] = GL_FULL_SUPPORT;

      break;
   case GL_TEXTURE_VIEW:
   case GL_VIEW_COMPATIBILITY_CLASS:
      if (!ctx->Extensions.ARB_texture_view)
         goto end;

      if (target == GL_TEXTURE_BUFFER || target == GL_RENDERBUFFER)
         goto end;

      if (pname == GL_TEXTURE_VIEW) {
         /* @FIXME: is full support the correct answer ? */
         buffer[0] = GL_FULL_SUPPORT;
      } else {
         GLenum view_class = _mesa_texture_view_lookup_view_class(ctx,
                                                                  internalformat);
         if (view_class == GL_FALSE)
            goto end;

         buffer[0] = view_class;
      }

      break;
   default:
      unreachable("bad param");
   }

 end:
   if (bufSize != 0 && params == NULL) {
      /* Emit a warning to aid application debugging, but go ahead and do the
       * memcpy (and probably crash) anyway.
       */
      _mesa_warning(ctx,
                    "glGetInternalformativ(bufSize = %d, but params = NULL)",
                    bufSize);
   }

   /* Copy the data from the temporary buffer to the buffer supplied by the
    * application.  Clamp the size of the copy to minimum between the size
    * supplied by the application and the maximum of 16 elements of the
    * temporary buffer.
    */
   memcpy(params, buffer, MIN2(bufSize, 16) * sizeof(GLint));
}

/* MAX_COMBINED_DIMENSIONS is the only pname that the spec specifies that can
 * be a 64-bit query. Due that we maintain the 32-bit query as default, and
 * implement the 64-bit query as a wrap over the 32-bit query. To handle
 * MAX_COMBINED_DIMENSIONS, the driver packs the 64-bit integer on two 32-bit
 * integers at params, and the wrapper here unpacks it */
void GLAPIENTRY
_mesa_GetInternalformati64v(GLenum target, GLenum internalformat,
                            GLenum pname, GLsizei bufSize, GLint64 *params)
{
   GLint params32[16];
   GLsizei i = 0;
   GLsizei realSize = MIN2(bufSize, 16);
   GLsizei callSize;

   GET_CURRENT_CONTEXT(ctx);

   ASSERT_OUTSIDE_BEGIN_END(ctx);

   if (!ctx->Extensions.ARB_internalformat_query2) {
      _mesa_error(ctx, GL_INVALID_OPERATION, "glGetInternalformati64v");
      return;
   }

   /* For SAMPLES there are cases where params needs to remain unmodified. As
    * no pname can return a negative value, we fill params32 with negative
    * values as reference values, that can be used to know what copy-back to
    * params */
   memset(params32, -1, 16);

   /* For GL_MAX_COMBINED_DIMENSIONS we need to get back 2 32-bit integers,
    * and at the same time we only need 2. So for that pname, we call the
    * 32-bit query with bufSize 2, except on the case of bufSize 0, that is
    * basically like asking to not get the value, but that is a caller
    * problem. */
   if (pname == GL_MAX_COMBINED_DIMENSIONS && bufSize > 0)
      callSize = 2;
   else
      callSize = bufSize;

   _mesa_GetInternalformativ(target, internalformat, pname, callSize, params32);

   if (pname == GL_MAX_COMBINED_DIMENSIONS) {
      memcpy(params, params32, sizeof(GLint64));
   } else {
      for (i = 0; i < realSize; i++) {
         /* We only copy back the values that changed */
         if (params32[i] < 0)
            break;
         params[i] = (GLint64) params32[i];
      }
   }
}
