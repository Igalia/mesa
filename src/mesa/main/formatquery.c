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

/* An element of the Table 3.22, in OpenGL 4.2 Core specification */
struct imageformat {
   GLenum format;
   int texel_size;
   GLenum pixel_format;
   GLenum pixel_type;
   GLenum compatibility_class;
};

/* Table 3.22, in OpenGL 4.2 Core specification */
static const struct imageformat imageformat_table[] = {
   {GL_RGBA32F, 128, GL_RGBA, GL_FLOAT, GL_IMAGE_CLASS_4_X_32},
   {GL_RGBA16F, 64, GL_RGBA, GL_HALF_FLOAT, GL_IMAGE_CLASS_4_X_16},
   {GL_RG32F, 64, GL_RG, GL_FLOAT, GL_IMAGE_CLASS_2_X_32},
   {GL_RG16F, 32, GL_RG, GL_HALF_FLOAT, GL_IMAGE_CLASS_4_X_16},
   {GL_R11F_G11F_B10F, 32, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_IMAGE_CLASS_11_11_10},
   {GL_R32F, 32, GL_RED, GL_FLOAT, GL_IMAGE_CLASS_1_X_32},
   {GL_R16F, 16, GL_RED, GL_HALF_FLOAT, GL_IMAGE_CLASS_1_X_16},
   {GL_RGBA32UI, 128, GL_RGBA_INTEGER, GL_UNSIGNED_INT, GL_IMAGE_CLASS_4_X_32},
   {GL_RGBA16UI, 64, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_4_X_16},
   {GL_RGB10_A2UI, 32, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV, GL_IMAGE_CLASS_10_10_10_2},
   {GL_RGBA8UI, 32, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_4_X_8},
   {GL_RG32UI, 64, GL_RG_INTEGER, GL_UNSIGNED_INT, GL_IMAGE_CLASS_2_X_32},
   {GL_RG16UI, 32, GL_RG_INTEGER, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_2_X_16},
   {GL_RG8UI, 16, GL_RG_INTEGER, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_2_X_8},
   {GL_R32UI, 32, GL_RED_INTEGER, GL_UNSIGNED_INT, GL_IMAGE_CLASS_1_X_32},
   {GL_R16UI, 16, GL_RED_INTEGER, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_1_X_16},
   {GL_R8UI, 8, GL_RED_INTEGER, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_1_X_8},
   {GL_RGBA32I, 128, GL_RGBA_INTEGER, GL_INT, GL_IMAGE_CLASS_4_X_32},
   {GL_RGBA16I, 64, GL_RGBA_INTEGER, GL_SHORT, GL_IMAGE_CLASS_4_X_16},
   {GL_RGBA8I, 32, GL_RGBA_INTEGER, GL_BYTE, GL_IMAGE_CLASS_4_X_8},
   {GL_RG32I, 64, GL_RG_INTEGER, GL_INT, GL_IMAGE_CLASS_2_X_32},
   {GL_RG16I, 32, GL_RG_INTEGER, GL_SHORT, GL_IMAGE_CLASS_2_X_16},
   {GL_RG8I, 16, GL_RG_INTEGER, GL_BYTE, GL_IMAGE_CLASS_2_X_8},
   {GL_R32I, 32, GL_RED_INTEGER, GL_INT, GL_IMAGE_CLASS_1_X_32},
   {GL_R16I, 16, GL_RED_INTEGER, GL_SHORT, GL_IMAGE_CLASS_1_X_16},
   {GL_R8I, 8, GL_RED_INTEGER, GL_BYTE, GL_IMAGE_CLASS_1_X_8},
   {GL_RGBA16, 64, GL_RGBA, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_4_X_16},
   {GL_RGB10_A2, 32, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, GL_IMAGE_CLASS_10_10_10_2},
   {GL_RGBA8, 32, GL_RGBA, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_4_X_8},
   {GL_RG16, 32, GL_RG, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_2_X_16},
   {GL_RG8, 16, GL_RG, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_2_X_8},
   {GL_R16, 16, GL_RED, GL_UNSIGNED_SHORT, GL_IMAGE_CLASS_1_X_16},
   {GL_R8, 8, GL_RED, GL_UNSIGNED_BYTE, GL_IMAGE_CLASS_1_X_8},
   {GL_RGBA16_SNORM, 64, GL_RGBA, GL_SHORT, GL_IMAGE_CLASS_4_X_16},
   {GL_RGBA8_SNORM, 32, GL_RGBA, GL_BYTE, GL_IMAGE_CLASS_4_X_8},
   {GL_RG16_SNORM, 32, GL_RG, GL_SHORT, GL_IMAGE_CLASS_2_X_16},
   {GL_RG8_SNORM, 16, GL_RG, GL_BYTE, GL_IMAGE_CLASS_2_X_8},
   {GL_R16_SNORM, 16, GL_RED, GL_SHORT, GL_IMAGE_CLASS_1_X_16},
   {GL_R8_SNORM, 8, GL_RED, GL_BYTE, GL_IMAGE_CLASS_1_X_8},
};

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

static bool
_is_texture_target(GLenum target)
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
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
   case GL_TEXTURE_BUFFER:
      return true;
   case GL_RENDERBUFFER:
      return false;
   default:
      unreachable("bad target");
   }
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

/* Implements the Dependencies section of the ARB_internalformat_query2 spec.
 * Returns 'true' if everything went fine, 'false' otherwise.
 *
 * @FIXME: this method and _legal_parameters could be possibly be joint.
 */
static bool
_check_dependencies(struct gl_context *ctx, GLenum target,
                    GLenum pname, GLint buffer[16], GLsizei *count)
{
   switch(target){
   case GL_TEXTURE_2D:
   case GL_TEXTURE_3D:
      break;
   case GL_TEXTURE_1D:
      /* Taken from "legal_teximage_target" method */
      if (!_mesa_is_desktop_gl(ctx))
         return false;

      break;
   case GL_TEXTURE_1D_ARRAY:
      /* Taken from "legal_teximage_target" method */
      if (!(_mesa_is_desktop_gl(ctx) && ctx->Extensions.EXT_texture_array))
         return false;

      break;
   case GL_TEXTURE_2D_ARRAY:
      /* Taken from "legal_teximage_target" method"*/
      if (!((_mesa_is_desktop_gl(ctx) && ctx->Extensions.EXT_texture_array) || _mesa_is_gles3(ctx)))
          return false;

      break;
   case GL_TEXTURE_CUBE_MAP:
      /* Addittional check (not mentioned in the ARB_internalformat_query2 spec)
       * Taken from "legal_teximage_target" method
       */
      if (!(_mesa_is_desktop_gl(ctx) && ctx->Extensions.ARB_texture_cube_map))
         return false;

      break;
   case GL_TEXTURE_CUBE_MAP_ARRAY:
      /* Taken from "legal_teximage_target" method */
      if (!(_mesa_is_desktop_gl(ctx) && ctx->Extensions.ARB_texture_cube_map_array))
         return false;

      break;
   case GL_TEXTURE_RECTANGLE:
      /* Taken from "legal_teximage_target" method */
      if (!(_mesa_is_desktop_gl(ctx) && ctx->Extensions.NV_texture_rectangle))
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
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 20) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 20))
          return false;

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
      /* @FIXME: Also check EXT_framebuffer_object */
      if  ((_mesa_is_desktop_gl(ctx) && ctx->Version <= 20) ||
           !ctx->Extensions.ARB_framebuffer_object)
      return false;

      break;
   case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
      /* @FIXME: Also check EXT_framebuffer_object */
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 20) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 20) ||
          !ctx->Extensions.ARB_framebuffer_object ||
          !ctx->Extensions.EXT_texture_array)
         return false;

      break;
   case GL_FRAMEBUFFER_BLEND:
      /* @FIXME: Also check EXT_framebuffer_object */
      if  ((_mesa_is_desktop_gl(ctx) && ctx->Version <= 20)  ||
          !ctx->Extensions.ARB_framebuffer_object)
         return false;

      break;
   case GL_READ_PIXELS:
      break;
   case GL_READ_PIXELS_FORMAT:
      break;
   case GL_READ_PIXELS_TYPE:
      break;
   case GL_TEXTURE_IMAGE_FORMAT:
      if (ctx->API == API_OPENGLES2 && ctx->Version <= 30)
         return false;

      break;
   case GL_TEXTURE_IMAGE_TYPE:
      if (ctx->API == API_OPENGLES2 && ctx->Version <= 30)
         return false;

      break;
   case GL_GET_TEXTURE_IMAGE_FORMAT:
      break;
   case GL_GET_TEXTURE_IMAGE_TYPE:
      break;
   case GL_MIPMAP:
      break;
   case GL_MANUAL_GENERATE_MIPMAP:
      /* @FIXME: Also check EXT_framebuffer_object */
      if  ((_mesa_is_desktop_gl(ctx) && ctx->Version <= 20) ||
         !ctx->Extensions.ARB_framebuffer_object)
         return false;

      break;
   case GL_AUTO_GENERATE_MIPMAP:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30)  ||
          (ctx->API = API_OPENGL_CORE && ctx->Version >= 32))
         return false;

      break;
   case GL_COLOR_ENCODING:
      break;
   case GL_SRGB_READ:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 20) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 20) ||
          !ctx->Extensions.EXT_texture_sRGB)
         return false;

      break;
   case GL_SRGB_WRITE:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 20) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 20) ||
          !ctx->Extensions.EXT_framebuffer_sRGB)
         return false;

      break;
   case GL_SRGB_DECODE_ARB:
      break;
   case GL_FILTER:
      break;
   case GL_VERTEX_TEXTURE:
      break;
   case GL_TESS_CONTROL_TEXTURE:
   case GL_TESS_EVALUATION_TEXTURE:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 30) ||
          !ctx->Extensions.ARB_tessellation_shader)
         return false;

      break;
   case GL_GEOMETRY_TEXTURE:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 30) ||
          !ctx->Extensions.ARB_geometry_shader4)
         return false;

      break;
   case GL_FRAGMENT_TEXTURE:
      break;
   case GL_COMPUTE_TEXTURE:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 40) ||
          !ctx->Extensions.ARB_compute_shader)
         return false;

      break;
   case GL_TEXTURE_SHADOW:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 20) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 23))
         return false;

      break;
   case GL_TEXTURE_GATHER:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 30) ||
          !ctx->Extensions.ARB_texture_gather)
         return false;

      break;
   case GL_TEXTURE_GATHER_SHADOW:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 30))
         return false;

      break;
   case GL_SHADER_IMAGE_LOAD:
   case GL_SHADER_IMAGE_STORE:
   case GL_SHADER_IMAGE_ATOMIC:
   case GL_IMAGE_TEXEL_SIZE:
   case GL_IMAGE_COMPATIBILITY_CLASS:
   case GL_IMAGE_PIXEL_FORMAT:
   case GL_IMAGE_PIXEL_TYPE:
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
    if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 40) ||
          !ctx->Extensions.ARB_shader_image_load_store)
         return false;

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
      /* @FIXME: All the drivers in mesa implement ARB_clear_buffer_object */
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 40))
         return false;

      break;
   case GL_TEXTURE_VIEW:
   case GL_VIEW_COMPATIBILITY_CLASS:
      if ((ctx->API == API_OPENGLES2 && ctx->Version <= 30) ||
          (_mesa_is_desktop_gl(ctx) && ctx->Version <= 40) ||
          !ctx->Extensions.ARB_texture_view)
         return false;

      break;
   default:
      unreachable("invalid pname");
   }

   return true;
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
   bool unsupported = false;
   GET_CURRENT_CONTEXT(ctx);

   ASSERT_OUTSIDE_BEGIN_END(ctx);

   if (!_legal_parameters(ctx, target, internalformat, pname, bufSize, params))
      return;

#if 0
   unsupported = !_check_dependencies(ctx, target, pname, buffer, &count);
   if (unsupported)
      goto end;
#endif

   switch(pname){
   case GL_SAMPLES:
   case GL_NUM_SAMPLE_COUNTS:
      switch (target) {
      case GL_RENDERBUFFER:
         if (!_mesa_base_fbo_format(ctx, internalformat))
            unsupported = true;
         break;
      case GL_TEXTURE_2D_MULTISAMPLE:
      case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
         if (!_mesa_is_renderable_texture_format(ctx, internalformat))
            unsupported = true;
         break;
      default:
         unsupported = true;
         break;
      }

      if (unsupported)
         goto end;

      /* @ TODO: call the driver */

      break;
   case GL_INTERNALFORMAT_SUPPORTED:
      if (_is_texture_target(target)) {
         unsupported = _mesa_base_tex_format(ctx, internalformat) < 0;
      } else {
         unsupported = !_mesa_base_fbo_format(ctx, internalformat);
      }

      if (!unsupported) {
         buffer[0] = GL_TRUE;
         count = 1;
      }

      break;
   case GL_INTERNALFORMAT_PREFERRED:
     /* @TODO: ask the driver */
      break;
   case GL_INTERNALFORMAT_RED_SIZE:
   case GL_INTERNALFORMAT_GREEN_SIZE:
   case GL_INTERNALFORMAT_BLUE_SIZE:
   case GL_INTERNALFORMAT_ALPHA_SIZE:
   case GL_INTERNALFORMAT_DEPTH_SIZE:
   case GL_INTERNALFORMAT_STENCIL_SIZE:
   case GL_INTERNALFORMAT_SHARED_SIZE:
      if (_is_texture_target(target)) {
         /* From the ARB_internalformat_query2 spec:
          *
          * For textures this query will return the same information as querying
          * GetTexLevelParameter{if}v for TEXTURE_*_SIZE would return.
          */
         if (target ==  GL_TEXTURE_BUFFER) {
            /* @TODO: get_tex_level_parameter_buffer */
            /* @FIXME: same than in the 'else' branch */
         } else {
            /* @TODO: Behave as get_tex_level_parameter_image  */
            /* @FIXME: We do not have the texFormat */
            /* baseformat = _mesa_base_tex_format(ctx, internalformat); */
            /* if (_mesa_base_format_has_channel(baseformat, pname)) */
            /*    *params = _mesa_get_format_bits(texFormat, pname); */
            /* else */
            /*    *params = 0; */
         }
      } else {
         /* @TODO */

         /* For uncompressed internal formats, queries of these values return the
         * actual resolutions that would be used for storing image array components
         * for the resource.
         */
         /*
         * For compressed internal formats, the resolutions returned specify the
         * component resolution of an uncompressed internal format that produces
         * an image of roughly the same quality as the compressed algorithm.
         */
      }

      break;

   case GL_INTERNALFORMAT_RED_TYPE:
   case GL_INTERNALFORMAT_GREEN_TYPE:
   case GL_INTERNALFORMAT_BLUE_TYPE:
   case GL_INTERNALFORMAT_ALPHA_TYPE:
   case GL_INTERNALFORMAT_DEPTH_TYPE:
   case GL_INTERNALFORMAT_STENCIL_TYPE:
      /* @TODO */
      /* @FIXME: same problem than for the *SIZE queries */
      break;
   case GL_MAX_WIDTH:
      /* @TODO: ask the driver */
      break;
   case GL_MAX_HEIGHT:
      /* @TODO: ask the driver */
      break;
   case GL_MAX_DEPTH:
      /* @TODO: ask the driver */
      break;
   case GL_MAX_LAYERS:
      /* @TODO: ask the driver */
      break;
   case GL_MAX_COMBINED_DIMENSIONS:
      /* @TODO: ask the driver */
      break;
   case GL_COLOR_COMPONENTS:
      /* @FIXME: _mesa_is_color_format, considers luminance and
       *  intensity colors, also GL_YCBCR_MESA, as color formats.
       * Some of them do not have and explicit R,G,B or A component,
       * they are calculated implicitly. Should we return FALSE for them?
      */
      if (_mesa_is_color_format(internalformat)) {
         buffer[0] = GL_TRUE;
         count = 1;
      } else {
         buffer[0] = GL_FALSE;
         count = 1;
      }

      break;
   case GL_DEPTH_COMPONENTS:
      if (_mesa_is_depth_format(internalformat) ||
          _mesa_is_depthstencil_format(internalformat)) {
         buffer[0] = GL_TRUE;
         count = 1;
      } else {
         buffer[0] = GL_FALSE;
         count = 1;
      }

      break;
   case GL_STENCIL_COMPONENTS:
      if (_mesa_is_stencil_format(internalformat) ||
          _mesa_is_depthstencil_format(internalformat)) {
         buffer[0] = GL_TRUE;
         count = 1;
      } else {
         buffer[0] = GL_FALSE;
         count = 1;
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
      count = 1;
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
      count = 1;
   }

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
      if (_mesa_is_color_format(internalformat)) {
         if (_mesa_get_linear_internalformat(internalformat) != internalformat) {
            buffer[0] = GL_SRGB;
         } else {
            buffer[0] = GL_LINEAR;
         }
         count = 1;
      } else {
         unsupported = true;
      }

      break;
   case GL_SRGB_READ:
      /* @TODO */
      break;
   case GL_SRGB_WRITE:
      /* @TODO */
      break;
   case GL_SRGB_DECODE_ARB:
      /* ARB_texture_sRGB_decode is supported, we won't reach this point otherwise
      * (the check is done in _legal_parameters)
      */
      buffer[0] = GL_FULL_SUPPORT;
      count = 1;
      /* @FIXME: Should we ask the driver to know if the support is: FULL, CAVEAT, etc? */
      break;
   case GL_FILTER:
      /* @TODO */
      break;
   case GL_VERTEX_TEXTURE:
      /* @TODO: ask the driver */
      break;
   case GL_TESS_CONTROL_TEXTURE:
      /* @TODO: ask the driver */
      break;
   case GL_TESS_EVALUATION_TEXTURE:
      /* @TODO: ask the driver */
      break;
   case GL_GEOMETRY_TEXTURE:
      /* @TODO: ask the driver */
      break;
   case GL_FRAGMENT_TEXTURE:
      /* @TODO: ask the driver*/
      break;
   case GL_COMPUTE_TEXTURE:
      /* @TODO: ask the driver*/
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
         for (int i = 0; i < ARRAY_SIZE(imageformat_table); ++i) {
            if (imageformat_table[i].format == internalformat) {
               buffer[0] = imageformat_table[i].texel_size;
               count = 1;
               break;
            }
         }

         if (count < 1) {
            unsupported = true;
         }

      break;
   case GL_IMAGE_COMPATIBILITY_CLASS:
      /* @FIXME: I am adding the implementation taking into account values in Table 3.22
       * of the OpenGL 4.2, but I do not think we support them, as I can not see any references
       * in the code about IMAGE_CLASS*.
      */
      for (int i = 0; i < ARRAY_SIZE(imageformat_table); ++i) {
         if (imageformat_table[i].format == internalformat) {
            buffer[0] = imageformat_table[i].compatibility_class;
            count = 1;
            break;
         }
      }

      if (count < 1) {
         unsupported = true;
      }

      break;
   case GL_IMAGE_PIXEL_FORMAT:
        for (int i = 0; i < ARRAY_SIZE(imageformat_table); ++i) {
            if (imageformat_table[i].format == internalformat) {
               buffer[0] = imageformat_table[i].pixel_format;
               count = 1;
               break;
            }
         }

         if (count < 1) {
            unsupported = true;
         }

      break;
   case GL_IMAGE_PIXEL_TYPE:
      for (int i = 0; i < ARRAY_SIZE(imageformat_table); ++i) {
         if (imageformat_table[i].format == internalformat) {
            buffer[0] = imageformat_table[i].pixel_type;
            count = 1;
            break;
         }
      }

      if (count < 1) {
         unsupported = true;
      }

      break;
   case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
      /* @TODO */
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
      /* @TODO: ask the driver */
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
      /* @TODO: ask the driver */
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
      /* @TODO: ask the driver */
      break;
   case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
      /* @TODO: ask the driver */
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
      /* All drivers in Mesa support  ARB_clear_buffer_object,
       * no check is needed.
       */
      if (target != GL_TEXTURE_BUFFER ||
          _mesa_validate_texbuffer_format(ctx, internalformat) == MESA_FORMAT_NONE) {
         unsupported = true;
         goto end;
      }

      /* @FIXME: is full support the correct answer ? */
      buffer[0] = GL_FULL_SUPPORT;
      count = 1;

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

 end:
   if (bufSize != 0 && params == NULL) {
      /* Emit a warning to aid application debugging, but go ahead and do the
       * memcpy (and probably crash) anyway.
       */
      _mesa_warning(ctx,
                    "glGetInternalformativ(bufSize = %d, but params = NULL)",
                    bufSize);
   }

   if (unsupported)
      _set_unsupported(pname, buffer, &count);

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
