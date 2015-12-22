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
#include "brw_state.h"
#include "main/formatquery.h"
#include "main/glformats.h"

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
   /* The Mesa layer gives us a temporary params buffer that is guaranteed
    * to be non-NULL, and have at least 16 elements.
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

   case GL_INTERNALFORMAT_PREFERRED: {
      params[0] = GL_NONE;

      /* We need to resolve an internal format that is compatible with
       * the passed internal format, and is "optimal" to the driver. This is
       * still poorly defined to us, so right now we just validate that the
       * passed internal format is supported. If so, we return the same
       * internal format, otherwise GL_NONE.
       */

      /* We need to "come up" with a type, to obtain a mesa_format out of
       * the passed internal format. Here, we get one from the internal
       * format itself, that is generic enough.
       */
      GLenum type;
      if (_mesa_is_enum_format_unsigned_int(internalFormat))
         type = GL_UNSIGNED_BYTE;
      else if (_mesa_is_enum_format_signed_int(internalFormat))
         type = GL_BYTE;
      else
         type = GL_FLOAT;

      /* Get a mesa_format from the internal format. */
      mesa_format mesa_format =
         _mesa_format_from_format_and_type(internalFormat, type);
      if (mesa_format < MESA_FORMAT_COUNT) {
         uint32_t brw_format = brw_format_for_mesa_format(mesa_format);
         if (brw_format != 0)
            params[0] = internalFormat;
      }
      break;
   }

   default:
      /* By default, we call the driver hook's fallback function from the frontend,
       * which has generic implementation for all pnames.
       */
      _mesa_query_internal_format_default(ctx, target, internalFormat, pname,
                                          params);
      break;
   }
}
