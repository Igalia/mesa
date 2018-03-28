/*
 * Copyright © 2018 Intel Corporation
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
 *
 */
#include "main/mtypes.h"
#include "linker_util.h"

void
linker_error(struct gl_shader_program *prog, const char *fmt, ...)
{
   va_list ap;

   ralloc_strcat(&prog->data->InfoLog, "error: ");
   va_start(ap, fmt);
   ralloc_vasprintf_append(&prog->data->InfoLog, fmt, ap);
   va_end(ap);

   prog->data->LinkStatus = LINKING_FAILURE;
}


void
linker_warning(struct gl_shader_program *prog, const char *fmt, ...)
{
   va_list ap;

   ralloc_strcat(&prog->data->InfoLog, "warning: ");
   va_start(ap, fmt);
   ralloc_vasprintf_append(&prog->data->InfoLog, fmt, ap);
   va_end(ap);

}
