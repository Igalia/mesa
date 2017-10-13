/*
 * Copyright © 2017 Intel Corporation
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

#include "spirv.h"
#include "spirv_extensions.h"

const char *
spirv_extensions_to_string(enum SpvExtension ext)
{
#define STR(x) case x: return #x;
   switch (ext) {
   STR(SPV_KHR_16bit_storage);
   STR(SPV_KHR_device_group);
   STR(SPV_KHR_multiview);
   STR(SPV_KHR_shader_ballot);
   STR(SPV_KHR_shader_draw_parameters);
   STR(SPV_KHR_storage_buffer_storage_class);
   STR(SPV_KHR_subgroup_vote);
   STR(SPV_KHR_variable_pointers);
   case SPV_EXTENSIONS_COUNT:
      unreachable("Unknown SPIR-V extension");
   }
#undef STR

   return "unknown";
}
