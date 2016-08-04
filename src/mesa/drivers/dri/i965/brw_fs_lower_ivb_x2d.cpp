/*
 * Copyright © 2016 Intel Corporation
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

#include "brw_fs.h"
#include "brw_cfg.h"
#include "brw_fs_builder.h"

using namespace brw;

bool
fs_visitor::lower_ivb_x2d()
{
   bool progress = false;

   assert(devinfo->gen == 7 && !devinfo->is_haswell);

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      if (inst->opcode != BRW_OPCODE_MOV)
         continue;

      if (inst->dst.type != BRW_REGISTER_TYPE_DF)
         continue;

      if (inst->src[0].type != BRW_REGISTER_TYPE_F &&
          inst->src[0].type != BRW_REGISTER_TYPE_D &&
          inst->src[0].type != BRW_REGISTER_TYPE_UD)
         continue;

      assert(inst->dst.file == VGRF);
      assert(inst->saturate == false);

      fs_reg dst = inst->dst;

      const fs_builder ibld(this, block, inst);

      /* In Ivybridge, converting 4 single-precision type values to 4
       * double-precision type values require to set exec_size to 8 in the
       * generated assembler:
       *
       * mov(8)   g9<1>:DF   g5<4,4,1>
       *
       * Internally, the hardware duplicates the horizontal stride, hence
       * converting just one out of two values. To avoid missing values, we
       * copy first the values in a temporal register strided to 2, and then
       * perform the conversion from there.
       */
      fs_reg temp = ibld.vgrf(inst->dst.type, 1);
      fs_reg strided_temp = subscript(temp, inst->src[0].type, 0);
      ibld.MOV(strided_temp, inst->src[0]);
      ibld.MOV(dst, strided_temp);

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}
