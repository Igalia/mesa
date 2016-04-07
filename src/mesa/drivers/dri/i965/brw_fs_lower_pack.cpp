/*
 * Copyright © 2015 Connor Abbott
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
fs_visitor::lower_pack()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      if (inst->opcode != FS_OPCODE_PACK)
         continue;

      assert(inst->dst.file == VGRF);
      assert(inst->saturate == false);

      const fs_builder ibld(this, block, inst);

      /* In gen7 we need to split multi-register single-precision writes
       * that don't write all channels in each GRF to instructions with a
       * width of 4 to work around a hardware bug. For this to work we need
       * to make these writes to a temporary register with WE_all and then
       * copy the result to the actual destination.
       */
      fs_reg dst;
      bool force_all = false;
      if (devinfo->gen >= 8) {
         dst = inst->dst;
      } else {
         force_all = true;
         dst = ibld.vgrf(BRW_REGISTER_TYPE_DF);
      }

      for (unsigned i = 0; i < inst->sources; i++) {
         fs_inst *linst =
            ibld.MOV(stride(horiz_offset(retype(dst, inst->src[i].type), i),
                            inst->sources),
                     inst->src[i]);
         linst->force_writemask_all = linst->force_writemask_all || force_all;
      }

      if (force_all)
         ibld.MOV(inst->dst, dst);

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}
