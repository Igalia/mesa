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
fs_visitor::lower_d2x()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      if (get_exec_type_size(inst) != 8 ||
          type_sz(inst->dst.type) >= get_exec_type_size(inst) ||
          inst->dst.stride > 1)
         continue;

      assert(inst->dst.file == VGRF);
      assert(inst->saturate == false);
      fs_reg dst = inst->dst;

      const fs_builder ibld(this, block, inst);

      /* From the Broadwell PRM, 3D Media GPGPU, "Double Precision Float to
       * Single Precision Float":
       *
       *    The upper Dword of every Qword will be written with undefined
       *    value when converting DF to F.
       *
       * So we need to allocate a temporary that's two registers, and then do
       * a strided MOV to get the lower DWord of every Qword that has the
       * result.
       *
       * This pass legalizes all the DF conversions to narrower types.
       */
      switch (inst->opcode) {
      case SHADER_OPCODE_MOV_INDIRECT:
      case BRW_OPCODE_MOV: {
         fs_reg temp = ibld.vgrf(inst->src[0].type, 1);
         fs_reg strided_temp = subscript(temp, inst->dst.type, 0);
         /* We clone the original instruction as we are going to modify it
          * and emit another one after it.
          */
         fs_inst *strided_inst = new(ibld.shader->mem_ctx) fs_inst(*inst);
         strided_inst->dst = strided_temp;
         /* As it is an strided destination, we write n-times more
          * being n the size difference between source and destination types.
          */
         strided_inst->size_written *= (type_sz(inst->src[0].type) / type_sz(inst->dst.type));
         ibld.emit(strided_inst);
         ibld.MOV(dst, strided_temp);
         /* Remove original instruction, now that is superseded. */
         inst->remove(block);
         break;
      }
      case BRW_OPCODE_SEL: {
         fs_reg temp0 = ibld.vgrf(inst->src[0].type, 1);
         fs_reg strided_temp0 = subscript(temp0, inst->dst.type, 0);
         fs_reg temp1 = ibld.vgrf(inst->src[1].type, 1);
         fs_reg strided_temp1 = subscript(temp1, inst->dst.type, 0);

         /* Let's convert the operands to the destination type first */
         ibld.MOV(strided_temp0, inst->src[0]);
         ibld.MOV(strided_temp1, inst->src[1]);
         inst->src[0] = strided_temp0;
         inst->src[1] = strided_temp1;
         break;
      }
      default:
         /* FIXME: If this is not a supported instruction, then we need to support it. */
         unreachable("Not supported instruction");
      }
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}
