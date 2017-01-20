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

static bool
supports_type_conversion(fs_inst *inst) {
   switch(inst->opcode) {
   case BRW_OPCODE_MOV:
   case SHADER_OPCODE_MOV_INDIRECT:
      return true;
   case BRW_OPCODE_SEL:
      return false;
   default:
      /* FIXME: We assume the opcodes don't explicitly mentioned
       * before just work fine with arbitrary conversions.
       */
      return true;
   }
}

bool
fs_visitor::lower_d2x()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      bool inst_support_conversion = supports_type_conversion(inst);
      bool supported_conversion =
         inst_support_conversion &&
         (get_exec_type_size(inst) != 8 ||
          type_sz(inst->dst.type) > 4 ||
          type_sz(inst->dst.type) >= get_exec_type_size(inst));

      /* If the conversion is supported or there is no conversion then
       * do nothing.
       */
      if (supported_conversion ||
          (!inst_support_conversion && inst->dst.type == inst->src[0].type) ||
          inst->dst.file == BAD_FILE || inst->src[0].file == BAD_FILE)
         continue;

      /* This pass only supports conversion to narrower or equal size types. */
      if (get_exec_type_size(inst) < type_sz(inst->dst.type))
          continue;

      assert(inst->saturate == false);

      const fs_builder ibld(this, block, inst);
      fs_reg dst = inst->dst;

      if (inst_support_conversion && !supported_conversion) {
         /* From the Broadwell PRM, 3D Media GPGPU, "Double Precision Float to
          * Single Precision Float":
          *
          *    The upper Dword of every Qword will be written with undefined
          *    value when converting DF to F.
          *
          * So we need to allocate a temporary that's two registers, and then do
          * a strided MOV to get the lower DWord of every Qword that has the
          * result.
          */
         fs_reg temp = ibld.vgrf(inst->src[0].type, 1);
         fs_reg strided_temp = subscript(temp, dst.type, 0);

         /* We clone the original instruction as we are going to modify it
          * and emit another one after it.
          */
         inst->dst = strided_temp;
         /* As it is an strided destination, we write n-times more being n the
          * size difference between source and destination types. Update
          * size_written with the new destination.
          */
         inst->size_written = inst->dst.component_size(inst->exec_size);
         ibld.at(block, inst->next).MOV(dst, strided_temp);
      } else {
         fs_reg temp0 = ibld.vgrf(inst->src[0].type, 1);
         fs_reg temp1 = ibld.vgrf(inst->src[0].type, 1);
         fs_reg strided_temp1 = subscript(temp1, dst.type, 0);

         inst->dst = temp0;
         /* As it is an strided destination, we write n-times more being n the
          * size difference between source and destination types. Update
          * size_written with the new destination.
          */
         inst->size_written = inst->dst.component_size(inst->exec_size);

         /* Now, do the conversion to original destination's type. */
         fs_inst *mov = ibld.at(block, inst->next).MOV(strided_temp1, temp0);
         ibld.at(block, mov->next).MOV(dst, strided_temp1);
      }
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}
