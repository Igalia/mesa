/*
 * Copyright © 2010 Intel Corporation
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

/** @file brw_fs_visitor.cpp
 *
 * This file supports generating the FS LIR from the GLSL IR.  The LIR
 * makes it easier to do backend-specific optimizations than doing so
 * in the GLSL IR or in the native code.
 */
#include <sys/types.h>

#include "main/macros.h"
#include "main/shaderobj.h"
#include "program/prog_parameter.h"
#include "program/prog_print.h"
#include "program/prog_optimize.h"
#include "util/register_allocate.h"
#include "program/hash_table.h"
#include "brw_context.h"
#include "brw_eu.h"
#include "brw_wm.h"
#include "brw_cs.h"
#include "brw_vec4.h"
#include "brw_fs.h"
#include "main/uniforms.h"
#include "glsl/glsl_types.h"
#include "glsl/ir_optimization.h"
#include "program/sampler.h"

using namespace brw;

fs_reg *
fs_visitor::emit_vs_system_value(int location)
{
   fs_reg *reg = new(this->mem_ctx)
      fs_reg(ATTR, VERT_ATTRIB_MAX, BRW_REGISTER_TYPE_D);
   brw_vs_prog_data *vs_prog_data = (brw_vs_prog_data *) prog_data;

   switch (location) {
   case SYSTEM_VALUE_BASE_VERTEX:
      reg->reg_offset = 0;
      vs_prog_data->uses_vertexid = true;
      break;
   case SYSTEM_VALUE_VERTEX_ID:
   case SYSTEM_VALUE_VERTEX_ID_ZERO_BASE:
      reg->reg_offset = 2;
      vs_prog_data->uses_vertexid = true;
      break;
   case SYSTEM_VALUE_INSTANCE_ID:
      reg->reg_offset = 3;
      vs_prog_data->uses_instanceid = true;
      break;
   default:
      unreachable("not reached");
   }

   return reg;
}

fs_inst *
fs_visitor::emit_texture_gen4(ir_texture_opcode op, fs_reg dst,
                              fs_reg coordinate, int coord_components,
                              fs_reg shadow_c,
                              fs_reg lod, fs_reg dPdy, int grad_components,
                              uint32_t sampler)
{
   int mlen;
   int base_mrf = 1;
   bool simd16 = false;
   fs_reg orig_dst;

   /* g0 header. */
   mlen = 1;

   if (shadow_c.file != BAD_FILE) {
      for (int i = 0; i < coord_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i), coordinate);
	 coordinate = offset(coordinate, bld, 1);
      }

      /* gen4's SIMD8 sampler always has the slots for u,v,r present.
       * the unused slots must be zeroed.
       */
      for (int i = coord_components; i < 3; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i), fs_reg(0.0f));
      }
      mlen += 3;

      if (op == ir_tex) {
	 /* There's no plain shadow compare message, so we use shadow
	  * compare with a bias of 0.0.
	  */
         bld.MOV(fs_reg(MRF, base_mrf + mlen), fs_reg(0.0f));
	 mlen++;
      } else if (op == ir_txb || op == ir_txl) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen), lod);
	 mlen++;
      } else {
         unreachable("Should not get here.");
      }

      bld.MOV(fs_reg(MRF, base_mrf + mlen), shadow_c);
      mlen++;
   } else if (op == ir_tex) {
      for (int i = 0; i < coord_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i), coordinate);
	 coordinate = offset(coordinate, bld, 1);
      }
      /* zero the others. */
      for (int i = coord_components; i<3; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i), fs_reg(0.0f));
      }
      /* gen4's SIMD8 sampler always has the slots for u,v,r present. */
      mlen += 3;
   } else if (op == ir_txd) {
      fs_reg &dPdx = lod;

      for (int i = 0; i < coord_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i), coordinate);
	 coordinate = offset(coordinate, bld, 1);
      }
      /* the slots for u and v are always present, but r is optional */
      mlen += MAX2(coord_components, 2);

      /*  P   = u, v, r
       * dPdx = dudx, dvdx, drdx
       * dPdy = dudy, dvdy, drdy
       *
       * 1-arg: Does not exist.
       *
       * 2-arg: dudx   dvdx   dudy   dvdy
       *        dPdx.x dPdx.y dPdy.x dPdy.y
       *        m4     m5     m6     m7
       *
       * 3-arg: dudx   dvdx   drdx   dudy   dvdy   drdy
       *        dPdx.x dPdx.y dPdx.z dPdy.x dPdy.y dPdy.z
       *        m5     m6     m7     m8     m9     m10
       */
      for (int i = 0; i < grad_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen), dPdx);
	 dPdx = offset(dPdx, bld, 1);
      }
      mlen += MAX2(grad_components, 2);

      for (int i = 0; i < grad_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen), dPdy);
	 dPdy = offset(dPdy, bld, 1);
      }
      mlen += MAX2(grad_components, 2);
   } else if (op == ir_txs) {
      /* There's no SIMD8 resinfo message on Gen4.  Use SIMD16 instead. */
      simd16 = true;
      bld.MOV(fs_reg(MRF, base_mrf + mlen, BRW_REGISTER_TYPE_UD), lod);
      mlen += 2;
   } else {
      /* Oh joy.  gen4 doesn't have SIMD8 non-shadow-compare bias/lod
       * instructions.  We'll need to do SIMD16 here.
       */
      simd16 = true;
      assert(op == ir_txb || op == ir_txl || op == ir_txf);

      for (int i = 0; i < coord_components; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i * 2, coordinate.type),
                 coordinate);
	 coordinate = offset(coordinate, bld, 1);
      }

      /* Initialize the rest of u/v/r with 0.0.  Empirically, this seems to
       * be necessary for TXF (ld), but seems wise to do for all messages.
       */
      for (int i = coord_components; i < 3; i++) {
         bld.MOV(fs_reg(MRF, base_mrf + mlen + i * 2), fs_reg(0.0f));
      }

      /* lod/bias appears after u/v/r. */
      mlen += 6;

      bld.MOV(fs_reg(MRF, base_mrf + mlen, lod.type), lod);
      mlen++;

      /* The unused upper half. */
      mlen++;
   }

   if (simd16) {
      /* Now, since we're doing simd16, the return is 2 interleaved
       * vec4s where the odd-indexed ones are junk. We'll need to move
       * this weirdness around to the expected layout.
       */
      orig_dst = dst;
      dst = fs_reg(GRF, alloc.allocate(8), orig_dst.type);
   }

   enum opcode opcode;
   switch (op) {
   case ir_tex: opcode = SHADER_OPCODE_TEX; break;
   case ir_txb: opcode = FS_OPCODE_TXB; break;
   case ir_txl: opcode = SHADER_OPCODE_TXL; break;
   case ir_txd: opcode = SHADER_OPCODE_TXD; break;
   case ir_txs: opcode = SHADER_OPCODE_TXS; break;
   case ir_txf: opcode = SHADER_OPCODE_TXF; break;
   default:
      unreachable("not reached");
   }

   fs_inst *inst = bld.emit(opcode, dst, reg_undef, fs_reg(sampler));
   inst->base_mrf = base_mrf;
   inst->mlen = mlen;
   inst->header_size = 1;
   inst->regs_written = simd16 ? 8 : 4;

   if (simd16) {
      for (int i = 0; i < 4; i++) {
         bld.MOV(orig_dst, dst);
	 orig_dst = offset(orig_dst, bld, 1);
	 dst = offset(dst, bld, 2);
      }
   }

   return inst;
}

fs_inst *
fs_visitor::emit_texture_gen4_simd16(ir_texture_opcode op, fs_reg dst,
                                     fs_reg coordinate, int vector_elements,
                                     fs_reg shadow_c, fs_reg lod,
                                     uint32_t sampler)
{
   fs_reg message(MRF, 2, BRW_REGISTER_TYPE_F);
   bool has_lod = op == ir_txl || op == ir_txb || op == ir_txf || op == ir_txs;

   if (has_lod && shadow_c.file != BAD_FILE)
      no16("TXB and TXL with shadow comparison unsupported in SIMD16.");

   if (op == ir_txd)
      no16("textureGrad unsupported in SIMD16.");

   /* Copy the coordinates. */
   for (int i = 0; i < vector_elements; i++) {
      bld.MOV(retype(offset(message, bld, i), coordinate.type), coordinate);
      coordinate = offset(coordinate, bld, 1);
   }

   fs_reg msg_end = offset(message, bld, vector_elements);

   /* Messages other than sample and ld require all three components */
   if (vector_elements > 0 && (has_lod || shadow_c.file != BAD_FILE)) {
      for (int i = vector_elements; i < 3; i++) {
         bld.MOV(offset(message, bld, i), fs_reg(0.0f));
      }
      msg_end = offset(message, bld, 3);
   }

   if (has_lod) {
      fs_reg msg_lod = retype(msg_end, op == ir_txf ?
                              BRW_REGISTER_TYPE_UD : BRW_REGISTER_TYPE_F);
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);
   }

   if (shadow_c.file != BAD_FILE) {
      fs_reg msg_ref = offset(message, bld, 3 + has_lod);
      bld.MOV(msg_ref, shadow_c);
      msg_end = offset(msg_ref, bld, 1);
   }

   enum opcode opcode;
   switch (op) {
   case ir_tex: opcode = SHADER_OPCODE_TEX; break;
   case ir_txb: opcode = FS_OPCODE_TXB;     break;
   case ir_txd: opcode = SHADER_OPCODE_TXD; break;
   case ir_txl: opcode = SHADER_OPCODE_TXL; break;
   case ir_txs: opcode = SHADER_OPCODE_TXS; break;
   case ir_txf: opcode = SHADER_OPCODE_TXF; break;
   default: unreachable("not reached");
   }

   fs_inst *inst = bld.emit(opcode, dst, reg_undef, fs_reg(sampler));
   inst->base_mrf = message.reg - 1;
   inst->mlen = msg_end.reg - inst->base_mrf;
   inst->header_size = 1;
   inst->regs_written = 8;

   return inst;
}

/* gen5's sampler has slots for u, v, r, array index, then optional
 * parameters like shadow comparitor or LOD bias.  If optional
 * parameters aren't present, those base slots are optional and don't
 * need to be included in the message.
 *
 * We don't fill in the unnecessary slots regardless, which may look
 * surprising in the disassembly.
 */
fs_inst *
fs_visitor::emit_texture_gen5(ir_texture_opcode op, fs_reg dst,
                              fs_reg coordinate, int vector_elements,
                              fs_reg shadow_c,
                              fs_reg lod, fs_reg lod2, int grad_components,
                              fs_reg sample_index, uint32_t sampler,
                              bool has_offset)
{
   int reg_width = dispatch_width / 8;
   unsigned header_size = 0;

   fs_reg message(MRF, 2, BRW_REGISTER_TYPE_F);
   fs_reg msg_coords = message;

   if (has_offset) {
      /* The offsets set up by the ir_texture visitor are in the
       * m1 header, so we can't go headerless.
       */
      header_size = 1;
      message.reg--;
   }

   for (int i = 0; i < vector_elements; i++) {
      bld.MOV(retype(offset(msg_coords, bld, i), coordinate.type), coordinate);
      coordinate = offset(coordinate, bld, 1);
   }
   fs_reg msg_end = offset(msg_coords, bld, vector_elements);
   fs_reg msg_lod = offset(msg_coords, bld, 4);

   if (shadow_c.file != BAD_FILE) {
      fs_reg msg_shadow = msg_lod;
      bld.MOV(msg_shadow, shadow_c);
      msg_lod = offset(msg_shadow, bld, 1);
      msg_end = msg_lod;
   }

   enum opcode opcode;
   switch (op) {
   case ir_tex:
      opcode = SHADER_OPCODE_TEX;
      break;
   case ir_txb:
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);

      opcode = FS_OPCODE_TXB;
      break;
   case ir_txl:
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);

      opcode = SHADER_OPCODE_TXL;
      break;
   case ir_txd: {
      /**
       *  P   =  u,    v,    r
       * dPdx = dudx, dvdx, drdx
       * dPdy = dudy, dvdy, drdy
       *
       * Load up these values:
       * - dudx   dudy   dvdx   dvdy   drdx   drdy
       * - dPdx.x dPdy.x dPdx.y dPdy.y dPdx.z dPdy.z
       */
      msg_end = msg_lod;
      for (int i = 0; i < grad_components; i++) {
         bld.MOV(msg_end, lod);
         lod = offset(lod, bld, 1);
         msg_end = offset(msg_end, bld, 1);

         bld.MOV(msg_end, lod2);
         lod2 = offset(lod2, bld, 1);
         msg_end = offset(msg_end, bld, 1);
      }

      opcode = SHADER_OPCODE_TXD;
      break;
   }
   case ir_txs:
      msg_lod = retype(msg_end, BRW_REGISTER_TYPE_UD);
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);

      opcode = SHADER_OPCODE_TXS;
      break;
   case ir_query_levels:
      msg_lod = msg_end;
      bld.MOV(retype(msg_lod, BRW_REGISTER_TYPE_UD), fs_reg(0u));
      msg_end = offset(msg_lod, bld, 1);

      opcode = SHADER_OPCODE_TXS;
      break;
   case ir_txf:
      msg_lod = offset(msg_coords, bld, 3);
      bld.MOV(retype(msg_lod, BRW_REGISTER_TYPE_UD), lod);
      msg_end = offset(msg_lod, bld, 1);

      opcode = SHADER_OPCODE_TXF;
      break;
   case ir_txf_ms:
      msg_lod = offset(msg_coords, bld, 3);
      /* lod */
      bld.MOV(retype(msg_lod, BRW_REGISTER_TYPE_UD), fs_reg(0u));
      /* sample index */
      bld.MOV(retype(offset(msg_lod, bld, 1), BRW_REGISTER_TYPE_UD), sample_index);
      msg_end = offset(msg_lod, bld, 2);

      opcode = SHADER_OPCODE_TXF_CMS;
      break;
   case ir_lod:
      opcode = SHADER_OPCODE_LOD;
      break;
   case ir_tg4:
      opcode = SHADER_OPCODE_TG4;
      break;
   default:
      unreachable("not reached");
   }

   fs_inst *inst = bld.emit(opcode, dst, reg_undef, fs_reg(sampler));
   inst->base_mrf = message.reg;
   inst->mlen = msg_end.reg - message.reg;
   inst->header_size = header_size;
   inst->regs_written = 4 * reg_width;

   if (inst->mlen > MAX_SAMPLER_MESSAGE_SIZE) {
      fail("Message length >" STRINGIFY(MAX_SAMPLER_MESSAGE_SIZE)
           " disallowed by hardware\n");
   }

   return inst;
}

static bool
is_high_sampler(const struct brw_device_info *devinfo, fs_reg sampler)
{
   if (devinfo->gen < 8 && !devinfo->is_haswell)
      return false;

   return sampler.file != IMM || sampler.fixed_hw_reg.dw1.ud >= 16;
}

fs_inst *
fs_visitor::emit_texture_gen7(ir_texture_opcode op, fs_reg dst,
                              fs_reg coordinate, int coord_components,
                              fs_reg shadow_c,
                              fs_reg lod, fs_reg lod2, int grad_components,
                              fs_reg sample_index, fs_reg mcs, fs_reg sampler,
                              fs_reg offset_value)
{
   int reg_width = dispatch_width / 8;
   unsigned header_size = 0;

   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, MAX_SAMPLER_MESSAGE_SIZE);
   for (int i = 0; i < MAX_SAMPLER_MESSAGE_SIZE; i++) {
      sources[i] = vgrf(glsl_type::float_type);
   }
   int length = 0;

   if (op == ir_tg4 || offset_value.file != BAD_FILE ||
       is_high_sampler(devinfo, sampler)) {
      /* For general texture offsets (no txf workaround), we need a header to
       * put them in.  Note that for SIMD16 we're making space for two actual
       * hardware registers here, so the emit will have to fix up for this.
       *
       * * ir4_tg4 needs to place its channel select in the header,
       * for interaction with ARB_texture_swizzle
       *
       * The sampler index is only 4-bits, so for larger sampler numbers we
       * need to offset the Sampler State Pointer in the header.
       */
      header_size = 1;
      sources[0] = fs_reg(GRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
      length++;
   }

   if (shadow_c.file != BAD_FILE) {
      bld.MOV(sources[length], shadow_c);
      length++;
   }

   bool has_nonconstant_offset =
      offset_value.file != BAD_FILE && offset_value.file != IMM;
   bool coordinate_done = false;

   /* The sampler can only meaningfully compute LOD for fragment shader
    * messages. For all other stages, we change the opcode to ir_txl and
    * hardcode the LOD to 0.
    */
   if (stage != MESA_SHADER_FRAGMENT && op == ir_tex) {
      op = ir_txl;
      lod = fs_reg(0.0f);
   }

   /* Set up the LOD info */
   switch (op) {
   case ir_tex:
   case ir_lod:
      break;
   case ir_txb:
      bld.MOV(sources[length], lod);
      length++;
      break;
   case ir_txl:
      bld.MOV(sources[length], lod);
      length++;
      break;
   case ir_txd: {
      no16("Gen7 does not support sample_d/sample_d_c in SIMD16 mode.");

      /* Load dPdx and the coordinate together:
       * [hdr], [ref], x, dPdx.x, dPdy.x, y, dPdx.y, dPdy.y, z, dPdx.z, dPdy.z
       */
      for (int i = 0; i < coord_components; i++) {
         bld.MOV(sources[length], coordinate);
	 coordinate = offset(coordinate, bld, 1);
	 length++;

         /* For cube map array, the coordinate is (u,v,r,ai) but there are
          * only derivatives for (u, v, r).
          */
         if (i < grad_components) {
            bld.MOV(sources[length], lod);
            lod = offset(lod, bld, 1);
            length++;

            bld.MOV(sources[length], lod2);
            lod2 = offset(lod2, bld, 1);
            length++;
         }
      }

      coordinate_done = true;
      break;
   }
   case ir_txs:
      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), lod);
      length++;
      break;
   case ir_query_levels:
      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), fs_reg(0u));
      length++;
      break;
   case ir_txf:
      /* Unfortunately, the parameters for LD are intermixed: u, lod, v, r.
       * On Gen9 they are u, v, lod, r
       */

      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), coordinate);
      coordinate = offset(coordinate, bld, 1);
      length++;

      if (devinfo->gen >= 9) {
         if (coord_components >= 2) {
            bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), coordinate);
            coordinate = offset(coordinate, bld, 1);
         }
         length++;
      }

      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), lod);
      length++;

      for (int i = devinfo->gen >= 9 ? 2 : 1; i < coord_components; i++) {
         bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), coordinate);
	 coordinate = offset(coordinate, bld, 1);
	 length++;
      }

      coordinate_done = true;
      break;
   case ir_txf_ms:
      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), sample_index);
      length++;

      /* data from the multisample control surface */
      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), mcs);
      length++;

      /* there is no offsetting for this message; just copy in the integer
       * texture coordinates
       */
      for (int i = 0; i < coord_components; i++) {
         bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), coordinate);
         coordinate = offset(coordinate, bld, 1);
         length++;
      }

      coordinate_done = true;
      break;
   case ir_tg4:
      if (has_nonconstant_offset) {
         if (shadow_c.file != BAD_FILE)
            no16("Gen7 does not support gather4_po_c in SIMD16 mode.");

         /* More crazy intermixing */
         for (int i = 0; i < 2; i++) { /* u, v */
            bld.MOV(sources[length], coordinate);
            coordinate = offset(coordinate, bld, 1);
            length++;
         }

         for (int i = 0; i < 2; i++) { /* offu, offv */
            bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), offset_value);
            offset_value = offset(offset_value, bld, 1);
            length++;
         }

         if (coord_components == 3) { /* r if present */
            bld.MOV(sources[length], coordinate);
            coordinate = offset(coordinate, bld, 1);
            length++;
         }

         coordinate_done = true;
      }
      break;
   }

   /* Set up the coordinate (except for cases where it was done above) */
   if (!coordinate_done) {
      for (int i = 0; i < coord_components; i++) {
         bld.MOV(sources[length], coordinate);
         coordinate = offset(coordinate, bld, 1);
         length++;
      }
   }

   int mlen;
   if (reg_width == 2)
      mlen = length * reg_width - header_size;
   else
      mlen = length * reg_width;

   fs_reg src_payload = fs_reg(GRF, alloc.allocate(mlen),
                               BRW_REGISTER_TYPE_F);
   bld.LOAD_PAYLOAD(src_payload, sources, length, header_size);

   /* Generate the SEND */
   enum opcode opcode;
   switch (op) {
   case ir_tex: opcode = SHADER_OPCODE_TEX; break;
   case ir_txb: opcode = FS_OPCODE_TXB; break;
   case ir_txl: opcode = SHADER_OPCODE_TXL; break;
   case ir_txd: opcode = SHADER_OPCODE_TXD; break;
   case ir_txf: opcode = SHADER_OPCODE_TXF; break;
   case ir_txf_ms: opcode = SHADER_OPCODE_TXF_CMS; break;
   case ir_txs: opcode = SHADER_OPCODE_TXS; break;
   case ir_query_levels: opcode = SHADER_OPCODE_TXS; break;
   case ir_lod: opcode = SHADER_OPCODE_LOD; break;
   case ir_tg4:
      if (has_nonconstant_offset)
         opcode = SHADER_OPCODE_TG4_OFFSET;
      else
         opcode = SHADER_OPCODE_TG4;
      break;
   default:
      unreachable("not reached");
   }
   fs_inst *inst = bld.emit(opcode, dst, src_payload, sampler);
   inst->base_mrf = -1;
   inst->mlen = mlen;
   inst->header_size = header_size;
   inst->regs_written = 4 * reg_width;

   if (inst->mlen > MAX_SAMPLER_MESSAGE_SIZE) {
      fail("Message length >" STRINGIFY(MAX_SAMPLER_MESSAGE_SIZE)
           " disallowed by hardware\n");
   }

   return inst;
}

fs_reg
fs_visitor::rescale_texcoord(fs_reg coordinate, int coord_components,
                             bool is_rect, uint32_t sampler, int texunit)
{
   bool needs_gl_clamp = true;
   fs_reg scale_x, scale_y;

   /* The 965 requires the EU to do the normalization of GL rectangle
    * texture coordinates.  We use the program parameter state
    * tracking to get the scaling factor.
    */
   if (is_rect &&
       (devinfo->gen < 6 ||
        (devinfo->gen >= 6 && (key_tex->gl_clamp_mask[0] & (1 << sampler) ||
                               key_tex->gl_clamp_mask[1] & (1 << sampler))))) {
      struct gl_program_parameter_list *params = prog->Parameters;
      int tokens[STATE_LENGTH] = {
	 STATE_INTERNAL,
	 STATE_TEXRECT_SCALE,
	 texunit,
	 0,
	 0
      };

      no16("rectangle scale uniform setup not supported on SIMD16\n");
      if (dispatch_width == 16) {
	 return coordinate;
      }

      GLuint index = _mesa_add_state_reference(params,
					       (gl_state_index *)tokens);
      /* Try to find existing copies of the texrect scale uniforms. */
      for (unsigned i = 0; i < uniforms; i++) {
         if (stage_prog_data->param[i] ==
             &prog->Parameters->ParameterValues[index][0]) {
            scale_x = fs_reg(UNIFORM, i);
            scale_y = fs_reg(UNIFORM, i + 1);
            break;
         }
      }

      /* If we didn't already set them up, do so now. */
      if (scale_x.file == BAD_FILE) {
         scale_x = fs_reg(UNIFORM, uniforms);
         scale_y = fs_reg(UNIFORM, uniforms + 1);

         stage_prog_data->param[uniforms++] =
            &prog->Parameters->ParameterValues[index][0];
         stage_prog_data->param[uniforms++] =
            &prog->Parameters->ParameterValues[index][1];
      }
   }

   /* The 965 requires the EU to do the normalization of GL rectangle
    * texture coordinates.  We use the program parameter state
    * tracking to get the scaling factor.
    */
   if (devinfo->gen < 6 && is_rect) {
      fs_reg dst = fs_reg(GRF, alloc.allocate(coord_components));
      fs_reg src = coordinate;
      coordinate = dst;

      bld.MUL(dst, src, scale_x);
      dst = offset(dst, bld, 1);
      src = offset(src, bld, 1);
      bld.MUL(dst, src, scale_y);
   } else if (is_rect) {
      /* On gen6+, the sampler handles the rectangle coordinates
       * natively, without needing rescaling.  But that means we have
       * to do GL_CLAMP clamping at the [0, width], [0, height] scale,
       * not [0, 1] like the default case below.
       */
      needs_gl_clamp = false;

      for (int i = 0; i < 2; i++) {
	 if (key_tex->gl_clamp_mask[i] & (1 << sampler)) {
	    fs_reg chan = coordinate;
	    chan = offset(chan, bld, i);

            set_condmod(BRW_CONDITIONAL_GE,
                        bld.emit(BRW_OPCODE_SEL, chan, chan, fs_reg(0.0f)));

	    /* Our parameter comes in as 1.0/width or 1.0/height,
	     * because that's what people normally want for doing
	     * texture rectangle handling.  We need width or height
	     * for clamping, but we don't care enough to make a new
	     * parameter type, so just invert back.
	     */
	    fs_reg limit = vgrf(glsl_type::float_type);
            bld.MOV(limit, i == 0 ? scale_x : scale_y);
            bld.emit(SHADER_OPCODE_RCP, limit, limit);

            set_condmod(BRW_CONDITIONAL_L,
                        bld.emit(BRW_OPCODE_SEL, chan, chan, limit));
	 }
      }
   }

   if (coord_components > 0 && needs_gl_clamp) {
      for (int i = 0; i < MIN2(coord_components, 3); i++) {
	 if (key_tex->gl_clamp_mask[i] & (1 << sampler)) {
	    fs_reg chan = coordinate;
	    chan = offset(chan, bld, i);
            set_saturate(true, bld.MOV(chan, chan));
	 }
      }
   }
   return coordinate;
}

/* Sample from the MCS surface attached to this multisample texture. */
fs_reg
fs_visitor::emit_mcs_fetch(fs_reg coordinate, int components, fs_reg sampler)
{
   int reg_width = dispatch_width / 8;
   fs_reg payload = fs_reg(GRF, alloc.allocate(components * reg_width),
                           BRW_REGISTER_TYPE_F);
   fs_reg dest = vgrf(glsl_type::uvec4_type);
   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, components);

   /* parameters are: u, v, r; missing parameters are treated as zero */
   for (int i = 0; i < components; i++) {
      sources[i] = vgrf(glsl_type::float_type);
      bld.MOV(retype(sources[i], BRW_REGISTER_TYPE_D), coordinate);
      coordinate = offset(coordinate, bld, 1);
   }

   bld.LOAD_PAYLOAD(payload, sources, components, 0);

   fs_inst *inst = bld.emit(SHADER_OPCODE_TXF_MCS, dest, payload, sampler);
   inst->base_mrf = -1;
   inst->mlen = components * reg_width;
   inst->header_size = 0;
   inst->regs_written = 4 * reg_width; /* we only care about one reg of
                                        * response, but the sampler always
                                        * writes 4/8
                                        */

   return dest;
}

void
fs_visitor::emit_texture(ir_texture_opcode op,
                         const glsl_type *dest_type,
                         fs_reg coordinate, int coord_components,
                         fs_reg shadow_c,
                         fs_reg lod, fs_reg lod2, int grad_components,
                         fs_reg sample_index,
                         fs_reg offset_value,
                         fs_reg mcs,
                         int gather_component,
                         bool is_cube_array,
                         bool is_rect,
                         uint32_t sampler,
                         fs_reg sampler_reg, int texunit)
{
   fs_inst *inst = NULL;

   if (op == ir_tg4) {
      /* When tg4 is used with the degenerate ZERO/ONE swizzles, don't bother
       * emitting anything other than setting up the constant result.
       */
      int swiz = GET_SWZ(key_tex->swizzles[sampler], gather_component);
      if (swiz == SWIZZLE_ZERO || swiz == SWIZZLE_ONE) {

         fs_reg res = vgrf(glsl_type::vec4_type);
         this->result = res;

         for (int i=0; i<4; i++) {
            bld.MOV(res, fs_reg(swiz == SWIZZLE_ZERO ? 0.0f : 1.0f));
            res = offset(res, bld, 1);
         }
         return;
      }
   }

   if (coordinate.file != BAD_FILE) {
      /* FINISHME: Texture coordinate rescaling doesn't work with non-constant
       * samplers.  This should only be a problem with GL_CLAMP on Gen7.
       */
      coordinate = rescale_texcoord(coordinate, coord_components, is_rect,
                                    sampler, texunit);
   }

   /* Writemasking doesn't eliminate channels on SIMD8 texture
    * samples, so don't worry about them.
    */
   fs_reg dst = vgrf(glsl_type::get_instance(dest_type->base_type, 4, 1));

   if (devinfo->gen >= 7) {
      inst = emit_texture_gen7(op, dst, coordinate, coord_components,
                               shadow_c, lod, lod2, grad_components,
                               sample_index, mcs, sampler_reg,
                               offset_value);
   } else if (devinfo->gen >= 5) {
      inst = emit_texture_gen5(op, dst, coordinate, coord_components,
                               shadow_c, lod, lod2, grad_components,
                               sample_index, sampler,
                               offset_value.file != BAD_FILE);
   } else if (dispatch_width == 16) {
      inst = emit_texture_gen4_simd16(op, dst, coordinate, coord_components,
                                      shadow_c, lod, sampler);
   } else {
      inst = emit_texture_gen4(op, dst, coordinate, coord_components,
                               shadow_c, lod, lod2, grad_components,
                               sampler);
   }

   if (shadow_c.file != BAD_FILE)
      inst->shadow_compare = true;

   if (offset_value.file == IMM)
      inst->offset = offset_value.fixed_hw_reg.dw1.ud;

   if (op == ir_tg4) {
      inst->offset |=
         gather_channel(gather_component, sampler) << 16; /* M0.2:16-17 */

      if (devinfo->gen == 6)
         emit_gen6_gather_wa(key_tex->gen6_gather_wa[sampler], dst);
   }

   /* fixup #layers for cube map arrays */
   if (op == ir_txs && is_cube_array) {
      fs_reg depth = offset(dst, bld, 2);
      fs_reg fixed_depth = vgrf(glsl_type::int_type);
      bld.emit(SHADER_OPCODE_INT_QUOTIENT, fixed_depth, depth, fs_reg(6));

      fs_reg *fixed_payload = ralloc_array(mem_ctx, fs_reg, inst->regs_written);
      int components = inst->regs_written / (inst->exec_size / 8);
      for (int i = 0; i < components; i++) {
         if (i == 2) {
            fixed_payload[i] = fixed_depth;
         } else {
            fixed_payload[i] = offset(dst, bld, i);
         }
      }
      bld.LOAD_PAYLOAD(dst, fixed_payload, components, 0);
   }

   swizzle_result(op, dest_type->vector_elements, dst, sampler);
}

/**
 * Apply workarounds for Gen6 gather with UINT/SINT
 */
void
fs_visitor::emit_gen6_gather_wa(uint8_t wa, fs_reg dst)
{
   if (!wa)
      return;

   int width = (wa & WA_8BIT) ? 8 : 16;

   for (int i = 0; i < 4; i++) {
      fs_reg dst_f = retype(dst, BRW_REGISTER_TYPE_F);
      /* Convert from UNORM to UINT */
      bld.MUL(dst_f, dst_f, fs_reg((float)((1 << width) - 1)));
      bld.MOV(dst, dst_f);

      if (wa & WA_SIGN) {
         /* Reinterpret the UINT value as a signed INT value by
          * shifting the sign bit into place, then shifting back
          * preserving sign.
          */
         bld.SHL(dst, dst, fs_reg(32 - width));
         bld.ASR(dst, dst, fs_reg(32 - width));
      }

      dst = offset(dst, bld, 1);
   }
}

/**
 * Set up the gather channel based on the swizzle, for gather4.
 */
uint32_t
fs_visitor::gather_channel(int orig_chan, uint32_t sampler)
{
   int swiz = GET_SWZ(key_tex->swizzles[sampler], orig_chan);
   switch (swiz) {
      case SWIZZLE_X: return 0;
      case SWIZZLE_Y:
         /* gather4 sampler is broken for green channel on RG32F --
          * we must ask for blue instead.
          */
         if (key_tex->gather_channel_quirk_mask & (1 << sampler))
            return 2;
         return 1;
      case SWIZZLE_Z: return 2;
      case SWIZZLE_W: return 3;
      default:
         unreachable("Not reached"); /* zero, one swizzles handled already */
   }
}

/**
 * Swizzle the result of a texture result.  This is necessary for
 * EXT_texture_swizzle as well as DEPTH_TEXTURE_MODE for shadow comparisons.
 */
void
fs_visitor::swizzle_result(ir_texture_opcode op, int dest_components,
                           fs_reg orig_val, uint32_t sampler)
{
   if (op == ir_query_levels) {
      /* # levels is in .w */
      this->result = offset(orig_val, bld, 3);
      return;
   }

   this->result = orig_val;

   /* txs,lod don't actually sample the texture, so swizzling the result
    * makes no sense.
    */
   if (op == ir_txs || op == ir_lod || op == ir_tg4)
      return;

   if (dest_components == 1) {
      /* Ignore DEPTH_TEXTURE_MODE swizzling. */
   } else if (key_tex->swizzles[sampler] != SWIZZLE_NOOP) {
      fs_reg swizzled_result = vgrf(glsl_type::vec4_type);
      swizzled_result.type = orig_val.type;

      for (int i = 0; i < 4; i++) {
	 int swiz = GET_SWZ(key_tex->swizzles[sampler], i);
	 fs_reg l = swizzled_result;
	 l = offset(l, bld, i);

	 if (swiz == SWIZZLE_ZERO) {
            bld.MOV(l, fs_reg(0.0f));
	 } else if (swiz == SWIZZLE_ONE) {
            bld.MOV(l, fs_reg(1.0f));
	 } else {
            bld.MOV(l, offset(orig_val, bld,
                                  GET_SWZ(key_tex->swizzles[sampler], i)));
	 }
      }
      this->result = swizzled_result;
   }
}

/**
 * Try to replace IF/MOV/ELSE/MOV/ENDIF with SEL.
 *
 * Many GLSL shaders contain the following pattern:
 *
 *    x = condition ? foo : bar
 *
 * The compiler emits an ir_if tree for this, since each subexpression might be
 * a complex tree that could have side-effects or short-circuit logic.
 *
 * However, the common case is to simply select one of two constants or
 * variable values---which is exactly what SEL is for.  In this case, the
 * assembly looks like:
 *
 *    (+f0) IF
 *    MOV dst src0
 *    ELSE
 *    MOV dst src1
 *    ENDIF
 *
 * which can be easily translated into:
 *
 *    (+f0) SEL dst src0 src1
 *
 * If src0 is an immediate value, we promote it to a temporary GRF.
 */
bool
fs_visitor::try_replace_with_sel()
{
   fs_inst *endif_inst = (fs_inst *) instructions.get_tail();
   assert(endif_inst->opcode == BRW_OPCODE_ENDIF);

   /* Pattern match in reverse: IF, MOV, ELSE, MOV, ENDIF. */
   int opcodes[] = {
      BRW_OPCODE_IF, BRW_OPCODE_MOV, BRW_OPCODE_ELSE, BRW_OPCODE_MOV,
   };

   fs_inst *match = (fs_inst *) endif_inst->prev;
   for (int i = 0; i < 4; i++) {
      if (match->is_head_sentinel() || match->opcode != opcodes[4-i-1])
         return false;
      match = (fs_inst *) match->prev;
   }

   /* The opcodes match; it looks like the right sequence of instructions. */
   fs_inst *else_mov = (fs_inst *) endif_inst->prev;
   fs_inst *then_mov = (fs_inst *) else_mov->prev->prev;
   fs_inst *if_inst = (fs_inst *) then_mov->prev;

   /* Check that the MOVs are the right form. */
   if (then_mov->dst.equals(else_mov->dst) &&
       !then_mov->is_partial_write() &&
       !else_mov->is_partial_write()) {

      /* Remove the matched instructions; we'll emit a SEL to replace them. */
      while (!if_inst->next->is_tail_sentinel())
         if_inst->next->exec_node::remove();
      if_inst->exec_node::remove();

      /* Only the last source register can be a constant, so if the MOV in
       * the "then" clause uses a constant, we need to put it in a temporary.
       */
      fs_reg src0(then_mov->src[0]);
      if (src0.file == IMM) {
         src0 = vgrf(glsl_type::float_type);
         src0.type = then_mov->src[0].type;
         bld.MOV(src0, then_mov->src[0]);
      }

      if (if_inst->conditional_mod) {
         /* Sandybridge-specific IF with embedded comparison */
         bld.CMP(bld.null_reg_d(), if_inst->src[0], if_inst->src[1],
                 if_inst->conditional_mod);
         set_predicate(BRW_PREDICATE_NORMAL,
                       bld.emit(BRW_OPCODE_SEL, then_mov->dst,
                                src0, else_mov->src[0]));
      } else {
         /* Separate CMP and IF instructions */
         set_predicate_inv(if_inst->predicate, if_inst->predicate_inverse,
                           bld.emit(BRW_OPCODE_SEL, then_mov->dst,
                                    src0, else_mov->src[0]));
      }

      return true;
   }

   return false;
}

void
fs_visitor::emit_untyped_atomic(unsigned atomic_op, unsigned surf_index,
                                fs_reg dst, fs_reg offset, fs_reg src0,
                                fs_reg src1)
{
   int reg_width = dispatch_width / 8;
   int length = 0;

   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, 4);

   sources[0] = fs_reg(GRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
   /* Initialize the sample mask in the message header. */
   bld.exec_all().MOV(sources[0], fs_reg(0u));

   if (stage == MESA_SHADER_FRAGMENT) {
      if (((brw_wm_prog_data*)this->prog_data)->uses_kill) {
         bld.exec_all()
            .MOV(component(sources[0], 7), brw_flag_reg(0, 1));
      } else {
         bld.exec_all()
            .MOV(component(sources[0], 7),
                 retype(brw_vec1_grf(1, 7), BRW_REGISTER_TYPE_UD));
      }
   } else {
      /* The execution mask is part of the side-band information sent together with
       * the message payload to the data port. It's implicitly ANDed with the sample
       * mask sent in the header to compute the actual set of channels that execute
       * the atomic operation.
       */
      assert(stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_COMPUTE);
      bld.exec_all()
         .MOV(component(sources[0], 7), fs_reg(0xffffu));
   }
   length++;

   /* Set the atomic operation offset. */
   sources[1] = vgrf(glsl_type::uint_type);
   bld.MOV(sources[1], offset);
   length++;

   /* Set the atomic operation arguments. */
   if (src0.file != BAD_FILE) {
      sources[length] = vgrf(glsl_type::uint_type);
      bld.MOV(sources[length], src0);
      length++;
   }

   if (src1.file != BAD_FILE) {
      sources[length] = vgrf(glsl_type::uint_type);
      bld.MOV(sources[length], src1);
      length++;
   }

   int mlen = 1 + (length - 1) * reg_width;
   fs_reg src_payload = fs_reg(GRF, alloc.allocate(mlen),
                               BRW_REGISTER_TYPE_UD);
   bld.LOAD_PAYLOAD(src_payload, sources, length, 1);

   /* Emit the instruction. */
   fs_inst *inst = bld.emit(SHADER_OPCODE_UNTYPED_ATOMIC, dst, src_payload,
                            fs_reg(surf_index), fs_reg(atomic_op));
   inst->mlen = mlen;
}

void
fs_visitor::emit_untyped_surface_read(unsigned surf_index, fs_reg dst,
                                      fs_reg offset)
{
   int reg_width = dispatch_width / 8;

   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, 1);

   /* Set the surface read offset. */
   sources[0] = vgrf(glsl_type::uint_type);
   bld.MOV(sources[0], offset);

   int mlen = reg_width;
   fs_reg src_payload = fs_reg(GRF, alloc.allocate(mlen),
                               BRW_REGISTER_TYPE_UD, dispatch_width);
   fs_inst *inst = bld.LOAD_PAYLOAD(src_payload, sources, 1, 0);

   /* Emit the instruction. */
   inst = bld.emit(SHADER_OPCODE_UNTYPED_SURFACE_READ, dst, src_payload,
                   fs_reg(surf_index), fs_reg(1));
   inst->mlen = mlen;
}

/** Emits a dummy fragment shader consisting of magenta for bringup purposes. */
void
fs_visitor::emit_dummy_fs()
{
   int reg_width = dispatch_width / 8;

   /* Everyone's favorite color. */
   const float color[4] = { 1.0, 0.0, 1.0, 0.0 };
   for (int i = 0; i < 4; i++) {
      bld.MOV(fs_reg(MRF, 2 + i * reg_width, BRW_REGISTER_TYPE_F),
              fs_reg(color[i]));
   }

   fs_inst *write;
   write = bld.emit(FS_OPCODE_FB_WRITE);
   write->eot = true;
   if (devinfo->gen >= 6) {
      write->base_mrf = 2;
      write->mlen = 4 * reg_width;
   } else {
      write->header_size = 2;
      write->base_mrf = 0;
      write->mlen = 2 + 4 * reg_width;
   }

   /* Tell the SF we don't have any inputs.  Gen4-5 require at least one
    * varying to avoid GPU hangs, so set that.
    */
   brw_wm_prog_data *wm_prog_data = (brw_wm_prog_data *) this->prog_data;
   wm_prog_data->num_varying_inputs = devinfo->gen < 6 ? 1 : 0;
   memset(wm_prog_data->urb_setup, -1,
          sizeof(wm_prog_data->urb_setup[0]) * VARYING_SLOT_MAX);

   /* We don't have any uniforms. */
   stage_prog_data->nr_params = 0;
   stage_prog_data->nr_pull_params = 0;
   stage_prog_data->curb_read_length = 0;
   stage_prog_data->dispatch_grf_start_reg = 2;
   wm_prog_data->dispatch_grf_start_reg_16 = 2;
   grf_used = 1; /* Gen4-5 don't allow zero GRF blocks */

   calculate_cfg();
}

/* The register location here is relative to the start of the URB
 * data.  It will get adjusted to be a real location before
 * generate_code() time.
 */
struct brw_reg
fs_visitor::interp_reg(int location, int channel)
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_data *prog_data = (brw_wm_prog_data*) this->prog_data;
   int regnr = prog_data->urb_setup[location] * 2 + channel / 2;
   int stride = (channel & 1) * 4;

   assert(prog_data->urb_setup[location] != -1);

   return brw_vec1_grf(regnr, stride);
}

/** Emits the interpolation for the varying inputs. */
void
fs_visitor::emit_interpolation_setup_gen4()
{
   struct brw_reg g1_uw = retype(brw_vec1_grf(1, 0), BRW_REGISTER_TYPE_UW);

   fs_builder abld = bld.annotate("compute pixel centers");
   this->pixel_x = vgrf(glsl_type::uint_type);
   this->pixel_y = vgrf(glsl_type::uint_type);
   this->pixel_x.type = BRW_REGISTER_TYPE_UW;
   this->pixel_y.type = BRW_REGISTER_TYPE_UW;
   abld.ADD(this->pixel_x,
            fs_reg(stride(suboffset(g1_uw, 4), 2, 4, 0)),
            fs_reg(brw_imm_v(0x10101010)));
   abld.ADD(this->pixel_y,
            fs_reg(stride(suboffset(g1_uw, 5), 2, 4, 0)),
            fs_reg(brw_imm_v(0x11001100)));

   abld = bld.annotate("compute pixel deltas from v0");

   this->delta_xy[BRW_WM_PERSPECTIVE_PIXEL_BARYCENTRIC] =
      vgrf(glsl_type::vec2_type);
   const fs_reg &delta_xy = this->delta_xy[BRW_WM_PERSPECTIVE_PIXEL_BARYCENTRIC];
   const fs_reg xstart(negate(brw_vec1_grf(1, 0)));
   const fs_reg ystart(negate(brw_vec1_grf(1, 1)));

   if (devinfo->has_pln && dispatch_width == 16) {
      for (unsigned i = 0; i < 2; i++) {
         abld.half(i).ADD(half(offset(delta_xy, abld, i), 0),
                          half(this->pixel_x, i), xstart);
         abld.half(i).ADD(half(offset(delta_xy, abld, i), 1),
                          half(this->pixel_y, i), ystart);
      }
   } else {
      abld.ADD(offset(delta_xy, abld, 0), this->pixel_x, xstart);
      abld.ADD(offset(delta_xy, abld, 1), this->pixel_y, ystart);
   }

   abld = bld.annotate("compute pos.w and 1/pos.w");
   /* Compute wpos.w.  It's always in our setup, since it's needed to
    * interpolate the other attributes.
    */
   this->wpos_w = vgrf(glsl_type::float_type);
   abld.emit(FS_OPCODE_LINTERP, wpos_w, delta_xy,
             interp_reg(VARYING_SLOT_POS, 3));
   /* Compute the pixel 1/W value from wpos.w. */
   this->pixel_w = vgrf(glsl_type::float_type);
   abld.emit(SHADER_OPCODE_RCP, this->pixel_w, wpos_w);
}

/** Emits the interpolation for the varying inputs. */
void
fs_visitor::emit_interpolation_setup_gen6()
{
   struct brw_reg g1_uw = retype(brw_vec1_grf(1, 0), BRW_REGISTER_TYPE_UW);

   fs_builder abld = bld.annotate("compute pixel centers");
   if (devinfo->gen >= 8 || dispatch_width == 8) {
      /* The "Register Region Restrictions" page says for BDW (and newer,
       * presumably):
       *
       *     "When destination spans two registers, the source may be one or
       *      two registers. The destination elements must be evenly split
       *      between the two registers."
       *
       * Thus we can do a single add(16) in SIMD8 or an add(32) in SIMD16 to
       * compute our pixel centers.
       */
      fs_reg int_pixel_xy(GRF, alloc.allocate(dispatch_width / 8),
                          BRW_REGISTER_TYPE_UW);

      const fs_builder dbld = abld.exec_all().group(dispatch_width * 2, 0);
      dbld.ADD(int_pixel_xy,
               fs_reg(stride(suboffset(g1_uw, 4), 1, 4, 0)),
               fs_reg(brw_imm_v(0x11001010)));

      this->pixel_x = vgrf(glsl_type::float_type);
      this->pixel_y = vgrf(glsl_type::float_type);
      abld.emit(FS_OPCODE_PIXEL_X, this->pixel_x, int_pixel_xy);
      abld.emit(FS_OPCODE_PIXEL_Y, this->pixel_y, int_pixel_xy);
   } else {
      /* The "Register Region Restrictions" page says for SNB, IVB, HSW:
       *
       *     "When destination spans two registers, the source MUST span two
       *      registers."
       *
       * Since the GRF source of the ADD will only read a single register, we
       * must do two separate ADDs in SIMD16.
       */
      fs_reg int_pixel_x = vgrf(glsl_type::uint_type);
      fs_reg int_pixel_y = vgrf(glsl_type::uint_type);
      int_pixel_x.type = BRW_REGISTER_TYPE_UW;
      int_pixel_y.type = BRW_REGISTER_TYPE_UW;
      abld.ADD(int_pixel_x,
               fs_reg(stride(suboffset(g1_uw, 4), 2, 4, 0)),
               fs_reg(brw_imm_v(0x10101010)));
      abld.ADD(int_pixel_y,
               fs_reg(stride(suboffset(g1_uw, 5), 2, 4, 0)),
               fs_reg(brw_imm_v(0x11001100)));

      /* As of gen6, we can no longer mix float and int sources.  We have
       * to turn the integer pixel centers into floats for their actual
       * use.
       */
      this->pixel_x = vgrf(glsl_type::float_type);
      this->pixel_y = vgrf(glsl_type::float_type);
      abld.MOV(this->pixel_x, int_pixel_x);
      abld.MOV(this->pixel_y, int_pixel_y);
   }

   abld = bld.annotate("compute pos.w");
   this->pixel_w = fs_reg(brw_vec8_grf(payload.source_w_reg, 0));
   this->wpos_w = vgrf(glsl_type::float_type);
   abld.emit(SHADER_OPCODE_RCP, this->wpos_w, this->pixel_w);

   for (int i = 0; i < BRW_WM_BARYCENTRIC_INTERP_MODE_COUNT; ++i) {
      uint8_t reg = payload.barycentric_coord_reg[i];
      this->delta_xy[i] = fs_reg(brw_vec16_grf(reg, 0));
   }
}

void
fs_visitor::setup_color_payload(fs_reg *dst, fs_reg color, unsigned components,
                                unsigned exec_size, bool use_2nd_half)
{
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   fs_inst *inst;

   if (key->clamp_fragment_color) {
      fs_reg tmp = vgrf(glsl_type::vec4_type);
      assert(color.type == BRW_REGISTER_TYPE_F);
      for (unsigned i = 0; i < components; i++) {
         inst = bld.MOV(offset(tmp, bld, i), offset(color, bld, i));
         inst->saturate = true;
      }
      color = tmp;
   }

   if (exec_size < dispatch_width) {
      unsigned half_idx = use_2nd_half ? 1 : 0;
      for (unsigned i = 0; i < components; i++)
         dst[i] = half(offset(color, bld, i), half_idx);
   } else {
      for (unsigned i = 0; i < components; i++)
         dst[i] = offset(color, bld, i);
   }
}

static enum brw_conditional_mod
cond_for_alpha_func(GLenum func)
{
   switch(func) {
      case GL_GREATER:
         return BRW_CONDITIONAL_G;
      case GL_GEQUAL:
         return BRW_CONDITIONAL_GE;
      case GL_LESS:
         return BRW_CONDITIONAL_L;
      case GL_LEQUAL:
         return BRW_CONDITIONAL_LE;
      case GL_EQUAL:
         return BRW_CONDITIONAL_EQ;
      case GL_NOTEQUAL:
         return BRW_CONDITIONAL_NEQ;
      default:
         unreachable("Not reached");
   }
}

/**
 * Alpha test support for when we compile it into the shader instead
 * of using the normal fixed-function alpha test.
 */
void
fs_visitor::emit_alpha_test()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   const fs_builder abld = bld.annotate("Alpha test");

   fs_inst *cmp;
   if (key->alpha_test_func == GL_ALWAYS)
      return;

   if (key->alpha_test_func == GL_NEVER) {
      /* f0.1 = 0 */
      fs_reg some_reg = fs_reg(retype(brw_vec8_grf(0, 0),
                                      BRW_REGISTER_TYPE_UW));
      cmp = abld.CMP(bld.null_reg_f(), some_reg, some_reg,
                     BRW_CONDITIONAL_NEQ);
   } else {
      /* RT0 alpha */
      fs_reg color = offset(outputs[0], bld, 3);

      /* f0.1 &= func(color, ref) */
      cmp = abld.CMP(bld.null_reg_f(), color, fs_reg(key->alpha_test_ref),
                     cond_for_alpha_func(key->alpha_test_func));
   }
   cmp->predicate = BRW_PREDICATE_NORMAL;
   cmp->flag_subreg = 1;
}

fs_inst *
fs_visitor::emit_single_fb_write(const fs_builder &bld,
                                 fs_reg color0, fs_reg color1,
                                 fs_reg src0_alpha, unsigned components,
                                 unsigned exec_size, bool use_2nd_half)
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_data *prog_data = (brw_wm_prog_data*) this->prog_data;
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   int header_size = 2, payload_header_size;

   /* We can potentially have a message length of up to 15, so we have to set
    * base_mrf to either 0 or 1 in order to fit in m0..m15.
    */
   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, 15);
   int length = 0;

   /* From the Sandy Bridge PRM, volume 4, page 198:
    *
    *     "Dispatched Pixel Enables. One bit per pixel indicating
    *      which pixels were originally enabled when the thread was
    *      dispatched. This field is only required for the end-of-
    *      thread message and on all dual-source messages."
    */
   if (devinfo->gen >= 6 &&
       (devinfo->is_haswell || devinfo->gen >= 8 || !prog_data->uses_kill) &&
       color1.file == BAD_FILE &&
       key->nr_color_regions == 1) {
      header_size = 0;
   }

   if (header_size != 0) {
      assert(header_size == 2);
      /* Allocate 2 registers for a header */
      length += 2;
   }

   if (payload.aa_dest_stencil_reg) {
      sources[length] = fs_reg(GRF, alloc.allocate(1));
      bld.group(8, 0).exec_all().annotate("FB write stencil/AA alpha")
         .MOV(sources[length],
              fs_reg(brw_vec8_grf(payload.aa_dest_stencil_reg, 0)));
      length++;
   }

   prog_data->uses_omask =
      prog->OutputsWritten & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
   if (prog_data->uses_omask) {
      assert(this->sample_mask.file != BAD_FILE);
      /* Hand over gl_SampleMask. Only lower 16 bits are relevant.  Since
       * it's unsinged single words, one vgrf is always 16-wide.
       */
      sources[length] = fs_reg(GRF, alloc.allocate(1),
                               BRW_REGISTER_TYPE_UW);
      bld.exec_all().annotate("FB write oMask")
         .emit(FS_OPCODE_SET_OMASK, sources[length], this->sample_mask);
      length++;
   }

   payload_header_size = length;

   if (color0.file == BAD_FILE) {
      /* Even if there's no color buffers enabled, we still need to send
       * alpha out the pipeline to our null renderbuffer to support
       * alpha-testing, alpha-to-coverage, and so on.
       */
      if (this->outputs[0].file != BAD_FILE)
         setup_color_payload(&sources[length + 3],
                             offset(this->outputs[0], bld, 3),
                             1, exec_size, false);
      length += 4;
   } else if (color1.file == BAD_FILE) {
      if (src0_alpha.file != BAD_FILE) {
         setup_color_payload(&sources[length], src0_alpha, 1, exec_size, false);
         length++;
      }

      setup_color_payload(&sources[length], color0, components,
                          exec_size, use_2nd_half);
      length += 4;
   } else {
      setup_color_payload(&sources[length], color0, components,
                          exec_size, use_2nd_half);
      length += 4;
      setup_color_payload(&sources[length], color1, components,
                          exec_size, use_2nd_half);
      length += 4;
   }

   if (source_depth_to_render_target) {
      if (devinfo->gen == 6) {
	 /* For outputting oDepth on gen6, SIMD8 writes have to be
	  * used.  This would require SIMD8 moves of each half to
	  * message regs, kind of like pre-gen5 SIMD16 FB writes.
	  * Just bail on doing so for now.
	  */
	 no16("Missing support for simd16 depth writes on gen6\n");
      }

      if (prog->OutputsWritten & BITFIELD64_BIT(FRAG_RESULT_DEPTH)) {
	 /* Hand over gl_FragDepth. */
	 assert(this->frag_depth.file != BAD_FILE);
         if (exec_size < dispatch_width) {
            sources[length] = half(this->frag_depth, use_2nd_half);
         } else {
            sources[length] = this->frag_depth;
         }
      } else {
	 /* Pass through the payload depth. */
         sources[length] = fs_reg(brw_vec8_grf(payload.source_depth_reg, 0));
      }
      length++;
   }

   if (payload.dest_depth_reg)
      sources[length++] = fs_reg(brw_vec8_grf(payload.dest_depth_reg, 0));

   const fs_builder ubld = bld.group(exec_size, use_2nd_half);
   fs_inst *load;
   fs_inst *write;
   if (devinfo->gen >= 7) {
      /* Send from the GRF */
      fs_reg payload = fs_reg(GRF, -1, BRW_REGISTER_TYPE_F);
      load = ubld.LOAD_PAYLOAD(payload, sources, length, payload_header_size);
      payload.reg = alloc.allocate(load->regs_written);
      load->dst = payload;
      write = ubld.emit(FS_OPCODE_FB_WRITE, reg_undef, payload);
      write->base_mrf = -1;
   } else {
      /* Send from the MRF */
      load = ubld.LOAD_PAYLOAD(fs_reg(MRF, 1, BRW_REGISTER_TYPE_F),
                               sources, length, payload_header_size);

      /* On pre-SNB, we have to interlace the color values.  LOAD_PAYLOAD
       * will do this for us if we just give it a COMPR4 destination.
       */
      if (devinfo->gen < 6 && exec_size == 16)
         load->dst.reg |= BRW_MRF_COMPR4;

      write = ubld.emit(FS_OPCODE_FB_WRITE);
      write->exec_size = exec_size;
      write->base_mrf = 1;
   }

   write->mlen = load->regs_written;
   write->header_size = header_size;
   if (prog_data->uses_kill) {
      write->predicate = BRW_PREDICATE_NORMAL;
      write->flag_subreg = 1;
   }
   return write;
}

void
fs_visitor::emit_fb_writes()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_data *prog_data = (brw_wm_prog_data*) this->prog_data;
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;

   fs_inst *inst = NULL;
   if (do_dual_src) {
      const fs_builder abld = bld.annotate("FB dual-source write");

      inst = emit_single_fb_write(abld, this->outputs[0],
                                  this->dual_src_output, reg_undef, 4, 8);
      inst->target = 0;

      /* SIMD16 dual source blending requires to send two SIMD8 dual source
       * messages, where each message contains color data for 8 pixels. Color
       * data for the first group of pixels is stored in the "lower" half of
       * the color registers, so in SIMD16, the previous message did:
       * m + 0: r0
       * m + 1: g0
       * m + 2: b0
       * m + 3: a0
       *
       * Here goes the second message, which packs color data for the
       * remaining 8 pixels. Color data for these pixels is stored in the
       * "upper" half of the color registers, so we need to do:
       * m + 0: r1
       * m + 1: g1
       * m + 2: b1
       * m + 3: a1
       */
      if (dispatch_width == 16) {
         inst = emit_single_fb_write(abld, this->outputs[0],
                                     this->dual_src_output, reg_undef, 4, 8,
                                     true);
         inst->target = 0;
      }

      prog_data->dual_src_blend = true;
   } else {
      for (int target = 0; target < key->nr_color_regions; target++) {
         /* Skip over outputs that weren't written. */
         if (this->outputs[target].file == BAD_FILE)
            continue;

         const fs_builder abld = bld.annotate(
            ralloc_asprintf(this->mem_ctx, "FB write target %d", target));

         fs_reg src0_alpha;
         if (devinfo->gen >= 6 && key->replicate_alpha && target != 0)
            src0_alpha = offset(outputs[0], bld, 3);

         inst = emit_single_fb_write(abld, this->outputs[target], reg_undef,
                                     src0_alpha,
                                     this->output_components[target],
                                     dispatch_width);
         inst->target = target;
      }
   }

   if (inst == NULL) {
      /* Even if there's no color buffers enabled, we still need to send
       * alpha out the pipeline to our null renderbuffer to support
       * alpha-testing, alpha-to-coverage, and so on.
       */
      inst = emit_single_fb_write(bld, reg_undef, reg_undef, reg_undef, 0,
                                  dispatch_width);
      inst->target = 0;
   }

   inst->eot = true;
}

void
fs_visitor::setup_uniform_clipplane_values(gl_clip_plane *clip_planes)
{
   const struct brw_vue_prog_key *key =
      (const struct brw_vue_prog_key *) this->key;

   for (int i = 0; i < key->nr_userclip_plane_consts; i++) {
      this->userplane[i] = fs_reg(UNIFORM, uniforms);
      for (int j = 0; j < 4; ++j) {
         stage_prog_data->param[uniforms + j] =
            (gl_constant_value *) &clip_planes[i][j];
      }
      uniforms += 4;
   }
}

/**
 * Lower legacy fixed-function and gl_ClipVertex clipping to clip distances.
 *
 * This does nothing if the shader uses gl_ClipDistance or user clipping is
 * disabled altogether.
 */
void fs_visitor::compute_clip_distance(gl_clip_plane *clip_planes)
{
   struct brw_vue_prog_data *vue_prog_data =
      (struct brw_vue_prog_data *) prog_data;
   const struct brw_vue_prog_key *key =
      (const struct brw_vue_prog_key *) this->key;

   /* Bail unless some sort of legacy clipping is enabled */
   if (!key->userclip_active || prog->UsesClipDistanceOut)
      return;

   /* From the GLSL 1.30 spec, section 7.1 (Vertex Shader Special Variables):
    *
    *     "If a linked set of shaders forming the vertex stage contains no
    *     static write to gl_ClipVertex or gl_ClipDistance, but the
    *     application has requested clipping against user clip planes through
    *     the API, then the coordinate written to gl_Position is used for
    *     comparison against the user clip planes."
    *
    * This function is only called if the shader didn't write to
    * gl_ClipDistance.  Accordingly, we use gl_ClipVertex to perform clipping
    * if the user wrote to it; otherwise we use gl_Position.
    */

   gl_varying_slot clip_vertex = VARYING_SLOT_CLIP_VERTEX;
   if (!(vue_prog_data->vue_map.slots_valid & VARYING_BIT_CLIP_VERTEX))
      clip_vertex = VARYING_SLOT_POS;

   /* If the clip vertex isn't written, skip this.  Typically this means
    * the GS will set up clipping. */
   if (outputs[clip_vertex].file == BAD_FILE)
      return;

   setup_uniform_clipplane_values(clip_planes);

   const fs_builder abld = bld.annotate("user clip distances");

   this->outputs[VARYING_SLOT_CLIP_DIST0] = vgrf(glsl_type::vec4_type);
   this->outputs[VARYING_SLOT_CLIP_DIST1] = vgrf(glsl_type::vec4_type);

   for (int i = 0; i < key->nr_userclip_plane_consts; i++) {
      fs_reg u = userplane[i];
      fs_reg output = outputs[VARYING_SLOT_CLIP_DIST0 + i / 4];
      output.reg_offset = i & 3;

      abld.MUL(output, outputs[clip_vertex], u);
      for (int j = 1; j < 4; j++) {
         u.reg = userplane[i].reg + j;
         abld.MAD(output, output, offset(outputs[clip_vertex], bld, j), u);
      }
   }
}

void
fs_visitor::emit_urb_writes()
{
   int slot, urb_offset, length;
   struct brw_vs_prog_data *vs_prog_data =
      (struct brw_vs_prog_data *) prog_data;
   const struct brw_vs_prog_key *key =
      (const struct brw_vs_prog_key *) this->key;
   const GLbitfield64 psiz_mask =
      VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT | VARYING_BIT_PSIZ;
   const struct brw_vue_map *vue_map = &vs_prog_data->base.vue_map;
   bool flush;
   fs_reg sources[8];

   /* If we don't have any valid slots to write, just do a minimal urb write
    * send to terminate the shader.  This includes 1 slot of undefined data,
    * because it's invalid to write 0 data:
    *
    * From the Broadwell PRM, Volume 7: 3D Media GPGPU, Shared Functions -
    * Unified Return Buffer (URB) > URB_SIMD8_Write and URB_SIMD8_Read >
    * Write Data Payload:
    *
    *    "The write data payload can be between 1 and 8 message phases long."
    */
   if (vue_map->slots_valid == 0) {
      fs_reg payload = fs_reg(GRF, alloc.allocate(2), BRW_REGISTER_TYPE_UD);
      bld.exec_all().MOV(payload, fs_reg(retype(brw_vec8_grf(1, 0),
                                                BRW_REGISTER_TYPE_UD)));

      fs_inst *inst = bld.emit(SHADER_OPCODE_URB_WRITE_SIMD8, reg_undef, payload);
      inst->eot = true;
      inst->mlen = 2;
      inst->offset = 1;
      return;
   }

   length = 0;
   urb_offset = 0;
   flush = false;
   for (slot = 0; slot < vue_map->num_slots; slot++) {
      fs_reg reg, src, zero;

      int varying = vue_map->slot_to_varying[slot];
      switch (varying) {
      case VARYING_SLOT_PSIZ:

         /* The point size varying slot is the vue header and is always in the
          * vue map.  But often none of the special varyings that live there
          * are written and in that case we can skip writing to the vue
          * header, provided the corresponding state properly clamps the
          * values further down the pipeline. */
         if ((vue_map->slots_valid & psiz_mask) == 0) {
            assert(length == 0);
            urb_offset++;
            break;
         }

         zero = fs_reg(GRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
         bld.MOV(zero, fs_reg(0u));

         sources[length++] = zero;
         if (vue_map->slots_valid & VARYING_BIT_LAYER)
            sources[length++] = this->outputs[VARYING_SLOT_LAYER];
         else
            sources[length++] = zero;

         if (vue_map->slots_valid & VARYING_BIT_VIEWPORT)
            sources[length++] = this->outputs[VARYING_SLOT_VIEWPORT];
         else
            sources[length++] = zero;

         if (vue_map->slots_valid & VARYING_BIT_PSIZ)
            sources[length++] = this->outputs[VARYING_SLOT_PSIZ];
         else
            sources[length++] = zero;
         break;

      case BRW_VARYING_SLOT_NDC:
      case VARYING_SLOT_EDGE:
         unreachable("unexpected scalar vs output");
         break;

      case BRW_VARYING_SLOT_PAD:
         break;

      default:
         /* gl_Position is always in the vue map, but isn't always written by
          * the shader.  Other varyings (clip distances) get added to the vue
          * map but don't always get written.  In those cases, the
          * corresponding this->output[] slot will be invalid we and can skip
          * the urb write for the varying.  If we've already queued up a vue
          * slot for writing we flush a mlen 5 urb write, otherwise we just
          * advance the urb_offset.
          */
         if (this->outputs[varying].file == BAD_FILE) {
            if (length > 0)
               flush = true;
            else
               urb_offset++;
            break;
         }

         if ((varying == VARYING_SLOT_COL0 ||
              varying == VARYING_SLOT_COL1 ||
              varying == VARYING_SLOT_BFC0 ||
              varying == VARYING_SLOT_BFC1) &&
             key->clamp_vertex_color) {
            /* We need to clamp these guys, so do a saturating MOV into a
             * temp register and use that for the payload.
             */
            for (int i = 0; i < 4; i++) {
               reg = fs_reg(GRF, alloc.allocate(1), outputs[varying].type);
               src = offset(this->outputs[varying], bld, i);
               set_saturate(true, bld.MOV(reg, src));
               sources[length++] = reg;
            }
         } else {
            for (int i = 0; i < 4; i++)
               sources[length++] = offset(this->outputs[varying], bld, i);
         }
         break;
      }

      const fs_builder abld = bld.annotate("URB write");

      /* If we've queued up 8 registers of payload (2 VUE slots), if this is
       * the last slot or if we need to flush (see BAD_FILE varying case
       * above), emit a URB write send now to flush out the data.
       */
      int last = slot == vue_map->num_slots - 1;
      if (length == 8 || last)
         flush = true;
      if (flush) {
         fs_reg *payload_sources = ralloc_array(mem_ctx, fs_reg, length + 1);
         fs_reg payload = fs_reg(GRF, alloc.allocate(length + 1),
                                 BRW_REGISTER_TYPE_F);
         payload_sources[0] =
            fs_reg(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD));

         memcpy(&payload_sources[1], sources, length * sizeof sources[0]);
         abld.LOAD_PAYLOAD(payload, payload_sources, length + 1, 1);

         fs_inst *inst =
            abld.emit(SHADER_OPCODE_URB_WRITE_SIMD8, reg_undef, payload);
         inst->eot = last;
         inst->mlen = length + 1;
         inst->offset = urb_offset;
         urb_offset = slot + 1;
         length = 0;
         flush = false;
      }
   }
}

void
fs_visitor::emit_cs_terminate()
{
   assert(devinfo->gen >= 7);

   /* We are getting the thread ID from the compute shader header */
   assert(stage == MESA_SHADER_COMPUTE);

   /* We can't directly send from g0, since sends with EOT have to use
    * g112-127. So, copy it to a virtual register, The register allocator will
    * make sure it uses the appropriate register range.
    */
   struct brw_reg g0 = retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD);
   fs_reg payload = fs_reg(GRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
   bld.exec_all().MOV(payload, g0);

   /* Send a message to the thread spawner to terminate the thread. */
   fs_inst *inst = bld.exec_all()
                      .emit(CS_OPCODE_CS_TERMINATE, reg_undef, payload);
   inst->eot = true;
}

void
fs_visitor::emit_barrier()
{
   assert(devinfo->gen >= 7);

   /* We are getting the barrier ID from the compute shader header */
   assert(stage == MESA_SHADER_COMPUTE);

   fs_reg payload = fs_reg(GRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);

   /* Clear the message payload */
   bld.exec_all().MOV(payload, fs_reg(0u));

   /* Copy bits 27:24 of r0.2 (barrier id) to the message payload reg.2 */
   fs_reg r0_2 = fs_reg(retype(brw_vec1_grf(0, 2), BRW_REGISTER_TYPE_UD));
   bld.exec_all().AND(component(payload, 2), r0_2, fs_reg(0x0f000000u));

   /* Emit a gateway "barrier" message using the payload we set up, followed
    * by a wait instruction.
    */
   bld.exec_all().emit(SHADER_OPCODE_BARRIER, reg_undef, payload);
}

fs_visitor::fs_visitor(const struct brw_compiler *compiler, void *log_data,
                       void *mem_ctx,
                       gl_shader_stage stage,
                       const void *key,
                       struct brw_stage_prog_data *prog_data,
                       struct gl_shader_program *shader_prog,
                       struct gl_program *prog,
                       unsigned dispatch_width,
                       int shader_time_index)
   : backend_shader(compiler, log_data, mem_ctx,
                    shader_prog, prog, prog_data, stage),
     key(key), prog_data(prog_data),
     dispatch_width(dispatch_width),
     shader_time_index(shader_time_index),
     promoted_constants(0),
     bld(fs_builder(this, dispatch_width).at_end())
{
   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      key_tex = &((const brw_wm_prog_key *) key)->tex;
      break;
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_GEOMETRY:
      key_tex = &((const brw_vue_prog_key *) key)->tex;
      break;
   case MESA_SHADER_COMPUTE:
      key_tex = &((const brw_cs_prog_key*) key)->tex;
      break;
   default:
      unreachable("unhandled shader stage");
   }

   this->failed = false;
   this->simd16_unsupported = false;
   this->no16_msg = NULL;

   this->nir_locals = NULL;
   this->nir_ssa_values = NULL;
   this->nir_globals = NULL;

   memset(&this->payload, 0, sizeof(this->payload));
   memset(this->outputs, 0, sizeof(this->outputs));
   memset(this->output_components, 0, sizeof(this->output_components));
   this->source_depth_to_render_target = false;
   this->runtime_check_aads_emit = false;
   this->first_non_payload_grf = 0;
   this->max_grf = devinfo->gen >= 7 ? GEN7_MRF_HACK_START : BRW_MAX_GRF;

   this->virtual_grf_start = NULL;
   this->virtual_grf_end = NULL;
   this->live_intervals = NULL;
   this->regs_live_at_ip = NULL;

   this->uniforms = 0;
   this->last_scratch = 0;
   this->pull_constant_loc = NULL;
   this->push_constant_loc = NULL;

   this->spilled_any_registers = false;
   this->do_dual_src = false;

   if (dispatch_width == 8)
      this->param_size = rzalloc_array(mem_ctx, int, stage_prog_data->nr_params);
}

fs_visitor::~fs_visitor()
{
}
