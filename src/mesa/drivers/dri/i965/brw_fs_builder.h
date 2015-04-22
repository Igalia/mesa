/* -*- c++ -*- */
/*
 * Copyright © 2010-2015 Intel Corporation
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

#ifndef BRW_FS_BUILDER_H
#define BRW_FS_BUILDER_H

#include "brw_ir_fs.h"
#include "brw_ir_allocator.h"
#include "brw_device_info.h"

namespace brw {
   /**
    * Toolbox to assemble an FS IR program out of individual instructions.
    *
    * This object is meant to have an interface consistent with
    * brw::vec4_builder.  They cannot be fully interchangeable because
    * brw::fs_builder generates scalar code while brw::vec4_builder generates
    * vector code.
    */
   class fs_builder {
   public:
      /** Type used in this IR to represent a source of an instruction. */
      typedef fs_reg src_reg;

      /** Type used in this IR to represent the destination of an instruction. */
      typedef fs_reg dst_reg;

      /** Type used in this IR to represent an instruction. */
      typedef fs_inst instruction;

      /**
       * Construct an fs_builder appending instructions at the end of the list
       * \p instructions.  \p alloc provides book-keeping of virtual registers
       * allocated through the builder, \p dispatch_width, \p stage and \p
       * uses_kill are required because they may have an effect on code
       * generation.
       */
      fs_builder(const brw_device_info *devinfo,
                 void *mem_ctx,
                 simple_allocator &alloc,
                 exec_list &instructions,
                 unsigned dispatch_width,
                 gl_shader_stage stage,
                 bool uses_kill) :
         devinfo(devinfo), mem_ctx(mem_ctx),
         alloc(&alloc), block(NULL),
         cursor((exec_node *)&instructions.tail),
         native_width(dispatch_width),
         _half(0), force_uncompressed(false),
         stage(stage), uses_kill(uses_kill)
      {
      }

      /**
       * Construct an fs_builder that inserts instructions before \p cursor in
       * basic block \p block, inheriting other code generation parameters
       * from this.
       */
      fs_builder
      at(bblock_t *block, instruction *cursor) const
      {
         fs_builder bld = *this;
         bld.block = block;
         bld.cursor = cursor;
         return bld;
      }

      /**
       * Construct a builder of half-SIMD-width instructions inheriting other
       * code generation parameters from this.  Predication and control flow
       * masking will use the enable signals for the i-th half.
       */
      fs_builder
      half(unsigned i) const
      {
         fs_builder bld = *this;
         bld.force_uncompressed = true;
         bld._half = i;
         return bld;
      }

      /**
       * Get the SIMD width in use.
       */
      unsigned
      dispatch_width() const
      {
         return (force_uncompressed ? 8 : native_width);
      }

      /**
       * Allocate a virtual register of natural vector size (one for this IR)
       * and SIMD width.  \p n gives the amount of space to allocate in
       * dispatch_width units (which is just enough space for one logical
       * component in this IR).
       */
      dst_reg
      vgrf(enum brw_reg_type type, unsigned n = 1) const
      {
         return dst_reg(GRF, alloc->allocate(
                           DIV_ROUND_UP(n * type_sz(type) * dispatch_width(),
                                        REG_SIZE)),
                        type, dispatch_width());
      }

      /**
       * Create a null register of floating type.
       */
      dst_reg
      null_reg_f() const
      {
         return dst_reg(retype(brw_null_vec(dispatch_width()),
                               BRW_REGISTER_TYPE_F));
      }

      /**
       * Create a null register of signed integer type.
       */
      dst_reg
      null_reg_d() const
      {
         return dst_reg(retype(brw_null_vec(dispatch_width()),
                               BRW_REGISTER_TYPE_D));
      }

      /**
       * Create a null register of unsigned integer type.
       */
      dst_reg
      null_reg_ud() const
      {
         return dst_reg(retype(brw_null_vec(dispatch_width()),
                               BRW_REGISTER_TYPE_UD));
      }

      /**
       * Get the mask of SIMD channels enabled by dispatch and not yet
       * disabled by discard.
       */
      src_reg
      sample_mask_reg() const
      {
         return (stage != MESA_SHADER_FRAGMENT ? src_reg(0xffff) :
                 uses_kill ? brw_flag_reg(0, 1) :
                 retype(brw_vec1_grf(1, 7), BRW_REGISTER_TYPE_UD));
      }

      /**
       * Insert an instruction into the program.
       */
      instruction *
      emit(const instruction &inst) const
      {
         return emit(new(mem_ctx) instruction(inst));
      }

      /**
       * Create and insert a nullary control instruction into the program.
       */
      instruction *
      emit(enum opcode opcode) const
      {
         return emit(instruction(opcode, dispatch_width()));
      }

      /**
       * Create and insert a nullary instruction into the program.
       */
      instruction *
      emit(enum opcode opcode, const dst_reg &dst) const
      {
         return emit(instruction(opcode, dst));
      }

      /**
       * Create and insert a unary instruction into the program.
       */
      instruction *
      emit(enum opcode opcode, const dst_reg &dst, const src_reg &src0) const
      {
         switch (opcode) {
         case SHADER_OPCODE_RCP:
         case SHADER_OPCODE_RSQ:
         case SHADER_OPCODE_SQRT:
         case SHADER_OPCODE_EXP2:
         case SHADER_OPCODE_LOG2:
         case SHADER_OPCODE_SIN:
         case SHADER_OPCODE_COS:
            return fix_math_instruction(
               emit(instruction(opcode, dst.width, dst,
                                fix_math_operand(src0))));

         default:
            return emit(instruction(opcode, dst.width, dst, src0));
         }
      }

      /**
       * Create and insert a binary instruction into the program.
       */
      instruction *
      emit(enum opcode opcode, const dst_reg &dst, const src_reg &src0,
           const src_reg &src1) const
      {
         switch (opcode) {
         case SHADER_OPCODE_POW:
         case SHADER_OPCODE_INT_QUOTIENT:
         case SHADER_OPCODE_INT_REMAINDER:
            return fix_math_instruction(
               emit(instruction(opcode, dst.width, dst,
                                fix_math_operand(src0),
                                fix_math_operand(src1))));

         default:
            return emit(instruction(opcode, dst.width, dst, src0, src1));

         }
      }

      /**
       * Create and insert a ternary instruction into the program.
       */
      instruction *
      emit(enum opcode opcode, const dst_reg &dst, const src_reg &src0,
           const src_reg &src1, const src_reg &src2) const
      {
         switch (opcode) {
         case BRW_OPCODE_BFE:
         case BRW_OPCODE_BFI2:
         case BRW_OPCODE_MAD:
         case BRW_OPCODE_LRP:
            return emit(instruction(opcode, dst.width, dst,
                                    fix_3src_operand(src0),
                                    fix_3src_operand(src1),
                                    fix_3src_operand(src2)));

         default:
            return emit(instruction(opcode, dst.width, dst, src0, src1, src2));
         }
      }

      /**
       * Insert a preallocated instruction into the program.
       */
      instruction *
      emit(instruction *inst) const
      {
         assert(inst->dst.width <= dispatch_width());

         inst->force_uncompressed = (native_width == 16 &&
                                     inst->exec_size <= 8);
         inst->force_sechalf = (_half == 1);
         inst->annotation = current_annotation;
         inst->ir = base_ir;

         if (block)
            static_cast<instruction *>(cursor)->insert_before(block, inst);
         else
            cursor->insert_before(inst);

         return inst;
      }

      /**
       * Select \p src0 if the comparison of both sources with the given
       * conditional mod evaluates to true, otherwise select \p src1.
       *
       * Generally useful to get the minimum or maximum of two values.
       */
      void
      emit_minmax(const dst_reg &dst, const src_reg &src0,
                  const src_reg &src1, brw_conditional_mod mod) const
      {
         if (devinfo->gen >= 6) {
            exec_condmod(mod, SEL(dst, fix_unsigned_negate(src0),
                                  fix_unsigned_negate(src1)));
         } else {
            CMP(null_reg_d(), src0, src1, mod);
            exec_predicate(BRW_PREDICATE_NORMAL,
                           SEL(dst, src0, src1));
         }
      }

      /**
       * Copy any live channel from \p src to the first channel of \p dst.
       */
      void
      emit_uniformize(const dst_reg &dst, const src_reg &src) const
      {
         const dst_reg chan_index = vgrf(BRW_REGISTER_TYPE_UD);

         emit(SHADER_OPCODE_FIND_LIVE_CHANNEL, component(chan_index, 0))
            ->force_writemask_all = true;
         emit(SHADER_OPCODE_BROADCAST, component(dst, 0),
              src, component(chan_index, 0))
            ->force_writemask_all = true;
      }

      /**
       * Assorted arithmetic ops.
       * @{
       */
#define ALU1(op)                                        \
      instruction *                                     \
      op(const dst_reg &dst, const src_reg &src0) const \
      {                                                 \
         return emit(BRW_OPCODE_##op, dst, src0);       \
      }

#define ALU2(op)                                                        \
      instruction *                                                     \
      op(const dst_reg &dst, const src_reg &src0, const src_reg &src1) const \
      {                                                                 \
         return emit(BRW_OPCODE_##op, dst, src0, src1);                 \
      }

#define ALU2_ACC(op)                                                    \
      instruction *                                                     \
      op(const dst_reg &dst, const src_reg &src0, const src_reg &src1) const \
      {                                                                 \
         instruction *inst = emit(BRW_OPCODE_##op, dst, src0, src1);    \
         inst->writes_accumulator = true;                               \
         return inst;                                                   \
      }

#define ALU3(op)                                                        \
      instruction *                                                     \
      op(const dst_reg &dst, const src_reg &src0, const src_reg &src1,  \
         const src_reg &src2) const                                     \
      {                                                                 \
         return emit(BRW_OPCODE_##op, dst, src0, src1, src2);           \
      }

      ALU2(ADD)
      ALU2_ACC(ADDC)
      ALU2(AND)
      ALU2(ASR)
      ALU2(AVG)
      ALU3(BFE)
      ALU2(BFI1)
      ALU3(BFI2)
      ALU1(BFREV)
      ALU1(CBIT)
      ALU2(CMPN)
      ALU3(CSEL)
      ALU2(DP2)
      ALU2(DP3)
      ALU2(DP4)
      ALU2(DPH)
      ALU1(F16TO32)
      ALU1(F32TO16)
      ALU1(FBH)
      ALU1(FBL)
      ALU1(FRC)
      ALU2(LINE)
      ALU1(LZD)
      ALU2(MAC)
      ALU2_ACC(MACH)
      ALU3(MAD)
      ALU1(MOV)
      ALU2(MUL)
      ALU1(NOT)
      ALU2(OR)
      ALU2(PLN)
      ALU1(RNDD)
      ALU1(RNDE)
      ALU1(RNDU)
      ALU1(RNDZ)
      ALU2(SAD2)
      ALU2_ACC(SADA2)
      ALU2(SEL)
      ALU2(SHL)
      ALU2(SHR)
      ALU2_ACC(SUBB)
      ALU2(XOR)

#undef ALU3
#undef ALU2_ACC
#undef ALU2
#undef ALU1
      /** @} */

      /**
       * CMP: Sets the low bit of the destination channels with the result
       * of the comparison, while the upper bits are undefined, and updates
       * the flag register with the packed 16 bits of the result.
       */
      instruction *
      CMP(dst_reg dst, const src_reg &src0, const src_reg &src1,
          brw_conditional_mod condition) const
      {
         /* Take the instruction:
          *
          * CMP null<d> src0<f> src1<f>
          *
          * Original gen4 does type conversion to the destination type before
          * comparison, producing garbage results for floating point comparisons.
          * gen5 does the comparison on the execution type (resolved source types),
          * so dst type doesn't matter.  gen6 does comparison and then uses the
          * result as if it was the dst type with no conversion, which happens to
          * mostly work out for float-interpreted-as-int since our comparisons are
          * for >0, =0, <0.
          */
         if (devinfo->gen == 4)
            dst = retype(dst, src0.type);

         return exec_condmod(condition,
                             emit(BRW_OPCODE_CMP, dst,
                                  fix_unsigned_negate(src0),
                                  fix_unsigned_negate(src1)));
      }

      /**
       * Gen4 predicated IF.
       */
      instruction *
      IF(brw_predicate predicate) const
      {
         instruction *inst = emit(BRW_OPCODE_IF);
         return exec_predicate(predicate, inst);
      }

      /**
       * Gen6 IF with embedded comparison.
       */
      instruction *
      IF(const src_reg &src0, const src_reg &src1,
         brw_conditional_mod condition) const
      {
         assert(devinfo->gen == 6);
         return exec_condmod(condition,
                             emit(BRW_OPCODE_IF,
                                  null_reg_d(),
                                  fix_unsigned_negate(src0),
                                  fix_unsigned_negate(src1)));
      }

      /**
       * Emit a linear interpolation instruction.
       */
      instruction *
      LRP(const dst_reg &dst, const src_reg &x, const src_reg &y,
          const src_reg &a) const
      {
         if (devinfo->gen >= 6) {
            /* The LRP instruction actually does op1 * op0 + op2 * (1 - op0), so
             * we need to reorder the operands.
             */
            return emit(BRW_OPCODE_LRP, dst, a, y, x);

         } else {
            /* We can't use the LRP instruction.  Emit x*(1-a) + y*a. */
            const dst_reg y_times_a = vgrf(dst.type);
            const dst_reg one_minus_a = vgrf(dst.type);
            const dst_reg x_times_one_minus_a = vgrf(dst.type);

            MUL(y_times_a, y, a);
            ADD(one_minus_a, negate(a), src_reg(1.0f));
            MUL(x_times_one_minus_a, x, src_reg(one_minus_a));
            return ADD(dst, src_reg(x_times_one_minus_a), src_reg(y_times_a));
         }
      }

      /**
       * Collect a number of registers in a contiguous range of registers.
       */
      instruction *
      LOAD_PAYLOAD(const dst_reg &dst, const src_reg *src,
                   unsigned sources, unsigned header_size) const
      {
         assert(dst.width % 8 == 0);
         instruction *inst = emit(instruction(SHADER_OPCODE_LOAD_PAYLOAD,
                                              dst.width, dst, src, sources));
         inst->header_size = header_size;

         for (unsigned i = 0; i < header_size; i++)
            assert(src[i].file != GRF ||
                   src[i].width * type_sz(src[i].type) == 32);
         inst->regs_written = header_size;

         for (unsigned i = header_size; i < sources; ++i)
            assert(src[i].file != GRF ||
                   src[i].width == dst.width);
         inst->regs_written += (sources - header_size) * (dst.width / 8);

         return inst;
      }

      /** @{ debug annotation info */
      void
      set_annotation(const char *s) {
         current_annotation = s;
      }

      void
      set_base_ir(const void *ir) {
         base_ir = ir;
      }
      /** @} */

      const brw_device_info *const devinfo;

   private:
      /**
       * Workaround for negation of UD registers.  See comment in
       * fs_generator::generate_code() for more details.
       */
      src_reg
      fix_unsigned_negate(const src_reg &src) const
      {
         if (src.type == BRW_REGISTER_TYPE_UD &&
             src.negate) {
            dst_reg temp = vgrf(BRW_REGISTER_TYPE_UD);
            MOV(temp, src);
            return src_reg(temp);
         } else {
            return src;
         }
      }

      /**
       * Workaround for source register modes not supported by the ternary
       * instruction encoding.
       */
      src_reg
      fix_3src_operand(const src_reg &src) const
      {
         if (src.file == GRF || src.file == UNIFORM || src.stride > 1) {
            return src;
         } else {
            dst_reg expanded = vgrf(src.type);
            MOV(expanded, src);
            return expanded;
         }
      }

      /**
       * Workaround for source register modes not supported by the math
       * instruction.
       */
      src_reg
      fix_math_operand(const src_reg &src) const
      {
         /* Can't do hstride == 0 args on gen6 math, so expand it out. We
          * might be able to do better by doing execsize = 1 math and then
          * expanding that result out, but we would need to be careful with
          * masking.
          *
          * Gen6 hardware ignores source modifiers (negate and abs) on math
          * instructions, so we also move to a temp to set those up.
          *
          * Gen7 relaxes most of the above restrictions, but still can't use IMM
          * operands to math
          */
         if ((devinfo->gen == 6 && (src.file == IMM || src.file == UNIFORM ||
                                    src.abs || src.negate)) ||
             (devinfo->gen == 7 && src.file == IMM)) {
            const dst_reg tmp = vgrf(src.type);
            MOV(tmp, src);
            return tmp;
         } else {
            return src;
         }
      }

      /**
       * Workaround other weirdness of the math instruction.
       */
      instruction *
      fix_math_instruction(instruction *instr) const
      {
         if (devinfo->gen < 6) {
            instr->base_mrf = 2;
            instr->mlen = instr->sources * dispatch_width() / 8;
         }

         return instr;
      }

      void *const mem_ctx;

      simple_allocator *alloc;
      bblock_t *block;
      exec_node *cursor;

      unsigned native_width;
      unsigned _half;
      bool force_uncompressed;
      gl_shader_stage stage;
      bool uses_kill;

      /** @{ debug annotation info */
      const char *current_annotation;
      const void *base_ir;
      /** @} */
   };
}

#endif
