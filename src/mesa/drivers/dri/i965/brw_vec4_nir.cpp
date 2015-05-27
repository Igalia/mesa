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

#include "glsl/ir.h"
#include "glsl/ir_optimization.h"
#include "glsl/nir/glsl_to_nir.h"
#include "brw_vec4.h"
#include "brw_nir.h"
#include "brw_vs.h"
#include "brw_fs.h"

namespace brw {

void
vec4_visitor::emit_nir_code()
{
   nir_shader *nir = prog->nir;

   nir_inputs = ralloc_array(mem_ctx, src_reg, nir->num_inputs);
   nir_setup_inputs(nir);

   nir_outputs = ralloc_array(mem_ctx, int, nir->num_outputs);
   nir_output_types = ralloc_array(mem_ctx, brw_reg_type, nir->num_outputs);
   nir_setup_outputs(nir);

   nir_emit_system_values(nir);

   if (nir->num_uniforms > 0)
      nir_setup_uniforms(nir);

   /* get the main function and emit it */
   nir_foreach_overload(nir, overload) {
      assert(strcmp(overload->function->name, "main") == 0);
      assert(overload->impl);
      nir_emit_impl(overload->impl);
   }
}

static bool
emit_system_values_block(nir_block *block, void *void_visitor)
{
   vec4_visitor *v = (vec4_visitor *)void_visitor;
   dst_reg *reg;

   nir_foreach_instr(block, instr) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      /* @FIXME: brw_fs_nir included asserts of being at vertex shader
       * stage. I assume that those would not be needed in this case. */
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_vertex_id:
         /* @FIXME: note that lower_vertex_id is an ir lowering. So when the
          * intermediate ir pass get removed, a equivalent lowering would be
          * needed on nir (probably on nir_lower_system_values.c) */
         unreachable("should be lowered by lower_vertex_id().");

      case nir_intrinsic_load_vertex_id_zero_base:
         reg = &v->nir_system_values[SYSTEM_VALUE_VERTEX_ID_ZERO_BASE];
         if (reg->file == BAD_FILE)
            *reg = *v->make_reg_for_system_value(SYSTEM_VALUE_VERTEX_ID_ZERO_BASE, NULL);
         break;

      case nir_intrinsic_load_base_vertex:
         reg = &v->nir_system_values[SYSTEM_VALUE_BASE_VERTEX];
         if (reg->file == BAD_FILE)
            *reg = *v->make_reg_for_system_value(SYSTEM_VALUE_BASE_VERTEX, NULL);
         break;

      case nir_intrinsic_load_instance_id:
         reg = &v->nir_system_values[SYSTEM_VALUE_INSTANCE_ID];
         if (reg->file == BAD_FILE)
            *reg = *v->make_reg_for_system_value(SYSTEM_VALUE_INSTANCE_ID, NULL);
         break;

      default:
         break;
      }
   }

   return true;
}

/* @FIXME: this name is somewhat misleading. I doesn't do the emission yet,
 * but just some setups. Probably nir_setup_system_values would be a better
 * name (like nir_setup_inputs or nir_setup_outputs).
 *
 * Using nir_emit_system_values as is the one used at brw_fs_nir, for
 * consistency.
 *
 **/
void
vec4_visitor::nir_emit_system_values(nir_shader *shader)
{
   /* @FIXME: vec4_visitor doesn't support fragment shader system values, so
    * we could use a smaller array. Keeping that way for now for simplicity
    * sake */
   nir_system_values = ralloc_array(mem_ctx, dst_reg, SYSTEM_VALUE_MAX);
   nir_foreach_overload(shader, overload) {
      assert(strcmp(overload->function->name, "main") == 0);
      assert(overload->impl);
      nir_foreach_block(overload->impl, emit_system_values_block, this);
   }
}

static int
type_size(const struct glsl_type *type)
{
   unsigned int i;
   int size;

   switch (type->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_BOOL:
      if (type->is_matrix()) {
	 return type->matrix_columns;
      } else {
	 /* Regardless of size of vector, it gets a vec4. This is bad
	  * packing for things like floats, but otherwise arrays become a
	  * mess.  Hopefully a later pass over the code can pack scalars
	  * down if appropriate.
	  */
	 return 1;
      }
   case GLSL_TYPE_ARRAY:
      assert(type->length > 0);
      return type_size(type->fields.array) * type->length;
   case GLSL_TYPE_STRUCT:
      size = 0;
      for (i = 0; i < type->length; i++) {
	 size += type_size(type->fields.structure[i].type);
      }
      return size;
   case GLSL_TYPE_SAMPLER:
      /* Samplers take up no register space, since they're baked in at
       * link time.
       */
      return 0;
   case GLSL_TYPE_ATOMIC_UINT:
      return 0;
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_INTERFACE:
      unreachable("not reached");
   }

   return 0;
}

void
vec4_visitor::nir_setup_inputs(nir_shader *shader)
{
   foreach_list_typed(nir_variable, var, node, &shader->inputs) {
      int offset = var->data.driver_location;
      int vector_elements =
         var->type->is_array() ? var->type->fields.array->vector_elements
                               : var->type->vector_elements;

      unsigned size = type_size(var->type);
      for (unsigned i = 0; i < size; i++) {
         src_reg src = src_reg(ATTR, var->data.location + i, var->type);
         src = retype(src, brw_type_for_base_type(var->type));
         nir_inputs[offset] = src;
         offset += vector_elements;
      }
   }
}

void
vec4_visitor::nir_setup_outputs(nir_shader *shader)
{
   foreach_list_typed(nir_variable, var, node, &shader->outputs) {
      int offset = var->data.driver_location;
      unsigned size = type_size(var->type);
      brw_reg_type type = brw_type_for_base_type(var->type);

      for (unsigned i = 0; i < size; i++) {
         nir_outputs[offset + i * 4] = var->data.location + i;
         nir_output_types[offset + i * 4] = type;
      }
   }
}

void
vec4_visitor::nir_setup_uniforms(nir_shader *shader)
{
   uniforms = 0;

   nir_uniform_offsets = rzalloc_array(mem_ctx, int, this->uniform_array_size * 4);
   memset(nir_uniform_offsets, 0, this->uniform_array_size * 4 * sizeof(int));

   if (shader_prog) {
      foreach_list_typed(nir_variable, var, node, &shader->uniforms) {
         /* UBO's, atomics and samplers don't take up space in the
            uniform file */
         if (var->interface_type != NULL || var->type->contains_atomic() ||
             type_size(var->type) == 0) {
            continue;
         }

         assert(uniforms < uniform_array_size);
         this->uniform_size[uniforms] = type_size(var->type);

         if (strncmp(var->name, "gl_", 3) == 0)
            nir_setup_builtin_uniform(var);
         else
            nir_setup_uniform(var);
      }
   } else {
      /* prog_to_nir doesn't create uniform variables; set param up directly. */
      for (unsigned p = 0; p < prog->Parameters->NumParameters; p++) {
         for (unsigned int i = 0; i < 4; i++) {
            stage_prog_data->param[4 * p + i] =
               &prog->Parameters->ParameterValues[p][i];
         }

         nir_uniform_offsets[p * 4] = uniforms;
         uniforms++;
      }
   }
}

void
vec4_visitor::nir_setup_uniform(nir_variable *var)
{
   int namelen = strlen(var->name);

   /* The data for our (non-builtin) uniforms is stored in a series of
    * gl_uniform_driver_storage structs for each subcomponent that
    * glGetUniformLocation() could name.  We know it's been set up in the same
    * order we'd walk the type, so walk the list of storage and find anything
    * with our name, or the prefix of a component that starts with our name.
    */

    unsigned offset = 0;
    for (unsigned u = 0; u < shader_prog->NumUniformStorage; u++) {
       struct gl_uniform_storage *storage = &shader_prog->UniformStorage[u];

       if (storage->builtin)
          continue;

       if (strncmp(var->name, storage->name, namelen) != 0 ||
           (storage->name[namelen] != 0 &&
            storage->name[namelen] != '.' &&
            storage->name[namelen] != '[')) {
          continue;
       }

       gl_constant_value *components = storage->storage;
       unsigned vector_count = (MAX2(storage->array_elements, 1) *
                                storage->type->matrix_columns);

       for (unsigned s = 0; s < vector_count; s++) {
          assert(uniforms < uniform_array_size);
          uniform_vector_size[uniforms] = storage->type->vector_elements;

          int i;
          for (i = 0; i < uniform_vector_size[uniforms]; i++) {
             stage_prog_data->param[uniforms * 4 + i] = components;
             components++;
          }
          for (; i < 4; i++) {
             static gl_constant_value zero = { 0.0 };
             stage_prog_data->param[uniforms * 4 + i] = &zero;
          }

          int uniform_offset = var->data.driver_location + offset;
          nir_uniform_offsets[uniform_offset] = uniforms;
          offset += uniform_vector_size[uniforms];

          uniforms++;
       }
    }
}

void
vec4_visitor::nir_setup_builtin_uniform(nir_variable *var)
{
   const nir_state_slot *const slots = var->state_slots;
   assert(var->state_slots != NULL);

   unsigned offset = 0;
   for (unsigned int i = 0; i < var->num_state_slots; i++) {
      /* This state reference has already been setup by ir_to_mesa,
       * but we'll get the same index back here.  We can reference
       * ParameterValues directly, since unlike brw_fs.cpp, we never
       * add new state references during compile.
       */
      int index = _mesa_add_state_reference(this->prog->Parameters,
					    (gl_state_index *)slots[i].tokens);
      gl_constant_value *values =
         &this->prog->Parameters->ParameterValues[index][0];

      assert(this->uniforms < uniform_array_size);

      for (unsigned j = 0; j < 4; j++)
	 stage_prog_data->param[this->uniforms * 4 + j] =
            &values[GET_SWZ(slots[i].swizzle, j)];

      this->uniform_vector_size[this->uniforms] =
         (var->type->is_scalar() || var->type->is_vector() ||
          var->type->is_matrix() ? var->type->vector_elements : 4);

      int uniform_offset = var->data.driver_location + offset;
      nir_uniform_offsets[uniform_offset] = uniforms;
      if (!var->type->is_record())
          offset += uniform_vector_size[uniforms];
      else
         offset += type_size(var->type->fields.structure[i].type);

      this->uniforms++;
   }
}

void
vec4_visitor::nir_emit_impl(nir_function_impl *impl)
{
   nir_locals = ralloc_array(mem_ctx, dst_reg, impl->reg_alloc);
   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      unsigned array_elems =
         reg->num_array_elems == 0 ? 1 : reg->num_array_elems;
      unsigned size = array_elems * reg->num_components;

      nir_locals[reg->index] = dst_reg(GRF, alloc.allocate(size));
   }

   nir_emit_cf_list(&impl->body);
}

void
vec4_visitor::nir_emit_cf_list(exec_list *list)
{
   exec_list_validate(list);
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if:
         nir_emit_if(nir_cf_node_as_if(node));
         break;

      case nir_cf_node_loop:
         nir_emit_loop(nir_cf_node_as_loop(node));
         break;

      case nir_cf_node_block:
         nir_emit_block(nir_cf_node_as_block(node));
         break;

      default:
         unreachable("Invalid CFG node block");
      }
   }
}

void
vec4_visitor::nir_emit_if(nir_if *if_stmt)
{
   /* First, put the condition in f0 */
   src_reg condition =
      retype(get_nir_src(if_stmt->condition), BRW_REGISTER_TYPE_D);

   int num_components = if_stmt->condition.is_ssa ?
      if_stmt->condition.ssa->num_components :
      if_stmt->condition.reg.reg->num_components;

   condition.swizzle = brw_swizzle_for_size(num_components);

   vec4_instruction *inst = emit(MOV(dst_null_d(), condition));
   inst->conditional_mod = BRW_CONDITIONAL_NZ;

   emit(IF(BRW_PREDICATE_NORMAL));

   nir_emit_cf_list(&if_stmt->then_list);

   /* note: if the else is empty, dead CF elimination will remove it */
   emit(BRW_OPCODE_ELSE);

   nir_emit_cf_list(&if_stmt->else_list);

   emit(BRW_OPCODE_ENDIF);
}

void
vec4_visitor::nir_emit_loop(nir_loop *loop)
{
   emit(BRW_OPCODE_DO);

   nir_emit_cf_list(&loop->body);

   emit(BRW_OPCODE_WHILE);
}

void
vec4_visitor::nir_emit_block(nir_block *block)
{
   nir_foreach_instr(block, instr) {
      nir_emit_instr(instr);
   }
}

void
vec4_visitor::nir_emit_instr(nir_instr *instr)
{
   this->base_ir = instr;

   switch (instr->type) {
   case nir_instr_type_load_const:
      /* We can hit these, but we do nothing now and use them as
       * immediates later in get_nir_src().
       * @FIXME: while this is what fs_nir does, we can do this better in the VS
       * stage because we can emit vector operations and save some MOVs in
       * cases where the constants are representable in 8 bits.
       */
      break;

   case nir_instr_type_intrinsic:
      nir_emit_intrinsic(nir_instr_as_intrinsic(instr));
      break;

   case nir_instr_type_alu:
      nir_emit_alu(nir_instr_as_alu(instr));
      break;

   case nir_instr_type_jump:
      nir_emit_jump(nir_instr_as_jump(instr));
      break;

   case nir_instr_type_tex:
      nir_emit_texture(nir_instr_as_tex(instr));
      break;

   default:
      fprintf(stderr, "VS instruction not yet implemented by NIR->vec4\n");
      break;
   }
}

static dst_reg
dst_reg_for_nir_reg(vec4_visitor *v, nir_register *nir_reg,
                    unsigned base_offset, nir_src *indirect)
{
   dst_reg reg;

   reg = v->nir_locals[nir_reg->index];

   reg = offset(reg, base_offset * nir_reg->num_components);
   if (indirect) {
      int multiplier = nir_reg->num_components;

      reg.reladdr = new(v->mem_ctx) src_reg(dst_reg(GRF, v->alloc.allocate(1)));
      v->emit(v->MUL(dst_reg(*reg.reladdr),
                     retype(v->get_nir_src(*indirect), BRW_REGISTER_TYPE_D),
                     src_reg(multiplier)));
   }

   return reg;
}

dst_reg
vec4_visitor::get_nir_dest(nir_dest dest)
{
   return dst_reg_for_nir_reg (this, dest.reg.reg, dest.reg.base_offset,
                               dest.reg.indirect);
}

src_reg
vec4_visitor::get_nir_src(nir_src src, nir_alu_type type)
{
   dst_reg reg;

   if (src.is_ssa) {
      assert(src.ssa->parent_instr->type == nir_instr_type_load_const);
      nir_load_const_instr *load = nir_instr_as_load_const(src.ssa->parent_instr);

      reg = dst_reg(GRF, alloc.allocate(src.ssa->num_components));
      reg = retype(reg, brw_type_for_nir_type(type));

      for (unsigned i = 0; i < src.ssa->num_components; ++i) {
         reg.writemask = 1 << i;

         switch (reg.type) {
         case BRW_REGISTER_TYPE_F:
            emit(MOV(reg, src_reg(load->value.f[i])));
            break;
         case BRW_REGISTER_TYPE_D:
            emit(MOV(reg, src_reg(load->value.i[i])));
            break;
         case BRW_REGISTER_TYPE_UD:
            emit(MOV(reg, src_reg(load->value.u[i])));
            break;
         default:
            unreachable("invalid register type");
         }
      }
   }
   else {
     reg = dst_reg_for_nir_reg(this, src.reg.reg, src.reg.base_offset,
                               src.reg.indirect);
     reg = retype(reg, brw_type_for_nir_type(type));
   }

   return src_reg(reg);
}

src_reg
vec4_visitor::get_nir_src(nir_src src)
{
   /* if type is not specified, default to signed int */
   return get_nir_src(src, nir_type_int);
}

void
vec4_visitor::nir_emit_intrinsic(nir_intrinsic_instr *instr)
{
   dst_reg dest;
   src_reg src;

   bool has_indirect = false;

   switch (instr->intrinsic) {

   case nir_intrinsic_load_input_indirect:
      has_indirect = true;
      /* fallthrough */
   case nir_intrinsic_load_input: {
      dest = get_nir_dest(instr->dest);

      dest.writemask = 0;
      for (int i = 0; i < instr->num_components; i++)
         dest.writemask |= 1 << i;

      int offset = instr->const_index[0];
      src = nir_inputs[offset];

      if (has_indirect)
         src.reladdr = new(mem_ctx) src_reg(get_nir_src(instr->src[0]));

      dest = retype(dest, src.type);
      emit(MOV(dest, src));
      break;
   }

   case nir_intrinsic_store_output_indirect:
      has_indirect = true;
      /* fallthrough */
   case nir_intrinsic_store_output: {
      src = get_nir_src(instr->src[0]);
      dest = dst_reg(src);

      dest.writemask = 0;
      for (unsigned i = 0; i < instr->num_components; i++)
         dest.writemask |= (1 << i);

      int offset = instr->const_index[0];
      int output = nir_outputs[offset];

      dest = retype(dest, nir_output_types[offset]);

      if (has_indirect)
         dest.reladdr = new(mem_ctx) src_reg(get_nir_src(instr->src[1]));

      output_reg[output] = dest;
      break;
   }

   case nir_intrinsic_load_vertex_id:
      unreachable("should be lowered by lower_vertex_id()");

   case nir_intrinsic_load_vertex_id_zero_base: {
      src_reg vertex_id = src_reg(nir_system_values[SYSTEM_VALUE_VERTEX_ID_ZERO_BASE]);
      dest = get_nir_dest(instr->dest);
      assert(vertex_id.file != BAD_FILE);
      dest.type = vertex_id.type;
      emit(MOV(dest, vertex_id));

      break;
   }

   case nir_intrinsic_load_base_vertex: {
      src_reg base_vertex = src_reg(nir_system_values[SYSTEM_VALUE_BASE_VERTEX]);
      dest = get_nir_dest(instr->dest);
      assert(base_vertex.file != BAD_FILE);
      dest.type = base_vertex.type;
      emit(MOV(dest, base_vertex));

      break;
   }

   case nir_intrinsic_load_instance_id: {
      src_reg instance_id = src_reg(nir_system_values[SYSTEM_VALUE_INSTANCE_ID]);
      dest = get_nir_dest(instr->dest);
      assert(instance_id.file != BAD_FILE);
      dest.type = instance_id.type;
      emit(MOV(dest, instance_id));

      break;
   }

   case nir_intrinsic_load_uniform_indirect:
      has_indirect = true;
      /* fallthrough */
   case nir_intrinsic_load_uniform: {
      unsigned index = instr->const_index[0];
      unsigned offset = nir_uniform_offsets[index];

      dest = get_nir_dest(instr->dest);
      src = src_reg(dst_reg(UNIFORM, offset));

      /* @FIXME: this has not been tested yet, just copied from fs_nir */
      if (has_indirect) {
         src_reg tmp = retype(get_nir_src(instr->src[0]), BRW_REGISTER_TYPE_D);
         src.reladdr = new(mem_ctx) src_reg(tmp);
      }
      emit(MOV(dest, src));
      break;
   }

   case nir_intrinsic_atomic_counter_read:
   case nir_intrinsic_atomic_counter_inc:
   case nir_intrinsic_atomic_counter_dec: {
      unsigned surf_index = prog_data->base.binding_table.abo_start +
         (unsigned) instr->const_index[0];
      src_reg offset = src_reg(get_nir_src(instr->src[0], nir_type_int));
      dest = get_nir_dest(instr->dest);

      switch (instr->intrinsic) {
         case nir_intrinsic_atomic_counter_inc:
            emit_untyped_atomic(BRW_AOP_INC, surf_index, dest, offset,
                                src_reg(), src_reg());
            break;
         case nir_intrinsic_atomic_counter_dec:
            emit_untyped_atomic(BRW_AOP_PREDEC, surf_index, dest, offset,
                                src_reg(), src_reg());
            break;
         case nir_intrinsic_atomic_counter_read:
            emit_untyped_surface_read(surf_index, dest, offset);
            break;
         default:
            unreachable("Unreachable");
      }
   }
      break;
   case nir_intrinsic_load_ubo_indirect:
      has_indirect = true;
      /* fallthrough */
   case nir_intrinsic_load_ubo: {
      src_reg surf_index;
      nir_const_value *const_block_index = nir_src_as_const_value(instr->src[0]);
      dest = get_nir_dest(instr->dest);

      if (const_block_index) {
         /* The block index is a constant, so just emit the binding table entry
          * as an immediate.
          */
         surf_index = src_reg(prog_data->base.binding_table.ubo_start +
                              const_block_index->u[0]);
      } else {
         /* The block index is not a constant. Evaluate the index expression
          * per-channel and add the base UBO index; the generator will select
          * a value from any live channel.
          */
         surf_index = src_reg(this, glsl_type::uint_type);
         emit(ADD(dst_reg(surf_index), get_nir_src(instr->src[0], nir_type_int),
                  src_reg(prog_data->base.binding_table.ubo_start)));

         /* Assume this may touch any UBO. It would be nice to provide
          * a tighter bound, but the array information is already lowered away.
          */
         brw_mark_surface_used(&prog_data->base,
                               prog_data->base.binding_table.ubo_start +
                               shader_prog->NumUniformBlocks - 1);
      }

      unsigned const_offset = instr->const_index[0];
      src_reg offset;

      if  (!has_indirect)  {
         if (devinfo->gen >= 8) {
            /* Store the offset in a GRF so we can send-from-GRF. */
            offset = src_reg(this, glsl_type::int_type);
            emit(MOV(dst_reg(offset), src_reg(const_offset / 16)));
         } else {
            /* Immediates are fine on older generations since they'll be moved
             * to a (potentially fake) MRF at the generator level.
             */
            offset = src_reg(const_offset / 16);
         }
      } else {
         offset = src_reg(this, glsl_type::uint_type);
         emit(SHR(dst_reg(offset), get_nir_src(instr->src[1], nir_type_int), src_reg(4)));
      }

      src_reg packed_consts = src_reg(this, glsl_type::vec4_type);
      packed_consts.type = dest.type;

      emit_pull_constant_load_reg(dst_reg(packed_consts),
                                  surf_index,
                                  offset,
                                  NULL, NULL /* before_block/inst */);

      packed_consts.swizzle = brw_swizzle_for_size(instr->num_components);
      packed_consts.swizzle += BRW_SWIZZLE4(const_offset % 16 / 4,
                                            const_offset % 16 / 4,
                                            const_offset % 16 / 4,
                                            const_offset % 16 / 4);

      emit(MOV(dest, packed_consts));

      break;
   }

   default:
      fprintf(stderr,
              "Non-implemented intrinsic instruction in NIR->vec4 (%d)\n",
              instr->intrinsic);
      break;
   }
}

static unsigned
brw_swizzle_for_nir_swizzle(uint8_t swizzle[4])
{
   return  BRW_SWIZZLE4(swizzle[0], swizzle[1], swizzle[2], swizzle[3]);
}

void
vec4_visitor::nir_emit_alu(nir_alu_instr *instr)
{
   vec4_instruction *inst;

   dst_reg dst = get_nir_dest(instr->dest.dest);
   dst = retype(dst,
                brw_type_for_nir_type(nir_op_infos[instr->op].output_type));
   dst.writemask = instr->dest.write_mask;

   src_reg op[4];
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      op[i] = get_nir_src(instr->src[i].src,
                          nir_op_infos[instr->op].input_types[i]);
      op[i].swizzle = brw_swizzle_for_nir_swizzle(instr->src[i].swizzle);
      op[i].abs = instr->src[i].abs;
      op[i].negate = instr->src[i].negate;
   }

   switch(instr->op) {
   case nir_op_imov:
   case nir_op_fmov:
      dst.writemask = instr->dest.write_mask;

      inst = emit(MOV(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
         unreachable("not reached: should be handled by lower_vec_to_movs()");

   case nir_op_i2f:
   case nir_op_u2f:
      inst = emit(MOV(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_f2i:
   case nir_op_f2u:
      inst = emit(MOV(dst, op[0]));
      break;

   case nir_op_fadd:
      op[0] = retype(op[0], BRW_REGISTER_TYPE_F);
      op[1] = retype(op[1], BRW_REGISTER_TYPE_F);
      /* fall through */
   case nir_op_iadd:
      inst = emit(ADD(dst, op[0], op[1]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmul:
      inst = emit(MUL(dst, op[0], op[1]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_imul:
      if (brw->gen >= 8) {
	 emit(MUL(dst, op[0], op[1]));
      } else {
         nir_const_value *value0 = nir_src_as_const_value(instr->src[0].src);
         nir_const_value *value1 = nir_src_as_const_value(instr->src[1].src);

         if (value0 && value0->u[0] < (1 << 16)) {
            if (brw->gen < 7)
               emit(MUL(dst,  op[0], op[1]));
            else
               emit(MUL(dst, op[1], op[0]));
         } else if  (value1 && value1->u[0] < (1 << 16)) {
            if (brw->gen < 7)
               emit(MUL(dst, op[1], op[0]));
            else
               emit(MUL(dst, op[0], op[1]));
         } else {
            struct brw_reg acc = retype(brw_acc_reg(8), dst.type);

            emit(MUL(acc, op[0], op[1]));
            emit(MACH(dst_null_d(), op[0], op[1]));
            emit(MOV(dst, src_reg(acc)));
         }
      }
      break;

   case nir_op_imul_high:
   case nir_op_umul_high: {
      struct brw_reg acc = retype(brw_acc_reg(8), dst.type);

      emit(MUL(acc, op[0], op[1]));
      emit(MACH(dst, op[0], op[1]));
      break;
   }

   case nir_op_frcp:
      inst = emit_math(SHADER_OPCODE_RCP, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fexp2:
      inst = emit_math(SHADER_OPCODE_EXP2, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flog2:
      inst = emit_math(SHADER_OPCODE_LOG2, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fsin:
      inst = emit_math(SHADER_OPCODE_SIN, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fcos:
      inst = emit_math(SHADER_OPCODE_COS, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_idiv:
   case nir_op_udiv:
      emit_math(SHADER_OPCODE_INT_QUOTIENT, dst, op[0], op[1]);
      break;

   case nir_op_umod:
      emit_math(SHADER_OPCODE_INT_REMAINDER, dst, op[0], op[1]);
      break;

   case nir_op_uadd_carry: {
      struct brw_reg acc = retype(brw_acc_reg(8), BRW_REGISTER_TYPE_UD);

      emit(ADDC(dst_null_ud(), op[0], op[1]));
      emit(MOV(dst, src_reg(acc)));
      break;
   }

   case nir_op_usub_borrow: {
      struct brw_reg acc = retype(brw_acc_reg(8), BRW_REGISTER_TYPE_UD);

      emit(SUBB(dst_null_ud(), op[0], op[1]));
      emit(MOV(dst, src_reg(acc)));
      break;
   }

   case nir_op_ldexp:
      unreachable("not reached: should be handled by ldexp_to_arith()");

   case nir_op_fsqrt:
      inst = emit_math(SHADER_OPCODE_SQRT, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_frsq:
      inst = emit_math(SHADER_OPCODE_RSQ, dst, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fpow:
      inst = emit_math(SHADER_OPCODE_POW, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_ftrunc:
      inst = emit(RNDZ(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fceil: {
      int num_components;

      if (instr->src[0].src.is_ssa)
         num_components = instr->src[0].src.ssa->num_components;
      else
         num_components = instr->src[0].src.reg.reg->num_components;

      src_reg tmp = src_reg(this, glsl_type::float_type, num_components);
      if (num_components > 0)
         tmp.swizzle = brw_swizzle_for_size(num_components);

      op[0].negate = !op[0].negate;
      emit(RNDD(dst_reg(tmp), op[0]));
      tmp.negate = true;
      inst = emit(MOV(dst, tmp));
      inst->saturate = instr->dest.saturate;
      break;
   }

   case nir_op_ffloor:
      inst = emit(RNDD(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_ffract:
      inst = emit(FRC(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fround_even:
      inst = emit(RNDE(dst, op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmin:
   case nir_op_imin:
   case nir_op_umin:
      /* @FIXME: I took the code for max and min from the vec4_visitor,
       * in fs_nir and fs_visitor  was a bit different. First, in the fs_visitor
       * before  performing the operation a call to resolve_ud_negate was
       * made for every operand. Second, the emit_minmax code of both
       * the fs_visitor and fs_nir was equivalent to the vec4 emit_minmax
       * code with the exception that (if gen < 6) they do:
       *    emit(CMP(reg_null_d, src0, src1, conditionalmod)); ...
       * instead of:
       *    emit(CMP(dst, src0, src1, conditionalmod)); ...
       * See the fs_visitor and vec4_visitor emit_minmax code for more details.
       */
      inst = emit_minmax(BRW_CONDITIONAL_L, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmax:
   case nir_op_imax:
   case nir_op_umax:
      inst = emit_minmax(BRW_CONDITIONAL_GE, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fddx:
   case nir_op_fddx_coarse:
   case nir_op_fddx_fine:
   case nir_op_fddy:
   case nir_op_fddy_coarse:
   case nir_op_fddy_fine:
      unreachable("derivatives not valid in vertex shader");

   case nir_op_flt:
   case nir_op_ilt:
   case nir_op_ult:
   case nir_op_fge:
   case nir_op_ige:
   case nir_op_uge:
   case nir_op_feq:
   case nir_op_ieq:
   case nir_op_fne:
   case nir_op_ine:
      /* @FIXME: if (gen <=5) both fs_visitor and vec4_visitor call the function
       * resolve_bool_comparison for every operand before doing the emit.
       * This check and function calls are not done in the brw_fs_nir.
       * Check if we need to add it, it could be the case that NIR is not available
       * for (gen <= 5) or this check and calls not needed for other reasons.
       */
      emit(CMP(dst, op[0], op[1],
               brw_conditional_for_nir_comparison(instr->op)));
      break;

   case nir_op_ball_fequal2:
   case nir_op_ball_iequal2:
   case nir_op_ball_fequal3:
   case nir_op_ball_iequal3:
   case nir_op_ball_fequal4:
   case nir_op_ball_iequal4: {
      /* @FIXME: if (gen <= 5) the vec4_visitor and fs_visitor call to
       * resolve_bool_comparison for every operand. To check if we
       * want to add it.
       */

      /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      op[1].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[1].swizzle);

      emit(CMP(dst_null_d(), op[0], op[1],
               brw_conditional_for_nir_comparison(instr->op)));
      emit(MOV(dst, src_reg(0)));
      inst = emit(MOV(dst, src_reg(~0)));
      inst->predicate = BRW_PREDICATE_ALIGN16_ALL4H;
      break;
   }

   case nir_op_bany_fnequal2:
   case nir_op_bany_inequal2:
   case nir_op_bany_fnequal3:
   case nir_op_bany_inequal3:
   case nir_op_bany_fnequal4:
   case nir_op_bany_inequal4: {
      /* @FIXME: if (gen <= 5) the vec4_visitor and fs_visitor call to
       * resolve_bool_comparison for every operand. To check if we
       * want to add it.
       */

      /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      op[1].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[1].swizzle);

      emit(CMP(dst_null_d(), op[0], op[1],
               brw_conditional_for_nir_comparison(instr->op)));
      emit(MOV(dst, src_reg(0)));
      inst = emit(MOV(dst, src_reg(~0)));
      inst->predicate = BRW_PREDICATE_ALIGN16_ANY4H;
      break;
   }
      /* @FIXME: for the following logical operations: inot, ixor, ior, iand,
       * brw_fs_nir calls to resolve_source_modifiers for every operand
       * if (gen > =8). This call and check is not present in the fs and vec4
       * visitors. Decide if we want to add it.
       */
   case nir_op_inot:
      inst = emit(NOT(dst, op[0]));
      break;

   case nir_op_ixor:
      emit(XOR(dst, op[0], op[1]));
      break;

   case nir_op_ior:
      emit(OR(dst, op[0], op[1]));
      break;

   case nir_op_iand:
      emit(AND(dst, op[0], op[1]));
      break;

   case nir_op_b2i:
      emit(AND(dst, op[0], src_reg(1)));
      break;
   case nir_op_b2f:
      /* @FIXME: as for the relational operations,  both the fs_visitor and vec4_visitor
       * call to resolve_bool_comparison for the operand if (gen <= 5). Check if it is
       * needed.
       */

      /* @FIXME: fs_visitor and vec4_visitor, do:
       *      op[0].type = BRW_REGISTER_TYPE_D;
       *      result_dst.type = BRW_REGISTER_TYPE_D;
       *      emit(AND(result_dst, op[0], src_reg(0x3f800000u)));
       *      result_dst.type = BRW_REGISTER_TYPE_F;
       * instead of the following emit (C&P from brw_fs_nir):
       */
      emit(AND(retype(dst, BRW_REGISTER_TYPE_UD), op[0], src_reg(0x3f800000u)));
      break;

   case nir_op_f2b:
      emit(CMP(dst, op[0], src_reg(0.0f), BRW_CONDITIONAL_NZ));
      break;
   case nir_op_i2b:
      emit(CMP(dst, op[0],  src_reg(0), BRW_CONDITIONAL_NZ));
      break;

   case nir_op_fnoise1_1:
   case nir_op_fnoise1_2:
   case nir_op_fnoise1_3:
   case nir_op_fnoise1_4:
   case nir_op_fnoise2_1:
   case nir_op_fnoise2_2:
   case nir_op_fnoise2_3:
   case nir_op_fnoise2_4:
   case nir_op_fnoise3_1:
   case nir_op_fnoise3_2:
   case nir_op_fnoise3_3:
   case nir_op_fnoise3_4:
   case nir_op_fnoise4_1:
   case nir_op_fnoise4_2:
   case nir_op_fnoise4_3:
   case nir_op_fnoise4_4:
      unreachable("not reached: should be handled by lower_noise");

   case nir_op_unpack_half_2x16_split_x:
   case nir_op_unpack_half_2x16_split_y:
   case nir_op_pack_half_2x16_split:
      unreachable("not reached: should not occur in vertex shader");

   case nir_op_pack_snorm_2x16:
   case nir_op_pack_unorm_2x16:
   case nir_op_unpack_snorm_2x16:
   case nir_op_unpack_unorm_2x16:
      unreachable("not reached: should be handled by lower_packing_builtins");

   case nir_op_pack_half_2x16: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      emit_pack_half_2x16(dst, op[0]);
      break;
   }

   case nir_op_unpack_half_2x16: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      emit_unpack_half_2x16(dst, op[0]);
      break;

   }

   case nir_op_unpack_unorm_4x8: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      emit_unpack_unorm_4x8(dst, op[0]);
      break;
   }

   case nir_op_unpack_snorm_4x8: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);

      emit_unpack_snorm_4x8(dst, op[0]);
      break;
   }

   case nir_op_pack_unorm_4x8: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      emit_pack_unorm_4x8(dst, op[0]);
      break;
   }

   case nir_op_pack_snorm_4x8: {
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);
      emit_pack_snorm_4x8(dst, op[0]);
      break;
   }

   case nir_op_bitfield_reverse:
      emit(BFREV(dst, op[0]));
      break;

   case nir_op_bit_count:
      emit(CBIT(dst, op[0]));
      break;
   case nir_op_ufind_msb:
   case nir_op_ifind_msb: {
      /* @FIXME: FBH only supports UD type for dst, in the fs_visitor
       * and the vec4_visitor, instead of using a retype in the FBH call
       * like in fs_nir, they use a temporary UD register to make the call
       * and a MOV to convert back UD to D after the FBH. Something like this:
       *       src_reg temp = src_reg(this, glsl_type::uint_type);
       *       inst = emit(FBH(dst_reg(temp), op[0]));
       *       emit(MOV(dst, temp));
       * Apart from this, the vec4_visitor also set before the MOV:
       *       inst->dst.writemask = WRITEMASK_XYZW;
       *       temp.swizzle = BRW_SWIZZLE_NOOP;
       * Check if it is enough doing the retype (like in fs_nir) or we need to do the MOV.
       */
      emit(FBH(retype(dst, BRW_REGISTER_TYPE_UD), op[0]));

      /* FBH counts from the MSB side, while GLSL's findMSB() wants the count
       * from the LSB side. If FBH didn't return an error (0xFFFFFFFF), then
       * subtract the result from 31 to convert the MSB count into an LSB count.
       */
      src_reg tmp = src_reg(dst);

      emit(CMP(dst_null_d(), tmp, src_reg(-1), BRW_CONDITIONAL_NZ));
      tmp.negate = true;
      inst = emit(ADD(dst, tmp, src_reg(31)));
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;
   }

   case nir_op_find_lsb:
      emit(FBL(dst, op[0]));
      break;

   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      emit(BFE(dst, op[2], op[1], op[0]));
      break;

   case nir_op_bfm:
      emit(BFI1(dst, op[0], op[1]));
      break;

   case nir_op_bfi:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      emit(BFI2(dst, op[0], op[1], op[2]));
      break;

   case nir_op_bitfield_insert:
      unreachable("not reached: should be handled by "
                  "lower_instructions::bitfield_insert_to_bfm_bfi");

   case nir_op_fsign:
      /* AND(val, 0x80000000) gives the sign bit.
       *
       * Predicated OR ORs 1.0 (0x3f800000) with the sign bit if val is not
       * zero.
       */
      emit(CMP(dst_null_f(), op[0], src_reg(0.0f), BRW_CONDITIONAL_NZ));

      op[0].type = BRW_REGISTER_TYPE_UD;
      dst.type = BRW_REGISTER_TYPE_UD;
      emit(AND(dst, op[0], src_reg(0x80000000u)));

      inst = emit(OR(dst, src_reg(dst), src_reg(0x3f800000u)));
      inst->predicate = BRW_PREDICATE_NORMAL;
      /* @FIXME: brw_fs_nir adds the following code:
       *     if (instr->dest.saturate) {
       *         inst = emit(MOV(dst, dst));
       *         inst->saturate = true;
       *     }
       * and it also uses a temporary result_int register to emit
       * the previous operations.
       * The code related to the saturate and the int register are
       * not present in the fs and vec4 visitors.
       * However, both the fs and vec4 visitors add:
       *       this->result.type = BRW_REGISTER_TYPE_F;
       */
      break;
   case nir_op_isign:
      /*  ASR(val, 31) -> negative val generates 0xffffffff (signed -1).
       *               -> non-negative val generates 0x00000000.
       *  Predicated OR sets 1 if val is positive.
       */
      emit(CMP(dst_null_d(), op[0], src_reg(0), BRW_CONDITIONAL_G));
      emit(ASR(dst, op[0], src_reg(31)));
      inst = emit(OR(dst, src_reg(dst), src_reg(1)));
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;

   case nir_op_ishl:
      emit(SHL(dst, op[0], op[1]));
      break;

   case nir_op_ishr:
      emit(ASR(dst, op[0], op[1]));
      break;

   case nir_op_ushr:
      emit(SHR(dst, op[0], op[1]));
      break;

   case nir_op_ffma:
      /* @FIXME: vec4_visitor adds:
       *       op[0] = fix_3src_operand(op[0]);
       *       op[1] = fix_3src_operand(op[1]);
       *       op[2] = fix_3src_operand(op[2]);
       * We probably want to add it too. To confirm.
       */
      inst = emit(MAD(dst, op[2], op[1], op[0]));
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flrp:
      /* @FIXME: emit_lrp is implicitly calling to fix_3src_operand.
       * Notice that we are not doing it for other operations.
       * Probably we want to do it.
       */
      inst = emit_lrp(dst, op[0], op[1], op[2]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_bcsel:
      emit(CMP(dst_null_d(), op[0], src_reg(0), BRW_CONDITIONAL_NZ));
      inst = emit(BRW_OPCODE_SEL, dst, op[1], op[2]);
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;

   case nir_op_fdot2:
      inst = emit(BRW_OPCODE_DP2, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fdot3:
      inst = emit(BRW_OPCODE_DP3, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fdot4:
      inst = emit(BRW_OPCODE_DP4, dst, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_bany2:
   case nir_op_bany3:
   case nir_op_bany4: {
      /* @FIXME: if (gen <= 5) the vec4_visitor calls to resolve_bool_comparison for
       * the operand. To check if we want to add it.
       */
     /* Update the swizzle to take into account the size of the operand */
      unsigned size = nir_op_infos[instr->op].input_sizes[0];
      op[0].swizzle = brw_compose_swizzle(brw_swizzle_for_size(size),
                                          op[0].swizzle);

      emit(CMP(dst_null_d(), op[0], src_reg(0), BRW_CONDITIONAL_NZ));
      emit(MOV(dst, src_reg(0)));

      inst = emit(MOV(dst, src_reg((int)ctx->Const.UniformBooleanTrue)));
      inst->predicate = BRW_PREDICATE_ALIGN16_ANY4H;
      break;
   }

   case nir_op_fabs:
   case nir_op_iabs:
   case nir_op_fneg:
   case nir_op_ineg:
   case nir_op_fsat:
      unreachable("not reached: should be lowered by lower_source mods");

   case nir_op_fdiv:
     unreachable("not reached: should be lowered by DIV_TO_MUL_RCP in the compiler");

   case nir_op_fmod:
     unreachable("not reached: should be lowered by MOD_TO_FLOOR in the compiler");

   case nir_op_fsub:
   case nir_op_isub:
      unreachable("not reached: should be handled by ir_sub_to_add_neg");

   default:
      fprintf(stderr, "Non-implemented ALU operation (%d)\n", instr->op);
      break;
   }
}

void
vec4_visitor::nir_emit_jump(nir_jump_instr *instr)
{
   switch (instr->type) {
   case nir_jump_break:
      emit(BRW_OPCODE_BREAK);
      break;
   case nir_jump_continue:
      emit(BRW_OPCODE_CONTINUE);
      break;
   case nir_jump_return:
   default:
      unreachable("unknown jump");
   }
}


void
vec4_visitor::nir_emit_texture(nir_tex_instr *instr)
{
   unsigned sampler = instr->sampler_index;
   src_reg sampler_reg = src_reg(sampler);
   src_reg coordinate;
   src_reg shadow_comparitor;
   int shadow_compare = 0;
   int offset_components = 0;
   src_reg tex_offset;
   src_reg lod;

   /* Get the parameters */
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      src_reg src = get_nir_src(instr->src[i].src);
      switch (instr->src[i].src_type) {
      case nir_tex_src_comparitor:
         shadow_comparitor = retype (src, BRW_REGISTER_TYPE_F);
         shadow_compare = 1;
         break;
      case nir_tex_src_coord:
         switch (instr->op) {
         case nir_texop_txf:
            fprintf(stderr, "WIP: \tnir_texop_txf\n");
         case nir_texop_txf_ms:
            fprintf(stderr, "\tnir_texop_txf_ms\n");
            coordinate = retype(src, BRW_REGISTER_TYPE_D);
            break;
         default:
            coordinate = retype(src, BRW_REGISTER_TYPE_F);
            break;
         }
         break;
      case nir_tex_src_ddx:
         fprintf(stderr, "WIP: nir_tex_src_ddx\n");
         break;
      case nir_tex_src_ddy:
         fprintf(stderr, "WIP: nir_tex_src_ddy\n");
         break;
      case nir_tex_src_lod:
         switch (instr->op) {
         case nir_texop_txs:
            fprintf(stderr, "\t WIP: nir_tex_src_lod:nir_texop_txs\n");
            break;
         case nir_texop_txf:
            fprintf(stderr, "\t WIP: nir_tex_src_lod:nir_texop_txf\n");
            break;
         default:
            lod = retype(src, BRW_REGISTER_TYPE_F);
            break;
         }
         break;
      case nir_tex_src_ms_index:
         fprintf(stderr, "WIP: nir_tex_src_ms_index\n");
         break;
      case nir_tex_src_offset:
         fprintf(stderr, "WIP: nir_tex_src_offset\n");
         break;
      case nir_tex_src_projector:
         fprintf(stderr, "WIP: nir_tex_src_projector\n");
         unreachable("should be lowered");
      case nir_tex_src_sampler_offset:
         fprintf(stderr, "WIP: nir_tex_src_sampler_offset\n");
      case nir_tex_src_bias:
         unreachable("LOD bias is not valid for vertex shaders.\n");
         break;
      default:
         unreachable("unknown texture source");
      }
   }

   for (unsigned i = 0; i < 3; i++) {
      if (instr->const_offset[i] != 0) {
         /* @FIXME: right now offset_components will be always 0,
            as nir_tex_src_offset is not supported yet */
         assert(offset_components == 0);
         tex_offset = src_reg(brw_texture_offset(instr->const_offset, 3));
         break;
      }
   }

   /* Get the texture operation */
   /*@FIME (comment to be removed) On brw_fs_visitor this switch makes a
    * nir_texop=>ir_texop conversion, as it relies on brw_fs_visitor ir-based
    * emit_texture (that relies on different gen versions). For now we are
    * being "nir-pure" so we can do a direct shader opcode conversion */
   enum opcode opcode;
   switch (instr->op) {
   case nir_texop_query_levels: opcode = SHADER_OPCODE_TXS; break;
   case nir_texop_tex:
      lod = src_reg(0.0f);
      opcode = SHADER_OPCODE_TXL;
      break;
      /* @FIXME: for tg4 we need to check if has a non constant offset */
   case nir_texop_tg4: opcode = SHADER_OPCODE_TG4; break;
   case nir_texop_txd: opcode = SHADER_OPCODE_TXD; break;
   case nir_texop_txf: opcode = SHADER_OPCODE_TXF; break;
   case nir_texop_txf_ms: opcode = SHADER_OPCODE_TXF_CMS; break;
   case nir_texop_txl: opcode = SHADER_OPCODE_TXL; break;
   case nir_texop_txs: opcode = SHADER_OPCODE_TXS; break;
   case nir_texop_txb:
      unreachable("TXB (ie: texture() with bias on glsl) is not valid for vertex shaders.\n");
   case nir_texop_lod:
      unreachable("LOD (ie: textureQueryLOD on glsl) is not valid for vertex shaders.\n");
      break;
   default:
      unreachable("unknown texture opcode");
   }

   enum glsl_base_type dest_base_type;
   switch (instr->dest_type) {
   case nir_type_float:
      dest_base_type = GLSL_TYPE_FLOAT;
      break;
   case nir_type_int:
      dest_base_type = GLSL_TYPE_INT;
      break;
   case nir_type_unsigned:
      dest_base_type = GLSL_TYPE_UINT;
      break;
   default:
      unreachable("bad type");
   }

   const glsl_type *dest_type =
      glsl_type::get_instance(dest_base_type, nir_tex_instr_dest_size(instr), 1);

   vec4_instruction *inst = new(mem_ctx)
      vec4_instruction(opcode, dst_reg(this, dest_type));

   if (tex_offset.file == IMM)
      inst->offset = tex_offset.fixed_hw_reg.dw1.ud;

   /* @FIXME: wip consider all cases for header_size (see brw_vec4_visitor) */
   inst->header_size = (inst->offset != 0) ? 1 : 0;
   inst->base_mrf = 2;
   inst->mlen = inst->header_size + 1;
   inst->dst.writemask = WRITEMASK_XYZW;
   inst->shadow_compare = shadow_compare;

   inst->src[1] = sampler_reg;

   /* Load the coordinate */
   int param_base = inst->base_mrf + inst->header_size;
   int coord_mask = (1 << instr->coord_components) - 1;
   int zero_mask = 0xf & ~coord_mask;

   emit(MOV(dst_reg(MRF, param_base, coordinate.type, coord_mask), coordinate));

   if (zero_mask != 0) {
      emit(MOV(dst_reg(MRF, param_base, coordinate.type, zero_mask), src_reg(0)));
   }

   /* Load the shadow comparitor */
   if (shadow_compare) { /*@FIXME: this conditional is assuming only tex op support */
      emit(MOV(dst_reg(MRF, param_base + 1, shadow_comparitor.type, WRITEMASK_X),
               shadow_comparitor));
      inst->mlen++;
   }
   /* Load the LOD info */
   if (instr->op == nir_texop_tex || instr->op == nir_texop_txl) {
      int mrf, writemask;
      mrf = param_base + 1; /* @FIXME: asumming devinfo->gen >= 5 */
      if (shadow_compare) {
         writemask = WRITEMASK_Y;
         /* mlen already incremented on shadow comparitor loading */
      } else {
         writemask = WRITEMASK_X;
         inst->mlen++;
      }
      emit(MOV(dst_reg(MRF, mrf, lod.type, writemask), lod));
   } else {
      /* @FIXME: WIP */
      fprintf(stderr, "WIP: lod only supported for tex or txl texop\n");
   }

   emit(inst);

   dst_reg dest = get_nir_dest(instr->dest);
   /* @FIXME: get_nir_dest calls dst_reg_for_nir_reg that sets a hardcoded
    * type. It is needed to set the proper type to get things working. */
   dest.type = brw_type_for_base_type (dest_type);

   /* @FIXME: here brw_vec4_visitor call swizzle_result, that does a swizzle
    * on the source and/or the destination if needed, and then it emit the
    * src->dest mov. swizzle not supported yet, so just doing a direct mov */
   emit(MOV(dest, src_reg(inst->dst)));
}

}
