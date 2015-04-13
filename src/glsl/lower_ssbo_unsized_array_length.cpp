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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file lower_ssbo_unsized_array_length.cpp
 *
 * IR lower pass to replace ir_unop_ssbo_unsized_array_length expressions 
 * in the HIR with usage of ir_triop_ssbo_unsized_array_length expressions
 * because, at this point, all the needed data is available.
 *
 * ir_triop_ssbo_unsized_array_length expression implementation is driver's work
 * because unsized array's lenght should be calculated in run-time because the
 * bound buffer to a shader storage block can change.
 */

#include "ir.h"
#include "ir_builder.h"
#include "ir_rvalue_visitor.h"
#include "main/macros.h"

using namespace ir_builder;

namespace {

class lower_ssbo_unsized_array_length_visitor : public ir_hierarchical_visitor {
public:
    lower_ssbo_unsized_array_length_visitor(struct gl_shader *shader)
      : shader(shader)
   {
   }

   virtual ir_visitor_status visit_leave(class ir_expression *);
   virtual ir_visitor_status visit_leave(class ir_assignment *);
   ir_expression *process(ir_rvalue **, ir_dereference *, ir_variable *);
   ir_expression *emit_ssbo_unsized_array_length(ir_variable *base_offset,
                                       unsigned int deref_offset,
                                       unsigned int unsized_array_stride);
   ir_expression *ssbo_unsized_array_length(ir_rvalue *offset, ir_rvalue *stride);
   unsigned calculate_unsized_array_stride(ir_dereference *deref);

   void *mem_ctx;
   struct gl_shader *shader;
   struct gl_uniform_buffer_variable *ubo_var;
   ir_rvalue *uniform_block;
   bool progress;
};

} /* anonymous namespace */

ir_expression *
lower_ssbo_unsized_array_length_visitor::ssbo_unsized_array_length(ir_rvalue *offset,
                                                                   ir_rvalue *stride)
{
   ir_rvalue *block_ref = this->uniform_block->clone(mem_ctx, NULL);
   return new(mem_ctx) ir_expression(ir_triop_ssbo_unsized_array_length,
                                     glsl_type::int_type, block_ref, offset, stride);
}

ir_expression *
lower_ssbo_unsized_array_length_visitor::emit_ssbo_unsized_array_length(
                                            ir_variable *base_offset,
                                            unsigned int deref_offset,
                                            unsigned int unsized_array_stride)
{
   ir_rvalue *offset =
      add(base_offset, new(mem_ctx) ir_constant(deref_offset));
   ir_rvalue *stride = new(mem_ctx) ir_constant(unsized_array_stride);
   return ssbo_unsized_array_length(offset, stride);
}

static bool
is_dereferenced_thing_row_major(const ir_dereference *deref)
{
   bool matrix = false;
   const ir_rvalue *ir = deref;

   while (true) {
      matrix = matrix || ir->type->without_array()->is_matrix();

      switch (ir->ir_type) {
      case ir_type_dereference_array: {
         const ir_dereference_array *const array_deref =
            (const ir_dereference_array *) ir;

         ir = array_deref->array;
         break;
      }

      case ir_type_dereference_record: {
         const ir_dereference_record *const record_deref =
            (const ir_dereference_record *) ir;

         ir = record_deref->record;

         const int idx = ir->type->field_index(record_deref->field);
         assert(idx >= 0);

         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(ir->type->fields.structure[idx].matrix_layout);

         switch (matrix_layout) {
         case GLSL_MATRIX_LAYOUT_INHERITED:
            break;
         case GLSL_MATRIX_LAYOUT_COLUMN_MAJOR:
            return false;
         case GLSL_MATRIX_LAYOUT_ROW_MAJOR:
            return matrix || deref->type->without_array()->is_record();
         }

         break;
      }

      case ir_type_dereference_variable: {
         const ir_dereference_variable *const var_deref =
            (const ir_dereference_variable *) ir;

         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(var_deref->var->data.matrix_layout);

         switch (matrix_layout) {
         case GLSL_MATRIX_LAYOUT_INHERITED:
            assert(!matrix);
            return false;
         case GLSL_MATRIX_LAYOUT_COLUMN_MAJOR:
            return false;
         case GLSL_MATRIX_LAYOUT_ROW_MAJOR:
            return matrix || deref->type->without_array()->is_record();
         }

         unreachable("invalid matrix layout");
         break;
      }

      default:
         return false;
      }
   }

   /* The tree must have ended with a dereference that wasn't an
    * ir_dereference_variable.  That is invalid, and it should be impossible.
    */
   unreachable("invalid dereference tree");
   return false;
}

static const char *
interface_field_name(void *mem_ctx, char *base_name, ir_dereference *d,
                     ir_rvalue **nonconst_block_index)
{
   ir_rvalue *previous_index = NULL;
   *nonconst_block_index = NULL;

   while (d != NULL) {
      switch (d->ir_type) {
      case ir_type_dereference_variable: {
         ir_dereference_variable *v = (ir_dereference_variable *) d;
         if (previous_index
             && v->var->is_interface_instance()
             && v->var->type->is_array()) {
            ir_constant *const_index = previous_index->as_constant();
            if (!const_index) {
               *nonconst_block_index = previous_index;
               return ralloc_asprintf(mem_ctx, "%s[0]", base_name);
            } else {
               return ralloc_asprintf(mem_ctx,
                                      "%s[%d]",
                                      base_name,
                                      const_index->get_uint_component(0));
            }
         } else {
            return base_name;
         }
         break;
      }

      case ir_type_dereference_record: {
         ir_dereference_record *r = (ir_dereference_record *) d;
         d = r->record->as_dereference();
         break;
      }

      case ir_type_dereference_array: {
         ir_dereference_array *a = (ir_dereference_array *) d;
         d = a->array->as_dereference();
         previous_index = a->array_index;
         break;
      }

      default:
         assert(!"Should not get here.");
         break;
      }
   }

   assert(!"Should not get here.");
   return NULL;
}

unsigned
lower_ssbo_unsized_array_length_visitor::calculate_unsized_array_stride(ir_dereference *deref)
{
   assert(deref->ir_type == ir_type_dereference_variable);
   ir_dereference_variable *deref_var = (ir_dereference_variable *)deref;
   const struct glsl_type *unsized_array_type = deref_var->var->type->fields.array;//->fields.
   unsigned array_stride = 0;
   bool row_major = is_dereferenced_thing_row_major(deref);

   /* FIXME: Do we need matrix_columns here?? and in the other place of this file? 
    * Can I reuse this function in process()??
    */
   int matrix_columns = 1;

   if (unsized_array_type->is_matrix() && row_major) {
      /* When loading a vector out of a row major matrix, the
      * step between the columns (vectors) is the size of a
      * float, while the step between the rows (elements of a
      * vector) is handled below in emit_ubo_loads.
      */
      array_stride = 4;
      if (unsized_array_type->is_double())
         array_stride *= 2;
      matrix_columns = unsized_array_type->matrix_columns;
   } else {
      /* Whether or not the field is row-major (because it might be a
      * bvec2 or something) does not affect the array itself. We need
      * to know whether an array element in its entirety is row-major.
      */
      const bool array_row_major =
         is_dereferenced_thing_row_major(deref_var);

      array_stride = unsized_array_type->std140_size(array_row_major);
      array_stride = glsl_align(array_stride, 16);
   }
   return array_stride;
}

ir_expression *
lower_ssbo_unsized_array_length_visitor::process(ir_rvalue **rvalue,
                                                 ir_dereference *deref,
                                                 ir_variable *var)
{
   mem_ctx = ralloc_parent(*rvalue);

   /* Find out the name of the interface block*/
   ir_rvalue *nonconst_block_index;
   const char *const field_name =
      interface_field_name(mem_ctx, (char *) var->get_interface_type()->name,
                           deref, &nonconst_block_index);

   /* Locate the ssbo block by interface name */
   this->uniform_block = NULL;
   for (unsigned i = 0; i < shader->NumUniformBlocks; i++) {
      if (strcmp(field_name, shader->UniformBlocks[i].Name) == 0) {

         ir_constant *index = new(mem_ctx) ir_constant(i);

         if (nonconst_block_index) {
            if (nonconst_block_index->type != glsl_type::uint_type)
               nonconst_block_index = i2u(nonconst_block_index);
            this->uniform_block = add(nonconst_block_index, index);
         } else {
            this->uniform_block = index;
         }

         assert(shader->UniformBlocks[i].IsBuffer);

         struct gl_uniform_block *block = &shader->UniformBlocks[i];

         this->ubo_var = var->is_interface_instance()
            ? &block->Uniforms[0] : &block->Uniforms[var->data.location];

         break;
      }
   }

   assert(this->uniform_block);

   ir_rvalue *offset = new(mem_ctx) ir_constant(0u);
   unsigned const_offset = 0;
   bool row_major = is_dereferenced_thing_row_major(deref);
   int matrix_columns = 1;
   unsigned unsized_array_stride = calculate_unsized_array_stride(deref);

   /* Calculate the offset to the start of the region of the UBO
    * dereferenced by *rvalue.  This may be a variable offset if an
    * array dereference has a variable index.
    */
   while (deref) {
      switch (deref->ir_type) {
      case ir_type_dereference_variable: {
         const_offset += ubo_var->Offset;
         deref = NULL;
         break;
      }

      case ir_type_dereference_array: {
         ir_dereference_array *deref_array = (ir_dereference_array *) deref;
         unsigned array_stride;
         if (deref_array->array->type->is_matrix() && row_major) {
            /* When loading a vector out of a row major matrix, the
             * step between the columns (vectors) is the size of a
             * float, while the step between the rows (elements of a
             * vector) is handled below in emit_ubo_loads.
             */
            array_stride = 4;
            if (deref_array->array->type->is_double())
               array_stride *= 2;
            matrix_columns = deref_array->array->type->matrix_columns;
         } else if (deref_array->type->is_interface()) {
            /* We're processing an array dereference of an interface instance
             * array. The thing being dereferenced *must* be a variable
             * dereference because interfaces cannot be embedded in other
             * types. In terms of calculating the offsets for the lowering
             * pass, we don't care about the array index. All elements of an
             * interface instance array will have the same offsets relative to
             * the base of the block that backs them.
             */
            assert(deref_array->array->as_dereference_variable());
            deref = deref_array->array->as_dereference();
            break;
         } else {
            /* Whether or not the field is row-major (because it might be a
             * bvec2 or something) does not affect the array itself. We need
             * to know whether an array element in its entirety is row-major.
             */
            const bool array_row_major =
               is_dereferenced_thing_row_major(deref_array);

            array_stride = deref_array->type->std140_size(array_row_major);
            array_stride = glsl_align(array_stride, 16);
         }

         ir_rvalue *array_index = deref_array->array_index;
         if (array_index->type->base_type == GLSL_TYPE_INT)
            array_index = i2u(array_index);

         ir_constant *const_index =
            array_index->constant_expression_value(NULL);
         if (const_index) {
            const_offset += array_stride * const_index->value.u[0];
         } else {
            offset = add(offset,
                         mul(array_index,
                             new(mem_ctx) ir_constant(array_stride)));
         }
         deref = deref_array->array->as_dereference();
         break;
      }

      case ir_type_dereference_record: {
         ir_dereference_record *deref_record = (ir_dereference_record *) deref;
         const glsl_type *struct_type = deref_record->record->type;
         unsigned intra_struct_offset = 0;

         for (unsigned int i = 0; i < struct_type->length; i++) {
            const glsl_type *type = struct_type->fields.structure[i].type;

            ir_dereference_record *field_deref = new(mem_ctx)
               ir_dereference_record(deref_record->record,
                                     struct_type->fields.structure[i].name);
            const bool field_row_major =
               is_dereferenced_thing_row_major(field_deref);

            ralloc_free(field_deref);

            unsigned field_align = type->std140_base_alignment(field_row_major);

            intra_struct_offset = glsl_align(intra_struct_offset, field_align);

            if (strcmp(struct_type->fields.structure[i].name,
                       deref_record->field) == 0)
               break;

            intra_struct_offset += type->std140_size(field_row_major);

            /* If the field just examined was itself a structure, apply rule
             * #9:
             *
             *     "The structure may have padding at the end; the base offset
             *     of the member following the sub-structure is rounded up to
             *     the next multiple of the base alignment of the structure."
             */
            if (type->without_array()->is_record()) {
               intra_struct_offset = glsl_align(intra_struct_offset,
                                                field_align);

            }
         }

         const_offset += intra_struct_offset;
         deref = deref_record->record->as_dereference();
         break;
      }

      default:
         assert(!"not reached");
         deref = NULL;
         break;
      }
   }

   /* Now that we've calculated the offset to the start of the
    * dereference, emit writes from the temporary to memory
    */
   ir_variable *write_offset = new(mem_ctx) ir_variable(glsl_type::uint_type,
                                                        "ssbo_write_temp_offset",
                                                        ir_var_temporary);
   base_ir->insert_after(write_offset);
   base_ir->insert_after(assign(write_offset, offset));

   ir_expression *new_ssbo = emit_ssbo_unsized_array_length(write_offset,
                                                            const_offset,
                                                            unsized_array_stride);

   return new_ssbo;
}

ir_visitor_status
lower_ssbo_unsized_array_length_visitor::visit_leave(ir_assignment *ir)
{
   if (!ir->rhs || ir->rhs->ir_type != ir_type_expression)
      return visit_continue;

   ir_expression *expr = (ir_expression *) ir->rhs;
   if (expr->operation == ir_expression_operation(ir_unop_ssbo_unsized_array_length)) {
      ir_rvalue *rvalue = expr->operands[0]->as_rvalue();
      if (!rvalue || !rvalue->type->is_array() || !rvalue->type->is_unsized_array())
         return visit_continue;

      ir_dereference *deref = expr->operands[0]->as_dereference();
      if (!deref)
         return visit_continue;

      ir_variable *var = expr->operands[0]->variable_referenced();
      if (!var || !var->is_in_shader_storage_block())
         return visit_continue;
      /* Now replace the unop instruction for the binop */
      ir_expression *temp = process(&rvalue, deref, var);
      /* FIXME: Is this 'delete' needed?*/
      delete expr;
      ir->rhs = temp;
      return visit_continue;
   }
   return visit_leave((ir_expression *)ir->rhs);
}

ir_visitor_status
lower_ssbo_unsized_array_length_visitor::visit_leave(ir_expression *former_ir)
{
   unsigned i = 0;

   if (former_ir->operation == ir_expression_operation(ir_unop_ssbo_unsized_array_length)) {
         /* Don't replace this unop if it is found alone. It is going to be
          * removed by the optimization passes or replaced if it is part of
          * an ir_assignment or another ir_expression.
          */
         return visit_continue;
   }

   for (i = 0; i < 4; i++) {
      if (!former_ir->operands[i] || former_ir->operands[0]->ir_type != ir_type_expression)
         continue;
      ir_expression *ir = (ir_expression *) former_ir->operands[i];
      if (ir->operation == ir_expression_operation(ir_unop_ssbo_unsized_array_length)) {
         ir_rvalue *rvalue = ir->operands[i]->as_rvalue();
         if (!rvalue || !rvalue->type->is_array() || !rvalue->type->is_unsized_array())
            return visit_continue;

         ir_dereference *deref = ir->operands[i]->as_dereference();
         if (!deref)
            return visit_continue;

         ir_variable *var = ir->operands[i]->variable_referenced();
         if (!var || !var->is_in_shader_storage_block())
            return visit_continue;
         /* Now replace the unop instruction for the binop */
         ir_expression *temp = process(&rvalue, deref, var);
         /* FIXME: Is this 'delete' needed?*/
         delete ir;
         former_ir->operands[i] = temp;
      }
   }
   return visit_continue;
}

void
lower_ssbo_unsized_array_length(struct gl_shader *shader, exec_list *instructions)
{
   lower_ssbo_unsized_array_length_visitor v(shader);
   visit_list_elements(&v, instructions);
}