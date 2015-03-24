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

#include "ir.h"
#include "ir_builder.h"
#include "ir_rvalue_visitor.h"
#include "main/macros.h"
#include "program/prog_instruction.h"

using namespace ir_builder;

/**
 * \file lower_ssbo_writes.cpp
 */

namespace {

class lower_ssbo_writes_visitor : public ir_hierarchical_visitor {
public:
   lower_ssbo_writes_visitor(struct gl_shader *shader)
      : shader(shader)
   {
   }

   virtual ir_visitor_status visit_leave(class ir_assignment *);
   void process(ir_rvalue **, ir_dereference *, ir_variable *, ir_variable *, unsigned write_mask);
   void emit_ssbo_writes(ir_dereference *deref, ir_variable *base_offset,
                         unsigned int deref_offset, bool row_major,
                         int matrix_columns, unsigned write_mask);
   ir_ssbo_store *ssbo_write(ir_rvalue *deref, ir_rvalue *offset, unsigned write_mask);

   void *mem_ctx;
   struct gl_shader *shader;
   struct gl_uniform_buffer_variable *ubo_var;
   ir_rvalue *uniform_block;
   bool progress;
};

} /* anonymous namespace */

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

ir_ssbo_store *
lower_ssbo_writes_visitor::ssbo_write(ir_rvalue *deref, ir_rvalue *offset, unsigned write_mask)
{
   /* FIXME: We probably need type information */
   ir_rvalue *block_ref = this->uniform_block->clone(mem_ctx, NULL);
   ir_rvalue *val_ref = deref->clone(mem_ctx, NULL);
   return new(mem_ctx) ir_ssbo_store(block_ref, offset, val_ref, write_mask);
}

void
lower_ssbo_writes_visitor::emit_ssbo_writes(ir_dereference *deref,
                                            ir_variable *base_offset,
                                            unsigned int deref_offset,
                                            bool row_major,
                                            int matrix_columns,
                                            unsigned write_mask)
{
   assert(deref->type->is_scalar() || deref->type->is_vector());

   if (!row_major) {
      ir_rvalue *offset =
         add(base_offset, new(mem_ctx) ir_constant(deref_offset));
      base_ir->insert_after(ssbo_write(deref, offset, write_mask));
   } else {
      assert("Not implemented yet!");
      unsigned N = deref->type->is_double() ? 8 : 4;

      /* We're dereffing a column out of a row-major matrix, so we
       * gather the vector from each stored row.
      */
      assert(deref->type->base_type == GLSL_TYPE_FLOAT ||
             deref->type->base_type == GLSL_TYPE_DOUBLE);
      /* Matrices, row_major or not, are stored as if they were
       * arrays of vectors of the appropriate size in std140.
       * Arrays have their strides rounded up to a vec4, so the
       * matrix stride is always 16. However a double matrix may either be 16
       * or 32 depending on the number of columns.
       */
      assert(matrix_columns <= 4);
      unsigned matrix_stride = glsl_align(matrix_columns * N, 16);

      for (unsigned i = 0; i < deref->type->vector_elements; i++) {
         ir_rvalue *chan_offset =
            add(base_offset,
                new(mem_ctx) ir_constant(deref_offset + i * matrix_stride));

         base_ir->insert_after(ssbo_write(deref, chan_offset, write_mask));
      }
   }
}

void
lower_ssbo_writes_visitor::process(ir_rvalue **rvalue, ir_dereference *deref,
                                   ir_variable *var, ir_variable *write_var,
                                   unsigned write_mask)
{
   mem_ctx = ralloc_parent(*rvalue);

   /* Finx out the name of the interface block*/
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
   base_ir->insert_before(write_offset);
   base_ir->insert_before(assign(write_offset, offset));

   deref = new(mem_ctx) ir_dereference_variable(write_var);
   emit_ssbo_writes(deref, write_offset, const_offset, row_major, matrix_columns, write_mask);
}

ir_visitor_status
lower_ssbo_writes_visitor::visit_leave(ir_assignment *ir)
{
   if (!ir || !ir->lhs)
      return visit_continue;

   ir_rvalue *rvalue = ir->lhs->as_rvalue();
   if (!rvalue)
      return visit_continue;

   ir_dereference *deref = ir->lhs->as_dereference();
   if (!deref)
      return visit_continue;

   ir_variable *var = ir->lhs->variable_referenced();
   if (!var || !var->is_in_uniform_block())
      return visit_continue;

   /* Write the result of the assignment to a temporary */
   const glsl_type *type = rvalue->type;
   ir_variable *write_var = new(mem_ctx) ir_variable(type,
                                                     "ssbo_write_temp",
                                                     ir_var_temporary);
   base_ir->insert_before(write_var);
   ir->lhs = new(mem_ctx) ir_dereference_variable(write_var);

   /* Now write from the temporary to memory */
   process(&rvalue, deref, var, write_var, ir->write_mask);

   return visit_continue;
}

void
lower_ssbo_writes(struct gl_shader *shader, exec_list *instructions)
{
   lower_ssbo_writes_visitor v(shader);
   visit_list_elements(&v, instructions);
}
