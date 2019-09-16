/*
 * Copyright © 2014 Connor Abbott
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
 *
 * Authors:
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#ifndef NIR_H
#define NIR_H

#include "util/hash_table.h"
#include "compiler/glsl/list.h"
#include "GL/gl.h" /* GLenum */
#include "util/list.h"
#include "util/ralloc.h"
#include "util/set.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/macros.h"
#include "compiler/nir_types.h"
#include "compiler/shader_enums.h"
#include "compiler/shader_info.h"
#include <stdio.h>

#ifndef NDEBUG
#include "util/debug.h"
#endif /* NDEBUG */

#include "nir_opcodes.h"

#if defined(_WIN32) && !defined(snprintf)
#define snprintf _snprintf
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NIR_FALSE 0u
#define NIR_TRUE (~0u)
#define NIR_MAX_VEC_COMPONENTS 4
#define NIR_MAX_MATRIX_COLUMNS 4
typedef uint8_t nir_component_mask_t;

/** Defines a cast function
 *
 * This macro defines a cast function from in_type to out_type where
 * out_type is some structure type that contains a field of type out_type.
 *
 * Note that you have to be a bit careful as the generated cast function
 * destroys constness.
 */
#define NIR_DEFINE_CAST(name, in_type, out_type, field, \
                        type_field, type_value)         \
static inline out_type *                                \
name(const in_type *parent)                             \
{                                                       \
   assert(parent && parent->type_field == type_value);  \
   return exec_node_data(out_type, parent, field);      \
}

struct nir_function;
struct nir_shader;
struct nir_instr;
struct nir_builder;


/**
 * Description of built-in state associated with a uniform
 *
 * \sa nir_variable::state_slots
 */
typedef struct {
   gl_state_index16 tokens[STATE_LENGTH];
   int swizzle;
} nir_state_slot;

typedef enum {
   nir_var_shader_in       = (1 << 0),
   nir_var_shader_out      = (1 << 1),
   nir_var_shader_temp     = (1 << 2),
   nir_var_function_temp   = (1 << 3),
   nir_var_uniform         = (1 << 4),
   nir_var_mem_ubo         = (1 << 5),
   nir_var_system_value    = (1 << 6),
   nir_var_mem_ssbo        = (1 << 7),
   nir_var_mem_shared      = (1 << 8),
   nir_var_mem_global      = (1 << 9),
   nir_var_mem_push_const  = (1 << 10), /* not actually used for variables */
   nir_num_variable_modes  = 11,
   nir_var_all             = (1 << nir_num_variable_modes) - 1,
} nir_variable_mode;

/**
 * Rounding modes.
 */
typedef enum {
   nir_rounding_mode_undef = 0,
   nir_rounding_mode_rtne  = 1, /* round to nearest even */
   nir_rounding_mode_ru    = 2, /* round up */
   nir_rounding_mode_rd    = 3, /* round down */
   nir_rounding_mode_rtz   = 4, /* round towards zero */
} nir_rounding_mode;

typedef union {
   bool b;
   float f32;
   double f64;
   int8_t i8;
   uint8_t u8;
   int16_t i16;
   uint16_t u16;
   int32_t i32;
   uint32_t u32;
   int64_t i64;
   uint64_t u64;
} nir_const_value;

#define nir_const_value_to_array(arr, c, components, m) \
{ \
   for (unsigned i = 0; i < components; ++i) \
      arr[i] = c[i].m; \
} while (false)

static inline nir_const_value
nir_const_value_for_raw_uint(uint64_t x, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   switch (bit_size) {
   case 1:  v.b   = x;  break;
   case 8:  v.u8  = x;  break;
   case 16: v.u16 = x;  break;
   case 32: v.u32 = x;  break;
   case 64: v.u64 = x;  break;
   default:
      unreachable("Invalid bit size");
   }

   return v;
}

static inline nir_const_value
nir_const_value_for_int(int64_t i, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   assert(bit_size <= 64);
   if (bit_size < 64) {
      assert(i >= (-(1ll << (bit_size - 1))));
      assert(i < (1ll << (bit_size - 1)));
   }

   return nir_const_value_for_raw_uint(i, bit_size);
}

static inline nir_const_value
nir_const_value_for_uint(uint64_t u, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   assert(bit_size <= 64);
   if (bit_size < 64)
      assert(u < (1ull << bit_size));

   return nir_const_value_for_raw_uint(u, bit_size);
}

static inline nir_const_value
nir_const_value_for_bool(bool b, unsigned bit_size)
{
   /* Booleans use a 0/-1 convention */
   return nir_const_value_for_int(-(int)b, bit_size);
}

/* This one isn't inline because it requires half-float conversion */
nir_const_value nir_const_value_for_float(double b, unsigned bit_size);

static inline int64_t
nir_const_value_as_int(nir_const_value value, unsigned bit_size)
{
   switch (bit_size) {
   /* int1_t uses 0/-1 convention */
   case 1:  return -(int)value.b;
   case 8:  return value.i8;
   case 16: return value.i16;
   case 32: return value.i32;
   case 64: return value.i64;
   default:
      unreachable("Invalid bit size");
   }
}

static inline int64_t
nir_const_value_as_uint(nir_const_value value, unsigned bit_size)
{
   switch (bit_size) {
   case 1:  return value.b;
   case 8:  return value.u8;
   case 16: return value.u16;
   case 32: return value.u32;
   case 64: return value.u64;
   default:
      unreachable("Invalid bit size");
   }
}

static inline bool
nir_const_value_as_bool(nir_const_value value, unsigned bit_size)
{
   int64_t i = nir_const_value_as_int(value, bit_size);

   /* Booleans of any size use 0/-1 convention */
   assert(i == 0 || i == -1);

   return i;
}

/* This one isn't inline because it requires half-float conversion */
double nir_const_value_as_float(nir_const_value value, unsigned bit_size);

typedef struct nir_constant {
   /**
    * Value of the constant.
    *
    * The field used to back the values supplied by the constant is determined
    * by the type associated with the \c nir_variable.  Constants may be
    * scalars, vectors, or matrices.
    */
   nir_const_value values[NIR_MAX_VEC_COMPONENTS];

   /* we could get this from the var->type but makes clone *much* easier to
    * not have to care about the type.
    */
   unsigned num_elements;

   /* Array elements / Structure Fields */
   struct nir_constant **elements;
} nir_constant;

/**
 * \brief Layout qualifiers for gl_FragDepth.
 *
 * The AMD/ARB_conservative_depth extensions allow gl_FragDepth to be redeclared
 * with a layout qualifier.
 */
typedef enum {
    nir_depth_layout_none, /**< No depth layout is specified. */
    nir_depth_layout_any,
    nir_depth_layout_greater,
    nir_depth_layout_less,
    nir_depth_layout_unchanged
} nir_depth_layout;

/**
 * Enum keeping track of how a variable was declared.
 */
typedef enum {
   /**
    * Normal declaration.
    */
   nir_var_declared_normally = 0,

   /**
    * Variable is implicitly generated by the compiler and should not be
    * visible via the API.
    */
   nir_var_hidden,
} nir_var_declaration_type;

/**
 * Either a uniform, global variable, shader input, or shader output. Based on
 * ir_variable - it should be easy to translate between the two.
 */

typedef struct nir_variable {
   struct exec_node node;

   /**
    * Declared type of the variable
    */
   const struct glsl_type *type;

   /**
    * Declared name of the variable
    */
   char *name;

   struct nir_variable_data {
      /**
       * Storage class of the variable.
       *
       * \sa nir_variable_mode
       */
      nir_variable_mode mode;

      /**
       * Is the variable read-only?
       *
       * This is set for variables declared as \c const, shader inputs,
       * and uniforms.
       */
      unsigned read_only:1;
      unsigned centroid:1;
      unsigned sample:1;
      unsigned patch:1;
      unsigned invariant:1;

      /**
       * Can this variable be coalesced with another?
       *
       * This is set by nir_lower_io_to_temporaries to say that any
       * copies involving this variable should stay put. Propagating it can
       * duplicate the resulting load/store, which is not wanted, and may
       * result in a load/store of the variable with an indirect offset which
       * the backend may not be able to handle.
       */
      unsigned cannot_coalesce:1;

      /**
       * When separate shader programs are enabled, only input/outputs between
       * the stages of a multi-stage separate program can be safely removed
       * from the shader interface. Other input/outputs must remains active.
       *
       * This is also used to make sure xfb varyings that are unused by the
       * fragment shader are not removed.
       */
      unsigned always_active_io:1;

      /**
       * Interpolation mode for shader inputs / outputs
       *
       * \sa glsl_interp_mode
       */
      unsigned interpolation:2;

      /**
       * If non-zero, then this variable may be packed along with other variables
       * into a single varying slot, so this offset should be applied when
       * accessing components.  For example, an offset of 1 means that the x
       * component of this variable is actually stored in component y of the
       * location specified by \c location.
       */
      unsigned location_frac:2;

      /**
       * If true, this variable represents an array of scalars that should
       * be tightly packed.  In other words, consecutive array elements
       * should be stored one component apart, rather than one slot apart.
       */
      unsigned compact:1;

      /**
       * Whether this is a fragment shader output implicitly initialized with
       * the previous contents of the specified render target at the
       * framebuffer location corresponding to this shader invocation.
       */
      unsigned fb_fetch_output:1;

      /**
       * Non-zero if this variable is considered bindless as defined by
       * ARB_bindless_texture.
       */
      unsigned bindless:1;

      /**
       * Was an explicit binding set in the shader?
       */
      unsigned explicit_binding:1;

      /**
       * Was a transfer feedback buffer set in the shader?
       */
      unsigned explicit_xfb_buffer:1;

      /**
       * Was a transfer feedback stride set in the shader?
       */
      unsigned explicit_xfb_stride:1;

      /**
       * Was an explicit offset set in the shader?
       */
      unsigned explicit_offset:1;

      /**
       * \brief Layout qualifier for gl_FragDepth.
       *
       * This is not equal to \c ir_depth_layout_none if and only if this
       * variable is \c gl_FragDepth and a layout qualifier is specified.
       */
      nir_depth_layout depth_layout;

      /**
       * Storage location of the base of this variable
       *
       * The precise meaning of this field depends on the nature of the variable.
       *
       *   - Vertex shader input: one of the values from \c gl_vert_attrib.
       *   - Vertex shader output: one of the values from \c gl_varying_slot.
       *   - Geometry shader input: one of the values from \c gl_varying_slot.
       *   - Geometry shader output: one of the values from \c gl_varying_slot.
       *   - Fragment shader input: one of the values from \c gl_varying_slot.
       *   - Fragment shader output: one of the values from \c gl_frag_result.
       *   - Uniforms: Per-stage uniform slot number for default uniform block.
       *   - Uniforms: Index within the uniform block definition for UBO members.
       *   - Non-UBO Uniforms: uniform slot number.
       *   - Other: This field is not currently used.
       *
       * If the variable is a uniform, shader input, or shader output, and the
       * slot has not been assigned, the value will be -1.
       */
      int location;

      /**
       * The actual location of the variable in the IR. Only valid for inputs
       * and outputs.
       */
      unsigned int driver_location;

      /**
       * Vertex stream output identifier.
       *
       * For packed outputs, bit 31 is set and bits [2*i+1,2*i] indicate the
       * stream of the i-th component.
       */
      unsigned stream;

      /**
       * output index for dual source blending.
       */
      int index;

      /**
       * Descriptor set binding for sampler or UBO.
       */
      int descriptor_set;

      /**
       * Initial binding point for a sampler or UBO.
       *
       * For array types, this represents the binding point for the first element.
       */
      int binding;

      /**
       * Location an atomic counter or transform feedback is stored at.
       */
      unsigned offset;

      /**
       * Transform feedback buffer.
       */
      unsigned xfb_buffer;

      /**
       * Transform feedback stride.
       */
      unsigned xfb_stride;

      /**
       * How the variable was declared.  See nir_var_declaration_type.
       *
       * This is used to detect variables generated by the compiler, so should
       * not be visible via the API.
       */
      unsigned how_declared:2;

      /**
       * ARB_shader_image_load_store qualifiers.
       */
      struct {
         enum gl_access_qualifier access;

         /** Image internal format if specified explicitly, otherwise GL_NONE. */
         GLenum format;
      } image;
   } data;

   /**
    * Built-in state that backs this uniform
    *
    * Once set at variable creation, \c state_slots must remain invariant.
    * This is because, ideally, this array would be shared by all clones of
    * this variable in the IR tree.  In other words, we'd really like for it
    * to be a fly-weight.
    *
    * If the variable is not a uniform, \c num_state_slots will be zero and
    * \c state_slots will be \c NULL.
    */
   /*@{*/
   unsigned num_state_slots;    /**< Number of state slots used */
   nir_state_slot *state_slots;  /**< State descriptors. */
   /*@}*/

   /**
    * Constant expression assigned in the initializer of the variable
    *
    * This field should only be used temporarily by creators of NIR shaders
    * and then lower_constant_initializers can be used to get rid of them.
    * Most of the rest of NIR ignores this field or asserts that it's NULL.
    */
   nir_constant *constant_initializer;

   /**
    * For variables that are in an interface block or are an instance of an
    * interface block, this is the \c GLSL_TYPE_INTERFACE type for that block.
    *
    * \sa ir_variable::location
    */
   const struct glsl_type *interface_type;

   /**
    * Description of per-member data for per-member struct variables
    *
    * This is used for variables which are actually an amalgamation of
    * multiple entities such as a struct of built-in values or a struct of
    * inputs each with their own layout specifier.  This is only allowed on
    * variables with a struct or array of array of struct type.
    */
   unsigned num_members;
   struct nir_variable_data *members;
} nir_variable;

#define nir_foreach_variable(var, var_list) \
   foreach_list_typed(nir_variable, var, node, var_list)

#define nir_foreach_variable_safe(var, var_list) \
   foreach_list_typed_safe(nir_variable, var, node, var_list)

static inline bool
nir_variable_is_global(const nir_variable *var)
{
   return var->data.mode != nir_var_function_temp;
}

typedef struct nir_register {
   struct exec_node node;

   unsigned num_components; /** < number of vector components */
   unsigned num_array_elems; /** < size of array (0 for no array) */

   /* The bit-size of each channel; must be one of 8, 16, 32, or 64 */
   uint8_t bit_size;

   /** generic register index. */
   unsigned index;

   /** only for debug purposes, can be NULL */
   const char *name;

   /** set of nir_srcs where this register is used (read from) */
   struct list_head uses;

   /** set of nir_dests where this register is defined (written to) */
   struct list_head defs;

   /** set of nir_ifs where this register is used as a condition */
   struct list_head if_uses;
} nir_register;

#define nir_foreach_register(reg, reg_list) \
   foreach_list_typed(nir_register, reg, node, reg_list)
#define nir_foreach_register_safe(reg, reg_list) \
   foreach_list_typed_safe(nir_register, reg, node, reg_list)

typedef enum PACKED {
   nir_instr_type_alu,
   nir_instr_type_deref,
   nir_instr_type_call,
   nir_instr_type_tex,
   nir_instr_type_intrinsic,
   nir_instr_type_load_const,
   nir_instr_type_jump,
   nir_instr_type_ssa_undef,
   nir_instr_type_phi,
   nir_instr_type_parallel_copy,
} nir_instr_type;

typedef struct nir_instr {
   struct exec_node node;
   struct nir_block *block;
   nir_instr_type type;

   /* A temporary for optimization and analysis passes to use for storing
    * flags.  For instance, DCE uses this to store the "dead/live" info.
    */
   uint8_t pass_flags;

   /** generic instruction index. */
   unsigned index;
} nir_instr;

static inline nir_instr *
nir_instr_next(nir_instr *instr)
{
   struct exec_node *next = exec_node_get_next(&instr->node);
   if (exec_node_is_tail_sentinel(next))
      return NULL;
   else
      return exec_node_data(nir_instr, next, node);
}

static inline nir_instr *
nir_instr_prev(nir_instr *instr)
{
   struct exec_node *prev = exec_node_get_prev(&instr->node);
   if (exec_node_is_head_sentinel(prev))
      return NULL;
   else
      return exec_node_data(nir_instr, prev, node);
}

static inline bool
nir_instr_is_first(const nir_instr *instr)
{
   return exec_node_is_head_sentinel(exec_node_get_prev_const(&instr->node));
}

static inline bool
nir_instr_is_last(const nir_instr *instr)
{
   return exec_node_is_tail_sentinel(exec_node_get_next_const(&instr->node));
}

typedef struct nir_ssa_def {
   /** for debugging only, can be NULL */
   const char* name;

   /** generic SSA definition index. */
   unsigned index;

   /** Index into the live_in and live_out bitfields */
   unsigned live_index;

   /** Instruction which produces this SSA value. */
   nir_instr *parent_instr;

   /** set of nir_instrs where this register is used (read from) */
   struct list_head uses;

   /** set of nir_ifs where this register is used as a condition */
   struct list_head if_uses;

   uint8_t num_components;

   /* The bit-size of each channel; must be one of 8, 16, 32, or 64 */
   uint8_t bit_size;
} nir_ssa_def;

struct nir_src;

typedef struct {
   nir_register *reg;
   struct nir_src *indirect; /** < NULL for no indirect offset */
   unsigned base_offset;

   /* TODO use-def chain goes here */
} nir_reg_src;

typedef struct {
   nir_instr *parent_instr;
   struct list_head def_link;

   nir_register *reg;
   struct nir_src *indirect; /** < NULL for no indirect offset */
   unsigned base_offset;

   /* TODO def-use chain goes here */
} nir_reg_dest;

struct nir_if;

typedef struct nir_src {
   union {
      /** Instruction that consumes this value as a source. */
      nir_instr *parent_instr;
      struct nir_if *parent_if;
   };

   struct list_head use_link;

   union {
      nir_reg_src reg;
      nir_ssa_def *ssa;
   };

   bool is_ssa;
} nir_src;

static inline nir_src
nir_src_init(void)
{
   nir_src src = { { NULL } };
   return src;
}

#define NIR_SRC_INIT nir_src_init()

#define nir_foreach_use(src, reg_or_ssa_def) \
   list_for_each_entry(nir_src, src, &(reg_or_ssa_def)->uses, use_link)

#define nir_foreach_use_safe(src, reg_or_ssa_def) \
   list_for_each_entry_safe(nir_src, src, &(reg_or_ssa_def)->uses, use_link)

#define nir_foreach_if_use(src, reg_or_ssa_def) \
   list_for_each_entry(nir_src, src, &(reg_or_ssa_def)->if_uses, use_link)

#define nir_foreach_if_use_safe(src, reg_or_ssa_def) \
   list_for_each_entry_safe(nir_src, src, &(reg_or_ssa_def)->if_uses, use_link)

typedef struct {
   union {
      nir_reg_dest reg;
      nir_ssa_def ssa;
   };

   bool is_ssa;
} nir_dest;

static inline nir_dest
nir_dest_init(void)
{
   nir_dest dest = { { { NULL } } };
   return dest;
}

#define NIR_DEST_INIT nir_dest_init()

#define nir_foreach_def(dest, reg) \
   list_for_each_entry(nir_dest, dest, &(reg)->defs, reg.def_link)

#define nir_foreach_def_safe(dest, reg) \
   list_for_each_entry_safe(nir_dest, dest, &(reg)->defs, reg.def_link)

static inline nir_src
nir_src_for_ssa(nir_ssa_def *def)
{
   nir_src src = NIR_SRC_INIT;

   src.is_ssa = true;
   src.ssa = def;

   return src;
}

static inline nir_src
nir_src_for_reg(nir_register *reg)
{
   nir_src src = NIR_SRC_INIT;

   src.is_ssa = false;
   src.reg.reg = reg;
   src.reg.indirect = NULL;
   src.reg.base_offset = 0;

   return src;
}

static inline nir_dest
nir_dest_for_reg(nir_register *reg)
{
   nir_dest dest = NIR_DEST_INIT;

   dest.reg.reg = reg;

   return dest;
}

static inline unsigned
nir_src_bit_size(nir_src src)
{
   return src.is_ssa ? src.ssa->bit_size : src.reg.reg->bit_size;
}

static inline unsigned
nir_src_num_components(nir_src src)
{
   return src.is_ssa ? src.ssa->num_components : src.reg.reg->num_components;
}

static inline bool
nir_src_is_const(nir_src src)
{
   return src.is_ssa &&
          src.ssa->parent_instr->type == nir_instr_type_load_const;
}

static inline unsigned
nir_dest_bit_size(nir_dest dest)
{
   return dest.is_ssa ? dest.ssa.bit_size : dest.reg.reg->bit_size;
}

static inline unsigned
nir_dest_num_components(nir_dest dest)
{
   return dest.is_ssa ? dest.ssa.num_components : dest.reg.reg->num_components;
}

void nir_src_copy(nir_src *dest, const nir_src *src, void *instr_or_if);
void nir_dest_copy(nir_dest *dest, const nir_dest *src, nir_instr *instr);

typedef struct {
   nir_src src;

   /**
    * \name input modifiers
    */
   /*@{*/
   /**
    * For inputs interpreted as floating point, flips the sign bit. For
    * inputs interpreted as integers, performs the two's complement negation.
    */
   bool negate;

   /**
    * Clears the sign bit for floating point values, and computes the integer
    * absolute value for integers. Note that the negate modifier acts after
    * the absolute value modifier, therefore if both are set then all inputs
    * will become negative.
    */
   bool abs;
   /*@}*/

   /**
    * For each input component, says which component of the register it is
    * chosen from. Note that which elements of the swizzle are used and which
    * are ignored are based on the write mask for most opcodes - for example,
    * a statement like "foo.xzw = bar.zyx" would have a writemask of 1101b and
    * a swizzle of {2, x, 1, 0} where x means "don't care."
    */
   uint8_t swizzle[NIR_MAX_VEC_COMPONENTS];
} nir_alu_src;

typedef struct {
   nir_dest dest;

   /**
    * \name saturate output modifier
    *
    * Only valid for opcodes that output floating-point numbers. Clamps the
    * output to between 0.0 and 1.0 inclusive.
    */

   bool saturate;

   unsigned write_mask : NIR_MAX_VEC_COMPONENTS; /* ignored if dest.is_ssa is true */
} nir_alu_dest;

/** NIR sized and unsized types
 *
 * The values in this enum are carefully chosen so that the sized type is
 * just the unsized type OR the number of bits.
 */
typedef enum {
   nir_type_invalid = 0, /* Not a valid type */
   nir_type_int =       2,
   nir_type_uint =      4,
   nir_type_bool =      6,
   nir_type_float =     128,
   nir_type_bool1 =     1  | nir_type_bool,
   nir_type_bool32 =    32 | nir_type_bool,
   nir_type_int1 =      1  | nir_type_int,
   nir_type_int8 =      8  | nir_type_int,
   nir_type_int16 =     16 | nir_type_int,
   nir_type_int32 =     32 | nir_type_int,
   nir_type_int64 =     64 | nir_type_int,
   nir_type_uint1 =     1  | nir_type_uint,
   nir_type_uint8 =     8  | nir_type_uint,
   nir_type_uint16 =    16 | nir_type_uint,
   nir_type_uint32 =    32 | nir_type_uint,
   nir_type_uint64 =    64 | nir_type_uint,
   nir_type_float16 =   16 | nir_type_float,
   nir_type_float32 =   32 | nir_type_float,
   nir_type_float64 =   64 | nir_type_float,
} nir_alu_type;

#define NIR_ALU_TYPE_SIZE_MASK 0x79
#define NIR_ALU_TYPE_BASE_TYPE_MASK 0x86

static inline unsigned
nir_alu_type_get_type_size(nir_alu_type type)
{
   return type & NIR_ALU_TYPE_SIZE_MASK;
}

static inline unsigned
nir_alu_type_get_base_type(nir_alu_type type)
{
   return type & NIR_ALU_TYPE_BASE_TYPE_MASK;
}

static inline nir_alu_type
nir_get_nir_type_for_glsl_base_type(enum glsl_base_type base_type)
{
   switch (base_type) {
   case GLSL_TYPE_BOOL:
      return nir_type_bool1;
      break;
   case GLSL_TYPE_UINT:
      return nir_type_uint32;
      break;
   case GLSL_TYPE_INT:
      return nir_type_int32;
      break;
   case GLSL_TYPE_UINT16:
      return nir_type_uint16;
      break;
   case GLSL_TYPE_INT16:
      return nir_type_int16;
      break;
   case GLSL_TYPE_UINT8:
      return nir_type_uint8;
   case GLSL_TYPE_INT8:
      return nir_type_int8;
   case GLSL_TYPE_UINT64:
      return nir_type_uint64;
      break;
   case GLSL_TYPE_INT64:
      return nir_type_int64;
      break;
   case GLSL_TYPE_FLOAT:
      return nir_type_float32;
      break;
   case GLSL_TYPE_FLOAT16:
      return nir_type_float16;
      break;
   case GLSL_TYPE_DOUBLE:
      return nir_type_float64;
      break;

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_ARRAY:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_SUBROUTINE:
   case GLSL_TYPE_FUNCTION:
   case GLSL_TYPE_ERROR:
      return nir_type_invalid;
   }

   unreachable("unknown type");
}

static inline nir_alu_type
nir_get_nir_type_for_glsl_type(const struct glsl_type *type)
{
   return nir_get_nir_type_for_glsl_base_type(glsl_get_base_type(type));
}

nir_op nir_type_conversion_op(nir_alu_type src, nir_alu_type dst,
                              nir_rounding_mode rnd);

static inline nir_op
nir_op_vec(unsigned components)
{
   switch (components) {
   case  1: return nir_op_mov;
   case  2: return nir_op_vec2;
   case  3: return nir_op_vec3;
   case  4: return nir_op_vec4;
   default: unreachable("bad component count");
   }
}

typedef enum {
   /**
    * Operation where the first two sources are commutative.
    *
    * For 2-source operations, this just mathematical commutativity.  Some
    * 3-source operations, like ffma, are only commutative in the first two
    * sources.
    */
   NIR_OP_IS_2SRC_COMMUTATIVE = (1 << 0),
   NIR_OP_IS_ASSOCIATIVE = (1 << 1),
} nir_op_algebraic_property;

typedef struct {
   const char *name;

   unsigned num_inputs;

   /**
    * The number of components in the output
    *
    * If non-zero, this is the size of the output and input sizes are
    * explicitly given; swizzle and writemask are still in effect, but if
    * the output component is masked out, then the input component may
    * still be in use.
    *
    * If zero, the opcode acts in the standard, per-component manner; the
    * operation is performed on each component (except the ones that are
    * masked out) with the input being taken from the input swizzle for
    * that component.
    *
    * The size of some of the inputs may be given (i.e. non-zero) even
    * though output_size is zero; in that case, the inputs with a zero
    * size act per-component, while the inputs with non-zero size don't.
    */
   unsigned output_size;

   /**
    * The type of vector that the instruction outputs. Note that the
    * staurate modifier is only allowed on outputs with the float type.
    */

   nir_alu_type output_type;

   /**
    * The number of components in each input
    */
   unsigned input_sizes[NIR_MAX_VEC_COMPONENTS];

   /**
    * The type of vector that each input takes. Note that negate and
    * absolute value are only allowed on inputs with int or float type and
    * behave differently on the two.
    */
   nir_alu_type input_types[NIR_MAX_VEC_COMPONENTS];

   nir_op_algebraic_property algebraic_properties;

   /* Whether this represents a numeric conversion opcode */
   bool is_conversion;
} nir_op_info;

extern const nir_op_info nir_op_infos[nir_num_opcodes];

typedef struct nir_alu_instr {
   nir_instr instr;
   nir_op op;

   /** Indicates that this ALU instruction generates an exact value
    *
    * This is kind of a mixture of GLSL "precise" and "invariant" and not
    * really equivalent to either.  This indicates that the value generated by
    * this operation is high-precision and any code transformations that touch
    * it must ensure that the resulting value is bit-for-bit identical to the
    * original.
    */
   bool exact:1;

   /**
    * Indicates that this instruction do not cause wrapping to occur, in the
    * form of overflow or underflow.
    */
   bool no_signed_wrap:1;
   bool no_unsigned_wrap:1;

   nir_alu_dest dest;
   nir_alu_src src[];
} nir_alu_instr;

void nir_alu_src_copy(nir_alu_src *dest, const nir_alu_src *src,
                      nir_alu_instr *instr);
void nir_alu_dest_copy(nir_alu_dest *dest, const nir_alu_dest *src,
                       nir_alu_instr *instr);

/* is this source channel used? */
static inline bool
nir_alu_instr_channel_used(const nir_alu_instr *instr, unsigned src,
                           unsigned channel)
{
   if (nir_op_infos[instr->op].input_sizes[src] > 0)
      return channel < nir_op_infos[instr->op].input_sizes[src];

   return (instr->dest.write_mask >> channel) & 1;
}

static inline nir_component_mask_t
nir_alu_instr_src_read_mask(const nir_alu_instr *instr, unsigned src)
{
   nir_component_mask_t read_mask = 0;
   for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; c++) {
      if (!nir_alu_instr_channel_used(instr, src, c))
         continue;

      read_mask |= (1 << instr->src[src].swizzle[c]);
   }
   return read_mask;
}

/**
 * Get the number of channels used for a source
 */
static inline unsigned
nir_ssa_alu_instr_src_components(const nir_alu_instr *instr, unsigned src)
{
   if (nir_op_infos[instr->op].input_sizes[src] > 0)
      return nir_op_infos[instr->op].input_sizes[src];

   return nir_dest_num_components(instr->dest.dest);
}

static inline bool
nir_alu_instr_is_comparison(const nir_alu_instr *instr)
{
   switch (instr->op) {
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fne:
   case nir_op_ilt:
   case nir_op_ult:
   case nir_op_ige:
   case nir_op_uge:
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_i2b1:
   case nir_op_f2b1:
   case nir_op_inot:
      return true;
   default:
      return false;
   }
}

bool nir_const_value_negative_equal(nir_const_value c1, nir_const_value c2,
                                    nir_alu_type full_type);

bool nir_alu_srcs_equal(const nir_alu_instr *alu1, const nir_alu_instr *alu2,
                        unsigned src1, unsigned src2);

bool nir_alu_srcs_negative_equal(const nir_alu_instr *alu1,
                                 const nir_alu_instr *alu2,
                                 unsigned src1, unsigned src2);

typedef enum {
   nir_deref_type_var,
   nir_deref_type_array,
   nir_deref_type_array_wildcard,
   nir_deref_type_ptr_as_array,
   nir_deref_type_struct,
   nir_deref_type_cast,
} nir_deref_type;

typedef struct {
   nir_instr instr;

   /** The type of this deref instruction */
   nir_deref_type deref_type;

   /** The mode of the underlying variable */
   nir_variable_mode mode;

   /** The dereferenced type of the resulting pointer value */
   const struct glsl_type *type;

   union {
      /** Variable being dereferenced if deref_type is a deref_var */
      nir_variable *var;

      /** Parent deref if deref_type is not deref_var */
      nir_src parent;
   };

   /** Additional deref parameters */
   union {
      struct {
         nir_src index;
      } arr;

      struct {
         unsigned index;
      } strct;

      struct {
         unsigned ptr_stride;
      } cast;
   };

   /** Destination to store the resulting "pointer" */
   nir_dest dest;
} nir_deref_instr;

static inline nir_deref_instr *nir_src_as_deref(nir_src src);

static inline nir_deref_instr *
nir_deref_instr_parent(const nir_deref_instr *instr)
{
   if (instr->deref_type == nir_deref_type_var)
      return NULL;
   else
      return nir_src_as_deref(instr->parent);
}

static inline nir_variable *
nir_deref_instr_get_variable(const nir_deref_instr *instr)
{
   while (instr->deref_type != nir_deref_type_var) {
      instr = nir_deref_instr_parent(instr);
      if (!instr)
         return NULL;
   }

   return instr->var;
}

bool nir_deref_instr_has_indirect(nir_deref_instr *instr);
bool nir_deref_instr_is_known_out_of_bounds(nir_deref_instr *instr);
bool nir_deref_instr_has_complex_use(nir_deref_instr *instr);

bool nir_deref_instr_remove_if_unused(nir_deref_instr *instr);

unsigned nir_deref_instr_ptr_as_array_stride(nir_deref_instr *instr);

typedef struct {
   nir_instr instr;

   struct nir_function *callee;

   unsigned num_params;
   nir_src params[];
} nir_call_instr;

#include "nir_intrinsics.h"

#define NIR_INTRINSIC_MAX_CONST_INDEX 4

/** Represents an intrinsic
 *
 * An intrinsic is an instruction type for handling things that are
 * more-or-less regular operations but don't just consume and produce SSA
 * values like ALU operations do.  Intrinsics are not for things that have
 * special semantic meaning such as phi nodes and parallel copies.
 * Examples of intrinsics include variable load/store operations, system
 * value loads, and the like.  Even though texturing more-or-less falls
 * under this category, texturing is its own instruction type because
 * trying to represent texturing with intrinsics would lead to a
 * combinatorial explosion of intrinsic opcodes.
 *
 * By having a single instruction type for handling a lot of different
 * cases, optimization passes can look for intrinsics and, for the most
 * part, completely ignore them.  Each intrinsic type also has a few
 * possible flags that govern whether or not they can be reordered or
 * eliminated.  That way passes like dead code elimination can still work
 * on intrisics without understanding the meaning of each.
 *
 * Each intrinsic has some number of constant indices, some number of
 * variables, and some number of sources.  What these sources, variables,
 * and indices mean depends on the intrinsic and is documented with the
 * intrinsic declaration in nir_intrinsics.h.  Intrinsics and texture
 * instructions are the only types of instruction that can operate on
 * variables.
 */
typedef struct {
   nir_instr instr;

   nir_intrinsic_op intrinsic;

   nir_dest dest;

   /** number of components if this is a vectorized intrinsic
    *
    * Similarly to ALU operations, some intrinsics are vectorized.
    * An intrinsic is vectorized if nir_intrinsic_infos.dest_components == 0.
    * For vectorized intrinsics, the num_components field specifies the
    * number of destination components and the number of source components
    * for all sources with nir_intrinsic_infos.src_components[i] == 0.
    */
   uint8_t num_components;

   int const_index[NIR_INTRINSIC_MAX_CONST_INDEX];

   nir_src src[];
} nir_intrinsic_instr;

static inline nir_variable *
nir_intrinsic_get_var(nir_intrinsic_instr *intrin, unsigned i)
{
   return nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[i]));
}

/**
 * \name NIR intrinsics semantic flags
 *
 * information about what the compiler can do with the intrinsics.
 *
 * \sa nir_intrinsic_info::flags
 */
typedef enum {
   /**
    * whether the intrinsic can be safely eliminated if none of its output
    * value is not being used.
    */
   NIR_INTRINSIC_CAN_ELIMINATE = (1 << 0),

   /**
    * Whether the intrinsic can be reordered with respect to any other
    * intrinsic, i.e. whether the only reordering dependencies of the
    * intrinsic are due to the register reads/writes.
    */
   NIR_INTRINSIC_CAN_REORDER = (1 << 1),
} nir_intrinsic_semantic_flag;

/**
 * \name NIR intrinsics const-index flag
 *
 * Indicates the usage of a const_index slot.
 *
 * \sa nir_intrinsic_info::index_map
 */
typedef enum {
   /**
    * Generally instructions that take a offset src argument, can encode
    * a constant 'base' value which is added to the offset.
    */
   NIR_INTRINSIC_BASE = 1,

   /**
    * For store instructions, a writemask for the store.
    */
   NIR_INTRINSIC_WRMASK,

   /**
    * The stream-id for GS emit_vertex/end_primitive intrinsics.
    */
   NIR_INTRINSIC_STREAM_ID,

   /**
    * The clip-plane id for load_user_clip_plane intrinsic.
    */
   NIR_INTRINSIC_UCP_ID,

   /**
    * The amount of data, starting from BASE, that this instruction may
    * access.  This is used to provide bounds if the offset is not constant.
    */
   NIR_INTRINSIC_RANGE,

   /**
    * The Vulkan descriptor set for vulkan_resource_index intrinsic.
    */
   NIR_INTRINSIC_DESC_SET,

   /**
    * The Vulkan descriptor set binding for vulkan_resource_index intrinsic.
    */
   NIR_INTRINSIC_BINDING,

   /**
    * Component offset.
    */
   NIR_INTRINSIC_COMPONENT,

   /**
    * Interpolation mode (only meaningful for FS inputs).
    */
   NIR_INTRINSIC_INTERP_MODE,

   /**
    * A binary nir_op to use when performing a reduction or scan operation
    */
   NIR_INTRINSIC_REDUCTION_OP,

   /**
    * Cluster size for reduction operations
    */
   NIR_INTRINSIC_CLUSTER_SIZE,

   /**
    * Parameter index for a load_param intrinsic
    */
   NIR_INTRINSIC_PARAM_IDX,

   /**
    * Image dimensionality for image intrinsics
    *
    * One of GLSL_SAMPLER_DIM_*
    */
   NIR_INTRINSIC_IMAGE_DIM,

   /**
    * Non-zero if we are accessing an array image
    */
   NIR_INTRINSIC_IMAGE_ARRAY,

   /**
    * Image format for image intrinsics
    */
   NIR_INTRINSIC_FORMAT,

   /**
    * Access qualifiers for image and memory access intrinsics
    */
   NIR_INTRINSIC_ACCESS,

   /**
    * Alignment for offsets and addresses
    *
    * These two parameters, specify an alignment in terms of a multiplier and
    * an offset.  The offset or address parameter X of the intrinsic is
    * guaranteed to satisfy the following:
    *
    *                (X - align_offset) % align_mul == 0
    */
   NIR_INTRINSIC_ALIGN_MUL,
   NIR_INTRINSIC_ALIGN_OFFSET,

   /**
    * The Vulkan descriptor type for a vulkan_resource_[re]index intrinsic.
    */
   NIR_INTRINSIC_DESC_TYPE,

   /**
    * The nir_alu_type of a uniform/input/output
    */
   NIR_INTRINSIC_TYPE,

   /**
    * The swizzle mask for the instructions
    * SwizzleInvocationsAMD and SwizzleInvocationsMaskedAMD
    */
   NIR_INTRINSIC_SWIZZLE_MASK,

   /* Separate source/dest access flags for copies */
   NIR_INTRINSIC_SRC_ACCESS,
   NIR_INTRINSIC_DST_ACCESS,

   NIR_INTRINSIC_NUM_INDEX_FLAGS,

} nir_intrinsic_index_flag;

#define NIR_INTRINSIC_MAX_INPUTS 5

typedef struct {
   const char *name;

   unsigned num_srcs; /** < number of register/SSA inputs */

   /** number of components of each input register
    *
    * If this value is 0, the number of components is given by the
    * num_components field of nir_intrinsic_instr.  If this value is -1, the
    * intrinsic consumes however many components are provided and it is not
    * validated at all.
    */
   int src_components[NIR_INTRINSIC_MAX_INPUTS];

   bool has_dest;

   /** number of components of the output register
    *
    * If this value is 0, the number of components is given by the
    * num_components field of nir_intrinsic_instr.
    */
   unsigned dest_components;

   /** bitfield of legal bit sizes */
   unsigned dest_bit_sizes;

   /** the number of constant indices used by the intrinsic */
   unsigned num_indices;

   /** indicates the usage of intr->const_index[n] */
   unsigned index_map[NIR_INTRINSIC_NUM_INDEX_FLAGS];

   /** semantic flags for calls to this intrinsic */
   nir_intrinsic_semantic_flag flags;
} nir_intrinsic_info;

extern const nir_intrinsic_info nir_intrinsic_infos[nir_num_intrinsics];

static inline unsigned
nir_intrinsic_src_components(nir_intrinsic_instr *intr, unsigned srcn)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   assert(srcn < info->num_srcs);
   if (info->src_components[srcn] > 0)
      return info->src_components[srcn];
   else if (info->src_components[srcn] == 0)
      return intr->num_components;
   else
      return nir_src_num_components(intr->src[srcn]);
}

static inline unsigned
nir_intrinsic_dest_components(nir_intrinsic_instr *intr)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   if (!info->has_dest)
      return 0;
   else if (info->dest_components)
      return info->dest_components;
   else
      return intr->num_components;
}

#define INTRINSIC_IDX_ACCESSORS(name, flag, type)                             \
static inline type                                                            \
nir_intrinsic_##name(const nir_intrinsic_instr *instr)                        \
{                                                                             \
   const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];   \
   assert(info->index_map[NIR_INTRINSIC_##flag] > 0);                         \
   return (type)instr->const_index[info->index_map[NIR_INTRINSIC_##flag] - 1]; \
}                                                                             \
static inline void                                                            \
nir_intrinsic_set_##name(nir_intrinsic_instr *instr, type val)                \
{                                                                             \
   const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];   \
   assert(info->index_map[NIR_INTRINSIC_##flag] > 0);                         \
   instr->const_index[info->index_map[NIR_INTRINSIC_##flag] - 1] = val;       \
}

INTRINSIC_IDX_ACCESSORS(write_mask, WRMASK, unsigned)
INTRINSIC_IDX_ACCESSORS(base, BASE, int)
INTRINSIC_IDX_ACCESSORS(stream_id, STREAM_ID, unsigned)
INTRINSIC_IDX_ACCESSORS(ucp_id, UCP_ID, unsigned)
INTRINSIC_IDX_ACCESSORS(range, RANGE, unsigned)
INTRINSIC_IDX_ACCESSORS(desc_set, DESC_SET, unsigned)
INTRINSIC_IDX_ACCESSORS(binding, BINDING, unsigned)
INTRINSIC_IDX_ACCESSORS(component, COMPONENT, unsigned)
INTRINSIC_IDX_ACCESSORS(interp_mode, INTERP_MODE, unsigned)
INTRINSIC_IDX_ACCESSORS(reduction_op, REDUCTION_OP, unsigned)
INTRINSIC_IDX_ACCESSORS(cluster_size, CLUSTER_SIZE, unsigned)
INTRINSIC_IDX_ACCESSORS(param_idx, PARAM_IDX, unsigned)
INTRINSIC_IDX_ACCESSORS(image_dim, IMAGE_DIM, enum glsl_sampler_dim)
INTRINSIC_IDX_ACCESSORS(image_array, IMAGE_ARRAY, bool)
INTRINSIC_IDX_ACCESSORS(access, ACCESS, enum gl_access_qualifier)
INTRINSIC_IDX_ACCESSORS(src_access, SRC_ACCESS, enum gl_access_qualifier)
INTRINSIC_IDX_ACCESSORS(dst_access, DST_ACCESS, enum gl_access_qualifier)
INTRINSIC_IDX_ACCESSORS(format, FORMAT, unsigned)
INTRINSIC_IDX_ACCESSORS(align_mul, ALIGN_MUL, unsigned)
INTRINSIC_IDX_ACCESSORS(align_offset, ALIGN_OFFSET, unsigned)
INTRINSIC_IDX_ACCESSORS(desc_type, DESC_TYPE, unsigned)
INTRINSIC_IDX_ACCESSORS(type, TYPE, nir_alu_type)
INTRINSIC_IDX_ACCESSORS(swizzle_mask, SWIZZLE_MASK, unsigned)

static inline void
nir_intrinsic_set_align(nir_intrinsic_instr *intrin,
                        unsigned align_mul, unsigned align_offset)
{
   assert(util_is_power_of_two_nonzero(align_mul));
   assert(align_offset < align_mul);
   nir_intrinsic_set_align_mul(intrin, align_mul);
   nir_intrinsic_set_align_offset(intrin, align_offset);
}

/** Returns a simple alignment for a load/store intrinsic offset
 *
 * Instead of the full mul+offset alignment scheme provided by the ALIGN_MUL
 * and ALIGN_OFFSET parameters, this helper takes both into account and
 * provides a single simple alignment parameter.  The offset X is guaranteed
 * to satisfy X % align == 0.
 */
static inline unsigned
nir_intrinsic_align(const nir_intrinsic_instr *intrin)
{
   const unsigned align_mul = nir_intrinsic_align_mul(intrin);
   const unsigned align_offset = nir_intrinsic_align_offset(intrin);
   assert(align_offset < align_mul);
   return align_offset ? 1 << (ffs(align_offset) - 1) : align_mul;
}

/* Converts a image_deref_* intrinsic into a image_* one */
void nir_rewrite_image_intrinsic(nir_intrinsic_instr *instr,
                                 nir_ssa_def *handle, bool bindless);

/* Determine if an intrinsic can be arbitrarily reordered and eliminated. */
static inline bool
nir_intrinsic_can_reorder(nir_intrinsic_instr *instr)
{
   if (instr->intrinsic == nir_intrinsic_load_deref ||
       instr->intrinsic == nir_intrinsic_load_ssbo ||
       instr->intrinsic == nir_intrinsic_bindless_image_load ||
       instr->intrinsic == nir_intrinsic_image_deref_load ||
       instr->intrinsic == nir_intrinsic_image_load) {
      return nir_intrinsic_access(instr) & ACCESS_CAN_REORDER;
   } else {
      const nir_intrinsic_info *info =
         &nir_intrinsic_infos[instr->intrinsic];
      return (info->flags & NIR_INTRINSIC_CAN_ELIMINATE) &&
             (info->flags & NIR_INTRINSIC_CAN_REORDER);
   }
}

/**
 * \group texture information
 *
 * This gives semantic information about textures which is useful to the
 * frontend, the backend, and lowering passes, but not the optimizer.
 */

typedef enum {
   nir_tex_src_coord,
   nir_tex_src_projector,
   nir_tex_src_comparator, /* shadow comparator */
   nir_tex_src_offset,
   nir_tex_src_bias,
   nir_tex_src_lod,
   nir_tex_src_min_lod,
   nir_tex_src_ms_index, /* MSAA sample index */
   nir_tex_src_ms_mcs, /* MSAA compression value */
   nir_tex_src_ddx,
   nir_tex_src_ddy,
   nir_tex_src_texture_deref, /* < deref pointing to the texture */
   nir_tex_src_sampler_deref, /* < deref pointing to the sampler */
   nir_tex_src_texture_offset, /* < dynamically uniform indirect offset */
   nir_tex_src_sampler_offset, /* < dynamically uniform indirect offset */
   nir_tex_src_texture_handle, /* < bindless texture handle */
   nir_tex_src_sampler_handle, /* < bindless sampler handle */
   nir_tex_src_plane,          /* < selects plane for planar textures */
   nir_num_tex_src_types
} nir_tex_src_type;

typedef struct {
   nir_src src;
   nir_tex_src_type src_type;
} nir_tex_src;

typedef enum {
   nir_texop_tex,                /**< Regular texture look-up */
   nir_texop_txb,                /**< Texture look-up with LOD bias */
   nir_texop_txl,                /**< Texture look-up with explicit LOD */
   nir_texop_txd,                /**< Texture look-up with partial derivatives */
   nir_texop_txf,                /**< Texel fetch with explicit LOD */
   nir_texop_txf_ms,             /**< Multisample texture fetch */
   nir_texop_txf_ms_fb,          /**< Multisample texture fetch from framebuffer */
   nir_texop_txf_ms_mcs,         /**< Multisample compression value fetch */
   nir_texop_txs,                /**< Texture size */
   nir_texop_lod,                /**< Texture lod query */
   nir_texop_tg4,                /**< Texture gather */
   nir_texop_query_levels,       /**< Texture levels query */
   nir_texop_texture_samples,    /**< Texture samples query */
   nir_texop_samples_identical,  /**< Query whether all samples are definitely
                                  * identical.
                                  */
} nir_texop;

typedef struct {
   nir_instr instr;

   enum glsl_sampler_dim sampler_dim;
   nir_alu_type dest_type;

   nir_texop op;
   nir_dest dest;
   nir_tex_src *src;
   unsigned num_srcs, coord_components;
   bool is_array, is_shadow;

   /**
    * If is_shadow is true, whether this is the old-style shadow that outputs 4
    * components or the new-style shadow that outputs 1 component.
    */
   bool is_new_style_shadow;

   /* gather component selector */
   unsigned component : 2;

   /* gather offsets */
   int8_t tg4_offsets[4][2];

   /* True if the texture index or handle is not dynamically uniform */
   bool texture_non_uniform;

   /* True if the sampler index or handle is not dynamically uniform */
   bool sampler_non_uniform;

   /** The texture index
    *
    * If this texture instruction has a nir_tex_src_texture_offset source,
    * then the texture index is given by texture_index + texture_offset.
    */
   unsigned texture_index;

   /** The size of the texture array or 0 if it's not an array */
   unsigned texture_array_size;

   /** The sampler index
    *
    * The following operations do not require a sampler and, as such, this
    * field should be ignored:
    *    - nir_texop_txf
    *    - nir_texop_txf_ms
    *    - nir_texop_txs
    *    - nir_texop_lod
    *    - nir_texop_query_levels
    *    - nir_texop_texture_samples
    *    - nir_texop_samples_identical
    *
    * If this texture instruction has a nir_tex_src_sampler_offset source,
    * then the sampler index is given by sampler_index + sampler_offset.
    */
   unsigned sampler_index;
} nir_tex_instr;

static inline unsigned
nir_tex_instr_dest_size(const nir_tex_instr *instr)
{
   switch (instr->op) {
   case nir_texop_txs: {
      unsigned ret;
      switch (instr->sampler_dim) {
         case GLSL_SAMPLER_DIM_1D:
         case GLSL_SAMPLER_DIM_BUF:
            ret = 1;
            break;
         case GLSL_SAMPLER_DIM_2D:
         case GLSL_SAMPLER_DIM_CUBE:
         case GLSL_SAMPLER_DIM_MS:
         case GLSL_SAMPLER_DIM_RECT:
         case GLSL_SAMPLER_DIM_EXTERNAL:
         case GLSL_SAMPLER_DIM_SUBPASS:
            ret = 2;
            break;
         case GLSL_SAMPLER_DIM_3D:
            ret = 3;
            break;
         default:
            unreachable("not reached");
      }
      if (instr->is_array)
         ret++;
      return ret;
   }

   case nir_texop_lod:
      return 2;

   case nir_texop_texture_samples:
   case nir_texop_query_levels:
   case nir_texop_samples_identical:
      return 1;

   default:
      if (instr->is_shadow && instr->is_new_style_shadow)
         return 1;

      return 4;
   }
}

/* Returns true if this texture operation queries something about the texture
 * rather than actually sampling it.
 */
static inline bool
nir_tex_instr_is_query(const nir_tex_instr *instr)
{
   switch (instr->op) {
   case nir_texop_txs:
   case nir_texop_lod:
   case nir_texop_texture_samples:
   case nir_texop_query_levels:
   case nir_texop_txf_ms_mcs:
      return true;
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_txf_ms_fb:
   case nir_texop_tg4:
      return false;
   default:
      unreachable("Invalid texture opcode");
   }
}

static inline bool
nir_tex_instr_has_implicit_derivative(const nir_tex_instr *instr)
{
   switch (instr->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_lod:
      return true;
   default:
      return false;
   }
}

static inline nir_alu_type
nir_tex_instr_src_type(const nir_tex_instr *instr, unsigned src)
{
   switch (instr->src[src].src_type) {
   case nir_tex_src_coord:
      switch (instr->op) {
      case nir_texop_txf:
      case nir_texop_txf_ms:
      case nir_texop_txf_ms_fb:
      case nir_texop_txf_ms_mcs:
      case nir_texop_samples_identical:
         return nir_type_int;

      default:
         return nir_type_float;
      }

   case nir_tex_src_lod:
      switch (instr->op) {
      case nir_texop_txs:
      case nir_texop_txf:
         return nir_type_int;

      default:
         return nir_type_float;
      }

   case nir_tex_src_projector:
   case nir_tex_src_comparator:
   case nir_tex_src_bias:
   case nir_tex_src_min_lod:
   case nir_tex_src_ddx:
   case nir_tex_src_ddy:
      return nir_type_float;

   case nir_tex_src_offset:
   case nir_tex_src_ms_index:
   case nir_tex_src_plane:
      return nir_type_int;

   case nir_tex_src_ms_mcs:
   case nir_tex_src_texture_deref:
   case nir_tex_src_sampler_deref:
   case nir_tex_src_texture_offset:
   case nir_tex_src_sampler_offset:
   case nir_tex_src_texture_handle:
   case nir_tex_src_sampler_handle:
      return nir_type_uint;

   case nir_num_tex_src_types:
      unreachable("nir_num_tex_src_types is not a valid source type");
   }

   unreachable("Invalid texture source type");
}

static inline unsigned
nir_tex_instr_src_size(const nir_tex_instr *instr, unsigned src)
{
   if (instr->src[src].src_type == nir_tex_src_coord)
      return instr->coord_components;

   /* The MCS value is expected to be a vec4 returned by a txf_ms_mcs */
   if (instr->src[src].src_type == nir_tex_src_ms_mcs)
      return 4;

   if (instr->src[src].src_type == nir_tex_src_ddx ||
       instr->src[src].src_type == nir_tex_src_ddy) {
      if (instr->is_array)
         return instr->coord_components - 1;
      else
         return instr->coord_components;
   }

   /* Usual APIs don't allow cube + offset, but we allow it, with 2 coords for
    * the offset, since a cube maps to a single face.
    */
   if (instr->src[src].src_type == nir_tex_src_offset) {
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
         return 2;
      else if (instr->is_array)
         return instr->coord_components - 1;
      else
         return instr->coord_components;
   }

   return 1;
}

static inline int
nir_tex_instr_src_index(const nir_tex_instr *instr, nir_tex_src_type type)
{
   for (unsigned i = 0; i < instr->num_srcs; i++)
      if (instr->src[i].src_type == type)
         return (int) i;

   return -1;
}

void nir_tex_instr_add_src(nir_tex_instr *tex,
                           nir_tex_src_type src_type,
                           nir_src src);

void nir_tex_instr_remove_src(nir_tex_instr *tex, unsigned src_idx);

bool nir_tex_instr_has_explicit_tg4_offsets(nir_tex_instr *tex);

typedef struct {
   nir_instr instr;

   nir_ssa_def def;

   nir_const_value value[];
} nir_load_const_instr;

typedef enum {
   nir_jump_return,
   nir_jump_break,
   nir_jump_continue,
} nir_jump_type;

typedef struct {
   nir_instr instr;
   nir_jump_type type;
} nir_jump_instr;

/* creates a new SSA variable in an undefined state */

typedef struct {
   nir_instr instr;
   nir_ssa_def def;
} nir_ssa_undef_instr;

typedef struct {
   struct exec_node node;

   /* The predecessor block corresponding to this source */
   struct nir_block *pred;

   nir_src src;
} nir_phi_src;

#define nir_foreach_phi_src(phi_src, phi) \
   foreach_list_typed(nir_phi_src, phi_src, node, &(phi)->srcs)
#define nir_foreach_phi_src_safe(phi_src, phi) \
   foreach_list_typed_safe(nir_phi_src, phi_src, node, &(phi)->srcs)

typedef struct {
   nir_instr instr;

   struct exec_list srcs; /** < list of nir_phi_src */

   nir_dest dest;
} nir_phi_instr;

typedef struct {
   struct exec_node node;
   nir_src src;
   nir_dest dest;
} nir_parallel_copy_entry;

#define nir_foreach_parallel_copy_entry(entry, pcopy) \
   foreach_list_typed(nir_parallel_copy_entry, entry, node, &(pcopy)->entries)

typedef struct {
   nir_instr instr;

   /* A list of nir_parallel_copy_entrys.  The sources of all of the
    * entries are copied to the corresponding destinations "in parallel".
    * In other words, if we have two entries: a -> b and b -> a, the values
    * get swapped.
    */
   struct exec_list entries;
} nir_parallel_copy_instr;

NIR_DEFINE_CAST(nir_instr_as_alu, nir_instr, nir_alu_instr, instr,
                type, nir_instr_type_alu)
NIR_DEFINE_CAST(nir_instr_as_deref, nir_instr, nir_deref_instr, instr,
                type, nir_instr_type_deref)
NIR_DEFINE_CAST(nir_instr_as_call, nir_instr, nir_call_instr, instr,
                type, nir_instr_type_call)
NIR_DEFINE_CAST(nir_instr_as_jump, nir_instr, nir_jump_instr, instr,
                type, nir_instr_type_jump)
NIR_DEFINE_CAST(nir_instr_as_tex, nir_instr, nir_tex_instr, instr,
                type, nir_instr_type_tex)
NIR_DEFINE_CAST(nir_instr_as_intrinsic, nir_instr, nir_intrinsic_instr, instr,
                type, nir_instr_type_intrinsic)
NIR_DEFINE_CAST(nir_instr_as_load_const, nir_instr, nir_load_const_instr, instr,
                type, nir_instr_type_load_const)
NIR_DEFINE_CAST(nir_instr_as_ssa_undef, nir_instr, nir_ssa_undef_instr, instr,
                type, nir_instr_type_ssa_undef)
NIR_DEFINE_CAST(nir_instr_as_phi, nir_instr, nir_phi_instr, instr,
                type, nir_instr_type_phi)
NIR_DEFINE_CAST(nir_instr_as_parallel_copy, nir_instr,
                nir_parallel_copy_instr, instr,
                type, nir_instr_type_parallel_copy)


#define NIR_DEFINE_SRC_AS_CONST(type, suffix)               \
static inline type                                          \
nir_src_comp_as_##suffix(nir_src src, unsigned comp)        \
{                                                           \
   assert(nir_src_is_const(src));                           \
   nir_load_const_instr *load =                             \
      nir_instr_as_load_const(src.ssa->parent_instr);       \
   assert(comp < load->def.num_components);                 \
   return nir_const_value_as_##suffix(load->value[comp],    \
                                      load->def.bit_size);  \
}                                                           \
                                                            \
static inline type                                          \
nir_src_as_##suffix(nir_src src)                            \
{                                                           \
   assert(nir_src_num_components(src) == 1);                \
   return nir_src_comp_as_##suffix(src, 0);                 \
}

NIR_DEFINE_SRC_AS_CONST(int64_t,    int)
NIR_DEFINE_SRC_AS_CONST(uint64_t,   uint)
NIR_DEFINE_SRC_AS_CONST(bool,       bool)
NIR_DEFINE_SRC_AS_CONST(double,     float)

#undef NIR_DEFINE_SRC_AS_CONST


typedef struct {
   nir_ssa_def *def;
   unsigned comp;
} nir_ssa_scalar;

static inline bool
nir_ssa_scalar_is_const(nir_ssa_scalar s)
{
   return s.def->parent_instr->type == nir_instr_type_load_const;
}

static inline nir_const_value
nir_ssa_scalar_as_const_value(nir_ssa_scalar s)
{
   assert(s.comp < s.def->num_components);
   nir_load_const_instr *load = nir_instr_as_load_const(s.def->parent_instr);
   return load->value[s.comp];
}

#define NIR_DEFINE_SCALAR_AS_CONST(type, suffix)                     \
static inline type                                                   \
nir_ssa_scalar_as_##suffix(nir_ssa_scalar s)                         \
{                                                                    \
   return nir_const_value_as_##suffix(                               \
      nir_ssa_scalar_as_const_value(s), s.def->bit_size);            \
}

NIR_DEFINE_SCALAR_AS_CONST(int64_t,    int)
NIR_DEFINE_SCALAR_AS_CONST(uint64_t,   uint)
NIR_DEFINE_SCALAR_AS_CONST(bool,       bool)
NIR_DEFINE_SCALAR_AS_CONST(double,     float)

#undef NIR_DEFINE_SCALAR_AS_CONST

static inline bool
nir_ssa_scalar_is_alu(nir_ssa_scalar s)
{
   return s.def->parent_instr->type == nir_instr_type_alu;
}

static inline nir_op
nir_ssa_scalar_alu_op(nir_ssa_scalar s)
{
   return nir_instr_as_alu(s.def->parent_instr)->op;
}

static inline nir_ssa_scalar
nir_ssa_scalar_chase_alu_src(nir_ssa_scalar s, unsigned alu_src_idx)
{
   nir_ssa_scalar out = { NULL, 0 };

   nir_alu_instr *alu = nir_instr_as_alu(s.def->parent_instr);
   assert(alu_src_idx < nir_op_infos[alu->op].num_inputs);

   /* Our component must be written */
   assert(s.comp < s.def->num_components);
   assert(alu->dest.write_mask & (1u << s.comp));

   assert(alu->src[alu_src_idx].src.is_ssa);
   out.def = alu->src[alu_src_idx].src.ssa;

   if (nir_op_infos[alu->op].input_sizes[alu_src_idx] == 0) {
      /* The ALU src is unsized so the source component follows the
       * destination component.
       */
      out.comp = alu->src[alu_src_idx].swizzle[s.comp];
   } else {
      /* This is a sized source so all source components work together to
       * produce all the destination components.  Since we need to return a
       * scalar, this only works if the source is a scalar.
       */
      assert(nir_op_infos[alu->op].input_sizes[alu_src_idx] == 1);
      out.comp = alu->src[alu_src_idx].swizzle[0];
   }
   assert(out.comp < out.def->num_components);

   return out;
}


/*
 * Control flow
 *
 * Control flow consists of a tree of control flow nodes, which include
 * if-statements and loops. The leaves of the tree are basic blocks, lists of
 * instructions that always run start-to-finish. Each basic block also keeps
 * track of its successors (blocks which may run immediately after the current
 * block) and predecessors (blocks which could have run immediately before the
 * current block). Each function also has a start block and an end block which
 * all return statements point to (which is always empty). Together, all the
 * blocks with their predecessors and successors make up the control flow
 * graph (CFG) of the function. There are helpers that modify the tree of
 * control flow nodes while modifying the CFG appropriately; these should be
 * used instead of modifying the tree directly.
 */

typedef enum {
   nir_cf_node_block,
   nir_cf_node_if,
   nir_cf_node_loop,
   nir_cf_node_function
} nir_cf_node_type;

typedef struct nir_cf_node {
   struct exec_node node;
   nir_cf_node_type type;
   struct nir_cf_node *parent;
} nir_cf_node;

typedef struct nir_block {
   nir_cf_node cf_node;

   struct exec_list instr_list; /** < list of nir_instr */

   /** generic block index; generated by nir_index_blocks */
   unsigned index;

   /*
    * Each block can only have up to 2 successors, so we put them in a simple
    * array - no need for anything more complicated.
    */
   struct nir_block *successors[2];

   /* Set of nir_block predecessors in the CFG */
   struct set *predecessors;

   /*
    * this node's immediate dominator in the dominance tree - set to NULL for
    * the start block.
    */
   struct nir_block *imm_dom;

   /* This node's children in the dominance tree */
   unsigned num_dom_children;
   struct nir_block **dom_children;

   /* Set of nir_blocks on the dominance frontier of this block */
   struct set *dom_frontier;

   /*
    * These two indices have the property that dom_{pre,post}_index for each
    * child of this block in the dominance tree will always be between
    * dom_pre_index and dom_post_index for this block, which makes testing if
    * a given block is dominated by another block an O(1) operation.
    */
   unsigned dom_pre_index, dom_post_index;

   /* live in and out for this block; used for liveness analysis */
   BITSET_WORD *live_in;
   BITSET_WORD *live_out;
} nir_block;

static inline nir_instr *
nir_block_first_instr(nir_block *block)
{
   struct exec_node *head = exec_list_get_head(&block->instr_list);
   return exec_node_data(nir_instr, head, node);
}

static inline nir_instr *
nir_block_last_instr(nir_block *block)
{
   struct exec_node *tail = exec_list_get_tail(&block->instr_list);
   return exec_node_data(nir_instr, tail, node);
}

static inline bool
nir_block_ends_in_jump(nir_block *block)
{
   return !exec_list_is_empty(&block->instr_list) &&
          nir_block_last_instr(block)->type == nir_instr_type_jump;
}

#define nir_foreach_instr(instr, block) \
   foreach_list_typed(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_reverse(instr, block) \
   foreach_list_typed_reverse(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_safe(instr, block) \
   foreach_list_typed_safe(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_reverse_safe(instr, block) \
   foreach_list_typed_reverse_safe(nir_instr, instr, node, &(block)->instr_list)

typedef enum {
   nir_selection_control_none = 0x0,
   nir_selection_control_flatten = 0x1,
   nir_selection_control_dont_flatten = 0x2,
} nir_selection_control;

typedef struct nir_if {
   nir_cf_node cf_node;
   nir_src condition;
   nir_selection_control control;

   struct exec_list then_list; /** < list of nir_cf_node */
   struct exec_list else_list; /** < list of nir_cf_node */
} nir_if;

typedef struct {
   nir_if *nif;

   /** Instruction that generates nif::condition. */
   nir_instr *conditional_instr;

   /** Block within ::nif that has the break instruction. */
   nir_block *break_block;

   /** Last block for the then- or else-path that does not contain the break. */
   nir_block *continue_from_block;

   /** True when ::break_block is in the else-path of ::nif. */
   bool continue_from_then;
   bool induction_rhs;

   /* This is true if the terminators exact trip count is unknown. For
    * example:
    *
    *    for (int i = 0; i < imin(x, 4); i++)
    *       ...
    *
    * Here loop analysis would have set a max_trip_count of 4 however we dont
    * know for sure that this is the exact trip count.
    */
   bool exact_trip_count_unknown;

   struct list_head loop_terminator_link;
} nir_loop_terminator;

typedef struct {
   /* Estimated cost (in number of instructions) of the loop */
   unsigned instr_cost;

   /* Guessed trip count based on array indexing */
   unsigned guessed_trip_count;

   /* Maximum number of times the loop is run (if known) */
   unsigned max_trip_count;

   /* Do we know the exact number of times the loop will be run */
   bool exact_trip_count_known;

   /* Unroll the loop regardless of its size */
   bool force_unroll;

   /* Does the loop contain complex loop terminators, continues or other
    * complex behaviours? If this is true we can't rely on
    * loop_terminator_list to be complete or accurate.
    */
   bool complex_loop;

   nir_loop_terminator *limiting_terminator;

   /* A list of loop_terminators terminating this loop. */
   struct list_head loop_terminator_list;
} nir_loop_info;

typedef enum {
   nir_loop_control_none = 0x0,
   nir_loop_control_unroll = 0x1,
   nir_loop_control_dont_unroll = 0x2,
} nir_loop_control;

typedef struct {
   nir_cf_node cf_node;

   struct exec_list body; /** < list of nir_cf_node */

   nir_loop_info *info;
   nir_loop_control control;
   bool partially_unrolled;
} nir_loop;

/**
 * Various bits of metadata that can may be created or required by
 * optimization and analysis passes
 */
typedef enum {
   nir_metadata_none = 0x0,
   nir_metadata_block_index = 0x1,
   nir_metadata_dominance = 0x2,
   nir_metadata_live_ssa_defs = 0x4,
   nir_metadata_not_properly_reset = 0x8,
   nir_metadata_loop_analysis = 0x10,
} nir_metadata;

typedef struct {
   nir_cf_node cf_node;

   /** pointer to the function of which this is an implementation */
   struct nir_function *function;

   struct exec_list body; /** < list of nir_cf_node */

   nir_block *end_block;

   /** list for all local variables in the function */
   struct exec_list locals;

   /** list of local registers in the function */
   struct exec_list registers;

   /** next available local register index */
   unsigned reg_alloc;

   /** next available SSA value index */
   unsigned ssa_alloc;

   /* total number of basic blocks, only valid when block_index_dirty = false */
   unsigned num_blocks;

   nir_metadata valid_metadata;
} nir_function_impl;

ATTRIBUTE_RETURNS_NONNULL static inline nir_block *
nir_start_block(nir_function_impl *impl)
{
   return (nir_block *) impl->body.head_sentinel.next;
}

ATTRIBUTE_RETURNS_NONNULL static inline nir_block *
nir_impl_last_block(nir_function_impl *impl)
{
   return (nir_block *) impl->body.tail_sentinel.prev;
}

static inline nir_cf_node *
nir_cf_node_next(nir_cf_node *node)
{
   struct exec_node *next = exec_node_get_next(&node->node);
   if (exec_node_is_tail_sentinel(next))
      return NULL;
   else
      return exec_node_data(nir_cf_node, next, node);
}

static inline nir_cf_node *
nir_cf_node_prev(nir_cf_node *node)
{
   struct exec_node *prev = exec_node_get_prev(&node->node);
   if (exec_node_is_head_sentinel(prev))
      return NULL;
   else
      return exec_node_data(nir_cf_node, prev, node);
}

static inline bool
nir_cf_node_is_first(const nir_cf_node *node)
{
   return exec_node_is_head_sentinel(node->node.prev);
}

static inline bool
nir_cf_node_is_last(const nir_cf_node *node)
{
   return exec_node_is_tail_sentinel(node->node.next);
}

NIR_DEFINE_CAST(nir_cf_node_as_block, nir_cf_node, nir_block, cf_node,
                type, nir_cf_node_block)
NIR_DEFINE_CAST(nir_cf_node_as_if, nir_cf_node, nir_if, cf_node,
                type, nir_cf_node_if)
NIR_DEFINE_CAST(nir_cf_node_as_loop, nir_cf_node, nir_loop, cf_node,
                type, nir_cf_node_loop)
NIR_DEFINE_CAST(nir_cf_node_as_function, nir_cf_node,
                nir_function_impl, cf_node, type, nir_cf_node_function)

static inline nir_block *
nir_if_first_then_block(nir_if *if_stmt)
{
   struct exec_node *head = exec_list_get_head(&if_stmt->then_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_if_last_then_block(nir_if *if_stmt)
{
   struct exec_node *tail = exec_list_get_tail(&if_stmt->then_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

static inline nir_block *
nir_if_first_else_block(nir_if *if_stmt)
{
   struct exec_node *head = exec_list_get_head(&if_stmt->else_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_if_last_else_block(nir_if *if_stmt)
{
   struct exec_node *tail = exec_list_get_tail(&if_stmt->else_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

static inline nir_block *
nir_loop_first_block(nir_loop *loop)
{
   struct exec_node *head = exec_list_get_head(&loop->body);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_loop_last_block(nir_loop *loop)
{
   struct exec_node *tail = exec_list_get_tail(&loop->body);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

/**
 * Return true if this list of cf_nodes contains a single empty block.
 */
static inline bool
nir_cf_list_is_empty_block(struct exec_list *cf_list)
{
   if (exec_list_is_singular(cf_list)) {
      struct exec_node *head = exec_list_get_head(cf_list);
      nir_block *block =
         nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
      return exec_list_is_empty(&block->instr_list);
   }
   return false;
}

typedef struct {
   uint8_t num_components;
   uint8_t bit_size;
} nir_parameter;

typedef struct nir_function {
   struct exec_node node;

   const char *name;
   struct nir_shader *shader;

   unsigned num_params;
   nir_parameter *params;

   /** The implementation of this function.
    *
    * If the function is only declared and not implemented, this is NULL.
    */
   nir_function_impl *impl;

   bool is_entrypoint;
} nir_function;

typedef enum {
   nir_lower_imul64 = (1 << 0),
   nir_lower_isign64 = (1 << 1),
   /** Lower all int64 modulus and division opcodes */
   nir_lower_divmod64 = (1 << 2),
   /** Lower all 64-bit umul_high and imul_high opcodes */
   nir_lower_imul_high64 = (1 << 3),
   nir_lower_mov64 = (1 << 4),
   nir_lower_icmp64 = (1 << 5),
   nir_lower_iadd64 = (1 << 6),
   nir_lower_iabs64 = (1 << 7),
   nir_lower_ineg64 = (1 << 8),
   nir_lower_logic64 = (1 << 9),
   nir_lower_minmax64 = (1 << 10),
   nir_lower_shift64 = (1 << 11),
   nir_lower_imul_2x32_64 = (1 << 12),
   nir_lower_extract64 = (1 << 13),
} nir_lower_int64_options;

typedef enum {
   nir_lower_drcp = (1 << 0),
   nir_lower_dsqrt = (1 << 1),
   nir_lower_drsq = (1 << 2),
   nir_lower_dtrunc = (1 << 3),
   nir_lower_dfloor = (1 << 4),
   nir_lower_dceil = (1 << 5),
   nir_lower_dfract = (1 << 6),
   nir_lower_dround_even = (1 << 7),
   nir_lower_dmod = (1 << 8),
   nir_lower_dsub = (1 << 9),
   nir_lower_ddiv = (1 << 10),
   nir_lower_fp64_full_software = (1 << 11),
} nir_lower_doubles_options;

typedef enum {
   nir_divergence_single_prim_per_subgroup = (1 << 0),
   nir_divergence_single_patch_per_tcs_subgroup = (1 << 1),
   nir_divergence_single_patch_per_tes_subgroup = (1 << 2),
   nir_divergence_view_index_uniform = (1 << 3),
} nir_divergence_options;

typedef struct nir_shader_compiler_options {
   bool lower_fdiv;
   bool lower_ffma;
   bool fuse_ffma;
   bool lower_flrp16;
   bool lower_flrp32;
   /** Lowers flrp when it does not support doubles */
   bool lower_flrp64;
   bool lower_fpow;
   bool lower_fsat;
   bool lower_fsqrt;
   bool lower_sincos;
   bool lower_fmod;
   /** Lowers ibitfield_extract/ubitfield_extract to ibfe/ubfe. */
   bool lower_bitfield_extract;
   /** Lowers ibitfield_extract/ubitfield_extract to compares, shifts. */
   bool lower_bitfield_extract_to_shifts;
   /** Lowers bitfield_insert to bfi/bfm */
   bool lower_bitfield_insert;
   /** Lowers bitfield_insert to compares, and shifts. */
   bool lower_bitfield_insert_to_shifts;
   /** Lowers bitfield_insert to bfm/bitfield_select. */
   bool lower_bitfield_insert_to_bitfield_select;
   /** Lowers bitfield_reverse to shifts. */
   bool lower_bitfield_reverse;
   /** Lowers bit_count to shifts. */
   bool lower_bit_count;
   /** Lowers ifind_msb to compare and ufind_msb */
   bool lower_ifind_msb;
   /** Lowers find_lsb to ufind_msb and logic ops */
   bool lower_find_lsb;
   bool lower_uadd_carry;
   bool lower_usub_borrow;
   /** Lowers imul_high/umul_high to 16-bit multiplies and carry operations. */
   bool lower_mul_high;
   /** lowers fneg and ineg to fsub and isub. */
   bool lower_negate;
   /** lowers fsub and isub to fadd+fneg and iadd+ineg. */
   bool lower_sub;

   /* lower {slt,sge,seq,sne} to {flt,fge,feq,fne} + b2f: */
   bool lower_scmp;

   /* lower fall_equalN/fany_nequalN (ex:fany_nequal4 to sne+fdot4+fsat) */
   bool lower_vector_cmp;

   /** enables rules to lower idiv by power-of-two: */
   bool lower_idiv;

   /** enable rules to avoid bit ops */
   bool lower_bitops;

   /** enables rules to lower isign to imin+imax */
   bool lower_isign;

   /** enables rules to lower fsign to fsub and flt */
   bool lower_fsign;

   /* lower fdph to fdot4 */
   bool lower_fdph;

   /** lower fdot to fmul and fsum/fadd. */
   bool lower_fdot;

   /* Does the native fdot instruction replicate its result for four
    * components?  If so, then opt_algebraic_late will turn all fdotN
    * instructions into fdot_replicatedN instructions.
    */
   bool fdot_replicates;

   /** lowers ffloor to fsub+ffract: */
   bool lower_ffloor;

   /** lowers ffract to fsub+ffloor: */
   bool lower_ffract;

   /** lowers fceil to fneg+ffloor+fneg: */
   bool lower_fceil;

   bool lower_ftrunc;

   bool lower_ldexp;

   bool lower_pack_half_2x16;
   bool lower_pack_unorm_2x16;
   bool lower_pack_snorm_2x16;
   bool lower_pack_unorm_4x8;
   bool lower_pack_snorm_4x8;
   bool lower_unpack_half_2x16;
   bool lower_unpack_unorm_2x16;
   bool lower_unpack_snorm_2x16;
   bool lower_unpack_unorm_4x8;
   bool lower_unpack_snorm_4x8;

   bool lower_extract_byte;
   bool lower_extract_word;

   bool lower_all_io_to_temps;
   bool lower_all_io_to_elements;

   /* Indicates that the driver only has zero-based vertex id */
   bool vertex_id_zero_based;

   /**
    * If enabled, gl_BaseVertex will be lowered as:
    * is_indexed_draw (~0/0) & firstvertex
    */
   bool lower_base_vertex;

   /**
    * If enabled, gl_HelperInvocation will be lowered as:
    *
    *   !((1 << sample_id) & sample_mask_in))
    *
    * This depends on some possibly hw implementation details, which may
    * not be true for all hw.  In particular that the FS is only executed
    * for covered samples or for helper invocations.  So, do not blindly
    * enable this option.
    *
    * Note: See also issue #22 in ARB_shader_image_load_store
    */
   bool lower_helper_invocation;

   /**
    * Convert gl_SampleMaskIn to gl_HelperInvocation as follows:
    *
    *   gl_SampleMaskIn == 0 ---> gl_HelperInvocation
    *   gl_SampleMaskIn != 0 ---> !gl_HelperInvocation
    */
   bool optimize_sample_mask_in;

   bool lower_cs_local_index_from_id;
   bool lower_cs_local_id_from_index;

   bool lower_device_index_to_zero;

   /* Set if nir_lower_wpos_ytransform() should also invert gl_PointCoord. */
   bool lower_wpos_pntc;

   bool lower_hadd;
   bool lower_add_sat;

   /**
    * Should IO be re-vectorized?  Some scalar ISAs still operate on vec4's
    * for IO purposes and would prefer loads/stores be vectorized.
    */
   bool vectorize_io;

   /**
    * Should nir_lower_io() create load_interpolated_input intrinsics?
    *
    * If not, it generates regular load_input intrinsics and interpolation
    * information must be inferred from the list of input nir_variables.
    */
   bool use_interpolated_input_intrinsics;

   /* Lowers when 32x32->64 bit multiplication is not supported */
   bool lower_mul_2x32_64;

   /* Lowers when rotate instruction is not supported */
   bool lower_rotate;

   /**
    * Is this the Intel vec4 backend?
    *
    * Used to inhibit algebraic optimizations that are known to be harmful on
    * the Intel vec4 backend.  This is generally applicable to any
    * optimization that might cause more immediate values to be used in
    * 3-source (e.g., ffma and flrp) instructions.
    */
   bool intel_vec4;

   unsigned max_unroll_iterations;

   nir_lower_int64_options lower_int64_options;
   nir_lower_doubles_options lower_doubles_options;
} nir_shader_compiler_options;

typedef struct nir_shader {
   /** list of uniforms (nir_variable) */
   struct exec_list uniforms;

   /** list of inputs (nir_variable) */
   struct exec_list inputs;

   /** list of outputs (nir_variable) */
   struct exec_list outputs;

   /** list of shared compute variables (nir_variable) */
   struct exec_list shared;

   /** Set of driver-specific options for the shader.
    *
    * The memory for the options is expected to be kept in a single static
    * copy by the driver.
    */
   const struct nir_shader_compiler_options *options;

   /** Various bits of compile-time information about a given shader */
   struct shader_info info;

   /** list of global variables in the shader (nir_variable) */
   struct exec_list globals;

   /** list of system value variables in the shader (nir_variable) */
   struct exec_list system_values;

   struct exec_list functions; /** < list of nir_function */

   /**
    * the highest index a load_input_*, load_uniform_*, etc. intrinsic can
    * access plus one
    */
   unsigned num_inputs, num_uniforms, num_outputs, num_shared;

   /** Size in bytes of required scratch space */
   unsigned scratch_size;

   /** Constant data associated with this shader.
    *
    * Constant data is loaded through load_constant intrinsics.  See also
    * nir_opt_large_constants.
    */
   void *constant_data;
   unsigned constant_data_size;
} nir_shader;

#define nir_foreach_function(func, shader) \
   foreach_list_typed(nir_function, func, node, &(shader)->functions)

static inline nir_function_impl *
nir_shader_get_entrypoint(nir_shader *shader)
{
   nir_function *func = NULL;

   nir_foreach_function(function, shader) {
      assert(func == NULL);
      if (function->is_entrypoint) {
         func = function;
#ifndef NDEBUG
         break;
#endif
      }
   }

   if (!func)
      return NULL;

   assert(func->num_params == 0);
   assert(func->impl);
   return func->impl;
}

nir_shader *nir_shader_create(void *mem_ctx,
                              gl_shader_stage stage,
                              const nir_shader_compiler_options *options,
                              shader_info *si);

nir_register *nir_local_reg_create(nir_function_impl *impl);

void nir_reg_remove(nir_register *reg);

/** Adds a variable to the appropriate list in nir_shader */
void nir_shader_add_variable(nir_shader *shader, nir_variable *var);

static inline void
nir_function_impl_add_variable(nir_function_impl *impl, nir_variable *var)
{
   assert(var->data.mode == nir_var_function_temp);
   exec_list_push_tail(&impl->locals, &var->node);
}

/** creates a variable, sets a few defaults, and adds it to the list */
nir_variable *nir_variable_create(nir_shader *shader,
                                  nir_variable_mode mode,
                                  const struct glsl_type *type,
                                  const char *name);
/** creates a local variable and adds it to the list */
nir_variable *nir_local_variable_create(nir_function_impl *impl,
                                        const struct glsl_type *type,
                                        const char *name);

/** creates a function and adds it to the shader's list of functions */
nir_function *nir_function_create(nir_shader *shader, const char *name);

nir_function_impl *nir_function_impl_create(nir_function *func);
/** creates a function_impl that isn't tied to any particular function */
nir_function_impl *nir_function_impl_create_bare(nir_shader *shader);

nir_block *nir_block_create(nir_shader *shader);
nir_if *nir_if_create(nir_shader *shader);
nir_loop *nir_loop_create(nir_shader *shader);

nir_function_impl *nir_cf_node_get_function(nir_cf_node *node);

/** requests that the given pieces of metadata be generated */
void nir_metadata_require(nir_function_impl *impl, nir_metadata required, ...);
/** dirties all but the preserved metadata */
void nir_metadata_preserve(nir_function_impl *impl, nir_metadata preserved);

/** creates an instruction with default swizzle/writemask/etc. with NULL registers */
nir_alu_instr *nir_alu_instr_create(nir_shader *shader, nir_op op);

nir_deref_instr *nir_deref_instr_create(nir_shader *shader,
                                        nir_deref_type deref_type);

nir_jump_instr *nir_jump_instr_create(nir_shader *shader, nir_jump_type type);

nir_load_const_instr *nir_load_const_instr_create(nir_shader *shader,
                                                  unsigned num_components,
                                                  unsigned bit_size);

nir_intrinsic_instr *nir_intrinsic_instr_create(nir_shader *shader,
                                                nir_intrinsic_op op);

nir_call_instr *nir_call_instr_create(nir_shader *shader,
                                      nir_function *callee);

nir_tex_instr *nir_tex_instr_create(nir_shader *shader, unsigned num_srcs);

nir_phi_instr *nir_phi_instr_create(nir_shader *shader);

nir_parallel_copy_instr *nir_parallel_copy_instr_create(nir_shader *shader);

nir_ssa_undef_instr *nir_ssa_undef_instr_create(nir_shader *shader,
                                                unsigned num_components,
                                                unsigned bit_size);

nir_const_value nir_alu_binop_identity(nir_op binop, unsigned bit_size);

/**
 * NIR Cursors and Instruction Insertion API
 * @{
 *
 * A tiny struct representing a point to insert/extract instructions or
 * control flow nodes.  Helps reduce the combinatorial explosion of possible
 * points to insert/extract.
 *
 * \sa nir_control_flow.h
 */
typedef enum {
   nir_cursor_before_block,
   nir_cursor_after_block,
   nir_cursor_before_instr,
   nir_cursor_after_instr,
} nir_cursor_option;

typedef struct {
   nir_cursor_option option;
   union {
      nir_block *block;
      nir_instr *instr;
   };
} nir_cursor;

static inline nir_block *
nir_cursor_current_block(nir_cursor cursor)
{
   if (cursor.option == nir_cursor_before_instr ||
       cursor.option == nir_cursor_after_instr) {
      return cursor.instr->block;
   } else {
      return cursor.block;
   }
}

bool nir_cursors_equal(nir_cursor a, nir_cursor b);

static inline nir_cursor
nir_before_block(nir_block *block)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_before_block;
   cursor.block = block;
   return cursor;
}

static inline nir_cursor
nir_after_block(nir_block *block)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_after_block;
   cursor.block = block;
   return cursor;
}

static inline nir_cursor
nir_before_instr(nir_instr *instr)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_before_instr;
   cursor.instr = instr;
   return cursor;
}

static inline nir_cursor
nir_after_instr(nir_instr *instr)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_after_instr;
   cursor.instr = instr;
   return cursor;
}

static inline nir_cursor
nir_after_block_before_jump(nir_block *block)
{
   nir_instr *last_instr = nir_block_last_instr(block);
   if (last_instr && last_instr->type == nir_instr_type_jump) {
      return nir_before_instr(last_instr);
   } else {
      return nir_after_block(block);
   }
}

static inline nir_cursor
nir_before_src(nir_src *src, bool is_if_condition)
{
   if (is_if_condition) {
      nir_block *prev_block =
         nir_cf_node_as_block(nir_cf_node_prev(&src->parent_if->cf_node));
      assert(!nir_block_ends_in_jump(prev_block));
      return nir_after_block(prev_block);
   } else if (src->parent_instr->type == nir_instr_type_phi) {
#ifndef NDEBUG
      nir_phi_instr *cond_phi = nir_instr_as_phi(src->parent_instr);
      bool found = false;
      nir_foreach_phi_src(phi_src, cond_phi) {
         if (phi_src->src.ssa == src->ssa) {
            found = true;
            break;
         }
      }
      assert(found);
#endif
      /* The LIST_ENTRY macro is a generic container-of macro, it just happens
       * to have a more specific name.
       */
      nir_phi_src *phi_src = LIST_ENTRY(nir_phi_src, src, src);
      return nir_after_block_before_jump(phi_src->pred);
   } else {
      return nir_before_instr(src->parent_instr);
   }
}

static inline nir_cursor
nir_before_cf_node(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_before_block(nir_cf_node_as_block(node));

   return nir_after_block(nir_cf_node_as_block(nir_cf_node_prev(node)));
}

static inline nir_cursor
nir_after_cf_node(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_after_block(nir_cf_node_as_block(node));

   return nir_before_block(nir_cf_node_as_block(nir_cf_node_next(node)));
}

static inline nir_cursor
nir_after_phis(nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_phi)
         return nir_before_instr(instr);
   }
   return nir_after_block(block);
}

static inline nir_cursor
nir_after_cf_node_and_phis(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_after_block(nir_cf_node_as_block(node));

   nir_block *block = nir_cf_node_as_block(nir_cf_node_next(node));

   return nir_after_phis(block);
}

static inline nir_cursor
nir_before_cf_list(struct exec_list *cf_list)
{
   nir_cf_node *first_node = exec_node_data(nir_cf_node,
                                            exec_list_get_head(cf_list), node);
   return nir_before_cf_node(first_node);
}

static inline nir_cursor
nir_after_cf_list(struct exec_list *cf_list)
{
   nir_cf_node *last_node = exec_node_data(nir_cf_node,
                                           exec_list_get_tail(cf_list), node);
   return nir_after_cf_node(last_node);
}

/**
 * Insert a NIR instruction at the given cursor.
 *
 * Note: This does not update the cursor.
 */
void nir_instr_insert(nir_cursor cursor, nir_instr *instr);

static inline void
nir_instr_insert_before(nir_instr *instr, nir_instr *before)
{
   nir_instr_insert(nir_before_instr(instr), before);
}

static inline void
nir_instr_insert_after(nir_instr *instr, nir_instr *after)
{
   nir_instr_insert(nir_after_instr(instr), after);
}

static inline void
nir_instr_insert_before_block(nir_block *block, nir_instr *before)
{
   nir_instr_insert(nir_before_block(block), before);
}

static inline void
nir_instr_insert_after_block(nir_block *block, nir_instr *after)
{
   nir_instr_insert(nir_after_block(block), after);
}

static inline void
nir_instr_insert_before_cf(nir_cf_node *node, nir_instr *before)
{
   nir_instr_insert(nir_before_cf_node(node), before);
}

static inline void
nir_instr_insert_after_cf(nir_cf_node *node, nir_instr *after)
{
   nir_instr_insert(nir_after_cf_node(node), after);
}

static inline void
nir_instr_insert_before_cf_list(struct exec_list *list, nir_instr *before)
{
   nir_instr_insert(nir_before_cf_list(list), before);
}

static inline void
nir_instr_insert_after_cf_list(struct exec_list *list, nir_instr *after)
{
   nir_instr_insert(nir_after_cf_list(list), after);
}

void nir_instr_remove_v(nir_instr *instr);

static inline nir_cursor
nir_instr_remove(nir_instr *instr)
{
   nir_cursor cursor;
   nir_instr *prev = nir_instr_prev(instr);
   if (prev) {
      cursor = nir_after_instr(prev);
   } else {
      cursor = nir_before_block(instr->block);
   }
   nir_instr_remove_v(instr);
   return cursor;
}

/** @} */

nir_ssa_def *nir_instr_ssa_def(nir_instr *instr);

typedef bool (*nir_foreach_ssa_def_cb)(nir_ssa_def *def, void *state);
typedef bool (*nir_foreach_dest_cb)(nir_dest *dest, void *state);
typedef bool (*nir_foreach_src_cb)(nir_src *src, void *state);
bool nir_foreach_ssa_def(nir_instr *instr, nir_foreach_ssa_def_cb cb,
                         void *state);
bool nir_foreach_dest(nir_instr *instr, nir_foreach_dest_cb cb, void *state);
bool nir_foreach_src(nir_instr *instr, nir_foreach_src_cb cb, void *state);

nir_const_value *nir_src_as_const_value(nir_src src);

#define NIR_SRC_AS_(name, c_type, type_enum, cast_macro)                \
static inline c_type *                                                  \
nir_src_as_ ## name (nir_src src)                                       \
{                                                                       \
    return src.is_ssa && src.ssa->parent_instr->type == type_enum       \
           ? cast_macro(src.ssa->parent_instr) : NULL;                  \
}

NIR_SRC_AS_(alu_instr, nir_alu_instr, nir_instr_type_alu, nir_instr_as_alu)
NIR_SRC_AS_(intrinsic, nir_intrinsic_instr,
            nir_instr_type_intrinsic, nir_instr_as_intrinsic)
NIR_SRC_AS_(deref, nir_deref_instr, nir_instr_type_deref, nir_instr_as_deref)

bool nir_src_is_dynamically_uniform(nir_src src);
bool nir_srcs_equal(nir_src src1, nir_src src2);
bool nir_instrs_equal(const nir_instr *instr1, const nir_instr *instr2);
void nir_instr_rewrite_src(nir_instr *instr, nir_src *src, nir_src new_src);
void nir_instr_move_src(nir_instr *dest_instr, nir_src *dest, nir_src *src);
void nir_if_rewrite_condition(nir_if *if_stmt, nir_src new_src);
void nir_instr_rewrite_dest(nir_instr *instr, nir_dest *dest,
                            nir_dest new_dest);

void nir_ssa_dest_init(nir_instr *instr, nir_dest *dest,
                       unsigned num_components, unsigned bit_size,
                       const char *name);
void nir_ssa_def_init(nir_instr *instr, nir_ssa_def *def,
                      unsigned num_components, unsigned bit_size,
                      const char *name);
static inline void
nir_ssa_dest_init_for_type(nir_instr *instr, nir_dest *dest,
                           const struct glsl_type *type,
                           const char *name)
{
   assert(glsl_type_is_vector_or_scalar(type));
   nir_ssa_dest_init(instr, dest, glsl_get_components(type),
                     glsl_get_bit_size(type), name);
}
void nir_ssa_def_rewrite_uses(nir_ssa_def *def, nir_src new_src);
void nir_ssa_def_rewrite_uses_after(nir_ssa_def *def, nir_src new_src,
                                    nir_instr *after_me);

nir_component_mask_t nir_ssa_def_components_read(const nir_ssa_def *def);

/*
 * finds the next basic block in source-code order, returns NULL if there is
 * none
 */

nir_block *nir_block_cf_tree_next(nir_block *block);

/* Performs the opposite of nir_block_cf_tree_next() */

nir_block *nir_block_cf_tree_prev(nir_block *block);

/* Gets the first block in a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_first(nir_cf_node *node);

/* Gets the last block in a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_last(nir_cf_node *node);

/* Gets the next block after a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_next(nir_cf_node *node);

/* Macros for loops that visit blocks in source-code order */

#define nir_foreach_block(block, impl) \
   for (nir_block *block = nir_start_block(impl); block != NULL; \
        block = nir_block_cf_tree_next(block))

#define nir_foreach_block_safe(block, impl) \
   for (nir_block *block = nir_start_block(impl), \
        *next = nir_block_cf_tree_next(block); \
        block != NULL; \
        block = next, next = nir_block_cf_tree_next(block))

#define nir_foreach_block_reverse(block, impl) \
   for (nir_block *block = nir_impl_last_block(impl); block != NULL; \
        block = nir_block_cf_tree_prev(block))

#define nir_foreach_block_reverse_safe(block, impl) \
   for (nir_block *block = nir_impl_last_block(impl), \
        *prev = nir_block_cf_tree_prev(block); \
        block != NULL; \
        block = prev, prev = nir_block_cf_tree_prev(block))

#define nir_foreach_block_in_cf_node(block, node) \
   for (nir_block *block = nir_cf_node_cf_tree_first(node); \
        block != nir_cf_node_cf_tree_next(node); \
        block = nir_block_cf_tree_next(block))

/* If the following CF node is an if, this function returns that if.
 * Otherwise, it returns NULL.
 */
nir_if *nir_block_get_following_if(nir_block *block);

nir_loop *nir_block_get_following_loop(nir_block *block);

void nir_index_local_regs(nir_function_impl *impl);
void nir_index_ssa_defs(nir_function_impl *impl);
unsigned nir_index_instrs(nir_function_impl *impl);

void nir_index_blocks(nir_function_impl *impl);

void nir_print_shader(nir_shader *shader, FILE *fp);
void nir_print_shader_annotated(nir_shader *shader, FILE *fp, struct hash_table *errors);
void nir_print_instr(const nir_instr *instr, FILE *fp);
void nir_print_deref(const nir_deref_instr *deref, FILE *fp);

/** Shallow clone of a single ALU instruction. */
nir_alu_instr *nir_alu_instr_clone(nir_shader *s, const nir_alu_instr *orig);

nir_shader *nir_shader_clone(void *mem_ctx, const nir_shader *s);
nir_function_impl *nir_function_impl_clone(nir_shader *shader,
                                           const nir_function_impl *fi);
nir_constant *nir_constant_clone(const nir_constant *c, nir_variable *var);
nir_variable *nir_variable_clone(const nir_variable *c, nir_shader *shader);

void nir_shader_replace(nir_shader *dest, nir_shader *src);

void nir_shader_serialize_deserialize(nir_shader *s);

#ifndef NDEBUG
void nir_validate_shader(nir_shader *shader, const char *when);
void nir_metadata_set_validation_flag(nir_shader *shader);
void nir_metadata_check_validation_flag(nir_shader *shader);

static inline bool
should_skip_nir(const char *name)
{
   static const char *list = NULL;
   if (!list) {
      /* Comma separated list of names to skip. */
      list = getenv("NIR_SKIP");
      if (!list)
         list = "";
   }

   if (!list[0])
      return false;

   return comma_separated_list_contains(list, name);
}

static inline bool
should_clone_nir(void)
{
   static int should_clone = -1;
   if (should_clone < 0)
      should_clone = env_var_as_boolean("NIR_TEST_CLONE", false);

   return should_clone;
}

static inline bool
should_serialize_deserialize_nir(void)
{
   static int test_serialize = -1;
   if (test_serialize < 0)
      test_serialize = env_var_as_boolean("NIR_TEST_SERIALIZE", false);

   return test_serialize;
}

static inline bool
should_print_nir(void)
{
   static int should_print = -1;
   if (should_print < 0)
      should_print = env_var_as_boolean("NIR_PRINT", false);

   return should_print;
}
#else
static inline void nir_validate_shader(nir_shader *shader, const char *when) { (void) shader; (void)when; }
static inline void nir_metadata_set_validation_flag(nir_shader *shader) { (void) shader; }
static inline void nir_metadata_check_validation_flag(nir_shader *shader) { (void) shader; }
static inline bool should_skip_nir(UNUSED const char *pass_name) { return false; }
static inline bool should_clone_nir(void) { return false; }
static inline bool should_serialize_deserialize_nir(void) { return false; }
static inline bool should_print_nir(void) { return false; }
#endif /* NDEBUG */

#define _PASS(pass, nir, do_pass) do {                               \
   if (should_skip_nir(#pass)) {                                     \
      printf("skipping %s\n", #pass);                                \
      break;                                                         \
   }                                                                 \
   do_pass                                                           \
   nir_validate_shader(nir, "after " #pass);                         \
   if (should_clone_nir()) {                                         \
      nir_shader *clone = nir_shader_clone(ralloc_parent(nir), nir); \
      nir_shader_replace(nir, clone);                                \
   }                                                                 \
   if (should_serialize_deserialize_nir()) {                         \
      nir_shader_serialize_deserialize(nir);                         \
   }                                                                 \
} while (0)

#define NIR_PASS(progress, nir, pass, ...) _PASS(pass, nir,          \
   nir_metadata_set_validation_flag(nir);                            \
   if (should_print_nir())                                           \
      printf("%s\n", #pass);                                         \
   if (pass(nir, ##__VA_ARGS__)) {                                   \
      progress = true;                                               \
      if (should_print_nir())                                        \
         nir_print_shader(nir, stdout);                              \
      nir_metadata_check_validation_flag(nir);                       \
   }                                                                 \
)

#define NIR_PASS_V(nir, pass, ...) _PASS(pass, nir,                  \
   if (should_print_nir())                                           \
      printf("%s\n", #pass);                                         \
   pass(nir, ##__VA_ARGS__);                                         \
   if (should_print_nir())                                           \
      nir_print_shader(nir, stdout);                                 \
)

#define NIR_SKIP(name) should_skip_nir(#name)

/** An instruction filtering callback
 *
 * Returns true if the instruction should be processed and false otherwise.
 */
typedef bool (*nir_instr_filter_cb)(const nir_instr *, const void *);

/** A simple instruction lowering callback
 *
 * Many instruction lowering passes can be written as a simple function which
 * takes an instruction as its input and returns a sequence of instructions
 * that implement the consumed instruction.  This function type represents
 * such a lowering function.  When called, a function with this prototype
 * should either return NULL indicating that no lowering needs to be done or
 * emit a sequence of instructions using the provided builder (whose cursor
 * will already be placed after the instruction to be lowered) and return the
 * resulting nir_ssa_def.
 */
typedef nir_ssa_def *(*nir_lower_instr_cb)(struct nir_builder *,
                                           nir_instr *, void *);

/**
 * Special return value for nir_lower_instr_cb when some progress occurred
 * (like changing an input to the instr) that didn't result in a replacement
 * SSA def being generated.
 */
#define NIR_LOWER_INSTR_PROGRESS ((nir_ssa_def *)(uintptr_t)1)

/** Iterate over all the instructions in a nir_function_impl and lower them
 *  using the provided callbacks
 *
 * This function implements the guts of a standard lowering pass for you.  It
 * iterates over all of the instructions in a nir_function_impl and calls the
 * filter callback on each one.  If the filter callback returns true, it then
 * calls the lowering call back on the instruction.  (Splitting it this way
 * allows us to avoid some save/restore work for instructions we know won't be
 * lowered.)  If the instruction is dead after the lowering is complete, it
 * will be removed.  If new instructions are added, the lowering callback will
 * also be called on them in case multiple lowerings are required.
 *
 * The metadata for the nir_function_impl will also be updated.  If any blocks
 * are added (they cannot be removed), dominance and block indices will be
 * invalidated.
 */
bool nir_function_impl_lower_instructions(nir_function_impl *impl,
                                          nir_instr_filter_cb filter,
                                          nir_lower_instr_cb lower,
                                          void *cb_data);
bool nir_shader_lower_instructions(nir_shader *shader,
                                   nir_instr_filter_cb filter,
                                   nir_lower_instr_cb lower,
                                   void *cb_data);

void nir_calc_dominance_impl(nir_function_impl *impl);
void nir_calc_dominance(nir_shader *shader);

nir_block *nir_dominance_lca(nir_block *b1, nir_block *b2);
bool nir_block_dominates(nir_block *parent, nir_block *child);
bool nir_block_is_unreachable(nir_block *block);

void nir_dump_dom_tree_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_dom_tree(nir_shader *shader, FILE *fp);

void nir_dump_dom_frontier_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_dom_frontier(nir_shader *shader, FILE *fp);

void nir_dump_cfg_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_cfg(nir_shader *shader, FILE *fp);

int nir_gs_count_vertices(const nir_shader *shader);

bool nir_shrink_vec_array_vars(nir_shader *shader, nir_variable_mode modes);
bool nir_split_array_vars(nir_shader *shader, nir_variable_mode modes);
bool nir_split_var_copies(nir_shader *shader);
bool nir_split_per_member_structs(nir_shader *shader);
bool nir_split_struct_vars(nir_shader *shader, nir_variable_mode modes);

bool nir_lower_returns_impl(nir_function_impl *impl);
bool nir_lower_returns(nir_shader *shader);

void nir_inline_function_impl(struct nir_builder *b,
                              const nir_function_impl *impl,
                              nir_ssa_def **params);
bool nir_inline_functions(nir_shader *shader);

bool nir_propagate_invariant(nir_shader *shader);

void nir_lower_var_copy_instr(nir_intrinsic_instr *copy, nir_shader *shader);
void nir_lower_deref_copy_instr(struct nir_builder *b,
                                nir_intrinsic_instr *copy);
bool nir_lower_var_copies(nir_shader *shader);

void nir_fixup_deref_modes(nir_shader *shader);

bool nir_lower_global_vars_to_local(nir_shader *shader);

typedef enum {
   nir_lower_direct_array_deref_of_vec_load     = (1 << 0),
   nir_lower_indirect_array_deref_of_vec_load   = (1 << 1),
   nir_lower_direct_array_deref_of_vec_store    = (1 << 2),
   nir_lower_indirect_array_deref_of_vec_store  = (1 << 3),
} nir_lower_array_deref_of_vec_options;

bool nir_lower_array_deref_of_vec(nir_shader *shader, nir_variable_mode modes,
                                  nir_lower_array_deref_of_vec_options options);

bool nir_lower_indirect_derefs(nir_shader *shader, nir_variable_mode modes);

bool nir_lower_locals_to_regs(nir_shader *shader);

void nir_lower_io_to_temporaries(nir_shader *shader,
                                 nir_function_impl *entrypoint,
                                 bool outputs, bool inputs);

bool nir_lower_vars_to_scratch(nir_shader *shader,
                               nir_variable_mode modes,
                               int size_threshold,
                               glsl_type_size_align_func size_align);

void nir_shader_gather_info(nir_shader *shader, nir_function_impl *entrypoint);

void nir_gather_ssa_types(nir_function_impl *impl,
                          BITSET_WORD *float_types,
                          BITSET_WORD *int_types);

void nir_assign_var_locations(struct exec_list *var_list, unsigned *size,
                              int (*type_size)(const struct glsl_type *, bool));

/* Some helpers to do very simple linking */
bool nir_remove_unused_varyings(nir_shader *producer, nir_shader *consumer);
bool nir_remove_unused_io_vars(nir_shader *shader, struct exec_list *var_list,
                               uint64_t *used_by_other_stage,
                               uint64_t *used_by_other_stage_patches);
void nir_compact_varyings(nir_shader *producer, nir_shader *consumer,
                          bool default_to_smooth_interp);
void nir_link_xfb_varyings(nir_shader *producer, nir_shader *consumer);
bool nir_link_opt_varyings(nir_shader *producer, nir_shader *consumer);


void nir_assign_io_var_locations(struct exec_list *var_list,
                                 unsigned *size,
                                 gl_shader_stage stage);

typedef enum {
   /* If set, this causes all 64-bit IO operations to be lowered on-the-fly
    * to 32-bit operations.  This is only valid for nir_var_shader_in/out
    * modes.
    */
   nir_lower_io_lower_64bit_to_32 = (1 << 0),

   /* If set, this forces all non-flat fragment shader inputs to be
    * interpolated as if with the "sample" qualifier.  This requires
    * nir_shader_compiler_options::use_interpolated_input_intrinsics.
    */
   nir_lower_io_force_sample_interpolation = (1 << 1),
} nir_lower_io_options;
bool nir_lower_io(nir_shader *shader,
                  nir_variable_mode modes,
                  int (*type_size)(const struct glsl_type *, bool),
                  nir_lower_io_options);

bool nir_io_add_const_offset_to_base(nir_shader *nir, nir_variable_mode mode);

bool
nir_lower_vars_to_explicit_types(nir_shader *shader,
                                 nir_variable_mode modes,
                                 glsl_type_size_align_func type_info);

typedef enum {
   /**
    * An address format which is a simple 32-bit global GPU address.
    */
   nir_address_format_32bit_global,

   /**
    * An address format which is a simple 64-bit global GPU address.
    */
   nir_address_format_64bit_global,

   /**
    * An address format which is a bounds-checked 64-bit global GPU address.
    *
    * The address is comprised as a 32-bit vec4 where .xy are a uint64_t base
    * address stored with the low bits in .x and high bits in .y, .z is a
    * size, and .w is an offset.  When the final I/O operation is lowered, .w
    * is checked against .z and the operation is predicated on the result.
    */
   nir_address_format_64bit_bounded_global,

   /**
    * An address format which is comprised of a vec2 where the first
    * component is a buffer index and the second is an offset.
    */
   nir_address_format_32bit_index_offset,

   /**
    * An address format which is a simple 32-bit offset.
    */
   nir_address_format_32bit_offset,

   /**
    * An address format representing a purely logical addressing model.  In
    * this model, all deref chains must be complete from the dereference
    * operation to the variable.  Cast derefs are not allowed.  These
    * addresses will be 32-bit scalars but the format is immaterial because
    * you can always chase the chain.
    */
   nir_address_format_logical,
} nir_address_format;

static inline unsigned
nir_address_format_bit_size(nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:           return 32;
   case nir_address_format_64bit_global:           return 64;
   case nir_address_format_64bit_bounded_global:   return 32;
   case nir_address_format_32bit_index_offset:     return 32;
   case nir_address_format_32bit_offset:           return 32;
   case nir_address_format_logical:                return 32;
   }
   unreachable("Invalid address format");
}

static inline unsigned
nir_address_format_num_components(nir_address_format addr_format)
{
   switch (addr_format) {
   case nir_address_format_32bit_global:           return 1;
   case nir_address_format_64bit_global:           return 1;
   case nir_address_format_64bit_bounded_global:   return 4;
   case nir_address_format_32bit_index_offset:     return 2;
   case nir_address_format_32bit_offset:           return 1;
   case nir_address_format_logical:                return 1;
   }
   unreachable("Invalid address format");
}

static inline const struct glsl_type *
nir_address_format_to_glsl_type(nir_address_format addr_format)
{
   unsigned bit_size = nir_address_format_bit_size(addr_format);
   assert(bit_size == 32 || bit_size == 64);
   return glsl_vector_type(bit_size == 32 ? GLSL_TYPE_UINT : GLSL_TYPE_UINT64,
                           nir_address_format_num_components(addr_format));
}

const nir_const_value *nir_address_format_null_value(nir_address_format addr_format);

nir_ssa_def *nir_build_addr_ieq(struct nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                                nir_address_format addr_format);

nir_ssa_def *nir_build_addr_isub(struct nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                                 nir_address_format addr_format);

nir_ssa_def * nir_explicit_io_address_from_deref(struct nir_builder *b,
                                                 nir_deref_instr *deref,
                                                 nir_ssa_def *base_addr,
                                                 nir_address_format addr_format);
void nir_lower_explicit_io_instr(struct nir_builder *b,
                                 nir_intrinsic_instr *io_instr,
                                 nir_ssa_def *addr,
                                 nir_address_format addr_format);

bool nir_lower_explicit_io(nir_shader *shader,
                           nir_variable_mode modes,
                           nir_address_format);

nir_src *nir_get_io_offset_src(nir_intrinsic_instr *instr);
nir_src *nir_get_io_vertex_index_src(nir_intrinsic_instr *instr);

bool nir_is_per_vertex_io(const nir_variable *var, gl_shader_stage stage);

bool nir_lower_regs_to_ssa_impl(nir_function_impl *impl);
bool nir_lower_regs_to_ssa(nir_shader *shader);
bool nir_lower_vars_to_ssa(nir_shader *shader);

bool nir_remove_dead_derefs(nir_shader *shader);
bool nir_remove_dead_derefs_impl(nir_function_impl *impl);
bool nir_remove_dead_variables(nir_shader *shader, nir_variable_mode modes);
bool nir_lower_constant_initializers(nir_shader *shader,
                                     nir_variable_mode modes);

bool nir_move_vec_src_uses_to_dest(nir_shader *shader);
bool nir_lower_vec_to_movs(nir_shader *shader);
void nir_lower_alpha_test(nir_shader *shader, enum compare_func func,
                          bool alpha_to_one);
bool nir_lower_alu(nir_shader *shader);

bool nir_lower_flrp(nir_shader *shader, unsigned lowering_mask,
                    bool always_precise, bool have_ffma);

bool nir_lower_alu_to_scalar(nir_shader *shader, nir_instr_filter_cb cb, const void *data);
bool nir_lower_bool_to_float(nir_shader *shader);
bool nir_lower_bool_to_int32(nir_shader *shader);
bool nir_lower_int_to_float(nir_shader *shader);
bool nir_lower_load_const_to_scalar(nir_shader *shader);
bool nir_lower_read_invocation_to_scalar(nir_shader *shader);
bool nir_lower_phis_to_scalar(nir_shader *shader);
void nir_lower_io_arrays_to_elements(nir_shader *producer, nir_shader *consumer);
void nir_lower_io_arrays_to_elements_no_indirects(nir_shader *shader,
                                                  bool outputs_only);
void nir_lower_io_to_scalar(nir_shader *shader, nir_variable_mode mask);
void nir_lower_io_to_scalar_early(nir_shader *shader, nir_variable_mode mask);
bool nir_lower_io_to_vector(nir_shader *shader, nir_variable_mode mask);

void nir_lower_fragcoord_wtrans(nir_shader *shader);
void nir_lower_viewport_transform(nir_shader *shader);
bool nir_lower_uniforms_to_ubo(nir_shader *shader, int multiplier);

typedef struct nir_lower_subgroups_options {
   uint8_t subgroup_size;
   uint8_t ballot_bit_size;
   bool lower_to_scalar:1;
   bool lower_vote_trivial:1;
   bool lower_vote_eq_to_ballot:1;
   bool lower_subgroup_masks:1;
   bool lower_shuffle:1;
   bool lower_shuffle_to_32bit:1;
   bool lower_quad:1;
} nir_lower_subgroups_options;

bool nir_lower_subgroups(nir_shader *shader,
                         const nir_lower_subgroups_options *options);

bool nir_lower_system_values(nir_shader *shader);

enum PACKED nir_lower_tex_packing {
   nir_lower_tex_packing_none = 0,
   /* The sampler returns up to 2 32-bit words of half floats or 16-bit signed
    * or unsigned ints based on the sampler type
    */
   nir_lower_tex_packing_16,
   /* The sampler returns 1 32-bit word of 4x8 unorm */
   nir_lower_tex_packing_8,
};

typedef struct nir_lower_tex_options {
   /**
    * bitmask of (1 << GLSL_SAMPLER_DIM_x) to control for which
    * sampler types a texture projector is lowered.
    */
   unsigned lower_txp;

   /**
    * If true, lower away nir_tex_src_offset for all texelfetch instructions.
    */
   bool lower_txf_offset;

   /**
    * If true, lower away nir_tex_src_offset for all rect textures.
    */
   bool lower_rect_offset;

   /**
    * If true, lower rect textures to 2D, using txs to fetch the
    * texture dimensions and dividing the texture coords by the
    * texture dims to normalize.
    */
   bool lower_rect;

   /**
    * If true, convert yuv to rgb.
    */
   unsigned lower_y_uv_external;
   unsigned lower_y_u_v_external;
   unsigned lower_yx_xuxv_external;
   unsigned lower_xy_uxvx_external;
   unsigned lower_ayuv_external;
   unsigned lower_xyuv_external;

   /**
    * To emulate certain texture wrap modes, this can be used
    * to saturate the specified tex coord to [0.0, 1.0].  The
    * bits are according to sampler #, ie. if, for example:
    *
    *   (conf->saturate_s & (1 << n))
    *
    * is true, then the s coord for sampler n is saturated.
    *
    * Note that clamping must happen *after* projector lowering
    * so any projected texture sample instruction with a clamped
    * coordinate gets automatically lowered, regardless of the
    * 'lower_txp' setting.
    */
   unsigned saturate_s;
   unsigned saturate_t;
   unsigned saturate_r;

   /* Bitmask of textures that need swizzling.
    *
    * If (swizzle_result & (1 << texture_index)), then the swizzle in
    * swizzles[texture_index] is applied to the result of the texturing
    * operation.
    */
   unsigned swizzle_result;

   /* A swizzle for each texture.  Values 0-3 represent x, y, z, or w swizzles
    * while 4 and 5 represent 0 and 1 respectively.
    */
   uint8_t swizzles[32][4];

   /* Can be used to scale sampled values in range required by the format. */
   float scale_factors[32];

   /**
    * Bitmap of textures that need srgb to linear conversion.  If
    * (lower_srgb & (1 << texture_index)) then the rgb (xyz) components
    * of the texture are lowered to linear.
    */
   unsigned lower_srgb;

   /**
    * If true, lower nir_texop_tex on shaders that doesn't support implicit
    * LODs to nir_texop_txl.
    */
   bool lower_tex_without_implicit_lod;

   /**
    * If true, lower nir_texop_txd on cube maps with nir_texop_txl.
    */
   bool lower_txd_cube_map;

   /**
    * If true, lower nir_texop_txd on 3D surfaces with nir_texop_txl.
    */
   bool lower_txd_3d;

   /**
    * If true, lower nir_texop_txd on shadow samplers (except cube maps)
    * with nir_texop_txl. Notice that cube map shadow samplers are lowered
    * with lower_txd_cube_map.
    */
   bool lower_txd_shadow;

   /**
    * If true, lower nir_texop_txd on all samplers to a nir_texop_txl.
    * Implies lower_txd_cube_map and lower_txd_shadow.
    */
   bool lower_txd;

   /**
    * If true, lower nir_texop_txb that try to use shadow compare and min_lod
    * at the same time to a nir_texop_lod, some math, and nir_texop_tex.
    */
   bool lower_txb_shadow_clamp;

   /**
    * If true, lower nir_texop_txd on shadow samplers when it uses min_lod
    * with nir_texop_txl.  This includes cube maps.
    */
   bool lower_txd_shadow_clamp;

   /**
    * If true, lower nir_texop_txd on when it uses both offset and min_lod
    * with nir_texop_txl.  This includes cube maps.
    */
   bool lower_txd_offset_clamp;

   /**
    * If true, lower nir_texop_txd with min_lod to a nir_texop_txl if the
    * sampler is bindless.
    */
   bool lower_txd_clamp_bindless_sampler;

   /**
    * If true, lower nir_texop_txd with min_lod to a nir_texop_txl if the
    * sampler index is not statically determinable to be less than 16.
    */
   bool lower_txd_clamp_if_sampler_index_not_lt_16;

   /**
    * If true, lower nir_texop_txs with a non-0-lod into nir_texop_txs with
    * 0-lod followed by a nir_ishr.
    */
   bool lower_txs_lod;

   /**
    * If true, apply a .bagr swizzle on tg4 results to handle Broadcom's
    * mixed-up tg4 locations.
    */
   bool lower_tg4_broadcom_swizzle;

   /**
    * If true, lowers tg4 with 4 constant offsets to 4 tg4 calls
    */
   bool lower_tg4_offsets;

   enum nir_lower_tex_packing lower_tex_packing[32];
} nir_lower_tex_options;

bool nir_lower_tex(nir_shader *shader,
                   const nir_lower_tex_options *options);

enum nir_lower_non_uniform_access_type {
   nir_lower_non_uniform_ubo_access     = (1 << 0),
   nir_lower_non_uniform_ssbo_access    = (1 << 1),
   nir_lower_non_uniform_texture_access = (1 << 2),
   nir_lower_non_uniform_image_access   = (1 << 3),
};

bool nir_lower_non_uniform_access(nir_shader *shader,
                                  enum nir_lower_non_uniform_access_type);

enum nir_lower_idiv_path {
   nir_lower_idiv_nv50,
   nir_lower_idiv_llvm,
};

bool nir_lower_idiv(nir_shader *shader, enum nir_lower_idiv_path path);

bool nir_lower_input_attachments(nir_shader *shader, bool use_fragcoord_sysval);

bool nir_lower_clip_vs(nir_shader *shader, unsigned ucp_enables, bool use_vars);
bool nir_lower_clip_gs(nir_shader *shader, unsigned ucp_enables);
bool nir_lower_clip_fs(nir_shader *shader, unsigned ucp_enables);
bool nir_lower_clip_cull_distance_arrays(nir_shader *nir);

bool nir_lower_frexp(nir_shader *nir);

void nir_lower_two_sided_color(nir_shader *shader);

bool nir_lower_clamp_color_outputs(nir_shader *shader);

void nir_lower_passthrough_edgeflags(nir_shader *shader);
bool nir_lower_patch_vertices(nir_shader *nir, unsigned static_count,
                              const gl_state_index16 *uniform_state_tokens);

typedef struct nir_lower_wpos_ytransform_options {
   gl_state_index16 state_tokens[STATE_LENGTH];
   bool fs_coord_origin_upper_left :1;
   bool fs_coord_origin_lower_left :1;
   bool fs_coord_pixel_center_integer :1;
   bool fs_coord_pixel_center_half_integer :1;
} nir_lower_wpos_ytransform_options;

bool nir_lower_wpos_ytransform(nir_shader *shader,
                               const nir_lower_wpos_ytransform_options *options);
bool nir_lower_wpos_center(nir_shader *shader, const bool for_sample_shading);

bool nir_lower_fb_read(nir_shader *shader);

typedef struct nir_lower_drawpixels_options {
   gl_state_index16 texcoord_state_tokens[STATE_LENGTH];
   gl_state_index16 scale_state_tokens[STATE_LENGTH];
   gl_state_index16 bias_state_tokens[STATE_LENGTH];
   unsigned drawpix_sampler;
   unsigned pixelmap_sampler;
   bool pixel_maps :1;
   bool scale_and_bias :1;
} nir_lower_drawpixels_options;

void nir_lower_drawpixels(nir_shader *shader,
                          const nir_lower_drawpixels_options *options);

typedef struct nir_lower_bitmap_options {
   unsigned sampler;
   bool swizzle_xxxx;
} nir_lower_bitmap_options;

void nir_lower_bitmap(nir_shader *shader, const nir_lower_bitmap_options *options);

bool nir_lower_atomics_to_ssbo(nir_shader *shader, unsigned ssbo_offset);

typedef enum  {
   nir_lower_int_source_mods = 1 << 0,
   nir_lower_float_source_mods = 1 << 1,
   nir_lower_triop_abs = 1 << 2,
   nir_lower_all_source_mods = (1 << 3) - 1
} nir_lower_to_source_mods_flags;


bool nir_lower_to_source_mods(nir_shader *shader, nir_lower_to_source_mods_flags options);

bool nir_lower_gs_intrinsics(nir_shader *shader);

typedef unsigned (*nir_lower_bit_size_callback)(const nir_alu_instr *, void *);

bool nir_lower_bit_size(nir_shader *shader,
                        nir_lower_bit_size_callback callback,
                        void *callback_data);

nir_lower_int64_options nir_lower_int64_op_to_options_mask(nir_op opcode);
bool nir_lower_int64(nir_shader *shader, nir_lower_int64_options options);

nir_lower_doubles_options nir_lower_doubles_op_to_options_mask(nir_op opcode);
bool nir_lower_doubles(nir_shader *shader, const nir_shader *softfp64,
                       nir_lower_doubles_options options);
bool nir_lower_pack(nir_shader *shader);

bool nir_lower_point_size(nir_shader *shader, float min, float max);

typedef enum {
   nir_lower_interpolation_at_sample = (1 << 1),
   nir_lower_interpolation_at_offset = (1 << 2),
   nir_lower_interpolation_centroid  = (1 << 3),
   nir_lower_interpolation_pixel     = (1 << 4),
   nir_lower_interpolation_sample    = (1 << 5),
} nir_lower_interpolation_options;

bool nir_lower_interpolation(nir_shader *shader,
                             nir_lower_interpolation_options options);

bool nir_normalize_cubemap_coords(nir_shader *shader);

void nir_live_ssa_defs_impl(nir_function_impl *impl);

void nir_loop_analyze_impl(nir_function_impl *impl,
                           nir_variable_mode indirect_mask);

bool nir_ssa_defs_interfere(nir_ssa_def *a, nir_ssa_def *b);

bool nir_repair_ssa_impl(nir_function_impl *impl);
bool nir_repair_ssa(nir_shader *shader);

void nir_convert_loop_to_lcssa(nir_loop *loop);
bool nir_convert_to_lcssa(nir_shader *shader, bool skip_invariants, bool skip_bool_invariants);
bool* nir_divergence_analysis(nir_shader *shader, nir_divergence_options options);

/* If phi_webs_only is true, only convert SSA values involved in phi nodes to
 * registers.  If false, convert all values (even those not involved in a phi
 * node) to registers.
 */
bool nir_convert_from_ssa(nir_shader *shader, bool phi_webs_only);

bool nir_lower_phis_to_regs_block(nir_block *block);
bool nir_lower_ssa_defs_to_regs_block(nir_block *block);
bool nir_rematerialize_derefs_in_use_blocks_impl(nir_function_impl *impl);

bool nir_lower_samplers(nir_shader *shader);

/* This is here for unit tests. */
bool nir_opt_comparison_pre_impl(nir_function_impl *impl);

bool nir_opt_comparison_pre(nir_shader *shader);

bool nir_opt_algebraic(nir_shader *shader);
bool nir_opt_algebraic_before_ffma(nir_shader *shader);
bool nir_opt_algebraic_late(nir_shader *shader);
bool nir_opt_constant_folding(nir_shader *shader);

bool nir_opt_combine_stores(nir_shader *shader, nir_variable_mode modes);

bool nir_copy_prop(nir_shader *shader);

bool nir_opt_copy_prop_vars(nir_shader *shader);

bool nir_opt_cse(nir_shader *shader);

bool nir_opt_dce(nir_shader *shader);

bool nir_opt_dead_cf(nir_shader *shader);

bool nir_opt_dead_write_vars(nir_shader *shader);

bool nir_opt_deref_impl(nir_function_impl *impl);
bool nir_opt_deref(nir_shader *shader);

bool nir_opt_find_array_copies(nir_shader *shader);

bool nir_opt_gcm(nir_shader *shader, bool value_number);

bool nir_opt_idiv_const(nir_shader *shader, unsigned min_bit_size);

bool nir_opt_if(nir_shader *shader, bool aggressive_last_continue);

bool nir_opt_intrinsics(nir_shader *shader);

bool nir_opt_large_constants(nir_shader *shader,
                             glsl_type_size_align_func size_align,
                             unsigned threshold);

bool nir_opt_loop_unroll(nir_shader *shader, nir_variable_mode indirect_mask);

typedef enum {
    nir_move_const_undef = (1 << 0),
    nir_move_load_ubo    = (1 << 1),
    nir_move_load_input  = (1 << 2),
    nir_move_comparisons = (1 << 3),
} nir_move_options;

bool nir_can_move_instr(nir_instr *instr, nir_move_options options);

bool nir_opt_sink(nir_shader *shader, nir_move_options options);

bool nir_opt_move(nir_shader *shader, nir_move_options options);

bool nir_opt_peephole_select(nir_shader *shader, unsigned limit,
                             bool indirect_load_ok, bool expensive_alu_ok);

bool nir_opt_rematerialize_compares(nir_shader *shader);

bool nir_opt_remove_phis(nir_shader *shader);
bool nir_opt_remove_phis_block(nir_block *block);

bool nir_opt_shrink_load(nir_shader *shader);

bool nir_opt_trivial_continues(nir_shader *shader);

bool nir_opt_undef(nir_shader *shader);

bool nir_opt_vectorize(nir_shader *shader);

bool nir_opt_conditional_discard(nir_shader *shader);

bool nir_opt_load_store_vectorize(nir_shader *shader, nir_variable_mode modes,
                                  int (*type_size)(nir_variable_mode mode, const struct glsl_type *),
                                  int (*align)(nir_variable_mode, bool is_store, unsigned bit_size, unsigned num_components));

void nir_strip(nir_shader *shader);

void nir_sweep(nir_shader *shader);

void nir_remap_dual_slot_attributes(nir_shader *shader,
                                    uint64_t *dual_slot_inputs);
uint64_t nir_get_single_slot_attribs_mask(uint64_t attribs, uint64_t dual_slot);

bool* nir_uniform_analysis(nir_shader *shader);

nir_intrinsic_op nir_intrinsic_from_system_value(gl_system_value val);
gl_system_value nir_system_value_from_intrinsic(nir_intrinsic_op intrin);

static inline bool
nir_variable_is_in_ubo(const nir_variable *var)
{
   return (var->data.mode == nir_var_mem_ubo &&
           var->interface_type != NULL);
}

static inline bool
nir_variable_is_in_ssbo(const nir_variable *var)
{
   return (var->data.mode == nir_var_mem_ssbo &&
           var->interface_type != NULL);
}

static inline bool
nir_variable_is_in_block(const nir_variable *var)
{
   return nir_variable_is_in_ubo(var) || nir_variable_is_in_ssbo(var);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIR_H */
