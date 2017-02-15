/*
 * Copyright 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 */
/* based on pieces from si_pipe.c and radeon_llvm_emit.c */
#include "ac_llvm_util.h"

#include <llvm-c/Core.h>

#include "c11/threads.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void ac_init_llvm_target()
{
#if HAVE_LLVM < 0x0307
	LLVMInitializeR600TargetInfo();
	LLVMInitializeR600Target();
	LLVMInitializeR600TargetMC();
	LLVMInitializeR600AsmPrinter();
#else
	LLVMInitializeAMDGPUTargetInfo();
	LLVMInitializeAMDGPUTarget();
	LLVMInitializeAMDGPUTargetMC();
	LLVMInitializeAMDGPUAsmPrinter();
#endif
}

static once_flag ac_init_llvm_target_once_flag = ONCE_FLAG_INIT;

static LLVMTargetRef ac_get_llvm_target(const char *triple)
{
	LLVMTargetRef target = NULL;
	char *err_message = NULL;

	call_once(&ac_init_llvm_target_once_flag, ac_init_llvm_target);

	if (LLVMGetTargetFromTriple(triple, &target, &err_message)) {
		fprintf(stderr, "Cannot find target for triple %s ", triple);
		if (err_message) {
			fprintf(stderr, "%s\n", err_message);
		}
		LLVMDisposeMessage(err_message);
		return NULL;
	}
	return target;
}

static const char *ac_get_llvm_processor_name(enum radeon_family family)
{
	switch (family) {
	case CHIP_TAHITI:
		return "tahiti";
	case CHIP_PITCAIRN:
		return "pitcairn";
	case CHIP_VERDE:
		return "verde";
	case CHIP_OLAND:
		return "oland";
	case CHIP_HAINAN:
		return "hainan";
	case CHIP_BONAIRE:
		return "bonaire";
	case CHIP_KABINI:
		return "kabini";
	case CHIP_KAVERI:
		return "kaveri";
	case CHIP_HAWAII:
		return "hawaii";
	case CHIP_MULLINS:
		return "mullins";
	case CHIP_TONGA:
		return "tonga";
	case CHIP_ICELAND:
		return "iceland";
	case CHIP_CARRIZO:
		return "carrizo";
#if HAVE_LLVM <= 0x0307
	case CHIP_FIJI:
		return "tonga";
	case CHIP_STONEY:
		return "carrizo";
#else
	case CHIP_FIJI:
		return "fiji";
	case CHIP_STONEY:
		return "stoney";
#endif
#if HAVE_LLVM <= 0x0308
	case CHIP_POLARIS10:
		return "tonga";
	case CHIP_POLARIS11:
		return "tonga";
#else
	case CHIP_POLARIS10:
		return "polaris10";
	case CHIP_POLARIS11:
		return "polaris11";
#endif
	default:
		return "";
	}
}

LLVMTargetMachineRef ac_create_target_machine(enum radeon_family family, bool supports_spill)
{
	assert(family >= CHIP_TAHITI);

	const char *triple = supports_spill ? "amdgcn-mesa-mesa3d" : "amdgcn--";
	LLVMTargetRef target = ac_get_llvm_target(triple);
	LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
	                             target,
	                             triple,
	                             ac_get_llvm_processor_name(family),
	                             "+DumpCode,+vgpr-spilling",
	                             LLVMCodeGenLevelDefault,
	                             LLVMRelocDefault,
	                             LLVMCodeModelDefault);

	return tm;
}


#if HAVE_LLVM < 0x0400
static LLVMAttribute ac_attr_to_llvm_attr(enum ac_func_attr attr)
{
   switch (attr) {
   case AC_FUNC_ATTR_ALWAYSINLINE: return LLVMAlwaysInlineAttribute;
   case AC_FUNC_ATTR_BYVAL: return LLVMByValAttribute;
   case AC_FUNC_ATTR_INREG: return LLVMInRegAttribute;
   case AC_FUNC_ATTR_NOALIAS: return LLVMNoAliasAttribute;
   case AC_FUNC_ATTR_NOUNWIND: return LLVMNoUnwindAttribute;
   case AC_FUNC_ATTR_READNONE: return LLVMReadNoneAttribute;
   case AC_FUNC_ATTR_READONLY: return LLVMReadOnlyAttribute;
   default:
	   fprintf(stderr, "Unhandled function attribute: %x\n", attr);
	   return 0;
   }
}

#else

static const char *attr_to_str(enum ac_func_attr attr)
{
   switch (attr) {
   case AC_FUNC_ATTR_ALWAYSINLINE: return "alwaysinline";
   case AC_FUNC_ATTR_BYVAL: return "byval";
   case AC_FUNC_ATTR_INREG: return "inreg";
   case AC_FUNC_ATTR_NOALIAS: return "noalias";
   case AC_FUNC_ATTR_NOUNWIND: return "nounwind";
   case AC_FUNC_ATTR_READNONE: return "readnone";
   case AC_FUNC_ATTR_READONLY: return "readonly";
   default:
	   fprintf(stderr, "Unhandled function attribute: %x\n", attr);
	   return 0;
   }
}

#endif

void
ac_add_function_attr(LLVMValueRef function,
                     int attr_idx,
                     enum ac_func_attr attr)
{

#if HAVE_LLVM < 0x0400
   LLVMAttribute llvm_attr = ac_attr_to_llvm_attr(attr);
   if (attr_idx == -1) {
      LLVMAddFunctionAttr(function, llvm_attr);
   } else {
      LLVMAddAttribute(LLVMGetParam(function, attr_idx - 1), llvm_attr);
   }
#else
   LLVMContextRef context = LLVMGetModuleContext(LLVMGetGlobalParent(function));
   const char *attr_name = attr_to_str(attr);
   unsigned kind_id = LLVMGetEnumAttributeKindForName(attr_name,
                                                      strlen(attr_name));
   LLVMAttributeRef llvm_attr = LLVMCreateEnumAttribute(context, kind_id, 0);
   LLVMAddAttributeAtIndex(function, attr_idx, llvm_attr);
#endif
}

void
ac_dump_module(LLVMModuleRef module)
{
	char *str = LLVMPrintModuleToString(module);
	fprintf(stderr, "%s", str);
	LLVMDisposeMessage(str);
}
