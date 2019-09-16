
#include <iomanip>
#include "aco_ir.h"
#include "llvm-c/Disassembler.h"
#include "ac_llvm_util.h"

#include <llvm/ADT/StringRef.h>

namespace aco {

void print_asm(Program *program, std::vector<uint32_t>& binary,
               unsigned exec_size, enum radeon_family family, std::ostream& out)
{
   std::vector<bool> referenced_blocks(program->blocks.size());
   referenced_blocks[0] = true;
   for (Block& block : program->blocks) {
      for (unsigned succ : block.linear_succs)
         referenced_blocks[succ] = true;
   }

   std::vector<std::tuple<uint64_t, llvm::StringRef, uint8_t>> symbols;
   std::vector<std::array<char,16>> block_names;
   block_names.reserve(program->blocks.size());
   for (Block& block : program->blocks) {
      if (!referenced_blocks[block.index])
         continue;
      std::array<char, 16> name;
      sprintf(name.data(), "BB%u", block.index);
      block_names.push_back(name);
      symbols.emplace_back(block.offset * 4, llvm::StringRef(block_names[block_names.size() - 1].data()), 0);
   }

   LLVMDisasmContextRef disasm = LLVMCreateDisasmCPU("amdgcn-mesa-mesa3d",
                                                     ac_get_llvm_processor_name(family),
                                                     &symbols, 0, NULL, NULL);

   char outline[1024];
   size_t pos = 0;
   bool invalid = false;
   unsigned next_block = 0;
   while (pos < exec_size) {
      while (next_block < program->blocks.size() && pos == program->blocks[next_block].offset) {
         if (referenced_blocks[next_block])
            out << "BB" << std::dec << next_block << ":" << std::endl;
         next_block++;
      }

      size_t l = LLVMDisasmInstruction(disasm, (uint8_t *) &binary[pos],
                                       (exec_size - pos) * sizeof(uint32_t), pos * 4,
                                       outline, sizeof(outline));

      size_t new_pos;
      const int align_width = 60;
      if (program->chip_class == GFX9 && !l && ((binary[pos] & 0xffff8000) == 0xd1348000)) { /* not actually an invalid instruction */
         out << std::left << std::setw(align_width) << std::setfill(' ') << "\tv_add_u32_e64 + clamp";
         new_pos = pos + 2;
      } else if (!l) {
         out << std::left << std::setw(align_width) << std::setfill(' ') << "(invalid instruction)";
         new_pos = pos + 1;
         invalid = true;
      } else {
         out << std::left << std::setw(align_width) << std::setfill(' ') << outline;
         assert(l % 4 == 0);
         new_pos = pos + l / 4;
      }
      out << std::right;

      out << " ;";
      for (; pos < new_pos; pos++)
         out << " " << std::setfill('0') << std::setw(8) << std::hex << binary[pos];
      out << std::endl;
   }
   out << std::setfill(' ') << std::setw(0) << std::dec;
   assert(next_block == program->blocks.size());

   LLVMDisasmDispose(disasm);

   if (program->constant_data.size()) {
      out << std::endl << "/* constant data */" << std::endl;
      for (unsigned i = 0; i < program->constant_data.size(); i += 32) {
         out << '[' << std::setw(6) << std::setfill('0') << std::dec << i << ']';
         unsigned line_size = std::min<size_t>(program->constant_data.size() - i, 32);
         for (unsigned j = 0; j < line_size; j += 4) {
            unsigned size = std::min<size_t>(program->constant_data.size() - (i + j), 4);
            uint32_t v = 0;
            memcpy(&v, &program->constant_data[i + j], size);
            out << " " << std::setw(8) << std::setfill('0') << std::hex << v;
         }
         out << std::endl;
      }
   }

   out << std::setfill(' ') << std::setw(0) << std::dec;

   if (invalid) {
      /* Invalid instructions usually lead to GPU hangs, which can make
       * getting the actual invalid instruction hard. Abort here so that we
       * can find the problem.
       */
      abort();
   }
}

}
