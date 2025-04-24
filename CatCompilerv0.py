# test.py
import os
import sys
import re
import struct

# --- Instruction Set Definition ---

# Opcodes (8-bit) - Extended
# We'll use different opcodes for different operand types for simplicity here.
# A more complex design might use flags within the instruction bytes.
OPCODES = {
    # Original Register-Register Ops (2 bytes total)
    # Format: Opcode (8) | SrcReg (4) | DestReg (4)
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03,
    'cmp': 0x04,
    'mov': 0x05, # Added mov reg, reg

    # Register-Immediate Ops (4 bytes total)
    # Format: Opcode (8) | DestReg (4) | 0000 | Immediate (16)
    'mov_imm': 0x10,
    'add_imm': 0x11,
    'sub_imm': 0x12,
    'cmp_imm': 0x13,

    # Jump Instructions (4 bytes total)
    # Format: Opcode (8) | 0000 0000 | Target Address (16)
    'jmp': 0x20,
    'je':  0x21, # Jump if Equal (ZF=1)
    'jne': 0x22, # Jump if Not Equal (ZF=0)
    # Add more jumps as needed (jg, jl, etc.)
}

# Registers (4-bit codes)
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}

# --- Assembler Class ---

class Assembler:
    def __init__(self):
        self.symbol_table = {} # label -> address (byte offset)
        self.intermediate_code = [] # [(line_num, opcode, [operand1, operand2]), ...]
        self.binary_code = bytearray()
        self.current_address = 0 # Byte offset during generation

    def _parse_operand(self, operand_str):
        """Identifies operand type (register, immediate, label)."""
        operand_str = operand_str.strip()
        if not operand_str:
            return None, None

        # Check for register
        if operand_str.lower() in REGISTERS:
            return 'register', operand_str.lower()

        # Check for immediate (decimal or hex)
        try:
            if operand_str.lower().startswith('0x'):
                value = int(operand_str, 16)
            else:
                value = int(operand_str)
            # For simplicity, treat all immediates as 16-bit for now
            if 0 <= value <= 0xFFFF:
                 return 'immediate', value
            elif -0x8000 <= value < 0: # Handle negative numbers if needed (2's complement)
                 return 'immediate', value & 0xFFFF
            else:
                 return 'error', f"Immediate value out of 16-bit range: {operand_str}"
        except ValueError:
            pass # Not an integer

        # Assume it's a label (validation happens in pass 2)
        # Basic label format check (alphanumeric + _)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand_str):
             return 'label', operand_str
        else:
             return 'error', f"Invalid operand format: {operand_str}"

    def pass1(self, source_lines):
        """
        First pass:
        - Parses lines into instructions and operands.
        - Builds the symbol table (labels and their *instruction indices* initially).
        - Creates an intermediate representation.
        - Handles basic syntax checks.
        """
        print("--- Pass 1: Parsing and Symbol Table ---")
        instruction_index = 0
        for line_num, line in enumerate(source_lines, 1):
            line = line.strip()
            # Remove comments (everything after ';')
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos].strip()

            if not line:
                continue # Skip empty lines

            # Check for labels
            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$', line)
            instruction_part = line
            if label_match:
                label, instruction_part = label_match.groups()
                label = label.lower()
                if label in self.symbol_table:
                    print(f"Error line {line_num}: Label '{label}' redefined.")
                    # Decide on error handling: continue or exit? Let's continue.
                else:
                    # Store instruction index for now, byte address calculated later
                    self.symbol_table[label] = instruction_index
                    print(f"  Found label '{label}' at index {instruction_index}")
                instruction_part = instruction_part.strip()

            if not instruction_part:
                continue # Line only contained a label

            # Parse instruction and operands
            parts = re.split(r'[,\s]+', instruction_part, maxsplit=2)
            opcode_str = parts[0].lower()
            operands = [p.strip() for p in parts[1:] if p.strip()]

            # Basic validation
            if not opcode_str:
                 print(f"Warning line {line_num}: Empty instruction part '{instruction_part}'")
                 continue

            # Store in intermediate representation
            self.intermediate_code.append({
                'line': line_num,
                'opcode': opcode_str,
                'operands': operands,
                'index': instruction_index # Store original index
            })
            instruction_index += 1

        print(f"Symbol Table (indices): {self.symbol_table}")
        print("--- Pass 1 Complete ---")
        return not any('error' in str(item) for item in self.intermediate_code) # Basic check

    def optimize(self):
        """
        Simple peephole optimization pass.
        - Removes redundant 'mov reg, reg' instructions.
        """
        print("--- Optimization Pass ---")
        optimized_code = []
        removed_count = 0
        i = 0
        while i < len(self.intermediate_code):
            instr = self.intermediate_code[i]
            is_redundant_mov = False

            # Pattern: mov reg, reg (where reg is the same)
            if instr['opcode'] == 'mov' and len(instr['operands']) == 2:
                op1_type, op1_val = self._parse_operand(instr['operands'][0])
                op2_type, op2_val = self._parse_operand(instr['operands'][1])
                if op1_type == 'register' and op2_type == 'register' and op1_val == op2_val:
                    print(f"  Optimizing line {instr['line']}: Removing redundant mov {op1_val}, {op1_val}")
                    is_redundant_mov = True
                    removed_count += 1

            # Add more optimization patterns here if desired

            if not is_redundant_mov:
                optimized_code.append(instr)
            i += 1

        self.intermediate_code = optimized_code
        # Update instruction indices and label addresses after potential removals
        new_symbol_table = {}
        current_index = 0
        # Create a map from old index to new index
        index_map = {instr['index']: i for i, instr in enumerate(self.intermediate_code)}

        # Remap labels
        for label, old_index in self.symbol_table.items():
            if old_index in index_map:
                 new_symbol_table[label] = index_map[old_index]
            else:
                 # This label pointed to an instruction that got optimized away.
                 # This is tricky. A simple approach is to point it to the *next*
                 # available instruction index. More complex optimizers need
                 # careful label management.
                 # Find the next instruction's new index.
                 next_new_index = -1
                 for i in range(old_index + 1, len(self.intermediate_code) + removed_count): # Check original indices
                     if i in index_map:
                         next_new_index = index_map[i]
                         break
                 if next_new_index != -1:
                     new_symbol_table[label] = next_new_index
                     print(f"  Warning: Label '{label}' pointed to optimized instruction, remapped to index {next_new_index}")
                 else:
                     # Label pointed to the last instruction(s) which were removed
                     new_symbol_table[label] = len(self.intermediate_code) # Point past the end? Or error?
                     print(f"  Warning: Label '{label}' pointed to optimized instruction at end, remapped past end.")


        self.symbol_table = new_symbol_table
        # Update the 'index' field in the intermediate code itself
        for i, instr in enumerate(self.intermediate_code):
            instr['index'] = i

        print(f"Optimization removed {removed_count} instructions.")
        print(f"Updated Symbol Table (indices): {self.symbol_table}")
        print("--- Optimization Complete ---")


    def _get_instruction_size(self, opcode, operands):
        """ Calculates the size (in bytes) of an encoded instruction. """
        op_lower = opcode.lower()

        # Register-Register (2 bytes)
        if op_lower in ('add', 'sub', 'mul', 'div', 'cmp', 'mov'):
             # Basic check, detailed validation in pass2
             if len(operands) == 2:
                 op1_type, _ = self._parse_operand(operands[0])
                 op2_type, _ = self._parse_operand(operands[1])
                 if op1_type == 'register' and op2_type == 'register':
                     return 2
             # Handle mov reg, imm case separately below

        # Register-Immediate (4 bytes)
        if op_lower in ('mov', 'add', 'sub', 'cmp'):
             if len(operands) == 2:
                 op1_type, _ = self._parse_operand(operands[0])
                 op2_type, _ = self._parse_operand(operands[1])
                 if op1_type == 'register' and op2_type == 'immediate':
                     return 4

        # Jumps (4 bytes)
        if op_lower in ('jmp', 'je', 'jne'):
             if len(operands) == 1:
                 op1_type, _ = self._parse_operand(operands[0])
                 if op1_type == 'label':
                     return 4

        # If none matched, it's an error or unsupported
        return -1 # Indicate error or unknown size


    def pass2(self):
        """
        Second pass:
        - Calculates final byte addresses for labels based on instruction sizes.
        - Resolves label operands to addresses.
        - Encodes instructions into binary format.
        - Performs final validation.
        """
        print("--- Pass 2: Encoding and Address Resolution ---")
        self.binary_code = bytearray()
        self.current_address = 0
        errors = False

        # First, calculate final byte addresses for all labels
        temp_address = 0
        instruction_addresses = {} # instruction index -> byte address
        for instr in self.intermediate_code:
            instruction_addresses[instr['index']] = temp_address
            size = self._get_instruction_size(instr['opcode'], instr['operands'])
            if size == -1:
                 # Error detected here means pass1 logic or _get_instruction_size is incomplete
                 print(f"Error line {instr['line']}: Cannot determine size for '{instr['opcode']} {' '.join(instr['operands'])}'. Unsupported format?")
                 errors = True
                 size = 0 # Avoid cascading errors, but output will be wrong
            temp_address += size

        # Update symbol table with byte addresses
        final_symbol_table = {}
        for label, index in self.symbol_table.items():
            if index in instruction_addresses:
                final_symbol_table[label] = instruction_addresses[index]
            else:
                # Should not happen if optimization pass updated correctly
                print(f"Error: Label '{label}' index {index} not found in instruction addresses after optimization.")
                final_symbol_table[label] = 0xFFFF # Indicate error
                errors = True
        self.symbol_table = final_symbol_table
        print(f"Final Symbol Table (byte addresses): {self.symbol_table}")


        # Now, encode instructions
        for instr in self.intermediate_code:
            opcode_str = instr['opcode']
            operands = instr['operands']
            line_num = instr['line']
            encoded = None

            try:
                # --- Register-Register Instructions ---
                if opcode_str in ('add', 'sub', 'mul', 'div', 'cmp', 'mov') and len(operands) == 2:
                    op1_type, op1_val = self._parse_operand(operands[0])
                    op2_type, op2_val = self._parse_operand(operands[1])

                    if op1_type == 'register' and op2_type == 'register':
                        if opcode_str not in OPCODES: # Should exist
                             raise ValueError(f"Internal error: Opcode '{opcode_str}' not found")
                        op_byte = OPCODES[opcode_str]
                        src_reg_code = REGISTERS[op1_val]
                        dest_reg_code = REGISTERS[op2_val]
                        # Format: Opcode (8) | SrcReg (4) | DestReg (4)
                        instruction_word = (op_byte << 8) | (src_reg_code << 4) | dest_reg_code
                        encoded = struct.pack('>H', instruction_word) # Big-endian short (2 bytes)

                    # --- Register-Immediate Instructions ---
                    elif op1_type == 'register' and op2_type == 'immediate':
                        imm_opcode_str = f"{opcode_str}_imm"
                        if imm_opcode_str not in OPCODES:
                            raise ValueError(f"Unsupported immediate operation: {opcode_str}")
                        op_byte = OPCODES[imm_opcode_str]
                        dest_reg_code = REGISTERS[op1_val]
                        imm_val = op2_val
                        # Format: Opcode (8) | DestReg (4) | 0000 | Immediate (16)
                        instruction_part1 = (op_byte << 8) | (dest_reg_code << 4)
                        # Pack opcode+reg byte and immediate separately
                        encoded = struct.pack('>H', instruction_part1) + struct.pack('>H', imm_val) # Big-endian short + short (4 bytes)

                    else:
                         raise ValueError(f"Invalid operand types for {opcode_str}: {op1_type}, {op2_type}")

                # --- Jump Instructions ---
                elif opcode_str in ('jmp', 'je', 'jne') and len(operands) == 1:
                    op1_type, op1_val = self._parse_operand(operands[0])
                    if op1_type == 'label':
                        if op1_val not in self.symbol_table:
                            raise ValueError(f"Undefined label: '{op1_val}'")
                        target_addr = self.symbol_table[op1_val]
                        if not (0 <= target_addr <= 0xFFFF):
                             raise ValueError(f"Target address {target_addr} for label '{op1_val}' out of 16-bit range")

                        op_byte = OPCODES[opcode_str]
                        # Format: Opcode (8) | 0000 0000 | Target Address (16)
                        instruction_part1 = op_byte << 8
                        encoded = struct.pack('>H', instruction_part1) + struct.pack('>H', target_addr) # Big-endian short + short (4 bytes)
                    else:
                        raise ValueError(f"Invalid operand type for {opcode_str}: {op1_type}")

                else:
                    raise ValueError(f"Unknown or invalid instruction format: {opcode_str} {' '.join(operands)}")

            except ValueError as e:
                print(f"Error line {line_num}: {e}")
                errors = True

            if encoded:
                #print(f"  Encoding line {line_num}: {instr} -> {encoded.hex()}")
                self.binary_code.extend(encoded)
                self.current_address += len(encoded)
            else:
                 # Add placeholder or handle error to keep addresses consistent?
                 # For now, errors mean the output is likely unusable.
                 pass


        print(f"--- Pass 2 Complete ({len(self.binary_code)} bytes generated) ---")
        return not errors


    def write_output(self, filename):
        """Writes the generated binary code to a file."""
        try:
            with open(filename, 'wb') as f:
                f.write(self.binary_code)
            print(f"Successfully wrote {len(self.binary_code)} bytes to '{filename}'")
        except IOError as e:
            print(f"Error writing output file '{filename}': {e}")

    def compile(self, source_filename, output_filename):
        """Runs the full assembly process."""
        try:
            with open(source_filename, 'r') as f:
                source_lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Source file not found: '{source_filename}'")
            return
        except IOError as e:
            print(f"Error reading source file '{source_filename}': {e}")
            return

        if self.pass1(source_lines):
            self.optimize()
            if self.pass2():
                self.write_output(output_filename)
            else:
                print("Assembly failed due to errors in Pass 2.")
        else:
            print("Assembly failed due to errors in Pass 1.")


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py <source.asm> <output.o>")
        sys.exit(1)

    source_file = sys.argv[1]
    output_file = sys.argv[2]

    assembler = Assembler()
    assembler.compile(source_file, output_file)

    # Example of how to create a sample source file:
    if not os.path.exists("sample.asm"):
        print("\nCreating sample source file 'sample.asm'...")
        sample_asm = """
; Example Assembly Code for test.py Assembler

start:
    mov rax, 10      ; Load 10 into rax (mov_imm)
    mov rbx, 0x20    ; Load 32 into rbx (hex immediate)
    mov rcx, rax     ; Copy rax to rcx (redundant, will be optimized)
    mov rcx, rax     ; Copy rax to rcx (mov reg, reg)

loop_top:
    cmp rax, rbx     ; Compare rax and rbx
    je end_loop      ; Jump to end_loop if rax == rbx

    add rax, 1       ; Increment rax (add_imm)
    sub rbx, 0x02    ; Decrement rbx by 2 (sub_imm)
    jmp loop_top     ; Jump back to loop_top

end_loop:
    mov rdx, rax     ; Store final rax value in rdx
    mul rdx, 2       ; Multiply rdx by 2 (mul_imm - assuming we add it)
                     ; Currently only mul reg, reg exists. Let's use that.
    mov rcx, 2       ; Need a register for the multiplier
    mul rdx, rcx     ; rdx = rdx * rcx (rdx = rdx * 2)

    ; End of program (no explicit halt instruction defined)
"""
        with open("sample.asm", "w") as f:
            f.write(sample_asm)
        print("Created 'sample.asm'. You can now run:")
        print(f"python test.py sample.asm sample.o")

