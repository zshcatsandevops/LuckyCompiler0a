# test.py
import os
import sys
import re
import struct

# --- Instruction Set Definition ---

# Opcodes (8-bit)
# Format (RR):  Opcode (8) | SrcReg (4) | DestReg (4)
# Format (RI):  Opcode (8) | DestReg (4) | 0000 | Immediate (16)
# Format (JMP): Opcode (8) | 0000 0000   | Target Address (16)
OPCODES = {
    # Register-Register
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03,
    'cmp': 0x04,
    'mov': 0x05,

    # Register-Immediate
    'mov_imm': 0x10,
    'add_imm': 0x11,
    'sub_imm': 0x12,
    'cmp_imm': 0x13,
    'mul_imm': 0x14,  # added
    'div_imm': 0x15,  # added

    # Jumps
    'jmp': 0x20,
    'je':  0x21,
    'jne': 0x22,
}

# Registers (4-bit codes)
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}

class Assembler:
    def __init__(self):
        self.symbol_table = {}            # label -> address (byte offset)
        self.intermediate_code = []       # list of dicts with parsed instructions
        self.binary_code = bytearray()
        self.current_address = 0

    def _parse_operand(self, operand_str):
        """Detect operand type (register, immediate, label)."""
        operand_str = operand_str.strip()
        if not operand_str:
            return None, None

        low = operand_str.lower()

        # Register
        if low in REGISTERS:
            return 'register', low

        # Immediate (dec/hex, with negative support)
        try:
            if low.startswith('0x'):
                value = int(low, 16)
            else:
                value = int(low, 10)
            if 0 <= value <= 0xFFFF:
                return 'immediate', value
            elif -0x8000 <= value < 0:
                return 'immediate', value & 0xFFFF
            else:
                return 'error', f"Immediate value out of 16-bit range: {operand_str}"
        except ValueError:
            pass

        # Label (basic validation)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand_str):
            return 'label', operand_str
        return 'error', f"Invalid operand format: {operand_str}"

    def pass1(self, source_lines):
        """Parse, collect labels (by instruction index), and build IR."""
        print("--- Pass 1: Parsing and Symbol Table ---")
        instruction_index = 0

        for line_num, raw in enumerate(source_lines, 1):
            line = raw.strip()
            # strip comments
            cpos = line.find(';')
            if cpos != -1:
                line = line[:cpos].strip()
            if not line:
                continue

            # Label
            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$', line)
            instruction_part = line
            if label_match:
                label, instruction_part = label_match.groups()
                label = label.lower()
                if label in self.symbol_table:
                    print(f"Error line {line_num}: Label '{label}' redefined.")
                else:
                    self.symbol_table[label] = instruction_index
                    print(f"  Found label '{label}' at index {instruction_index}")
                instruction_part = instruction_part.strip()

            if not instruction_part:
                continue

            # Instruction + operands
            parts = re.split(r'[,\s]+', instruction_part, maxsplit=2)
            opcode_str = parts[0].lower()
            operands = [p.strip() for p in parts[1:] if p.strip()]

            if not opcode_str:
                print(f"Warning line {line_num}: Empty instruction part '{instruction_part}'")
                continue

            self.intermediate_code.append({
                'line': line_num,
                'opcode': opcode_str,
                'operands': operands,
                'index': instruction_index
            })
            instruction_index += 1

        print(f"Symbol Table (indices): {self.symbol_table}")
        print("--- Pass 1 Complete ---")
        return True

    def optimize(self):
        """Simple peephole optimizer."""
        print("--- Optimization Pass ---")
        optimized = []
        removed = 0
        i = 0
        while i < len(self.intermediate_code):
            instr = self.intermediate_code[i]
            op = instr['opcode']
            ops = instr['operands']

            drop = False

            # Remove 'mov r, r'
            if op == 'mov' and len(ops) == 2:
                k1, v1 = self._parse_operand(ops[0])
                k2, v2 = self._parse_operand(ops[1])
                if k1 == k2 == 'register' and v1 == v2:
                    print(f"  Optimizing line {instr['line']}: drop redundant 'mov {v1}, {v2}'")
                    drop = True
                    removed += 1

            # Remove consecutive duplicate 'mov dest, src'
            if not drop and op == 'mov' and i + 1 < len(self.intermediate_code):
                nxt = self.intermediate_code[i + 1]
                if nxt['opcode'] == 'mov' and nxt['operands'] == ops:
                    print(f"  Optimizing lines {instr['line']} & {nxt['line']}: drop duplicate second mov")
                    # keep the first, drop the second
                    optimized.append(instr)
                    i += 2
                    removed += 1
                    continue

            if not drop:
                optimized.append(instr)
            i += 1

        self.intermediate_code = optimized

        # Remap label indices after removals
        index_map = {old['index']: i for i, old in enumerate(self.intermediate_code)}
        new_symbols = {}
        for label, old_idx in self.symbol_table.items():
            if old_idx in index_map:
                new_symbols[label] = index_map[old_idx]
            else:
                # find next surviving instruction
                next_new = -1
                for j in range(old_idx + 1, len(self.intermediate_code) + removed):
                    if j in index_map:
                        next_new = index_map[j]
                        break
                if next_new != -1:
                    new_symbols[label] = next_new
                    print(f"  Warning: Label '{label}' remapped to index {next_new} (target optimized away)")
                else:
                    new_symbols[label] = len(self.intermediate_code)
                    print(f"  Warning: Label '{label}' remapped past end (target optimized away at end)")

        self.symbol_table = new_symbols
        for i, instr in enumerate(self.intermediate_code):
            instr['index'] = i

        print(f"Optimization removed {removed} instructions.")
        print(f"Updated Symbol Table (indices): {self.symbol_table}")
        print("--- Optimization Complete ---")

    def _get_instruction_size(self, opcode, operands):
        """Return encoded size in bytes, or -1 if unsupported."""
        op = opcode.lower()

        # RR: reg, reg (dest, src in source)
        if op in ('add', 'sub', 'mul', 'div', 'cmp', 'mov'):
            if len(operands) == 2:
                t1, _ = self._parse_operand(operands[0])
                t2, _ = self._parse_operand(operands[1])
                if t1 == 'register' and t2 == 'register':
                    return 2

        # RI: reg, imm (dest, imm)
        if op in ('mov', 'add', 'sub', 'cmp', 'mul', 'div'):
            if len(operands) == 2:
                t1, _ = self._parse_operand(operands[0])
                t2, _ = self._parse_operand(operands[1])
                if t1 == 'register' and t2 == 'immediate':
                    return 4

        # Jumps: label
        if op in ('jmp', 'je', 'jne'):
            if len(operands) == 1:
                t1, _ = self._parse_operand(operands[0])
                if t1 == 'label':
                    return 4

        return -1

    def pass2(self):
        """Compute byte addresses, resolve labels, and encode."""
        print("--- Pass 2: Encoding and Address Resolution ---")
        self.binary_code = bytearray()
        self.current_address = 0
        errors = False

        # Instruction index -> byte address
        addr = 0
        instr_addr = {}
        for instr in self.intermediate_code:
            instr_addr[instr['index']] = addr
            size = self._get_instruction_size(instr['opcode'], instr['operands'])
            if size == -1:
                print(f"Error line {instr['line']}: Unsupported form '{instr['opcode']} {' '.join(instr['operands'])}'")
                errors = True
                size = 0
            addr += size

        # Label -> byte address
        final_symbols = {}
        for label, idx in self.symbol_table.items():
            if idx in instr_addr:
                final_symbols[label] = instr_addr[idx]
            else:
                print(f"Error: Label '{label}' index {idx} not found post-optimization.")
                final_symbols[label] = 0xFFFF
                errors = True
        self.symbol_table = final_symbols
        print(f"Final Symbol Table (byte addresses): {self.symbol_table}")

        # Encode
        for instr in self.intermediate_code:
            op = instr['opcode']
            ops = instr['operands']
            line = instr['line']
            encoded = None

            try:
                # RR: dest, src  ->  fields: src | dest
                if op in ('add', 'sub', 'mul', 'div', 'cmp', 'mov') and len(ops) == 2:
                    t1, v1 = self._parse_operand(ops[0])
                    t2, v2 = self._parse_operand(ops[1])

                    if t1 == 'register' and t2 == 'register':
                        if op not in OPCODES:
                            raise ValueError(f"Internal error: opcode '{op}' not found")
                        op_byte = OPCODES[op]
                        src_reg = REGISTERS[v2]   # src is operand 2
                        dest_reg = REGISTERS[v1]  # dest is operand 1
                        word = (op_byte << 8) | (src_reg << 4) | dest_reg
                        encoded = struct.pack('>H', word)

                    # RI: dest, imm
                    elif t1 == 'register' and t2 == 'immediate':
                        imm_op = f"{op}_imm"
                        if imm_op not in OPCODES:
                            raise ValueError(f"Unsupported immediate operation: {op}")
                        op_byte = OPCODES[imm_op]
                        dest_reg = REGISTERS[v1]
                        part1 = (op_byte << 8) | (dest_reg << 4)
                        encoded = struct.pack('>H', part1) + struct.pack('>H', v2)

                    else:
                        raise ValueError(f"Invalid operand types for {op}: {t1}, {t2}")

                # JMP label
                elif op in ('jmp', 'je', 'jne') and len(ops) == 1:
                    t1, v1 = self._parse_operand(ops[0])
                    if t1 != 'label':
                        raise ValueError(f"Invalid operand for {op}: expected label, got {t1}")
                    if v1 not in self.symbol_table:
                        raise ValueError(f"Undefined label: '{v1}'")
                    target = self.symbol_table[v1]
                    if not (0 <= target <= 0xFFFF):
                        raise ValueError(f"Target address {target} out of range for label '{v1}'")
                    op_byte = OPCODES[op]
                    part1 = op_byte << 8
                    encoded = struct.pack('>H', part1) + struct.pack('>H', target)

                else:
                    raise ValueError(f"Unknown/invalid instruction: {op} {' '.join(ops)}")

            except ValueError as e:
                print(f"Error line {line}: {e}")
                errors = True

            if encoded:
                self.binary_code.extend(encoded)
                self.current_address += len(encoded)

        print(f"--- Pass 2 Complete ({len(self.binary_code)} bytes generated) ---")
        return not errors

    def write_output(self, filename):
        try:
            with open(filename, 'wb') as f:
                f.write(self.binary_code)
            print(f"Successfully wrote {len(self.binary_code)} bytes to '{filename}'")
        except IOError as e:
            print(f"Error writing output file '{filename}': {e}")

    def compile(self, source_filename, output_filename):
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

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py <source.asm> <output.bin>")
        sys.exit(1)

    source_file = sys.argv[1]
    output_file = sys.argv[2]

    assembler = Assembler()
    assembler.compile(source_file, output_file)

    # Example sample file (created once)
    if not os.path.exists("sample.asm"):
        print("\nCreating sample source file 'sample.asm'...")
        sample_asm = """
; Example Assembly Code for test.py Assembler
; All instructions use Intel-style operand order: dest, src

start:
    mov rax, 10      ; rax = 10
    mov rbx, 0x20    ; rbx = 32
    mov rcx, rax     ; rcx = rax

loop_top:
    cmp rax, rbx     ; compare rax vs rbx
    je end_loop      ; if equal, jump to end_loop

    add rax, 1       ; rax += 1
    sub rbx, 0x02    ; rbx -= 2
    jmp loop_top     ; loop

end_loop:
    mov rdx, rax     ; rdx = rax
    mul rdx, 2       ; rdx *= 2    (mul_imm now supported)

; No explicit halt instruction
"""
        with open("sample.asm", "w") as f:
            f.write(sample_asm)
        print("Created 'sample.asm'. Run:")
        print("  python test.py sample.asm sample.bin")
