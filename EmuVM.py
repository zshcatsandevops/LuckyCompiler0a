# test.py
import os
import sys
import re
import struct
import argparse
import time # For potential step delay

# --- Constants ---
MEM_SIZE = 0x10000 # 64KB Memory for the VM
ROM_MAGIC = b'T3VM' # Magic bytes to identify our ROM
ROM_VERSION = 1

# --- Instruction Set Definition ---

# Opcodes (8-bit) - Extended
OPCODES = {
    # Register-Register Ops (2 bytes total)
    # Format: Opcode (8) | SrcReg (4) | DestReg (4)
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03, # Note: div often affects two registers (quotient/remainder) - simplified here
    'cmp': 0x04,
    'mov': 0x05,

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

    # System Instructions
    # Format: Opcode (8) | Syscall ID (8) | 0000 0000 0000
    'syscall': 0xF0, # Syscall instruction (2 bytes for now)
    # Format: Opcode (8) | 0000 0000
    'hlt': 0xFF, # Halt execution (1 byte) - Let's make it 1 byte for simplicity
}

# Syscall IDs (used with 'syscall' opcode)
SYSCALLS = {
    'print_rax': 0x01, # Print the decimal value in RAX
    # Add more syscalls here (e.g., print_char, read_int)
}

# Registers (4-bit codes)
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}
# Reverse lookup for emulator
REG_CODE_TO_NAME = {v: k for k, v in REGISTERS.items()}

# --- Assembler Class ---

class Assembler:
    def __init__(self):
        self.symbol_table = {} # label -> address (byte offset)
        self.intermediate_code = [] # [(line_num, opcode, [operand1, operand2]), ...]
        self.binary_code = bytearray()
        self.current_address = 0 # Byte offset during generation
        self.entry_point_label = None # Optional entry point label

    def _parse_operand(self, operand_str):
        """Identifies operand type (register, immediate, label, syscall_id)."""
        operand_str = operand_str.strip()
        if not operand_str:
            return None, None

        # Check for register
        if operand_str.lower() in REGISTERS:
            return 'register', operand_str.lower()

        # Check for syscall ID (used only by 'syscall')
        if operand_str.lower() in SYSCALLS:
             return 'syscall_id', operand_str.lower()

        # Check for immediate (decimal or hex)
        try:
            value = 0
            if operand_str.lower().startswith('0x'):
                value = int(operand_str, 16)
            else:
                value = int(operand_str)
            # Allow signed 16-bit for immediates if needed, but pack as unsigned
            if -0x8000 <= value <= 0xFFFF:
                 # Pack as unsigned 16-bit
                 return 'immediate', value & 0xFFFF
            else:
                 return 'error', f"Immediate value out of 16-bit range: {operand_str}"
        except ValueError:
            pass # Not an integer

        # Assume it's a label (validation happens in pass 2)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand_str):
             return 'label', operand_str
        else:
             return 'error', f"Invalid operand format: {operand_str}"

    def pass1(self, source_lines):
        """First pass: Parse, build symbol table (indices), create IR."""
        print("--- Pass 1: Parsing and Symbol Table ---")
        instruction_index = 0
        for line_num, line in enumerate(source_lines, 1):
            line = line.strip()
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos].strip()
            if not line: continue

            # Check for directives like .entry
            if line.lower().startswith(".entry"):
                parts = line.split()
                if len(parts) == 2:
                    self.entry_point_label = parts[1].lower()
                    print(f"  Found entry point directive: '{self.entry_point_label}'")
                else:
                    print(f"Error line {line_num}: Invalid .entry directive format.")
                continue # Directive processed

            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$', line)
            instruction_part = line
            label = None
            if label_match:
                label, instruction_part = label_match.groups()
                label = label.lower()
                if label in self.symbol_table:
                    print(f"Error line {line_num}: Label '{label}' redefined.")
                else:
                    self.symbol_table[label] = instruction_index
                    print(f"  Found label '{label}' at index {instruction_index}")
                instruction_part = instruction_part.strip()

            if not instruction_part: continue

            parts = re.split(r'[,\s]+', instruction_part, maxsplit=2)
            opcode_str = parts[0].lower()
            operands = [p.strip() for p in parts[1:] if p.strip()]

            # Special handling for hlt (no operands)
            if opcode_str == 'hlt' and operands:
                 print(f"Warning line {line_num}: 'hlt' instruction takes no operands.")
                 operands = []

            self.intermediate_code.append({
                'line': line_num,
                'opcode': opcode_str,
                'operands': operands,
                'index': instruction_index
            })
            instruction_index += 1

        # Default entry point if not specified
        if not self.entry_point_label:
             if 'start' in self.symbol_table:
                 self.entry_point_label = 'start'
             elif self.intermediate_code: # First instruction if no 'start' label
                 # Find the label associated with the first instruction, if any
                 first_instr_idx = self.intermediate_code[0]['index']
                 entry_label_found = False
                 for lbl, idx in self.symbol_table.items():
                     if idx == first_instr_idx:
                         self.entry_point_label = lbl
                         entry_label_found = True
                         break
                 if not entry_label_found:
                     # No label for the first instruction, entry point will be address 0
                     self.entry_point_label = None # Indicate address 0 needed
                     print("  No '.entry' or 'start' label found, defaulting entry point to address 0.")
             else:
                 self.entry_point_label = None # Empty file
                 print("Warning: No code found.")


        print(f"Symbol Table (indices): {self.symbol_table}")
        print(f"Entry Point Label: {self.entry_point_label if self.entry_point_label else 'Address 0'}")
        print("--- Pass 1 Complete ---")
        return True # Basic success

    def optimize(self):
        """Simple peephole optimization pass."""
        print("--- Optimization Pass ---")
        # (Optimization logic from previous example - can be expanded)
        optimized_code = []
        removed_count = 0
        i = 0
        original_indices = {instr['index']: instr for instr in self.intermediate_code}

        while i < len(self.intermediate_code):
            instr = self.intermediate_code[i]
            is_redundant_mov = False

            if instr['opcode'] == 'mov' and len(instr['operands']) == 2:
                op1_type, op1_val = self._parse_operand(instr['operands'][0])
                op2_type, op2_val = self._parse_operand(instr['operands'][1])
                if op1_type == 'register' and op2_type == 'register' and op1_val == op2_val:
                    print(f"  Optimizing line {instr['line']}: Removing redundant mov {op1_val}, {op1_val}")
                    is_redundant_mov = True
                    removed_count += 1
                    del original_indices[instr['index']] # Remove from map

            if not is_redundant_mov:
                optimized_code.append(instr)
            i += 1

        self.intermediate_code = optimized_code
        # Update instruction indices and label addresses
        new_symbol_table = {}
        current_index = 0
        index_map = {} # old_index -> new_index

        # Build the index map based on remaining instructions
        for i, instr in enumerate(self.intermediate_code):
             index_map[instr['index']] = i
             instr['index'] = i # Update index in the IR itself
             current_index += 1


        # Remap labels
        for label, old_index in self.symbol_table.items():
            if old_index in index_map:
                 new_symbol_table[label] = index_map[old_index]
            else:
                 # Label pointed to an optimized instruction. Find the *next* instruction's *new* index.
                 next_new_index = -1
                 # Search through original indices greater than the removed one
                 sorted_original_indices = sorted(original_indices.keys())
                 for original_idx in sorted_original_indices:
                      if original_idx > old_index:
                          if original_idx in index_map: # Should be true if it wasn't removed
                              next_new_index = index_map[original_idx]
                              break

                 if next_new_index != -1:
                     new_symbol_table[label] = next_new_index
                     print(f"  Warning: Label '{label}' pointed to optimized instruction, remapped to index {next_new_index}")
                 else:
                     # Label pointed to the last instruction(s) which were removed
                     new_symbol_table[label] = len(self.intermediate_code) # Point past the end
                     print(f"  Warning: Label '{label}' pointed to optimized instruction at end, remapped past end.")


        self.symbol_table = new_symbol_table

        print(f"Optimization removed {removed_count} instructions.")
        print(f"Updated Symbol Table (indices): {self.symbol_table}")
        print("--- Optimization Complete ---")


    def _get_instruction_size(self, opcode, operands):
        """ Calculates the size (in bytes) of an encoded instruction. """
        op_lower = opcode.lower()

        if op_lower == 'hlt': return 1 # hlt is 1 byte

        if op_lower in ('add', 'sub', 'mul', 'div', 'cmp', 'mov'):
             if len(operands) == 2:
                 op1_type, _ = self._parse_operand(operands[0])
                 op2_type, _ = self._parse_operand(operands[1])
                 if op1_type == 'register' and op2_type == 'register': return 2
                 if op1_type == 'register' and op2_type == 'immediate': return 4

        if op_lower in ('jmp', 'je', 'jne'):
             if len(operands) == 1:
                 op1_type, _ = self._parse_operand(operands[0])
                 if op1_type == 'label': return 4

        if op_lower == 'syscall':
             if len(operands) == 1:
                 op1_type, _ = self._parse_operand(operands[0])
                 if op1_type == 'syscall_id': return 2 # Opcode + Syscall ID

        return -1 # Error or unknown size

    def pass2(self):
        """Second pass: Encode, resolve addresses, final validation."""
        print("--- Pass 2: Encoding and Address Resolution ---")
        self.binary_code = bytearray()
        self.current_address = 0
        errors = False
        entry_point_address = 0 # Default to 0

        # Calculate final byte addresses for labels
        temp_address = 0
        instruction_addresses = {} # instruction index -> byte address
        for instr in self.intermediate_code:
            instruction_addresses[instr['index']] = temp_address
            size = self._get_instruction_size(instr['opcode'], instr['operands'])
            if size == -1:
                 print(f"Error line {instr['line']}: Cannot determine size for '{instr['opcode']} {' '.join(instr['operands'])}'. Unsupported format?")
                 errors = True
                 size = 0
            temp_address += size

        # Update symbol table with byte addresses
        final_symbol_table = {}
        for label, index in self.symbol_table.items():
            if index in instruction_addresses:
                final_symbol_table[label] = instruction_addresses[index]
            else:
                # This label might point past the last instruction if it was optimized away
                if index == len(self.intermediate_code): # Check if it points past the end
                    final_symbol_table[label] = temp_address # Address after the last instruction
                    print(f"  Label '{label}' points past the last instruction to address 0x{temp_address:04X}")
                else:
                    print(f"Error: Label '{label}' index {index} has no corresponding address.")
                    final_symbol_table[label] = 0xFFFF # Indicate error
                    errors = True
        self.symbol_table = final_symbol_table
        print(f"Final Symbol Table (byte addresses): { {k: f'0x{v:04X}' for k, v in self.symbol_table.items()} }")

        # Determine entry point address
        if self.entry_point_label:
            if self.entry_point_label in self.symbol_table:
                entry_point_address = self.symbol_table[self.entry_point_label]
            else:
                print(f"Error: Entry point label '{self.entry_point_label}' not found.")
                errors = True
        else:
             # Defaulted to address 0 earlier
             entry_point_address = 0
        print(f"Entry Point Address: 0x{entry_point_address:04X}")


        # Encode instructions
        for instr in self.intermediate_code:
            opcode_str = instr['opcode']
            operands = instr['operands']
            line_num = instr['line']
            encoded = None

            try:
                op_byte = OPCODES.get(opcode_str) # Get base opcode byte

                # --- HLT ---
                if opcode_str == 'hlt':
                    if op_byte is None: raise ValueError("Internal error: HLT opcode missing")
                    encoded = struct.pack('>B', op_byte) # 1 byte

                # --- Register-Register ---
                elif opcode_str in ('add', 'sub', 'mul', 'div', 'cmp', 'mov') and len(operands) == 2:
                    op1_type, op1_val = self._parse_operand(operands[0])
                    op2_type, op2_val = self._parse_operand(operands[1])

                    if op1_type == 'register' and op2_type == 'register':
                        if op_byte is None: raise ValueError(f"Internal error: Opcode '{opcode_str}' missing")
                        src_reg = REGISTERS[op1_val]
                        dest_reg = REGISTERS[op2_val]
                        instruction_word = (op_byte << 8) | (src_reg << 4) | dest_reg
                        encoded = struct.pack('>H', instruction_word) # 2 bytes

                    # --- Register-Immediate ---
                    elif op1_type == 'register' and op2_type == 'immediate':
                        imm_opcode_str = f"{opcode_str}_imm"
                        op_byte = OPCODES.get(imm_opcode_str)
                        if op_byte is None: raise ValueError(f"Unsupported immediate operation: {opcode_str}")
                        dest_reg = REGISTERS[op1_val]
                        imm_val = op2_val
                        instruction_part1 = (op_byte << 8) | (dest_reg << 4) # Upper 16 bits
                        encoded = struct.pack('>H', instruction_part1) + struct.pack('>H', imm_val) # 4 bytes

                    else:
                         raise ValueError(f"Invalid operand types for {opcode_str}: {op1_type}, {op2_type}")

                # --- Jumps ---
                elif opcode_str in ('jmp', 'je', 'jne') and len(operands) == 1:
                    op1_type, op1_val = self._parse_operand(operands[0])
                    if op1_type == 'label':
                        if op_byte is None: raise ValueError(f"Internal error: Opcode '{opcode_str}' missing")
                        if op1_val not in self.symbol_table: raise ValueError(f"Undefined label: '{op1_val}'")
                        target_addr = self.symbol_table[op1_val]
                        if not (0 <= target_addr <= 0xFFFF): raise ValueError(f"Target address 0x{target_addr:04X} out of range")
                        instruction_part1 = op_byte << 8 # Upper 16 bits
                        encoded = struct.pack('>H', instruction_part1) + struct.pack('>H', target_addr) # 4 bytes
                    else:
                        raise ValueError(f"Invalid operand type for {opcode_str}: {op1_type}")

                # --- Syscall ---
                elif opcode_str == 'syscall' and len(operands) == 1:
                     op1_type, op1_val = self._parse_operand(operands[0])
                     if op1_type == 'syscall_id':
                         if op_byte is None: raise ValueError("Internal error: SYSCALL opcode missing")
                         syscall_id_byte = SYSCALLS.get(op1_val)
                         if syscall_id_byte is None: raise ValueError(f"Unknown syscall ID: '{op1_val}'")
                         # Format: Opcode (8) | Syscall ID (8)
                         instruction_word = (op_byte << 8) | syscall_id_byte
                         encoded = struct.pack('>H', instruction_word) # 2 bytes
                     else:
                         raise ValueError(f"Invalid operand type for syscall: {op1_type}")

                else:
                    # Check if it was just a label definition line
                    is_label_only = False
                    for lbl, addr in self.symbol_table.items():
                         # This check isn't perfect, assumes labels align exactly with IR entries
                         # A better check would be needed if labels could be mid-instruction
                         if instr['index'] in self.symbol_table.values():
                              is_label_only = True # Likely just a label
                              break
                    if not is_label_only:
                         raise ValueError(f"Unknown or invalid instruction format: {opcode_str} {' '.join(operands)}")

            except ValueError as e:
                print(f"Error line {line_num}: {e}")
                errors = True

            if encoded:
                self.binary_code.extend(encoded)
                self.current_address += len(encoded)
            elif not errors and not is_label_only:
                 # This case should ideally not be reached if all valid instructions encode something
                 print(f"Warning line {line_num}: Instruction did not generate code: {instr['opcode']} {' '.join(instr['operands'])}")


        print(f"--- Pass 2 Complete ({len(self.binary_code)} bytes generated) ---")
        return not errors, entry_point_address


    def write_output(self, filename, entry_point_address):
        """Writes the ROM header and binary code to a file."""
        try:
            with open(filename, 'wb') as f:
                # --- ROM Header ---
                # Magic (4 bytes)
                # Version (2 bytes)
                # Entry Point Address (2 bytes)
                # Code Size (4 bytes)
                # Reserved (4 bytes) - for future use
                header = struct.pack(
                    '>4sHHII', # Big-endian format string
                    ROM_MAGIC,
                    ROM_VERSION,
                    entry_point_address,
                    len(self.binary_code),
                    0 # Reserved
                )
                f.write(header)
                f.write(self.binary_code)
            print(f"Successfully wrote {len(header)} header bytes + {len(self.binary_code)} code bytes to '{filename}'")
        except IOError as e:
            print(f"Error writing output file '{filename}': {e}")

    def assemble(self, source_filename, output_filename):
        """Runs the full assembly process."""
        try:
            with open(source_filename, 'r') as f:
                source_lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Source file not found: '{source_filename}'")
            return False
        except IOError as e:
            print(f"Error reading source file '{source_filename}': {e}")
            return False

        if self.pass1(source_lines):
            self.optimize()
            success, entry_point = self.pass2()
            if success:
                self.write_output(output_filename, entry_point)
                return True
            else:
                print("Assembly failed due to errors in Pass 2.")
                return False
        else:
            print("Assembly failed due to errors in Pass 1.")
            return False

# --- EmuVM Class ---

class EmuVM:
    def __init__(self, memory_size=MEM_SIZE):
        self.memory = bytearray(memory_size)
        self.registers = {
            'rax': 0, 'rbx': 0, 'rcx': 0, 'rdx': 0,
            'pc': 0, # Program Counter
        }
        # Flags
        self.flags = {
            'ZF': 0, # Zero Flag (1 if result is zero, 0 otherwise)
            # Add CF (Carry), SF (Sign), OF (Overflow) as needed
        }
        self.halted = False
        self.debug = False
        self.step_mode = False
        self.rom_header = None

    def load_rom(self, filename):
        """Loads a .rom file with header into memory."""
        try:
            with open(filename, 'rb') as f:
                # Read header (16 bytes)
                header_fmt = '>4sHHII' # Magic, Version, Entry, CodeSize, Reserved
                header_size = struct.calcsize(header_fmt)
                header_data = f.read(header_size)
                if len(header_data) < header_size:
                    print("Error: ROM file too small for header.")
                    return False

                magic, version, entry, code_size, _ = struct.unpack(header_fmt, header_data)

                if magic != ROM_MAGIC:
                    print(f"Error: Invalid ROM magic number. Expected {ROM_MAGIC}, got {magic}")
                    return False

                if version != ROM_VERSION:
                     print(f"Warning: ROM version mismatch. Expected {ROM_VERSION}, got {version}. Attempting to load anyway.")
                     # Add compatibility checks if versions diverge significantly

                self.rom_header = {'magic': magic, 'version': version, 'entry': entry, 'code_size': code_size}
                print(f"ROM Header: {self.rom_header}")

                # Read code
                code = f.read(code_size)
                if len(code) < code_size:
                    print(f"Warning: ROM file code section is smaller ({len(code)} bytes) than header indicates ({code_size} bytes).")

                if entry + len(code) > len(self.memory):
                    print(f"Error: Code size ({len(code)} bytes) exceeds memory capacity when loaded at entry point (0x{entry:04X}).")
                    return False

                # Load code into memory at the entry point address
                self.memory[entry : entry + len(code)] = code
                self.registers['pc'] = entry # Set Program Counter to entry point
                self.halted = False
                print(f"Loaded {len(code)} bytes of code into memory starting at 0x{entry:04X}.")
                return True

        except FileNotFoundError:
            print(f"Error: ROM file not found: '{filename}'")
            return False
        except IOError as e:
            print(f"Error reading ROM file '{filename}': {e}")
            return False
        except struct.error:
             print("Error: Failed to unpack ROM header. File might be corrupted.")
             return False

    def _fetch_byte(self):
        """Fetches a byte from memory at PC and increments PC."""
        if self.registers['pc'] >= len(self.memory):
            print(f"Error: PC (0x{self.registers['pc']:04X}) out of bounds!")
            self.halted = True
            return None
        byte = self.memory[self.registers['pc']]
        self.registers['pc'] += 1
        return byte

    def _fetch_word(self):
        """Fetches a 16-bit word (big-endian) from memory at PC and increments PC by 2."""
        b1 = self._fetch_byte()
        if b1 is None: return None
        b2 = self._fetch_byte()
        if b2 is None: return None
        return (b1 << 8) | b2

    def _get_reg_name(self, code):
        """Gets register name from 4-bit code."""
        return REG_CODE_TO_NAME.get(code & 0x0F, '???') # Mask to be sure

    def _read_reg(self, reg_name):
        """Reads value from a register."""
        return self.registers.get(reg_name, 0) # Default to 0 if invalid?

    def _write_reg(self, reg_name, value):
        """Writes value to a register (handling 16-bit overflow)."""
        if reg_name in self.registers:
            self.registers[reg_name] = value & 0xFFFF # Keep it 16-bit
        else:
            print(f"Warning: Attempt to write to unknown register '{reg_name}'")

    def _set_flags_cmp(self, val1, val2):
        """Sets flags based on a comparison (val1 - val2)."""
        result = (val1 - val2) & 0xFFFF # Simulate 16-bit subtraction
        self.flags['ZF'] = 1 if result == 0 else 0
        # Add other flags (SF, CF, OF) if needed for other instructions

    def _print_state(self):
        """Prints the current state of registers and flags."""
        regs_str = " ".join([f"{name.upper()}:0x{val:04X}" for name, val in self.registers.items()])
        flags_str = " ".join([f"{name}:{val}" for name, val in self.flags.items()])
        print(f"State | PC:0x{self.registers['pc']:04X} | {regs_str} | Flags:[{flags_str}]")

    def run(self):
        """Executes the loaded program."""
        if self.halted:
            print("VM is halted.")
            return

        print("\n--- Starting Execution ---")
        while not self.halted:
            if self.step_mode or self.debug:
                self._print_state()

            start_pc = self.registers['pc'] # PC before fetching instruction
            opcode_byte = self._fetch_byte()

            if opcode_byte is None: break # Error fetching

            if self.debug:
                print(f"Debug | @0x{start_pc:04X}: Fetched Opcode 0x{opcode_byte:02X}")

            executed = False # Flag to check if any instruction matched

            # --- Decode and Execute ---

            # 1-Byte Instructions
            if opcode_byte == OPCODES['hlt']:
                if self.debug: print(f"Debug | Executing HLT")
                self.halted = True
                executed = True

            # 2-Byte Instructions (Opcode | Ops)
            elif opcode_byte in (OPCODES['add'], OPCODES['sub'], OPCODES['mul'],
                                 OPCODES['div'], OPCODES['cmp'], OPCODES['mov'],
                                 OPCODES['syscall']):
                operand_byte = self._fetch_byte()
                if operand_byte is None: break

                if self.debug: print(f"Debug | Fetched Operand Byte 0x{operand_byte:02X}")

                # Register-Register Ops
                if opcode_byte != OPCODES['syscall']: # Syscall handled separately
                    src_reg_code = (operand_byte >> 4) & 0x0F
                    dest_reg_code = operand_byte & 0x0F
                    src_reg_name = self._get_reg_name(src_reg_code)
                    dest_reg_name = self._get_reg_name(dest_reg_code)
                    src_val = self._read_reg(src_reg_name)
                    dest_val = self._read_reg(dest_reg_name)

                    if self.debug: print(f"Debug | Executing {list(OPCODES.keys())[list(OPCODES.values()).index(opcode_byte)].upper()} {src_reg_name.upper()}, {dest_reg_name.upper()}")

                    result = 0
                    if opcode_byte == OPCODES['add']:
                        result = (dest_val + src_val)
                        self._write_reg(dest_reg_name, result)
                    elif opcode_byte == OPCODES['sub']:
                        result = (dest_val - src_val)
                        self._write_reg(dest_reg_name, result)
                        self._set_flags_cmp(dest_val, src_val) # Set flags based on sub
                    elif opcode_byte == OPCODES['mul']:
                        result = (dest_val * src_val)
                        self._write_reg(dest_reg_name, result) # Note: High bits lost
                    elif opcode_byte == OPCODES['div']:
                        if src_val == 0:
                            print(f"Error @0x{start_pc:04X}: Division by zero!")
                            self.halted = True
                        else:
                            # Simple integer division, result in dest, remainder lost
                            result = int(dest_val / src_val)
                            self._write_reg(dest_reg_name, result)
                    elif opcode_byte == OPCODES['cmp']:
                        self._set_flags_cmp(dest_val, src_val) # Only sets flags
                    elif opcode_byte == OPCODES['mov']:
                        self._write_reg(dest_reg_name, src_val)
                    executed = True

                # Syscall Op
                elif opcode_byte == OPCODES['syscall']:
                    syscall_id = operand_byte
                    if self.debug: print(f"Debug | Executing SYSCALL ID: 0x{syscall_id:02X}")

                    if syscall_id == SYSCALLS['print_rax']:
                        print(f"Syscall Output: {self._read_reg('rax')}") # Print decimal value
                    else:
                        print(f"Warning @0x{start_pc:04X}: Unknown syscall ID 0x{syscall_id:02X}")
                    executed = True


            # 4-Byte Instructions (Opcode | DestReg/0 | Immediate/Address)
            elif opcode_byte in (OPCODES['mov_imm'], OPCODES['add_imm'], OPCODES['sub_imm'],
                                 OPCODES['cmp_imm'], OPCODES['jmp'], OPCODES['je'], OPCODES['jne']):
                op_byte2 = self._fetch_byte() # Contains DestReg or is 0 for jumps
                imm_or_addr = self._fetch_word() # 16-bit immediate or address
                if op_byte2 is None or imm_or_addr is None: break

                if self.debug: print(f"Debug | Fetched OpByte2 0x{op_byte2:02X}, Imm/Addr 0x{imm_or_addr:04X}")

                # Register-Immediate Ops
                if opcode_byte in (OPCODES['mov_imm'], OPCODES['add_imm'], OPCODES['sub_imm'], OPCODES['cmp_imm']):
                    dest_reg_code = (op_byte2 >> 4) & 0x0F
                    dest_reg_name = self._get_reg_name(dest_reg_code)
                    dest_val = self._read_reg(dest_reg_name)
                    imm_val = imm_or_addr # Immediate value

                    if self.debug: print(f"Debug | Executing {list(OPCODES.keys())[list(OPCODES.values()).index(opcode_byte)].upper()} {dest_reg_name.upper()}, 0x{imm_val:04X}")

                    result = 0
                    if opcode_byte == OPCODES['mov_imm']:
                        self._write_reg(dest_reg_name, imm_val)
                    elif opcode_byte == OPCODES['add_imm']:
                        result = (dest_val + imm_val)
                        self._write_reg(dest_reg_name, result)
                    elif opcode_byte == OPCODES['sub_imm']:
                        result = (dest_val - imm_val)
                        self._write_reg(dest_reg_name, result)
                        self._set_flags_cmp(dest_val, imm_val)
                    elif opcode_byte == OPCODES['cmp_imm']:
                         self._set_flags_cmp(dest_val, imm_val)
                    executed = True

                # Jump Ops
                elif opcode_byte in (OPCODES['jmp'], OPCODES['je'], OPCODES['jne']):
                    target_addr = imm_or_addr
                    jump_taken = False
                    jump_type = list(OPCODES.keys())[list(OPCODES.values()).index(opcode_byte)].upper()

                    if opcode_byte == OPCODES['jmp']:
                        jump_taken = True
                    elif opcode_byte == OPCODES['je']:
                        jump_taken = (self.flags['ZF'] == 1)
                    elif opcode_byte == OPCODES['jne']:
                        jump_taken = (self.flags['ZF'] == 0)

                    if self.debug: print(f"Debug | Executing {jump_type} 0x{target_addr:04X} (ZF={self.flags['ZF']}) -> Taken: {jump_taken}")

                    if jump_taken:
                        if target_addr >= len(self.memory):
                             print(f"Error @0x{start_pc:04X}: Jump target 0x{target_addr:04X} out of bounds!")
                             self.halted = True
                        else:
                             self.registers['pc'] = target_addr # Set PC to target
                    executed = True

            # --- End Decode ---

            if not executed and not self.halted:
                print(f"Error @0x{start_pc:04X}: Unknown or unimplemented opcode 0x{opcode_byte:02X}")
                self.halted = True

            if self.step_mode and not self.halted:
                input("Press Enter to step...") # Wait for user

        print("--- Execution Halted ---")
        self._print_state() # Print final state


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T3 Assembler and Emulator")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Assembler command
    parser_asm = subparsers.add_parser('assemble', help='Assemble a .asm file into a .rom file')
    parser_asm.add_argument('source', help='Input assembly file (.asm)')
    parser_asm.add_argument('output', help='Output ROM file (.rom)')

    # Emulator command
    parser_emu = subparsers.add_parser('run', help='Run a .rom file in the emulator')
    parser_emu.add_argument('romfile', help='Input ROM file (.rom)')
    parser_emu.add_argument('--debug', action='store_true', help='Enable detailed execution logging')
    parser_emu.add_argument('--step', action='store_true', help='Step through execution instruction by instruction')

    args = parser.parse_args()

    if args.command == 'assemble':
        assembler = Assembler()
        assembler.assemble(args.source, args.output)

    elif args.command == 'run':
        vm = EmuVM()
        vm.debug = args.debug
        vm.step_mode = args.step
        if vm.load_rom(args.romfile):
            vm.run()

    # Example of how to create a sample source file:
    sample_asm_file = "sample.asm"
    if not os.path.exists(sample_asm_file):
        print(f"\nCreating sample source file '{sample_asm_file}'...")
        sample_asm_code = """
; Example Assembly Code for T3 Assembler/Emulator
.entry start      ; Define the entry point

start:
    mov rax, 10      ; rax = 10
    mov rbx, 5       ; rbx = 5
    mov rcx, rax     ; rcx = rax (redundant, optimized out)
    mov rcx, rax     ; rcx = rax (kept)

loop_top:
    cmp rax, 0       ; Compare rax with 0
    je print_result  ; Jump to print_result if rax == 0 (ZF=1)

    syscall print_rax ; Print current value of rax
    sub rax, 1       ; rax = rax - 1
    add rbx, 10      ; rbx = rbx + 10 (just to do something)
    jmp loop_top     ; Jump back to loop_top

print_result:
    mov rax, rbx     ; Move final rbx value to rax
    syscall print_rax ; Print the final rbx value stored in rax
    hlt              ; Halt execution
"""
        try:
            with open(sample_asm_file, "w") as f:
                f.write(sample_asm_code)
            print(f"Created '{sample_asm_file}'. You can now run:")
            print(f"1. Assemble: python test.py assemble {sample_asm_file} sample.rom")
            print(f"2. Run:      python test.py run sample.rom")
            print(f"3. Debug Run: python test.py run sample.rom --debug --step")
        except IOError as e:
            print(f"Error creating sample file: {e}")

