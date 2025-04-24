import os, sys

# x86 operations
opcodes = {
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03,
    'cmp': 0x04
}

# Registers
registers = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D
}

# Load the assembly code file
if len(sys.argv) < 2:
    print("Usage: python compiler.py <source.asm>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    code = f.readlines()

# Clear output before writing
with open('output_file.o', 'wb') as f:
    pass

# Main compiler loop
def compile():
    for line in code:
        parts = line.strip().split()
        if len(parts) != 3:
            print(f"Invalid instruction: {line.strip()}")
            continue

        op, src1, dest = parts

        if op not in opcodes or src1 not in registers or dest not in registers:
            print(f"Unknown symbol in line: {line.strip()}")
            continue

        op_code = opcodes[op]
        src1_code = registers[src1]
        dest_code = registers[dest]

        instruction = (op_code << 8) | (src1_code << 4) | dest_code

        # Write as binary
        with open('output_file.o', 'ab') as f:
            f.write(instruction.to_bytes(2, byteorder='big'))

compile()
