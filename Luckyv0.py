import os, sys

# x86 operations 
opcodes = { 
    'add': 0x0000,
    'sub': 0x0001, 
    'mul': 0x0002, 
    'div': 0x0003, 
    'cmp': 0x0004
}

# Registers 
registers = { 
    'rax': 0x000A, 
    'rbx': 0x000B, 
    'rcx': 0x000C, 
    'rdx': 0x000D
}

# Load the assembly code file 
with open(sys.argv[1], 'r') as f:
    code = f.readlines()

# Main compiler loop 
def compile():
    for line in code: 
        parts = line.split()

        #operation
        op = parts[0]
        if op in opcodes: 
            op_code = opcodes[op]
        
        #source
        src1 = parts[1]
        if src1 in registers: 
            src1_code = registers[src1]

        #destination
        dest = parts[2]
        if dest in registers:
            dest_code = registers[dest]
            
        #write instruction to file
        with open('output_file.o', 'a') as f: 
            f.write((op_code << 4) + src1_code + dest_code)

compile()