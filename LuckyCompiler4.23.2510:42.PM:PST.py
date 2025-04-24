import argparse
import sys
import logging
import os

# Define opcodes for a custom x86-like instruction set
OPCODES = {
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03,
    'cmp': 0x04,
    'mov': 0x05,
    'jmp': 0x06,
    'push': 0x07,
    'pop': 0x08,
    'nop': 0x09,
}

# Define register codes
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple x86-like assembler to binary .o file'
    )
    parser.add_argument('source', help='Path to source assembly file (.asm)')
    parser.add_argument('-o', '--output', default='output_file.o',
                        help='Output object file (default: %(default)s)')
    parser.add_argument('--hexdump', action='store_true',
                        help='Print the assembled output as a hex dump to stdout')
    parser.add_argument('--strict', action='store_true',
                        help='Halt execution on the first assembly error')
    return parser.parse_args()

def parse_instruction(line):
    """Parse a line into (op, src, dest) or (op, src, None) or (op, None, None)."""
    code = line.split('#', 1)[0].strip()
    if not code:
        return None, None, None
    tokens = code.replace(',', ' ').split()
    if not tokens:
        return None, None, None
    op = tokens[0].lower()
    src = tokens[1].lower() if len(tokens) > 1 else None
    dest = tokens[2].lower() if len(tokens) > 2 else None
    return op, src, dest

def encode_instruction(op, src, dest, line_no, strict_mode):
    """Encode instruction to 2 bytes, handling errors and strict mode."""
    if op not in OPCODES:
        msg = f"Unknown opcode '{op}' at line {line_no}"
        if strict_mode:
            logging.error(msg)
            sys.exit(1)
        logging.warning(msg)
        return None

    opcode = OPCODES[op]

    # Two-operand instructions
    if op in {'add', 'sub', 'mul', 'div', 'cmp', 'mov'}:
        if src not in REGISTERS or dest not in REGISTERS:
            msg = f"Invalid register(s) at line {line_no}: '{src}', '{dest}'"
            if strict_mode:
                logging.error(msg)
                sys.exit(1)
            logging.warning(msg)
            return None
        src_code = REGISTERS[src]
        dest_code = REGISTERS[dest]
        return ((opcode << 8) | (src_code << 4) | dest_code).to_bytes(2, 'big')

    # Single-operand instructions
    if op in {'push', 'pop', 'jmp'}:
        if src not in REGISTERS:
            msg = f"Invalid register '{src}' at line {line_no}"
            if strict_mode:
                logging.error(msg)
                sys.exit(1)
            logging.warning(msg)
            return None
        src_code = REGISTERS[src]
        # Encode: [8 bits opcode][4 bits src][4 bits zero]
        return ((opcode << 8) | (src_code << 4)).to_bytes(2, 'big')

    # Zero-operand instructions
    if op == 'nop':
        return ((opcode << 8)).to_bytes(2, 'big')

    msg = f"Unsupported instruction format at line {line_no}: '{op}'"
    if strict_mode:
        logging.error(msg)
        sys.exit(1)
    logging.warning(msg)
    return None

def assemble_file(source_path, strict_mode=False):
    if not os.path.isfile(source_path):
        logging.error(f"Source file not found: {source_path}")
        sys.exit(1)

    output_bytes = bytearray()
    instruction_count = 0

    with open(source_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            op, src, dest = parse_instruction(line)
            if not op:
                continue
            encoded = encode_instruction(op, src, dest, idx, strict_mode)
            if encoded:
                output_bytes.extend(encoded)
                instruction_count += 1

    logging.info(f"Assembled {instruction_count} instruction(s)")
    return output_bytes

def write_output(output_path, data):
    try:
        with open(output_path, 'wb') as f:
            f.write(data)
        logging.info(f"Wrote {len(data)} bytes to '{output_path}'")
    except IOError as e:
        logging.error(f"Failed to write output file '{output_path}': {e}")
        sys.exit(1)

def print_hexdump(data):
    print("--- Hex Dump ---")
    if not data:
        print("(empty)")
        return
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        print(' '.join(f"{b:02X}" for b in chunk))
    print("----------------")

def main():
    args = parse_args()
    assembled = assemble_file(args.source, args.strict)
    write_output(args.output, assembled)
    if args.hexdump:
        print_hexdump(assembled)

if __name__ == '__main__':
    main()
