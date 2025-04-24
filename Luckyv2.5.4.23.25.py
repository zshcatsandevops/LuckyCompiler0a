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
}

# Define register codes
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple x86-like assembler to binary .o file'
    )
    parser.add_argument('source', help='Path to source assembly file (.asm)')
    parser.add_argument('-o', '--output', default='output_file.o',
                        help='Output object file (default: %(default)s)')
    return parser.parse_args()


def assemble_line(line, line_no):
    """
    Parse a single assembly line and return its 2-byte encoding.
    Expected syntax: OP SRC,DEST
    """
    # Strip comments and whitespace
    code_part = line.split('#', 1)[0].strip()
    if not code_part:
        return None

    # Tokenize
    try:
        op_and_args = code_part.replace(',', ' ').split()
        op, src, dest = op_and_args
    except ValueError:
        logging.warning(f"Skipping invalid syntax at line {line_no}: '{line.strip()}'")
        return None

    op = op.lower()
    src = src.lower()
    dest = dest.lower()

    if op not in OPCODES:
        logging.error(f"Unknown opcode '{op}' at line {line_no}")
        return None
    if src not in REGISTERS:
        logging.error(f"Unknown register '{src}' at line {line_no}")
        return None
    if dest not in REGISTERS:
        logging.error(f"Unknown register '{dest}' at line {line_no}")
        return None

    op_code = OPCODES[op]
    src_code = REGISTERS[src]
    dest_code = REGISTERS[dest]

    # Encode: [8 bits opcode][4 bits src][4 bits dest]
    instruction = (op_code << 8) | (src_code << 4) | dest_code
    return instruction.to_bytes(2, byteorder='big')


def assemble_file(source_path):
    """
    Read the source file and assemble into a bytearray of code.
    """
    if not os.path.isfile(source_path):
        logging.error(f"Source file not found: {source_path}")
        sys.exit(1)

    with open(source_path, 'r') as f:
        lines = f.readlines()

    output_bytes = bytearray()
    count = 0

    for idx, line in enumerate(lines, start=1):
        encoded = assemble_line(line, idx)
        if encoded:
            output_bytes.extend(encoded)
            count += 1

    logging.info(f"Assembled {count} instruction(s)")
    return output_bytes


def write_output(output_path, data):
    """
    Write assembled bytes to the output file in one go.
    """
    with open(output_path, 'wb') as f:
        f.write(data)
    logging.info(f"Wrote {len(data)} bytes to '{output_path}'")


def main():
    args = parse_args()
    assembled = assemble_file(args.source)
    write_output(args.output, assembled)


if __name__ == '__main__':
    main()
