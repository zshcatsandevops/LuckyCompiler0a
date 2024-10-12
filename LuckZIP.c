import re
import subprocess
import sys

# Optional: For colored outputs
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = ''
        CYAN = ''
        RED = ''
    class Style:
        BRIGHT = ''
        RESET_ALL = ''

# Dictionary to map simple x86 assembly instructions to C equivalents
ASM_TO_C_MAP = {
    'mov': ('{dst} = {src};', 2),
    'add': ('{dst} += {src};', 2),
    'sub': ('{dst} -= {src};', 2),
    'mul': ('{dst} *= {src};', 2),
    'div': ('{dst} /= {src};', 2),
    'inc': ('{dst}++;', 1),
    'dec': ('{dst}--;', 1)
}

def is_variable(operand):
    try:
        int(operand, 0)  # Supports decimal, hex, octal, binary literals
        return False
    except ValueError:
        return True

def asm_to_c_line(line, variables):
    line = line.strip()
    if not line or line.startswith(';'):
        return ''

    # Regex to match assembly instructions
    match = re.match(r'(\w+)\s+([^,\s]+)(?:,\s*([^,\s]+))?', line)
    if match:
        instr = match.group(1).lower()
        dst = match.group(2)
        src = match.group(3) if match.lastindex >= 3 else ''

        if instr in ASM_TO_C_MAP:
            template, num_operands = ASM_TO_C_MAP[instr]
            if num_operands == 2 and not src:
                return f'// Error: Instruction "{instr}" requires two operands.'
            if num_operands == 1 and src:
                return f'// Error: Instruction "{instr}" requires one operand.'

            # Collect variables
            if is_variable(dst):
                variables.add(dst)
            if num_operands == 2 and is_variable(src):
                variables.add(src)

            # Format the C code using the map
            c_line = template.format(dst=dst, src=src)
            return c_line
        else:
            return f'// Unsupported instruction: {line}'
    else:
        return f'// Could not parse line: {line}'

def generate_c_code(c_statements, variables):
    # Generate variable declarations
    decl_code = [f'int {var} = 0;' for var in variables]

    # Wrap the code into a main function
    full_c_code = []
    full_c_code.append('#include <stdio.h>')
    full_c_code.append('int main() {')
    full_c_code.extend(['    ' + line for line in decl_code])
    full_c_code.extend(['    ' + line for line in c_statements])
    # Print the value of variables at the end
    for var in variables:
        full_c_code.append(f'    printf("{var} = %d\\n", {var});')
    full_c_code.append('    return 0;')
    full_c_code.append('}')
    return '\n'.join(full_c_code)

def main():
    print(Fore.CYAN + Style.BRIGHT + "Welcome to the Assembly-to-C Compiler!")
    print("Type your assembly instructions below.")
    print("Type '/generate' to generate and compile the C code.")
    print("Type '/exit' to quit." + Style.RESET_ALL)

    asm_lines = []
    variables = set()
    while True:
        try:
            user_input = input(Fore.GREEN + ">> " + Style.RESET_ALL)
            if user_input.strip() == '':
                continue
            if user_input.startswith('/'):
                command = user_input.strip().lower()
                if command == '/exit':
                    print(Fore.CYAN + "Exiting. Goodbye!" + Style.RESET_ALL)
                    break
                elif command == '/generate':
                    c_statements = []
                    for asm_line in asm_lines:
                        c_line = asm_to_c_line(asm_line, variables)
                        if c_line:
                            c_statements.append(c_line)
                    c_code = generate_c_code(c_statements, variables)
                    print(Fore.CYAN + "\nGenerated C code:\n" + Style.RESET_ALL)
                    print(c_code)
                    # Save to output.c
                    with open('output.c', 'w') as f:
                        f.write(c_code)
                    # Compile the generated C code using GCC
                    compile_result = subprocess.run(['gcc', 'output.c', '-o', 'output'], capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        print(Fore.RED + "Compilation failed:" + Style.RESET_ALL)
                        print(compile_result.stderr)
                    else:
                        print(Fore.CYAN + "Compilation succeeded. Running the program..." + Style.RESET_ALL)
                        # Run the compiled program
                        if sys.platform.startswith('win'):
                            executable = 'output.exe'
                        else:
                            executable = './output'
                        run_result = subprocess.run([executable], capture_output=True, text=True)
                        print(Fore.CYAN + "Program output:" + Style.RESET_ALL)
                        print(run_result.stdout)
                elif command == '/help':
                    print(Fore.CYAN + "Available commands:" + Style.RESET_ALL)
                    print("/generate - Generate and compile the C code from the assembly instructions.")
                    print("/exit - Exit the compiler.")
                    print("/help - Display this help message.")
                else:
                    print(Fore.RED + "Unknown command. Type '/help' for a list of commands." + Style.RESET_ALL)
            else:
                asm_lines.append(user_input)
        except KeyboardInterrupt:
            print("\n" + Fore.CYAN + "Exiting. Goodbye!" + Style.RESET_ALL)
            break

if __name__ == "__main__":
    main()
