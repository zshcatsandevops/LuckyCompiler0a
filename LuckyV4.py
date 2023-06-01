import subprocess

def lucky_help():
  """Prints the help message for the Lucky compiler."""

  command = ["lucky", "-help"]
  output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")

  print(output)

def lucky_version():
  """Prints the version of the Lucky compiler."""

  command = ["lucky", "-version"]
  output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")

  print(output)

def lucky_copyright():
  """Prints the copyright information for the Lucky compiler."""

  command = ["lucky", "-copyright"]
  output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode("utf-8")

  print(output)

def lucky_compile(source_file, output_file):
  """Compiles a C source file using the Lucky compiler."""

  command = ["lucky", "-o", output_file, source_file]
  subprocess.run(command)

def lucky_run(source_file):
  """Compiles and runs a C source file using the Lucky compiler."""
import sys
import subprocess
output_file = "lucky_output.out"


if __name__ == "__main__":
  if len(sys.argv) == 1:
    lucky_help()
  elif sys.argv[1] == "-version":
    lucky_version()
  elif sys.argv[1] == "-copyright":
    lucky_copyright()
  elif sys.argv[1] == "-compile":
    lucky_compile(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == "-run":
    lucky_run(sys.argv[2])
  else:
    print("Unknown command")
