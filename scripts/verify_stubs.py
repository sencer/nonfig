from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
  """Verify stubs by generating them and running basedpyright."""
  # Ensure dependencies are installed
  if not shutil.which("nonfig-stubgen"):
    print(
      "Error: nonfig-stubgen command not found. Install package in editable mode first."
    )
    sys.exit(1)

  if not shutil.which("basedpyright"):
    print("Error: basedpyright command not found.")
    sys.exit(1)

  project_root = Path(__file__).parent.parent
  examples_dir = project_root / "examples"

  if not examples_dir.exists():
    print(f"Error: Examples directory not found at {examples_dir}")
    sys.exit(1)

  print(f"Generating stubs for {examples_dir}...")
  try:
    subprocess.run(
      ["nonfig-stubgen", str(examples_dir)],
      check=True,
      capture_output=True,
      text=True,
    )
  except subprocess.CalledProcessError as e:
    print("Error generating stubs:")
    print(e.stdout)
    print(e.stderr)
    sys.exit(1)

  print("Running basedpyright on examples...")
  try:
    # Check all python files in examples
    result = subprocess.run(
      ["basedpyright", str(examples_dir), "--level", "error"],
      check=False,  # We handle return code manually
      capture_output=True,
      text=True,
    )

    if result.returncode != 0:
      print("Type checking failed:")
      print(result.stdout)
      print(result.stderr)
      sys.exit(1)

    print("Success! Stubs pass type checking.")

  except Exception as e:  # noqa: BLE001
    print(f"An error occurred: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
