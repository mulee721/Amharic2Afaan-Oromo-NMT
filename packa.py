import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Mulee\Documents\AI-MT")
VENVS = [".venv", ".venv-1", ".venv-2"]
OUTPUT_FILE = PROJECT_ROOT / "requirements.txt"


def get_venv_python(venv_path):
    """Finds the Python executable inside a virtual environment (Windows or Unix)."""
    win_python = venv_path / "Scripts" / "python.exe"
    if win_python.exists():
        return win_python

    unix_python = venv_path / "bin" / "python"
    if unix_python.exists():
        return unix_python

    return None


def get_installed_packages(python_exe):
    """Executes 'pip freeze' using the target virtual environment's Python interpreter."""
    try:
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error running pip freeze with {python_exe}: {e}")
        return []


def main():
    combined_packages = {}

    for venv_name in VENVS:
        venv_path = PROJECT_ROOT / venv_name

        if not venv_path.exists():
            print(f"⚠️ Skipped: Directory '{venv_name}' does not exist.")
            continue

        python_exe = get_venv_python(venv_path)
        if not python_exe:
            print(f"⚠️ Skipped: Python executable not found in '{venv_name}'.")
            continue

        print(f"Scanning installed packages in '{venv_name}'...")
        packages = get_installed_packages(python_exe)

        for line in packages:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract normalized package name for deduplication
            if "==" in line:
                pkg_name = line.split("==")[0].lower()
            elif "@" in line:
                pkg_name = line.split("@")[0].strip().lower()
            else:
                pkg_name = line.lower()

            if pkg_name in combined_packages:
                if combined_packages[pkg_name] != line:
                    print(
                        f"  ℹ Duplicate found for '{pkg_name}': Keeping '{combined_packages[pkg_name]}' (ignored '{line}' from {venv_name})"
                    )
            else:
                combined_packages[pkg_name] = line

    # Sort packages alphabetically
    sorted_requirements = sorted(
        combined_packages.values(), key=lambda s: s.lower()
    )

    # Save to root requirements.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for req in sorted_requirements:
            f.write(f"{req}\n")

    print(
        f"\n✓ Successfully saved {len(sorted_requirements)} unique packages to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()