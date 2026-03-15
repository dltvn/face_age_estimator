from __future__ import annotations

from pathlib import Path
import tomllib


def main() -> None:
    """Generate requirements.txt from project dependencies in pyproject.toml."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    pyproject_path = project_root / "pyproject.toml"
    requirements_path = project_root / "requirements.txt"

    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject_data.get("project", {}).get("dependencies", [])

    if not dependencies:
        raise ValueError("No project.dependencies found in pyproject.toml")

    content = "\n".join(dependencies) + "\n"
    requirements_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
