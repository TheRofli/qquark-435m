from pathlib import Path

def detect_project_context(root: str | Path = ".") -> str:
    root = Path(root).resolve()
    found: list[str] = []

    def exists(name: str) -> bool:
        return (root / name).exists()

    if list(root.glob("*.uproject")):
        found.append("Project type: Unreal Engine project. Likely C++/Blueprints. If UI is involved, prefer UMG/Slate. Do not suggest HTML/CSS/React unless web files exist.")

    if exists("package.json"):
        found.append("Project type: JavaScript/TypeScript project. Inspect package.json to infer React, Vite, Next.js, Node, or tooling.")

    if exists("vite.config.ts") or exists("vite.config.js"):
        found.append("Framework hint: Vite frontend project.")

    if exists("next.config.js") or exists("next.config.mjs"):
        found.append("Framework hint: Next.js project.")

    if exists("pyproject.toml") or exists("requirements.txt"):
        found.append("Project type: Python project.")

    if exists("CMakeLists.txt"):
        found.append("Project type: C/C++ project using CMake.")

    if exists("Cargo.toml"):
        found.append("Project type: Rust project.")

    if exists("docker-compose.yml") or exists("compose.yml"):
        found.append("Infrastructure hint: Docker Compose is present.")

    if exists("android") and exists("ios"):
        found.append("Project type: likely React Native or mobile app project.")

    if exists("__manifest__.py") or exists("__openerp__.py"):
        found.append("Project type: Odoo module. Prefer Python/XML views/assets, not React, unless frontend assets exist.")

    if not found:
        return (
            "Project context: unknown. "
            "The agent should inspect relevant files before choosing a stack or framework."
        )

    return "\n".join(f"- {item}" for item in found)
