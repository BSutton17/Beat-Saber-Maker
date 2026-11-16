"""Copy the current mapping engine from the AI Mapper folder into this program, so the program always maps with the newest model and still works on a computer without AI Mapper.

    python sync_engine.py "<AI Mapper folder>"          copy, if anything changed
    python sync_engine.py "<AI Mapper folder>" --check  only report whether anything changed

AI Mapper/models/CURRENT.json names the builder, extra modules and model folders. The builder and every tools module it imports go to core/, the models to models/.
mapfiles.py is never copied: this program has its own, which points the engine at this folder and the chosen output folder.
"""
import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE, MODELS = HERE / "core", HERE / "models"
NEVER_COPY = {"mapfiles"}


def closure(tools, start):
    seen, todo = set(), list(start)
    while todo:
        name = todo.pop()
        if name in seen or not (tools / f"{name}.py").exists():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((tools / f"{name}.py").read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                todo += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                todo.append(node.module.split(".")[0])
    return sorted(seen - NEVER_COPY)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(source):
    current = json.loads((source / "models" / "CURRENT.json").read_text(encoding="utf-8"))
    tools = source / "tools"
    files = {}
    for name in closure(tools, [current["builder"], *current.get("extra_modules", [])]):
        files[f"core/{name}.py"] = tools / f"{name}.py"
    for model in current["models"]:
        for file_name in current.get("model_files", ["weights.json"]):
            if (source / "models" / model / file_name).exists():
                files[f"models/{model}/{file_name}"] = source / "models" / model / file_name
    return current, files


def main():
    source = Path(sys.argv[1]).resolve()
    current, files = plan(source)
    changed = [rel for rel, src in files.items() if not (HERE / rel).exists() or digest(HERE / rel) != digest(src)]
    stale = [p for p in CORE.glob("*.py") if f"core/{p.name}" not in files]
    if "--check" in sys.argv:
        print(json.dumps({"changed": changed, "stale": [p.name for p in stale]}))
        return
    for rel in changed:
        (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[rel], HERE / rel)
    for p in stale:
        p.unlink()
    record = {"builder": current["builder"], "models": current["models"], "synced": datetime.now().isoformat(timespec="seconds"),
              "files": {rel: digest(HERE / rel) for rel in files}}
    (HERE / "CURRENT.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({"changed": changed, "removed": [p.name for p in stale], "builder": current["builder"], "models": current["models"]}))


if __name__ == "__main__":
    main()
