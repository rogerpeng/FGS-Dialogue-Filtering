"""Explicit allowlist: no private corpus, checkpoints, cached API output, or .env."""
from pathlib import Path
import zipfile


def bundle(out):
    root = Path(__file__).resolve().parents[1]
    out = Path(out)
    if out.exists():
        raise ValueError("Bundle already exists")
    included = []
    for directory, suffixes in [("fgs", {".py"}), ("tests", {".py"}), ("docs", {".md"}),
                                ("examples", {".csv", ".jsonl", ".md"}),
                                ("scripts", {".ps1", ".py"})]:
        included.extend(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    included.extend(root / name for name in ["README.md", "pyproject.toml", ".gitignore", "requirements-tested.txt"] if (root / name).exists())
    included.extend((root / "configs").glob("*.example.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, "revision3_code/" + path.relative_to(root).as_posix())
    return {"path": str(out), "files": len(included), "published": False,
            "note": "Review authorship/licensing and source before public release. This bundle excludes private data and run artifacts."}
