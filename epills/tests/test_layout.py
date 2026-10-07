"""The hosted code must run without the training side: epills.core and serving/ never import epills.training."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("epills.training",)


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_hosted_code_does_not_import_training():
    hosted = list((ROOT / "src" / "epills" / "core").rglob("*.py")) + list((ROOT / "serving").rglob("*.py"))
    assert hosted, "no hosted source files found"
    offenders = [f"{p.relative_to(ROOT)} imports {m}" for p in hosted for m in imported_modules(p)
                 if m.startswith(FORBIDDEN)]
    assert not offenders, "hosted code imports training code:\n" + "\n".join(offenders)
