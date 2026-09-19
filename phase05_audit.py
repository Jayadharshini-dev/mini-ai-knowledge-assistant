#!/usr/bin/env python3
"""
Phase 05 structural audit — static analysis only, no imports, no network.

Usage:
    python phase05_audit.py <path-to-repo-root>

Checks the mechanical invariants agreed for Phase 05. It cannot judge semantics
(only a human reading pipeline.py can do that), but it settles the questions that
are objectively decidable from the source, and it does so without trusting any
summary of the code.

Exit code 0 = no BLOCKERs found. Exit code 1 = at least one BLOCKER.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

BLOCKERS: list[str] = []
WARNINGS: list[str] = []
INFO: list[str] = []


def blocker(msg: str) -> None:
    BLOCKERS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def info(msg: str) -> None:
    INFO.append(msg)


# AST helpers
def find_funcs(tree: ast.AST, name: str) -> list[ast.AST]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]


def yields_in_finally(tree: ast.AST) -> list[int]:
    """A yield inside finally raises RuntimeError on generator aclose()."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Yield, ast.YieldFrom)):
                        hits.append(sub.lineno)
    return hits


def bad_handlers(tree: ast.AST) -> list[tuple[int, str]]:
    """Bare except: and except BaseException: swallow CancelledError/GeneratorExit."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                hits.append((node.lineno, "bare except:"))
            elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
                hits.append((node.lineno, "except BaseException:"))
            elif isinstance(node.type, ast.Tuple):
                for e in node.type.elts:
                    if isinstance(e, ast.Name) and e.id == "BaseException":
                        hits.append((node.lineno, "except (... BaseException ...):"))
    return hits


def to_thread_targets(tree: ast.AST) -> list[tuple[int, str]]:
    """Find asyncio.to_thread(<callable>, ...) and report the callable expression."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_tt = (isinstance(f, ast.Attribute) and f.attr == "to_thread") or (
            isinstance(f, ast.Name) and f.id == "to_thread"
        )
        if is_tt and node.args:
            out.append((node.lineno, ast.unparse(node.args[0])))
    return out


def awaited_calls(fn: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Await):
            out.append((node.lineno, ast.unparse(node.value)))
    return out


def instantiations(tree: ast.AST, cls: str) -> list[tuple[int, str]]:
    """Where is <cls>(...) constructed, and inside which enclosing function?"""
    out = []
    parents: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(fn):
                parents[id(sub)] = fn.name
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == cls
        ):
            out.append((node.lineno, parents.get(id(node), "<module>")))
    return out


# --------------------------------------------------------------------------- checks
FORBIDDEN_IMPORTS = [
    "fastapi",
    "starlette",
    "uvicorn",
    "sse_starlette",
    "flask",
    "redis",
    "sqlalchemy",
    "psycopg",
    "sqlite3",
    "celery",
    "kombu",
    "requests",
    "httpx",
    "aiohttp",
    "prometheus_client",
    "opentelemetry",
    "rank_bm25",
    "sklearn.feature_extraction",
]
FORBIDDEN_TOKENS = [
    (r"\bsleep\s*\(", "sleep() — artificial delay"),
    (r"\btime\.time\s*\(", "time.time() — use perf_counter"),
    (r"\bdatetime\.now\s*\(", "datetime.now() — non-deterministic in payloads"),
    (r"\buuid4\s*\(", "uuid4() — non-deterministic identifiers"),
    (r"\brandom\.", "random — non-deterministic"),
    (r"slots\s*=\s*True", "dataclass(slots=True) is Python 3.10+; this project is 3.9"),
    (r"\bfrom\s+.*config\s+import\s+settings\b", "settings import inside backend/rag"),
    (r"\bbuild_pipeline\b", "factory inside backend/rag (belongs to Phase 06)"),
]


def audit_module(path: Path, src: str) -> ast.AST:
    tree = ast.parse(src, filename=str(path))
    rel = path.as_posix()

    for lineno in yields_in_finally(tree):
        blocker(
            f"{rel}:{lineno} yield inside finally — aclose() will raise RuntimeError"
        )

    for lineno, kind in bad_handlers(tree):
        blocker(f"{rel}:{lineno} {kind} swallows CancelledError / GeneratorExit")

    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            for bad in FORBIDDEN_IMPORTS:
                if m == bad or m.startswith(bad + "."):
                    blocker(f"{rel}:{node.lineno} forbidden import for Phase 05: {m}")

    for pattern, why in FORBIDDEN_TOKENS:
        for m in re.finditer(pattern, src):
            line = src[: m.start()].count("\n") + 1
            blocker(f"{rel}:{line} {why}  ->  {m.group(0)!r}")

    return tree


def main(root: Path) -> int:
    rag = root / "backend" / "rag"
    if not rag.is_dir():
        print(f"ERROR: {rag} does not exist. Pass the repo root as argv[1].")
        return 2

    trees: dict[str, ast.AST] = {}
    for py in sorted(rag.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        trees[py.name] = audit_module(py.relative_to(root), src)
        info(f"{py.relative_to(root)}  {len(src.splitlines())} lines")

    # ---- pipeline.py specifics
    pipe = trees.get("pipeline.py")
    if pipe is None:
        blocker("backend/rag/pipeline.py not found")
    else:
        runs = find_funcs(pipe, "run")
        if not runs:
            blocker("pipeline.py: no function named run()")
        for fn in runs:
            if not isinstance(fn, ast.AsyncFunctionDef):
                blocker("pipeline.py: run() is not an async def")
            has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(fn))
            if not has_yield:
                blocker("pipeline.py: run() contains no yield — not an async generator")
            n_yields = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Yield))
            info(f"pipeline.run() yields at {n_yields} sites")

            tt = to_thread_targets(fn)
            info(
                "to_thread targets in run(): "
                + (", ".join(f"L{line_no}:{t}" for line_no, t in tt) or "NONE")
            )
            blob = " ".join(t for _, t in tt).lower()
            for need, label in [
                ("retrieve", "retrieval"),
                ("generate", "provider generation"),
            ]:
                if need not in blob:
                    blocker(
                        f"pipeline.run(): no asyncio.to_thread call appears to wrap "
                        f"{label} (looked for {need!r} among to_thread targets)"
                    )

            awaits = awaited_calls(fn)
            non_tt = [
                (line_no, e)
                for line_no, e in awaits
                if "to_thread" not in e and "sleep" not in e
            ]
            if non_tt:
                warn(
                    "awaits in run() that are not to_thread "
                    "(verify each is non-blocking): "
                    + ", ".join(f"L{line_no}:{e}" for line_no, e in non_tt)
                )

        # emitter must be per-run
        for cls in ("EventEmitter", "TraceEmitter", "Emitter"):
            for lineno, enclosing in instantiations(pipe, cls):
                if enclosing == "__init__":
                    blocker(
                        f"pipeline.py:{lineno} {cls}() constructed in __init__ — "
                        "seq/t0 shared across concurrent runs"
                    )
                else:
                    info(
                        f"pipeline.py:{lineno} {cls}() constructed in "
                        f"{enclosing}()  [OK]"
                    )

    # ---- CONTRACTS.md drift
    contracts = root / "docs" / "CONTRACTS.md"
    ev = trees.get("events.py")
    if contracts.is_file() and ev is not None:
        doc = contracts.read_text(encoding="utf-8")
        doc_types = set(re.findall(r"`([A-Z][A-Z_]{3,})`", doc))
        code_types = set()
        for node in ast.walk(ev):
            if isinstance(node, ast.ClassDef) and "EventType" in node.name:
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(
                        stmt.targets[0], ast.Name
                    ):
                        code_types.add(stmt.targets[0].id)
        only_code = code_types - doc_types
        if only_code:
            blocker(f"EventType members absent from CONTRACTS.md: {sorted(only_code)}")
        info(f"EventType members in code: {len(code_types)}")

        # terminal-event amendment
        term_rows = [
            ln
            for ln in doc.splitlines()
            if re.search(r"\b(ABSTAINED|GENERATION_SKIPPED)\b", ln)
            and "yes" in ln.lower()
        ]
        if term_rows:
            blocker(
                "CONTRACTS.md marks ABSTAINED / GENERATION_SKIPPED as terminal:\n    "
                + "\n    ".join(term_rows)
            )
    else:
        warn(
            "docs/CONTRACTS.md or backend/rag/events.py not found — drift check skipped"
        )

    # ---- report
    print("=" * 72)
    for line in INFO:
        print("INFO    ", line)
    print("-" * 72)
    for line in WARNINGS:
        print("WARN    ", line)
    for line in BLOCKERS:
        print("BLOCKER ", line)
    print("=" * 72)
    print(f"{len(BLOCKERS)} blockers, {len(WARNINGS)} warnings")
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]).resolve()))
