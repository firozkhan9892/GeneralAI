"""Runtime Verification & Code Quality Audit for GeneralAI.

Performs comprehensive static and dynamic analysis:
- TODO/FIXME/HACK/XXX comment audit
- Dead code detection (unused imports, unreachable code, unused functions)
- Unsafe pattern detection (bare except, print statements, mutable defaults, global state)
- Performance benchmarks (startup time, agent latency, memory usage)
- Dependency verification (missing/unused packages)

Run from project root:
    python -m tests.runtime_verification_code_audit
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── Project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
TESTS_DIR = PROJECT_ROOT / "tests"


# ── Data classes for structured findings ──────────────────────────
@dataclass
class Finding:
    category: str
    severity: str  # CRITICAL, WARNING, INFO
    file: str
    line: int
    message: str
    code_snippet: str = ""


@dataclass
class AuditReport:
    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)

    def add(
        self,
        category: str,
        severity: str,
        file: str,
        line: int,
        message: str,
        snippet: str = "",
    ):
        self.findings.append(Finding(category, severity, file, line, message, snippet))

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("  GENERALAI — CODE QUALITY AUDIT REPORT")
        lines.append("=" * 80)

        # ── Stats ──
        lines.append("\n## 1. SUMMARY STATISTICS\n")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")

        # ── Performance ──
        lines.append("\n## 2. PERFORMANCE BENCHMARKS\n")
        for k, v in self.performance.items():
            lines.append(f"  {k}: {v}")

        # ── Findings grouped by category ──
        by_cat: Dict[str, List[Finding]] = defaultdict(list)
        for f in self.findings:
            by_cat[f.category].append(f)

        cat_order = [
            "TODO/FIXME AUDIT",
            "DEAD CODE — Unused Imports",
            "DEAD CODE — Unreachable Code",
            "DEAD CODE — Unused Functions/Classes",
            "UNSAFE PATTERNS — Bare Except",
            "UNSAFE PATTERNS — Print Statements",
            "UNSAFE PATTERNS — Mutable Defaults",
            "UNSAFE PATTERNS — Global State",
            "DEPENDENCY CHECK",
        ]

        section = 3
        for cat in cat_order:
            items = by_cat.get(cat, [])
            if not items:
                continue
            crit = sum(1 for f in items if f.severity == "CRITICAL")
            warn = sum(1 for f in items if f.severity == "WARNING")
            info = sum(1 for f in items if f.severity == "INFO")
            lines.append(
                f"\n## {section}. {cat}  ({len(items)} findings: {crit} critical, {warn} warning, {info} info)\n"
            )
            for f in items:
                rel = os.path.relpath(f.file, PROJECT_ROOT).replace("\\", "/")
                sev_tag = f"[{f.severity}]"
                lines.append(f"  {sev_tag:12s}  {rel}:{f.line}  — {f.message}")
                if f.code_snippet:
                    for sl in f.code_snippet.strip().splitlines():
                        lines.append(f"               | {sl}")
            section += 1

        # ── Top 10 Critical ──
        criticals = [f for f in self.findings if f.severity == "CRITICAL"]
        lines.append(f"\n## {section}. TOP 10 MOST CRITICAL ISSUES\n")
        for i, f in enumerate(criticals[:10], 1):
            rel = os.path.relpath(f.file, PROJECT_ROOT).replace("\\", "/")
            lines.append(f"  {i:2d}. [{f.category}] {rel}:{f.line} — {f.message}")

        lines.append("\n" + "=" * 80)
        lines.append("  END OF AUDIT REPORT")
        lines.append("=" * 80)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 1. TODO / FIXME / HACK / XXX AUDIT
# ══════════════════════════════════════════════════════════════════
TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)


def audit_todo_fixme(report: AuditReport) -> None:
    """Scan app/*.py for TODO, FIXME, HACK, XXX comments."""
    counts: Dict[str, int] = defaultdict(int)
    by_module: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_hits: List[Tuple[Path, int, str, str]] = []

    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, 1):
            m = TODO_PATTERN.search(line)
            if m:
                tag = m.group(1).upper()
                counts[tag] += 1
                rel = py.relative_to(PROJECT_ROOT).as_posix()
                module = rel.split("/")[1] if len(rel.split("/")) > 1 else rel
                by_module[module][tag] += 1
                all_hits.append((py, lineno, tag, line.strip()))

    report.stats["TODO comments"] = counts.get("TODO", 0)
    report.stats["FIXME comments"] = counts.get("FIXME", 0)
    report.stats["HACK comments"] = counts.get("HACK", 0)
    report.stats["XXX comments"] = counts.get("XXX", 0)
    report.stats["Total action-item comments"] = sum(counts.values())

    # Per-module breakdown
    report.stats["TODO/FIXME by module"] = dict(by_module)

    # Severity: FIXME/HACK/XXX = CRITICAL, TODO = WARNING
    for py, lineno, tag, snippet in all_hits:
        sev = "CRITICAL" if tag in ("FIXME", "HACK", "XXX") else "WARNING"
        report.add(
            "TODO/FIXME AUDIT", sev, str(py), lineno, f"{tag} comment found", snippet
        )


# ══════════════════════════════════════════════════════════════════
# 2. DEAD CODE — UNUSED IMPORTS
# ══════════════════════════════════════════════════════════════════
def audit_unused_imports(report: AuditReport) -> None:
    """Parse each .py file and flag imports that are never referenced."""
    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        # Collect imports
        imports: List[Tuple[str, int, str]] = []  # (name, lineno, full_line)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports.append(
                        (name, node.lineno, ast.get_source_segment(source, node) or "")
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports.append(
                        (name, node.lineno, ast.get_source_segment(source, node) or "")
                    )

        # Collect all names used in the file
        all_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # x.y.z → add 'x'
                n = node
                while isinstance(n, ast.Attribute):
                    n = n.value  # type: ignore[assignment]
                if isinstance(n, ast.Name):
                    all_names.add(n.id)

        for name, lineno, snippet in imports:
            if name not in all_names and name != "*":
                report.add(
                    "DEAD CODE — Unused Imports",
                    "WARNING",
                    str(py),
                    lineno,
                    f"Unused import: '{name}'",
                    snippet,
                )


# ══════════════════════════════════════════════════════════════════
# 3. DEAD CODE — UNREACHABLE CODE
# ══════════════════════════════════════════════════════════════════
TERMINATING = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def audit_unreachable_code(report: AuditReport) -> None:
    """Find code blocks that follow a return/raise/break/continue."""
    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            body_lists = []
            if hasattr(node, "body") and isinstance(node.body, list):
                body_lists.append(("body", node.body))
            if hasattr(node, "orelse") and isinstance(node.orelse, list):
                body_lists.append(("orelse", node.orelse))
            if hasattr(node, "handlers") and isinstance(node.handlers, list):
                for h in node.handlers:
                    if hasattr(h, "body"):
                        body_lists.append(("handler", h.body))

            for section_name, body in body_lists:
                for i, stmt in enumerate(body):
                    if isinstance(stmt, TERMINATING):
                        # Check if there are more statements after this
                        remaining = body[i + 1 :]
                        if remaining:
                            # Filter out only Pass statements (which are harmless after return)
                            real_remaining = [
                                s for s in remaining if not isinstance(s, ast.Pass)
                            ]
                            if real_remaining:
                                first = real_remaining[0]
                                report.add(
                                    "DEAD CODE — Unreachable Code",
                                    "WARNING",
                                    str(py),
                                    first.lineno,
                                    f"Unreachable code after {type(stmt).__name__} (line {stmt.lineno})",
                                )


# ══════════════════════════════════════════════════════════════════
# 4. DEAD CODE — UNUSED FUNCTIONS / CLASSES
# ══════════════════════════════════════════════════════════════════
def audit_unused_definitions(report: AuditReport) -> None:
    """Heuristic: functions/classes defined in app/ but never referenced anywhere else."""
    # Build a set of all names used across the entire app/ codebase
    all_names_used: Set[str] = set()
    all_files: List[Path] = []

    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        all_files.append(py)
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names_used.add(node.id)
            elif isinstance(node, ast.Attribute):
                n = node
                while isinstance(n, ast.Attribute):
                    n = n.value  # type: ignore[assignment]
                if isinstance(n, ast.Name):
                    all_names_used.add(n.id)

    # Also gather names from test files (they reference app names)
    for py in sorted(TESTS_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                all_names_used.add(node.id)
            elif isinstance(node, ast.Attribute):
                n = node
                while isinstance(n, ast.Attribute):
                    n = n.value  # type: ignore[assignment]
                if isinstance(n, ast.Name):
                    all_names_used.add(n.id)

    # Check definitions
    for py in all_files:
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_") and not name.startswith("__"):
                    # Private functions — lower bar
                    continue
                if name.startswith("__") and name.endswith("__"):
                    continue
                if name not in all_names_used:
                    report.add(
                        "DEAD CODE — Unused Functions/Classes",
                        "INFO",
                        str(py),
                        node.lineno,
                        f"Function '{name}' is never referenced elsewhere",
                        f"def {name}(...)"
                        if isinstance(node, ast.FunctionDef)
                        else f"async def {name}(...)",
                    )
            elif isinstance(node, ast.ClassDef):
                name = node.name
                if name.startswith("_"):
                    continue
                if name not in all_names_used:
                    report.add(
                        "DEAD CODE — Unused Functions/Classes",
                        "INFO",
                        str(py),
                        node.lineno,
                        f"Class '{name}' is never referenced elsewhere",
                        f"class {name}:",
                    )


# ══════════════════════════════════════════════════════════════════
# 5. UNSAFE PATTERNS — BARE EXCEPT
# ══════════════════════════════════════════════════════════════════
def audit_bare_except(report: AuditReport) -> None:
    """Find bare 'except:' clauses (no exception type specified)."""
    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                report.add(
                    "UNSAFE PATTERNS — Bare Except",
                    "CRITICAL",
                    str(py),
                    node.lineno,
                    "Bare except: clause catches all exceptions including SystemExit/KeyboardInterrupt",
                    "except:",
                )


# ══════════════════════════════════════════════════════════════════
# 6. UNSAFE PATTERNS — PRINT STATEMENTS
# ══════════════════════════════════════════════════════════════════
def audit_print_statements(report: AuditReport) -> None:
    """Find print() calls in app/ code — should use logging instead."""
    # Allow prints in main.py (CLI output) and test files
    allowed_files = {"main.py"}

    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        rel = py.relative_to(PROJECT_ROOT).as_posix()
        if any(rel.endswith(a) for a in allowed_files):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    report.add(
                        "UNSAFE PATTERNS — Print Statements",
                        "WARNING",
                        str(py),
                        node.lineno,
                        "Use logging instead of print() in application code",
                    )


# ══════════════════════════════════════════════════════════════════
# 7. UNSAFE PATTERNS — MUTABLE DEFAULT ARGUMENTS
# ══════════════════════════════════════════════════════════════════
MUTABLE_DEFAULTS = (ast.List, ast.Dict, ast.Set)


def audit_mutable_defaults(report: AuditReport) -> None:
    """Find function signatures with mutable default arguments (list, dict, set)."""
    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = node.args.defaults
                # defaults are right-aligned; skip **kwargs
                for default in defaults:
                    if isinstance(default, MUTABLE_DEFAULTS):
                        report.add(
                            "UNSAFE PATTERNS — Mutable Defaults",
                            "CRITICAL",
                            str(py),
                            node.lineno,
                            f"Mutable default argument in '{node.name}()' — use None and initialize inside",
                        )


# ══════════════════════════════════════════════════════════════════
# 8. UNSAFE PATTERNS — GLOBAL STATE MUTATIONS
# ══════════════════════════════════════════════════════════════════
def audit_global_state(report: AuditReport) -> None:
    """Find module-level mutable assignments outside of DI container patterns."""
    # Heuristic: look for module-level assignments that look like singletons or registries
    # outside of __init__.py and known container/registry files.

    for py in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        rel = py.relative_to(PROJECT_ROOT).as_posix()
        # Skip container/registry infrastructure — these are supposed to manage global state
        if any(x in rel for x in ["container", "registry", "__init__", "bootstrap"]):
            continue

        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = None
                    if isinstance(target, ast.Name):
                        name = target.id
                    if name and name.isupper():
                        # Module-level CONSTANT or singleton
                        if isinstance(
                            node.value, (ast.List, ast.Dict, ast.Set, ast.Call)
                        ):
                            report.add(
                                "UNSAFE PATTERNS — Global State",
                                "WARNING",
                                str(py),
                                node.lineno,
                                f"Module-level mutable global '{name}' — prefer DI container",
                            )


# ══════════════════════════════════════════════════════════════════
# 9. DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════════
def audit_dependencies(report: AuditReport) -> None:
    """Compare requirements.txt against actual imports in app/."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        report.add(
            "DEPENDENCY CHECK",
            "CRITICAL",
            str(req_file),
            0,
            "requirements.txt not found!",
        )
        return

    req_text = req_file.read_text(encoding="utf-8")
    # Parse package names from requirements.txt (strip version specifiers)
    pkg_to_pip: Dict[str, str] = {}
    for line in req_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Extract package name (before >=, <=, ==, etc.)
        match = re.match(r"^([a-zA-Z0-9_][a-zA-Z0-9._-]*)", line)
        if match:
            raw_name = match.group(1)
            # Normalize: pydantic-settings → pydantic_settings for import
            import_name = raw_name.replace("-", "_")
            pkg_to_pip[import_name] = raw_name

    # Collect all imports from app/ and main.py
    imported_modules: Set[str] = set()
    files_to_scan = list(APP_DIR.rglob("*.py")) + [PROJECT_ROOT / "main.py"]
    for py in files_to_scan:
        if "__pycache__" in str(py):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

    # Standard library modules (Python 3.9+ stdlib)
    stdlib = {
        "abc",
        "argparse",
        "asyncio",
        "ast",
        "base64",
        "bisect",
        "calendar",
        "collections",
        "concurrent",
        "configparser",
        "contextlib",
        "copy",
        "csv",
        "ctypes",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "email",
        "enum",
        "errno",
        "fcntl",
        "fnmatch",
        "fractions",
        "functools",
        "gc",
        "glob",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "imaplib",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "math",
        "mimetypes",
        "mmap",
        "multiprocessing",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "queue",
        "random",
        "re",
        "readline",
        "secrets",
        "select",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtplib",
        "socket",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "struct",
        "subprocess",
        "sys",
        "sysconfig",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "token",
        "traceback",
        "tracemalloc",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
        "zipimport",
        "__future__",
        "typing_extensions",
    }

    # Check for missing dependencies (imported but not in requirements)
    third_party_imported: Set[str] = set()
    for mod in imported_modules:
        if mod in stdlib or mod in ("app", "tests", "models"):
            continue
        third_party_imported.add(mod)

    missing = third_party_imported - set(pkg_to_pip.keys())
    unnecessary = set(pkg_to_pip.keys()) - third_party_imported

    report.stats["Packages in requirements.txt"] = len(pkg_to_pip)
    report.stats["Third-party imports found in code"] = len(third_party_imported)
    report.stats["Missing dependencies (imported but not in requirements.txt)"] = len(
        missing
    )
    report.stats["Unnecessary dependencies (in requirements.txt but not imported)"] = (
        len(unnecessary)
    )

    for mod in sorted(missing):
        report.add(
            "DEPENDENCY CHECK",
            "CRITICAL",
            str(req_file),
            0,
            f"Package '{mod}' is imported but NOT listed in requirements.txt",
        )
    for mod in sorted(unnecessary):
        report.add(
            "DEPENDENCY CHECK",
            "INFO",
            str(req_file),
            0,
            f"Package '{mod}' is in requirements.txt but never imported in app/",
        )


# ══════════════════════════════════════════════════════════════════
# 10. PERFORMANCE BENCHMARKS
# ══════════════════════════════════════════════════════════════════
def benchmark_performance(report: AuditReport) -> None:
    """Measure server startup time, agent latency, and memory usage."""
    results: Dict[str, str] = {}

    # ── 10a. Server startup time ──
    try:
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        t0 = time.perf_counter()
        sys.path.insert(0, str(PROJECT_ROOT))
        # Import the app factory
        spec = importlib.util.spec_from_file_location(
            "app.server.app", str(APP_DIR / "server" / "app.py")
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass
        t1 = time.perf_counter()
        mem_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        results["Server module import time"] = f"{(t1 - t0) * 1000:.1f} ms"
        results["Server module memory delta"] = (
            f"{(mem_after - mem_before) / 1024:.1f} KB"
        )
    except Exception as e:
        results["Server module import time"] = f"ERROR: {e}"
        tracemalloc.stop()

    # ── 10b. Agent execution (kernel bootstrap) ──
    try:
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        t0 = time.perf_counter()
        try:
            from app.core.container import DependencyContainer
            from app.core.lifecycle import LifecycleManager
            from app.kernel.bootstrap import bootstrap_kernel

            container = DependencyContainer()
            lifecycle = LifecycleManager()
            _ = bootstrap_kernel(container, lifecycle)
            t1 = time.perf_counter()
            mem_after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            results["Kernel bootstrap time"] = f"{(t1 - t0) * 1000:.1f} ms"
            results["Kernel bootstrap memory delta"] = (
                f"{(mem_after - mem_before) / 1024:.1f} KB"
            )
        except Exception as e:
            t1 = time.perf_counter()
            tracemalloc.stop()
            results["Kernel bootstrap time"] = f"ERROR: {e}"
    except Exception as e:
        results["Agent benchmark"] = f"ERROR: {e}"

    # ── 10c. Thread / task leak check ──
    try:
        import threading

        threads_before = threading.active_count()
        # Run a quick kernel bootstrap cycle
        try:
            from app.core.container import DependencyContainer
            from app.core.lifecycle import LifecycleManager
            from app.kernel.bootstrap import bootstrap_kernel

            container = DependencyContainer()
            lifecycle = LifecycleManager()
            bootstrap_kernel(container, lifecycle)
        except Exception:
            pass
        time.sleep(0.1)
        threads_after = threading.active_count()
        delta = threads_after - threads_before
        results["Thread count before"] = str(threads_before)
        results["Thread count after"] = str(threads_after)
        results["Thread leak delta"] = f"{delta} threads"
        if delta > 2:
            report.add(
                "PERFORMANCE",
                "WARNING",
                str(PROJECT_ROOT),
                0,
                f"Potential thread leak: {delta} threads created during bootstrap",
            )
    except Exception as e:
        results["Thread leak check"] = f"ERROR: {e}"

    report.performance.update(results)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def run_audit() -> AuditReport:
    report = AuditReport()

    print("Running TODO/FIXME audit...")
    audit_todo_fixme(report)

    print("Running unused imports audit...")
    audit_unused_imports(report)

    print("Running unreachable code audit...")
    audit_unreachable_code(report)

    print("Running unused definitions audit...")
    audit_unused_definitions(report)

    print("Running bare except audit...")
    audit_bare_except(report)

    print("Running print statement audit...")
    audit_print_statements(report)

    print("Running mutable default arguments audit...")
    audit_mutable_defaults(report)

    print("Running global state audit...")
    audit_global_state(report)

    print("Running dependency check...")
    audit_dependencies(report)

    print("Running performance benchmarks...")
    benchmark_performance(report)

    report.stats["Total findings"] = len(report.findings)
    report.stats["Critical findings"] = sum(
        1 for f in report.findings if f.severity == "CRITICAL"
    )
    report.stats["Warning findings"] = sum(
        1 for f in report.findings if f.severity == "WARNING"
    )
    report.stats["Info findings"] = sum(
        1 for f in report.findings if f.severity == "INFO"
    )

    return report


if __name__ == "__main__":
    report = run_audit()
    output = report.summary()
    print(output)

    # Also write to file
    out_path = PROJECT_ROOT / "tests" / "audit_report.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"\nReport also written to: {out_path}")
