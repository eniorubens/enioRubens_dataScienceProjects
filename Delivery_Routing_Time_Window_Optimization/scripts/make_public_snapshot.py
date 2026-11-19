"""Build a public snapshot of the repository without third-party files.

The private repository (and its history) is left untouched. The snapshot is a fresh folder, optionally
with a new one-commit git history::

    python scripts/make_public_snapshot.py <target-dir> [--ref HEAD | --from-worktree] [--git-init]
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Third-party files that stay in the private repository only (see ``scripts/fetch_data.py``).
EXCLUDED = (
    "data/raw/sintef/solomon-100.zip",
    "data/raw/cvrplib_solomon_100/Solomon.7z",
    "data/raw/cvrplib_solomon_100/Solomon-solutions.7z",
    "docs/Solomon-html descriptions.pdf",
    "data/reference/cvrplib_bks_100.csv",
)

GITIGNORE_HEADER = "# Third-party data: not redistributed here; obtain it with `python scripts/fetch_data.py`."


def git(args: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    """Run git and return the completed process (bytes output)."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=check)


def is_excluded(relpath: str) -> bool:
    """Return True if a repository-relative path is one of the excluded third-party files."""
    return relpath.replace("\\", "/") in EXCLUDED


def check_target(target: Path, private_root: Path) -> Path:
    """Validate the target folder: absent or empty, and outside the private repository."""
    target = target.resolve()
    if target == private_root or private_root in target.parents or target in private_root.parents:
        raise SystemExit(f"target {target} must not be inside (or contain) the private repository {private_root}")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise SystemExit(f"target {target} exists and is not an empty directory")
    return target


def extract_archive(ref: str, target: Path, root: Path = ROOT) -> None:
    """Extract ``git archive <ref>`` into ``target``."""
    data = git(["archive", "--format=tar", ref], cwd=root).stdout
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(target, filter="data")


def copy_worktree(target: Path, root: Path = ROOT) -> None:
    """Copy tracked and new non-ignored files of the working tree into ``target``."""
    listing = git(["ls-files", "-co", "--exclude-standard", "-z"], cwd=root).stdout
    target.mkdir(parents=True, exist_ok=True)
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        source = root / rel
        if not source.is_file():  # tracked but deleted in the working tree
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def remove_excluded(target: Path) -> list[str]:
    """Delete the excluded files from ``target``; return the ones that were present."""
    removed = []
    for rel in EXCLUDED:
        path = target / rel
        if path.exists():
            path.unlink()
            removed.append(rel)
    return removed


def write_gitignore(target: Path) -> None:
    """Append the excluded paths (and the extraction folders) to the snapshot's ``.gitignore``."""
    path = target / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current and not current.endswith("\n"):
        current += "\n"
    block = [GITIGNORE_HEADER, *(f"/{rel}" for rel in EXCLUDED), ""]
    path.write_text(current + "\n" + "\n".join(block), encoding="utf-8", newline="\n")


def assert_clean(target: Path) -> None:
    """Fail if any excluded file exists in ``target``."""
    present = [rel for rel in EXCLUDED if (target / rel).exists()]
    if present:
        raise SystemExit(f"excluded files still present in the snapshot: {present}")


def tree_summary(target: Path) -> tuple[int, int]:
    """Return ``(file count, total bytes)`` of ``target`` ignoring ``.git``."""
    files = [p for p in target.rglob("*") if p.is_file() and ".git" not in p.relative_to(target).parts[:1]]
    return len(files), sum(p.stat().st_size for p in files)


def init_repo(target: Path, label: str, user: tuple[str, str]) -> None:
    """Initialise a one-commit repository in ``target`` using the private repository's git user."""
    name, email = user
    git(["init", "-b", "main"], cwd=target)
    git(["add", "-A"], cwd=target)
    message = f"Snapshot público de {label} (sem arquivos de terceiros)"
    git(["-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", message], cwd=target)


def configured_user(root: Path = ROOT) -> tuple[str, str]:
    """Return the git user name and email configured for the private repository."""
    def get(key: str) -> str:
        return git(["config", key], cwd=root, check=False).stdout.decode("utf-8").strip()

    return get("user.name"), get("user.email")


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="snapshot folder (must not exist or be empty; outside this repo)")
    parser.add_argument("--ref", default="HEAD", help="git ref to archive (default HEAD)")
    parser.add_argument(
        "--from-worktree", action="store_true", help="copy tracked + new non-ignored files of the working tree"
    )
    parser.add_argument("--git-init", action="store_true", help="git init -b main and make one commit")
    args = parser.parse_args(argv)

    private_root = Path(git(["rev-parse", "--show-toplevel"]).stdout.decode().strip()).resolve()
    target = check_target(args.target, private_root)
    short = git(["rev-parse", "--short", "HEAD" if args.from_worktree else args.ref]).stdout.decode().strip()

    if args.from_worktree:
        copy_worktree(target, private_root)
        label = f"{short}+worktree"
    else:
        extract_archive(args.ref, target, private_root)
        label = short
    removed = remove_excluded(target)
    write_gitignore(target)
    assert_clean(target)
    if args.git_init:
        init_repo(target, label, configured_user(private_root))

    count, size = tree_summary(target)
    print(f"snapshot: {target}\nsource: {'working tree' if args.from_worktree else args.ref} ({label})")
    print(f"files: {count}  size: {size / 1e6:.2f} MB  git history: {'1 commit' if args.git_init else 'none'}")
    print("excluded (removed):" if removed else "excluded: none were present")
    for rel in removed:
        print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
