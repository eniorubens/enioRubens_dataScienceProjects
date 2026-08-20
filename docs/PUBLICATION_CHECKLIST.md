# Publication review checklist

This file defines the final human review gate. It does not authorize a remote,
a commit, or publication.

## Repository boundary

- [ ] Create a standalone repository whose root is this project directory.
- [ ] Do not add a GitHub remote to the current parent repository at
  `D:\Cursos`.
- [ ] Confirm that no files from sibling projects appear in the first commit.

## Scientific claims

- [ ] Preserve `S8: NOT_CONFIRMED (2/3)` and `V2.1-C: NOT_CONFIRMED (3/4)`.
- [ ] Preserve `deployment_authorized: false` and `deploy=false`.
- [ ] Do not reopen `test`, `stress`, or the sealed `monitor` partition.
- [ ] Treat bootstrap intervals as diagnostic, not as replacement gates.
- [ ] Do not claim row-level D1 pool identity; D1 has no persisted signature.

## Public files

- [ ] Include aggregate JSON evidence under `temp/` and both frozen `.joblib`
  bundles under `artifacts/`.
- [ ] Exclude all raw data, Parquet files, narratives, identifiers, caches,
  logs, Kaggle credentials, and local environments.
- [ ] Verify every pinned SHA-256 after a clean clone on Windows and Linux.
- [ ] Load `.joblib` files only from this trusted repository and verify hashes
  before deserialization.

## Documentation and notebooks

- [ ] Review both `README.md` and `README.pt-BR.md` for editorial parity.
- [ ] Render every notebook in GitHub's preview and inspect outputs.
- [ ] Confirm that no notebook output contains complaint narratives, local
  absolute paths, warning traces, or stale pre-result messages.
- [ ] Review `docs/DATA_PROVENANCE.md` and all ADR amendments.

## Validation

- [ ] Run `python -m unittest discover -s tests -t . -v` in the worktree so
  each test reports progress.
- [ ] Run the same command in a clean standalone clone containing only public
  files.
- [ ] Record skipped tests that require the undistributed raw/index Parquet.
- [ ] Review `git status`, `git diff --check`, and the complete staged file list.

Publication remains a manual decision after all boxes are reviewed.
