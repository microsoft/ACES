# This PR contains

## Description

## Checklist

- [ ] Are you adding a new domain?
  - [ ] If yes, does it include tests in `domains/<domain>/tests/`?
  - [ ] Does it include an `eval.yaml` with metadata?

- [ ] Are you modifying the framework (`src/saber/`)?
  - [ ] If yes, do all framework tests pass? (`uv run pytest tests/`)
  - [ ] Are there new tests for the changed functionality?

- [ ] Are you modifying an existing domain?
  - [ ] If yes, do all existing tests still pass?
  - [ ] If task behavior changed, has the task YAML been updated?

- [ ] Does this change affect scoring?
  - [ ] If yes, have scoring changes been validated against known-good results?

- [ ] Does this change affect Docker images?
  - [ ] If yes, have images been rebuilt and tested?

- [ ] Code quality
  - [ ] Lint passes (`uv run ruff check`)
  - [ ] Type hints on all public functions
  - [ ] No `Any` types in public APIs
  - [ ] Tests cover new/changed functionality
