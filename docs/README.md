# Development Guidelines

This document defines the coding conventions for implementation, refactoring, and
testing in this repository. Apply them together with task-specific instructions unless
explicitly directed otherwise.

## Implementation Guidelines

- Treat the current HEAD and working tree as the primary sources of truth.
- Reuse existing services, helpers, records, and repositories whenever possible.
- Do not duplicate the same decision logic in multiple places.
- Keep distinct responsibilities, such as numerical computation, analysis, and plotting,
  separate.
- Avoid cleanup, large-scale refactoring, or renaming that is unnecessary for the task.
- Do not add new dependencies to legacy paths scheduled for removal.
- For changes that affect research results, verify not only the test results but also that
  the physical and numerical meaning and computational conditions remain unchanged.

## Python Style

- Do not use future annotations such as `from __future__ import annotations`.
- Add a docstring to every function.
- Prefer modern Python syntax for type hints.
  - Use `A | B` instead of `typing.Union`.
  - Use `T | None` instead of `typing.Optional`.
  - Use `list[T]` instead of `typing.List`.
  - Use `dict[K, V]` instead of `typing.Dict`.
  - Prefer built-in generics and PEP 604 syntax in other cases as well.

## Comments and Docstrings

- Generate and write all code comments in English.
- Write exception messages in English.
- Add a short one-line comment to a long operation when needed to clarify the intent of
  the processing block.
- Do not add comments that merely restate obvious operations line by line.
- Use comments when needed to record design intent, assumptions, constraints, or why an
  operation is necessary.
- Match the comment density of the surrounding code and avoid unnecessary comments.
- Keep docstring style consistent with the surrounding code.
- For public fields, place an attribute docstring immediately after each field declaration.

## Commit Messages

- Generate and write all commit messages in English.
- Keep commit messages concise and describe the purpose of the change.

## Testing

- Give each test a docstring that explains what it verifies.
- Run only the relevant pytest tests during development.
- Run the full pytest suite once the implementation is stable and nearly ready to commit.
- Do not change scientific or numerical specifications solely to make tests pass.
- Do not run production workloads or expensive numerical computations unless explicitly
  requested.

## Completion Checklist

- [ ] All new or modified comments are written in English.
- [ ] The commit message is written in English.
- [ ] All new or modified exception messages are written in English.
- [ ] Every new or modified function has a docstring.
- [ ] Every new or modified public field has an attribute docstring immediately after it.
- [ ] No future annotations were added.
- [ ] No new uses of `typing.Union`, `Optional`, `List`, or `Dict` were added.
- [ ] No logic was duplicated.
- [ ] No out-of-scope changes were made.
- [ ] Relevant pytest tests pass.
- [ ] The full pytest suite was run at the final stage.
- [ ] No production workload was run without explicit instructions.
