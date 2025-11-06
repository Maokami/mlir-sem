# MLIR-Sem: Formal Semantics Framework for MLIR

## Project Overview

This project builds an **extensible and compositional framework in Coq for defining the formal semantics of MLIR dialects**, leveraging **ITrees** (Interaction Trees). The framework enables:

- Formal verification of MLIR optimizations
- Extraction of reference interpreters
- Modular dialect semantics composition (mirroring MLIR's design philosophy)

**Goal**: Make MLIR dialect semantics explicit and formal (currently implicit in syntax/lowering passes). Inspired by Vellvm for LLVM IR.

## Core Principles

1. **Single source of truth in Coq** - syntax, types, and semantics defined once
2. **Small core + compositional extensions** - dialects, events, and handlers are modular
3. **TDD and property-based testing** - tests guide development
4. **Functional programming preferred** - leverage Coq's strengths
5. **Formalization first, extraction second** - correctness before performance
6. **No axioms** - unless explicitly documented and justified
7. **Living documentation** - docs and proofs evolve with code

## Repository Structure

```
mlir-sem/
├── deps/           # External dependencies (vellvm, etc.)
├── src/
│   ├── Syntax/     # MLIR syntax, types, dialect declarations
│   ├── Semantics/  # ITree-based semantics definitions
│   ├── Pass/       # Optimization passes (TODO)
│   ├── Theory/     # Theorems and proofs (TODO)
│   ├── Extraction/ # Coq → OCaml extraction configuration
│   └── Utils/      # Common tactics, lemmas, utilities
├── test/
│   ├── unit/       # Unit tests
│   ├── property/   # QuickChick property-based tests
│   └── oracle/     # Golden tests vs MLIR toolchain
├── tools/          # Scripts, converters, build utilities
├── docs/
│   ├── adr/        # Architectural Decision Records
│   ├── design/     # Module design details
│   └── howto/      # Contribution guides
└── CI/             # CI workflows and scripts
```

**Dependency Direction**: `Syntax → Semantics → Passes`

## Coding Style

### Coq Conventions

- **Module naming**: By purpose (`Dialect.X`, `Semantics.Core`)
- **File organization**: One core concept per file when possible
- **Common tactics**: Store in `Utils/Tactics.v`, avoid duplication
- **Library preference**: Use standard library; justify MathComp/ssreflect usage
- **Proof naming**: Descriptive (e.g., `arith_add_commutes`)
- **Hint management**: Use local scopes, avoid global hint pollution
- **Lemma files**: Separate reusable lemmas into `*_lemmas.v`

### ITree Guidelines

- **Observation equivalence**: Clearly document which equivalence is used (`eutt`, `eqit`, etc.) in each proof file
- **Handler modularity**: Keep handlers modular and replaceable
- **Effect composition**: Design effects to compose cleanly

### Extraction

- **Target**: Reference interpreter + CLI wrapper
- **Validation**: Compare against MLIR tools in golden tests
- **Smoke tests**: Every extraction must be executable

## Definition of Done

A feature is complete ONLY if it includes:

1. ✓ **Specification and core definitions** in Coq
2. ✓ **Unit tests** and QuickChick properties
3. ✓ **Documentation update** (design notes + ADR if architectural)
4. ✓ **Passing CI checks** and extraction smoke test

**No exceptions.** Incomplete features should not be merged.

## Testing Strategy

- **TDD approach**: Write unit tests first
- **Property testing**: Use QuickChick for invariants and roundtrip properties
- **Golden tests**: Compare extracted interpreter output with MLIR execution
- **CI seed storage**: Store failing seeds and inputs for reproducibility
- **Goal**: Port every relevant test from the official MLIR repository
- **Command**: `dune test -f` (force re-run all tests)

## Commit Policy

Follow conventional commit style:

```
type(scope): summary

Body: what changed and why
Refs: ADR links / issue numbers

Types: feat, fix, proof, refactor, docs, test, chore, build

Example:
proof(Semantics): prove eutt equivalence for cf.cond_br

Added commutativity lemma and refactored proof using
itree_rewrite tactics. Refs: ADR-0001, #42
```

**Guidelines**:
- Small, meaningful commits
- Clear what + why in body
- Link to relevant issues/ADRs

## Issue and PR Workflow

**Language**: All communication (issues, PRs, commits, comments) must be in **English**.

### Creating Issues

Every meaningful work unit needs a GitHub issue first:

- **Goal**: High-level objective (feature, bug fix, etc.)
- **Rationale**: Why the change is needed
- **Plan/Design**: Brief overview of proposed solution
- **Tasks**: Actionable checklist
- **Links**: Relevant ADRs or design docs

### Pull Request Requirements

Submit contributions via **Pull Requests only**:

- **Summary**: Concise description of changes
- **Implementation Details**: What and how
- **Testing**: How changes were verified
- **Related Issue**: Link to issue (e.g., `Fixes #123`)

**Process**:
1. Keep PR scope small and focused
2. Ensure Definition of Done is satisfied
3. Request **GitHub Copilot** review for every PR
4. Use AI review as first pass, refine manually for rigor
5. Merge only after CI passes and feedback is addressed

## Checklist for New Dialect or Pass

- [ ] Syntax, semantics, and effects defined
- [ ] Unit + property tests written
- [ ] Golden oracle tests updated
- [ ] ADR + design doc created
- [ ] Extraction config updated
- [ ] CI passing

## Reference

Study `deps/vellvm` structure and document differences in `docs/design/vellvm-comparison.md`.

## CI Policy

- **Matrix builds**: Test across Coq and OCaml versions
- **Steps**:
  1. Format check
  2. Build and proof verification
  3. Unit + property + golden tests
  4. Extraction & smoke run
  5. Docs and diagrams validation

Mark slow tactics separately in CI for efficient testing.

## GitHub Project Board

- Maintain a GitHub Project board to track progress
- **Columns**: Backlog → Spec/Design → In Progress → Review → Done
- **Tagging**: By category (Dialect, Semantics, Extraction, Proof, Docs, CI)
- **Milestones**:
  - Core semantics MVP
  - First extracted reference interpreter
  - First verified optimization pass
- Keep it lightweight for coordination and visibility
