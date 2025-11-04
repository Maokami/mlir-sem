# Project Development Guide

This project aims to build an **extensible and compositional denotational semantics framework for MLIR in Coq**, leveraging **ITrees** and supporting **formal verification of MLIR optimizations** and **reference interpreter extraction**, similar in spirit to Vellvm for LLVM IR.

## Core Principles

- Single source of truth in Coq (syntax, types, semantics)
- Small core + compositional extensions (dialects, events, handlers)
- TDD and property-based testing culture
- functional programming preferred
- Formalization first, extraction second
- No axioms unless explicitly documented and justified
- Documentation and proofs evolve with the code

## Definition of Done

A feature is complete only if it includes:

- Specification and core definitions in Coq
- Unit tests and QuickChick properties
- Documentation update (design notes + ADR)
- Passing CI checks and extraction smoke test

## Repository Structure

```
├─ deps/                  # Referenced projects (e.g., vellvm)
├─ driver/                # (TODO)
├─ src/
│  ├─ Syntax/             # MLIR 구문, 타입, 다이얼렉트 선언
│  ├─ Semantics/          # ITree-based semantics
│  ├─ Pass/               # (TODO)
│  ├─ Theory/             # (TODO)
│  ├─ Extraction/         # Coq → OCaml 추출 설정
│  └─ Utils/              # 공통 전개, 전술, 전형 정리
├─ test/
│  ├─ unit/               # 단위 테스트
│  ├─ property/           # QuickChick 속성
│  └─ oracle/             # MLIR 툴체인과의 대조(golden) 산출물
├─ tools/                 # 스크립트, 변환기, 빌드 유틸
├─ docs/
│  ├─ adr/                # Architectural Decision Records
│  ├─ design/             # 모듈 설계 상세
│  └─ howto/              # 기여 가이드, 패스 추가 방법
├─ CI/                    # CI 워크플로우와 스크립트
├─ dune / _CoqProject     # 빌드 설정 (dune 권장)
└─ opam                   # 재현 가능한 환경
```


## Style and Coding Rules

### Coq Style

- Modules named by purpose (`Dialect.X`, `Semantics.Core`)
- One core concept per file if possible
- Dependency direction: `Syntax → Semantics → Passes`
- Common tactics in `Utils/Tactics.v`
- Prefer standard library; justify MathComp/ssreflect usage
- Document equivalence used (`eutt`, `eqit`, etc.)

### ITree Guidelines

- Clear observation equivalence in each proof file
- Handlers modular and replaceable

### Extraction

- Extract reference interpreter and CLI wrapper
- Compare against MLIR tools for golden tests

## Testing Strategy

- Unit tests first (TDD)
- QuickChick properties for invariants
- Golden tests comparing extracted interpreter with MLIR execution
- CI stores failing seeds and inputs
- Goal : port every testset as possible from the official MLIR repo

## Documentation Rules

- ADR format: `ADR-XXXX-title.md`
- Each dialect, effect, pass documented with examples
- Mermaid diagrams recommended
- English comments; Korean allowed for intuition remarks

## CI Policy

- Matrix builds across Coq and OCaml versions
- Steps:
  1. Format check
  2. Build and proof
  3. Unit + property + golden tests
  4. Extraction & smoke run
  5. Docs and diagrams validation

## Commit Policy

- Small, meaningful commits
- Follow conventional style:
```
type(scope): summary
Body: what + why
Refs: ADR links / issues
Types: `feat, fix, proof, refactor, docs, test, chore, build`

Example:  
`proof(Semantics): prove eutt equivalence for IfOp`
```

## Proof Engineering Notes

- Names like `arith_add_commutes`
- Local hint scopes; avoid global pollution
- Separate reusable lemmas into `*_lemmas.v`
- Mark slow tactics and separate in CI

## Checklist for New Dialect or Pass

- [ ] Syntax, semantics, effects
- [ ] Unit + property tests
- [ ] Golden oracle updates
- [ ] ADR + design doc
- [ ] Extraction config updated
- [ ] CI passing

## Reference

- Study `deps/vellvm` structure and document differences in  
  `docs/design/vellvm-comparison.md`

## Issue and PR Workflow
- Create a **GitHub issue** for each meaningful work unit before starting development. A good issue should include:
  - **Goal:** The high-level objective (e.g., new feature, bug fix).
  - **Rationale:** Why the change is needed.
  - **Plan/Design:** A brief overview of the proposed solution.
  - **Tasks:** A checklist of actionable tasks.
  - **Links:** References to relevant ADRs or design docs.

- Submit contributions via **Pull Requests only**, tied to the relevant issue. A good PR should include:
  - **Summary:** A concise summary of the changes.
  - **Implementation Details:** What was implemented and how.
  - **Testing:** How the changes were verified.
  - **Related Issue:** A link to the issue that this PR resolves (e.g., `Fixes #123`).

- Keep PR scope small. Each PR must satisfy the **Definition of Done**.
- Request an automated review from **GitHub Copilot** for every PR.
- Use the review to improve clarity, safety, and adherence to project conventions.
- Treat AI review as a **first pass**, and refine manually for rigor.
- Merge only after CI passes and review feedback is addressed.

## GitHub Project Board
- Maintain a GitHub Project board to track planning, design, implementation, and verification progress.
- Use columns such as:
  - Backlog → Spec/Design → In Progress → Review → Done
- Link Issues and PRs to the board so progress is automatically reflected.
- Tag work items by category (e.g., Dialect, Semantics, Extraction, Proof, Docs, CI).
- Use Milestones to track major goals such as:
  - Core semantics MVP
  - First extracted reference interpreter
  - First verified optimization pass
- Keep the board lightweight; treat it as a coordination and visibility tool, not bureaucracy.
