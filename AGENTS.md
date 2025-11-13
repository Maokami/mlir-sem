# MLIR-Sem Agent Guide (AGENTS.md)

This AGENTS file gives Codex CLI the same project context that Claude Code received via `.claude/*.md`. When in doubt, open the source docs (e.g., `.claude/CLAUDE.md`, `.claude/skills/*/SKILL.md`) for full detail.

## Project Purpose & Definition of Done
- Build an extensible Coq framework (ITrees-based) for MLIR dialect semantics, enabling translation validation, proofs, and extraction of a reference interpreter.
- Dependency direction is strict: `Syntax → Semantics → Pass/Extraction`.
- A feature is complete only when **spec + proofs**, **unit + property tests**, **docs/ADR**, and **CI + extraction smoke tests** all pass.
- No axioms unless explicitly documented; proofs must use structured skeletons (never blind admits).

## Repo & Coding Conventions
- Layout mirrors `.claude/CLAUDE.md` description (deps/, src/, test/, docs/, tools/, etc.). Dialect code lives under `src/Syntax/Dialect/*` and `src/Semantics/Dialect/*`.
- Coq style: one concept per file, modules named by purpose (e.g., `Dialect.Arith`, `Semantics.Core`). Keep reusable lemmas in `*_lemmas.v`. Manage hints locally.
- ITree semantics must document the equivalence used (`eutt`, `eqit`, etc.) and keep handlers modular.
- Extraction config lives in `src/Extraction/`. Every extraction change requires a smoke run of the generated interpreter.

## Build, Testing, and Proof Workflow
- Practice TDD: add/adjust tests alongside every code change. Core commands: `dune build`, `dune test -f`, `./tools/check_admitted.sh --details`, and `dune build @extract`.
- Build frequently (after 1–2 lemmas) or run `dune build --watch`. Use `tools/proof_skeleton.sh <file> <lemma>` to scaffold proofs before admits.
- Enforce admitted-proof hygiene: each admit needs a TODO comment plus context. Total admitted proofs must stay under the repo’s documented limit (warning at 20).
- Golden/oracle tests compare the extracted interpreter with MLIR tool output. Run the oracle suites in `test/oracle/` (e.g., via `dune test -f` or task-specific scripts) before merging semantics changes.
- When diagnosing toolchain diffs, validate MLIR input (`mlir-opt --verify-diagnostics`), run both MLIR and extracted interpreters, and diff the outputs. Document any intentional divergences in the test files.

## Collaboration Workflow
- All issues, PRs, commits, and comments must be in English. Every work item starts with a GitHub issue describing goal, rationale, design outline, tasks, and references.
- Pull requests must stay small, satisfy the Definition of Done, request GitHub Copilot review, and summarize implementation + testing. Reference the matching issue (e.g., `Fixes #123`).
- AI review policy: prefer **GitHub Copilot** and **Codex CLI**. Disable Gemini. AI bots review once per PR unless explicitly re-requested. Treat critical feedback (security/logic correctness) as blocking; track non-blocking items via issues.

## Task Playbooks (from `.claude/skills`)
### Add MLIR Dialect (`.claude/skills/add-dialect/SKILL.md`)
1. Gather dialect scope: operations, types, attributes, priority examples.
2. Define syntax in `src/Syntax/Dialect/<Dialect>` (types, ops, attrs) following MLIR naming.
3. Define semantics: events + handlers + op interpreter in `src/Semantics/Dialect/`. Document equivalence and keep handlers composable.
4. Tests: unit (`test/unit/`), QuickChick properties (`test/property/`), and golden/oracle tests (`test/oracle/<dialect>/`).
5. Docs: create `docs/design/<dialect>-semantics.md`, add ADR if architectural choices were made, and update extraction config.
6. Verify CI (format, build, proofs, tests, extraction, docs) before merge.

### Verify Optimization Pass / Translation Validation (`.claude/skills/add-pass*/`)
1. Identify pass (e.g., SCCP, DCE) and target dialects; collect MLIR tests demonstrating its transformations.
2. Run `mlir-opt --<pass>` on each test to capture before/after MLIR files.
3. Parse both versions via the OCaml driver, export Coq ASTs, and place equivalence proofs under `src/TranslationValidation/<Pass>.v`.
4. Prove semantic equivalence using itree tactics (`prog_equiv` / `eutt`). Each test case should become a lemma; compose results into the main theorem (`pass_preserves_semantics`).
5. Integrate with golden tests plus docs/ADR updates describing verification scope and assumptions (e.g., trusting mlir-opt output as `is_pass_output`).

### Run Golden Tests (`.claude/skills/run-golden-test/SKILL.md`)
1. Check prerequisites: `mlir-opt`, `mlir-translate`, OCaml/dune toolchain.
2. Build Coq project and extracted interpreter (`dune build`, `dune build @extract`, `dune build driver/mlir_interp.exe`).
3. Choose test scope (`test/oracle/<dialect>/*.mlir` or specific files). For each test: validate with MLIR, produce expected output via MLIR toolchain, run `_build/default/driver/mlir_interp.exe`, and diff results.
4. Record metadata (dialect, features, known issues) at the top of each `.mlir` file. Any regression must add/update tests documenting the fix.

### Create ADR (`.claude/skills/create-adr/SKILL.md`)
1. Collect title, context, decision, alternatives, consequences, and authors.
2. Determine the next ADR number by scanning `docs/adr/` (`ADR-XXXX-title.md`).
3. Use the provided template (status/date/authors + Context, Decision, Alternatives, Consequences, Implementation Notes, References, Revision History).
4. Fairly evaluate alternatives (pros/cons/why not chosen) and connect to related ADRs, design docs, or issues.
5. Update ADR status as decisions evolve (Proposed → Accepted → Deprecated/Superseded). Link implementation issues in the References section.

## References & Tooling
- Workflow improvements, admitted-proof scripts, and hook setup live under `tools/` (see `tools/README.md`).
- Detailed proof workflow, testing strategy, and policy rationale remain in `.claude/CLAUDE.md`; refer to it when deeper context is needed.
- Keep `.claude/` intact for Anthropics workflows; Codex should treat this AGENTS.md as the canonical entry point and cross-link to `.claude` files when more detail is required.
