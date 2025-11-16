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
- All issues, PRs, commits, and comments must be in English.
- Every work item starts with a GitHub issue covering goal, rationale, design outline, tasks, and references.
- Pull requests stay small, satisfy the Definition of Done, summarize implementation + testing, and reference the matching issue (e.g., `Fixes #123`) while requesting GitHub Copilot review.
- Critical AI or human feedback on security/logic/blocking items must be resolved before merge; track non-blocking items via follow-up issues.

### Commit Policy
- Use **conventional commits**: ``type(scope): summary`` with the summary in present tense and <72 chars.
- Allowed types: `feat`, `fix`, `proof`, `refactor`, `docs`, `test`, `chore`, `build`; omit `scope` only when it would be redundant.
- Commit body must explain **what changed** and **why**, plus references (ADRs, issues) under a `Refs:` line.
- Keep commits small and meaningful so that diff + message tell a coherent story.
- Example:
  ```
  proof(Semantics): prove eutt equivalence for cf.cond_br

  Added commutativity lemma and refactored proof using itree_rewrite tactics.
  Refs: ADR-0001, #42
  ```

### Issue & PR Workflow
- **Issues**: document goal, rationale, plan/design sketch, task checklist, and links to ADRs/docs before coding starts.
- **PRs** must include: concise summary, implementation details (highlighting key files/decisions), explicit testing section (commands or suites), and a reference to the driving issue (`Fixes #123`).
- Process for each PR:
  1. Keep scope tight; split work rather than shipping mega-PRs.
  2. Ensure Definition of Done (spec+proofs, tests, docs/ADR, CI+extraction smoke) is met.
  3. Request required AI reviews, respond to all critical feedback, and rerun CI if needed.
  4. Merge only after CI is green and reviewers approve. Update ADRs/docs if implementation diverges from approved plan.

### AI Review Policy
- Enable **GitHub Copilot** and **Codex CLI** as the default AI reviewers; disable Gemini to avoid noisy refactor churn (Repo Settings → Code security and analysis → Disable Gemini Code Assist).
- Limit AI reviewers to **one pass per PR** (draft or ready). Do not request re-reviews unless feedback specifically asks for it.
- Document in the PR description how critical vs. non-critical feedback was handled; file follow-up issues for any accepted non-blocking comments.

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
1. **Prereqs**: Ensure `mlir-opt`, `mlir-translate`, `dune`, and `ocaml` are on PATH (`which mlir-opt`, etc.); help user install missing tools before proceeding.
2. **Build**: Run `dune build`, `dune build @extract`, and `dune build driver/mlir_interp.exe`. If extraction fails, inspect `src/Extraction/` config, unresolved admits, or non-computational defs that block extraction.
3. **Select Scope & Annotate Tests**: Decide whether to run all `test/oracle/**/*.mlir`, a dialect subset, or a single file. Each `.mlir` must start with metadata covering dialect, features, and known issues.
4. **Produce Expected Output**: For each test, validate the MLIR input (`mlir-opt --verify-diagnostics <file>`), then run the relevant MLIR tool (`mlir-opt`, `mlir-cpu-runner`, pass pipeline, etc.) to capture the expected output.
5. **Run Extracted Interpreter**: Execute `_build/default/driver/mlir_interp.exe <file>` (or the dune test driver) to obtain the actual output.
6. **Compare & Document Strategy**: Diff expected vs. actual (plain `diff`, `dune test`, or custom script). State whether the comparison is exact, semantically equivalent, or approximate (e.g., floats) within the test file.
7. **Triage Differences**: Confirm MLIR behavior, minimize failing MLIR input, inspect Coq semantics and extracted OCaml, and rerun builds. Check for formatting differences (normalize outputs) before assuming semantic divergence.
8. **CI & Artifacts**: Commit expected outputs, store failing diffs/logs for debugging, and integrate `dune build @extract` + `dune test` into CI. Re-run full suite after MLIR upgrades and update docs/ADRs describing intentional divergences.

### Create ADR (`.claude/skills/create-adr/SKILL.md`)
1. **Gather Inputs**: Title, context, concrete decision statement, alternatives considered, consequences (positive/negative/neutral), authors, and related issues/docs.
2. **Assign Number**: List `docs/adr/ADR-*.md`, pick the next zero-padded number, and choose a descriptive slug (e.g., `ADR-0004-handler-architecture.md`).
3. **Template**: Each ADR starts with Status (Proposed/Accepted/Deprecated/Superseded), Date, and Authors, followed by sections for Context, Decision (“We will …”), Alternatives (each with pros/cons and why not chosen), Consequences (positive/negative/neutral lists), Implementation Notes, References (issues, ADRs, design docs), and a Revision History table.
4. **Populate & Link**: Fill every section using the gathered info, reference related ADRs/docs/code paths (e.g., `src/Semantics/...`), and link implementation issues in References. Keep evaluation balanced—“do nothing” counts as an alternative.
5. **Maintain Lifecycle**: Create ADRs for architectural/proof/extraction/test strategy decisions (not minor fixes), update status as decisions evolve, and ensure resulting implementation work references the ADR. Specific, trade-off-aware ADRs become living documentation for future contributors.

## References & Tooling
- Workflow improvements, admitted-proof scripts, and hook setup live under `tools/` (see `tools/README.md`).
- Detailed proof workflow, testing strategy, and policy rationale remain in `.claude/CLAUDE.md`; refer to it when deeper context is needed.
- Keep `.claude/` intact for Anthropics workflows; Codex should treat this AGENTS.md as the canonical entry point and cross-link to `.claude` files when more detail is required.
