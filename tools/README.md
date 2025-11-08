# Development Tools

This directory contains helper scripts for proof development workflow.

## Scripts

### `check_admitted.sh`

Count and track admitted proofs in the project.

```bash
# Basic usage - show count
./tools/check_admitted.sh

# Show detailed list of admitted proofs
./tools/check_admitted.sh --details

# Set custom limit
./tools/check_admitted.sh --max 10

# Fail if limit exceeded (useful in CI)
./tools/check_admitted.sh --max 15 --fail
```

**Output:**
```
📊 Admitted Proof Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total admitted proofs: 18
Maximum allowed: 15
Status: ⚠️  EXCEEDED LIMIT
```

### `proof_skeleton.sh`

Generate proof skeleton template for a specific lemma.

```bash
./tools/proof_skeleton.sh <file.v> <lemma_name>
```

**Example:**
```bash
./tools/proof_skeleton.sh src/Utils/InterpLemmas.v interpret_read_pure
```

**Output:**
```
📝 Generating proof skeleton for: interpret_read_pure

Lemma signature:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lemma interpret_read_pure :
  forall (s : interpreter_state) (var : string),
    s.(call_stack) <> [] ->
    exists v, interpret (trigger (inl1 (LocalRead var))) s ≈ Ret (s, v).

Status: ⚠️  Currently admitted

Suggested proof skeleton:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Proof.
  intros.
  unfold key_definition.
  ...
```

### `setup-hooks.sh`

Install git hooks for the project.

```bash
./tools/setup-hooks.sh
```

**What it does:**
- Copies `tools/pre-commit.template` to `.git/hooks/pre-commit`
- Makes the hook executable
- Shows what the hook will do

**When to use:** Once after cloning the repository

### `validate_pass.sh`

(Existing script) Validate MLIR pass definitions.

## Pre-commit Hook

The pre-commit hook automatically runs when you commit:

1. ✅ Builds all modified Coq files
2. ✅ Checks admitted proof count
3. ✅ Provides helpful error messages

**Setup:**
```bash
./tools/setup-hooks.sh
```

This installs the pre-commit hook from `tools/pre-commit.template` to `.git/hooks/pre-commit`.

**To bypass (not recommended):**
```bash
git commit --no-verify
```

## Workflow

### Recommended Development Flow

```bash
# 0. Setup git hooks (first time only)
./tools/setup-hooks.sh

# 1. Start watch mode in one terminal
dune build --watch

# 2. Edit files in another terminal
vim src/Utils/InterpLemmas.v

# 3. Check admitted count periodically
./tools/check_admitted.sh

# 4. Generate skeleton for new proof
./tools/proof_skeleton.sh src/Utils/InterpLemmas.v my_new_lemma

# 5. Commit (pre-commit hook runs automatically)
git add src/Utils/InterpLemmas.v
git commit -m "feat(proofs): add my_new_lemma"
```

### Common Issues and Solutions

**Issue:** Type error in lemma statement
```
Error: The term "var" has type "string" while it is expected to have type "Type"
```

**Solution:** Add explicit type applications
```coq
(* WRONG *)
trigger (inl1 (LocalWrite var value))

(* CORRECT *)
trigger (inl1 (@LocalWrite string mlir_value var value))
```

**Issue:** Missing import
```
Error: The reference interp_state was not found
```

**Solution:** Add required import
```coq
From ITree Require Import Events.State Events.StateFacts.
```

**Issue:** Name clash
```
Error: The term "eq" has type "arith_cmp_pred"
```

**Solution:** Use qualified name
```coq
eutt (@Coq.Init.Logic.eq R) t1 t2
```

## CI Integration

These tools are integrated into CI:

- `check_admitted.sh --max 15 --fail` runs on every PR
- Pre-commit hook enforces build success
- Admitted count is tracked over time

## See Also

- [CLAUDE.md](../.claude/CLAUDE.md#proof-development-workflow) - Proof development rules
- [Proof Development Guide](../docs/howto/proof-development.md) - (TODO)
