# Oracle Testing for Optimization Passes

## Overview

This document describes **oracle testing** (also called golden testing) for MLIR optimization passes. Oracle testing compares the execution outputs of programs before and after optimization to detect semantic differences.

**Important**: This is NOT formal translation validation, which requires proving semantic equivalence in Coq (see [Future Work](#future-work-true-translation-validation) below). Oracle testing is a pragmatic first step that helps catch bugs, but does not provide formal guarantees.

## How It Works

Oracle tests compare the execution outputs of MLIR programs before and after optimization:

1. **Original program**: An unoptimized MLIR file (e.g., `test.mlir`)
2. **Optimized program**: Either a pre-generated optimized file (e.g., `test.opt.mlir`) or dynamically generated using `mlir-opt`
3. **Execution**: Both programs are executed through our extracted OCaml interpreter
4. **Comparison**: If outputs match, the test passes

**Limitations**: This only tests specific inputs, not all possible executions. A passing test does not prove correctness.

## Test Infrastructure

### Components

- **test_driver.ml**: Contains test suite with `make_translation_validation_test` helper (naming will be updated)
- **mlir-opt**: MLIR optimization tool (from LLVM project) that applies passes
- **Extracted interpreter**: OCaml interpreter extracted from Coq semantics

### Environment Variables

- `MLIR_OPT_PATH`: Path to mlir-opt binary (defaults to `mlir-opt` from PATH)
- `RUN_EXE_PATH`: Path to extracted interpreter (set by dune)

## Writing Oracle Tests

### Option 1: Pre-generated Optimized File

Use this when you want to commit the optimized file to the repository:

```ocaml
make_translation_validation_test
  ~name:"SCCP constant propagation"
  ~mlir_file:"sccp_addi.mlir"
  ~opt_mlir_file:(Some "sccp_addi.opt.mlir")
  ~pass_pipeline:"builtin.module(func.func(sccp))"
```

**Steps:**
1. Create the original MLIR file: `test/sccp_addi.mlir`
2. Generate optimized version:
   ```bash
   mlir-opt test/sccp_addi.mlir -pass-pipeline='builtin.module(func.func(sccp))' -o test/sccp_addi.opt.mlir
   ```
3. Add both files to `test/dune` deps
4. Add test case to `test_driver.ml`

### Option 2: Dynamically Generated Optimization

Use this for quick tests or when the optimized file is large:

```ocaml
make_translation_validation_test
  ~name:"SCCP with addi (dynamic)"
  ~mlir_file:"sccp_addi.mlir"
  ~opt_mlir_file:None
  ~pass_pipeline:"builtin.module(func.func(sccp))"
```

The test will automatically run `mlir-opt` during execution.

## Example Test Cases

### Example 1: Constant Folding

**test/sccp_addi.mlir**:
```mlir
func.func @constant_prop_addi() -> i32 {
  %c1 = arith.constant 10 : i32
  %c2 = arith.constant 20 : i32
  %result = arith.addi %c1, %c2 : i32
  return %result : i32
}
```

**After SCCP optimization**:
```mlir
func.func @constant_prop_addi() -> i32 {
  %c30_i32 = arith.constant 30 : i32
  return %c30_i32 : i32
}
```

Both programs return `30`, validation passes ✓

### Example 2: Dead Branch Elimination

**test/sccp_branch.mlir**:
```mlir
func.func @constant_branch(%arg0: i32) -> i32 {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  cf.cond_br %true, ^bb1, ^bb2

^bb1:
  return %c1 : i32

^bb2:
  return %c2 : i32
}
```

SCCP determines that `%true` is always true, so `^bb2` is dead code. Both versions return `1` ✓

## Pass Pipeline Syntax

Pass pipelines use MLIR's textual pass pipeline syntax:

```
builtin.module(func.func(sccp))
```

This means:
- Apply to `builtin.module` operations
- Within modules, apply to `func.func` operations
- Run the `sccp` (Sparse Conditional Constant Propagation) pass

Other examples:
- `builtin.module(func.func(cse))` - Common Subexpression Elimination
- `builtin.module(func.func(canonicalize))` - Canonicalization
- `builtin.module(func.func(sccp,cse))` - Multiple passes in sequence

See [MLIR Pass Infrastructure docs](https://mlir.llvm.org/docs/PassManagement/) for more details.

## Supported Operations

Currently, our interpreter supports:
- `arith.constant` - Integer constants
- `arith.addi` - Addition
- `arith.cmpi` - Comparison
- `func.return` - Function return
- `cf.br` - Unconditional branch
- `cf.cond_br` - Conditional branch

**Note**: Tests using unsupported operations (e.g., `arith.select`) will fail during interpretation. Add semantics for new operations in `src/Semantics/` before testing them.

## Running Tests

### Run all tests:
```bash
dune test
```

### Run only translation validation tests:
```bash
dune exec test/test_driver.exe -- test "Translation Validation"
```

### Verbose output:
```bash
dune test --verbose
```

### Set custom mlir-opt path:
```bash
export MLIR_OPT_PATH=/path/to/custom/mlir-opt
dune test
```

## Adding Tests to CI

Translation validation tests are automatically included in the CI pipeline. Ensure:

1. All test files are checked into git
2. Tests pass locally before pushing
3. If using external passes, document dependencies

## Limitations and Future Work

### Current Limitations

1. **Not formal verification**: Oracle testing only checks specific test cases, not all possible inputs
2. **Interpreter coverage**: Only supports basic arith and cf dialect operations
3. **Output comparison**: Simple string equality (no semantic equivalence checking)
4. **Single-function tests**: Multi-function programs not yet fully tested
5. **No state**: Tests without memory/state operations only

### Future Work: True Translation Validation

True translation validation requires proving semantic equivalence in Coq, similar to the [Alive2](https://github.com/AliveToolkit/alive2) project for LLVM. The workflow would be:

```
input.mlir  ──parse──> AST₁ ─┐
                              ├──> Coq Proof: ⟦AST₁⟧ ≈ ⟦AST₂⟧
output.mlir ──parse──> AST₂ ─┘
```

**Key differences from oracle testing**:

| Aspect | Oracle Testing (Current) | Translation Validation (Future) |
|--------|--------------------------|----------------------------------|
| **Scope** | Specific test inputs | All possible executions |
| **Method** | Execute & compare outputs | Prove semantic equivalence |
| **Guarantee** | Bug detection | Correctness proof |
| **Location** | OCaml test driver | Coq theorems |
| **Automation** | Fully automatic | Semi-automatic (tactics) |

**Implementation approach**:

1. **Parsing**: Convert MLIR text to our Coq AST definitions
2. **Semantic equivalence**: Use ITree bisimulation (`eutt`, `eqit`) to state equivalence
3. **Proof structure**: One Coq file per optimization pass (e.g., `src/Theory/SCCP_correct.v`)
4. **Automation**: Develop tactics for common proof patterns
5. **Organization**: Not in `test/` but in `src/Theory/` or `src/Pass/`

**Example proof structure**:

```coq
(* src/Theory/SCCP_correct.v *)
Require Import Semantics.Core.
Require Import Pass.SCCP.

Theorem sccp_preserves_semantics :
  forall (original optimized : program),
    sccp original = optimized ->
    eutt (semantics original) (semantics optimized).
Proof.
  (* ... proof using ITree tactics ... *)
Qed.
```

**This is a major undertaking** and requires:
- Implementing optimization passes in Coq (not just testing external `mlir-opt`)
- Developing proof automation tactics
- Potentially extracting verified optimizers

For now, oracle testing serves as:
1. Bug detection during development
2. Regression testing
3. Specification by example (test cases can guide formal proofs)

### Other Future Enhancements

1. **Property-based testing**: Generate random programs and validate optimizations (QuickChick)
2. **More dialect support**: memref, scf, affine, etc.
3. **MLIR test suite integration**: Port relevant tests from LLVM project
4. **Dead code detection**: Flag when optimized code has unused definitions

## Troubleshooting

### Test fails: "Unsupported operation"

**Solution**: The operation is not implemented in the semantics. Either:
- Add semantics for the operation in `src/Semantics/`
- Use a different test case with supported operations

### Test fails: "mlir-opt not found"

**Solution**: Install LLVM/MLIR or set `MLIR_OPT_PATH`:
```bash
export MLIR_OPT_PATH=/usr/local/opt/llvm/bin/mlir-opt
```

### Test fails: "Interpreter execution failed"

**Solution**: The program may not be executable (e.g., missing function name). Check:
- Function is named and has proper signature
- All operations are supported
- Program is well-formed MLIR

### Outputs don't match but should be equivalent

**Potential causes**:
- Dead code differences (extra constants not used)
- Reordering of independent operations
- Different intermediate values (but same final result)

**Solution**: Manually inspect both outputs. If semantically equivalent, this indicates a limitation of string comparison - consider implementing more sophisticated equivalence checking.

## References

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [SCCP Pass Documentation](https://mlir.llvm.org/docs/Passes/#-sccp-sparse-conditional-constant-propagation)
- [Translation Validation Overview](https://en.wikipedia.org/wiki/Translation_validation)
- ADR-0001: Translation Validation Framework
