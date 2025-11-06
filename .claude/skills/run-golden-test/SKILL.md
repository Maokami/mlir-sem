---
name: run-golden-test
description: Execute golden tests that compare the extracted OCaml interpreter against MLIR toolchain output. Use when validating semantics implementation or debugging differences between formal semantics and MLIR behavior.
---

# Run Golden Tests

This skill guides you through running and maintaining golden tests that validate the extracted interpreter against MLIR's official toolchain.

## Overview

Golden tests ensure that:
- The extracted interpreter matches MLIR's behavior
- Semantics definitions are faithful to MLIR specification
- Changes don't introduce regressions

The workflow involves:
1. Building the extracted interpreter
2. Running test cases through both interpreters
3. Comparing outputs
4. Analyzing and resolving differences

## Workflow Steps

### Step 1: Ensure Prerequisites

Check that required tools are available:

```bash
# Check MLIR tools
which mlir-opt
which mlir-translate

# Check build system
dune --version

# Check OCaml
ocaml --version
```

If tools are missing, guide user to install them.

### Step 2: Build the Extracted Interpreter

Extract and build the OCaml interpreter:

```bash
# Build Coq project
dune build

# Extract to OCaml
dune build @extract

# Build extracted interpreter
dune build driver/mlir_interp.exe
```

If extraction fails:
- Check extraction configuration in `src/Extraction/`
- Verify all Coq modules compile successfully
- Look for non-computational definitions that can't be extracted

### Step 3: Identify Test Cases

Determine which tests to run:

1. **All tests**: `test/oracle/**/*.mlir`
2. **Specific dialect**: `test/oracle/arith/*.mlir`
3. **Specific test**: `test/oracle/arith/addi_simple.mlir`

Ask user which scope to test if not specified.

### Step 4: Run MLIR Toolchain on Test Cases

For each `.mlir` test file:

```bash
# Validate MLIR file
mlir-opt --verify-diagnostics test/oracle/arith/addi_simple.mlir

# Run through MLIR interpreter (if available)
mlir-cpu-runner test/oracle/arith/addi_simple.mlir

# Or use custom MLIR pass/tool
mlir-opt --some-pass test/oracle/arith/addi_simple.mlir -o expected.mlir
```

Save MLIR's output as the "expected" result.

### Step 5: Run Extracted Interpreter on Test Cases

Run the extracted interpreter:

```bash
# Run interpreter on test file
_build/default/driver/mlir_interp.exe test/oracle/arith/addi_simple.mlir
```

Save the extracted interpreter's output as the "actual" result.

### Step 6: Compare Outputs

Compare expected vs actual results:

```bash
# Direct comparison
diff expected.txt actual.txt

# Or use test driver
dune test
```

**Comparison strategies**:
- **Exact match**: Outputs should be identical
- **Semantic equivalence**: Allow different but equivalent representations
- **Approximate match**: For floating-point values

Document the comparison strategy for each test.

### Step 7: Analyze Differences

If outputs differ, investigate:

1. **Is MLIR behavior correct?**
   - Check MLIR documentation
   - Review dialect specifications
   - Test with MLIR's own test suite

2. **Is our semantics definition correct?**
   - Review Coq definitions
   - Check ITree handler implementations
   - Verify extraction configuration

3. **Is this an acceptable difference?**
   - Some differences may be intentional (e.g., different error messages)
   - Document known differences in test metadata

### Step 8: Update Test Expectations (if needed)

If the difference is expected:

1. **Update expected output**: If MLIR behavior changed
2. **Fix semantics**: If our definition was wrong
3. **Document difference**: If difference is acceptable

Add comments to test files explaining expected behavior:

```mlir
// RUN: mlir-opt %s | FileCheck %s
// EXPECTED: Result should be constant 42
func.func @test() -> i32 {
  %c1 = arith.constant 1 : i32
  %c41 = arith.constant 41 : i32
  %result = arith.addi %c1, %c41 : i32
  return %result : i32
}
// CHECK: %[[RES:.*]] = arith.constant 42 : i32
```

### Step 9: Create Regression Tests

For each fixed bug or resolved difference:

1. Add a specific test case to `test/oracle/`
2. Document what the test validates
3. Ensure test is run in CI

## Test Organization

Organize tests by dialect and complexity:

```
test/oracle/
├── arith/
│   ├── constant.mlir          # Simple constant operations
│   ├── addi_simple.mlir        # Basic addition
│   ├── addi_overflow.mlir      # Edge case: overflow
│   └── complex_expr.mlir       # Complex expressions
├── scf/
│   ├── if_simple.mlir          # Basic conditionals
│   ├── for_loop.mlir           # Loop constructs
│   └── nested_control_flow.mlir # Nested structures
└── integration/
    ├── multi_dialect.mlir      # Cross-dialect tests
    └── optimization.mlir       # Pass integration tests
```

## Test Metadata

Add metadata to each test file:

```mlir
// TEST: arith.addi semantics
// DIALECT: arith
// FEATURES: integer arithmetic, constant folding
// MLIR-VERSION: 1.0
// EXPECTED-BEHAVIOR: Addition should be commutative and associative
// KNOWN-ISSUES: None
```

## Automation with Test Driver

The test driver (`test/test_driver.ml`) should:

1. Discover all `.mlir` files in `test/oracle/`
2. Run each through both interpreters
3. Compare outputs
4. Report results (pass/fail)
5. Store failing cases for debugging

Example test driver usage:

```bash
# Run all golden tests
dune test

# Run specific test
dune exec -- test/test_driver.exe test/oracle/arith/addi_simple.mlir

# Update all expectations
dune test --force
```

## Debugging Differences

When a test fails:

### Step 1: Isolate the Issue
```bash
# Run single test
_build/default/driver/mlir_interp.exe test/oracle/failing_test.mlir -v

# Enable debug output
MLIR_INTERP_DEBUG=1 _build/default/driver/mlir_interp.exe test/oracle/failing_test.mlir
```

### Step 2: Examine Intermediate States
- Add logging to ITree handlers
- Print intermediate values in transformation
- Compare step-by-step execution

### Step 3: Minimize Test Case
Reduce failing test to minimal example:
```mlir
// Original (complex)
func.func @complex() { ... 20 operations ... }

// Minimized
func.func @minimal() {
  %c1 = arith.constant 1 : i32  // This line causes failure
  return %c1 : i32
}
```

### Step 4: Check Semantics Definition
Review Coq definitions:
```coq
(* Is this the correct semantics for arith.constant? *)
Definition interp_constant (t : type) (v : value) : itree E value :=
  Ret v.  (* Maybe this should trigger some event? *)
```

### Step 5: Verify Extraction
Check that extracted OCaml code is correct:
```ocaml
(* _build/default/src/Semantics/Arith.ml *)
let interp_constant t v = Ret v  (* Is this what we expect? *)
```

## Continuous Integration

Integrate golden tests into CI:

```yaml
# CI/.github/workflows/test.yml
- name: Run Golden Tests
  run: |
    dune build @extract
    dune test
    # Fail if any differences found
    if [ $? -ne 0 ]; then
      echo "Golden tests failed!"
      exit 1
    fi
```

**Store test artifacts**:
- Failing test outputs
- Diff results
- Debug logs

## Updating Tests After MLIR Changes

When MLIR version updates:

1. **Re-run all tests**: `dune test --force`
2. **Review failures**: Some may be due to MLIR behavior changes
3. **Update expectations**: If changes are correct
4. **Update semantics**: If our definitions need adjustment
5. **Document changes**: Note MLIR version and behavior changes in ADR

## Best Practices

1. **Test Early**: Add golden tests as soon as dialect support is added
2. **Test Often**: Run tests on every commit (CI)
3. **Test Thoroughly**: Cover edge cases, not just happy paths
4. **Document Tests**: Explain what each test validates
5. **Keep Tests Small**: One concept per test file
6. **Version Control Expectations**: Commit expected outputs
7. **Automate Everything**: Use test driver, don't manually compare

## Common Issues

### Extraction Fails
- **Cause**: Non-computational definitions, missing extraction hints
- **Fix**: Add extraction configuration, refactor definitions

### Output Format Mismatch
- **Cause**: Different pretty-printing between MLIR and extracted interpreter
- **Fix**: Normalize outputs before comparison, or use semantic comparison

### Test Flakiness
- **Cause**: Non-deterministic behavior, timing issues
- **Fix**: Make tests deterministic, use fixed seeds for random testing

### Performance Issues
- **Cause**: Extracted code is slow, large test suite
- **Fix**: Profile and optimize hot paths, parallelize tests

## Integration with Other Workflows

Golden tests complement:
- **Unit tests**: Unit tests check small pieces, golden tests check integration
- **Property tests**: Properties check invariants, golden tests check concrete behavior
- **Proof verification**: Proofs ensure correctness, golden tests validate implementation

## Example: Adding a Golden Test

```bash
# 1. Create test file
cat > test/oracle/arith/addi_simple.mlir <<EOF
// TEST: Basic integer addition
func.func @test_addi() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return %sum : i32
}
// EXPECTED: 3
EOF

# 2. Run through MLIR
mlir-opt test/oracle/arith/addi_simple.mlir

# 3. Build interpreter
dune build driver/mlir_interp.exe

# 4. Run test
dune exec -- test/test_driver.exe test/oracle/arith/addi_simple.mlir

# 5. Check result
echo "Test passed!"
```

## After Running Tests

Once tests complete:
1. **Record results**: Pass/fail counts, timing
2. **Update dashboard**: If you have test result tracking
3. **Fix failures**: Don't let failures accumulate
4. **Celebrate success**: When all tests pass!
