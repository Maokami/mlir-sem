# Test Suite

This directory contains the test suite for mlir-sem.

## Structure

```
test/
├── test_driver.ml          # Main test driver (Alcotest)
├── dune                    # Build configuration
├── *.mlir                  # Test MLIR files
├── *.opt.mlir              # Optimized MLIR files (for translation validation)
└── expect/                 # Golden files for output comparison
    ├── *.ast.expect       # Expected AST outputs
    └── *.output.expect    # Expected interpreter outputs
```

## Test Suites

### 1. Parser and Transformer
Tests the MLIR-to-AST transformation pipeline. Compares transformed AST against golden files.

### 2. Interpreter Execution
Tests the extracted OCaml interpreter. Compares program output against golden files.

### 3. Oracle Testing (Pass Validation)
Tests optimization passes by comparing execution outputs before/after optimization. Note: This is oracle testing, not formal translation validation (which requires Coq proofs).

## Running Tests

```bash
# Run all tests
dune test

# Force re-run all tests (ignore cache)
dune test -f

# Run specific test suite
dune exec test/test_driver.exe -- test "Oracle Testing"

# Verbose output
dune test --verbose
```

## Adding New Tests

### Parser/Transformer Test
1. Create `test/example.mlir`
2. Generate expected AST: Run parser and save output to `test/expect/example.ast.expect`
3. Add test case in `test_driver.ml`:
   ```ocaml
   make_parse_and_transform_test
     ~name:"Parse example"
     ~mlir_file:"example.mlir"
     ~expect_file:"example.ast.expect"
   ```

### Interpreter Test
1. Create `test/example.mlir`
2. Run interpreter and save output to `test/expect/example.output.expect`
3. Add test case in `test_driver.ml`:
   ```ocaml
   make_interpreter_execution_test
     ~name:"Execute example"
     ~mlir_file:"example.mlir"
     ~expect_file:"example.output.expect"
   ```

### Oracle Test (Pass Validation)
1. Create `test/example.mlir` (original program)
2. Generate optimized version: `mlir-opt test/example.mlir -pass-pipeline='...' -o test/example.opt.mlir`
3. Add both files to `dune` deps
4. Add test case in `test_driver.ml`:
   ```ocaml
   make_translation_validation_test  (* naming will be updated *)
     ~name:"Example optimization"
     ~mlir_file:"example.mlir"
     ~opt_mlir_file:(Some "example.opt.mlir")
     ~pass_pipeline:"builtin.module(func.func(pass-name))"
   ```

See [Oracle Testing Guide](../docs/howto/translation-validation-testing.md) for details.

## Environment Variables

- `RUN_EXE_PATH`: Path to interpreter executable (set by dune)
- `MLIR_OPT_PATH`: Path to mlir-opt (defaults to PATH lookup)
- `DUNE_SOURCEROOT`: Project root (set by dune)

## Current Test Coverage

- ✅ Basic arithmetic operations (constant, addi)
- ✅ Comparisons (cmpi)
- ✅ Control flow (br, cond_br)
- ✅ Function definitions and returns
- ✅ SCCP optimization pass
- ❌ Memory operations (memref dialect)
- ❌ Loop optimizations (scf dialect)
- ❌ Multiple functions in one module
- ❌ Function calls

## CI Integration

Tests run automatically in CI on every push. See `.github/workflows/` for configuration.
