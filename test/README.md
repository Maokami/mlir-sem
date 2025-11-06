# Test Suite

This directory contains tests for **our tools**: parser, semantics, and interpreter.

**For oracle testing and pass validation**, see [`validation/`](../validation/) directory.

## Purpose

Tests in this directory validate that:
- Our MLIR parser correctly transforms MLIR text to our Coq AST
- Our semantics correctly execute programs
- Extraction works and produces valid OCaml code

Tests here do NOT validate optimization passes—that's done in `validation/`.

## Structure

```
test/
├── test_driver.ml          # Main test driver (Alcotest)
├── dune                    # Build configuration
├── *.mlir                  # Test MLIR files
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
Tests optimization passes by comparing execution outputs before/after optimization.

**Note**: Oracle tests are now in [`validation/oracle/`](../validation/) directory for better organization.

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

Oracle tests are now organized in the [`validation/`](../validation/) directory.

See:
- [Validation README](../validation/README.md) for adding oracle tests
- [Oracle Testing Guide](../docs/howto/translation-validation-testing.md) for detailed instructions

## Environment Variables

- `RUN_EXE_PATH`: Path to interpreter executable (set by dune)
- `MLIR_OPT_PATH`: Path to mlir-opt (defaults to PATH lookup)
- `DUNE_SOURCEROOT`: Project root (set by dune)

## Current Test Coverage

- ✅ Basic arithmetic operations (constant, addi)
- ✅ Comparisons (cmpi)
- ✅ Control flow (br, cond_br)
- ✅ Function definitions and returns
- ✅ SCCP optimization pass (constant propagation, semantic preservation)
- ❌ Memory operations (memref dialect)
- ❌ Loop optimizations (scf dialect)
- ❌ Multiple functions in one module
- ❌ Function calls

## CI Integration

Tests run automatically in CI on every push. See `.github/workflows/` for configuration.
