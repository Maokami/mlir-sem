# Property-Based Tests

This directory contains **property-based tests** using QuickChick for the MLIR-Sem project.

## Overview

Property-based testing automatically generates random test inputs to verify properties of our code. Unlike unit tests that check specific examples, property tests verify general invariants across many randomly generated inputs.

## Installation

To use these tests, you need to install QuickChick:

```bash
opam install coq-quickchick
```

QuickChick requires:
- Coq >= 8.12
- OCaml >= 4.08

## Current Tests

### AST_Properties.v

Tests fundamental properties of the AST definitions:

1. **Type Decidability**: Integer type equality is decidable
2. **Predicate Validity**: All generated comparison predicates are valid
3. **Constant Preservation**: Constant operations preserve their values
4. **Structural Properties**: Operations maintain expected structure

## Running Tests

### From Coq

```coq
From PropertyTests Require Import AST_Properties.
QuickChick test_ast_properties.
```

### From Command Line (once QuickChick is installed)

```bash
# Build the property tests
dune build test/property

# Run extracted tests (after uncommenting in dune file)
dune exec test/property/property_tests.exe
```

## Writing New Property Tests

### Step 1: Create Generators

Define generators for your types:

```coq
Definition gen_my_type : G my_type :=
  (* Use QuickChick combinators: elements, oneOf, liftGen, etc. *)
```

### Step 2: Write Properties

Express your invariants as boolean properties:

```coq
Definition prop_my_property : Checker :=
  forAllShrink gen_my_type shrink (fun x =>
    (* Your property here *)
    some_predicate x ?
  ).
```

### Step 3: Add to Test Suite

Include your property in the main test suite:

```coq
Definition test_my_module : Checker :=
  conjoin [
    whenFail "Error message" prop_my_property;
    (* ... other properties ... *)
  ].
```

## Example Properties to Add

High-priority properties for future development:

1. **Parser/Printer Roundtrip**:
   ```
   forall ast, parse (print ast) = Some ast
   ```

2. **Semantic Preservation**:
   ```
   forall op, eval (optimize op) = eval op
   ```

3. **Type Safety**:
   ```
   forall op input, typecheck op input -> eval op input ≠ error
   ```

4. **Pass Correctness**:
   ```
   forall prog, semantics (pass prog) ≈ semantics prog
   ```

## References

- [QuickChick Tutorial](https://softwarefoundations.cis.upenn.edu/qc-current/index.html)
- [QuickChick Documentation](https://github.com/QuickChick/QuickChick)
- [Vellvm QuickChick Tests](https://github.com/vellvm/vellvm) - See `deps/vellvm/doc/intern/vellvm-quickchick-overview.org`

## Testing Strategy

As outlined in CLAUDE.md:

- ✅ **TDD approach**: Write properties first when designing new features
- ✅ **Property testing**: Use QuickChick for invariants and roundtrip properties
- ✅ **CI seed storage**: Store failing seeds in issues for reproducibility
- ⚠️ **Comprehensive coverage**: Port relevant tests from MLIR test suite

## Status

**Current State**: Infrastructure set up, basic AST properties defined
**Next Steps**:
1. Install QuickChick
2. Enable property tests in CI
3. Add roundtrip properties for parser/printer
4. Add semantic preservation properties for passes
