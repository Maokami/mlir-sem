---
name: add-dialect
description: Guide through adding a new MLIR dialect with syntax, semantics, tests, and documentation. Use when the user wants to add support for a new MLIR dialect (e.g., arith, scf, memref) to the formal semantics framework.
---

# Add MLIR Dialect

This skill guides you through the complete workflow of adding formal semantics for a new MLIR dialect to the mlir-sem project.

## Overview

Adding a dialect involves:
1. Defining syntax and types in Coq
2. Defining ITree-based semantics (events + handlers)
3. Writing unit and property tests
4. Creating documentation (ADR + design notes)
5. Updating extraction configuration
6. Verifying CI passes

## Workflow Steps

### Step 1: Gather Dialect Information

Ask the user:
- Which dialect are we adding? (e.g., `arith`, `scf`, `memref`, `func`)
- Which operations from this dialect should be supported?
- Are there any specific use cases or examples to prioritize?

### Step 2: Define Syntax (src/Syntax/)

Create or update files in `src/Syntax/Dialect/`:

1. **Types**: Define dialect-specific types
   - File: `src/Syntax/Dialect/<DialectName>Types.v`
   - Example: integer types, index types, memref types

2. **Operations**: Define operation syntax
   - File: `src/Syntax/Dialect/<DialectName>Ops.v`
   - Include operation name, operands, attributes, result types
   - Follow MLIR's operation definition structure

3. **Attributes**: Define dialect-specific attributes if needed
   - File: `src/Syntax/Dialect/<DialectName>Attrs.v`

**Key Principles**:
- One core concept per file when possible
- Use descriptive names matching MLIR conventions
- Document any deviations from MLIR spec

### Step 3: Define Semantics (src/Semantics/)

Create semantics files in `src/Semantics/Dialect/`:

1. **Events**: Define ITree events for dialect operations
   - File: `src/Semantics/Dialect/<DialectName>Events.v`
   - Each operation should trigger appropriate events
   - Example: arithmetic ops → computation events

2. **Handlers**: Implement event handlers
   - File: `src/Semantics/Dialect/<DialectName>Handlers.v`
   - Handlers interpret events into concrete behavior
   - Keep handlers modular and replaceable

3. **Semantics**: Connect syntax to events
   - File: `src/Semantics/Dialect/<DialectName>.v`
   - Map each operation to its ITree semantics
   - Document observation equivalence used (`eutt`, `eqit`)

**Key Principles**:
- Clear documentation of equivalence relations
- Modular handler design for composition
- Prove basic properties (determinism, commutativity where applicable)

### Step 4: Write Tests

Create test files in `test/`:

1. **Unit Tests** (`test/unit/`)
   - File: `test/unit/test_<dialect_name>.ml`
   - Test individual operations with concrete inputs
   - Verify expected outputs

2. **Property Tests** (`test/property/`)
   - File: `test/property/<DialectName>Properties.v`
   - Use QuickChick for invariant checking
   - Test roundtrip properties, commutativity, etc.

3. **Golden Tests** (`test/oracle/`)
   - Create `.mlir` test files
   - Compare extracted interpreter with MLIR toolchain output
   - Document expected behavior

**Run tests**: `dune test -f`

### Step 5: Document the Dialect

Create documentation:

1. **Design Doc** (`docs/design/`)
   - File: `docs/design/<dialect_name>-semantics.md`
   - Explain design decisions
   - Document operation semantics
   - Include examples

2. **ADR** (if architectural decisions were made)
   - File: `docs/adr/ADR-XXXX-<title>.md`
   - Use create-adr skill to generate template
   - Document why certain semantic choices were made

### Step 6: Update Extraction Configuration

Update extraction settings in `src/Extraction/`:

1. Add dialect modules to extraction list
2. Ensure dialect operations are extractable
3. Run smoke test: build and execute extracted interpreter

### Step 7: Verify CI

Ensure all CI checks pass:

1. Format check
2. Build and proof verification
3. All tests (unit + property + golden)
4. Extraction and smoke run
5. Documentation validation

## Definition of Done Checklist

Before marking the dialect addition complete:

- [ ] Syntax definitions in `src/Syntax/Dialect/`
- [ ] Semantics (events + handlers) in `src/Semantics/Dialect/`
- [ ] Unit tests in `test/unit/`
- [ ] QuickChick properties in `test/property/`
- [ ] Golden tests in `test/oracle/`
- [ ] Design documentation in `docs/design/`
- [ ] ADR created if needed
- [ ] Extraction configuration updated
- [ ] CI passing
- [ ] Code reviewed (request GitHub Copilot review)

## Example: Adding Arithmetic Dialect

```coq
(* src/Syntax/Dialect/ArithOps.v *)
Inductive ArithOp : Type :=
  | AddIOp : ArithOp
  | MulIOp : ArithOp
  | CmpIOp : comparison_predicate -> ArithOp.

(* src/Semantics/Dialect/ArithEvents.v *)
Variant ArithE : Type -> Type :=
  | BinOp : binop -> value -> value -> ArithE value
  | Compare : cmp -> value -> value -> ArithE bool.

(* src/Semantics/Dialect/Arith.v *)
Definition interp_arith_op (op : ArithOp) (args : list value) : itree E value :=
  match op, args with
  | AddIOp, [v1; v2] => trigger (BinOp Add v1 v2)
  | MulIOp, [v1; v2] => trigger (BinOp Mul v1 v2)
  | ...
  end.
```

## Common Issues

- **Missing dependencies**: Ensure dialect dependencies are properly imported
- **Type mismatches**: Check that value types align between syntax and semantics
- **Test failures**: Verify MLIR test files are valid with `mlir-opt --verify-diagnostics`
- **Extraction errors**: Check that all constructors and functions are extractable

## Next Steps After Completion

Once the dialect is added:
1. Consider adding optimization passes (use `add-pass` skill)
2. Verify dialect composes with existing dialects
3. Add more comprehensive examples to documentation
