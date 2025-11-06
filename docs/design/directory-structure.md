# Directory Structure Design

## Overview

This document describes the directory structure of mlir-sem, inspired by the Vellvm project architecture.

## Current Structure Issues

The current `test/` directory conflates multiple concerns:
- Unit tests for our tools (parser, semantics, interpreter)
- Oracle testing (comparing optimized vs original programs)
- Integration tests

This makes it unclear what each test validates and where new tests should go.

## Proposed Structure

```
mlir-sem/
├── src/
│   ├── Syntax/         # MLIR AST definitions
│   │   ├── Dialect/    # Dialect-specific syntax (arith, cf, func, etc.)
│   │   └── Core.v      # Common syntax elements
│   │
│   ├── Semantics/      # ITree-based semantics
│   │   ├── Dialect/    # Dialect-specific semantics
│   │   ├── Handlers/   # Event handlers
│   │   └── Core.v      # Core semantic framework
│   │
│   ├── Theory/         # Metatheory and proofs
│   │   ├── Equivalence.v     # ITree bisimulation utilities
│   │   ├── Refinement.v      # Refinement relations
│   │   ├── Properties.v      # General properties
│   │   └── Tactics.v         # Proof automation
│   │
│   ├── Pass/           # Optimization passes (Coq implementations)
│   │   ├── SCCP.v            # SCCP implementation
│   │   ├── SCCP_correct.v    # SCCP correctness proof
│   │   ├── DCE.v             # Dead code elimination
│   │   └── DCE_correct.v     # DCE correctness proof
│   │
│   ├── Utils/          # Common utilities, tactics, lemmas
│   └── Extraction/     # Coq → OCaml extraction configuration
│
├── driver/             # OCaml driver and utilities
│   ├── parser/         # MLIR text → Coq AST parser
│   ├── interpreter/    # Reference interpreter runner
│   └── cli/            # Command-line interface
│
├── test/               # Unit and integration tests for OUR TOOLS
│   ├── unit/           # Unit tests
│   │   ├── syntax/     # Syntax tests (parser, AST)
│   │   ├── semantics/  # Semantics tests (interpreter correctness)
│   │   └── extraction/ # Extraction smoke tests
│   │
│   ├── integration/    # End-to-end integration tests
│   │   ├── *.mlir      # Test MLIR programs
│   │   └── expect/     # Expected outputs
│   │
│   └── property/       # QuickChick property-based tests
│       └── *.v         # Property tests
│
├── validation/         # Oracle testing and pass validation
│   ├── oracle/         # Oracle tests (execution comparison)
│   │   ├── sccp/       # SCCP oracle tests
│   │   │   ├── input.mlir
│   │   │   └── optimized.mlir
│   │   └── dce/        # DCE oracle tests
│   │
│   ├── cross-check/    # Cross-validation with MLIR tools
│   │   └── README.md   # How to compare against mlir-opt
│   │
│   └── benchmarks/     # Performance benchmarks
│       └── *.mlir
│
├── verify/             # Translation validation experiments (future)
│   ├── passes/         # Pass verification workflows
│   │   └── sccp/
│   │       ├── test1.mlir          # Original
│   │       ├── test1.opt.mlir      # Optimized
│   │       └── test1_proof.v       # Equivalence proof (future)
│   │
│   └── tools/          # Verification tooling
│       └── equiv_check.ml          # Automated equivalence checking
│
├── docs/
│   ├── adr/            # Architectural Decision Records
│   ├── design/         # Design documents
│   └── howto/          # Guides and tutorials
│
└── tools/              # Build scripts, utilities, helpers

```

## Directory Responsibilities

### `src/` - Coq Formalization (Verified Core)

**Responsibility**: All formally verified code lives here.

- **Syntax/**: MLIR AST definitions (single source of truth)
- **Semantics/**: ITree-based semantics (compositional, modular)
- **Theory/**: Metatheory, proofs about semantics, general theorems
- **Pass/**: Optimization passes implemented in Coq WITH correctness proofs
- **Utils/**: Shared tactics, lemmas, utilities
- **Extraction/**: Extraction configuration for OCaml

**Philosophy**: Everything here should be:
1. Formally specified in Coq
2. Proven correct where applicable
3. Extractable to OCaml

---

### `driver/` - OCaml Runtime (Unverified)

**Responsibility**: OCaml code for running extracted interpreters and tooling.

- **parser/**: MLIR text parser (converts text → Coq AST)
- **interpreter/**: Reference interpreter runner
- **cli/**: Command-line interface

**Philosophy**: Unverified "last-mile" infrastructure. Keep minimal and well-tested.

---

### `test/` - Testing Our Tools

**Responsibility**: Tests that validate OUR implementations (parser, semantics, extraction).

**What goes here:**
- ✅ Does our parser correctly parse MLIR?
- ✅ Does our interpreter correctly execute programs?
- ✅ Does extraction work?
- ✅ Do our semantic functions behave as expected?

**What does NOT go here:**
- ❌ Comparing mlir-opt output against our interpreter (that's `validation/`)
- ❌ Testing external MLIR passes (that's `validation/`)

**Subdirectories:**
- `unit/`: Fast, focused tests
- `integration/`: End-to-end tests of complete workflows
- `property/`: QuickChick property-based tests

---

### `validation/` - Oracle Testing and External Validation

**Responsibility**: Compare our semantics against external tools (mlir-opt, MLIR execution).

**What goes here:**
- ✅ Running mlir-opt and comparing outputs
- ✅ Differential testing against MLIR C++ interpreter
- ✅ Performance benchmarks
- ✅ Cross-validation with upstream MLIR tests

**Subdirectories:**
- `oracle/`: Execute original + optimized, compare outputs
- `cross-check/`: Compare against MLIR toolchain
- `benchmarks/`: Performance measurements

**Philosophy**: These tests do NOT verify our Coq code—they validate that our semantics matches real-world MLIR behavior.

---

### `verify/` - Translation Validation (Future)

**Responsibility**: Formal verification experiments for proving pass correctness.

**What goes here (future work):**
- Parse both original and optimized MLIR
- Generate Coq equivalence goals
- Semi-automated proof attempts
- Example proofs for case studies

**Philosophy**: This is experimental and forward-looking. Not part of core infrastructure yet.

---

## Migration Plan

### Phase 1: Reorganize `test/`
1. Keep existing tests in `test/integration/`
2. Move oracle tests to `validation/oracle/`
3. Update test driver accordingly

### Phase 2: Add `Pass/` directory
1. Stub out `src/Pass/` with placeholder files
2. Document expected structure

### Phase 3: Create `validation/`
1. Move oracle testing infrastructure
2. Add cross-validation scripts

### Phase 4: Expand `Theory/`
1. Add equivalence checking utilities
2. Add proof automation tactics

## Comparison with Vellvm

| Aspect | Vellvm | mlir-sem (proposed) |
|--------|--------|---------------------|
| Coq sources | `src/rocq/` | `src/` |
| OCaml driver | `src/ml/libvellvm/` | `driver/` |
| Tests | `tests/` (integration) | `test/` (unit+integration) |
| Oracle testing | `tests/alive2/` | `validation/oracle/` |
| Pass proofs | `src/rocq/Transformations/` | `src/Pass/` + `src/Theory/` |
| Property tests | `src/rocq/QC/` | `test/property/` |

## Benefits of This Structure

1. **Clear separation of concerns**: Testing our tools vs validating against external tools
2. **Vellvm-inspired**: Proven structure from mature project
3. **Scalable**: Easy to add new dialects, passes, validations
4. **Explicit about verification status**: `src/` is verified, `driver/` is not, `validation/` is empirical
5. **Future-proof**: Room for translation validation experiments

## References

- [Vellvm Directory Structure](https://github.com/vellvm/vellvm)
- ADR-0002: Translation Validation Strategy
