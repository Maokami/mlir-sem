# MLIR-Sem Design Documentation

This directory contains comprehensive design and architecture documentation for the mlir-sem project.

## Documents

### 1. ARCHITECTURE-ANALYSIS.md (590 lines)
**Comprehensive technical analysis of the entire system**

- **Section 1: Executive Summary** - High-level overview of the Hybrid Validation Strategy
- **Section 2: OCaml Driver Infrastructure** - Detailed breakdown of:
  - bindings.ml (MLIR C API FFI)
  - transformer.ml (C API → OCaml AST conversion)
  - ast_printer.ml (visualization)
  - run.ml (main executable)
  - Data flow examples
- **Section 3: Coq Extraction Process** - How Coq code is extracted to OCaml
  - Extraction configuration
  - Extracted module structure
  - Build workflow
  - Linking with OCaml driver
- **Section 4: Oracle Testing Infrastructure** - How oracle tests work
  - Current test cases (SCCP)
  - Test execution flow
  - Limitations by design
- **Section 5: Translation Validation (Future Path)** - Long-term formal verification strategy
  - Current framework
  - Planned architecture
  - Why this approach is correct
- **Section 6-10: Architecture review, workflow, file summary, recommendations, conclusions**

**Best for**: Understanding the complete system, how components interact, and the verification strategy

### 2. ARCHITECTURE-SUMMARY.md (375 lines)
**Quick reference guide with visual diagrams**

- **System Overview** - Data flow diagrams
- **Directory Structure Map** - Visual layout of source tree
- **Component Responsibilities** - Table of who does what
- **Data Type Flow** - Step-by-step example of MLIR → AST → Execution
- **Translation Validation Workflow** - Oracle vs formal verification stages
- **Key Design Principles** - 5 core architecture principles
- **Testing Tiers** - Unit tests, oracle tests, translation validation
- **Checklist for Adding New Passes** - Step-by-step guide
- **References** - Links to related docs

**Best for**: Quick understanding, visual learners, implementation checklists

### 3. directory-structure.md (223 lines)
**Detailed explanation of the directory structure**

- Current structure issues (what it originally tried to solve)
- Proposed structure (following Vellvm model)
- Directory responsibilities and philosophy
- Migration plan
- Comparison with Vellvm
- Benefits of this structure

**Best for**: Understanding why directories are organized as they are

### Other Design Documents

- [../adr/ADR-0001-translation-validation-framework.md](../adr/ADR-0001-translation-validation-framework.md) - Why we chose translation validation strategy
- [../adr/ADR-0002-hybrid-validation-strategy.md](../adr/ADR-0002-hybrid-validation-strategy.md) - Why we use oracle tests + formal proofs
- [../howto/translation-validation-testing.md](../howto/translation-validation-testing.md) - How to write oracle tests

---

## Key Findings from Analysis

### ✅ CORRECT ARCHITECTURE CONFIRMED

The mlir-sem project has a fundamentally sound architecture:

1. **Coq (src/)** - Verified semantics
   - Single source of truth: MLIR AST
   - Formal semantics using ITrees
   - No axioms except justified ones
   - Extracted to OCaml automatically

2. **OCaml Driver (driver/)** - Unverified infrastructure
   - MLIR C API bindings (Ctypes FFI)
   - AST transformation (C pointers → OCaml types)
   - Well-tested, calls extracted Coq only

3. **Oracle Tests (validation/)** - Empirical validation
   - Differential testing: original vs optimized
   - Currently tests SCCP optimization
   - Catches bugs pragmatically
   - Guides proof development

4. **Translation Validation Framework (src/TranslationValidation/)** - Future formal proofs
   - Infrastructure ready
   - Awaiting implementation of pass proofs
   - Will prove semantic equivalence formally

### NO MAJOR INCONSISTENCIES FOUND

The architecture correctly follows the Vellvm (LLVM formal semantics) model:

```
Vellvm                          MLIR-Sem
src/rocq/          ←→           src/
src/rocq/Transformations/ ←→    src/Pass/ + src/Theory/
tests/             ←→           test/
tests/alive2/      ←→           validation/oracle/
```

Minor observations are all appropriate by design:
- MLIR C API parsing (not custom parser) - correct, uses official parser
- TranslationValidation/Framework.v is partial - OK, it's a sketch
- AST supports 6 operations - designed for extension

---

## For Different Readers

### I want to understand the overall architecture
→ Start with **ARCHITECTURE-SUMMARY.md** for diagrams and overview

### I need detailed technical information
→ Read **ARCHITECTURE-ANALYSIS.md** section by section

### I'm implementing a new optimization pass
→ See **ARCHITECTURE-SUMMARY.md** "Checklist for Adding New Optimization Pass"

### I need to understand the design decisions
→ Read **directory-structure.md** and the ADR documents (ADR-0001, ADR-0002)

### I want to add oracle tests
→ Read [../howto/translation-validation-testing.md](../howto/translation-validation-testing.md)

---

## Architecture Highlights

### Data Flow (MLIR Text → JSON Output)

```
MLIR File
   ↓
MLIR C API (bindings.ml)
   ↓
transformer.ml (C → OCaml AST)
   ↓
Interp.run_program (extracted from Coq)
   ↓
run_tree (OCaml ITree executor)
   ↓
JSON {"result": [values]}
```

### Oracle Testing (Validation)

```
Original.mlir → Execute → Output₁
                            Compare!
Optimized.mlir → Execute → Output₂
```

### Translation Validation (Future)

```
Original.mlir → Parse to AST₁ ─┐
                              ├─→ Prove AST₁ ≈ AST₂ in Coq
Optimized.mlir → Parse to AST₂ ─┘
```

---

## Key Components

### OCaml Driver (driver/)
- **bindings.ml** (237 lines): MLIR C API FFI
- **transformer.ml** (325 lines): C API → OCaml AST
- **run.ml** (79 lines): Main executable
- **ast_printer.ml** (TBD): AST visualization

### Coq Semantics (src/)
- **Syntax/AST.v** (50 lines): MLIR AST definitions
- **Semantics/Values.v**: Value representation
- **Semantics/Events.v**: ITree effect types
- **Semantics/Denotation.v**: Operation semantics
- **Semantics/Interp.v** (90 lines): Interpreter + run_program
- **Extraction/Extract.v** (14 lines): Extraction config

### Oracle Tests (validation/oracle/sccp/)
- sccp_addi.mlir, sccp_addi.opt.mlir (constant folding)
- sccp_branch.mlir, sccp_branch.opt.mlir (conditional branches)
- 3 active test cases in test/test_driver.ml

---

## Design Principles

1. **Single Source of Truth** - AST defined once in Coq, extracted to OCaml
2. **Pragmatic + Rigorous** - Oracle tests now, formal proofs later
3. **Clear Separation** - Verified (Coq), unverified (driver), empirical (validation)
4. **Extraction-Based** - Automatic, semantic-preserving translation Coq → OCaml
5. **No Axioms** - Constructive proofs, executable code

---

## Next Steps

To implement Translation Validation passes:

1. Use existing oracle tests as specifications
2. Complete TranslationValidation/Framework.v
3. Implement pass proofs in src/Theory/PassName_correct.v
4. Prove semantic equivalence using ITree bisimulation
5. Extract verified passes if needed

See ARCHITECTURE-SUMMARY.md "Checklist for Adding New Optimization Pass" for details.

---

## References

- **[ARCHITECTURE-ANALYSIS.md](./ARCHITECTURE-ANALYSIS.md)** - Full technical analysis (590 lines)
- **[ARCHITECTURE-SUMMARY.md](./ARCHITECTURE-SUMMARY.md)** - Quick reference with diagrams (375 lines)
- **[directory-structure.md](./directory-structure.md)** - Directory organization rationale
- **[../adr/ADR-0001-translation-validation-framework.md](../adr/ADR-0001-translation-validation-framework.md)** - Strategy decision
- **[../adr/ADR-0002-hybrid-validation-strategy.md](../adr/ADR-0002-hybrid-validation-strategy.md)** - Validation approach
- **[../howto/translation-validation-testing.md](../howto/translation-validation-testing.md)** - How to write oracle tests
