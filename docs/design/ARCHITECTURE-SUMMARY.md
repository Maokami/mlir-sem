# MLIR-Sem Architecture: Quick Reference

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLIR Formalization                         │
│                    (Coq + ITree + OCaml)                        │
└─────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  MLIR .mlir File │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
            ┌───────▼───────┐         ┌─────────▼────────┐
            │  MLIR C API   │         │  mlir-opt Tool   │
            │  (from LLVM)  │         │  (External)      │
            └───────┬───────┘         └─────────┬────────┘
                    │                           │
        ┌───────────▼───────────┐   ┌──────────▼──────────┐
        │  driver/bindings.ml   │   │  Optimized .mlir    │
        │  (Ctypes FFI Layer)   │   │  (After transform)  │
        └───────────┬───────────┘   └──────────┬──────────┘
                    │                           │
        ┌───────────▼───────────────────────────▼────────────┐
        │     driver/transformer.ml                          │
        │  (C API Object → OCaml AST Conversion)            │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────▼────────────────────┐
        │  Interp.mlir_program           │
        │  (Coq-extracted OCaml type)    │
        └───────────┬────────────────────┘
                    │
        ┌───────────▼────────────────────┐
        │  Interp.run_program()          │
        │  (Extracted from Coq)          │
        │  Returns: itree monad          │
        └───────────┬────────────────────┘
                    │
        ┌───────────▼────────────────────┐
        │  driver/run.ml                 │
        │  run_tree() - itree interpreter│
        └───────────┬────────────────────┘
                    │
        ┌───────────▼────────────────────┐
        │  JSON Output                   │
        │  {"result": [value]}           │
        └────────────────────────────────┘

COMPARISON FLOW (Oracle Testing):
═════════════════════════════════

Original.mlir  ──[transform]──> AST₁ ──[run_program]──> Output₁
                                                          │
                                                     Compare!
                                                          │
Optimized.mlir ──[transform]──> AST₂ ──[run_program]──> Output₂
```

---

## Directory Structure Map

```
src/
├── Syntax/AST.v              ← MLIR AST (single source of truth)
├── Semantics/
│   ├── Values.v              ← mlir_value = Z (big integers)
│   ├── Events.v              ← Effect types (LocalE, FunctionE, ControlE, FailureE)
│   ├── Denotation.v          ← Operation semantics (ITrees)
│   └── Interp.v              ← Interpreter framework + run_program
├── Extraction/Extract.v      ← Extraction config → OCaml
├── TranslationValidation/
│   ├── Framework.v           ← Proof tactics & lemmas
│   └── SCCP_Examples.v       ← Example SCCP proofs
└── Pass/ (planned)           ← Future: certified pass implementations

driver/
├── bindings.ml               ← MLIR C API FFI (Ctypes)
├── transformer.ml            ← C API → OCaml AST
├── ast_printer.ml            ← AST pretty-printing
└── run.ml                    ← Main executable

test/
├── test_driver.ml            ← Test suite (Alcotest)
├── simple_arith.mlir         ← Unit test MLIR files
├── control_flow.mlir
└── expect/
    ├── *.ast.expect          ← Golden AST outputs
    └── *.output.expect       ← Golden execution outputs

validation/
├── oracle/sccp/
│   ├── sccp_addi.mlir        ← Original program
│   ├── sccp_addi.opt.mlir    ← Optimized version
│   ├── sccp_branch.mlir
│   └── sccp_branch.opt.mlir
├── cross-check/              ← (Planned) Compare vs MLIR toolchain
└── benchmarks/               ← (Planned) Performance tests
```

---

## Component Responsibilities

### Coq (`src/`) - VERIFIED
**Role**: Formal specification of MLIR semantics

| Module | Purpose | Key Exports |
|--------|---------|------------|
| `Syntax.AST.v` | MLIR syntax | `mlir_type`, `operation`, `block`, `mlir_func`, `mlir_program` |
| `Semantics.Values.v` | Value representation | `mlir_value` (= Z) |
| `Semantics.Events.v` | Computation effects | `LocalE`, `FunctionE`, `ControlE`, `FailureE`, `MlirSemE` |
| `Semantics.Denotation.v` | Meaning of ops | `denote_general_op`, `denote_block`, `denote_func` |
| `Semantics.Interp.v` | Execution model | `run_program`, `interpret` |
| `TranslationValidation.Framework.v` | TV proofs | `prog_equiv`, `pass_correct`, proof tactics |

**Properties**:
- ✅ Formally verified with Coq
- ✅ No axioms (except justified ones documented)
- ✅ Extracted to OCaml without modification
- ✅ Single source of truth for semantics

### OCaml Driver (`driver/`) - UNVERIFIED
**Role**: Unverified "last-mile" infrastructure for executing extracted code

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `bindings.ml` | FFI to MLIR C API | `context_create`, `module_create_parse`, operation/block/value accessors |
| `transformer.ml` | C → OCaml AST | `transform_module`, `transform_operation`, `transform_block` |
| `ast_printer.ml` | AST visualization | `string_of_mlir_program` |
| `run.ml` | Main entry point | Orchestrates parse → transform → execute → output |

**Properties**:
- ❌ Not formally verified (uses C FFI)
- ✅ Well-tested with golden tests
- ✅ Stateless transformation (no side effects)
- ✅ Calls extracted Coq code only

### Oracle Tests (`validation/`) - EMPIRICAL
**Role**: Differential testing against external tools

| Directory | Purpose | What It Tests |
|-----------|---------|--------------|
| `oracle/sccp/` | SCCP correctness | Original vs mlir-opt output equivalence |
| `cross-check/` | Vs MLIR tools | Compare extracted interpreter vs LLVM IR interpreter |
| `benchmarks/` | Performance | Speed and resource usage |

**Properties**:
- ✅ Fast pragmatic validation
- ❌ Not formal proofs (limited to test cases)
- ✅ Catches implementation bugs
- ✅ Guides formal proof development

---

## Data Type Flow

```
MLIR Text (UTF-8)
       │
       └──[bindings.ml: mlirModuleCreateParse]──→ mlir_module (C pointer)
              │
              ├─ mlir_context created
              ├─ Dialects registered (func, arith, cf)
              └─ String parsed by LLVM's official parser

mlir_module (opaque C pointer)
       │
       └──[transformer.ml]──→ Coq-extracted OCaml types
              │
              ├─ transform_module
              │   └─ For each function:
              │       ├─ Get name, type, body region
              │       └─ transform_region
              │           └─ For each block:
              │               ├─ Get name, arguments
              │               └─ transform_operations_in_block
              │                   └─ For each operation:
              │                       └─ Match on operation name
              │                           ├─ "arith.constant" → Arith_Constant
              │                           ├─ "arith.addi" → Arith_AddI
              │                           ├─ "cf.br" → Cf_Branch
              │                           └─ ... (dispatch by name)
              │
              └─ Map C pointers to SSA names
                 ├─ %0, %1, %2, ... (value names)
                 └─ block0, block1, ... (block names)

OCaml AST (extracted from Coq)
       │
       ├─ type mlir_type = Integer of Z | FunctionType of ...
       ├─ type general_op = Arith_Constant of Z * mlir_type | ...
       ├─ type operation = Op of value_id list * general_op | Term of terminator_op
       ├─ type block = { block_name: string; block_ops: operation list; ... }
       ├─ type region = block list
       ├─ type mlir_func = FuncOp of string * mlir_type * region
       └─ type mlir_program = mlir_func list

Execution
       │
       └──[Interp.run_program(mlir_program, "main")]──→ itree monad
              │
              └─ Set up interpreter state:
                 ├─ build_program_context (name → function body)
                 ├─ Initial empty call frame
                 └─ Denote main function as itree

ITree Execution
       │
       └──[run_tree : itree → mlir_value list option]
              │
              ├─ observe tree:
              │   ├─ RetF(result) → return Some(result)
              │   ├─ TauF(next) → recursively interpret next
              │   └─ VisF(event, k) → handle event
              │       └─ FailureE → return None
              │
              └─ Reconstruct state through execution

JSON Output
       │
       └── {"result": [<mlir_value>, <mlir_value>, ...]}
```

---

## Translation Validation Workflow (Complete)

```
STAGE 1: Oracle Testing (Current)
════════════════════════════════

Input programs:
┌─────────────────┐        ┌──────────────────┐
│ Original MLIR   │        │ Optimized MLIR   │
│ (unoptimized)   │        │ (from mlir-opt)  │
└────────┬────────┘        └────────┬─────────┘
         │                          │
         └──[Parse + Execute]──────┘
                 │
        ┌────────▼────────┐
        │ Compare Outputs │
        └────────┬────────┘
                 │
        ✅ If outputs match → SCCP is likely correct for this case
        ❌ If outputs differ → SCCP has a bug (or test case is wrong)


STAGE 2: Translation Validation (Future)
════════════════════════════════════════

Input programs:
┌─────────────────┐        ┌──────────────────┐
│ Original MLIR   │        │ Optimized MLIR   │
│ (unoptimized)   │        │ (from mlir-opt)  │
└────────┬────────┘        └────────┬─────────┘
         │                          │
         └──[Parse to Coq AST]─────┘
                 │
        ┌────────▼─────────────────┐
        │ Generate Coq Goal:       │
        │ denote_program AST₁ ≈    │
        │ denote_program AST₂      │
        └────────┬─────────────────┘
                 │
        ┌────────▼────────────────────┐
        │ Prove Equivalence in Coq:   │
        │ ITree bisimulation lemmas   │
        │ + SMT solver hints (future) │
        └────────┬────────────────────┘
                 │
        ✅ Proof succeeds → SCCP correct for ALL inputs
        ❌ Proof fails → Either SCCP is wrong, or semantics doesn't match
```

---

## Key Design Principles

### 1. Single Source of Truth
- MLIR AST defined once in Coq (`src/Syntax/AST.v`)
- Extracted to OCaml without modification
- No separate parser in OCaml - just conversion from C API

### 2. Pragmatic + Rigorous
- **Pragmatic now**: Oracle tests catch bugs empirically
- **Rigorous later**: Translation validation proofs provide guarantees
- Hybrid approach follows ADR-0002

### 3. Clear Separation of Concerns
```
Verified Code (Coq)           Unverified Code (OCaml)           Empirical Tests
─────────────────────────────────────────────────────────────────────────────
    src/                             driver/                      validation/
  Semantics                      Infrastructure                  Oracle tests
  Definition                      Last-mile glue                Differential
  (ITrees)                        (C FFI, runner)               Testing
  
   Trust: ✅ High             Trust: 🟡 Medium (tested)         Trust: ❌ Low
   (Proven correct)           (Golden tests validate)           (Pragmatic)
```

### 4. Extraction from Coq to OCaml
- Extraction is **automatic** and **bidirectional** (semantic meaning preserved)
- No manual rewrites or "unverified OCaml patches"
- If you verify in Coq, it's automatically verified in extracted code

### 5. No Axioms (Unless Documented)
- Coq proofs are constructive
- Extraction yields executable code
- Justified axioms only for external laws (e.g., big integer arithmetic)

---

## Testing Tiers

### Tier 1: Unit Tests (test/)
**Purpose**: Verify our implementations (parser, semantics, extraction)

Examples:
- Does parser correctly transform MLIR text to AST?
- Does interpreter execute correctly?
- Does extraction work?

**Run**: `dune test`

### Tier 2: Oracle Tests (validation/oracle/)
**Purpose**: Verify semantics matches real MLIR behavior

Examples:
- Does SCCP produce same output as original?
- Does CSE compute correct results?
- Do optimizations preserve behavior?

**Run**: `dune test` (also runs oracle tests)

### Tier 3: Translation Validation (src/TranslationValidation/)
**Purpose**: FORMAL proofs of pass correctness

Examples:
- Prove: ∀ P, denote_program(P) ≈ denote_program(sccp(P))
- Prove: ∀ P, denote_program(P) ≈ denote_program(dce(P))

**Status**: Framework exists, awaiting implementation

---

## Checklist for Adding New Optimization Pass

- [ ] Add oracle tests in `validation/oracle/pass-name/`
- [ ] Run oracle tests: `dune test`
- [ ] Verify all tests pass
- [ ] If formal verification desired:
  - [ ] Create `src/Pass/PassName.v` (implementation in Coq, optional)
  - [ ] Create `src/Theory/PassName_correct.v` (correctness proof)
  - [ ] Prove: `Theorem pass_correct : pass_correct pass_name`
  - [ ] Update `src/Extraction/Extract.v` if new types/functions
  - [ ] Run: `dune build src` to regenerate extraction
  - [ ] Run: `dune test` to verify extraction works
- [ ] Document proof strategy in comments
- [ ] Create ADR if new architectural pattern

---

## References

- **Full Analysis**: [ARCHITECTURE-ANALYSIS.md](./ARCHITECTURE-ANALYSIS.md)
- **ADR-0001**: Translation Validation Framework (strategy)
- **ADR-0002**: Hybrid Validation Strategy (pragmatic + rigorous)
- **Design Docs**: [directory-structure.md](./directory-structure.md)
- **Testing Docs**: [../howto/translation-validation-testing.md](../howto/translation-validation-testing.md)
