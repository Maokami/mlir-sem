# MLIR-Sem Project: Comprehensive Architecture Analysis

**Date**: 2025-11-07  
**Scope**: OCaml driver infrastructure, Translation Validation approach, extraction process

---

## 1. EXECUTIVE SUMMARY

### Current State
The mlir-sem project implements a **Hybrid Validation Strategy** (per ADR-0002) combining:
- **Oracle Testing**: Differential testing comparing execution outputs before/after optimization
- **Translation Validation (Future)**: Formal proofs of semantic equivalence in Coq

### Key Insight: The Project Uses the CORRECT Architecture

The architecture properly separates:
1. **Coq Formalization** (`src/`) - The verified semantics core
2. **OCaml Driver** (`driver/`) - Unverified "last-mile" MLIR C API bindings
3. **Oracle Testing** (`validation/`) - Empirical validation against external tools
4. **Translation Validation** (Future via `src/Pass/`, `src/Theory/`) - Formal proofs

---

## 2. OCaml DRIVER INFRASTRUCTURE

### 2.1 Architecture Overview

```
MLIR Text File
     ↓
[MLIR C API] ← bindings.ml (Ctypes FFI)
     ↓
transformer.ml (C → OCaml AST conversion)
     ↓
Interp.mlir_program (Coq-extracted AST)
     ↓
Interp.run_program (Extracted interpreter)
     ↓
itree execution (OCaml runner in run.ml)
     ↓
JSON output {"result": [...]}
```

### 2.2 Component Breakdown

#### 2.2.1 `bindings.ml` - MLIR C API Bindings
**Purpose**: FFI layer to LLVM/MLIR C API using Ctypes

**Key Types**:
- `mlir_context` - Global MLIR compilation context
- `mlir_module` - Top-level MLIR module
- `mlir_operation` - MLIR operation (nodes in IR)
- `mlir_region` - Regions (blocks container)
- `mlir_block` - Control flow blocks
- `mlir_value` - SSA values
- `mlir_type` - Type information
- `mlir_attribute` - Operation metadata

**Dialects Registered**:
- `func` - Function operations
- `arith` - Arithmetic operations
- `cf` - Control flow operations

**Key Functions** (Sample):
```ocaml
let context_create : void -> mlir_context
let module_create_parse : mlir_context → mlir_string_ref → mlir_module
let operation_get_identifier : mlir_operation → mlir_identifier
let block_get_first_operation : mlir_block → mlir_operation
```

#### 2.2.2 `transformer.ml` - MLIR C API → OCaml AST Conversion

**Purpose**: Recursively traverse MLIR C API structures and build Coq-extracted AST

**Key Functions**:

1. **Type Conversion** (`transform_mlir_type`)
   - Converts `mlir_type` → `Interp.mlir_type`
   - Handles: Integer types, Function types
   - Maps MLIR width to Z (big integers)

2. **Operation Conversion** (`transform_operation`)
   - Dispatch based on operation name string
   - Handles:
     - `arith.constant` → Arith_Constant
     - `arith.addi` → Arith_AddI
     - `arith.cmpi` → Arith_CmpI (with predicate translation)
     - `func.return` → Func_Return
     - `cf.br` → Cf_Branch
     - `cf.cond_br` → Cf_CondBranch
   - Maps C pointers to SSA value names via hashtables

3. **Block Conversion** (`transform_block`)
   - Extracts block name and block arguments
   - Recursively transforms all operations in block

4. **Region Conversion** (`transform_region`)
   - Two-pass: First populates block name map, then transforms
   - Ensures block references are properly resolved

5. **Function Conversion** (`transform_func`)
   - Extracts function type from `function_type` attribute
   - Builds region body
   - Returns `Interp.mlir_func` with name, type, body

6. **Module Conversion** (`transform_module`)
   - Entry point: Resets all maps
   - Traverses top-level operations
   - Returns `Interp.mlir_program` (list of functions)

**State Management**:
```ocaml
let value_map : (mlir_value, string) Hashtbl.t  (* C pointer → "%" name *)
let value_counter = ref 0
let block_map : (mlir_block, string) Hashtbl.t  (* C pointer → "blockN" name *)
let block_counter = ref 0

let reset_maps () = (* Called at start of each module transformation *)
```

**Important Design Notes**:
- SSA values are renamed to canonical form (`%0`, `%1`, etc.)
- Block names are generated as `block0`, `block1`, etc.
- This abstraction is necessary because MLIR C API returns raw pointers, not names

#### 2.2.3 `ast_printer.ml` - Coq AST → Text (for testing)

**Purpose**: Pretty-print extracted AST for golden testing

**Key Function**:
```ocaml
let string_of_mlir_program : Interp.mlir_program -> string
```

Used to verify parser output matches expected AST structure.

#### 2.2.4 `run.ml` - Interpreter Execution

**Purpose**: Main executable that orchestrates the full pipeline

**Workflow**:
1. Parse MLIR file using C API
2. Transform to OCaml AST
3. Call `Interp.run_program` (extracted from Coq)
4. Execute the returned itree using `run_tree`
5. Format output as JSON

**ITree Execution** (`run_tree`):
```ocaml
let rec run_tree (tree : itree) : mlir_value list option =
  match Interp.observe tree with
  | RetF (_, result_vals) -> Some result_vals  (* Success *)
  | TauF t -> run_tree t                        (* Tau/silent step *)
  | VisF (event, k) ->
      match event with
      | FailureE msg -> None                    (* Error *)
      | _ -> None                               (* Unhandled event *)
```

This is the **OCaml runner** that interprets the ITree returned by the Coq semantics.

### 2.3 Data Flow Example

Input: `simple_arith.mlir`
```mlir
func.func @main() -> i32 {
  %c1 = arith.constant 10 : i32
  %c2 = arith.constant 20 : i32
  %result = arith.addi %c1, %c2 : i32
  return %result : i32
}
```

**Step 1: Parsing (bindings.ml → MLIR C API)**
```
mlirModuleCreateParse ctx "func.func @main() ..."
→ returns mlir_module (opaque pointer)
```

**Step 2: Traversal (transformer.ml)**
```
transform_module(mlir_module)
  → get_operation(module)
    → get_region(operation, 0)
      → get_first_block(region)
        → get_first_operation(block) → transform_operation
          (1) arith.constant 10 → Op(["%" ], Arith_Constant(10, Integer 32))
          (2) arith.constant 20 → Op(["%1"], Arith_Constant(20, Integer 32))
          (3) arith.addi %0, %1 → Op(["%2"], Arith_AddI("%0", "%1", Integer 32))
          (4) func.return %2  → Term(Func_Return(["%2"]))
        → collect into block
      → collect into region
    → FuncOp("main", FunctionType([], [Integer 32]), [region])
  → [FuncOp(...)]  ← mlir_program
```

**Step 3: Execution (Coq-extracted interpreter)**
```
Interp.run_program(mlir_program, "main")
→ returns itree MlirSemE (interpreter_state * [mlir_value])

run_tree(itree)
→ Ret(_, [30])  ← final result after executing operations
```

**Step 4: Output (run.ml)**
```json
{"result": [30]}
```

---

## 3. COQA EXTRACTION PROCESS

### 3.1 Extraction Configuration

**File**: `src/Extraction/Extract.v`

```coq
Extraction Language OCaml.
Set Extraction AccessOpaque.

Extraction "AST.ml" 
  mlir_type arith_cmp_pred general_op terminator_op operation block region mlir_func mlir_program.

Extraction "Values.ml" mlir_value.

Extraction "Events.ml" 
  LocalE FunctionE ControlE FailureE MlirSemE raise.

Extraction "Denotation.ml" 
  denote_general_op denote_terminator denote_block denote_func.

Extraction "Interp.ml" 
  call_frame function_def program_context interpreter_state 
  handle_event interpret build_program_context run_program.
```

### 3.2 Extracted Modules

#### AST.ml (from Syntax.AST.v)
- Type definitions: `mlir_type`, `arith_cmp_pred`, `general_op`, `terminator_op`, `operation`, `block`, `region`, `mlir_func`, `mlir_program`
- Single source of truth from Coq

#### Values.ml (from Semantics.Values.v)
- `mlir_value` = Z (big integer)
- Simple extraction from Coq type

#### Events.ml (from Semantics.Events.v)
- ITree effect types: `LocalE`, `FunctionE`, `ControlE`, `FailureE`, `MlirSemE`
- Effect constructors for all computation types

#### Denotation.ml (from Semantics.Denotation.v)
- `denote_general_op` - Semantics of arithmetic/memory operations
- `denote_terminator` - Semantics of control flow
- `denote_block` - Block execution
- `denote_func` - Function execution
- Uses ITree to encode semantics as functions returning itrees

#### Interp.ml (from Semantics.Interp.v)
- **Key extraction**: `run_program : mlir_program → string → itree option`
- This is what the OCaml driver calls!
- Includes event handler: `handle_event : MlirSemE → State monad`
- Sets up initial interpreter state

### 3.3 Linking with OCaml Driver

**Dune Build**:
```dune
(library
  (name driver)
  (modules bindings transformer ast_printer)
  (libraries ctypes ctypes.foreign mlir_sem_extracted zarith)
  (wrapped true))

(executable
  (name run)
  (public_name driver_run)
  (modules run)
  (libraries driver mlir_sem_extracted)
  ...LLVM linking flags...
)
```

**Key Dependencies**:
- `mlir_sem_extracted` - Coq extraction (built separately)
- `ctypes`, `ctypes.foreign` - C FFI
- `zarith` - Big integers (for `mlir_value` type)

### 3.4 Build Workflow

```
Coq source files (src/*.v)
    ↓ coqc compile
Extracted OCaml (src/Extraction/*.ml)
    ↓ Extract.v commands
{AST.ml, Values.ml, Events.ml, Denotation.ml, Interp.ml}
    ↓ dune build
mlir_sem_extracted.a (OCaml library)
    ↓ link with driver
driver_run (executable)
```

---

## 4. ORACLE TESTING INFRASTRUCTURE

### 4.1 Current State

**Location**: `validation/oracle/sccp/`

Files:
- `sccp_addi.mlir` - Original program (constant folding test)
- `sccp_addi.opt.mlir` - Optimized version (pre-generated with mlir-opt)
- `sccp_branch.mlir` - Conditional branch test
- `sccp_branch.opt.mlir` - Optimized version

### 4.2 Test Structure (test/test_driver.ml)

**Test Helper**: `make_translation_validation_test`

```ocaml
let make_translation_validation_test 
  ~name ~mlir_file ~opt_mlir_file ~pass_pipeline =
  test_case name `Quick (fun () ->
    let original_output = run_interpreter original_path in
    let optimized_output = 
      match opt_mlir_file with
      | Some path -> run_interpreter path
      | None -> 
          let temp = Filename.temp_file "mlir_opt" ".mlir" in
          run_mlir_opt original_path pass_pipeline temp;
          run_interpreter temp
    in
    check string "outputs match" original_output optimized_output)
```

**Test Execution Flow**:
1. Parse original MLIR → Transform to AST → Execute → Get output
2. Parse optimized MLIR → Transform to AST → Execute → Get output
3. Compare outputs as strings

### 4.3 Current Test Cases

```ocaml
make_translation_validation_test
  ~name:"SCCP constant propagation with addi"
  ~mlir_file:"oracle/sccp/sccp_addi.mlir"
  ~opt_mlir_file:(Some "oracle/sccp/sccp_addi.opt.mlir")
  ~pass_pipeline:"builtin.module(func.func(sccp))";
```

**What This Tests**:
- Original: 10 + 20 (two constants, addition)
- Optimized: 30 (constant folded)
- Both should return [30]

### 4.4 Limitations (By Design)

These are **oracle tests**, NOT formal translation validation:

| Aspect | Oracle Testing | True Translation Validation |
|--------|---|---|
| Scope | Specific test inputs | All possible inputs |
| Method | Execute and compare | Prove equivalence in Coq |
| Guarantee | Empirical | Formal proof |
| Can find bugs | ✅ Yes | ✅ Yes |
| Proves correctness | ❌ No | ✅ Yes |

---

## 5. TRANSLATION VALIDATION (FUTURE PATH)

### 5.1 Current Infrastructure

**File**: `src/TranslationValidation/Framework.v`

Provides:
- `prog_equiv` - Two programs produce same observables
- `func_equiv` - Two functions produce same result for all inputs
- `block_equiv` - Two blocks are control-flow equivalent
- `pass_correct` - Pass preserves semantics

**Proof Tactics**:
- `tv_intro` - Set up equiv proof
- `tv_simpl` - Simplify goals
- `tv_step` - Take one equivalence step
- `tv_auto` - Automated proof attempt

### 5.2 Planned Architecture

The CORRECT approach combines both:

**Short-term** (now):
- Parse MLIR text using C API (`bindings.ml` → `transformer.ml`)
- Execute both original and optimized programs
- Compare outputs (oracle testing)
- Catch bugs, build test corpus

**Long-term** (future):
- Parse both MLIR versions to Coq AST definitions
- Generate Coq equality goals from parsed AST
- **Prove** semantic equivalence using ITree bisimulation
- Extract verified passes (if needed)

### 5.3 Why This Is the Right Approach

Per ADR-0001 and ADR-0002:

1. **Leverage existing MLIR passes**: Don't re-implement, use `mlir-opt` as oracle
2. **Practical validation**: Oracle tests catch bugs and guide proof development
3. **Formal guarantees**: Translation validation proofs provide soundness for critical passes
4. **Scalable**: Don't need to certify every pass, focus on high-value ones
5. **Separation of concerns**:
   - `src/` = Verified semantics (formal)
   - `validation/` = Empirical validation (pragmatic)
   - `driver/` = Unverified infrastructure (untrusted)

---

## 6. ARCHITECTURAL INCONSISTENCIES & CORRECTIONS

### 6.1 ✅ NO MAJOR INCONSISTENCIES FOUND

The current architecture is **fundamentally sound**. The project follows the Vellvm model correctly:

**Vellvm Model**:
```
src/rocq/           ← Coq semantics
src/rocq/Transformations/  ← Pass implementations + proofs
tests/              ← Unit tests of our tools
tests/alive2/       ← Oracle tests vs external tools
```

**MLIR-Sem Model** (correctly mirrors Vellvm):
```
src/                ← Coq semantics
src/Pass/           ← Pass implementations (planned)
src/Theory/         ← Pass proofs (planned)
src/TranslationValidation/  ← Proof framework (exists)
test/               ← Unit tests of our tools
validation/         ← Oracle tests vs external tools
driver/             ← OCaml infrastructure (unverified)
```

### 6.2 Minor Observations

**Note 1**: `TranslationValidation/Framework.v` exists but is incomplete
- Defines equivalence relations and tactics
- Example lemmas (constant_folding_correct, dce_unreachable_correct) use undefined symbols
- This is OK - it's a framework sketch awaiting full implementation

**Note 2**: Parser location is correct
- MLIR C API parsing happens in `transformer.ml` via FFI
- NOT parsing MLIR text directly with Ocamllex/Menhir
- This is appropriate because:
  - MLIR C API is the official, maintained parser
  - Avoids duplicate parser implementation
  - Guarantees conformance with upstream MLIR

**Note 3**: AST is complete
- Currently supports: arith (constant, addi, cmpi), cf (br, cond_br), func (return)
- Design allows easy extension for new dialects
- No inconsistencies

---

## 7. THE CORRECT WORKFLOW FOR TRANSLATION VALIDATION

### 7.1 High-Level Process

```
Given: original.mlir and optimized.mlir (from mlir-opt)

Step 1: Parse both using MLIR C API
        original.mlir  ──[bindings/transformer]──> AST₁ : mlir_program
        optimized.mlir ──[bindings/transformer]──> AST₂ : mlir_program

Step 2: Coq proof of equivalence
        Goal: prog_equiv AST₁ AST₂
        Proof: By ITree bisimulation (eutt) of run_program outputs

Step 3: Extract proof to OCaml (if needed)
        Extract witness to demonstrate equivalence
```

### 7.2 Implementation Roadmap

**Phase 1: Oracle Testing** (Current)
- Infrastructure exists: `validation/oracle/`, `test_driver.ml`
- Catches bugs empirically
- Example: SCCP constant folding test

**Phase 2: Translation Validation (Framework Setup)**
- Framework exists: `src/TranslationValidation/Framework.v`
- Needs: Completion of equivalence lemmas
- Example: Prove `sccp_preserves_semantics : pass_correct sccp`

**Phase 3: Formal Pass Verification**
- Create `src/Pass/SCCP.v` and `src/Pass/SCCP_correct.v`
- Implement SCCP in Coq (optional, for certified passes)
- Prove: For any program P, `denote_program P ≈ denote_program (sccp P)`

**Phase 4: Extraction & Meta-Validation**
- Extract verified SCCP to OCaml
- Run both mlir-opt and extracted SCCP on test suite
- Compare results (meta-validation: proving the prover is correct)

---

## 8. KEY FILES SUMMARY

| File | Purpose | Type | Status |
|------|---------|------|--------|
| `driver/bindings.ml` | MLIR C API FFI | OCaml | ✅ Complete |
| `driver/transformer.ml` | C API → OCaml AST | OCaml | ✅ Complete |
| `driver/run.ml` | Main executable | OCaml | ✅ Complete |
| `driver/ast_printer.ml` | AST pretty-printing | OCaml | ✅ Complete |
| `src/Syntax/AST.v` | MLIR AST definitions | Coq | ✅ Complete |
| `src/Semantics/Denotation.v` | Operation semantics | Coq | ✅ Complete |
| `src/Semantics/Interp.v` | Interpreter framework | Coq | ✅ Complete |
| `src/Extraction/Extract.v` | Extraction config | Coq | ✅ Complete |
| `src/TranslationValidation/Framework.v` | TV proof framework | Coq | 🟡 Partial |
| `validation/oracle/` | Oracle tests | Test | ✅ Active |
| `test/test_driver.ml` | Test suite | OCaml | ✅ Active |
| `docs/adr/ADR-0002-hybrid-validation-strategy.md` | Design decision | Doc | ✅ Complete |

---

## 9. RECOMMENDATIONS

### 9.1 For Implementing Translation Validation Passes

1. **DO NOT** implement passes in Coq from scratch initially
   - Use ADR-0002 hybrid approach
   - Let mlir-opt do the transformation
   - Write Coq proofs that verify the result

2. **DO** use the oracle test cases as specifications
   - Each oracle test defines expected behavior
   - Proof should match test case semantics

3. **DO** complete `TranslationValidation/Framework.v`
   - Define `denote_program` and friends
   - Implement missing helper lemmas
   - Develop reusable proof tactics

4. **DO** document each pass proof in Coq comments
   - Explain the proof strategy
   - Cite relevant optimization papers

### 9.2 For Adding New Dialects

1. Update `src/Syntax/AST.v` with new operation types
2. Update `src/Semantics/Denotation.v` with operation semantics
3. Update `driver/bindings.ml` with C API bindings
4. Update `driver/transformer.ml` with transformation rules
5. Add oracle tests in `validation/oracle/dialect-name/`
6. Add unit tests in `test/test_driver.ml`

### 9.3 For Architecture Reviews

The current approach is sound. Any architectural changes should:
1. Maintain the Vellvm-inspired structure
2. Keep Coq and OCaml concerns separated
3. Clearly mark what is verified vs unverified
4. Document decisions in ADR format

---

## 10. CONCLUSION

The mlir-sem project demonstrates **correct architecture** for formal verification of optimizations:

1. ✅ **Single source of truth**: MLIR AST defined once in Coq
2. ✅ **Pragmatic + rigorous**: Oracle testing now, formal proofs later
3. ✅ **Proper separation**: Coq (verified), OCaml driver (unverified), validation (empirical)
4. ✅ **Scalable approach**: Can verify many passes without re-implementation
5. ✅ **Proven design**: Follows Vellvm (LLVM IR formal semantics) model

The OCaml driver correctly:
- Uses MLIR C API for robust parsing
- Transforms C API objects to Coq AST
- Calls extracted interpreter
- Executes ITrees in OCaml

Translation Validation can proceed with confidence that the infrastructure is sound.

