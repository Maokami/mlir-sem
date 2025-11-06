# ADR-0002: Hybrid Validation Strategy for MLIR Optimization Passes

**Status**: Accepted
**Date**: 2025-11-06
**Authors**: Jaeho Choi, Claude
**Supersedes**: None
**Related**: ADR-0001

## Context

We need to validate the correctness of MLIR optimization passes. There are two fundamentally different approaches:

### 1. **Oracle Testing** (Differential Testing)
- Compare execution outputs before/after optimization
- Uses external tools (mlir-opt) as black boxes
- Tests specific inputs only
- **Pros**: Pragmatic, immediate, catches bugs
- **Cons**: No formal guarantees, incomplete coverage

### 2. **Translation Validation** (Formal Verification)
- Prove semantic equivalence in Coq using ITree bisimulation
- Requires implementing passes in Coq
- **Pros**: Soundness for all inputs, verified correctness
- **Cons**: High effort, requires proof engineering

### Key Questions

1. Should we focus on oracle testing or formal verification?
2. Where should oracle tests live? (Currently in `test/`, but that's for testing our tools)
3. How do we structure the project for both approaches?

### Inspiration: Vellvm and Alive2

**Vellvm** uses:
- `src/rocq/Transformations/`: Coq implementations + correctness proofs
- `tests/alive2/`: Oracle testing against Alive2
- Hybrid approach: formal verification for core, oracle for validation

**Alive2** (for LLVM):
- Automated translation validation using SMT solvers
- Proves equivalence for specific test cases automatically
- No Coq, but inspiration for our formal approach

## Decision

We adopt **Option C: Hybrid Strategy** with clear separation of concerns:

### 1. Short-term: Oracle Testing (Pragmatic)

**Purpose**: Catch bugs, build test corpus, understand MLIR passes

**Approach**:
- Execute original and optimized MLIR through our interpreter
- Compare outputs
- Use `mlir-opt` as oracle (black box)
- Build test suite incrementally

**Location**: `validation/oracle/`
- NOT in `test/` (which is for testing our tools)
- Clear distinction: validation vs testing

### 2. Long-term: Translation Validation (Rigorous)

**Purpose**: Formally verify critical passes

**Approach**:
- Implement optimization passes in Coq (`src/Pass/`)
- Prove semantic equivalence using ITree bisimulation (`src/Theory/`)
- Extract verified optimizers to OCaml
- Focus on high-value passes (SCCP, DCE, etc.)

**Location**: `src/Pass/` and `src/Theory/`

### 3. Complementary Roles

| Aspect | Oracle Testing | Translation Validation |
|--------|----------------|------------------------|
| **Purpose** | Bug detection, regression | Soundness guarantee |
| **Coverage** | Specific inputs | All inputs |
| **Effort** | Low (automated) | High (proof engineering) |
| **Speed** | Fast | Slow (proof development) |
| **Trust** | Empirical | Formal |
| **Location** | `validation/oracle/` | `src/Pass/`, `src/Theory/` |

**Together**: Oracle testing finds issues quickly; formal verification ensures critical passes are correct.

## Consequences

### Directory Structure

Reorganize project to separate concerns:

```
src/
├── Pass/           # Coq implementations of passes
│   ├── SCCP.v     # Implementation
│   └── SCCP_correct.v  # Correctness proof
└── Theory/         # Metatheory for proofs
    ├── Equivalence.v
    └── Tactics.v

test/               # Testing OUR TOOLS
├── unit/
└── integration/

validation/         # Testing EXTERNAL BEHAVIOR
├── oracle/         # Differential testing
└── cross-check/    # Compare with MLIR toolchain

verify/             # Future: Automated verification
```

See [Directory Structure Design](../design/directory-structure.md) for details.

### Positive Consequences

1. **Pragmatic progress**: Oracle testing lets us move forward immediately
2. **Rigorous foundation**: Translation validation ensures soundness for critical passes
3. **Clear separation**: No confusion about what each test validates
4. **Incremental path**: Start with oracle, add formal proofs later
5. **Best of both worlds**: Empirical validation + formal guarantees

### Negative Consequences

1. **Dual infrastructure**: Must maintain both oracle tests and formal proofs
2. **Potential duplication**: Same passes tested both ways
3. **Coordination**: Oracle tests should inform formal verification priorities

### Mitigation Strategies

1. **Prioritize passes**: Not everything needs formal verification
   - High-value: SCCP, DCE, inlining
   - Lower priority: Canonicalization, peephole opts

2. **Test corpus reuse**: Oracle tests become specifications for proofs

3. **Automation**: Develop tactics to reduce proof burden

4. **Documentation**: Clear guidelines on when to use each approach

## Implementation Roadmap

### Phase 1: Oracle Testing Infrastructure (Current)
- ✅ Implement oracle test driver
- ✅ Add SCCP oracle tests
- ✅ Document oracle testing approach
- ⏳ Move to `validation/oracle/` directory

### Phase 2: Directory Restructuring
- Create `validation/` directory structure
- Move oracle tests from `test/`
- Update documentation

### Phase 3: Formal Verification Foundation
- Create `src/Pass/` directory
- Implement first pass in Coq (e.g., simple DCE)
- Prove correctness using ITree bisimulation
- Develop proof tactics library

### Phase 4: Scale Formal Verification
- Add more passes
- Extract verified optimizers
- Compare extracted vs mlir-opt (meta-validation!)

## Examples

### Oracle Test (validation/oracle/sccp/addi.mlir)
```ocaml
(* Run both programs, compare outputs *)
let original = run_interpreter "input.mlir" in
let optimized = run_mlir_opt "input.mlir" "sccp"
                |> run_interpreter in
assert (original = optimized)
```

### Translation Validation (src/Pass/SCCP_correct.v)
```coq
Theorem sccp_preserves_semantics :
  forall (prog : program),
    eutt (semantics prog) (semantics (sccp prog)).
Proof.
  (* Formal proof using ITree bisimulation *)
Qed.
```

## Alternatives Considered

### Alternative A: Oracle Testing Only
**Rejected**: No formal guarantees, only bug detection

**Pros**:
- Low effort
- Immediate results
- Easy to automate

**Cons**:
- No soundness guarantees
- Incomplete coverage
- Not aligned with project goals (formal semantics)

### Alternative B: Translation Validation Only
**Rejected**: Too ambitious upfront, blocks progress

**Pros**:
- Strongest guarantees
- Aligned with formal methods

**Cons**:
- High upfront cost
- Blocks near-term progress
- Requires mature proof infrastructure first

### Alternative C: Hybrid (CHOSEN)
**Rationale**: Balances pragmatism and rigor

## References

- [Alive2: Bounded Translation Validation for LLVM](https://github.com/AliveToolkit/alive2)
- [Vellvm: Formal Verification of LLVM IR](https://github.com/vellvm/vellvm)
- [Translation Validation: From Simulink to C](https://dl.acm.org/doi/10.1145/3385412.3386007)
- ADR-0001: Translation Validation Framework
- [Directory Structure Design](../design/directory-structure.md)
- [Oracle Testing Guide](../howto/translation-validation-testing.md)

## Decision Log

- **2025-11-06**: Initial draft and acceptance
- Discussion: Hybrid approach provides best balance for mlir-sem project goals
