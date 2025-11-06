# ADR-0001: Translation Validation Framework for MLIR Passes

**Status**: Accepted

**Date**: 2025-11-04

**Authors**: Maokami

## Context

The project's primary goal is to create a formal semantics for MLIR and use it to verify compiler optimizations. Two main approaches were considered for achieving this goal:

1. **Certified Pass Implementation**: Implement optimization passes directly in Coq, prove them correct, and extract them to executable code
2. **Translation Validation**: Verify existing MLIR passes by proving semantic equivalence between input and output programs

### Problem Statement

MLIR is a large, complex compiler infrastructure with numerous optimization passes. Each pass has been carefully optimized for performance over many years. The question is: how can we formally verify these passes?

### Constraints

- MLIR passes are highly optimized and battle-tested
- Re-implementing passes in Coq would be time-consuming and may not match performance
- The MLIR codebase evolves rapidly with new passes and optimizations
- We need a scalable approach that can verify multiple passes

### Requirements

- Formal verification of MLIR optimization passes
- Leverage existing MLIR toolchain (`mlir-opt`)
- Scalable to multiple passes and dialects
- Practical and maintainable approach

## Decision

We will build a **Translation Validation framework** to verify MLIR optimization passes. This approach proves that the semantics of an MLIR program are preserved after a pass is applied, without re-implementing the pass itself.

**Workflow:**

1. **Input**: An MLIR program (`program.mlir`)
2. **Transformation**: Apply a specific MLIR pass using external tool (e.g., `mlir-opt --sccp`) to produce optimized program (`program.opt.mlir`)
3. **Parsing**: Parse both `program.mlir` and `program.opt.mlir` into Coq AST representations
4. **Verification**: Prove in Coq that the denotational semantics of the two ASTs are equivalent

**Central Theorem:**

```coq
Theorem pass_correct : forall p_before p_after,
  is_pass_output("sccp", p_before, p_after) ->
  denote_program(p_before) = denote_program(p_after).
```

**First Target**: Sparse Conditional Constant Propagation (`sccp`) pass

## Alternatives Considered

### Alternative 1: Certified Pass Implementation in Coq

**Description**: Implement optimization passes directly in Coq, prove them correct, extract to OCaml, and use the extracted passes in a verified compiler pipeline.

**Pros**:
- End-to-end verification from specification to implementation
- Extracted passes are provably correct by construction
- No dependency on external unverified tools
- Aligns with projects like CompCert

**Cons**:
- Requires re-implementing every pass from scratch
- Extracted code may not match MLIR's performance
- High development cost for each pass
- Difficult to keep up with MLIR's rapid evolution
- Cannot leverage MLIR's existing, battle-tested implementations

**Why not chosen**: While valuable for critical passes, this approach does not scale well for verifying a large, existing compiler infrastructure like MLIR. The development effort would be prohibitive, and we would lose the benefit of MLIR's highly optimized implementations.

### Alternative 2: Hybrid Approach (Certified + Translation Validation)

**Description**: Use certified passes for critical optimizations and translation validation for others.

**Pros**:
- Best of both worlds for different scenarios
- Flexibility in verification strategy
- Can extract verified passes when needed

**Cons**:
- More complex framework to maintain
- Duplicated effort for some passes
- Unclear when to use which approach

**Why not chosen**: We decided to focus on translation validation first. If specific passes require certified implementation in the future, we can add that capability as needed. Starting with a single, focused approach reduces complexity.

### Alternative 3: Testing-Only Approach (No Formal Verification)

**Description**: Use differential testing and property-based testing without formal proofs.

**Pros**:
- Lower implementation effort
- Can test more scenarios quickly
- Practical bug-finding tool

**Cons**:
- No formal guarantees of correctness
- Cannot prove absence of bugs
- Does not achieve project's goal of formal verification

**Why not chosen**: While testing is valuable (and we will use differential testing as a complementary technique), it does not provide the formal guarantees that are the project's primary goal.

## Consequences

### Positive

- **Leverage existing MLIR infrastructure**: Use highly optimized, battle-tested passes
- **Scalable verification**: Can verify many passes without re-implementation
- **Practical approach**: Aligns with successful projects like Vellvm for LLVM
- **Focus on semantics**: Effort goes into formal semantics definition, not pass implementation
- **Easier maintenance**: MLIR toolchain handles pass updates

### Negative

- **Dependency on external tools**: Requires `mlir-opt` and MLIR toolchain
- **Per-example verification**: Must prove equivalence for each test case (not a universal proof)
- **Parser complexity**: Must handle syntactic variations in optimized code
- **No extracted passes**: Cannot generate verified executable passes

### Neutral

- **Different verification guarantees**: Proves specific instances rather than universal pass correctness
- **Complementary to testing**: Translation validation + differential testing provide strong assurance
- **Incremental approach**: Can add certified passes later if needed

## Implementation Notes

### Test Infrastructure (`test/`)

- Update `test_driver.ml` to invoke `mlir-opt` on input files
- Support pairs of MLIR files (before/after optimization)
- Parse both original and optimized files
- Feed to semantic equivalence test cases

**Key files affected**:
- `test/test_driver.ml`
- `test/dune`
- `test/oracle/*.mlir` (test cases)

### Coq Framework (`src/`)

**Syntax/Parsing** (`src/Syntax/`):
- Parser must robustly handle pre- and post-optimization MLIR variations
- Support all constructs used by target passes

**Semantics** (`src/Semantics/`):
- Existing denotational semantics is foundation for equivalence proofs
- May need to refine semantics as we encounter edge cases

**Theory** (`src/Theory/`):
- `Correctness.v`: Formal statements and proofs of semantic equivalence
- `PassLemmas.v`: Reusable lemmas for pass verification
- Per-pass proof files (e.g., `SCCPCorrect.v`)

### Long-Term Goal

- Verify a wide range of MLIR passes using this framework
- Port relevant tests from official MLIR repository for broad coverage
- Build library of reusable proof patterns for pass verification
- Consider differential testing framework as complementary validation (see Epic #8)

## References

- Related GitHub Issues:
  - Epic #3: Implement Translation Validation for `sccp` pass
  - Task #4: Set up test infrastructure for translation validation
  - Task #5: Formalize the `sccp` correctness theorem in Coq
  - Task #6: Port a simple constant propagation test from MLIR repo
- Related Projects:
  - [Vellvm](https://github.com/vellvm/vellvm): Translation validation for LLVM
  - [CompCert](https://compcert.org/): Certified compiler (alternative approach)
- MLIR Documentation:
  - [SCCP Pass](https://mlir.llvm.org/docs/Passes/#-sccp-sparse-conditional-constant-propagation)

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2025-11-04 | Maokami | Initial version |
| 2025-11-06 | Claude Code | Reformatted to standard ADR template |
