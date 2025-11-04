# ADR-0001: Translation Validation Framework

**Status:** Proposed

## Context

The project's primary goal is to create a formal semantics for MLIR and use it to verify compiler optimizations. The initial proposal was to implement certified passes directly in Coq. However, after discussion, a **Translation Validation** approach was chosen as a more practical and scalable strategy. This allows us to leverage the existing, highly-optimized passes in the official MLIR toolchain (`mlir-opt`) and verify their behavior without re-implementing them.

This decision aligns with the successful methodology of projects like Vellvm for LLVM.

## Decision

We will build a framework to perform translation validation on MLIR optimization passes. The core idea is to prove that the semantics of an MLIR program are preserved after a pass is applied.

**The workflow will be:**

1.  **Input:** An MLIR program (`program.mlir`).
2.  **Transformation:** Apply a specific MLIR pass using an external tool (e.g., `mlir-opt --sccp`) to produce an optimized program (`program.opt.mlir`).
3.  **Parsing:** Parse both `program.mlir` and `program.opt.mlir` into their respective Coq AST representations within our framework.
4.  **Verification:** Prove in Coq that the denotational semantics of the two ASTs are equivalent. The central theorem will be of the form:
    ```coq
    Theorem pass_correct : forall p_before p_after,
      is_pass_output("sccp", p_before, p_after) ->
      denote_program(p_before) = denote_program(p_after).
    ```

Our first target for this framework will be the **Sparse Conditional Constant Propagation (`sccp`)** pass.

## Consequences

*   **Test Infrastructure (`test/`):**
    *   The test driver (`test_driver.ml`) and `dune` configuration must be updated.
    *   The test harness will need to be able to invoke `mlir-opt` on a given input file.
    *   It will then need to parse both the original and the optimized file and feed them to a new set of test cases designed for semantic equivalence checks.

*   **Coq Framework (`src/`):**
    *   **Syntax/Parsing:** The parser must be robust enough to handle the syntactic variations between pre- and post-optimization MLIR code.
    *   **Semantics:** The existing denotational semantics will be the foundation for the equivalence proofs.
    *   **Theory (`src/Theory/Correctness.v`):** This file will contain the formal statements and proofs of semantic equivalence for the validated passes.

*   **Long-Term Goal:**
    *   This framework will be the basis for verifying a wide range of MLIR passes.
    *   The test suite will evolve to incorporate more examples, with the eventual goal of porting relevant tests from the official MLIR repository to ensure broad coverage.

*   **Alternative (Rejected):**
    *   We will not be implementing optimization passes directly in Coq (i.e., creating certified passes) at this time. While valuable, that approach is less scalable for verifying a large, existing compiler like MLIR.
