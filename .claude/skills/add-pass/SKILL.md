---
name: add-pass
description: Guide through verifying an MLIR optimization pass using Translation Validation - proving semantic equivalence between programs before and after optimization. Use when the user wants to verify an MLIR pass (e.g., sccp, constant folding, DCE) or add pass verification support.
---

# Verify MLIR Optimization Pass (Translation Validation)

This skill guides you through verifying MLIR optimization passes using **Translation Validation**. Rather than implementing passes in Coq, we prove that existing MLIR passes preserve semantics by showing equivalence between input and output programs.

## Overview

Translation Validation workflow:
1. Run MLIR pass on test program (`mlir-opt`)
2. Parse both input and output MLIR files into Coq ASTs
3. Prove semantic equivalence in Coq
4. Automate testing infrastructure

**Key Decision**: See ADR-0001 for why we chose Translation Validation over implementing passes in Coq.

## Workflow Steps

### Step 1: Select Target Pass

Ask the user:
- **Which pass to verify?** (e.g., `sccp`, `constant-fold`, `dce`, `cse`)
- **Which dialects does it operate on?** (e.g., arith, scf, func)
- **What transformations does it perform?**
- **Are there specific test cases or benchmarks?**

**First-time setup**: If this is the first pass being verified, prioritize `sccp` (Sparse Conditional Constant Propagation) as it's our reference implementation.

### Step 2: Set Up Test Cases

Create test MLIR files in `test/oracle/<pass_name>/`:

1. **Start simple**: Basic cases that clearly show the transformation
   ```mlir
   // test/oracle/sccp/constant_simple.mlir
   func.func @test_constant() -> i32 {
     %c1 = arith.constant 1 : i32
     %c2 = arith.constant 2 : i32
     %sum = arith.addi %c1, %c2 : i32
     return %sum : i32
   }
   // Expected after sccp: return constant 3
   ```

2. **Add complexity gradually**:
   - Conditional branches with constant conditions
   - Loop unrolling opportunities
   - Dead code scenarios
   - Inter-procedural cases (if applicable)

3. **Port from MLIR repository**: Find relevant tests in MLIR's official test suite and adapt them

**Organization**:
```
test/oracle/sccp/
├── constant_simple.mlir          # Basic constant folding
├── constant_branch.mlir          # Conditional with constant
├── constant_loop.mlir            # Loop with constant bounds
├── interprocedural.mlir          # Cross-function analysis
└── README.md                     # Document test coverage
```

### Step 3: Run MLIR Pass on Test Cases

Set up infrastructure to run `mlir-opt`:

1. **Manual testing first**:
   ```bash
   # Run pass on test file
   mlir-opt --sccp test/oracle/sccp/constant_simple.mlir \
     -o test/oracle/sccp/constant_simple.opt.mlir

   # Verify output is valid MLIR
   mlir-opt --verify-diagnostics test/oracle/sccp/constant_simple.opt.mlir
   ```

2. **Inspect transformations**:
   - Compare input vs output manually
   - Understand what the pass did
   - Document expected transformations

3. **Automate pass execution**:
   - Update `test/test_driver.ml` to invoke `mlir-opt`
   - Generate `.opt.mlir` files automatically
   - Store both versions for comparison

### Step 4: Ensure Parser Support

Verify that the parser handles both pre- and post-optimization MLIR:

1. **Check syntax coverage**:
   - Can parser handle all constructs in input file?
   - Can parser handle all constructs in output file?
   - Optimized code may use different syntax (e.g., folded constants)

2. **Update parser if needed** (`src/Syntax/`):
   - Add missing operation support
   - Handle dialect-specific attributes
   - Support SSA value representations

3. **Test parsing**:
   ```ocaml
   (* Test in OCaml extracted code *)
   let p_before = parse_mlir_file "constant_simple.mlir" in
   let p_after = parse_mlir_file "constant_simple.opt.mlir" in
   (* Both should parse successfully *)
   ```

### Step 5: Formalize Correctness Theorem

State the main equivalence theorem in Coq (`src/Theory/<PassName>Correct.v`):

```coq
(* src/Theory/SCCPCorrect.v *)

Require Import Syntax.Program.
Require Import Semantics.Denote.

(** Main correctness theorem for SCCP pass *)
Theorem sccp_preserves_semantics :
  forall (p_before p_after : program) (env : environment),
    (* Precondition: p_after is result of running sccp on p_before *)
    is_sccp_output p_before p_after ->
    (* Conclusion: Both programs have equivalent semantics *)
    eutt eq
      (denote_program p_after env)
      (denote_program p_before env).
```

**Key components**:

1. **Precondition** (`is_sccp_output`):
   - May be axiomatized initially
   - Could be refined to parse actual `mlir-opt` output
   - Documents relationship between input and output

2. **Equivalence relation**:
   - Use `eutt eq` for semantic equivalence (up to tau)
   - May need weaker equivalence for some passes
   - Document choice in theorem comments

3. **Environment handling**:
   - Prove equivalence holds for all environments
   - Or parameterize by specific initial states

### Step 6: Develop the Proof

Prove the correctness theorem in `src/Theory/<PassName>Correct.v`:

#### Common Proof Strategies

**Strategy 1: Case Analysis on Program Structure**

For simple passes that transform specific patterns:

```coq
Proof.
  intros p_before p_after env H_output.
  unfold is_sccp_output in H_output.
  (* Analyze structure of p_before and p_after *)
  destruct p_before; destruct p_after.
  (* Case analysis on transformation patterns *)
  - (* Constant folding case *)
    simpl. rewrite constant_fold_correct. reflexivity.
  - (* Constant propagation case *)
    simpl. rewrite constant_prop_correct. reflexivity.
  (* ... *)
Qed.
```

**Strategy 2: Simulation Relation**

For complex passes with multiple transformation rules:

```coq
(* Define simulation relation between states *)
Definition sccp_simulation (s1 s2 : state) : Prop :=
  (* s2 simulates s1 if... *)
  forall v, eval_expr s1 v = eval_expr s2 (sccp_transform v).

(* Prove simulation is preserved by execution *)
Lemma sccp_simulation_step :
  forall s1 s2,
    sccp_simulation s1 s2 ->
    forall s1', step s1 s1' ->
    exists s2', step s2 s2' /\ sccp_simulation s1' s2'.

(* Use simulation to prove main theorem *)
Theorem sccp_preserves_semantics : ...
Proof.
  apply simulation_implies_equivalence.
  apply sccp_simulation_step.
Qed.
```

**Strategy 3: Lemma-Driven Approach**

Break proof into manageable pieces:

```coq
(* Prove equivalence for individual operations *)
Lemma sccp_constant_ops_equiv :
  forall c, denote_constant c ≈ denote_constant (sccp_fold_constant c).

Lemma sccp_branch_ops_equiv :
  forall b, denote_branch b ≈ denote_branch (sccp_simplify_branch b).

(* Compose lemmas for full proof *)
Theorem sccp_preserves_semantics : ...
Proof.
  induction p_before; simpl.
  - apply sccp_constant_ops_equiv.
  - apply sccp_branch_ops_equiv.
  - (* ... *)
Qed.
```

#### Proof Development Tips

1. **Start with specific examples**:
   - Prove equivalence for one test case first
   - Generalize once pattern is clear

2. **Use automation**:
   - Develop custom tactics in `Utils/Tactics.v`
   - Automate repetitive case analyses

3. **Reusable lemmas**:
   - Store in `src/Theory/<PassName>Lemmas.v`
   - Lemmas about constant folding, propagation rules, etc.

4. **Iterative refinement**:
   - Start with partial proofs (use `admit` for hard parts)
   - Document assumptions and TODOs
   - Refine incrementally

### Step 7: Integrate with Test Infrastructure

Update test harness (`test/test_driver.ml`):

1. **Pass execution**:
   ```ocaml
   let run_pass pass_name input_file =
     (* Run mlir-opt *)
     let cmd = Printf.sprintf "mlir-opt --%s %s" pass_name input_file in
     let output = Unix.open_process_in cmd in
     (* Parse output *)
     parse_mlir_string (read_all output)
   ```

2. **Equivalence checking**:
   ```ocaml
   let test_pass_correctness pass_name test_file =
     let p_before = parse_mlir_file test_file in
     let p_after = run_pass pass_name test_file in
     (* Check semantic equivalence *)
     assert (semantically_equivalent p_before p_after)
   ```

3. **Test discovery**:
   ```ocaml
   let discover_pass_tests pass_name =
     (* Find all .mlir files in test/oracle/<pass_name>/ *)
     find_files (Printf.sprintf "test/oracle/%s/*.mlir" pass_name)
   ```

4. **Dune integration**:
   ```lisp
   ; test/dune
   (test
    (name test_sccp)
    (deps ../src/extraction/mlir_interp.exe)
    (action (run ./test_driver.exe sccp)))
   ```

### Step 8: Create Documentation

Document the verified pass:

1. **Design Document** (`docs/design/<pass_name>-verification.md`):
   ```markdown
   # SCCP Pass Verification

   ## Pass Overview
   - What transformations does SCCP perform?
   - Which dialects are involved?

   ## Verification Approach
   - Proof strategy used
   - Key lemmas and insights
   - Challenges encountered

   ## Test Coverage
   - List of test cases
   - Coverage metrics
   - Known limitations

   ## Examples
   [Show before/after examples]
   ```

2. **Update ADR-0001** if verification revealed insights about the framework

3. **Proof documentation**:
   - Add Coqdoc comments to proof files
   - Explain non-obvious steps
   - Reference relevant MLIR documentation

### Step 9: Validate and Review

Final checks before completion:

1. **CI verification**:
   ```bash
   # All tests pass
   dune test

   # Proofs verify
   dune build @check

   # Extraction works
   dune build @extract
   ```

2. **Coverage analysis**:
   - What percentage of pass behavior is covered?
   - Are edge cases tested?
   - Document known limitations

3. **Code review**:
   - Request GitHub Copilot review
   - Check proof clarity and maintainability
   - Verify documentation completeness

## Definition of Done Checklist

Before marking pass verification complete:

- [ ] Test cases created in `test/oracle/<pass_name>/`
- [ ] Parser handles both pre- and post-optimization syntax
- [ ] Correctness theorem stated in `src/Theory/<PassName>Correct.v`
- [ ] Theorem **proven** (or partial proof with documented TODOs)
- [ ] Key lemmas in `src/Theory/<PassName>Lemmas.v`
- [ ] Test infrastructure updated (`test/test_driver.ml`, `test/dune`)
- [ ] All tests passing (`dune test`)
- [ ] Design documentation in `docs/design/`
- [ ] Proof documentation with Coqdoc comments
- [ ] CI passing
- [ ] Code reviewed

## Example: SCCP Pass Verification

### Test Case

```mlir
// test/oracle/sccp/constant_simple.mlir
func.func @test() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return %sum : i32
}
```

### After SCCP

```mlir
// Generated by: mlir-opt --sccp constant_simple.mlir
func.func @test() -> i32 {
  %c3 = arith.constant 3 : i32
  return %c3 : i32
}
```

### Correctness Theorem

```coq
(* Both programs return 3 for all environments *)
Theorem constant_simple_equiv :
  forall env,
    denote_program p_before env ≈ denote_program p_after env.
Proof.
  intros. unfold p_before, p_after.
  simpl. (* Simplify denotations *)
  (* LHS: denote(1) + denote(2) *)
  (* RHS: denote(3) *)
  rewrite constant_add_folds. (* 1 + 2 = 3 *)
  reflexivity.
Qed.
```

## Common Challenges

### Challenge 1: Parser Limitations

**Problem**: Parser doesn't handle optimized MLIR syntax

**Solution**:
- Extend parser incrementally as needed
- Test parser on actual `mlir-opt` output
- Consider using MLIR's C API for parsing (future work)

### Challenge 2: Complex Transformations

**Problem**: Pass performs many different optimizations

**Solution**:
- Break into smaller lemmas (one per transformation)
- Prove lemmas independently
- Compose for main theorem

### Challenge 3: Axiomatized Preconditions

**Problem**: `is_sccp_output` is axiomatized, not verified

**Solution**:
- Acceptable for translation validation
- We trust `mlir-opt` to correctly apply the pass
- Focus verification effort on semantic equivalence
- Document this assumption clearly

### Challenge 4: Proof Complexity

**Problem**: Equivalence proof is too complex

**Solution**:
- Start with simplest test cases
- Build library of reusable tactics
- Consider weaker equivalence relations if appropriate
- Break into multiple theorems (one per test case initially)

## Integration with Other Workflows

Translation Validation complements:

- **Differential Testing** (Epic #8): Runtime validation of pass behavior
- **Property Testing**: QuickChick can generate test programs
- **Golden Tests**: Compare with MLIR toolchain output

## Scaling to Multiple Passes

After verifying the first pass (SCCP):

1. **Identify common patterns**:
   - Reusable proof tactics
   - Common lemmas (constant folding, dead code, etc.)
   - Test infrastructure patterns

2. **Build proof library**:
   - `src/Theory/PassCommon.v`: Shared definitions
   - `src/Theory/PassTactics.v`: Reusable tactics
   - `test/PassTestLib.ml`: Common test utilities

3. **Verify more passes**:
   - Constant folding
   - Dead code elimination
   - Common subexpression elimination
   - Loop optimizations

4. **Pass pipelines**:
   - Prove correctness of pass sequences
   - `sccp → dce → cse` preserves semantics

## References

- **ADR-0001**: Translation Validation Framework decision
- **MLIR Pass Documentation**: https://mlir.llvm.org/docs/Passes/
- **Vellvm**: Translation validation for LLVM (similar approach)
- **GitHub Issues**:
  - Epic #3: SCCP pass verification
  - Task #4: Test infrastructure
  - Task #5: Formalize correctness theorem
  - Task #6: Port MLIR tests

## After Completion

Once a pass is verified:

1. **Announce success**: Update project board and team
2. **Gather metrics**: LOC, proof time, coverage percentage
3. **Write experience report**: What worked? What didn't?
4. **Plan next pass**: Apply lessons learned
5. **Consider extraction**: Could we extract a verified validator?
