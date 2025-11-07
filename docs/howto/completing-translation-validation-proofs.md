# How to Complete Translation Validation Proofs

This guide documents the strategy and infrastructure needed to complete formal proofs of MLIR optimization pass correctness using Translation Validation.

## Overview

**Goal**: Prove that an optimized MLIR program has the same semantics as the original program.

**Formal Statement**: `prog_equiv program_before program_after`

```coq
Definition prog_equiv (p1 p2 : mlir_program) : Prop :=
  forall func_name,
    match run_program p1 func_name, run_program p2 func_name with
    | Some t1, Some t2 => eutt Logic.eq t1 t2
    | None, None => True
    | _, _ => False
    end.
```

## Current Status (as of 2025-11-08)

### ✅ Completed

1. **Framework Infrastructure** (`src/TranslationValidation/Framework.v`)
   - `prog_equiv` definition
   - `block_equiv` definition
   - Basic tactics: `tv_simp`, `tv_intro`, `tv_step`

2. **Example Proofs** (`src/TranslationValidation/Examples/`)
   - `ConstantFold_Addi.v` - Original example with admitted proof
   - `ConstantFold_Simple.v` - Simplified examples showing proof structure
   - `ConstantFold_Compute.v` - Computational approach exploration

3. **SCCP Simple Case** (`src/TranslationValidation/SCCP_Simple.v`)
   - `sccp_simple_correct` theorem stated
   - Helper lemmas `before_computes_130` and `after_computes_130` identified
   - Proof structure documented
   - Missing infrastructure clearly identified

4. **Utility Lemmas** (`src/Utils/`)
   - `Lemmas.v` - Basic value and list lemmas
   - `InterpLemmas.v` - Skeleton for interpreter reasoning
   - `DenotationLemmas.v` - Basic denotation properties

### 🔄 In Progress

**Missing Infrastructure** - The key blocker for completing proofs:

1. **Interpreter Lemmas** (`src/Utils/InterpLemmas.v`)
   - How `interp_state` handles LocalE events
   - Read/write semantics for local variables
   - Composition properties

2. **ITree Reasoning Tactics**
   - Stepping through interpreted programs
   - Automatic handling of bind sequences
   - State manipulation patterns

## Proof Strategy

### High-Level Approach

For a simple constant folding optimization like SCCP:

```
Original:  %0 = const 10; %1 = const 20; %2 = add %0, %1; return %2
Optimized: %result = const 30; return %result
```

**Proof Structure**:

1. **Show both programs execute without errors**
   - No FailureE events triggered
   - All operations are well-defined

2. **Show both programs produce the same result**
   - Original: computes 10 + 20 = 30 at runtime
   - Optimized: returns pre-computed 30
   - Both yield `[IntVal 30]`

3. **State differences don't matter**
   - Different intermediate local variables
   - But same final return value

### Detailed Steps

```coq
Theorem constant_fold_correct :
  prog_equiv prog_original prog_optimized.
Proof.
  unfold prog_equiv.
  intros func_name.
  destruct (string_dec func_name "main").

  (* Step 1: Both programs have "main" function *)
  - subst. simpl.

    (* Step 2: Show equivalence of interpreted ITrees *)
    (* Need: original_computes and optimized_computes lemmas *)

    (* Step 3: Apply transitivity if needed *)
    (* Both ≈ Ret (_, [IntVal result]) *)

    admit.

  (* Step 4: Handle non-existent functions *)
  - simpl. destruct (string_dec ...); trivial.
Qed.
```

### Helper Lemmas Pattern

For each program, prove a computational lemma:

```coq
Lemma original_computes :
  forall (s0 : interpreter_state),
    exists (s_final : interpreter_state),
      interpret (denote_func original_main) s0 ≈
      Ret (s_final, [IntVal expected_result]).
```

**Proof of Helper Lemma**:

1. Unfold `denote_func`, `denote_block`, `denote_general_op`
2. For each operation:
   - Show how LocalWrite modifies the state
   - Show how LocalRead retrieves the value
   - Show how arithmetic computes the result
3. Combine using bind associativity
4. Show final return value

## Missing Infrastructure

### 1. Interpreter State Lemmas

**File**: `src/Utils/InterpLemmas.v`

**Needed Lemmas**:

```coq
(* Ret doesn't change state *)
Lemma interp_ret : forall s v,
  interp_state h (Ret v) s = Ret (s, v).

(* Write then read same variable *)
Lemma write_read_same : forall s var val,
  interp_state h (
    trigger (LocalWrite var val) ;;
    trigger (LocalRead var)
  ) s ≈
  interp_state h (
    trigger (LocalWrite var val) ;;
    Ret val
  ) s.

(* Bind associativity under interp_state *)
Lemma interp_bind_assoc : forall {A B C} (m : itree E A) (f : A -> itree E B) (g : B -> itree E C) s,
  interp_state h (x <- m ;; f x) s >>= (fun '(s', x) => interp_state h (g x) s') ≈
  interp_state h (x <- m ;; y <- f x ;; g y) s.
```

**References to Study**:
- `ITree.Interp.InterpFacts` - Existing ITree interpretation lemmas
- `ITree.Events.State` - State-specific reasoning
- `Paco` library - For coinductive reasoning with `eutt`

### 2. Denotation Simplification Lemmas

**File**: `src/Utils/DenotationLemmas.v`

**Needed Lemmas**:

```coq
(* Arithmetic is deterministic *)
Lemma addi_deterministic : forall lhs rhs lval rval ty,
  (* Under environment with lhs=lval, rhs=rval *)
  denote_general_op (Arith_AddI lhs rhs ty) ≈
  (* ... produces lval + rval ... *)

(* Constants are pure *)
Lemma constant_pure : forall val ty,
  denote_general_op (Arith_Constant val ty) ≈
  Ret [IntVal val].
```

### 3. Proof Automation Tactics

**File**: `src/Utils/Tactics.v` (extend existing)

**Needed Tactics**:

```coq
(* Step through one operation in denote_block *)
Ltac denote_step :=
  unfold denote_block, denote_general_op;
  simpl;
  (* Handle common patterns *)
  ...

(* Automatically simplify arithmetic *)
Ltac arith_compute :=
  repeat match goal with
  | [ |- context[IntVal ?x + IntVal ?y] ] =>
      replace (IntVal x + IntVal y) with (IntVal (x + y))
  end.

(* Handle LocalWrite/LocalRead sequences *)
Ltac local_state_solve :=
  repeat match goal with
  | [ |- context[LocalWrite ?v ?val ;; LocalRead ?v] ] =>
      rewrite write_read_same
  end.
```

## Recommended Workflow

### Phase 1: Build Core Lemmas (Current Priority)

1. **Start with simplest case**: Prove `interp_ret`
   - Study ITree library documentation
   - Look at existing `interp_state` proofs
   - Get this ONE lemma to `Qed.`

2. **Build up incrementally**: Once `interp_ret` works:
   - Prove `interp_constant` (constants don't change state)
   - Prove `write_read_same` (write then read)
   - Each lemma builds on the previous

3. **Validate with simple example**:
   - Use `ConstantFold_Simple.v` test cases
   - Try to complete ONE trivial proof end-to-end
   - Learn from any issues

### Phase 2: Develop Tactics

1. **Extract common patterns** from Phase 1 proofs
2. **Create tactics** to automate repetitive steps
3. **Test tactics** on multiple examples

### Phase 3: Scale to Complex Cases

1. **Complete SCCP_Simple.v**:
   - Use lemmas from Phase 1
   - Use tactics from Phase 2
   - Document any new patterns

2. **Generalize** to other passes (DCE, etc.)

## Alternative: Pragmatic Approach

If full formal proofs remain too complex:

1. **Use admitted lemmas** with clear documentation
2. **Rely on oracle testing** for practical validation
3. **Focus framework development** on:
   - Clear proof structure
   - Identified missing lemmas
   - Tactics for future completion

This is acceptable for a research project and demonstrates:
- Understanding of the problem
- Clear path to completion
- Working validation framework (oracle tests)

## Learning Resources

### ITree Documentation

- [ITree Tutorial](https://github.com/DeepSpec/InteractionTrees/blob/master/tutorial/Introduction.v)
- [ITree Examples](https://github.com/DeepSpec/InteractionTrees/tree/master/examples)
- Key modules:
  - `ITree.Eq` - Equivalence relations
  - `ITree.Interp` - Interpretation
  - `ITree.Events.State` - State effects

### Related Projects

- **Vellvm**: Study `src/coq/Theory/` for LLVM IR proof patterns
- **CompCert**: Classic certified compiler, different approach but good tactics
- **DeepSpec**: General ITree reasoning patterns

### Coq Tactics

- **Paco**: For coinductive proofs with `eutt`
- **Setoid rewriting**: For rewriting under equivalence relations
- **Computational reflection**: For arithmetic simplification

## Summary

**Current Achievement**: We have:
1. ✅ Clear framework structure
2. ✅ Identified proof obligations
3. ✅ Documented missing infrastructure
4. ✅ Working oracle tests as validation

**Next Critical Step**: Prove ONE simple lemma completely (suggest: `interp_ret`)

**Long-term Goal**: Build up infrastructure incrementally until SCCP_Simple.v is complete without admits

**Timeline Estimate**:
- Phase 1 (Core lemmas): 2-4 weeks
- Phase 2 (Tactics): 1-2 weeks
- Phase 3 (SCCP completion): 1-2 weeks

**Resources Needed**:
- ITree library expertise
- Paco/coinduction knowledge
- Time for deep Coq work

---

*Document created: 2025-11-08*
*See also*: `docs/adr/ADR-0001-translation-validation-framework.md`, `docs/adr/ADR-0002-hybrid-validation-strategy.md`
