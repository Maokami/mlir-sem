(** Proof by computation for simple constant folding

    Strategy: For simple examples without loops or function calls,
    we can use Coq's computation to evaluate the programs and show
    they produce the same result.
*)

From Stdlib Require Import String ZArith List.
From ITree Require Import ITree Eq.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Values Semantics.Events.
From MlirSem Require Import Semantics.Interp Semantics.Denotation.
From MlirSem Require Import TranslationValidation.Framework.
From MlirSem Require Import Utils.Tactics.

Import ListNotations.
Open Scope Z_scope.
Open Scope string_scope.

(** Simplest example: 10 + 20 vs 30 *)

Definition prog_simple_add : mlir_program :=
  [
    FuncOp "main" (FunctionType [] [(Integer 64)])
    [
      {| block_name := "entry";
         block_args := [];
         block_ops := [
           (Op ["%a"] (Arith_Constant 10 (Integer 64)));
           (Op ["%b"] (Arith_Constant 20 (Integer 64)));
           (Op ["%c"] (Arith_AddI "%a" "%b" (Integer 64)));
           (Term (Func_Return ["%c"]))
         ] |}
    ]
  ].

Definition prog_simple_folded : mlir_program :=
  [
    FuncOp "main" (FunctionType [] [(Integer 64)])
    [
      {| block_name := "entry";
         block_args := [];
         block_ops := [
           (Op ["%result"] (Arith_Constant 30 (Integer 64)));
           (Term (Func_Return ["%result"]))
         ] |}
    ]
  ].

(** Let's examine what run_program produces *)
Definition tree_add := run_program prog_simple_add "main".
Definition tree_folded := run_program prog_simple_folded "main".

(** Both should be Some *)
Example both_programs_have_main :
  tree_add <> None /\ tree_folded <> None.
Proof.
  unfold tree_add, tree_folded.
  split; discriminate.
Qed.

(** Extract the trees *)
Definition get_tree (opt : option (itree MlirSemE (interpreter_state * list mlir_value))) :
  itree MlirSemE (interpreter_state * list mlir_value) :=
  match opt with
  | Some t => t
  | None => Ret ({| prog_ctx := ZStringMap.empty _; call_stack := [] |}, [])
  end.

Definition t_add := get_tree tree_add.
Definition t_folded := get_tree tree_folded.

(**
   KEY INSIGHT:

   These trees still have MlirSemE effects (though not LocalE, which was interpreted).
   To prove eutt, we need to show they produce the same observations.

   For programs without FunctionE, ControlE, or FailureE events,
   the trees should reduce to Ret values.

   Let's try a different approach: prove a weaker property first.
*)

(** Weaker property: Both programs terminate without errors *)
(** This would require reasoning about the absence of FailureE events *)

(**
   ALTERNATIVE: Use simulation/bisimulation

   We can prove that every step in one program corresponds to a step
   in the other, maintaining the invariant that they will produce the same result.

   But this still requires deep reasoning about ITree semantics.

   MOST PRACTICAL APPROACH FOR NOW:

   1. Admit the equivalence for this simple case
   2. Document what we would need to complete the proof
   3. Use this as a template for understanding the proof obligations
   4. Build up the necessary lemmas over time
*)

Axiom interp_state_preserves_eutt :
  forall {E1 E2 S R} (h : E1 ~> Monads.stateT S (itree E2))
         (t1 t2 : itree E1 R) (s : S),
    eutt eq t1 t2 ->
    eutt eq (interp_state h t1 s) (interp_state h t2 s).

Axiom denote_func_deterministic :
  forall (f1 f2 : mlir_func),
    (* If f1 and f2 have the same structure but different var names *)
    (* and compute the same arithmetic result *)
    (* then their denotations are eutt *)
    True. (* TODO: state this properly *)

(**
   ROADMAP TO COMPLETING PROOFS:

   1. Prove lemmas about LocalE handler preserving equivalence
   2. Prove that arithmetic operations are deterministic
   3. Prove that variable renaming doesn't affect semantics
   4. Prove that constant folding preserves the computed value

   For (4), the key lemma would be:

   Lemma constant_fold_correct :
     forall (ops_original ops_folded : list operation)
            (original_computes : denote_block ops_original ≈ ... Ret val ...)
            (folded_computes : denote_block ops_folded ≈ ... Ret val ...),
       interpret (denote_block ops_original) s ≈
       interpret (denote_block ops_folded) s.
*)

Theorem simple_add_equiv :
  prog_equiv prog_simple_add prog_simple_folded.
Proof.
  unfold prog_equiv.
  intros func_name.
  destruct (string_dec func_name "main").
  - (* main function *)
    subst.
    simpl.

    (* At this point, both trees are complex ITree expressions
       involving interp_state, denote_func, etc.

       To complete this proof, we would need to:
       1. Unfold all definitions
       2. Apply lemmas about interp_state
       3. Reason about the state transformations
       4. Show both produce Ret (_, [IntVal 30])

       This is a substantial proof effort. *)

    admit.

  - (* not main *)
    simpl.
    destruct (string_dec func_name "main"); try contradiction.
    trivial.

Admitted.

(**
   LESSON LEARNED:

   The proof is conceptually straightforward but technically challenging.
   We need infrastructure:

   1. Lemmas about interp_state behavior
   2. Lemmas about LocalE read/write semantics
   3. Tactics for automatic unfolding and simplification
   4. Possibly computational reflection to evaluate simple cases

   NEXT STEPS:

   1. Build up Utils/InterpLemmas.v with concrete, provable lemmas
   2. Develop tactics in Utils/Tactics.v for common patterns
   3. Start with the absolute simplest case (just return a constant)
   4. Gradually build up to addition, then to SCCP

   Or: Accept that full formal proofs will take time, and focus on:
   - Oracle testing (already have)
   - Partial proofs with admitted lemmas (document what's needed)
   - Build the framework for future complete proofs
*)
