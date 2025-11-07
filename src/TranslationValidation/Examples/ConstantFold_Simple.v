(** Simplified constant folding example to validate proof approach

    This file contains a minimal example to test our proof strategy
    before tackling the full SCCP_Simple.v proof.
*)

From Stdlib Require Import ZArith List String.
From ITree Require Import ITree Eq.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Interp.
From MlirSem Require Import Semantics.Denotation.
From MlirSem Require Import Semantics.Events.
From MlirSem Require Import TranslationValidation.Framework.
From MlirSem Require Import Utils.Tactics.
From MlirSem Require Import Utils.Lemmas.

Import ListNotations.
Open Scope Z_scope.
Open Scope string_scope.

(** Simplest possible example: constant vs constant *)
Definition prog_constant_direct : mlir_program :=
  [
    FuncOp "main" (FunctionType [] [(Integer 64)])
    [
      {| block_name := "entry";
         block_args := [];
         block_ops := [
           (Op ["%0"] (Arith_Constant 42 (Integer 64)));
           (Term (Func_Return ["%0"]))
         ] |}
    ]
  ].

Definition prog_constant_folded : mlir_program :=
  [
    FuncOp "main" (FunctionType [] [(Integer 64)])
    [
      {| block_name := "entry";
         block_args := [];
         block_ops := [
           (Op ["%result"] (Arith_Constant 42 (Integer 64)));
           (Term (Func_Return ["%result"]))
         ] |}
    ]
  ].

(** These two programs differ only in variable naming,
    which should not affect semantics *)
Theorem constant_renaming_equiv :
  prog_equiv prog_constant_direct prog_constant_folded.
Proof.
  unfold prog_equiv.
  intros func_name.

  (* Both programs have "main" *)
  destruct (string_dec func_name "main").
  - (* func_name = "main" *)
    subst.
    simpl.

    (* At this point, we need to show that two ITrees are equivalent.
       Both trees:
       1. Write a constant to a local variable
       2. Read that variable
       3. Return it

       The only difference is the variable name ("%0" vs "%result"),
       but the semantics should be the same.

       TODO: This requires reasoning about interpret and interp_state.
       For now, we admit this to validate the structure. *)
    admit.

  - (* func_name ≠ "main" *)
    simpl.
    destruct (string_dec func_name "main"); try contradiction.
    trivial.
Admitted.

(** Next: actual constant folding - computing at compile time *)
Definition prog_add_constants_unfolded : mlir_program :=
  [
    FuncOp "main" (FunctionType [] [(Integer 64)])
    [
      {| block_name := "entry";
         block_args := [];
         block_ops := [
           (Op ["%0"] (Arith_Constant 10 (Integer 64)));
           (Op ["%1"] (Arith_Constant 20 (Integer 64)));
           (Op ["%2"] (Arith_AddI "%0" "%1" (Integer 64)));
           (Term (Func_Return ["%2"]))
         ] |}
    ]
  ].

Definition prog_add_constants_folded : mlir_program :=
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

(** This is the core constant folding equivalence we want to prove *)
Theorem add_constants_equiv :
  prog_equiv prog_add_constants_unfolded prog_add_constants_folded.
Proof.
  unfold prog_equiv.
  intros func_name.

  destruct (string_dec func_name "main").
  - subst.
    simpl.

    (* The key insight:
       - Unfolded version: LocalWrite 10 → LocalWrite 20 → LocalRead 10 → LocalRead 20 → compute 30 → LocalWrite 30 → LocalRead 30 → return [30]
       - Folded version: LocalWrite 30 → LocalRead 30 → return [30]

       After interpret processes these, both should yield the same final state and result.

       TODO: Prove this using ITree reasoning about interp_state *)
    admit.

  - simpl.
    destruct (string_dec func_name "main"); try contradiction.
    trivial.
Admitted.

(**
   LESSONS LEARNED:

   1. We need lemmas about how `interpret` handles sequences of LocalWrite/LocalRead
   2. We need to reason about state changes in interp_state
   3. Key property: If we write then read the same variable, we get what we wrote
   4. Sequential execution: Effects don't interfere if they use different variables

   NEXT STEPS:

   1. Develop lemmas in Utils/Lemmas.v for LocalWrite/LocalRead behavior
   2. Prove properties about `interpret` and `interp_state`
   3. Build up tactics for handling these common patterns
   4. Then tackle SCCP_Simple.v with these tools
*)
