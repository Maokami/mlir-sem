(*
 * Common Tactics for MLIR-Sem Proofs
 *
 * This file provides reusable tactics for reasoning about ITrees,
 * semantic equivalence, and program transformations.
 *)

From Stdlib Require Import ZArith List String.
From ITree Require Import ITree.
From Paco Require Import paco.

Import ListNotations.
Import ITreeNotations.

Local Open Scope itree_scope.

(* ========================================================================= *)
(* Basic Simplification Tactics *)
(* ========================================================================= *)

(*
 * inv H: Inversion on hypothesis H, introducing all new hypotheses
 * and equalities, then substituting and clearing.
 *)
Ltac inv H := inversion H; subst; clear H.

(*
 * destr E: Destruct E with automatic simplification.
 * Cleans up trivial goals and simplifies equalities.
 *)
Ltac destr E :=
  destruct E; try discriminate; try congruence; auto.

(*
 * simpl_goal: Simplify the goal using standard tactics.
 *)
Ltac simpl_goal :=
  simpl; try reflexivity; try congruence; auto.

(* ========================================================================= *)
(* ITree-Specific Tactics *)
(* ========================================================================= *)

(*
 * itree_simp: Main ITree simplification tactic.
 * Combines unfolding and standard simplification.
 *)
Ltac itree_simp :=
  unfold ITree.bind, ITree.map, ITree.trigger in *;
  simpl;
  try reflexivity;
  auto.

(* ========================================================================= *)
(* Case Analysis Tactics *)
(* ========================================================================= *)

(*
 * case_val v: Destruct a value and simplify.
 *)
Ltac case_val v :=
  destruct v; simpl; try reflexivity; auto.

(* ========================================================================= *)
(* Forward Reasoning *)
(* ========================================================================= *)

(*
 * forward H: Apply hypothesis H and introduce the result.
 * Useful for chaining lemmas.
 *)
Ltac forward H :=
  match type of H with
  | ?P -> ?Q =>
      let H' := fresh in
      assert (H' : P); [| specialize (H H'); clear H']
  end.

(* ========================================================================= *)
(* List Reasoning *)
(* ========================================================================= *)

(*
 * list_induction l: Set up induction on list l with sensible names.
 *)
Ltac list_induction l :=
  induction l as [| x xs IHxs]; simpl; auto.

(* ========================================================================= *)
(* Automation Configuration *)
(* ========================================================================= *)

(*
 * Create hints database for MLIR-Sem proofs.
 *)
Create HintDb mlir_sem discriminated.

(*
 * Hint: Common reflexivity and congruence lemmas.
 *)
#[global] Hint Resolve Z.eq_dec : mlir_sem.
#[global] Hint Resolve String.string_dec : mlir_sem.
#[global] Hint Resolve eq_refl : mlir_sem.

(*
 * mlir_auto: Main automation tactic for MLIR-Sem proofs.
 * Combines standard automation with domain-specific tactics.
 *)
Ltac mlir_auto :=
  simpl_goal;
  try itree_simp;
  auto with mlir_sem.
