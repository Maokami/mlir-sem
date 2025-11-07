(*
 * Common Tactics for MLIR-Sem Proofs
 *
 * This file provides reusable tactics for reasoning about ITrees,
 * semantic equivalence, and program transformations.
 *)

From Stdlib Require Import ZArith List String Lia.
From ITree Require Import ITree Eq.
From Paco Require Import paco.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Denotation.

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

(*
 * unfold_denote: Unfold all denotation functions in the goal.
 * Useful for exposing the ITree structure of programs.
 *)
Ltac unfold_denote :=
  unfold denote_func, denote_block,
         denote_general_op, denote_terminator in *;
  simpl.

(*
 * eutt_refl: Reflexivity for eutt (equivalence up to taus).
 *)
Ltac eutt_refl :=
  reflexivity.

(*
 * eutt_sym: Apply symmetry of eutt.
 *)
Ltac eutt_sym :=
  symmetry.

(*
 * eutt_trans H: Apply transitivity of eutt using hypothesis H.
 *)
Ltac eutt_trans H :=
  etransitivity; [| apply H].

(*
 * rewrite_eutt H: Rewrite using eutt equivalence H.
 * Works in both directions and under bind.
 *)
Ltac rewrite_eutt H :=
  rewrite H.

(*
 * itree_case: Case analysis on ITree structure (Ret, Tau, Vis).
 * Simplifies goals based on the ITree constructor.
 *)
Ltac itree_case t :=
  destruct (observe t); simpl; auto.

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
(* Option Type Reasoning *)
(* ========================================================================= *)

(*
 * destr_option o: Destruct an option value and handle both cases.
 * Automatically simplifies None cases and introduces the value in Some cases.
 *)
Ltac destr_option o :=
  destruct o as [?val|]; simpl; auto; try discriminate.

(*
 * match_option: Simplify match expressions on options in the goal.
 *)
Ltac match_option :=
  repeat match goal with
  | |- context[match ?o with Some _ => _ | None => _ end] =>
      destruct o; simpl; auto
  | H : context[match ?o with Some _ => _ | None => _ end] |- _ =>
      destruct o; simpl in H; auto
  end.

(* ========================================================================= *)
(* List Reasoning *)
(* ========================================================================= *)

(*
 * list_induction l: Set up induction on list l with sensible names.
 *)
Ltac list_induction l :=
  induction l as [| x xs IHxs]; simpl; auto.

(*
 * list_length_eq: Prove two lists have equal length using arithmetic.
 *)
Ltac list_length_eq :=
  repeat rewrite length_map;  (* Use standard library lemma *)
  try reflexivity;
  try lia.

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

(* ========================================================================= *)
(* Debugging and Inspection *)
(* ========================================================================= *)

(*
 * show_goal: Display the current goal in a readable format.
 * Useful for debugging complex ITree goals.
 *)
Ltac show_goal :=
  match goal with
  | |- ?G => idtac "Current goal:" G
  end.

(*
 * show_context: Display all hypotheses.
 * Useful for understanding the proof state.
 *)
Ltac show_context :=
  match goal with
  | H : ?T |- _ => idtac "Hypothesis:" H ":" T; fail
  | _ => idtac "No more hypotheses"
  end.
