(*
 * Infrastructure Quick Tests
 *
 * This file contains quick sanity checks for the ITree reasoning
 * infrastructure (tactics and lemmas) to ensure they work correctly.
 *)

From Stdlib Require Import ZArith List.
From ITree Require Import ITree Eq.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Values.
From MlirSem Require Import Semantics.Events.
From MlirSem Require Import Semantics.Denotation.
From MlirSem Require Import Utils.Tactics.
From MlirSem Require Import Utils.Lemmas.

Import ListNotations.
Import ITreeNotations.

Local Open Scope itree_scope.
Local Open Scope Z_scope.

(* ========================================================================= *)
(* Basic Tactic Tests *)
(* ========================================================================= *)

(*
 * Test: inv tactic works on simple inversion
 *)
Goal forall (x y : Z) (H : Some x = Some y), x = y.
Proof.
  intros x y H.
  inv H.
  reflexivity.
Qed.

(*
 * Test: simpl_goal tactic
 *)
Goal forall x : Z, x = x.
Proof.
  simpl_goal.
Qed.

(*
 * Test: list_induction tactic
 *)
Goal forall (l : list Z), length l = length l.
Proof.
  list_induction l.
Qed.

(* ========================================================================= *)
(* Lemma Application Tests *)
(* ========================================================================= *)

(*
 * Test: mlir_value_eq_dec can be used
 *)
Goal forall (v1 v2 : mlir_value), {v1 = v2} + {v1 <> v2}.
Proof.
  apply mlir_value_eq_dec.
Qed.

(*
 * Test: mlir_map_length lemma
 *)
Goal forall (f : Z -> Z) (l : list Z),
  length (map f l) = length l.
Proof.
  intros.
  apply mlir_map_length.
Qed.

(*
 * Test: nth_error_map lemma
 *)
Goal forall (f : Z -> Z) (l : list Z) (n : nat),
  nth_error (map f l) n = option_map f (nth_error l n).
Proof.
  intros.
  apply nth_error_map.
Qed.

(* ========================================================================= *)
(* ITree-Specific Tactic Tests *)
(* ========================================================================= *)

(*
 * Test: itree_simp works
 *)
Goal forall (E : Type -> Type) (t : itree E unit),
  ITree.bind t (fun _ => Ret tt) = ITree.bind t (fun _ => Ret tt).
Proof.
  intros.
  itree_simp.
Qed.

(*
 * Test: eutt_refl works
 *)
Goal forall (t : itree MlirSemE unit),
  eutt Logic.eq t t.
Proof.
  intros.
  eutt_refl.
Qed.

(* ========================================================================= *)
(* Option Type Reasoning Tests *)
(* ========================================================================= *)

(*
 * Test: match_option tactic
 *)
Goal forall (o : option Z),
  match o with
  | Some v => Some v
  | None => None
  end = o.
Proof.
  intros o.
  match_option.
Qed.

(*
 * Test: destr_option tactic
 *)
Goal forall (o : option Z) (default : Z),
  match o with
  | Some v => v
  | None => default
  end = match o with Some v => v | None => default end.
Proof.
  intros o default.
  destr_option o; reflexivity.
Qed.

(* ========================================================================= *)
(* Arithmetic Operation Lemmas *)
(* ========================================================================= *)

(*
 * Test: arith_constant_deterministic
 *)
Goal forall (val : Z) (res_type : mlir_type),
  denote_general_op (Arith_Constant val res_type) ≈ Ret [IntVal val].
Proof.
  intros.
  apply arith_constant_deterministic.
Qed.

(* ========================================================================= *)
(* Block Execution Lemmas *)
(* ========================================================================= *)

(*
 * Test: denote_block_single_term
 *)
Goal forall (t : terminator_op),
  denote_block [Term t] ≈ denote_terminator t.
Proof.
  intros.
  apply denote_block_single_term.
Qed.

(*
 * Test: denote_block_nil_error
 *)
Goal exists msg, denote_block [] ≈ trigger (inr1 (inr1 (inr1 (Throw msg)))).
Proof.
  apply denote_block_nil_error.
Qed.

(* ========================================================================= *)
(* Integration Test: mlir_auto *)
(* ========================================================================= *)

(*
 * Test: mlir_auto can solve simple goals
 *)
Goal forall (x y : Z), x = y -> x = y.
Proof.
  mlir_auto.
Qed.

Goal forall (l : list Z), l = l.
Proof.
  mlir_auto.
Qed.

(* ========================================================================= *)
(* Summary *)
(* ========================================================================= *)

(*
 * All tests passed! The infrastructure is working correctly:
 *
 * ✓ Basic tactics (inv, simpl_goal, list_induction)
 * ✓ List lemmas (mlir_map_length, nth_error_map)
 * ✓ Value lemmas (mlir_value_eq_dec)
 * ✓ ITree tactics (itree_simp, eutt_refl)
 * ✓ Option tactics (match_option, destr_option)
 * ✓ Arithmetic lemmas (arith_constant_deterministic)
 * ✓ Block execution lemmas (denote_block_single_term, denote_block_nil_error)
 * ✓ Automation (mlir_auto)
 *
 * The ITree reasoning infrastructure is ready for use in proofs!
 *)
