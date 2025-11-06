(*
 * Common Lemmas for MLIR-Sem Proofs
 *
 * This file contains reusable lemmas for reasoning about:
 * - Semantic equivalence
 * - ITree properties
 * - Program transformations
 *)

From Stdlib Require Import ZArith List Nat.
From ITree Require Import ITree.
From Paco Require Import paco.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Values.
From MlirSem Require Import Utils.Tactics.

Import ListNotations.
Import ITreeNotations.

Local Open Scope itree_scope.

(* ========================================================================= *)
(* Value Lemmas *)
(* ========================================================================= *)

(*
 * MLIR value equality is decidable
 *)
Lemma mlir_value_eq_dec (v1 v2 : mlir_value) :
  {v1 = v2} + {v1 <> v2}.
Proof.
  decide equality; apply Z.eq_dec.
Qed.

(* ========================================================================= *)
(* List Lemmas *)
(* ========================================================================= *)

(*
 * Length of map is preserved
 *)
Lemma map_length {A B : Type} (f : A -> B) (l : list A) :
  length (map f l) = length l.
Proof.
  induction l; simpl; auto.
Qed.

(*
 * nth_error on mapped list
 *)
Lemma nth_error_map {A B : Type} (f : A -> B) (l : list A) (n : nat) :
  nth_error (map f l) n = option_map f (nth_error l n).
Proof.
  revert n. induction l; intros [|n]; simpl; auto.
Qed.

(* ========================================================================= *)
(* Hints *)
(* ========================================================================= *)

#[global] Hint Resolve mlir_value_eq_dec : mlir_sem.
#[global] Hint Resolve map_length : mlir_sem.
