(*
 * Common Lemmas for MLIR-Sem Proofs
 *
 * This file contains reusable lemmas for reasoning about:
 * - Semantic equivalence
 * - ITree properties
 * - Program transformations
 *)

From Stdlib Require Import ZArith List Nat.
From ITree Require Import ITree Eq.
From Paco Require Import paco.
From MlirSem Require Import Syntax.AST.
From MlirSem Require Import Semantics.Values.
From MlirSem Require Import Semantics.Events.
From MlirSem Require Import Semantics.Denotation.
From MlirSem Require Import Utils.Tactics.

Import ListNotations.
Import ITreeNotations.

Local Open Scope itree_scope.
Local Open Scope Z_scope.

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
Lemma mlir_map_length {A B : Type} (f : A -> B) (l : list A) :
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
(* Arithmetic Operation Lemmas *)
(* ========================================================================= *)

(*
 * Arith_Constant is deterministic: always returns the same value
 *)
Lemma arith_constant_deterministic (val : Z) (res_type : mlir_type) :
  denote_general_op (Arith_Constant val res_type) ≈ Ret [IntVal val].
Proof.
  unfold denote_general_op. reflexivity.
Qed.

(*
 * Arith_AddI computes integer addition
 * This lemma states that if we have two integer values in locals,
 * addition produces their sum.
 *)
Lemma arith_addi_computes (lhs rhs : value_id) (l r : Z) (res_type : mlir_type) :
  forall (env : value_id -> option mlir_value),
    env lhs = Some (IntVal l) ->
    env rhs = Some (IntVal r) ->
    (* Under an interpreter that reads from env *)
    (* denote_general_op returns the sum *)
    True. (* TODO: Complete when we have interp_state *)
Admitted.

(*
 * Arith_CmpI computes comparisons correctly
 *)
Lemma arith_cmpi_correct (pred : arith_cmp_pred) (lhs rhs : value_id)
                         (l r : Z) (res_type : mlir_type) :
  forall (env : value_id -> option mlir_value),
    env lhs = Some (IntVal l) ->
    env rhs = Some (IntVal r) ->
    (* The comparison result matches the predicate *)
    True. (* TODO: Complete when we have interp_state *)
Admitted.

(* ========================================================================= *)
(* Block Execution Lemmas *)
(* ========================================================================= *)

(*
 * Executing a single terminator is equivalent to denoting it
 *)
Lemma denote_block_single_term (t : terminator_op) :
  denote_block [Term t] ≈ denote_terminator t.
Proof.
  unfold denote_block. reflexivity.
Qed.

(*
 * Executing operations in sequence
 * denote_block (op1 :: op2 :: rest) executes op1 then continues
 *)
Lemma denote_block_cons (results : list value_id) (g_op : general_op) (rest : list operation) :
  forall ops_rest,
    rest = ops_rest ->
    (* denote_block will first execute g_op, write results, then continue *)
    True. (* TODO: State and prove the sequential execution property *)
Admitted.

(*
 * Empty block list is an error
 *)
Lemma denote_block_nil_error :
  exists msg, denote_block [] ≈ trigger (inr1 (inr1 (inr1 (Throw msg)))).
Proof.
  Require Import String.
  Open Scope string_scope.
  exists "Block with no terminator"%string.
  reflexivity.
Qed.

(* ========================================================================= *)
(* Hints *)
(* ========================================================================= *)

#[global] Hint Resolve mlir_value_eq_dec : mlir_sem.
#[global] Hint Resolve mlir_map_length : mlir_sem.
#[global] Hint Resolve arith_constant_deterministic : mlir_sem.
#[global] Hint Resolve denote_block_single_term : mlir_sem.
