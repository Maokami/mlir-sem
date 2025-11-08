(** Lemmas about denotation semantics (before interpretation)

    This file provides lemmas for reasoning about the denotational semantics
    of MLIR operations at the ITree level, before they are interpreted.

    Strategy: Prove equivalences at the denote level, then lift to interpret level.
*)

From Stdlib Require Import String ZArith List.
From ITree Require Import ITree Eq.
From MlirSem Require Import Syntax.AST Semantics.Events Semantics.Denotation.
From MlirSem Require Import Utils.Tactics.

Import ListNotations.
Import ITreeNotations.

Open Scope itree_scope.

(** * Basic Denotation Properties *)

(** Two constants with the same value denote the same thing *)
Lemma denote_constant_eq :
  forall (val : Z) (ty1 ty2 : mlir_type),
    denote_general_op (Arith_Constant val ty1) ≈
    denote_general_op (Arith_Constant val ty2).
Proof.
  intros.
  unfold denote_general_op.
  reflexivity.
Qed.

(** Constant operations are deterministic *)
Lemma denote_constant_deterministic :
  forall (val : Z) (ty : mlir_type),
    denote_general_op (Arith_Constant val ty) ≈ Ret [IntVal val].
Proof.
  intros.
  unfold denote_general_op.
  reflexivity.
Qed.

(** * Block Denotation Properties *)

(** A block that just returns a constant *)
Lemma denote_block_constant_return :
  forall (var : value_id) (val : Z) (ty : mlir_type),
    denote_block [
      Op [var] (Arith_Constant val ty);
      Term (Func_Return [var])
    ] ≈
    (vals <- denote_general_op (Arith_Constant val ty) ;;
     trigger (inl1 (LocalWrite var (List.hd (IntVal 0) vals))) ;;
     v <- trigger (inl1 (LocalRead var)) ;;
     Ret (inr [v])).
Proof.
  intros.
  unfold denote_block. simpl.
  unfold denote_general_op, denote_terminator, read_locals, map_monad_.
  simpl.
  (* This should be provable by unfolding definitions *)
  (* The key insight: we can rewrite the bind sequence step by step *)
  reflexivity.
Qed.

(** Key lemma: write-then-read with constant can be simplified *)
Lemma write_read_constant_simplifies :
  forall (var : value_id) (val : Z),
    (trigger (inl1 (LocalWrite var (IntVal val))) ;;
     v <- trigger (inl1 (LocalRead var)) ;;
     Ret (inr [v]))
    ≈
    (trigger (inl1 (LocalWrite var (IntVal val))) ;;
     Ret (inr [IntVal val])).
Proof.
  intros.
  (* This requires reasoning about the LocalE effect semantics
     Under the handler, LocalWrite followed by LocalRead of the same var
     should return what was written.

     However, proving this requires handler-specific reasoning.
     For now, we state this as an axiom to make progress. *)
  admit.
Admitted.

(** * Arithmetic Simplification Lemmas *)

(** Adding two constants can be computed at denote time *)
Lemma denote_addi_constants :
  forall (lhs rhs : value_id) (lval rval : Z) (ty : mlir_type),
    (* If we know lhs will be lval and rhs will be rval *)
    (* Then the addition denotes to lval + rval *)
    (* But we need to express this in terms of the state... *)
    True. (* Placeholder - requires state reasoning *)
Proof.
  admit.
Admitted.

(** * Tactics for Denotation Reasoning *)

(** Simplify denote_general_op on constants *)
Ltac denote_constant_simp :=
  repeat match goal with
  | [ |- context[denote_general_op (Arith_Constant ?v ?ty)] ] =>
      rewrite denote_constant_deterministic
  end.

(** Unfold and simplify denote_block *)
Ltac denote_block_simp :=
  unfold denote_block, denote_general_op, denote_terminator;
  simpl;
  try reflexivity.

(**
   KEY REALIZATION:

   Proving equivalence at the denote level is still complex because we need
   to reason about LocalE effects and how they interact with binds.

   Alternative approach: Use the fact that ITree interpretation is compositional.
   If we can show that the OBSERVABLE EFFECTS are the same (same sequence of
   events triggered), then the interpreted trees will be equivalent.

   For constant folding:
   - Original: Write 10, Write 20, Read 10, Read 20, compute, Write 30, Read 30, Return
   - Folded:   Write 30, Read 30, Return

   These produce DIFFERENT effect sequences, but the FINAL RESULT is the same.

   This is actually NOT a simple eutt - we need to reason about the semantics
   of the effects themselves, i.e., we need the INTERPRETATION.

   REVISED STRATEGY:
   We should use a different equivalence notion that accounts for semantics:
   "After interpretation, both programs produce the same final value"

   This means working at the interpreted level is actually necessary.
*)
