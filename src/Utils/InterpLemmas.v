(** Lemmas for reasoning about the interpreter and interp_state

    This file provides lemmas for proving equivalence of programs
    after they've been interpreted using interp_state.
*)

From Stdlib Require Import String ZArith List.
From Stdlib.Structures Require Import OrderedTypeEx.
From ExtLib Require Import Structures.Monad.
From ITree Require Import ITree Eq Events.State Interp.Interp Interp.InterpFacts.
From Stdlib.FSets Require Import FMapWeakList.
From MlirSem Require Import Syntax.AST Semantics.Events Semantics.Denotation Semantics.Interp.
From MlirSem Require Import Utils.Tactics.

Import ListNotations.
Import ITreeNotations.

Open Scope itree_scope.

(** * Basic Properties of interpret *)

(** The interpret function is defined using interp_state.
    We need lemmas about how it handles common operation patterns. *)

(** Write-then-read on the same variable returns the written value *)
Lemma interpret_write_read_same :
  forall (s : interpreter_state) (var : string) (val : mlir_value),
    s.(call_stack) <> [] ->
    (* TODO: State the property precisely
       Writing val to var, then reading var, should return val *)
    True.
Proof.
  (* This will require understanding the call_frame update semantics *)
  admit.
Admitted.

(** Writing to different variables doesn't interfere *)
Lemma interpret_write_write_different :
  forall (s : interpreter_state) (var1 var2 : string) (val1 val2 : mlir_value),
    var1 <> var2 ->
    s.(call_stack) <> [] ->
    (* TODO: Writing to var1 then var2 is independent of order if they're different *)
    True.
Proof.
  admit.
Admitted.

(** Reading doesn't change state *)
Lemma interpret_read_pure :
  forall (s : interpreter_state) (var : string),
    s.(call_stack) <> [] ->
    (* TODO: Reading a variable doesn't modify the state *)
    True.
Proof.
  admit.
Admitted.

(** * Properties of denote_general_op after interpretation *)

(** Constants are pure - they don't depend on state *)
Lemma interpret_constant_pure :
  forall (s : interpreter_state) (val : Z) (ty : mlir_type),
    s.(call_stack) <> [] ->
    interpret (denote_general_op (Arith_Constant val ty)) s ≈
    Ret (s, [IntVal val]).
Proof.
  intros s val ty _.
  unfold interpret, denote_general_op.
  (* Arith_Constant simply returns Ret [IntVal val], no effects *)
  (* interp_state on a Ret is just Ret (state, value) *)
  unfold interp_state.
  (* The key is that Ret doesn't trigger any events, so interp just returns it *)
  rewrite interp_ret.
  reflexivity.
Qed.

(** Addition reads two values and computes their sum *)
Lemma interpret_addi_computes :
  forall (s : interpreter_state) (lhs rhs : value_id) (lval rval : Z) (ty : mlir_type),
    s.(call_stack) <> [] ->
    (* If the current frame has lhs mapped to lval and rhs mapped to rval *)
    (exists frame rest,
        s.(call_stack) = frame :: rest /\
        ZStringMap.find lhs frame = Some (IntVal lval) /\
        ZStringMap.find rhs frame = Some (IntVal rval)) ->
    interpret (denote_general_op (Arith_AddI lhs rhs ty)) s ≈
    Ret (s, [IntVal (lval + rval)]).
Proof.
  intros s lhs rhs lval rval ty Hstack [frame [rest [Hframe [Hlhs Hrhs]]]].
  unfold interpret, denote_general_op.
  (* This requires:
     1. Unfolding interp_state
     2. Showing LocalRead gets the right values
     3. Showing the computation proceeds correctly
  *)
  admit.
Admitted.

(** * Simplification tactics for interpreter reasoning *)

(** Tactic to simplify goals involving interpret and constants *)
Ltac interp_constant_simp :=
  repeat match goal with
  | [ |- context[interpret (denote_general_op (Arith_Constant ?v ?ty)) ?s] ] =>
      rewrite interpret_constant_pure; auto
  end.

(** Tactic to handle write-read sequences *)
Ltac interp_write_read :=
  repeat match goal with
  | [ |- context[interpret (trigger (inl1 (LocalWrite ?v ?val));; trigger (inl1 (LocalRead ?v))) ?s] ] =>
      rewrite interpret_write_read_same; auto
  end.

(**
   STRATEGY FOR COMPLETING THESE LEMMAS:

   1. Study ITree.Interp.InterpFacts for existing lemmas about interp_state
   2. Use ITree.Events.State for state-specific reasoning
   3. Leverage eutt properties and rewriting lemmas
   4. Build up from simple cases (constants) to complex (arithmetic with state)

   These lemmas are CRITICAL for proving prog_equiv theorems.
   Without them, we cannot reason about the interpreter's behavior.
*)
