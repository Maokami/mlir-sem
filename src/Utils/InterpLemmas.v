(** Lemmas for reasoning about the interpreter and interp_state

    This file provides lemmas for proving equivalence of programs
    after they've been interpreted using interp_state.
*)

From Stdlib Require Import String ZArith List.
From Stdlib.Structures Require Import OrderedTypeEx.
From ExtLib Require Import Structures.Monad.
From ITree Require Import ITree Eq Events.State Events.StateFacts Interp.Interp Interp.InterpFacts.
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
  forall (s : interpreter_state) (var : string) (value : mlir_value),
    s.(call_stack) <> [] ->
    interpret (trigger (inl1 (@LocalWrite string mlir_value var value)) ;;
               trigger (inl1 (@LocalRead string mlir_value var))) s ≈
    interpret (trigger (inl1 (@LocalWrite string mlir_value var value))) s >>= fun stt =>
      Ret (fst stt, value).
Proof.
  intros s var value Hstack.
  (* TODO: Complete this proof
     SIGNATURE CHECKED: ✓ (intros succeeded)
     IMPORTS NEEDED: Events.StateFacts (already imported)
     KEY LEMMAS: interp_state_bind, interp_state_trigger, ZStringMap.find_1, ZStringMap.add_1
     STRATEGY:
       1. Unfold interpret and use interp_state_bind
       2. Apply interp_state_trigger to both LocalWrite and LocalRead
       3. Use ZStringMap.add_1 to show written value is found
       4. Simplify the bind chain
     BLOCKERS: Need to understand proper bind rewriting in eutt context
  *)
  admit.
Admitted.

(** Writing to different variables doesn't interfere *)
Lemma interpret_write_write_different :
  forall (s : interpreter_state) (var1 var2 : string) (val1 val2 : mlir_value),
    var1 <> var2 ->
    s.(call_stack) <> [] ->
    (* Writing to var1 doesn't change the value of var2 *)
    interpret (trigger (inl1 (@LocalWrite string mlir_value var1 val1)) ;;
               trigger (inl1 (@LocalRead string mlir_value var2))) s ≈
    interpret (trigger (inl1 (@LocalRead string mlir_value var2))) s >>= fun sv2 =>
      interpret (trigger (inl1 (@LocalWrite string mlir_value var1 val1))) s >>= fun _ =>
        Ret (fst sv2, snd sv2).
Proof.
  intros s var1 var2 val1 val2 Hneq Hstack.
  (* TODO: Complete this proof
     SIGNATURE CHECKED: ✓ (intros succeeded)
     IMPORTS NEEDED: Events.StateFacts (already imported)
     KEY LEMMAS: interp_state_bind, interp_state_trigger, ZStringMap.add_2, ZStringMap.add_3
     STRATEGY:
       1. Similar to interpret_write_read_same but use ZStringMap.add_2
       2. Show that writing var1 doesn't affect reading var2 when var1 <> var2
       3. Use ZStringMap properties to prove frame independence
     BLOCKERS: Complex bind reasoning with multiple state operations
  *)
  admit.
Admitted.

(** Reading doesn't change state *)
Lemma interpret_read_pure :
  forall (s : interpreter_state) (var : string),
    s.(call_stack) <> [] ->
    exists v,
      interpret (trigger (inl1 (@LocalRead string mlir_value var))) s ≈ Ret (s, v).
Proof.
  intros s var Hstack.
  (* TODO: Complete this proof
     SIGNATURE CHECKED: ✓ (intros succeeded)
     IMPORTS NEEDED: Events.StateFacts (already imported)
     KEY LEMMAS: interp_state_trigger
     STRATEGY:
       1. Apply interp_state_trigger
       2. Unfold handle_event
       3. Case analysis on call_stack using Hstack
       4. Show state remains unchanged (only value is returned)
     BLOCKERS: None - should be straightforward
  *)
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
  rewrite interp_state_ret.
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
  (* TODO: Complete this proof
     SIGNATURE CHECKED: ✓ (intros and destructing succeeded)
     IMPORTS NEEDED: Events.StateFacts (already imported)
     KEY LEMMAS: interp_state_bind, interp_state_trigger, bind_ret_l
     STRATEGY:
       1. Unfold interpret and denote_general_op (Arith_AddI)
       2. Use interp_state_bind to handle the bind chain
       3. Apply interp_state_trigger to first LocalRead, use Hlhs
       4. Simplify with bind_ret_l
       5. Apply interp_state_trigger to second LocalRead, use Hrhs
       6. Simplify and apply interp_state_ret for final Ret
     BLOCKERS: None - similar pattern to interpret_constant_pure
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
  | [ |- context[interpret (trigger (inl1 (@LocalWrite string mlir_value ?v ?val));; trigger (inl1 (@LocalRead string mlir_value ?v))) ?s] ] =>
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
