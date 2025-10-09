(* src/Theory/Correctness.v *)
From Stdlib Require Import String ZArith .
From Stdlib.Structures Require Import OrderedTypeEx.
From ITree Require Import ITree Eq.Eqit.
From Paco Require Import paco.

From MlirSem Require Import Syntax.AST Semantics.Events Semantics.Values Semantics.Denotation Semantics.Interpreter.

From ExtLib.Structures Require Import Monad.
Import MonadNotation.

From ITree.Interp Require Import InterpFacts.
From ITree.Events Require Import StateFacts.
From ITree Require Import ITreeFacts.
From Stdlib.FSets Require Import FMapFacts.
Module ZFacts := FMapFacts.WFacts_fun String_as_OT ZStringMap.

Open Scope itree_scope.
Open Scope string_scope.

(** An example computation to test our semantics. *)
Module Example.

  (** 1. Define the operation to test: arith.addi %0, %1 *)
  Definition add_op : mlir_operation :=
    Arith_AddI "%0" "%1" (Integer 32).

  (** 2. Define the initial environment where %0 = 10 and %1 = 32. *)
  Definition initial_env : environment :=
    ZStringMap.add "%1" (IntVal 32) (ZStringMap.add "%0" (IntVal 10) (ZStringMap.empty mlir_value)).

  (** 3. Get the itree for the operation using the denotation function. *)
  Definition add_itree : itree ArithE mlir_value :=
    denote_op add_op.

  (** 4. Run the itree with the handler and the initial environment. *)
  Definition result_tree : itree FailureE (environment * mlir_value) :=
    run add_itree initial_env.

  (** 5. We prove that the result of running the example is indeed 42. *)
  Theorem example_correct :
    result_tree ≈ Ret (initial_env, IntVal 42).
  Proof.
    unfold result_tree, run, add_itree, denote_op, add_op.

    setoid_rewrite interp_state_bind.
    setoid_rewrite interp_state_trigger.
    cbn.
    rewrite bind_ret_l.

    setoid_rewrite interp_state_bind.
    setoid_rewrite interp_state_trigger.
    cbn.
    rewrite bind_ret_l.

    cbn.
    rewrite interp_state_ret.
    reflexivity.
  Qed.

End Example.
