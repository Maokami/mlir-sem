From Stdlib Require Import String ZArith Strings.String ZArith.BinInt.
From Stdlib.Structures Require Import OrderedTypeEx.
From ExtLib Require Import Structures.Monads Data.Monads.EitherMonad StateMonad.
From ITree Require Import ITree Eq.Eqit Events.State Core.ITreeDefinition.
From Stdlib.FSets Require Import FMapWeakList.
From Paco Require Import paco.

Import MonadNotation.
Import ITreeNotations.

Open Scope string_scope.
Open Scope Z_scope.
Open Scope monad_scope.

(* Events for MLIR Semantics *)

(** [mlir_value] represents a runtime value.
    For now, it only contains integer values. *)
Inductive mlir_value : Set :=
| IntVal (v: Z).

(** [LocalE] defines events for interacting with the local environment,
    i.e., reading and writing SSA values. *)
Variant LocalE (k v: Type) : Type -> Type :=
| LocalRead (id: k) : LocalE k v v
| LocalWrite (id: k) (val: v) : LocalE k v unit.

(** [FailureE] defines events for error handling. *)
Variant FailureE : Type -> Type :=
| Throw {A: Type} (msg: string) : FailureE A.

(** [ArithE] is the final event type for our initial arithmetic semantics.
    It combines local environment events and failure events. *)
Definition ArithE :=
  (LocalE string mlir_value) +' FailureE.

(**
(** [raise] is a helper to trigger a failure event. *)
Definition raise {E A} `{FailureE -< E} (msg: string) : itree E A :=
  v <- trigger (Throw msg);; match v with end.
*)

(* MLIR Abstract Syntax Tree Definitions *)

(** A [value_id] is the identifier for an SSA value in MLIR (e.g., "%0", "%res").
    For now, we model it simply as a string. *)
Definition value_id := string.

(** The [mlir_type] inductive represents types in MLIR.
    We start with just integer types. *)
Inductive mlir_type : Set :=
| Integer (width: Z).

(** Predicates for the arith.cmpi operation. *)
Inductive arith_cmp_pred : Set :=
| eq | ne | slt | sle | sgt | sge | ult | ule.

(** The [mlir_operation] inductive represents MLIR operations.
    We start with a few basic operations from the 'arith' dialect. *)
Inductive mlir_operation : Set :=
| Arith_Constant (val: Z) (res_type: mlir_type)
| Arith_AddI (lhs: value_id) (rhs: value_id) (res_type: mlir_type)
| Arith_CmpI (pred: arith_cmp_pred) (lhs: value_id) (rhs: value_id) (res_type: mlir_type).

(* Denotational Semantics for MLIR AST *)

(** [denote_op] is the core denotation function.
    It takes an [mlir_operation] and returns an [itree] that represents
    the meaning of that operation. The itree can trigger events from [ArithE]. *)
Definition denote_op (op: mlir_operation) : itree ArithE mlir_value :=
  match op with
  | Arith_Constant val res_type =>
    (* A constant is a pure value. We just return it wrapped in an mlir_value. *)
    Ret (IntVal val)

  | Arith_AddI lhs rhs res_type =>
    (* To add, we first need to read the values of the left and right operands.
       We trigger the LocalRead event for each. The `inl1` indicates that we are
       triggering the first type of event in the sum type `ArithE`, which is `LocalE`. *)
    lhs_val <- trigger (inl1 (@LocalRead string mlir_value lhs)) ;; 
    rhs_val <- trigger (inl1 (@LocalRead string mlir_value rhs)) ;; 

    (* After getting the values, we pattern match to ensure they are integers,
       then return the sum. If they are not integers, we trigger a failure event.
       The `inr1` indicates the second type of event in `ArithE`, which is `FailureE`. *)
    match lhs_val, rhs_val with
    | IntVal l, IntVal r => Ret (IntVal (l + r))
    end

  | Arith_CmpI pred lhs rhs res_type =>
    (* CmpI is similar to AddI: read both operands first. *)
    lhs_val <- trigger (inl1 (@LocalRead string mlir_value lhs)) ;;
    rhs_val <- trigger (inl1 (@LocalRead string mlir_value rhs)) ;;

    match lhs_val, rhs_val with
    | IntVal l, IntVal r =>
      (* After getting the values, perform the comparison based on the predicate. *)
      let res := match pred with
                 | eq  => if Z.eqb l r then 1 else 0
                 | ne  => if negb (Z.eqb l r) then 1 else 0
                 | slt => if Z.ltb l r then 1 else 0
                 | sle => if Z.leb l r then 1 else 0
                 | sgt => if Z.gtb l r then 1 else 0
                 | sge => if Z.geb l r then 1 else 0
                 (* For now, we treat unsigned comparisons the same as signed. *)
                 | ult => if Z.ltb l r then 1 else 0
                 | ule => if Z.leb l r then 1 else 0
                 end
      in
      Ret (IntVal res)
(*     | _, _ => raise "Type error in Arith_CmpI: operands are not integers" *)
    end
  end.

(* Handlers for MLIR Events *)

Module ZStringMap := FMapWeakList.Make(String_as_OT).

Definition environment := ZStringMap.t mlir_value.

(** [run] interprets an itree with ArithE events. *)
Definition run {A: Type} (t: itree ArithE A) (env: environment)
  : itree FailureE (environment * A) :=
  let handler (T:Type) (e: ArithE T)
    : Monads.stateT environment (itree FailureE) T :=
    match e with
    | inl1 loc_e =>
        (* LocalE *)
        match loc_e in (LocalE _ _ T0) return Monads.stateT environment (itree FailureE) T0 with
        | @LocalRead _ _ id =>
            fun s =>
              let v :=
                match ZStringMap.find id s with
                | Some v => v
                | None => IntVal 0
                end in
              Ret (s, v)
        | @LocalWrite _ _ id val =>
            fun s => Ret (ZStringMap.add id val s, tt)
        end
    | inr1 fail_e =>
        (* FailureE *)
        Monads.liftState (trigger fail_e)
    end
  in
  Monads.run_stateT (interp_state handler t) env.

(* Main file for MLIR Semantics in Coq *)

From ITree.Interp Require Import InterpFacts.
From ITree.Events Require Import StateFacts.
From ITree Require Import ITreeFacts.
From Stdlib.FSets Require Import FMapFacts.
Module ZFacts := FMapFacts.WFacts_fun String_as_OT ZStringMap.

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
