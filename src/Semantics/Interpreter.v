(* src/Semantics/Interpreter.v *)
From Stdlib Require Import String ZArith.
From Stdlib.Structures Require Import OrderedTypeEx.
From ExtLib Require Import Structures.Monads Data.Monads.EitherMonad StateMonad.
From ITree Require Import ITree Events.State Core.ITreeDefinition.
From Stdlib.FSets Require Import FMapWeakList.
From MlirSem Require Import Semantics.Events.

Import MonadNotation.
Open Scope monad_scope.

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
