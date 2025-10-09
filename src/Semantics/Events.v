(* src/Semantics/Events.v *)
From Stdlib Require Import String ZArith.
From MlirSem.Semantics Require Import Values.
Export MlirSem.Semantics.Values.

From ITree Require Import ITree.

(** [LocalE] defines events for interacting with the local environment,
    i.e., reading and writing SSA values. *)
Variant LocalE (k v: Type) : Type -> Type := 
| LocalRead (id: k) : LocalE k v v
| LocalWrite (id: k) (val: v) : LocalE k v unit.

(** [FailureE] defines events for error handling. *)
Variant FailureE : Type -> Type :=
| Throw {A : Type} (msg : string) : FailureE A.

(** [ArithE] is the final event type for our initial arithmetic semantics.
    It combines local environment events and failure events. *)
Definition ArithE :=
  (LocalE string mlir_value) +' FailureE.

(** [raise] is a helper to trigger a failure event. *)
Definition raise {E A} `{FailureE -< E} (msg : string) : itree E A :=
  trigger (Throw (A:=A) msg).
