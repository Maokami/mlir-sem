From Stdlib Require Import String ZArith.
From MlirSem.Syntax Require Import AST.
From MlirSem.Semantics Require Import Values.
Export MlirSem.Semantics.Values.

From ITree Require Import ITree.

(** [LocalE] defines events for interacting with the local environment,
    i.e., reading and writing SSA values. *)
Variant LocalE (k v: Type) : Type -> Type := 
| LocalRead (id: k) : LocalE k v v
| LocalWrite (id: k) (val: v) : LocalE k v unit.

(** [FunctionE] defines events for function calls and returns. *)
Variant FunctionE (v: Type) : Type -> Type :=
| Call (name: string) (args: list v) : FunctionE v (list v)
| Return (vals: list v) : FunctionE v void.

(** [ControlE] defines events for control flow. *)
Variant ControlE (k v: Type) : Type -> Type :=
| Branch (dest: k) (args: list v) : ControlE k v void
| CondBranch (cond: v) (true_dest: k) (true_args: list v) (false_dest: k) (false_args: list v) : ControlE k v void.

(** [FailureE] defines events for error handling. *)
Variant FailureE : Type -> Type :=
| Throw {A : Type} (msg : string) : FailureE A.

(** [MlirSemE] is the new combined event type. *)
Definition MlirSemE :=
  (LocalE string mlir_value) +'
  (FunctionE mlir_value) +'
  (ControlE block_id mlir_value) +'
  FailureE.

(** [raise] is a helper to trigger a failure event. *)
Definition raise {E A} `{FailureE -< E} (msg : string) : itree E A :=
  trigger (Throw (A:=A) msg).