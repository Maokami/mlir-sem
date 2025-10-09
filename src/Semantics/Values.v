(* src/Semantics/Values.v *)
From Stdlib Require Import ZArith.

(** [mlir_value] represents a runtime value.
    For now, it only contains integer values. *)
Inductive mlir_value : Set :=
| IntVal (v: Z).
