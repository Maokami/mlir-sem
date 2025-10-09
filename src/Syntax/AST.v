(* src/Syntax/AST.v *)
From Stdlib Require Import String ZArith.

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
