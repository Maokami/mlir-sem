From Stdlib Require Import String ZArith List.

Definition value_id := string.
Definition block_id := string.

Inductive mlir_type : Set :=
| Integer (width: Z)
| FunctionType (args: list mlir_type) (rets: list mlir_type).

Inductive arith_cmp_pred : Set :=
| eq | ne | slt | sle | sgt | sge | ult | ule.

(** [general_op] represents general MLIR operations that do not terminate a block. *)
Inductive general_op : Set :=
| Arith_Constant (val: Z) (res_type: mlir_type)
| Arith_AddI (lhs: value_id) (rhs: value_id) (res_type: mlir_type)
| Arith_CmpI (pred: arith_cmp_pred) (lhs: value_id) (rhs: value_id) (res_type: mlir_type)
(* TODO: The operand type for Arith_CmpI is currently lost. This should be added for perfect pretty-printing. *)
| Func_Call (callee: string) (args: list value_id) (res_type: mlir_type).

(** [terminator_op] represents operations that terminate a block. *)
Inductive terminator_op : Set :=
| Func_Return (vals: list value_id)
| Cf_Branch (dest: block_id) (args: list value_id)
| Cf_CondBranch (cond: value_id)
                 (true_dest: block_id) (true_args: list value_id)
                 (false_dest: block_id) (false_args: list value_id).

(** An [operation] is either a general operation with its results, or a terminator. *)
Inductive operation : Set :=
| Op (results: list value_id) (op: general_op)
| Term (op: terminator_op).

(** A [block] is a sequence of operations. *)
Record block : Set := {
  block_name : block_id;
  block_args: list (value_id * mlir_type);
  block_ops : list operation;
}.

(** A [region] is a list of blocks. *)
Definition region := list block.

(** [mlir_func] represents a single function definition. *)
Inductive mlir_func : Set :=
| FuncOp (name: string) (type: mlir_type) (body: region).

(** [mlir_program] is the top-level representation, a list of functions. *)
Definition mlir_program := list mlir_func.
