(* src/Semantics/Denotation.v *)
From Stdlib Require Import String ZArith List.
From ITree Require Import ITree.
From MlirSem Require Import Syntax.AST Semantics.Events.

From ExtLib.Structures Require Import
     Monad.
Import MonadNotation.
Import ITreeNotations.
Import ListNotations.


Open Scope Z_scope.
Open Scope itree_scope.

(** Monadic list helpers (local definitions) *)
Fixpoint map_monad {E A B} (f : A -> itree E B) (xs : list A)
  : itree E (list B) :=
  match xs with
  | [] => Ret []
  | x :: xt =>
      y <- f x;;
      ys <- map_monad f xt;;
      Ret (y :: ys)
  end.

Fixpoint map_monad_ {E A} (f : A -> itree E unit) (xs : list A)
  : itree E unit :=
  match xs with
  | [] => Ret tt
  | x :: xt =>
      f x;;
      map_monad_ f xt
  end.

(** Specialized helpers to avoid implicit E inference issues *)
Fixpoint read_locals (xs : list value_id) : itree MlirSemE (list mlir_value) :=
  match xs with
  | [] => Ret []
  | x :: xt =>
      v <- trigger (inl1 (@LocalRead string mlir_value x)) ;;
      vs <- read_locals xt ;;
      Ret (v :: vs)
  end.


(** [denote_general_op] is the denotation function for general operations.
    It returns a list of resulting values. *)
Definition denote_general_op (op: general_op) : itree MlirSemE (list mlir_value) :=
  match op with
  | Arith_Constant val res_type => Ret [IntVal val]
  | Arith_AddI lhs rhs res_type =>
    lhs_val <- trigger (inl1 (@LocalRead string mlir_value lhs)) ;;
    rhs_val <- trigger (inl1 (@LocalRead string mlir_value rhs)) ;;
    match lhs_val, rhs_val with
    | IntVal l, IntVal r => Ret [IntVal (l + r)]
(*     | _, _ => trigger (inr1 (inr1 (inr1 (Throw "add arguments are not integers")))) *)
    end

  | Arith_CmpI pred lhs rhs res_type =>
    lhs_val <- trigger (inl1 (@LocalRead string mlir_value lhs)) ;;
    rhs_val <- trigger (inl1 (@LocalRead string mlir_value rhs)) ;;
    match lhs_val, rhs_val with
    | IntVal l, IntVal r =>
      let res := match pred with
                 | eq  => if Z.eqb l r then 1 else 0
                 | ne  => if negb (Z.eqb l r) then 1 else 0
                 | slt => if Z.ltb l r then 1 else 0
                 | sle => if Z.leb l r then 1 else 0
                 | sgt => if Z.gtb l r then 1 else 0
                 | sge => if Z.geb l r then 1 else 0
                 | ult => if Z.ltb l r then 1 else 0
                 | ule => if Z.leb l r then 1 else 0
                 end
      in
      Ret [IntVal res]
(*     | _, _ => trigger (inr1 (inr1 (inr1 (Throw "cmpi arguments are not integers")))) *)
    end

  | Func_Call callee args res_type =>
      (* 1. Read arguments from the local environment *)
      arg_vals <- read_locals args;;
      (* 2. Trigger the call event *)
      ret_vals <- trigger ((inr1 (inl1 (@Call mlir_value callee arg_vals))) : MlirSemE (list mlir_value)) ;;
      Ret ret_vals
  end.

(** [denote_terminator] is the denotation function for terminator operations. *)
Definition denote_terminator (op: terminator_op) : itree MlirSemE void :=
  match op with
  | Func_Return vals =>
      ret_vals <- read_locals vals;;
      trigger ((inr1 (inl1 (@Return mlir_value ret_vals))) : MlirSemE void)
  | Cf_Branch dest args =>
      arg_vals <- read_locals args;;
      trigger ((inr1 (inr1 (inl1 (@Branch block_id mlir_value dest arg_vals)))) : MlirSemE void)
  | Cf_CondBranch cond true_dest true_args false_dest false_args =>
      cond_val <- trigger (inl1 (@LocalRead string mlir_value cond)) ;;
      true_vals <- read_locals true_args;;
      false_vals <- read_locals false_args;;
      trigger ((inr1 (inr1 (inl1 (@CondBranch block_id mlir_value cond_val true_dest true_vals false_dest false_vals)))) : MlirSemE void)
  end.

(** [denote_operation] denotes a single operation, handling result binding. *)
Definition denote_operation (op: operation) : itree MlirSemE unit :=
  match op with
  | Op results g_op =>
    vals <- denote_general_op g_op;;
    (* Bind results to values. Assumes results and vals have same length. *)
    map_monad_ (fun p =>
      let '(id, vl) := p in
      trigger (inl1 (@LocalWrite string mlir_value id vl)))
      (combine results vals)
  | Term t_op =>
    v <- denote_terminator t_op;;
    match v with end (* Terminators return void *)
  end.

(** [denote_block] denotes all operations in a block. *)
Fixpoint denote_block (ops: list operation) : itree MlirSemE unit :=
  match ops with
  | [] => Ret tt
  | op :: rest =>
    denote_operation op;;
    denote_block rest
  end.

(** [denote_func] is the top-level denotation function for a single function. *)
Definition denote_func (f: mlir_func) : itree MlirSemE unit :=
  match f with
  | FuncOp name type body =>
    (* For a function, we just denote its body, which is a region.
       For now, we assume a region has one block and we denote it. *)
    match body with
    | [b] => denote_block (block_ops b)
    | _ => trigger (inr1 (inr1 (inr1 (Throw "functions with multiple blocks not supported yet"))))
    end
  end.