(* src/Semantics/Denotation.v *)
From Stdlib Require Import String ZArith List.
From ITree Require Import ITree.
From ITree.Interp Require Import Recursion.
From Stdlib.FSets Require Import FMapWeakList.
From MlirSem Require Import Syntax.AST Semantics.Events.
From Stdlib.Structures Require Import OrderedTypeEx.

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
Definition denote_terminator (op: terminator_op) : itree MlirSemE (block_id + list mlir_value) :=
  match op with
  | Func_Return vals =>
      ret_vals <- read_locals vals;;
      Ret (inr ret_vals)
  | Cf_Branch dest args =>
      (* TODO: args are ignored for now *)
      Ret (inl dest)
  | Cf_CondBranch cond true_dest true_args false_dest false_args =>
      trigger (inr1 (inr1 (inr1 (Throw "Cf_CondBranch not supported yet"))))
  end.

(** [denote_block] denotes all operations in a block, returning either the next block to execute or the final return value of the function. *)
Fixpoint denote_block (ops: list operation) : itree MlirSemE (block_id + list mlir_value) :=
  match ops with
  | [] => trigger (inr1 (inr1 (inr1 (Throw "Block with no terminator"))))
  | [Term t_op] => denote_terminator t_op
  | (Op results g_op) :: rest =>
      vals <- denote_general_op g_op;;
      map_monad_ (fun p =>
        let '(id, vl) := p in
        trigger (inl1 (@LocalWrite string mlir_value id vl)))
        (combine results vals);;
      denote_block rest
  | _ => trigger (inr1 (inr1 (inr1 (Throw "Malformed block: terminator is not the last operation"))))
  end.

Module BlockMap := FMapWeakList.Make(String_as_OT).

(** [denote_func] is the top-level denotation for a single function.
    It uses the [iter] combinator to model control flow between blocks. *)
Definition denote_func (f: mlir_func) : itree MlirSemE (list mlir_value) :=
  match f with
  | FuncOp name type body =>
      let block_map := List.fold_right
        (fun b map => BlockMap.add (block_name b) b map)
        (BlockMap.empty block)
        body
      in
      match body with
      | [] => trigger (inr1 (inr1 (inr1 (Throw "Function with empty body"))))
      | entry_block :: _ =>
          iter (C := ktree _) (bif := sum)
               (fun (current_block_id : block_id) =>
                  match BlockMap.find current_block_id block_map with
                  | None => trigger (inr1 (inr1 (inr1 (Throw ("Target block not found: " ++ current_block_id)))))
                  | Some current_block => denote_block (block_ops current_block)
                  end)
               (block_name entry_block)
      end
  end.