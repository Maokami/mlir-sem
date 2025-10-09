(* src/Semantics/Denotation.v *)
From Stdlib Require Import String ZArith.
From ITree Require Import ITree.
From MlirSem Require Import Syntax.AST Semantics.Events.

From ExtLib.Structures Require Import
     Monad.
Import MonadNotation.
Import ITreeNotations.

Open Scope Z_scope.
Open Scope itree_scope.

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
    end
  end.
