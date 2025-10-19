(* src/Theory/Correctness.v *)
From Stdlib Require Import String ZArith List.
From MlirSem Require Import Syntax.AST Semantics.Interpreter Semantics.Values.

Import ListNotations.
Local Open Scope string_scope.
(** An example computation to test our new semantics structure. *)
Module Example.

  (**
    func.func @add(%arg0: i32, %arg1: i32) -> i32 {
      %res = arith.addi %arg0, %arg1
      func.return %res
    }
  *)
  Definition add_func : mlir_func :=
    FuncOp "add" (FunctionType [Integer 32; Integer 32] [Integer 32])
      [ {| block_name := "entry";
           block_args := [("%arg0", Integer 32); ("%arg1", Integer 32)];
           block_ops :=
             [ Op ["%res"] (Arith_AddI "%arg0" "%arg1" (Integer 32));
               Term (Func_Return ["%res"]) ]
         |} ].

  (**
    func.func @main() {
      %c10 = arith.constant 10
      %c32 = arith.constant 32
      %res = func.call @add(%c10, %c32) : (i32, i32) -> i32
      func.return
    }
  *)
  Definition main_func : mlir_func :=
    FuncOp "main" (FunctionType [] [])
      [ {| block_name := "entry";
           block_args := [];
           block_ops :=
             [ Op ["%c10"] (Arith_Constant 10 (Integer 32));
               Op ["%c32"] (Arith_Constant 32 (Integer 32));
               Op ["%res"] (Func_Call "add" ["%c10"; "%c32"] (FunctionType [Integer 32; Integer 32] [Integer 32]));
               Term (Func_Return []) ]
         |} ].

  Definition program : mlir_program := [add_func; main_func].

  (** We want to prove that running this program results in 42. *)
  Theorem example_correct :
    run_program program "main" = Some []. (* Should be Some [IntVal 42] eventually *)
  Proof.
    (* The interpreter is not fully implemented yet. *)
    Admitted.

End Example.