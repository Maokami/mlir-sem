(* Auto-generated translation validation pair *)
Require Import MlirSem.Syntax.AST.
Require Import ZArith.
Open Scope Z_scope.

(* Original program from: test/TranslationValidation/sccp_simple_before.mlir *)
Definition program_before : mlir_program :=
  [
(FuncOp "main" (FunctionType [] [(Integer (64)%Z)])
  [
  {| block_name := "block0";
     block_args := [];
     block_ops := [
      (Op ["%0"] (Arith_Constant (10)%Z (Integer (64)%Z)));
      (Op ["%1"] (Arith_Constant (20)%Z (Integer (64)%Z)));
      (Op ["%2"] (Arith_AddI "%0" "%1" (Integer (64)%Z)));
      (Op ["%3"] (Arith_Constant (100)%Z (Integer (64)%Z)));
      (Op ["%4"] (Arith_AddI "%2" "%3" (Integer (64)%Z)));
      (Term (Func_Return ["%4"]))
    ] |}
  ])
].

(* Optimized program from: test/TranslationValidation/sccp_simple_after.mlir *)
Definition program_after : mlir_program :=
  [
(FuncOp "main" (FunctionType [] [(Integer (64)%Z)])
  [
  {| block_name := "block0";
     block_args := [];
     block_ops := [
      (Op ["%0"] (Arith_Constant (130)%Z (Integer (64)%Z)));
      (Term (Func_Return ["%0"]))
    ] |}
  ])
].

(* End of auto-generated definitions *)
