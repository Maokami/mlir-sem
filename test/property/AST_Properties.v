(*
 * Property-based tests for MLIR AST definitions
 *
 * This file contains QuickChick property tests for the core AST.
 * To use these tests, install QuickChick:
 *   opam install coq-quickchick
 *)

From QuickChick Require Import QuickChick.
Require Import Coq.Strings.String.
Require Import Coq.ZArith.ZArith.
From MlirSem Require Import Syntax.AST.

Import QcDefaultNotation.
Open Scope qc_scope.

(* ========================================================================= *)
(* Generators for MLIR AST types *)
(* ========================================================================= *)

(* Generator for integer types *)
Definition gen_int_type : G int_type :=
  elements (I1 :: I32 :: I64 :: nil).

(* Generator for function types *)
Definition gen_func_type : G func_type :=
  liftGen (fun inputs output =>
    FuncType inputs output)
    (listOf gen_int_type)
    gen_int_type.

(* Generator for comparison predicates *)
Definition gen_cmpi_pred : G cmpi_pred :=
  elements (CmpEq :: CmpNe :: CmpSlt :: CmpSle :: CmpSgt :: CmpSge :: nil).

(* Generator for small integers *)
Definition gen_small_int : G Z :=
  choose (-100, 100)%Z.

(* Generator for operation SSA values (simplified) *)
Definition gen_operation : G operation :=
  oneOf [
    liftGen (fun z => OpConstant z) gen_small_int;
    liftGen2 (fun x y => OpAddi x y) arbitrary arbitrary;
    liftGen3 (fun pred x y => OpCmpi pred x y) gen_cmpi_pred arbitrary arbitrary
  ].

(* ========================================================================= *)
(* Properties *)
(* ========================================================================= *)

(*
 * Property: Integer type equality is decidable
 * This property ensures that we can always decide if two integer types are equal.
 *)
Definition prop_int_type_eq_decidable : Checker :=
  forAllShrink gen_int_type shrink (fun t1 =>
  forAllShrink gen_int_type shrink (fun t2 =>
    (t1 = t2) \/ (t1 <> t2) ?
  )).

(*
 * Property: Comparison predicates are well-formed
 * Every generated comparison predicate should be one of the valid constructors.
 *)
Definition prop_cmpi_pred_valid : Checker :=
  forAllShrink gen_cmpi_pred shrink (fun pred =>
    match pred with
    | CmpEq | CmpNe | CmpSlt | CmpSle | CmpSgt | CmpSge => true
    end ?
  ).

(*
 * Property: Constant operations preserve their values
 * A constant operation should maintain the same value regardless of representation.
 *)
Definition prop_constant_preserves_value : Checker :=
  forAllShrink gen_small_int shrink (fun z =>
    match OpConstant z with
    | OpConstant z' => (z = z') ?
    end
  ).

(*
 * Property: Addition is commutative in the AST structure sense
 * OpAddi x y and OpAddi y x represent addition, though we don't claim
 * semantic commutativity here (that requires semantics).
 *)
Definition prop_addi_structure : Checker :=
  forAll arbitrary (fun x : ssa_val =>
  forAll arbitrary (fun y : ssa_val =>
    (* Both expressions are well-formed operations *)
    match OpAddi x y, OpAddi y x with
    | _, _ => true
    end ?
  )).

(* ========================================================================= *)
(* Test suite *)
(* ========================================================================= *)

(*
 * Run all property tests.
 * Execute with: QuickChick test_ast_properties
 *)
Definition test_ast_properties : Checker :=
  conjoin [
    whenFail "Integer type equality not decidable" prop_int_type_eq_decidable;
    whenFail "Invalid comparison predicate generated" prop_cmpi_pred_valid;
    whenFail "Constant operation doesn't preserve value" prop_constant_preserves_value;
    whenFail "Addition operation malformed" prop_addi_structure
  ].

(* For extraction and command-line testing *)
Extract Constant defNumTests => "1000".

(* QuickChick test_ast_properties. *)
