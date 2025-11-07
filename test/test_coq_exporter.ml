open Alcotest
open Driver.Coq_exporter
open Mlir_sem_extracted.Interp

(* Test z_to_coq conversion *)
let test_z_to_coq_positive () =
  let z = Big_int_Z.big_int_of_int 42 in
  let result = z_to_coq z in
  check string "Positive integer converts correctly" "(42)%Z" result

let test_z_to_coq_negative () =
  let z = Big_int_Z.big_int_of_int (-42) in
  let result = z_to_coq z in
  check string "Negative integer converts correctly" "(-42)%Z" result

let test_z_to_coq_zero () =
  let z = Big_int_Z.big_int_of_int 0 in
  let result = z_to_coq z in
  check string "Zero converts correctly" "(0)%Z" result

(* Test value_id_to_coq conversion *)
let test_value_id_to_coq () =
  let result = value_id_to_coq "%0" in
  check string "Value ID converts correctly" "\"%0\"" result

(* Test block_id_to_coq conversion *)
let test_block_id_to_coq () =
  let result = block_id_to_coq "block0" in
  check string "Block ID converts correctly" "\"block0\"" result

(* Test mlir_type_to_coq conversion *)
let test_mlir_type_integer () =
  let typ = Integer (Big_int_Z.big_int_of_int 64) in
  let result = mlir_type_to_coq typ in
  check string "Integer type converts correctly" "(Integer (64)%Z)" result

let test_mlir_type_function () =
  let typ = FunctionType (
    [Integer (Big_int_Z.big_int_of_int 64)],
    [Integer (Big_int_Z.big_int_of_int 32)]
  ) in
  let result = mlir_type_to_coq typ in
  check string "Function type converts correctly"
    "(FunctionType [(Integer (64)%Z)] [(Integer (32)%Z)])" result

(* Test arith_cmp_pred_to_coq conversion *)
let test_arith_cmp_pred_eq () =
  let result = arith_cmp_pred_to_coq Eq0 in
  check string "Eq predicate converts correctly" "eq" result

let test_arith_cmp_pred_slt () =
  let result = arith_cmp_pred_to_coq Slt in
  check string "Slt predicate converts correctly" "slt" result

(* Test general_op_to_coq conversion *)
let test_general_op_constant () =
  let op = Arith_Constant (
    Big_int_Z.big_int_of_int 42,
    Integer (Big_int_Z.big_int_of_int 64)
  ) in
  let result = general_op_to_coq op in
  check string "Arith_Constant converts correctly"
    "(Arith_Constant (42)%Z (Integer (64)%Z))" result

let test_general_op_addi () =
  let op = Arith_AddI (
    "%0", "%1",
    Integer (Big_int_Z.big_int_of_int 64)
  ) in
  let result = general_op_to_coq op in
  check string "Arith_AddI converts correctly"
    "(Arith_AddI \"%0\" \"%1\" (Integer (64)%Z))" result

let test_general_op_cmpi () =
  let op = Arith_CmpI (
    Eq0, "%0", "%1",
    Integer (Big_int_Z.big_int_of_int 1)
  ) in
  let result = general_op_to_coq op in
  check string "Arith_CmpI converts correctly"
    "(Arith_CmpI eq \"%0\" \"%1\" (Integer (1)%Z))" result

(* Test terminator_op_to_coq conversion *)
let test_terminator_return () =
  let op = Func_Return ["%0"; "%1"] in
  let result = terminator_op_to_coq op in
  check string "Func_Return converts correctly"
    "(Func_Return [\"%0\"; \"%1\"])" result

let test_terminator_branch () =
  let op = Cf_Branch ("block1", ["%0"]) in
  let result = terminator_op_to_coq op in
  check string "Cf_Branch converts correctly"
    "(Cf_Branch \"block1\" [\"%0\"])" result

let test_terminator_cond_branch () =
  let op = Cf_CondBranch ("%cond", "block_true", ["%0"], "block_false", ["%1"]) in
  let result = terminator_op_to_coq op in
  check string "Cf_CondBranch converts correctly"
    "(Cf_CondBranch \"%cond\" \"block_true\" [\"%0\"] \"block_false\" [\"%1\"])" result

(* Test operation_to_coq conversion *)
let test_operation_op () =
  let op = Op (
    ["%0"],
    Arith_Constant (Big_int_Z.big_int_of_int 10, Integer (Big_int_Z.big_int_of_int 64))
  ) in
  let result = operation_to_coq op in
  check string "Op converts correctly"
    "(Op [\"%0\"] (Arith_Constant (10)%Z (Integer (64)%Z)))" result

let test_operation_term () =
  let op = Term (Func_Return ["%0"]) in
  let result = operation_to_coq op in
  check string "Term converts correctly"
    "(Term (Func_Return [\"%0\"]))" result

(* Test block_to_coq conversion *)
let test_block_simple () =
  let block = {
    block_name = "entry";
    block_args = [];
    block_ops = [
      Op (["%0"], Arith_Constant (Big_int_Z.big_int_of_int 42, Integer (Big_int_Z.big_int_of_int 64)));
      Term (Func_Return ["%0"])
    ]
  } in
  let result = block_to_coq block in
  check (string)
    "Simple block converts correctly"
    (String.trim "  {| block_name := \"entry\";\n     block_args := [];\n     block_ops := [\n      (Op [\"%0\"] (Arith_Constant (42)%Z (Integer (64)%Z)));\n      (Term (Func_Return [\"%0\"]))\n    ] |}")
    (String.trim result)

(* Test mlir_program_to_coq conversion *)
let test_mlir_program_empty () =
  let prog = [] in
  let result = mlir_program_to_coq "test_prog" prog in
  check string "Empty program converts correctly"
    "Definition test_prog : mlir_program :=\n  [].\n" result

(* File export tests *)
let test_export_to_coq_file () =
  let temp_file = Filename.temp_file "test_export" ".v" in
  Fun.protect
    ~finally:(fun () -> Sys.remove temp_file)
    (fun () ->
      let prog = [] in
      export_to_coq_file temp_file "test_prog" prog;
      check bool "Export file exists" true (Sys.file_exists temp_file);
      let content = In_channel.with_open_bin temp_file In_channel.input_all in
      check bool "Export file contains expected header" true
        (String.contains content '('))

let test_export_translation_pair () =
  let temp_file = Filename.temp_file "test_translation" ".v" in
  Fun.protect
    ~finally:(fun () -> Sys.remove temp_file)
    (fun () ->
      let prog_before = [] in
      let prog_after = [] in
      export_translation_pair
        ~before_file:"before.mlir"
        ~after_file:"after.mlir"
        ~before_name:"program_before"
        ~after_name:"program_after"
        ~before_prog:prog_before
        ~after_prog:prog_after
        ~output_file:temp_file;
      check bool "Translation pair file exists" true (Sys.file_exists temp_file);
      let content = In_channel.with_open_bin temp_file In_channel.input_all in
      check bool "Translation pair contains both programs" true
        (String.contains content 'p' && String.contains content 'b'))

(* Test suite *)
let () =
  run "Coq Exporter"
    [
      ( "Integer Conversion",
        [
          test_case "Positive integer" `Quick test_z_to_coq_positive;
          test_case "Negative integer" `Quick test_z_to_coq_negative;
          test_case "Zero" `Quick test_z_to_coq_zero;
        ] );
      ( "Identifier Conversion",
        [
          test_case "Value ID" `Quick test_value_id_to_coq;
          test_case "Block ID" `Quick test_block_id_to_coq;
        ] );
      ( "Type Conversion",
        [
          test_case "Integer type" `Quick test_mlir_type_integer;
          test_case "Function type" `Quick test_mlir_type_function;
        ] );
      ( "Comparison Predicate Conversion",
        [
          test_case "Eq predicate" `Quick test_arith_cmp_pred_eq;
          test_case "Slt predicate" `Quick test_arith_cmp_pred_slt;
        ] );
      ( "General Operation Conversion",
        [
          test_case "Arith_Constant" `Quick test_general_op_constant;
          test_case "Arith_AddI" `Quick test_general_op_addi;
          test_case "Arith_CmpI" `Quick test_general_op_cmpi;
        ] );
      ( "Terminator Operation Conversion",
        [
          test_case "Func_Return" `Quick test_terminator_return;
          test_case "Cf_Branch" `Quick test_terminator_branch;
          test_case "Cf_CondBranch" `Quick test_terminator_cond_branch;
        ] );
      ( "Operation Conversion",
        [
          test_case "Op" `Quick test_operation_op;
          test_case "Term" `Quick test_operation_term;
        ] );
      ( "Block Conversion",
        [
          test_case "Simple block" `Quick test_block_simple;
        ] );
      ( "Program Conversion",
        [
          test_case "Empty program" `Quick test_mlir_program_empty;
        ] );
      ( "File Export",
        [
          test_case "Export to Coq file" `Quick test_export_to_coq_file;
          test_case "Export translation pair" `Quick test_export_translation_pair;
        ] );
    ]
