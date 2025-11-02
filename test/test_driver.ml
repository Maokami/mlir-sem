open Alcotest
open Driver.Bindings
open Driver.Transformer
open Driver.Ast_printer

(* Helper to read file content into a string *)
let read_file filename =
  let ch = open_in filename in
  let s = really_input_string ch (in_channel_length ch) in
  close_in ch;
  s

(* Helper to read expect file content *)
let read_expect_file filename_base =
  let expect_dir =
    match Sys.getenv_opt "DUNE_SOURCEROOT" with
    | Some root -> Filename.concat root "test/expect"
    | None -> "test/expect"
  in
  read_file (Filename.concat expect_dir filename_base)

let get_mlir_test_file filename_base () =
  let default_path =
    match Sys.getenv_opt "DUNE_SOURCEROOT" with
    | Some root -> Filename.concat root (Filename.concat "test" filename_base)
    | None -> Filename.concat "test" filename_base
  in
  default_path

(* Test case for parsing and transforming MLIR to AST, using golden files *)
let test_parse_and_transform_golden () =
  let filename = get_mlir_test_file "simple_arith.mlir" () in
  let file_content = read_file filename in

  let ctx = context_create () in
  try
    let func_dialect = get_func_dialect () in
    let arith_dialect = get_arith_dialect () in
    let cf_dialect = get_cf_dialect () in
    register_dialect func_dialect ctx;
    register_dialect arith_dialect ctx;
    register_dialect cf_dialect ctx;

    let mlir_string = string_ref_create_from_string file_content in
    let c_module = module_create_parse ctx mlir_string in

    check bool "Module should not be null" false (module_is_null c_module);

    let ocaml_prog = transform_module c_module in
    let pretty_printed_ast = string_of_mlir_program ocaml_prog in
    let expected_ast = read_expect_file "simple_arith.ast.expect" in

    check string "Transformed AST should match golden file" expected_ast
      pretty_printed_ast;

    module_destroy c_module;
    context_destroy ctx
  with e ->
    context_destroy ctx;
    raise e

(* Test case for interpreter execution, using golden files *)
let test_interpreter_execution_golden () =
  let mlir_file = get_mlir_test_file "simple_arith.mlir" () in
  let run_exe_path =
    match Sys.getenv_opt "RUN_EXE_PATH" with
    | Some p -> p
    | None -> Alcotest.fail "RUN_EXE_PATH environment variable not set"
  in
  let command = Printf.sprintf "%s %s" run_exe_path mlir_file in

  let ic = Unix.open_process_in command in
  let result_output = input_line ic in
  let _ = Unix.close_process_in ic in

  let expected_output = read_expect_file "simple_arith.output.expect" in

  check string "Interpreter output should match golden file" expected_output
    result_output

let test_parse_and_transform_control_flow_golden () =
  let filename = get_mlir_test_file "control_flow.mlir" () in
  let file_content = read_file filename in

  let ctx = context_create () in
  try
    let func_dialect = get_func_dialect () in
    let arith_dialect = get_arith_dialect () in
    let cf_dialect = get_cf_dialect () in
    register_dialect func_dialect ctx;
    register_dialect arith_dialect ctx;
    register_dialect cf_dialect ctx;

    let mlir_string = string_ref_create_from_string file_content in
    let c_module = module_create_parse ctx mlir_string in

    check bool "Module should not be null" false (module_is_null c_module);

    let ocaml_prog = transform_module c_module in
    let pretty_printed_ast = string_of_mlir_program ocaml_prog in
    let expected_ast = read_expect_file "control_flow.ast.expect" in

    check string "Transformed AST should match golden file" expected_ast
      pretty_printed_ast;

    module_destroy c_module;
    context_destroy ctx
  with e ->
    context_destroy ctx;
    raise e

let test_interpreter_execution_control_flow_golden () =
  let mlir_file = get_mlir_test_file "control_flow.mlir" () in
  let run_exe_path =
    match Sys.getenv_opt "RUN_EXE_PATH" with
    | Some p -> p
    | None -> Alcotest.fail "RUN_EXE_PATH environment variable not set"
  in
  let command = Printf.sprintf "%s %s" run_exe_path mlir_file in

  let ic = Unix.open_process_in command in
  let result_output = input_line ic in
  let _ = Unix.close_process_in ic in

  let expected_output = read_expect_file "control_flow.output.expect" in

  check string "Interpreter output should match golden file" expected_output
    result_output

(* The test suite *)
let () =
  let () = Printexc.record_backtrace true in
  run "MLIR Driver"
    [
      ( "Parser and Transformer",
        [
          test_case "Parse and transform simple file (golden)" `Quick
            test_parse_and_transform_golden;
          test_case "Parse and transform control flow file (golden)" `Quick
            test_parse_and_transform_control_flow_golden;
        ] );
      ( "Interpreter Execution",
        [
          test_case "Execute simple arith (golden)" `Quick
            test_interpreter_execution_golden;
          test_case "Execute control flow (golden)" `Quick
            test_interpreter_execution_control_flow_golden;
        ] );
    ]
