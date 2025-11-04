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

let with_mlir_context (f : mlir_context -> unit) : unit =
  let ctx = context_create () in
  Fun.protect
    ~finally:(fun () -> context_destroy ctx)
    (fun () ->
      let func_dialect = get_func_dialect () in
      let arith_dialect = get_arith_dialect () in
      let cf_dialect = get_cf_dialect () in
      register_dialect func_dialect ctx;
      register_dialect arith_dialect ctx;
      register_dialect cf_dialect ctx;
      f ctx)

(* Generic test case for parsing and transforming MLIR to AST *)
let make_parse_and_transform_test ~name ~mlir_file ~expect_file =
  test_case name `Quick (fun () ->
      with_mlir_context (fun ctx ->
          let filename = get_mlir_test_file mlir_file () in
          let file_content = read_file filename in
          let mlir_string = string_ref_create_from_string file_content in
          let c_module = module_create_parse ctx mlir_string in
          Fun.protect
            ~finally:(fun () -> module_destroy c_module)
            (fun () ->
              check bool "Module should not be null" false
                (module_is_null c_module);
              let ocaml_prog = transform_module c_module in
              let pretty_printed_ast = string_of_mlir_program ocaml_prog in
              let expected_ast = read_expect_file expect_file in
              check string "Transformed AST should match golden file"
                expected_ast pretty_printed_ast)))

(* Generic test case for interpreter execution *)
let make_interpreter_execution_test ~name ~mlir_file ~expect_file =
  test_case name `Quick (fun () ->
      let mlir_file_path = get_mlir_test_file mlir_file () in
      let run_exe_path =
        match Sys.getenv_opt "RUN_EXE_PATH" with
        | Some p -> p
        | None -> Alcotest.fail "RUN_EXE_PATH environment variable not set"
      in
      let command = Printf.sprintf "%s %s" run_exe_path mlir_file_path in
      let ic = Unix.open_process_in command in
      let result_output = input_line ic in
      let _ = Unix.close_process_in ic in
      let expected_output = read_expect_file expect_file in
      check string "Interpreter output should match golden file"
        expected_output result_output)

(* The test suite *)
let () =
  let () = Printexc.record_backtrace true in
  run "MLIR Driver"
    [
      ( "Parser and Transformer",
        [
          make_parse_and_transform_test ~name:"Parse and transform simple file (golden)"
            ~mlir_file:"simple_arith.mlir" ~expect_file:"simple_arith.ast.expect";
          make_parse_and_transform_test
            ~name:"Parse and transform control flow file (golden)"
            ~mlir_file:"control_flow.mlir"
            ~expect_file:"control_flow.ast.expect";
          make_parse_and_transform_test
            ~name:"Parse and transform conditional branch file (golden)"
            ~mlir_file:"cond_branch.mlir"
            ~expect_file:"cond_branch.ast.expect";
        ] );
      ( "Interpreter Execution",
        [
          make_interpreter_execution_test
            ~name:"Execute simple arith (golden)" ~mlir_file:"simple_arith.mlir"
            ~expect_file:"simple_arith.output.expect";
          make_interpreter_execution_test
            ~name:"Execute control flow (golden)"
            ~mlir_file:"control_flow.mlir"
            ~expect_file:"control_flow.output.expect";
          make_interpreter_execution_test
            ~name:"Execute conditional branch (golden)"
            ~mlir_file:"cond_branch.mlir"
            ~expect_file:"cond_branch.output.expect";
        ] );
    ]