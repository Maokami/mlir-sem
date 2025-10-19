open Alcotest
open Driver
open AST

(* Alcotest testable for Z.t *)
let z_testable = testable (Fmt.of_to_string Z.to_string) Z.equal

(* Read file content into a string *)
let read_file filename =
  let ch = open_in filename in
  let s = really_input_string ch (in_channel_length ch) in
  close_in ch;
  s

let get_test_file () =
  match Sys.getenv_opt "MLIR_TEST_FILE" with
  | Some p -> p
  | None ->
      let default =
        match Sys.getenv_opt "DUNE_SOURCEROOT" with
        | Some root -> Filename.concat root "test/simple_arith.mlir"
        | None -> "test/simple_arith.mlir"
      in
      default

(* A test case that checks if a simple MLIR file can be parsed and
   transformed. *)
let test_parse_and_transform () =
  let filename = get_test_file () in
  let file_content = read_file filename in

  let ctx = Bindings.context_create () in
  try
    let func_dialect = Bindings.get_func_dialect () in
    let arith_dialect = Bindings.get_arith_dialect () in
    Bindings.register_dialect func_dialect ctx;
    Bindings.register_dialect arith_dialect ctx;

    let mlir_string = Bindings.string_ref_create_from_string file_content in
    let c_module = Bindings.module_create_parse ctx mlir_string in

    (* Check 1: Parsing was successful *)
    check bool "Module should not be null" false
      (Bindings.module_is_null c_module);

    (* Check 2: Transform the module and check its structure *)
    let ocaml_prog = Transformer.transform_module c_module in
    check (list string) "Should contain one function named 'main'" [ "main" ]
      (List.map (function FuncOp (name, _, _) -> name) ocaml_prog);

    (* Check 3: Check the function body *)
    let func = List.hd ocaml_prog in
    let (FuncOp (_, _, body)) = func in
    check int "Function body should contain one block" 1 (List.length body);
    let block = List.hd body in
    check int "Block should have 2 operations" 2 (List.length block.block_ops);

    let op1 = List.hd block.block_ops in
    (match op1 with
    | Op ([ "%0" ], Arith_Constant (v, _)) ->
        check z_testable "Constant should be 42" (Z.of_int 42) v
    | _ ->
        Alcotest.fail "First operation should be arith.constant with result %0");

    let op2 = List.nth block.block_ops 1 in
    (match op2 with
    | Term (Func_Return [ "%0" ]) -> ()
    | _ ->
        Alcotest.fail "Second operation should be func.return with operand %0");

    Bindings.module_destroy c_module;
    Bindings.context_destroy ctx
  with e ->
    Bindings.context_destroy ctx;
    raise e

(* The test suite *)
let () =
  let () = Printexc.record_backtrace true in
  run "MLIR Driver"
    [
      ( "Parser and Transformer",
        [
          test_case "Parse and transform simple file" `Quick
            test_parse_and_transform;
        ] );
    ]
