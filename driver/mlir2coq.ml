(* Command-line tool to convert MLIR files to Coq definitions *)
open Driver.Transformer
open Driver.Bindings
open Driver.Coq_exporter

let read_file filename =
  try
    let ch = open_in filename in
    Fun.protect
      ~finally:(fun () -> close_in ch)
      (fun () -> really_input_string ch (in_channel_length ch))
  with
  | Sys_error msg ->
      Printf.eprintf "Error reading file %s: %s\n" filename msg;
      exit 1

let parse_mlir ctx mlir_string =
  let c_module = module_create_parse ctx mlir_string in
  if module_is_null c_module then
    Error "Failed to parse MLIR"
  else
    Fun.protect
      ~finally:(fun () -> module_destroy c_module)
      (fun () -> Ok (transform_module c_module))

let process_single_file ctx input_file output_file prog_name =
  let file_content = read_file input_file in
  let mlir_string = string_ref_create_from_string file_content in
  match parse_mlir ctx mlir_string with
  | Error msg ->
      Printf.eprintf "Error processing %s: %s\n" input_file msg;
      false
  | Ok mlir_prog ->
      export_to_coq_file output_file prog_name mlir_prog;
      true

let process_translation_pair ctx before_file after_file output_file =
  let before_content = read_file before_file in
  let after_content = read_file after_file in
  let before_string = string_ref_create_from_string before_content in
  let after_string = string_ref_create_from_string after_content in

  match parse_mlir ctx before_string, parse_mlir ctx after_string with
  | Ok before_prog, Ok after_prog ->
      export_translation_pair
        ~before_file ~after_file
        ~before_name:"program_before"
        ~after_name:"program_after"
        ~before_prog ~after_prog
        ~output_file;
      true
  | Error msg, _ | _, Error msg ->
      Printf.eprintf "Error: %s\n" msg;
      false

let usage () =
  Printf.eprintf "Usage:\n";
  Printf.eprintf "  %s <input.mlir> <output.v> <module_name>  - Convert single file\n" Sys.argv.(0);
  Printf.eprintf "  %s --pair <before.mlir> <after.mlir> <output.v>  - Convert translation validation pair\n" Sys.argv.(0);
  exit 1

let main () =
  let argc = Array.length Sys.argv in
  if argc < 3 then usage ()
  else
    let ctx = context_create () in
    Fun.protect
      ~finally:(fun () -> context_destroy ctx)
      (fun () ->
        (* Register dialects *)
        register_dialect (get_func_dialect ()) ctx;
        register_dialect (get_arith_dialect ()) ctx;
        register_dialect (get_cf_dialect ()) ctx;

        let success =
          if Sys.argv.(1) = "--pair" then
            if argc <> 5 then usage ()
            else process_translation_pair ctx Sys.argv.(2) Sys.argv.(3) Sys.argv.(4)
          else if argc = 4 then
            process_single_file ctx Sys.argv.(1) Sys.argv.(2) Sys.argv.(3)
          else
            usage ()
        in
        exit (if success then 0 else 1))

let () = main ()