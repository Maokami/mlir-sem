open Driver.Transformer
open Driver.Bindings
open Mlir_sem_extracted

(* Helper to print an mlir_value *)
let print_value (v : Interp.mlir_value) : string = Z.to_string v

(* The OCaml runner for the ITree. *)
let rec run_tree
    (tree :
      ( Obj.t Interp.mlirSemE,
        Interp.interpreter_state * Interp.mlir_value list )
      Interp.itree) : Interp.mlir_value list option =
  match Interp.observe tree with
  | Interp.RetF state_and_result ->
      let _st, result_vals = state_and_result in
      Some result_vals
  | Interp.TauF t -> run_tree t (* Direct recursion on the next tree *)
  | Interp.VisF (ev, _k) -> (
      match ev with
      | Interp.Inr1 (Interp.Inr1 (Interp.Inr1 msg)) ->
          (* FailureE *)
          Printf.eprintf "Interpreter failed with: %s\n" msg;
          None
      | _ ->
          Printf.eprintf "Unhandled event in OCaml runner.\n";
          None)

(* Read file content into a string *)
let read_file filename =
  let ch = open_in filename in
  let s = really_input_string ch (in_channel_length ch) in
  close_in ch;
  s

let run_with_module ctx mlir_string =
  let c_module = module_create_parse ctx mlir_string in
  if module_is_null c_module then Error `Parse_failed
  else
    Fun.protect
      ~finally:(fun () -> module_destroy c_module)
      (fun () ->
        let ocaml_prog = transform_module c_module in
        match Interp.run_program ocaml_prog "main" with
        | None -> Error `No_main
        | Some itree -> (
            match run_tree itree with
            | None -> Error `No_value
            | Some vals -> Ok vals))

let main () =
  let argc = Array.length Sys.argv in
  if argc <> 2 then Printf.eprintf "Usage: %s <file.mlir>\n" Sys.argv.(0)
  else
    let filename = Sys.argv.(1) in
    let file_content = read_file filename in
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
        let mlir_string = string_ref_create_from_string file_content in
        match run_with_module ctx mlir_string with
        | Ok vals ->
            Printf.printf "{\"result\": [%s]}\n"
              (String.concat ", " (List.map print_value vals))
        | Error `Parse_failed -> prerr_endline "Error parsing MLIR file"
        | Error `No_main ->
            prerr_endline "Main function not found or setup failed."
        | Error `No_value ->
            Printf.printf
              "{\"error\": \"Interpreter did not return a value.\"}\n")

let () = main ()
