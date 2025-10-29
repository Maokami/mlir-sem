open Driver.Transformer
open Driver.Bindings
open Mlir_sem_extracted

(* Helper to print an mlir_value *)
let print_value (v : Values.mlir_value) : string =
  Z.to_string v

(* The OCaml runner for the ITree, analogous to vellvm's `step` function.
   It does not use fuel and will loop forever on non-terminating programs. *)
let rec run_tree (tree : (Obj.t Interp.mlirSemE, Interp.interpreter_state * unit) Interp.itree) : Values.mlir_value list option =
  match Interp.observe tree with
  | Interp.RetF (_st, _) ->
      Printf.eprintf "Interpreter finished without a return value.\n";
      None
  | Interp.TauF t ->
      run_tree t (* Direct recursion on the next tree *)
  | Interp.VisF (ev, _k) ->
      (* The runner handles events that the Coq interpreter passed through. *)
      (match Obj.magic ev with
      | Events.Inr1 (Events.Inl1 (Events.Return vals)) ->
          (* This is the final result of the main function. *)
          Some vals
      | _ ->
          Printf.eprintf "Unhandled event in OCaml runner.\n";
          None)

(* Read file content into a string *)
let read_file filename =
  let ch = open_in filename in
  let s = really_input_string ch (in_channel_length ch) in
  close_in ch;
  s

let main () =
  let argc = Array.length Sys.argv in
  if argc <> 2 then Printf.eprintf "Usage: %s <file.mlir>\n" Sys.argv.(0)
  else
    let filename = Sys.argv.(1) in
    let file_content = read_file filename in

    let ctx = context_create () in
    try
      let func_dialect = get_func_dialect () in
      let arith_dialect = get_arith_dialect () in
      register_dialect func_dialect ctx;
      register_dialect arith_dialect ctx;

      let mlir_string = string_ref_create_from_string file_content in
      let c_module = module_create_parse ctx mlir_string in

      if module_is_null c_module then
        Printf.eprintf "Error parsing MLIR file\n"
      else (
        let ocaml_prog = transform_module c_module in

        (* Get the initial ITree from the Coq implementation *)
        match Interp.run_program ocaml_prog "main" with
        | None -> Printf.eprintf "Main function not found or setup failed.\n"
        | Some initial_tree -> (
            (* Run the ITree using the OCaml runner *)
            match run_tree initial_tree with
            | Some result_vals ->
                Printf.printf "Execution result: [%s]\n"
                  (String.concat ", " (List.map print_value result_vals))
            | None ->
                Printf.eprintf "Interpreter did not return a value.\n");

        module_destroy c_module);
      context_destroy ctx
    with e ->
      context_destroy ctx;
      raise e

let () = main ()
