open Ctypes
open AST

(* Helper to convert MlirStringRef to OCaml string *)
let string_of_mlir_string_ref
    (str_ref : Bindings.mlir_string_ref Ctypes.structure) : string =
  let data_ptr = Ctypes.getf str_ref Bindings.data in
  let length = Ctypes.getf str_ref Bindings.length |> Unsigned.Size_t.to_int in
  string_from_ptr data_ptr ~length

(* State for mapping C pointers to SSA names *)
let value_map : (Bindings.mlir_value, string) Hashtbl.t = Hashtbl.create 16
let value_counter = ref 0

let reset_value_map () =
  Hashtbl.clear value_map;
  value_counter := 0

let get_value_name (v : Bindings.mlir_value) : string =
  if Hashtbl.mem value_map v then Hashtbl.find value_map v
  else
    let name = "%" ^ string_of_int !value_counter in
    incr value_counter;
    Hashtbl.add value_map v name;
    name

let rec transform_operation (c_op : Bindings.mlir_operation) : operation =
  let op_name_ident = Bindings.operation_get_identifier c_op in
  let op_name_ref = Bindings.operation_get_name op_name_ident in
  let op_name = string_of_mlir_string_ref op_name_ref in

  match op_name with
  | "arith.constant" ->
      let value_attr =
        Bindings.operation_get_attribute_by_name c_op
          (Bindings.string_ref_create_from_string "value")
      in
      let value =
        Bindings.integer_attr_get_value_int value_attr |> Z.of_int64
      in

      let num_results =
        Bindings.operation_get_num_results c_op |> Intptr.to_int
      in
      let results =
        if num_results > 0 then
          let res_val = Bindings.operation_get_result c_op (Intptr.of_int 0) in
          [ get_value_name res_val ]
        else []
      in

      (* TODO: get type properly *)
      Op (results, Arith_Constant (value, Integer (Z.of_int 32)))
  | "func.return" ->
      let num_operands =
        Bindings.operation_get_num_operands c_op |> Intptr.to_int
      in
      let operands =
        if num_operands > 0 then
          let op_val = Bindings.operation_get_operand c_op (Intptr.of_int 0) in
          [ get_value_name op_val ]
        else []
      in
      Term (Func_Return operands)
  | _ -> failwith ("Unsupported operation: " ^ op_name)

and transform_operations_in_block (c_op : Bindings.mlir_operation) :
    operation list =
  if is_null c_op then []
  else
    let ocaml_op = transform_operation c_op in
    let next_op = Bindings.operation_get_next_in_block c_op in
    ocaml_op :: transform_operations_in_block next_op

and transform_block (c_block : Bindings.mlir_block) : block =
  let first_op = Bindings.block_get_first_operation c_block in
  let ops = transform_operations_in_block first_op in
  (* TODO: get block name and args *)
  {
    block_name = "entry";
    (* placeholder *)
    block_args = [];
    (* placeholder *)
    block_ops = ops;
  }

and transform_region (c_region : Bindings.mlir_region) : region =
  let first_block = Bindings.region_get_first_block c_region in
  if is_null first_block then [] else [ transform_block first_block ]
(* TODO: Handle multiple blocks *)

(* Transforms a func.func operation into our AST *)
let transform_func (c_op : Bindings.mlir_operation) : mlir_func =
  let op_name_ident = Bindings.operation_get_identifier c_op in
  let op_name_ref = Bindings.operation_get_name op_name_ident in
  let _op_name = string_of_mlir_string_ref op_name_ref in

  (* This is a simplified placeholder. *)
  let func_name = "main" in
  (* Placeholder *)

  let num_regions = Bindings.operation_get_num_regions c_op |> Intptr.to_int in
  let body =
    if num_regions > 0 then
      let region = Bindings.operation_get_region c_op (Intptr.of_int 0) in
      transform_region region
    else []
  in
  FuncOp (func_name, FunctionType ([], []), body)
(* Placeholder for type *)

(* Transforms a C-API mlirModule into our OCaml AST for mlir_program *)
let transform_module (c_module : Bindings.mlir_module) : mlir_program =
  reset_value_map ();
  let top_level_op = Bindings.module_get_operation c_module in

  (* A module has one region with one block containing the top-level operations
     (functions, etc.) *)
  let region = Bindings.operation_get_region top_level_op (Intptr.of_int 0) in
  let block = Bindings.region_get_first_block region in

  let rec transform_top_level_ops (c_op : Bindings.mlir_operation) :
      mlir_program =
    if is_null c_op then []
    else
      let func = transform_func c_op in
      let next_op = Bindings.operation_get_next_in_block c_op in
      func :: transform_top_level_ops next_op
  in

  let first_op = Bindings.block_get_first_operation block in
  transform_top_level_ops first_op
