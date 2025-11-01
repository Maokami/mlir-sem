open Ctypes
open Mlir_sem_extracted

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

let rec transform_mlir_type (c_type : Bindings.mlir_type) : Interp.mlir_type =
  if Bindings.type_is_a_integer c_type then
    let width =
      Bindings.integer_type_get_width c_type |> Unsigned.UInt.to_int
    in
    Interp.Integer (Z.of_int width)
  else if Bindings.type_is_a_function c_type then
    let num_inputs =
      Bindings.function_type_get_num_inputs c_type |> Intptr.to_int
    in
    let input_types =
      List.init num_inputs (fun i ->
          let input_c_type =
            Bindings.function_type_get_input c_type (Intptr.of_int i)
          in
          transform_mlir_type input_c_type)
    in
    let num_results =
      Bindings.function_type_get_num_results c_type |> Intptr.to_int
    in
    let result_types =
      List.init num_results (fun i ->
          let result_c_type =
            Bindings.function_type_get_result c_type (Intptr.of_int i)
          in
          transform_mlir_type result_c_type)
    in
    Interp.FunctionType (input_types, result_types)
  else failwith "Unsupported MLIR type"

let rec transform_operation (c_op : Bindings.mlir_operation) : Interp.operation
    =
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
      let res_val = Bindings.operation_get_result c_op (Intptr.of_int 0) in
      let res_c_type = Bindings.value_get_type res_val in
      let res_type = transform_mlir_type res_c_type in
      Interp.Op (results, Arith_Constant (value, res_type))
  | "arith.addi" ->
      let op1_val = Bindings.operation_get_operand c_op (Intptr.of_int 0) in
      let op2_val = Bindings.operation_get_operand c_op (Intptr.of_int 1) in
      let res_val = Bindings.operation_get_result c_op (Intptr.of_int 0) in
      let op1_name = get_value_name op1_val in
      let op2_name = get_value_name op2_val in
      let res_name = get_value_name res_val in
      let res_c_type = Bindings.value_get_type res_val in
      let res_type = transform_mlir_type res_c_type in
      Interp.Op ([ res_name ], Arith_AddI (op1_name, op2_name, res_type))
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
      Interp.Term (Func_Return operands)
  | _ -> failwith ("Unsupported operation: " ^ op_name)

and transform_operations_in_block (c_op : Bindings.mlir_operation) :
    Interp.operation list =
  if is_null c_op then []
  else
    let ocaml_op = transform_operation c_op in
    let next_op = Bindings.operation_get_next_in_block c_op in
    ocaml_op :: transform_operations_in_block next_op

and transform_block (c_block : Bindings.mlir_block) : Interp.block =
  let first_op = Bindings.block_get_first_operation c_block in
  let ops = transform_operations_in_block first_op in
  (* TODO: get block name and args *)
  {
    Interp.block_name = "entry";
    (* placeholder *)
    Interp.block_args = [];
    (* placeholder *)
    Interp.block_ops = ops;
  }

and transform_region (c_region : Bindings.mlir_region) : Interp.region =
  let first_block = Bindings.region_get_first_block c_region in
  if is_null first_block then [] else [ transform_block first_block ]
(* TODO: Handle multiple blocks *)

(* Transforms a func.func operation into our AST *)
let transform_func (c_op : Bindings.mlir_operation) : Interp.mlir_func =
  let op_name_ident = Bindings.operation_get_identifier c_op in
  let op_name_ref = Bindings.operation_get_name op_name_ident in
  let _op_name = string_of_mlir_string_ref op_name_ref in

  (* This is a simplified placeholder. *)
  let func_name = "main" in
  (* Placeholder *)

  let func_c_type_attr =
    Bindings.operation_get_attribute_by_name c_op
      (Bindings.string_ref_create_from_string "function_type")
  in
  let func_c_type =
    if Bindings.attribute_is_null func_c_type_attr then
      failwith "func.func operation missing 'function_type' attribute (null)"
    else if Bindings.attribute_is_a_type func_c_type_attr then
      Bindings.type_attr_get_value func_c_type_attr
    else
      failwith
        "func.func operation 'function_type' attribute is not a type attribute"
  in
  let func_type = transform_mlir_type func_c_type in

  let num_regions = Bindings.operation_get_num_regions c_op |> Intptr.to_int in
  let body =
    if num_regions > 0 then
      let region = Bindings.operation_get_region c_op (Intptr.of_int 0) in
      transform_region region
    else []
  in
  Interp.FuncOp (func_name, func_type, body)

(* Transforms a C-API mlirModule into our OCaml AST for mlir_program *)
let transform_module (c_module : Bindings.mlir_module) : Interp.mlir_program =
  reset_value_map ();
  let top_level_op = Bindings.module_get_operation c_module in

  (* A module has one region with one block containing the top-level operations
     (functions, etc.) *)
  let region = Bindings.operation_get_region top_level_op (Intptr.of_int 0) in
  let block = Bindings.region_get_first_block region in

  let rec transform_top_level_ops (c_op : Bindings.mlir_operation) :
      Interp.mlir_program =
    if is_null c_op then []
    else
      let func = transform_func c_op in
      let next_op = Bindings.operation_get_next_in_block c_op in
      func :: transform_top_level_ops next_op
  in

  let first_op = Bindings.block_get_first_operation block in
  transform_top_level_ops first_op
