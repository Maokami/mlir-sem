(* Exports OCaml AST to Coq definition files *)
open Mlir_sem_extracted.Interp
open Printf

(* Helper to convert mlir_value (Big_int_Z) to Coq numeral notation *)
let z_to_coq (z : mlir_value) : string =
  let str = Big_int_Z.string_of_big_int z in
  if Big_int_Z.sign_big_int z >= 0 then
    sprintf "(%s)%%Z" str
  else
    sprintf "(-%s)%%Z" (Big_int_Z.string_of_big_int (Big_int_Z.abs_big_int z))

(* Convert value_id to Coq *)
let value_id_to_coq (id : value_id) : string =
  sprintf "\"%s\"" id

(* Convert block_id to Coq *)
let block_id_to_coq (id : block_id) : string =
  sprintf "\"%s\"" id

(* Convert mlir_type to Coq *)
let rec mlir_type_to_coq (typ : mlir_type) : string =
  match typ with
  | Integer width ->
      sprintf "(Integer %s)" (z_to_coq width)
  | FunctionType (args, rets) ->
      sprintf "(FunctionType [%s] [%s])"
        (String.concat "; " (List.map mlir_type_to_coq args))
        (String.concat "; " (List.map mlir_type_to_coq rets))

(* Convert comparison predicate to Coq *)
let arith_cmp_pred_to_coq (pred : arith_cmp_pred) : string =
  match pred with
  | Eq0 -> "eq"
  | Ne -> "ne"
  | Slt -> "slt"
  | Sle -> "sle"
  | Sgt -> "sgt"
  | Sge -> "sge"
  | Ult -> "ult"
  | Ule -> "ule"

(* Convert general_op to Coq *)
let general_op_to_coq (op : general_op) : string =
  match op with
  | Arith_Constant (value, res_type) ->
      sprintf "(Arith_Constant %s %s)"
        (z_to_coq value) (mlir_type_to_coq res_type)
  | Arith_AddI (lhs, rhs, res_type) ->
      sprintf "(Arith_AddI %s %s %s)"
        (value_id_to_coq lhs) (value_id_to_coq rhs) (mlir_type_to_coq res_type)
  | Arith_CmpI (pred, lhs, rhs, res_type) ->
      sprintf "(Arith_CmpI %s %s %s %s)"
        (arith_cmp_pred_to_coq pred) (value_id_to_coq lhs)
        (value_id_to_coq rhs) (mlir_type_to_coq res_type)
  | Func_Call (callee, args, res_type) ->
      sprintf "(Func_Call \"%s\" [%s] %s)"
        callee
        (String.concat "; " (List.map value_id_to_coq args))
        (mlir_type_to_coq res_type)

(* Convert terminator_op to Coq *)
let terminator_op_to_coq (op : terminator_op) : string =
  match op with
  | Func_Return vals ->
      sprintf "(Func_Return [%s])"
        (String.concat "; " (List.map value_id_to_coq vals))
  | Cf_Branch (dest, args) ->
      sprintf "(Cf_Branch %s [%s])"
        (block_id_to_coq dest)
        (String.concat "; " (List.map value_id_to_coq args))
  | Cf_CondBranch (cond, true_dest, true_args, false_dest, false_args) ->
      sprintf "(Cf_CondBranch %s %s [%s] %s [%s])"
        (value_id_to_coq cond)
        (block_id_to_coq true_dest)
        (String.concat "; " (List.map value_id_to_coq true_args))
        (block_id_to_coq false_dest)
        (String.concat "; " (List.map value_id_to_coq false_args))

(* Convert operation to Coq *)
let operation_to_coq (op : operation) : string =
  match op with
  | Op (results, gen_op) ->
      let results_str =
        sprintf "[%s]" (String.concat "; " (List.map value_id_to_coq results))
      in
      sprintf "(Op %s %s)" results_str (general_op_to_coq gen_op)
  | Term term_op ->
      sprintf "(Term %s)" (terminator_op_to_coq term_op)

(* Convert block to Coq *)
let block_to_coq (blk : block) : string =
  let args_str =
    if blk.block_args = [] then "[]"
    else
      let arg_to_coq (name, typ) =
        sprintf "(%s, %s)" (value_id_to_coq name) (mlir_type_to_coq typ)
      in
      sprintf "[%s]" (String.concat "; " (List.map arg_to_coq blk.block_args))
  in
  let ops_str =
    if blk.block_ops = [] then "[]"
    else
      sprintf "[\n      %s\n    ]"
        (String.concat ";\n      " (List.map operation_to_coq blk.block_ops))
  in
  sprintf "  {| block_name := %s;\n     block_args := %s;\n     block_ops := %s |}"
    (block_id_to_coq blk.block_name)
    args_str
    ops_str

(* Convert region to Coq *)
let region_to_coq (region : region) : string =
  if region = [] then "[]"
  else
    sprintf "[\n%s\n  ]" (String.concat ";\n" (List.map block_to_coq region))

(* Convert mlir_func to Coq *)
let mlir_func_to_coq (func : mlir_func) : string =
  match func with
  | FuncOp (name, typ, body) ->
      sprintf "(FuncOp \"%s\" %s\n  %s)"
        name (mlir_type_to_coq typ) (region_to_coq body)

(* Convert mlir_program to Coq definition *)
let mlir_program_to_coq (name : string) (prog : mlir_program) : string =
  let funcs_str =
    if prog = [] then "[]"
    else
      sprintf "[\n%s\n]" (String.concat ";\n" (List.map mlir_func_to_coq prog))
  in
  sprintf "Definition %s : mlir_program :=\n  %s.\n" name funcs_str

(* Export a program to a Coq file *)
let export_to_coq_file (filename : string) (prog_name : string) (prog : mlir_program) =
  try
    let oc = open_out filename in
    fprintf oc "(* Auto-generated from MLIR by coq_exporter.ml *)\n";
    fprintf oc "Require Import MlirSem.Syntax.AST.\n";
    fprintf oc "Require Import ZArith.\n";
    fprintf oc "Open Scope Z_scope.\n\n";
    fprintf oc "%s\n" (mlir_program_to_coq prog_name prog);
    close_out oc;
    printf "Exported Coq definition to %s\n" filename
  with
  | Sys_error msg ->
      eprintf "Error writing to file %s: %s\n" filename msg;
      exit 1

(* Export two programs for translation validation *)
let export_translation_pair ~before_file ~after_file ~before_name ~after_name
    ~before_prog ~after_prog ~output_file =
  try
    let oc = open_out output_file in
    fprintf oc "(* Auto-generated translation validation pair *)\n";
    fprintf oc "Require Import MlirSem.Syntax.AST.\n";
    fprintf oc "Require Import ZArith.\n";
    fprintf oc "Open Scope Z_scope.\n\n";
    fprintf oc "(* Original program from: %s *)\n" before_file;
    fprintf oc "%s\n" (mlir_program_to_coq before_name before_prog);
    fprintf oc "(* Optimized program from: %s *)\n" after_file;
    fprintf oc "%s\n" (mlir_program_to_coq after_name after_prog);
    fprintf oc "(* End of auto-generated definitions *)\n";
    close_out oc;
    printf "Exported translation validation pair to %s\n" output_file
  with
  | Sys_error msg ->
      eprintf "Error writing to file %s: %s\n" output_file msg;
      exit 1