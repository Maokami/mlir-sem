open Mlir_sem_extracted

let rec string_of_mlir_type (t : Interp.mlir_type) : string =
  match t with
  | Interp.Integer width -> "i" ^ Z.to_string width
  | Interp.FunctionType (args, rets) ->
      "("
      ^ String.concat ", " (List.map string_of_mlir_type args)
      ^ ") -> ("
      ^ String.concat ", " (List.map string_of_mlir_type rets)
      ^ ")"

let string_of_arith_cmp_pred (p : Interp.arith_cmp_pred) : string =
  match p with
  | Interp.Eq0 -> "eq"
  | Interp.Ne -> "ne"
  | Interp.Slt -> "slt"
  | Interp.Sle -> "sle"
  | Interp.Sgt -> "sgt"
  | Interp.Sge -> "sge"
  | Interp.Ult -> "ult"
  | Interp.Ule -> "ule"

let string_of_general_op (op : Interp.general_op) : string =
  match op with
  | Interp.Arith_Constant (value, ty) ->
      "arith.constant " ^ Z.to_string value ^ " : " ^ string_of_mlir_type ty
  | Interp.Arith_AddI (lhs, rhs, ty) ->
      "arith.addi " ^ lhs ^ ", " ^ rhs ^ " : " ^ string_of_mlir_type ty
  | Interp.Arith_CmpI (pred, lhs, rhs, ty) ->
      "arith.cmpi \""
      ^ string_of_arith_cmp_pred pred
      ^ ", " ^ lhs ^ ", " ^ rhs ^ " : " ^ string_of_mlir_type ty
  | Interp.Func_Call (callee, args, fty) -> (
      let args_s = String.concat ", " args in
      match fty with
      | Interp.FunctionType (arg_tys, ret_tys) ->
          let arg_tys_s =
            "("
            ^ String.concat ", " (List.map string_of_mlir_type arg_tys)
            ^ ")"
          in
          let ret_tys_s =
            "("
            ^ String.concat ", " (List.map string_of_mlir_type ret_tys)
            ^ ")"
          in
          "func.call @" ^ callee ^ "(" ^ args_s ^ ") : " ^ arg_tys_s ^ " -> "
          ^ ret_tys_s
      | _ ->
          "func.call @" ^ callee ^ "(" ^ args_s ^ ") : "
          ^ string_of_mlir_type fty)

let string_of_terminator_op (op : Interp.terminator_op) : string =
  match op with
  | Interp.Func_Return vals -> "func.return " ^ String.concat ", " vals
  | Interp.Cf_Branch (dest, args) ->
      "cf.br ^" ^ dest ^ "(" ^ String.concat ", " args ^ ")"
  | Interp.Cf_CondBranch (cond, true_dest, true_args, false_dest, false_args) ->
      "cf.cond_br " ^ cond ^ ", ^" ^ true_dest ^ "("
      ^ String.concat ", " true_args
      ^ "), ^" ^ false_dest ^ "("
      ^ String.concat ", " false_args
      ^ ")"

let string_of_operation (op : Interp.operation) : string =
  match op with
  | Interp.Op (results, g_op) ->
      (if List.length results > 0 then String.concat ", " results ^ " = "
       else "")
      ^ string_of_general_op g_op
  | Interp.Term t_op -> string_of_terminator_op t_op

let string_of_block (block : Interp.block) : string =
  let args_str =
    if List.length block.block_args > 0 then
      "("
      ^ String.concat ", "
          (List.map
             (fun (id, ty) -> id ^ ":" ^ string_of_mlir_type ty)
             block.block_args)
      ^ ")"
    else ""
  in
  let ops_str =
    String.concat "\n  " (List.map string_of_operation block.block_ops)
  in
  block.block_name ^ args_str ^ ":\n  " ^ ops_str

let string_of_mlir_func (f : Interp.mlir_func) : string =
  match f with
  | Interp.FuncOp (name, ty, body) ->
      "func.func @" ^ name ^ " " ^ string_of_mlir_type ty ^ " {\n"
      ^ String.concat "\n" (List.map string_of_block body)
      ^ "\n}"

let string_of_mlir_program (prog : Interp.mlir_program) : string =
  String.concat "\n\n" (List.map string_of_mlir_func prog)
