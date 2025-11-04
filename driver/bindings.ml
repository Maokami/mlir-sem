open Ctypes
open Foreign

(* Opaque pointers for MLIR C API types *)
type mlir_context = unit ptr

let mlir_context : mlir_context typ = ptr void

type mlir_dialect_handle = unit ptr

let dialect_handle : mlir_dialect_handle typ = ptr void

type mlir_operation = unit ptr

let mlir_operation : mlir_operation typ = ptr void

type mlir_module = unit ptr

let mlir_module : mlir_module typ = ptr void

type mlir_region = unit ptr

let mlir_region : mlir_region typ = ptr void

type mlir_block = unit ptr

let mlir_block : mlir_block typ = ptr void

type mlir_value = unit ptr

let mlir_value : mlir_value typ = ptr void

type mlir_type = unit ptr

let mlir_type : mlir_type typ = ptr void

type mlir_type_id = unit ptr

let mlir_type_id : mlir_type_id typ = ptr void

type mlir_attribute = unit ptr

let mlir_attribute : mlir_attribute typ = ptr void

type mlir_identifier = unit ptr

let mlir_identifier : mlir_identifier typ = ptr void

(* A C-style string reference *)
type mlir_string_ref

let mlir_string_ref : mlir_string_ref structure typ = structure "MlirStringRef"
let data = field mlir_string_ref "data" (ptr char)
let length = field mlir_string_ref "length" size_t
let () = seal mlir_string_ref

(* Callback for printing *)
type mlir_string_callback = mlir_string_ref -> unit ptr -> unit

let mlir_string_callback =
  funptr (mlir_string_ref @-> ptr void @-> returning void)

(* === Function bindings === *)

(* Context, Dialect, and Module Management *)
let context_create =
  foreign "mlirContextCreate" (void @-> returning mlir_context)

let context_destroy =
  foreign "mlirContextDestroy" (mlir_context @-> returning void)

let register_dialect =
  foreign "mlirDialectHandleRegisterDialect"
    (dialect_handle @-> mlir_context @-> returning void)

let get_func_dialect =
  foreign "mlirGetDialectHandle__func__" (void @-> returning dialect_handle)

let get_arith_dialect =
  foreign "mlirGetDialectHandle__arith__" (void @-> returning dialect_handle)

let get_cf_dialect =
  foreign "mlirGetDialectHandle__cf__" (void @-> returning dialect_handle)

let string_ref_create_from_string =
  foreign "mlirStringRefCreateFromCString" (string @-> returning mlir_string_ref)

let module_create_parse =
  foreign "mlirModuleCreateParse"
    (mlir_context @-> mlir_string_ref @-> returning mlir_module)

let module_is_null (m : mlir_module) = Ctypes.is_null m
let module_destroy = foreign "mlirModuleDestroy" (mlir_module @-> returning void)

(* IR Traversal *)
let module_get_operation =
  foreign "mlirModuleGetOperation" (mlir_module @-> returning mlir_operation)

let operation_get_name =
  foreign "mlirIdentifierStr" (mlir_identifier @-> returning mlir_string_ref)

let operation_get_identifier =
  foreign "mlirOperationGetName" (mlir_operation @-> returning mlir_identifier)

let operation_get_num_regions =
  foreign "mlirOperationGetNumRegions" (mlir_operation @-> returning intptr_t)

let operation_get_region =
  foreign "mlirOperationGetRegion"
    (mlir_operation @-> intptr_t @-> returning mlir_region)

let region_get_first_block =
  foreign "mlirRegionGetFirstBlock" (mlir_region @-> returning mlir_block)

let block_get_first_operation =
  foreign "mlirBlockGetFirstOperation" (mlir_block @-> returning mlir_operation)

let operation_get_next_in_block =
  foreign "mlirOperationGetNextInBlock"
    (mlir_operation @-> returning mlir_operation)

let operation_get_num_successors =
  foreign "mlirOperationGetNumSuccessors" (mlir_operation @-> returning intptr_t)

let operation_get_successor =
  foreign "mlirOperationGetSuccessor"
    (mlir_operation @-> intptr_t @-> returning mlir_block)

let block_get_num_arguments =
  foreign "mlirBlockGetNumArguments" (mlir_block @-> returning intptr_t)

let block_get_argument =
  foreign "mlirBlockGetArgument"
    (mlir_block @-> intptr_t @-> returning mlir_value)

let block_get_next_in_region =
  foreign "mlirBlockGetNextInRegion" (mlir_block @-> returning mlir_block)

(* Operations and Attributes *)
let operation_get_num_results =
  foreign "mlirOperationGetNumResults" (mlir_operation @-> returning intptr_t)

let operation_get_result =
  foreign "mlirOperationGetResult"
    (mlir_operation @-> intptr_t @-> returning mlir_value)

let operation_get_num_operands =
  foreign "mlirOperationGetNumOperands" (mlir_operation @-> returning intptr_t)

let operation_get_operand =
  foreign "mlirOperationGetOperand"
    (mlir_operation @-> intptr_t @-> returning mlir_value)

let operation_get_attribute_by_name =
  foreign "mlirOperationGetAttributeByName"
    (mlir_operation @-> mlir_string_ref @-> returning mlir_attribute)

let attribute_is_a_integer =
  foreign "mlirAttributeIsAInteger" (mlir_attribute @-> returning bool)

let attribute_is_a_string =
  foreign "mlirAttributeIsAString" (mlir_attribute @-> returning bool)

let integer_attr_get_value_int =
  foreign "mlirIntegerAttrGetValueInt" (mlir_attribute @-> returning int64_t)

let attribute_is_a_dense_elements =
  foreign "mlirAttributeIsADenseElements" (mlir_attribute @-> returning bool)

let attribute_is_a_dense_int_elements =
  foreign "mlirAttributeIsADenseIntElements" (mlir_attribute @-> returning bool)

let dense_elements_attr_is_splat =
  foreign "mlirDenseElementsAttrIsSplat" (mlir_attribute @-> returning bool)

let dense_elements_attr_get_int64_splat_value =
  foreign "mlirDenseElementsAttrGetInt64SplatValue"
    (mlir_attribute @-> returning int64_t)

let dense_elements_attr_get_int64_value =
  foreign "mlirDenseElementsAttrGetInt64Value"
    (mlir_attribute @-> intptr_t @-> returning int64_t)

let string_attr_get_value =
  foreign "mlirStringAttrGetValue" (mlir_attribute @-> returning mlir_string_ref)

let attribute_is_a_type =
  foreign "mlirAttributeIsAType" (mlir_attribute @-> returning bool)

let attribute_get_type =
  foreign "mlirAttributeGetType" (mlir_attribute @-> returning mlir_type)

let attribute_is_null (attr : mlir_attribute) = Ctypes.is_null attr

let type_attr_get_value =
  foreign "mlirTypeAttrGetValue" (mlir_attribute @-> returning mlir_type)

(* Types *)
let operation_get_type =
  foreign "mlirOperationGetTypeID" (mlir_operation @-> returning mlir_type_id)

let value_get_type =
  foreign "mlirValueGetType" (mlir_value @-> returning mlir_type)

let type_is_a_function =
  foreign "mlirTypeIsAFunction" (mlir_type @-> returning bool)

let function_type_get_num_inputs =
  foreign "mlirFunctionTypeGetNumInputs" (mlir_type @-> returning intptr_t)

let function_type_get_input =
  foreign "mlirFunctionTypeGetInput"
    (mlir_type @-> intptr_t @-> returning mlir_type)

let function_type_get_num_results =
  foreign "mlirFunctionTypeGetNumResults" (mlir_type @-> returning intptr_t)

let function_type_get_result =
  foreign "mlirFunctionTypeGetResult"
    (mlir_type @-> intptr_t @-> returning mlir_type)

let type_is_a_integer =
  foreign "mlirTypeIsAInteger" (mlir_type @-> returning bool)

let integer_type_get_width =
  foreign "mlirIntegerTypeGetWidth" (mlir_type @-> returning uint)

(* Printing *)
let operation_print =
  foreign "mlirOperationPrint"
    (mlir_operation @-> mlir_string_callback @-> ptr void @-> returning void)

let operation_dump =
  foreign "mlirOperationDump" (mlir_operation @-> returning void)

let type_dump = foreign "mlirTypeDump" (mlir_type @-> returning void)
