From Stdlib Require Import String ZArith List Extraction.
From Stdlib Require Import ExtrOcamlBasic ExtrOcamlNativeString ExtrOcamlZBigInt.
From MlirSem Require Import Syntax.AST Semantics.Values Semantics.Events Semantics.Denotation Semantics.Interp.

Extraction Language OCaml.
Set Extraction AccessOpaque.
Set Extraction Output Directory ".".

(* Extract specific definitions to avoid Monolithic Extraction error *)
Extraction "AST.ml" mlir_type arith_cmp_pred general_op terminator_op operation block region mlir_func mlir_program.
Extraction "Values.ml" mlir_value.
Extraction "Events.ml" LocalE FunctionE ControlE FailureE MlirSemE raise.
Extraction "Denotation.ml" denote_general_op denote_terminator denote_block denote_func.
Extraction "Interp.ml" call_frame function_def program_context interpreter_state handle_event interpret build_program_context run_program.