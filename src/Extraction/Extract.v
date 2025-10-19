From Stdlib Require Import String ZArith List Extraction.
From Stdlib Require Import ExtrOcamlBasic ExtrOcamlNativeString ExtrOcamlZBigInt.
From MlirSem Require Import Syntax.AST.

Extraction Language OCaml.
Set Extraction AccessOpaque.
Set Extraction Output Directory ".".

Extraction "AST.ml" MlirSem.Syntax.AST.mlir_program.