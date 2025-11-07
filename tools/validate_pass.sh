#!/bin/bash
# Script to validate an MLIR optimization pass using translation validation

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <pass-name> <input.mlir> <output-dir>"
    echo "Example: $0 sccp test.mlir validation_output/"
    exit 1
fi

PASS_NAME=$1
INPUT_FILE=$2
OUTPUT_DIR=$3

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# File paths
BEFORE_FILE="$OUTPUT_DIR/before.mlir"
AFTER_FILE="$OUTPUT_DIR/after.mlir"
COQ_FILE="$OUTPUT_DIR/${PASS_NAME}_validation.v"

echo "=== Translation Validation for $PASS_NAME pass ==="

# Copy original file
echo "1. Preparing original program..."
cp "$INPUT_FILE" "$BEFORE_FILE"

# Apply optimization pass
echo "2. Applying $PASS_NAME pass..."
mlir-opt "--$PASS_NAME" "$BEFORE_FILE" > "$AFTER_FILE" || {
    echo "Error: Failed to apply $PASS_NAME pass"
    echo "Make sure mlir-opt is installed and the pass name is correct"
    exit 1
}

# Convert to Coq
echo "3. Converting to Coq definitions..."
dune exec mlir2coq -- --pair "$BEFORE_FILE" "$AFTER_FILE" "$COQ_FILE" || {
    echo "Error: Failed to convert MLIR to Coq"
    echo "Make sure the project is built with: dune build"
    exit 1
}

echo "4. Generated Coq file: $COQ_FILE"

# Create validation proof template
PROOF_FILE="$OUTPUT_DIR/${PASS_NAME}_proof.v"
cat > "$PROOF_FILE" << 'EOF'
(** Proof of correctness for PASS_NAME optimization *)

Require Import MlirSem.TranslationValidation.Framework.
Require Import PASS_NAME_validation.

Theorem PASS_NAME_correct :
  prog_equiv program_before program_after.
Proof.
  unfold prog_equiv.
  intros func_name.

  (* Analyze both programs *)
  simpl.

  (* TODO: Complete the proof *)
  (* Hint: Use tv_simp and other tactics from Framework.v *)
Admitted.
EOF

sed "s/PASS_NAME/$PASS_NAME/g" "$PROOF_FILE" > "$PROOF_FILE.tmp" && mv "$PROOF_FILE.tmp" "$PROOF_FILE"

echo "5. Generated proof template: $PROOF_FILE"
echo ""
echo "=== Next Steps ==="
echo "1. Review the generated Coq definitions in: $COQ_FILE"
echo "2. Complete the proof in: $PROOF_FILE"
echo "3. Add the proof to src/TranslationValidation/ and update dune"
echo ""
echo "To run the oracle test comparison:"
echo "  dune exec driver_run -- $BEFORE_FILE"
echo "  dune exec driver_run -- $AFTER_FILE"
echo "And verify they produce the same output."