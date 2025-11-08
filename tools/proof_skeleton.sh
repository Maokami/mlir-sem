#!/bin/bash
# Generate proof skeleton template for admitted proofs
# Usage: ./tools/proof_skeleton.sh <file.v> <lemma_name>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file.v> <lemma_name>"
    echo ""
    echo "Example: $0 src/Utils/InterpLemmas.v interpret_read_pure"
    exit 1
fi

FILE="$1"
LEMMA="$2"

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found"
    exit 1
fi

# Find the lemma in the file
if ! grep -q "^Lemma $LEMMA\|^Theorem $LEMMA\|^Axiom $LEMMA" "$FILE"; then
    echo "Error: Lemma '$LEMMA' not found in $FILE"
    exit 1
fi

echo "📝 Generating proof skeleton for: $LEMMA"
echo ""

# Extract lemma signature
echo "Lemma signature:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sed -n "/^Lemma $LEMMA/,/^Proof\./p" "$FILE" | head -n -1
echo ""

# Check if it's already admitted
if grep -A 2 "^Lemma $LEMMA" "$FILE" | grep -q "admit\."; then
    echo "Status: ⚠️  Currently admitted"
else
    echo "Status: ℹ️  Proof may be incomplete"
fi

echo ""
echo "Suggested proof skeleton:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat <<'EOF'
Proof.
  (* Step 1: Introduce all variables *)
  intros.

  (* Step 2: Unfold key definitions *)
  unfold (* list definitions here *).

  (* Step 3: Simplify *)
  simpl.

  (* Step 4: Rewrite with known lemmas *)
  (* rewrite lemma_name. *)

  (* Step 5: Case analysis if needed *)
  (* destruct ... *)

  (* Step 6: Complete the proof *)
  (* reflexivity / auto / ... *)

  (* TODO: Complete from here *)
  admit.
Admitted.
EOF

echo ""
echo "💡 Tips:"
echo "  1. Start with 'intros' to check variable types"
echo "  2. Use 'unfold' to expand definitions and catch type errors early"
echo "  3. Build incrementally - run 'dune build' after each step"
echo "  4. Add explicit type applications if you see 'The term ... has type ... while it is expected to have type Type'"

