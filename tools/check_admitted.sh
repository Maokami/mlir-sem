#!/bin/bash
# Check admitted proofs in the project
# Usage: ./tools/check_admitted.sh [--max N] [--fail]

set -e

# Default values
MAX_ADMITTED=15
FAIL_ON_EXCEED=false
SHOW_DETAILS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max)
            MAX_ADMITTED="$2"
            shift 2
            ;;
        --fail)
            FAIL_ON_EXCEED=true
            shift
            ;;
        --details)
            SHOW_DETAILS=true
            shift
            ;;
        *)
            echo "Usage: $0 [--max N] [--fail] [--details]"
            echo "  --max N      Set maximum allowed admitted proofs (default: 15)"
            echo "  --fail       Exit with error if limit exceeded"
            echo "  --details    Show list of all admitted proofs"
            exit 1
            ;;
    esac
done

# Count admitted proofs
ADMITTED_FILES=$(grep -rl "Admitted\." src/ 2>/dev/null || true)
ADMITTED_COUNT=0

if [ -n "$ADMITTED_FILES" ]; then
    ADMITTED_COUNT=$(echo "$ADMITTED_FILES" | xargs -d '\n' grep "Admitted\." 2>/dev/null | wc -l | tr -d ' ')
fi

echo "📊 Admitted Proof Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total admitted proofs: $ADMITTED_COUNT"
echo "Maximum allowed: $MAX_ADMITTED"

if [ "$ADMITTED_COUNT" -le "$MAX_ADMITTED" ]; then
    echo "Status: ✅ PASS"
else
    echo "Status: ⚠️  EXCEEDED LIMIT"
fi

if [ "$SHOW_DETAILS" = true ] && [ "$ADMITTED_COUNT" -gt 0 ]; then
    echo ""
    echo "Admitted proofs by file:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for file in $ADMITTED_FILES; do
        count=$(grep "Admitted\." "$file" | wc -l | tr -d ' ')
        if [ "$count" -gt 0 ]; then
            echo "  $file: $count"

            # Show lemma names
            grep -B 5 "Admitted\." "$file" | grep "^Lemma\|^Theorem\|^Axiom" | sed 's/^/    /'
        fi
    done
fi

echo ""

# Exit with error if requested and limit exceeded
if [ "$FAIL_ON_EXCEED" = true ] && [ "$ADMITTED_COUNT" -gt "$MAX_ADMITTED" ]; then
    echo "❌ Error: Too many admitted proofs ($ADMITTED_COUNT > $MAX_ADMITTED)"
    echo "   Please complete some proofs before adding more."
    exit 1
fi

exit 0
