#!/usr/bin/env bash
# Pre-commit hook: run architect-sim and fail if grade drops
# Install: cp scripts/pre-commit-check.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -e

echo "Running architecture analysis..."
RESULT=$(architect-sim --root . --analyze-only --format json 2>/dev/null)

GRADE=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['grade']['letter'])")
SCORE=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['grade']['overall'])")
CRITICAL=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['summary']['findings_critical'])")

echo "Grade: $GRADE ($SCORE/100) | Critical: $CRITICAL"

# Fail if grade is F
if [ "$GRADE" = "F" ]; then
    echo "Architecture grade is F. Fix critical issues before committing."
    exit 1
fi

echo "Architecture check passed."
