#!/usr/bin/env bash
# Retry kaggle competitions submit when API returns 429 (rate limit).
# Usage: ./scripts/kaggle_submit_retry.sh /ABS/PATH/to.csv "message"
set -euo pipefail
FILE="${1:?csv path}"
MSG="${2:-submit}"
COMP="${KAGGLE_COMP:-hrt-eth-zurich-datathon-2026}"
CFG="${KAGGLE_CONFIG_DIR:-/Users/mgershman/Desktop/datathon/.kaggle}"
KAGGLE="${KAGGLE_BIN:-/Users/mgershman/Desktop/datathon/.venv/bin/kaggle}"
unset KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY || true
export KAGGLE_CONFIG_DIR="$CFG"

for attempt in 1 2 3 4 5 6 7 8; do
  echo "attempt $attempt ..."
  if env -i PATH="/Users/mgershman/Desktop/datathon/.venv/bin:/usr/bin:/bin" HOME="${HOME:-}" \
    KAGGLE_CONFIG_DIR="$CFG" \
    "$KAGGLE" competitions submit -c "$COMP" -f "$FILE" -m "$MSG"; then
    exit 0
  fi
  sleep "${KAGGLE_RETRY_SLEEP:-90}"
done
exit 1
