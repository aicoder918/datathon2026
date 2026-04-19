#!/usr/bin/env bash
set -u
unset KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/Users/mgershman/Desktop/datathon/.kaggle}"
KAGGLE=/Users/mgershman/Desktop/datathon/.venv/bin/kaggle
SUB=/Users/mgershman/Desktop/datathon/datathon2026/submissions
COMP=hrt-eth-zurich-datathon-2026

submit() {
  local f="$1" m="$2"
  for i in 1 2 3 4 5; do
    out=$(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        -u GIT_HTTP_PROXY -u GIT_HTTPS_PROXY \
        -u SOCKS_PROXY -u SOCKS5_PROXY -u socks_proxy -u socks5_proxy \
        "$KAGGLE" competitions submit -c "$COMP" -f "$SUB/$f" -m "$m" 2>&1 | tail -1)
    echo "[$f] $out"
    if echo "$out" | grep -q "Successfully submitted"; then
      sleep 8; return 0
    fi
    echo "retry $i for $f ..."
    sleep 30
  done
  return 1
}

submit sub_base_plus_ridgea50k_935_065.csv      "w1 sub ridgealpha50k 935/065"
submit sub_base_plus_ridgefitq30_935_065.csv    "w1 sub ridgefullimp_tq30 935/065"
submit sub_base_plus_plsc5_935_065.csv          "w1 sub plsc5 935/065"
submit sub_base_plus_plsc3_935_065.csv          "w1 sub plsc3 935/065"
submit champ_plus_ridgea50k_990_010.csv         "w1 3rd champ+ridgea50k 990/010"
submit champ_plus_ridgea50k_980_020.csv         "w1 3rd champ+ridgea50k 980/020"
submit champ_plus_plsc5_990_010.csv             "w1 3rd champ+plsc5 990/010"
submit champ_plus_plsc5_980_020.csv             "w1 3rd champ+plsc5 980/020"
submit champ_plus_cb30_990_010.csv              "w1 3rd champ+cb_bootstrap30 990/010"
submit champ_plus_cb30_980_020.csv              "w1 3rd champ+cb_bootstrap30 980/020"
