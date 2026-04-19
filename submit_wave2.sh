#!/usr/bin/env bash
set -u
unset KAGGLE_API_TOKEN
export KAGGLE_USERNAME='ldldoodidodood'
export KAGGLE_KEY='b4d2167892794aa0e9a1941504ffe65e'
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
    if echo "$out" | grep -q "Successfully submitted"; then sleep 8; return 0; fi
    sleep 30
  done
  return 1
}

for f in sub_base_plus_ridge_top10_905_095.csv sub_base_plus_ridge_top10_910_090.csv \
         sub_base_plus_ridge_top10_915_085.csv sub_base_plus_ridge_top10_925_075.csv \
         sub_base_plus_ridge_top10_928_072.csv sub_base_plus_ridge_top10_932_068.csv \
         sub_base_plus_ridge_top10_948_052.csv sub_base_plus_ridge_top10_960_040.csv \
         sub_base_plus_ridge_top10_965_035.csv; do
  submit "$f" "w2 rt10 sweep $f"
done

for f in sub_base_rt10_ra50_935_055_010.csv sub_base_rt10_ra50_935_060_005.csv \
         sub_base_rt10_ra50_930_060_010.csv sub_base_rt10_ra50_930_055_015.csv \
         sub_base_rt10_ra50_920_065_015.csv sub_base_rt10_ra50_925_065_010.csv; do
  submit "$f" "w2 3-branch base+rt10+ra50 $f"
done

for f in champ_amp1005.csv champ_amp1010.csv champ_amp1015.csv champ_amp1020.csv \
         champ_amp0995.csv champ_amp0990.csv; do
  submit "$f" "w2 amp on champion $f"
done
