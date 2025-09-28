#!/usr/bin/env bash
set -euo pipefail

# 可配置参数
SCRIPT="run_basic.py"         
MODEL="SASRec"              
DATASET="Amazon_Beauty"
BASE_CFG="sasrec.yaml"  

# 随机种子范围（更大范围与步长）
START_SEED=200
END_SEED=2000
STEP=150

echo "Running $SCRIPT model=$MODEL dataset=$DATASET seeds ${START_SEED}..${END_SEED} step ${STEP}" 

s=$START_SEED
while [ $s -le $END_SEED ]; do
  TMP="tmp_seed_${s}.yaml"
  # 复制基础配置并覆盖/追加 seed 字段
  if grep -Eq '^\s*seed\s*:\s*' "$BASE_CFG"; then
    sed -E "s/^\s*seed\s*:\s*\S+/seed: ${s}/" "$BASE_CFG" > "$TMP"
  else
    cat "$BASE_CFG" > "$TMP"
    echo "seed: ${s}" >> "$TMP"
  fi

  echo "==> Seed ${s}"
  echo "python ${SCRIPT} --model ${MODEL} --dataset ${DATASET} --config_files ${TMP}"
  python "$SCRIPT" --model "$MODEL" --dataset "$DATASET" --config_files "$TMP"

  rm -f "$TMP"
  s=$(( s + STEP ))
done

echo "All runs completed."







