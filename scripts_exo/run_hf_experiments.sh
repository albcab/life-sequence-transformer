#!/usr/bin/env bash
# Prepare only: --dry-run. Execute: DEVICE=2 bash scripts/run_hf_experiments.sh
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE="$REPO/generated_smc-batched-20260922T083211Z/vno12_norm_pos_100sp/decoder_only"
SELECTION=()
if [[ -n ${EXPERIMENTS:-} ]]; then
    IFS=',' read -r -a SELECTED <<< "$EXPERIMENTS"
    SELECTION=(--experiments "${SELECTED[@]}")
fi
if [[ ${1:-} == --dry-run ]]; then
    exec python "$REPO/scripts/run_hf_suite.py" --plan --repo "$REPO" --baseline "$BASELINE" "${SELECTION[@]}"
fi
if [[ $# -gt 0 ]]; then
    echo 'Usage: run_hf_experiments.sh [--dry-run]' >&2
    exit 2
fi
DEVICE=${DEVICE:-2}
IMAGE=${IMAGE:-albcab/l2v:smc}
MODE=""
if [[ ${DFA_SMC:-0} == 1 ]]; then MODE="-dfa_smc"; fi
NAME="hf_causal_lm_exo-all${MODE}-gpu${DEVICE}-rolling_window_mode-$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
RUN_DIR="$REPO/logs/$NAME"
# Validate cohort/baseline coverage before creating a container.
python "$REPO/scripts/run_hf_suite.py" --plan --repo "$REPO" --baseline "$BASELINE" "${SELECTION[@]}" >/dev/null
USED=$(nvidia-smi --id="$DEVICE" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( USED > 100 )); then
    echo "GPU $DEVICE is occupied ($USED MiB); select a free DEVICE." >&2
    exit 1
fi
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
mkdir -p "$RUN_DIR" "$REPO/.cache/hf_dfa_smc"
# Freeze code/config used by the run; saved outputs remain under their own mounts.
cp -a "$REPO/scripts" "$REPO/src" "$REPO/conf" "$REPO/counter.py" "$RUN_DIR/"
EXTRA=("${SELECTION[@]}")
if [[ ${DFA_SMC:-0} == 1 ]]; then EXTRA+=(--dfa-smc); fi
COMMAND=(python -m scripts.run_hf_suite --repo /usr/src
    --baseline /usr/src/generated_smc-batched-20260922T083211Z/vno12_norm_pos_100sp/decoder_only
    --checkpoint /usr/src/checkpoints/hf_causal_lm_exo/hf_causal_lm_exo/1.0/1.0-epoch=14.ckpt
    --output /evaluation "${EXTRA[@]}")
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
docker run -d --name "$NAME" --gpus "device=$DEVICE" --shm-size=2g \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1 -e HF_DFA_CACHE=/dfa-cache -w /usr/src \
    --mount "type=bind,src=$REPO,dst=/usr/src,readonly" \
    --mount "type=bind,src=$RUN_DIR/scripts,dst=/usr/src/scripts,readonly" \
    --mount "type=bind,src=$RUN_DIR/src,dst=/usr/src/src,readonly" \
    --mount "type=bind,src=$RUN_DIR/conf,dst=/usr/src/conf,readonly" \
    --mount "type=bind,src=$RUN_DIR/counter.py,dst=/usr/src/counter.py,readonly" \
    --mount "type=bind,src=$REPO/logs/hf-llm-gpu2-20261001t074056z/huggingface,dst=/hf-cache,readonly" \
    --mount "type=bind,src=$REPO/.cache/hf_dfa_smc,dst=/dfa-cache" \
    --mount "type=bind,src=$RUN_DIR,dst=/evaluation" \
    "$IMAGE_ID" bash -o pipefail -c '"$@" 2>&1 | tee /evaluation/console.log; status=${PIPESTATUS[0]}; echo "$status" > /evaluation/exit_code.txt; exit "$status"' bash "${COMMAND[@]}"
echo "Logs: $RUN_DIR"
echo "Follow: docker logs -f $NAME"
