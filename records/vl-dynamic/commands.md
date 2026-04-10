cd /workspace/golf/parameter-golf

# V1 seed 42
RUN_ID=codebook_v1_seed42 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook.py \
  > logs/codebook_v1_seed42.log 2>&1 &

tail -f logs/codebook_v1_seed42.log

# V1 seed 2024
RUN_ID=codebook_v1_seed2024 \
SEED=2024 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook.py \
  > logs/codebook_v1_seed2024.log 2>&1 &

tail -f logs/codebook_v1_seed2024.log

# Baseline seed 42
RUN_ID=baseline_seed42 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  > logs/baseline_seed42.log 2>&1 &

# Baseline seed 2024
RUN_ID=baseline_seed2024 \
SEED=2024 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  > logs/baseline_seed2024.log 2>&1 &