cd /workspace/golf/parameter-golf

# V1 seed 42
RUN_ID=codebook_v1_seed42 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook_v1.py \
  > logs/codebook_v1_seed42.log 2>&1 &

tail -f logs/codebook_v1_seed42.log

# V1 seed 2024
RUN_ID=codebook_v1_seed2024 \
SEED=2024 \
MAX_WALLCLOCK_SECONDS=600 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook_v1.py \
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

archivoqué estrain_gpt_codebook_v1.pycodebook puro, INT8 original, MLP3 tiedtrain_gpt_codebook_v4alpha.pycodebook + repulsión volumétricatrain_gpt_codebook_v5.pycodebook + SDClip de Kevin + 11L + MLP4 untiedtrain_gpt_codebook_v6.pycodebook + VSingularity

----------------------

# V5 — codebook + SDClip + 11L MLP4 untied
RUN_ID=codebook_v5_final \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=11 \
MLP_MULT=4 \
TIE_EMBEDDINGS=0 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook_v5.py \
  > logs/codebook_v5_final.log 2>&1 &

# V4-Alpha — codebook + repulsión
RUN_ID=codebook_v4alpha \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=11 \
TIE_EMBEDDINGS=0 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook_v4alpha.py \
  > logs/codebook_v4alpha.log 2>&1 &

# V6 — codebook + VSingularity
RUN_ID=codebook_v6 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=11 \
TIE_EMBEDDINGS=0 \
nohup torchrun --standalone --nproc_per_node=2 \
  records/vl-dynamic/train_gpt_codebook_v6.py \
  > logs/codebook_v6.log 2>&1 &