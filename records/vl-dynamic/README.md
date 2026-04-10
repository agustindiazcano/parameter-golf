https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th

# 1. Clone
cd /workspace
git clone https://github.com/agustindiazcano/parameter-golf.git
cd parameter-golf

# 2. Dependencies
pip install sentencepiece scikit-learn

# 3. Data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# 4. Logs
mkdir -p logs

