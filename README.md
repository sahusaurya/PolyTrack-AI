# 🏎️ PolyTrack-AI
Machine Learning Project to beat Polytrack.

An imitation + reinforcement learning agent trained to master the web game **[PolyTrack](https://www.kodub.com/apps/polytrack)**.

## 🎯 Overview
The project uses:
- **Imitation Learning (Behavioral Cloning)** to mimic human gameplay.
- **Reinforcement Learning (PPO)** to fine-tune performance and surpass human-level scores.

## 🧠 Key Components
- `scripts/record_polytrack.py` – Records gameplay with screenshots + actions.
- `scripts/train_bc.py` – Trains a CNN policy on recorded demos.
- `envs/polytrack_env.py` – Custom Gym wrapper for PolyTrack.
- `scripts/fine_tune_rl.py` – Reinforcement Learning fine-tuning (coming soon).

## ⚙️ Setup
```bash
git clone https://github.com/<your-username>/polytrack-ai.git
cd polytrack-ai
python3 -m venv polytrack-env
source polytrack-env/bin/activate
pip install -r requirements.txt
