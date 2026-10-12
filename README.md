# Lunar Lander DQN - model + web app (NLDL project P3)

Team: Arup (model lead), Aditya (application lead)

A DQN agent trained with our own PyTorch code (replay buffer, target network, epsilon-greedy,
Huber loss, Adam) plus a Double DQN comparison over 3 seeds. A FastAPI service serves the trained
checkpoints and a web UI lets you replay landings, change gravity/wind, and view reward curves.

## Repository layout
- `notebooks/DQN_LunarLander.ipynb` - training and evaluation (Arup)
- `app/main.py` - API: `/api/play`, `/api/evaluate`, `/api/curves`, `/api/checkpoints`
- `app/static/index.html` - web UI
- `checkpoints/` - `dqn_untrained.pth`, `dqn_mid.pth`, `dqn_final.pth`
- `data/curves.json` - 3-seed reward curves for the chart
- `logs/predictions.csv` - every run served by the API is logged here

## Setup files from the notebook
1. Download from Colab into `checkpoints/`: `dqn_untrained.pth`, `dqn_mid.pth` and the final model as `dqn_final.pth`.
2. Run this cell at the end of the notebook and put the file in `data/`:
```python
import json, os
os.makedirs("data", exist_ok=True)
json.dump({"DQN": {"mean": dqn_mean.tolist(), "std": dqn_std.tolist()},
           "DDQN": {"mean": ddqn_mean.tolist(), "std": ddqn_std.tolist()}},
          open("data/curves.json", "w"))
```

## Run locally
```bash
pip install swig && pip install -r requirements.txt
uvicorn app.main:app --reload      # open http://127.0.0.1:8000
```

## Deploy (Hugging Face Spaces)
Create a new Space with SDK **Docker**, add `sdk: docker` and `app_port: 7860` in the README front matter,
and push this repo (the included Dockerfile serves on port 7860).

## Results (fill in from final evaluation)
| Agent | Mean reward over 100 eval episodes |
|---|---|
| DQN (3-seed mean) | 208.0 |
| Double DQN (3-seed mean) | 145.3 |
