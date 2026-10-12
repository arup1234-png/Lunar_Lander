"""FastAPI inference server for the LunarLander DQN project."""
import base64
import csv
import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_FILE = ROOT / "logs" / "predictions.csv"
CURVES_FILE = ROOT / "data" / "curves.json"
CKPT_FILES = {"untrained": "dqn_untrained.pth", "mid": "dqn_mid.pth", "final": "dqn_final.pth"}


class DQN(nn.Module):  # must match the architecture in Arup's notebook
    def __init__(self, state_size=8, action_size=4):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Scenario(BaseModel):
    checkpoint: Literal["untrained", "mid", "final"] = "final"
    gravity: float = Field(-10.0, ge=-11.5, le=-1.0)
    wind: float = Field(0.0, ge=0.0, le=20.0)
    seed: Optional[int] = None


class EvalRequest(Scenario):
    episodes: int = Field(20, ge=1, le=50)


app = FastAPI(title="LunarLander DQN API")
LOCK = threading.Lock()  # pygame rendering is not thread-safe
_models = {}


def get_model(stage):
    if stage not in _models:
        path = CKPT_DIR / CKPT_FILES[stage]
        if not path.exists():
            raise HTTPException(404, f"Checkpoint missing: checkpoints/{path.name}")
        net = DQN()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own trusted files
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        _models[stage] = net
    return _models[stage]


def run_episode(model, gravity, wind, seed, render=False):
    env = gym.make(
        "LunarLander-v3", continuous=False, gravity=gravity,
        enable_wind=wind > 0, wind_power=wind, turbulence_power=1.5,
        render_mode="rgb_array" if render else None,
    )
    state, _ = env.reset(seed=seed)
    total, steps, frames = 0.0, 0, []
    while True:
        if render and steps % 2 == 0:
            frames.append(env.render())
        with torch.no_grad():
            action = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break
    if render:
        frames.append(env.render())
    env.close()
    # The env overwrites the last reward with -100 (crash) or +100 (came to rest).
    if terminated and reward <= -100:
        outcome = "crashed"
    elif terminated and reward >= 100:
        outcome = "landed"
    else:
        outcome = "timeout"
    return total, steps, outcome, frames


def make_gif(frames):
    skip = max(1, len(frames) // 250)
    imgs = [Image.fromarray(f).resize((360, 240)) for f in frames[::skip]]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=40 * skip, loop=0)
    return base64.b64encode(buf.getvalue()).decode()


def log_run(**row):
    LOG_FILE.parent.mkdir(exist_ok=True)
    row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


@app.get("/api/checkpoints")
def checkpoints():
    return {k: (CKPT_DIR / v).exists() for k, v in CKPT_FILES.items()}


@app.post("/api/play")
def play(req: Scenario):
    model = get_model(req.checkpoint)
    seed = req.seed if req.seed is not None else int(np.random.randint(0, 1_000_000))
    with LOCK:
        reward, steps, outcome, frames = run_episode(model, req.gravity, req.wind, seed, render=True)
    log_run(endpoint="play", checkpoint=req.checkpoint, gravity=req.gravity, wind=req.wind,
            seed=seed, episodes=1, mean_reward=round(reward, 2), result=outcome)
    return {"reward": round(reward, 2), "steps": steps, "outcome": outcome,
            "seed": seed, "solved": reward >= 200, "gif": make_gif(frames)}


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    model = get_model(req.checkpoint)
    base = req.seed if req.seed is not None else 10_000
    with LOCK:
        runs = [run_episode(model, req.gravity, req.wind, base + i) for i in range(req.episodes)]
    rewards = np.array([r[0] for r in runs])
    landed = sum(r[2] == "landed" for r in runs)
    solved = int((rewards >= 200).sum())
    log_run(endpoint="evaluate", checkpoint=req.checkpoint, gravity=req.gravity, wind=req.wind,
            seed=base, episodes=req.episodes, mean_reward=round(float(rewards.mean()), 2),
            result=f"{solved}/{req.episodes} >=200")
    return {"episodes": req.episodes, "mean": round(float(rewards.mean()), 2),
            "std": round(float(rewards.std()), 2), "solved": solved, "landed": landed}


@app.get("/api/curves")
def curves():
    if not CURVES_FILE.exists():
        raise HTTPException(404, "data/curves.json missing - export it from the notebook (see README)")
    return json.loads(CURVES_FILE.read_text())


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
