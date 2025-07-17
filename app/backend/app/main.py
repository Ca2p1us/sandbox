# backend/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import csv
import json
from datetime import datetime
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 音パラメータの読み込み
with open("backend/parameters.json", "r") as f:
    PARAMETERS = json.load(f)

@app.post("/submit")
async def submit_rating(request: Request):
    data = await request.json()
    user_id = data["user_id"]
    sound_id = data["sound_id"]
    rating = data["rating"]
    timestamp = datetime.now().isoformat()

    params = PARAMETERS.get(sound_id, {})

    os.makedirs("backend/logs", exist_ok=True)
    with open("backend/logs/results.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, sound_id, rating, json.dumps(params)])

    return {"status": "ok"}
