# ./orchestrator/server.py
from fastapi import FastAPI
from run_daily import main  # <-- must exist in run_daily.py

app = FastAPI()

@app.get("/")
def health():
    return {"ok": True}

@app.post("/run")
def run():
    main()
    return {"status": "ok"}