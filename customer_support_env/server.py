from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
from .env.environment import CustomerSupportEnv   # ← relative import
from .env.models import Action                    # ← relative import

app = FastAPI(
    title="CustomerSupportEnv",
    description="Autonomous Customer Support Ops — OpenEnv v1",
    version="1.0.0",
)

env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None   # ← Optional


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def health():
    return {"status": "ok", "env_id": env.ENV_ID, "version": env.VERSION}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = env.reset(req.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(**req.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.score,
            "done":        done,
            "info":        info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": env.list_tasks()}   # ← dict, not list


@app.get("/action_space")
def action_space():
    return env.action_space_description()


@app.get("/obs_space")
def obs_space():
    return env.observation_space_description()