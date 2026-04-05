from fastapi import FastAPI
from pydantic import BaseModel
from customer_support_env.env.environment import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    action: dict


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(req.task_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    from customer_support_env.env.models import Action
    action = Action(**req.action)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.score,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    return env.list_tasks()