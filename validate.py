#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script for CustomerSupportEnv
=====================================================================
Run before submitting to verify OpenEnv compliance.

Usage:
    python validate.py

Exit code 0 = all checks passed. Exit code 1 = failures found.
"""

from __future__ import annotations
import importlib
import inspect
import json
import os
import sys
import yaml
from typing import List, Tuple

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: List[Tuple[str, str, str]] = []  # (check_name, status, detail)


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = PASS if condition else (WARN if warn_only else FAIL)
    results.append((name, status, detail))
    symbol = "✅" if condition else ("⚠️ " if warn_only else "❌")
    print(f"  {symbol} {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. File structure checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] File Structure")
required_files = [
    "customer_support_env/env/environment.py",
    "customer_support_env/env/models.py",
    "customer_support_env/env/reward_function.py",
    "customer_support_env/tasks/task_definitions.py",
    "customer_support_env/graders/graders.py",
    "customer_support_env/server.py",
    "inference.py",
    "openenv.yaml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))

# ─────────────────────────────────────────────────────────────────────────────
# 2. openenv.yaml schema
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] openenv.yaml Compliance")
try:
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)
    check("openenv.yaml parseable", True)
    check("env_id defined",    bool(spec.get("env_id")),    spec.get("env_id", "MISSING"))
    check("version defined",   bool(spec.get("version")),   spec.get("version", "MISSING"))
    check("tasks section present", "tasks" in spec and len(spec["tasks"]) >= 3,
          f"{len(spec.get('tasks', []))} tasks")
    check("reward section present", "reward" in spec)
    check("observation_space present", "observation_space" in spec)
    check("action_space present", "action_space" in spec)
    task_ids = [t["task_id"] for t in spec.get("tasks", [])]
    check("3+ tasks in yaml", len(task_ids) >= 3, str(task_ids))
    difficulties = [t.get("difficulty") for t in spec.get("tasks", [])]
    check("easy/medium/hard difficulties", set(difficulties) == {"easy", "medium", "hard"})
    for task in spec.get("tasks", []):
        check(f"  task {task['task_id']} has pass_threshold",
              "pass_threshold" in task)
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Python package imports
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Python Package Imports")
try:
    sys.path.insert(0, os.getcwd())
    from customer_support_env.env.environment import CustomerSupportEnv
    from customer_support_env.env.models import Action, ActionType, Observation, Reward, EpisodeState
    from customer_support_env.tasks.task_definitions import list_tasks, get_task
    from customer_support_env.graders.graders import grade_episode
    check("All imports succeed", True)
except ImportError as e:
    check("All imports succeed", False, str(e))
    print("  Cannot continue — fix imports first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. OpenEnv interface compliance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] OpenEnv Interface")
env = CustomerSupportEnv()

check("CustomerSupportEnv.reset() exists",   hasattr(env, "reset"))
check("CustomerSupportEnv.step() exists",    hasattr(env, "step"))
check("CustomerSupportEnv.state() exists",   hasattr(env, "state"))

# reset() signature
sig = inspect.signature(env.reset)
check("reset() accepts task_id param", "task_id" in sig.parameters)

# reset() returns Observation
try:
    obs = env.reset()
    check("reset() returns Observation", isinstance(obs, Observation),
          type(obs).__name__)
except Exception as e:
    check("reset() returns Observation", False, str(e))

# step() returns (obs, reward, done, info)
try:
    obs = env.reset()
    result = env.step(Action(action_type=ActionType.RESOLVE,
                             resolution_code="solved", resolution_note="test"))
    check("step() returns 4-tuple", len(result) == 4)
    obs2, reward, done, info = result
    check("step() obs is Observation", isinstance(obs2, Observation))
    check("step() reward is Reward",   isinstance(reward, Reward))
    check("step() done is bool",       isinstance(done, bool))
    check("step() info is dict",       isinstance(info, dict))
    check("reward.score in [0,1]",     0.0 <= reward.score <= 1.0, str(reward.score))
except Exception as e:
    check("step() interface", False, str(e))

# state()
try:
    env.reset()
    st = env.state()
    check("state() returns EpisodeState", isinstance(st, EpisodeState))
    check("state() serialisable",
          bool(json.dumps(st.model_dump())))
except Exception as e:
    check("state() interface", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Grader checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Grader Checks")
for task_id in list_tasks():
    try:
        task_def = get_task(task_id)
        ticket   = task_def["ticket"].model_copy(deep=True)

        # Empty history → low but valid score
        result = grade_episode(task_id, [], ticket, 0, task_def["max_steps"])
        check(f"  {task_id}: grader runs", True, f"score={result.score:.3f}")
        check(f"  {task_id}: score in [0,1]", 0.0 <= result.score <= 1.0,
              str(result.score))
        check(f"  {task_id}: score not always 1.0", result.score < 1.0,
              "grader would be trivially maxed")
        check(f"  {task_id}: score not always 0.0", result.score >= 0.0)

        # Determinism check
        result2 = grade_episode(task_id, [], ticket, 0, task_def["max_steps"])
        check(f"  {task_id}: deterministic", result.score == result2.score)
    except Exception as e:
        check(f"  {task_id}: grader runs", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 6. inference.py checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] inference.py Checks")
try:
    with open("inference.py") as fh:
        inf_src = fh.read()
    check("inference.py exists and readable", True)
    check("[START] log format present", '"[START]"' in inf_src)
    check("[STEP] log format present",  '"[STEP]"'  in inf_src)
    check("[END] log format present",   '"[END]"'   in inf_src)
    check("Uses OpenAI client",        "from openai import OpenAI" in inf_src)
    check("Reads API_BASE_URL env var", "API_BASE_URL" in inf_src)
    check("Reads MODEL_NAME env var",   "MODEL_NAME"   in inf_src)
    check("Reads HF_TOKEN env var",     "HF_TOKEN"     in inf_src)
    check("temperature=0 (deterministic)", "temperature=0" in inf_src)
except FileNotFoundError:
    check("inference.py exists and readable", False, "File not found")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total   = len(results)
passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
warned  = sum(1 for _, s, _ in results if s == WARN)

print(f"VALIDATION SUMMARY: {passed}/{total} passed | {failed} failed | {warned} warnings")
if failed == 0:
    print("🎉 ALL CHECKS PASSED — ready to submit!")
else:
    print(f"🚨 {failed} FAILURES — fix before submitting.")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  ❌ {name}: {detail}")

sys.exit(0 if failed == 0 else 1)
