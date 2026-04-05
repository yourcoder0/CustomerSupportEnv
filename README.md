---
title: CustomerSupportEnv
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---


# 🎧 CustomerSupportEnv — Autonomous Customer Support Ops

## 🧠 What Makes This Environment Different

CustomerSupportEnv goes beyond scripted workflows and evaluates **decision-making under real business constraints**.

Unlike toy environments, agents must:
- Balance **policy compliance vs. customer satisfaction**
- Handle **regulatory scenarios (GDPR, chargebacks)**
- Make **irreversible decisions** (resolve vs escalate)
- Operate under **SLA pressure and incomplete information**

This environment is designed to test whether an agent is **deployable in real-world support systems — not just correct in isolation**.

> **OpenEnv v1 compliant** · **Hugging Face Spaces ready** · **Deterministic graders** · **3 tasks: easy → hard**

An AI agent environment that simulates a real customer support operations queue. Agents must triage incoming tickets, apply business policies, draft customer-facing replies, and resolve or escalate cases — all while respecting SLA windows, customer tier constraints, and compliance rules (GDPR, chargeback policy).

---

## 🎯 Why This Environment?

Customer support is a **$350B industry** where AI agents are already being deployed. This environment trains and evaluates agents on the nuanced decisions real support reps make daily:

- When do you deny a refund vs. offer a credit?
- How do you respond to a GDPR breach notice in under 2 steps?
- What's the difference between a `billing` escalation and an `engineering` one?

These are **policy-constrained, consequence-laden decisions** — not toy tasks. A weak agent that promises refunds to ineligible customers or escalates a breach to billing instead of engineering causes real business damage.

---

## 🚀 Quick Start

### Local (Python)
```bash
git clone https://huggingface.co/spaces/<your-org>/customer-support-env
cd customer-support-env
pip install -r requirements.txt

# Start the API server
python -m uvicorn customer_support_env.server:app --host 0.0.0.0 --port 7860

# Run baseline inference (requires API key)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

### Docker
```bash
docker build -t customer-support-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=sk-... \
  customer-support-env
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Reset episode `{"task_id": "..."}` |
| POST | `/step` | Take action `{"action": {...}}` |
| GET | `/state` | Full serialised state |
| GET | `/tasks` | List all task IDs |
| GET | `/action_space` | Action schema |
| GET | `/obs_space` | Observation schema |

---

## 📦 Project Structure

```
customer_support_env/
├── env/
│   ├── environment.py       # Main CustomerSupportEnv class
│   ├── models.py            # Typed Pydantic schemas (Observation, Action, Reward, State)
│   └── reward_function.py   # Shaped reward with 7 components + penalties
├── tasks/
│   └── task_definitions.py  # TASK_1 (easy), TASK_2 (medium), TASK_3 (hard)
├── graders/
│   └── graders.py           # Deterministic episode graders per task
├── tests/
│   └── test_environment.py  # 30+ unit + integration tests
├── server.py                # FastAPI HTTP server
├── inference.py             # ← Baseline agent script (strict log format)
├── openenv.yaml             # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔭 Observation Space

```python
class Observation(BaseModel):
    task_id:              str
    step:                 int
    max_steps:            int
    ticket:               Ticket          # Full ticket with history, tags, status
    available_macros:     List[str]       # Canned-response macro IDs
    available_categories: List[str]       # Valid classification categories
    knowledge_base:       Dict[str, str]  # Policy lookup (billing, escalation, SLA)
    last_action_result:   Optional[str]   # Human-readable feedback on last action
    done:                 bool
    info:                 Dict[str, Any]  # cumulative_reward, steps_remaining
```

**Ticket fields:**
- `ticket_id`, `subject`, `body`, `customer_tier` (free/pro/enterprise)
- `history`: `List[Message]` with role (customer/agent/system) and content
- `status`, `priority`, `assigned_category`, `tags`
- `draft_reply`, `resolution_note`, `internal_summary`

---

## ⚡ Action Space

```python
class Action(BaseModel):
    action_type: ActionType  # One of 8 types below

    # CLASSIFY
    category:   Optional[str]       # e.g. "billing/invoice", "security/breach"
    priority:   Optional[Priority]  # low | medium | high | critical

    # DRAFT_REPLY / REQUEST_INFO
    reply_text: Optional[str]

    # ESCALATE
    escalation_tier:   Optional[str]  # tier2 | billing | engineering | legal
    escalation_reason: Optional[str]

    # RESOLVE
    resolution_note: Optional[str]
    resolution_code: Optional[str]   # solved | workaround | wont_fix | duplicate | spam

    # APPLY_MACRO
    macro_id: Optional[str]

    # TAG
    tags: Optional[List[str]]

    # SUMMARIZE
    summary_text: Optional[str]
```

**Action types:** `classify`, `draft_reply`, `escalate`, `resolve`, `request_info`, `apply_macro`, `tag`, `summarize`

**Terminal actions:** `resolve`, `escalate` — end the episode immediately.

---

## 🏆 Reward Function

All rewards are deterministic (no LLM calls). Scores always in `[0.0, 1.0]`.

| Component | Max Weight | Description |
|-----------|-----------|-------------|
| `classification_accuracy` | 0.20 | Exact match = 1.0; same namespace = 0.5; wrong = 0.0 |
| `priority_accuracy` | 0.10 | Adjacent level = 0.5; enterprise under-prioritisation penalised harder |
| `reply_quality` | 0.25 | Rubric: required topics covered, forbidden phrases absent, length, greeting |
| `resolution_appropriateness` | 0.20 | Correct code + note = 1.0; partial credit for close attempts |
| `sla_compliance` | 0.10 | Full credit ≤ SLA budget steps; degrades linearly to 2× budget |
| `policy_compliance` | 0.10 | Tier rules, escalation path rules, classify-before-resolve |
| `efficiency_bonus` | 0.05 | Bonus for resolving early in episode |

**Penalties:**

| Violation | Penalty |
|-----------|---------|
| Wrong escalation tier | −0.10 |
| Missing required field | −0.05 |
| Contradictory actions (resolve → escalate) | −0.15 |
| Spam / empty reply | −0.20 × 0.5 |

---

## 📋 Tasks

### Task 1 — Easy: Invoice Lookup (max 3 steps)

**Scenario:** A Pro-tier customer can't find their invoice download button.

**Objectives:**
1. Classify as `billing/invoice`, priority `low`
2. Draft a reply mentioning the download location
3. Resolve with code `solved`

**Pass threshold:** 0.60 | **Baseline expected:** ~0.75

```python
# Optimal action sequence
env.reset("task_easy_classify_reply")
env.step(Action(action_type="classify", category="billing/invoice", priority="low"))
env.step(Action(action_type="draft_reply", reply_text="Hello! You can find your invoice under Billing > Invoice History..."))
env.step(Action(action_type="resolve", resolution_code="solved", resolution_note="Directed to billing portal"))
```

---

### Task 2 — Medium: Policy-Constrained Refund Denial (max 5 steps)

**Scenario:** A Free-tier customer threatens chargeback and demands a refund 45 days after purchase. Policy: Free tier = ineligible; >30 days = ineligible.

**Objectives:**
1. Classify as `billing/refund`, priority `high`
2. Reply must NOT promise a refund (policy gate, 0.25 weight)
3. Reply SHOULD offer account credit as alternative
4. Resolve with code `wont_fix`

**Pass threshold:** 0.55 | **Baseline expected:** ~0.60

**Key failure mode:** Agents that promise "refund approved" score 0 on the policy component (0.25 weight) — a significant penalty.

---

### Task 3 — Hard: GDPR Breach Escalation (max 6 steps)

**Scenario:** An Enterprise customer ($2M ARR) reports suspicious API key usage at 03:00 UTC, invokes GDPR Article 33 (72h notification window).

**Objectives:**
1. Classify as `security/breach`, priority `critical`
2. Escalate **only** to `engineering` tier (billing = wrong path)
3. Reply must acknowledge GDPR / 72-hour duty
4. Produce internal summary for legal (≥20 words)
5. Apply tags: `gdpr`, `security`, `breach`, `enterprise`
6. First response or escalation must happen within step 2 (critical SLA)

**Pass threshold:** 0.50 | **Baseline expected:** ~0.45

**Key failure modes:**
- Escalating to `billing` instead of `engineering` = −0.10 + 0 on escalation component
- Skipping GDPR acknowledgement = −0.15 on reply
- No internal summary = missing 0.10 weight
- Late first action (step > 2) = 0 SLA score

---

## 📊 Baseline Scores (gpt-4o-mini)

| Task | Difficulty | Baseline Score | Passed |
|------|-----------|---------------|--------|
| task_easy_classify_reply | Easy | ~0.75 | ✅ |
| task_medium_refund_policy | Medium | ~0.60 | ✅ |
| task_hard_security_escalation | Hard | ~0.45 | ⚠️ Stress-test scenario |
> The hard task intentionally simulates high-risk regulatory incidents where baseline models struggle — highlighting the need for structured reasoning and escalation policies.
| **Average** | | **~0.60** | **2/3** |

---

## 🔬 Edge Cases & Failure Scenarios

| Scenario | Behaviour |
|----------|-----------|
| Empty reply text | `reply_quality = 0.0`, no penalty beyond missed score |
| Reply > 400 words | Length penalty degrades reply_quality proportionally |
| Wrong macro ID | No-op, feedback logged, no crash |
| Resolve before classify | `must_classify_before_resolve` policy → −0.05 penalty |
| Escalate after partial resolve path | Contradiction detected → −0.15 |
| Missing escalation reason | −0.05 missing field penalty |
| Step limit exceeded | Forced RESOLVE action, episode terminates, grader runs |
| Unknown task_id | `ValueError` raised with available task list |
| Step after done | `RuntimeError("Episode is done")` |
| `reset()` on running episode | Safe — clears all state, fresh ticket |

---

## 🧪 Running Tests

```bash
pytest tests/ -v

# Expected output:
# tests/test_environment.py::TestSmoke::test_env_creates PASSED
# tests/test_environment.py::TestSmoke::test_three_tasks_registered PASSED
# ... (30+ tests)
```

---

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / OpenAI API key | (required) |
| `PORT` | Server port | `7860` |

---
## 🚀 Key Innovations

- **Multi-objective reward shaping** (7 components: policy, SLA, quality, efficiency)
- **Deterministic grading pipeline** for reproducibility
- **Policy-gated decision making** (refund rules, escalation paths)
- **Regulatory awareness simulation** (GDPR breach handling)
- **Tier-aware behavior** (free vs enterprise customers)

## 📜 Inference Log Format

The `inference.py` script emits structured JSON logs per OpenEnv spec:

```jsonc
// Episode start
{"event": "[START]", "task_id": "...", "model": "...", "api_base": "...", "timestamp": "..."}

// Each step
{"event": "[STEP]", "task_id": "...", "step": 1, "action": {...}, "reward": 0.2300,
 "done": false, "info": {"action_feedback": "...", "episode_complete": false, "final_grade": null}}

// Episode end
{"event": "[END]", "task_id": "...", "total_steps": 3, "final_score": 0.7800,
 "cumulative_reward": 1.1200, "passed": true, "violations": [], "timestamp": "..."}

// All-tasks summary
{"event": "[SUMMARY]", "tasks_run": 3, "tasks_passed": 2, "average_score": 0.6033,
 "results": [...], "model": "gpt-4o-mini"}
```

---

## 🏗️ Design Decisions

**Why deterministic graders?** LLM-based graders introduce variance that makes reproducible scoring impossible. All graders use keyword matching, rubric checks, and structural analysis — no API calls.

**Why partial credit?** Binary pass/fail provides no training signal for intermediate steps. The 7-component reward lets an agent learn that "correct classification + bad reply" scores ~0.30, while "correct classification + good reply + wrong resolution" scores ~0.65.

**Why these three tasks?** They represent the three most common failure modes in production support agents: (1) insufficient KB usage, (2) policy non-compliance, (3) regulatory awareness. Agents that pass all three are genuinely deployable.

---
## 🤖 What a High-Performing Agent Does

A strong agent in this environment:

- Adapts responses based on **customer tier and sentiment**
- Avoids policy violations while maintaining **customer trust**
- Escalates **only when necessary and to the correct team**
- Balances **speed (SLA)** with **decision quality**
- Handles ambiguous or high-risk scenarios with **structured reasoning**

This moves beyond rule-following into **context-aware decision-making**.

## 🏁 Why This Matters

Most AI agents fail not because they lack knowledge, but because they make **poor decisions under constraints**.

CustomerSupportEnv evaluates exactly that.

It bridges the gap between:
- ✔ Academic benchmarks  
- ✔ Real-world deployment readiness  

This makes it a strong foundation for training **production-grade autonomous agents**.

## 📄 License

Apache 2.0
