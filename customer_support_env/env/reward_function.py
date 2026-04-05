"""
Reward Function — Autonomous Customer Support Ops
=================================================

Weight Table
------------
Component                   Max Weight  Description
--------------------------  ----------  ----------------------------------------
classification_accuracy     0.20        Correct issue category assignment
priority_accuracy           0.10        Correct urgency level (SLA-aware)
reply_quality               0.25        BLEU-free semantic rubric on draft reply
resolution_appropriateness  0.20        Right resolution path chosen
sla_compliance              0.10        Action taken within SLA window
policy_compliance           0.10        No forbidden actions for customer tier
efficiency_bonus            0.05        Resolved in fewer steps than max

Penalties
---------
Wrong escalation path      -0.10  (escalated to wrong tier)
Missing required fields    -0.05  per field
Contradictory actions      -0.15  (resolve then escalate same ticket)
Spam/junk reply            -0.20  (reply unrelated to ticket)

Total possible before bonus: 0.95
Efficiency can push to 1.0
All scores clamped to [0.0, 1.0].
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
from customer_support_env.env.models import (
    Action, ActionType, Priority, RewardBreakdown, Reward, Ticket, TicketStatus
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_WEIGHT        = 0.20
PRIORITY_WEIGHT        = 0.10
REPLY_WEIGHT           = 0.25
RESOLUTION_WEIGHT      = 0.20
SLA_WEIGHT             = 0.10
POLICY_WEIGHT          = 0.10
EFFICIENCY_WEIGHT      = 0.05

WRONG_ESCALATION_PEN   = -0.10
MISSING_FIELD_PEN      = -0.05
CONTRADICTION_PEN      = -0.15
SPAM_REPLY_PEN         = -0.20

# SLA budgets (in steps) per priority
SLA_STEPS: Dict[str, int] = {
    "critical": 2,
    "high":     4,
    "medium":   6,
    "low":      10,
}


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def score_classification(action: Action, expected_category: str) -> float:
    """
    Exact match = 1.0. Same top-level family = 0.5. Wrong = 0.0.
    Categories are namespaced: 'billing/refund', 'billing/charge', 'technical/login'.
    """
    if not action.category:
        return 0.0
    agent_cat = action.category.strip().lower()
    exp_cat = expected_category.strip().lower()
    if agent_cat == exp_cat:
        return 1.0
    # Partial: same top-level namespace
    agent_top = agent_cat.split("/")[0]
    exp_top = exp_cat.split("/")[0]
    if agent_top == exp_top:
        return 0.5
    return 0.0


def score_priority(action: Action, expected_priority: str, customer_tier: str) -> float:
    """
    Correct = 1.0.  Adjacent level = 0.5.  Wrong by 2+ = 0.0.
    Enterprise tier: under-prioritizing is penalised harder (×0.5 partial credit).
    """
    if not action.priority:
        return 0.0
    levels = ["low", "medium", "high", "critical"]
    try:
        agent_idx = levels.index(action.priority.value)
        exp_idx   = levels.index(expected_priority)
    except ValueError:
        return 0.0

    diff = abs(agent_idx - exp_idx)
    if diff == 0:
        return 1.0
    if diff == 1:
        partial = 0.5
        # Under-prioritising enterprise = harsher
        if customer_tier == "enterprise" and agent_idx < exp_idx:
            partial = 0.25
        return partial
    return 0.0


def score_reply_quality(reply_text: Optional[str], rubric: Dict) -> float:
    """
    Deterministic rubric-based scoring (no LLM, no randomness).
    rubric keys: required_topics (list), forbidden_phrases (list),
                 min_length (int), max_length (int), greeting_required (bool).
    Returns [0.0, 1.0].
    """
    if not reply_text:
        return 0.0

    text_lower = reply_text.lower()
    score = 0.0
    total_checks = 0

    # 1. Length check (25% of reply score)
    length = len(reply_text.split())
    min_len = rubric.get("min_length", 20)
    max_len = rubric.get("max_length", 300)
    total_checks += 1
    if min_len <= length <= max_len:
        score += 1.0
    elif length < min_len:
        score += max(0.0, length / min_len)
    else:
        score += max(0.0, 1.0 - (length - max_len) / max_len)

    # 2. Required topics (50% of reply score)
    required_topics: list = rubric.get("required_topics", [])
    if required_topics:
        hits = sum(1 for t in required_topics if t.lower() in text_lower)
        topic_score = hits / len(required_topics)
        score += topic_score * 2  # weight x2
        total_checks += 2
    else:
        total_checks += 2
        score += 2.0  # no requirements = full marks

    # 3. Forbidden phrases (25%)
    forbidden: list = rubric.get("forbidden_phrases", [])
    total_checks += 1
    if not any(f.lower() in text_lower for f in forbidden):
        score += 1.0

    # 4. Greeting check
    if rubric.get("greeting_required", True):
        total_checks += 1
        greetings = ["hello", "hi ", "dear", "greetings", "thank you for reaching"]
        if any(g in text_lower for g in greetings):
            score += 1.0

    return min(1.0, score / max(total_checks, 1))


def score_resolution(action: Action, expected_resolution: str, ticket: Ticket) -> float:
    """
    Correct resolution code = 1.0.
    Resolved with note = 0.8 if no code expected.
    Escalated instead of resolved = 0.3 (some credit for not dropping).
    Missing resolution note = 0.4.
    """
    if action.action_type == ActionType.RESOLVE:
        if action.resolution_code and action.resolution_code == expected_resolution:
            note_bonus = 0.2 if action.resolution_note and len(action.resolution_note) > 20 else 0.0
            return min(1.0, 0.8 + note_bonus)
        if action.resolution_note and len(action.resolution_note) > 20:
            return 0.6
        return 0.4
    if action.action_type == ActionType.ESCALATE:
        # Partial credit: at least didn't close without action
        return 0.3
    return 0.0


def score_sla(step: int, priority: Optional[str], resolved: bool) -> float:
    """
    Full marks if resolved within SLA steps budget.
    Degrades linearly to 0 at 2× budget.
    """
    if not priority or not resolved:
        return 0.0
    budget = SLA_STEPS.get(priority, 6)
    if step <= budget:
        return 1.0
    if step <= budget * 2:
        return max(0.0, 1.0 - (step - budget) / budget)
    return 0.0


def score_policy(action: Action, ticket: Ticket, policy_rules: Dict) -> Tuple[float, float]:
    """
    Returns (policy_score, penalty).
    policy_rules: {
        "no_refund_for_free": bool,
        "escalation_tiers_allowed": list,
        "must_classify_before_resolve": bool,
    }
    """
    penalty = 0.0
    policy_score = 1.0  # start full, deduct violations

    # Rule: free tier cannot get direct refund promise
    if policy_rules.get("no_refund_for_free") and ticket.customer_tier == "free":
        if action.action_type == ActionType.DRAFT_REPLY:
            if action.reply_text and "refund" in action.reply_text.lower():
                policy_score -= 0.5
                penalty += 0.05

    # Rule: escalation tier must be allowed
    allowed_tiers = policy_rules.get("escalation_tiers_allowed", [])
    if action.action_type == ActionType.ESCALATE and allowed_tiers:
        if action.escalation_tier not in allowed_tiers:
            policy_score -= 1.0
            penalty += WRONG_ESCALATION_PEN

    # Rule: must classify before resolve
    if policy_rules.get("must_classify_before_resolve"):
        if action.action_type == ActionType.RESOLVE and not ticket.assigned_category:
            policy_score -= 0.3
            penalty += MISSING_FIELD_PEN

    return max(0.0, policy_score), penalty


def score_efficiency(step: int, max_steps: int) -> float:
    """Bonus for resolving quickly. 0 if at max_steps, 1.0 if in first 20%."""
    if max_steps <= 0:
        return 0.0
    ratio = step / max_steps
    if ratio <= 0.2:
        return 1.0
    if ratio >= 1.0:
        return 0.0
    return max(0.0, 1.0 - ratio)


# ---------------------------------------------------------------------------
# Main reward aggregator
# ---------------------------------------------------------------------------

def compute_reward(
    action: Action,
    ticket: Ticket,
    step: int,
    max_steps: int,
    grading_spec: Dict,
    episode_done: bool,
    action_history: list,
) -> Reward:
    """
    grading_spec keys:
        expected_category (str)
        expected_priority (str)
        expected_resolution (str)
        reply_rubric (dict)
        policy_rules (dict)
        is_terminal_action (bool)  — True when this action should close episode
    """
    bd = RewardBreakdown()
    cumulative_penalty = 0.0

    # --- Classification ---
    if action.action_type == ActionType.CLASSIFY:
        raw = score_classification(action, grading_spec.get("expected_category", ""))
        bd.classification_accuracy = raw * CATEGORY_WEIGHT
        raw_p = score_priority(action, grading_spec.get("expected_priority", "medium"),
                               ticket.customer_tier)
        bd.priority_accuracy = raw_p * PRIORITY_WEIGHT

    # --- Reply quality ---
    if action.action_type in (ActionType.DRAFT_REPLY, ActionType.REQUEST_INFO):
        rubric = grading_spec.get("reply_rubric", {})
        reply = action.reply_text or ""
        # Spam detection: reply shorter than 5 words and unrelated
        if len(reply.split()) < 5:
            cumulative_penalty += SPAM_REPLY_PEN * 0.5
            bd.reply_quality = 0.0
        else:
            raw_r = score_reply_quality(reply, rubric)
            bd.reply_quality = raw_r * REPLY_WEIGHT

    # --- Resolution ---
    if action.action_type in (ActionType.RESOLVE, ActionType.ESCALATE):
        raw_res = score_resolution(action, grading_spec.get("expected_resolution", "solved"), ticket)
        bd.resolution_appropriateness = raw_res * RESOLUTION_WEIGHT
        resolved = action.action_type == ActionType.RESOLVE
        priority_str = (ticket.priority.value if ticket.priority else
                        grading_spec.get("expected_priority", "medium"))
        bd.sla_compliance = score_sla(step, priority_str, resolved) * SLA_WEIGHT

    # --- Policy ---
    policy_score, pen = score_policy(action, ticket, grading_spec.get("policy_rules", {}))
    bd.policy_compliance = policy_score * POLICY_WEIGHT
    cumulative_penalty += pen

    # --- Efficiency (only at episode end) ---
    if episode_done:
        bd.efficiency_bonus = score_efficiency(step, max_steps) * EFFICIENCY_WEIGHT

    # --- Contradiction check ---
    prev_types = [a.get("action_type") for a in action_history]
    if "resolve" in prev_types and action.action_type == ActionType.ESCALATE:
        cumulative_penalty += CONTRADICTION_PEN
    if "escalate" in prev_types and action.action_type == ActionType.RESOLVE:
        cumulative_penalty += CONTRADICTION_PEN * 0.5

    # --- Missing required fields ---
    if action.action_type == ActionType.DRAFT_REPLY and not action.reply_text:
        cumulative_penalty += MISSING_FIELD_PEN
    if action.action_type == ActionType.ESCALATE and not action.escalation_reason:
        cumulative_penalty += MISSING_FIELD_PEN

    bd.penalty = cumulative_penalty

    raw_total = (
        bd.classification_accuracy
        + bd.priority_accuracy
        + bd.reply_quality
        + bd.resolution_appropriateness
        + bd.sla_compliance
        + bd.policy_compliance
        + bd.efficiency_bonus
        + bd.penalty
    )

    final_score = max(0.0, min(1.0, raw_total))

    rationale_parts = [
        f"classify={bd.classification_accuracy:.3f}",
        f"priority={bd.priority_accuracy:.3f}",
        f"reply={bd.reply_quality:.3f}",
        f"resolution={bd.resolution_appropriateness:.3f}",
        f"sla={bd.sla_compliance:.3f}",
        f"policy={bd.policy_compliance:.3f}",
        f"efficiency={bd.efficiency_bonus:.3f}",
        f"penalty={bd.penalty:.3f}",
        f"total={final_score:.3f}",
    ]

    return Reward(
        score=final_score,
        breakdown=bd,
        rationale=" | ".join(rationale_parts),
    )
