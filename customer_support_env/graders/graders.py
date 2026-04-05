"""
Grader Functions — Autonomous Customer Support Ops
===================================================
Each grader takes the full episode trajectory (action history + final ticket state)
and returns a deterministic score in [0.0, 1.0] with a detailed breakdown.

All graders are DETERMINISTIC — no LLM calls, no randomness, no external I/O.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from customer_support_env.env.models import (
    Action, ActionType, Priority, Ticket, TicketStatus
)

from customer_support_env.env.reward_function import score_reply_quality


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    task_id: str
    score: float                          # Final [0.0, 1.0]
    passed: bool                          # score >= pass_threshold
    pass_threshold: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":          self.task_id,
            "score":            round(self.score, 4),
            "passed":           self.passed,
            "pass_threshold":   self.pass_threshold,
            "component_scores": {k: round(v, 4) for k, v in self.component_scores.items()},
            "violations":       self.violations,
            "notes":            self.notes,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_actions_of_type(history: List[Dict], atype: str) -> List[Dict]:
    return [a for a in history if a.get("action_type") == atype]


def _text_contains_all(text: str, topics: List[str]) -> List[str]:
    """Return list of missing topics."""
    tl = text.lower()
    return [t for t in topics if t.lower() not in tl]


def _text_contains_any(text: str, phrases: List[str]) -> List[str]:
    """Return list of forbidden phrases found."""
    tl = text.lower()
    return [p for p in phrases if p.lower() in tl]


# ---------------------------------------------------------------------------
# Task 1 Grader — Easy: Classify and Reply
# ---------------------------------------------------------------------------

def grade_task1(
    action_history: List[Dict],
    final_ticket: Ticket,
    total_steps: int,
    max_steps: int,
) -> GradeResult:
    """
    Scoring rubric:
      classification_correct   (0.25)
      priority_correct         (0.10)
      reply_present            (0.25)
      reply_topics_covered     (0.20)
      ticket_resolved          (0.15)
      step_efficiency          (0.05)
    """
    task_id = "task_easy_classify_reply"
    components: Dict[str, float] = {}
    violations: List[str] = []

    # --- Classification ---
    classify_acts = _find_actions_of_type(action_history, "classify")
    if classify_acts:
        cat = classify_acts[-1].get("category", "").lower()
        if cat == "billing/invoice":
            components["classification_correct"] = 0.25
        elif cat.startswith("billing"):
            components["classification_correct"] = 0.125
            violations.append(f"Partial classification: got '{cat}', expected 'billing/invoice'")
        else:
            components["classification_correct"] = 0.0
            violations.append(f"Wrong classification: '{cat}'")
    else:
        components["classification_correct"] = 0.0
        violations.append("No classify action taken")

    # --- Priority ---
    if classify_acts:
        pri = classify_acts[-1].get("priority", "")
        if pri == "low":
            components["priority_correct"] = 0.10
        elif pri == "medium":
            components["priority_correct"] = 0.05
            violations.append("Priority should be 'low' for invoice lookup")
        else:
            components["priority_correct"] = 0.0
            violations.append(f"Wrong priority: '{pri}'")
    else:
        components["priority_correct"] = 0.0

    # --- Reply quality ---
    reply_acts = _find_actions_of_type(action_history, "draft_reply")
    if reply_acts:
        reply_text = reply_acts[-1].get("reply_text", "")
        rubric = {
            "required_topics": ["billing", "invoice", "download"],
            "forbidden_phrases": ["i don't know", "cannot help", "call us"],
            "min_length": 25, "max_length": 200, "greeting_required": True,
        }
        raw_quality = score_reply_quality(reply_text, rubric)
        components["reply_present"]       = 0.25 if reply_text else 0.0
        components["reply_topics_covered"] = raw_quality * 0.20
        missing = _text_contains_all(reply_text, ["download"])
        if missing:
            violations.append(f"Reply missing required topics: {missing}")
    else:
        components["reply_present"]       = 0.0
        components["reply_topics_covered"] = 0.0
        violations.append("No draft_reply action taken")

    # --- Resolution ---
    resolve_acts = _find_actions_of_type(action_history, "resolve")
    if resolve_acts:
        code = resolve_acts[-1].get("resolution_code", "")
        if code == "solved":
            components["ticket_resolved"] = 0.15
        elif code:
            components["ticket_resolved"] = 0.08
            violations.append(f"Resolution code '{code}' suboptimal; expected 'solved'")
        else:
            components["ticket_resolved"] = 0.07
            violations.append("Resolved without a resolution_code")
    else:
        components["ticket_resolved"] = 0.0
        violations.append("Ticket not resolved")

    # --- Efficiency ---
    ratio = total_steps / max(max_steps, 1)
    components["step_efficiency"] = max(0.0, (1.0 - ratio)) * 0.05

    score = min(1.0, sum(components.values()))
    return GradeResult(
        task_id=task_id, score=score, passed=score >= 0.60,
        pass_threshold=0.60, component_scores=components,
        violations=violations,
        notes=f"Steps used: {total_steps}/{max_steps}",
    )


# ---------------------------------------------------------------------------
# Task 2 Grader — Medium: Policy-constrained refund denial
# ---------------------------------------------------------------------------

def grade_task2(
    action_history: List[Dict],
    final_ticket: Ticket,
    total_steps: int,
    max_steps: int,
) -> GradeResult:
    """
    Scoring rubric:
      classification_correct   (0.20)
      priority_high_or_crit    (0.10)
      no_refund_promised        (0.25)  ← critical policy gate
      credit_offered            (0.15)
      correct_resolution_code   (0.20)
      no_wrong_escalation       (0.05)
      step_efficiency           (0.05)
    """
    task_id = "task_medium_refund_policy"
    components: Dict[str, float] = {}
    violations: List[str] = []

    # --- Classification ---
    classify_acts = _find_actions_of_type(action_history, "classify")
    if classify_acts:
        cat = classify_acts[-1].get("category", "").lower()
        components["classification_correct"] = 0.20 if cat == "billing/refund" else (
            0.10 if cat.startswith("billing") else 0.0
        )
        if not cat.startswith("billing"):
            violations.append(f"Wrong classification: '{cat}'")
    else:
        components["classification_correct"] = 0.0
        violations.append("No classify action")

    # --- Priority ---
    if classify_acts:
        pri = classify_acts[-1].get("priority", "")
        components["priority_high_or_crit"] = 0.10 if pri in ("high", "critical") else (
            0.05 if pri == "medium" else 0.0
        )
        if pri not in ("high", "critical"):
            violations.append(f"Priority '{pri}' too low for chargeback-threat ticket")
    else:
        components["priority_high_or_crit"] = 0.0

    # --- Policy: no refund promised (critical) ---
    all_replies = _find_actions_of_type(action_history, "draft_reply")
    refund_promised = False
    credit_mentioned = False
    for r in all_replies:
        text = (r.get("reply_text") or "").lower()
        forbidden = _text_contains_any(text, ["refund approved", "we will refund", "processed your refund"])
        if forbidden:
            refund_promised = True
            violations.append(f"Policy violation: reply contains '{forbidden}'")
        if "credit" in text or "compensation" in text:
            credit_mentioned = True

    components["no_refund_promised"] = 0.0 if refund_promised else 0.25
    components["credit_offered"]     = 0.15 if credit_mentioned else 0.0
    if not credit_mentioned and not refund_promised:
        violations.append("Reply should offer account credit as alternative")

    # --- Resolution code ---
    resolve_acts = _find_actions_of_type(action_history, "resolve")
    if resolve_acts:
        code = resolve_acts[-1].get("resolution_code", "")
        components["correct_resolution_code"] = (
            0.20 if code == "wont_fix" else
            0.10 if code in ("solved", "workaround") else
            0.05 if code else 0.0
        )
        if code != "wont_fix":
            violations.append(f"Resolution code '{code}'; expected 'wont_fix' for policy denial")
    else:
        components["correct_resolution_code"] = 0.0
        violations.append("Ticket not resolved")

    # --- No wrong escalation ---
    escalate_acts = _find_actions_of_type(action_history, "escalate")
    wrong_esc = [e for e in escalate_acts if e.get("escalation_tier") == "engineering"]
    components["no_wrong_escalation"] = 0.0 if wrong_esc else 0.05
    if wrong_esc:
        violations.append("Escalated to engineering for a billing/refund issue")

    # --- Efficiency ---
    ratio = total_steps / max(max_steps, 1)
    components["step_efficiency"] = max(0.0, (1.0 - ratio)) * 0.05

    score = min(1.0, sum(components.values()))
    return GradeResult(
        task_id=task_id, score=score, passed=score >= 0.55,
        pass_threshold=0.55, component_scores=components,
        violations=violations,
        notes=f"Steps: {total_steps}/{max_steps} | refund_promised={refund_promised}",
    )


# ---------------------------------------------------------------------------
# Task 3 Grader — Hard: Security breach escalation + GDPR
# ---------------------------------------------------------------------------

def grade_task3(
    action_history: List[Dict],
    final_ticket: Ticket,
    total_steps: int,
    max_steps: int,
) -> GradeResult:
    """
    Scoring rubric:
      classified_security       (0.15)
      priority_critical         (0.10)
      escalated_to_engineering  (0.20)  ← required path
      escalation_has_reason     (0.05)
      gdpr_in_reply             (0.15)
      reply_quality             (0.10)
      internal_summary          (0.10)
      correct_tags              (0.10)
      sla_within_2_steps        (0.05)
    """
    task_id = "task_hard_security_escalation"
    components: Dict[str, float] = {}
    violations: List[str] = []

    # --- Classification ---
    classify_acts = _find_actions_of_type(action_history, "classify")
    if classify_acts:
        cat = classify_acts[-1].get("category", "").lower()
        components["classified_security"] = (
            0.15 if cat == "security/breach" else
            0.08 if cat.startswith("security") else 0.0
        )
        if not cat.startswith("security"):
            violations.append(f"Wrong category '{cat}'; breach must be 'security/breach'")
    else:
        components["classified_security"] = 0.0
        violations.append("No classify action — critical for security triage")

    # --- Priority critical ---
    if classify_acts:
        pri = classify_acts[-1].get("priority", "")
        components["priority_critical"] = 0.10 if pri == "critical" else (
            0.05 if pri == "high" else 0.0
        )
        if pri != "critical":
            violations.append(f"Priority '{pri}'; breach at enterprise tier must be 'critical'")
    else:
        components["priority_critical"] = 0.0

    # --- Escalation to engineering ---
    escalate_acts = _find_actions_of_type(action_history, "escalate")
    eng_escalations = [e for e in escalate_acts if e.get("escalation_tier") == "engineering"]
    components["escalated_to_engineering"] = 0.20 if eng_escalations else 0.0
    if not eng_escalations:
        violations.append("Must escalate to 'engineering' tier for security breach")

    # Escalation has reason
    if eng_escalations:
        reason = eng_escalations[-1].get("escalation_reason", "")
        components["escalation_has_reason"] = 0.05 if (reason and len(reason) > 15) else 0.02
        if not reason:
            violations.append("Escalation missing reason text")
    else:
        components["escalation_has_reason"] = 0.0

    # --- GDPR in reply ---
    reply_acts = _find_actions_of_type(action_history, "draft_reply")
    gdpr_mentioned = False
    reply_quality_score = 0.0
    for r in reply_acts:
        text = (r.get("reply_text") or "").lower()
        if "gdpr" in text or "article 33" in text or "72 hour" in text or "72h" in text:
            gdpr_mentioned = True
        rubric = {
            "required_topics": ["gdpr", "escalat", "engineer", "incident"],
            "forbidden_phrases": ["billing", "invoice", "self-service"],
            "min_length": 50, "max_length": 400, "greeting_required": True,
        }
        reply_quality_score = max(reply_quality_score, score_reply_quality(text, rubric))

    components["gdpr_in_reply"] = 0.15 if gdpr_mentioned else 0.0
    if not gdpr_mentioned:
        violations.append("Reply must acknowledge GDPR Article 33 / 72h notification duty")
    components["reply_quality"] = reply_quality_score * 0.10

    # --- Internal summary ---
    summarize_acts = _find_actions_of_type(action_history, "summarize")
    if summarize_acts:
        summary = summarize_acts[-1].get("summary_text", "") or ""
        if len(summary.split()) >= 20:
            components["internal_summary"] = 0.10
        else:
            components["internal_summary"] = 0.05
            violations.append("Internal summary too short (< 20 words)")
    else:
        components["internal_summary"] = 0.0
        violations.append("No summarize action — required for legal paper trail")

    # --- Correct tags ---
    required_tags = {"gdpr", "security", "breach", "enterprise"}
    tag_acts = _find_actions_of_type(action_history, "tag")
    applied_tags: set = set()
    for t in tag_acts:
        applied_tags.update([x.lower() for x in (t.get("tags") or [])])
    # Also check ticket tags set by env
    applied_tags.update([x.lower() for x in (final_ticket.tags or [])])

    tag_coverage = len(required_tags & applied_tags) / len(required_tags)
    components["correct_tags"] = tag_coverage * 0.10
    missing_tags = required_tags - applied_tags
    if missing_tags:
        violations.append(f"Missing required tags: {missing_tags}")

    # --- SLA: acknowledge + escalate within 2 steps ---
    early_steps = [i for i, a in enumerate(action_history, 1)
                   if a.get("action_type") in ("draft_reply", "escalate")]
    sla_met = bool(early_steps and early_steps[0] <= 2)
    components["sla_within_2_steps"] = 0.05 if sla_met else 0.0
    if not sla_met:
        violations.append("Critical SLA: first response or escalation must happen within step 2")

    score = min(1.0, sum(components.values()))
    return GradeResult(
        task_id=task_id, score=score, passed=score >= 0.50,
        pass_threshold=0.50, component_scores=components,
        violations=violations,
        notes=f"Steps: {total_steps}/{max_steps} | GDPR_ack={gdpr_mentioned} | eng_esc={bool(eng_escalations)}",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS = {
    "task_easy_classify_reply":     grade_task1,
    "task_medium_refund_policy":    grade_task2,
    "task_hard_security_escalation": grade_task3,
}


def grade_episode(
    task_id: str,
    action_history: List[Dict],
    final_ticket: Ticket,
    total_steps: int,
    max_steps: int,
) -> GradeResult:
    if task_id not in GRADERS:
        raise ValueError(f"No grader registered for task_id='{task_id}'")
    return GRADERS[task_id](action_history, final_ticket, total_steps, max_steps)
