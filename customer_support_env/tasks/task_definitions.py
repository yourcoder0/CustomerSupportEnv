"""
Task Definitions — Autonomous Customer Support Ops
====================================================
Three tasks of increasing complexity:
  Task 1 (easy):   Single-ticket classification + canned reply
  Task 2 (medium): Multi-step resolution with policy constraints
  Task 3 (hard):   Escalation decision tree + cross-ticket summarisation
"""

from __future__ import annotations
from typing import Any, Dict, List
from customer_support_env.env.models import (
    Message, Ticket, TicketStatus, Priority
)


# ---------------------------------------------------------------------------
# Shared knowledge base & macros
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: Dict[str, str] = {
    "billing/refund_policy": (
        "Refunds are available within 30 days of purchase for Pro and Enterprise tiers. "
        "Free tier users are not eligible for refunds but may receive credits."
    ),
    "technical/password_reset": (
        "Users can reset passwords via Settings > Security > Reset Password. "
        "If blocked, issue a temporary link via admin console."
    ),
    "technical/api_rate_limit": (
        "API rate limits: Free=60 req/min, Pro=600 req/min, Enterprise=unlimited. "
        "Rate limit errors return HTTP 429."
    ),
    "billing/invoice_download": (
        "Invoices available under Billing > Invoice History. "
        "Enterprise clients receive monthly PDF invoices by email."
    ),
    "technical/data_export": (
        "Data export via Settings > Data > Export. GDPR requests must be processed in 72h."
    ),
    "escalation/engineering": (
        "Escalate to engineering for: data loss, security breaches, API outages."
    ),
    "escalation/billing_team": (
        "Escalate to billing for: disputed charges > $500, fraud reports, invoicing disputes."
    ),
    "policy/sla": (
        "Critical: 1h, High: 4h, Medium: 24h, Low: 72h first response SLA."
    ),
}

MACROS: List[str] = [
    "macro_billing_refund_denied_free",
    "macro_password_reset_link",
    "macro_api_docs_link",
    "macro_escalation_ack",
    "macro_close_no_response",
    "macro_request_order_number",
    "macro_gdpr_ack",
]

CATEGORIES: List[str] = [
    "billing/refund",
    "billing/invoice",
    "billing/charge_dispute",
    "technical/login",
    "technical/api",
    "technical/data_export",
    "technical/performance",
    "account/cancellation",
    "account/upgrade",
    "general/feedback",
    "security/breach",
]


# ---------------------------------------------------------------------------
# Task 1 — EASY: Classify and Reply
# ---------------------------------------------------------------------------
# Objective: Given a simple billing inquiry from a Pro customer,
#   (a) classify it correctly, (b) draft an appropriate reply.
# Max steps: 3  (classify → draft reply → resolve)
# ---------------------------------------------------------------------------

TASK_1: Dict[str, Any] = {
    "task_id": "task_easy_classify_reply",
    "description": (
        "A Pro-tier customer cannot find their invoice. "
        "The agent must: (1) classify the ticket, (2) draft a helpful reply "
        "pointing to the correct self-serve location, (3) resolve the ticket."
    ),
    "difficulty": "easy",
    "max_steps": 3,
    "ticket": Ticket(
        ticket_id="TK-001",
        subject="Where can I download my invoice?",
        body=(
            "Hi, I was charged last week and need the invoice for my accountant. "
            "I looked in the billing section but couldn't find a download button. "
            "Please help. Thanks."
        ),
        customer_tier="pro",
        product="SaaS Platform",
        history=[
            Message(role="system",   content="Ticket opened via web form.",     timestamp="2024-01-15T09:00:00Z"),
            Message(role="customer", content="I need my invoice urgently.",      timestamp="2024-01-15T09:01:00Z"),
        ],
        status=TicketStatus.OPEN,
    ),
    "grading_spec": {
        "expected_category":   "billing/invoice",
        "expected_priority":   "low",
        "expected_resolution": "solved",
        "reply_rubric": {
            "required_topics": ["billing", "invoice", "download"],
            "forbidden_phrases": ["i don't know", "cannot help", "call us"],
            "min_length": 25,
            "max_length": 200,
            "greeting_required": True,
        },
        "policy_rules": {
            "no_refund_for_free": False,
            "escalation_tiers_allowed": ["billing"],
            "must_classify_before_resolve": True,
        },
        "is_terminal_action": True,
    },
    "optimal_sequence": ["classify", "draft_reply", "resolve"],
    "baseline_expected_score": 0.75,
}


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: Policy-constrained refund handling
# ---------------------------------------------------------------------------
# Objective: Free-tier customer demands a refund after 45 days.
#   Policy: free tier ineligible, >30 days ineligible even for Pro.
#   Agent must: classify, apply correct policy, deny politely with credit offer,
#   resolve correctly without promising a refund.
# Max steps: 5
# ---------------------------------------------------------------------------

TASK_2: Dict[str, Any] = {
    "task_id": "task_medium_refund_policy",
    "description": (
        "A Free-tier customer purchased 45 days ago and is demanding a full refund. "
        "Policy prohibits refunds for Free tier and for requests >30 days old. "
        "The agent must classify, identify policy violation, deny appropriately, "
        "offer an account credit instead, and resolve without escalating."
    ),
    "difficulty": "medium",
    "max_steps": 5,
    "ticket": Ticket(
        ticket_id="TK-002",
        subject="I want a refund NOW - this is unacceptable",
        body=(
            "I signed up 45 days ago and the product is terrible. "
            "I demand a full refund of $0 (free plan) immediately. "
            "Your product didn't work as advertised. I will charge-back if needed."
        ),
        customer_tier="free",
        product="SaaS Platform",
        history=[
            Message(role="system",   content="Ticket flagged as potential chargeback risk.", timestamp="2024-01-15T10:00:00Z"),
            Message(role="customer", content="Nobody has replied yet! This is a scam.",      timestamp="2024-01-15T10:30:00Z"),
            Message(role="agent",    content="We've received your request and are reviewing.", timestamp="2024-01-15T10:32:00Z"),
        ],
        status=TicketStatus.IN_PROGRESS,
    ),
    "grading_spec": {
        "expected_category":   "billing/refund",
        "expected_priority":   "high",
        "expected_resolution": "wont_fix",
        "reply_rubric": {
            "required_topics": ["policy", "understand", "credit"],
            "forbidden_phrases": ["refund approved", "we will refund", "processed your refund"],
            "min_length": 40,
            "max_length": 300,
            "greeting_required": True,
        },
        "policy_rules": {
            "no_refund_for_free": True,
            "escalation_tiers_allowed": ["billing", "tier2"],
            "must_classify_before_resolve": True,
        },
        "is_terminal_action": True,
    },
    "optimal_sequence": ["classify", "draft_reply", "resolve"],
    "baseline_expected_score": 0.60,
}


# ---------------------------------------------------------------------------
# Task 3 — HARD: Security breach escalation + GDPR summary
# ---------------------------------------------------------------------------
# Objective: Enterprise customer reports potential data breach, mentions GDPR.
#   Agent must: classify as security/breach at CRITICAL priority, immediately
#   escalate to engineering tier (NOT billing), draft an acknowledgement
#   within SLA (step ≤ 2), produce an internal summary for legal review.
#   Any delay, wrong tier, or missing GDPR mention = score penalty.
# Max steps: 6
# ---------------------------------------------------------------------------

TASK_3: Dict[str, Any] = {
    "task_id": "task_hard_security_escalation",
    "description": (
        "An Enterprise customer reports suspicious activity suggesting their API keys "
        "were compromised. They invoke GDPR Article 33 (72h breach notification). "
        "The agent must: (1) classify as security/breach with CRITICAL priority, "
        "(2) escalate to the engineering tier immediately, "
        "(3) draft a GDPR-aware acknowledgement reply within 2 steps (SLA critical), "
        "(4) produce an internal summary for legal review, "
        "(5) tag ticket with [gdpr, security, breach, enterprise]."
    ),
    "difficulty": "hard",
    "max_steps": 6,
    "ticket": Ticket(
        ticket_id="TK-003",
        subject="URGENT: Possible data breach - GDPR Article 33",
        body=(
            "We detected that our API keys were used from an unknown IP in Eastern Europe "
            "at 03:00 UTC today. We immediately rotated keys but fear customer PII may have "
            "been accessed. Per GDPR Article 33, we have 72 hours to notify authorities. "
            "We need your incident response team NOW. This is a $2M ARR account."
        ),
        customer_tier="enterprise",
        product="API Platform",
        history=[
            Message(role="customer", content="We're also calling our DPO right now.", timestamp="2024-01-15T05:00:00Z"),
            Message(role="system",   content="VIP flag set. Account value: $2M ARR.",  timestamp="2024-01-15T05:01:00Z"),
        ],
        status=TicketStatus.OPEN,
    ),
    "grading_spec": {
        "expected_category":   "security/breach",
        "expected_priority":   "critical",
        "expected_resolution": "solved",   # resolved via engineering escalation path
        "reply_rubric": {
            "required_topics": ["gdpr", "escalat", "engineer", "24 hours", "incident"],
            "forbidden_phrases": ["billing", "invoice", "we cannot help", "self-service"],
            "min_length": 50,
            "max_length": 400,
            "greeting_required": True,
        },
        "policy_rules": {
            "no_refund_for_free": False,
            "escalation_tiers_allowed": ["engineering"],   # ONLY engineering is correct
            "must_classify_before_resolve": True,
        },
        "is_terminal_action": True,
        "required_tags": ["gdpr", "security", "breach", "enterprise"],
        "required_actions": ["classify", "escalate", "draft_reply", "summarize", "tag"],
    },
    "optimal_sequence": ["classify", "escalate", "draft_reply", "summarize", "tag"],
    "baseline_expected_score": 0.45,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    TASK_1["task_id"]: TASK_1,
    TASK_2["task_id"]: TASK_2,
    TASK_3["task_id"]: TASK_3,
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    return list(TASKS.keys())
