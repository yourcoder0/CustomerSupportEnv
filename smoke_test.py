"""
smoke_test.py — Quick sanity check for all 3 tasks (no API key needed)
=======================================================================
Uses the same mock sequences as inference.py --mock mode.
Run: python smoke_test.py
"""
import sys
sys.path.insert(0, ".")

from environment import CustomerSupportEnv
from models import Action

env = CustomerSupportEnv()

SEPARATOR = "─" * 55

# ─────────────────────────────────────────────────────────────
# Task 1 — Easy: Invoice lookup
# ─────────────────────────────────────────────────────────────
print(SEPARATOR)
print("TASK 1 — EASY: Invoice Lookup")
print(SEPARATOR)

obs = env.reset("task_easy_classify_reply")
print(f"  Ticket : {obs.ticket.subject}")
print(f"  Tier   : {obs.ticket.customer_tier}")

obs, r, done, info = env.step(Action(
    action_type="classify", category="billing/invoice", priority="low"
))
print(f"  Step 1 | classify billing/invoice low | reward={r.score:.3f} | {info['action_feedback']}")

obs, r, done, info = env.step(Action(
    action_type="draft_reply",
    reply_text=(
        "Hello! Thank you for reaching out. You can download your invoice by going to "
        "Billing > Invoice History in your dashboard. There you will find a Download PDF button. "
        "Let us know if you need anything else!"
    )
))
print(f"  Step 2 | draft_reply                  | reward={r.score:.3f}")

obs, r, done, info = env.step(Action(
    action_type="resolve",
    resolution_code="solved",
    resolution_note="Directed customer to Billing > Invoice History for invoice download."
))
grade = info["grade"]
print(f"  Step 3 | resolve(solved)              | reward={r.score:.3f}")
print(f"  ► Final score : {grade['score']:.3f}  |  Passed: {grade['passed']}")
if grade["violations"]:
    for v in grade["violations"]:
        print(f"    ⚠  {v}")
else:
    print("    ✓  No violations")

# ─────────────────────────────────────────────────────────────
# Task 2 — Medium: Refund policy denial
# ─────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("TASK 2 — MEDIUM: Refund Policy (Free Tier)")
print(SEPARATOR)

obs = env.reset("task_medium_refund_policy")
print(f"  Ticket : {obs.ticket.subject}")
print(f"  Tier   : {obs.ticket.customer_tier}")

obs, r, done, info = env.step(Action(
    action_type="classify", category="billing/refund", priority="high"
))
print(f"  Step 1 | classify billing/refund high  | reward={r.score:.3f}")

obs, r, done, info = env.step(Action(
    action_type="draft_reply",
    reply_text=(
        "Hello, we understand your frustration and sincerely appreciate you reaching out. "
        "After reviewing your account, we are unable to process a refund under our policy — "
        "refunds are not available for Free tier accounts, and requests made more than 30 days "
        "after purchase fall outside our eligibility window. "
        "As a gesture of goodwill, we would be happy to offer you an account credit toward "
        "a future Pro upgrade. Please reply if you would like us to apply this credit."
    )
))
print(f"  Step 2 | draft_reply (deny + credit)   | reward={r.score:.3f}")

obs, r, done, info = env.step(Action(
    action_type="resolve",
    resolution_code="wont_fix",
    resolution_note="Refund denied: Free tier not eligible + 45 days > 30-day window. Credit offered."
))
grade = info["grade"]
print(f"  Step 3 | resolve(wont_fix)             | reward={r.score:.3f}")
print(f"  ► Final score : {grade['score']:.3f}  |  Passed: {grade['passed']}")
if grade["violations"]:
    for v in grade["violations"]:
        print(f"    ⚠  {v}")
else:
    print("    ✓  No violations")

# ─────────────────────────────────────────────────────────────
# Task 3 — Hard: GDPR Security Breach Escalation
# KEY INSIGHT: escalate is a TERMINAL action — it ends the episode.
# All non-terminal actions (draft_reply, summarize, tag) must fire
# BEFORE escalate, or the grader never sees them.
# Correct order: classify → draft_reply → summarize → tag → escalate
# ─────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("TASK 3 — HARD: GDPR Security Breach Escalation")
print(SEPARATOR)

obs = env.reset("task_hard_security_escalation")
print(f"  Ticket : {obs.ticket.subject}")
print(f"  Tier   : {obs.ticket.customer_tier}")

# Step 1 — classify (non-terminal)
obs, r, done, info = env.step(Action(
    action_type="classify", category="security/breach", priority="critical"
))
print(f"  Step 1 | classify security/breach crit  | reward={r.score:.3f} | done={done}")

# Step 2 — GDPR-ack reply to customer (non-terminal, satisfies SLA ≤ 2 steps check)
obs, r, done, info = env.step(Action(
    action_type="draft_reply",
    reply_text=(
        "Hello, we are treating this as a critical incident and have immediately engaged our "
        "engineering incident response team. We fully acknowledge your GDPR Article 33 "
        "obligations and the 72-hour notification window for reporting to the DPA. "
        "Our engineers will contact you within the hour with an initial incident assessment "
        "and will assist with any authority notification requirements. "
        "We are fully committed to resolving this as a top priority."
    )
))
print(f"  Step 2 | draft_reply (GDPR ack, SLA≤2)  | reward={r.score:.3f} | done={done}")

# Step 3 — internal legal summary (non-terminal)
obs, r, done, info = env.step(Action(
    action_type="summarize",
    summary_text=(
        "CRITICAL SECURITY INCIDENT — Enterprise TK-003 ($2M ARR). "
        "Customer reports API keys accessed from unknown Eastern European IP at 03:00 UTC. "
        "Keys rotated immediately. Potential PII exposure confirmed. "
        "GDPR Article 33 invoked — 72-hour DPA notification window active. "
        "Legal team must be notified. Customer DPO is engaged."
    )
))
print(f"  Step 3 | summarize (legal paper trail)  | reward={r.score:.3f} | done={done}")

# Step 4 — required tags (non-terminal)
obs, r, done, info = env.step(Action(
    action_type="tag",
    tags=["gdpr", "security", "breach", "enterprise"]
))
print(f"  Step 4 | tag [gdpr,security,breach,...] | reward={r.score:.3f} | done={done}")

# Step 5 — escalate to engineering (TERMINAL — closes episode with all checks satisfied)
obs, r, done, info = env.step(Action(
    action_type="escalate",
    escalation_tier="engineering",
    escalation_reason=(
        "Enterprise account ($2M ARR) reports API key compromise from unknown IP. "
        "GDPR Article 33 invoked — 72-hour DPA window. Potential PII breach. "
        "Immediate IR and legal notification required."
    )
))
grade = info["grade"]
print(f"  Step 5 | escalate → engineering (END)   | reward={r.score:.3f} | done={done}")
print(f"  ► Final score : {grade['score']:.3f}  |  Passed: {grade['passed']}")
if grade["violations"]:
    for v in grade["violations"]:
        print(f"    ⚠  {v}")
else:
    print("    ✓  No violations")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("SMOKE TEST COMPLETE")
print("=" * 55)
print("Expected scores:")
print("  Task 1 (easy)   ≥ 0.75")
print("  Task 2 (medium) ≥ 0.55")
print("  Task 3 (hard)   ≥ 0.80  (near-optimal mock actions)")
print()
print("To run the full inference with live LLM:")
print("  python inference.py")
print()
print("To run mock mode (no API key):")
print("  python inference.py --mock")
print()
print("To run a single task:")
print("  python inference.py --mock --task easy")
print("  python inference.py --mock --task medium")
print("  python inference.py --mock --task hard")