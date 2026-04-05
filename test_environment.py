"""
tests/test_environment.py — Unit + integration tests for CustomerSupportEnv

Run: pytest tests/ -v
"""

from __future__ import annotations
import sys
import os
import pytest

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import CustomerSupportEnv
from models import Action, ActionType, Priority, Reward, TicketStatus
from reward_function import (
    score_classification, score_priority, score_reply_quality,
    score_sla, score_efficiency, compute_reward,
)
from graders import (
    grade_task1, grade_task2, grade_task3, grade_episode,
)
from task_definitions import (
    list_tasks, get_task, TASK_1, TASK_2, TASK_3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return CustomerSupportEnv(seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoke:
    def test_env_creates(self, env):
        assert env is not None

    def test_three_tasks_registered(self, env):
        tasks = env.list_tasks()
        assert len(tasks) == 3

    def test_reset_returns_observation(self, env):
        obs = env.reset("task_easy_classify_reply")
        assert obs.task_id == "task_easy_classify_reply"
        assert obs.step == 0
        assert obs.done is False
        assert obs.ticket is not None

    def test_state_before_reset_raises(self):
        fresh_env = CustomerSupportEnv()
        with pytest.raises(RuntimeError):
            fresh_env.state()

    def test_step_before_reset_raises(self):
        fresh_env = CustomerSupportEnv()
        with pytest.raises(RuntimeError):
            fresh_env.step(Action(action_type=ActionType.RESOLVE))


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskDefinitions:
    def test_task_ids_unique(self):
        tasks = list_tasks()
        assert len(tasks) == len(set(tasks))

    def test_all_tasks_have_required_keys(self):
        required = {"task_id", "description", "difficulty", "max_steps",
                    "ticket", "grading_spec", "optimal_sequence"}
        for tid in list_tasks():
            task = get_task(tid)
            assert required.issubset(set(task.keys())), f"{tid} missing keys"

    def test_difficulties_progression(self):
        diffs = [get_task(t)["difficulty"] for t in list_tasks()]
        assert diffs == ["easy", "medium", "hard"]

    def test_max_steps_within_budget(self):
        for tid in list_tasks():
            task = get_task(tid)
            assert 1 <= task["max_steps"] <= 10, f"{tid} max_steps out of range"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")


# ─────────────────────────────────────────────────────────────────────────────
# Reward function unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardFunction:
    def test_classification_exact_match(self):
        a = Action(action_type=ActionType.CLASSIFY, category="billing/invoice")
        assert score_classification(a, "billing/invoice") == 1.0

    def test_classification_partial_namespace(self):
        a = Action(action_type=ActionType.CLASSIFY, category="billing/refund")
        assert score_classification(a, "billing/invoice") == 0.5

    def test_classification_wrong(self):
        a = Action(action_type=ActionType.CLASSIFY, category="technical/login")
        assert score_classification(a, "billing/invoice") == 0.0

    def test_priority_exact(self):
        a = Action(action_type=ActionType.CLASSIFY, priority=Priority.LOW)
        assert score_priority(a, "low", "pro") == 1.0

    def test_priority_adjacent(self):
        a = Action(action_type=ActionType.CLASSIFY, priority=Priority.MEDIUM)
        assert score_priority(a, "low", "pro") == 0.5

    def test_priority_enterprise_under_penalty(self):
        # Under-prioritising enterprise = harsher (0.25 not 0.5)
        a = Action(action_type=ActionType.CLASSIFY, priority=Priority.HIGH)
        score = score_priority(a, "critical", "enterprise")
        assert score == 0.25

    def test_reply_quality_all_topics(self):
        rubric = {
            "required_topics": ["billing", "invoice", "download"],
            "forbidden_phrases": ["cannot help"],
            "min_length": 10, "max_length": 200, "greeting_required": True,
        }
        good_reply = "Hello! You can find your invoice in Billing > Invoice History and download it there."
        score = score_reply_quality(good_reply, rubric)
        assert score >= 0.7

    def test_reply_quality_forbidden_phrase(self):
        rubric = {
            "required_topics": [],
            "forbidden_phrases": ["cannot help"],
            "min_length": 5, "max_length": 200, "greeting_required": False,
        }
        bad_reply = "Sorry we cannot help with this issue."
        score = score_reply_quality(bad_reply, rubric)
        # forbidden phrase should penalise
        assert score < 0.9

    def test_sla_within_budget(self):
        assert score_sla(step=2, priority="critical", resolved=True) == 1.0

    def test_sla_over_budget(self):
        assert score_sla(step=5, priority="critical", resolved=True) < 1.0

    def test_sla_not_resolved(self):
        assert score_sla(step=1, priority="high", resolved=False) == 0.0

    def test_efficiency_early(self):
        assert score_efficiency(step=1, max_steps=10) == 1.0

    def test_efficiency_late(self):
        assert score_efficiency(step=10, max_steps=10) == 0.0

    def test_reward_score_in_range(self, env):
        """compute_reward must always return score in [0, 1]."""
        from task_definitions import TASK_1
        ticket = TASK_1["ticket"].model_copy(deep=True)
        action = Action(action_type=ActionType.CLASSIFY,
                        category="billing/invoice", priority=Priority.LOW)
        reward = compute_reward(
            action=action, ticket=ticket, step=1, max_steps=3,
            grading_spec=TASK_1["grading_spec"],
            episode_done=False, action_history=[],
        )
        assert 0.0 <= reward.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Grader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraders:
    def _make_ticket(self, task_id: str):
        return get_task(task_id)["ticket"].model_copy(deep=True)

    # ── Task 1 ──────────────────────────────────────────────────────────────

    def test_grade_task1_optimal(self):
        ticket = self._make_ticket("task_easy_classify_reply")
        ticket.assigned_category = "billing/invoice"
        history = [
            {"action_type": "classify", "category": "billing/invoice", "priority": "low"},
            {"action_type": "draft_reply",
             "reply_text": "Hello! You can download your invoice from Billing > Invoice History."},
            {"action_type": "resolve", "resolution_code": "solved",
             "resolution_note": "Directed customer to invoice download page."},
        ]
        result = grade_task1(history, ticket, total_steps=3, max_steps=3)
        assert result.score >= 0.75
        assert result.passed

    def test_grade_task1_no_classify(self):
        ticket = self._make_ticket("task_easy_classify_reply")
        history = [
            {"action_type": "draft_reply", "reply_text": "Hello! Check billing."},
            {"action_type": "resolve", "resolution_code": "solved", "resolution_note": "done"},
        ]
        result = grade_task1(history, ticket, total_steps=2, max_steps=3)
        assert result.component_scores.get("classification_correct", 0) == 0.0
        assert any("No classify action" in v for v in result.violations)

    def test_grade_task1_wrong_category(self):
        ticket = self._make_ticket("task_easy_classify_reply")
        history = [{"action_type": "classify", "category": "technical/login", "priority": "low"}]
        result = grade_task1(history, ticket, total_steps=1, max_steps=3)
        assert result.component_scores["classification_correct"] == 0.0

    def test_grade_task1_partial_category(self):
        ticket = self._make_ticket("task_easy_classify_reply")
        history = [{"action_type": "classify", "category": "billing/refund", "priority": "low"}]
        result = grade_task1(history, ticket, total_steps=1, max_steps=3)
        assert result.component_scores["classification_correct"] == 0.125

    # ── Task 2 ──────────────────────────────────────────────────────────────

    def test_grade_task2_policy_violation(self):
        """Promising refund should score 0 on no_refund_promised component."""
        ticket = self._make_ticket("task_medium_refund_policy")
        history = [
            {"action_type": "classify", "category": "billing/refund", "priority": "high"},
            {"action_type": "draft_reply",
             "reply_text": "Hello! Refund approved, we will refund your payment today."},
            {"action_type": "resolve", "resolution_code": "solved"},
        ]
        result = grade_task2(history, ticket, total_steps=3, max_steps=5)
        assert result.component_scores.get("no_refund_promised", 0) == 0.0
        assert any("Policy violation" in v for v in result.violations)

    def test_grade_task2_optimal(self):
        ticket = self._make_ticket("task_medium_refund_policy")
        ticket.assigned_category = "billing/refund"
        history = [
            {"action_type": "classify", "category": "billing/refund", "priority": "high"},
            {"action_type": "draft_reply",
             "reply_text": (
                 "Hello, we understand your frustration. Unfortunately, our policy does not allow "
                 "refunds for free-tier accounts or purchases older than 30 days. We'd be happy to "
                 "offer you an account credit for future use instead."
             )},
            {"action_type": "resolve", "resolution_code": "wont_fix",
             "resolution_note": "Policy: free tier ineligible, >30 days."},
        ]
        result = grade_task2(history, ticket, total_steps=3, max_steps=5)
        assert result.score >= 0.55
        assert result.passed

    def test_grade_task2_wrong_escalation(self):
        ticket = self._make_ticket("task_medium_refund_policy")
        history = [
            {"action_type": "classify", "category": "billing/refund", "priority": "high"},
            {"action_type": "escalate", "escalation_tier": "engineering",
             "escalation_reason": "Customer angry"},
        ]
        result = grade_task2(history, ticket, total_steps=2, max_steps=5)
        assert result.component_scores.get("no_wrong_escalation", 0) == 0.0

    # ── Task 3 ──────────────────────────────────────────────────────────────

    def test_grade_task3_optimal(self):
        ticket = self._make_ticket("task_hard_security_escalation")
        ticket.tags = ["gdpr", "security", "breach", "enterprise"]
        ticket.assigned_category = "security/breach"
        ticket.internal_summary = "Enterprise account reported potential API key breach. GDPR Article 33 invoked. Escalated to engineering team for incident response."
        history = [
            {"action_type": "classify", "category": "security/breach", "priority": "critical"},
            {"action_type": "escalate", "escalation_tier": "engineering",
             "escalation_reason": "Potential data breach with GDPR implications. Enterprise account $2M ARR."},
            {"action_type": "draft_reply",
             "reply_text": (
                 "Hello, we take this extremely seriously. We have escalated your case to our "
                 "engineering incident response team immediately. In line with GDPR Article 33, "
                 "we will ensure notification within the 72-hour window. Our team will contact "
                 "you within the hour."
             )},
            {"action_type": "summarize",
             "summary_text": "Enterprise account reports possible API key compromise from unknown IP. Customer invoked GDPR Article 33. Escalated to engineering. Legal notification required within 72h."},
            {"action_type": "tag", "tags": ["gdpr", "security", "breach", "enterprise"]},
        ]
        result = grade_task3(history, ticket, total_steps=5, max_steps=6)
        assert result.score >= 0.80
        assert result.passed

    def test_grade_task3_wrong_escalation_tier(self):
        ticket = self._make_ticket("task_hard_security_escalation")
        history = [
            {"action_type": "classify", "category": "security/breach", "priority": "critical"},
            {"action_type": "escalate", "escalation_tier": "billing",
             "escalation_reason": "Charge dispute"},
        ]
        result = grade_task3(history, ticket, total_steps=2, max_steps=6)
        assert result.component_scores.get("escalated_to_engineering", 0) == 0.0
        assert any("engineering" in v.lower() for v in result.violations)

    def test_grade_task3_no_gdpr_mention(self):
        ticket = self._make_ticket("task_hard_security_escalation")
        history = [
            {"action_type": "classify", "category": "security/breach", "priority": "critical"},
            {"action_type": "escalate", "escalation_tier": "engineering",
             "escalation_reason": "Potential breach."},
            {"action_type": "draft_reply",
             "reply_text": "Hello! We have escalated your ticket to engineering. Thank you for reaching out."},
        ]
        result = grade_task3(history, ticket, total_steps=3, max_steps=6)
        assert result.component_scores.get("gdpr_in_reply", 0) == 0.0

    def test_grade_task3_missing_summary(self):
        ticket = self._make_ticket("task_hard_security_escalation")
        history = [
            {"action_type": "classify", "category": "security/breach", "priority": "critical"},
            {"action_type": "escalate", "escalation_tier": "engineering",
             "escalation_reason": "Security breach with GDPR implications."},
        ]
        result = grade_task3(history, ticket, total_steps=2, max_steps=6)
        assert result.component_scores.get("internal_summary", 0) == 0.0
        assert "summarize" in " ".join(result.violations).lower()

    def test_grade_registry_coverage(self):
        """All tasks must have a registered grader."""
        for tid in list_tasks():
            ticket = get_task(tid)["ticket"].model_copy(deep=True)
            result = grade_episode(tid, [], ticket, 0, 10)
            assert 0.0 <= result.score <= 1.0, f"{tid} grader out of range"


# ─────────────────────────────────────────────────────────────────────────────
# Environment integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentIntegration:
    def test_full_episode_task1(self, env):
        obs = env.reset("task_easy_classify_reply")
        assert obs.step == 0

        # Step 1: classify
        obs, r, done, info = env.step(Action(
            action_type=ActionType.CLASSIFY,
            category="billing/invoice",
            priority=Priority.LOW,
        ))
        assert not done
        assert 0.0 <= r.score <= 1.0

        # Step 2: reply
        obs, r, done, info = env.step(Action(
            action_type=ActionType.DRAFT_REPLY,
            reply_text="Hello! You can download invoices from Billing > Invoice History.",
        ))
        assert not done

        # Step 3: resolve — should terminate
        obs, r, done, info = env.step(Action(
            action_type=ActionType.RESOLVE,
            resolution_code="solved",
            resolution_note="Directed customer to invoice download.",
        ))
        assert done
        assert "grade" in info
        assert 0.0 <= info["grade"]["score"] <= 1.0

    def test_terminal_action_ends_episode(self, env):
        env.reset("task_easy_classify_reply")
        _, _, done, _ = env.step(Action(
            action_type=ActionType.RESOLVE,
            resolution_code="solved",
            resolution_note="Immediate resolve.",
        ))
        assert done

    def test_step_after_done_raises(self, env):
        env.reset("task_easy_classify_reply")
        env.step(Action(action_type=ActionType.RESOLVE, resolution_code="solved",
                        resolution_note="done"))
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.CLASSIFY, category="billing/invoice"))

    def test_state_is_consistent(self, env):
        env.reset("task_easy_classify_reply")
        env.step(Action(action_type=ActionType.CLASSIFY,
                        category="billing/invoice", priority=Priority.LOW))
        s = env.state()
        assert s.step == 1
        assert s.ticket.assigned_category == "billing/invoice"

    def test_reset_clears_state(self, env):
        env.reset("task_easy_classify_reply")
        env.step(Action(action_type=ActionType.CLASSIFY,
                        category="billing/invoice", priority=Priority.LOW))
        obs = env.reset("task_easy_classify_reply")
        assert obs.step == 0
        assert obs.ticket.assigned_category is None

    def test_all_tasks_run_to_completion(self, env):
        """Each task must complete without error with a greedy policy."""
        for task_id in list_tasks():
            obs = env.reset(task_id)
            task_def = get_task(task_id)
            done = False
            for _ in range(task_def["max_steps"] + 1):
                if done:
                    break
                # Greedy: always resolve
                _, r, done, info = env.step(Action(
                    action_type=ActionType.RESOLVE,
                    resolution_code="solved",
                    resolution_note="Test greedy agent.",
                ))
                assert 0.0 <= r.score <= 1.0

    def test_reward_always_in_range(self, env):
        """Stress test: random-ish actions should never produce out-of-range reward."""
        import random
        random.seed(42)
        actions = [
            Action(action_type=ActionType.CLASSIFY, category="billing/refund", priority=Priority.HIGH),
            Action(action_type=ActionType.DRAFT_REPLY, reply_text="Hello! " + "x " * 30),
            Action(action_type=ActionType.TAG, tags=["test"]),
            Action(action_type=ActionType.ESCALATE, escalation_tier="billing",
                   escalation_reason="Test escalation"),
        ]
        for task_id in list_tasks():
            obs = env.reset(task_id)
            for action in actions:
                if env._state.done:
                    break
                _, r, done, _ = env.step(action)
                assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range on {task_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases and failure scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_reply_penalised(self, env):
        env.reset("task_easy_classify_reply")
        _, r, _, _ = env.step(Action(action_type=ActionType.DRAFT_REPLY, reply_text=""))
        assert r.score < 0.3

    def test_very_long_reply_penalised(self, env):
        env.reset("task_easy_classify_reply")
        long_text = "word " * 400  # well over max_length
        _, r, _, _ = env.step(Action(action_type=ActionType.DRAFT_REPLY, reply_text=long_text))
        # Quality should be reduced for excessive length
        assert r.breakdown.reply_quality < 0.25

    def test_contradictory_actions_penalised(self, env):
        env.reset("task_medium_refund_policy")
        env.step(Action(action_type=ActionType.CLASSIFY,
                        category="billing/refund", priority=Priority.HIGH))
        env.step(Action(action_type=ActionType.DRAFT_REPLY,
                        reply_text="Hello we understand your request and appreciate your patience with us."))
        # Escalate after partial resolution path — contradiction
        _, r, done, _ = env.step(Action(
            action_type=ActionType.ESCALATE,
            escalation_tier="billing",
            escalation_reason="Need billing review.",
        ))
        assert done  # escalate terminates episode
        # Penalty should be visible in score

    def test_missing_escalation_reason_penalised(self, env):
        env.reset("task_hard_security_escalation")
        env.step(Action(action_type=ActionType.CLASSIFY,
                        category="security/breach", priority=Priority.CRITICAL))
        _, r, done, info = env.step(Action(
            action_type=ActionType.ESCALATE,
            escalation_tier="engineering",
            escalation_reason=None,  # missing!
        ))
        # Penalty applied for missing field
        assert r.breakdown.penalty < 0.0 or info.get("grade", {}).get("score", 1.0) < 0.9

    def test_wrong_macro_id_is_noop(self, env):
        """Applying a non-existent macro should not crash the env."""
        env.reset("task_easy_classify_reply")
        obs, r, done, info = env.step(Action(
            action_type=ActionType.APPLY_MACRO,
            macro_id="macro_does_not_exist",
        ))
        assert not done
        assert "no-op" in info.get("action_feedback", "").lower()

    def test_score_deterministic(self, env):
        """Running same sequence twice must produce identical scores."""
        scores = []
        for _ in range(2):
            env.reset("task_easy_classify_reply")
            env.step(Action(action_type=ActionType.CLASSIFY,
                            category="billing/invoice", priority=Priority.LOW))
            _, r, done, info = env.step(Action(
                action_type=ActionType.RESOLVE, resolution_code="solved",
                resolution_note="Directed to invoice page.",
            ))
            scores.append(info["grade"]["score"])
        assert scores[0] == scores[1], "Grader is not deterministic!"

    def test_task3_sla_violation_reduces_score(self):
        """SLA within 2 steps must fail if first action is step 4."""
        from task_definitions import TASK_3
        ticket = TASK_3["ticket"].model_copy(deep=True)
        ticket.assigned_category = "security/breach"
        # Action history with escalation at step 4 (late)
        history = [
            {"action_type": "classify",   "category": "security/breach", "priority": "critical"},
            {"action_type": "tag",        "tags": ["security"]},
            {"action_type": "summarize",  "summary_text": "Internal note about breach."},
            {"action_type": "escalate",   "escalation_tier": "engineering",
             "escalation_reason": "Data breach GDPR."},
        ]
        result = grade_task3(history, ticket, total_steps=4, max_steps=6)
        # Step 4 is beyond SLA step 2 threshold
        assert result.component_scores.get("sla_within_2_steps", 0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
