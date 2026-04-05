"""
CustomerSupportEnv — Autonomous Customer Support Ops
=====================================================
Full OpenEnv-compliant environment class.

Interface:
    env = CustomerSupportEnv()
    obs = env.reset(task_id="task_easy_classify_reply")
    obs, reward, done, info = env.step(action)
    state = env.state()
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from customer_support_env.env.models import (
    Action, ActionType, EpisodeState, Observation, Priority,
    Reward, RewardBreakdown, Ticket, TicketStatus,
)

from customer_support_env.env.reward_function import compute_reward

from customer_support_env.tasks.task_definitions import (
    CATEGORIES, KNOWLEDGE_BASE, MACROS, get_task, list_tasks,
)

from customer_support_env.graders.graders import grade_episode

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """
    Autonomous Customer Support Operations environment.

    Simulates a real customer support queue where an agent must triage,
    respond to, and resolve customer tickets following business policies.

    Observation space:  See Observation model
    Action space:       See Action model
    Reward range:       [0.0, 1.0]
    Episode termination: RESOLVE or ESCALATE action, or max_steps exceeded
    """

    VERSION = "1.0.0"
    ENV_ID   = "customer-support-ops-v1"

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._state: Optional[EpisodeState] = None
        self._task_def: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset environment to initial state for given task_id.
        Defaults to the first registered task if none specified.
        """
        if task_id is None:
            task_id = list_tasks()[0]

        self._task_def = get_task(task_id)
        ticket_template: Ticket = self._task_def["ticket"]
        # Deep copy so mutations don't affect task definition
        ticket = ticket_template.model_copy(deep=True)

        self._state = EpisodeState(
            task_id=task_id,
            step=0,
            max_steps=self._task_def["max_steps"],
            done=False,
            ticket=ticket,
            cumulative_reward=0.0,
            reward_history=[],
            action_history=[],
            available_macros=MACROS.copy(),
            available_categories=CATEGORIES.copy(),
            knowledge_base=dict(KNOWLEDGE_BASE),
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action. Returns (observation, reward, done, info).
        Mutates internal ticket state according to action semantics.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step += 1
        info: Dict[str, Any] = {"step": self._state.step}

        # --- Apply action to ticket state ---
        feedback = self._apply_action(action)
        info["action_feedback"] = feedback

        # --- Determine if episode is terminal ---
        terminal_action = action.action_type in (ActionType.RESOLVE, ActionType.ESCALATE)
        step_limit_reached = self._state.step >= self._state.max_steps

        episode_done = terminal_action or step_limit_reached

        # --- Compute per-step reward ---
        reward = compute_reward(
            action=action,
            ticket=self._state.ticket,
            step=self._state.step,
            max_steps=self._state.max_steps,
            grading_spec=self._task_def["grading_spec"],
            episode_done=episode_done,
            action_history=self._state.action_history,
        )

        # --- Record action ---
        self._state.action_history.append(action.model_dump())
        self._state.reward_history.append(reward.score)
        self._state.cumulative_reward += reward.score

        # --- Finalise episode ---
        if episode_done:
            self._state.done = True
            # Run deterministic end-of-episode grader
            grade = grade_episode(
                task_id=self._state.task_id,
                action_history=self._state.action_history,
                final_ticket=self._state.ticket,
                total_steps=self._state.step,
                max_steps=self._state.max_steps,
            )
            info["grade"] = grade.to_dict()
            # Override final reward with grader score for episode-end consistency
            reward = Reward(
                score=grade.score,
                breakdown=reward.breakdown,
                rationale=f"[GRADED] {grade.notes} | violations={len(grade.violations)}",
            )
            info["episode_complete"] = True
            info["final_grade"] = grade.score
        else:
            info["episode_complete"] = False

        obs = self._build_observation(last_action_result=feedback)
        return obs, reward, self._state.done, info

    def state(self) -> EpisodeState:
        """Return full serialisable state snapshot."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Additional utility methods
    # ------------------------------------------------------------------

    def list_tasks(self) -> list:
        return list_tasks()

    def action_space_description(self) -> Dict[str, Any]:
        return {
            "action_types": [a.value for a in ActionType],
            "fields": Action.model_json_schema(),
        }

    def observation_space_description(self) -> Dict[str, Any]:
        return Observation.model_json_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> str:
        """Mutate ticket state based on action. Returns human-readable feedback."""
        ticket = self._state.ticket
        atype = action.action_type

        if atype == ActionType.CLASSIFY:
            if action.category:
                ticket.assigned_category = action.category
            if action.priority:
                ticket.priority = action.priority
            ticket.status = TicketStatus.IN_PROGRESS
            return f"Ticket classified as '{action.category}' with priority '{action.priority}'"

        elif atype == ActionType.DRAFT_REPLY:
            ticket.draft_reply = action.reply_text
            ticket.status = TicketStatus.IN_PROGRESS
            return "Draft reply recorded"

        elif atype == ActionType.ESCALATE:
            ticket.status = TicketStatus.ESCALATED
            ticket.escalation_reason = action.escalation_reason
            return f"Ticket escalated to '{action.escalation_tier}': {action.escalation_reason}"

        elif atype == ActionType.RESOLVE:
            ticket.status = TicketStatus.RESOLVED
            ticket.resolution_note = action.resolution_note
            return f"Ticket resolved [{action.resolution_code}]: {action.resolution_note}"

        elif atype == ActionType.REQUEST_INFO:
            ticket.status = TicketStatus.PENDING_CUSTOMER
            ticket.requested_info = action.reply_text
            return f"Requested info from customer: {action.reply_text}"

        elif atype == ActionType.APPLY_MACRO:
            if action.macro_id in self._state.available_macros:
                return f"Macro '{action.macro_id}' applied"
            return f"Unknown macro '{action.macro_id}' — no-op"

        elif atype == ActionType.TAG:
            if action.tags:
                current = set(ticket.tags or [])
                current.update(action.tags)
                ticket.tags = list(current)
            return f"Tags applied: {action.tags}"

        elif atype == ActionType.SUMMARIZE:
            ticket.internal_summary = action.summary_text
            return "Internal summary saved"

        return "Unknown action — no-op"

    def _build_observation(self, last_action_result: Optional[str] = None) -> Observation:
        s = self._state
        return Observation(
            task_id=s.task_id,
            step=s.step,
            max_steps=s.max_steps,
            ticket=s.ticket.model_copy(deep=True),
            available_macros=s.available_macros,
            available_categories=s.available_categories,
            knowledge_base=s.knowledge_base,
            last_action_result=last_action_result,
            done=s.done,
            info={
                "cumulative_reward": s.cumulative_reward,
                "steps_remaining": s.max_steps - s.step,
            },
        )
