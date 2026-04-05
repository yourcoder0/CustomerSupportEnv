"""
Typed Pydantic models for the Autonomous Customer Support Ops environment.
Defines Observation, Action, Reward, and State schemas per OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class ActionType(str, Enum):
    CLASSIFY       = "classify"        # Assign category + priority
    DRAFT_REPLY    = "draft_reply"     # Write customer-facing reply
    ESCALATE       = "escalate"        # Escalate to human / tier-2
    RESOLVE        = "resolve"         # Mark ticket resolved with reason
    REQUEST_INFO   = "request_info"    # Ask customer for more info
    APPLY_MACRO    = "apply_macro"     # Apply a canned-response macro
    TAG            = "tag"             # Add metadata tags
    SUMMARIZE      = "summarize"       # Produce internal summary note


# ---------------------------------------------------------------------------
# Ticket & Conversation
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["customer", "agent", "system"]
    content: str
    timestamp: str  # ISO-8601 string, deterministic in tests


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    product: str
    history: List[Message] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    status: TicketStatus = TicketStatus.OPEN
    priority: Optional[Priority] = None
    assigned_category: Optional[str] = None
    resolution_note: Optional[str] = None
    escalation_reason: Optional[str] = None
    draft_reply: Optional[str] = None
    internal_summary: Optional[str] = None
    requested_info: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    max_steps: int
    ticket: Ticket
    available_macros: List[str] = Field(default_factory=list)
    available_categories: List[str] = Field(default_factory=list)
    knowledge_base: Dict[str, str] = Field(default_factory=dict)
    last_action_result: Optional[str] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """What the agent submits at each step."""
    action_type: ActionType
    # For CLASSIFY
    category: Optional[str] = None
    priority: Optional[Priority] = None
    # For DRAFT_REPLY / REQUEST_INFO
    reply_text: Optional[str] = None
    # For ESCALATE
    escalation_reason: Optional[str] = None
    escalation_tier: Optional[Literal["tier2", "billing", "engineering", "legal"]] = None
    # For RESOLVE
    resolution_note: Optional[str] = None
    resolution_code: Optional[Literal["solved", "workaround", "wont_fix", "duplicate", "spam"]] = None
    # For APPLY_MACRO
    macro_id: Optional[str] = None
    # For TAG
    tags: Optional[List[str]] = None
    # For SUMMARIZE
    summary_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed reward components (weights documented in reward_function.py)."""
    classification_accuracy: float = 0.0    # max 0.20
    priority_accuracy: float = 0.0           # max 0.10
    reply_quality: float = 0.0               # max 0.25
    resolution_appropriateness: float = 0.0  # max 0.20
    sla_compliance: float = 0.0              # max 0.10
    policy_compliance: float = 0.0           # max 0.10
    efficiency_bonus: float = 0.0            # max 0.05
    penalty: float = 0.0                     # negative


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized total reward [0,1]")
    breakdown: RewardBreakdown
    rationale: str = ""


# ---------------------------------------------------------------------------
# Episode State
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """Full serializable state snapshot."""
    task_id: str
    step: int
    max_steps: int
    done: bool
    ticket: Ticket
    cumulative_reward: float
    reward_history: List[float]
    action_history: List[Dict[str, Any]]
    available_macros: List[str]
    available_categories: List[str]
    knowledge_base: Dict[str, str]
