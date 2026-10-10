"""
ReAgent Taxonomy Definitions
Defines Failure Types, Recovery Strategies, and Policy Rules.
"""

from enum import Enum


class FailureType(str, Enum):
    AGENT_REASONING = "AGENT_REASONING"
    TOOL_FAILURE = "TOOL_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    INPUT_CONTEXT = "INPUT_CONTEXT"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    TIMEOUT = "TIMEOUT"
    AUTHORIZATION = "AUTHORIZATION"
    DEPENDENCY = "DEPENDENCY"
    TRANSIENT_EXTERNAL = "TRANSIENT_EXTERNAL"
    UNKNOWN = "UNKNOWN"


class RecoveryStrategyType(str, Enum):
    RETRY = "RETRY"                             # alias used for pre-execution policy check
    RETRY_SAME = "RETRY_SAME"
    RETRY_MODIFIED_INPUT = "RETRY_MODIFIED_INPUT"
    RESELECT_TOOL = "RESELECT_TOOL"
    REPAIR_CONTEXT = "REPAIR_CONTEXT"
    FALLBACK_TOOL = "FALLBACK_TOOL"
    WAIT_AND_RETRY = "WAIT_AND_RETRY"
    ESCALATE = "ESCALATE"
    ABORT = "ABORT"


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"                       # all postconditions passed
    ESCALATED = "ESCALATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
