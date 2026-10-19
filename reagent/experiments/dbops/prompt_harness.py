"""
prompt_harness.py — Deterministic Offline Prompt & Schema Harness for DBOPS-M1.

Constructs prompts deterministically for the 12 pre-registered DBOPS tasks,
computes stable SHA-256 hashes, records schema version metadata, and estimates
request tokens using ReAgent's token estimation heuristics.

Zero network access. Zero model provider calls. Zero live API access.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from reagent.experiments.dbops.manifest_dbops_v1 import (
    ALL_TASK_IDS,
    MANIFEST_VERSION,
)
from reagent.experiments.dbops.tasks import get_task, DbopsTask
from reagent.policy import estimate_prompt_tokens
from reagent.providers.nebius import (
    STRUCTURED_DECISION_SCHEMA,
    ALLOWED_STRUCTURED_DECISION_KEYS,
)
from reagent.taxonomy import FailureType, RecoveryStrategyType


@dataclass(frozen=True)
class TaskPromptBundle:
    """
    Immutable representation of a generated task prompt and its provenance.
    """
    task_id: str
    manifest_version: str
    system_prompt: str
    user_prompt: str
    full_prompt_text: str
    prompt_hash: str
    estimated_tokens: int
    token_estimator_type: str  # 'exact_bpe' or 'conservative_char_ratio_proxy'
    token_limit_safe: bool     # True if < 1500 tokens


class DbopsPromptHarness:
    """
    Harness for constructing deterministic prompts for all 12 DBOPS tasks.
    """
    DECISION_SCHEMA_VERSION = "structured_decision_v1"
    MAX_PROMPT_TOKEN_CEILING = 1500

    def __init__(self, available_tools: Optional[List[str]] = None):
        self.available_tools = available_tools or ["dbops_database"]

    def build_system_prompt(self) -> str:
        """
        Build the canonical system prompt used by ReAgent decision engine.
        Matches NebiusNemotronProvider.make_structured_decision system prompt.
        """
        return (
            "You are ReAgent's Decision Engine. "
            "You produce a SINGLE structured JSON object describing the next action. "
            "Output ONLY valid JSON matching exactly the provided schema. "
            "Do NOT include explanation, markdown, or chain-of-thought in your output. "
            f"Available tools: {self.available_tools}. "
            f"Failure taxonomy: {[t.value for t in FailureType]}. "
            f"Recovery strategies: {[s.value for s in RecoveryStrategyType]}."
        )

    def build_user_prompt(self, task: DbopsTask, context_summary: Optional[str] = None) -> str:
        """
        Build user prompt from task's natural language human description.
        Does not hardcode gold outcomes or internal fault states.
        """
        return (
            f"Task: {task.human_description}\n"
            + (f"Context: {context_summary}\n" if context_summary else "")
            + f"Schema to fill:\n{STRUCTURED_DECISION_SCHEMA}"
        )

    def construct_task_prompt(
        self,
        task_id: str,
        context_summary: Optional[str] = None,
    ) -> TaskPromptBundle:
        """
        Construct the deterministic prompt bundle for a given task ID.
        """
        task = get_task(task_id)
        sys_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(task, context_summary)
        full_text = f"SYSTEM:\n{sys_prompt}\n\nUSER:\n{user_prompt}"

        # Deterministic SHA-256 hash of the complete request
        prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        # Token estimation: ReAgent conservative character heuristic
        est_tokens = estimate_prompt_tokens(prompt=user_prompt, system_prompt=sys_prompt)

        return TaskPromptBundle(
            task_id=task.task_id,
            manifest_version=MANIFEST_VERSION,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            full_prompt_text=full_text,
            prompt_hash=prompt_hash,
            estimated_tokens=est_tokens,
            token_estimator_type="conservative_char_ratio_proxy",
            token_limit_safe=(est_tokens < self.MAX_PROMPT_TOKEN_CEILING),
        )

    def construct_all_prompts(self) -> Dict[str, TaskPromptBundle]:
        """
        Construct prompt bundles for all 12 pre-registered tasks.
        """
        return {task_id: self.construct_task_prompt(task_id) for task_id in ALL_TASK_IDS}
