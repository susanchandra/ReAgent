"""
ReAgent Success Contract Manager (Milestone 3)

Responsibility:
  - Build a SuccessContract from caller-supplied Postconditions.
  - Freeze it (mark frozen_at timestamp) before any tool execution.
  - The frozen contract may not be mutated after freezing.
  - Provide contract→AssertionRule bridging for the DeterministicVerifier.
"""

from typing import List, Optional
from reagent.models import SuccessContract, Postcondition, get_utc_now
from reagent.verification.verifier import AssertionRule


# Map Postcondition predicates → AssertionRule assertion_types understood by
# DeterministicVerifier.  Predicates with identical names pass through directly.
_PREDICATE_MAP: dict = {
    "STATUS_SUCCESS":       "STATUS_SUCCESS",
    "KEY_EXISTS":           "KEY_EXISTS",
    "VALUE_EQUALS":         "VALUE_EQUALS",
    "FILE_EXISTS":          "FILE_EXISTS",
    "FILE_CONTAINS":        "FILE_CONTAINS",
    "FILE_SIZE_GTE":        "FILE_SIZE_GTE",
    "JSON_VALID":           "JSON_VALID",
    "JSON_PATH_EQUALS":     "JSON_PATH_EQUALS",
    "HTTP_STATUS":          "HTTP_STATUS",
    "NO_ERROR_PAYLOAD":     "NO_ERROR_PAYLOAD",
}


class ContractManager:
    """
    Builds and freezes SuccessContracts.
    Once freeze() is called the returned SuccessContract object is immutable
    (callers must not modify its fields).
    """

    def build_contract(
        self,
        task_id: str,
        postconditions: List[Postcondition],
        provenance: str = "fixture",
    ) -> SuccessContract:
        """
        Build a new contract for a task.
        The contract is not frozen until freeze() is called.
        """
        if not postconditions:
            raise ValueError(
                f"SuccessContract for task '{task_id}' requires at least one postcondition."
            )
        return SuccessContract(
            task_id=task_id,
            postconditions=postconditions,
            provenance=provenance,
        )

    def freeze(self, contract: SuccessContract) -> SuccessContract:
        """
        Return a new SuccessContract with frozen_at set to now.
        The original object is not mutated (Pydantic models are value-like).
        """
        return SuccessContract(
            contract_id=contract.contract_id,
            task_id=contract.task_id,
            postconditions=contract.postconditions,
            provenance=contract.provenance,
            frozen_at=get_utc_now(),
        )

    def to_assertion_rules(self, contract: SuccessContract) -> List[AssertionRule]:
        """
        Convert all postconditions in a frozen contract into AssertionRule objects
        understood by the DeterministicVerifier.
        Predicates not in the supported map are skipped with a warning.
        """
        rules: List[AssertionRule] = []
        for pc in contract.postconditions:
            assertion_type = _PREDICATE_MAP.get(pc.predicate)
            if assertion_type is None:
                import logging
                logging.getLogger(__name__).warning(
                    "Unsupported predicate '%s' in contract %s — skipping.",
                    pc.predicate, contract.contract_id,
                )
                continue
            rules.append(
                AssertionRule(
                    assertion_type=assertion_type,
                    key_path=pc.target,
                    expected_value=pc.expected,
                    description=pc.description,
                )
            )
        return rules
