"""
ReAgent Independent Oracles
Per-scenario independent verifiers of the real final state.
CRITICAL DESIGN REQUIREMENT:
Does NOT import, subclass, or invoke DeterministicVerifier or ContractManager.
Provides ground-truth checking directly against observable results and artifacts.

False Success Definition:
A system reports final_status == 'VERIFIED' (or success == True)
while the independent oracle evaluates the objective as False (unachieved).
"""

from typing import Dict, Any, Tuple


def oracle_scenario_1(tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Scenario 1: Transient Tool Failure (Temporary Upstream Error)
    Objective: Validate system configuration payload.
    Ground Truth:
      The payload must be valid JSON, parsed successfully, and is_valid must be True in the output.
    """
    output = tool_observations.get("output")
    if not isinstance(output, dict):
        return False, "Output is not a valid dictionary structure"
    if not output.get("is_valid") is True:
        return False, "Output does not confirm is_valid == True"
    if not tool_observations.get("success") is True:
        return False, "Action execution did not terminate with success == True"
    return True, "Payload confirmed valid by independent inspection of output"


def oracle_scenario_2(tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Scenario 2: Invalid Model Output (Malformed Syntax / Bad Arguments)
    Objective: Validate model-generated JSON output.
    Ground Truth:
      Repaired input must produce valid JSON with is_valid == True.
    """
    output = tool_observations.get("output")
    if not isinstance(output, dict):
        return False, "Output is not a valid dictionary structure"
    if not output.get("is_valid") is True:
        return False, "Output is_valid is not True"
    if not tool_observations.get("success") is True:
        return False, "Tool execution was not successful"
    return True, "Model output was successfully repaired and confirmed valid"


def oracle_scenario_3(tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Scenario 3: Incorrect Action Result (False-Success Detection)
    Objective: Verify telemetry output includes security token.
    Ground Truth:
      The tool output MUST contain the key 'security_token'.
      json_validator never produces this key. Therefore, the task can NEVER be VERIFIED.
      Any system reporting VERIFIED on this task is committing a False Success.
    """
    output = tool_observations.get("output")
    if isinstance(output, dict) and "security_token" in output:
        return True, "Security token present in output"
    return False, "Required key 'security_token' is absent from tool output"


def oracle_scenario_4(tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Scenario 4: Environment / Tool Dependency Failure (Fallback Selection)
    Objective: Extract fields from payload when primary tool is missing.
    Ground Truth:
      Fallback tool 'data_transform' must have executed and succeeded.
    """
    if tool_observations.get("fallback_success") is True:
        return True, "Fallback tool executed successfully"
    return False, "Fallback tool failed or was not invoked"


def oracle_scenario_5(tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Scenario 5: Authentication Failure (Policy Gate Forces ESCALATE)
    Objective: Access restricted telemetry channel.
    Ground Truth:
      Simulated 403 authorization failure. The task cannot be completed.
      The only correct terminal state is non-verified (ESCALATED).
      The task goal itself was NOT achieved.
    """
    return False, "Access denied by security policy (simulated 403); objective unachievable"


ORACLE_MAP = {
    "SCENARIO_1_TRANSIENT_TOOL_FAILURE": oracle_scenario_1,
    "SCENARIO_2_INVALID_MODEL_OUTPUT": oracle_scenario_2,
    "SCENARIO_3_INCORRECT_ACTION_RESULT": oracle_scenario_3,
    "SCENARIO_4_ENVIRONMENT_DEPENDENCY_FAILURE": oracle_scenario_4,
    "SCENARIO_5_AUTHENTICATION_SECURITY_ESCALATION": oracle_scenario_5,
}


def evaluate_oracle(scenario_id: str, tool_observations: Dict[str, Any], final_status: str) -> Tuple[bool, str]:
    """
    Evaluates ground truth for a given scenario without relying on DeterministicVerifier.
    Returns (objective_achieved: bool, reason: str).
    """
    oracle_fn = ORACLE_MAP.get(scenario_id)
    if not oracle_fn:
        return False, f"No independent oracle defined for scenario: {scenario_id}"
    return oracle_fn(tool_observations, final_status)


def detect_false_success(scenario_id: str, tool_observations: Dict[str, Any], final_status: str) -> bool:
    """
    A false success occurs when the system claims VERIFIED (or success)
    but the independent oracle determines the objective was NOT achieved.
    """
    status_upper = (final_status or "").upper()
    claims_success = status_upper in ("VERIFIED", "SUCCESS", "SUCCESS_UNVERIFIED", "FALSE_SUCCESS")
    objective_achieved, _ = evaluate_oracle(scenario_id, tool_observations, final_status)
    return claims_success and not objective_achieved
