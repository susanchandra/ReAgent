"""
ReAgent Deterministic Failure Scenario Definitions
Formal schemas for Scenarios A through F.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from reagent.taxonomy import FailureType, RecoveryStrategyType, AgentStatus
from reagent.verification.verifier import AssertionRule


class ScenarioDefinition(BaseModel):
    scenario_id: str
    name: str
    description: str
    initial_status: AgentStatus = AgentStatus.RUNNING
    tool_name: str
    initial_payload: Dict[str, Any]
    injected_error: Optional[str] = None
    expected_failure_type: FailureType
    expected_strategy: RecoveryStrategyType
    recovery_payload: Optional[Dict[str, Any]] = None
    fallback_tool_name: Optional[str] = None
    assertions: List[AssertionRule] = Field(default_factory=list)
    expected_final_status: AgentStatus = AgentStatus.RUNNING


def get_all_scenarios() -> List[ScenarioDefinition]:
    """Returns the 6 required deterministic failure scenarios."""
    return [
        # Scenario A: Malformed tool input
        ScenarioDefinition(
            scenario_id="scenario_a_malformed_input",
            name="Scenario A: Malformed Tool Input",
            description="Agent provides missing required parameters to allowlisted tool.",
            initial_status=AgentStatus.RUNNING,
            tool_name="data_transform",
            initial_payload={"source_data": {"user_id": 42, "role": "admin"}},  # missing fields_to_extract
            injected_error="ParameterValidationError: Missing required parameter: 'fields_to_extract'",
            expected_failure_type=FailureType.INPUT_CONTEXT,
            expected_strategy=RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            recovery_payload={"source_data": {"user_id": 42, "role": "admin"}, "fields_to_extract": ["user_id"]},
            assertions=[
                AssertionRule(assertion_type="STATUS_SUCCESS"),
                AssertionRule(assertion_type="KEY_EXISTS", key_path="extracted_data"),
            ],
            expected_final_status=AgentStatus.RUNNING,
        ),

        # Scenario B: Transient external / tool failure
        ScenarioDefinition(
            scenario_id="scenario_b_transient_failure",
            name="Scenario B: Transient External Failure",
            description="Remote endpoint returns 502 Bad Gateway / Connection Reset.",
            initial_status=AgentStatus.RUNNING,
            tool_name="data_transform",
            initial_payload={"source_data": {"telemetry": "raw"}, "fields_to_extract": ["telemetry"]},
            injected_error="502 Bad Gateway: Upstream server temporarily unavailable",
            expected_failure_type=FailureType.TRANSIENT_EXTERNAL,
            expected_strategy=RecoveryStrategyType.WAIT_AND_RETRY,
            recovery_payload={"source_data": {"telemetry": "raw"}, "fields_to_extract": ["telemetry"]},
            assertions=[
                AssertionRule(assertion_type="STATUS_SUCCESS"),
            ],
            expected_final_status=AgentStatus.RUNNING,
        ),

        # Scenario C: Unavailable dependency / environment
        ScenarioDefinition(
            scenario_id="scenario_c_unavailable_dependency",
            name="Scenario C: Unavailable Dependency",
            description="Agent attempts to invoke non-allowlisted / unavailable tool; falls back to alternative.",
            initial_status=AgentStatus.RUNNING,
            tool_name="legacy_unregistered_extractor",
            initial_payload={"data": "raw_blob"},
            injected_error="SecurityViolation: Tool 'legacy_unregistered_extractor' is not in the approved allowlist.",
            expected_failure_type=FailureType.ENVIRONMENT_FAILURE,
            expected_strategy=RecoveryStrategyType.FALLBACK_TOOL,
            fallback_tool_name="data_transform",
            recovery_payload={"source_data": {"data": "raw_blob"}, "fields_to_extract": ["data"]},
            assertions=[
                AssertionRule(assertion_type="STATUS_SUCCESS"),
            ],
            expected_final_status=AgentStatus.RUNNING,
        ),

        # Scenario D: Invalid model / tool output
        ScenarioDefinition(
            scenario_id="scenario_d_invalid_model_output",
            name="Scenario D: Invalid Model/Tool Output",
            description="Model output produces unparseable JSON violating schema constraints.",
            initial_status=AgentStatus.RUNNING,
            tool_name="json_validator",
            initial_payload={"json_payload": "{unquoted_key: 123", "required_keys": ["unquoted_key"]},
            injected_error="json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes",
            expected_failure_type=FailureType.MODEL_OUTPUT,
            expected_strategy=RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            recovery_payload={"json_payload": '{"unquoted_key": 123}', "required_keys": ["unquoted_key"]},
            assertions=[
                AssertionRule(assertion_type="STATUS_SUCCESS"),
                AssertionRule(assertion_type="VALUE_EQUALS", key_path="is_valid", expected_value=True),
            ],
            expected_final_status=AgentStatus.RUNNING,
        ),

        # Scenario E: Repeated failure requiring escalation
        ScenarioDefinition(
            scenario_id="scenario_e_repeated_failure_escalation",
            name="Scenario E: Repeated Failure Requiring Escalation",
            description="Repeated authorization failures exhaust retry policy budget and force escalation.",
            initial_status=AgentStatus.RUNNING,
            tool_name="data_transform",
            initial_payload={"source_data": {"secret": True}, "fields_to_extract": ["secret"]},
            injected_error="403 Forbidden: Invalid bearer token permissions",
            expected_failure_type=FailureType.AUTHORIZATION,
            expected_strategy=RecoveryStrategyType.ESCALATE,
            assertions=[],
            expected_final_status=AgentStatus.ESCALATED,
        ),

        # Scenario F: Model claims success but post-condition verification fails
        ScenarioDefinition(
            scenario_id="scenario_f_false_success_verification_failure",
            name="Scenario F: False Success Detection",
            description="Tool/Model returns 200 OK or claims success, but post-condition assertion detects missing file / error payload.",
            initial_status=AgentStatus.RUNNING,
            tool_name="sandboxed_file",
            initial_payload={"action": "write", "relative_path": "reports/summary.txt", "content": "Sample Report"},
            injected_error=None,
            expected_failure_type=FailureType.TOOL_FAILURE,
            expected_strategy=RecoveryStrategyType.RETRY_MODIFIED_INPUT,
            recovery_payload={"action": "write", "relative_path": "reports/missing_target.pdf", "content": "Recovered PDF content"},
            assertions=[
                AssertionRule(assertion_type="STATUS_SUCCESS"),
                AssertionRule(assertion_type="FILE_EXISTS", expected_value="D:/Jarvis/workspace/sandbox/reports/missing_target.pdf", description="Must verify target file exists on disk"),
            ],
            expected_final_status=AgentStatus.RUNNING,
        ),
    ]
