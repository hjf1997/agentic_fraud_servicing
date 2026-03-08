"""Shared simulation infrastructure: Scenario dataclass, CCP agent, and turn generators.

Provides the common framework for all simulation scenarios. Each scenario module
(e.g., scenario_scam_techvault.py, scenario_doordash_fraud.py) creates a Scenario
instance with scenario-specific evidence, prompts, and system events.
"""

from dataclasses import dataclass, field
from typing import Callable

from agents import Agent, RunConfig, Runner

from agentic_fraud_servicing.gateway.tool_gateway import ToolGateway
from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.providers.base import ModelProvider

# ---------------------------------------------------------------------------
# Scenario dataclass — encapsulates all scenario-specific configuration
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A simulation scenario with all data needed to run an E2E simulation.

    Attributes:
        name: Short identifier (e.g., 'scam_techvault', 'doordash_fraud').
        title: Display title for the simulation banner.
        description: Multi-line description printed at the start.
        case_id: Unique case identifier for this simulation run.
        call_id: Unique call identifier for this simulation run.
        cm_system_prompt: System prompt for the cardmember simulator agent.
        system_event_auth: Text for the identity verification SYSTEM event.
        system_event_evidence: Text for the evidence retrieval SYSTEM event.
        max_turns: Maximum conversation turns (CCP + CM + SYSTEM combined).
        seed_evidence_fn: Function that seeds evidence into the gateway.
        create_case_fn: Function that creates the initial Case.
    """

    name: str
    title: str
    description: str
    case_id: str
    call_id: str
    cm_system_prompt: str
    system_event_auth: str
    system_event_evidence: str
    max_turns: int = 14
    seed_evidence_fn: Callable[[ToolGateway, str], None] = field(repr=False, default=None)  # type: ignore[assignment]
    create_case_fn: Callable[[ToolGateway, str, str], Case] = field(repr=False, default=None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CCP simulator agent (shared across all scenarios)
# ---------------------------------------------------------------------------

CCP_SYSTEM_PROMPT = """\
You are role-playing as Sarah, an American Express Contact Center Professional \
(CCP) handling an incoming fraud dispute call. You are professional, empathetic, \
and thorough.

Procedure:
- Greet the caller warmly and verify their identity (ask for card number, name).
- Ask about the disputed transaction: what charge, when, amount, merchant.
- Listen carefully for inconsistencies in the caller's story.
- Follow standard AMEX dispute handling: gather facts, verify claims against \
system records, assess credibility.

Copilot Integration:
- Before each of your turns, you will receive a copilot context note enclosed \
in [COPILOT] tags. This contains hypothesis scores (fraud/dispute/scam \
probabilities), suggested questions, risk flags, and a running summary.
- Incorporate the copilot's suggested questions naturally into the conversation \
— do not read them verbatim or mention the copilot system to the caller.
- If the copilot flags inconsistencies or scam indicators, gently probe those \
areas without being accusatory.
- If impersonation risk is elevated, proceed with additional verification.

Style:
- Professional and empathetic but thorough.
- 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.
- Address the caller by name once identity is established.\
"""

ccp_agent = Agent(name="ccp_simulator", instructions=CCP_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Turn generation functions
# ---------------------------------------------------------------------------


async def generate_cm_turn(
    cm_agent: Agent, conversation_history: str, model_provider: ModelProvider
) -> str:
    """Generate the cardmember's next dialogue turn via LLM.

    Args:
        cm_agent: The scenario-specific CM simulator agent.
        conversation_history: Full conversation so far, formatted as text.
        model_provider: The ModelProvider instance (e.g. BedrockModelProvider).

    Returns:
        The cardmember's response text.
    """
    result = await Runner.run(
        cm_agent,
        input=conversation_history,
        run_config=RunConfig(model_provider=model_provider),
    )
    return result.final_output


async def generate_ccp_turn(
    conversation_history: str, copilot_context: str, model_provider: ModelProvider
) -> str:
    """Generate the CCP's next dialogue turn via LLM, informed by copilot suggestions.

    Args:
        conversation_history: Full conversation so far, formatted as text.
        copilot_context: Formatted copilot suggestions for the CCP to incorporate.
        model_provider: The ModelProvider instance (e.g. BedrockModelProvider).

    Returns:
        The CCP's response text.
    """
    input_text = f"[COPILOT]\n{copilot_context}\n[/COPILOT]\n\n{conversation_history}"
    result = await Runner.run(
        ccp_agent,
        input=input_text,
        run_config=RunConfig(model_provider=model_provider),
    )
    return result.final_output


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, Callable[[], Scenario]] = {}


def register_scenario(name: str, factory: Callable[[], Scenario]) -> None:
    """Register a scenario factory function by name."""
    _SCENARIOS[name] = factory


def get_scenario(name: str) -> Scenario:
    """Get a scenario by name. Raises KeyError if not found."""
    if name not in _SCENARIOS:
        available = ", ".join(sorted(_SCENARIOS.keys()))
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return _SCENARIOS[name]()


def list_scenarios() -> list[str]:
    """Return sorted list of registered scenario names."""
    return sorted(_SCENARIOS.keys())
