from typing import Any
from .base import (
    ManagementDecision,
    BuyDecision,
    AuctionBid,
    JailDecision,
    DebtDecision,
    TradeProposal,
    TradeResponse,
)

class InvalidAgentError(Exception):
    """Raised when an agent violates the game interface (e.g. invalid action)."""

class LLMAPIError(Exception):
    """Raised when an LLM API fails (fatal)."""

def is_raw_kbench_llm(obj: Any) -> bool:
    """Duck-type check for raw kbench.llms object."""
    return hasattr(obj, "prompt") and hasattr(obj, "respond") and not hasattr(obj, "bind")

def validate_agent_shape(agent: Any) -> None:
    required = [
        "bind",
        "choose_management",
        "choose_buy",
        "choose_auction_bid",
        "choose_jail_action",
        "choose_debt_action",
        "choose_trade_response",
    ]
    for method in required:
        if not hasattr(agent, method) or not callable(getattr(agent, method)):
            raise InvalidAgentError(f"Agent {type(agent).__name__} is missing required method '{method}'")

def bind_and_validate_agent(agent_instance: Any, participant, game) -> Any:
    validate_agent_shape(agent_instance)
    bound_agent = agent_instance.bind(participant, game)
    if bound_agent is None:
        bound_agent = agent_instance
    validate_agent_shape(bound_agent)
    if not hasattr(bound_agent, "participant") or bound_agent.participant is not participant:
        raise InvalidAgentError(f"Agent {type(bound_agent).__name__} failed to bind participant properly.")
    if hasattr(bound_agent, "setup") and callable(bound_agent.setup):
        bound_agent.setup()
    return bound_agent

def likely_api_failure(err: Exception) -> bool:
    text = str(err).lower()
    return any(
        term in text
        for term in (
            "api_key",
            "unauthorized",
            "rate limit",
            "quota",
            "connection",
            "timeout",
            "not found",
            "internal server error",
            "bad gateway",
            "googleapi",
            "openaierror",
        )
    )

def coerce_management_decision(obj: Any) -> ManagementDecision:
    if isinstance(obj, ManagementDecision):
        return obj
    if isinstance(obj, dict):
        return ManagementDecision(**obj)
    return ManagementDecision(
        action=getattr(obj, "action"),
        target_property=getattr(obj, "target_property", None),
        trade_proposal=getattr(obj, "trade_proposal", None),
        reason=getattr(obj, "reason", ""),
    )

def coerce_buy_decision(obj: Any) -> BuyDecision:
    if isinstance(obj, BuyDecision):
        return obj
    if isinstance(obj, dict):
        return BuyDecision(**obj)
    return BuyDecision(
        will_buy=getattr(obj, "will_buy"),
        reason=getattr(obj, "reason", ""),
    )

def coerce_auction_bid(obj: Any) -> AuctionBid:
    if isinstance(obj, AuctionBid):
        return obj
    if isinstance(obj, dict):
        return AuctionBid(**obj)
    return AuctionBid(
        bid_amount=getattr(obj, "bid_amount"),
        reason=getattr(obj, "reason", ""),
    )

def coerce_jail_decision(obj: Any) -> JailDecision:
    if isinstance(obj, JailDecision):
        return obj
    if isinstance(obj, dict):
        return JailDecision(**obj)
    return JailDecision(
        action=getattr(obj, "action"),
        reason=getattr(obj, "reason", ""),
    )

def coerce_debt_decision(obj: Any) -> DebtDecision:
    if isinstance(obj, DebtDecision):
        return obj
    if isinstance(obj, dict):
        return DebtDecision(**obj)
    return DebtDecision(
        action=getattr(obj, "action"),
        target_property=getattr(obj, "target_property", None),
        reason=getattr(obj, "reason", ""),
    )

def coerce_trade_response(obj: Any) -> TradeResponse:
    if isinstance(obj, TradeResponse):
        return obj
    if isinstance(obj, dict):
        return TradeResponse(**obj)
    return TradeResponse(
        action=getattr(obj, "action"),
        reason=getattr(obj, "reason", ""),
    )

def validate_decision_against_legal_actions(decision_action: str, legal_actions: list[dict], target: int | None = None) -> None:
    valid_actions = [la["action"] for la in legal_actions]
    if decision_action not in valid_actions:
        raise InvalidAgentError(f"Action '{decision_action}' is not in legal actions: {valid_actions}")
    
    action_def = next(la for la in legal_actions if la["action"] == decision_action)
    if "allowed_targets" in action_def and action_def["allowed_targets"]:
        if target not in action_def["allowed_targets"]:
            raise InvalidAgentError(f"Target '{target}' is not allowed for action '{decision_action}'. Allowed: {action_def['allowed_targets']}")


def coerce_and_validate_decision(obj: Any, phase: str, context) -> Any:
    """Normalize custom-agent output and enforce the same rules as LLM output."""
    try:
        if phase in ("pre_roll", "post_roll"):
            decision = coerce_management_decision(obj)
            validate_decision_against_legal_actions(
                decision.action, context.legal_actions, decision.target_property
            )
            return decision
        if phase == "buy":
            decision = coerce_buy_decision(obj)
            action = "buy" if decision.will_buy else "decline"
            validate_decision_against_legal_actions(action, context.legal_actions)
            if decision.will_buy and context.money < context.landed_space["price"]:
                raise InvalidAgentError("Player cannot afford this property.")
            return decision
        if phase == "auction":
            decision = coerce_auction_bid(obj)
            if decision.bid_amount < 0:
                raise InvalidAgentError("Auction bid cannot be negative.")
            if decision.bid_amount > context.money:
                raise InvalidAgentError("Auction bid cannot exceed the player's cash.")
            return decision
        if phase == "jail_decision":
            decision = coerce_jail_decision(obj)
            validate_decision_against_legal_actions(decision.action, context.legal_actions)
            return decision
        if phase == "debt_resolution":
            decision = coerce_debt_decision(obj)
            validate_decision_against_legal_actions(
                decision.action, context.legal_actions, decision.target_property
            )
            return decision
        if phase == "trade_response":
            decision = coerce_trade_response(obj)
            validate_decision_against_legal_actions(decision.action, context.legal_actions)
            return decision
    except InvalidAgentError:
        raise
    except Exception as exc:
        raise InvalidAgentError(
            f"Agent returned a malformed decision during {phase}: {exc}"
        ) from exc
    raise InvalidAgentError(f"Unsupported decision phase: {phase}")
