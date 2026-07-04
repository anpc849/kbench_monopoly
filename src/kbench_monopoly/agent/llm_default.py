import time
from typing import Any
from .base import (
    BaseAgent, AgentContext,
    ManagementDecision, BuyDecision, AuctionBid,
    JailDecision, DebtDecision, TradeResponse,
)
from .validation import (
    likely_api_failure, LLMAPIError, InvalidAgentError,
    coerce_management_decision, coerce_buy_decision, coerce_auction_bid,
    coerce_jail_decision, coerce_debt_decision, coerce_trade_response,
    validate_decision_against_legal_actions
)

BASE_RULES_TEXT = """You are playing Monopoly. The game is deterministic except for dice rolls and card draws.
Win condition: Be the last player standing.
Houses and Hotels: Must own all properties in a color group to build. Must build evenly.
Mortgages: Unimproved properties can be mortgaged for 50% of price. Unmortgaging costs mortgage value + 10%. Mortgaged properties collect no rent.
Auctions: If a property is landed on and declined, every living player submits exactly one private sealed bid. Highest bid wins; bid 0 means no bid.
Jail: Roll doubles to get out, pay $50, or explicitly choose a held Community Chest or Chance jail card. 3rd consecutive doubles sends you to jail.
"""

class DefaultLLMAgent(BaseAgent):
    """Wraps raw kbench LLM objects to provide structured Monopoly decisions."""

    def __init__(
        self,
        llm: Any,
        max_retries: int = 5,
        sleep_seconds: float = 1.0,
        record_prompts: bool = True,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.record_prompts = record_prompts
        self.chat_name = f"monopoly_agent_{id(self)}"
        self.decision_log: list[dict[str, Any]] = []
        self._decision_sequence = 0

    def model_name(self) -> str:
        for attr in ("model", "name", "id"):
            val = getattr(self.llm, attr, None)
            if val: return str(val)
        return "UnknownLLM"

    def _sleep_after_llm_call(self):
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _rules_text(self) -> str:
        game = getattr(self, "game", None)
        trading_enabled = bool(
            game is not None and getattr(game.config, "enable_trading", False)
        )
        trading_rule = (
            "Trading: Trading is enabled. You may propose a bounded trade only when "
            "'propose_trade' appears in Legal Actions."
            if trading_enabled
            else "Trading: Trading is disabled for this game."
        )
        return f"{BASE_RULES_TEXT}{trading_rule}\n"

    def _custom_prompt_text(self) -> str:
        participant = getattr(self, "participant", {}) or {}
        custom_prompt = ""
        if isinstance(participant, dict):
            custom_prompt = str(participant.get("custom_prompt", "") or "").strip()
        if not custom_prompt:
            return ""
        return (
            "\n\nCustom player prompt for this seat:\n"
            f"{custom_prompt}\n"
            "Use it for strategy and voice, but never override rules, legal actions, "
            "or the requested JSON schema."
        )

    def _prompt_prefix(self, context: AgentContext, prompt_content: str) -> str:
        """Stable prompt prefix kept ahead of per-turn state for prompt caching."""
        return (
            f"{self._rules_text()}\n"
            f"{context.board_reference_text()}\n\n"
            f"--- Decision Instruction ---\n"
            f"{prompt_content}"
            f"{self._custom_prompt_text()}\n\n"
            f"--- Dynamic Decision Context ---\n"
        )

    def _invoke_llm(self, context: AgentContext, prompt_content: str, schema_class: type, coerce_func: callable, validate_func: callable = None) -> Any:
        self._decision_sequence += 1
        decision_sequence = self._decision_sequence
        error_hint = ""
        for attempt in range(1, self.max_retries + 1):
            full_prompt = (
                f"{self._prompt_prefix(context, prompt_content)}"
                f"{context.to_text(include_board_reference=False)}"
                f"{error_hint}"
            )
            response = None
            try:
                try:
                    from kaggle_benchmarks import chats
                except ImportError:
                    response = self.llm.prompt(full_prompt, schema=schema_class)
                else:
                    chat_name = (
                        f"{self.chat_name}-{context.player_name}-{context.phase}-"
                        f"{context.turn_number}-decision-{decision_sequence}-attempt-{attempt}"
                    )
                    with chats.new(name=chat_name, orphan=False):
                        response = self.llm.prompt(full_prompt, schema=schema_class)
                
                self._sleep_after_llm_call()
                decision = coerce_func(response)

                reason = getattr(decision, "reason", None)
                if not isinstance(reason, str) or not reason.strip():
                    raise InvalidAgentError(
                        "LLM decisions must include a non-empty reason for analysis."
                    )
                
                if validate_func:
                    validate_func(decision)

                self.decision_log.append({
                    "round_id": context.round_number,
                    "turn_id": context.turn_number,
                    "phase": context.phase,
                    "attempt": attempt,
                    "prompt": full_prompt if self.record_prompts else None,
                    "requested": self._safe_payload(response),
                    "decision": self._safe_payload(decision),
                    "reason": getattr(decision, "reason", ""),
                    "error": None,
                })
                return decision

            except LLMAPIError:
                raise
            except Exception as e:
                if likely_api_failure(e):
                    self._record_failure(context, attempt, response, e, full_prompt)
                    raise LLMAPIError(f"LLM API failure on attempt {attempt}: {e}") from e

                self._record_failure(context, attempt, response, e, full_prompt)
                # Validation or logic error
                error_hint = f"\n\nERROR ON PREVIOUS ATTEMPT:\n{e}\nPlease correct your output and select exactly from the legal actions."
                if attempt == self.max_retries:
                    raise InvalidAgentError(f"Agent failed to produce valid decision after {self.max_retries} attempts. Last error: {e}") from e

    def _record_failure(
        self,
        context: AgentContext,
        attempt: int,
        response: Any,
        error: Exception,
        prompt: str,
    ) -> None:
        self.decision_log.append({
            "round_id": context.round_number,
            "turn_id": context.turn_number,
            "phase": context.phase,
            "attempt": attempt,
            "prompt": prompt if self.record_prompts else None,
            "requested": self._safe_payload(response),
            "decision": None,
            "reason": "",
            "error": str(error),
        })

    @staticmethod
    def _safe_payload(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return repr(value)

    def get_log(self) -> dict[str, Any] | None:
        return {"decisions": list(self.decision_log)}

    def choose_management(self, context: AgentContext) -> ManagementDecision:
        trading_available = any(
            action["action"] == "propose_trade" for action in context.legal_actions
        )
        trading_clause = (
            " You may also initiate one bounded trade."
            if trading_available
            else " Trading is not available in this phase."
        )
        trade_output_clause = (
            " For propose_trade, provide one complete trade_proposal."
            if trading_available
            else ""
        )
        prompt = (
            "You are in the management phase. You can use only the actions shown "
            "in Legal Actions, then end management to proceed."
            f"{trading_clause}\n"
            "Choose exactly one action from the Legal Actions list. "
            "If your action requires a target property, provide its integer index."
            f"{trade_output_clause}"
        )
        def validate(decision: ManagementDecision):
            validate_decision_against_legal_actions(decision.action, context.legal_actions, decision.target_property)
        return self._invoke_llm(context, prompt, ManagementDecision, coerce_management_decision, validate)

    def choose_buy(self, context: AgentContext) -> BuyDecision:
        prompt = (
            "You have landed on an unowned property. Will you buy it for the listed price? "
            "If you decline, it will go to auction."
        )
        def validate(decision: BuyDecision):
            if decision.will_buy and context.money < context.landed_space["price"]:
                raise InvalidAgentError("You do not have enough money to buy this property.")
        return self._invoke_llm(context, prompt, BuyDecision, coerce_buy_decision, validate)

    def choose_auction_bid(self, context: AgentContext) -> AuctionBid:
        prompt = (
            "A property is being auctioned by sealed bid. Submit exactly one blind bid amount now. "
            "You will not see other players' bids and you will not get another chance to change or raise your bid. "
            "The highest bidder wins and pays their bid. Bid 0 if you are not interested."
        )
        def validate(decision: AuctionBid):
            if decision.bid_amount < 0:
                raise InvalidAgentError("Bid cannot be negative.")
            if decision.bid_amount > context.money:
                raise InvalidAgentError(f"Cannot bid more than your current money (${context.money}).")
        return self._invoke_llm(context, prompt, AuctionBid, coerce_auction_bid, validate)

    def choose_jail_action(self, context: AgentContext) -> JailDecision:
        prompt = (
            "You are in jail. You must decide how to try to get out.\n"
            "Choose exactly one action from the Legal Actions list ('roll', "
            "'pay_fine', 'use_cc_card', or 'use_chance_card')."
        )
        def validate(decision: JailDecision):
            validate_decision_against_legal_actions(decision.action, context.legal_actions)
        return self._invoke_llm(context, prompt, JailDecision, coerce_jail_decision, validate)

    def choose_debt_action(self, context: AgentContext) -> DebtDecision:
        prompt = (
            "You are in debt! You must raise funds by selling houses or mortgaging properties. "
            "If you have no assets left, you must declare bankruptcy.\n"
            "Choose exactly one action from the Legal Actions list ('sell_house', 'mortgage', 'declare_bankruptcy')."
        )
        def validate(decision: DebtDecision):
            validate_decision_against_legal_actions(decision.action, context.legal_actions, decision.target_property)
        return self._invoke_llm(context, prompt, DebtDecision, coerce_debt_decision, validate)

    def choose_trade_response(self, context: AgentContext) -> TradeResponse:
        prompt = (
            "Another player has proposed a trade. Choose exactly one action from "
            "the Legal Actions list ('accept' or 'reject')."
        )
        def validate(decision: TradeResponse):
            validate_decision_against_legal_actions(decision.action, context.legal_actions)
        return self._invoke_llm(context, prompt, TradeResponse, coerce_trade_response, validate)
