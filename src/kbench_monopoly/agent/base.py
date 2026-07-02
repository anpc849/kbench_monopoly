from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from pydantic import BaseModel, Field, StrictBool, StrictInt

class TradeProposal(BaseModel):
    """A proposed trade between two players."""
    target_player: str
    properties_offered: list[StrictInt] = Field(default_factory=list)
    properties_requested: list[StrictInt] = Field(default_factory=list)
    money_offered: StrictInt = 0
    money_requested: StrictInt = 0
    offer_cc_jail_card: StrictBool = False
    offer_chance_jail_card: StrictBool = False
    request_cc_jail_card: StrictBool = False
    request_chance_jail_card: StrictBool = False


class TradeResponse(BaseModel):
    """Agent response to an incoming trade proposal."""
    action: str
    reason: str = ""


class ManagementDecision(BaseModel):
    """Agent decision during pre-roll or post-roll management phase."""
    action: str
    target_property: StrictInt | None = None
    trade_proposal: TradeProposal | None = None
    reason: str = ""


class BuyDecision(BaseModel):
    """Agent decision when landing on unowned property."""
    will_buy: StrictBool
    reason: str = ""


class AuctionBid(BaseModel):
    """Agent decision during a sealed-bid auction."""
    bid_amount: StrictInt
    reason: str = ""


class JailDecision(BaseModel):
    """Agent decision when in jail at start of turn."""
    action: str
    reason: str = ""


class DebtDecision(BaseModel):
    """Agent decision when in debt and must raise funds."""
    action: str
    target_property: StrictInt | None = None
    reason: str = ""


@dataclass
class AgentContext:
    """Private, structured game context for one player at one decision point."""
    player_name: str
    phase: str
    turn_number: int
    round_number: int

    # Player's own state (always visible)
    position: int
    position_name: str
    money: int
    properties_owned: list[dict[str, Any]]
    in_jail: bool
    has_cc_jail_card: bool
    has_chance_jail_card: bool

    # Board state (public information)
    all_players: list[dict[str, Any]]
    property_ownership: list[dict[str, Any]]

    # Phase-specific context
    legal_actions: list[dict[str, Any]] = field(default_factory=list)
    dice_result: tuple[int, int] | None = None
    landed_space: dict[str, Any] | None = None
    auction_state: dict[str, Any] | None = None
    trade_proposal: TradeProposal | None = None
    debt_amount: int = 0
    creditor_name: str = ""

    # History
    public_history: list[dict[str, Any]] = field(default_factory=list)
    private_decision_history: list[dict[str, Any]] = field(default_factory=list)
    extra_hint: str = ""
    round_result: dict[str, Any] | None = None

    def legal_actions_text(self) -> str:
        if not self.legal_actions:
            return "None"
        parts = []
        for action in self.legal_actions:
            desc = f"- action: '{action['action']}'"
            if 'allowed_targets' in action and action['allowed_targets']:
                desc += f" (allowed targets: {action['allowed_targets']})"
            if 'description' in action:
                desc += f" -> {action['description']}"
            parts.append(desc)
        return "\n".join(parts)

    def board_summary_text(self) -> str:
        parts = ["--- Players ---"]
        for p in self.all_players:
            status = []
            if not p['alive']:
                status.append("BANKRUPT")
            else:
                if p['in_jail']:
                    status.append("IN JAIL")
                status.append(f"Money: ${p['money']}")
                status.append(f"Position: {p['position']} ({p['position_name']})")
                status.append(f"Properties Owned: {p['property_count']}")
            parts.append(f"- {p['name']}: {', '.join(status)}")

        parts.append("\n--- Property Ownership ---")
        if not self.property_ownership:
            parts.append("No properties owned yet.")
        else:
            for prop in self.property_ownership:
                state = []
                if prop['mortgaged']:
                    state.append("MORTGAGED")
                if prop['houses'] == 5:
                    state.append("1 Hotel")
                elif prop['houses'] > 0:
                    state.append(f"{prop['houses']} Houses")
                state_str = f" [{', '.join(state)}]" if state else ""
                parts.append(f"- {prop['name']} (Group {prop['group']}): Owned by {prop['owner_name']}{state_str}")
        return "\n".join(parts)

    def public_history_text(self) -> str:
        if not self.public_history:
            return "- No public actions yet."
        rows = []
        for item in self.public_history[-20:]:  # Last 20 events
            kind = item.get("type", "event")
            if kind == "roll":
                rows.append(f"- {item['player']} rolled {item['dice']} and landed on {item['space_name']}.")
            elif kind == "buy":
                rows.append(f"- {item['player']} bought {item['space_name']} for ${item['price']}.")
            elif kind == "rent":
                rows.append(f"- {item['player']} paid ${item['amount']} rent to {item['creditor']}.")
            elif kind == "tax":
                rows.append(f"- {item['player']} paid ${item['amount']} in tax.")
            elif kind == "auction_start":
                rows.append(f"- Auction started for {item['space_name']}.")
            elif kind == "auction_win":
                rows.append(f"- {item['player']} won auction for {item['space_name']} with bid ${item['amount']}.")
            elif kind == "auction_nobid":
                rows.append(f"- Nobody bid on {item['space_name']}.")
            elif kind == "build":
                rows.append(f"- {item['player']} built a house/hotel on {item['space_name']}.")
            elif kind == "sell_house":
                rows.append(f"- {item['player']} sold a house/hotel from {item['space_name']}.")
            elif kind == "mortgage":
                rows.append(f"- {item['player']} mortgaged {item['space_name']}.")
            elif kind == "unmortgage":
                rows.append(f"- {item['player']} unmortgaged {item['space_name']}.")
            elif kind == "card":
                rows.append(f"- {item['player']} drew {item['deck']} card: '{item['text']}'.")
            elif kind == "jail":
                rows.append(f"- {item['player']} was sent to jail!")
            elif kind == "jail_out":
                rows.append(f"- {item['player']} got out of jail (Method: {item['method']}).")
            elif kind == "bankrupt":
                rows.append(f"- {item['player']} went bankrupt to {item['creditor']}!")
            else:
                rows.append(f"- {item}")
        return "\n".join(rows)

    def private_decision_history_text(self) -> str:
        if not self.private_decision_history:
            return "- No prior private decisions."
        rows = []
        for item in self.private_decision_history[-10:]:
            phase = item.get("phase", "decision")
            decision = item.get("decision", {})
            if isinstance(decision, dict):
                decision = {
                    key: value for key, value in decision.items() if key != "reason"
                }
            action_desc = str(decision)
            rows.append(
                f"- Round {item.get('round_id')}, Turn {item.get('turn_id')}, Phase {phase}: "
                f"Action: {action_desc} | Reason: {item.get('reason', '')}"
            )
        return "\n".join(rows)
        
    def specific_context_text(self) -> str:
        parts = []
        if self.phase == "buy":
            ls = self.landed_space
            parts.append(f"Space: {ls['name']} | Price: ${ls['price']}")
            if ls.get('space_type') == 'property':
                parts.append(f"  Rent: ${ls['rent']} | 1H: ${ls['rent_1h']} | 2H: ${ls['rent_2h']} | 3H: ${ls['rent_3h']} | 4H: ${ls['rent_4h']} | Hotel: ${ls['rent_hotel']}")
                parts.append(f"  House Cost: ${ls['house_price']} | Mortgage Value: ${ls['mortgage_value']}")
            elif ls.get('space_type') == 'railroad':
                parts.append(f"  Rent: $25 (1), $50 (2), $100 (3), $200 (4) | Mortgage Value: ${ls['mortgage_value']}")
            elif ls.get('space_type') == 'utility':
                parts.append(f"  Rent: 4x dice (1), 10x dice (2) | Mortgage Value: ${ls['mortgage_value']}")
                
        elif self.phase == "auction":
            ls = self.auction_state
            parts.append(f"Auction for: {ls['space_name']} | Normal Price: ${ls['price']}")
            parts.append("  Auction format: one private sealed bid only. You will not see other bids and will not get a second chance to raise.")
            if ls.get('space_type') == 'property':
                parts.append(f"  Rent: ${ls['rent']} | 1H: ${ls['rent_1h']} | 2H: ${ls['rent_2h']} | 3H: ${ls['rent_3h']} | 4H: ${ls['rent_4h']} | Hotel: ${ls['rent_hotel']}")
                parts.append(f"  House Cost: ${ls['house_price']} | Mortgage Value: ${ls['mortgage_value']}")
            elif ls.get('space_type') == 'railroad':
                parts.append(f"  Rent: $25 (1), $50 (2), $100 (3), $200 (4) | Mortgage Value: ${ls['mortgage_value']}")
            elif ls.get('space_type') == 'utility':
                parts.append(f"  Rent: 4x dice (1), 10x dice (2) | Mortgage Value: ${ls['mortgage_value']}")
        elif self.phase == "debt_resolution":
            parts.append(f"You owe ${self.debt_amount} to {self.creditor_name}. You must raise funds or declare bankruptcy.")
        elif self.phase == "trade_response" and self.trade_proposal:
            tp = self.trade_proposal
            parts.append("Incoming trade proposal:")
            parts.append(f"  They offer: Properties {tp.properties_offered}, ${tp.money_offered}")
            parts.append(f"  They request: Properties {tp.properties_requested}, ${tp.money_requested}")
            parts.append(
                "  Jail cards: "
                f"offer CC={tp.offer_cc_jail_card}, offer Chance={tp.offer_chance_jail_card}, "
                f"request CC={tp.request_cc_jail_card}, request Chance={tp.request_chance_jail_card}"
            )
            
        if self.extra_hint:
            parts.append(f"\nEXTRA HINT: {self.extra_hint}")
            
        return "\n".join(parts)

    def board_reference_text(self) -> str:
        from kbench_monopoly.board_data import BOARD, COLOR_NAMES
        parts = ["--- Board Reference ---"]
        groups = {}
        for p in BOARD:
            if p.group >= 1:
                if p.group not in groups:
                    groups[p.group] = []
                groups[p.group].append(f"{p.index}: {p.name}")
        
        for g_id, g_props in sorted(groups.items()):
            g_name = "Railroad" if g_id == 1 else "Utility" if g_id == 2 else COLOR_NAMES.get(g_id, "Unknown")
            parts.append(f"- Group {g_id} ({g_name}): {', '.join(g_props)}")
        return "\n".join(parts)

    def to_text(self) -> str:
        return (
            f"Player: {self.player_name}\n"
            f"Phase: {self.phase}\n"
            f"Round: {self.round_number} | Turn: {self.turn_number}\n\n"
            f"--- Your Status ---\n"
            f"Position: {self.position} ({self.position_name})\n"
            f"Money: ${self.money}\n"
            f"In Jail: {self.in_jail}\n"
            f"Jail Cards: Community Chest={self.has_cc_jail_card}, Chance={self.has_chance_jail_card}\n"
            f"Properties Owned (indices): {[p['index'] for p in self.properties_owned]}\n\n"
            f"{self.board_summary_text()}\n\n"
            f"--- Phase Context ---\n"
            f"{self.specific_context_text()}\n\n"
            f"--- Legal Actions ---\n"
            f"{self.legal_actions_text()}\n\n"
            f"--- Public History ---\n{self.public_history_text()}\n\n"
            f"--- Your Private Decision History ---\n"
            f"{self.private_decision_history_text()}\n\n"
            f"{self.board_reference_text()}"
        )


class BaseAgent:
    """Base class for Monopoly-compatible agents."""

    def bind(self, participant, game):
        self.participant = participant
        self.game = game
        return self

    def setup(self) -> None:
        return None

    def choose_action(self, context: AgentContext):
        """Route to appropriate decision method based on context.phase."""
        if context.phase in ("pre_roll", "post_roll"):
            return self.choose_management(context)
        elif context.phase == "jail_decision":
            return self.choose_jail_action(context)
        elif context.phase == "buy":
            return self.choose_buy(context)
        elif context.phase == "auction":
            return self.choose_auction_bid(context)
        elif context.phase == "debt_resolution":
            return self.choose_debt_action(context)
        elif context.phase == "trade_response":
            return self.choose_trade_response(context)
        else:
            raise ValueError(f"Unknown phase: {context.phase}")

    def choose_management(self, context: AgentContext) -> ManagementDecision:
        raise NotImplementedError

    def choose_buy(self, context: AgentContext) -> BuyDecision:
        raise NotImplementedError

    def choose_auction_bid(self, context: AgentContext) -> AuctionBid:
        raise NotImplementedError

    def choose_jail_action(self, context: AgentContext) -> JailDecision:
        raise NotImplementedError

    def choose_debt_action(self, context: AgentContext) -> DebtDecision:
        raise NotImplementedError

    def choose_trade_response(self, context: AgentContext) -> TradeResponse:
        raise NotImplementedError

    def get_log(self) -> dict[str, Any] | None:
        return None

    def agent_name(self) -> str:
        return type(self).__name__

    def model_name(self) -> str:
        return ""
