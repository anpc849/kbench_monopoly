import random
from dataclasses import dataclass, field
from typing import Any
import copy

from .board_data import (
    BOARD, COMMUNITY_CHEST_CARDS, CHANCE_CARDS, GROUP_MEMBERS,
    GO_POSITION, JAIL_POSITION, FREE_PARKING_POSITION, GO_TO_JAIL_POSITION
)
from .config import GameConfig
from .agent import AgentContext, InvalidAgentError
from .agent.validation import (
    bind_and_validate_agent,
    coerce_and_validate_decision,
    is_raw_kbench_llm,
)
from .agent.llm_default import DefaultLLMAgent

@dataclass
class PlayerState:
    name: str
    agent: Any
    seat: int
    model_id: str
    evaluated: bool
    position: int = 0
    money: int = 1500
    alive: bool = True
    in_jail: bool = False
    jail_turns: int = 0
    has_cc_jail_card: bool = False
    has_chance_jail_card: bool = False
    doubles_count: int = 0

@dataclass
class PropertyState:
    owner: int | None = None
    houses: int = 0
    is_mortgaged: bool = False

class MonopolyGame:
    def __init__(self, config: GameConfig, ui_observer=None):
        self.config = config
        self.ui = ui_observer
        self.rng = random.Random(config.seed)
        
        self.players: list[PlayerState] = []
        for i, pconf in enumerate(config.player_configs):
            agent = pconf["agent"]
            if is_raw_kbench_llm(agent):
                agent = DefaultLLMAgent(
                    agent,
                    max_retries=config.llm_max_attempts,
                    sleep_seconds=config.llm_pause_seconds,
                    record_prompts=config.record_llm_prompts,
                )
            agent = bind_and_validate_agent(agent, pconf, self)
            self.players.append(PlayerState(
                name=pconf["name"],
                agent=agent,
                seat=i,
                model_id=pconf.get("model_id", ""),
                evaluated=pconf.get("evaluated", False),
                money=config.starting_money
            ))
            
        self.properties: dict[int, PropertyState] = {
            s.index: PropertyState() for s in BOARD if s.space_type in ("property", "railroad", "utility")
        }
        
        self.houses_available = config.max_houses
        self.hotels_available = config.max_hotels
        
        self.cc_deck = list(range(len(COMMUNITY_CHEST_CARDS)))
        self.chance_deck = list(range(len(CHANCE_CARDS)))
        self.rng.shuffle(self.cc_deck)
        self.rng.shuffle(self.chance_deck)
        self.cc_index = 0
        self.chance_index = 0
        
        self.round_number = 1
        self.turn_number = 1
        self.turns_completed = 0
        self.winner = None
        self.timeout = False
        self.end_reason: str | None = None
        
        self.public_history: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []
        self.game_log = {
            "schema_version": "monopoly-game-log-v1",
            "seed": config.seed,
            "events": [],
        }
        
    def _log_event(self, event_type: str, **kwargs):
        event = {"type": event_type, "round": self.round_number, "turn": self.turn_number, **kwargs}
        self.public_history.append(event)
        self.game_log["events"].append({
            "event_id": len(self.game_log["events"]) + 1,
            "round": self.round_number,
            "turn": self.turn_number,
            **event,
        })
        if self.ui:
            self.ui.report(f"Event: {event}")
            if callable(getattr(self.ui, "draw_game", None)):
                self.ui.draw_game(self._build_snapshot())

    def _request_decision(self, player: PlayerState, context: AgentContext):
        try:
            raw = player.agent.choose_action(context)
            decision = coerce_and_validate_decision(raw, context.phase, context)
            if (
                context.phase in ("pre_roll", "post_roll")
                and decision.action == "propose_trade"
            ):
                self._validate_trade_proposal(player, decision.trade_proposal)
        except Exception as exc:
            self._record_decision(player, context, None, error=str(exc))
            raise
        self._record_decision(player, context, decision, error=None)
        return decision

    def _record_decision(self, player: PlayerState, context: AgentContext, decision, error: str | None):
        if decision is None:
            payload = None
        elif hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "__dict__"):
            payload = dict(decision.__dict__)
        else:
            payload = repr(decision)
        event = {
            "player": player.name,
            "round_id": self.round_number,
            "turn_id": self.turn_number,
            "phase": context.phase,
            "decision": payload,
            "reason": getattr(decision, "reason", "") if decision is not None else "",
            "error": error,
            "private": True,
            "visible_to": [player.name],
        }
        self.decision_log.append(event)
        self.game_log["events"].append({
            "event_id": len(self.game_log["events"]) + 1,
            "round": self.round_number,
            "turn": self.turn_number,
            "type": "agent_decision",
            **event,
        })

    def _check_stop(self):
        if self.ui and hasattr(self.ui, "check_stop"):
            if self.ui.check_stop():
                raise InterruptedError("Game stopped by UI")
        if self.ui and hasattr(self.ui, "draw_game"):
            self.ui.draw_game(self._build_snapshot())

    def _build_snapshot(self):
        private_logs = {}
        for p in self.players:
            private_logs[p.name] = self._private_decision_history_for(p.name)
            
        return {
            "round": self.round_number,
            "turn": self.turn_number,
            "turns_completed": self.turns_completed,
            "winner": self.winner,
            "timeout": self.timeout,
            "end_reason": self.end_reason,
            "players": [
                {
                    "name": p.name, "money": p.money, "position": p.position,
                    "alive": p.alive, "in_jail": p.in_jail,
                    "model_id": p.model_id, "evaluated": p.evaluated,
                    "has_cc_jail_card": p.has_cc_jail_card,
                    "has_chance_jail_card": p.has_chance_jail_card,
                    "properties_owned": [i for i, prop in self.properties.items() if prop.owner == p.seat],
                    "net_worth": self._calculate_net_worth(p)
                } for p in self.players
            ],
            "properties": {
                idx: {"owner": prop.owner, "houses": prop.houses, "mortgaged": prop.is_mortgaged}
                for idx, prop in self.properties.items()
            },
            "public_history": self.public_history,
            "private_logs": private_logs,
            "decision_log": copy.deepcopy(self.decision_log),
        }

    def start(self):
        self._log_event("game_start", players=[p.name for p in self.players])
        current_player_idx = 0
        
        while True:
            self._check_stop()
            active_alive_players = [p for p in self.players if p.alive]
            if len(active_alive_players) <= 1:
                if active_alive_players:
                    self.winner = active_alive_players[0].name
                    self.end_reason = "last_standing"
                    self._log_event("game_over", winner=self.winner, reason="last_standing")
                else:
                    self.winner = None
                    self.end_reason = "draw"
                    self._log_event("game_over", winner=None, reason="draw")
                break
                
            if (
                self.round_number > self.config.max_rounds
                or self.turns_completed >= self.config.max_turns
            ):
                self.timeout = True
                self.winner = None
                self.end_reason = (
                    "max_rounds"
                    if self.round_number > self.config.max_rounds
                    else "max_turns"
                )
                standings = {
                    p.name: self._calculate_net_worth(p) for p in active_alive_players
                }
                self._log_event(
                    "game_over",
                    winner=None,
                    reason="timeout",
                    limit_reached=self.end_reason,
                    net_worth=standings,
                )
                break

            current_player = self.players[current_player_idx]
            if current_player.alive:
                self._play_turn(current_player)
                self.turns_completed += 1
                self.turn_number += 1
                
                # Check if we should advance to next player (no doubles)
                if current_player.doubles_count == 0 or not current_player.alive or current_player.in_jail:
                    current_player.doubles_count = 0
                    current_player_idx = (current_player_idx + 1) % len(self.players)
                    if current_player_idx == 0:
                        self.round_number += 1
            else:
                current_player_idx = (current_player_idx + 1) % len(self.players)
                if current_player_idx == 0:
                    self.round_number += 1

    def _play_turn(self, player: PlayerState):
        self._log_event("turn_start", player=player.name, round=self.round_number)
        
        # Pre-roll management
        if player.alive and not player.in_jail:
            self._management_phase(player, "pre_roll")
            
        if not player.alive: return
            
        # Jail decision
        if player.in_jail:
            out_of_jail = self._resolve_jail(player)
            if not out_of_jail:
                return # Turn ends if still in jail
                
        if not player.alive: return

        # Dice roll
        die1 = self.rng.randint(1, 6)
        die2 = self.rng.randint(1, 6)
        is_doubles = (die1 == die2)
        
        if is_doubles:
            player.doubles_count += 1
        else:
            player.doubles_count = 0

        if player.doubles_count == 3:
            self._log_event("jail", player=player.name, reason="three_doubles")
            player.in_jail = True
            player.position = JAIL_POSITION
            player.doubles_count = 0
            return

        self._move_player(player, die1 + die2, direct=False, dice_roll=(die1, die2))
        if not player.alive: return

        self._resolve_landing(player, die1 + die2)
        if not player.alive: return

        # Post-roll management
        self._management_phase(player, "post_roll")

    def _move_player(
        self,
        player: PlayerState,
        steps: int,
        direct: bool = False,
        collect_go: bool = False,
        dice_roll: tuple[int, int] | None = None,
    ):
        old_pos = player.position
        if direct:
            player.position = steps % 40
            if collect_go and (player.position < old_pos or player.position == GO_POSITION):
                player.money += self.config.go_salary
                self._log_event("pass_go", player=player.name, amount=self.config.go_salary)
        else:
            player.position += steps
            if player.position >= 40:
                player.position -= 40
                player.money += self.config.go_salary
                self._log_event("pass_go", player=player.name, amount=self.config.go_salary)
                
        space = BOARD[player.position]
        if not direct:
            roll_payload = {"player": player.name, "dice": steps, "space_name": space.name, "position": player.position}
            if dice_roll is not None:
                roll_payload["die1"], roll_payload["die2"] = dice_roll
            self._log_event("roll", **roll_payload)
        else:
            self._log_event("move_direct", player=player.name, space_name=space.name, position=player.position)

    def _resolve_landing(self, player: PlayerState, dice_total: int, special_multiplier: int = 1):
        space = BOARD[player.position]
        
        if space.space_type in ("property", "railroad", "utility"):
            prop = self.properties[player.position]
            if prop.owner is None:
                self._offer_property(player, player.position)
            elif prop.owner != player.seat and not prop.is_mortgaged:
                owner = self.players[prop.owner]
                rent = self._calculate_rent(player.position, dice_total, special_multiplier)
                self._log_event("rent_owed", player=player.name, creditor=owner.name, amount=rent, space=space.name)
                self._resolve_debt(player, rent, prop.owner)
                if player.alive and owner.alive:
                    owner.money += rent
                    
        elif space.space_type == "tax":
            self._log_event("tax_owed", player=player.name, amount=space.tax_amount, space=space.name)
            self._resolve_debt(player, space.tax_amount, None)
            
        elif space.space_type == "go_to_jail":
            self._log_event("jail", player=player.name, reason="landed_go_to_jail")
            player.in_jail = True
            player.position = JAIL_POSITION
            player.doubles_count = 0
            
        elif space.space_type == "chance":
            self._resolve_card(player, "chance", dice_total)
            
        elif space.space_type == "community_chest":
            self._resolve_card(player, "community_chest", dice_total)

    def _offer_property(self, player: PlayerState, space_index: int):
        space = BOARD[space_index]
        if player.money >= space.price:
            ctx = self._build_context(player, "buy", space_index=space_index)
            decision = self._request_decision(player, ctx)
            if decision.will_buy:
                player.money -= space.price
                self.properties[space_index].owner = player.seat
                self._log_event("buy", player=player.name, space_name=space.name, price=space.price)
                return
                
        if self.config.enable_auctions:
            self._run_auction(space_index, starting_seat=player.seat)

    def _run_auction(self, space_index: int, starting_seat: int = 0):
        """Run a sealed-bid auction using one decision call per active player."""
        space = BOARD[space_index]
        self._log_event("auction_start", space_name=space.name)

        active_players = [
            self.players[(starting_seat + offset) % len(self.players)]
            for offset in range(len(self.players))
            if self.players[(starting_seat + offset) % len(self.players)].alive
        ]
        bids = {}

        for p in active_players:
            ctx = self._build_context(p, "auction", space_index=space_index)
            decision = self._request_decision(p, ctx)
            bids[p.seat] = decision.bid_amount if decision.bid_amount > 0 else 0

        highest_bid = max(bids.values(), default=0)
        winner_seat = next(
            (p.seat for p in active_players if bids[p.seat] == highest_bid), None
        )

        if highest_bid > 0 and winner_seat is not None:
            winner = self.players[winner_seat]
            winner.money -= highest_bid
            self.properties[space_index].owner = winner_seat
            self._log_event("auction_win", player=winner.name, space_name=space.name, amount=highest_bid)
        else:
            self._log_event("auction_nobid", space_name=space.name)

    def _management_phase(self, player: PlayerState, phase_name: str):
        actions_taken = 0
        max_actions = self.config.max_management_actions
        while actions_taken < max_actions and player.alive:
            ctx = self._build_context(player, phase_name)
            if len(ctx.legal_actions) == 1 and ctx.legal_actions[0]["action"] == "end_management":
                self._log_event(
                    "management_skipped",
                    player=player.name,
                    phase=phase_name,
                    reason="no_meaningful_action",
                )
                break
                
            decision = self._request_decision(player, ctx)
            if decision.action == "end_management":
                break
                
            self._execute_management_action(player, decision)
            actions_taken += 1
        if actions_taken == max_actions:
            self._log_event(
                "management_limit_reached",
                player=player.name,
                phase=phase_name,
                limit=max_actions,
            )

    def _execute_management_action(self, player: PlayerState, decision):
        if decision.action == "mortgage":
            space = BOARD[decision.target_property]
            prop = self.properties[decision.target_property]
            prop.is_mortgaged = True
            player.money += space.mortgage_value
            self._log_event("mortgage", player=player.name, space_name=space.name)
            
        elif decision.action == "unmortgage":
            space = BOARD[decision.target_property]
            prop = self.properties[decision.target_property]
            cost = int(space.mortgage_value * 1.1)
            player.money -= cost
            prop.is_mortgaged = False
            self._log_event("unmortgage", player=player.name, space_name=space.name)
            
        elif decision.action == "build_house":
            space = BOARD[decision.target_property]
            prop = self.properties[decision.target_property]
            player.money -= space.house_price
            prop.houses += 1
            if prop.houses == 5:
                self.houses_available += 4
                self.hotels_available -= 1
            else:
                self.houses_available -= 1
            self._log_event("build", player=player.name, space_name=space.name)
            
        elif decision.action == "sell_house":
            space = BOARD[decision.target_property]
            prop = self.properties[decision.target_property]
            player.money += space.house_price // 2
            prop.houses -= 1
            if prop.houses == 4:
                self.houses_available -= 4
                self.hotels_available += 1
            else:
                self.houses_available += 1
            self._log_event("sell_house", player=player.name, space_name=space.name)

        elif decision.action == "propose_trade":
            self._handle_trade(player, decision.trade_proposal)

    def _handle_trade(self, initiator: PlayerState, proposal):
        recipient = self._validate_trade_proposal(initiator, proposal)
        context = self._build_context(
            recipient,
            "trade_response",
            trade_proposal=proposal,
            initiator_name=initiator.name,
        )
        response = self._request_decision(recipient, context)
        if response.action == "reject":
            self._log_event(
                "trade_rejected", player=recipient.name, initiator=initiator.name
            )
            return

        # Revalidate after the recipient's decision in case a custom observer
        # mutated state while the response was pending.
        recipient = self._validate_trade_proposal(initiator, proposal)
        initiator.money += proposal.money_requested - proposal.money_offered
        recipient.money += proposal.money_offered - proposal.money_requested
        for index in proposal.properties_offered:
            self.properties[index].owner = recipient.seat
        for index in proposal.properties_requested:
            self.properties[index].owner = initiator.seat

        if proposal.offer_cc_jail_card:
            initiator.has_cc_jail_card = False
            recipient.has_cc_jail_card = True
        if proposal.offer_chance_jail_card:
            initiator.has_chance_jail_card = False
            recipient.has_chance_jail_card = True
        if proposal.request_cc_jail_card:
            recipient.has_cc_jail_card = False
            initiator.has_cc_jail_card = True
        if proposal.request_chance_jail_card:
            recipient.has_chance_jail_card = False
            initiator.has_chance_jail_card = True

        self._log_event(
            "trade_accepted",
            player=recipient.name,
            initiator=initiator.name,
            properties_offered=list(proposal.properties_offered),
            properties_requested=list(proposal.properties_requested),
            money_offered=proposal.money_offered,
            money_requested=proposal.money_requested,
        )

    def _validate_trade_proposal(self, initiator: PlayerState, proposal):
        if proposal is None:
            raise InvalidAgentError("propose_trade requires a trade_proposal.")
        recipient = next(
            (
                p
                for p in self.players
                if p.name == proposal.target_player
                and p.alive
                and p.seat != initiator.seat
            ),
            None,
        )
        if recipient is None:
            raise InvalidAgentError("Trade target must be another living player.")
        if proposal.money_offered < 0 or proposal.money_requested < 0:
            raise InvalidAgentError("Trade money values cannot be negative.")
        if proposal.money_offered > initiator.money:
            raise InvalidAgentError("Initiator cannot afford the offered money.")
        if proposal.money_requested > recipient.money:
            raise InvalidAgentError("Recipient cannot afford the requested money.")

        if not any(
            (
                proposal.properties_offered,
                proposal.properties_requested,
                proposal.money_offered,
                proposal.money_requested,
                proposal.offer_cc_jail_card,
                proposal.offer_chance_jail_card,
                proposal.request_cc_jail_card,
                proposal.request_chance_jail_card,
            )
        ):
            raise InvalidAgentError("A trade must exchange at least one asset.")

        offered = set(proposal.properties_offered)
        requested = set(proposal.properties_requested)
        if len(offered) != len(proposal.properties_offered) or len(requested) != len(proposal.properties_requested):
            raise InvalidAgentError("Trade property lists cannot contain duplicates.")
        if offered & requested:
            raise InvalidAgentError("A property cannot be both offered and requested.")
        for index, expected_owner in [
            *((index, initiator.seat) for index in offered),
            *((index, recipient.seat) for index in requested),
        ]:
            if index not in self.properties or self.properties[index].owner != expected_owner:
                raise InvalidAgentError(f"Trade includes property {index} not owned by its offeror.")
            group = BOARD[index].group
            if any(self.properties[i].houses > 0 for i in GROUP_MEMBERS.get(group, [index])):
                raise InvalidAgentError("Properties from an improved group cannot be traded.")

        card_checks = (
            (proposal.offer_cc_jail_card, initiator.has_cc_jail_card),
            (proposal.offer_chance_jail_card, initiator.has_chance_jail_card),
            (proposal.request_cc_jail_card, recipient.has_cc_jail_card),
            (proposal.request_chance_jail_card, recipient.has_chance_jail_card),
        )
        if any(requested_card and not owned for requested_card, owned in card_checks):
            raise InvalidAgentError("Trade includes a Get Out of Jail Free card not owned by its offeror.")
        return recipient

    def _resolve_jail(self, player: PlayerState) -> bool:
        player.jail_turns += 1
        ctx = self._build_context(player, "jail_decision")
        decision = self._request_decision(player, ctx)
        
        if decision.action in ("use_cc_card", "use_chance_card"):
            if decision.action == "use_cc_card":
                player.has_cc_jail_card = False
                # Reinsert card
                cc_jail_idx = next(i for i, c in enumerate(COMMUNITY_CHEST_CARDS) if c.effect_type == "jail_free")
                self.cc_deck.append(cc_jail_idx)
            else:
                player.has_chance_jail_card = False
                chance_jail_idx = next(i for i, c in enumerate(CHANCE_CARDS) if c.effect_type == "jail_free")
                self.chance_deck.append(chance_jail_idx)
                
            player.in_jail = False
            player.jail_turns = 0
            self._log_event("jail_out", player=player.name, method=decision.action)
            return True
            
        elif decision.action == "pay_fine":
            self._resolve_debt(player, 50, None)
            if player.alive:
                player.in_jail = False
                player.jail_turns = 0
                self._log_event("jail_out", player=player.name, method="paid_fine")
                return True
            return False
            
        elif decision.action == "roll":
            die1 = self.rng.randint(1, 6)
            die2 = self.rng.randint(1, 6)
            if die1 == die2:
                player.in_jail = False
                player.jail_turns = 0
                self._log_event("jail_out", player=player.name, method="rolled_doubles")
                self._move_player(player, die1 + die2, direct=False)
                if player.alive:
                    self._resolve_landing(player, die1 + die2)
                return False # Don't take normal turn
            else:
                if player.jail_turns == 3:
                    self._log_event("jail_out", player=player.name, method="forced_pay")
                    self._resolve_debt(player, 50, None)
                    if player.alive:
                        player.in_jail = False
                        player.jail_turns = 0
                        self._move_player(player, die1 + die2, direct=False)
                        if player.alive:
                            self._resolve_landing(player, die1 + die2)
                return False

    def _resolve_debt(self, player: PlayerState, amount: int, creditor_seat: int | None):
        while player.money < amount and player.alive:
            ctx = self._build_context(player, "debt_resolution", debt_amount=amount, creditor_seat=creditor_seat)
            decision = self._request_decision(player, ctx)
            if decision.action == "declare_bankruptcy":
                self._bankruptcy(player, creditor_seat)
                break
            else:
                self._execute_management_action(player, decision)
                
        if player.alive:
            player.money -= amount
            self._log_event("debt_paid", player=player.name, amount=amount)

    def _bankruptcy(self, player: PlayerState, creditor_seat: int | None):
        player.alive = False
        creditor_name = self.players[creditor_seat].name if creditor_seat is not None else "Bank"
        self._log_event("bankrupt", player=player.name, creditor=creditor_name)
        
        owned = [i for i, p in self.properties.items() if p.owner == player.seat]
        
        if creditor_seat is not None:
            creditor = self.players[creditor_seat]
            creditor.money += max(player.money, 0)
            player.money = 0
            mortgage_interest = 0
            for i in owned:
                prop = self.properties[i]
                # Sell houses to bank
                if prop.houses > 0:
                    val = (prop.houses if prop.houses <= 4 else 5) * (BOARD[i].house_price // 2)
                    creditor.money += val
                    if prop.houses == 5:
                        self.hotels_available += 1
                    else:
                        self.houses_available += prop.houses
                    prop.houses = 0
                prop.owner = creditor_seat
                if prop.is_mortgaged:
                    mortgage_interest += max(1, int(BOARD[i].mortgage_value * 0.1))
            
            if player.has_cc_jail_card: creditor.has_cc_jail_card = True
            if player.has_chance_jail_card: creditor.has_chance_jail_card = True
            player.has_cc_jail_card = False
            player.has_chance_jail_card = False
            if mortgage_interest and creditor.alive:
                self._log_event(
                    "mortgage_interest_owed",
                    player=creditor.name,
                    amount=mortgage_interest,
                )
                self._resolve_debt(creditor, mortgage_interest, None)
        else:
            # Bankrupt to bank
            player.money = 0
            auction_indices = []
            for i in owned:
                prop = self.properties[i]
                if prop.houses > 0:
                    if prop.houses == 5:
                        self.hotels_available += 1
                    else:
                        self.houses_available += prop.houses
                prop.houses = 0
                prop.is_mortgaged = False
                prop.owner = None
                auction_indices.append(i)

            # Release the complete portfolio before asking any auction bidder.
            # This prevents agents from observing a half-bankrupt transient state.
            if self.config.enable_auctions:
                for i in auction_indices:
                    self._run_auction(
                        i, starting_seat=(player.seat + 1) % len(self.players)
                    )
                    
            if player.has_cc_jail_card:
                idx = next(i for i, c in enumerate(COMMUNITY_CHEST_CARDS) if c.effect_type == "jail_free")
                self.cc_deck.append(idx)
            if player.has_chance_jail_card:
                idx = next(i for i, c in enumerate(CHANCE_CARDS) if c.effect_type == "jail_free")
                self.chance_deck.append(idx)
            player.has_cc_jail_card = False
            player.has_chance_jail_card = False

    def _resolve_card(self, player: PlayerState, deck: str, dice_total: int):
        if deck == "community_chest":
            card_idx = self.cc_deck[self.cc_index]
            self.cc_index = (self.cc_index + 1) % len(self.cc_deck)
            card = COMMUNITY_CHEST_CARDS[card_idx]
        else:
            card_idx = self.chance_deck[self.chance_index]
            self.chance_index = (self.chance_index + 1) % len(self.chance_deck)
            card = CHANCE_CARDS[card_idx]
            
        self._log_event("card", player=player.name, deck=deck, text=card.text)
        
        if card.effect_type == "jail_free":
            if deck == "community_chest":
                player.has_cc_jail_card = True
                self.cc_deck.remove(card_idx)
                self.cc_index -= 1
            else:
                player.has_chance_jail_card = True
                self.chance_deck.remove(card_idx)
                self.chance_index -= 1
                
        elif card.effect_type == "collect":
            player.money += card.amount
            
        elif card.effect_type == "pay":
            self._resolve_debt(player, card.amount, None)
            
        elif card.effect_type == "move":
            self._move_player(
                player, card.destination, direct=True, collect_go=True
            )
            if player.alive:
                self._resolve_landing(player, dice_total)
                
        elif card.effect_type == "go_back":
            self._move_player(player, player.position - card.amount, direct=True)
            if player.alive:
                self._resolve_landing(player, dice_total)
                
        elif card.effect_type == "jail":
            player.in_jail = True
            player.position = JAIL_POSITION
            player.doubles_count = 0
            
        elif card.effect_type == "move_nearest":
            dest = card.destination # Group 1 or 2
            pos = player.position
            # Find nearest space of that group going forward
            next_space = next(i for i in range(1, 41) if BOARD[(pos + i) % 40].group == dest)
            target = (pos + next_space) % 40
            
            self._move_player(player, target, direct=True, collect_go=True)
            
            if player.alive:
                if BOARD[target].space_type == "railroad":
                    self._resolve_landing(player, dice_total, special_multiplier=2)
                else: # Utility
                    self._resolve_landing(player, dice_total, special_multiplier=10)
                    
        elif card.effect_type == "repairs":
            houses = 0
            hotels = 0
            for prop in self.properties.values():
                if prop.owner == player.seat:
                    if prop.houses == 5: hotels += 1
                    elif prop.houses > 0: houses += prop.houses
            cost = houses * card.per_house + hotels * card.per_hotel
            if cost > 0:
                self._resolve_debt(player, cost, None)
                
        elif card.effect_type == "pay_each":
            recipients = [
                p for p in self.players if p.alive and p.seat != player.seat
            ]
            for recipient in recipients:
                if not player.alive:
                    break
                self._resolve_debt(player, card.amount, recipient.seat)
                if player.alive:
                    recipient.money += card.amount
                            
        elif card.effect_type == "collect_from_each":
            for p in self.players:
                if p.alive and p.seat != player.seat:
                    self._resolve_debt(p, card.amount, player.seat)
                    if p.alive:
                        player.money += card.amount

    def _calculate_rent(self, space_idx: int, dice_total: int, special_multiplier: int) -> int:
        space = BOARD[space_idx]
        prop = self.properties[space_idx]
        if prop.is_mortgaged: return 0
        
        if space.space_type == "railroad":
            owned = sum(1 for i in GROUP_MEMBERS[1] if self.properties[i].owner == prop.owner)
            rent = 25 * (2 ** (owned - 1))
            if special_multiplier == 2: rent *= 2
            return rent
            
        elif space.space_type == "utility":
            owned = sum(1 for i in GROUP_MEMBERS[2] if self.properties[i].owner == prop.owner)
            if special_multiplier == 10:
                return 10 * dice_total
            return (4 * dice_total) if owned == 1 else (10 * dice_total)
            
        else: # Property
            if prop.houses == 0:
                if self._check_monopoly(prop.owner, space.group):
                    return space.base_rent * 2
                return space.base_rent
            elif prop.houses == 1: return space.rent_1h
            elif prop.houses == 2: return space.rent_2h
            elif prop.houses == 3: return space.rent_3h
            elif prop.houses == 4: return space.rent_4h
            else: return space.rent_hotel

    def _check_monopoly(self, owner: int, group: int) -> bool:
        if group < 3: return False
        return all(self.properties[i].owner == owner for i in GROUP_MEMBERS[group])

    def _calculate_net_worth(self, player: PlayerState) -> int:
        nw = player.money
        for i, p in self.properties.items():
            if p.owner == player.seat:
                space = BOARD[i]
                if p.is_mortgaged:
                    nw += space.mortgage_value
                else:
                    nw += space.price
                    if p.houses > 0:
                        nw += p.houses * space.house_price
        return nw

    def validate_invariants(self, *, full_history: bool = False) -> None:
        """Raise AssertionError if stable game state violates a core rule."""
        valid_seats = set(range(len(self.players)))
        for player in self.players:
            assert 0 <= player.position < len(BOARD), "player position is off board"
            assert player.money >= 0, "player money cannot be negative after resolution"
            if player.in_jail:
                assert player.position == JAIL_POSITION, "jailed player must be at Jail"
            if not player.alive:
                assert player.money == 0, "bankrupt player must have no cash"

        for index, prop in self.properties.items():
            assert prop.owner is None or prop.owner in valid_seats, "invalid property owner"
            assert 0 <= prop.houses <= 5, "invalid improvement count"
            if prop.owner is not None:
                assert self.players[prop.owner].alive, "bankrupt player still owns property"
            if prop.is_mortgaged:
                assert prop.houses == 0, "mortgaged property cannot be improved"
            if prop.houses:
                assert BOARD[index].space_type == "property", "special property is improved"

        for group, indices in GROUP_MEMBERS.items():
            if group < 3:
                continue
            states = [self.properties[index] for index in indices]
            if any(state.houses for state in states):
                owners = {state.owner for state in states}
                assert len(owners) == 1 and None not in owners, "improved group lacks one owner"
                assert not any(state.is_mortgaged for state in states), "improved group is mortgaged"
                levels = [state.houses for state in states]
                assert max(levels) - min(levels) <= 1, "improvements are not even"

        houses_on_board = sum(
            prop.houses for prop in self.properties.values() if 1 <= prop.houses <= 4
        )
        hotels_on_board = sum(1 for prop in self.properties.values() if prop.houses == 5)
        assert self.houses_available + houses_on_board == self.config.max_houses, "house supply drift"
        assert self.hotels_available + hotels_on_board == self.config.max_hotels, "hotel supply drift"

        cc_card = next(
            index for index, card in enumerate(COMMUNITY_CHEST_CARDS)
            if card.effect_type == "jail_free"
        )
        chance_card = next(
            index for index, card in enumerate(CHANCE_CARDS)
            if card.effect_type == "jail_free"
        )
        cc_holders = sum(player.has_cc_jail_card for player in self.players)
        chance_holders = sum(player.has_chance_jail_card for player in self.players)
        assert self.cc_deck.count(cc_card) + cc_holders == 1, "Community Chest jail card duplicated or lost"
        assert self.chance_deck.count(chance_card) + chance_holders == 1, "Chance jail card duplicated or lost"

        public_events = self.public_history if full_history else self.public_history[-1:]
        private_decisions = self.decision_log if full_history else self.decision_log[-1:]
        assert not any(
            event.get("type") == "agent_decision" for event in public_events
        ), "private agent decision leaked into public history"
        for decision in private_decisions:
            assert decision.get("visible_to") == [decision.get("player")], "decision visibility leak"

    def _build_context(self, player: PlayerState, phase: str, **kwargs) -> AgentContext:
        ctx = AgentContext(
            player_name=player.name,
            phase=phase,
            turn_number=self.turn_number,
            round_number=self.round_number,
            position=player.position,
            position_name=BOARD[player.position].name,
            money=player.money,
            properties_owned=[
                {"index": i, "name": BOARD[i].name, "group": BOARD[i].group, "houses": p.houses, "mortgaged": p.is_mortgaged}
                for i, p in self.properties.items() if p.owner == player.seat
            ],
            in_jail=player.in_jail,
            has_cc_jail_card=player.has_cc_jail_card,
            has_chance_jail_card=player.has_chance_jail_card,
            all_players=[
                {
                    "name": p.name, "money": p.money, "position": p.position, "position_name": BOARD[p.position].name,
                    "alive": p.alive, "in_jail": p.in_jail, "property_count": sum(1 for prop in self.properties.values() if prop.owner == p.seat)
                } for p in self.players
            ],
            property_ownership=[
                {"index": i, "name": BOARD[i].name, "group": BOARD[i].group, "owner_name": self.players[p.owner].name, "houses": p.houses, "mortgaged": p.is_mortgaged}
                for i, p in self.properties.items() if p.owner is not None
            ],
            public_history=copy.deepcopy(
                self.public_history[-self.config.context_public_history_limit :]
            ),
            private_decision_history=self._private_decision_history_for(player.name),
        )
        
        # Populate legal actions based on phase
        if phase in ("pre_roll", "post_roll"):
            ctx.legal_actions = [{"action": "end_management", "description": "End management phase and proceed."}]
            buildable = []
            sellable = []
            mortgageable = []
            unmortgageable = []
            
            # Compute valid targets
            for grp, indices in GROUP_MEMBERS.items():
                if grp < 3: continue
                if all(self.properties[i].owner == player.seat for i in indices):
                    # Check mortgages in group
                    if not any(self.properties[i].is_mortgaged for i in indices):
                        # Can build if even
                        for i in indices:
                            h = self.properties[i].houses
                            if h < 5 and all(self.properties[j].houses >= h for j in indices):
                                piece_available = (
                                    self.houses_available > 0
                                    if h < 4
                                    else self.hotels_available > 0
                                )
                                if player.money >= BOARD[i].house_price and piece_available:
                                    buildable.append(i)
                            can_break_hotel = h < 5 or self.houses_available >= 4
                            if h > 0 and can_break_hotel and all(self.properties[j].houses <= h for j in indices):
                                sellable.append(i)
                                
            for i, p in self.properties.items():
                if p.owner == player.seat:
                    if p.is_mortgaged:
                        if player.money >= int(BOARD[i].mortgage_value * 1.1):
                            unmortgageable.append(i)
                    else:
                        if p.houses == 0:
                            # Check no houses in entire group
                            if all(self.properties[j].houses == 0 for j in GROUP_MEMBERS.get(BOARD[i].group, [i])):
                                mortgageable.append(i)
                                
            if buildable: ctx.legal_actions.append({"action": "build_house", "allowed_targets": buildable})
            if sellable: ctx.legal_actions.append({"action": "sell_house", "allowed_targets": sellable})
            if mortgageable: ctx.legal_actions.append({"action": "mortgage", "allowed_targets": mortgageable})
            if unmortgageable: ctx.legal_actions.append({"action": "unmortgage", "allowed_targets": unmortgageable})
            if self.config.enable_trading and any(
                p.alive and p.seat != player.seat for p in self.players
            ):
                ctx.legal_actions.append({
                    "action": "propose_trade",
                    "description": "Propose one bounded trade to another living player.",
                })
            
        elif phase == "buy":
            space = BOARD[kwargs["space_index"]]
            ctx.landed_space = {
                "name": space.name,
                "price": space.price,
                "rent": space.base_rent,
                "rent_1h": space.rent_1h,
                "rent_2h": space.rent_2h,
                "rent_3h": space.rent_3h,
                "rent_4h": space.rent_4h,
                "rent_hotel": space.rent_hotel,
                "house_price": space.house_price,
                "mortgage_value": space.mortgage_value,
                "space_type": space.space_type,
            }
            ctx.legal_actions = [{"action": "buy"}, {"action": "decline"}]
            
        elif phase == "auction":
            space = BOARD[kwargs["space_index"]]
            ctx.auction_state = {
                "space_name": space.name,
                "price": space.price,
                "rent": space.base_rent,
                "rent_1h": space.rent_1h,
                "rent_2h": space.rent_2h,
                "rent_3h": space.rent_3h,
                "rent_4h": space.rent_4h,
                "rent_hotel": space.rent_hotel,
                "house_price": space.house_price,
                "mortgage_value": space.mortgage_value,
                "space_type": space.space_type,
            }
            ctx.legal_actions = [{"action": "bid"}]
            
        elif phase == "jail_decision":
            ctx.legal_actions = [{"action": "roll"}]
            ctx.legal_actions.append({"action": "pay_fine"})
            if player.has_cc_jail_card:
                ctx.legal_actions.append({"action": "use_cc_card"})
            if player.has_chance_jail_card:
                ctx.legal_actions.append({"action": "use_chance_card"})
            
        elif phase == "debt_resolution":
            ctx.debt_amount = kwargs["debt_amount"]
            ctx.creditor_name = self.players[kwargs["creditor_seat"]].name if kwargs["creditor_seat"] is not None else "Bank"
            ctx.legal_actions = [{"action": "declare_bankruptcy", "description": "Surrender and lose the game."}]
            
            sellable = []
            mortgageable = []
            for grp, indices in GROUP_MEMBERS.items():
                if grp < 3: continue
                if all(self.properties[i].owner == player.seat for i in indices):
                    for i in indices:
                        h = self.properties[i].houses
                        can_break_hotel = h < 5 or self.houses_available >= 4
                        if h > 0 and can_break_hotel and all(self.properties[j].houses <= h for j in indices):
                            sellable.append(i)
                            
            for i, p in self.properties.items():
                if p.owner == player.seat and not p.is_mortgaged and p.houses == 0:
                    if all(self.properties[j].houses == 0 for j in GROUP_MEMBERS.get(BOARD[i].group, [i])):
                        mortgageable.append(i)
                        
            if sellable: ctx.legal_actions.append({"action": "sell_house", "allowed_targets": sellable})
            if mortgageable: ctx.legal_actions.append({"action": "mortgage", "allowed_targets": mortgageable})

        elif phase == "trade_response":
            ctx.trade_proposal = kwargs["trade_proposal"]
            ctx.extra_hint = f"Trade proposed by {kwargs['initiator_name']}."
            ctx.legal_actions = [
                {"action": "accept", "description": "Accept and execute the trade."},
                {"action": "reject", "description": "Reject the trade."},
            ]
            
        return ctx

    def _private_decision_history_for(self, player_name: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(event)
            for event in self.decision_log
            if player_name in event.get("visible_to", [])
        ][-self.config.context_private_history_limit :]

def _net_worth_rankings(players_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(players_summary, key=lambda item: (-item["net_worth"], item["seat"]))
    rankings = []
    previous_net_worth = None
    current_rank = 0
    for index, player in enumerate(ordered, start=1):
        if player["net_worth"] != previous_net_worth:
            current_rank = index
            previous_net_worth = player["net_worth"]
        rankings.append({
            "rank": current_rank,
            "name": player["name"],
            "seat": player["seat"],
            "evaluated": player["evaluated"],
            "alive": player["alive"],
            "net_worth": player["net_worth"],
        })
    return rankings


def evaluated_player_ranked_first(result: dict[str, Any], player_name: str) -> bool:
    """Return True when the named player has final net-worth rank 1."""
    return any(
        row["name"] == player_name and row["rank"] == 1
        for row in result.get("final_rankings", [])
    )


def score_evaluated_player(result: dict[str, Any], player_name: str) -> bool:
    """Win normally; on a safety limit, score final net-worth rank one."""
    if result.get("timeout"):
        return evaluated_player_ranked_first(result, player_name)
    return result.get("winner") == player_name


def run_monopoly_game(game_config: GameConfig, ui=None) -> dict[str, Any]:
    game = MonopolyGame(game_config, ui)
    game.start()
    rounds_played = (
        min(game.round_number, game.config.max_rounds)
        if game.end_reason == "max_rounds"
        else game.round_number
    )
    
    players_summary = []
    for p in game.players:
        players_summary.append({
            "name": p.name,
            "seat": p.seat,
            "model_id": p.model_id,
            "evaluated": p.evaluated,
            "final_money": p.money,
            "final_position": p.position,
            "alive": p.alive,
            "properties_owned": [i for i, prop in game.properties.items() if prop.owner == p.seat],
            "net_worth": game._calculate_net_worth(p)
        })
        
    final_rankings = _net_worth_rankings(players_summary)
    final_state = {
        "rounds_played": rounds_played,
        "turns_played": game.turns_completed,
        "end_reason": game.end_reason,
        "players": copy.deepcopy(players_summary),
        "rankings": copy.deepcopy(final_rankings),
        "properties": {
            index: {
                "owner": prop.owner,
                "houses": prop.houses,
                "mortgaged": prop.is_mortgaged,
            }
            for index, prop in game.properties.items()
        },
        "bank": {
            "houses_available": game.houses_available,
            "hotels_available": game.hotels_available,
        },
    }

    game.game_log["winner"] = game.winner
    game.game_log["timeout"] = game.timeout
    game.game_log["end_reason"] = game.end_reason
    game.game_log["final_state"] = copy.deepcopy(final_state)
    agent_logs = {}
    for p in game.players:
        if callable(getattr(p.agent, "get_log", None)):
            log = p.agent.get_log()
            if log is not None:
                agent_logs[p.name] = log

    return {
        "winner": game.winner,
        "timeout": game.timeout,
        "end_reason": game.end_reason,
        "rounds_played": rounds_played,
        "turns_played": game.turns_completed,
        "players": players_summary,
        "final_rankings": final_rankings,
        "final_state": final_state,
        "public_history": game.public_history,
        "decision_log": game.decision_log,
        "agent_logs": agent_logs,
        "game_log": game.game_log,
    }
