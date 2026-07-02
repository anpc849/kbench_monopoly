import unittest
from contextlib import nullcontext
from unittest.mock import patch

from kbench_monopoly.agent import (
    AuctionBid,
    BaseAgent,
    BuyDecision,
    DebtDecision,
    InvalidAgentError,
    JailDecision,
    ManagementDecision,
    TradeProposal,
    TradeResponse,
    DefaultLLMAgent,
)
from kbench_monopoly.board_data import BOARD, CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from kbench_monopoly.config import GameConfig, build_benchmark_config
from kbench_monopoly.runner import MonopolyGame, run_monopoly_game
from kbench_monopoly.runner import evaluated_player_ranked_first, score_evaluated_player


class FirstLegalAgent(BaseAgent):
    """Small deterministic agent used for engine tests."""

    def choose_management(self, context) -> ManagementDecision:
        return ManagementDecision(action="end_management")

    def choose_buy(self, context) -> BuyDecision:
        return BuyDecision(will_buy=True)

    def choose_auction_bid(self, context) -> AuctionBid:
        return AuctionBid(bid_amount=0)

    def choose_jail_action(self, context) -> JailDecision:
        return JailDecision(action="roll")

    def choose_debt_action(self, context) -> DebtDecision:
        return DebtDecision(action="declare_bankruptcy")

    def choose_trade_response(self, context) -> TradeResponse:
        return TradeResponse(action="reject")


class DeclineAgent(FirstLegalAgent):
    def choose_buy(self, context) -> BuyDecision:
        return BuyDecision(will_buy=False)


class BidAgent(FirstLegalAgent):
    def __init__(self, amount: int):
        self.amount = amount
        self.bid_calls = 0

    def choose_auction_bid(self, context) -> AuctionBid:
        self.bid_calls += 1
        return AuctionBid(bid_amount=self.amount)


class AcceptTradeAgent(FirstLegalAgent):
    def choose_trade_response(self, context) -> TradeResponse:
        return TradeResponse(action="accept")


class MissingDecisionAgent(FirstLegalAgent):
    def choose_buy(self, context):
        return object()

    def choose_auction_bid(self, context):
        return object()


class JailCardAgent(FirstLegalAgent):
    def __init__(self, action="use_chance_card"):
        self.action = action

    def choose_jail_action(self, context) -> JailDecision:
        return JailDecision(action=self.action)


class FineMortgageAgent(FirstLegalAgent):
    def __init__(self):
        self.debt_calls = 0

    def choose_jail_action(self, context) -> JailDecision:
        return JailDecision(action="pay_fine")

    def choose_debt_action(self, context) -> DebtDecision:
        self.debt_calls += 1
        mortgage = next(
            (a for a in context.legal_actions if a["action"] == "mortgage"), None
        )
        if mortgage:
            return DebtDecision(
                action="mortgage", target_property=mortgage["allowed_targets"][0]
            )
        return DebtDecision(action="declare_bankruptcy")


class CountingDebtAgent(FirstLegalAgent):
    def __init__(self):
        self.debt_calls = 0

    def choose_debt_action(self, context) -> DebtDecision:
        self.debt_calls += 1
        return DebtDecision(action="declare_bankruptcy")


class CyclingManagementAgent(FirstLegalAgent):
    def __init__(self):
        self.management_calls = 0

    def choose_management(self, context) -> ManagementDecision:
        self.management_calls += 1
        for action_name in ("mortgage", "unmortgage"):
            action = next(
                (a for a in context.legal_actions if a["action"] == action_name), None
            )
            if action:
                return ManagementDecision(
                    action=action_name, target_property=action["allowed_targets"][0]
                )
        return ManagementDecision(action="end_management")


class InspectBidAgent(BidAgent):
    def __init__(self, amount: int):
        super().__init__(amount)
        self.seen_public_history = []

    def choose_auction_bid(self, context) -> AuctionBid:
        self.seen_public_history = list(context.public_history)
        return super().choose_auction_bid(context)


def make_config(agent_a=None, agent_b=None, **updates):
    data = {
        "player_configs": [
            {"name": "P1", "agent": agent_a or FirstLegalAgent(), "evaluated": True},
            {"name": "P2", "agent": agent_b or FirstLegalAgent()},
        ],
        "seed": 42,
        "max_rounds": 20,
        "max_turns": 50,
    }
    data.update(updates)
    return GameConfig(**data)


class MonopolyTests(unittest.TestCase):
    def test_board_and_decks_are_complete(self):
        self.assertEqual(len(BOARD), 40)
        self.assertEqual(len(CHANCE_CARDS), 16)
        self.assertEqual(len(COMMUNITY_CHEST_CARDS), 16)
        self.assertEqual(BOARD[0].space_type, "go")
        self.assertEqual(BOARD[10].space_type, "jail")
        self.assertEqual(BOARD[30].space_type, "go_to_jail")

    def test_config_rejects_invalid_players(self):
        with self.assertRaises(ValueError):
            GameConfig(player_configs=[])
        with self.assertRaises(ValueError):
            GameConfig(
                player_configs=[
                    {"name": "Same", "agent": FirstLegalAgent()},
                    {"name": "Same", "agent": FirstLegalAgent()},
                ]
            )

    def test_timeout_has_no_winner(self):
        result = run_monopoly_game(make_config(max_turns=1))
        self.assertTrue(result["timeout"])
        self.assertIsNone(result["winner"])
        self.assertEqual(result["turns_played"], 1)

    def test_max_rounds_uses_net_worth_ranking_and_explicit_end_reason(self):
        result = run_monopoly_game(
            make_config(max_rounds=1, max_turns=100)
        )
        self.assertTrue(result["timeout"])
        self.assertEqual(result["end_reason"], "max_rounds")
        self.assertIsNone(result["winner"])
        self.assertEqual(result["final_rankings"][0]["rank"], 1)
        evaluated_name = next(
            player["name"] for player in result["players"] if player["evaluated"]
        )
        expected = next(
            row["rank"] == 1
            for row in result["final_rankings"]
            if row["name"] == evaluated_name
        )
        self.assertEqual(score_evaluated_player(result, evaluated_name), expected)

    def test_tied_best_net_worth_counts_as_rank_one(self):
        result = {
            "timeout": True,
            "winner": None,
            "final_rankings": [
                {"rank": 1, "name": "P1", "net_worth": 1500},
                {"rank": 1, "name": "P2", "net_worth": 1500},
            ],
        }
        self.assertTrue(evaluated_player_ranked_first(result, "P1"))
        self.assertTrue(evaluated_player_ranked_first(result, "P2"))
        self.assertTrue(score_evaluated_player(result, "P1"))

    def test_non_timeout_scoring_uses_last_standing_winner(self):
        result = {
            "timeout": False,
            "winner": "P2",
            "final_rankings": [{"rank": 1, "name": "P1", "net_worth": 2000}],
        }
        self.assertFalse(score_evaluated_player(result, "P1"))
        self.assertTrue(score_evaluated_player(result, "P2"))

    def test_passing_go_collects_salary(self):
        game = MonopolyGame(make_config(enable_auctions=False))
        player = game.players[0]
        player.position = 39
        before = player.money
        game._move_player(player, 2)
        self.assertEqual(player.position, 1)
        self.assertEqual(player.money, before + game.config.go_salary)

    def test_chance_direct_move_collects_for_passing_go(self):
        game = MonopolyGame(
            make_config(DeclineAgent(), DeclineAgent(), enable_auctions=False)
        )
        player = game.players[0]
        player.position = 36
        before = player.money
        game.chance_deck = [14]  # Advance to St. Charles Place.
        game.chance_index = 0
        game._resolve_card(player, "chance", dice_total=7)
        self.assertEqual(player.position, 11)
        self.assertEqual(player.money, before + game.config.go_salary)

    def test_sealed_auction_uses_one_call_per_player_and_stable_tie_break(self):
        first = BidAgent(100)
        second = BidAgent(100)
        game = MonopolyGame(make_config(first, second))
        game._run_auction(1, starting_seat=1)
        self.assertEqual(first.bid_calls, 1)
        self.assertEqual(second.bid_calls, 1)
        self.assertEqual(game.properties[1].owner, 1)
        self.assertEqual(game.players[1].money, 1400)

    def test_illegal_custom_agent_bid_is_rejected(self):
        game = MonopolyGame(make_config(BidAgent(2000), BidAgent(0)))
        with self.assertRaises(InvalidAgentError):
            game._run_auction(1)

    def test_rent_calculation(self):
        game = MonopolyGame(make_config())
        game.properties[1].owner = 0
        self.assertEqual(game._calculate_rent(1, 7, 1), 2)
        game.properties[3].owner = 0
        self.assertEqual(game._calculate_rent(1, 7, 1), 4)
        game.properties[1].houses = 1
        self.assertEqual(game._calculate_rent(1, 7, 1), 10)

    def test_piece_supply_blocks_building_and_hotel_breaking(self):
        game = MonopolyGame(make_config())
        game.properties[1].owner = game.properties[3].owner = 0
        game.houses_available = 0
        context = game._build_context(game.players[0], "pre_roll")
        self.assertNotIn("build_house", [a["action"] for a in context.legal_actions])

        game.properties[1].houses = game.properties[3].houses = 5
        game.houses_available = 3
        context = game._build_context(game.players[0], "pre_roll")
        self.assertNotIn("sell_house", [a["action"] for a in context.legal_actions])

    def test_bankruptcy_transfers_cash_property_and_mortgage_interest(self):
        game = MonopolyGame(make_config())
        debtor, creditor = game.players
        debtor.money = 30
        creditor.money = 100
        game.properties[1].owner = debtor.seat
        game.properties[1].is_mortgaged = True
        game._bankruptcy(debtor, creditor.seat)
        self.assertFalse(debtor.alive)
        self.assertEqual(debtor.money, 0)
        self.assertEqual(game.properties[1].owner, creditor.seat)
        self.assertEqual(creditor.money, 127)

    def test_bounded_trade_executes_after_acceptance(self):
        game = MonopolyGame(
            make_config(FirstLegalAgent(), AcceptTradeAgent(), enable_trading=True)
        )
        initiator, recipient = game.players
        game.properties[1].owner = initiator.seat
        game.properties[3].owner = recipient.seat
        proposal = TradeProposal(
            target_player=recipient.name,
            properties_offered=[1],
            properties_requested=[3],
            money_offered=50,
            money_requested=20,
        )
        game._handle_trade(initiator, proposal)
        self.assertEqual(game.properties[1].owner, recipient.seat)
        self.assertEqual(game.properties[3].owner, initiator.seat)
        self.assertEqual(initiator.money, 1470)
        self.assertEqual(recipient.money, 1530)

    def test_improved_property_cannot_be_traded(self):
        game = MonopolyGame(make_config(enable_trading=True))
        game.properties[1].owner = game.players[0].seat
        game.properties[1].houses = 1
        proposal = TradeProposal(target_player="P2", properties_offered=[1])
        with self.assertRaises(InvalidAgentError):
            game._handle_trade(game.players[0], proposal)

    def test_result_contains_decisions_and_versioned_replay(self):
        result = run_monopoly_game(make_config(max_turns=2))
        self.assertTrue(result["decision_log"])
        self.assertEqual(
            result["game_log"]["schema_version"], "monopoly-game-log-v1"
        )
        self.assertIn("winner", result["game_log"])
        self.assertEqual(result["final_state"], result["game_log"]["final_state"])
        self.assertIn("rankings", result["final_state"])
        self.assertIn("bank", result["final_state"])
        self.assertTrue(
            any(e["type"] == "agent_decision" for e in result["game_log"]["events"])
        )

    def test_private_history_contains_only_own_decisions(self):
        game = MonopolyGame(make_config())
        p1, p2 = game.players
        c1 = game._build_context(p1, "pre_roll")
        game._request_decision(p1, c1)
        c2 = game._build_context(p2, "pre_roll")
        self.assertFalse(c2.private_decision_history)
        next_c1 = game._build_context(p1, "post_roll")
        self.assertEqual(len(next_c1.private_decision_history), 1)

    def test_invalid_agent_shape_is_rejected(self):
        with self.assertRaises(InvalidAgentError):
            MonopolyGame(make_config(object(), FirstLegalAgent()))

    def test_missing_buy_field_is_rejected_without_decline_fallback(self):
        game = MonopolyGame(
            make_config(MissingDecisionAgent(), FirstLegalAgent(), enable_auctions=False)
        )
        with self.assertRaises(InvalidAgentError):
            game._offer_property(game.players[0], 1)
        self.assertIsNone(game.properties[1].owner)

    def test_missing_bid_field_is_rejected_without_zero_bid_fallback(self):
        game = MonopolyGame(make_config(MissingDecisionAgent(), BidAgent(1)))
        with self.assertRaises(InvalidAgentError):
            game._run_auction(1)
        self.assertIsNone(game.properties[1].owner)

    def test_sealed_bids_are_not_visible_to_later_bidders(self):
        first = BidAgent(75)
        second = InspectBidAgent(50)
        game = MonopolyGame(make_config(first, second))
        game._run_auction(1)
        event_types = [event["type"] for event in second.seen_public_history]
        self.assertEqual(event_types, ["auction_start"])
        self.assertFalse(any("bid_amount" in event for event in second.seen_public_history))

    def test_management_limit_is_configurable_and_logged(self):
        agent = CyclingManagementAgent()
        game = MonopolyGame(make_config(agent, FirstLegalAgent(), max_management_actions=2))
        game.properties[5].owner = 0
        game._management_phase(game.players[0], "pre_roll")
        self.assertEqual(agent.management_calls, 2)
        self.assertEqual(game.public_history[-1]["type"], "management_limit_reached")
        self.assertEqual(game.public_history[-1]["limit"], 2)

    def test_no_choice_management_progression_is_explicit_and_has_no_agent_call(self):
        agent = CyclingManagementAgent()
        game = MonopolyGame(make_config(agent, FirstLegalAgent()))
        game._management_phase(game.players[0], "pre_roll")
        self.assertEqual(agent.management_calls, 0)
        self.assertEqual(game.public_history[-1]["type"], "management_skipped")

    def test_only_legal_bankruptcy_still_comes_from_agent(self):
        agent = CountingDebtAgent()
        game = MonopolyGame(make_config(agent, FirstLegalAgent()))
        game.players[0].money = 0
        game._resolve_debt(game.players[0], 10, None)
        self.assertEqual(agent.debt_calls, 1)
        self.assertFalse(game.players[0].alive)
        self.assertEqual(game.decision_log[-1]["decision"]["action"], "declare_bankruptcy")

    def test_jail_card_choice_is_explicit_when_both_cards_are_owned(self):
        game = MonopolyGame(make_config(JailCardAgent(), FirstLegalAgent()))
        player = game.players[0]
        player.in_jail = True
        player.has_cc_jail_card = True
        player.has_chance_jail_card = True
        context = game._build_context(player, "jail_decision")
        actions = {a["action"] for a in context.legal_actions}
        self.assertIn("use_cc_card", actions)
        self.assertIn("use_chance_card", actions)
        self.assertTrue(game._resolve_jail(player))
        self.assertTrue(player.has_cc_jail_card)
        self.assertFalse(player.has_chance_jail_card)

    def test_jail_fine_can_be_selected_before_liquid_cash_is_available(self):
        agent = FineMortgageAgent()
        game = MonopolyGame(make_config(agent, FirstLegalAgent()))
        player = game.players[0]
        player.in_jail = True
        player.money = 0
        game.properties[5].owner = player.seat
        context = game._build_context(player, "jail_decision")
        self.assertIn("pay_fine", [a["action"] for a in context.legal_actions])
        self.assertTrue(game._resolve_jail(player))
        self.assertTrue(player.alive)
        self.assertFalse(player.in_jail)
        self.assertEqual(player.money, 50)
        self.assertEqual(agent.debt_calls, 1)

    def test_tax_and_go_to_jail_spaces_resolve_automatically(self):
        game = MonopolyGame(make_config())
        player = game.players[0]
        player.position = 4
        game._resolve_landing(player, 7)
        self.assertEqual(player.money, 1300)
        player.position = 30
        game._resolve_landing(player, 8)
        self.assertTrue(player.in_jail)
        self.assertEqual(player.position, 10)

    def test_railroad_and_utility_rent_tables(self):
        game = MonopolyGame(make_config())
        for index in (5, 15, 25, 35):
            game.properties[index].owner = 0
        self.assertEqual(game._calculate_rent(5, 7, 1), 200)
        self.assertEqual(game._calculate_rent(5, 7, 2), 400)
        game.properties[12].owner = 0
        self.assertEqual(game._calculate_rent(12, 7, 1), 28)
        game.properties[28].owner = 0
        self.assertEqual(game._calculate_rent(12, 7, 1), 70)
        self.assertEqual(game._calculate_rent(12, 7, 10), 70)

    def test_mortgage_and_unmortgage_use_declared_values(self):
        game = MonopolyGame(make_config())
        player = game.players[0]
        game.properties[5].owner = player.seat
        game._execute_management_action(
            player, ManagementDecision(action="mortgage", target_property=5)
        )
        self.assertTrue(game.properties[5].is_mortgaged)
        self.assertEqual(player.money, 1600)
        game._execute_management_action(
            player, ManagementDecision(action="unmortgage", target_property=5)
        )
        self.assertFalse(game.properties[5].is_mortgaged)
        self.assertEqual(player.money, 1490)

    def test_even_building_targets_only_least_improved_property(self):
        game = MonopolyGame(make_config())
        game.properties[1].owner = game.properties[3].owner = 0
        game.properties[1].houses = 1
        context = game._build_context(game.players[0], "pre_roll")
        build = next(a for a in context.legal_actions if a["action"] == "build_house")
        self.assertEqual(build["allowed_targets"], [3])

    def test_mortgaged_property_collects_no_rent(self):
        game = MonopolyGame(make_config())
        game.properties[39].owner = 0
        game.properties[39].is_mortgaged = True
        self.assertEqual(game._calculate_rent(39, 8, 1), 0)

    def test_noop_trade_is_rejected(self):
        game = MonopolyGame(make_config(enable_trading=True))
        with self.assertRaises(InvalidAgentError):
            game._handle_trade(game.players[0], TradeProposal(target_player="P2"))

    def test_random_seed_reproduces_public_game_history(self):
        first = run_monopoly_game(make_config(max_turns=8))
        second = run_monopoly_game(make_config(max_turns=8))
        self.assertEqual(first["public_history"], second["public_history"])

    def test_starting_player_is_seeded_random_not_always_evaluated_first(self):
        starts = set()
        for seed in range(12):
            result = run_monopoly_game(make_config(seed=seed, max_turns=1))
            game_start = result["public_history"][0]
            first_turn = next(
                event for event in result["public_history"] if event["type"] == "turn_start"
            )
            starts.add(game_start["starting_player"])
            self.assertEqual(game_start["starting_player"], first_turn["player"])
        self.assertGreater(len(starts), 1)

    def test_randomized_round_start_still_allows_full_turn_cycle(self):
        selected_seed = None
        for seed in range(50):
            result = run_monopoly_game(make_config(seed=seed, max_turns=1))
            if result["public_history"][0]["starting_seat"] != 0:
                selected_seed = seed
                break
        self.assertIsNotNone(selected_seed)

        result = run_monopoly_game(
            make_config(seed=selected_seed, max_rounds=1, max_turns=20)
        )
        self.assertEqual(result["end_reason"], "max_rounds")
        self.assertGreaterEqual(result["turns_played"], len(result["players"]))

    def test_supported_player_counts_smoke(self):
        for count in (2, 3, 4):
            with self.subTest(count=count):
                configs = [
                    {"name": f"P{i}", "agent": FirstLegalAgent(), "evaluated": i == 0}
                    for i in range(count)
                ]
                result = run_monopoly_game(
                    GameConfig(
                        player_configs=configs,
                        seed=10,
                        max_rounds=2,
                        max_turns=3,
                    )
                )
                self.assertEqual(len(result["players"]), count)
                self.assertTrue(result["game_log"]["events"])

    def test_agent_decisions_are_private_not_public_events(self):
        result = run_monopoly_game(make_config(max_turns=2))
        self.assertFalse(
            any(event.get("type") == "agent_decision" for event in result["public_history"])
        )
        self.assertTrue(
            any(event.get("type") == "agent_decision" for event in result["game_log"]["events"])
        )

    def test_private_reason_is_rendered_once_not_duplicated(self):
        game = MonopolyGame(make_config())
        player = game.players[0]
        context = game._build_context(player, "pre_roll")
        unique_reason = "UNIQUE_RESEARCH_REASON_12345"
        game._record_decision(
            player,
            context,
            ManagementDecision(action="end_management", reason=unique_reason),
            error=None,
        )
        next_context = game._build_context(player, "post_roll")
        self.assertEqual(next_context.private_decision_history_text().count(unique_reason), 1)


class MockLLM:
    def prompt(self, message, schema=None, **kwargs):
        return None

    def respond(self, **kwargs):
        return None


class SequenceLLM(MockLLM):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompt_calls = 0

    def prompt(self, message, schema=None, **kwargs):
        self.prompt_calls += 1
        return self.responses.pop(0)


class MockKBench:
    llm = MockLLM()
    llms = {"opp1": MockLLM(), "opp2": MockLLM()}


class BenchmarkConfigTests(unittest.TestCase):
    def test_build_benchmark_config(self):
        kbench = MockKBench()
        config = build_benchmark_config(
            kbench, kbench.llm, opponent_model_ids=["opp1", "opp2"], seed=42
        )
        self.assertEqual(len(config.player_configs), 3)
        self.assertTrue(config.player_configs[0]["evaluated"])
        self.assertEqual(config.player_configs[1]["model_id"], "opp1")

    def test_benchmark_config_rejects_invalid_opponent_count(self):
        with self.assertRaises(ValueError):
            build_benchmark_config(MockKBench(), MockKBench.llm, opponent_model_ids=[])

    def test_benchmark_config_rejects_duplicate_and_missing_opponents(self):
        with self.assertRaises(RuntimeError):
            build_benchmark_config(
                MockKBench(), MockKBench.llm, opponent_model_ids=["opp1", "opp1"]
            )
        with self.assertRaises(RuntimeError):
            build_benchmark_config(
                MockKBench(), MockKBench.llm, opponent_model_ids=["missing"]
            )

    def test_benchmark_config_preserves_explicit_names_and_seed(self):
        config = build_benchmark_config(
            MockKBench(),
            MockKBench.llm,
            opponent_model_ids=["opp1"],
            player_names=["Research", "Control"],
            seed=123,
        )
        self.assertEqual(config.seed, 123)
        self.assertEqual(
            [spec["name"] for spec in config.player_configs],
            ["Research", "Control"],
        )

    def test_benchmark_config_accepts_game_config_overrides(self):
        config = build_benchmark_config(
            MockKBench(),
            MockKBench.llm,
            opponent_model_ids=["opp1"],
            max_rounds=1,
            max_turns=2,
            enable_auctions=False,
        )
        self.assertEqual(config.max_rounds, 1)
        self.assertEqual(config.max_turns, 2)
        self.assertFalse(config.enable_auctions)

    def test_benchmark_config_rejects_unknown_game_config_overrides(self):
        with self.assertRaisesRegex(TypeError, "not_a_config_field"):
            build_benchmark_config(
                MockKBench(),
                MockKBench.llm,
                opponent_model_ids=["opp1"],
                not_a_config_field=True,
            )


class DefaultLLMAgentTests(unittest.TestCase):
    def _auction_context(self):
        game = MonopolyGame(make_config())
        return game._build_context(game.players[0], "auction", space_index=1)

    def test_default_retry_settings_are_explicit(self):
        agent = DefaultLLMAgent(MockLLM())
        self.assertEqual(agent.max_retries, 5)
        self.assertEqual(agent.sleep_seconds, 1.0)

    def test_game_config_controls_raw_llm_retry_and_pause_settings(self):
        game = MonopolyGame(
            make_config(
                MockLLM(),
                MockLLM(),
                llm_max_attempts=2,
                llm_pause_seconds=0,
            )
        )
        self.assertEqual(game.players[0].agent.max_retries, 2)
        self.assertEqual(game.players[0].agent.sleep_seconds, 0)

    def test_invalid_bid_retries_and_never_becomes_zero_bid(self):
        llm = SequenceLLM([{}, {"bid_amount": 40, "reason": "explicit bid"}])
        agent = DefaultLLMAgent(llm, max_retries=2, sleep_seconds=0)
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            decision = agent.choose_auction_bid(self._auction_context())
        self.assertEqual(decision.bid_amount, 40)
        self.assertEqual(llm.prompt_calls, 2)
        self.assertIsNotNone(agent.decision_log[0]["error"])
        self.assertEqual(agent.decision_log[1]["decision"]["bid_amount"], 40)

    def test_repeated_invalid_bid_raises_without_fallback(self):
        llm = SequenceLLM([{}, {}])
        agent = DefaultLLMAgent(llm, max_retries=2, sleep_seconds=0)
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            with self.assertRaises(InvalidAgentError):
                agent.choose_auction_bid(self._auction_context())
        self.assertEqual(llm.prompt_calls, 2)
        self.assertTrue(all(item["decision"] is None for item in agent.decision_log))

    def test_string_bid_is_not_silently_coerced_to_integer(self):
        llm = SequenceLLM(
            [
                {"bid_amount": "40", "reason": "wrong type"},
                {"bid_amount": 40, "reason": "explicit integer bid"},
            ]
        )
        agent = DefaultLLMAgent(llm, max_retries=2, sleep_seconds=0)
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            decision = agent.choose_auction_bid(self._auction_context())
        self.assertEqual(decision.bid_amount, 40)
        self.assertEqual(llm.prompt_calls, 2)
        self.assertIsNotNone(agent.decision_log[0]["error"])

    def test_empty_reason_is_retried_for_research_logging(self):
        llm = SequenceLLM(
            [
                {"bid_amount": 25, "reason": ""},
                {"bid_amount": 25, "reason": "I value the property at $25."},
            ]
        )
        agent = DefaultLLMAgent(llm, max_retries=2, sleep_seconds=0)
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            decision = agent.choose_auction_bid(self._auction_context())
        self.assertEqual(decision.bid_amount, 25)
        self.assertEqual(llm.prompt_calls, 2)

    def test_auction_prompt_explains_one_shot_sealed_bid(self):
        llm = SequenceLLM(
            [{"bid_amount": 25, "reason": "This is my only sealed bid."}]
        )
        agent = DefaultLLMAgent(llm, max_retries=1, sleep_seconds=0)
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            agent.choose_auction_bid(self._auction_context())
        prompt = agent.decision_log[0]["prompt"]
        self.assertIn("exactly one private sealed bid", prompt)
        self.assertIn("Submit exactly one blind bid amount now", prompt)
        self.assertIn("you will not get another chance to change or raise your bid", prompt)
        self.assertIn("one private sealed bid only", prompt)

    def test_every_llm_decision_uses_a_distinct_isolated_visible_chat(self):
        llm = SequenceLLM(
            [
                {"bid_amount": 10, "reason": "first decision"},
                {"bid_amount": 20, "reason": "second decision"},
            ]
        )
        agent = DefaultLLMAgent(llm, max_retries=1, sleep_seconds=0)
        chat_calls = []

        def fake_new(*, name, orphan):
            chat_calls.append({"name": name, "orphan": orphan})
            return nullcontext()

        with patch("kaggle_benchmarks.chats.new", side_effect=fake_new):
            agent.choose_auction_bid(self._auction_context())
            agent.choose_auction_bid(self._auction_context())
        self.assertEqual(len(chat_calls), 2)
        self.assertNotEqual(chat_calls[0]["name"], chat_calls[1]["name"])
        self.assertTrue(all(call["orphan"] is False for call in chat_calls))

    def test_disabled_trading_is_not_advertised_to_llm(self):
        game = MonopolyGame(make_config(enable_trading=False))
        llm = SequenceLLM(
            [{"action": "end_management", "reason": "No management action needed."}]
        )
        agent = DefaultLLMAgent(llm, max_retries=1, sleep_seconds=0)
        agent.bind({}, game)
        context = game._build_context(game.players[0], "pre_roll")
        with patch("kaggle_benchmarks.chats.new", return_value=nullcontext()):
            agent.choose_management(context)
        prompt = agent.decision_log[0]["prompt"]
        self.assertIn("Trading is disabled for this game", prompt)
        self.assertNotIn("initiate one bounded trade", prompt)
        self.assertNotIn("provide one complete trade_proposal", prompt)


if __name__ == "__main__":
    unittest.main()
