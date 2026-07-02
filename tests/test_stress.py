import argparse
import random

from kbench_monopoly import (
    AuctionBid,
    BaseAgent,
    BuyDecision,
    DebtDecision,
    GameConfig,
    JailDecision,
    ManagementDecision,
    TradeProposal,
    TradeResponse,
)
from kbench_monopoly.runner import MonopolyGame


class RandomLegalAgent(BaseAgent):
    """Seeded agent that samples only actions advertised by the environment."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.decisions = 0

    def _check(self):
        self.game.validate_invariants()
        self.decisions += 1

    def choose_management(self, context):
        self._check()
        legal = list(context.legal_actions)
        chosen = self.rng.choice(legal)
        action = chosen["action"]
        if action == "propose_trade":
            targets = [
                player["name"]
                for player in context.all_players
                if player["alive"] and player["name"] != context.player_name
            ]
            if not targets or context.money < 1:
                return ManagementDecision(
                    action="end_management", reason="No valid bounded cash gift."
                )
            return ManagementDecision(
                action="propose_trade",
                trade_proposal=TradeProposal(
                    target_player=self.rng.choice(targets),
                    money_offered=self.rng.randint(1, min(context.money, 25)),
                ),
                reason="Exercise the bounded trade path.",
            )
        targets = chosen.get("allowed_targets", [])
        target = self.rng.choice(targets) if targets else None
        return ManagementDecision(
            action=action,
            target_property=target,
            reason="Seeded random legal management action.",
        )

    def choose_buy(self, context):
        self._check()
        return BuyDecision(
            will_buy=bool(self.rng.getrandbits(1)),
            reason="Seeded random buy decision.",
        )

    def choose_auction_bid(self, context):
        self._check()
        upper = min(context.money, context.auction_state["price"] * 2)
        return AuctionBid(
            bid_amount=self.rng.randint(0, upper),
            reason="Seeded random sealed bid.",
        )

    def choose_jail_action(self, context):
        self._check()
        action = self.rng.choice(context.legal_actions)["action"]
        return JailDecision(action=action, reason="Seeded random jail action.")

    def choose_debt_action(self, context):
        self._check()
        non_bankruptcy = [
            action
            for action in context.legal_actions
            if action["action"] != "declare_bankruptcy"
        ]
        chosen = (
            self.rng.choice(non_bankruptcy)
            if non_bankruptcy and self.rng.random() < 0.85
            else next(
                action
                for action in context.legal_actions
                if action["action"] == "declare_bankruptcy"
            )
        )
        targets = chosen.get("allowed_targets", [])
        target = self.rng.choice(targets) if targets else None
        return DebtDecision(
            action=chosen["action"],
            target_property=target,
            reason="Seeded random legal debt action.",
        )

    def choose_trade_response(self, context):
        self._check()
        return TradeResponse(
            action=self.rng.choice(["accept", "reject"]),
            reason="Seeded random bounded trade response.",
        )


def run_campaign(game_count: int) -> dict[str, int]:
    total_turns = 0
    total_decisions = 0
    completed = 0
    timed_out = 0
    for seed in range(game_count):
        player_count = 2 + seed % 3
        agents = [RandomLegalAgent(seed * 100 + seat) for seat in range(player_count)]
        config = GameConfig(
            player_configs=[
                {
                    "name": f"P{seat + 1}",
                    "agent": agent,
                    "evaluated": seat == 0,
                }
                for seat, agent in enumerate(agents)
            ],
            seed=seed,
            max_rounds=30,
            max_turns=60,
            starting_money=750 + (seed % 4) * 250,
            max_management_actions=4,
            context_public_history_limit=20,
            context_private_history_limit=10,
            enable_trading=seed % 4 == 0,
            enable_auctions=True,
        )
        game = MonopolyGame(config)
        game.start()
        game.validate_invariants(full_history=True)
        total_turns += game.turns_completed
        total_decisions += sum(agent.decisions for agent in agents)
        if game.timeout:
            timed_out += 1
        else:
            completed += 1
            assert game.winner is not None
        assert len([player for player in game.players if player.alive]) >= 1

    return {
        "games": game_count,
        "turns": total_turns,
        "decisions": total_decisions,
        "completed": completed,
        "timed_out": timed_out,
    }


def test_seeded_randomized_invariant_campaign():
    summary = run_campaign(10)
    assert summary["games"] == 10
    assert summary["decisions"] > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()
    print(run_campaign(args.games))
