from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from faker import Faker


@dataclass
class GameConfig:
    """Experiment config for an explicit-player Monopoly run."""

    player_configs: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None
    max_rounds: int = 50
    max_turns: int = 200
    starting_money: int = 1500
    max_houses: int = 32
    max_hotels: int = 12
    go_salary: int = 200
    max_management_actions: int = 10
    llm_max_attempts: int = 5
    llm_pause_seconds: float = 1.0
    context_public_history_limit: int = 100
    context_private_history_limit: int = 50
    record_llm_prompts: bool = True
    enable_trading: bool = False
    enable_auctions: bool = True
    evaluated_player_name: str = "Evaluated"
    opponent_model_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 2 <= len(self.player_configs) <= 4:
            raise ValueError("Monopoly requires 2 to 4 player configurations.")

        names = []
        evaluated_names = []
        for index, spec in enumerate(self.player_configs):
            if not isinstance(spec, dict):
                raise TypeError(f"player_configs[{index}] must be a dictionary.")
            name = str(spec.get("name", "")).strip()
            if not name:
                raise ValueError(f"player_configs[{index}] must define a non-empty name.")
            if spec.get("agent") is None:
                raise ValueError(f"player_configs[{index}] must define an agent.")
            names.append(name)
            if spec.get("evaluated", False):
                evaluated_names.append(name)

        if len(set(names)) != len(names):
            raise ValueError("Player names must be distinct.")
        if len(evaluated_names) > 1:
            raise ValueError("At most one player may be marked as evaluated.")
        if evaluated_names and self.evaluated_player_name == "Evaluated":
            self.evaluated_player_name = evaluated_names[0]

        for field_name in (
            "max_rounds",
            "max_turns",
            "max_management_actions",
            "llm_max_attempts",
            "context_public_history_limit",
            "context_private_history_limit",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1.")
        for field_name in ("starting_money", "max_houses", "max_hotels", "go_salary"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative.")
        if self.llm_pause_seconds < 0:
            raise ValueError("llm_pause_seconds cannot be negative.")

    def with_updates(self, **updates) -> "GameConfig":
        data = {
            "player_configs": [dict(spec) for spec in self.player_configs],
            "seed": self.seed,
            "max_rounds": self.max_rounds,
            "max_turns": self.max_turns,
            "starting_money": self.starting_money,
            "max_houses": self.max_houses,
            "max_hotels": self.max_hotels,
            "go_salary": self.go_salary,
            "max_management_actions": self.max_management_actions,
            "llm_max_attempts": self.llm_max_attempts,
            "llm_pause_seconds": self.llm_pause_seconds,
            "context_public_history_limit": self.context_public_history_limit,
            "context_private_history_limit": self.context_private_history_limit,
            "record_llm_prompts": self.record_llm_prompts,
            "enable_trading": self.enable_trading,
            "enable_auctions": self.enable_auctions,
            "evaluated_player_name": self.evaluated_player_name,
            "opponent_model_ids": list(self.opponent_model_ids),
        }
        data.update(updates)
        return GameConfig(**data)


def build_benchmark_config(
    kbench,
    evaluated_llm,
    *,
    opponent_model_ids: list[str],
    seed: int | None = None,
    player_names: list[str] | None = None,
) -> GameConfig:
    """Build the default mixed-LLM benchmark configuration.

    The evaluated model is seat 1. Opponents are resolved from provider-qualified
    `kbench.llms` keys supplied by the task author. The task may provide one
    to three opponents, producing a two- to four-player game.
    """

    if not 1 <= len(opponent_model_ids) <= 3:
        raise ValueError(
            "Monopoly benchmark tasks must define 1 to 3 opponent models "
            f"for a 2- to 4-player game; got {len(opponent_model_ids)}."
        )
    available = dict(getattr(kbench, "llms", {}) or {})
    selected = _select_opponent_models(
        available,
        evaluated_llm,
        opponent_model_ids,
    )
    resolved_names = player_names or generate_player_names(len(selected) + 1, seed=seed)
    if len(resolved_names) != len(selected) + 1:
        raise ValueError(
            "player_names must contain exactly one name for the evaluated LLM "
            f"plus each opponent; expected {len(selected) + 1}, got "
            f"{len(resolved_names)}."
        )
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("player_names must be distinct.")
    player_configs = [
        {
            "name": resolved_names[0],
            "agent": evaluated_llm,
            "model_id": _model_name(evaluated_llm),
            "evaluated": True,
        }
    ]
    for index, (model_id, llm) in enumerate(selected, start=1):
        player_configs.append(
            {
                "name": resolved_names[index],
                "agent": llm,
                "model_id": model_id,
                "evaluated": False,
            }
        )
    return GameConfig(
        player_configs=player_configs,
        seed=seed,
        evaluated_player_name=resolved_names[0],
        opponent_model_ids=list(opponent_model_ids),
    )


def _select_opponent_models(available, evaluated_llm, opponent_model_ids):
    selected = []
    excluded_names = {_model_name(evaluated_llm)}

    for model_id in opponent_model_ids:
        llm = available.get(model_id)
        if llm is None:
            raise RuntimeError(
                f"Opponent model {model_id!r} is not available in kbench.llms."
            )
        if llm is evaluated_llm or model_id in excluded_names:
            raise RuntimeError(
                f"Opponent model {model_id!r} resolves to the evaluated model; "
                "choose distinct opponent models."
            )
        selected.append((model_id, llm))

    if len({model_id for model_id, _ in selected}) != len(selected):
        raise RuntimeError("Opponent model IDs must be distinct values.")
    return selected


def _model_name(llm) -> str:
    for attr in ("model", "name", "id"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return type(llm).__name__


def generate_player_names(count: int, *, seed: int | None = None) -> list[str]:
    if not 2 <= count <= 8:
        raise ValueError("Monopoly player name generation expects 2 to 8 players.")
    faker = Faker()
    if seed is not None:
        faker.seed_instance(seed)
    names = []
    attempts = 0
    while len(names) < count and attempts < 100:
        attempts += 1
        name = faker.first_name()
        if name not in names:
            names.append(name)
    if len(names) != count:
        fallback = ["Arthur", "Mika", "Noa", "Rin", "Alex", "Sam", "Leo", "Mia"]
        for name in fallback:
            if name not in names:
                names.append(name)
            if len(names) == count:
                break
    return names
