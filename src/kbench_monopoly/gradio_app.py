from __future__ import annotations

import argparse
import html
import importlib
import json
import os
import pprint
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from .agent import (
    AgentContext,
    AuctionBid,
    BaseAgent,
    BuyDecision,
    DebtDecision,
    JailDecision,
    ManagementDecision,
    TradeProposal,
    TradeResponse,
)
from .agent.llm_default import DefaultLLMAgent
from .board_data import BOARD
from .config import GameConfig, generate_player_names
from .runner import run_monopoly_game


MAX_PLAYERS = 4
HUMAN_MODEL_ID = "human"
DEFAULT_MAX_ROUNDS = 10
DEFAULT_MAX_TURNS = 100
DEFAULT_MODEL_IDS = [
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4-nano-2026-03-17",
    "anthropic/claude-haiku-4-5@20251001",
    "google/gemini-3.5-flash",
]

RUN_STOP_EVENTS: dict[str, threading.Event] = {}
RUN_HUMAN_INPUTS: dict[str, queue.Queue] = {}
RUN_HUMAN_PENDING: dict[str, str] = {}


class GameStopped(Exception):
    """Raised at a safe game boundary after the user requests Stop."""


@dataclass
class GradioSnapshot:
    report_text: str
    result: dict[str, Any]


@dataclass
class HumanRequest:
    run_id: str
    request_id: str
    player_name: str
    phase: str
    prompt: str
    legal_actions: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "player_name": self.player_name,
            "phase": self.phase,
            "prompt": self.prompt,
            "legal_actions": list(self.legal_actions),
        }


class PersonalityLLMAgent(DefaultLLMAgent):
    """Default model adapter with a seat-specific strategy prompt."""

    def __init__(self, llm, personality: str = "", **kwargs):
        super().__init__(llm, **kwargs)
        self.personality = str(personality or "").strip()

    def _invoke_llm(self, context, prompt_content, schema_class, coerce_func, validate_func=None):
        if self.personality:
            prompt_content += (
                "\n\nCustom player prompt for this seat:\n"
                f"{self.personality}\n"
                "Use it for strategy and voice, but never override rules or legal actions."
            )
        return super()._invoke_llm(
            context, prompt_content, schema_class, coerce_func, validate_func
        )

    def get_log(self) -> dict[str, Any] | None:
        payload = super().get_log() or {}
        payload["custom_prompt"] = self.personality
        return payload


class HumanGradioAgent(BaseAgent):
    """Human agent that waits on the per-run Gradio input queue."""

    def __init__(self, run_id: str):
        self.run_id = str(run_id)

    def _ask(self, context: AgentContext) -> dict[str, Any]:
        ui = getattr(getattr(self, "game", None), "ui", None)
        if ui is None or not callable(getattr(ui, "request_human_decision", None)):
            raise RuntimeError("Human players require the Gradio app.")
        details = context.specific_context_text() or "Review the board and choose a legal action."
        request = HumanRequest(
            run_id=self.run_id,
            request_id=f"{self.run_id}:{context.player_name}:{context.phase}:{context.turn_number}:{time.time_ns()}",
            player_name=context.player_name,
            phase=context.phase,
            prompt=f"{context.player_name} — {context.phase}\n\n{details}",
            legal_actions=list(context.legal_actions),
        )
        ui.request_human_decision(request)
        inputs = RUN_HUMAN_INPUTS.setdefault(self.run_id, queue.Queue())
        while True:
            ui.check_stop()
            try:
                response = inputs.get(timeout=0.2)
            except queue.Empty:
                continue
            if (
                response.get("request_id") == request.request_id
                and response.get("phase") == context.phase
                and response.get("player_name") == context.player_name
            ):
                return response

    def choose_management(self, context):
        response = self._ask(context)
        proposal = response.get("trade_proposal")
        return ManagementDecision(
            action=response["action"],
            target_property=response.get("target_property"),
            trade_proposal=TradeProposal(**proposal) if proposal else None,
            reason="Human decision submitted through Gradio.",
        )

    def choose_buy(self, context):
        response = self._ask(context)
        return BuyDecision(
            will_buy=response["action"] == "buy",
            reason="Human buy decision submitted through Gradio.",
        )

    def choose_auction_bid(self, context):
        response = self._ask(context)
        return AuctionBid(
            bid_amount=int(response.get("bid_amount") or 0),
            reason="Human sealed bid submitted through Gradio.",
        )

    def choose_jail_action(self, context):
        response = self._ask(context)
        return JailDecision(
            action=response["action"],
            reason="Human jail decision submitted through Gradio.",
        )

    def choose_debt_action(self, context):
        response = self._ask(context)
        return DebtDecision(
            action=response["action"],
            target_property=response.get("target_property"),
            reason="Human debt decision submitted through Gradio.",
        )

    def choose_trade_response(self, context):
        response = self._ask(context)
        return TradeResponse(
            action=response["action"],
            reason="Human trade response submitted through Gradio.",
        )

    def model_name(self) -> str:
        return HUMAN_MODEL_ID

    def get_log(self) -> dict[str, Any] | None:
        return {"human": True}


class GradioGameUI:
    """Liar's Bar-style observer that streams snapshots and human requests."""

    def __init__(self, updates: queue.Queue, stop_event: threading.Event):
        self.updates = updates
        self.stop_event = stop_event
        self.report_text = ""

    def check_stop(self):
        if self.stop_event.is_set():
            raise GameStopped("Game stopped by user.")
        return False

    def report(self, text: str):
        self.check_stop()
        if text:
            self.report_text = str(text)

    def draw_game(self, snapshot: dict[str, Any]):
        self.check_stop()
        self.updates.put(GradioSnapshot(self.report_text, snapshot))

    def request_human_decision(self, request: HumanRequest):
        self.check_stop()
        RUN_HUMAN_PENDING[request.run_id] = request.request_id
        self.updates.put(request)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_kbench():
    local_src = workspace_root() / "kaggle-benchmarks" / "src"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    import kaggle_benchmarks as kbench

    try:
        if len(list(kbench.llms.keys())) > 0:
            return kbench
    except Exception as exc:
        raise RuntimeError("Unable to inspect kbench.llms.") from exc
    load_env_file(workspace_root() / "kaggle-benchmarks" / ".env")
    kbench = importlib.reload(kbench)
    if len(list(kbench.llms.keys())) == 0:
        raise RuntimeError("kbench.llms is empty after loading the local .env file.")
    return kbench


def model_choices(kbench) -> list[str]:
    choices = list(getattr(kbench, "llms", {}).keys()) if kbench else []
    if not choices:
        choices = list(DEFAULT_MODEL_IDS)
    if HUMAN_MODEL_ID not in choices:
        choices.append(HUMAN_MODEL_ID)
    return choices


def default_state(choices: list[str]) -> dict[str, Any]:
    names = generate_player_names(MAX_PLAYERS)
    models = [choice for choice in choices if choice != HUMAN_MODEL_ID] or [HUMAN_MODEL_ID]
    return {
        "player_count": 2,
        "evaluated_index": 0,
        "seed": "",
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "max_turns": DEFAULT_MAX_TURNS,
        "starting_money": 1500,
        "max_management_actions": 3,
        "enable_trading": False,
        "enable_auctions": True,
        "players": [
            {
                "name": names[index],
                "model": models[index % len(models)],
                "personality": "",
            }
            for index in range(MAX_PLAYERS)
        ],
    }


def collect_state(*values) -> dict[str, Any]:
    state = {
        "player_count": int(values[0]),
        "evaluated_index": int(values[1]),
        "seed": str(values[2] or "").strip(),
        "max_rounds": int(values[3]),
        "max_turns": int(values[4]),
        "starting_money": int(values[5]),
        "max_management_actions": int(values[6]),
        "enable_trading": bool(values[7]),
        "enable_auctions": bool(values[8]),
        "players": [],
    }
    cursor = 9
    for _ in range(MAX_PLAYERS):
        state["players"].append(
            {
                "name": str(values[cursor] or "").strip(),
                "model": str(values[cursor + 1] or "").strip(),
                "personality": str(values[cursor + 2] or "").strip(),
            }
        )
        cursor += 3
    state["players"] = state["players"][: state["player_count"]]
    return state


def validate_state(state: dict[str, Any]) -> None:
    if not 2 <= int(state["player_count"]) <= 4:
        raise ValueError("Player count must be between 2 and 4.")
    if not 0 <= int(state["evaluated_index"]) < int(state["player_count"]):
        raise ValueError("Evaluated seat must be active.")
    for key in ("max_rounds", "max_turns", "starting_money", "max_management_actions"):
        if int(state[key]) < 1:
            raise ValueError(f"{key} must be at least 1.")
    names = [player["name"] for player in state["players"]]
    if any(not name for name in names):
        raise ValueError("Every active player needs a name.")
    if len(set(names)) != len(names):
        raise ValueError("Active player names must be distinct.")
    if any(not player["model"] for player in state["players"]):
        raise ValueError("Every active player needs a model or human.")
    if state["seed"]:
        int(state["seed"])


def make_game_config(kbench, state: dict[str, Any]) -> GameConfig:
    validate_state(state)
    run_id = str(state.get("run_id") or "")
    player_configs = []
    for index, player in enumerate(state["players"]):
        model_id = player["model"]
        if model_id == HUMAN_MODEL_ID:
            agent = HumanGradioAgent(run_id)
        else:
            if model_id not in kbench.llms:
                raise RuntimeError(f"Model {model_id!r} is not available in kbench.llms.")
            agent = PersonalityLLMAgent(
                kbench.llms[model_id],
                personality=player.get("personality", ""),
            )
        player_configs.append(
            {
                "name": player["name"],
                "agent": agent,
                "model_id": model_id,
                "evaluated": index == state["evaluated_index"],
            }
        )
    return GameConfig(
        player_configs=player_configs,
        seed=int(state["seed"]) if state["seed"] else None,
        max_rounds=int(state["max_rounds"]),
        max_turns=int(state["max_turns"]),
        starting_money=int(state["starting_money"]),
        max_management_actions=int(state["max_management_actions"]),
        enable_trading=bool(state["enable_trading"]),
        enable_auctions=bool(state["enable_auctions"]),
        evaluated_player_name=state["players"][state["evaluated_index"]]["name"],
    )


def export_config_payload(state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    return {
        "settings": {
            key: state[key]
            for key in (
                "player_count",
                "evaluated_index",
                "seed",
                "max_rounds",
                "max_turns",
                "starting_money",
                "max_management_actions",
                "enable_trading",
                "enable_auctions",
            )
        },
        "players": [dict(player) for player in state["players"]],
    }


def game_config_from_export(payload: dict[str, Any], kbench) -> GameConfig:
    settings = dict(payload["settings"])
    state = settings | {"players": [dict(player) for player in payload["players"]]}
    if any(player.get("model") == HUMAN_MODEL_ID for player in state["players"]):
        raise RuntimeError("Exported scripts cannot recreate interactive human seats.")
    return make_game_config(kbench, state)


def export_config_code(state: dict[str, Any]) -> str:
    payload = export_config_payload(state)
    return (
        "import kbench_monopoly as monopoly\n"
        "from kbench_monopoly.gradio_app import game_config_from_export, load_kbench\n\n"
        "kbench = load_kbench()\n"
        "game_config_payload = "
        + pprint.pformat(payload, width=100, sort_dicts=False)
        + "\n\ngame_config = game_config_from_export(game_config_payload, kbench)\n"
        "result = monopoly.run_monopoly_game(game_config)\n"
        "print(result['final_rankings'])\n"
    )


def randomize_names(_player_count: int):
    names = generate_player_names(MAX_PLAYERS, seed=time.time_ns() % 1_000_000)
    return [gr.update(value=name) for name in names]


def update_player_visibility(count):
    return [gr.update(visible=index < int(count)) for index in range(MAX_PLAYERS)]


def short_model_name(model: str) -> str:
    value = str(model or "")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if "@" in value:
        value = value.split("@", 1)[0]
    return value[:24]


def dice_values(event: dict[str, Any]) -> tuple[int | None, int | None]:
    if "die1" in event and "die2" in event:
        return int(event["die1"]), int(event["die2"])
    total = event.get("dice")
    if isinstance(total, int) and 2 <= total <= 12:
        first = max(1, min(6, total // 2))
        second = max(1, min(6, total - first))
        return first, second
    return None, None


def render_die(value: int | None) -> str:
    if value not in {1, 2, 3, 4, 5, 6}:
        return "<span class='die unknown'>?</span>"
    return f"<span class='die die-{value}' aria-label='die {value}'><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span>"


PLAYER_COLORS = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6"]
GROUP_COLORS = {
    3: "#7c3f22", 4: "#69c5e8", 5: "#df5da8", 6: "#f28c28",
    7: "#e33b3b", 8: "#f2cf32", 9: "#2f9b62", 10: "#3156b8",
}


def player_money(player: dict[str, Any]) -> int:
    return int(player.get("money", player.get("final_money", 0)) or 0)


def player_position(player: dict[str, Any]) -> int:
    """Read a position from either a live snapshot or a final result."""
    return int(player.get("position", player.get("final_position", 0)) or 0)


def board_coordinates(index: int) -> tuple[int, int]:
    if index == 0:
        return 11, 11
    if 1 <= index <= 9:
        return 11, 11 - index
    if index == 10:
        return 11, 1
    if 11 <= index <= 19:
        return 21 - index, 1
    if index == 20:
        return 1, 1
    if 21 <= index <= 29:
        return 1, index - 19
    if index == 30:
        return 1, 11
    return index - 29, 11


def latest_event(result: dict[str, Any], event_type: str | None = None):
    for event in reversed(result.get("public_history", [])):
        if event_type is None or event.get("type") == event_type:
            return event
    return None


def current_player_name(result: dict[str, Any]) -> str | None:
    if result.get("end_reason") or result.get("winner") or result.get("timeout"):
        return None
    for event in reversed(result.get("public_history", [])):
        player = event.get("player")
        if player:
            return str(player)
    return None


def render_board(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None or not snapshot.result:
        return "<section class='mono-board empty'><div>Configure players and press Play.</div></section>"
    result = snapshot.result
    players = result.get("players", [])
    final_state = result.get("final_state") or {}
    properties = result.get("properties") or final_state.get("properties") or {}
    latest = latest_event(result) or {}
    latest_roll = latest_event(result, "roll") or {}
    positions: dict[int, list[int]] = {}
    for seat, player in enumerate(players):
        if player.get("alive", True):
            positions.setdefault(player_position(player), []).append(seat)

    spaces = []
    for space in BOARD:
        row, column = board_coordinates(space.index)
        state = properties.get(space.index, properties.get(str(space.index), {})) or {}
        owner = state.get("owner")
        houses = int(state.get("houses", 0) or 0)
        owner_style = f"--owner:{PLAYER_COLORS[int(owner) % 4]};" if owner is not None else ""
        color = GROUP_COLORS.get(space.group)
        color_bar = f"<span class='color-bar' style='background:{color}'></span>" if color else ""
        improvement = ""
        if houses == 5:
            improvement = "<span class='hotel'>H</span>"
        elif houses:
            improvement = "<span class='houses'>" + "".join("●" for _ in range(houses)) + "</span>"
        owner_house = ""
        if owner is not None:
            owner_index = int(owner)
            owner_name = players[owner_index].get("name", f"Player {owner_index}") if owner_index < len(players) else f"Player {owner_index}"
            owner_house = (
                f"<span class='owner-house' style='--owner:{PLAYER_COLORS[owner_index % 4]}' "
                f"title='Owned by {html.escape(owner_name)}' aria-label='Owned by {html.escape(owner_name)}'>&#8962;</span>"
            )
        tokens = "".join(
            f"<span class='token {'active' if latest.get('player') == players[seat].get('name') else ''}' "
            f"style='background:{PLAYER_COLORS[seat % 4]}' title='{html.escape(players[seat].get('name', ''))}'></span>"
            for seat in positions.get(space.index, [])
        )
        flags = " mortgaged" if state.get("mortgaged") else ""
        spaces.append(
            f"<article class='board-space{flags}' style='grid-row:{row};grid-column:{column};{owner_style}'>"
            f"{color_bar}<span class='space-index'>{space.index}</span>"
            f"<strong>{html.escape(space.name)}</strong>{improvement}"
            f"<small>{'$' + str(space.price) if space.price else space.space_type.replace('_', ' ')}</small>"
            f"{owner_house}<span class='tokens'>{tokens}</span></article>"
        )

    ranking = sorted(players, key=lambda player: -int(player.get("net_worth", player_money(player))))
    current_name = current_player_name(result)
    player_cards = "".join(
        f"<div class='mini-player {'current' if current_name == player.get('name') else ''} {'bankrupt' if not player.get('alive', True) else ''}' style='--player:{PLAYER_COLORS[seat % 4]}'>"
        f"<span class='avatar' style='background:{PLAYER_COLORS[seat % 4]}'></span>"
        f"<div class='player-ident'><b>{html.escape(player.get('name', ''))}</b><small>{html.escape(str(player.get('model_id', '')))}</small></div>"
        f"<div class='money'>${player_money(player):,}<small>NW ${int(player.get('net_worth', 0)):,}</small></div></div>"
        for seat, player in enumerate(players)
    )
    last_text = event_summary(latest) if latest else "Waiting for the game to start."
    card = latest if latest.get("type") == "card" else None
    overlay = (
        f"<div class='event-overlay card-pop'><b>{html.escape(str(card.get('deck', '')).replace('_', ' ').title())}</b>"
        f"<span>{html.escape(str(card.get('text', '')))}</span></div>"
        if card else ""
    )
    banner = ""
    if result.get("end_reason") or result.get("winner") or result.get("timeout"):
        if result.get("winner"):
            title = f"Winner — {result['winner']}"
        else:
            leader = ranking[0].get("name") if ranking else "No leader"
            title = f"Rank 1 — {leader}"
        subtitle = "Last player standing" if result.get("winner") else "Final net-worth leader"
        banner = (
            "<div class='winner-banner'>"
            "<span>GAME OVER</span>"
            f"<b>{html.escape(title)}</b>"
            f"<small>{html.escape(subtitle)}</small>"
            "</div>"
        )

    die1, die2 = dice_values(latest_roll)
    dice_html = (
        f"<div class='dice-pair {'shake' if latest.get('type') == 'roll' else ''}'>"
        f"{render_die(die1)}{render_die(die2)}"
        f"<small>latest roll {latest_roll.get('dice', 'â€”')}</small></div>"
    )

    return (
        "<section class='mono-board'>" + banner + "<div class='board-grid'>"
        + "".join(spaces)
        + "<div class='board-center'>"
        "<div class='brand'><span>KAGGLE BENCHMARK</span><b>MONOPOLY</b></div>"
        f"<div class='turn-chip'>Round {result.get('round', result.get('rounds_played', 0))} · "
        f"Turn {result.get('turn', result.get('turns_played', 0))}</div>"
        f"{dice_html}"
        f"<div class='dice {'shake' if latest.get('type') == 'roll' else ''}'><span>{latest_roll.get('dice', '—')}</span><small>latest roll</small></div>"
        f"<div class='latest-event'>{html.escape(last_text)}</div>"
        f"<div class='player-stack'>{player_cards}</div>"
        "</div></div>" + overlay + "</section>"
    )


def decision_source(result: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = result.get("decision_log")
    if isinstance(decisions, list):
        return decisions
    rows = []
    for player, logs in (result.get("private_logs") or {}).items():
        for item in logs:
            rows.append(dict(item) | {"player": player})
    return rows


def item_round(item: dict[str, Any], fallback: int = 0) -> int:
    return int(item.get("round_id", item.get("round", fallback)) or fallback)


def item_turn(item: dict[str, Any], fallback: int = 0) -> int:
    return int(item.get("turn_id", item.get("turn", fallback)) or fallback)


def round_groups(items: list[dict[str, Any]], fallback_round: int = 0) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item_round(item, fallback_round), []).append(item)
    return [(round_id, grouped[round_id]) for round_id in sorted(grouped, reverse=True)]


def render_decision_room(result: dict[str, Any]) -> str:
    players = {player.get("name"): player for player in result.get("players", [])}
    has_human = any(player.get("model_id") == HUMAN_MODEL_ID for player in players.values())
    ended = bool(result.get("end_reason") or result.get("winner") or result.get("timeout"))
    decisions = []
    for item in decision_source(result)[-60:]:
        player = str(item.get("player", "Agent"))
        if has_human and not ended and players.get(player, {}).get("model_id") != HUMAN_MODEL_ID:
            continue
        decisions.append(item)
    groups = []
    for round_id, items in round_groups(decisions, int(result.get("round", 0) or 0)):
        bubbles = []
        for item in reversed(items):
            player = str(item.get("player", "Agent"))
            decision = item.get("decision") or {}
            action = decision.get("action") if isinstance(decision, dict) else None
            if action is None and isinstance(decision, dict) and "will_buy" in decision:
                action = "buy" if decision["will_buy"] else "decline"
            if action is None and isinstance(decision, dict) and "bid_amount" in decision:
                action = f"bid ${decision['bid_amount']}"
            reason = str(item.get("reason") or "No reason recorded.")
            seat = next((index for index, p in enumerate(result.get("players", [])) if p.get("name") == player), 0)
            bubbles.append(
                f"<div class='decision-bubble'><div class='decision-head'>"
                f"<span style='background:{PLAYER_COLORS[seat % 4]}'></span><b>{html.escape(player)}</b>"
                f"<em>Turn {item_turn(item)} Â· {html.escape(str(item.get('phase', 'decision')).replace('_', ' '))}</em></div>"
                f"<strong>{html.escape(str(action or 'decision'))}</strong><p>{html.escape(reason)}</p></div>"
            )
        groups.append(
            f"<details class='round-group' open><summary>Round {round_id}<span>{len(items)} decisions</span></summary>"
            f"<div class='round-body'>{''.join(bubbles)}</div></details>"
        )
    if not groups:
        message = "AI reasoning is hidden during human play." if has_human and not ended else "No decisions yet."
        groups.append(f"<div class='side-empty'>{message}</div>")
    return "".join(groups)


def event_summary(event: dict[str, Any]) -> str:
    kind = event.get("type", "event")
    player = event.get("player", "")
    if kind == "roll":
        return f"{player} rolled {event.get('dice')} and landed on {event.get('space_name')}"
    if kind == "buy":
        return f"{player} bought {event.get('space_name')} for ${event.get('price')}"
    if kind == "auction_win":
        return f"{player} won {event.get('space_name')} for ${event.get('amount')}"
    if kind in {"rent_owed", "debt_paid", "tax_owed"}:
        return f"{player} — {kind.replace('_', ' ')} ${event.get('amount')}"
    if kind == "card":
        return f"{player} drew: {event.get('text')}"
    if kind == "game_over":
        return f"Game over — {event.get('reason')}"
    detail = event.get("space_name") or event.get("reason") or ""
    return f"{player} {kind.replace('_', ' ')} {detail}".strip()


def render_side(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None or not snapshot.result:
        return "<section class='side-shell'><div class='side-empty'>Game status and AI reasoning appear here.</div></section>"
    result = snapshot.result
    events = result.get("public_history", [])[-80:]
    feed_groups = []
    for round_id, items in round_groups(events, int(result.get("round", 0) or 0)):
        rows = "".join(
            f"<div class='feed-row'><span>{html.escape(str(event.get('type', 'event')).replace('_', ' '))}</span>"
            f"<p><em>T{item_turn(event)}</em>{html.escape(event_summary(event))}</p></div>"
            for event in reversed(items)
        )
        feed_groups.append(
            f"<details class='round-group feed-round' open><summary>Round {round_id}<span>{len(items)} events</span></summary>"
            f"<div class='round-body'>{rows}</div></details>"
        )
    feed = "".join(feed_groups) or "<div class='side-empty'>No public events yet.</div>"
    return (
        "<section class='side-shell'>"
        "<div class='side-panel decision-room'><h3><span>●</span> AI DECISION ROOM</h3>"
        f"<div class='scroll-area'>{render_decision_room(result)}</div></div>"
        "<div class='side-panel game-feed'><h3>GAME FEED</h3>"
        f"<div class='scroll-area'>{feed}</div></div></section>"
    )


def public_rows(snapshot: GradioSnapshot | None) -> list[list[Any]]:
    if snapshot is None:
        return []
    return [
        [event.get("round", snapshot.result.get("round")), event.get("type"), event.get("player", ""), event_summary(event)]
        for event in snapshot.result.get("public_history", [])
    ]


def state_has_human(state: dict[str, Any] | None) -> bool:
    return any(player.get("model") == HUMAN_MODEL_ID for player in (state or {}).get("players", []))


def public_result(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(result or {})
    for key in ("decision_log", "agent_logs", "private_logs"):
        payload.pop(key, None)
    return payload


def json_safe(value: Any) -> Any:
    """Return a Gradio JSON-compatible copy with string dictionary keys."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def result_json_update(result: dict[str, Any] | None, has_human: bool):
    result = result or {}
    ended = bool(result.get("end_reason") or result.get("winner") or result.get("timeout"))
    visible_result = result if ended or not has_human else public_result(result)
    return json_safe(visible_result)


def human_detail_control_updates(legal_actions: list[dict[str, Any]], action: str | None):
    selected = next((item for item in legal_actions if item.get("action") == action), None) or {}
    targets = sorted(selected.get("allowed_targets", []) or [])
    return (
        gr.update(choices=targets, value=None, visible=bool(targets)),
        gr.update(value=0, visible=True, interactive=action == "bid"),
        gr.update(value="", visible=action == "propose_trade"),
    )


def hidden_human_controls():
    return (
        gr.update(visible=False), gr.update(value=""),
        gr.update(choices=[], value=None), gr.update(choices=[], value=None, visible=False),
        gr.update(value=0, visible=True, interactive=False), gr.update(value="", visible=False),
        gr.update(interactive=False), gr.update(value=""),
    )


def visible_human_controls(request: HumanRequest):
    actions = [item["action"] for item in request.legal_actions]
    selected_action = actions[0] if actions else None
    target_update, amount_update, trade_update = human_detail_control_updates(request.legal_actions, selected_action)
    return (
        gr.update(visible=True),
        gr.update(value=request.prompt),
        gr.update(choices=actions, value=selected_action),
        target_update,
        amount_update,
        trade_update,
        gr.update(interactive=True),
        gr.update(value=""),
    )


def queue_human_submission(state_value, action, target, amount, trade_text):
    state_value = state_value or {}
    request = state_value.get("human_request") or {}
    run_id = str(request.get("run_id") or state_value.get("run_id") or "")
    inputs = RUN_HUMAN_INPUTS.get(run_id)
    if not request or inputs is None:
        return state_value, "No pending human decision.", gr.update(interactive=False)
    request_id = str(request.get("request_id") or "")
    pending_request_id = RUN_HUMAN_PENDING.get(run_id)
    if not request_id or pending_request_id != request_id:
        return state_value, "This prompt is no longer active. Please use the latest human decision panel.", gr.update(interactive=False)
    legal = {item["action"]: item for item in request.get("legal_actions", [])}
    if action not in legal:
        return state_value, "Select a legal action.", gr.update(interactive=True)
    allowed = legal[action].get("allowed_targets", [])
    target_value = None
    if allowed:
        target_value = int(target) if target not in (None, "") else None
        if target_value not in allowed:
            return state_value, f"Choose one of these targets: {allowed}", gr.update(interactive=True)
    trade_proposal = None
    if action == "propose_trade":
        try:
            trade_proposal = json.loads(str(trade_text or ""))
        except Exception as exc:
            return state_value, f"Trade JSON is invalid: {exc}", gr.update(interactive=True)
    if action == "bid" and amount in (None, ""):
        return state_value, "Enter a sealed bid amount. Use 0 only if you intentionally do not want to bid.", gr.update(interactive=True)
    inputs.put({
        "phase": request["phase"], "player_name": request["player_name"],
        "request_id": request_id,
        "action": action, "target_property": target_value,
        "bid_amount": int(amount or 0) if action == "bid" else None, "trade_proposal": trade_proposal,
    })
    RUN_HUMAN_PENDING.pop(run_id, None)
    state_value["human_request"] = None
    return state_value, "Submitted. Waiting for the gameâ€¦", gr.update(interactive=False)


CSS = """
:root { --bg:#090d14; --panel:#111827; --line:#273247; --muted:#94a3b8; --text:#edf2f7; --gold:#f5c451; }
html, body, gradio-app { background:#090d14 !important; }
.gradio-container { width:100vw !important; max-width:none !important; margin:0 !important; padding:18px 38px 24px !important; background:radial-gradient(circle at 20% 0%,#172235 0,#090d14 42%) !important; color:var(--text) !important; }
.gradio-container, .gradio-container .prose, .gradio-container label, .gradio-container input, .gradio-container textarea, .gradio-container select { color:var(--text); }
.mono-title { text-align:center; letter-spacing:.14em; }
.mono-title h1 { font-size:clamp(28px,4vw,52px); margin:0; color:white; text-shadow:0 0 28px rgba(245,196,81,.28); }
.mono-title p { color:var(--muted); margin:.4rem 0 1.2rem; }
.mono-board { position:relative; min-height:min(82vh,980px); padding:16px; border:1px solid var(--line); border-radius:20px; background:#080c12; overflow:hidden; box-shadow:0 28px 80px rgba(0,0,0,.45); }
.mono-board.empty { display:grid; place-items:center; color:var(--muted); }
.board-grid { width:min(100%,1500px); height:min(82vh,940px); margin:auto; display:grid; grid-template:repeat(11,1fr)/repeat(11,1fr); background:#0d1520; border:3px solid #334155; border-radius:14px; overflow:hidden; }
.board-space { --owner:transparent; position:relative; display:flex; flex-direction:column; align-items:center; justify-content:flex-start; gap:2px; padding:3px; overflow:hidden; background:linear-gradient(180deg,#1e332a 0%,#162820 100%); color:#e8fff4 !important; border:1px solid #315043; text-align:center; }
.board-space strong { color:#f0fff7 !important; font-size:clamp(7px,.58vw,11px); line-height:1.05; text-transform:uppercase; text-shadow:0 1px 2px rgba(0,0,0,.65); }
.board-space small { color:#9fc7b2 !important; font-size:clamp(6px,.48vw,9px); opacity:.95; margin-top:auto; }
.space-index { position:absolute; left:2px; bottom:2px; z-index:3; color:#8bb8a2 !important; font-size:7px; opacity:.82; }
.color-bar { width:calc(100% + 6px); height:17%; margin:-3px -3px 1px; border-bottom:1px solid rgba(0,0,0,.35); }
.board-space.mortgaged { filter:saturate(.15); opacity:.7; }
.owner-house { position:absolute; right:3px; bottom:3px; z-index:5; display:grid; place-items:center; width:15px; height:15px; border-radius:5px; background:var(--owner); color:white !important; font-size:12px; line-height:1; font-weight:900; border:1px solid rgba(255,255,255,.9); box-shadow:0 2px 7px rgba(0,0,0,.35); text-shadow:0 1px 2px rgba(0,0,0,.45); }
.houses,.hotel { position:absolute; top:1px; right:2px; z-index:6; color:#146c43; font-size:8px; letter-spacing:-1px; }
.hotel { background:#c53030; color:white; padding:0 3px; border-radius:2px; }
.tokens { position:absolute; inset:25% 3px 12%; display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:2px; pointer-events:none; }
.token { width:13px; height:13px; border-radius:50%; border:2px solid white; box-shadow:0 2px 8px #000; animation:token-arrive .45s ease-out; }
.token.active { animation:token-pulse .8s ease-in-out infinite alternate; }
.board-center { grid-area:2/2/11/11; display:flex; flex-direction:column; align-items:center; gap:10px; padding:18px; background:radial-gradient(circle,#162437,#0b111b 68%); color:white; overflow:hidden; }
.brand { display:flex; flex-direction:column; align-items:center; margin-top:2%; }
.brand span { color:var(--gold); font-size:10px; letter-spacing:.28em; }
.brand b { margin-top:5px; padding:4px 18px; background:#c72f38; border:3px solid white; font-size:clamp(20px,3vw,44px); letter-spacing:.1em; transform:rotate(-2deg); box-shadow:5px 6px 0 #6e1820; }
.turn-chip { padding:5px 12px; border:1px solid #3b4b62; border-radius:999px; color:#cbd5e1; font-size:12px; }
.dice { display:none !important; }
.dice-pair { display:grid; grid-template-columns:repeat(2,58px); justify-content:center; align-items:center; gap:12px; padding:10px 14px 18px; border:1px solid rgba(245,196,81,.45); border-radius:18px; background:linear-gradient(180deg,rgba(245,196,81,.18),rgba(15,23,42,.9)); box-shadow:0 16px 38px rgba(0,0,0,.38), inset 0 0 24px rgba(245,196,81,.08); position:relative; }
.dice-pair small { position:absolute; left:0; right:0; bottom:4px; text-align:center; color:#f8d778; font-size:8px; text-transform:uppercase; letter-spacing:.08em; }
.die { width:58px; height:58px; display:grid; grid-template:repeat(3,1fr)/repeat(3,1fr); gap:4px; padding:8px; border-radius:14px; background:radial-gradient(circle at 30% 25%,#fff7cc,#f6c453 62%,#b56b12); border:1px solid #fef3c7; box-shadow:0 9px 20px rgba(0,0,0,.42), inset 0 -4px 9px rgba(120,53,15,.45); }
.die.unknown { display:grid; place-items:center; color:#111827; font-size:28px; font-weight:900; }
.die i { display:block; width:9px; height:9px; align-self:center; justify-self:center; border-radius:50%; background:#172033; opacity:0; box-shadow:0 1px 1px rgba(255,255,255,.25); }
.die-1 i:nth-child(5), .die-2 i:nth-child(1), .die-2 i:nth-child(9), .die-3 i:nth-child(1), .die-3 i:nth-child(5), .die-3 i:nth-child(9), .die-4 i:nth-child(1), .die-4 i:nth-child(3), .die-4 i:nth-child(7), .die-4 i:nth-child(9), .die-5 i:nth-child(1), .die-5 i:nth-child(3), .die-5 i:nth-child(5), .die-5 i:nth-child(7), .die-5 i:nth-child(9), .die-6 i:nth-child(1), .die-6 i:nth-child(3), .die-6 i:nth-child(4), .die-6 i:nth-child(6), .die-6 i:nth-child(7), .die-6 i:nth-child(9) { opacity:1; }
.dice-pair.shake { animation:dice-shake .55s ease; }
.latest-event { min-height:32px; color:#cbd5e1; font-size:12px; text-align:center; max-width:520px; }
.player-stack { width:min(96%,980px); display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
.mini-player { display:grid; grid-template-columns:auto minmax(0,1fr) auto; align-items:center; gap:8px; padding:9px; background:rgba(17,24,39,.88); border:1px solid #334155; border-radius:10px; }
.mini-player .avatar { width:14px; height:14px; border-radius:50%; box-shadow:0 0 10px currentColor; }.mini-player b { display:block; font-size:11px; }.mini-player small { display:block; color:var(--muted); font-size:8px; white-space:normal; overflow-wrap:anywhere; line-height:1.15; }.mini-player .money { color:#86efac; font-weight:800; font-size:11px; text-align:right; }.mini-player.bankrupt { opacity:.4; filter:grayscale(1); }
.mini-player.current { position:relative; border-color:var(--player); background:linear-gradient(90deg,rgba(255,255,255,.04),rgba(17,24,39,.92)); box-shadow:0 0 0 1px var(--player), 0 0 22px color-mix(in srgb,var(--player) 54%,transparent), inset 0 0 18px rgba(255,255,255,.04); animation:current-player-pulse 1.05s ease-in-out infinite alternate; }
.mini-player.current::after { content:"CURRENT TURN"; position:absolute; top:-9px; right:10px; padding:2px 7px; border-radius:999px; background:var(--player); color:#07111f; font-size:7px; font-weight:900; letter-spacing:.08em; box-shadow:0 4px 12px rgba(0,0,0,.35); }
.mini-player.current .avatar { animation:token-pulse .8s ease-in-out infinite alternate; }
.winner-banner { position:absolute; z-index:30; top:50%; left:50%; transform:translate(-50%,-50%); display:flex; flex-direction:column; align-items:center; gap:6px; min-width:min(72%,560px); padding:26px 34px; border:2px solid rgba(245,196,81,.95); border-radius:26px; background:radial-gradient(circle at 50% 0%,rgba(245,196,81,.24),rgba(5,8,14,.97) 58%); color:var(--gold); text-align:center; font-weight:900; box-shadow:0 0 0 9999px rgba(2,6,12,.32), 0 30px 110px rgba(245,196,81,.25), inset 0 0 35px rgba(245,196,81,.1); animation:winner-in .8s cubic-bezier(.2,1.2,.2,1), winner-glow 1.5s ease-in-out infinite alternate; }
.winner-banner span { color:#fef3c7; font-size:12px; letter-spacing:.28em; }
.winner-banner b { font-size:clamp(30px,4.3vw,64px); line-height:1.02; text-shadow:0 0 26px rgba(245,196,81,.48); }
.winner-banner small { color:#cbd5e1; font-size:13px; letter-spacing:.08em; text-transform:uppercase; }
.event-overlay { position:absolute; z-index:18; left:50%; top:48%; transform:translate(-50%,-50%); width:min(70%,500px); padding:18px; border:1px solid var(--gold); border-radius:14px; background:rgba(9,13,20,.96); text-align:center; box-shadow:0 20px 80px #000; }.event-overlay b,.event-overlay span { display:block; }.event-overlay b { color:var(--gold); }.event-overlay span { margin-top:8px; }
.card-pop { animation:card-pop 1.8s ease both; }
.side-shell { display:grid; grid-template-rows:1.1fr .9fr; gap:12px; height:820px; }
.side-panel { min-height:0; padding:14px; border:1px solid var(--line); border-radius:16px; background:rgba(14,21,33,.94); box-shadow:0 18px 45px rgba(0,0,0,.25); }.side-panel h3 { margin:0 0 10px; color:#cbd5e1; font-size:12px; letter-spacing:.12em; }.side-panel h3 span { color:#34d399; }
.scroll-area { height:calc(100% - 30px); overflow:auto; padding-right:4px; scrollbar-color:#334155 transparent; }
.round-group { margin:0 0 10px; border:1px solid #26364d; border-radius:12px; background:rgba(7,12,21,.34); overflow:hidden; }
.round-group summary { cursor:pointer; display:flex; align-items:center; justify-content:space-between; gap:10px; padding:8px 10px; color:#dbeafe; font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; background:linear-gradient(90deg,rgba(51,65,85,.72),rgba(15,23,42,.36)); }
.round-group summary span { color:#93c5fd; font-size:9px; font-weight:700; letter-spacing:0; text-transform:none; }
.round-body { padding:8px; }
.decision-bubble { margin:0 0 10px; padding:10px; border:1px solid #2d3a50; border-radius:10px; background:#161f2f; animation:bubble-in .32s ease-out; }.decision-head { display:flex; align-items:center; gap:6px; }.decision-head span { width:9px; height:9px; border-radius:50%; }.decision-head b { font-size:11px; }.decision-head em { margin-left:auto; color:var(--muted); font-size:8px; font-style:normal; text-transform:uppercase; }.decision-bubble>strong { display:block; color:var(--gold); margin:5px 0 2px; font-size:11px; }.decision-bubble p { margin:0; color:#cbd5e1; font-size:10px; line-height:1.42; }
.feed-row { display:grid; grid-template-columns:82px 1fr; gap:8px; padding:8px 0; border-bottom:1px solid #202c3d; }.feed-row span { color:#7dd3fc; font-size:8px; text-transform:uppercase; }.feed-row p { margin:0; color:#cbd5e1; font-size:10px; line-height:1.35; }.feed-row p em { display:inline-block; min-width:24px; margin-right:6px; color:#fbbf24; font-style:normal; font-weight:800; }.side-empty { display:grid; place-items:center; height:100%; color:var(--muted); text-align:center; }
@keyframes token-arrive { from{transform:translateY(-14px) scale(.5);opacity:0} to{transform:none;opacity:1} } @keyframes token-pulse { to{transform:scale(1.35);box-shadow:0 0 15px white} } @keyframes current-player-pulse { from{filter:brightness(1)} to{filter:brightness(1.18)} }
@keyframes dice-shake { 20%{transform:rotate(10deg)}40%{transform:rotate(-9deg)}60%{transform:rotate(7deg)}80%{transform:rotate(-4deg)} }
@keyframes card-pop { 0%{opacity:0;transform:translate(-50%,-40%) scale(.7)}15%,75%{opacity:1;transform:translate(-50%,-50%) scale(1)}100%{opacity:0;transform:translate(-50%,-58%) scale(.95)} }
@keyframes bubble-in { from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none} } @keyframes winner-in { from{opacity:0;transform:translate(-50%,-50%) scale(.72)}to{opacity:1;transform:translate(-50%,-50%) scale(1)} } @keyframes winner-glow { to{box-shadow:0 0 0 9999px rgba(2,6,12,.4), 0 36px 130px rgba(245,196,81,.42), inset 0 0 42px rgba(245,196,81,.16)} }
@media(max-width:1100px){.side-shell{height:auto;grid-template-rows:420px 360px}.mono-board{min-height:680px}.board-grid{height:680px}.player-stack{grid-template-columns:1fr}.board-space strong{font-size:6px}}
"""


def build_app():
    try:
        kbench = load_kbench()
        choices = model_choices(kbench)
        load_status = f"Loaded {len(choices) - 1} benchmark models plus human play."
    except Exception as exc:
        kbench = None
        choices = model_choices(None)
        load_status = f"Models unavailable: {exc}"
    initial = default_state(choices)

    with gr.Blocks(title="Monopoly Kaggle Benchmark") as demo:
        state = gr.State(initial)
        latest_snapshot = gr.State(None)
        gr.HTML("<div class='mono-title'><h1>MONOPOLY</h1><p>Kaggle Benchmark Arena · observable decisions · replayable state</p></div>")

        with gr.Row():
            with gr.Column(scale=7):
                game_html = gr.HTML(render_board(None))
            with gr.Column(scale=3):
                with gr.Group(visible=False) as human_panel:
                    gr.Markdown("### Human Decision")
                    human_prompt = gr.Textbox(label="Current request", lines=4, interactive=False)
                    human_action = gr.Dropdown(label="Legal action", choices=[])
                    human_target = gr.Dropdown(label="Property target", choices=[], visible=False)
                    human_amount = gr.Number(label="Bid amount (auction only; 0 = no bid)", value=0, precision=0, visible=True, interactive=False)
                    human_trade = gr.Textbox(label="Trade proposal JSON", lines=5, visible=False)
                    human_submit = gr.Button("Submit Human Action", variant="primary", interactive=False)
                    human_error = gr.Markdown("")
                side_html = gr.HTML(render_side(None))

        with gr.Group(visible=True) as config_scene:
            gr.Markdown("## Game Config")
            gr.Markdown(load_status)
            with gr.Row():
                with gr.Column(scale=2):
                    player_count = gr.Slider(2, 4, value=initial["player_count"], step=1, label="Players")
                    evaluated_index = gr.Number(value=0, precision=0, label="Evaluated seat (0-based)")
                    seed = gr.Textbox(value="", label="Seed", placeholder="Blank = random")
                    with gr.Row():
                        max_rounds = gr.Number(value=initial["max_rounds"], precision=0, label="Max rounds")
                        max_turns = gr.Number(value=initial["max_turns"], precision=0, label="Max turns")
                    with gr.Row():
                        starting_money = gr.Number(value=1500, precision=0, label="Starting cash")
                        management_actions = gr.Number(value=3, precision=0, label="Actions / phase")
                    with gr.Row():
                        enable_trading = gr.Checkbox(value=False, label="Trading")
                        enable_auctions = gr.Checkbox(value=True, label="Sealed auctions")
                    randomize_btn = gr.Button("Randomize Names")
                with gr.Column(scale=3):
                    player_rows = []
                    name_inputs = []
                    model_inputs = []
                    personality_inputs = []
                    for index in range(MAX_PLAYERS):
                        with gr.Group(visible=index < initial["player_count"]) as row:
                            with gr.Row():
                                name = gr.Textbox(initial["players"][index]["name"], label=f"Seat {index + 1} name")
                                model = gr.Dropdown(choices, value=initial["players"][index]["model"], label=f"Seat {index + 1} model", allow_custom_value=True)
                            personality = gr.Textbox(label=f"Seat {index + 1} custom prompt", placeholder="Optional strategy, personality, or voice.", lines=2)
                        player_rows.append(row); name_inputs.append(name); model_inputs.append(model); personality_inputs.append(personality)
            validation = gr.Markdown("")
            with gr.Row():
                validate_btn = gr.Button("Validate Config")
                export_btn = gr.Button("Export Config")
                play_btn = gr.Button("Play", variant="primary")
            export_box = gr.Code(label="Copyable Python config", language="python", lines=20, value="# Click Export Config")

        with gr.Group(visible=False) as gameplay_scene:
            with gr.Row():
                stop_btn = gr.Button("Stop", variant="stop", interactive=False)
                restart_btn = gr.Button("Back to Config", interactive=False)
            with gr.Accordion("Detailed public log and result", open=False):
                public_log = gr.Dataframe(headers=["Round", "Type", "Actor", "Summary"], datatype=["number", "str", "str", "str"], label="Public Log")
                result_json = gr.JSON(label="Result JSON")

        collect_inputs = [player_count, evaluated_index, seed, max_rounds, max_turns, starting_money, management_actions, enable_trading, enable_auctions]
        for index in range(MAX_PLAYERS):
            collect_inputs.extend([name_inputs[index], model_inputs[index], personality_inputs[index]])

        player_count.change(update_player_visibility, player_count, player_rows)
        randomize_btn.click(randomize_names, player_count, name_inputs)

        def validate_only(*values):
            new_state = collect_state(*values)
            try:
                validate_state(new_state)
                return new_state, "### Config valid"
            except Exception as exc:
                return new_state, f"### Validation failed\n```text\n{exc}\n```"

        validate_btn.click(validate_only, collect_inputs, [state, validation])

        def export_only(*values):
            new_state = collect_state(*values)
            try:
                return new_state, "### Export ready", export_config_code(new_state)
            except Exception as exc:
                return new_state, f"### Export failed\n```text\n{exc}\n```", ""

        export_btn.click(export_only, collect_inputs, [state, validation, export_box])

        def run_game_stream(*values):
            new_state = collect_state(*values)
            has_human = state_has_human(new_state)
            if kbench is None:
                yield (new_state, None, gr.update(visible=True), gr.update(visible=False), render_board(None), "<section class='side-shell'><div class='side-empty'>kbench models are unavailable.</div></section>", [], {}, gr.update(interactive=False), gr.update(interactive=True), *hidden_human_controls())
                return
            run_id = str(time.time_ns())
            new_state["run_id"] = run_id
            new_state["human_request"] = None
            stop_event = threading.Event()
            RUN_STOP_EVENTS[run_id] = stop_event
            if has_human:
                RUN_HUMAN_INPUTS[run_id] = queue.Queue()
            try:
                config = make_game_config(kbench, new_state)
            except Exception as exc:
                RUN_STOP_EVENTS.pop(run_id, None); RUN_HUMAN_INPUTS.pop(run_id, None); RUN_HUMAN_PENDING.pop(run_id, None)
                yield (new_state, None, gr.update(visible=True), gr.update(visible=False), render_board(None), f"<section class='side-shell'><div class='side-empty'>{html.escape(str(exc))}</div></section>", [], {}, gr.update(interactive=False), gr.update(interactive=True), *hidden_human_controls())
                return

            updates: queue.Queue[Any] = queue.Queue()
            ui = GradioGameUI(updates, stop_event)
            result_box: dict[str, Any] = {}

            def target():
                try:
                    result_box["result"] = run_monopoly_game(config, ui=ui)
                except GameStopped:
                    result_box["stopped"] = True
                except Exception as exc:
                    result_box["error"] = exc
                finally:
                    updates.put(None)

            threading.Thread(target=target, daemon=True).start()
            last_snapshot = None
            yield (new_state, None, gr.update(visible=False), gr.update(visible=True), render_board(None), "<section class='side-shell'><div class='side-empty'>Game starting…</div></section>", [], {}, gr.update(interactive=True), gr.update(interactive=False), *hidden_human_controls())
            while True:
                item = updates.get()
                if item is None:
                    break
                if isinstance(item, HumanRequest):
                    new_state["human_request"] = item.to_payload()
                    yield (new_state, last_snapshot, gr.update(visible=False), gr.update(visible=True), render_board(last_snapshot), render_side(last_snapshot), public_rows(last_snapshot), result_json_update(last_snapshot.result if last_snapshot else {}, has_human), gr.update(interactive=True), gr.update(interactive=False), *visible_human_controls(item))
                    continue
                last_snapshot = item
                yield (new_state, last_snapshot, gr.update(visible=False), gr.update(visible=True), render_board(last_snapshot), render_side(last_snapshot), public_rows(last_snapshot), result_json_update(last_snapshot.result, has_human), gr.update(interactive=not stop_event.is_set()), gr.update(interactive=False), *hidden_human_controls())

            RUN_STOP_EVENTS.pop(run_id, None); RUN_HUMAN_INPUTS.pop(run_id, None); RUN_HUMAN_PENDING.pop(run_id, None)
            if result_box.get("stopped"):
                text = "Game stopped at the latest safe action boundary."
                yield (new_state, last_snapshot, gr.update(visible=False), gr.update(visible=True), render_board(last_snapshot), f"<section class='side-shell'><div class='side-empty'>{text}</div></section>", public_rows(last_snapshot), result_json_update(last_snapshot.result if last_snapshot else {}, has_human), gr.update(interactive=False), gr.update(interactive=True), *hidden_human_controls())
                return
            if "error" in result_box:
                text = html.escape(str(result_box["error"]))
                yield (new_state, last_snapshot, gr.update(visible=False), gr.update(visible=True), render_board(last_snapshot), f"<section class='side-shell'><div class='side-empty'>Game failed: {text}</div></section>", public_rows(last_snapshot), result_json_update(last_snapshot.result if last_snapshot else {}, has_human), gr.update(interactive=False), gr.update(interactive=True), *hidden_human_controls())
                return
            final_result = result_box.get("result", {})
            final_snapshot = GradioSnapshot(final_result.get("end_reason") or "Game ended", final_result)
            new_state["game_log"] = final_result.get("game_log")
            yield (new_state, final_snapshot, gr.update(visible=False), gr.update(visible=True), render_board(final_snapshot), render_side(final_snapshot), public_rows(final_snapshot), result_json_update(final_result, has_human), gr.update(interactive=False), gr.update(interactive=True), *hidden_human_controls())

        stream_outputs = [state, latest_snapshot, config_scene, gameplay_scene, game_html, side_html, public_log, result_json, stop_btn, restart_btn, human_panel, human_prompt, human_action, human_target, human_amount, human_trade, human_submit, human_error]
        play_btn.click(run_game_stream, collect_inputs, stream_outputs)

        def stop_game(state_value, snapshot_value):
            state_value = state_value or {}
            run_id = str(state_value.get("run_id") or "")
            if run_id in RUN_STOP_EVENTS:
                RUN_STOP_EVENTS[run_id].set()
            RUN_HUMAN_PENDING.pop(run_id, None)
            state_value["human_request"] = None
            return (state_value, snapshot_value, render_board(snapshot_value), "<section class='side-shell'><div class='side-empty'>Stop requested. Waiting for the current action boundary.</div></section>", public_rows(snapshot_value), gr.update(interactive=False), gr.update(interactive=False), *hidden_human_controls())

        stop_btn.click(stop_game, [state, latest_snapshot], [state, latest_snapshot, game_html, side_html, public_log, stop_btn, restart_btn, human_panel, human_prompt, human_action, human_target, human_amount, human_trade, human_submit, human_error])

        def refresh_human_action_controls(state_value, action):
            request = (state_value or {}).get("human_request") or {}
            legal_actions = request.get("legal_actions") or []
            return human_detail_control_updates(legal_actions, action)

        human_action.change(refresh_human_action_controls, [state, human_action], [human_target, human_amount, human_trade])

        def submit_human_action(state_value, action, target, amount, trade_text):
            state_value = state_value or {}
            request = state_value.get("human_request") or {}
            run_id = str(request.get("run_id") or state_value.get("run_id") or "")
            inputs = RUN_HUMAN_INPUTS.get(run_id)
            if not request or inputs is None:
                return state_value, "No pending human decision.", gr.update(interactive=False)
            request_id = str(request.get("request_id") or "")
            pending_request_id = RUN_HUMAN_PENDING.get(run_id)
            if not request_id or pending_request_id != request_id:
                return state_value, "This prompt is no longer active. Please use the latest human decision panel.", gr.update(interactive=False)
            legal = {item["action"]: item for item in request.get("legal_actions", [])}
            if action not in legal:
                return state_value, "Select a legal action.", gr.update(interactive=True)
            allowed = legal[action].get("allowed_targets", [])
            target_value = None
            if allowed:
                target_value = int(target) if target not in (None, "") else None
                if target_value not in allowed:
                    return state_value, f"Choose one of these targets: {allowed}", gr.update(interactive=True)
            trade_proposal = None
            if action == "propose_trade":
                try:
                    trade_proposal = json.loads(str(trade_text or ""))
                except Exception as exc:
                    return state_value, f"Trade JSON is invalid: {exc}", gr.update(interactive=True)
            if action == "bid" and amount in (None, ""):
                return state_value, "Enter a sealed bid amount. Use 0 only if you intentionally do not want to bid.", gr.update(interactive=True)
            inputs.put({
                "phase": request["phase"], "player_name": request["player_name"],
                "request_id": request_id,
                "action": action, "target_property": target_value,
                "bid_amount": int(amount or 0) if action == "bid" else None, "trade_proposal": trade_proposal,
            })
            RUN_HUMAN_PENDING.pop(run_id, None)
            state_value["human_request"] = None
            return state_value, "Submitted. Waiting for the game…", gr.update(interactive=False)

        human_submit.click(submit_human_action, [state, human_action, human_target, human_amount, human_trade], [state, human_error, human_submit])

        def restart(state_value):
            state_value = state_value or initial
            run_id = str(state_value.get("run_id") or "")
            if run_id in RUN_STOP_EVENTS:
                RUN_STOP_EVENTS[run_id].set()
            RUN_HUMAN_PENDING.pop(run_id, None)
            state_value["human_request"] = None
            return (state_value, None, gr.update(visible=True), gr.update(visible=False), render_board(None), render_side(None), [], {}, gr.update(interactive=False), gr.update(interactive=False), *hidden_human_controls())

        restart_btn.click(restart, state, stream_outputs)

    return demo


def main(argv: list[str] | None = None):
    warnings.filterwarnings(
        "ignore",
        message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
        module=r"gradio\.routes",
    )
    parser = argparse.ArgumentParser(description="Launch the kbench_monopoly Gradio app.")
    parser.add_argument("--share", default=False)
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args(argv)
    share = str(args.share).lower() in {"1", "true", "yes", "y"}
    app = build_app()
    try:
        app.launch(
            share=share,
            server_name=args.server_name,
            server_port=args.server_port,
            css=CSS,
        )
    finally:
        for event in list(RUN_STOP_EVENTS.values()):
            event.set()


if __name__ == "__main__":
    main()
