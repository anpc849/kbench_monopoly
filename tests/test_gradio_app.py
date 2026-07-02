from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import gradio as gr
import pytest

from kbench_monopoly.agent import AgentContext
from kbench_monopoly.gradio_app import (
    HUMAN_MODEL_ID,
    RUN_HUMAN_INPUTS,
    RUN_HUMAN_PENDING,
    GameStopped,
    GradioGameUI,
    GradioSnapshot,
    HumanGradioAgent,
    HumanRequest,
    PersonalityLLMAgent,
    build_app,
    collect_state,
    default_state,
    export_config_code,
    export_config_payload,
    game_config_from_export,
    human_detail_control_updates,
    make_game_config,
    model_choices,
    player_position,
    public_result,
    queue_human_submission,
    render_board,
    render_decision_room,
    render_side,
    result_json_update,
    validate_state,
)


def configured_state(models=("provider/model-a", "provider/model-b")):
    state = default_state([*models, HUMAN_MODEL_ID])
    state["players"] = [
        {"name": "Ada", "model": models[0], "personality": "Protect cash."},
        {"name": "Lin", "model": models[1], "personality": "Prefer railroads."},
    ]
    state["seed"] = "42"
    return state


def test_model_choices_prefers_runtime_registry_and_adds_human():
    kbench = SimpleNamespace(llms={"live/model": object()})
    assert model_choices(kbench) == ["live/model", HUMAN_MODEL_ID]


def test_collect_and_validate_configuration():
    values = [2, 1, "17", 10, 100, 1500, 3, False, True]
    values += [
        "Ada", "model-a", "steady",
        "Lin", "model-b", "bold",
        "Unused 1", "model-c", "",
        "Unused 2", "model-d", "",
    ]
    state = collect_state(*values)
    validate_state(state)
    assert len(state["players"]) == 2
    assert state["players"][1]["personality"] == "bold"
    assert state["seed"] == "17"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.update(player_count=1), "between 2 and 4"),
        (lambda state: state.update(evaluated_index=3), "must be active"),
        (lambda state: state.update(max_rounds=0), "at least 1"),
        (lambda state: state["players"][1].update(name="Ada"), "distinct"),
        (lambda state: state["players"][0].update(model=""), "model or human"),
        (lambda state: state.update(seed="not-an-int"), "invalid literal"),
    ],
)
def test_invalid_configuration_is_rejected(mutate, message):
    state = configured_state()
    mutate(state)
    with pytest.raises((ValueError, TypeError), match=message):
        validate_state(state)


def test_make_game_config_preserves_models_prompts_and_evaluated_seat():
    llms = {"provider/model-a": object(), "provider/model-b": object()}
    state = configured_state()
    state["evaluated_index"] = 1
    config = make_game_config(SimpleNamespace(llms=llms), state)

    assert config.seed == 42
    assert config.max_rounds == 10
    assert config.evaluated_player_name == "Lin"
    assert [item["model_id"] for item in config.player_configs] == list(llms)
    assert isinstance(config.player_configs[0]["agent"], PersonalityLLMAgent)
    assert config.player_configs[0]["agent"].personality == "Protect cash."
    assert config.player_configs[1]["evaluated"] is True


def test_human_seat_uses_human_agent_and_export_rejects_it():
    state = configured_state()
    state["players"][1]["model"] = HUMAN_MODEL_ID
    config = make_game_config(
        SimpleNamespace(llms={"provider/model-a": object()}), state | {"run_id": "run-1"}
    )
    assert isinstance(config.player_configs[1]["agent"], HumanGradioAgent)

    payload = export_config_payload(state)
    with pytest.raises(RuntimeError, match="cannot recreate interactive human seats"):
        game_config_from_export(payload, SimpleNamespace(llms={}))


def test_export_round_trip_and_code_include_final_rankings():
    state = configured_state()
    kbench = SimpleNamespace(
        llms={"provider/model-a": object(), "provider/model-b": object()}
    )
    config = game_config_from_export(export_config_payload(state), kbench)
    code = export_config_code(state)

    assert config.seed == 42
    assert [item["name"] for item in config.player_configs] == ["Ada", "Lin"]
    assert "run_monopoly_game(game_config)" in code
    assert "final_rankings" in code


def test_decision_room_shows_exact_recorded_action_and_reason():
    result = {
        "players": [
            {"name": "Ada", "model_id": "model-a"},
            {"name": "Lin", "model_id": "model-b"},
        ],
        "decision_log": [
            {
                "player": "Ada",
                "phase": "buy",
                "decision": {"will_buy": False},
                "reason": "Cash reserve is more valuable than this utility.",
            }
        ],
    }
    rendered = render_decision_room(result)
    assert "decline" in rendered
    assert "Cash reserve is more valuable than this utility." in rendered


def test_decision_room_is_nested_by_round_and_turn():
    result = {
        "players": [{"name": "Ada", "model_id": "model-a"}],
        "decision_log": [
            {"player": "Ada", "round_id": 1, "turn_id": 1, "phase": "buy", "decision": {"will_buy": True}, "reason": "round one"},
            {"player": "Ada", "round_id": 2, "turn_id": 3, "phase": "auction", "decision": {"bid_amount": 80}, "reason": "round two"},
        ],
    }
    rendered = render_decision_room(result)
    assert "class='round-group'" in rendered
    assert rendered.index("Round 2") < rendered.index("Round 1")
    assert "Turn 3" in rendered


def test_game_feed_is_nested_by_round_and_turn():
    snapshot = GradioSnapshot(
        "live",
        {
            "players": [],
            "round": 2,
            "public_history": [
                {"round": 1, "turn": 1, "type": "turn_start", "player": "Ada"},
                {"round": 2, "turn": 3, "type": "roll", "player": "Ada", "dice": 6, "space_name": "Chance"},
            ],
        },
    )
    rendered = render_side(snapshot)
    assert "Round 2" in rendered and "Round 1" in rendered
    assert rendered.index("Round 2") < rendered.index("Round 1")
    assert "<em>T3</em>" in rendered


def test_board_highlights_current_player_card_until_game_end():
    result = {
        "players": [
            {"name": "Sarah", "model_id": "model-a", "money": 1380, "position": 8, "net_worth": 1500, "alive": True},
            {"name": "Patrick", "model_id": "model-b", "money": 1500, "position": 0, "net_worth": 1500, "alive": True},
        ],
        "round": 1,
        "turn": 2,
        "properties": {},
        "public_history": [
            {"type": "turn_start", "player": "Patrick", "round": 1, "turn": 2},
            {"type": "roll", "player": "Patrick", "round": 1, "turn": 2, "dice": 8, "die1": 6, "die2": 2, "space_name": "Vermont Avenue", "position": 8},
        ],
    }
    rendered = render_board(GradioSnapshot("live", result))
    assert "mini-player current" in rendered
    assert "--player:#f59e0b" in rendered

    result["end_reason"] = "max_rounds"
    ended = render_board(GradioSnapshot("done", result))
    assert "mini-player current" not in ended


def test_ai_reasoning_privacy_during_human_play_and_release_after_end():
    result = {
        "players": [
            {"name": "Human", "model_id": HUMAN_MODEL_ID},
            {"name": "Bot", "model_id": "model-a"},
        ],
        "decision_log": [
            {"player": "Bot", "phase": "buy", "decision": {"will_buy": True}, "reason": "private bot reason"},
            {"player": "Human", "phase": "buy", "decision": {"will_buy": False}, "reason": "human reason"},
        ],
        "agent_logs": {"Bot": {"prompt": "private prompt"}},
    }
    live_html = render_decision_room(result)
    assert "private bot reason" not in live_html
    assert "human reason" in live_html
    assert "agent_logs" not in public_result(result)
    assert "decision_log" not in result_json_update(result, has_human=True)

    result["end_reason"] = "max_rounds"
    assert "private bot reason" in render_decision_room(result)
    assert result_json_update(result, has_human=True)["agent_logs"]


def test_human_detail_controls_follow_selected_action_only():
    legal_actions = [
        {"action": "end_management"},
        {"action": "mortgage", "allowed_targets": [5, 15]},
        {"action": "bid"},
        {"action": "propose_trade"},
    ]
    target, amount, trade = human_detail_control_updates(legal_actions, "end_management")
    assert target["visible"] is False
    assert amount["visible"] is True
    assert amount["interactive"] is False
    assert trade["visible"] is False

    target, amount, trade = human_detail_control_updates(legal_actions, "mortgage")
    assert target["visible"] is True and target["choices"] == [5, 15]
    assert amount["visible"] is True
    assert amount["interactive"] is False
    assert trade["visible"] is False

    target, amount, trade = human_detail_control_updates(legal_actions, "bid")
    assert target["visible"] is False
    assert amount["visible"] is True
    assert amount["value"] == 0
    assert amount["interactive"] is True
    assert trade["visible"] is False


def test_human_submission_rejects_stale_request_id_before_auction_queue():
    run_id = "stale-submit"
    RUN_HUMAN_INPUTS[run_id] = queue.Queue()
    RUN_HUMAN_PENDING[run_id] = "new-auction-request"
    state = {
        "run_id": run_id,
        "human_request": {
            "run_id": run_id,
            "request_id": "old-buy-request",
            "player_name": "Ada",
            "phase": "auction",
            "legal_actions": [{"action": "bid"}],
        },
    }
    try:
        _, message, update = queue_human_submission(state, "bid", None, 0, "")
        assert "no longer active" in message
        assert RUN_HUMAN_INPUTS[run_id].empty()
        assert update["interactive"] is False
    finally:
        RUN_HUMAN_INPUTS.pop(run_id, None)
        RUN_HUMAN_PENDING.pop(run_id, None)


def test_human_submission_requires_explicit_bid_for_current_auction_request():
    run_id = "fresh-auction-submit"
    RUN_HUMAN_INPUTS[run_id] = queue.Queue()
    RUN_HUMAN_PENDING[run_id] = "auction-request"
    state = {
        "run_id": run_id,
        "human_request": {
            "run_id": run_id,
            "request_id": "auction-request",
            "player_name": "Ada",
            "phase": "auction",
            "legal_actions": [{"action": "bid"}],
        },
    }
    try:
        _, message, update = queue_human_submission(dict(state), "bid", None, None, "")
        assert "Enter a sealed bid amount" in message
        assert update["interactive"] is True
        assert RUN_HUMAN_INPUTS[run_id].empty()

        next_state, message, update = queue_human_submission(state, "bid", None, 0, "")
        assert "Submitted" in message
        assert update["interactive"] is False
        queued = RUN_HUMAN_INPUTS[run_id].get_nowait()
        assert queued["request_id"] == "auction-request"
        assert queued["bid_amount"] == 0
        assert next_state["human_request"] is None
        assert run_id not in RUN_HUMAN_PENDING
    finally:
        RUN_HUMAN_INPUTS.pop(run_id, None)
        RUN_HUMAN_PENDING.pop(run_id, None)


def test_final_result_schema_renders_rank_owner_and_positions():
    result = {
        "players": [
            {"name": "Ada", "model_id": "a", "final_money": 1200, "final_position": 1, "net_worth": 1600, "alive": True},
            {"name": "Lin", "model_id": "b", "final_money": 900, "final_position": 20, "net_worth": 900, "alive": True},
        ],
        "timeout": True,
        "end_reason": "max_rounds",
        "rounds_played": 10,
        "turns_played": 20,
        "final_state": {"properties": {1: {"owner": 0, "houses": 2, "mortgaged": False}}},
        "public_history": [],
    }
    rendered = render_board(GradioSnapshot("done", result))
    assert player_position(result["players"][0]) == 1
    assert "Rank 1" in rendered and "Ada" in rendered
    assert "--owner:#60a5fa" in rendered
    assert "class='owner-house'" in rendered
    assert "Owned by Ada" in rendered
    assert rendered.count("<span class='token ") == 2


def test_board_renders_two_dice_and_full_model_id():
    result = {
        "players": [
            {
                "name": "Ada",
                "model_id": "provider/very-long-model-id-that-should-not-be-truncated",
                "money": 1200,
                "position": 1,
                "net_worth": 1600,
                "alive": True,
            },
            {"name": "Lin", "model_id": "short", "money": 900, "position": 20, "net_worth": 900, "alive": True},
        ],
        "round": 1,
        "turn": 1,
        "properties": {},
        "public_history": [
            {"type": "roll", "player": "Ada", "dice": 7, "die1": 3, "die2": 4, "space_name": "Chance", "position": 7}
        ],
    }
    rendered = render_board(GradioSnapshot("live", result))
    assert "class='dice-pair shake'" in rendered
    assert "die die-3" in rendered
    assert "die die-4" in rendered
    assert "very-long-model-id-that-should-not-be-truncated" in rendered


def test_result_json_update_converts_integer_keys_for_gradio_json():
    payload = {
        "round": 1,
        "properties": {15: {"owner": 0}},
        "private_logs": {0: [{"reason": "hidden while active"}]},
    }
    visible = result_json_update(payload, has_human=False)
    assert "15" in visible["properties"]
    assert 15 not in visible["properties"]
    assert "0" in visible["private_logs"]


def test_streaming_ui_emits_snapshots_and_honors_stop():
    updates = queue.Queue()
    stop_event = threading.Event()
    ui = GradioGameUI(updates, stop_event)
    ui.report("rolled")
    ui.draw_game({"round": 1})
    snapshot = updates.get_nowait()
    assert snapshot.report_text == "rolled"
    assert snapshot.result == {"round": 1}

    stop_event.set()
    with pytest.raises(GameStopped, match="stopped by user"):
        ui.check_stop()


def test_gradio_ui_registers_pending_human_request_id():
    updates = queue.Queue()
    stop_event = threading.Event()
    ui = GradioGameUI(updates, stop_event)
    request = HumanRequest(
        run_id="pending-run",
        request_id="request-1",
        player_name="Human",
        phase="auction",
        prompt="Bid once.",
        legal_actions=[{"action": "bid"}],
    )
    try:
        ui.request_human_decision(request)
        assert RUN_HUMAN_PENDING["pending-run"] == "request-1"
        assert updates.get_nowait() is request
    finally:
        RUN_HUMAN_PENDING.pop("pending-run", None)


def test_human_agent_waits_for_matching_gradio_response_id():
    run_id = "human-test"
    RUN_HUMAN_INPUTS[run_id] = queue.Queue()

    class FakeUI:
        def check_stop(self):
            return False

        def request_human_decision(self, request):
            RUN_HUMAN_INPUTS[run_id].put(
                {
                    "request_id": "stale-request",
                    "phase": request.phase,
                    "player_name": request.player_name,
                    "action": "mortgage",
                }
            )
            RUN_HUMAN_INPUTS[run_id].put(
                {
                    "request_id": request.request_id,
                    "phase": request.phase,
                    "player_name": request.player_name,
                    "action": "end_management",
                }
            )

    agent = HumanGradioAgent(run_id)
    agent.game = SimpleNamespace(ui=FakeUI())
    context = AgentContext(
        player_name="Human",
        phase="pre_roll",
        turn_number=1,
        round_number=1,
        position=0,
        position_name="GO",
        money=1500,
        properties_owned=[],
        in_jail=False,
        has_cc_jail_card=False,
        has_chance_jail_card=False,
        all_players=[],
        property_ownership=[],
        legal_actions=[{"action": "end_management"}],
    )
    try:
        decision = agent.choose_management(context)
        assert decision.action == "end_management"
        assert "Gradio" in decision.reason
    finally:
        RUN_HUMAN_INPUTS.pop(run_id, None)


def test_app_builds_all_components_without_starting_a_server():
    app = build_app()
    assert isinstance(app, gr.Blocks)
    assert len(app.blocks) >= 70
