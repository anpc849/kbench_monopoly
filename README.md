# kbench_monopoly

Configurable **Monopoly** environment for Kaggle Benchmarks LLM agents.

The environment owns the game rules: board movement, property ownership,
sealed-bid auctions, rent, jail, Chance and Community Chest cards, property
management, debt resolution, bankruptcy, replay logs, and scoring. Agents only
provide decisions.

## Installation

```bash
git clone https://github.com/anpc849/kbench_monopoly
cd kbench_monopoly
pip install -e .
```

## Gradio App

Launch the interactive Gradio app:

```bash
kbench_monopoly_gradio --share True
```

The Gradio app expects `kaggle_benchmarks` to be importable. In Kaggle
notebooks, models are usually loaded automatically by the benchmark
environment. For local desktop testing, the app falls back to a local
`kaggle-benchmarks/.env` file and `kaggle-benchmarks/src` checkout when needed.

The UI supports:

- 2 to 4 players
- LLM players and optional human players
- custom or randomized player names
- custom extra prompt per player
- configurable maximum rounds and turns
- deterministic seeds for reproducible games
- dark-mode board visualization with live player positions
- nested round-by-round decision room and game feed
- replay inspection and exportable config code

## Basic Usage

```python
from kbench_monopoly import GameConfig, run_monopoly_game

config = GameConfig(
    player_configs=[
        {"name": "Mira", "agent": agent_1, "model_id": "model-a", "evaluated": True},
        {"name": "Theo", "agent": agent_2, "model_id": "model-b"},
    ],
    seed=None,
)

result = run_monopoly_game(game_config=config)
evaluated_won = result["winner"] == "Mira"
score = 1 if evaluated_won else 0
```

If a safety limit is reached before one player wins by bankruptcy elimination,
`score_evaluated_player()` scores the evaluated player by final net-worth rank.
The evaluated player receives a winning score when they are ranked first.

```python
from kbench_monopoly import score_evaluated_player

score = 1 if score_evaluated_player(result, config.evaluated_player_name) else 0
```

## Configuration

`GameConfig` controls one full game.

```python
GameConfig(
    player_configs=[...],
    seed=None,
    max_rounds=50,
    max_turns=200,
    starting_money=1500,
    max_houses=32,
    max_hotels=12,
    go_salary=200,
    max_management_actions=10,
    llm_max_attempts=5,
    llm_pause_seconds=1.0,
    context_public_history_limit=100,
    context_private_history_limit=50,
    record_llm_prompts=True,
    enable_trading=False,
    enable_auctions=True,
    evaluated_player_name="Evaluated",
    opponent_model_ids=[],
)
```

### `player_configs`

Required. A list of 2 to 4 player specs.

Each player spec supports:

- `name`: public player name. Names must be distinct.
- `agent`: object that implements the Monopoly agent methods, or a Kaggle
  Benchmarks LLM that can be wrapped by the default agent.
- `model_id`: optional public model label for UI and logs.
- `evaluated`: optional boolean. Mark the benchmarked player with `True`.
- `custom_prompt`: optional extra instruction appended to the default LLM
  prompt. It does not replace the rules prompt.

Example:

```python
player_configs = [
    {
        "name": "Mira",
        "agent": evaluated_llm,
        "model_id": "openai/gpt-5-mini",
        "evaluated": True,
        "custom_prompt": "Prioritize long-term net worth over short-term cash.",
    },
    {
        "name": "Theo",
        "agent": opponent_llm,
        "model_id": "google/gemini-3.5-flash",
    },
]
```

### `seed`

Optional. Use an integer for deterministic games, or `None` for fresh
randomization.

- `seed=123`: the same seed gives the same player names, deck order, dice rolls,
  and random events across runs when the same agents make the same decisions.
- `seed=None`: each run uses fresh randomness.

### `max_rounds` and `max_turns`

Safety limits. The game stops if either limit is reached before a single winner
is found. When this happens, final net worth is used for benchmark scoring.

### `starting_money`

Initial cash for each player. The default is the standard Monopoly starting
amount of `$1500`.

### `max_houses` and `max_hotels`

Bank supply limits for property improvements.

### `max_management_actions`

Maximum number of management actions a player may perform in one management
phase. This prevents unbounded repeated LLM calls.

### `llm_max_attempts` and `llm_pause_seconds`

Controls retry behavior for malformed or invalid LLM decisions.

### `context_public_history_limit` and `context_private_history_limit`

Limits how much public and private history is sent back to agents in each
decision context.

### `record_llm_prompts`

When `True`, raw LLM prompts and responses are recorded in `result["agent_logs"]`
for inspection.

### `enable_trading`

When `True`, players may propose bounded trades during management phases. The
default is `False` to keep benchmark behavior simpler and easier to analyze.

### `enable_auctions`

When `True`, declined unowned properties enter a sealed-bid auction. Each living
player submits exactly one private bid. Players do not see other bids and do not
get a second chance to raise.

### `evaluated_player_name`

Name of the evaluated player. This is mainly used in result summaries and
benchmark scoring.

### `opponent_model_ids`

Metadata used by `build_benchmark_config()` to record which configured Kaggle
Benchmarks models were selected as opponents.

## Result Logs

Detailed end-of-game inspection is available through:

- `result["players"]`: final cash, position, owned properties, alive state, and
  net worth
- `result["final_rankings"]`: stable final rank ordered by net worth
- `result["final_state"]`: players, rankings, properties, improvements,
  mortgages, and bank supply
- `result["public_history"]`: public gameplay events
- `result["decision_log"]`: private validated agent decisions and visibility
  metadata
- `result["agent_logs"]`: raw LLM attempts, prompts, responses, retries, and
  errors
- `result["game_log"]`: versioned replay events plus the complete final state

## Kaggle Benchmark Usage

`build_benchmark_config()` creates a 2 to 4 player game where the evaluated LLM
is the first player and the task author chooses 1 to 3 opponent models.

```python
import kaggle_benchmarks as kbench
import kbench_monopoly as monopoly

OPPONENT_MODEL_IDS = [
    "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "xai/grok-4.20-0309-non-reasoning",
]


@kbench.task(name="kbench-monopoly")
def kbench_monopoly(llm) -> int:
    config = monopoly.build_benchmark_config(
        kbench,
        llm,
        opponent_model_ids=OPPONENT_MODEL_IDS,
        player_names=["Mira", "Theo", "Nora", "Caleb"],
        seed=None,
    )
    result = monopoly.run_monopoly_game(game_config=config)
    return 1 if monopoly.score_evaluated_player(result, config.evaluated_player_name) else 0
```

`opponent_model_ids` must contain 1 to 3 distinct models. The game supports 2 to
4 total players, so an empty opponent list or more than 3 opponents raises a
setup error.

## Custom Agents

Custom agents can subclass `BaseAgent` or implement the same methods:

```python
from kbench_monopoly import BaseAgent, BuyDecision, ManagementDecision


class MyAgent(BaseAgent):
    def choose_buy(self, context):
        return BuyDecision(
            will_buy=True,
            reason="The property is affordable and improves my position.",
        )

    def choose_management(self, context):
        return ManagementDecision(
            action="end_management",
            reason="No useful management action is available.",
        )
```

Invalid custom-agent or LLM output raises an error instead of falling back to a
default action. This keeps behavior analysis faithful to the actual player.

## Local Development

```bash
python -m compileall -q src
python -m pytest
```

## Acknowledgements

This project reimplements Monopoly game mechanics for Kaggle Benchmarks and was
informed by the local base Monopoly implementation used during development.

This project is intended for research and benchmarking purposes only.
