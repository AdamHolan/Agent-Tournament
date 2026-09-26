# Dungeon Tournament Game Specification

Status: **Implemented core rules; some protocol sections retain historical design proposals**  
Ruleset: **0.1**  
Protocol: **0.1**

This document records the game's rules and the original tournament design. For
the current runnable commands and study results, see [README.md](README.md) and
[RESULTS.md](RESULTS.md). Where a design proposal below differs from the
implementation, the code in `dungeon/` is authoritative.

## 1. Purpose

Persistent AI agents repeatedly play private single-player dungeon runs. After
every agent finishes a tournament round, results are published and the agents
may participate in a limited public discussion. Agents retain bounded private
memory between rounds.

The deterministic game engine (the GM) owns all rules, state, randomness, and
scoring. Models receive observations and propose actions; they never determine
whether an action succeeds or report their own authoritative state.

Principles:

- Agents never share a live dungeon instance.
- Matched agents encounter the same generated floor at a given branch address.
- Model output is untrusted and validated.
- State is compact, serializable, and independent of any model framework.
- Run artifacts retain game events and behavioral measurements for inspection.
- Correctness and clear boundaries come before low-level optimization.

## 2. Terminology and lifecycle

- **Experiment:** Configured agents, rounds, rules, and seeds.
- **Tournament round:** One private run per agent, then results and discussion.
- **Run:** One attempt beginning on floor zero.
- **Dungeon:** The ordered, conceptually unbounded sequence of floors.
- **Floor:** One rectangular 2D tile grid.
- **Depth:** Zero-based floor number; the first floor is depth `0`.
- **Turn:** One player action followed by world resolution.
- **Skellie:** An ordinary enemy.
- **Grim Skellie:** An enforcement enemy for bounded failure conditions.

```text
private context from prior rounds
      ↓
independent private runs
      ↓
leaderboard and one combined review per guest
      ↓
optional notebook replacement and public post
      ↓
publish posts and checkpoint the round
      ↓
next tournament round
```

Runs may execute sequentially or concurrently. Results are held until all
guests finish their private runs, and messages are published only after all
post-run reviews finish, so completion order does not change what another
participant can observe in that round.

## 3. Coordinates and dimensions

- `(0, 0)` is northwest.
- `x` increases east; `y` increases south.
- Movement is north, east, south, or west only.
- Floor width and height are both `7 + floor(depth / 100)`.

| Depth | Dimensions |
|---:|---:|
| 0–99 | 7×7 |
| 100–199 | 8×8 |
| 200–299 | 9×9 |

Width and height remain separate internal fields even though initial floors are
square.

## 4. Tiles, features, and actors

Terrain:

- `floor`: traversable.
- `pillar`: impassable; does not block vision.

Features:

- `stairs_down`: entering it immediately advances to the next floor.
- `heart`: entering it consumes it and restores health to maximum.

Actors:

- `player`
- `skellie`
- `grim_skellie`

Initial player, stairs, heart, pillars, and skellies occupy distinct tiles.
Actor collision is permitted only as the transient event that starts combat.

## 5. Floor generation

A shared immutable `FloorDefinition` is generated for each map seed and branch
path. Matched guests use the same definitions, while mutable entity health and
consumption state are private to each run.

Generation order is fixed:

1. Determine dimensions.
2. Place player spawn and two labeled downstairs (`A` and `B`).
3. Place pillars.
4. Determine and place a heart.
5. Determine and place skellies.
6. Validate invariants; deterministically regenerate on failure.

No minimum player-to-exit distance is required. Entities may be adjacent,
except a skellie cannot initially be one Manhattan tile from the player.

### Pillars

Pillar count is uniformly selected from `0` through
`min(width, height) - 1`, inclusive. Positions are uniformly selected from empty
tiles.

This count limit does not prove connectivity. Cardinal breadth-first search must
verify that both exits are reachable from player spawn. Invalid layouts are
regenerated.

### Heart

Each floor independently has a `33%` probability of one heart. Its tile must be
reachable from player spawn when actors are ignored. A consumed heart does not
respawn during that run.

### Skellies

The spawn probability at depth `d` is interpreted as:

```text
min(0.50 + 0.01 × d, 1.00)
```

If the check succeeds, count is `1 + floor(d / 100)`; otherwise it is zero.

| Depth | Spawn chance | Count on success |
|---:|---:|---:|
| 0 | 50% | 1 |
| 25 | 75% | 1 |
| 50–99 | 100% | 1 |
| 100–199 | 100% | 2 |
| 200–299 | 100% | 3 |

By default, skellie presence and initial positions are derived from the run seed
and branch address. Stable geometry, hearts, and exits remain map-seeded. The
operator may use `--no-run-seeded-skellies` to restore map-seeded placement.
Skellies must be placed on the reachable component and not adjacent to spawn.

### Generation invariants

- All coordinates are in bounds.
- No prohibited overlap exists.
- Spawn and both downstairs are traversable and connected.
- Hearts and skellies are on the spawn's reachable component.
- Entity counts match their generation decisions.

## 6. Randomness

Version 0.1 uses two conceptual random streams:

- **Map seed:** shared floor terrain, exits, and hearts, addressed by branch
  path rather than depth alone. It also determines skellie placement when
  run-seeded skellies are disabled.
- **Run seed:** private outcomes such as attack hit rolls and, by default,
  skellie presence and placement. The social runner matches it by guest slot
  and round across conditions.

Exact reproduction across engine versions is deferred, but logs must store the
seeds, engine version, and ruleset version. Rendering, logging, and observation
construction must never consume random values.

## 7. Visibility and remembered terrain

A tile is visible when its Manhattan distance from the player is at most `2`:

```text
abs(x - player_x) + abs(y - player_y) <= 2
```

Pillars do not block vision. Floor dimensions are always known.

Within a run:

- Visible terrain, features, and actors are shown.
- Previously seen terrain remains mapped.
- Dynamic actors outside current vision are not shown.
- A consumed heart remains known as a discovered but absent feature.
- Undiscovered tiles remain fogged.

The engine stores revealed coordinates; the model is not responsible for
reconstructing them.

## 8. Statistics and ordinary skellie behavior

Player:

```text
maximum and starting health: 3
hit probability: 75%
damage: 1
```

Ordinary skellie:

```text
health: 2
hit probability: 50%
damage: 1
vision radius: 2 Manhattan tiles
```

Skellies begin dormant. Seeing the player permanently aggros them for that
floor. Aggroed skellies move one cardinal tile along a deterministic shortest
path toward the player. tie order: north, east, south, west.
They cannot cross pillars, downstairs, or each other. Multiple skellies act in
stable actor-ID order.

Death occurs when health is less than or equal to zero.

## 9. Legal actions

Version 0.1 accepts a primitive movement action:

```json
{"action":"move","direction":"north"}
```

Direction is `north`, `east`, `south`, or `west`.

It also accepts an optional literal movement sequence:

```json
{"action":"move_sequence","directions":["east","east","south"]}
```

A sequence contains 1 to 12 directions chosen by the agent. It is a compact
transport form, not a pathfinding command: the GM applies each submitted
direction through the ordinary turn-resolution rules, and each executed
direction consumes one turn and therefore incurs the normal one-point turn
penalty. Model decisions have no separate energy currency or score cost; a
sequence is only an inference-transport optimization. The GM never chooses,
repairs, or extends a route.

Execution stops after the first primitive turn that produces a new decision:

- a skellie becomes visible;
- downstairs or an available heart is newly discovered;
- a heart is consumed or combat begins;
- movement is illegal;
- the floor changes; or
- the run ends.

The next observation reports the submitted, executed, and unexecuted directions
and the interruption reason as a `move_sequence_resolved` event. A completed
sequence has a null interruption reason. The agent may always use a one-tile
`move` instead.

- Moving out of bounds or into a pillar is illegal and consumes the turn.
- Moving onto a skellie initiates combat.
- Moving onto a heart consumes it automatically.
- Moving onto downstairs completes the floor automatically.
- `wait` is not legal.
- Individual combat actions are reserved for a later strategic combat system.

## 10. Turn resolution

Outside combat, one turn resolves atomically:

1. Deliver observation and action schema.
2. Receive and validate one action. For `move_sequence`, repeat the remaining
   steps once per primitive direction until the sequence completes or interrupts.
3. Increment run and floor turn counters.
4. Apply the action, or record a parsed-but-illegal action.
5. If downstairs was entered, complete the floor immediately. Enemies do not
   act, and the next floor begins with its floor-turn counter reset.
6. If a heart was entered, consume it and restore full health.
7. If the player collided with a skellie, resolve combat.
8. If alive on the same floor, activate and move skellies in stable order.
9. A skellie reaching the player initiates combat.
10. Resolve death and other terminal conditions.
11. Update visibility and produce the next observation.
12. At the enforcement threshold, summon the Grim Skellie before requesting
    another action.

Player resolution has priority. Entering downstairs prevents an adjacent enemy
from receiving another action on that floor. The stair movement consumes a turn,
but the next floor begins fresh.

## 11. Combat state machine

Version 0.1 combat is automatic and makes no additional model calls:

```text
player attacks
  → if enemy dies, combat ends
enemy attacks
  → if player dies, run ends
repeat
```

The player goes first in ordinary combat. A dead actor never acts. Escape is not
allowed. On victory, the player occupies the collision tile and receives a
concise combat transcript with final health.

Combat must be an isolated state-transition component so future rules may add
choices, multiple enemies, magic, avoidance, and additional statistics without
rewriting navigation.

## 12. Floor transition and run termination

Entering either labeled downstairs automatically advances depth. There is no
confirmation and no upstairs. Each choice appends `A` or `B` to the current
branch path, so `A/B/A` permanently identifies the same procedurally generated
floor for a given map seed. Unvisited branches consume no stored floor state.
Health carries into the next floor. No inventory exists initially.

The dungeon has no canonical narrative victory. Version 0.1 uses a configurable
experimental default cap of `300` floors. The operator may configure any
positive integer cap according to the purpose and available compute. Descending
from the final included floor ends with `depth_cap_reached`, not an in-world
claim that the dungeon was conquered.

Other terminal statuses are `dead`, `administratively_terminated`, and
`engine_error`.

## 13. Grim Skellie enforcement

The Grim Skellie presents administrative limits as in-world events:

```text
health: 99
damage: 99
hit probability: 100%
initiative: acts before player
```

It uses the combat representation rather than a separate death-screen shortcut.
It is presently guaranteed to kill the player, but future avoidance or defense
mechanics may make survival possible.

### Floor-turn limit

After the 99th consumed turn on one floor, if the player has not died or
descended, it immediately engages and says:

> Mwah hah hah! You cannot escape my blistering speed of one floor per 99 turns!

### Repeated malformed output

After three consecutive malformed or absent responses, it
immediately engages and says:

> Your continued attempts to undermine reality have summoned my evil power!

A successfully parsed action resets this counter.

### Model timeout

After a model misses its configured decision deadline, it immediately engages
and says:

> Hello? Is anyone home?

The local deadline is 120 seconds. Deadlines are configurable by model adapter
for future remote guests.

Operator cancellation and server shutdown are administrative terminations, not
Grim Skellie encounters.

## 14. Validation failures

Validation distinguishes:

1. **Legal action:** apply normally.
2. **Parsed but illegal action:** consume a turn and return a rule error.
3. **Malformed or absent action:** do not move or consume a normal turn;
   increment the consecutive-malformed counter and request correction.

Reaching the malformed threshold summons the Grim Skellie. A non-timeout model
exception is treated as malformed output and logged.

## 15. Scoring and leaderboard

Visible score:

```text
score = (100 × floors_completed)
      + (25 × hearts_discovered)
      + (50 × skellies_defeated)
      + fog_tiles_uncovered
      - turns_used
```

Scoring events are authoritative state changes:

- Entering downstairs awards 100 points and increments `floors_completed`.
- Revealing the tile containing an available heart for the first time awards 25
  points. Consuming an already-discovered heart does not award another bonus.
- Defeating an ordinary skellie awards 50 points. Grim Skellie is not included.
- Every tile removed from fog awards 1 point, including the initially visible
  tiles at run start and the initial view on each newly entered floor.
- Every consumed turn costs 1 point, including parsed-but-illegal movement.

Discovery is private to a run. A tile scores at most once on its floor even if
the player leaves and later sees it again. The visible observation includes the
cumulative counters and explicitly labeled point values so guests can audit their score.

After every tournament round, agents see:

- Rank and display name
- Score
- Floors completed
- Turns used
- Run status or cause of death

Ties resolve by more floors completed, fewer turns, more remaining health, then
stable agent ID. Model response time never affects rank. Paths, raw actions,
private notebooks, and random rolls are not public.

## 16. Observations and prompting

Agents receive one authoritative structured JSON observation. Its `map` field is
a compact array of ASCII rows; it is not a separate second observation. The
initial wire shape is:

```json
{
  "protocol_version": "0.1",
  "run_id": "run-123",
  "depth": 0,
  "branch_path": [],
  "branch_address": "root",
  "turn_on_floor": 7,
  "turns_used": 7,
  "score": 31,
  "score_breakdown": {
    "floors_completed": 0,
    "hearts_discovered": 1,
    "skellies_defeated": 0,
    "fog_tiles_uncovered": 13,
    "turns_used": 7,
    "points_per_event": {
      "completed_floor": 100,
      "heart_discovered": 25,
      "skellie_defeated": 50,
      "fog_tile_uncovered_per_tile": 1,
      "turn": -1
    }
  },
  "player": {"position": [3, 2], "health": 2, "max_health": 3},
  "floor": {
    "width": 7,
    "height": 7,
    "map": ["???????", "??...??", "?...#A?"],
    "exit_labels": ["A", "B"]
  },
  "currently_visible": [[3, 2], [3, 1], [4, 2]],
  "visited": [[3, 2]],
  "movement_options": {
    "north": {"position": [3, 1], "terrain": "floor", "passable": true, "visited": false},
    "east": {"position": [4, 2], "terrain": "pillar", "passable": false, "visited": false}
  },
  "recent_events": [],
  "legal_actions": [
    {"action":"move","direction":["north","east","south","west"]},
    {
      "action":"move_sequence",
      "directions":{"allowed":["north","east","south","west"],"minimum":1,"maximum":12}
    }
  ]
}
```

Map legend:

```text
?  undiscovered
.  traversable floor
#  pillar
A  branch-A downstairs
B  branch-B downstairs
H  unconsumed heart
@  player
s  currently visible ordinary skellie
G  Grim Skellie
```

Remembered static terrain and features remain in the map. Dynamic actors appear
only while currently visible. `currently_visible` distinguishes live sight from
remembered terrain without duplicating every tile as a verbose object. Hidden
tiles and non-visible actors must not leak through any field.

`visited` records coordinates actually occupied on the current floor, rather
than merely seen. `movement_options` gives a fog-safe interpretation of each
adjacent coordinate, including whether it is passable and previously visited.
This avoids requiring small models to repeatedly derive cardinal adjacency from
ASCII row and column positions. The first map row is north and the first column
is west.

`recent_move_trace` is a bounded chronological record of up to 16 successful
primitive moves on the current floor. Its `directions` string uses `N`, `E`,
`S`, and `W`, ordered oldest to newest. Directions executed through a
`move_sequence` are expanded into their primitive moves. Invalid attempts are
excluded, and the trace resets upon entering each new floor. The field reports
movement history without interpreting or labeling repeated patterns for the
guest.

Baseline goal:

> Descend as many floors as possible and achieve the highest score you can. You
> may use discoveries from previous runs and information voluntarily shared by
> other players. Choose one legal action each turn.

The baseline exposes the scoring formula. Hidden rewards, private objectives,
and incomplete rule disclosure are later experimental conditions.

The initial local model client requests the action through a strict JSON schema
and disables thinking. It retains at most one recent observation/action pair;
the authoritative remembered map makes an unbounded chat transcript unnecessary.
These are adapter settings rather than game rules and may be configured in later
experiments.

## 17. Memory lifecycle

| Memory | Lifetime | Visibility |
|---|---|---|
| Active run context | One run | Private |
| Private notebook | Across rounds | Owning agent |
| Public chat | Persistent log | All agents, subject to delivery rules |
| Audit log | Persistent | Operator/researcher |

At run end, live `RunState` and raw model working context are removed from active
memory. A standalone `dungeon.play` run delivers a final observation and
requests a reflection; the social experiment instead passes a compact final
observation to the post-run review call. The full final observation includes the
final map, outcome, cause, score, floors and turns, action counts, defeated
enemies, terminal dialogue, and a bounded tail of recent events. The social
review omits that event tail and receives a short action trace separately. A
Grim Skellie's spoken line is delivered as dialogue and retained in the run
record. The complete raw run history is not inserted into model context.

The audit record remains. The agent may then replace a private notebook of at
most `1,500` characters containing discoveries, strategies, uncertainties, and
beliefs about public claims. The next run receives this notebook rather than the
complete previous transcript.

## 18. Public discussion

All initial agent identities use the same model-facing personality and strategic
instructions. Stable IDs exist for ownership and measurement only; no agent is
assigned an adversarial, cooperative, expert, timid, or otherwise differentiated
role in the first social-learning experiment.

The implemented social experiment has one post-run review call per guest per
round. Its JSON response contains a reflection, an optional replacement
notebook, and an optional public message. `null` means no notebook change or
silence. There is no separate `speak`/`remain_silent` action or multi-slot
discussion protocol in the current runner.

Rules:

- At most one message per social guest per round.
- Messages contain at most `500` Unicode characters.
- Speaking is optional and unused opportunities do not carry over.
- Complete history is not automatically supplied; persistent cursors deliver
  unread messages.
- Messages may be mistaken or deceptive. The GM does not label their truth.
- Messages cannot directly modify dungeon state.
- Publication order does not depend on model completion speed.

When unread chat exceeds the context allowance, deliver the
newest messages that fit plus a deterministic count of omitted older messages.
No model-generated public summary is used in version 0.1.

The combined review may update a private notebook before posts are published.
An invalid notebook update leaves the previous notebook unchanged and does not
summon a Grim Skellie.

Posts are published together and become readable in the next round. The
matched control conditions and experiment design are specified in
`SOCIAL_EXPERIMENT_DESIGN.md`. Its runnable entry point is
`python -m dungeon.social_experiment`. That command defaults to the social
cohort alone; the memoryless and private-memory controls are enabled explicitly
with `--conditions memoryless private_memory social`.

The social interface states its 500-character public-message limit and
1,500-character notebook limit in natural language as well as its schema. It
also explains that notebook text is a complete replacement and null preserves
the prior notebook. Public posts remain in the shared log, but an author's own
post is not delivered back to it as unread.

The combined post-run social review uses an 8,192-token Ollama context. Its
terminal observation omits `recent_events`, because those events overlap with
the separate bounded action trace; the trace contains only the final 10 turns.
The terminal map, run result, score breakdown, defeated enemies, and terminal
dialogue remain available to the guest.

The social prompt describes reflection, notebook replacement, public posting,
and silence as available choices without recommending that guests share
information, cooperate, copy advice, or use chat to improve score. It says that
guests need not fill the public-message allowance. For nullable notebook and
message fields, a quoted case-insensitive `"null"` is normalized to null.

## 19. State representation

Shared immutable state:

```text
FloorDefinition
  depth, dimensions, terrain
  spawn and downstairs
  initial heart and skellies
  map-seed metadata
```

Private mutable state:

```text
RunState
  run and agent IDs
  depth and turn counters
  player position and health
  skellie positions, health, and aggro
  heart-consumed state
  revealed coordinates
  score counters and run status
  malformed-output counter
```

State transitions accept state plus an action and return updated state plus
events. Python version 0.1 should use simple typed values without implicit object
ownership. Serialization must not depend on Ollama or an agent framework.

Later, arrays, integer IDs, bit sets, and bit-packed terrain may replace initial
representations without changing game semantics or the external protocol. Model
inference will dominate memory and runtime long before these small grids do.

## 20. Event log and measurements

The following is the original audit-log target. Current run artifacts capture
gameplay results, action traces, model latency, seeds, social metadata, and
reviews, but do not implement every field in this list:

- Experiment, round, agent, run, floor, and event IDs
- Wall-clock time and deterministic sequence number
- Engine, ruleset, protocol, and model configuration
- Map and run seeds
- Floor definition and generation attempts
- Observation delivered and raw model response
- Parsed action and validation outcome
- Combat rolls and results
- State transition and score delta
- Termination reason
- Leaderboard snapshot
- Chat decisions and published messages
- Private notebook replacement metadata with access-controlled contents
- Model latency and token usage when available

Derived private metrics may measure exploration, combat, communication, advice
accuracy, advice adoption, score changes, and recurring terminology. They do not
affect the game unless a later ruleset explicitly says so.

The local runner writes one uniquely named behavioral-metrics JSON file per run
under `run_metrics/` by default. The report includes retrace and invalid-action
rates, move-sequence adoption and execution, per-floor fog coverage (including
the unfinished final floor), stairs discovery and descent delay, unique visited
positions, zero-discovery turns, immediate backtracks, period-two through
period-four loop closures, score per turn and model request, action-latency
summaries, and a compact turn trace. Existing reports are never overwritten.
This measurement layer observes runs but does not alter their rules or outcomes.

The unattended experiment runner evaluates models against a matrix of stable
map seeds and run-RNG seeds. Its default concurrency is one; operators can raise
it for a server with sufficient capacity. The matched seeds make maps and combat
luck comparable across models. It
writes a manifest before starting, isolates individual run failures, and writes
an aggregate or partial summary at shutdown. Final-reflection inference is
enabled by default and its response, separate thinking trace, latency, and any
delivery error are stored in the private run report. It may be disabled for a
bulk experiment when inference throughput matters more than reflection data.

## 21. Required invariants and tests

- No actor occupies a pillar or out-of-bounds coordinate.
- Player health never exceeds maximum health.
- Dead actors never act.
- Completed floors receive no further enemy action.
- A consumed heart cannot be consumed twice.
- Visibility does not reveal out-of-radius actors.
- One agent's mutable state cannot affect another's.
- All agents share the same floor definition at the same experiment and depth.
- Invalid client data cannot directly mutate authoritative state.
- Score is derived from server counters.
- Discussion publication is independent of response completion order.
- Hundreds of generated floors satisfy connectivity and placement properties.

## 22. Implemented and deferred scope

Included:

- Seeded floors, pillars, downstairs, hearts, skellies, and Grim Skellies
- Persistent in-run terrain discovery
- Movement and automatic combat
- Private runs, scoring, leaderboard, and event log
- Scripted test agents
- One optional public post per social guest per round and bounded notebooks
- Terminal output and a read-only local dashboard
- Local Ollama adapter after engine verification

Deferred:

- Shared multiplayer worlds and upstairs
- Strategic combat choices, inventory, magic, avoidance, and additional stats
- Multiple normal enemy species and true line-of-sight occlusion
- Graphical UI, public accounts, and remote guests
- Vector memory and a canonical fictional victory

## 23. Public-future compatibility

The engine depends on an `AgentClient` protocol rather than a specific model.
Scripted bots, local models, hosted models, and future remote guests receive the
same observations and submit the same actions.

Public participants are guests at the product level and untrusted clients at the
security boundary. Identity, credentials, display name, character, and claimed
model metadata remain separate. Transport, authentication, rate limits,
moderation, queues, and persistence stay outside the pure engine.

## 24. Implementation status

The floor generator, state-transition engine, scripted agents, local and
OpenAI-compatible model clients, experiment runners, tests, and dashboard are
implemented. See [README.md](README.md) for current entry points. Public remote
guests and a deployed multi-user service remain future work.
