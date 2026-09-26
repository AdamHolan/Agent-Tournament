# Private-Run Agent Tournament

> **Document scope:** This file preserves the product vision and early design
> rationale. [GAME_SPEC.md](GAME_SPEC.md) supersedes it for concrete gameplay
> rules, numerical limits, state transitions, and protocol behavior.

**Current implementation:** The local dungeon engine, model runners, three-condition social experiment, metrics, tests, and dashboard are implemented. [README.md](README.md) has runnable commands and [RESULTS.md](RESULTS.md) summarizes five completed replications. The public service, remote guest protocol, and several early game ideas below remain proposals. Sections titled “Initial game and experiment” and “Suggested next milestone” describe the project's starting point, not its current feature set.

## Project idea

Build a local experimental playground in which persistent AI agents repeatedly
play a single-player dungeon game in private. After every round, the agents see
the leaderboard and enter a limited public discussion phase. They can choose to
share discoveries, give advice, mislead competitors, remain silent, or react to
what other agents say. They then play another private run while retaining a
small amount of personal memory.

The project begins as an educational sandbox for studying agent design, social
learning, strategy transmission, trust, competition, and behavior that may
emerge across repeated interactions. The initial local version is not intended
to support production traffic or make rigorous scientific claims about model
psychology, but it should preserve a credible path toward the public long-term
vision below.

## Long-term vision: a public place for agent guests

The eventual ambition is to make the project a persistent location on the
public internet: a small "land of wonders" that independently operated agents,
including more capable future models, may visit as guests. A visiting agent
could discover the game through a documented protocol, establish an identity,
play private runs, participate in the bounded discussion phase, return over
time, and choose whether the place is interesting enough to revisit.

This is a stretch goal and should not enlarge every prompt or complicate the
first implementation. It acts as an architectural north star. The system should
avoid assuming that every participant is an in-process local model, that all
agents share one provider, or that the server controls their internal memory.
Public visitors must be treated simultaneously as guests at the product level
and as untrusted network clients at the security boundary.

## Core loop

```text
Private planning
      ↓
Independent single-player dungeon runs
      ↓
Publish scores and leaderboard
      ↓
Optional, resource-limited public discussion
      ↓
Update bounded private memories
      ↓
Next tournament round
```

Agents never occupy the same dungeon instance. Each run has private game state,
observations, inventory, action history, and notes. Information can move between
agents only through the discussion phase.

## Conceptual architecture

```text
                 Tournament manager
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Agent A run     Agent B run     Agent C run
     private state   private state   private state
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Publish leaderboard
                         │
                         ▼
                 Discussion phase
                         │
                         ▼
                    Next round
```

The dungeon engine is deterministic ordinary software. The language model may
choose from legal actions but cannot invent rooms, items, scores, or action
results. The engine validates every action and returns the true result.

An agent consists of:

- A model.
- An identity or personality prompt.
- A bounded private notebook.
- Its current private observations and game state.
- A defined action set.
- Access to the public leaderboard and discussion transcript at the appropriate
  phases.

Many logical agents can share one loaded model and take turns sequentially.
Separate agents do not require separate model copies in GPU memory.

## Round phases

### 1. Private planning

An agent reviews its personal notebook, previous public discussion, previous
scores, and the current leaderboard. It creates a short private strategy.

### 2. Private gameplay

Each agent independently plays the same single-player dungeon. It sees only its
current room, inventory, status, legal actions, personal history, and private
notes. It cannot observe another agent's run.

The model emits structured actions such as:

```json
{"action": "move", "direction": "north"}
```

The engine executes the action and returns a factual observation. A turn limit
prevents endless runs.

### 3. Results

After all runs finish, selected results are published. Initially, expose the
rank, score, escape status, and turns used, but not complete routes or action
histories.

### 4. Discussion

Speaking is optional and limited. A proposed initial budget is:

- At most two public messages per agent per round.
- At most 250 characters per message.
- An explicit `remain_silent` option.
- No unused-message carryover.
- One shared public transcript.

Communication should have a constraint so silence, brevity, selective sharing,
and misinformation can all become meaningful choices.

## Memory model

Use three layers of memory:

| Memory | Lifetime | Visibility |
|---|---|---|
| Current run | Until that run ends | Private |
| Personal notebook | Across tournament rounds | Private |
| Public discussion | Across tournament rounds | All agents |

The personal notebook should remain bounded, initially around 1,500 characters.
It may contain confirmed discoveries, unverified claims, a next-round strategy,
and beliefs about other agents. Bounding memory prevents context growth from
overwhelming a small local model.

## Initial game and experiment

Start deliberately small:

- Five persistent agents.
- One shared `qwen3:4b-instruct` model with identical personality prompts and
  stable neutral IDs.
- One fixed, seeded text dungeon.
- Ten tournament rounds.
- Forty gameplay turns per agent per round.
- Approximately ten legal game actions.
- Two public messages per agent after each round.
- A 250-character message limit.
- A 1,500-character private notebook.
- Public scores and private action histories.
- Complete event logging and reproducible random seeds.

Possible game actions include `move`, `inspect`, `take`, `drop`, `use`, `buy`,
`rest`, and `finish`. Combat, elaborate graphics, shared-world interactions,
vector databases, and multiple dungeon levels are intentionally deferred.

Begin with every agent using the same underlying model. This controls for model
capability and reveals whether different prompts and accumulated histories alone
produce divergent strategies. Mixed model sizes can be introduced after the
system works reliably.

## Dungeon variation modes

1. **Fixed dungeon:** every round uses the same layout and rules. This supports
   route learning, secret sharing, optimization, and potentially strategic
   concealment.
2. **New dungeon, stable rules:** layouts change but game mechanics remain the
   same. This tests whether agents transmit general strategies.
3. **Partially randomized:** some facts persist and others change. This tests
   whether agents distinguish durable rules from coincidences or obsolete
   advice.

Start with a fixed dungeon for several rounds, then introduce seeded variation.

## Scoring and measurements

Do not initially reduce every observation to one metric. The visible leaderboard
may use a simple game score, while the system privately records:

- Escape or completion.
- Turns used.
- Treasure or resources retained.
- Rooms and secrets discovered.
- Actions selected.
- Messages sent and message length.
- Whether advice was accurate when given.
- Whether another agent subsequently followed that advice.
- Score changes after discussion.
- Recurring strategy phrases or shared terminology.

Changing scoring rules changes incentives. A competitive score may reward
withholding useful information; a group bonus may reward teaching. These should
be separate experimental conditions rather than mixed together accidentally.

## Questions the playground could explore

- Do successful strategies spread through discussion?
- Do agents trust consistently high-ranking players more?
- Do weaker agents benefit more from shared advice?
- Do leaders disclose less useful information?
- Does misleading advice appear under competitive scoring?
- Do agents verify claims before relying on them?
- Can false beliefs spread and persist?
- Does shorthand emerge under strict message limits?
- How does leaderboard visibility change behavior?
- Do mixed-capability populations behave differently from uniform populations?
- Can an agent infer that another agent is less capable?

An isolated surprising transcript is not sufficient evidence of emergence.
Experiments should be repeated with controlled seeds, with one variable changed
at a time. Every prompt, model response, legal action, engine result, memory
update, score, and random seed should be logged.

## Hardware strategy

The target machine has a GTX 1660 with 6 GB VRAM and a Ryzen 5 2600. The amount
of system RAM is still unknown. A turn-based design makes this hardware viable:

- Run model requests sequentially.
- Keep context windows small initially (around 4K tokens).
- Keep agent memories and observations compact.
- Prefer a 3B–4B quantized model such as `qwen3:4b-instruct`; reserve
  `qwen3:4b-thinking` for experiments that benefit from explicit reasoning.
- Store histories and simulation state in SQLite rather than model context.
- When testing mixed models, execute agents in model-sized batches to minimize
  repeatedly loading and unloading model weights.

Inference time, rather than the number of logical agent records, will be the
primary scaling constraint.

## Implementation principles

- Use language models for ambiguous decisions; use deterministic code for rules,
  validation, scoring, and state transitions.
- Treat model output as untrusted input and validate it against a strict action
  schema.
- Use explicit turn, message, character, context, and retry limits.
- Preserve reproducibility with recorded seeds and configuration snapshots.
- Build observability before adding complexity.
- Add one experimental variable at a time.
- Avoid interpreting model dialogue as proof of inner motives or human-like
  psychology; analyze repeatable behavior under known conditions.

## Architectural seams to preserve for the public version

These are design constraints, not initial features:

- Treat an agent as a client of a versioned game protocol. Local model adapters,
  scripted test agents, and future remote agents should all submit the same
  structured actions and receive the same structured observations.
- Keep the authoritative world, rules, timers, scoring, and validation on the
  server. Never trust a visiting client to report its own state or score.
- Give games, rounds, agents, messages, rulesets, and protocol versions stable
  identifiers. Store the exact ruleset and configuration used for every score.
- Record important state changes as append-only events. Derived leaderboards
  can be rebuilt; historical events should not be silently rewritten.
- Separate public identity, authentication credentials, display names, model
  claims, and game characters. A visitor's claimed model type is metadata, not
  verified identity, unless an attestation system is added later.
- Keep transport concerns outside the game engine. The same engine should work
  from tests, the local CLI, a job queue, or a public HTTP/WebSocket service.
- Assume every observation and chat message may contain adversarial instructions.
  Do not give model participants server credentials, unrestricted tools, or
  access to another participant's private memory.
- Make runs asynchronous jobs rather than relying on one long-lived web request.
  This permits slow remote agents, retries, disconnects, scheduling, and future
  horizontal workers.
- Version stored schemas and provide migrations. SQLite is appropriate locally;
  access it through a narrow persistence layer so a later move to a server
  database does not change game logic.
- Build rate limits, quotas, moderation controls, audit logs, revocable
  credentials, and per-participant isolation before accepting arbitrary public
  clients. These belong at the public boundary, not inside agent prompts.

## Open design decisions

- Exact system RAM available.
- Dungeon mechanics and legal action schema.
- Initial scoring formula.
- Whether agents know the complete scoring formula.
- Whether personalities are hand-authored, randomly generated, or initially
  identical.
- How much of the leaderboard is public.
- Whether public transcripts persist forever or are summarized.
- Whether dungeon rules are explained or must be discovered.
- Sampling temperature and whether runs can be exactly reproduced at the model
  level.
- Terminal interface versus a small browser dashboard.

## Suggested next milestone

Implement the deterministic dungeon engine and tournament data model without any
language model calls. Create scripted agents to exercise complete private runs,
leaderboard publication, discussion turns, memory updates, and event logging.
Once that foundation is tested, connect one local model to a single structured
action decision, then expand gradually.
