# Social Learning Experiment Design

**Status:** Implemented experiment design with historical planning notes. The
current command defaults and output layout are in [README.md](README.md) and
`dungeon/social_experiment.py`; completed replications are summarized in
[RESULTS.md](RESULTS.md).

## Purpose

Test whether identical model instances can turn post-run reflection and shared
language into better behavior across repeated private dungeon runs.

The first social experiment uses neutral, identical instructions. Agents
receive no adversarial roles, secret objectives, personality variations, or
different strategic instructions. Stable identities exist only so notebooks,
chat cursors, scores, and longitudinal measurements have owners.

Primary question:

> Does persistent private or shared language reduce navigation loops and improve
> score and floor completion over later rounds?

Secondary questions include whether agents share correct map discoveries,
whether recipients act on useful advice, whether ineffective advice propagates,
and whether move-sequence usage emerges through communication.

## Identical agent identity policy

Initial agents are named `guest-001`, `guest-002`, and so on. Every guest uses:

- The same underlying model and model tag.
- The same gameplay system prompt.
- The same review prompt within each condition. The three conditions use
  different review prompts to describe their available capabilities.
- The same temperature, history bound, action schema, and context limits.
- No biography, temperament, role, expertise, or private objective.

The numeric ID is never described as a personality. Run order must not affect
what an agent can observe. Inference may be sequential or concurrent, but all run
results are withheld until every guest in the round has finished, and all valid
posts are published simultaneously in stable ID order.

## Experimental conditions

Use three isolated cohorts. No notebook or message crosses a cohort boundary.

| Condition | Persistent private notebook | Shared chat |
|---|---:|---:|
| `memoryless` | No | No |
| `private_memory` | Yes | No |
| `social` | Yes | Yes |

All three conditions still perform one post-run review inference. For
`memoryless`, the reflection is logged, while no notebook or message is carried
forward. This keeps the number and placement of model calls comparable. Every
condition receives the same
public leaderboard fields; only the social cohort receives message text.

Matched logical agent slots use the same map and run-RNG seed schedule across
conditions. For example, `memoryless/guest-002/round-004` and
`social/guest-002/round-004` face the same floor definitions and combat random
stream. Model sampling can still diverge and is part of the measured behavior.

## Round lifecycle

```text
checkpoint: round ready
        |
        v
deliver notebook and unread chat permitted by condition
        |
        v
run every guest's private dungeon (configurable concurrency, isolated state)
        |
        v
freeze results only after every guest finishes
        |
        v
deliver identical-format leaderboard
        |
        v
one private social-review inference per guest
  - reflection
  - replacement notebook or silence
  - one optional public message or silence
        |
        v
validate, then publish all messages simultaneously
        |
        v
advance round and atomically checkpoint
```

Messages published after round N become readable at the beginning of round
N+1. Agents cannot reply within the same discussion phase in version 0.1. This
one-round-lag forum design trades conversational depth for many more repeated
social rounds on limited hardware.

## Social-review model interface

Reflection, notebook replacement, and the optional public post share one model
call. The response uses a strict schema conceptually equivalent to:

```json
{
  "reflection": "What happened and what appears useful",
  "notebook": "Replacement private memory, or null",
  "public_message": "One optional message, or null"
}
```

Limits:

- Reflection: 750 Unicode characters.
- Notebook: 1,500 Unicode characters.
- Public message: 500 Unicode characters.
- One public message opportunity per agent per round in this experiment.
- Empty strings normalize to null.
- Invalid notebook output preserves the previous valid notebook.
- Invalid public output becomes silence; it does not summon Grim Skellie.
- Social-review failure is logged and does not invalidate the completed run.

The social-review prompt is identical for all identities. It neutrally explains
that reflection, notebook replacement, public posting, and silence are
available. It does not ask guests to share information or improve future score,
and does not tell them to cooperate, copy, deceive, avoid loops, use sequences,
or adopt a particular strategy. It states that a complete public message must
fit within 500 characters and need not fill the allowance. Quoted `"null"`
values in nullable fields normalize to actual null.

## Context delivered at the next run

Gameplay retains one ordinary observation/action history pair within a run.
Before the first observation of a new run, the model receives a bounded preamble:

```text
Identity: guest-NNN
Private notebook: <condition-dependent text or empty>
Unread public messages: <social condition only, or empty>
Previous round result: <score, floors, turns, terminal cause>
Current leaderboard: <rank, guest ID, score, floors, turns, cause>
```

The text explicitly labels public messages as untrusted statements by other
guests. It does not summarize, verify, or correct them. Unread delivery uses a
persistent per-agent cursor. If messages exceed the context allowance, deliver
the newest complete messages that fit and include a deterministic omitted-count.

## Initial experiment profile and later replications

The original three-condition design profile was:

```text
conditions:          3
agents per condition: 3
rounds:              8
maximum actions/run: 24
depth cap:           3
map seed:            one fixed dungeon shared by every run
run RNG:             unique by agent slot and round, matched across conditions
history turns:       1
temperature:         0.2
social reviews:      1 per completed run
posts:               at most 1 per agent per round
```

This profile produces 72 private runs, at most 1,728 gameplay decisions, and 72
review calls. It is a planning example rather than the CLI default. The current
CLI defaults to the `social` condition alone, three guests, eight rounds, 24
decisions per run, depth cap three, and concurrency one. The five completed
replications used all three conditions, ten rounds, 48 decisions per run, depth
cap 20, and a higher concurrency on a rented GPU. See [RESULTS.md](RESULTS.md).

The action cap is intentionally much lower than the prior 75-decision study. It
makes efficient navigation and accumulated knowledge valuable while purchasing
enough repeated rounds to observe longitudinal change. It is an administrative
experiment limit, not an energy mechanic: every executed primitive movement
still consumes one ordinary game turn, including movements inside a sequence.

Implemented overrides include `--agents`, `--rounds`, `--max-decisions`,
`--conditions`, `--depth-cap`, `--max-concurrency`, `--chat-delivery-chars`, and
`--experiment-seed`. The 500-character public-message and 1,500-character
notebook limits are currently fixed by the model schema, not CLI flags. A
smaller pilot can use two agents per condition, three rounds, and 18 decisions
per run.

## Persistence and resume

Social experiments are resumable. A current experiment directory contains:

```text
manifest.json
checkpoint.json
chat.jsonl
rounds/
  round-NNN/
    runs/*.json
    reviews/*.json
    leaderboard-<condition>.json
summary.json
```

The manifest records prompt hashes, model settings, seeds, limits, and
conditions. `checkpoint.json` holds the next round, agent notebooks and chat
cursors, and the latest leaderboards. `chat.jsonl` is append-only. Completed
run and review jobs are detected from their artifact files on resume.

JSON snapshots use temporary files followed by atomic replacement. On
`--resume PATH`, existing run and review artifacts are reused, an interrupted
run restarts from its original seeds, and published messages are deduplicated.
Ctrl+C writes a partial summary. Checkpoints advance after a complete round;
they do not capture model context mid-run.

Mid-run model context is not checkpointed. Restarting an incomplete run may
produce another sample at nonzero temperature, but it cannot corrupt completed
rounds or social state.

## Measurements

The design calls for retaining gameplay metrics per run and measuring:

- Condition, stable agent ID, and round number.
- Score, floors, loop closures, retracing, zero-discovery turns, stair delay,
  sequence adoption, and their change over rounds.
- Exact notebook versions and replacement decisions.
- Exact public messages, silence decisions, character counts, author, source
  round, delivery recipients, and cursor advancement.
- Speaking rate and notebook-update rate.
- Message exposure before each run.
- Performance slope by agent and condition.
- Matched outcome differences between conditions for the same agent slot/round.
- Map-coordinate and stair claims that can be checked deterministically.
- Whether later trajectories agree with exposed actionable claims.
- Repeated phrases or strategies spreading through notebooks and chat.
- Incorrect claims that persist or disappear.

Advice adoption would need an explicit, conservative proxy rather than an
interpretation of model intent. For example: after receiving a claim that the
stairs are at coordinate `[x,y]`, did the next run reduce time to that coordinate
relative to its matched memoryless run? Raw artifacts remain available for
qualitative inspection.

## Analysis plan

Compare conditions both in aggregate and as matched observations. Report early
rounds versus late rounds and show individual trajectories; a mean alone can
hide agents that enter persistent loops.

Primary outcomes:

1. Floors completed per run.
2. Score per model request.
3. Zero-discovery turns.
4. Period-four loop closures.
5. Turns between stairs discovery and descent.

Evidence of social learning requires improvement over rounds that is larger in
`social` than in both `memoryless` and `private_memory`. A clever message or one
successful run is an anecdote, not sufficient evidence. This remains an
exploratory local experiment rather than a claim about model consciousness or
human-like social cognition.

## Deferred variables

Do not add these to the first social experiment:

- Different personalities or adversarial prompts.
- Mixed model families or sizes.
- Private goals, deception rewards, or hidden scoring.
- Different map dimensions.
- Multiple chat slots per round.
- Model-generated chat summaries.
- Retrieval databases or semantic memory.
- More elaborate GPU scheduling beyond the implemented `--max-concurrency`
  control.

These become later experiments only after the identical-agent baseline is
working and measured.
