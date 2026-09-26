"""Coordinate an AgentClient with the pure dungeon engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .clients import AgentClient, AgentClientError, AgentDecision, AgentTimeoutError, FinalReceipt
from .engine import DungeonEngine
from .models import Direction, Event, MoveSequenceAction, RunState, Transition
from .observation import build_final_observation, build_observation


RunObserver = Callable[[str, object], None]


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    transitions: tuple[Transition, ...]
    events: tuple[Event, ...]
    final_observation: dict[str, object]
    final_receipt: FinalReceipt | None
    final_delivery_error: str | None = None
    observations: tuple[dict[str, object], ...] = ()
    decisions: tuple[AgentDecision, ...] = ()
    resolved_turns: tuple["ResolvedTurn", ...] = ()


@dataclass(frozen=True)
class ResolvedTurn:
    """One engine transition paired with its authoritative pre-transition state."""

    decision_index: int
    before: RunState
    transition: Transition


def run_agent(
    engine: DungeonEngine,
    client: AgentClient,
    run_id: str,
    agent_id: str,
    run_seed: int | str,
    max_decisions: int = 1_000,
    observer: RunObserver | None = None,
    deliver_final: bool = True,
) -> RunOutcome:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    notify = observer or (lambda _kind, _value: None)
    state = engine.start_run(run_id, agent_id, run_seed)
    transitions: list[Transition] = []
    events: list[Event] = []
    recent_events: tuple[Event, ...] = ()
    observations: list[dict[str, object]] = []
    decisions: list[AgentDecision] = []
    resolved_turns: list[ResolvedTurn] = []
    recent_move_trace: list[str] = []
    direction_glyphs = {
        Direction.NORTH.value: "N",
        Direction.EAST.value: "E",
        Direction.SOUTH.value: "S",
        Direction.WEST.value: "W",
    }

    for decision_index in range(max_decisions):
        if not state.active:
            break
        observation = build_observation(state, recent_events, recent_move_trace)
        observations.append(observation)
        notify("observation", observation)
        new_transitions: tuple[Transition, ...]
        supplemental_events: tuple[Event, ...] = ()
        try:
            decision = client.choose_action(observation)
        except AgentTimeoutError as exc:
            notify("client_error", str(exc))
            new_transitions = (engine.record_timeout(state),)
        except AgentClientError as exc:
            notify("client_error", str(exc))
            new_transitions = (engine.record_malformed(state, str(exc)),)
        else:
            decisions.append(decision)
            notify("decision", decision)
            if decision.action is None:
                new_transitions = (engine.record_malformed(state, decision.error or "invalid response"),)
            elif isinstance(decision.action, MoveSequenceAction):
                try:
                    sequence = engine.step_sequence(state, decision.action)
                except ValueError as exc:
                    new_transitions = (engine.record_malformed(state, str(exc)),)
                else:
                    new_transitions = sequence.transitions
                    supplemental_events = (Event("move_sequence_resolved", {
                        "submitted": [Direction(item).value for item in decision.action.directions],
                        "executed": [item.value for item in sequence.executed],
                        "remaining": [item.value for item in sequence.remaining],
                        "interrupted_by": sequence.interrupted_by,
                    }),)
                    notify("sequence_transition", sequence)
            else:
                new_transitions = (engine.step(state, decision.action),)

        decision_events: list[Event] = []
        for transition in new_transitions:
            resolved_turns.append(ResolvedTurn(decision_index, state, transition))
            transitions.append(transition)
            events.extend(transition.events)
            decision_events.extend(transition.events)
            state = transition.state
            notify("transition", transition)
            if any(event.kind == "floor_entered" for event in transition.events):
                recent_move_trace.clear()
            else:
                for event in transition.events:
                    if event.kind == "player_moved":
                        recent_move_trace.append(
                            direction_glyphs[str(event.data["direction"])]
                        )
                del recent_move_trace[:-16]
        events.extend(supplemental_events)
        decision_events.extend(supplemental_events)
        recent_events = tuple(decision_events)
    else:
        if state.active:
            transition = engine.terminate(state, "max_decisions_reached")
            transitions.append(transition)
            resolved_turns.append(ResolvedTurn(max_decisions, state, transition))
            events.extend(transition.events)
            state = transition.state
            notify("transition", transition)

    final_observation = build_final_observation(
        state, events, recent_move_trace=recent_move_trace
    )
    notify("final_observation", final_observation)
    receipt: FinalReceipt | None = None
    delivery_error: str | None = None
    if deliver_final:
        try:
            receipt = client.receive_final(final_observation)
            notify("final_receipt", receipt)
        except AgentClientError as exc:
            delivery_error = str(exc)
            notify("client_error", delivery_error)

    return RunOutcome(
        state=state,
        transitions=tuple(transitions),
        events=tuple(events),
        final_observation=final_observation,
        final_receipt=receipt,
        final_delivery_error=delivery_error,
        observations=tuple(observations),
        decisions=tuple(decisions),
        resolved_turns=tuple(resolved_turns),
    )
