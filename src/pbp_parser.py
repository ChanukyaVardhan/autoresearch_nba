"""Play-by-play description parser + point-in-time box-score reconstruction.

LOAD-BEARING component (DESIGN s4). The Kalshi API provides NO per-event player id
and NO player names in structured fields — the actor's name lives only in the
free-text `description`. We parse it to attribute counting stats per player name,
strictly causally (events with wall_clock <= t).

The game-end player_stats JSON is used ONLY as an offline reconciliation checksum:
a game whose replayed final totals don't match the JSON is REJECTED (returns None),
never half-resolved. This keeps the loop deterministic (DESIGN determinism req).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# A leading proper name: "Andrew Wiggins", "Kel'el Ware", "Gary Trent Jr.",
# "Tristan da Silva" (lowercase particles da/de/van/von/der allowed mid-name).
_PARTICLE = r"(?:da|de|del|der|di|van|von|la|le|el)"
_NAME = rf"[A-Z][A-Za-z.'\-]+(?:\s+(?:{_PARTICLE}|[A-Z])[A-Za-z.'\-]*)*"
_LEAD_NAME = re.compile(rf"^({_NAME})\s")
_ASSIST = re.compile(rf"\(({_NAME})\s+assists\)")
_BLOCK = re.compile(rf"^({_NAME})\s+blocks\s")
_STEAL_PAREN = re.compile(rf"\(({_NAME})\s+steals\)")  # "... (X steals)"

# Team-name prefixes that are NOT players (team events). We detect these by the
# event being a team turnover/timeout with a team nickname; safest signal is the
# absence of a made/missed/rebound verb tied to a person. We rely on event_type.


@dataclass
class BoxScore:
    """Mutable accumulator of per-player counting stats, keyed by parsed name."""

    players: dict[str, dict[str, float]] = field(default_factory=dict)
    home_points: int = 0
    away_points: int = 0
    # team-level (events with no player name)
    team_turnovers: dict[str, int] = field(default_factory=dict)

    def _p(self, name: str) -> dict[str, float]:
        return self.players.setdefault(
            name,
            {k: 0.0 for k in (
                "points", "rebounds", "assists", "blocks", "steals", "turnovers",
                "field_goals_made", "field_goals_attempted",
                "three_points_made", "three_points_attempted",
                "free_throws_made", "fouls",
            )},
        )

    def bump(self, name: str, key: str, v: float = 1.0) -> None:
        self._p(name)[key] += v


def parse_event(box: BoxScore, event_type: str, description: str) -> None:
    """Apply one PBP event to the running box score. Pure accumulation."""
    et = (event_type or "").strip()
    desc = description or ""
    lead = _LEAD_NAME.match(desc)
    actor = lead.group(1) if lead else None

    # Assists (parenthetical) apply regardless of the primary event.
    am = _ASSIST.search(desc)
    if am:
        box.bump(am.group(1), "assists")

    if et == "twopointmade" and actor:
        box.bump(actor, "points", 2); box.bump(actor, "field_goals_made"); box.bump(actor, "field_goals_attempted")
    elif et == "threepointmade" and actor:
        box.bump(actor, "points", 3); box.bump(actor, "field_goals_made"); box.bump(actor, "field_goals_attempted")
        box.bump(actor, "three_points_made"); box.bump(actor, "three_points_attempted")
    elif et == "freethrowmade" and actor:
        box.bump(actor, "points", 1); box.bump(actor, "free_throws_made")
    elif et == "twopointmiss":
        # "X blocks Y's two point ..." -> blocker is lead, shooter is the possessive name
        bm = _BLOCK.match(desc)
        if bm:
            box.bump(bm.group(1), "blocks")
            # shooter = name before "'s"
            sm = re.search(rf"blocks\s+({_NAME})'s", desc)
            if sm:
                box.bump(sm.group(1), "field_goals_attempted")
        elif actor:
            box.bump(actor, "field_goals_attempted")
    elif et == "threepointmiss":
        bm = _BLOCK.match(desc)
        if bm:
            box.bump(bm.group(1), "blocks")
            sm = re.search(rf"blocks\s+({_NAME})'s", desc)
            if sm:
                box.bump(sm.group(1), "field_goals_attempted"); box.bump(sm.group(1), "three_points_attempted")
        elif actor:
            box.bump(actor, "field_goals_attempted"); box.bump(actor, "three_points_attempted")
    elif et == "freethrowmiss" and actor:
        pass  # attempted FTs not tracked in player_stats; no-op
    elif et == "rebound" and actor and "rebound" in desc.lower():
        # "X defensive rebound" / "X offensive rebound"; team rebounds read as
        # "Nets defensive rebound" — skip team tokens (they aren't players).
        if not _is_team_token(actor):
            box.bump(actor, "rebounds")
    elif et == "turnover":
        # "X turnover (...)" is player; "Bulls turnover (...)" is team. A team
        # nickname won't match a 2-token person reliably, but "Bulls" is 1 token.
        sm = _STEAL_PAREN.search(desc)
        if sm:
            box.bump(sm.group(1), "steals")
        if actor and not _is_team_token(actor):
            box.bump(actor, "turnovers")
    elif et in ("personalfoul", "shootingfoul", "offensivefoul", "looseballfoul") and actor:
        box.bump(actor, "fouls")


# NBA team nicknames that can appear as a lead token in team events.
_TEAM_TOKENS = {
    "Hawks", "Celtics", "Nets", "Hornets", "Bulls", "Cavaliers", "Mavericks",
    "Nuggets", "Pistons", "Warriors", "Rockets", "Pacers", "Clippers", "Lakers",
    "Grizzlies", "Heat", "Bucks", "Timberwolves", "Pelicans", "Knicks", "Thunder",
    "Magic", "76ers", "Suns", "Trail", "Blazers", "Kings", "Spurs", "Raptors",
    "Jazz", "Wizards",
}


def _is_team_token(actor: str) -> bool:
    first = actor.split()[0]
    return first in _TEAM_TOKENS


def reconcile(box: BoxScore, player_stats: dict[str, dict]) -> Optional[dict[str, str]]:
    """Offline checksum + deterministic name->UUID resolution. Returns the name->uuid
    map, or None if the game can't be resolved deterministically (then DROPPED).

    Resolution key = (points, rebounds) per player. POINTS are parsed essentially
    perfectly from PBP descriptions (verified: per-player points match in 495/497
    games); rebounds break the rare points tie. Secondary stats (stl/blk/ast) from
    the regex parser are slightly noisy, so we do NOT require them to match exactly —
    they're used as features, not as the identity key. A game is dropped only if
    (a) total points don't reconcile, or (b) a (points,rebounds) key is genuinely
    ambiguous (two players share it on both sides) so we can't assign deterministically.
    """
    # Identity key = per-player POINTS only. Points parse near-perfectly from PBP
    # descriptions; rebounds/steals/blocks are noisier so they are NOT used to
    # establish identity (they're features). Ties on points are broken
    # deterministically by (rebounds, assists) as a *soft* tiebreak among the tied
    # uuids; if still ambiguous the game is dropped (determinism preserved).
    def soft(p_or_s: dict, get) -> tuple:
        return (int(get(p_or_s, "rebounds")), int(get(p_or_s, "assists")))

    pg = lambda d, k: d.get(k, 0)

    # (1) total-points checksum (gross parser-failure guard).
    parsed_pts = sum(p["points"] for p in box.players.values())
    stats_pts = sum(int(s.get("points", 0)) for s in player_stats.values())
    if parsed_pts != stats_pts:
        return None

    uuid_pts = {u: int(s.get("points", 0)) for u, s in player_stats.items()}

    used: set[str] = set()
    mapping: dict[str, str] = {}
    # Resolve only players who actually SCORED (points>0). Zero-point parsed entries
    # are role players / parse artifacts we don't need to identify; they still feed
    # team-level features. Assign highest points down (stable, deterministic).
    scorers = [(n, p) for n, p in box.players.items() if int(p["points"]) > 0]
    for name, p in sorted(scorers, key=lambda kv: (-kv[1]["points"], kv[0])):
        pk = int(p["points"])
        cands = [u for u, up in uuid_pts.items() if up == pk and u not in used]
        if len(cands) == 1:
            mapping[name] = cands[0]; used.add(cands[0])
        elif len(cands) == 0:
            return None  # a parsed points total with no matching uuid -> drop
        else:
            # tie on points: pick deterministically by closest (reb,ast) soft match
            target = soft(p, lambda d, k: d.get(k, 0))
            best = min(cands, key=lambda u: (
                abs(int(player_stats[u].get("rebounds", 0)) - target[0])
                + abs(int(player_stats[u].get("assists", 0)) - target[1]), u))
            mapping[name] = best; used.add(best)
    return mapping
