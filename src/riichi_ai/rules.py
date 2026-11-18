"""Versioned rule contract and auditor-only physical-tile checks.

These helpers do not implement or certify Mahjong rules. Full simulator state
is required for auditing and must never enter a policy observation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import json
import random
import re
from typing import Any, Mapping, Sequence

from riichienv import GameRule


RULE_FIELDS = (
    "allows_ron_on_ankan_for_kokushi_musou", "is_kokushi_musou_13machi_double",
    "is_suuankou_tanki_double", "is_junsei_chuurenpoutou_double",
    "is_daisuushii_double", "yakuman_pao_is_liability_only",
    "sanchaho_is_draw", "kuikae_forbidden", "dealer_first_discard_is_tedashi",
)


def resolved_rule_profile() -> dict[str, Any]:
    """Resolve all exposed GameRule flags plus the intended whole-match rules.

    The semantic contract is deliberately separate from the coverage report:
    stating a rule here does not imply that the corresponding path is tested.
    """
    rule = GameRule.default_tenhou()
    return {
        "schema": "riichi-ai-rule-profile-v1", "profile_version": 1,
        "engine": {"name": "riichienv", "version": version("riichienv")},
        "game_mode": "4p-red-half", "resolved_game_mode_id": 2,
        "detailed_preset": "GameRule.default_tenhou",
        "game_rule": {name: getattr(rule, name) for name in RULE_FIELDS},
        "match": {
            "players": 4, "initial_scores": [25000] * 4,
            "red_physical_tiles": [16, 52, 88], "open_tanyao": True,
            "scheduled_last_round": [1, 3], "extension_last_round": [2, 3],
            "target_score": 30000, "bankruptcy_below": 0,
            "dealer_win_continues": True, "dealer_tenpai_draw_continues": True,
            "last_dealer_top_at_target_stops": True,
            "abortive_draw_continues": True,
            "tie_priority": "initial_seat_ascending",
            "terminal_sticks_recipient": "first_place",
            "double_ron": True, "triple_ron": "abortive_draw",
            "double_ron_honba_and_sticks": "nearest_winner_after_discarder",
            "riichi_cost": 1000, "honba_ron": 300, "honba_tsumo_each": 100,
            "noten_pool": 3000, "kazoe_yakuman": True, "kiriage_mangan": False,
        },
    }


def rule_profile_hash() -> str:
    payload = json.dumps(resolved_rule_profile(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_physical_tiles(env: Any) -> dict[str, int]:
    """Require every physical tile exactly once in a genuine four-player state.

    RiichiEnv retains called tiles in the donor's discard history. Subtract
    precisely that historical alias for each open meld, after checking its
    owner. Dora indicators are aliases into the wall, not additional tiles.
    Raises ValueError rather than repairing invalid state.
    """
    wall = list(env.wall)
    hands = [list(hand) for hand in env.hands]
    discards = [list(river) for river in env.discards]
    if len(hands) != 4 or len(discards) != 4 or len(env.melds) != 4:
        raise ValueError("physical audit requires a four-player state")
    physical = wall + [tile for hand in hands for tile in hand]
    physical += [tile for river in discards for tile in river]
    aliases: set[int] = set()
    meld_count = 0
    for owner, melds in enumerate(env.melds):
        for meld in melds:
            meld_count += 1
            tiles = list(meld.tiles)
            physical.extend(tiles)
            if meld.opened:
                source = meld.from_who
                if source not in range(4) or source == owner:
                    raise ValueError("invalid open-meld source seat")
                called = set(tiles).intersection(discards[source])
                if len(called) != 1 or called.intersection(aliases):
                    raise ValueError("open meld must alias exactly one donor discard")
                aliases.update(called)
    counts = Counter(physical)
    for tile in aliases:
        counts[tile] -= 1
    expected = Counter(range(136))
    if counts != expected:
        missing = sorted((expected - counts).elements())
        extra = sorted((counts - expected).elements())
        raise ValueError(f"physical tile conservation failed: missing={missing}, extra={extra}")
    return {"physical_tiles": 136, "called_discard_aliases": len(aliases), "melds": meld_count}


def wall_for_hands(
    hands: Mapping[int, str], draws: Sequence[str] = (), *, seed: int = 0,
    dealer: int = 0, dead_wall: Mapping[int, str] | None = None,
) -> list[int]:
    """Build a complete coherent wall from compact tile strings, for audits.

    ``hands`` specifies exactly 13 tiles for each supplied absolute seat; other
    seats receive deterministic remaining tiles. ``draws`` starts with the
    dealer's first draw. ``dead_wall`` addresses original wall indices 122..135.
    Normal fives prefer non-red copies; ``0m/0p/0s`` requests the red copy.
    Allocation raises if a physical tile is unavailable. This is synthetic
    fixture construction, never a hidden-state posterior sampler.
    """
    if dealer not in range(4) or any(seat not in range(4) for seat in hands):
        raise ValueError("seat must be in 0..3")
    available = set(range(136))

    def allocate(text: str) -> list[int]:
        tokens = re.findall(r"([0-9]+)([mpsz])", text)
        if "".join(a + b for a, b in tokens) != text:
            raise ValueError(f"invalid tile notation: {text}")
        result = []
        for digits, suit in tokens:
            for char in digits:
                rank = int(char)
                if suit == "z" and rank not in range(1, 8):
                    raise ValueError("honors must be 1..7")
                if suit != "z" and rank not in range(10):
                    raise ValueError("suited tile must be 0..9")
                kind = "mpsz".index(suit) * 9 + ((rank or 5) - 1)
                copies = [kind * 4 + i for i in range(4)]
                if rank == 0:
                    copies = [kind * 4]
                elif kind in (4, 13, 22):
                    copies = copies[1:] + copies[:1]
                tile = next((tile for tile in copies if tile in available), None)
                if tile is None:
                    raise ValueError(f"no physical copy remains for {char}{suit}")
                available.remove(tile)
                result.append(tile)
        return result

    wall: list[int | None] = [None] * 136
    for seat, text in sorted(hands.items()):
        tiles = allocate(text)
        if len(tiles) != 13:
            raise ValueError("specified initial hands must contain 13 tiles")
        relative = (seat - dealer) % 4
        slots = [block * 16 + relative * 4 + i for block in range(3) for i in range(4)]
        slots.append(48 + relative)
        for slot, tile in zip(slots, tiles):
            wall[slot] = tile
    if len(draws) > 70:
        raise ValueError("initial wall has only 70 live draws")
    for offset, text in enumerate(draws):
        tiles = allocate(text)
        if len(tiles) != 1:
            raise ValueError("each draw must specify one tile")
        wall[52 + offset] = tiles[0]
    for index, text in (dead_wall or {}).items():
        if index not in range(122, 136):
            raise ValueError("dead-wall index must be in 122..135")
        tiles = allocate(text)
        if len(tiles) != 1:
            raise ValueError("each dead-wall slot must specify one tile")
        wall[index] = tiles[0]
    remaining = sorted(available)
    random.Random(seed).shuffle(remaining)
    result = [remaining.pop() if tile is None else tile for tile in wall]
    if sorted(result) != list(range(136)):
        raise ValueError("fixture wall is not a physical-tile permutation")
    return result
