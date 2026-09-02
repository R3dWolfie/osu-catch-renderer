"""Authoritative lazer-standardised total score for a catch replay.

R3D shows ONE score scale everywhere — osu!lazer's standardised (ScoreV3,
max ~1,000,000 × mod multiplier) — no matter which client set the play.
The number the in-video counter ends on, the results screen, and the
website card must all be the SAME standardised total, and it must be the
number the player recognises from lazer / the osu! website.

The .osr legacy header `score` field means different things per source:

  • true osu!stable play ............ the stable ScoreV1 total (e.g. 163,573,200)
  • lazer play downloaded from the
    osu! website (legacy .osr export,
    stable-style game_version) ...... the CLASSIC-converted display total
                                      (lazer ScoreInfoExtensions.GetDisplayScore
                                      → convertStandardisedToClassic)
  • lazer client local export ....... ScoreInfo.TotalScore = the standardised
                                      total itself (LegacyScoreEncoder.Encode
                                      writes `(int)score.ScoreInfo.TotalScore`)

Both legacy-looking cases carry stable-format game_version (YYYYMMDD), so the
client CANNOT be told apart from the header alone. We therefore convert the
header under every interpretation and pick the candidate closest to our own
standardised simulation (scene.CatchSim), which is client-agnostic and
accurate to a few percent — while the wrong interpretation is typically off
by 10×+ (a classic total read as ScoreV1, or vice versa).

Exact conversion sources (ppy/osu master, 2026-07, LATEST_VERSION 30000018):

  A. stable ScoreV1 → standardised:
     osu.Game/Database/StandardisedScoreMigrationTools.cs
       convertFromLegacyTotalScore (case 2) + estimateComboProportionForCatch
     osu.Game.Rulesets.Catch/Difficulty/CatchLegacyScoreSimulator.cs
       Simulate + GetLegacyScoreMultiplier   (ScoreV1 attribute sim)
     osu.Game/Rulesets/Objects/Legacy/LegacyRulesetExtensions.cs
       CalculateDifficultyPeppyStars          (decimal, banker's rounding)
     This is the SAME math osu-web/osu-queue-score-statistics runs server-side,
     so it reproduces the standardised number shown on the osu! website for
     stable plays (verified: 163,573,200 → 940,575 on Epitaph [Lament]).

  B. classic ↔ standardised (lazer display conversion, exact & monotonic):
     osu.Game/Scoring/Legacy/ScoreInfoExtensions.cs
       convertStandardisedToClassic case 2:
         classic = round((std/1e6 · objectCount)² · 21.62 + std/10)
       objectCount = maximum basic judgements = the map's FRUIT count
       (HitResult.Great maxima; ticks/bonus are not "basic").
     The +std/10 term makes the map strictly monotonic (lazer's own remark:
     "every 10 points in standardised mode converts to at least 1 point in
     classic mode ... does not reorder scores"), so the inverse — the positive
     root of the quadratic 21.62·(n/1e6)²·s² + s/10 − classic = 0 — is exact
     up to the ±0.5 header rounding.

Maximum-statistics folding for stable scores mirrors the server: max(Great) =
count_300 + count_miss (stable folds droplet misses into count_miss — the
"slightly incorrect ... trudge on" note in the lazer source), max(SmallTickHit)
= count_50 + count_katu.
"""
from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from struct import pack, unpack

from osu_catch_renderer.beatmap.models import ObjType

MAX_SCORE = 1_000_000.0

# ── stable (ScoreV1) legacy mod multipliers ────────────────────────────────
# CatchLegacyScoreSimulator.GetLegacyScoreMultiplier — the multipliers the
# STABLE client applied to ScoreV1 (NB: DT/NC are 1.06 here, unlike the
# current lazer standardised table where rate mods are 1.1).
_LEGACY_MULT = {
    1 << 0: 0.5,    # NF
    1 << 1: 0.5,    # EZ
    1 << 3: 1.06,   # HD
    1 << 4: 1.12,   # HR
    1 << 6: 1.06,   # DT
    1 << 8: 0.3,    # HT (and Daycore)
    1 << 9: 1.06,   # NC (same as DT; the NC bit implies DT — dedup below)
    1 << 10: 1.12,  # FL
}
_RELAX = 1 << 7
_SCORE_V2 = 1 << 29


def legacy_mod_multiplier(mods: int) -> float:
    mods = int(mods or 0)
    if mods & _RELAX:            # CatchModRelax → return 0
        return 0.0
    if mods & (1 << 9):          # NC is stored DT|NC — count the rate once
        mods &= ~(1 << 6)
    m = 1.0
    for bit, mult in _LEGACY_MULT.items():
        if mods & bit:
            m *= mult
    return m


# ── CalculateDifficultyPeppyStars (LegacyRulesetExtensions) ────────────────

def _f32(v: float) -> float:
    """C# BeatmapDifficulty stores float32; mirror (decimal)(double)(float)v."""
    return unpack("f", pack("f", float(v)))[0]


def difficulty_peppy_stars(hp: float, od: float, cs: float,
                           object_count: int, drain_length_s: int) -> int:
    if drain_length_s != 0:
        otd = Decimal(object_count) / Decimal(drain_length_s) * 8
        otd = max(Decimal(0), min(Decimal(16), otd))
    else:
        otd = Decimal(16)
    total = (Decimal(_f32(hp)) + Decimal(_f32(od)) + Decimal(_f32(cs)) + otd)
    return int((total / Decimal(38) * Decimal(5))
               .quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


# ── raw .osu facts for the peppy-stars inputs ──────────────────────────────
# The legacy sim uses the BASE beatmap (no mods): raw HP/OD/CS, the raw
# circle/slider/spinner count, and drain seconds = (last object start −
# first object start − Σ break length) / 1000 (integer division), with each
# time Math.Round()ed (banker's) first — CatchLegacyScoreSimulator.Simulate.

def _round_even(x: float) -> int:
    return int(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def parse_base_osu_facts(osu_path: Path) -> dict:
    text = Path(osu_path).read_text(encoding="utf-8", errors="replace")
    section = None
    hp = 5.0
    od = 7.0
    cs = 5.0
    n_objects = 0
    first_t: float | None = None
    last_t: float | None = None
    break_ms = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            section = m.group(1).lower()
            continue
        if section == "difficulty":
            k, _, v = line.partition(":")
            k = k.strip().lower()
            try:
                fv = float(v.strip())
            except ValueError:
                continue
            if k == "hpdrainrate":
                hp = fv
            elif k == "overalldifficulty":
                od = fv
            elif k == "circlesize":
                cs = fv
        elif section == "events":
            # break line: "2,start,end" (or the "Break,start,end" alias)
            parts = line.split(",")
            if len(parts) >= 3 and parts[0].strip() in ("2", "Break"):
                try:
                    bs, be = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                break_ms += _round_even(be) - _round_even(bs)
        elif section == "hitobjects":
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                t = float(parts[2])
            except ValueError:
                continue
            n_objects += 1
            if first_t is None:
                first_t = t
            last_t = t
    drain_s = 0
    if n_objects and first_t is not None and last_t is not None:
        drain_s = (_round_even(last_t) - _round_even(first_t) - break_ms) // 1000
    return {"hp": hp, "od": od, "cs": cs, "object_count": n_objects,
            "drain_length_s": int(drain_s)}


# ── CatchLegacyScoreSimulator.Simulate (ScoreV1 attributes) ────────────────

def legacy_attributes(objects, osu_facts: dict) -> dict:
    """Walk the CONVERTED catch objects (engine parse order == lazer nested
    order) accumulating stable ScoreV1 attributes. Returns the
    LegacyScoreAttributes equivalent:
      accuracy_score, combo_score, bonus_score, bonus_ratio, max_combo
    """
    peppy = float(difficulty_peppy_stars(
        osu_facts["hp"], osu_facts["od"], osu_facts["cs"],
        osu_facts["object_count"], osu_facts["drain_length_s"]))
    acc_score = 0
    combo_score = 0
    legacy_bonus = 0
    standardised_bonus = 0
    combo = 0
    for o in objects:
        if o.kind is ObjType.TINY_DROPLET:
            acc_score += 10                      # no combo
        elif o.kind is ObjType.DROPLET:
            acc_score += 100
            combo += 1                           # combo, but NOT combo-scored
        elif o.kind is ObjType.FRUIT:
            # (int)(max(0, combo-1) * (300/25 * peppy)) — 300/25 is C# int
            # division (=12), the outer cast truncates toward zero.
            combo_score += int(max(0, combo - 1) * (12 * peppy))
            acc_score += 300
            combo += 1
        elif o.kind is ObjType.BANANA:
            legacy_bonus += 1100
            standardised_bonus += 200            # CatchScoreProcessor LargeBonus
    return {
        "accuracy_score": acc_score,
        "combo_score": combo_score,
        "bonus_score": legacy_bonus,
        "bonus_ratio": (standardised_bonus / legacy_bonus) if legacy_bonus else 0.0,
        "max_combo": combo,
        "peppy_stars": int(peppy),
    }


# ── estimateComboProportionForCatch (StandardisedScoreMigrationTools) ──────

def _best_case_combo_total(max_combo: int) -> float:
    if max_combo == 0:
        return 1.0
    total = 0.5 * min(max_combo, 2)
    if max_combo <= 2:
        return total
    # ∫₂^x log₄(t) dt
    m = min(max_combo, 200)
    total += (m * (math.log(m) - 1) + 2 - math.log(4)) / math.log(4)
    if max_combo <= 200:
        return total
    total += (max_combo - 200) * math.log(200) / math.log(4)
    return total


def _dropped_after_miss(length_after: int) -> float:
    if length_after >= 200:
        length_after = 200
    # ∫₀^x (log₄(200) − log₄(t)) dt
    return length_after * (1 + math.log(200) - math.log(length_after)) / math.log(4)


def estimate_combo_proportion(beatmap_max_combo: int, score_max_combo: int,
                              score_miss_count: int) -> float:
    if beatmap_max_combo == 0:
        return 1.0
    if score_max_combo == 0:
        return 0.0
    if beatmap_max_combo == score_max_combo:
        return 1.0
    best_total = _best_case_combo_total(beatmap_max_combo)
    remaining = beatmap_max_combo - (score_max_combo + score_miss_count)
    dropped = 0.0
    # C#: (int)Math.Floor(remaining / missCount) — miss==0 gives +inf whose
    # int-cast lands the code in the else branch; mirror that explicitly.
    assumed_len = (math.floor(remaining / score_miss_count)
                   if score_miss_count > 0 else -1)
    if assumed_len > 0:
        combos_count = math.floor(remaining / assumed_len)
        dropped += combos_count * _dropped_after_miss(int(assumed_len))
        remaining -= combos_count * assumed_len
        if remaining > 0:
            dropped += _dropped_after_miss(int(remaining))
    else:
        dropped = best_total - _best_case_combo_total(score_max_combo)
    if best_total == 0:
        return 1.0
    return 1.0 - max(0.0, min(1.0, dropped / best_total))


# ── stable ScoreV1 total → standardised (case 2) ───────────────────────────

def stable_to_standardised(meta, attrs: dict, new_mod_multiplier: float) -> int:
    """convertFromLegacyTotalScore, ruleset 2. `meta` supplies the header
    total + counts; `attrs` comes from legacy_attributes()."""
    mods = int(getattr(meta, "mods", 0) or 0)
    legacy_total = int(meta.score)
    if mods & _SCORE_V2:
        # ScoreV2 mod: the header is already 1M-standardised (line 118-119
        # of StandardisedScoreMigrationTools) — TotalScore stays as-is.
        return legacy_total

    legacy_mult = legacy_mod_multiplier(mods)
    max_acc_score = attrs["accuracy_score"]
    max_combo_score = round(attrs["combo_score"] * legacy_mult)
    bonus_ratio = attrs["bonus_ratio"]
    max_base = max_acc_score + max_combo_score
    bonus_proportion = max(0.0, (legacy_total - max_base) * bonus_ratio)

    # server-side maximum-statistics folding for stable scores:
    #   max(Great) = count_300 + count_miss, max(SmallTick) = count_50 + katu
    max_great = int(meta.count_300) + int(meta.count_miss)
    max_tiny = int(meta.count_50) + int(meta.count_katu)
    divisor = max_tiny + max_great
    fruit_tiny_scale = (max_tiny / divisor) if divisor else 0.0

    max_tiny_portion = 400_000
    combo_portion = MAX_SCORE - max_tiny_portion + max_tiny_portion * (1 - fruit_tiny_scale)
    droplets_portion = max_tiny_portion * fruit_tiny_scale
    droplets_hit = (int(meta.count_50) / max_tiny) if max_tiny else 0.0

    ecp = estimate_combo_proportion(
        int(attrs["max_combo"]), int(meta.max_combo), int(meta.count_miss))
    without_mods = round(combo_portion * ecp
                         + droplets_portion * droplets_hit
                         + bonus_proportion)
    return int(round(without_mods * new_mod_multiplier))


# ── classic ↔ standardised (exact lazer display conversion) ────────────────

def standardised_to_classic(std: int, n_fruits: int) -> int:
    return int(round((std / MAX_SCORE * n_fruits) ** 2 * 21.62 + std / 10.0))


def classic_to_standardised(classic: int, n_fruits: int) -> int:
    """Positive root of 21.62·(n/1e6)²·s² + s/10 − classic = 0, then snapped
    to the exact preimage when one exists (classic headers are rounded)."""
    classic = int(classic)
    if classic <= 0:
        return 0
    if n_fruits <= 0:
        s = classic * 10
    else:
        a = 21.62 * (n_fruits / MAX_SCORE) ** 2
        s = (-0.1 + math.sqrt(0.01 + 4.0 * a * classic)) / (2.0 * a)
    best = int(round(s))
    for cand in range(max(0, best - 3), best + 4):   # snap to exact preimage
        if standardised_to_classic(cand, n_fruits) == classic:
            return cand
    return best


# ── candidate resolution ───────────────────────────────────────────────────

LAZER_GAME_VERSION_BOUNDARY = 30_000_000


def compute_candidates(meta, objects, osu_path, new_mod_multiplier: float) -> dict:
    """All defensible standardised interpretations of the header score."""
    n_fruits = sum(1 for o in objects if o.kind is ObjType.FRUIT)
    facts = parse_base_osu_facts(osu_path)
    attrs = legacy_attributes(objects, facts)
    gv = int(getattr(meta, "game_version", 0) or 0)
    header = int(meta.score)

    cands: dict[str, int | None] = {
        # header is a stable ScoreV1 total (true osu!stable play)
        "stable_v1": stable_to_standardised(meta, attrs, new_mod_multiplier),
        # header is lazer's classic display total (osu-web export of a lazer
        # play, or an older lazer client export)
        "lazer_classic": classic_to_standardised(header, n_fruits),
        # header is the standardised total itself (current lazer local export)
        "lazer_direct": header if gv >= LAZER_GAME_VERSION_BOUNDARY else None,
    }
    return {
        "candidates": cands,
        "n_fruits": n_fruits,
        "beatmap_max_combo": attrs["max_combo"],
        "legacy_attrs": attrs,
        "osu_facts": facts,
        "game_version": gv,
        "header_score": header,
        "mods": int(getattr(meta, "mods", 0) or 0),
    }


def resolve_authoritative(fid: dict, sim_final: int) -> tuple[int, str]:
    """Pick the authoritative standardised total. Returns (score, source_tag).

    game_version < 30M (stable-format .osr): the header is ALWAYS a
    legacy-space (ScoreV1) total — a true stable play stores its real ScoreV1
    total, and an osu-web download of a LAZER play stores the server's
    synthesized legacy_total_score, which round-trips through the same
    convertFromLegacyTotalScore math back to the standardised number the
    player saw (verified on real plays: 163,573,200 → 940,575 == osu-web;
    14,287,550 → 625,838 == the player's lazer ~625k). So stable_v1 is the
    correct decode for BOTH sub-30M cases; sim proximity is only a sanity
    guard here, not a selector (the sim tracks the play's REAL combo
    distribution while lazer's conversion deliberately uses a worst-case
    combo estimate, so near-FC stable plays legitimately sim above the
    converted value).

    game_version ≥ 30M (lazer client export): current encoders write the
    standardised total itself (LegacyScoreEncoder: `(int)TotalScore`); older
    ones wrote the classic display total. Pick whichever interpretation is
    closer to the sim — they differ by ~10× so the choice is unambiguous.
    """
    gv = int(fid.get("game_version", 0) or 0)
    ref = max(1, int(sim_final))
    # ScoreV2 play: the header IS the standardised total already --
    # StandardisedScoreMigrationTools keeps a ScoreV2 TotalScore unchanged
    # (`if (mods.Any(mod => mod is ModScoreV2)) return (..., score.TotalScore)`)
    # and stable_to_standardised mirrors that (returns the header as-is). It
    # needs no sim-proximity sanity gate: the sim is an estimate, the V2
    # header is the real standardised number. (Before the mods_score_multiplier
    # V2 fix, an NF+V2 play simmed to ~half and the 50% gate here rejected the
    # exact header, shipping the halved sim as score_v3.)
    if int(fid.get("mods", 0) or 0) & _SCORE_V2:
        val = fid["candidates"].get("stable_v1")
        if val is not None and val >= 0:
            return int(val), "stable_v1"
    if gv < LAZER_GAME_VERSION_BOUNDARY:
        val = fid["candidates"].get("stable_v1")
        if val is not None and val >= 0:
            err = abs(int(val) - ref) / ref
            if err <= 0.5 or sim_final <= 0:
                return int(val), "stable_v1"
        return int(sim_final), "sim"
    best: tuple[float, int, str] | None = None
    for tag in ("lazer_direct", "lazer_classic"):
        val = fid["candidates"].get(tag)
        if val is None or val < 0:
            continue
        err = abs(int(val) - ref) / ref
        if best is None or err < best[0]:
            best = (err, int(val), tag)
    if best is None or best[0] > 0.35:
        return int(sim_final), "sim"
    return best[1], best[2]
