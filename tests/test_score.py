"""The leaderboard's numbers, checked by hand on a four-point grid."""

import math

import numpy as np
import pandas as pd
import pytest

from loe.score import metrics

NAN = float("nan")

# weight, truth, truth_lakes, pred, p_land, mass, source, text
POINTS = [
    (1, 0, 0, 0, 0.2, 0.9, "logprobs", "Water"),
    (2, 1, 1, 1, 0.7, 0.8, "logprobs", "The answer is Land"),  # read from logprobs, text starts a sentence
    (1, 1, 0, 1, NAN, NAN, "text", "Land"),  # land in the mask, water once lakes count as water
    (4, 0, 0, NAN, NAN, NAN, None, None),  # never answered: counts as wrong
]


def grid(points=POINTS) -> pd.DataFrame:
    cols = ["weight", "truth", "truth_lakes", "pred", "p_land", "mass", "source", "text"]
    df = pd.DataFrame(points, columns=cols)
    df["point_id"] = range(len(df))
    return df


def test_area_weighted_scores():
    m = metrics(grid())
    assert m["points"] == 4 and m["answered"] == 3
    assert m["coverage"] == pytest.approx(4 / 8)
    assert m["always_water"] == pytest.approx(5 / 8)
    assert m["accuracy"] == pytest.approx(4 / 8)  # the heavy unanswered point weighs on it
    assert m["skill"] == pytest.approx((4 / 8 - 5 / 8) / (3 / 8))  # worse than always Water
    assert m["accuracy_lakes"] == pytest.approx(3 / 8)
    assert m["skill_lakes"] == pytest.approx((3 / 8 - 6 / 8) / (2 / 8))
    assert m["land_recall"] == pytest.approx(1.0) and m["land_precision"] == pytest.approx(1.0)
    assert m["land_share_predicted"] == pytest.approx(3 / 8)


def test_probability_scores_use_only_points_with_logprobs():
    m = metrics(grid())
    assert m["logprobs_share"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx((1 * 0.2**2 + 2 * 0.3**2) / 3)
    assert m["median_mass"] == pytest.approx(0.85)


def test_indirect_answers():
    m = metrics(grid())
    assert m["indirect_share"] == pytest.approx(1 / 3)
    assert m["indirect_median_mass"] == pytest.approx(0.8)
    assert m["indirect_example"] == "The answer…"


def test_always_water_has_zero_skill_and_no_land_precision():
    points = [p[:3] + (0, NAN, NAN, "text", "Water") for p in POINTS]
    m = metrics(grid(points))
    assert m["accuracy"] == pytest.approx(m["always_water"]) and m["skill"] == pytest.approx(0.0)
    assert m["land_recall"] == 0.0 and math.isnan(m["land_precision"])
    assert "brier" not in m and m["logprobs_share"] == 0.0 and m["indirect_share"] == 0.0


def test_land_answered_on_water_lowers_precision():
    points = [POINTS[0][:3] + (1, NAN, NAN, "text", "Land")] + POINTS[1:]
    m = metrics(grid(points))
    assert m["land_precision"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(3 / 8)
    assert np.isfinite(m["skill"])
