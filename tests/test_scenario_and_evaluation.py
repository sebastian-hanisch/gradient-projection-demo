"""Szenario (Kopie aus Stück 16) und Auswertung (Vergleich, Wegemengen, Schrittweite, Lastreihe, Verteilung, Eindeutigkeit)."""

import pytest

import gp_constants as C
import gp_evaluation as ev
import gp_frankwolfe as fw
import gp_paths as gp
import gp_scenario as sc

P = ev.DEFAULT_PARAMS


def test_generation_is_deterministic_and_seed_dependent():
    a, b, c = sc.generate(6, 5, 10, 5), sc.generate(6, 5, 10, 5), sc.generate(6, 5, 10, 6)
    assert a == b and a != c


def test_grid_structure():
    """8 x 8 Kreuzungen: 2 * (7 * 8 + 8 * 7) = 224 Kanten; 6 Zonen, 30 geordnete Zonenpaare, 6 Ursprünge."""
    n = sc.generate(8, 6, 10, 5)
    assert n.n == 64 and n.m == 224 and len(n.zones) == 6 == len(set(n.zones)) and len(n.demand) == 30 and len(n.origins()) == 6


def test_lessons_and_build_dispatch():
    assert set(sc.LESSONS) == set(C.FIXED_NETS) and sc.build("pigou", 9, 9, 9, 9).m == 2
    assert sc.build("grid", 6, 5, 10, 5) == sc.generate(6, 5, 10, 5)


def test_analyse_and_compare_shapes():
    a = ev.analyse(P._replace(iterations=30))
    assert a["result"].iterations <= 30 and a["net"].m == 224
    c = ev.compare(P, cap=40)
    assert set(c["results"]) == {"fw", "cfw", "gp"} and set(c["table"]) == set(c["ops"]) == {"fw", "cfw", "gp"} and all(len(v) == len(C.TOLS) for v in c["table"].values())
    assert c["ops"]["gp"][0] == c["results"]["gp"].ops[c["table"]["gp"][0]]


def test_paths_view_follows_frank_wolfe_to_1e_4():
    c = ev.compare(P, cap=110)
    pv = ev.paths_view(P, c)
    assert pv["upto"] == 102 and len(pv["fw_curve"]) == 103 and pv["gp_paths"][-1] < pv["fw_curve"][-1]


def test_frozen_setting_reaches_the_algorithm():
    r = ev.analyse(P._replace(frozen=True))["result"]
    assert max(r.npaths[C.FROZEN_UNTIL + 1:]) <= r.npaths[C.FROZEN_UNTIL + 1] and r.gaps[-1] > 1e-3


def test_run_gp_overrides_only_what_it_is_told():
    net = ev.network(P)
    a = ev.run_gp(net, P, 30, update="jacobi")
    assert a.update == "jacobi" and a.alpha == P.alpha
    assert ev.run_gp(net, P._replace(adaptive=True), 30).alphas[-1] <= P.alpha


def test_alpha_sweep_shape_and_the_window():
    s = ev.alpha_sweep(P._replace(side=5, zones=4), alphas=(0.5, 2.0), cap=60)
    assert [r["alpha"] for r in s["rows"]] == [0.5, 2.0] and set(s["adaptive"]) == set(gp.UPDATES)
    assert s["rows"][1]["gauss_seidel"] is None                   # Schritt 2 kommt nie an


def test_distribution_and_load_series_shapes():
    p = P._replace(side=5, zones=4)
    d = ev.distribution(p, seeds=C.SWEEP_SEEDS[:3], cap=60)
    assert d["n"] == 3 and all(0 <= d[m + "_missed"] <= 3 for m in ev.METHOD_KEYS) and d["cfw_beats_cfw"] == 0
    rows = ev.load_series(p, loads=(5, 15), seeds=C.LOAD_SEEDS[:2], cap=60)
    assert [r["load"] for r in rows] == [5, 15] and all(set(r["its"]) == set(ev.METHOD_KEYS) for r in rows) and rows[0]["its"]["fw"] <= rows[1]["its"]["fw"]


def test_uniqueness_on_a_small_grid():
    u = ev.uniqueness(P._replace(side=5, zones=4))
    assert u["max_diff"] < 1e-5 and u["deviation"] < 1e-4 and u["paths"] >= u["pairs"]


def test_lesson_starts_give_three_decompositions():
    rows = ev.lesson_starts()
    assert len(rows) == 5 and all(r["edges"] == [5.0] * 4 for r in rows) and len({frozenset(r["paths"]) for r in rows}) == 3


def test_pair_helpers():
    net = ev.network(P)
    assert ev.pair_label(net, 0).startswith("Z") and ev.pair_label(sc.pigou(), 0).startswith("A → B")
    r = ev.analyse(P)["result"]
    i = ev.busiest_pair(r)
    assert len(r.snaps[-1][i]) == max(len(s) for s in r.snaps[-1])
    assert ev.wardrop_check(net, r.x) < 1e-5
