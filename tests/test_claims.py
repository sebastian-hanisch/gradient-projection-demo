"""Jede Zahl, die README, Hilfetexte und Beispieltexte nennen, ist hier belegt (Stadtgitter 8 x 8, 6 Zonen, Last 1,0, Seed 5, feste Netze ab Seed 100000).
Die Rechnung nutzt nur + - * / auf Python-Zahlen: Iterationszahlen sind auf allen Plattformen dieselben (Bänder von wenigen Iterationen trotzdem gegen Rundungsunterschiede);
Mittel über Netze werden mit Bändern geprüft."""

import statistics
from collections import Counter

import pytest

import gp_constants as C
import gp_evaluation as ev
import gp_frankwolfe as fw
import gp_paths as gp
import gp_scenario as sc

P = ev.DEFAULT_PARAMS


def _params(name):
    p = C.PRESETS[name]
    return ev.Params(p["net"], p["side"], p["zones"], p["load"], p["mode"], p["update"], p["alpha"], p["adaptive"], p["frozen"], p["iterations"], p["seed"])


def _run(name):
    params = _params(name)
    a = ev.analyse(params)
    return params, a["net"], a["result"]


def _its(r, tols=C.TOLS):
    return [r.first_below(t) for t in tols]


def _close(got, want, band=2):
    assert len(got) == len(want) and all((g is None and w is None) or (g is not None and w is not None and abs(g - w) <= (0 if w <= 5 else band)) for g, w in zip(got, want)), (got, want)


@pytest.fixture(scope="module")
def default_compare():
    return ev.compare(P)


def test_the_default_grid_and_the_gap_table(default_compare):
    """Dasselbe Netz wie in Stück 16 (224 Kanten, 30 Zonenpaare, 6 Läufe je Durchgang). Frank-Wolfe 4/21/102 (1e-5 in 200 nicht), konjugiert 4/15/24/69/145, Gradient Projection (Gauss-Seidel, 0,5) 3/6/11/16/20."""
    params, net, r = _run("🏙️ Stadtgitter")
    assert (net.m, len(net.zones), len(net.demand), len(net.origins())) == (224, 6, 30, 6)
    c = default_compare
    _close(c["table"]["fw"], [4, 21, 102, None, None])
    _close(c["table"]["cfw"], [4, 15, 24, 69, 145])
    _close(c["table"]["gp"], [3, 6, 11, 16, 20])
    assert r.first_below(1e-6) == 20 and r.iterations == 20 and r.runs == 21 * 6
    assert r.npaths[-1] == 40 and r.deleted[-1] == 31 and max(len(s) for s in r.snaps[-1]) == 3
    assert sorted(Counter(len(s) for s in r.snaps[-1]).items()) == [(1, 21), (2, 8), (3, 1)]              # 21 Zonenpaare mit einem Weg, 8 mit zwei, eines mit drei


def test_the_effort_in_edge_operations(default_compare):
    """Bis 1e-4: 61 600 (Gradient Projection) gegen 641 312 (Frank-Wolfe) und 152 096 (konjugiert): das Zehnfache und das 2,5-fache."""
    ops = {m: default_compare["ops"][m][2] for m in ("fw", "cfw", "gp")}
    assert ops["gp"] == pytest.approx(61600, rel=0.01) and ops["fw"] == pytest.approx(641312, rel=0.01) and ops["cfw"] == pytest.approx(152096, rel=0.01)
    assert ops["fw"] / ops["gp"] == pytest.approx(10.4, abs=0.2) and ops["cfw"] / ops["gp"] == pytest.approx(2.5, abs=0.1)
    per = {m: ops[m] / default_compare["table"][m][2] for m in ops}
    assert per["gp"] == pytest.approx(5600, rel=0.02) and per["fw"] == pytest.approx(6300, rel=0.02)          # eine Iteration kostet etwa gleich viel: der Gewinn steckt in den Iterationen


def test_the_paths_of_frank_wolfe_and_gradient_projection(default_compare):
    """Frank-Wolfe hat bis 1e-4 (Iteration 102) 70 verschiedene Wege erzeugt, alle mit Fluss; Gradient Projection trägt bei 1e-4 43 Wege und am Ende 40 (bis zu 4 je Zonenpaar), 26 bzw. 31 gelöscht."""
    pv = ev.paths_view(P, default_compare)
    assert pv["upto"] == 102 and pv["fw_curve"][-1] == 70 and pv["fw_curve"][21] == 69
    g = default_compare["results"]["gp"]
    assert g.npaths[11] == 43 and g.deleted[11] == 26 and max(len(s) for s in g.snaps[11]) == 4


def test_jacobi_is_slower_and_more_fragile():
    _, _, r = _run("🔁 Jacobi")
    _close(_its(r), [4, 8, 14, 20, 24])
    net = ev.network(P)
    assert ev.run_gp(net, P, 200, update="jacobi", alpha=0.75, tol=1e-5).first_below(1e-4) is None
    assert ev.run_gp(net, P, 200, update="gauss_seidel", alpha=0.75, tol=1e-5).first_below(1e-4) == 25


def test_step_one_zigzags():
    """Der volle Newton-Schritt: unter 1e-2 nach 9 Iterationen, dann nicht weiter; nach 100 Lücke 1,6e-2, Gesamtfahrzeit 7 037 (2,0 % über 6 897), 228 gelöschte Wege."""
    _, net, r = _run("⚠️ Schritt 1,0")
    assert r.first_below(1e-2) == 9 and r.first_below(1e-3) is None and r.iterations == 100
    assert r.gaps[-1] == pytest.approx(1.6e-2, rel=0.05) and r.tstt[-1] == pytest.approx(7037, abs=1) and r.tstt[-1] / 6897.2 == pytest.approx(1.020, abs=0.001) and r.deleted[-1] == 228


def test_the_adaptive_step():
    _, _, r = _run("🛡️ Adaptiv")
    _close(_its(r), [7, 9, 17, 26, 30])
    assert r.alphas[-1] == 0.5 and r.tstt[-1] == pytest.approx(6897, abs=1)


def test_high_load_adaptive_against_frank_wolfe_and_fixed_steps():
    params, net, r = _run("🚗 Hohe Last, adaptiv")
    _close(_its(r), [11, 47, 89, 114, 157], band=3)
    assert r.alphas[-1] == 0.125 and r.npaths[-1] == 90 and max(len(s) for s in r.snaps[-1]) == 20 and max(fw.utilization(net, r.x)) == pytest.approx(2.23, abs=0.01)
    c = ev.compare(params)
    _close(c["table"]["fw"][:3], [35, None, None])
    _close(c["table"]["cfw"][:3], [22, 70, None], band=3)
    for a in (0.5, 1.0):
        fixed = ev.run_gp(net, params, 200, alpha=a, adaptive=False)
        assert fixed.first_below(1e-3) is None
    assert ev.run_gp(net, params, 200, alpha=0.5, adaptive=False).gaps[-1] == pytest.approx(9.3e-2, rel=0.05)
    assert ev.run_gp(net, params, 200, alpha=1.0, adaptive=False).gaps[-1] == pytest.approx(6.3e-1, rel=0.05)


def test_system_optimum():
    """Gradient Projection 8/16/25/36/48 (Gesamtfahrzeit 6 632); Frank-Wolfe 15/79 (1e-4 in 200 nicht), konjugiert 10/29/122; das konjugierte Verfahren steht nach 800 Iterationen bei Lücke 2e-5; Preis der Anarchie 1,040."""
    params, net, r = _run("🌐 Systemoptimum")
    _close(_its(r), [8, 16, 25, 36, 48])
    assert r.tstt[-1] == pytest.approx(6632, abs=1)
    c = ev.compare(params)
    _close(c["table"]["fw"][:3], [15, 79, None])
    _close(c["table"]["cfw"][:3], [10, 29, 122])
    assert fw.assign(net, "so", "cfw", 800, 1e-9).gaps[-1] == pytest.approx(2.1e-5, rel=0.1)
    ue = ev.analyse(P)["result"]
    assert ue.tstt[-1] / r.tstt[-1] == pytest.approx(1.040, abs=0.001) and ue.tstt[-1] == pytest.approx(6897, abs=1)


def test_two_stages_by_hand():
    """Nachfrage 10, je Straße 5 nach einer Iteration; fünf Starts, drei verschiedene Wegeaufteilungen."""
    _, net, r = _run("🪜 Zwei Stufen")
    assert r.iterations == 1 and r.x == (5.0, 5.0, 5.0, 5.0) and net.demand[0][2] == 10.0
    rows = ev.lesson_starts()
    assert [dict(x["paths"]) for x in rows] == [{(0, 2): 5.0, (1, 3): 5.0}, {(0, 3): 5.0, (1, 2): 5.0}, {(1, 2): 5.0, (0, 3): 5.0}, {(1, 3): 5.0, (0, 2): 5.0},
                                                {(0, 2): 2.5, (0, 3): 2.5, (1, 2): 2.5, (1, 3): 2.5}]


def test_frozen_lists():
    """Nur in den ersten 2 Iterationen neue Wege: unter 1e-2 nach 3, nach 100 Lücke 4,1e-3, Gesamtfahrzeit 6 983 (1,2 % über 6 897)."""
    _, _, r = _run("🧊 Wege eingefroren")
    assert r.first_below(1e-2) == 3 and r.first_below(1e-3) is None and r.iterations == 100
    assert r.gaps[-1] == pytest.approx(4.1e-3, rel=0.05) and r.tstt[-1] == pytest.approx(6983, abs=1) and r.tstt[-1] / 6897.2 == pytest.approx(1.012, abs=0.001)


def test_alpha_sweep_on_the_default_grid():
    """Gauss-Seidel bis 1e-4: 0,25 → 25, 0,5 → 11, 0,75 → 25, 1 und 1,5 nicht; Jacobi 28, 14, sonst nicht; adaptiv 17 (Gauss-Seidel) und 16 (Jacobi)."""
    s = ev.alpha_sweep(P)
    gs = {r["alpha"]: r["gauss_seidel"] for r in s["rows"]}
    jc = {r["alpha"]: r["jacobi"] for r in s["rows"]}
    _close([gs[0.25], gs[0.5], gs[0.75], gs[1.0], gs[1.5]], [25, 11, 25, None, None])
    _close([jc[0.25], jc[0.5], jc[0.75], jc[1.0], jc[1.5]], [28, 14, None, None, None])
    _close([s["adaptive"]["gauss_seidel"], s["adaptive"]["jacobi"]], [17, 16])


def test_load_series_on_five_fixed_nets():
    """Iterationen bis 1e-4 (Mittel über 5 Netze, nicht erreicht = 201): Frank-Wolfe, konjugiert, Schritt 0,5, adaptiv; ein fester Schritt versagt bei hoher Last, adaptiv kommt in 4 von 5 Netzen bei Last 2,0 an."""
    rows = {r["load"]: r for r in ev.load_series(P)}
    assert rows[10]["its"]["fw"] == pytest.approx(178, abs=3) and rows[10]["its"]["cfw"] == pytest.approx(23, abs=3) and rows[10]["its"]["gp"] == pytest.approx(48, abs=3) and rows[10]["its"]["gp_adaptive"] == pytest.approx(12.2, abs=1.5)
    assert rows[5]["its"]["gp"] == pytest.approx(2.8, abs=0.5) and rows[5]["missed"]["gp_alpha1"] == 2
    assert [rows[l]["missed"]["gp"] for l in (5, 10, 15, 20, 30)] == [0, 1, 4, 5, 5]
    assert [rows[l]["missed"]["gp_adaptive"] for l in (5, 10, 15, 20, 30)] == [0, 0, 0, 1, 3]
    assert [rows[l]["missed"]["fw"] for l in (5, 10, 15, 20, 30)] == [0, 2, 5, 5, 5] and [rows[l]["missed"]["cfw"] for l in (5, 10, 15, 20, 30)] == [0, 0, 2, 5, 5]
    assert [5 - rows[l]["missed"]["gp_adaptive"] for l in (5, 10, 15, 20, 30)] == [5, 5, 5, 4, 2] and [5 - rows[l]["missed"]["gp"] for l in (5, 10, 15, 20, 30)] == [5, 4, 1, 0, 0]
    assert rows[20]["its"]["gp_adaptive"] == pytest.approx(123, abs=8) and rows[30]["its"]["gp_adaptive"] == pytest.approx(179, abs=8)


def test_distribution_over_40_fixed_nets():
    """Bis 1e-4 in 200 Iterationen (nicht erreicht = 201): Frank-Wolfe Mittel 159, 22 von 40 nicht; konjugiert 70, 8; Schritt 0,5: Median 19 (konjugiert 38,5), Mittel 71, 11 nicht; adaptiv: Median 16, Mittel 20, alle 40 erreicht,
    schneller als das konjugierte Verfahren in 37, langsamer in 2; Schritt 1: 33 nicht erreicht."""
    d = ev.distribution(P)
    assert d["n"] == 40
    assert d["fw_mean"] == pytest.approx(159, abs=3) and d["fw_missed"] == 22 and d["cfw_mean"] == pytest.approx(70, abs=3) and d["cfw_missed"] == 8
    assert d["gp_median"] == pytest.approx(19, abs=2) and d["cfw_median"] == pytest.approx(38.5, abs=3) and d["gp_mean"] == pytest.approx(71, abs=4) and d["gp_missed"] == 11
    assert d["gp_adaptive_median"] == pytest.approx(16, abs=2) and d["gp_adaptive_mean"] == pytest.approx(20, abs=2) and d["gp_adaptive_missed"] == 0
    assert d["gp_adaptive_beats_cfw"] == pytest.approx(37, abs=1) and d["gp_adaptive_loses_to_cfw"] == pytest.approx(2, abs=1) and d["gp_alpha1_missed"] == pytest.approx(33, abs=2)
    assert d["gp_beats_cfw"] == pytest.approx(25, abs=2) and d["gp_loses_to_cfw"] == pytest.approx(13, abs=2)
    med = {m: statistics.median(r[m]["ops"] for r in d["rows"] if r[m]["ops"] is not None) for m in ("fw", "cfw", "gp_adaptive")}          # nur Netze, in denen 1e-4 erreicht wurde
    assert med["gp_adaptive"] == pytest.approx(90118, rel=0.03) and med["cfw"] == pytest.approx(202272, rel=0.03) and med["fw"] == pytest.approx(798112, rel=0.03)
    assert sum(1 for r in d["rows"] if r["cfw"]["ops"] is not None) == 32 and sum(1 for r in d["rows"] if r["fw"]["ops"] is not None) == 18 and d["gp_alpha1_missed"] == 33


def test_edge_flows_are_unique_and_the_wardrop_condition_holds_on_the_paths():
    u = ev.uniqueness(P)
    assert u["max_diff"] < 1e-5 and u["deviation"] < 1e-5 and u["paths"] == 43 and u["pairs"] == 30 and u["gap_b"] < 1e-8 and u["gap_a"] <= 1e-9
