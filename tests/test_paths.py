"""Gradient Projection: Lehrnetze von Hand, Flusserhaltung, Wardrop-Bedingung, Grenzwert gleich Frank-Wolfe, Kontrollen (Schritt zu groß, Wege eingefroren), Zähler."""

import pytest

import gp_evaluation as ev
import gp_frankwolfe as fw
import gp_paths as gp
import gp_scenario as sc

NET = sc.generate(8, 6, 10, 5)
TOTAL = NET.total_demand()


def _flows_are_valid(net, res):
    for snap in res.snaps:
        for i, paths in enumerate(snap):
            assert all(h >= 0.0 for _l, h in paths)
            assert abs(sum(h for _l, h in paths) - net.demand[i][2]) < 1e-9
    for k, x in enumerate(res.flows):
        again = gp.edge_flows(net, [dict(p) for p in res.snaps[k]])
        assert max(abs(a - b) for a, b in zip(x, again)) < 1e-9


def test_pigou_user_equilibrium_is_immediate_and_system_optimum_needs_one_step():
    net = sc.pigou()
    ue = gp.assign_paths(net, "ue", "gauss_seidel", 1.0, 50, 1e-9)
    assert ue.iterations == 0 and ue.tstt[-1] == 1.0 and ue.snaps[-1] == ((((1,), 1.0),),)
    so = gp.assign_paths(net, "so", "gauss_seidel", 1.0, 50, 1e-9)
    assert so.iterations == 1 and dict(so.snaps[-1][0]) == {(0,): 0.5, (1,): 0.5} and so.tstt[-1] == 0.75
    assert ue.tstt[-1] / so.tstt[-1] == pytest.approx(4 / 3)


def test_two_stages_reaches_five_on_every_street_from_every_start():
    net = sc.two_stages()
    distinct = set()
    for index in range(5):
        res = gp.assign_paths(net, "ue", "gauss_seidel", 1.0, 50, 1e-9, init=gp.lesson_start(net, index))
        assert res.x == (5.0, 5.0, 5.0, 5.0)
        distinct.add(frozenset(res.snaps[-1][0]))
    assert len(distinct) == 3                       # (1,3)+(2,4), (1,4)+(2,3), alle vier je 2,5: Wegeflüsse hängen vom Start ab
    assert gp.lesson_start(sc.pigou(), 0) is None
    assert ev.lesson_starts()[4]["iterations"] == 0


def test_the_start_has_one_path_per_pair_and_the_full_demand_on_it():
    res = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 1, 1e-6)
    assert all(len(p) == 1 and p[0][1] == NET.demand[i][2] for i, p in enumerate(res.snaps[0]))
    assert res.snaps[0][0][0][0] == gp.trace(NET, gp.shortest_tree(gp._adjacency(NET), NET.n, fw.Coefs(NET).costs("ue", [0.0] * NET.m), NET.demand[0][0])[1], NET.demand[0][0], NET.demand[0][1])


@pytest.mark.parametrize("update", gp.UPDATES)
@pytest.mark.parametrize("alpha", (0.5, 2.0))
def test_flow_conservation_and_nonnegative_flows_even_for_a_step_that_is_too_big(update, alpha):
    res = gp.assign_paths(NET, "ue", update, alpha, 30, 1e-6)
    _flows_are_valid(NET, res)


def test_adaptive_step_and_frozen_lists_keep_the_flows_valid():
    _flows_are_valid(NET, gp.assign_paths(NET, "ue", "gauss_seidel", 1.0, 30, 1e-6, adaptive=True))
    _flows_are_valid(NET, gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 30, 1e-6, grow_until=2))


@pytest.mark.parametrize("update", gp.UPDATES)
def test_wardrop_condition_holds_on_the_paths_and_the_limit_equals_frank_wolfe(update):
    res = gp.assign_paths(NET, "ue", update, 0.5, 200, 1e-7)
    assert res.gaps[-1] < 1e-7
    assert gp.wardrop_deviation(NET, "ue", res.x, res.snaps[-1]) < 1e-5
    ref = fw.assign(NET, "ue", "cfw", 800, 1e-9)
    assert max(abs(res.x[k] - ref.x[k]) for k in range(NET.m)) / TOTAL < 1e-5
    assert ev.wardrop_check(NET, res.x) < 1e-6


def test_system_optimum_is_the_equilibrium_on_marginal_costs_and_matches_frank_wolfe():
    res = gp.assign_paths(NET, "so", "gauss_seidel", 0.5, 200, 1e-7)
    ref = fw.assign(NET, "so", "cfw", 800, 1e-9)                    # kommt im Systemoptimum in 800 Iterationen nur auf Lücke 2e-5: die Fahrzeit stimmt auf 1e-5
    assert res.gaps[-1] < 1e-7 and abs(res.tstt[-1] - ref.tstt[-1]) / ref.tstt[-1] < 2e-5 and ref.gaps[-1] > 1e-6
    assert res.tstt[-1] < gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 200, 1e-7).tstt[-1]
    assert gp.wardrop_deviation(NET, "so", res.x, res.snaps[-1]) < 1e-5


def test_constant_costs_need_no_newton_step():
    """Ohne Kostenanstieg (Ableitung 0) verschiebt das Verfahren den ganzen Fluss auf einmal: die Absicherung gegen die Division durch null."""
    net = sc.Network(2, ("A", "B"), ((0.0, 0.0), (1.0, 0.0)), ((0, 1, 1.0, 0.0, 1.0, 1), (0, 1, 2.0, 0.0, 1.0, 1)), (0, 1), ((0, 1, 3.0),), "test")
    res = gp.assign_paths(net, "ue", "gauss_seidel", 0.5, 20, 1e-9, init=[{(1,): 3.0}])
    assert res.iterations == 1 and res.snaps[-1] == ((((0,), 3.0),),) and res.gaps[-1] == 0.0


def test_a_step_that_is_too_big_never_arrives_and_a_frozen_list_stalls():
    big = gp.assign_paths(NET, "ue", "gauss_seidel", 1.0, 100, 1e-6)
    assert big.first_below(1e-3) is None and big.deleted[-1] > 100         # zickzackt: Wege werden gelöscht und neu aufgenommen
    frozen = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 100, 1e-6, grow_until=2)
    assert frozen.first_below(1e-3) is None and max(frozen.npaths[3:]) <= frozen.npaths[3]
    good = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 100, 1e-6)
    assert good.first_below(1e-6) is not None


def test_the_adaptive_step_arrives_where_a_fixed_step_fails():
    net = sc.generate(8, 6, 20, 5)
    fixed = gp.assign_paths(net, "ue", "gauss_seidel", 1.0, 150, 1e-6)
    adaptive = gp.assign_paths(net, "ue", "gauss_seidel", 1.0, 150, 1e-6, adaptive=True)
    assert fixed.first_below(1e-3) is None and adaptive.first_below(1e-4) is not None
    assert adaptive.alphas[-1] < 1.0 and all(b <= a for a, b in zip(adaptive.alphas, adaptive.alphas[1:]))


def test_counters_and_reproducibility():
    a = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 40, 1e-6)
    b = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 40, 1e-6)
    assert a.flows == b.flows and a.gaps == b.gaps and a.ops == b.ops
    origins = len(NET.origins())
    assert a.runs == origins * (1 + a.iterations)
    assert all(y > x for x, y in zip(a.ops, a.ops[1:])) and len(a.ops) == len(a.flows) == len(a.npaths) == len(a.deleted) == len(a.gaps)
    assert all(y >= x for x, y in zip(a.deleted, a.deleted[1:]))
    j = gp.assign_paths(NET, "ue", "jacobi", 0.5, 40, 1e-6)
    assert j.runs == origins * (1 + j.iterations)


def test_frank_wolfe_copy_keeps_its_results_and_counts_operations():
    res = fw.assign(NET, "ue", "fw", 110, 1e-6)
    assert [res.first_below(t) for t in (1e-2, 1e-3, 1e-4)] == [4, 21, 102]
    assert len(res.ops) == len(res.flows) and all(y > x for x, y in zip(res.ops, res.ops[1:]))
    co = fw.Coefs(NET)
    x = [3.0] * NET.m
    assert co.derivs("ue", x)[0] == pytest.approx(fw.time_deriv(NET.links[0], 3.0)) and co.derivs("so", x)[5] == pytest.approx(fw.marginal_deriv(NET.links[5], 3.0))
    assert co.calls == 2


def test_tree_traces_agree_with_all_or_nothing():
    co = fw.Coefs(NET)
    cost = co.costs("ue", [4.0] * NET.m)
    y, _s, _r = fw.all_or_nothing(NET, cost)
    adj = gp._adjacency(NET)
    x = [0.0] * NET.m
    for o, dests in NET.origins().items():
        _d, pred, _sc = gp.shortest_tree(adj, NET.n, cost, o)
        for d, q in dests:
            for k in gp.trace(NET, pred, o, d):
                x[k] += q
    assert x == y


def test_generated_paths_of_frank_wolfe_exceed_the_active_paths_of_gradient_projection():
    res = fw.assign(NET, "ue", "fw", 110, 1e-6)
    curve = ev.fw_generated_paths(NET, "ue", res, 30)
    assert all(y >= x for x, y in zip(curve, curve[1:])) and curve[0] == len(NET.demand)
    g = gp.assign_paths(NET, "ue", "gauss_seidel", 0.5, 40, 1e-6)
    assert g.npaths[-1] < curve[-1]
