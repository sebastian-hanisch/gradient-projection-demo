"""Unabhängiges Orakel für Gradient Projection: (1) der Schritt der Formel Delta h = min(h, alpha (c_p - c_b) / (t'_p + t'_b)) auf parallelen Straßen
mit einer eigenen Rechnung (Jacobi und Gauss-Seidel fallen bei einem Zonenpaar zusammen); (2) die relative Lücke jeder Iteration mit einem eigenen
Kürzeste-Wege-Verfahren (Bellman-Ford statt Dijkstra); (3) die Gleichgewichts- und Systemoptimum-Kantenflüsse gegen eine Minimierung des Beckmann-Programms
bzw. der Gesamtfahrzeit über alle Wege (scipy, SLSQP) auf kleinen Gittern."""

import random

import numpy as np
import pytest

import gp_paths as gp
import gp_scenario as sc


def _t(l, x):
    return l[2] + l[3] * (x / l[4]) ** l[5]


def _dt(l, x):
    return l[3] * l[5] * (x / l[4]) ** (l[5] - 1) / l[4]


def _d2t(l, x):
    return l[3] * l[5] * (l[5] - 1) * (x / l[4]) ** (l[5] - 2) / l[4] ** 2


def _costs(net, mode, x):
    if mode == "ue":
        return [_t(l, xk) for l, xk in zip(net.links, x)]
    return [l[2] + l[3] * (l[5] + 1) * (xk / l[4]) ** l[5] for l, xk in zip(net.links, x)]


@pytest.mark.parametrize("seed", range(40))
def test_step_on_parallel_roads_matches_an_independent_computation(seed):
    rng = random.Random(seed)
    k, q = rng.randint(2, 5), rng.uniform(1, 30)
    links = tuple((0, 1, float(rng.randint(1, 6)), 0.15 * rng.randint(1, 6), float(rng.randint(5, 30)), rng.choice([2, 3, 4])) for _ in range(k))
    net = sc.Network(2, ("A", "B"), ((0, 0), (1, 1)), links, (0, 1), ((0, 1, q),), "pigou")
    mode, alpha, update = rng.choice(["ue", "so"]), rng.choice([0.25, 0.5, 1.0]), rng.choice(gp.UPDATES)
    h = [rng.random() for _ in range(k)]
    h = [q * v / sum(h) for v in h]
    res = gp.assign_paths(net, mode, update, alpha, 5, 1e-12, init=[{(j,): h[j] for j in range(k)}])
    cost = (lambda l, x: _t(l, x)) if mode == "ue" else (lambda l, x: _t(l, x) + x * _dt(l, x))
    der = (lambda l, x: _dt(l, x)) if mode == "ue" else (lambda l, x: 2 * _dt(l, x) + x * _d2t(l, x))
    for step in range(1, res.iterations + 1):
        c = [cost(links[j], h[j]) for j in range(k)]
        d = [der(links[j], h[j]) for j in range(k)]
        b = min(range(k), key=lambda j: (c[j], j))
        new = h[:]
        for j in range(k):
            if j != b and h[j] > 0 and c[j] > c[b]:
                delta = min(h[j], alpha * (c[j] - c[b]) / (d[j] + d[b]))
                new[j] -= delta
                new[b] += delta
        h = new
        snap = dict(res.snaps[step][0])
        assert max(abs(h[j] - snap.get((j,), 0.0)) for j in range(k)) < 1e-9


def _bellman_ford(net, cost, o):
    dist = [float("inf")] * net.n
    dist[o] = 0.0
    for _ in range(net.n):
        for (u, v, *_r), c in zip(net.links, cost):
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c
    return dist


@pytest.mark.parametrize("mode", ["ue", "so"])
@pytest.mark.parametrize("seed", range(6))
def test_gap_of_every_iteration_matches_an_independent_shortest_path_computation(mode, seed):
    net = sc.generate(4 + seed % 3, 4, 10 + 5 * (seed % 3), 1000 + seed)
    res = gp.assign_paths(net, mode, "gauss_seidel", 0.5, 12, 1e-12)
    for k, x in enumerate(res.flows):
        cost = _costs(net, mode, x)
        sx = sum(xk * c for xk, c in zip(x, cost))
        trees = {o: _bellman_ford(net, cost, o) for o in net.origins()}
        sy = sum(q * trees[o][d] for o, d, q in net.demand)
        assert res.gaps[k] == pytest.approx((sx - sy) / sx, abs=1e-12)


def _oracle_flows(net, mode):
    import networkx as nx
    from scipy.optimize import minimize

    graph = nx.DiGraph()
    index = {}
    for k, l in enumerate(net.links):
        graph.add_edge(l[0], l[1])
        index[(l[0], l[1])] = k
    paths = [(i, [index[(a, b)] for a, b in zip(p, p[1:])]) for i, (o, d, _q) in enumerate(net.demand) for p in nx.all_simple_paths(graph, o, d)]
    inc = np.zeros((net.m, len(paths)))
    for j, (_i, es) in enumerate(paths):
        inc[es, j] = 1.0
    a = np.array([l[2] for l in net.links])
    b = np.array([l[3] for l in net.links])
    c = np.array([l[4] for l in net.links])
    scale = 10.0

    def f(h):
        x = inc @ h * scale
        return ((a * x + b * c * (x / c) ** 5 / 5).sum() if mode == "ue" else (x * (a + b * (x / c) ** 4)).sum()) / scale

    def g(h):
        x = inc @ h * scale
        return inc.T @ (a + b * (x / c) ** 4 if mode == "ue" else a + 5 * b * (x / c) ** 4)

    rows = np.zeros((len(net.demand), len(paths)))
    for j, (i, _es) in enumerate(paths):
        rows[i, j] = 1.0
    rhs = np.array([q / scale for _o, _d, q in net.demand])
    h0 = np.array([rhs[i] / sum(1 for ii, _ in paths if ii == i) for i, _ in paths])
    res = minimize(f, h0, jac=g, constraints=[{"type": "eq", "fun": lambda h: rows @ h - rhs, "jac": lambda h: rows}], bounds=[(0, None)] * len(paths), method="SLSQP", options={"maxiter": 2000, "ftol": 1e-15})
    return inc @ res.x * scale


@pytest.mark.parametrize("mode", ["ue", "so"])
@pytest.mark.parametrize("seed", range(4))
def test_edge_flows_match_a_direct_minimisation_over_all_paths(mode, seed):
    pytest.importorskip("scipy.optimize")
    pytest.importorskip("networkx")
    net = sc.generate(3, 3, (10, 20, 15, 30)[seed], 7000 + seed)
    res = gp.assign_paths(net, mode, "gauss_seidel", 1.0, 800, 1e-8, adaptive=True)
    assert res.gaps[-1] < 1e-8
    assert np.abs(np.array(res.x) - _oracle_flows(net, mode)).max() / net.total_demand() < 2e-3
