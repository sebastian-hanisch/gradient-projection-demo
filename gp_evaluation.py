"""Auswertung: Gradient Projection gegen Frank-Wolfe, Wegemengen, Schrittweite, Lastreihe, Verteilung über feste Netze, Eindeutigkeit der Kantenflüsse, Lehrnetz mit verschiedenen Startwegen.
Alle Zufallsnetze kommen aus festen Seeds (gp_constants), unabhängig vom Nutzer-Seed."""

import statistics
from collections import namedtuple

import gp_constants as C
import gp_frankwolfe as fw
import gp_paths as gp
import gp_scenario as sc

Params = namedtuple("Params", "net side zones load mode update alpha adaptive frozen iterations seed")
DEFAULT_PARAMS = Params(C.DEFAULT_NET, C.DEFAULT_SIDE, C.DEFAULT_ZONES, C.DEFAULT_LOAD, C.DEFAULT_MODE, C.DEFAULT_UPDATE, C.DEFAULT_ALPHA, False, False, C.DEFAULT_ITERATIONS, C.DEFAULT_SEED)


def network(params):
    return sc.build(params.net, params.side, params.zones, params.load, params.seed)


def run_gp(net, params, max_iter=None, tol=C.GAP_STOP, **over):
    """Gradient Projection mit den Einstellungen; `over` überschreibt einzelne (update, alpha, adaptive, mode)."""
    cfg = dict(mode=params.mode, update=params.update, alpha=params.alpha, adaptive=params.adaptive)
    cfg.update(over)
    return gp.assign_paths(net, cfg["mode"], cfg["update"], cfg["alpha"], params.iterations if max_iter is None else max_iter, tol,
                           grow_until=C.FROZEN_UNTIL if params.frozen else None, adaptive=cfg["adaptive"])


def analyse(params):
    net = network(params)
    return dict(net=net, result=run_gp(net, params))


def compare(params, cap=C.CAP):
    """Frank-Wolfe, konjugiertes Frank-Wolfe und Gradient Projection auf demselben Netz und im selben Modus: Iterationen und Aufwand (Kantenoperationen) bis zur Lücke 1e-2 ... 1e-6."""
    net = network(params)
    res = {"fw": fw.assign(net, params.mode, "fw", cap, C.GAP_STOP), "cfw": fw.assign(net, params.mode, "cfw", cap, C.GAP_STOP), "gp": run_gp(net, params, cap)}
    table = {m: [r.first_below(t) for t in C.TOLS] for m, r in res.items()}
    ops = {m: [(r.ops[k] if k is not None else None) for k in table[m]] for m, r in res.items()}
    return dict(net=net, results=res, table=table, ops=ops)


def fw_generated_paths(net, mode, res, upto):
    """Zahl der verschiedenen Wege, die das Alles-oder-nichts des Frank-Wolfe-Verfahrens bis Iteration j erzeugt hat (j = 0 .. upto). Jeder solche Weg trägt Fluss: der Verkehr ist eine Mischung aller bisherigen
    Alles-oder-nichts-Lösungen, alte Wege schrumpfen nur, ohne null zu werden. Für das konjugierte Verfahren ist es eine Obergrenze."""
    co = fw.Coefs(net)
    adj = gp._adjacency(net)
    by_origin = {}
    for i, (o, _d, _q) in enumerate(net.demand):
        by_origin.setdefault(o, []).append(i)
    seen = [set() for _ in net.demand]
    out = []
    for j in range(upto + 1):
        cost = co.costs(net_mode(res), res.flows[j - 1] if j > 0 else [0.0] * net.m)
        for o, idx in by_origin.items():
            _dist, pred, _sc = gp.shortest_tree(adj, net.n, cost, o)
            for i in idx:
                seen[i].add(gp.trace(net, pred, o, net.demand[i][1]))
        out.append(sum(len(s) for s in seen))
    return out


def net_mode(res):
    return res.mode


def paths_view(params, cmp):
    """Wegemengen: Gradient Projection (nach Iteration k) gegen die vom Frank-Wolfe-Verfahren erzeugten Wege, bis zur Lücke 1e-4 (oder bis zum Ende)."""
    net = cmp["net"]
    f, g = cmp["results"]["fw"], cmp["results"]["gp"]
    upto = f.first_below(1e-4)
    upto = f.iterations if upto is None else upto
    return dict(fw_curve=fw_generated_paths(net, params.mode, f, upto), gp_paths=list(g.npaths), gp_deleted=list(g.deleted), upto=upto)


def alpha_sweep(params, alphas=C.ALPHAS, cap=C.CAP):
    """Iterationen bis Lücke 1e-4 je Schrittweite (Gauss-Seidel und Jacobi, fest) und mit halbierendem adaptivem Schritt (Start 1,0); nicht erreicht = None."""
    net = network(params)
    rows = []
    for a in alphas:
        row = dict(alpha=a)
        for upd in gp.UPDATES:
            r = run_gp(net, params, cap, update=upd, alpha=a, adaptive=False, tol=1e-5)
            row[upd] = r.first_below(1e-4)
        rows.append(row)
    adaptive = {upd: run_gp(net, params, cap, update=upd, alpha=1.0, adaptive=True, tol=1e-5).first_below(1e-4) for upd in gp.UPDATES}
    return dict(rows=rows, adaptive=adaptive, cap=cap)


def _methods(net, params, cap):
    """Iterationen bis 1e-4 und Aufwand je Verfahren auf einem Netz."""
    tol = 1e-5
    runs = {"fw": fw.assign(net, params.mode, "fw", cap, tol), "cfw": fw.assign(net, params.mode, "cfw", cap, tol), "gp": run_gp(net, params, cap, tol=tol),
            "gp_adaptive": run_gp(net, params, cap, update="gauss_seidel", alpha=1.0, adaptive=True, tol=tol), "gp_alpha1": run_gp(net, params, cap, update="gauss_seidel", alpha=1.0, adaptive=False, tol=tol)}
    out = {}
    for m, r in runs.items():
        k = r.first_below(1e-4)
        out[m] = dict(its=k, ops=None if k is None else r.ops[k])
    return out


METHOD_KEYS = ("fw", "cfw", "gp", "gp_adaptive", "gp_alpha1")


def load_series(params, loads=C.LOADS, seeds=C.LOAD_SEEDS, cap=C.CAP):
    """Mittlere Iterationen bis Lücke 1e-4 je Lastfaktor und Verfahren (nicht erreicht = cap + 1) und die Zahl der Netze, in denen sie nicht erreicht wird."""
    rows = []
    for load in loads:
        per = [_methods(network(params._replace(load=load, seed=s)), params, cap) for s in seeds]
        rows.append(dict(load=load, its={m: statistics.fmean(cap + 1 if r[m]["its"] is None else r[m]["its"] for r in per) for m in METHOD_KEYS},
                         missed={m: sum(1 for r in per if r[m]["its"] is None) for m in METHOD_KEYS}, n=len(per)))
    return rows


def distribution(params, seeds=C.SWEEP_SEEDS, cap=C.CAP):
    """Über feste Netze mit den Einstellungen: Iterationen bis 1e-4 je Verfahren (nicht erreicht = cap + 1), Zahl der nicht erreichten, Vergleich mit dem konjugierten Verfahren."""
    rows = [_methods(network(params._replace(seed=s)), params, cap) for s in seeds]

    def its(r, m):
        return cap + 1 if r[m]["its"] is None else r[m]["its"]

    out = dict(n=len(rows), cap=cap, rows=rows)
    for m in METHOD_KEYS:
        vals = [its(r, m) for r in rows]
        out[m + "_mean"] = statistics.fmean(vals)
        out[m + "_median"] = statistics.median(vals)
        out[m + "_missed"] = sum(1 for r in rows if r[m]["its"] is None)
        out[m + "_beats_cfw"] = sum(1 for r in rows if its(r, m) < its(r, "cfw"))
        out[m + "_loses_to_cfw"] = sum(1 for r in rows if its(r, m) > its(r, "cfw"))
    return out


def uniqueness(params):
    """Kantenflüsse sind eindeutig: Gradient Projection (adaptiv, damit es sicher ankommt) und konjugiertes Frank-Wolfe enden bei denselben Flüssen (größte Abweichung in Prozent der gesamten Nachfrage)."""
    net = network(params)
    a = fw.assign(net, params.mode, "cfw", 800, 1e-9)
    b = run_gp(net, params, 600, update="gauss_seidel", alpha=1.0, adaptive=True, tol=1e-8)
    total = net.total_demand()
    return dict(max_diff=max(abs(a.x[k] - b.x[k]) for k in range(net.m)) / total, gap_a=a.gaps[-1], gap_b=b.gaps[-1],
                deviation=gp.wardrop_deviation(net, params.mode, b.x, b.snaps[-1]), paths=b.npaths[-1], pairs=len(net.demand))


def lesson_starts(mode="ue"):
    """Lehrnetz 'Zwei Stufen': Gradient Projection von jedem der vier möglichen Startwege aus (der ganze Verkehr auf einem Weg) und von der Gleichverteilung auf alle vier. Kantenflüsse am Ende gleich (je Straße 5), Wegeflüsse verschieden."""
    net = sc.two_stages()
    rows = []
    for index in range(5):
        r = gp.assign_paths(net, mode, "gauss_seidel", 1.0, 50, 1e-9, init=gp.lesson_start(net, index))
        rows.append(dict(start=index, paths=[(links, h) for links, h in r.snaps[-1][0]], edges=list(r.x), iterations=r.iterations))
    return rows


def pair_label(net, i):
    o, d, q = net.demand[i]
    zone = {v: j + 1 for j, v in enumerate(net.zones)}
    if net.kind == "grid":
        return f"Z{zone[o]} → Z{zone[d]} (Nachfrage {q:.1f})".replace(".", ",")
    return f"{net.names[o]} → {net.names[d]} (Nachfrage {q:.1f})".replace(".", ",")


def busiest_pair(result):
    """Zonenpaar mit den meisten Wegen am Ende der Rechnung (das lehrreichste zum Ansehen)."""
    snap = result.snaps[-1]
    return max(range(len(snap)), key=lambda i: (len(snap[i]), -i))


def wardrop_check(net, x, mode="ue"):
    """Relative Lücke einer frischen Alles-oder-nichts-Zuordnung an x (0 im Gleichgewicht)."""
    cost = fw.link_costs(net, mode, x)
    y, _, _ = fw.all_or_nothing(net, cost)
    return fw.relative_gap(net, mode, x, y)
