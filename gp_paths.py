"""Wegebasierte Verkehrsumlegung: Gradient Projection (Jayakrishnan, Tsai, Prashker, Rajadhyaksha 1994) für das Nutzergleichgewicht und das Systemoptimum.

**Idee:** Frank-Wolfe merkt sich nur den Verkehr auf den Kanten. Jede Iteration mischt einen neuen schnellsten Weg mit dem Gewicht a in den Verkehr und lässt alle alten Wege im selben Verhältnis schrumpfen - Fluss auf einem schlechten
Weg wird geometrisch kleiner, aber nie null. Gradient Projection hält stattdessen je Zonenpaar eine **Liste von Wegen mit ihrem Fluss** und schiebt Fluss vom teureren Weg auf den billigsten der Liste (den "Basisweg"):

    Delta h_p = min( h_p , alpha (c_p - c_Basis) / Summe_{e in p Delta Basis} t'_e )

Der Nenner ist die Summe der Kostenableitungen t' auf den Kanten, die nur einer der beiden Wege benutzt (symmetrische Differenz): ein Newton-Schritt mit diagonaler Hesse-Matrix, der die Wegekosten angleichen soll.
Wird der Fluss eines Wegs dabei null, fällt er aus der Liste (die Projektion auf h >= 0). Neue Wege liefert dasselbe Kürzeste-Wege-Orakel wie im Frank-Wolfe-Verfahren (Dijkstra je Ursprungszone).

**Zwei Arten zu aktualisieren:** *Jacobi* rechnet alle Zonenpaare mit den Kosten vom Anfang der Iteration, *Gauss-Seidel* (wie im Original) rechnet Ursprung für Ursprung und aktualisiert Fluss und Kosten nach jedem Ursprung.
Die relative Lücke wird wie in `gp_frankwolfe` gemessen ((Summe x t - Summe y t) / Summe x t mit y = Alles-oder-nichts auf den Kosten am Iterationsanfang); der Kontrolllauf dafür zählt beim Gauss-Seidel-Verfahren nicht zum Aufwand.
**Aufwand** in Kantenoperationen: durchsuchte Kanten der Kürzeste-Wege-Läufe, eine Operation je Kante und Kostenauswertung (Kosten und Ableitungen), und die Kanten, die die Wegerechnung (Wegekosten, symmetrische Differenz, Flussbuchung) besucht.
Nur + - * / auf Python-Zahlen: auf allen Plattformen dieselben Iterationszahlen.
"""

import heapq
from dataclasses import dataclass

import gp_frankwolfe as fw

UPDATES = ("jacobi", "gauss_seidel")


def _adjacency(net):
    adj = [[] for _ in range(net.n)]
    for k, (u, v, *_rest) in enumerate(net.links):
        adj[u].append((v, k))
    return adj


def shortest_tree(adj, n, cost, o):
    """Dijkstra von o (Gleichstände nach der Kantenreihenfolge, wie in `fw.all_or_nothing`). Rückgabe (Abstände, Vorgängerkanten, durchsuchte Kanten)."""
    dist = [1e18] * n
    pred = [-1] * n
    dist[o] = 0.0
    pq = [(0.0, o)]
    scanned = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, k in adj[u]:
            scanned += 1
            nd = d + cost[k]
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = k
                heapq.heappush(pq, (nd, v))
    return dist, pred, scanned


def trace(net, pred, o, d):
    """Kantenfolge des Wegs von o nach d entlang der Vorgängerkanten."""
    out = []
    v = d
    while v != o:
        k = pred[v]
        out.append(k)
        v = net.links[k][0]
    out.reverse()
    return tuple(out)


def edge_flows(net, paths):
    """Kantenfluss aus den Wegeflüssen (paths[i] = {Kantenfolge: Fluss})."""
    x = [0.0] * net.m
    for pf in paths:
        for links, h in pf.items():
            for k in links:
                x[k] += h
    return x


def path_cost(cost, links):
    s = 0.0
    for k in links:
        s += cost[k]
    return s


@dataclass(frozen=True)
class PathResult:
    flows: tuple            # Kantenfluss nach jeder Iteration (flows[0] = Start)
    gaps: tuple             # relative Lücke des Verkehrs vor jeder Iteration (gaps[k] gehört zu flows[k])
    objective: tuple
    tstt: tuple
    snaps: tuple            # snaps[k][i] = ((Kantenfolge, Fluss), ...): die Wegeliste des Zonenpaars i nach Iteration k
    npaths: tuple           # Wege in allen Listen nach Iteration k
    deleted: tuple          # bis Iteration k aus den Listen gefallene Wege (Fluss auf null geschoben)
    ops: tuple              # Aufwand in Kantenoperationen bis flows[k]
    runs: int               # Kürzeste-Wege-Läufe insgesamt (ohne Kontrolllauf)
    mode: str
    update: str
    alpha: float
    alphas: tuple = ()      # der Schritt nach jeder Iteration (nur bei `adaptive` nicht konstant)

    @property
    def iterations(self):
        return len(self.flows) - 1

    @property
    def x(self):
        return self.flows[-1]

    def first_below(self, tol):
        for k, g in enumerate(self.gaps):
            if g < tol:
                return k
        return None


def lesson_start(net, index):
    """Startwege für das Lehrnetz 'Zwei Stufen': der ganze Verkehr auf den Weg Nummer `index` (0..3 = Straße 1 oder 2, dann Straße 3 oder 4) oder, für index 4, gleichmäßig auf alle vier Wege. Für andere Netze None (Standardstart)."""
    if net.kind != "zweistufen":
        return None
    q = net.demand[0][2]
    if index == 4:
        return [{(a, b): q / 4 for a in (0, 1) for b in (2, 3)}]
    return [{((0, 1)[index // 2], (2, 3)[index % 2]): q}]


def assign_paths(net, mode="ue", update="gauss_seidel", alpha=1.0, max_iter=100, tol=1e-6, init=None, grow_until=None, adaptive=False):
    """Gradient Projection. `alpha` skaliert den Newton-Schritt (1 = volle Angleichung nach der Näherung), `init` (Liste je Zonenpaar von {Kantenfolge: Fluss}) ersetzt den Alles-oder-nichts-Start,
    `grow_until` = k: nur in den ersten k Iterationen dürfen neue Wege in die Listen (Negativkontrolle: Wege fest).
    `adaptive`: `alpha` ist nur der Startwert und wird halbiert, sobald die Zielfunktion (Beckmann bzw. Gesamtfahrzeit) nach einer Iteration gestiegen ist."""
    assert update in UPDATES and mode in fw.MODES
    m, n = net.m, net.n
    co = fw.Coefs(net)
    adj = _adjacency(net)
    ods = list(net.demand)
    origins = list(net.origins().keys())
    by_origin = {}
    for i, (o, _d, _q) in enumerate(ods):
        by_origin.setdefault(o, []).append(i)
    scanned = runs = pathops = deleted = 0
    a_now = alpha

    def snapshot():
        return tuple(tuple(pf.items()) for pf in paths)

    if init is None:                                   # Alles-oder-nichts auf den Freifahrtzeiten
        paths = [dict() for _ in ods]
        cost0 = co.costs(mode, [0.0] * m)
        for o in origins:
            _dist, pred, sc = shortest_tree(adj, n, cost0, o)
            scanned += sc
            runs += 1
            for i in by_origin[o]:
                paths[i][trace(net, pred, o, ods[i][1])] = ods[i][2]
    else:
        paths = [dict(pf) for pf in init]
    x = edge_flows(net, paths)
    flows, gaps, objs, ts, snaps, npaths, dels, ops = [tuple(x)], [], [fw.objective(net, mode, x)], [fw.tstt(net, x)], [snapshot()], [sum(len(pf) for pf in paths)], [0], [scanned + m * co.calls]

    alphas = [a_now]

    def shift(i, cost, deriv, dist_pred, growing, xs):
        """Fluss im Zonenpaar i vom teureren auf den billigsten Weg schieben; aktualisiert `xs` (Kantenfluss) in place. Rückgabe Zahl der gelöschten Wege."""
        nonlocal pathops
        o, d, _q = ods[i]
        pf = paths[i]
        if growing:
            links = trace(net, dist_pred, o, d)
            pathops += len(links)
            if links not in pf:
                pf[links] = 0.0
        costs = {}
        for links in pf:
            costs[links] = path_cost(cost, links)
            pathops += len(links)
        base = min(costs, key=costs.get)               # bei Gleichstand der zuerst aufgenommene Weg
        gone = 0
        for links in list(pf):
            if links is base or links == base:
                continue
            h = pf[links]
            gap_cost = costs[links] - costs[base]
            if h > 0.0 and gap_cost > 0.0:
                diff = set(links) ^ set(base)
                pathops += len(links) + len(base)
                denom = 0.0
                for k in diff:
                    denom += deriv[k]
                delta = h if denom <= 0.0 else min(h, a_now * gap_cost / denom)
                if delta > 0.0:
                    pf[base] += delta
                    pf[links] = h - delta
                    for k in links:
                        xs[k] -= delta
                    for k in base:
                        xs[k] += delta
                    pathops += len(links) + len(base)
            if pf[links] == 0.0:
                del pf[links]
                gone += 1
        if pf[base] == 0.0 and len(pf) > 1:
            del pf[base]
        return gone

    for it in range(1, max_iter + 1):
        cost = co.costs(mode, x)
        trees = {}
        pass_scanned = 0
        for o in origins:
            dist, pred, sc = shortest_tree(adj, n, cost, o)
            trees[o] = (dist, pred)
            pass_scanned += sc
        sx = sum(x[k] * cost[k] for k in range(m))
        sy = sum(q * trees[o][0][d] for o, d, q in ods)
        gap = (sx - sy) / sx if sx > 0 else 0.0
        gaps.append(gap)
        if gap < tol:
            break
        growing = grow_until is None or it <= grow_until
        if update == "jacobi":
            scanned += pass_scanned
            runs += len(origins)
            deriv = co.derivs(mode, x)
            xs = list(x)
            for i, (o, _d, _q) in enumerate(ods):
                deleted += shift(i, cost, deriv, trees[o][1], growing, xs)
        else:
            xs = list(x)
            for o in origins:
                cost_o = co.costs(mode, xs)
                dist, pred, sc = shortest_tree(adj, n, cost_o, o)
                scanned += sc
                runs += 1
                deriv = co.derivs(mode, xs)
                for i in by_origin[o]:
                    deleted += shift(i, cost_o, deriv, pred, growing, xs)
        x = edge_flows(net, paths)                     # sauber neu aus den Wegeflüssen (keine Rundungsdrift)
        pathops += sum(len(links) for pf in paths for links in pf)
        flows.append(tuple(x))
        objs.append(fw.objective(net, mode, x))
        if adaptive and objs[-1] > objs[-2]:
            a_now /= 2
        alphas.append(a_now)
        ts.append(fw.tstt(net, x))
        snaps.append(snapshot())
        npaths.append(sum(len(pf) for pf in paths))
        dels.append(deleted)
        ops.append(scanned + m * co.calls + pathops)
    else:
        cost = co.costs(mode, x)
        y, _sc, _rn = fw.all_or_nothing(net, cost)
        gaps.append(fw.relative_gap(net, mode, x, y))
    return PathResult(tuple(flows), tuple(gaps), tuple(objs), tuple(ts), tuple(snaps), tuple(npaths), tuple(dels), tuple(ops), runs, mode, update, alpha, tuple(alphas))


def wardrop_deviation(net, mode, x, snap):
    """Größte Abweichung eines benutzten Wegs vom kürzesten Weg seines Zonenpaars, relativ zur kürzesten Zeit (0 im Gleichgewicht; im Systemoptimum bezogen auf die Grenzkosten)."""
    co = fw.Coefs(net)
    cost = co.costs(mode, x)
    adj = _adjacency(net)
    worst = 0.0
    trees = {}
    for i, (o, d, _q) in enumerate(net.demand):
        if o not in trees:
            trees[o] = shortest_tree(adj, net.n, cost, o)[0]
        best = trees[o][d]
        for links, h in snap[i]:
            if h > 0.0:
                worst = max(worst, (path_cost(cost, links) - best) / best)
    return worst
