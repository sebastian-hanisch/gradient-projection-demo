"""Verkehrsumlegung: Nutzergleichgewicht (Wardrop) und Systemoptimum mit Frank-Wolfe und seinen Varianten.

**Kopie aus frank-wolfe-demo (Stück 16), als Vergleichsbasis.** Unverändert bis auf zwei Zusätze für den Aufwandsvergleich: `Coefs` zählt Kostenauswertungen (`calls`) und liefert die Kostenableitungen (`derivs`, für die Wegeverfahren);
`Result.ops` hält den bis zur Iteration k angefallenen Aufwand in Kantenoperationen (durchsuchte Kanten im Kürzeste-Wege-Lauf plus eine Kante je Kostenauswertung).

**Nutzergleichgewicht (UE):** jeder Fahrer wählt einen schnellsten Weg, gegeben den Verkehr der anderen; im Gleichgewicht sind alle benutzten Wege je Zonenpaar gleich lang und nicht länger als jeder unbenutzte.
Das ist die Lösung des Beckmann-Programms min Summe_e Integral_0^{x_e} t_e(s) ds (konvex, weil t_e steigt); die Kantenflüsse sind eindeutig, die Wegeflüsse im Allgemeinen nicht.
**Systemoptimum (SO):** minimale Gesamtfahrzeit Summe x_e t_e(x_e); es ist dasselbe Programm mit den **Grenzkosten** t_e + x_e t_e' statt der Fahrzeit (Pigou, Beckmann).

**Frank-Wolfe:** in jedem Schritt die Kosten am aktuellen Verkehr x berechnen, jedes Zonenpaar **alles auf einen schnellsten Weg** legen (Alles-oder-nichts, ein Kürzeste-Wege-Lauf je Ursprungszone: dasselbe Orakel wie im Pricing
der Column-Generation-Demo) und den Verkehr ein Stück in diese Richtung schieben: x <- x + a (y - x). Schrittweite a: **exakte Suche** (Bisektion auf der Richtungsableitung), **MSA** (a = 1 / (k + 1)), **fest** (0,5, Negativkontrolle).
**Konjugiertes Frank-Wolfe** (Mitradjieva und Lindberg 2013) ersetzt y durch einen Punkt, der zur vorigen Richtung bezüglich der (diagonalen) Hesse-Matrix konjugiert ist, und zickzackt weniger.

**Relative Lücke:** (Summe x t - Summe y t) / Summe x t; sie ist 0 genau im Gleichgewicht. Aufwand wird in Kürzeste-Wege-Läufen und in durchsuchten Kanten gezählt, nie in Sekunden.
Die Rechnung nutzt nur + - * / auf Python-Zahlen (Potenzen als Produkte): auf allen Plattformen dieselben Iterationszahlen.
"""

import heapq
from dataclasses import dataclass

METHODS = ("fw", "msa", "cfw", "fixed")
MODES = ("ue", "so")


# --- Kostenfunktionen einer Kante (a + b (x / c)^p) ---------------------------------------------------------------------------------

def _pow(r, p):
    out = 1.0
    for _ in range(p):
        out *= r
    return out


def time(link, x):
    _, _, a, b, c, p = link
    return a + b * _pow(x / c, p)


def time_deriv(link, x):
    _, _, a, b, c, p = link
    return 0.0 if p == 0 else b * p * _pow(x / c, p - 1) / c


def time_deriv2(link, x):
    _, _, a, b, c, p = link
    return 0.0 if p < 2 else b * p * (p - 1) * _pow(x / c, p - 2) / (c * c)


def marginal(link, x):
    """Grenzkosten t + x t' (die Kosten, die ein zusätzlicher Fahrer der Allgemeinheit verursacht)."""
    return time(link, x) + x * time_deriv(link, x)


def marginal_deriv(link, x):
    return 2 * time_deriv(link, x) + x * time_deriv2(link, x)


def integral(link, x):
    _, _, a, b, c, p = link
    return a * x + b * c * _pow(x / c, p + 1) / (p + 1)


def cost_fn(mode):
    """(Kostenfunktion, ihre Ableitung, Zielfunktion) je Modus: UE nutzt die Fahrzeit und das Beckmann-Integral, SO die Grenzkosten und die Gesamtfahrzeit."""
    if mode == "ue":
        return time, time_deriv, integral
    return marginal, marginal_deriv, lambda link, x: x * time(link, x)


class Coefs:
    """Die Kostenkoeffizienten aller Kanten als Listen (schnellere Rechnung in den Schleifen; dieselben Werte wie `time`, `marginal` usw.)."""

    def __init__(self, net):
        self.m = net.m
        self.a = [l[2] for l in net.links]
        self.b = [l[3] for l in net.links]
        self.c = [l[4] for l in net.links]
        self.p = [l[5] for l in net.links]
        self.calls = 0                    # Zahl der Kostenauswertungen über alle Kanten (Aufwandszähler)

    def costs(self, mode, x, d=None, tau=0.0):
        """Kosten aller Kanten am Verkehr x + tau d: UE Fahrzeit a + b r^p, SO Grenzkosten a + b (p + 1) r^p mit r = x / c."""
        self.calls += 1
        out = []
        a, b, c, p = self.a, self.b, self.c, self.p
        for k in range(self.m):
            xk = x[k] if d is None else x[k] + tau * d[k]
            r = xk / c[k]
            pk = p[k]
            rp = r * r * r * r if pk == 4 else (r if pk == 1 else _pow(r, pk))
            out.append(a[k] + b[k] * rp if mode == "ue" else a[k] + b[k] * (pk + 1) * rp)
        return out


    def derivs(self, mode, x):
        """Ableitung der Kosten nach dem Fluss je Kante (UE: t', SO: 2 t' + x t''), dieselben Werte wie `time_deriv` und `marginal_deriv`."""
        self.calls += 1
        out = []
        for k in range(self.m):
            b, c, p = self.b[k], self.c[k], self.p[k]
            r = x[k] / c
            t1 = 0.0 if p == 0 else b * p * _pow(r, p - 1) / c
            if mode == "ue":
                out.append(t1)
            else:
                t2 = 0.0 if p < 2 else b * p * (p - 1) * _pow(r, p - 2) / (c * c)
                out.append(2 * t1 + x[k] * t2)
        return out


# --- Alles-oder-nichts -----------------------------------------------------------------------------------------------------------

def all_or_nothing(net, cost):
    """Jedes Zonenpaar komplett auf einen schnellsten Weg (Dijkstra je Ursprung, Gleichstände nach der Kantenreihenfolge). Rückgabe (Kantenfluss, durchsuchte Kanten, Zahl der Läufe)."""
    adj = [[] for _ in range(net.n)]
    for k, (u, v, *_rest) in enumerate(net.links):
        adj[u].append((v, k))
    y = [0.0] * net.m
    scanned = runs = 0
    for o, dests in net.origins().items():
        runs += 1
        dist = [1e18] * net.n
        pred = [-1] * net.n
        dist[o] = 0.0
        pq = [(0.0, o)]
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
        for d, q in dests:
            v = d
            while v != o:
                k = pred[v]
                y[k] += q
                v = net.links[k][0]
    return y, scanned, runs


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------

def link_costs(net, mode, x):
    f = cost_fn(mode)[0]
    return [f(net.links[k], x[k]) for k in range(net.m)]


def tstt(net, x):
    """Gesamtfahrzeit Summe x_e t_e(x_e)."""
    return sum(x[k] * time(net.links[k], x[k]) for k in range(net.m))


def objective(net, mode, x):
    g = cost_fn(mode)[2]
    return sum(g(net.links[k], x[k]) for k in range(net.m))


def relative_gap(net, mode, x, y):
    cost = link_costs(net, mode, x)
    sx = sum(x[k] * cost[k] for k in range(net.m))
    sy = sum(y[k] * cost[k] for k in range(net.m))
    return (sx - sy) / sx if sx > 0 else 0.0


def utilization(net, x):
    return [x[k] / net.links[k][4] for k in range(net.m)]


@dataclass(frozen=True)
class Result:
    flows: tuple            # Kantenfluss nach jeder Iteration (flows[0] = Start: Alles-oder-nichts auf den Freifahrtzeiten)
    gaps: tuple             # relative Lücke des Verkehrs vor jeder Iteration (gaps[k] gehört zu flows[k])
    objective: tuple        # Zielfunktion zu flows[k]
    tstt: tuple             # Gesamtfahrzeit zu flows[k]
    steps: tuple            # Schrittweite der Iteration k -> k + 1
    scanned: int
    runs: int               # Kürzeste-Wege-Läufe insgesamt
    mode: str
    method: str
    ops: tuple = ()         # Aufwand in Kantenoperationen bis flows[k]

    @property
    def iterations(self):
        return len(self.flows) - 1

    @property
    def x(self):
        return self.flows[-1]

    def first_below(self, tol):
        """Kleinste Iteration, in der die relative Lücke unter `tol` liegt (None, wenn nie)."""
        for k, g in enumerate(self.gaps):
            if g < tol:
                return k
        return None


def line_search(net, mode, x, d, lo=0.0, hi=1.0, rounds=20, coefs=None):
    """Schrittweite in [lo, hi], die die Zielfunktion längs x + a d minimiert (Bisektion auf der Ableitung Summe d_e cost_e(x_e + a d_e))."""
    co = coefs or Coefs(net)
    m = net.m
    for _ in range(rounds):
        mid = (lo + hi) / 2
        cost = co.costs(mode, x, d, mid)
        der = sum(d[k] * cost[k] for k in range(m))
        if der > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def assign(net, mode="ue", method="fw", max_iter=300, tol=1e-6, delta=1e-6):
    """Verkehrsumlegung. `method`: 'fw' (exakte Schrittweite), 'msa', 'cfw' (konjugiert), 'fixed' (Schritt 0,5). Stoppt bei relativer Lücke unter `tol` oder nach `max_iter` Iterationen."""
    assert method in METHODS and mode in MODES
    f, fd, _ = cost_fn(mode)
    m = net.m
    co = Coefs(net)
    x, scanned, runs = all_or_nothing(net, co.costs(mode, [0.0] * m))
    flows, gaps, objs, ts, steps = [tuple(x)], [], [objective(net, mode, x)], [tstt(net, x)], []
    gap_evals = 0
    ops = [scanned + m * co.calls]
    s_prev = list(x)
    for it in range(1, max_iter + 1):
        cost = co.costs(mode, x)
        y, sc, rn = all_or_nothing(net, cost)
        scanned += sc
        runs += rn
        gap = relative_gap(net, mode, x, y)
        gap_evals += 1
        gaps.append(gap)
        if gap < tol:
            break
        if method == "cfw" and it > 1:
            h = [fd(net.links[k], x[k]) for k in range(m)]
            num = sum((s_prev[k] - x[k]) * h[k] * (y[k] - x[k]) for k in range(m))
            den = sum((s_prev[k] - x[k]) * h[k] * (y[k] - s_prev[k]) for k in range(m))
            back = sum((s_prev[k] - x[k]) ** 2 * h[k] for k in range(m))
            fwd = sum((y[k] - x[k]) ** 2 * h[k] for k in range(m))
            alpha = 0.0 if (den == 0 or back <= 1e-12 * fwd) else max(0.0, min(num / den, 1 - delta))          # x liegt schon auf dem alten Zielpunkt: neu beginnen
            target = [alpha * s_prev[k] + (1 - alpha) * y[k] for k in range(m)]
            if sum((target[k] - x[k]) * cost[k] for k in range(m)) >= 0:       # keine Abstiegsrichtung mehr: zurück zur Frank-Wolfe-Richtung
                target = y
        else:
            target = y
        d = [target[k] - x[k] for k in range(m)]
        if method == "msa":
            a = 1.0 / (it + 1)
        elif method == "fixed":
            a = 0.5
        else:
            a = line_search(net, mode, x, d, coefs=co)
        x = [x[k] + a * d[k] for k in range(m)]
        if method == "cfw":
            s_prev = target
        flows.append(tuple(x))
        objs.append(objective(net, mode, x))
        ts.append(tstt(net, x))
        steps.append(a)
        ops.append(scanned + m * (co.calls + gap_evals))
    else:
        cost = co.costs(mode, x)
        y, sc, rn = all_or_nothing(net, cost)
        scanned += sc
        runs += rn
        gaps.append(relative_gap(net, mode, x, y))
    return Result(tuple(flows), tuple(gaps), tuple(objs), tuple(ts), tuple(steps), scanned, runs, mode, method, tuple(ops))


def price_of_anarchy(net, method="fw", max_iter=600, tol=1e-6):
    """Gesamtfahrzeit im Nutzergleichgewicht geteilt durch die im Systemoptimum. Rückgabe (PoA, UE-Ergebnis, SO-Ergebnis)."""
    ue = assign(net, "ue", method, max_iter, tol)
    so = assign(net, "so", method, max_iter, tol)
    return ue.tstt[-1] / so.tstt[-1], ue, so
