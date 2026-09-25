"""Plotly-Abbildungen: Stadtplan mit Verkehr und Auslastung, Wege eines Zonenpaars, Lückenkurven der Verfahren (über Iterationen und über den Aufwand), Wegemengen, Schrittweiten-Vergleich, Lastreihe, Verteilung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Karten haben gleichen Maßstab (scaleanchor) mit automatischem Bereich; der Rand kommt über zwei unsichtbare Punkte
(ein fest vorgegebener Bereich wird beim ersten Zeichnen in schmaler Breite eingefroren). Beschriftungen von Kanten sind Annotationen mit heller Hinterlegung."""

from math import hypot

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gp_constants as C

UTIL_BINS = ((0.0, 0.5, "Auslastung unter 50 %", "#2ca02c"), (0.5, 0.8, "50 bis 80 %", "#bcbd22"), (0.8, 1.0, "80 bis 100 %", "#ff7f0e"), (1.0, 1.3, "100 bis 130 %", "#d62728"), (1.3, 1e9, "über 130 %", "#7f0000"))


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.18), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _frame(fig, points, height, pad=8):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.add_trace(go.Scatter(x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    return _base(fig, height)


def _offsets(net):
    """Seitlicher Versatz je Kante: gegenläufige Kanten liegen nebeneinander, parallele Kanten zwischen denselben Knoten fächern auf."""
    groups = {}
    for k, l in enumerate(net.links):
        groups.setdefault((l[0], l[1]), []).append(k)
    off = {}
    for (u, v), ks in groups.items():
        base = 1.4 if net.kind == "grid" else 0.0
        for i, k in enumerate(ks):
            off[k] = base + (i - (len(ks) - 1) / 2) * 6.0
    return off


def _segment(net, k, off):
    u, v = net.links[k][0], net.links[k][1]
    (x0, y0), (x1, y1) = net.pos[u], net.pos[v]
    length = hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = (y1 - y0) / length, -(x1 - x0) / length
    o = off[k]
    return x0 + nx * o, y0 + ny * o, x1 + nx * o, y1 + ny * o


def _width(flow, top, lo=1.0, hi=7.0):
    return round(lo + (hi - lo) * flow / top) if top > 0 else lo


def build_map(net, x, height=460, label_flows=False):
    """Stadtplan: Kantenbreite ~ Verkehr, Farbe = Auslastung x / Kapazität; Zonen orange umrandet. `label_flows`: Verkehr an den Kanten beschriften (kleine Lehrnetze)."""
    fig = go.Figure()
    off = _offsets(net)
    top = max(x) if x else 1.0
    used = [k for k in range(net.m) if x[k] > 1e-9]
    unused = [k for k in range(net.m) if x[k] <= 1e-9]
    if unused:
        xs, ys = [], []
        for k in unused:
            x0, y0, x1, y1 = _segment(net, k, off)
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=1), hoverinfo="skip", name="unbenutzt"))
    for lo, hi, name, color in UTIL_BINS:
        ks = [k for k in used if lo <= x[k] / net.links[k][4] < hi]
        by_width = {}
        for k in ks:
            by_width.setdefault(_width(x[k], top), []).append(k)
        first = True
        for w, group in sorted(by_width.items()):
            xs, ys = [], []
            for k in group:
                x0, y0, x1, y1 = _segment(net, k, off)
                xs += [x0, x1, None]
                ys += [y0, y1, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=w), hoverinfo="skip", name=name, showlegend=first))
            first = False
    hx, hy, ht = [], [], []
    for k in range(net.m):
        x0, y0, x1, y1 = _segment(net, k, off)
        hx.append((x0 + x1) / 2)
        hy.append((y0 + y1) / 2)
        u, v = net.links[k][0], net.links[k][1]
        ht.append(f"{net.names[u]} → {net.names[v]}: Verkehr {x[k]:.1f}, Kapazität {net.links[k][4]:.0f}, Fahrzeit {net.links[k][2] + net.links[k][3] * (x[k] / net.links[k][4]) ** net.links[k][5]:.2f}")
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=9, opacity=0), hovertext=ht, hoverinfo="text", showlegend=False))
    if label_flows:
        for k in range(net.m):
            x0, y0, x1, y1 = _segment(net, k, off)
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{x[k]:.2f}", showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.85)", borderpad=1)
    zone_set = set(net.zones)
    dots = [v for v in range(net.n) if v not in zone_set]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in dots], y=[net.pos[v][1] for v in dots], mode="markers", marker=dict(size=4, color=C.COLORS["node"]), hovertext=[net.names[v] for v in dots], hoverinfo="text", showlegend=False))
    z = list(net.zones)
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in z], y=[net.pos[v][1] for v in z], mode="markers+text", text=[f"Z{i + 1}" if net.kind == "grid" else net.names[v] for i, v in enumerate(z)], textposition="top center", textfont=dict(size=9),
                             marker=dict(size=13, symbol="square", color=C.COLORS["zone"], line=dict(width=1.5, color="#333")), hovertext=[net.names[v] for v in z], hoverinfo="text", name="Zone"))
    return _frame(fig, net.pos, height)


PATH_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2")


def build_pair_map(net, paths, height=380):
    """Die Wege eines Zonenpaars auf dem Stadtplan: jeder Weg in eigener Farbe, Breite ~ Fluss; alle anderen Straßen grau."""
    fig = go.Figure()
    off = _offsets(net)
    xs, ys = [], []
    for k in range(net.m):
        x0, y0, x1, y1 = _segment(net, k, off)
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=1), hoverinfo="skip", showlegend=False))
    ordered = sorted(paths, key=lambda p: -p[1])
    top = max((h for _l, h in ordered), default=1.0) or 1.0
    for i, (links, h) in enumerate(ordered):
        xs, ys = [], []
        shift = (i - (len(ordered) - 1) / 2) * 1.6
        for k in links:
            x0, y0, x1, y1 = _segment(net, k, {k: off[k] + shift})
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=PATH_COLORS[i % len(PATH_COLORS)], width=_width(h, top, 2.0, 8.0)), hoverinfo="skip", name=f"Weg {i + 1}: Fluss {h:.2f}".replace(".", ",")))
    ends = [net.links[ordered[0][0][0]][0], net.links[ordered[0][0][-1]][1]] if ordered else []
    others = [v for v in net.zones if v not in ends]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in others], y=[net.pos[v][1] for v in others], mode="markers", marker=dict(size=9, symbol="square", color="rgba(255,127,14,0.45)", line=dict(width=1, color="#666")),
                             hoverinfo="skip", showlegend=False))
    if ends:
        fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in ends], y=[net.pos[v][1] for v in ends], mode="markers+text", text=["von", "nach"], textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=14, symbol="square", color=C.COLORS["zone"], line=dict(width=2, color="#333")), hoverinfo="skip", showlegend=False))
    return _frame(fig, net.pos, height)


def build_pair_bars(times, flows, height=260):
    """Fluss je Weg eines Zonenpaars mit seiner Zeit (im Systemoptimum die Grenzkosten) an der Beschriftung."""
    order = sorted(range(len(flows)), key=lambda i: -flows[i])
    fig = go.Figure(go.Bar(y=[f"Weg {r + 1}: Zeit {times[i]:.2f}".replace(".", ",") for r, i in enumerate(order)], x=[flows[i] for i in order], orientation="h",
                           marker_color=[PATH_COLORS[r % len(PATH_COLORS)] for r in range(len(order))], text=[f"{flows[i]:.2f}".replace(".", ",") for i in order], textposition="auto"))
    fig.update_xaxes(title="Fluss")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height)


def build_gap(results, current=None, height=340, axis="iterations", legend=True):
    """Relative Lücke (logarithmisch) je Verfahren gegen die Iteration oder gegen den Aufwand in Kantenoperationen; gepunktet 1e-2 ... 1e-6; `current`: Iteration des Bildes (nur bei axis = iterations)."""
    fig = go.Figure()
    for m, res in results.items():
        gaps = [max(g, 1e-12) for g in res.gaps]
        xs = list(range(len(gaps))) if axis == "iterations" else list(res.ops[:len(gaps)])
        fig.add_trace(go.Scatter(x=xs, y=gaps[:len(xs)], mode="lines", name=C.LABELS[m], line=dict(color=C.COLORS[m], width=2)))
    for tol in C.TOLS:
        fig.add_hline(y=tol, line=dict(color="rgba(80,80,80,0.35)", dash="dot", width=1))
    if current is not None and axis == "iterations":
        fig.add_vline(x=current, line=dict(color="#111", dash="dash"), annotation_text="Bild", annotation_position="top")
    fig.update_xaxes(title="Iteration" if axis == "iterations" else "Aufwand (Kantenoperationen)")
    fig.update_yaxes(title="relative Lücke", type="log", exponentformat="power")
    fig = _base(fig, height)
    fig.update_layout(showlegend=legend)
    return fig


def build_paths_curve(pv, height=320):
    """Wege insgesamt: Gradient Projection (nach Iteration k) gegen die vom Frank-Wolfe-Verfahren erzeugten (alle tragen Fluss), dazu die aus den Listen gefallenen Wege."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(pv["fw_curve"]))), y=pv["fw_curve"], mode="lines", name="Frank-Wolfe: erzeugte Wege (alle mit Fluss)", line=dict(color=C.COLORS["fw"], width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(pv["gp_paths"]))), y=pv["gp_paths"], mode="lines", name="Gradient Projection: Wege in den Listen", line=dict(color=C.COLORS["gp"], width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(pv["gp_deleted"]))), y=pv["gp_deleted"], mode="lines", name="Gradient Projection: gelöschte Wege (insgesamt)", line=dict(color="#8c8c8c", width=2, dash="dot")))
    fig.update_xaxes(title="Iteration", range=[0, pv["upto"]])
    fig.update_yaxes(title="Wege")
    return _base(fig, height)


def build_alpha_sweep(sweep, height=320):
    """Iterationen bis Lücke 1e-4 je Schrittweite, Gauss-Seidel und Jacobi (Balken in Grau: nicht erreicht, Höhe = Grenze); ganz rechts der adaptive Schritt."""
    cap = sweep["cap"]
    cats = [f"α = {r['alpha']:g}".replace(".", ",") for r in sweep["rows"]] + ["adaptiv (Start 1)"]
    fig = go.Figure()
    for upd, name, color in (("gauss_seidel", "Gauss-Seidel", "#2ca02c"), ("jacobi", "Jacobi", "#ff7f0e")):
        vals = [r[upd] for r in sweep["rows"]] + [sweep["adaptive"][upd]]
        fig.add_trace(go.Bar(x=cats, y=[v if v is not None else cap for v in vals], name=name, marker_color=[color if v is not None else "#c8c8c8" for v in vals],
                             text=[str(v) if v is not None else "nie" for v in vals], textposition="outside"))
    fig.update_yaxes(title="Iterationen bis Lücke 1e-4", range=[0, cap * 1.15])
    fig.update_layout(barmode="group")
    return _base(fig, height)


def build_load_series(rows, height=340):
    """Mittlere Iterationen bis Lücke 1e-4 je Lastfaktor und Verfahren (logarithmisch; nicht erreicht zählt als Grenze + 1)."""
    fig = go.Figure()
    loads = [r["load"] / 10 for r in rows]
    for m in ("fw", "cfw", "gp", "gp_alpha1", "gp_adaptive"):
        fig.add_trace(go.Scatter(x=loads, y=[r["its"][m] for r in rows], mode="lines+markers", name=C.LABELS[m], line=dict(color=C.COLORS[m])))
    fig.update_xaxes(title="Lastfaktor")
    fig.update_yaxes(title="Iterationen bis Lücke 1e-4 (Mittel)", type="log")
    return _base(fig, height)


def build_dist(dist, height=320):
    """Median und Mittel der Iterationen bis Lücke 1e-4 je Verfahren über die festen Netze (nicht erreicht = Grenze + 1)."""
    ms = ("fw", "cfw", "gp", "gp_alpha1", "gp_adaptive")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.LABELS[m] for m in ms], y=[dist[f"{m}_median"] for m in ms], name="Median", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=[C.LABELS[m] for m in ms], y=[dist[f"{m}_mean"] for m in ms], name="Mittel", marker_color="#d62728"))
    fig.update_yaxes(title="Iterationen bis Lücke 1e-4")
    fig.update_layout(barmode="group")
    return _base(fig, height)
