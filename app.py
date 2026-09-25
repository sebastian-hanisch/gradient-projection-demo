"""Gradient Projection - Wege statt Kantenflüsse: verschwindet der Endspurt von Frank-Wolfe? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - wegebasierte Verkehrsumlegung mit Gradient Projection - und lässt stattdessen das Beispiel wachsen.
Fünfte Erweiterung (Stück 17, E4) der Netzwerkfluss-Linie der "Konzepte"-Reihe: Nachfolger der Frank-Wolfe-Demo (Stück 16) mit demselben Netz und demselben Modell, aber einem anderen Löser. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import gp_constants as C
import gp_evaluation as ev
import gp_frankwolfe as fw
import gp_paths as gp
from gp_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from gp_visualization import build_alpha_sweep, build_dist, build_gap, build_load_series, build_map, build_pair_bars, build_pair_map, build_paths_curve

st.set_page_config(page_title="Gradient Projection – Sebastian Hanisch", layout="wide")


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return "–" if x is None else f"{int(round(x)):,}".replace(",", " ")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def _sci(x):
    return f"{x:.1e}".replace(".", ",")


def _iters(n):
    return "1 Iteration" if n == 1 else f"{n} Iterationen"


def _alpha(a):
    return f"{a:g}".replace(".", ",")


@st.cache_resource(show_spinner=False, max_entries=32)
def _analysis(params):
    return ev.analyse(ev.Params(*params))


@st.cache_resource(show_spinner=False, max_entries=32)
def _compare(params):
    cmp = ev.compare(ev.Params(*params))
    cmp["paths"] = ev.paths_view(ev.Params(*params), cmp)
    return cmp


st.title("🛤️ Gradient Projection – Wege statt Kantenflüsse")
st.markdown(
    """
Die Frank-Wolfe-Demo (Stück 16) hat gezeigt: das Nutzergleichgewicht im Stadtnetz lässt sich mit dem Kürzeste-Wege-Orakel lösen, aber der **Endspurt ist lang** - 21 Iterationen bis zur Lücke 1e-3, 102 bis 1e-4, und 1e-5 in 200 nicht.
Der Grund: Frank-Wolfe kennt nur **Kantenflüsse**. Jede Iteration mischt einen neuen schnellsten Weg in den Verkehr und lässt alle alten Wege im gleichen Verhältnis schrumpfen - ein schlechter Weg wird geometrisch kleiner, aber nie null.
**Gradient Projection** führt stattdessen je Zonenpaar eine **Liste von Wegen mit ihrem Fluss** und **schiebt Fluss vom teureren Weg auf den billigsten**, bis ein Weg leer ist und aus der Liste fällt. Die Wegekosten gleichen sich so schnell an; was der Schritt kostet, steht in der Ableitung der Fahrzeit.
Diese Demo rechnet dasselbe Netz und dasselbe Modell wie Stück 16 - und zeigt, was der Löserwechsel bringt und wo er **heikel** ist: die Schrittweite.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - fünfte Erweiterung (E4) der Netzwerkfluss-Linie der \"Konzepte\"-Reihe, Nachfolger von Frank-Wolfe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Modell (Wardrop-Gleichgewicht mit BPR-Fahrzeiten) ist das der Frank-Wolfe-Demo; neu ist der **Löser** und die Frage, was wegebasiertes Rechnen kostet und bringt. Bush-basierte Verfahren (Algorithm B, TAPAS) sind nur erwähnt."
)

with st.expander("So funktioniert Gradient Projection", expanded=True):
    st.markdown(
        r"""
1. **Wege je Zonenpaar:** jedes Zonenpaar hat eine Liste von Wegen mit Fluss $h_p\ge 0$, die zusammen die Nachfrage tragen. Start: alles auf den schnellsten Weg bei leerem Netz.
2. **Neuer Weg:** mit den aktuellen Fahrzeiten läuft ein Dijkstra je Ursprung (dasselbe Kürzeste-Wege-Orakel wie bei Frank-Wolfe); ist der schnellste Weg noch nicht in der Liste, kommt er hinein. Der billigste Weg der Liste heißt **Basisweg**.
3. **Schieben:** vom Weg $p$ geht der Fluss $\Delta h_p=\min\!\big(h_p,\ \alpha\,\dfrac{c_p-c_{\text{Basis}}}{\sum_{e\in p\,\triangle\,\text{Basis}} t_e'}\big)$ auf den Basisweg. Im Nenner stehen die Ableitungen der Fahrzeit auf den Kanten, die nur einer der beiden Wege benutzt: ein Newton-Schritt, der die beiden Wegezeiten angleichen soll.
4. **Projektion:** wird $h_p$ null, fällt der Weg aus der Liste - Fluss kann nicht negativ werden. Das ist der Unterschied zu Frank-Wolfe, dessen alte Wege nie ganz leer werden.
5. **Aktualisieren:** *Gauss-Seidel* rechnet Ursprung für Ursprung und aktualisiert die Fahrzeiten nach jedem; *Jacobi* rechnet alle Zonenpaare mit den Fahrzeiten vom Iterationsanfang. Die Schrittweite $\alpha$ skaliert den Newton-Schritt; **adaptiv** halbiert sie, sobald die Zielfunktion steigt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox("Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
                           help="Ein Stadtgitter mit Zonen (dasselbe wie in der Frank-Wolfe-Demo) oder ein festes Lehrnetz: das Pigou-Netz (zwei parallele Straßen) und zwei Stufen paralleler Straßen (Kantenflüsse eindeutig, Wegeflüsse nicht).")
    mode = st.radio("Ziel", list(C.MODES), key="mode_radio", format_func=lambda k: C.MODES[k],
                    help="Nutzergleichgewicht: jeder wählt für sich den schnellsten Weg (Fahrzeit als Kosten). Systemoptimum: kleinste Gesamtfahrzeit (Grenzkosten als Kosten); derselbe Löser, andere Kosten.")
    update = st.radio("Aktualisierung", list(C.UPDATES), key="update_radio", format_func=lambda k: C.UPDATES[k],
                      help="Gauss-Seidel: nach jedem Ursprung werden Verkehr und Fahrzeiten neu gerechnet (wie im Original). Jacobi: alle Zonenpaare rechnen mit den Fahrzeiten vom Anfang der Iteration; das ist einfacher zu parallelisieren, aber empfindlicher.")
    alpha = st.radio("Schritt α", list(C.ALPHAS), key="alpha_radio", horizontal=True, format_func=_alpha,
                     help="Skalierung des Newton-Schritts: 1 gleicht die beiden Wegezeiten nach der Näherung genau an, kleinere Werte sind vorsichtiger. Im Standardnetz ist 0,5 am schnellsten; bei 1 zickzackt das Verfahren. Bei hoher Last müssen die brauchbaren Werte noch kleiner sein.")
    adaptive = st.checkbox("α adaptiv (halbieren, wenn die Zielfunktion steigt)", key="adaptive_check",
                           help="α ist dann nur der Startwert und wird halbiert, sobald die Zielfunktion nach einer Iteration gestiegen ist. Eine einfache Absicherung, kein Verfahren aus der Literatur.")
    frozen = st.checkbox(f"Neue Wege nur in den ersten {C.FROZEN_UNTIL} Iterationen (Negativkontrolle)", key="frozen_check",
                         help="Ohne das Kürzeste-Wege-Orakel in späteren Iterationen können die Listen nicht mehr wachsen: der Verkehr wird nur noch zwischen den bekannten Wegen verschoben und erreicht das Gleichgewicht nicht.")
    iterations = st.radio("Iterationsgrenze", list(C.ITERATIONS), key="iterations_radio", horizontal=True, help="Höchstzahl der Iterationen; die Rechnung stoppt früher, wenn die relative Lücke unter 1e-6 fällt.")
    if net_key == "grid":
        seed_widget("side_slider")
        side = st.slider("Kreuzungen je Kante", *bounds("side_slider"), key="side_slider", help="Das Stadtgitter hat side × side Kreuzungen und Straßen in beide Richtungen (bei 8 × 8: 224 Kanten).")
        st.session_state[KEPT["side_slider"]] = side
        seed_widget("zones_slider")
        zones = st.slider("Zonen", *bounds("zones_slider"), key="zones_slider", help="Zonen mit Nachfrage; jede Zone schickt Verkehr zu jeder anderen (Zonen × (Zonen − 1) Zonenpaare, ein Kürzeste-Wege-Lauf je Zone und Iteration).")
        st.session_state[KEPT["zones_slider"]] = zones
        seed_widget("load_slider")
        load = st.slider("Last [Zehntel des Standards]", *bounds("load_slider"), key="load_slider", step=5, help="Nachfrage in Zehnteln des Standards (10 = 1,0). Je höher die Last, desto stärker die Staus - und desto kleiner muss der Schritt α sein, damit Gradient Projection nicht zickzackt.")
        st.session_state[KEPT["load_slider"]] = load
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed (neue Straßen, Zonen und Nachfrage). Die Verteilungen über 40 feste Netze weiter unten ändern sich dabei nicht.")
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
        zones = int(st.session_state.get(KEPT["zones_slider"], C.DEFAULT_ZONES))
        load = int(st.session_state.get(KEPT["load_slider"], C.DEFAULT_LOAD))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Kreuzungen, Zonen, Last und Seed gehören zum Stadtgitter.")

sync_query_params({"net_select": net_key, "mode_radio": mode, "update_radio": update, "alpha_radio": alpha, "adaptive_check": int(bool(adaptive)), "frozen_check": int(bool(frozen)), "iterations_radio": int(iterations),
                   "side_slider": int(side), "zones_slider": int(zones), "load_slider": int(load), "seed_input": int(seed)})

# feste Lehrnetze ignorieren die Zufallsregler: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
params = (net_key, int(side), int(zones), int(load), mode, update, float(alpha), bool(adaptive), bool(frozen), int(iterations), int(seed))
if net_key != "grid":
    params = (net_key, C.DEFAULT_SIDE, C.DEFAULT_ZONES, C.DEFAULT_LOAD, mode, update, float(alpha), bool(adaptive), bool(frozen), int(iterations), C.DEFAULT_SEED)
with st.spinner("Rechne..."):
    a = _analysis(params)
net, res = a["net"], a["result"]
N_IT = res.iterations
label = net.kind != "grid"
co = fw.Coefs(net)

# --- Fluss auf Wegen verschieben --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Fluss auf Wegen verschieben, Iteration für Iteration")
if st.session_state.get("gp_step_owner") != params:
    st.session_state["gp_step"] = N_IT
    st.session_state["gp_pair"] = ev.busiest_pair(res)
    st.session_state["gp_step_owner"] = params
step_col, play_col = st.columns([5, 2])
with step_col:
    if N_IT > 0:
        step = st.slider("Iteration", 0, N_IT, key="gp_step", help="Iteration 0: alle fahren die schnellsten Wege bei leerem Netz (ein Weg je Zonenpaar); danach schiebt jede Iteration Fluss von teureren Wegen auf den billigsten.")
    else:
        step = 0
        st.caption("Das Netz ist schon nach dem ersten Alles-oder-nichts im Gleichgewicht - es gibt nichts zu iterieren.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch", disabled=N_IT == 0)
pair = st.selectbox("Zonenpaar", list(range(len(net.demand))), key="gp_pair", format_func=lambda i: ev.pair_label(net, i),
                    help="Die Wege dieses Zonenpaars werden unten gezeigt. Vorgewählt ist das Zonenpaar mit den meisten Wegen am Ende der Rechnung.")
view_slot = st.empty()


def _render(k):
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_map(net, res.flows[k], label_flows=label), width="stretch", key=f"map_{k}")
        c2.plotly_chart(build_gap({"gp": res}, current=k, height=460), width="stretch", key=f"gap_{k}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Relative Lücke", _sci(res.gaps[k]), help="Anteil der Fahrzeit, den ein Wechsel aller auf die kürzesten Wege noch sparen würde; 0 im Gleichgewicht.")
        m2.metric("Gesamtfahrzeit", _f(res.tstt[k], 1), delta=None if k == 0 else _f(res.tstt[k] - res.tstt[k - 1], 1) + " gegen vorher", delta_color="off", help="Summe der Verkehrsmenge mal Fahrzeit über alle Kanten.")
        m3.metric("Wege in den Listen", _int(res.npaths[k]), delta=None if k == 0 else f"{res.deleted[k] - res.deleted[k - 1]} gelöscht", delta_color="off",
                  help="Alle Wege aller Zonenpaare, die nach dieser Iteration Fluss tragen; darunter die in dieser Iteration gelöschten (Fluss auf null geschoben).")
        m4.metric("Schritt α", "–" if k == 0 else _alpha(res.alphas[k - 1]), help="Der Schritt, mit dem Iteration k − 1 → k gerechnet wurde (bei adaptivem Schritt kann er sinken).")
        snap = res.snaps[k][pair]
        cost = co.costs(mode, res.flows[k])
        flows = [h for _l, h in snap]
        times = [gp.path_cost(cost, l) for l, _h in snap]
        p1, p2 = st.columns([3, 2])
        p1.plotly_chart(build_pair_map(net, list(snap), height=340), width="stretch", key=f"pairmap_{k}")
        p2.plotly_chart(build_pair_bars(times, flows, height=340), width="stretch", key=f"pairbars_{k}")


if auto_play:
    for k in range(N_IT + 1):
        _render(k)
        time.sleep(min(0.4, 6.0 / max(N_IT, 1)))
    step = N_IT
else:
    _render(step)
st.caption("Oben: Kantenbreite = Verkehr, Farbe = Auslastung (Verkehr / Kapazität); orange Quadrate sind Zonen. Rechts die relative Lücke gegen die Iteration (logarithmisch). "
           "Unten die Wege des gewählten Zonenpaars: jeder Weg in eigener Farbe, Breite = Fluss; die Balken zeigen Fluss und Zeit je Weg. Im Gleichgewicht haben alle Wege mit Fluss dieselbe Zeit (im Systemoptimum: dieselben Grenzkosten).")

if res.gaps[-1] < C.GAP_STOP:
    st.success(f"✅ Nach {_iters(res.iterations)} ist die relative Lücke unter 1e-6 ({_sci(res.gaps[-1])}); {_int(res.npaths[-1])} Wege tragen Fluss, {res.deleted[-1]} Wege sind unterwegs gelöscht worden, {_int(res.runs)} Kürzeste-Wege-Läufe insgesamt.")
elif frozen:
    st.warning(f"⚠️ Die Wegelisten wachsen nur in den ersten {C.FROZEN_UNTIL} Iterationen: die Lücke bleibt bei {_sci(res.gaps[-1])} stehen, die Gesamtfahrzeit bei {_f(res.tstt[-1], 1)}. Ohne das Kürzeste-Wege-Orakel fehlen dem Verfahren die guten Wege.")
elif res.gaps[-1] > 1e-3 and res.iterations >= 20:
    st.warning(f"⚠️ Nach {_iters(res.iterations)} liegt die Lücke bei {_sci(res.gaps[-1])} und sinkt nicht mehr: der Schritt α = {_alpha(alpha)} ist für dieses Netz zu groß, der Verkehr wird zwischen den Wegen hin und her geschoben ({res.deleted[-1]} Wege sind aus den Listen gelöscht worden). Kleineres α oder der adaptive Schritt helfen.")
else:
    st.info(f"ℹ️ Nach {_iters(res.iterations)} (Grenze) liegt die relative Lücke bei {_sci(res.gaps[-1])}; die Gesamtfahrzeit ist auf {_f(res.tstt[-1], 1)} gefallen.")
if net_key == "zweistufen":
    st.markdown("**Kantenflüsse eindeutig, Wegeflüsse nicht:** jede der vier Straßen trägt genau 5. Gradient Projection legt die Wege dabei nur so um, wie es muss - welche Aufteilung am Ende steht, hängt vom Start ab (Tabelle unten).")

st.markdown("---")

# --- Vergleich mit Frank-Wolfe ----------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Gradient Projection gegen Frank-Wolfe")
with st.spinner("Rechne die Verfahren..."):
    cmp = _compare(params)
methods = ("fw", "cfw", "gp")
g1, g2 = st.columns(2)
g1.plotly_chart(build_gap(cmp["results"], height=340), width="stretch", key="compare_iterations")
g2.plotly_chart(build_gap(cmp["results"], height=340, axis="ops", legend=False), width="stretch", key="compare_ops")
st.table({"Verfahren": [C.LABELS[m] for m in methods],
          **{f"Iterationen bis Lücke {t:.0e}": [("–" if cmp["table"][m][i] is None else str(cmp["table"][m][i])) for m in methods] for i, t in enumerate(C.SHOWN_TOLS)}})
st.table({"Verfahren": [C.LABELS[m] for m in methods],
          **{f"Aufwand bis Lücke {t:.0e}": [_int(cmp["ops"][m][i]) for m in methods] for i, t in enumerate(C.TOLS[1:4], start=1)}})
st.caption(f"Links über die Iterationen, rechts über den **Aufwand in Kantenoperationen**: durchsuchte Kanten der Kürzeste-Wege-Läufe, eine Operation je Kante und Kostenauswertung (Kosten und Ableitungen; die Schrittweitensuche von Frank-Wolfe wertet die Kosten je Bisektionsrunde aus) "
           f"und bei Gradient Projection die von der Wegerechnung besuchten Kanten. „–“ heißt: nicht innerhalb von {C.CAP} Iterationen. Gradient Projection: {C.UPDATES[update]}, α = {_alpha(alpha)}{', adaptiv' if adaptive else ''}. Eine Kantenoperation ist keine Sekunde; die Einheit ist so gewählt, dass sie auf jedem Rechner dieselbe Zahl gibt.")

st.markdown("### Wie viele Wege trägt das Verfahren?")
pv = cmp["paths"]
st.plotly_chart(build_paths_curve(pv), width="stretch", key="paths_curve")
k_gp = cmp["table"]["gp"][2]
q1, q2, q3 = st.columns(3)
q1.metric("Wege bei Frank-Wolfe", _int(pv["fw_curve"][-1]), help=f"Verschiedene Wege, die das Alles-oder-nichts bis Iteration {pv['upto']} erzeugt hat (bis Lücke 1e-4 bzw. bis zum Ende); alle tragen Fluss, die alten nur geometrisch weniger.")
q2.metric("Wege bei Gradient Projection", _int(cmp["results"]["gp"].npaths[k_gp if k_gp is not None else -1]), help="Wege mit Fluss zum Zeitpunkt, an dem die Lücke 1e-4 erstmals unterschritten wird (sonst am Ende).")
q3.metric("Zonenpaare", str(len(net.demand)), help="Je Zonenpaar mindestens ein Weg.")
st.caption("Die Wegemengen sind nicht so unterschiedlich, wie man denken könnte: im Standardnetz nutzen am Ende 21 der 30 Zonenpaare einen einzigen Weg, 8 zwei und eines drei. Gradient Projection wirft die überflüssigen Wege heraus (40 statt 70), Frank-Wolfe behält sie mit kleinem Fluss.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie wählt man den Schritt α?")
st.caption("Iterationen bis zur Lücke 1e-4 je Schrittweite, Gauss-Seidel und Jacobi, auf dem gewählten Netz. Graue Balken: in 200 Iterationen nicht erreicht. Ganz rechts der adaptive Schritt (Start 1, halbieren, wenn die Zielfunktion steigt).")
if net_key != "grid":
    st.info("Für dieses Experiment das Stadtgitter wählen.")
else:
    if st.button("Schrittweiten durchrechnen (dauert einige Sekunden)", key="alpha_start"):
        st.session_state["alpha_on"] = True
    if st.session_state.get("alpha_on"):
        with st.spinner("Rechne 12 Läufe..."):
            sweep = ev.alpha_sweep(ev.Params(*params))
        st.plotly_chart(build_alpha_sweep(sweep), width="stretch", key="alpha_chart")
        st.caption("Es gibt ein Fenster: zu klein ist langsam, zu groß zickzackt (der Verkehr springt zwischen zwei Verteilungen). Im Standardnetz liegt der volle Newton-Schritt (α = 1) außerhalb. Jacobi verträgt weniger als Gauss-Seidel, weil alle Zonenpaare gleichzeitig auf dieselben Fahrzeiten reagieren.")

st.subheader("🔬 Wie hängt das von der Last ab?")
st.caption(f"Iterationen bis zur Lücke 1e-4, Mittel über 5 feste Netze je Lastfaktor (nicht erreicht zählt als Grenze + 1 = {C.CAP + 1}). Gauss-Seidel; „gewählt“ ist das eingestellte Verfahren.")
if net_key != "grid":
    st.info("Für dieses Experiment das Stadtgitter wählen.")
else:
    if st.button("Lastreihe durchrechnen (dauert etwa eine Minute)", key="load_start"):
        st.session_state["load_on"] = True
    if st.session_state.get("load_on"):
        with st.spinner("Rechne 5 Lastfaktoren × 5 Netze × 5 Verfahren..."):
            rows = ev.load_series(ev.Params(*params))
        st.plotly_chart(build_load_series(rows), width="stretch", key="load_chart")
        st.table({"Lastfaktor": [_f(r["load"] / 10, 1) for r in rows], "Schritt 0,5 (gewählt): nicht erreicht": [f"{r['missed']['gp']} von {r['n']}" for r in rows],
                  "Schritt 1: nicht erreicht": [f"{r['missed']['gp_alpha1']} von {r['n']}" for r in rows], "adaptiv: nicht erreicht": [f"{r['missed']['gp_adaptive']} von {r['n']}" for r in rows],
                  "Frank-Wolfe: nicht erreicht": [f"{r['missed']['fw']} von {r['n']}" for r in rows]})
        st.caption("Je höher die Last, desto kleiner der brauchbare Schritt: ein fester Schritt, der bei Last 1,0 der schnellste ist, versagt bei hoher Last. Der adaptive Schritt kommt am weitesten; bei sehr hoher Last (3,0) erreicht aber auch er 1e-4 nur in 2 von 5 Netzen innerhalb von 200 Iterationen. Frank-Wolfe schafft es ab Last 1,5 in keinem Netz, das konjugierte Verfahren ab 2,0 (Voreinstellung: Stadtgitter, Zonen und Netze wie in der Frank-Wolfe-Demo).")

st.subheader("🔬 Ist Gradient Projection wirklich schneller?")
st.caption(f"Iterationen bis zur Lücke 1e-4 auf 40 festen Netzen mit den Einstellungen (nicht erreicht = Grenze + 1 = {C.CAP + 1}); Gauss-Seidel bei den festen Vergleichsverfahren „Schritt 1“ und „adaptiv“.")
if net_key != "grid":
    st.info("Für dieses Experiment das Stadtgitter wählen.")
else:
    if st.button("40 Netze durchrechnen (dauert etwa eine Minute)", key="dist_start"):
        st.session_state["dist_on"] = True
    if st.session_state.get("dist_on"):
        with st.spinner("Rechne 40 Netze × 5 Verfahren..."):
            dist = ev.distribution(ev.Params(*params))
        st.plotly_chart(build_dist(dist), width="stretch", key="dist_chart")
        ms = ("fw", "cfw", "gp", "gp_alpha1", "gp_adaptive")
        st.table({"Verfahren": [C.LABELS[m] for m in ms], "Lücke 1e-4 nicht erreicht": [f"{dist[m + '_missed']} von {dist['n']}" for m in ms]})
        st.table({"Verfahren": [C.LABELS[m] for m in ms], "schneller als konjugiertes FW": [f"{dist[m + '_beats_cfw']} von {dist['n']}" for m in ms]})
        st.caption("In der Voreinstellung (Last 1,0): ein fester Schritt ist im Median schnell, versagt aber in einem Teil der Netze; der adaptive Schritt kommt in allen an und ist in 37 von 40 Netzen schneller als das konjugierte Verfahren. Schneller heißt hier: weniger Iterationen bis 1e-4.")

st.subheader("🔬 Kantenflüsse gleich, Wege verschieden")
st.caption("Im Gleichgewicht sind die Kantenflüsse eindeutig, die Wegeflüsse nicht. Am Lehrnetz „Zwei Stufen“ (vier Wege, je Straße 5) kann man es von Hand nachrechnen; auf dem Stadtgitter prüft die Messung, dass Gradient Projection bei denselben Kantenflüssen ankommt wie das konjugierte Frank-Wolfe.")
starts = ev.lesson_starts()
start_names = ["alles auf Straße 1, dann 3", "alles auf Straße 1, dann 4", "alles auf Straße 2, dann 3", "alles auf Straße 2, dann 4", "gleichmäßig auf alle vier Wege"]
st.table({"Start": start_names, "Wege am Ende (Fluss)": ["; ".join(f"({a + 1},{b + 1}) {h:g}".replace(".", ",") for (a, b), h in r["paths"]) for r in starts]})
st.caption("Fünf Starts, fünf Rechnungen mit demselben Verfahren: jede Straße trägt am Ende 5, aber die Aufteilung auf die vier Wege ist eine andere. Gradient Projection ändert nur, was es ändern muss; das Gleichgewicht legt die Kantenflüsse fest, nicht die Wege.")
if net_key != "grid":
    st.info("Für den Test auf dem Stadtgitter das Stadtgitter wählen.")
else:
    if st.button("Eindeutigkeit prüfen (dauert einige Sekunden)", key="unique_start"):
        st.session_state["unique_on"] = True
    if st.session_state.get("unique_on"):
        with st.spinner("Rechne beide Verfahren bis zur hohen Genauigkeit..."):
            un = ev.uniqueness(ev.Params(*params))
        u1, u2, u3 = st.columns(3)
        u1.metric("Größte Abweichung der Kantenflüsse", _pct(un["max_diff"], 5), help="In Prozent der gesamten Nachfrage: adaptives Gradient Projection (Lücke unter 1e-8) gegen konjugiertes Frank-Wolfe (Lücke unter 1e-9).")
        u2.metric("Wegezeit-Abweichung", _sci(un["deviation"]), help="Größte relative Abweichung eines benutzten Wegs vom schnellsten Weg seines Zonenpaars: die Wardrop-Bedingung, direkt an den Wegen geprüft.")
        u3.metric("Wege mit Fluss", f"{un['paths']} für {un['pairs']} Zonenpaare")
        st.caption("Die Kantenflüsse stimmen überein; und weil Gradient Projection die Wege selbst kennt, lässt sich die Wardrop-Bedingung direkt an ihnen prüfen: alle benutzten Wege sind (bis auf die Restlücke) gleich schnell.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer ansetzt |
|---|---|
| **Ein Schritt α passt** | Er passt nur in einem Fenster, das mit der Last schrumpft; außerhalb zickzackt das Verfahren und kommt nie an. Der adaptive Schritt hier ist eine einfache Absicherung, kein Verfahren aus der Literatur. Frank-Wolfe hat diese Sorge nicht (langsam, aber sicher). |
| **Wegelisten bleiben klein** | Im Standardnetz liegen sie bei ein bis vier Wegen je Zonenpaar (bei doppelter Last bis zu 20). In großen Netzen mit vielen Zonen wachsen sie; dafür gibt es **bush-basierte** Verfahren (Algorithm B, TAPAS), die statt Wegen ganze Bäume je Ursprung halten (nur erwähnt, nicht gebaut). |
| **Aufwand in Kantenoperationen** | Die Einheit ist eine Zählung, keine Uhr: eine echte Implementierung hat andere Konstanten (Datenstrukturen, Zwischenspeicher). Die Reihenfolge der Verfahren sollte sich dadurch nicht ändern, die Faktoren schon. |
| **Jeder kennt alle Fahrzeiten** | Wardrop setzt perfekte Information und rationale, unendlich viele kleine Fahrer voraus. Lernen aus Erfahrung ist ein anderes Modell (Demo „No-Regret-Lernen“). |
| **Eine Fahrzeugklasse, statische Zuordnung** | Lkw und Pkw belasten die Straßen verschieden, und Staus bauen sich im Zeitverlauf auf; beides ist hier nicht abgebildet. |
| **Ein generiertes Netz** | Ein Gitter mit erzeugten Kapazitäten und Zonen, kein reales Stadtnetz und keine Fremddaten. |
"""
)
st.caption("Die Netzwerkfluss-Linie ist als Ganzes geplant: die zwölf Stücke der Hauptlinie, die Erweiterung E1 (Projektauswahl, Graph Cuts, Gomory-Hu-Baum) und die Erweiterung E4: **Frank-Wolfe** (Stück 16) und **Gradient Projection** (dieses Stück, gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell** wie in der Frank-Wolfe-Demo: Fahrzeit $t_e(x)=a_e+b_e(x/c_e)^p$, Nutzergleichgewicht als Lösung von $\min\sum_e\int_0^{x_e}t_e$, Systemoptimum mit den Grenzkosten $t_e+x_et_e'$ statt der Fahrzeit. Wegevariablen: $x_e=\sum_p\delta_{ep}h_p$, $\sum_{p\in P_w}h_p=q_w$, $h_p\ge0$.

**Gradient Projection** (Jayakrishnan, Tsai, Thomas, Lee 1994). Für das Zonenpaar $w$ mit Wegeliste $P_w$ und Basisweg $b=\arg\min_p c_p$ ($c_p=\sum_{e\in p}t_e(x_e)$):
$$\Delta h_p=\min\Big(h_p,\ \alpha\,\frac{c_p-c_b}{\sum_{e\in p\triangle b}t_e'(x_e)}\Big)\ \ (p\ne b),\qquad h_b\leftarrow h_b+\sum_{p\ne b}\Delta h_p,\quad h_p\leftarrow h_p-\Delta h_p.$$
Das ist ein Newton-Schritt auf dem Unterraum $h_p\ge0$ mit der diagonalen Näherung der Hesse-Matrix (Kopplungen zwischen Zonenpaaren und die gemeinsamen Kanten $p\cap b$ werden weggelassen: sie heben sich beim Verschieben von $p$ auf $b$ heraus). Ist $h_p$ auf null geschoben, fällt $p$ aus $P_w$; neue Wege liefert ein Dijkstra je Ursprung mit den aktuellen Fahrzeiten.
Die diagonale Näherung ignoriert, dass viele Zonenpaare gleichzeitig dieselben Kanten belasten: bei hoher Last überschätzt der volle Schritt ($\alpha=1$) die Wirkung und schießt über das Ziel hinaus, deshalb hängt der brauchbare Schritt von der Last ab.

**Adaptiver Schritt:** $\alpha\leftarrow\alpha/2$, sobald $Z(x^{k+1})>Z(x^k)$. **Relative Lücke** wie bei Frank-Wolfe, $g_k=\big(\sum x_et_e-\sum y_et_e\big)/\sum x_et_e$ mit $y$ = Alles-oder-nichts auf den Fahrzeiten am Iterationsanfang.

**Aufwand.** Kantenoperationen = durchsuchte Kanten der Dijkstra-Läufe + eine Operation je Kante und Kostenauswertung + bei Gradient Projection die von der Wegerechnung besuchten Kanten (Wegekosten, symmetrische Differenz, Flussbuchung).

Implementiert in `gp_scenario.py` (Netz, Zonen, Lehrnetze; Kopie aus Stück 16), `gp_frankwolfe.py` (Frank-Wolfe, MSA, konjugiert; Kopie aus Stück 16 mit Aufwandszähler), `gp_paths.py` (Wegelisten, Gradient Projection, Wardrop-Abweichung), `gp_evaluation.py` (Vergleiche, Lastreihe, Verteilungen).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
