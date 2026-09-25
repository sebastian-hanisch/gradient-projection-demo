"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Gradient Projection: Wege statt Kantenflüsse - verschwindet der Endspurt von Frank-Wolfe?"."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------
SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 10, 8            # Kantenlänge des Stadtgitters (Kreuzungen)
ZONES_MIN, ZONES_MAX, DEFAULT_ZONES = 3, 10, 6         # Zonen mit Nachfrage
LOAD_MIN, LOAD_MAX, DEFAULT_LOAD = 5, 30, 10           # Lastfaktor in Zehnteln (1,0 = Standard), Schritt 5
DEFAULT_SEED = 5                                       # dasselbe Standardnetz wie in der Frank-Wolfe-Demo (Stück 16)
SEED_MAX = 2_000_000_000

NETS = {
    "grid": "Stadtgitter mit Zonen",
    "pigou": "Pigou-Netz (zwei parallele Straßen)",
    "zweistufen": "Zwei Stufen (Wege nicht eindeutig)",
}
DEFAULT_NET = "grid"
FIXED_NETS = ("pigou", "zweistufen")
MODES = {"ue": "Nutzergleichgewicht (jeder wählt den schnellsten Weg)", "so": "Systemoptimum (kleinste Gesamtfahrzeit)"}
DEFAULT_MODE = "ue"
UPDATES = {"gauss_seidel": "Gauss-Seidel (Ursprung für Ursprung, Kosten danach neu)", "jacobi": "Jacobi (alle Zonenpaare mit den Kosten vom Iterationsanfang)"}
DEFAULT_UPDATE = "gauss_seidel"
ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5)                   # Skalierung des Newton-Schritts
DEFAULT_ALPHA = 0.5
FROZEN_UNTIL = 2                                       # Negativkontrolle: nur in den ersten Iterationen kommen neue Wege in die Listen
ITERATIONS = (50, 100, 200)
DEFAULT_ITERATIONS = 100
TOLS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
SHOWN_TOLS = TOLS[:4]                                  # Spalten der Vergleichstabelle
GAP_STOP = 1e-6                                        # Abbruch der Rechnung

# --- feste Seed-Mengen (dieselben wie in den Flussdemos; unabhängig vom Nutzer-Seed) ---------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
LOAD_SEEDS = DIST_SEEDS[:5]
LOADS = (5, 10, 15, 20, 30)
CAP = 200                                              # Iterationsgrenze der Experimente (nicht erreicht = CAP + 1)

COLORS = {"fw": "#1f77b4", "cfw": "#d62728", "gp": "#2ca02c", "gp_adaptive": "#9467bd", "gp_alpha1": "#8c564b", "ue": "#1f77b4", "so": "#d62728", "zone": "#ff7f0e", "node": "#111111"}
LABELS = {"fw": "Frank-Wolfe (exakte Schrittweite)", "cfw": "Konjugiertes Frank-Wolfe", "gp": "Gradient Projection (gewählt)", "gp_adaptive": "Gradient Projection, α adaptiv", "gp_alpha1": "Gradient Projection, α = 1"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net=DEFAULT_NET, side=DEFAULT_SIDE, zones=DEFAULT_ZONES, load=DEFAULT_LOAD, mode=DEFAULT_MODE, update=DEFAULT_UPDATE, alpha=DEFAULT_ALPHA, adaptive=False, frozen=False,
             iterations=DEFAULT_ITERATIONS, seed=DEFAULT_SEED)
PRESETS = {
    "🏙️ Stadtgitter": {**_BASE},
    "🔁 Jacobi": {**_BASE, "update": "jacobi"},
    "⚠️ Schritt 1,0": {**_BASE, "alpha": 1.0},
    "🛡️ Adaptiv": {**_BASE, "alpha": 1.0, "adaptive": True},
    "🚗 Hohe Last, adaptiv": {**_BASE, "load": 20, "alpha": 1.0, "adaptive": True, "iterations": 200},
    "🌐 Systemoptimum": {**_BASE, "mode": "so"},
    "🪜 Zwei Stufen": {**_BASE, "net": "zweistufen", "alpha": 1.0},
    "🧊 Wege eingefroren": {**_BASE, "frozen": True},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py belegt (Lehrnetz von Hand, Stadtgitter über die Seeds der Presets)
PRESET_HELP = {
    "🏙️ Stadtgitter": "Dasselbe Netz wie in der Frank-Wolfe-Demo (8 × 8 Kreuzungen, 224 Kanten, 6 Zonen, 30 Zonenpaare): Gradient Projection (Gauss-Seidel, Schritt 0,5) unterschreitet die relative Lücke 1e-2 nach 3, 1e-3 nach 6, 1e-4 nach 11, 1e-5 nach 16 und 1e-6 nach 20 Iterationen. Frank-Wolfe braucht für 1e-4 102 Iterationen, das konjugierte Verfahren 24. In Kantenoperationen bis 1e-4: 61 600 gegen 641 312 (Frank-Wolfe) und 152 096 (konjugiert). Am Ende tragen 40 Wege den Verkehr der 30 Zonenpaare.",
    "🔁 Jacobi": "Alle Zonenpaare rechnen mit den Fahrzeiten vom Anfang der Iteration: 1e-2 nach 4, 1e-3 nach 8, 1e-4 nach 14 und 1e-6 nach 24 Iterationen - etwas langsamer als Gauss-Seidel (11 bis 1e-4), dafür ohne Abhängigkeit zwischen den Zonen. Jacobi verträgt einen kleineren Schritt: bei 0,75 kommt es in 200 Iterationen nicht mehr bis 1e-4 (Gauss-Seidel schon).",
    "⚠️ Schritt 1,0": "Der volle Newton-Schritt: die Lücke fällt nach 9 Iterationen unter 1e-2, kommt aber danach nicht weiter und liegt nach 100 bei 1,6e-2; die Gesamtfahrzeit bleibt bei 7 037 (2,0 % über dem Gleichgewicht 6 897), 228 Wege werden aus den Listen gelöscht. Der Schritt schießt über das Ziel hinaus, weil viele Zonenpaare gleichzeitig dieselben Kanten entlasten.",
    "🛡️ Adaptiv": "Start mit Schritt 1, halbieren, sobald die Zielfunktion steigt: der Schritt landet bei 0,5, die Lücke fällt unter 1e-2 nach 7, 1e-3 nach 9, 1e-4 nach 17, 1e-5 nach 26 und 1e-6 nach 30 Iterationen. Das ist langsamer als der von Hand gewählte Schritt 0,5 (11 bis 1e-4), aber es kommt an, ohne dass man den Schritt kennen muss.",
    "🚗 Hohe Last, adaptiv": "Doppelte Nachfrage: Frank-Wolfe erreicht 1e-2 nach 35 Iterationen und 1e-3 in 200 nicht, das konjugierte Verfahren 1e-3 nach 70 und 1e-4 in 200 nicht. Gradient Projection mit adaptivem Schritt (er sinkt auf 0,125) erreicht 1e-2 nach 11, 1e-3 nach 47, 1e-4 nach 89 und 1e-6 nach 157 Iterationen. Ein fester Schritt versagt hier (0,5: Lücke 9,3e-2 nach 200, 1: 6,3e-1). Die Zonenpaare brauchen jetzt bis zu 20 Wege, insgesamt 90.",
    "🌐 Systemoptimum": "Die Grenzkosten als Kosten: Gradient Projection erreicht 1e-2 nach 8, 1e-3 nach 16, 1e-4 nach 25 und 1e-6 nach 48 Iterationen (Gesamtfahrzeit 6 632); Frank-Wolfe braucht für 1e-3 79 Iterationen und erreicht 1e-4 in 200 nicht, das konjugierte Verfahren 29 und 122. Im Systemoptimum ist Frank-Wolfe besonders langsam: das konjugierte Verfahren steht nach 800 Iterationen erst bei Lücke 2e-5. Der Preis der Anarchie bleibt 1,040.",
    "🪜 Zwei Stufen": "Vier Wege (Straße 1 oder 2, dann 3 oder 4), Nachfrage 10: jede Straße trägt am Ende 5, und Gradient Projection braucht dafür eine einzige Iteration. Welche Wege den Verkehr tragen, hängt vom Start ab: (1,3) und (2,4), oder (1,4) und (2,3), oder alle vier je 2,5 - die Tabelle unten rechnet es für fünf Starts durch.",
    "🧊 Wege eingefroren": "Neue Wege dürfen nur in den ersten 2 Iterationen in die Listen: die Lücke sinkt noch nach 3 Iterationen unter 1e-2, bleibt dann aber bei 4,1e-3 hängen, die Gesamtfahrzeit bei 6 983 (1,2 % über dem Gleichgewicht 6 897). Ohne das Kürzeste-Wege-Orakel fehlen dem Verfahren die guten Wege; verschieben allein reicht nicht.",
}
