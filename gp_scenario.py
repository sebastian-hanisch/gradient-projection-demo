"""Szenario (Kopie aus frank-wolfe-demo, dasselbe Vehikel wie Stück 16): ein Stadtnetz als Gitter mit Straßen in beide Richtungen, Zonen mit Nachfrage zwischen allen Zonenpaaren, und feste Lehrnetze (Pigou, Zwei Stufen).

Jede Kante hat eine Fahrzeit t(x) = a + b (x / c)^p in Abhängigkeit vom Verkehr x auf ihr (BPR-Form: a = Freifahrtzeit t0, b = 0,15 t0, p = 4, c = Kapazität). Die Nachfrage ist stetig (nicht-atomar): Verkehr lässt sich beliebig teilen.
Alle Parameter sind ganzzahlig und kommen aus einem eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt aus `numpy.random`: numpy garantiert keine über Versionen stabilen Zufallsströme, die CI installiert aber
wöchentlich die neueste Version. Die Rechnung selbst nutzt nur + - * / auf Python-Zahlen, so dass Iterationszahlen auf Windows und Linux dieselben sind.
"""

from dataclasses import dataclass

_MASK = (1 << 64) - 1
MAP = 100
BPR_ALPHA = 0.15


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 1e6 liegt unter 1e-13)."""
        return self.next() % n


@dataclass(frozen=True)
class Network:
    n: int
    names: tuple
    pos: tuple
    links: tuple          # ((u, v, a, b, c, p), ...): Fahrzeit a + b (x / c)^p
    zones: tuple          # Knoten mit Nachfrage
    demand: tuple         # ((Ursprung, Ziel, Menge), ...)
    kind: str = "grid"
    side: int = 0

    @property
    def m(self):
        return len(self.links)

    def origins(self):
        out = {}
        for o, d, q in self.demand:
            out.setdefault(o, []).append((d, q))
        return out

    def total_demand(self):
        return sum(q for _, _, q in self.demand)


def generate(side, zones, load, seed):
    """Gitter side x side, Straßen in beide Richtungen: Freifahrtzeit 2 bis 6, Kapazität 20 bis 45 je Straße (beide Richtungen gleich); `zones` Zonen auf verschiedenen Knoten; Nachfrage 5 bis 15 je geordnetem
    Zonenpaar mal `load` / 10 (Lastfaktor in Zehnteln). Die Zufallszahlen werden in fester Reihenfolge gezogen: der Lastfaktor ändert nur die Menge, nicht das Netz."""
    rng = SplitMix64(seed)
    n = side * side
    links = []
    for y in range(side):
        for x in range(side):
            i = y * side + x
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < side and y + dy < side:
                    j = (y + dy) * side + x + dx
                    t0 = 2 + rng.below(5)
                    c = 20 + rng.below(26)
                    for u, v in ((i, j), (j, i)):
                        links.append((u, v, float(t0), BPR_ALPHA * t0, float(c), 4))
    chosen = []
    while len(chosen) < min(zones, n):
        k = rng.below(n)
        if k not in chosen:
            chosen.append(k)
    base = {}
    for o in chosen:
        for d in chosen:
            if o != d:
                base[(o, d)] = 5 + rng.below(11)
    demand = tuple((o, d, base[(o, d)] * load / 10) for (o, d) in sorted(base))
    step = MAP / max(1, side - 1)
    pos = tuple((round(step * (i % side), 3), round(step * (side - 1 - i // side), 3)) for i in range(n))
    names = tuple(f"Kreuzung {i + 1}" for i in range(n))
    return Network(n, names, pos, tuple(links), tuple(chosen), demand, "grid", side)


# --- feste Lehrnetze ------------------------------------------------------------------------------------------------------------

def pigou():
    """Pigou-Netz: zwei parallele Straßen von A nach B, die eine mit fester Fahrzeit 1, die andere mit Fahrzeit x (der Verkehrsmenge auf ihr); Nachfrage 1. Nutzergleichgewicht: alles auf die zweite Straße, Fahrzeit 1;
    Systemoptimum: halb und halb, mittlere Fahrzeit 0,75. Preis der Anarchie 4/3."""
    return Network(2, ("A", "B"), ((10.0, 50.0), (90.0, 50.0)), ((0, 1, 1.0, 0.0, 1.0, 1), (0, 1, 0.0, 1.0, 1.0, 1)), (0, 1), ((0, 1, 1.0),), "pigou")


def two_stages():
    """Zwei Stufen: A -> M über zwei gleiche parallele Straßen, M -> B über zwei gleiche parallele Straßen (Fahrzeit 1 + x / 10), Nachfrage 10 von A nach B. Die Kantenflüsse sind eindeutig (je Straße 5),
    die Wegeflüsse nicht: vier Wege (Straße 1 oder 2, dann Straße 3 oder 4) lassen sich auf verschiedene Weisen so belegen, dass jede Straße 5 trägt."""
    links = ((0, 1, 1.0, 0.1, 1.0, 1), (0, 1, 1.0, 0.1, 1.0, 1), (1, 2, 1.0, 0.1, 1.0, 1), (1, 2, 1.0, 0.1, 1.0, 1))
    return Network(3, ("A", "M", "B"), ((10.0, 50.0), (50.0, 50.0), (90.0, 50.0)), links, (0, 2), ((0, 2, 10.0),), "zweistufen")


LESSONS = {"pigou": pigou, "zweistufen": two_stages}


def build(net, side, zones, load, seed):
    """Netz zu den Einstellungen; feste Lehrnetze ignorieren die Zufallsparameter."""
    if net != "grid":
        return LESSONS[net]()
    return generate(side, zones, load, seed)
