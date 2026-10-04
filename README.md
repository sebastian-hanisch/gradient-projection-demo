# Gradient Projection – Wege statt Kantenflüsse: verschwindet der Endspurt von Frank-Wolfe? – Streamlit-Demo

Fünfte Erweiterung (Stück 17, **E4 Verkehrsumlegung, pfadbasiert**) der **Netzwerkfluss-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger von [frank-wolfe-demo](https://github.com/sebastian-hanisch/frank-wolfe-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Gradient Projection** (Jayakrishnan, Tsai, Prashker, Rajadhyaksha 1994) für die Verkehrsumlegung – an einem wachsenden Beispiel.
Das Modell ist das der Frank-Wolfe-Demo (Wardrop-Nutzergleichgewicht mit BPR-Fahrzeiten auf einem Stadtgitter, dasselbe Netz und derselbe Seed); neu ist der **Löser**. Frank-Wolfe kennt nur Kantenflüsse: jede Iteration mischt einen neuen schnellsten Weg in den Verkehr und lässt alle alten Wege im selben Verhältnis schrumpfen, ein schlechter Weg wird nie ganz leer – daher der lange Endspurt.
Gradient Projection hält je Zonenpaar eine **Liste von Wegen mit ihrem Fluss** und **schiebt Fluss vom teureren Weg auf den billigsten**, gestutzt bei null; die Schrittgröße kommt aus den Kostenableitungen (ein Newton-Schritt mit diagonaler Hesse-Matrix). Die Demo zeigt, was das bringt – und wo es **heikel** ist: die Schrittweite.

**Einordnung in die Reihe (die Kanten des Graphen):** Nachfolger von `frank-wolfe-demo` (gleiches Modell, anderer Löser); dasselbe Kürzeste-Wege-Orakel (Dijkstra je Ursprungszone) liefert die neuen Wege wie im Pricing der Column-Generation-Demo, und die Wegelisten sind die Spalten von dort. Bush-basierte Verfahren (Algorithm B, TAPAS) sind nur erwähnt.
```
multicommodity-demo (mehrere Güter teilen Kapazität)                                    [gebaut]
  ├─ mcf-column-generation-demo, garg-koenemann-demo, fixkosten-netzdesign-demo …        [gebaut]
  └─ frank-wolfe-demo (konvexe Kosten: Nutzergleichgewicht, Alles-oder-nichts-Orakel)    [gebaut]
       └─ gradient-projection-demo (Wege statt Kantenflüsse, Endspurt)                   [dieses Stück]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Lehrnetze von Hand, Beispielnetze über ihre Seeds, Verteilungen über 40 feste Netze (Seeds ab 100000; die Lastreihe über 5 Netze). Standard: 8 × 8 Kreuzungen (224 Kanten), 6 Zonen, 30 Zonenpaare, Last 1,0, Seed 5 – **dasselbe Netz wie in Stück 16** –, Gradient Projection mit Gauss-Seidel und Schritt 0,5.
Die Rechnung nutzt nur + − × ÷ auf Python-Zahlen (keine Bibliotheksfunktionen): Iterationszahlen sind auf allen Plattformen dieselben. Aufwand zählt in **Kantenoperationen** (durchsuchte Kanten der Kürzeste-Wege-Läufe, eine Operation je Kante und Kostenauswertung, bei Gradient Projection die von der Wegerechnung besuchten Kanten), nie in Sekunden.

**Der Endspurt verschwindet – mit dem richtigen Schritt.** Relative Lücke unter 1e-2 / 1e-3 / 1e-4 / 1e-5 / 1e-6 nach **3 / 6 / 11 / 16 / 20** Iterationen; Frank-Wolfe braucht 4 / 21 / 102 / – / –, das konjugierte Verfahren 4 / 15 / 24 / 69 / 145. In Kantenoperationen bis 1e-4: **61 600** gegen 641 312 (Frank-Wolfe, das **Zehnfache**) und 152 096 (konjugiert, das **2,5-fache**).
Eine Iteration kostet dabei etwa gleich viel (5 600 gegen 6 300 Kantenoperationen): der Gewinn steckt in den Iterationen, nicht in einer billigeren Iteration.

**Aber der Schritt ist die neue Schwachstelle.** Der volle Newton-Schritt (α = 1) **zickzackt**: die Lücke fällt nach 9 Iterationen unter 1e-2 und kommt dann nicht weiter (nach 100 Iterationen 1,6e-2, Gesamtfahrzeit 7 037, 2,0 % über dem Gleichgewicht 6 897; 228 Wege werden aus den Listen gelöscht). Grund: die diagonale Näherung übersieht, dass viele Zonenpaare gleichzeitig dieselben Kanten entlasten.
Bis Lücke 1e-4 (Gauss-Seidel): α = 0,25 → 25, **0,5 → 11**, 0,75 → 25, 1 und 1,5 nie; Jacobi 28 / 14 / nie / nie / nie. Es gibt ein Fenster, und es schrumpft mit der Last: Mittel über 5 Netze, Iterationen bis 1e-4 (nicht erreicht = 201), Schritt 0,5: 2,8 bei Last 0,5, 48 bei 1,0; bei Last 1,5 wird 1e-4 nur in 1 von 5 Netzen erreicht, bei 2,0 und 3,0 in keinem – Frank-Wolfe schafft es dort (und bei 1,5) in 200 Iterationen in keinem Netz, das konjugierte Verfahren bei 1,5 in 3 von 5.
Über 40 feste Netze (200 Iterationen, nicht erreicht = 201): Schritt 0,5 hat den **Median 19** (konjugiert 38,5), aber 1e-4 wird in **11 von 40** Netzen nicht erreicht – im Mittel 71, **nicht besser als das konjugierte Verfahren (70)**. Schritt 1: 33 von 40 nicht erreicht. Frank-Wolfe: Mittel 159, 22 von 40 nicht; konjugiert: 70, 8 von 40.

**Adaptiver Schritt.** Start mit α = 1 und halbieren, sobald die Zielfunktion nach einer Iteration steigt (eine einfache Absicherung, kein Verfahren aus der Literatur): im Standardnetz 7 / 9 / 17 / 26 / 30 Iterationen (der Schritt endet bei 0,5) – langsamer als der von Hand gewählte Schritt, aber **er kommt an, ohne dass man ihn kennt**. Über 40 Netze: Median 16, Mittel 20, **alle 40 erreicht**, schneller als das konjugierte Verfahren in **37**, langsamer in 2. Im Median 90 118 Kantenoperationen bis 1e-4 gegen 202 272 (konjugiert, 32 Netze erreicht) und 798 112 (Frank-Wolfe, 18 Netze erreicht).
Bei doppelter Last (Seed 5): Frank-Wolfe 35 Iterationen bis 1e-2, 1e-3 in 200 nicht; konjugiert 22 / 70, 1e-4 nicht; Gradient Projection adaptiv **11 / 47 / 89** bis 1e-2 / 1e-3 / 1e-4 und 157 bis 1e-6 (der Schritt sinkt auf 0,125); ein fester Schritt versagt (0,5: Lücke 9,3e-2 nach 200; 1: 6,3e-1). Adaptiv erreicht in der Lastreihe 1e-4 in 5 / 5 / 5 / 4 / 2 von 5 Netzen (Last 0,5 bis 3,0), Schritt 0,5 in 5 / 4 / 1 / 0 / 0.

**Wege.** Frank-Wolfe hat bis Lücke 1e-4 (Iteration 102) **70 verschiedene Wege** erzeugt, und alle tragen Fluss (die alten nur geometrisch weniger); Gradient Projection trägt bei 1e-4 43 und am Ende **40 Wege** (21 der 30 Zonenpaare nutzen einen einzigen Weg, 8 zwei, eines drei), 31 Wege sind unterwegs gelöscht worden. Bei doppelter Last sind es bis zu 20 Wege je Zonenpaar, insgesamt 90.
**Kantenflüsse eindeutig, Wegeflüsse nicht:** Gradient Projection (adaptiv, Lücke unter 1e-8) und konjugiertes Frank-Wolfe (Lücke unter 1e-9) enden bei denselben Kantenflüssen (größte Abweichung unter 0,001 % der Nachfrage); weil das Verfahren die Wege kennt, lässt sich die **Wardrop-Bedingung direkt an den Wegen prüfen** (größte relative Abweichung eines benutzten Wegs vom schnellsten unter 1e-5). Im Lehrnetz „Zwei Stufen“ (Nachfrage 10) trägt jede Straße nach einer Iteration 5, und fünf Starts (ganzer Verkehr auf je einem der vier Wege, oder gleichmäßig) enden bei **drei verschiedenen** Wegeaufteilungen (von Hand im Test).

**Systemoptimum.** Dasselbe Verfahren mit Grenzkosten: Lücke unter 1e-2 / 1e-3 / 1e-4 / 1e-5 / 1e-6 nach **8 / 16 / 25 / 36 / 48** Iterationen (Gesamtfahrzeit 6 632); Frank-Wolfe braucht 15 / 79 / – und das konjugierte Verfahren 10 / 29 / 122 – es steht nach 800 Iterationen erst bei Lücke 2e-5. Preis der Anarchie wie in Stück 16: **1,040** (6 897 gegen 6 632). Pigou (von Hand): Gleichgewicht sofort, Systemoptimum nach **einer** Iteration halb und halb (0,75).
**Negativkontrolle Wegelisten eingefroren:** dürfen neue Wege nur in den ersten 2 Iterationen in die Listen, sinkt die Lücke nach 3 Iterationen unter 1e-2, bleibt dann aber bei 4,1e-3 hängen (Gesamtfahrzeit 6 983, 1,2 % über dem Gleichgewicht): ohne das Kürzeste-Wege-Orakel fehlen dem Verfahren die guten Wege.

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Bau standen acht Vermutungen im Plan. Gemessen:

- **„Gradient Projection erreicht 1e-4 (und 1e-6) in einem Bruchteil der Iterationen“ – mit dem richtigen Schritt ja (11 statt 102), aber nicht robust.** Der von Hand gewählte Schritt 0,5 versagt in 11 von 40 Netzen und ist im Mittel nicht besser als das konjugierte Verfahren; erst der adaptive Schritt (Median 16, alle 40 erreicht) hält, was der Median von Schritt 0,5 verspricht.
- **„Der Newton-Schritt α = 1 ist der richtige“ – widerlegt.** Der volle Schritt zickzackt schon im Standardnetz; der beste feste Schritt ist dort 0,5, und bei doppelter Last müssen es noch kleinere sein.
- **„Eine Iteration kostet mehr als bei Frank-Wolfe“ – in dieser Zähleinheit nicht.** Frank-Wolfes exakte Schrittweitensuche (20 Bisektionsrunden mit Kostenauswertung) wiegt die Wegerechnung auf: etwa 6 300 gegen 5 600 Kantenoperationen je Iteration. Die Einheit ist eine Zählung, keine Uhr; eine echte Implementierung hat andere Konstanten.
- **„Gradient Projection hält die Wegemengen spärlich (unter 10 % der Frank-Wolfe-Wege)“ – widerlegt:** 40 statt 70 Wege, also 57 %. Im Gleichgewicht brauchen die meisten Zonenpaare nur einen Weg, und Frank-Wolfe erzeugt gar nicht so viele verschiedene.
- **„Gauss-Seidel ist schneller als Jacobi, Jacobi instabiler“ – bestätigt:** 11 gegen 14 Iterationen bis 1e-4 bei Schritt 0,5; bei Schritt 0,75 kommt Jacobi in 200 Iterationen nicht mehr bis 1e-4, Gauss-Seidel nach 25.
- **„Im Systemoptimum gilt dasselbe“ – ja, deutlicher:** Frank-Wolfe ist dort besonders langsam (das konjugierte Verfahren nach 800 Iterationen erst bei 2e-5), Gradient Projection braucht 25 Iterationen bis 1e-4.
- **„Kantenflüsse eindeutig, Wegeflüsse nicht“ – bestätigt** (Kantenflüsse bis unter 0,001 %, drei Aufteilungen im Lehrnetz).
- **Kontrollen:** zu großer Schritt zickzackt, eingefrorene Wegelisten bleiben hängen, ohne Kostenanstieg (Ableitung 0) verschiebt das Verfahren den ganzen Fluss in einer Iteration (Test).
- **Abweichung vom Plan:** die Lastreihe rechnet 5 Netze und bis 200 Iterationen; die Aktualisierung „je Zonenpaar“ wurde in der Vormessung erwogen und nicht übernommen (die Iterationen je Ursprung sind die zum Aufwand von Frank-Wolfe passende Einheit); die Lücke des Gauss-Seidel-Verfahrens wird mit einem eigenen Kontrolllauf gemessen, der nicht zum Aufwand zählt.

## Was die Demo zeigt

- **Fluss auf Wegen verschieben:** Regler und Abspielen durch die Iterationen, Stadtplan (Kantenbreite = Verkehr, Farbe = Auslastung), Lückenkurve mit Marke, dazu die Wege eines wählbaren Zonenpaars (Karte und Balken mit Fluss und Zeit je Weg).
- **Gradient Projection gegen Frank-Wolfe:** Lückenkurven über Iterationen und über den Aufwand, Tabellen bis 1e-2 … 1e-5 und Aufwand, Wegemengen.
- **Experimente (auf Abruf):** Schritt α (Gauss-Seidel und Jacobi, adaptiv), Lastreihe, 40 Netze, Kantenflüsse gleich und Wege verschieden (Lehrnetz mit fünf Starts, Wardrop-Bedingung an den Wegen).
- **Wo die Annahmen enden:** ein Schritt passt, Wegelisten bleiben klein, Aufwandseinheit, perfekte Information, eine Fahrzeugklasse, ein generiertes Netz.

## Modell und Verfahren

- **Modell** wie in Stück 16: Fahrzeit $t_e(x)=a_e+b_e(x/c_e)^p$ (BPR), Nutzergleichgewicht $\min\sum_e\int_0^{x_e}t_e$, Systemoptimum mit den Grenzkosten $t_e+x_et_e'$.
- **Gradient Projection:** für das Zonenpaar $w$ mit Wegeliste $P_w$ und Basisweg $b=\arg\min c_p$: $\Delta h_p=\min\big(h_p,\ \alpha\,(c_p-c_b)/\sum_{e\in p\triangle b}t_e'\big)$, Fluss von $p$ auf $b$; $h_p=0$ entfernt den Weg. Neue Wege liefert ein Dijkstra je Ursprung. *Gauss-Seidel*: Ursprung für Ursprung mit neuen Kosten; *Jacobi*: alle Zonenpaare mit den Kosten vom Iterationsanfang. *Adaptiv*: $\alpha\leftarrow\alpha/2$, sobald die Zielfunktion steigt.
- **Relative Lücke** wie bei Frank-Wolfe: $(\sum x t-\sum y t)/\sum x t$ mit $y$ = Alles-oder-nichts auf den Kosten am Iterationsanfang.

## Ehrliche Grenzen

- **Ein Schritt α passt nur in einem Fenster,** das mit der Last schrumpft; der adaptive Schritt ist eine einfache Absicherung, kein Verfahren aus der Literatur. Frank-Wolfe hat diese Sorge nicht (langsam, aber sicher).
- **Wegelisten wachsen mit Netz und Last** (hier ein bis vier Wege je Zonenpaar, bei doppelter Last bis zu 20); bush-basierte Verfahren (Algorithm B, TAPAS) halten stattdessen einen Baum je Ursprung und sind nur erwähnt.
- **Aufwand in Kantenoperationen** ist eine Zählung, keine Uhr.
- **Perfekte Information, eine Fahrzeugklasse, statische Zuordnung, BPR-Fahrzeiten;** Lernen aus Erfahrung und dynamische Umlegung sind andere Modelle.
- **Synthetische Daten:** ein Gitter mit erzeugten Kapazitäten und Zonen, kein reales Netz, keine Fremddaten (etwa Sioux Falls).

## Bewusst nicht umgesetzt

- Bush-basierte Verfahren (Algorithm B, TAPAS), Gauss-Seidel je Zonenpaar, Schrittweitensuche für Gradient Projection, Mehrklassen- und dynamische Umlegung.

## Dateien

```
app.py                  Oberfläche (Streamlit)
gp_scenario.py          Stadtgitter, Zonen, Lehrnetze, Zufallsgenerator (Kopie aus frank-wolfe-demo)
gp_frankwolfe.py        Kosten, Alles-oder-nichts, Frank-Wolfe, konjugiert (Kopie aus frank-wolfe-demo, mit Aufwandszähler)
gp_paths.py             Wegelisten, Gradient Projection, Wardrop-Abweichung
gp_evaluation.py        Vergleiche, Wegemengen, Schritt, Lastreihe, Verteilungen, Lehrnetz-Starts
gp_visualization.py     Plotly-Abbildungen
gp_presets.py           Permalink, Presets, Zufalls-Seed
gp_constants.py         Konstanten, Regler-Grenzen, feste Seed-Mengen, Preset-Texte
tests/                  Kern, Auswertung, Presets, Behauptungen, App, Regler-Zustand
```

## Lokal starten

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
venv/Scripts/streamlit run app.py
```

## Tests ausführen

```bash
venv/Scripts/pip install -r requirements-dev.txt
venv/Scripts/python -m pytest tests/ -v
```

Gebaut mit Streamlit und Plotly; der Kern ist reines Python.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Netzwerkfluss: vom Max-Flow zum Netzdesign](https://sebastianhanisch.net/konzepte-netzwerkfluss.html).
