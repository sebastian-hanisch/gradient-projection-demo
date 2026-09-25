"""Presets: vollständig, in den Grenzen, und jedes Beispiel zeigt, was sein Hilfetext behauptet."""

import pytest

import gp_constants as C
import gp_evaluation as ev
import gp_presets as P

KEYS = set(P.PRESET_KEYS)


def _params(p):
    return ev.Params(p["net"], p["side"], p["zones"], p["load"], p["mode"], p["update"], p["alpha"], p["adaptive"], p["frozen"], p["iterations"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["mode"] in C.MODES and p["update"] in C.UPDATES and p["alpha"] in C.ALPHAS and p["iterations"] in C.ITERATIONS
    assert isinstance(p["adaptive"], bool) and isinstance(p["frozen"], bool)
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["load"] - C.LOAD_MIN) % 5 == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_permalink_casts_accept_only_valid_values():
    spec = P.SETTING_SPECS
    assert spec["alpha_radio"].caster("0.5") == 0.5 and spec["alpha_radio"].caster("1") == 1.0
    for bad in ("0.6", "x"):
        with pytest.raises(ValueError):
            spec["alpha_radio"].caster(bad)
    assert spec["adaptive_check"].caster("1") is True and spec["adaptive_check"].caster("0") is False
    with pytest.raises(ValueError):
        spec["adaptive_check"].caster("2")


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_defaults_equal_the_first_preset():
    p = C.PRESETS["🏙️ Stadtgitter"]
    assert _params(p) == ev.DEFAULT_PARAMS


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"🪜 Zwei Stufen"}


def test_the_presets_show_both_good_and_bad_news():
    """Gut: der Standard, Jacobi, adaptiv und das Systemoptimum erreichen 1e-6; schlecht: Schritt 1 (nie unter 1e-3), Wege eingefroren (nie unter 1e-3)."""
    res = {n: ev.analyse(_params(p))["result"] for n, p in C.PRESETS.items()}
    for good in ("🏙️ Stadtgitter", "🔁 Jacobi", "🛡️ Adaptiv", "🚗 Hohe Last, adaptiv", "🌐 Systemoptimum"):
        assert res[good].first_below(1e-6) is not None, good
    assert res["⚠️ Schritt 1,0"].first_below(1e-3) is None and res["🧊 Wege eingefroren"].first_below(1e-3) is None
    assert res["🪜 Zwei Stufen"].iterations == 1
