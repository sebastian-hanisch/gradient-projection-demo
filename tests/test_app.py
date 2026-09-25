"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Aktualisierungen und Modi, Randgrößen, Iterations-Regler, ausgeblendete Regler, Permalink, Experimente auf Abruf."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gp_constants as C
from gp_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _step_slider(at):
    found = [s for s in at.slider if s.key == "gp_step"]
    return found[0] if found else None


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def test_default_renders_without_exception():
    at = _run()
    assert _step_slider(at).value == _step_slider(at).max == 20 and _metric(at, "Wege bei Frank-Wolfe") == ["70"]
    assert any("relative Lücke unter 1e-6" in t and "40 Wege" in t for t in _texts(at))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and _metric(at, "Wege bei Frank-Wolfe")


@pytest.mark.parametrize("update", list(C.UPDATES))
@pytest.mark.parametrize("mode", list(C.MODES))
def test_every_update_and_mode_renders(update, mode):
    def setup(at):
        at.session_state["update_radio"] = update
        at.session_state["mode_radio"] = mode
        at.session_state["iterations_radio"] = 50
    at = _run(setup)
    assert not at.error and _metric(at, "Wege bei Frank-Wolfe")


def test_extreme_sizes_render():
    for vals in ((("side_slider", C.SIDE_MIN), ("zones_slider", C.ZONES_MIN), ("load_slider", C.LOAD_MIN)), (("side_slider", C.SIDE_MAX), ("zones_slider", C.ZONES_MAX), ("load_slider", C.LOAD_MAX))):
        def setup(at, vals=vals):
            for key, value in vals:
                at.session_state[key] = value
            at.session_state["iterations_radio"] = 50
        at = _run(setup)
        assert not at.error and _metric(at, "Wege bei Frank-Wolfe")


def test_iteration_slider_moves_through_frames():
    at = _run()
    top = int(_step_slider(at).max)
    for value in (0, 1, top // 2, top):
        _step_slider(at).set_value(value)
        at.run()
        assert not at.exception and _step_slider(at).value == value


def test_the_zone_pair_selector_changes_the_view():
    at = _run()
    box = next(s for s in at.selectbox if s.key == "gp_pair")
    box.set_value(0)
    at.run()
    assert not at.exception and next(s for s in at.selectbox if s.key == "gp_pair").value == 0


def test_pigou_has_no_iterations_for_the_equilibrium():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "pigou"))
    assert _step_slider(at) is None and not at.exception


def test_step_too_big_and_frozen_lists_warn():
    at = _run(lambda a: a.session_state.__setitem__("alpha_radio", 1.0))
    assert any("zu groß" in t for t in _texts(at))
    at = _run(lambda a: a.session_state.__setitem__("frozen_check", True))
    assert any("Wegelisten wachsen nur" in t for t in _texts(at))


def test_two_stages_explains_path_flows():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "zweistufen"))
    assert not at.exception and any("Wegeflüsse nicht" in m.value for m in at.markdown)


def test_hidden_controls_keep_their_values_across_a_net_switch():
    at = _run()
    at.sidebar.slider(key="side_slider").set_value(6)
    at.run()
    at.sidebar.selectbox(key="net_select").set_value("pigou")
    at.run()
    assert not at.exception and not [w for w in at.sidebar.slider if w.key == "side_slider"]
    at.sidebar.selectbox(key="net_select").set_value("grid")
    at.run()
    assert at.sidebar.slider(key="side_slider").value == 6 and not at.exception


def test_permalink_settings_are_loaded_and_clamped():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["net"] = "grid"
    at.query_params["side"] = "99"
    at.query_params["load"] = "17"
    at.query_params["update"] = "jacobi"
    at.query_params["alpha"] = "0.25"
    at.query_params["adaptive"] = "1"
    at.query_params["frozen"] = "0"
    at.query_params["mode"] = "so"
    at.query_params["iterations"] = "200"
    at.run()
    assert not at.exception
    assert at.sidebar.slider(key="side_slider").value == C.SIDE_MAX and at.sidebar.slider(key="load_slider").value == 15
    assert at.sidebar.radio(key="update_radio").value == "jacobi" and at.sidebar.radio(key="alpha_radio").value == 0.25 and at.sidebar.radio(key="mode_radio").value == "so"
    assert at.sidebar.checkbox(key="adaptive_check").value is True and at.sidebar.checkbox(key="frozen_check").value is False and at.sidebar.radio(key="iterations_radio").value == 200


def test_invalid_permalink_values_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["iterations"] = "123"
    at.query_params["alpha"] = "0.6"
    at.query_params["adaptive"] = "maybe"
    at.run()
    assert not at.exception and at.sidebar.radio(key="iterations_radio").value == C.DEFAULT_ITERATIONS and at.sidebar.radio(key="alpha_radio").value == C.DEFAULT_ALPHA
    assert at.sidebar.checkbox(key="adaptive_check").value is False


def test_experiments_run_on_demand(monkeypatch):
    import gp_evaluation as ev
    a_orig, l_orig, d_orig = ev.alpha_sweep, ev.load_series, ev.distribution
    monkeypatch.setattr(ev, "alpha_sweep", lambda params: a_orig(params, alphas=(0.5, 2.0), cap=30))
    monkeypatch.setattr(ev, "load_series", lambda params: l_orig(params, loads=(5, 15), seeds=C.LOAD_SEEDS[:1], cap=30))
    monkeypatch.setattr(ev, "distribution", lambda params: d_orig(params, seeds=C.SWEEP_SEEDS[:2], cap=30))
    at = _run(lambda a: (a.session_state.__setitem__("side_slider", 5), a.session_state.__setitem__("zones_slider", 4), a.session_state.__setitem__("iterations_radio", 50)))
    for key in ("alpha_start", "load_start", "dist_start", "unique_start"):
        next(b for b in at.button if b.key == key).click().run()
        assert not at.exception, key
    assert any(m.label == "Größte Abweichung der Kantenflüsse" for m in at.metric)


def test_experiments_need_the_grid_but_the_lesson_table_is_always_there():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "pigou"))
    assert not [b for b in at.button if b.key in ("alpha_start", "load_start", "dist_start", "unique_start")]
    assert any("Wege am Ende (Fluss)" in list(t.value.columns) for t in at.table)


def test_source_has_explicit_chart_keys_and_locked_axes():
    app = APP.read_text(encoding="utf-8")
    assert all(re.search(r"plotly_chart\(.*key=", line) for line in app.splitlines() if "st.plotly_chart(" in line)
    viz = (ROOT / "gp_visualization.py").read_text(encoding="utf-8")
    assert viz.count("return _base(fig") + viz.count("return _frame(fig") >= 8 and "def lock_axes" in viz
