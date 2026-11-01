"""DBD surface-damage diagnostics — web demo.

Run locally:
    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Upload oscilloscope captures (GW Instek GDS-2000 "Format 2.0A" CSV or plain
``t,v,i`` CSV) or generate a synthetic capture, and get the waveform, the
Lissajous figure, all electrical/plasma quantities (thesis definitions and
corrected ones), a baseline health check and, optionally, the prediction of a
model trained with ``dbd-diag train``.
"""
from __future__ import annotations

import io as _io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dbd_diagnostics import features, io, ml, plots, simulate

st.set_page_config(page_title="DBD diagnostics", page_icon="⚡", layout="wide")

REPO = "https://github.com/MJ-best/ms-thesis-dbd-plasma-diagnostics"
UNITS = {
    "frequency": ("Frequency", "kHz", 1e-3), "v_rms": ("V_rms", "V", 1), "i_rms": ("I_rms", "mA", 1e3),
    "energy": ("Energy per cycle", "mJ", 1e3), "power": ("Mean power", "W", 1),
    "c_dielectric": ("C_dielectric", "pF", 1e12), "c_total": ("C_total", "pF", 1e12), "c_gas": ("C_gas", "pF", 1e12),
    "q_max": ("Transferred charge ΔQ", "nC", 1e9), "v_max": ("V_max (peak applied)", "V", 1),
    "v_min": ("V_min (onset)", "V", 1), "u_b": ("Burning voltage U_b", "V", 1),
    "u_b_energy": ("U_b (energy method)", "V", 1), "v_breakdown": ("V_breakdown, thesis Eq.15 (≈ V_max)", "V", 1),
    "en_td": ("E/N with U_b", "Td", 1), "te_corrected": ("T_e with U_b", "eV", 1),
    "te": ("T_e, thesis chain (extrapolated)", "eV", 1), "n_discharges": ("Discharges per half cycle", "-", 1),
    "ne_corrected_mean": ("n_e (i_dis, U_b)", "cm⁻³", 1), "ne_mean": ("n_e, thesis chain", "cm⁻³", 1),
}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚡ DBD diagnostics")
st.sidebar.caption(f"Code & thesis: [{REPO.split('/')[-1]}]({REPO})")
source = st.sidebar.radio("Data / 데이터", ["Synthetic example / 합성 예제", "Upload CSV / 파일 업로드"])

wfs: dict[str, io.Waveform] = {}
if source.startswith("Synthetic"):
    st.sidebar.subheader("Equivalent-circuit parameters")
    v_amp = st.sidebar.slider("Applied voltage amplitude V_max [V]", 1500, 3000, 2240, 10)
    u_b = st.sidebar.slider("Gap breakdown voltage U_b [V]", 500, 1300, 900, 10)
    c_d = st.sidebar.slider("Dielectric capacitance C_d [pF]", 20.0, 50.0, 34.2, 0.1)
    c_g = st.sidebar.slider("Gas-gap capacitance C_g [pF]", 8.0, 30.0, 16.9, 0.1)
    model = st.sidebar.selectbox("Model", ["126-cell stochastic (realistic)", "ideal Manley (exact)"])
    seed = st.sidebar.number_input("Random seed", 0, 10_000, 0)
    p = simulate.CircuitParams(C_d=c_d * 1e-12, C_g=c_g * 1e-12, U_b=float(u_b), V_amp=float(v_amp))

    @st.cache_data(show_spinner="Simulating…")
    def _sim(kind, c_d, c_g, u_b, v_amp, seed):
        pp = simulate.CircuitParams(C_d=c_d, C_g=c_g, U_b=u_b, V_amp=v_amp)
        w = simulate.ideal(pp) if kind.startswith("ideal") else simulate.cells(pp, seed=int(seed), periods=2.3)
        return w.t, w.v, w.i

    t, v, i = _sim(model, p.C_d, p.C_g, p.U_b, p.V_amp, seed)
    wfs["synthetic"] = io.Waveform(t=t, v=v, i=i, meta={"source_file": "synthetic"})
else:
    files = st.sidebar.file_uploader("Oscilloscope CSV files", type=["csv"], accept_multiple_files=True)
    for f in files or []:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(f.getvalue())
        try:
            wfs[f.name] = io.load_csv(tmp.name)
        except Exception as exc:
            st.sidebar.error(f"{f.name}: {exc}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

st.sidebar.subheader("Device / 장치")
gap_um = st.sidebar.number_input("Electrode gap [µm]", 10.0, 2000.0, 100.0)
area_um2 = st.sidebar.number_input("Discharge cross-section [µm²]", 100.0, 1e6, 150.0 * 300.0)
peaks = st.sidebar.selectbox("Pulse detector", ["robust", "thesis"])
cfg = features.DeviceConfig(gap_cm=gap_um * 1e-4, cell_area_cm2=area_um2 * 1e-8, peak_method=peaks)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Electrical diagnostics of atmospheric-pressure DBD plasma")
st.caption("M.S. thesis, M. Kim (Kwangwoon Univ., 2023) — open re-implementation with corrected burning-voltage, "
           "E/N and T_e calculations. 대기압 DBD 플라즈마 전기적 진단 데모.")

if not wfs:
    st.info("Upload one or more oscilloscope CSV files in the sidebar, or choose the synthetic example. "
            "사이드바에서 CSV를 올리거나 합성 예제를 선택하세요.")
    st.stop()


@st.cache_data(show_spinner="Analysing…")
def _extract(name, t, v, i, gap_cm, area_cm2, peak_method):
    c = features.DeviceConfig(gap_cm=gap_cm, cell_area_cm2=area_cm2, peak_method=peak_method)
    return features.extract(io.Waveform(t=t, v=v, i=i, meta={"source_file": name}), c)


rows = {}
for name, wf in wfs.items():
    try:
        rows[name] = _extract(name, wf.t, wf.v, wf.i, cfg.gap_cm, cfg.cell_area_cm2, cfg.peak_method)
    except Exception as exc:
        st.error(f"{name}: analysis failed ({exc})")
if not rows:
    st.stop()
table = pd.DataFrame(rows).T

selected = st.selectbox("Capture to display / 표시할 캡처", list(rows)) if len(rows) > 1 else next(iter(rows))
r = rows[selected]
wf = wfs[selected]

c1, c2, c3, c4 = st.columns(4)
c1.metric("V_rms", f"{r['v_rms']:.0f} V")
c2.metric("Mean power", f"{r['power']:.2f} W", help=f"energy per cycle {r['energy'] * 1e3:.3f} mJ")
c3.metric("Burning voltage U_b", f"{r['u_b']:.0f} V",
          help=f"thesis Eq.15 gives {r['v_breakdown']:.0f} V ≈ V_max = {r['v_max']:.0f} V")
c4.metric("T_e (U_b)", "–" if np.isnan(r["te_corrected"]) else f"{r['te_corrected']:.1f} eV",
          help=f"E/N = {r['en_td']:.0f} Td; thesis chain (extrapolated): {r['te']:.1f} eV")

tab_w, tab_l, tab_f, tab_h, tab_m = st.tabs(
    ["Waveform", "Lissajous", "All quantities", "Health check / 상태 점검", "Model prediction / 모델 예측"])

with tab_w:
    st.pyplot(plots.waveform_figure(wf))
    st.caption("Blue: gas-gap voltage u_g = v − Q/C_d. During discharge it is clamped at ±U_b (dotted).")

with tab_l:
    st.pyplot(plots.lissajous_figure(wf))
    st.caption("The thesis formula V_min + Q_max/C_dielectric equals the peak applied voltage, because "
               "Q_max = C_dielectric (V_max − V_min). The burning voltage U_b is the voltage across the gas.")

with tab_f:
    disp = []
    for key, (label, unit, scale) in UNITS.items():
        if key in table:
            vals = table[key].astype(float) * scale
            disp.append({"quantity": label, "unit": unit, **{k: v for k, v in vals.items()}})
    st.dataframe(pd.DataFrame(disp).set_index("quantity"), width="stretch")
    buf = _io.StringIO()
    table.to_csv(buf)
    st.download_button("Download all features (CSV)", buf.getvalue(), "features.csv", "text/csv")

with tab_h:
    st.markdown(
        "Compare with the same device's **healthy baseline** (e.g. the first hours of operation). "
        "In the thesis data the strongest damage indicator was the drop of the applied RMS voltage "
        "(normal 1489 ± 10 V vs degraded 1396 ± 72 V; single-feature AUC ≈ 1.0 on two devices). "
        "Thresholds below are heuristics — calibrate them on your own devices.")
    b1, b2, b3 = st.columns(3)
    base_v = b1.number_input("Baseline V_rms [V]", 100.0, 10000.0, float(round(r["v_rms"], 0)))
    base_cd = b2.number_input("Baseline C_dielectric [pF]", 1.0, 1000.0, float(round(r["c_dielectric"] * 1e12, 1)))
    base_ub = b3.number_input("Baseline U_b [V]", 10.0, 10000.0, float(round(r["u_b"], 0)))
    thr = st.slider("Alarm threshold for the V_rms drop [%]", 1.0, 10.0, 3.0, 0.5)
    dv = 100 * (r["v_rms"] - base_v) / base_v
    dcd = 100 * (r["c_dielectric"] * 1e12 - base_cd) / base_cd
    dub = 100 * (r["u_b"] - base_ub) / base_ub
    h1, h2, h3 = st.columns(3)
    h1.metric("ΔV_rms", f"{dv:+.1f} %")
    h2.metric("ΔC_dielectric", f"{dcd:+.1f} %")
    h3.metric("ΔU_b", f"{dub:+.1f} %")
    if dv <= -thr:
        st.error(f"V_rms dropped by {-dv:.1f} % (≥ {thr} %): consistent with dielectric surface damage. "
                 "전압 실효값이 기준 대비 감소했습니다 — 표면 손상 가능성.")
    else:
        st.success("No significant V_rms drop relative to the baseline. 기준 대비 유의한 감소 없음.")

with tab_m:
    st.markdown("Load a model trained on **your own** labelled captures with "
                "`dbd-diag extract … -o features.csv` → add `label`/`device` columns → "
                "`dbd-diag train features.csv -o model.joblib`.")
    st.warning("Model files are pickles: only load files you created yourself or fully trust. "
               "직접 만든(신뢰할 수 있는) 모델 파일만 불러오세요.")
    up = st.file_uploader("model.joblib", type=["joblib", "pkl"])
    if up is not None:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp.write(up.getvalue())
        try:
            bundle = ml.load_bundle(tmp.name)
            pred = ml.predict_bundle(bundle, table.astype({k: float for k in bundle["features"]}))
            pred["state"] = pred["predicted_label"].map({0: "normal / 정상", 1: "degraded / 손상"})
            st.write(f"Model: **{bundle.get('model_name')}**, trained on {bundle.get('n_train')} rows "
                     f"{bundle.get('class_counts')}")
            st.dataframe(pred, width="stretch")
        except Exception as exc:
            st.error(f"Could not use this model: {exc}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

st.divider()
st.caption(f"Please cite: M. Kim, M.S. thesis, Kwangwoon University (2023). Code MIT · docs CC BY 4.0 · {REPO}")
