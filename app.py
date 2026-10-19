"""
Hispanic Heritage Historical Data Explorer
A classroom Streamlit app for World Bank historical data, polynomial modeling,
and an interactive function/calculus lab.

Data source: World Bank Indicators API. Polynomial predictions are mathematical
fits, not causal forecasts. All extrapolations are uncertain.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns persisted."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "GDP divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people, net", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in consumer prices."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "dollars", "time", "days", "A business revenue or cost model."),
    "Engineering": ("Displacement", "meters", "time", "seconds", "Motion or engineered-system response."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "A simplified concentration-over-time model; not medical advice."),
    "Sports science": ("Distance", "meters", "time", "seconds", "A movement or training-performance model."),
    "Biology": ("Population", "organisms", "time", "days", "A population or cell-colony model."),
    "Bioengineering": ("Cell yield", "cells/mL", "time", "hours", "A simplified cell-growth or bioprocess model."),
    "Economic forecasting": ("Index", "index points", "time", "years", "A simplified economic indicator model."),
    "Weather": ("Temperature", "°C", "time", "hours", "A simplified temperature-over-time model."),
    "Custom": ("Output", "units", "input", "units", "A user-defined function and interpretation."),
}
SUGGESTIONS = {
    "Business": "a*x**2 + b*x + c", "Engineering": "a*sin(b*x) + c",
    "Medical": "a*exp(-b*x) + c", "Sports science": "a*x*(T-x)",
    "Biology": "K/(1 + a*exp(-b*x))", "Bioengineering": "a*exp(b*x)",
    "Economic forecasting": "a*x**2 + b*x + c", "Weather": "a*cos(b*x) + c",
    "Custom": "a*x**2 + b*x + c",
}
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"

def wb_fetch(country: str, indicator: str, start: int, end: int) -> tuple[pd.DataFrame, str]:
    """Fetch and validate World Bank observations; no synthetic fallback."""
    url = WB_URL.format(country=country, indicator=indicator, start=start, end=end)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observation table for this selection."
        rows = []
        for item in payload[1]:
            if item and item.get("date") and item.get("value") is not None:
                try:
                    rows.append({"year": int(item["date"]), "value": float(item["value"])})
                except (TypeError, ValueError):
                    continue
        frame = pd.DataFrame(rows, columns=["year", "value"]).drop_duplicates("year").sort_values("year")
        return frame, ("" if not frame.empty else "The API returned no non-null observations in the selected period.")
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Check your connection and try again."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not read the API response: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country: str, indicator: str, start: int, end: int):
    frame, error = wb_fetch(country, indicator, start, end)
    return frame, error

def valid_observations(frame: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    problems = []
    if not {"year", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    out = frame.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out["year"].isna().any() or not np.all(np.isclose(out["year"].dropna(), np.round(out["year"].dropna()))):
        problems.append("Years must be whole-number integers.")
    out = out.dropna(subset=["year", "value"])
    out["year"] = out["year"].astype(int)
    if out["year"].duplicated().any():
        problems.append("Duplicate years found. Keep one row per year.")
    out = out.drop_duplicates("year").sort_values("year")
    if not np.isfinite(out["value"]).all():
        problems.append("Values must be finite numbers.")
        out = out[np.isfinite(out["value"])]
    if indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate", "Total reserves including gold"):
        if indicator_name == "Life expectancy at birth" and ((out.value < 0) | (out.value > 125)).any():
            problems.append("Life expectancy values should be between 0 and 125 years.")
        if indicator_name in ("Unemployment, national estimate",) and ((out.value < 0) | (out.value > 100)).any():
            problems.append("Unemployment percentage should be between 0 and 100.")
        if indicator_name in ("Crude birth rate", "Intentional homicide rate", "Total reserves including gold") and (out.value < 0).any():
            problems.append("This indicator should not have negative values.")
    return out, problems

def polynomial_fit(frame: pd.DataFrame, degree: int):
    clean = frame.dropna(subset=["year", "value"]).sort_values("year")
    if clean.year.nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; this selection has {clean.year.nunique()}.")
    base = int(clean.year.min())
    t = clean.year.to_numpy(dtype=float) - base
    vals = clean.value.to_numpy(dtype=float)
    coeff = np.polynomial.polynomial.polyfit(t, vals, degree)
    pred = np.polynomial.polynomial.polyval(t, coeff)
    r2 = float(r2_score(vals, pred)) if len(vals) > 1 else float("nan")
    mae = float(mean_absolute_error(vals, pred))
    return base, coeff, pred, r2, mae, clean

def poly_text(coeff, base, unit):
    pieces = []
    for power, value in enumerate(coeff):
        if abs(value) < 1e-12: continue
        term = f"{value:.8g}"
        if power == 1: term += f"(year − {base})"
        elif power > 1: term += f"(year − {base})^{power}"
        pieces.append(term)
    return "y = " + (" + ".join(pieces) if pieces else "0") + f"  [{unit}]"

def calculate_poly(year, base, coeff):
    return float(np.polynomial.polynomial.polyval(float(year)-base, coeff))

def make_chart(frame, base, coeff, degree, sample_step, query_years=None, indicator_name="", unit=""):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if not frame.empty:
        ax.scatter(frame.year, frame.value, s=24, label="Observed data", zorder=3)
        lo, hi = int(frame.year.min()), int(frame.year.max())
        smooth = np.linspace(lo, hi + 1, 500)
        ax.plot(smooth, [calculate_poly(y, base, coeff) for y in smooth], label=f"Degree {degree} fit (historical range)")
        sample = np.arange(lo, hi + 1, sample_step)
        ax.scatter(sample, [calculate_poly(y, base, coeff) for y in sample], marker="x", s=35, label=f"Model samples every {sample_step} year(s)")
        ax.axvline(hi, linestyle="--", linewidth=1, label="Historical boundary")
        if query_years:
            q = np.array(query_years, dtype=float)
            inside = q[(q >= lo) & (q <= hi)]
            outside = q[(q < lo) | (q > hi)]
            if len(inside): ax.scatter(inside, [calculate_poly(y, base, coeff) for y in inside], marker="D", s=45, label="Calculated within range")
            if len(outside):
                start = max(hi, min(outside))
                end = max(outside)
                if end > hi:
                    ex = np.linspace(hi, end, 120)
                    ax.plot(ex, [calculate_poly(y, base, coeff) for y in ex], linestyle=":", label="Extrapolated segment")
                ax.scatter(outside, [calculate_poly(y, base, coeff) for y in outside], marker="^", s=45, label="Calculated outside range")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({unit})")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def parse_expression(expression: str, params: dict):
    x = sp.symbols("x", real=True)
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
               "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "abs": sp.Abs}
    allowed.update({k: sp.Float(v) for k, v in params.items()})
    try:
        expr = sp.sympify(expression, locals=allowed, evaluate=True)
        if expr.has(sp.Function) and any(str(f.func) not in {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs"} for f in expr.atoms(sp.Function)):
            raise ValueError("Use only supported functions: sin, cos, tan, exp, log, sqrt, abs.")
        if expr.free_symbols - {x}:
            raise ValueError("Expression contains an unknown variable or parameter.")
        return x, expr
    except Exception as exc:
        raise ValueError(f"Could not parse function. Use syntax such as x**2, sin(x), exp(-x), or a*x+b. Details: {exc}")

def numeric_eval(expr, x_symbol, xs):
    f = sp.lambdify(x_symbol, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        vals = np.asarray(f(xs), dtype=float)
    if vals.ndim == 0: vals = np.full(np.asarray(xs).shape, float(vals))
    return np.where(np.isfinite(vals), vals, np.nan)

def make_function_plot(expr, d1, d2, x_symbol, xlo, xhi, current_x, a, b, show_f, show_d1, show_d2, tangent=True):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.linspace(xlo, xhi, 900)
    if show_f: ax.plot(xs, numeric_eval(expr, x_symbol, xs), label="f(x)")
    if show_d1: ax.plot(xs, numeric_eval(d1, x_symbol, xs), label="f′(x)")
    if show_d2: ax.plot(xs, numeric_eval(d2, x_symbol, xs), label="f″(x)")
    try:
        y0 = float(sp.N(expr.subs(x_symbol, current_x)))
        slope = float(sp.N(d1.subs(x_symbol, current_x)))
        if show_f and math.isfinite(y0): ax.scatter([current_x], [y0], marker="o", s=45, label="Current point")
        if tangent and show_f and math.isfinite(slope) and math.isfinite(y0):
            tx = np.linspace(max(xlo, current_x-(xhi-xlo)*.18), min(xhi, current_x+(xhi-xlo)*.18), 100)
            ax.plot(tx, y0+slope*(tx-current_x), linestyle="--", label="Tangent at current x")
        ya, yb = float(sp.N(expr.subs(x_symbol, a))), float(sp.N(expr.subs(x_symbol, b)))
        if math.isfinite(ya) and math.isfinite(yb) and a != b:
            sec_x = np.linspace(min(a,b), max(a,b), 100)
            ax.plot(sec_x, ya+(yb-ya)/(b-a)*(sec_x-a), linestyle=":", label="Secant over interval")
    except Exception:
        pass
    ax.axhline(0, color="gray", linewidth=.6); ax.axvline(0, color="gray", linewidth=.6)
    ax.set_xlabel("x"); ax.set_ylabel("Function / derivative value")
    ax.grid(True, alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    return fig

def approximate_analysis(expr, x, xlo, xhi):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    critical, inflections = [], []
    try:
        roots = sp.solve(sp.Eq(d1, 0), x)
        for r in roots:
            if r.is_real is False: continue
            rv = float(sp.N(r))
            if xlo <= rv <= xhi:
                eps = max(1e-5, (xhi-xlo)*1e-5)
                left = float(sp.N(d1.subs(x, rv-eps))); right = float(sp.N(d1.subs(x, rv+eps)))
                kind = "local minimum" if left < 0 < right else ("local maximum" if left > 0 > right else "stationary point; no sign-change classification")
                critical.append((rv, float(sp.N(expr.subs(x, rv))), kind))
    except Exception:
        pass
    try:
        roots2 = sp.solve(sp.Eq(d2, 0), x)
        for r in roots2:
            if r.is_real is False: continue
            rv = float(sp.N(r))
            if xlo <= rv <= xhi:
                eps = max(1e-5, (xhi-xlo)*1e-5)
                left = float(sp.N(d2.subs(x, rv-eps))); right = float(sp.N(d2.subs(x, rv+eps)))
                if left*right < 0:
                    inflections.append((rv, float(sp.N(expr.subs(x, rv)))))
    except Exception:
        pass
    return d1, d2, critical, inflections

st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • Authentic World Bank indicators + mathematical modeling")
st.info("Brazil is included as a Latin American comparison country. Its inclusion is geographic and comparative; it is not a Spanish-speaking country.")
tab_data, tab_lab, tab_help = st.tabs(["Historical Data Explorer", "Function & Calculus Lab", "Guide & Verification"])

with tab_data:
    st.subheader("1. Retrieve historical indicator data")
    c1, c2, c3 = st.columns(3)
    with c1:
        country_label = st.selectbox("Country", list(COUNTRIES), key="country")
    with c2:
        indicator_name = st.selectbox("Indicator", list(INDICATORS), key="indicator")
    with c3:
        degree = st.slider("Polynomial degree", 3, 8, 3, key="degree")
    now_year = date.today().year
    start_year, end_year = now_year-70, now_year-1
    code, unit, definition = INDICATORS[indicator_name]
    st.markdown(f"**Definition:** {definition}  \n**Unit:** {unit}  \n**World Bank indicator:** `{code}`")
    st.markdown(f"**Requested years:** {start_year}–{end_year} (current year − 70 through current year − 1).")
    country_code = COUNTRIES[country_label]
    with st.spinner("Requesting official World Bank data…"):
        source_frame, fetch_error = cached_wb_fetch(country_code, code, start_year, end_year)
    if fetch_error:
        st.warning(fetch_error)
    original = source_frame.copy()
    if not original.empty:
        st.session_state["source_data_key"] = (country_code, code, start_year, end_year)
        st.session_state["source_data"] = original.copy()
    source_key = (country_code, code, start_year, end_year)
    if "source_data_key" not in st.session_state or st.session_state.get("source_data_key") != source_key:
        original = source_frame.copy()
    st.caption(f"Source: World Bank API • Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')} • "
               f"API: https://api.worldbank.org/v2/country/{country_code}/indicator/{code}?date={start_year}:{end_year}&format=json")
    if not original.empty:
        st.write(f"**Actual coverage:** {int(original.year.min())}–{int(original.year.max())} • **Non-null observations:** {len(original)} of {end_year-start_year+1} requested calendar years.")
    else:
        st.write("**Actual coverage:** no observations retrieved. No replacement or synthetic data is generated.")
    st.markdown("#### Editable working table")
    if "editor_key" not in st.session_state: st.session_state.editor_key = 0
    if "working_key" not in st.session_state: st.session_state.working_key = None
    if st.session_state.working_key != source_key:
        st.session_state.working_data = original.copy()
        st.session_state.working_key = source_key
        st.session_state.editor_key += 1
    if original.empty:
        working_input = pd.DataFrame({"year": pd.Series(dtype=int), "value": pd.Series(dtype=float)})
    else:
        working_input = st.session_state.working_data
    edited = st.data_editor(working_input, num_rows="dynamic", use_container_width=True, key=f"data_editor_{st.session_state.editor_key}",
                            column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"), "value": st.column_config.NumberColumn("Value", format="%.8g")})
    btn1, btn2 = st.columns([1, 5])
    with btn1:
        reset = st.button("Reset to source", key="reset_source")
    with btn2:
        st.caption("Edit year/value rows for a scenario. Edited values are not official source data.")
    if reset:
        st.session_state.working_data = original.copy()
        st.session_state.editor_key += 1
        st.rerun()
    clean, problems = valid_observations(edited, indicator_name)
    modified = not clean.reset_index(drop=True).equals(original.reset_index(drop=True))
    if modified: st.warning("USER-MODIFIED SCENARIO: analysis and exports use the editable table, not only the original API data.")
    for problem in problems: st.error(problem)
    st.session_state.working_data = edited.copy()
    sample_step = st.slider("Model sample increment (years)", 1, 10, 5, key="sample_step")
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10, key="horizon")
    compare_label = st.selectbox("Comparison country (same indicator)", ["None"] + [x for x in COUNTRIES if x != country_label], key="compare")
    printer = st.checkbox("Printer-friendly report view", key="printer")
    fit_ok = False
    if len(clean) >= degree+1 and not problems:
        try:
            base, coeff, pred, r2, mae, fit_frame = polynomial_fit(clean, degree)
            fit_ok = True
        except Exception as exc: st.error(str(exc))
    else:
        st.warning(f"Need at least {degree+1} valid distinct years for degree {degree}; currently {clean.year.nunique() if 'year' in clean else 0}.")
    calculated = pd.DataFrame(columns=["year", "estimate", "classification"])
    if fit_ok:
        st.subheader("2. Polynomial model")
        st.code(poly_text(coeff, base, unit))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("In-sample R²", f"{r2:.5f}")
        m2.metric("In-sample MAE", f"{mae:.5g} {unit}")
        m3.metric("Observations", str(len(fit_frame)))
        m4.metric("Mean observed value", f"{fit_frame.value.mean():.5g} {unit}")
        st.caption(f"Centered time: t = year − {base}. Historical range: {fit_frame.year.min()}–{fit_frame.year.max()}. Fit scores describe the data used to fit the model, not future accuracy.")
        residuals = fit_frame[["year", "value"]].copy()
        residuals["fitted"] = pred
        residuals["residual (actual − fitted)"] = residuals.value - residuals.fitted
        with st.expander("Residuals and chronological holdout"):
            st.dataframe(residuals, use_container_width=True)
            if len(fit_frame) >= max(2*(degree+1), 10):
                split = max(degree+1, int(len(fit_frame)*.8))
                train, test = fit_frame.iloc[:split], fit_frame.iloc[split:]
                if len(test) and train.year.nunique() >= degree+1:
                    hb, hc, *_ = polynomial_fit(train, degree)
                    hp = np.array([calculate_poly(y, hb, hc) for y in test.year])
                    st.write(f"**Chronological holdout MAE:** {mean_absolute_error(test.value, hp):.5g} {unit} across {len(test)} later observations. This score uses a separate earlier-training/later-test split.")
                else: st.write("Not enough later observations for a chronological holdout score.")
            else: st.write("Not enough observations for a meaningful chronological holdout split.")
        st.subheader("3. Calculate a year or batch")
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(int(fit_frame.year.max()),1950),2100), step=1, key="single_year")
        if st.button("Calculate Value", key="calculate_single"):
            val = calculate_poly(single_year, base, coeff)
            st.session_state.single_result = (source_key, degree, modified, int(single_year), val)
        single_result = st.session_state.get("single_result")
        if single_result and single_result[0] == source_key and single_result[1] == degree and single_result[2] == modified:
            _, _, _, yr, val = single_result
            range_label = "within fitted historical range" if fit_frame.year.min() <= yr <= fit_frame.year.max() else "outside fitted historical range (extrapolation)"
            actual = fit_frame.loc[fit_frame.year == yr, "value"]
            st.success(f"{yr}: estimated {val:.8g} {unit} • {range_label}.")
            if len(actual): st.caption(f"An actual observation exists for this year: {actual.iloc[0]:.8g} {unit}. The model estimate is not the actual observation.")
        batch_mode = st.radio("Batch input", ["Year range", "Comma-separated custom years"], horizontal=True, key="batch_mode")
        years = []
        if batch_mode == "Year range":
            br1, br2, br3 = st.columns(3)
            with br1: yr_start = st.number_input("Start year", 1950, 2100, max(1950, int(fit_frame.year.max())), key="batch_start")
            with br2: yr_end = st.number_input("End year", 1950, 2100, min(2100, max(1950, int(fit_frame.year.max())+10)), key="batch_end")
            with br3: yr_step = st.slider("Range step", 1, 10, 1, key="batch_step")
            if yr_end < yr_start: st.error("End year must be greater than or equal to start year.")
            else: years = list(range(int(yr_start), int(yr_end)+1, int(yr_step)))
        else:
            raw_years = st.text_input("Years separated by commas", "2000, 2010, 2020, 2030", key="custom_years")
            try:
                years = [int(x.strip()) for x in raw_years.split(",") if x.strip()]
                if any(str(y) != x.strip() and not (x.strip().startswith("+") and int(x.strip()) == y) for y, x in zip(years, [x for x in raw_years.split(",") if x.strip()])): raise ValueError("Use integer years.")
                if any(y < 1950 or y > 2100 for y in years): raise ValueError("Each year must be from 1950 through 2100.")
                if len(set(years)) != len(years): st.warning("Repeated years were requested; duplicate rows will be shown.")
            except Exception as exc:
                st.error(f"Invalid custom year list: {exc}"); years = []
        if years:
            calculated = pd.DataFrame({"year": years, "estimate": [calculate_poly(y, base, coeff) for y in years]})
            lo, hi = int(fit_frame.year.min()), int(fit_frame.year.max())
            calculated["classification"] = np.where(calculated.year.between(lo, hi), "within fitted range", "outside fitted range")
            st.dataframe(calculated, use_container_width=True)
            if ((calculated.estimate.abs() > max(1, clean.value.abs().max())*100) | (calculated.estimate < 0)).any():
                st.warning("Some estimates may be implausible for this indicator. Predictions are not silently clipped; inspect the graph and use caution with extrapolation.")
            st.session_state.batch_result = (source_key, degree, modified, calculated.copy())
        else:
            st.session_state.batch_result = (source_key, degree, modified, calculated.copy())
        query_years = list(calculated.year) if not calculated.empty else ([single_result[3]] if single_result and single_result[0] == source_key else [])
        if horizon > 0:
            future = list(range(int(fit_frame.year.max())+1, int(fit_frame.year.max())+horizon+1))
            query_years = list(dict.fromkeys(query_years + future))
        st.subheader("4. Historical visualization")
        fig = make_chart(fit_frame, base, coeff, degree, sample_step, query_years, indicator_name, unit)
        if compare_label != "None":
            compare_code = COUNTRIES[compare_label]
            compare_frame, compare_error = cached_wb_fetch(compare_code, code, start_year, end_year)
            if compare_error: st.warning(f"{compare_label}: {compare_error}")
            elif not compare_frame.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 5.2))
                ax2.plot(fit_frame.year, fit_frame.value, marker="o", markersize=3, label=country_label)
                ax2.plot(compare_frame.year, compare_frame.value, marker="o", markersize=3, label=compare_label)
                ax2.set_xlabel("Year"); ax2.set_ylabel(f"{indicator_name} ({unit})")
                ax2.grid(True, alpha=.25); ax2.legend(); fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                st.caption(f"Comparison uses each country's available observations. Different gaps or coverage may make direct year-by-year comparison incomplete.")
                plt.close(fig2)
        st.pyplot(fig, use_container_width=True)
        st.caption("Observed points are official API observations; the curve is a fitted polynomial. The dashed boundary marks the last fitted year. Extrapolation is mathematical, not a validated forecast.")
        plt.close(fig)
        st.subheader("5. Course-level interpretation")
        yvals = np.array([calculate_poly(y, base, coeff) for y in np.linspace(fit_frame.year.min(), fit_frame.year.max(), 200)])
        st.write(f"**Domain used for the historical fit:** {int(fit_frame.year.min())}–{int(fit_frame.year.max())}. The polynomial is mathematically defined for all real years, but historical evidence only covers the displayed range.")
        st.write(f"**Range over the sampled historical interval (approximate):** {np.nanmin(yvals):.6g} to {np.nanmax(yvals):.6g} {unit}.")
        if degree % 2: st.write(f"**End behavior:** degree {degree} is odd; the ends go in opposite directions, determined by the leading coefficient {coeff[-1]:.5g}.")
        else: st.write(f"**End behavior:** degree {degree} is even; both ends go in the same direction, determined by the leading coefficient {coeff[-1]:.5g}.")
        avg_rate = (calculate_poly(fit_frame.year.max(), base, coeff)-calculate_poly(fit_frame.year.min(), base, coeff))/(fit_frame.year.max()-fit_frame.year.min())
        st.write(f"**Average rate of change across the fitted interval:** {avg_rate:.6g} {unit} per year.")
        if degree >= 4:
            st.write("Higher-degree terms can introduce additional turning points and rapid changes near or beyond the observed years; they can fit historical patterns while extrapolating poorly.")
        if degree >= 5:
            st.write("A polynomial's curvature is described by its second derivative; changing average rates can indicate changes in concavity.")
        if degree >= 3:
            st.write("**Model limitation:** the fitted polynomial describes association in the selected data; it does not explain historical causes.")
        # Export current selection and model
        st.subheader("6. Export")
        export_df = clean.copy()
        export_df["fitted"] = [calculate_poly(y, base, coeff) for y in export_df.year]
        export_df["residual"] = export_df.value-export_df.fitted
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download working data + fit (CSV)", csv_bytes, file_name="historical_model.csv", mime="text/csv")
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            pd.DataFrame([{"country": country_label, "country_code": country_code, "indicator": indicator_name, "indicator_code": code, "unit": unit,
                           "definition": definition, "source": "World Bank API", "retrieved": datetime.now().isoformat(timespec="minutes"),
                           "modified_scenario": modified, "degree": degree, "base_year": base, "r_squared_in_sample": r2, "mae_in_sample": mae,
                           "equation": poly_text(coeff, base, unit)}]).to_excel(writer, sheet_name="Summary", index=False)
            export_df.to_excel(writer, sheet_name="Working data", index=False)
            calculated.to_excel(writer, sheet_name="Calculated years", index=False)
        st.download_button("Download Excel workbook (.xlsx)", excel_buf.getvalue(), file_name="historical_model.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf_buf = io.BytesIO()
        pdf_doc = SimpleDocTemplate(pdf_buf, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
                 Paragraph(f"{country_label} • {indicator_name} ({code}) • Unit: {unit}", styles["Heading2"]),
                 Paragraph(f"Definition: {definition}", styles["BodyText"]),
                 Paragraph(f"Retrieved {datetime.now().strftime('%Y-%m-%d')} • Modified scenario: {modified}", styles["BodyText"]),
                 Paragraph(f"Coverage: {fit_frame.year.min()}–{fit_frame.year.max()} • n={len(fit_frame)} • Degree={degree} • Base year={base}", styles["BodyText"]),
                 Paragraph(poly_text(coeff, base, unit).replace("&","&amp;"), styles["BodyText"]),
                 Paragraph(f"In-sample R²={r2:.6g}; in-sample MAE={mae:.6g} {unit}. These are not future-accuracy guarantees.", styles["BodyText"]),
                 Spacer(1, 8)]
        pdf_chart = make_chart(fit_frame, base, coeff, degree, sample_step, query_years, indicator_name, unit)
        img_buf = io.BytesIO(); pdf_chart.savefig(img_buf, format="png", dpi=130, bbox_inches="tight"); plt.close(pdf_chart); img_buf.seek(0)
        story.append(RLImage(img_buf, width=650, height=330))
        story.append(Paragraph("Limitations: polynomial fits can be sensitive to degree and extrapolation. No causal explanation is inferred. API gaps remain gaps.", styles["BodyText"]))
        preview = export_df.head(15).round(5).values.tolist()
        preview.insert(0, list(export_df.columns))
        tbl = Table(preview, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story += [Spacer(1, 8), Paragraph("Data preview (first 15 rows)", styles["Heading2"]), tbl]
        if not calculated.empty:
            results = calculated.round(6).values.tolist(); results.insert(0, list(calculated.columns))
            story += [Spacer(1, 8), Paragraph("Calculated years", styles["Heading2"]), Table(results, repeatRows=1)]
        pdf_doc.build(story)
        st.download_button("Download PDF report", pdf_buf.getvalue(), file_name="historical_model.pdf", mime="application/pdf")
    if printer:
        st.markdown("---")
        st.markdown("### Printer-friendly summary")
        st.write(f"{country_label} — {indicator_name}. {definition}")
        st.write(f"Unit: {unit}. Source: World Bank API. Retrieved: {date.today().isoformat()}.")
        if fit_ok: st.write(f"Degree {degree}; equation: {poly_text(coeff, base, unit)}; in-sample R²={r2:.5f}; MAE={mae:.5g} {unit}.")
        st.write("Use caution with polynomial extrapolation. Estimates outside the observed range are not validated forecasts.")

with tab_lab:
    st.subheader("Function & Calculus Lab")
    st.write("Choose a context, use a suggested or data-derived function, adjust parameters, or type a function. Input uses Python-style mathematical notation (`x**2`, `sin(x)`, `exp(-x)`, `log(x)`).")
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], key="course")
    context = st.selectbox("Context", list(CONTEXTS), key="context")
    dep_default, dep_unit_default, indep_default, indep_unit_default, desc_default = CONTEXTS[context]
    desc = st.text_input("Context description", desc_default, key="context_desc")
    vars1, vars2, vars3, vars4 = st.columns(4)
    with vars1: dep_name = st.text_input("Dependent variable", dep_default, key="dep_name")
    with vars2: dep_unit = st.text_input("Dependent units", dep_unit_default, key="dep_unit")
    with vars3: indep_name = st.text_input("Independent variable", indep_default, key="indep_name")
    with vars4: indep_unit = st.text_input("Independent units", indep_unit_default, key="indep_unit")
    fmode = st.radio("Function source", ["Suggested function", "Historical-data polynomial", "Custom expression"], horizontal=True, key="fmode")
    a_param = st.number_input("Parameter a", value=2.0, key="param_a")
    b_param = st.number_input("Parameter b", value=0.25, key="param_b")
    c_param = st.number_input("Parameter c", value=1.0, key="param_c")
    if fmode == "Suggested function":
        expression = st.text_input("Function expression", SUGGESTIONS[context], key="suggested_expr")
    elif fmode == "Historical-data polynomial":
        # Re-fetch default context selection if historical tab hasn't been used.
        lab_country = st.selectbox("Country for data-derived model", list(COUNTRIES), key="lab_country")
        lab_indicator = st.selectbox("Indicator for data-derived model", list(INDICATORS), key="lab_indicator")
        lab_degree = st.slider("Data-derived polynomial degree", 1, 8, 3, key="lab_degree")
        lab_code, lab_unit, lab_def = INDICATORS[lab_indicator]
        lab_data, lab_error = cached_wb_fetch(COUNTRIES[lab_country], lab_code, date.today().year-70, date.today().year-1)
        if lab_error: st.warning(lab_error)
        if len(lab_data) >= lab_degree+1:
            try:
                lab_base, lab_coeff, _, _, _, _ = polynomial_fit(lab_data, lab_degree)
                expression = " + ".join([f"({v:.12g})*(x-{lab_base})**{i}" for i,v in enumerate(lab_coeff)])
                st.caption(f"Using World Bank data-derived polynomial; values are in {lab_unit}.")
            except Exception as exc: st.error(str(exc)); expression = "x"
        else:
            st.warning("Not enough valid source observations for this degree."); expression = "x"
    else:
        expression = st.text_input("Your function f(x)", "a*x**2 + b*x + c", key="custom_expr")
    params = {"a": a_param, "b": b_param, "c": c_param}
    xlo, xhi = st.columns(2)
    with xlo: xmin = st.number_input("Analysis interval minimum", value=-5.0, key="xmin")
    with xhi: xmax = st.number_input("Analysis interval maximum", value=5.0, key="xmax")
    if xmax <= xmin:
        st.error("The interval maximum must be greater than the minimum.")
    else:
        try:
            x, expr = parse_expression(expression, params)
            d1, d2, critical, inflections = approximate_analysis(expr, x, xmin, xmax)
            st.markdown("#### Function and derivatives")
            st.latex("f(x) = " + sp.latex(expr))
            if course == "Calculus":
                st.latex("f'(x) = " + sp.latex(d1))
                st.latex("f''(x) = " + sp.latex(d2))
            elif course == "Precalculus":
                st.write("Precalculus focus: degree and coefficients shape the graph; turning points and changing average rates describe trends and curvature.")
                st.latex("f'(x) = " + sp.latex(d1))
            else:
                st.write("Honors Algebra 2 focus: domain, range, end behavior, graph trends, and average rate of change.")
            st.markdown("#### Synchronized graph controls")
            current_x = st.slider("Current input x", float(xmin), float(xmax), float((xmin+xmax)/2), key="current_x")
            colA, colB = st.columns(2)
            with colA: left_x = st.slider("Average-rate interval start", float(xmin), float(xmax), float(xmin), key="left_x")
            with colB: right_x = st.slider("Average-rate interval end", float(xmin), float(xmax), float(xmax), key="right_x")
            s1, s2, s3 = st.columns(3)
            with s1: show_f = st.checkbox("Show f(x)", True, key="show_f")
            with s2: show_d1 = st.checkbox("Show f′(x)", course == "Calculus", key="show_d1")
            with s3: show_d2 = st.checkbox("Show f″(x)", False, key="show_d2")
            fig = make_function_plot(expr, d1, d2, x, xmin, xmax, current_x, left_x, right_x, show_f, show_d1, show_d2)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            try:
                fval = float(sp.N(expr.subs(x, current_x)))
                slope = float(sp.N(d1.subs(x, current_x)))
                second = float(sp.N(d2.subs(x, current_x)))
                st.metric(f"{dep_name} at x={current_x:.4g}", f"{fval:.6g} {dep_unit}")
                st.write(f"**First derivative:** {slope:.6g} {dep_unit} per {indep_unit}. At this input, the dependent variable is changing at this instantaneous rate.")
                st.write(f"**Second derivative:** {second:.6g} {dep_unit} per {indep_unit}². It describes how the rate itself changes.")
                if context == "Engineering":
                    st.caption("If the function is displacement over time, its first derivative is velocity and its second derivative is acceleration. This interpretation only applies when x represents time and f represents position.")
                if context in ("Business", "Economic forecasting"):
                    st.caption("A positive first derivative means the modeled quantity is increasing; a negative one means it is decreasing. The second derivative indicates whether that rate is increasing or decreasing.")
                if right_x != left_x:
                    fa = float(sp.N(expr.subs(x, left_x))); fb = float(sp.N(expr.subs(x, right_x)))
                    avg = (fb-fa)/(right_x-left_x)
                    st.write(f"**Average rate of change** from {left_x:.4g} to {right_x:.4g}: {avg:.6g} {dep_unit} per {indep_unit}.")
                else: st.info("Choose different interval endpoints to calculate average rate of change.")
            except Exception:
                st.warning("The function or one of its derivatives is undefined at the selected input. Try another point or interval.")
            if course == "Calculus":
                st.markdown("#### Numerical / symbolic analysis on selected interval")
                st.write("Critical points are roots of f′(x) found by symbolic solving when possible. Classification uses a nearby sign check; zeros alone are not automatically extrema.")
                if critical: st.dataframe(pd.DataFrame(critical, columns=["x (approx.)", "f(x) (approx.)", "classification"]), use_container_width=True)
                else: st.write("No classifiable critical points found in the interval (or symbolic solving did not return any).")
                if inflections: st.dataframe(pd.DataFrame(inflections, columns=["inflection x (approx.)", "f(x) (approx.)"]), use_container_width=True)
                else: st.write("No verified concavity-changing inflection points found in the interval.")
                try:
                    roots1 = sp.solve(sp.Eq(d1, 0), x)
                    roots2 = sp.solve(sp.Eq(d2, 0), x)
                    grid = np.linspace(xmin, xmax, 2001)
                    vals = numeric_eval(expr, x, grid)
                    finite = np.isfinite(vals)
                    if finite.any():
                        idx = np.where(finite)[0]
                        imin, imax = idx[np.nanargmin(vals[finite])], idx[np.nanargmax(vals[finite])]
                        st.write(f"**Approximate absolute extrema over sampled finite points:** minimum ≈ ({grid[imin]:.5g}, {vals[imin]:.5g}); maximum ≈ ({grid[imax]:.5g}, {vals[imax]:.5g}). This numerical grid can miss narrow features and does not certify extrema at discontinuities.")
                except Exception: pass
                st.caption("Domain restrictions and discontinuities: log(x) requires x>0; rational expressions are undefined where denominators are zero. Numerical plots may have gaps where values are undefined. Results are approximations unless explicitly represented by exact symbolic expressions.")
            else:
                st.write(f"**Observed graph trend:** over the selected interval, the function is evaluated numerically; use the graph and rate values to describe increasing/decreasing behavior.")
            st.markdown("#### Contextual visual")
            visual = st.selectbox("Choose visual simulation", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or drug concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"], key="visual")
            # A simple function-driven visual: bar length/height follows current f(x), normalized to visible scale.
            vals = numeric_eval(expr, x, np.linspace(xmin, xmax, 300))
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size:
                lo_v, hi_v = float(np.min(finite_vals)), float(np.max(finite_vals))
                scale = 0.5 if hi_v == lo_v else min(1.0, max(0.05, (fval-lo_v)/(hi_v-lo_v)))
                labels = {
                    "Walking person with prosthetic leg": "Motion indicator (normalized)",
                    "Moving object / projectile": "Object height / position (normalized)",
                    "Tank or drug concentration": "Concentration level (normalized)",
                    "Population / cell colony": "Population level (normalized)",
                    "Thermometer / weather": "Temperature level (normalized)",
                    "Business / economic indicator": "Business / economic level (normalized)",
                }
                st.write(f"**{visual}:** {labels[visual]}. The filled indicator maps the function value within the selected interval to a 0–100% display scale; it is illustrative, not a biomechanical, medical, or physical simulation.")
                st.progress(float(scale))
            st.caption("Playback note: use the current-input slider to move the point and synchronize the function value, tangent, derivative readouts, and contextual explanation. This build does not run a timed autoplay animation; the slider is the reliable manual playback control.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed safely for this expression: {exc}")

with tab_help:
    st.subheader("How to run and deploy")
    st.markdown("""
**Local setup**
1. Install Python 3.10 or newer.
2. Save `app.py` and `requirements.txt` in the same folder.
3. Run `pip install -r requirements.txt`.
4. Run `python -m streamlit run app.py`.

**Streamlit Community Cloud**
1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the main file.
3. Deploy. The app fetches World Bank data at runtime; no credentials are required.

**Render**
- Create a Python Web Service and connect the GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

**Data flow:** controls select country and indicator → World Bank API returns official observations → original observations are kept separate from the editable table → validation checks years and values → centered polynomial fit computes estimates and in-sample metrics → chart and exports use the current validated working data.

**Math assumptions and limitations:** the polynomial uses centered time `t = year − base year`, improving numerical conditioning. At least degree + 1 distinct years are required. R² and MAE are in-sample fit metrics; a chronological holdout score is reported separately when enough observations exist. Polynomials can oscillate and behave implausibly outside the historical range. Estimates are not causal explanations or validated forecasts. Missing observations are not filled with fabricated values.

**Verification checklist**
- Confirm source URL and indicator code; compare selected rows against the World Bank API page.
- Edit a year/value and confirm the modified-scenario warning and exports change; click Reset to source.
- Try duplicate years, non-integer years, nonnumeric values, and too few rows; check the validation messages.
- Change degree, sample increment, horizon, country, indicator, comparison country, and printer view; verify the related output changes.
- Calculate one year, then include that year in batch results and compare estimates.
- Try malformed custom years and expressions such as `log(x)` at x ≤ 0; check clear errors and undefined-value handling.
- Download CSV, Excel, and PDF and open each file in the corresponding application.
- If deployment fails, inspect Streamlit logs for missing dependencies, Python version mismatch, network timeout, or API response errors.
""")
    st.subheader("Important implementation notes")
    st.write("The function lab supports polynomial, rational, exponential, logarithmic, and trigonometric expressions through SymPy. Calculus critical points and inflections are determined symbolically when possible and classified with numerical sign checks; complicated functions may not solve symbolically. The current app includes manual slider-based point movement rather than timed autoplay. The selected visual is a normalized function-driven indicator, not a clinical or engineering-grade simulation.")
