"""
app.py — Streamlit UI: Systemic Banking Risk Simulator (2D layout)
"""
import streamlit as st
import pandas as pd
from engine import simulate
from intervention import heuristic_recapitalization
from utils import load_nodes, get_unique_countries, build_plotly_map, format_losses


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
def configure_page() -> None:
    st.set_page_config(
        page_title="Systemic Risk Simulator",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    :root {
        --bg-deep: #010119; --bg-card: #111827; --bg-side: #010119;
        --accent: #6366F1; --accent-2: #818CF8;
        --red: #EF4444; --green: #22C55E; --blue: #3B82F6;
        --text: #E2E8F0; --muted: #94A3B8; --border: rgba(148,163,184,0.15);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-deep); color: var(--text);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: var(--bg-side) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label { color: var(--muted) !important; }
    [data-testid="metric-container"] {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.78rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text); font-size: 1.65rem; font-weight: 700; }
    .section-header { display: flex; align-items: center; gap: 10px; margin: 24px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .section-header h3 { margin: 0; font-size: 1.05rem; font-weight: 600; color: var(--text); }
    .badge { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em; padding: 3px 9px; border-radius: 999px; text-transform: uppercase; }
    .badge-before { background: rgba(239,68,68,0.18); color: var(--red); border: 1px solid rgba(239,68,68,0.35); }
    .badge-after  { background: rgba(34,197,94,0.18); color: var(--green); border: 1px solid rgba(34,197,94,0.35); }
    .info-box { background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.35); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; font-size: 0.88rem; color: var(--text); line-height: 1.6; }
    .info-box b { color: var(--accent-2); }
    .hero { text-align: center; padding: 80px 40px; }
    .hero h1 { font-size: 2.8rem; font-weight: 800; margin-bottom: 12px; background: linear-gradient(135deg, #818CF8, #6EE7B7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p  { color: var(--muted); font-size: 1.05rem; max-width: 540px; margin: 0 auto 28px; }
    .stButton > button { width: 100%; background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white !important; border: none; border-radius: 8px; padding: 12px 0; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.04em; transition: opacity 0.2s ease; }
    .stButton > button:hover { opacity: 0.88; }
    hr { border-color: var(--border) !important; margin: 28px 0; }
    .js-plotly-plot { border-radius: 12px; }
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df_nodes: pd.DataFrame) -> dict:
    with st.sidebar:
        if st.session_state.get("focused_bank"):
            st.info(f"Focus: **{st.session_state.focused_bank}**")
            if st.button("✕ Clear Focus", key="clear_focus"):
                st.session_state.focused_bank = None
                st.rerun()
            st.divider()

        try:
            st.image("fenrir_logo.png", use_container_width=True)
        except Exception:
            st.markdown("### Systemic Risk")

        st.markdown("<p style='color:#94A3B8;font-size:0.82rem'>Configure shock parameters and run the simulation.</p>", unsafe_allow_html=True)
        st.divider()

        st.markdown("### Shock Parameters")
        alpha = st.slider("Alpha — Shock Severity", 0.005, 0.02, 0.010, 0.001, format="%.3f")
        theta = st.slider("Theta — Default Threshold (CR)", 0.025, 0.05, 0.035, 0.001, format="%.3f")
        lambda_ = st.slider("Lambda — Contagion Intensity", 0.05, 0.25, 0.10, 0.01, format="%.2f")

        st.markdown("### Shock Origin")
        countries = get_unique_countries(df_nodes)
        shock_country = st.selectbox(
            "Shocked Country", options=countries,
            index=countries.index("DE") if "DE" in countries else 0
        )

        st.markdown("### Rescue Budget")
        budget = st.slider("Public Budget (EUR millions)", 0, 500_000, 50_000, 10_000, format="%d")

        st.divider()
        if "sim_running" not in st.session_state:
            st.session_state.sim_running = False

        if st.button("▶  Run Simulation", use_container_width=True):
            st.session_state.sim_running = True

    return {
        "alpha": alpha, "theta": theta, "lambda_": lambda_,
        "budget": budget, "shock_country": shock_country,
        "run_clicked": st.session_state.sim_running,
    }


# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
def render_kpi_row(n_defaults: int, total_losses_eur_m: float, n_banks: int) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Defaults", f"{n_defaults} / {n_banks}")
    c2.metric("Survival Rate", f"{(n_banks - n_defaults) / n_banks:.1%}")
    c3.metric("Systemic Loss", format_losses(total_losses_eur_m * 1e3))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    configure_page()

    if "focused_bank" not in st.session_state:
        st.session_state.focused_bank = None

    df_nodes = load_nodes()
    params = render_sidebar(df_nodes)

    if not params["run_clicked"]:
        st.markdown("""
        <div class='hero'>
            <h1>European Interbank Risk</h1>
            <p>Configure a shock scenario in the sidebar and click <b>Run Simulation</b> to model contagion across the network.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Run engines ─────────────────────────────────────────────────────────
    with st.spinner("Running contagion model…"):
        sim_before = simulate(
            alpha=params["alpha"],
            theta=params["theta"],
            lambda_=params["lambda_"],
            shock_country=params["shock_country"],
            df_nodes=df_nodes,
        )
        intervention = heuristic_recapitalization(
            budget_B=params["budget"],
            df_nodes=df_nodes,
            shock_country=params["shock_country"],
            alpha=params["alpha"],
            theta=params["theta"],
            lambda_=params["lambda_"],
        )

    # engine.py returns: 'final_network_state' (has 'defaulted', 'C_final', 'rescued')
    # intervention.py returns: 'sim_after' (a full simulate() result), 'new_final_losses'
    before_state = sim_before["final_network_state"]
    after_state  = intervention["sim_after"]["final_network_state"]

    focused = st.session_state.focused_bank

    # ── BEFORE row ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'><h3>Before Intervention</h3><span class='badge badge-before'>Pre-rescue</span></div>", unsafe_allow_html=True)
    col_stats, col_map = st.columns([1, 2.5])
    with col_stats:
        render_kpi_row(
            int(before_state["defaulted"].sum()),
            float(sim_before["final_total_losses"]) / 1e3,
            len(df_nodes),
        )
        if focused:
            st.info(f"🔍 Focused on **{focused}**")

        # Bank list for focus selection
        st.markdown("#### Select bank focus")
        chosen = st.selectbox(
            "Bank", options=["All banks"] + sorted(df_nodes["BankName"].dropna().unique().tolist()),
            key="bank_selector_before"
        )
        if chosen != "All banks" and chosen != focused:
            st.session_state.focused_bank = chosen
            st.rerun()
        elif chosen == "All banks" and focused:
            st.session_state.focused_bank = None
            st.rerun()

    with col_map:
        fig_b = build_plotly_map(
            G=sim_before["G"],
            df_state=before_state,
            title="Before Intervention",
            selected_bank_name=focused,
        )
        ev_b = st.plotly_chart(fig_b, use_container_width=True, on_select="rerun", selection_mode="points")
        if ev_b and ev_b.selection.get("points"):
            pt = ev_b.selection["points"][0]
            clicked = pt.get("customdata")
            if clicked and clicked != focused:
                st.session_state.focused_bank = clicked
                st.rerun()

    st.divider()

    # ── AFTER row ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'><h3>After Intervention</h3><span class='badge badge-after'>Post-rescue</span></div>", unsafe_allow_html=True)
    col_stats2, col_map2 = st.columns([1, 2.5])
    with col_stats2:
        render_kpi_row(
            int(after_state["defaulted"].sum()),
            float(intervention["new_final_losses"]) / 1e3,
            len(df_nodes),
        )
        saved = intervention["banks_saved"]
        spent = intervention["capital_spent"]
        st.markdown(f"""
        <div class='info-box'>
            <b>Banks saved:</b> {saved}<br>
            <b>Capital injected:</b> {format_losses(spent * 1e3)}
        </div>
        """, unsafe_allow_html=True)

    with col_map2:
        fig_a = build_plotly_map(
            G=sim_before["G"],
            df_state=after_state,
            title="After Intervention",
            selected_bank_name=focused,
        )
        ev_a = st.plotly_chart(fig_a, use_container_width=True, on_select="rerun", selection_mode="points")
        if ev_a and ev_a.selection.get("points"):
            pt = ev_a.selection["points"][0]
            clicked = pt.get("customdata")
            if clicked and clicked != focused:
                st.session_state.focused_bank = clicked
                st.rerun()


if __name__ == "__main__":
    main()
