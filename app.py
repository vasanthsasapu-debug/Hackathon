"""MMIX Dashboard — Streamlit UI"""
import streamlit as st
import sys, os, io, json, re, pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "src": ROOT = os.path.dirname(ROOT)
for p in [ROOT, os.path.join(ROOT, "src")]:
    if p not in sys.path: sys.path.insert(0, p)

from config import MEDIA_CHANNELS, CHANNEL_GROUPS, get_paths

st.set_page_config(page_title="MMIX Dashboard", layout="wide", page_icon="📊")
CR = 1e7
CACHE_DIR = os.path.join(ROOT, "outputs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

for k, v in {"pipeline_states": {}, "pipeline_done": {}, "scenario_history": []}.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("📊 Marketing Mix Model Dashboard")
tab_a, tab_b = st.tabs(["🚀 Pipeline", "🔮 Scenarios"])

# ═══════════════════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════════════════
def _clean(obj):
    if obj is None or callable(obj): return None
    if hasattr(obj, 'simulator'): return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("simulator", "all_simulators", "run_custom", "builder", "analyzer")}
    if isinstance(obj, (list, tuple)): return type(obj)(_clean(x) for x in obj)
    return obj

def save_results(g, state):
    try:
        d = {k: getattr(state, k, None) for k in ["granularity", "top_n_scenarios", "model_filter", "iteration", "max_iterations", "spec_strategy", "reasoning_trace", "decisions", "quality_scores", "current_phase", "paths", "corr_matrix", "outlier_log", "assumptions"]}
        for k in ["model_result", "fe_result", "aggregated_data", "response_curves"]: d[k] = _clean(getattr(state, k, None))
        d["narrator_narratives"] = state.narrator.narratives if state.narrator and hasattr(state.narrator, "narratives") else None
        with open(os.path.join(CACHE_DIR, f"mmix_results_{g}.pkl"), "wb") as f:
            pickle.dump({"state_dict": d, "timestamp": datetime.now().isoformat(), "version": 2}, f)
    except Exception as e: st.warning(f"Cache failed: {e}")

def load_cached(g):
    path = os.path.join(CACHE_DIR, f"mmix_results_{g}.pkl")
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: d = pickle.load(f).get("state_dict", {})
        if not d: return None
        state = SimpleNamespace(**{k: d.get(k) for k in ["granularity", "iteration", "max_iterations", "spec_strategy", "reasoning_trace", "decisions", "quality_scores", "model_result", "fe_result", "aggregated_data", "response_curves"]})
        state.narrator = SimpleNamespace(narratives=d["narrator_narratives"]) if d.get("narrator_narratives") else None
        
        # Debug: check if response_curves has ROI data
        rc = d.get("response_curves")
        if rc and rc.get("roi_summary"):
            roi = rc["roi_summary"].get("overall_media_roi", 0)
            st.toast(f"Loaded {g}: ROI=₹{roi:.2f}/₹1", icon="✅")
        return state
    except Exception as e: 
        st.warning(f"Load {g} failed: {e}")
        return None

def get_cache_info():
    info = {}
    for g in ["weekly", "monthly"]:
        path = os.path.join(CACHE_DIR, f"mmix_results_{g}.pkl")
        try:
            with open(path, "rb") as f: c = pickle.load(f)
            r2 = c.get("state_dict", {}).get("model_result", {}).get("best_model", {}).get("train_result", {}).get("r_squared")
            info[g] = {"exists": True, "r2": r2}
        except: info[g] = {"exists": False}
    return info

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_curves(curves_json):
    if not curves_json or not curves_json.get("curves"): return
    fig = go.Figure()
    for i, c in enumerate(curves_json["curves"]):
        fig.add_trace(go.Scatter(x=c["spend_levels_cr"], y=c["incremental_gmv_cr"], mode="lines", name=c["channel"], line=dict(width=3)))
    fig.update_layout(title="Response Curves", xaxis_title="Spend (₹Cr)", yaxis_title="Incr GMV (₹Cr)", template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)

def plot_roi(curves_json):
    if not curves_json or not curves_json.get("curves"): return
    data = sorted(curves_json["curves"], key=lambda x: x.get("marginal_roi_at_current", 0), reverse=True)
    colors = {"under-investing": "#4CAF50", "over-investing": "#F44336", "near-optimal": "#FF9800", "negative-impact": "#9E9E9E"}
    fig = go.Figure(go.Bar(y=[d["channel"] for d in data], x=[d.get("marginal_roi_at_current", 0) for d in data], orientation="h",
        marker_color=[colors.get(d.get("investment_status"), "#999") for d in data],
        text=[f"₹{d.get('marginal_roi_at_current', 0):.2f}" for d in data], textposition="outside"))
    fig.update_layout(title="Marginal ROI (₹/₹1)", template="plotly_white", height=max(200, len(data)*40), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def display_results(state, label):
    if not state or not state.model_result: return
    best = state.model_result.get("best_model", {})
    tr, cv = best.get("train_result", {}), best.get("cv_result", {})
    scores = best.get("scores", {})
    
    # Metrics - show all composite score components
    st.subheader(f"📊 {label.title()} Model")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("R²", f"{tr.get('r_squared', 0):.3f}")
    c2.metric("Adj R²", f"{tr.get('adj_r_squared', 0):.3f}")
    c3.metric("CV MAPE", f"{cv.get('cv_mape', 0):.1f}%" if cv.get('cv_mape') else "N/A")
    c4.metric("Composite", f"{scores.get('composite', 0):.3f}")
    c5.metric("Stability", f"{scores.get('stability_score', 0):.3f}")
    c6.metric("Ordinality", "✓" if best.get('ordinality', {}).get('passed') else "✗")
    st.caption(f"{best.get('spec_name', '')} | {best.get('model_type', '')} | {best.get('transform', '')} | Fit: {scores.get('fit_score', 0):.3f} | VIF: {scores.get('vif_score', 0):.3f}")
    
    # Top Models
    ranked = state.model_result.get("ranked_models", [])
    if ranked:
        with st.expander("🏆 Top Models"):
            st.dataframe(pd.DataFrame([{
                "Rank": m.get("rank"), 
                "Spec": m.get("spec_name", "")[:20], 
                "Type": m.get("model_type"), 
                "R²": f"{m.get('train_result', {}).get('r_squared', 0):.3f}",
                "Composite": f"{m.get('scores', {}).get('composite', 0):.3f}",
                "Fit": f"{m.get('scores', {}).get('fit_score', 0):.3f}",
                "Stability": f"{m.get('scores', {}).get('stability_score', 0):.3f}",
                "Ordinality": "✓" if m.get("ordinality", {}).get("passed") else "✗"
            } for m in ranked[:10]]), hide_index=True)
    
    # Coefficients - fix the chart
    coefs = {k: v for k, v in tr.get("coefficients", {}).items() if k != "const"}
    if coefs:
        with st.expander("📊 Coefficients"):
            # Sort by absolute value
            sorted_coefs = dict(sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=list(sorted_coefs.keys()),
                x=list(sorted_coefs.values()),
                orientation='h',
                marker_color=['#4CAF50' if v > 0 else '#F44336' for v in sorted_coefs.values()],
                text=[f"{v:+.4f}" for v in sorted_coefs.values()],
                textposition='outside'
            ))
            fig.update_layout(
                height=max(250, len(coefs) * 40),
                template="plotly_white",
                xaxis_title="Coefficient Value",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=80, t=10, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ROI
    rc = getattr(state, 'response_curves', None)
    if rc:
        with st.expander("📈 Response Curves & ROI", expanded=True):
            roi = rc.get("roi_summary", {})
            
            # Calculate weighted marginal ROI from table
            table = roi.get("table", [])
            total_spend = sum(r.get("current_spend_cr", 0) for r in table)
            weighted_marginal = sum(r.get("marginal_roi", 0) * r.get("current_spend_cr", 0) for r in table) / total_spend if total_spend > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall ROI", f"₹{roi.get('overall_media_roi', 0):.2f}/₹1")
            c2.metric("Marginal ROI", f"₹{weighted_marginal:.2f}/₹1", help="Weighted avg marginal ROI")
            c3.metric("Attributed GMV", f"₹{roi.get('total_attributed_gmv', 0)/CR:.1f}Cr")
            c4.metric("Total Spend", f"₹{roi.get('total_current_spend', 0)/CR:.1f}Cr")
            
            if rc.get("curves_json"): 
                plot_curves(rc["curves_json"])
                plot_roi(rc["curves_json"])
            
            # ROI table
            if table:
                with st.expander("📋 ROI Details"):
                    st.dataframe(pd.DataFrame([{
                        "Channel": r.get("channel"),
                        "Spend (Cr)": f"₹{r.get('current_spend_cr', 0):.2f}",
                        "Marginal ROI": f"₹{r.get('marginal_roi', 0):.2f}",
                        "Total ROI": f"₹{r.get('total_roi', 0):.2f}",
                        "Status": r.get("investment_status", ""),
                        "Saturation": f"{r.get('saturation_pct', 0):.0f}%" if r.get('saturation_pct') else "N/A"
                    } for r in table]), hide_index=True)
        
        ch_json = rc.get("channel_curves_json", {})
        if ch_json and ch_json.get("curves"):
            with st.expander("📊 Per-Channel ROI"):
                st.caption("⚠️ Directional estimates")
                plot_curves(ch_json); plot_roi(ch_json)
    
    # Narratives - each section as sub-expander
    narrator = getattr(state, 'narrator', None)
    if narrator and hasattr(narrator, 'narratives') and narrator.narratives:
        with st.expander("📝 Narratives", expanded=False):
            narrative_sections = [
                ("scenarios", "🎯 Scenario Recommendations"),
                ("modeling", "🔬 Model Interpretation"),
                ("features", "⚙️ Feature Engineering"),
                ("eda", "📊 EDA Summary"),
                ("outliers", "🧹 Data Quality & Outliers"),
            ]
            for key, title in narrative_sections:
                text = narrator.narratives.get(key)
                if text:
                    with st.expander(title, expanded=False):
                        st.markdown(text)
    
    # Reasoning
    trace = getattr(state, 'reasoning_trace', [])
    if trace:
        with st.expander("🔍 Reasoning Trace", expanded=False):
            for e in trace:
                st.markdown(f"**[{e.get('phase')}]** {e.get('decision')} — {e.get('reasoning', '')[:150]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO PARSING
# ═══════════════════════════════════════════════════════════════════════════════
def parse_scenario(text):
    # Check for "all channels" pattern first
    all_match = re.search(r'(increase|reduce|cut|boost)\s+all\s+(?:channels?\s+)?(?:by\s+)?(\d+)%', text.lower())
    if all_match:
        action, pct = all_match.groups()
        mult = 1 + int(pct)/100 if action in ("increase", "boost") else 1 - int(pct)/100
        changes = {ch: mult for ch in MEDIA_CHANNELS}
        if "sale" in text.lower(): changes.update({"sale_flag": 1, "sale_days": 3})
        return changes
    
    try:
        from narrative_generator import get_llm_client, call_llm
        client = get_llm_client()
        if client:
            resp = call_llm(client, f'Parse to JSON multipliers. Channels: {MEDIA_CHANNELS}. "all channels" means apply to all. Input: "{text}". Return ONLY JSON like {{"TV": 1.5}}', max_tokens=100)
            return json.loads(re.sub(r'```json?|```', '', resp).strip())
    except: pass
    
    changes = {}
    for m in re.finditer(r'(increase|reduce|cut|boost)\s+(\w+[\.\w]*)\s+(?:by\s+)?(\d+)%', text.lower()):
        act, ch, pct = m.groups()
        ch_map = {"tv": "TV", "digital": "Digital", "sponsorship": "Sponsorship", "sem": "SEM", 
                  "online": "Online.marketing", "online.marketing": "Online.marketing",
                  "affiliates": "Affiliates", "radio": "Radio", "content": "Content.Marketing"}
        if ch in ch_map: 
            changes[ch_map[ch]] = 1 + int(pct)/100 if act in ("increase", "boost") else 1 - int(pct)/100
    if "sale" in text.lower() and "no sale" not in text.lower(): 
        changes.update({"sale_flag": 1, "sale_days": 3})
    return changes or None

def plot_impact(changes, simulator, baseline):
    """Bar chart showing impact of each changed channel."""
    items = []
    for ch, mult in changes.items():
        if ch in ("sale_flag", "sale_days", "sale_intensity"): continue
        r = simulator({ch: mult})
        delta = (r["predicted_gmv"] - baseline) / CR
        items.append({"channel": ch, "change_pct": (mult-1)*100, "delta_cr": delta})
    
    if changes.get("sale_flag"):
        r = simulator({k: v for k, v in changes.items() if k.startswith("sale")})
        items.append({"channel": "Sale Event", "change_pct": 0, "delta_cr": (r["predicted_gmv"] - baseline) / CR})
    
    if not items: return
    
    # Sort by impact
    items = sorted(items, key=lambda x: x["delta_cr"], reverse=True)
    
    fig = go.Figure(go.Bar(
        y=[f"{i['channel']} ({i['change_pct']:+.0f}%)" for i in items],
        x=[i["delta_cr"] for i in items],
        orientation="h",
        marker_color=["#4CAF50" if i["delta_cr"] >= 0 else "#F44336" for i in items],
        text=[f"₹{i['delta_cr']:+.2f} Cr" for i in items],
        textposition="outside"
    ))
    fig.update_layout(
        title="Impact by Channel",
        xaxis_title="GMV Change (₹ Cr)",
        template="plotly_white",
        height=max(150, len(items) * 50 + 50),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=100, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def get_narrative(changes, result, r2):
    try:
        from narrative_generator import get_llm_client, call_llm
        client = get_llm_client()
        if not client: return None
        ch_desc = ", ".join(f"{ch}: {(v-1)*100:+.0f}%" for ch, v in changes.items() if not ch.startswith("sale"))
        prompt = f"""Marketing scenario: {ch_desc}
Result: ₹{result['baseline_gmv']/CR:.1f}Cr → ₹{result['predicted_gmv']/CR:.1f}Cr ({result['change_pct']:+.1f}%)

Write 2-3 short bullet points. Be specific with ₹ numbers. Include one recommendation.
Format exactly as:
• Point one
• Point two
• Recommendation"""
        return call_llm(client, prompt, max_tokens=120)
    except: return None

def ensure_simulator(state):
    if not state or not state.model_result: return False
    if state.model_result.get("simulator"): return True
    try:
        from modeling_engine import build_scenario_simulator
        best = state.model_result.get("best_model", {})
        if best.get("train_result", {}).get("success") and state.fe_result:
            state.model_result["simulator"] = build_scenario_simulator(best, state.fe_result["data"], 
                best["spec_config"]["resolved_features"], best["spec_config"]["target"],
                (state.aggregated_data or {}).get("data", {}).get("monthly"))
            return True
    except: pass
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# TAB A — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_a:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Controls")
        granularity = st.selectbox("Granularity", ["weekly", "monthly", "both"])
        skip_eda = st.checkbox("Skip EDA", True)
        skip_narratives = st.checkbox("Skip Narratives", False)
        
        with st.expander("Advanced"):
            top_models = st.slider("Top models for scenarios", 1, 3, 1)
            model_filter = st.selectbox("Model types", ["all", "linear", "OLS,Ridge,Lasso"])
        
        run_btn = st.button("▶ Run", type="primary", use_container_width=True)
        
        st.divider()
        cache = get_cache_info()
        for g in ["weekly", "monthly"]:
            st.caption(f"{'✓' if cache[g]['exists'] else '✗'} {g}: R²={cache[g].get('r2', 0):.3f}" if cache[g]['exists'] else f"✗ {g}")
        if st.button("📥 Load Cache", use_container_width=True):
            for g in ["weekly", "monthly"]:
                s = load_cached(g)
                if s and s.model_result: st.session_state.pipeline_states[g] = s; st.session_state.pipeline_done[g] = True
            st.rerun()
    
    with col2:
        if run_btn:
            for g in (["weekly", "monthly"] if granularity == "both" else [granularity]):
                st.markdown(f"### Running {g.title()}")
                progress = st.progress(0)
                status = st.empty()
                log_container = st.empty()
                logs = []
                
                class Tracker(io.StringIO):
                    def write(self, s):
                        if s.strip(): 
                            logs.append(s.rstrip())
                            log_container.code("\n".join(logs[-20:]), language=None)
                        for m, p in {"DATA": 10, "EDA": 20, "OUTLIER": 30, "AGGREG": 40, "FEATURE": 50, "MODEL": 65, "SCENARIO": 80, "RESPONSE": 90, "DONE": 100}.items():
                            if m in s.upper(): 
                                progress.progress(p/100)
                                status.info(f"**{m.title()}** in progress...")
                        return len(s)
                    def flush(self): pass
                
                old = sys.stdout
                try:
                    sys.stdout = Tracker()
                    from agent_orchestrator import run_agentic_pipeline
                    state = run_agentic_pipeline(granularity=g, skip_eda=skip_eda, 
                        skip_narratives=skip_narratives, top_n_scenarios=top_models, model_filter=model_filter)
                    st.session_state.pipeline_states[g] = state
                    st.session_state.pipeline_done[g] = True
                    save_results(g, state)
                    progress.progress(1.0)
                    status.success(f"✓ {g.title()} complete — R²={state.model_result['best_model']['train_result']['r_squared']:.3f}")
                except Exception as e: 
                    status.error(f"Failed: {e}")
                finally: 
                    sys.stdout = old
                
                with st.expander("Full Logs"): 
                    st.code("\n".join(logs), language=None)
        
        # Results display
        done = [g for g, ok in st.session_state.pipeline_done.items() if ok]
        if done:
            st.divider()
            view = st.radio("View:", done, horizontal=True) if len(done) > 1 else done[0]
            display_results(st.session_state.pipeline_states.get(view), view)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB B — SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_b:
    available = {g: st.session_state.pipeline_states.get(g) for g in ["weekly", "monthly"]
                 if st.session_state.pipeline_done.get(g) and st.session_state.pipeline_states.get(g) 
                 and st.session_state.pipeline_states.get(g).model_result}
    
    if not available:
        st.warning("Run pipeline or load cache first.")
    else:
        st.subheader("What-If Scenarios")
        model_g = st.radio("Model:", list(available.keys()), horizontal=True) if len(available) > 1 else list(available.keys())[0]
        state = available[model_g]
        r2 = state.model_result["best_model"]["train_result"]["r_squared"]
        st.caption(f"{model_g} | R²={r2:.3f}")
        
        user_input = st.text_area("Scenario:", placeholder="Increase TV by 30%, reduce Digital by 20%")
        if st.button("🔮 Run") and user_input.strip():
            if not ensure_simulator(state): st.error("Simulator failed")
            else:
                changes = parse_scenario(user_input)
                if not changes: st.error("Could not parse")
                else:
                    result = state.model_result["simulator"](changes)
                    base, pred, pct = result["baseline_gmv"], result["predicted_gmv"], result["change_pct"]
                    
                    st.info(f"**Applied:** {', '.join(f'{ch}: {(v-1)*100:+.0f}%' for ch, v in changes.items() if not ch.startswith('sale'))}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Baseline", f"₹{base/CR:.2f}Cr")
                    c2.metric("Predicted", f"₹{pred/CR:.2f}Cr", f"{pct:+.1f}%")
                    c3.metric("Change", f"₹{(pred-base)/CR:+.2f}Cr")
                    
                    # Show channel-level impact if multiple channels
                    if len([c for c in changes if not c.startswith("sale")]) > 0:
                        plot_impact(changes, state.model_result["simulator"], base)
                    
                    narrative = get_narrative(changes, result, r2)
                    if narrative: 
                        st.markdown("**💡 Insight:**")
                        st.markdown(narrative)
                    
                    st.session_state.scenario_history.append({"input": user_input, "pct": pct, "model": model_g})
        
        if st.session_state.scenario_history:
            with st.expander("History"):
                for h in st.session_state.scenario_history[-5:]:
                    st.markdown(f"[{h['model'][0].upper()}] {h['input']} → **{h['pct']:+.1f}%**")