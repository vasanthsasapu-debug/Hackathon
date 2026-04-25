"""
response_curves.py -- Response Curves & ROI Optimizer (Model-Aware)

Three levels of ROI:
  1. Overall media ROI (from best model)
  2. Group/campaign ROI (from best model — group or total level)
  3. Per-channel ROI (from best individual-channel model in ranked_models)
  + Product decomposition as table (proportional, not curves)
"""
import matplotlib
matplotlib.use("Agg")
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
from config import MEDIA_CHANNELS, CHANNEL_GROUPS, LOG_TO_RAW_MAP, get_paths, logger

CR = 1e7  # crore divisor


class ResponseCurveAnalyzer:

    def __init__(self, model_result, feature_matrix, clean_data=None,
                 n_points=100, max_multiplier=2.0):
        self.model_result, self.feature_matrix = model_result, feature_matrix
        self.clean_data, self.n_points, self.max_mult = clean_data, n_points, max_multiplier

        self.best = model_result.get("best_model", model_result.get("ranked_models", [{}])[0])
        self.simulator = model_result.get("simulator")
        self.features = self.best.get("spec_config", {}).get("resolved_features", [])
        self.spec_name = self.best.get("spec_name", "unknown")

        # raw channel spend means
        self.raw_means = {}
        for ch in MEDIA_CHANNELS:
            for src in [feature_matrix,
                        clean_data.get("monthly") if clean_data else None,
                        clean_data.get("weekly") if clean_data else None]:
                if src is not None and ch in src.columns:
                    v = pd.to_numeric(src[ch], errors="coerce").mean()
                    if not np.isnan(v) and v > 0:
                        self.raw_means[ch] = v; break

        self.baseline_gmv = self.simulator({})["baseline_gmv"] if self.simulator else 0
        self.category_shares = self._category_shares()
        self.model_level = self._detect_level()
        self.ch_shares = self._channel_spend_shares()

    # --- detection -----------------------------------------------------------

    def _detect_level(self):
        f = self.features
        if any(LOG_TO_RAW_MAP.get(x) in MEDIA_CHANNELS for x in f if x.startswith("log_")):
            return "individual"
        if any(x.startswith("log_spend_") or x.startswith("spend_") for x in f):
            return "group"
        if any(x in ("log_Total_Investment", "Total.Investment") for x in f):
            return "total"
        return "none"

    def _channel_spend_shares(self):
        """Each channel's share of its group's spend."""
        shares = {}
        for gname, chs in CHANNEL_GROUPS.items():
            g_spend = sum(self.raw_means.get(ch, 0) for ch in chs)
            if g_spend <= 0: continue
            for ch in chs:
                if ch in self.raw_means:
                    shares[ch] = {"group": gname, "spend": self.raw_means[ch],
                                  "share_of_group": self.raw_means[ch] / g_spend}
        return shares

    def _find_channel_model(self):
        """Find best model with individual channel features (e.g. spec_K). Returns (model, sim) or (None, None)."""
        from modeling_engine import build_scenario_simulator
        monthly = self.clean_data.get("monthly") if self.clean_data else None
        best_m, best_sim, best_r2 = None, None, -1
        for m in self.model_result.get("ranked_models", []):
            if not m.get("train_result", {}).get("success"): continue
            feats = m.get("spec_config", {}).get("resolved_features", [])
            ch_feats = [f for f in feats if LOG_TO_RAW_MAP.get(f) in MEDIA_CHANNELS]
            if len(ch_feats) < 3: continue  # need at least 3 individual channels to be useful
            r2 = m["train_result"]["r_squared"]
            if r2 > best_r2:
                tc = m.get("transform_config", {}).get("target", "log_total_gmv")
                sim = build_scenario_simulator(m, self.feature_matrix, feats, tc, monthly)
                if sim:
                    best_m, best_sim, best_r2 = m, sim, r2
        if best_m:
            logger.info("  Channel model: %s (%s) R²=%.3f with %d channel features",
                        best_m["spec_name"], best_m["model_type"], best_r2,
                        len([f for f in best_m["spec_config"]["resolved_features"]
                             if LOG_TO_RAW_MAP.get(f) in MEDIA_CHANNELS]))
        return best_m, best_sim

    def _category_shares(self):
        for src in [self.feature_matrix,
                    self.clean_data.get("monthly") if self.clean_data else None]:
            if src is None: continue
            rc = [c for c in src.columns if c.startswith("Revenue_")]
            if not rc: continue
            totals = {c.replace("Revenue_", ""): pd.to_numeric(src[c], errors="coerce").sum() for c in rc}
            gt = sum(v for v in totals.values() if v > 0)
            if gt > 0: return {k: v/gt for k, v in totals.items() if v > 0}
        return {}

    # --- sweep engine --------------------------------------------------------

    def _sweep(self, label, changes_fn, spend, sim=None):
        """Sweep spend 0→2×, return curve dict."""
        sim = sim or self.simulator
        if not sim or spend <= 0: return None
        mults = np.linspace(0.01, self.max_mult, self.n_points)
        levels = mults * spend
        gmv = np.array([sim(changes_fn(m)).get("predicted_gmv", self.baseline_gmv) for m in mults])
        gmv0 = sim(changes_fn(0.01)).get("predicted_gmv", gmv[0])
        incr = gmv - gmv0
        marginal = np.gradient(gmv, levels)
        if self.n_points > 5:
            marginal = np.convolve(marginal, np.ones(5)/5, mode='same')
        cidx = np.argmin(np.abs(mults - 1.0))
        start = max(2, int(self.n_points * 0.05))
        oidx = start + np.argmax(marginal[start:])
        m_cur = float(marginal[cidx])
        opt = mults[oidx]
        if m_cur < 0:
            status, sat = "negative-impact", None
        elif opt > 1.15:
            status = "under-investing"
            sat = float(min(100, incr[cidx] / incr[-1] * 100)) if incr[-1] > 0 else 0.0
        elif opt < 0.85:
            status = "over-investing"
            sat = float(min(100, incr[cidx] / incr[-1] * 100)) if incr[-1] > 0 else 100.0
        else:
            status = "near-optimal"
            sat = float(min(100, incr[cidx] / incr[-1] * 100)) if incr[-1] > 0 else 50.0
        return dict(channel=label, current_spend=spend, spend_levels=levels, multipliers=mults,
                    gmv_values=gmv, incremental_gmv=incr, marginal_roi=marginal,
                    current_spend_idx=cidx, marginal_roi_at_current=m_cur,
                    optimal_spend=float(levels[oidx]), optimal_multiplier=float(opt),
                    total_roi_at_current=float(incr[cidx]/spend),
                    investment_status=status, saturation_pct=sat)

    # --- curve computation ---------------------------------------------------

    def compute_curves(self):
        """Group/total curves from best model."""
        if self.model_level == "individual":
            return self._individual()
        elif self.model_level == "group":
            return self._groups()
        elif self.model_level == "total":
            return self._total()
        return {"curves": {}, "level": "none", "description": "No spend features"}

    def _individual(self):
        curves = {}
        for f in self.features:
            ch = LOG_TO_RAW_MAP.get(f)
            if ch and ch in MEDIA_CHANNELS and ch in self.raw_means:
                c = self._sweep(ch, lambda m, c=ch: {c: m}, self.raw_means[ch])
                if c: curves[ch] = c
        return {"curves": curves, "level": "individual",
                "description": f"Per-channel ({len(curves)} channels)"}

    def _groups(self):
        curves = {}
        for f in self.features:
            g = f.replace("log_spend_", "").replace("spend_", "") if ("log_spend_" in f or "spend_" in f) else None
            if g and g in CHANNEL_GROUPS:
                chs = [c for c in CHANNEL_GROUPS[g] if c in self.raw_means]
                if not chs: continue
                spend = sum(self.raw_means[c] for c in chs)
                label = f"{g} ({', '.join(chs)})"
                c = self._sweep(label, lambda m, cs=chs: {c: m for c in cs}, spend)
                if c: curves[g] = c
        return {"curves": curves, "level": "group",
                "description": f"Per-group ({len(curves)} groups)"}

    def _total(self):
        all_chs = list(self.raw_means.keys())
        spend = sum(self.raw_means.values())
        c = self._sweep("Total Media Investment", lambda m: {ch: m for ch in all_chs}, spend)
        curves = {"total_investment": c} if c else {}
        # group shares for context
        shares = {}
        for g, chs in CHANNEL_GROUPS.items():
            gs = sum(self.raw_means.get(c, 0) for c in chs)
            if gs > 0:
                shares[g] = {"spend_cr": gs/CR, "share_pct": gs/spend*100, "channels": [c for c in chs if c in self.raw_means]}
        return {"curves": curves, "level": "total",
                "description": f"Aggregate only — {self.spec_name} uses total spend",
                "group_shares": shares}

    def compute_channel_roi(self, group_curves):
        """
        Per-channel ROI. Strategy:
          1. If an all-channels model exists with R²>0.40, use it (real per-channel curves)
          2. Otherwise, split group ROI by each channel's spend share
        """
        MIN_R2 = 0.40
        ch_model, ch_sim = self._find_channel_model()

        # Strategy 1: real per-channel model
        if ch_model and ch_model["train_result"]["r_squared"] >= MIN_R2:
            feats = ch_model["spec_config"]["resolved_features"]
            r2 = ch_model["train_result"]["r_squared"]
            name = ch_model["spec_name"]
            mtype = ch_model["model_type"]
            curves = {}
            for f in feats:
                ch = LOG_TO_RAW_MAP.get(f)
                if ch and ch in MEDIA_CHANNELS and ch in self.raw_means:
                    c = self._sweep(ch, lambda m, c=ch: {c: m}, self.raw_means[ch], ch_sim)
                    if c: curves[ch] = c
            if curves:
                logger.info("  Using channel model %s (R²=%.3f) for per-channel ROI", name, r2)
                return {"curves": curves, "level": "individual",
                        "model_used": f"{name} ({mtype}, R²={r2:.3f})",
                        "description": f"Per-channel ROI from {name} (R²={r2:.3f})."}

        # Strategy 2: proportional split from group ROI
        if not group_curves or not self.ch_shares:
            return {"curves": {}, "level": "none", "description": "No data for channel ROI"}
        curves = {}
        for ch, info in self.ch_shares.items():
            gname = info["group"]
            share = info["share_of_group"]
            if gname not in group_curves: continue
            gc = group_curves[gname]
            cidx = gc["current_spend_idx"]
            # Split the group's marginal ROI by spend share
            curves[ch] = dict(
                channel=ch, current_spend=info["spend"],
                spend_levels=gc["spend_levels"] * share,
                multipliers=gc["multipliers"],
                gmv_values=gc["gmv_values"],
                incremental_gmv=gc["incremental_gmv"] * share,
                marginal_roi=gc["marginal_roi"] * share,
                current_spend_idx=cidx,
                marginal_roi_at_current=float(gc["marginal_roi_at_current"] * share),
                optimal_spend=float(gc["optimal_spend"] * share),
                optimal_multiplier=gc["optimal_multiplier"],
                total_roi_at_current=float(gc["total_roi_at_current"] * share),
                investment_status=gc["investment_status"],
                saturation_pct=gc["saturation_pct"],
                group=gname, share_of_group=float(share))
        return {"curves": curves, "level": "channel_split",
                "model_used": f"Spend-share split from {self.spec_name}",
                "description": f"Per-channel ROI split by spend share within groups. ⚠ Assumes uniform effectiveness within group."}

    # --- product decomposition (table only) ----------------------------------

    def decompose_by_product(self, curve):
        if not self.category_shares: return []
        incr = curve["incremental_gmv"][curve["current_spend_idx"]]
        return sorted([{"category": k, "share_pct": v*100, "incremental_gmv_cr": float(incr*v/CR),
                        "marginal_roi": float(curve["marginal_roi_at_current"]*v)}
                       for k, v in self.category_shares.items()],
                      key=lambda x: x["incremental_gmv_cr"], reverse=True)

    # --- ROI summary ---------------------------------------------------------

    def get_roi_summary(self, result):
        curves = result["curves"]
        table = sorted([dict(channel=c["channel"], current_spend_cr=c["current_spend"]/CR,
                             marginal_roi=c["marginal_roi_at_current"], total_roi=c["total_roi_at_current"],
                             optimal_multiplier=c["optimal_multiplier"],
                             investment_status=c["investment_status"], saturation_pct=c["saturation_pct"])
                        for c in curves.values()], key=lambda x: x["marginal_roi"], reverse=True)
        recs = []
        for r in table:
            if r["investment_status"] == "negative-impact":
                recs.append(f"⚠ {r['channel']}: negative ROI (₹{r['marginal_roi']:.2f}/₹1) — reduce spend.")
            elif r["investment_status"] == "under-investing":
                sat = f"{r['saturation_pct']:.0f}%" if r['saturation_pct'] is not None else "N/A"
                recs.append(f"Increase {r['channel']} — {sat} saturated, ₹{r['marginal_roi']:.2f}/₹1.")
            elif r["investment_status"] == "over-investing":
                recs.append(f"Reduce {r['channel']} — past optimal ({r['optimal_multiplier']:.1f}×).")
        if result["level"] == "total":
            recs.insert(0, f"⚠ Model uses aggregate spend only — channel-level needs group specs.")
        total_spend = sum(c["current_spend"] for c in curves.values())
        total_attr = sum(c["incremental_gmv"][c["current_spend_idx"]] for c in curves.values())
        return dict(table=table, recommendations=recs, level=result["level"],
                    description=result["description"], total_current_spend=total_spend,
                    total_attributed_gmv=total_attr, baseline_gmv=self.baseline_gmv,
                    overall_media_roi=float(total_attr/total_spend) if total_spend > 0 else 0)

    # --- single reusable plot function ---------------------------------------

    def _plot(self, curves, title, filename, save_dir, subtitle=""):
        """Shared plot: curves (left) + ROI bars (right). Returns path."""
        save_dir = save_dir or get_paths()["plots_dir"]
        os.makedirs(save_dir, exist_ok=True)
        if not curves: return None
        n = len(curves)
        colors = plt.cm.Set1(np.linspace(0, 1, max(n, 3)))
        status_colors = {"under-investing": "#4CAF50", "over-investing": "#F44336",
                         "near-optimal": "#FF9800", "negative-impact": "#9E9E9E"}
        wide = n > 1
        fig, axes = plt.subplots(1, 2 if wide else 1, figsize=(20 if wide else 12, 8))
        if not wide: axes = [axes]
        fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=13, fontweight="bold")
        ax1 = axes[0]
        for i, (_, c) in enumerate(curves.items()):
            s, g = c["spend_levels"]/CR, c["incremental_gmv"]/CR
            ci = c["current_spend_idx"]
            ax1.plot(s, g, color=colors[i], lw=2.5, label=c["channel"])
            ax1.plot(s[ci], g[ci], 'o', color=colors[i], ms=10, mec='white', mew=2, zorder=5)
            oi = np.argmin(np.abs(c["spend_levels"] - c["optimal_spend"]))
            if abs(oi - ci) > 3:
                ax1.plot(s[oi], g[oi], 'D', color=colors[i], ms=10, mec='black', mew=.5, zorder=5)
        ax1.set(xlabel="Spend (₹ Cr)", ylabel="Incremental GMV (₹ Cr)")
        ax1.set_title("Diminishing Returns (● Current, ◆ Optimal)")
        ax1.legend(fontsize=9, loc="upper left"); ax1.grid(True, alpha=0.3)
        if wide:
            ax2 = axes[1]
            names = [curves[k]["channel"] for k in curves]
            mrois = [curves[k]["marginal_roi_at_current"] for k in curves]
            stats = [curves[k]["investment_status"] for k in curves]
            si = np.argsort(mrois)[::-1]
            ax2.barh([names[i] for i in si], [mrois[i] for i in si],
                     color=[status_colors.get(stats[i], "#999") for i in si], ec='white', lw=1.5)
            pad = max(abs(r) for r in mrois) * 0.02
            for j, i in enumerate(si):
                ax2.text(mrois[i]+pad, j, f"₹{mrois[i]:.2f} ({stats[i]})", va='center', fontsize=9)
            ax2.set(xlabel="Marginal ROI (₹/₹1)"); ax2.set_title("ROI Ranking")
            ax2.axvline(0, color='black', lw=.8); ax2.grid(True, alpha=.3, axis='x'); ax2.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close("all")
        logger.info("  [OK] Saved %s", filename)
        return path

    def plot_group(self, result, save_dir=None):
        return self._plot(result["curves"], "Response Curves — Group/Campaign Level",
                          "response_curves.png", save_dir, f"Model: {self.spec_name}")

    def plot_channels(self, ch_result, save_dir=None):
        return self._plot(ch_result["curves"], "Response Curves — Per-Channel",
                          "response_curves_by_channel.png", save_dir,
                          ch_result.get("model_used", ""))

    # --- serialization -------------------------------------------------------

    def curves_to_json(self, result):
        out = {"level": result["level"], "description": result["description"], "curves": []}
        for c in result["curves"].values():
            out["curves"].append(dict(
                channel=c["channel"], current_spend_cr=float(c["current_spend"]/CR),
                spend_levels_cr=(c["spend_levels"]/CR).tolist(), multipliers=c["multipliers"].tolist(),
                gmv_values_cr=(c["gmv_values"]/CR).tolist(),
                incremental_gmv_cr=(c["incremental_gmv"]/CR).tolist(),
                marginal_roi=c["marginal_roi"].tolist(),
                marginal_roi_at_current=c["marginal_roi_at_current"],
                optimal_multiplier=c["optimal_multiplier"],
                optimal_spend_cr=float(c["optimal_spend"]/CR),
                total_roi_at_current=c["total_roi_at_current"],
                investment_status=c["investment_status"], saturation_pct=c["saturation_pct"],
                product_breakdown=self.decompose_by_product(c)))
        if "group_shares" in result: out["group_shares"] = result["group_shares"]
        out["category_shares"] = self.category_shares
        return out

    def narrative_context(self, result, ch_result=None):
        s = self.get_roi_summary(result)
        lines = [f"RESPONSE CURVE ANALYSIS | Model: {self.spec_name} | Level: {result['level']}",
                 f"Baseline GMV: ₹{self.baseline_gmv/CR:.2f}Cr | Spend: ₹{s['total_current_spend']/CR:.2f}Cr | Overall ROI: ₹{s['overall_media_roi']:.2f}/₹1",
                 "", "GROUP ROI:"]
        for r in s["table"]:
            sat = f"{r['saturation_pct']:.0f}%" if r['saturation_pct'] is not None else "N/A"
            lines.append(f"  {r['channel']}: ₹{r['marginal_roi']:.2f}/₹1, {sat} sat, {r['investment_status']}")
        if ch_result and ch_result.get("curves"):
            lines += ["", f"CHANNEL ROI ({ch_result.get('model_used','')}):", "  ⚠ Directional estimates."]
            for ch, c in sorted(ch_result["curves"].items(), key=lambda x: x[1]["marginal_roi_at_current"], reverse=True):
                lines.append(f"  {ch}: ₹{c['marginal_roi_at_current']:.2f}/₹1, ₹{c['current_spend']/CR:.2f}Cr, {c['investment_status']}")
        if s["recommendations"]:
            lines += ["", "RECOMMENDATIONS:"] + [f"  - {r}" for r in s["recommendations"]]
        return "\n".join(lines)


# =============================================================================
def run_response_curve_analysis(model_result, feature_matrix, clean_data=None,
                                save_dir=None, n_points=100):
    save_dir = save_dir or get_paths()["plots_dir"]
    print("=" * 70); print("  RESPONSE CURVE & ROI ANALYSIS"); print("=" * 70)

    a = ResponseCurveAnalyzer(model_result, feature_matrix, clean_data, n_points)
    print(f"  Model: {a.spec_name} | Level: {a.model_level}")

    result = a.compute_curves()
    roi = a.get_roi_summary(result)
    print(f"\n  Overall Media ROI: ₹{roi['overall_media_roi']:.2f}/₹1")
    for r in roi["table"]:
        print(f"  {r['channel']:<40} ₹{r['current_spend_cr']:.2f}Cr  ₹{r['marginal_roi']:.2f}/₹1  {r['investment_status']}")

    plots = []
    p = a.plot_group(result, save_dir)
    if p: plots.append(p)

    ch_result = a.compute_channel_roi(result["curves"])
    if ch_result["curves"]:
        print(f"\n  Per-channel ({ch_result.get('model_used','')}):")
        for ch, c in sorted(ch_result["curves"].items(), key=lambda x: x[1]["marginal_roi_at_current"], reverse=True):
            extra = f"  ({c['group']} {c['share_of_group']:.0%})" if "group" in c else ""
            print(f"  {ch:<22} ₹{c['current_spend']/CR:.2f}Cr  ₹{c['marginal_roi_at_current']:.2f}/₹1  {c['investment_status']}{extra}")
        p = a.plot_channels(ch_result, save_dir)
        if p: plots.append(p)

    print("=" * 70)
    return dict(analyzer=a, result=result, channel_curves=ch_result["curves"],
                channel_result=ch_result, group_curves=result["curves"], roi_summary=roi,
                narrative_context=a.narrative_context(result, ch_result), plots=plots,
                curves_json=a.curves_to_json(result),
                channel_curves_json=a.curves_to_json(ch_result) if ch_result["curves"] else {})