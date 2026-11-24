import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

OUT = "figures_final"; REV = "revision_outputs"
os.makedirs(OUT, exist_ok=True)
FS = 7                       # one font size for all text in all figures
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0, "lines.markersize": 4.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
K = "black"; G1 = "0.25"; G2 = "0.55"; G3 = "0.8"
CH = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0892", "CADC0894"]
CHS = [c.replace("CADC0", "") for c in CH]          # 872, 873, ...
MM = 1 / 25.4
W2 = 180 * MM                                       # double-column width


def panel(ax, letter, dx=-0.0, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS, fontweight="bold", va="bottom", ha="right")


def plabel(fig, ax, letter, x=None, dy=0.012):
    """Panel letter placed in figure coordinates (far left, at the height of the axes titles)."""
    p = ax.get_position()
    fig.text(0.004 if x is None else x, p.y1 + dy, letter, fontsize=FS, fontweight="bold", ha="left", va="bottom")


def check_text(fig, name):
    """Report every pair of visible text objects whose bounding boxes overlap."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    items = []

    def add(t):
        if t is not None and t.get_visible() and t.get_text().strip():
            items.append((t.get_text().replace("\n", " ")[:30], t.get_window_extent(rend)))

    for ax in fig.axes:
        for t in ax.texts: add(t)
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label): add(t)
        if ax.axison:
            for t in ax.get_xticklabels() + ax.get_yticklabels(): add(t)
        lg = ax.get_legend()
        if lg:
            for t in lg.get_texts(): add(t)
    for lg in fig.legends:
        for t in lg.get_texts(): add(t)
    for t in fig.texts: add(t)
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].padded(-0.5).overlaps(items[j][1].padded(-0.5)):
                bad.append((items[i][0], items[j][0]))
    if bad:
        print(f"[check] {name}: {len(bad)} overlapping text pair(s)")
        for a, b in bad: print(f"        '{a}'  x  '{b}'")
    else:
        print(f"[check] {name}: no overlapping text")


def save(fig, name):
    check_text(fig, name)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    # All shapes lie inside the axes limits with a margin, and boxes use pad=0 so the drawn
    # edge equals the stated coordinates: nothing touches or crosses the canvas border.
    CW, CH_ = 184, 100
    fig = plt.figure(figsize=(CW * 180 / 184 * MM, CH_ * 180 / 184 * MM))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, CW); ax.set_ylim(0, CH_); ax.axis("off")
    LH = 3.3                                           # one 7-pt text line, in data units

    def box(x, y, w, h, body, title=None, ls="-", lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    ec=K, fc="white", lw=lw, ls=ls, clip_on=False))
        yc = y + h / 2; nb = len(body.split("\n")); n = nb + (1 if title else 0)
        top = yc + (n - 1) * LH / 2
        if title:
            ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=FS, fontweight="bold")
            ax.text(x + w / 2, top - LH * (nb + 1) / 2, body, ha="center", va="center", fontsize=FS, linespacing=1.3)
        else:
            ax.text(x + w / 2, yc, body, ha="center", va="center", fontsize=FS, linespacing=1.3)

    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=K, lw=0.9, solid_capstyle="butt", clip_on=False)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                     color=K, ls=ls, shrinkA=0, shrinkB=0, clip_on=False))

    X1, W1, X2, W2_, X3, W3 = 2, 54, 74, 40, 130, 52
    for x, t in ((X1, "Sample flow"), (X2, "Representations"), (X3, "Analyses and controls")):
        ax.text(x, 98.5, t, fontsize=FS, fontweight="bold", va="top")

    # column 1
    box(X1, 78, W1, 14, "9 channels")
    box(X1, 54, W1, 17, "420 anomalous and\n1520 nominal segments", title="6 analysable channels")
    box(X1, 30, W1, 17, "386 anomalous segments", title="5 channels with onset estimation")
    box(X1, 5, W1, 17, "(primary estimator)", title="177 segments with a detected onset", lw=1.6)
    arrow(X1 + W1 / 2, 78, X1 + W1 / 2, 71); arrow(X1 + W1 / 2, 54, X1 + W1 / 2, 47)
    arrow(X1 + W1 / 2, 30, X1 + W1 / 2, 22)

    # column 2: representations, joined by a bus
    rep_y = [83, 60, 37]
    for yc, t in zip(rep_y, ["Level", "First difference", "Second difference"]):
        box(X2, yc - 6, W2_, 12, t)
    bus1 = 65
    line(X1 + W1, 62.5, bus1, 62.5); line(bus1, rep_y[-1], bus1, rep_y[0])
    for yc in rep_y: arrow(bus1, yc, X2, yc)

    # column 3
    box(X3, 72, W3, 22, "Length-matched, grouped\ncross-validation, six channels", title="Segment-level analysis")
    box(X3, 32, W3, 28, "(five channels)\nRandom pivots (solid)\nPseudo-onsets (dashed)\nFour estimators",
        title="Onset-centred analysis")
    box(X3, 2.5, W3, 22, "Window scan, injected changes\n(oracle pivot), sequence-model\nsensitivity", ls="--")
    bus2 = 122
    for yc in rep_y: line(X2 + W2_, yc, bus2, yc)
    line(bus2, rep_y[-1], bus2, 83)
    arrow(bus2, 83, X3, 83); arrow(bus2, 46, X3, 46)
    arrow(X1 + W1, 13.5, X3, 13.5, ls="--")
    save(fig, "Fig1")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    d = pd.read_csv(f"{REV}/rev8b_segment_level.csv")
    d = d[d.stratum == "len_matched"]
    sets = ["cov_only", "level", "disp", "provided4", "deriv", "all"]
    lab = ["Covariates only", "Level", "Dispersion", "Distributed derivative", "Derivative set", "All"]
    mk = {"logreg": ("o", K, "Logistic regression"), "rf": ("^", G2, "Random forest")}
    fig = plt.figure(figsize=(W2, 125 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.6, wspace=0.28,
                          left=0.20, right=0.99, top=0.89, bottom=0.08)
    yy = np.arange(len(sets))[::-1]
    auc = d[d.kind == "auc"]
    xmin = max(0.0, np.floor((auc.lo.min() - 0.02) * 10) / 10)          # computed from the data
    xticks = [t for t in (0.5, 1.0) if t >= xmin]
    axA0 = None
    for j, ch in enumerate(CH):
        ax = fig.add_subplot(gs[0, j], sharey=axA0)
        if j == 0: axA0 = ax
        for m, off in (("logreg", 0.18), ("rf", -0.18)):
            x = d[(d.channel == ch) & (d.kind == "auc") & (d.model == m)].set_index("name")
            v = np.array([x.loc[s, "value"] for s in sets]); lo = np.array([x.loc[s, "lo"] for s in sets]); hi = np.array([x.loc[s, "hi"] for s in sets])
            ax.errorbar(v, yy + off, xerr=[v - lo, hi - v], fmt=mk[m][0], color=mk[m][1],
                        mfc=mk[m][1] if m == "logreg" else "white", mec=K, ms=4, elinewidth=0.8, capsize=0, label=mk[m][2])
        ax.axvline(0.5, color=G2, lw=0.8, ls=":")
        n = int(d[(d.channel == ch) & (d.kind == "auc")].n_pos.iloc[0])
        ax.set_title(f"{CHS[j]} (n = {n})", pad=3)
        ax.set_xlim(xmin, 1.02); ax.set_xticks(xticks); ax.set_xlabel("AUC")
        if j == 0:
            ax.set_yticks(yy); ax.set_yticklabels(lab)
        else:
            plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(sets) - 0.4)
    h, l = axA0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.945))

    dl = [("deriv-level", "Derivative set − level"), ("deriv-cov_only", "Derivative set − covariates"),
          ("deriv-provided4", "Derivative set − distributed")]
    cyy = np.arange(len(CH))[::-1]
    gsb = gs[1, :].subgridspec(1, 3, wspace=0.3)
    axb0 = None
    for k, (nm, tt) in enumerate(dl):
        ax = fig.add_subplot(gsb[0, k], sharey=axb0)
        if k == 0: axb0 = ax
        for m, off in (("logreg", 0.18), ("rf", -0.18)):
            x = d[(d.kind == "delta") & (d.name == nm) & (d.model == m)].set_index("channel")
            v = np.array([x.loc[c, "value"] for c in CH]); lo = np.array([x.loc[c, "lo"] for c in CH]); hi = np.array([x.loc[c, "hi"] for c in CH])
            ax.errorbar(v, cyy + off, xerr=[v - lo, hi - v], fmt=mk[m][0], color=mk[m][1],
                        mfc=mk[m][1] if m == "logreg" else "white", mec=K, ms=4, elinewidth=0.8, capsize=0)
        sub = d[(d.kind == "delta") & (d.name == nm)]
        lo_, hi_ = sub.lo.min(), sub.hi.max(); pad = 0.06 * (hi_ - lo_)
        ax.set_xlim(min(lo_, 0) - pad, max(hi_, 0) + pad)           # zero line never sits on the spine
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.axvline(0, color=K, lw=0.8, ls="--")
        ax.set_title(tt, pad=3); ax.set_xlabel("Difference in AUC")
        if k == 0:
            ax.set_yticks(cyy); ax.set_yticklabels(CHS); ax.set_ylabel("Channel (CADC0…)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(CH) - 0.4)
    plabel(fig, axA0, "a"); plabel(fig, axb0, "b")
    save(fig, "Fig2")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    w = pd.read_csv(f"{REV}/rev10_window_scan.csv"); w["h"] = w.h.astype(str)
    w = w[w.variant == "all"]
    hs = ["8", "10", "15", "20", "30", "50", "full"]; xs = np.arange(len(hs))
    sets = [("level_all", "Level", "o", "white", ":"), ("d1_all", "First difference", "s", "0.55", "--"),
            ("d2_all", "Second difference", "^", "0.55", "-."), ("all9", "All statistics", "o", K, "-")]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 105 * MM), sharey=True, sharex=True)
    for r, m in enumerate(["rf", "logreg"]):
        for c, e in enumerate(["PELT3", "CUSUM"]):
            ax = axes[r, c]
            x0 = w[(w.est == e) & (w.model == m)]
            for s, lab, mrk, fc, ls in sets:
                g = x0[x0.set == s].groupby("h").auc.agg(["mean", "std"]).reindex(hs)
                ax.plot(xs, g["mean"], color=K, ls=ls, marker=mrk, mfc=fc, mec=K, lw=1.0, ms=4, label=lab)
                ax.fill_between(xs, g["mean"] - g["std"], g["mean"] + g["std"], color="0.5", alpha=0.25, lw=0)
            ax.axvline(5.5, color=G2, lw=0.8, ls=":"); ax.axhline(0.5, color=G2, lw=0.8, ls=":")
            npos = x0[x0.set == "all9"].groupby("h").n_pos.first().reindex(hs)
            ax.set_xticks(xs); ax.set_xticklabels([f"±{h}\n({int(npos[h])})" if h != "full" else f"Full\n({int(npos[h])})" for h in hs])
            ax.set_ylim(0.4, 1.0)
            ax.set_title(f"{'PELT' if e=='PELT3' else 'CUSUM'}, {'random forest' if m=='rf' else 'logistic regression'}", pad=3)
            if c == 0: ax.set_ylabel("AUC")
            if r == 1: ax.set_xlabel("Window half-length, ticks (anomalous windows)")
            panel(ax, "abcd"[r * 2 + c], dx=-0.12 if c == 0 else -0.04)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "Fig3")


# ---------------------------------------------------------------- Fig. 4
def est_label(e):
    if "KF" in e: return "Primary"
    if e == "BOCPD_level": return "Level-input BOCPD"
    return e.replace("PELT3", "PELT (3)").replace("PELT5", "PELT (5)")


def fig4():
    m = pd.read_csv(f"{REV}/rev4_estimator_matrix.csv")
    m = m[(m.stat == "d1_lv") & (m.variant == "all")].copy(); m["lab"] = m.estimator.map(est_label)
    order = ["Primary", "PELT (3)", "PELT (5)", "CUSUM", "Level-input BOCPD"]
    chs = ["CADC0874", "CADC0888", "CADC0894"]
    # a on top, b below: the long pair labels of b no longer collide with panel a
    fig = plt.figure(figsize=(W2, 128 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.62, left=0.24, right=0.985, top=0.86, bottom=0.09)
    gsa = gs[0].subgridspec(1, 3, wspace=0.25)
    yy = np.arange(len(order))[::-1]; ax0 = None
    for j, ch in enumerate(chs):
        ax = fig.add_subplot(gsa[0, j], sharey=ax0)
        if j == 0: ax0 = ax
        for con, off, mk_, fc in (("true_vs_random", 0.14, "o", K), ("true_vs_pseudo", -0.14, "s", "white")):
            x = m[(m.channel == ch) & (m.contrast == con)].set_index("lab")
            x = x[~x.index.duplicated(keep="first")]
            xs, ys, lo, hi = [], [], [], []
            for i, o in enumerate(order):
                if o in x.index and np.isfinite(x.loc[o, "A"]) and x.loc[o, "n_b"] >= 10 and x.loc[o, "n_a"] >= 10:
                    xs.append(x.loc[o, "A"]); ys.append(yy[i] + off); lo.append(x.loc[o, "ci_lo"]); hi.append(x.loc[o, "ci_hi"])
            xs, lo, hi = map(np.array, (xs, lo, hi))
            if len(xs):
                ax.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt=mk_, color=K, mfc=fc, mec=K, ms=4, elinewidth=0.8, capsize=0)
        ax.axvline(0.5, color=K, lw=0.8, ls="--"); ax.set_xlim(0, 1.02); ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(ch, pad=3)
        if j == 1: ax.set_xlabel("Probability of superiority, A")      # one shared label, no clash between panels
        if j == 0:
            ax.set_yticks(yy); ax.set_yticklabels(order)
        else:
            plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
    hnd = [Line2D([], [], marker="o", color=K, mfc=K, mec=K, ms=4, lw=0.8, label="Onset versus random pivot"),
           Line2D([], [], marker="s", color=K, mfc="white", mec=K, ms=4, lw=0.8, label="Onset versus pseudo-onset")]
    fig.legend(handles=hnd, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.62, 0.925))

    # panel b: onset agreement
    a = pd.read_csv(f"{REV}/rev9_estimator_agreement.csv"); a = a[a.channel == "ALL"].copy()
    nm = {"KF_BOCPD": "Primary", "BOCPD_level": "Level-input BOCPD", "PELT3": "PELT (3)", "PELT5": "PELT (5)", "CUSUM": "CUSUM"}
    a["lab"] = a.pair.map(lambda p: " – ".join(nm.get(s, s) for s in p.split("|")))
    a = a.sort_values("within5")
    ax = fig.add_subplot(gs[1]); y = np.arange(len(a))
    for yi, (u, v) in enumerate(zip(a.within5, a.null_within5)): ax.plot([u, v], [yi, yi], color=G2, lw=0.8, zorder=1)
    ax.scatter(a.within5, y, marker="o", c=K, s=22, zorder=3, label="Observed")
    ax.scatter(a.null_within5, y, marker="o", facecolors="white", edgecolors=K, s=22, zorder=3, label="Chance expectation")
    ax.set_yticks(y); ax.set_yticklabels(a.lab)
    xmax = max(a.within5.max(), a.null_within5.max()) * 1.08
    ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(a) - 0.3)
    ax.set_xlabel("Fraction of segments with onsets within 5 ticks")
    ax.legend(loc="lower right", frameon=False)
    plabel(fig, ax0, "a"); plabel(fig, ax, "b")
    save(fig, "Fig4")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    r = pd.read_csv(f"{REV}/rev11_injection_auc.csv", keep_default_na=False)
    rc = pd.read_csv(f"{REV}/rev11_estimator_recovery.csv", keep_default_na=False)
    for d in (r, rc): d["kind"] = d.kind.replace({"": "null"})
    rows = [("null", 0.0, "No change"), ("mean_shift", 2.0, "Mean shift 2σ"), ("mean_shift", 5.0, "Mean shift 5σ"), ("mean_shift", 10.0, "Mean shift 10σ"),
            ("noise_var", 2.0, "Variance ×2"), ("noise_var", 3.0, "Variance ×3"), ("noise_var", 5.0, "Variance ×5"),
            ("slope", 0.05, "Slope 0.05σ per tick"), ("slope", 0.15, "Slope 0.15σ per tick"), ("slope", 0.5, "Slope 0.5σ per tick")]
    cols = [("15", "±15"), ("30", "±30"), ("full", "Full")]
    r["h"] = r.h.astype(str)
    M = np.full((len(rows), len(cols)), np.nan)
    for i, (k, mg, _) in enumerate(rows):
        for j, (h, _) in enumerate(cols):
            x = r[(r.kind == k) & (np.isclose(r.mag.astype(float), mg)) & (r.h == h)]
            if len(x): M[i, j] = float(x.cv_auc_all9.iloc[0])
    # colour bar goes below the heat map, so it can no longer run into panel b
    fig = plt.figure(figsize=(W2, 92 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 0.06], hspace=0.45, wspace=0.06,
                          left=0.20, right=0.98, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    cmap = plt.get_cmap("Greys"); norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                lum = cmap(norm(M[i, j]))[0]                          # text colour chosen from the cell's own grey level
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=FS, color="white" if lum < 0.45 else K)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([x[2] for x in rows])
    ax.set_xlabel("Window half-length, ticks")
    for s in ax.spines.values(): s.set_visible(True)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Cross-validated AUC"); cb.set_ticks(np.arange(0.4, 1.001, 0.1)); cb.outline.set_linewidth(0.8)

    yy = np.arange(len(rows)); allv = []
    for off, e, fc in ((-0.2, "CUSUM", "white"), (0.2, "PELT3", "0.45")):
        v = []
        for k, mg, _ in rows:
            x = rc[(rc.kind == k) & np.isclose(rc.mag.astype(float), mg) & (rc.est == e)]
            v.append(float(x.hit_rate_within5.iloc[0]) if len(x) else np.nan)
        allv += v
        ax2.barh(yy + off, v, height=0.36, color=fc, edgecolor=K, lw=0.8, label="PELT" if e == "PELT3" else e)
    ax2.set_xlim(0, np.nanmax(allv) * 1.15)                          # computed from the data
    step = 0.01 if np.nanmax(allv) <= 0.06 else 0.05                 # round ticks instead of 0.008, 0.016, ...
    ax2.xaxis.set_major_locator(MultipleLocator(step)); ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.setp(ax2.get_yticklabels(), visible=False); ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("Estimated change point within ±5 ticks (fraction)")
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    # separators between the change types (null | mean shift | variance | slope) in both panels
    for i in range(1, len(rows)):
        if rows[i][0] != rows[i - 1][0]:
            ax.axhline(i - 0.5, color="white", lw=2)
            ax2.axhline(i - 0.5, color=G2, lw=0.8, ls=":")
    plabel(fig, ax, "a"); plabel(fig, ax2, "b", x=ax2.get_position().x0)
    save(fig, "Fig5")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for t in todo: {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[t]()