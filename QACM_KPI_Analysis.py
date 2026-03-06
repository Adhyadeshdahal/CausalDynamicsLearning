import numpy as np
import matplotlib.pyplot as plt

thresholds = [
    (-100, 100), (-10, 50), (-20, -20), (60, 60),
    (-20, -20),  (-50, 150), (60, 60),  (-100, 150),
]

np.random.seed(42)
n = 3000

def sample_params(n):
    params = []
    for lo, hi in thresholds:
        if lo == hi:
            params.append(np.full(n, lo, dtype=float))
        else:
            params.append(np.random.uniform(lo, hi, n))
    return params

P = sample_params(n)
P1,P2,P3,P4,P5,P6,P7,P8 = P

def safe(x): 
    zero_mask = np.abs(x) < 1e-1
    x[zero_mask] = 1e-1
    return x

K1  = 80  * np.exp(- P1**2                           / (2 * safe(P2)**2))
K2  = 100 * np.exp(-(P1 + P3)**2                     / (2 * safe(P2)**2))
K3  = 120 * np.exp(-(P1 + 45)**2                     / (2 * safe(P4)**2))
K41 = 120 * np.exp(-(safe(P6) + safe(P2) - 30)**2    / (2 * safe(P5)**2))
K42 = 150 * np.exp(-(safe(P6) + safe(P2) - 50)**2    / (2 * safe(P5)**2))
K5  = -35 * np.exp(-(safe(P8) + P1 - 25)**2          / (2 * safe(P7)**2))

kpis      = [K1, K2, K3, K41, K42, K5]
kpi_names = ['K1','K2','K3','K41','K42','K5']
tholds    = [55, 95, 85, 75, 80, -25]
direction = [0,  0,  0,  0,  0,  1]

# Terminal summary
print(f"\n{'='*65}")
print(f"  KPI Statistics  (n={n} random samples within param thresholds)")
print(f"{'='*65}")
print(f"{'KPI':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}  {'Threshold':>9}  {'%Pass':>6}")
print(f"{'-'*65}")
for k, name, thr, d in zip(kpis, kpi_names, tholds, direction):
    m, s, mn, mx = k.mean(), k.std(), k.min(), k.max()
    pct = 100*(k >= thr).mean() if d == 0 else 100*(k <= thr).mean()
    cmp = ">=" if d == 0 else "<="
    print(f"{name:<6} {m:>10.3f} {s:>10.3f} {mn:>10.3f} {mx:>10.3f}  {cmp}{thr:>6}   {pct:>5.1f}%")
print(f"{'='*65}\n")

BG    = '#0d0d1f'
PANEL = '#13132b'
colors_kpi = ['#00d4ff','#ff6b6b','#69ff47','#ffd700','#ff69b4','#ff8c00']

# ── Figure 1: 3D scatter plots ────────────────────────────────────────────────
kpi_3d = [
    (K1,  'P1','P2',    P1, safe(P2),        'K1 = 80·exp(-P1²/2P2²)',          'plasma'),
    (K2,  'P1','P3',    P1, P3,              'K2 = 100·exp(-(P1+P3)²/2P2²)',    'plasma'),
    (K3,  'P1','P4',    P1, safe(P4),        'K3 = 120·exp(-(P1+45)²/2P4²)',    'plasma'),
    (K41, 'P6','P2',    safe(P6), safe(P2),  'K41 = 120·exp(-(P6+P2-30)²/2P5²)','inferno'),
    (K42, 'P6','P2',    safe(P6), safe(P2),  'K42 = 150·exp(-(P6+P2-50)²/2P5²)','inferno'),
    (K5,  'P1','P8',    P1, safe(P8),        'K5 = -35·exp(-(P8+P1-25)²/2P7²)', 'coolwarm'),
]

fig1 = plt.figure(figsize=(22, 14))
fig1.patch.set_facecolor(BG)
fig1.suptitle('KPI 3D Scatter — Sampled Parameter Space (n=3000)',
              fontsize=16, fontweight='bold', color='white', y=0.99)

for i, (k, xl, yl, xd, yd, title, cm) in enumerate(kpi_3d):
    ax = fig1.add_subplot(2, 3, i+1, projection='3d')
    ax.set_facecolor(PANEL)
    sc = ax.scatter(xd, yd, k, c=k, cmap=cm, alpha=0.35, s=5)
    ax.set_xlabel(xl, color='white', fontsize=8, labelpad=4)
    ax.set_ylabel(yl, color='white', fontsize=8, labelpad=4)
    ax.set_zlabel(kpi_names[i], color='white', fontsize=8, labelpad=4)
    ax.set_title(title, color='white', fontsize=8.5, pad=6)
    ax.tick_params(colors='white', labelsize=6)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#222244')
    ax.yaxis.pane.set_edgecolor('#222244')
    ax.zaxis.pane.set_edgecolor('#222244')
    cb = fig1.colorbar(sc, ax=ax, shrink=0.45, pad=0.12)
    cb.ax.yaxis.set_tick_params(color='white', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

fig1.tight_layout(rect=[0,0,1,0.97])
fig1.savefig('kpi_3d_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved kpi_3d_analysis.png")

# ── Figure 2: Histogram panel ─────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
fig2.patch.set_facecolor(BG)
fig2.suptitle('KPI Histograms — Mean / Std / Threshold  (n=3000 samples)',
              fontsize=15, fontweight='bold', color='white')

for ax, k, name, thr, d, col in zip(axes.flat, kpis, kpi_names, tholds, direction, colors_kpi):
    ax.set_facecolor(PANEL)
    m, s = k.mean(), k.std()
    ax.hist(k, bins=70, color=col, alpha=0.80, edgecolor='none', label='Samples')
    ymax = ax.get_ylim()[1]
    ax.axvline(m,     color='white',  lw=2.2, linestyle='-',  label=f'Mean = {m:.2f}')
    ax.axvline(m + s, color='#aaaaff', lw=1.5, linestyle='--', label=f'+1σ  = {m+s:.2f}')
    ax.axvline(m - s, color='#aaaaff', lw=1.5, linestyle='--', label=f'−1σ  = {m-s:.2f}')
    ax.axvline(thr,   color='yellow', lw=2.2, linestyle=':',  label=f'Thr  = {thr}')

    # shade pass / fail regions
    xlim = ax.get_xlim()
    if d == 0:
        ax.axvspan(thr, xlim[1], alpha=0.08, color='lime',   label='Pass zone')
        ax.axvspan(xlim[0], thr, alpha=0.08, color='red',    label='Fail zone')
        pct = 100*(k >= thr).mean()
    else:
        ax.axvspan(xlim[0], thr, alpha=0.08, color='lime',   label='Pass zone')
        ax.axvspan(thr, xlim[1], alpha=0.08, color='red',    label='Fail zone')
        pct = 100*(k <= thr).mean()

    ax.set_title(f'{name}   (Pass: {pct:.1f}%)', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value', color='white', fontsize=9)
    ax.set_ylabel('Count', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#333366')
    leg = ax.legend(fontsize=7.5, facecolor='#0d0d1f', edgecolor='#333366')
    for txt in leg.get_texts(): txt.set_color('white')

fig2.tight_layout(rect=[0,0,1,0.96])
fig2.savefig('kpi_histograms.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved kpi_histograms.png")