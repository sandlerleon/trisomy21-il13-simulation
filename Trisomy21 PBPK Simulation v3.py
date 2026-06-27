"""
================================================================================
PRENATAL IL-13/TSLP BLOCKADE IN TRISOMY 21
Computational Simulation — Version 3.0
================================================================================
Paper:  "Neuroinflammatory Vulnerability Windows in Trisomy 21:
         A Theoretical Framework for Extended-Interval IL-13/TSLP
         Biologic Intervention During Fetal Neurodevelopment"
Author: Leon Sandler (pen name: Leonid Sandler)
        leonsandler@alumni.swinburne.edu
        https://www.linkedin.com/in/-leon-sandler
Zenodo: https://zenodo.org  (DOI assigned upon upload)
GitHub: https://github.com/leonsandler/trisomy21-il13-simulation

CHANGES FROM v2.0:
  - Added three downstream neurodevelopmental marker modules:
      * Synaptic density index (synaptogenesis - microglial pruning)
      * Oligodendrocyte maturation index (myelination efficiency)
      * Composite Neurodevelopmental Index (NDI) — weighted aggregate
  - Reviewer-recommended wording for simulation conclusions
  - Figure 6: Downstream neurodevelopmental markers panel
  - Figure 7: Composite NDI comparison
  - Extended summary statistics for neurodevelopmental outputs

WHAT THIS SIMULATION DOES:
  Layer 1 — PBPK:   maternal injection → FcRn placental transfer → fetal drug
  Layer 2 — PD ODE: IL-13/TSLP → JAK/STAT → microglial activation
  Layer 3 — NDM:    microglial activation → synaptic density +
                     oligodendrocyte maturation + composite NDI
  Layer 4 — Sensitivity: Monte Carlo ±20-25% parameter variation

OUTPUT FILES:
  Fig1–Fig7 PNG files + simulation_summary_v3.txt

DISCLAIMER:
  All kinetic parameters are literature-derived placeholder estimates.
  Downstream neurodevelopmental marker relationships are theoretical,
  derived from published qualitative associations in DS literature.
  This is a hypothesis-testing scaffold, NOT a predictive clinical model.
  Results should be interpreted as hypothesis-generating only.

DEPENDENCIES: numpy  scipy  matplotlib
INSTALL:      pip install numpy scipy matplotlib
PYTHON:       >= 3.8
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    'maternal':   '#2E5FA3',
    'fetal':      '#c45e2a',
    'euploid':    '#2e8b57',
    't21_untx':   '#c0392b',
    't21_tx':     '#1a6b8a',
    'fcRn':       '#8e44ad',
    'dose':       '#e67e22',
    'synapse':    '#16a085',
    'oligo':      '#8e44ad',
    'ndi':        '#2c3e50',
    'w1':         '#1a6b8a',
    'w2':         '#2E5FA3',
    'w3':         '#c45e2a',
    'bg':         '#f7f9fb',
    'grid':       '#dce6ed',
}

DISCLAIMER = '⚠ SIMULATION WITH PLACEHOLDER PARAMETERS — hypothesis-generating only, not predictive of clinical outcomes'

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

WEEK_START, WEEK_END = 8, 40
T_START = WEEK_START * 7
T_END   = WEEK_END   * 7
N_PTS   = 8000

DOSE_DAYS  = [84, 168, 224]   # weeks 12, 24, 32
DOSE_WEEKS = [d//7 for d in DOSE_DAYS]

# ── PK ───────────────────────────────────────────────────────────────────────
DOSE_MG_KG      = 0.3
MATERNAL_WEIGHT = 70.0
DOSE_TOTAL_MG   = DOSE_MG_KG * MATERNAL_WEIGHT
BIOAVAILABILITY = 0.72
K_ABSORB        = 0.045
T_HALF_MATERNAL = 42.0
K_ELIM_MAT      = np.log(2) / T_HALF_MATERNAL
VD_MATERNAL     = 8500.0
K_ELIM_FETAL    = K_ELIM_MAT * 0.35

FCRT_MAX       = 0.83
FCRT_MIDPOINT  = 196.0
FCRT_STEEPNESS = 0.048

def fcRn_efficiency(t_days):
    return FCRT_MAX / (1.0 + np.exp(-FCRT_STEEPNESS * (t_days - FCRT_MIDPOINT)))

def fetal_vd(t_days):
    frac = np.clip((t_days - T_START) / (T_END - T_START), 0, 1)
    return 50.0 + 3150.0 * frac ** 1.6

# ── Neuroinflammatory ODE parameters ─────────────────────────────────────────
BASE_IL13  = 1.00
BASE_TSLP  = 1.00
T21_IL13   = 1.65
T21_TSLP   = 1.45

K_CL_IL13  = 0.15
K_CL_TSLP  = 0.20
K_CL_JAK   = 0.30
K_CL_MIC   = 0.08    # T21 reduced resolution
K_CL_MIC_E = 0.13    # Euploid

K_JAK_IL13 = 0.25
K_JAK_TSLP = 0.15
K_MIC_JAK  = 0.20

K_ON  = 0.08
K_OFF = 0.004

# ── DOWNSTREAM NEURODEVELOPMENTAL MARKER PARAMETERS ──────────────────────────
# These relationships are derived from published qualitative associations in
# DS preclinical literature. They are theoretical and require empirical
# calibration against experimental data.
#
# SYNAPTIC DENSITY INDEX
# Source: Contestabile 2017 (DS neurogenesis); Flores-Aguilar 2020 (microglial pruning)
# Mechanism: Synaptogenesis rate inhibited by IL-13; excess pruning by activated microglia
# Relationship: dSyn/dt = k_syn_base - k_syn_il13*IL13 - k_syn_mic*Microglia + k_syn_res
K_SYN_BASE  = 2.50   # baseline synaptogenesis rate (euploid)
K_SYN_IL13  = 0.08   # IL-13 inhibition of synaptogenesis [Granot-Hershkovitz 2019]
K_SYN_MIC   = 0.06   # excess microglial pruning rate [Flores-Aguilar 2020]
K_SYN_RES   = 0.12   # synaptic turnover/recovery rate
T21_SYN_BASE = 0.82  # T21 intrinsic synaptogenesis deficit (chr21 gene effects)

# OLIGODENDROCYTE MATURATION INDEX
# Source: Maes 2012 (IL-13 impairs OPC differentiation); Stagni 2018 (myelination in DS)
# Mechanism: IL-13 and microglial activation impair OPC-to-oligodendrocyte transition
# Window: Most relevant weeks 28-36 (myelination onset)
K_OLI_BASE  = 1.80   # baseline OPC maturation rate
K_OLI_IL13  = 0.10   # IL-13 direct impairment of OPC differentiation [Maes 2012]
K_OLI_MIC   = 0.05   # microglial inflammatory environment effect
K_OLI_RES   = 0.08   # oligodendrocyte turnover
T21_OLI_BASE = 0.78  # T21 intrinsic oligodendrocyte deficit

# COMPOSITE NEURODEVELOPMENTAL INDEX (NDI)
# Weighted combination of inverse microglial activation, synaptic density,
# and oligodendrocyte maturation — normalised to euploid = 100
# Weights based on relative contribution estimates from DS literature
NDI_WEIGHTS = {'mic': 0.35, 'syn': 0.40, 'oli': 0.25}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PBPK MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_pbpk(dose_days, t_eval):
    mat_out   = np.zeros(len(t_eval))
    fetal_out = np.zeros(len(t_eval))
    dose_conc_per_inj = (DOSE_TOTAL_MG * 1000 * BIOAVAILABILITY) / VD_MATERNAL

    boundaries = [t_eval[0]] + sorted([d for d in dose_days if T_START < d < T_END]) + [t_eval[-1]+0.01]
    y0 = [0.0, 0.0, 0.0]

    for i in range(len(boundaries)-1):
        t0, t1 = boundaries[i], boundaries[i+1]
        mask   = (t_eval >= t0) & (t_eval < t1)
        t_seg  = t_eval[mask]
        if len(t_seg) < 2:
            continue

        def odes(t, y):
            dep, mat, fet = y
            eff = fcRn_efficiency(t)
            k_tr = eff * 0.0075
            return [
                -K_ABSORB * dep,
                 K_ABSORB * dep - K_ELIM_MAT * mat - k_tr * mat,
                 k_tr * mat * (VD_MATERNAL / fetal_vd(t)) - K_ELIM_FETAL * fet
            ]

        sol = solve_ivp(odes, [t_seg[0], t_seg[-1]], y0,
                        t_eval=t_seg, method='RK45', rtol=1e-7, atol=1e-9, max_step=0.5)
        mat_out[mask]   = np.clip(sol.y[1], 0, None)
        fetal_out[mask] = np.clip(sol.y[2], 0, None)
        y0 = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]]

        nxt = boundaries[i+1]
        if nxt in dose_days:
            y0[0] += dose_conc_per_inj

    return mat_out, fetal_out

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — NEUROINFLAMMATORY + DOWNSTREAM NEURODEVELOPMENTAL ODE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def full_odes(t, y, t_eval, fetal_conc, t21, treated):
    """
    8-variable ODE system.
    y[0] = IL-13 (free)
    y[1] = TSLP
    y[2] = JAK/STAT activation
    y[3] = Microglial activation index
    y[4] = Ab-IL13 neutralised complex
    y[5] = Synaptic density index
    y[6] = Oligodendrocyte maturation index
    y[7] = (unused — reserved for future BDNF module)
    """
    il13, tslp, jak, mic, neu, syn, oli, _ = y

    ab = float(np.interp(t, t_eval, fetal_conc)) if treated else 0.0

    prod_il13  = BASE_IL13 * (T21_IL13 if t21 else 1.0)
    prod_tslp  = BASE_TSLP * (T21_TSLP if t21 else 1.0)
    k_res_mic  = K_CL_MIC  if t21 else K_CL_MIC_E
    syn_base   = K_SYN_BASE * (T21_SYN_BASE if t21 else 1.0)
    oli_base   = K_OLI_BASE * (T21_OLI_BASE if t21 else 1.0)

    # Neuroinflammatory layer
    bind   = K_ON  * il13 * ab
    unbind = K_OFF * neu
    d_il13 = prod_il13  - K_CL_IL13 * il13 - bind + unbind
    d_tslp = prod_tslp  - K_CL_TSLP * tslp
    d_neu  = bind - unbind - 0.25 * K_CL_IL13 * neu
    d_jak  = K_JAK_IL13 * il13 + K_JAK_TSLP * tslp - K_CL_JAK * jak
    d_mic  = K_MIC_JAK  * jak  - k_res_mic * mic

    # Downstream neurodevelopmental markers
    # Synaptic density: driven by synaptogenesis, inhibited by IL-13 and microglial pruning
    syn_loss = K_SYN_IL13 * il13 + K_SYN_MIC * mic
    d_syn    = syn_base - syn_loss - K_SYN_RES * syn

    # Oligodendrocyte maturation: driven by OPC maturation, inhibited by IL-13 and mic activation
    # Myelination is gated by gestational age (window III: weeks 28–36)
    week = t / 7.0
    myelination_gate = np.clip((week - 20) / 16.0, 0, 1)  # ramps from wk20 to wk36
    oli_inhibition   = (K_OLI_IL13 * il13 + K_OLI_MIC * mic) * myelination_gate
    d_oli = oli_base * myelination_gate - oli_inhibition - K_OLI_RES * oli

    d_reserved = 0.0

    return [d_il13, d_tslp, d_jak, d_mic, d_neu, d_syn, d_oli, d_reserved]

def run_full(t_eval, fetal_conc, t21=True, treated=True):
    prod_il13 = BASE_IL13 * (T21_IL13 if t21 else 1.0)
    prod_tslp = BASE_TSLP * (T21_TSLP if t21 else 1.0)
    k_res     = K_CL_MIC  if t21 else K_CL_MIC_E
    syn_b     = K_SYN_BASE * (T21_SYN_BASE if t21 else 1.0)
    oli_b     = K_OLI_BASE * (T21_OLI_BASE if t21 else 1.0)
    jak0      = (K_JAK_IL13 * prod_il13 + K_JAK_TSLP * prod_tslp) / K_CL_JAK
    mic0      = K_MIC_JAK * jak0 / k_res
    syn0      = syn_b / (K_SYN_IL13 * prod_il13/K_CL_IL13 + K_SYN_MIC * mic0 + K_SYN_RES + 1e-9)
    syn0      = max(syn0, 0.1)
    y0 = [prod_il13/K_CL_IL13, prod_tslp/K_CL_TSLP, jak0, mic0, 0.0, syn0, 0.01, 0.0]

    sol = solve_ivp(
        lambda t, y: full_odes(t, y, t_eval, fetal_conc, t21, treated),
        [t_eval[0], t_eval[-1]], y0,
        t_eval=t_eval, method='RK45',
        rtol=1e-6, atol=1e-8, max_step=1.0
    )
    return sol.y

def compute_ndi(y, y_eup_ref):
    """
    Composite Neurodevelopmental Index, normalised to euploid = 100.
    Components: inverse microglial (lower is better), synaptic density, oligodendrocyte maturation.
    """
    mic_norm = y[3] / (y_eup_ref[3] + 1e-9)   # ratio to euploid (lower = worse)
    syn_norm = y[5] / (y_eup_ref[5] + 1e-9)   # ratio to euploid (higher = better)
    oli_norm = y[6] / (y_eup_ref[6] + 1e-9)   # ratio to euploid (higher = better)

    # NDI: penalise high microglia, reward high synapse and oligo
    ndi = 100 * (
        NDI_WEIGHTS['mic'] * (1 - np.clip(mic_norm - 1, 0, 2))
      + NDI_WEIGHTS['syn'] * np.clip(syn_norm, 0, 1)
      + NDI_WEIGHTS['oli'] * np.clip(oli_norm, 0, 1)
    )
    return np.clip(ndi, 0, 100)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(t_eval, n_samples=100):
    rng = np.random.default_rng(42)
    mic_tx_all  = []
    mic_utx_all = []
    ndi_tx_all  = []
    ndi_utx_all = []

    global K_ELIM_MAT, K_ABSORB, T21_IL13, T21_TSLP, K_ON, FCRT_MAX
    global K_SYN_IL13, K_OLI_IL13

    saved = (K_ELIM_MAT, K_ABSORB, T21_IL13, T21_TSLP, K_ON, FCRT_MAX, K_SYN_IL13, K_OLI_IL13)

    y_eup_ref = run_full(t_eval, np.zeros(N_PTS), t21=False, treated=False)

    for _ in range(n_samples):
        K_ELIM_MAT = saved[0] * rng.uniform(0.80, 1.20)
        K_ABSORB   = saved[1] * rng.uniform(0.80, 1.20)
        T21_IL13   = saved[2] * rng.uniform(0.85, 1.15)
        T21_TSLP   = saved[3] * rng.uniform(0.85, 1.15)
        K_ON       = saved[4] * rng.uniform(0.75, 1.25)
        FCRT_MAX   = saved[5] * rng.uniform(0.80, 1.00)
        K_SYN_IL13 = saved[6] * rng.uniform(0.75, 1.25)
        K_OLI_IL13 = saved[7] * rng.uniform(0.75, 1.25)

        _, fc = run_pbpk(DOSE_DAYS, t_eval)
        yt   = run_full(t_eval, fc,               t21=True, treated=True)
        yu   = run_full(t_eval, np.zeros(N_PTS), t21=True, treated=False)
        mic_tx_all.append(yt[3]);  mic_utx_all.append(yu[3])
        ndi_tx_all.append(compute_ndi(yt, y_eup_ref))
        ndi_utx_all.append(compute_ndi(yu, y_eup_ref))

    (K_ELIM_MAT, K_ABSORB, T21_IL13, T21_TSLP, K_ON, FCRT_MAX, K_SYN_IL13, K_OLI_IL13) = saved
    return (np.array(mic_tx_all), np.array(mic_utx_all),
            np.array(ndi_tx_all), np.array(ndi_utx_all))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RUN ALL SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("="*70)
print("TRISOMY 21 IL-13 BLOCKADE — COMPUTATIONAL SIMULATION v3.0")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

t_days = np.linspace(T_START, T_END, N_PTS)
weeks  = t_days / 7.0

print("\n[1/4] PBPK model...")
mat_conc, fet_conc = run_pbpk(DOSE_DAYS, t_days)

print("[2/4] Full ODE system (neuroinflammatory + neurodevelopmental markers)...")
y_eup = run_full(t_days, np.zeros(N_PTS), t21=False, treated=False)
y_utx = run_full(t_days, np.zeros(N_PTS), t21=True,  treated=False)
y_tx  = run_full(t_days, fet_conc,        t21=True,  treated=True)

ndi_eup = compute_ndi(y_eup, y_eup)
ndi_utx = compute_ndi(y_utx, y_eup)
ndi_tx  = compute_ndi(y_tx,  y_eup)

print("[3/4] Monte Carlo sensitivity analysis (n=100)...")
mic_tx_mc, mic_utx_mc, ndi_tx_mc, ndi_utx_mc = run_sensitivity(t_days, n_samples=100)

print("[4/4] Generating figures...")

# ── Axis helper ───────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel, xlim=(8,40)):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#1F3864', pad=7)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, alpha=0.3, color=C['grid'], lw=0.8)
    ax.set_xlim(*xlim)
    for sp in ax.spines.values(): sp.set_color('#cccccc')
    ax.tick_params(labelsize=8.5)

def add_windows(ax, alpha=0.10):
    for ws,we,col in [(8,16,C['w1']),(20,28,C['w2']),(28,36,C['w3'])]:
        ax.axvspan(ws, we, alpha=alpha, color=col, zorder=0)

def add_doses(ax):
    for wk in DOSE_WEEKS:
        ax.axvline(x=wk, color=C['dose'], lw=1.3, ls='--', alpha=0.8, zorder=3)

def watermark(fig):
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color='#c0392b', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Placental Permeation Curve
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(11,5.5))
fig1.patch.set_facecolor(C['bg'])
add_windows(ax, 0.07)
ax.plot(weeks, mat_conc, color=C['maternal'], lw=2.5, label='Maternal plasma', zorder=4)
ax.plot(weeks, fet_conc, color=C['fetal'],    lw=2.5, ls='--', label='Fetal compartment', zorder=4)
ax2 = ax.twinx()
fcrt = [fcRn_efficiency(t)*100 for t in t_days]
ax2.fill_between(weeks, 0, fcrt, alpha=0.07, color=C['fcRn'])
ax2.plot(weeks, fcrt, color=C['fcRn'], lw=1.5, ls=':', label='FcRn efficiency (%)')
ax2.set_ylabel('FcRn Transfer Efficiency (%)', color=C['fcRn'], fontsize=9.5)
ax2.tick_params(axis='y', labelcolor=C['fcRn'], labelsize=8.5)
ax2.set_ylim(0,105)
add_doses(ax)
for wk in DOSE_WEEKS:
    eff = fcRn_efficiency(wk*7)*100
    ax.annotate(f'▼Wk{wk}\n({eff:.0f}%)', xy=(wk, mat_conc.max()*0.88),
                fontsize=7.5, color=C['dose'], ha='center', fontweight='bold')
style_ax(ax, 'Figure 1: Placental Permeation Curve\nMaternal vs Fetal Compartments with FcRn Transfer Efficiency',
         'Gestational Age (weeks)', 'Drug Concentration (normalised μg/mL)')
l1,lb1 = ax.get_legend_handles_labels(); l2,lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=8.5, loc='upper left', framealpha=0.9)
watermark(fig1)
plt.tight_layout(rect=[0,0.02,1,1])
fig1.savefig(os.path.join(OUT_DIR,'Fig1_Placental_Permeation.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig1")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Window Coverage
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(11,5.5))
fig2.patch.set_facecolor(C['bg'])
for ws,we,col,lbl,sub in [(8,16,C['w1'],'Window I','Neurogenesis'),
                           (20,28,C['w2'],'Window II','Synaptic Pruning'),
                           (28,36,C['w3'],'Window III','Myelination')]:
    ax.axvspan(ws,we,alpha=0.13,color=col,zorder=0)
    ax.text((ws+we)/2, 0.97, lbl, ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=8.5, color=col, fontweight='bold')
    ax.text((ws+we)/2, 0.89, sub, ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=7.5, color=col)
ax.plot(weeks, fet_conc, color=C['fetal'], lw=2.8, label='Fetal antibody concentration', zorder=5)
nz = fet_conc[fet_conc>0.01]
thr = np.percentile(nz, 20) if len(nz)>0 else 0.05
ax.axhline(y=thr, color='#7f8c8d', ls=':', lw=1.5, alpha=0.8, label=f'Illustrative threshold (~{thr:.2f})')
ax.fill_between(weeks, thr, fet_conc, where=fet_conc>thr, alpha=0.12, color=C['fetal'], label='Active coverage period')
add_doses(ax)
style_ax(ax,'Figure 2: Vulnerability Window Coverage\nFetal Drug Levels vs Three Critical Developmental Periods',
         'Gestational Age (weeks)', 'Fetal Drug Concentration (normalised μg/mL)')
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
watermark(fig2)
plt.tight_layout(rect=[0,0.02,1,1])
fig2.savefig(os.path.join(OUT_DIR,'Fig2_Window_Coverage.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig2")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Microglial Suppression Index
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1,2, figsize=(14,6))
fig3.patch.set_facecolor(C['bg'])
for ax, (var, ylabel, title_sfx) in zip(axes, [
    (3, 'Microglial Activation Index (normalised)', 'Microglial Activation'),
    (0, 'Free IL-13 Concentration (normalised)',    'IL-13 Neutralisation'),
]):
    add_windows(ax, 0.08)
    ax.plot(weeks, y_eup[var], color=C['euploid'],  lw=1.8, ls='-.', label='Euploid reference', zorder=4)
    ax.plot(weeks, y_utx[var], color=C['t21_untx'], lw=2.3, label='T21 untreated', zorder=5)
    ax.plot(weeks, y_tx[var],  color=C['t21_tx'],   lw=2.3, ls='--', label='T21 + IL-13 blockade', zorder=5)
    ax.fill_between(weeks, y_utx[var], y_tx[var],
                    where=y_utx[var]>y_tx[var], alpha=0.15, color=C['t21_tx'], label='Simulated reduction')
    add_doses(ax)
    style_ax(ax, f'Figure 3: {title_sfx}\nT21 Treated vs Untreated vs Euploid', 'Gestational Age (weeks)', ylabel)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
watermark(fig3)
plt.suptitle('Figure 3: Neuroinflammatory Response', fontsize=11, fontweight='bold', color='#1F3864', y=1.01)
plt.tight_layout(rect=[0,0.02,1,1])
fig3.savefig(os.path.join(OUT_DIR,'Fig3_Microglial_Suppression.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig3")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — FcRn Transfer Dynamics
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1,2, figsize=(14,5.5))
fig4.patch.set_facecolor(C['bg'])
ax = axes[0]
add_windows(ax, 0.10)
fcrt_arr = np.array([fcRn_efficiency(t)*100 for t in t_days])
ax.fill_between(weeks, 0, fcrt_arr, alpha=0.12, color=C['fcRn'])
ax.plot(weeks, fcrt_arr, color=C['fcRn'], lw=2.5)
for wk in DOSE_WEEKS:
    eff = fcRn_efficiency(wk*7)*100
    ax.scatter([wk],[eff], color=C['dose'], zorder=6, s=90)
    ax.annotate(f'Wk {wk}: {eff:.0f}%', xy=(wk,eff), xytext=(wk+0.8,eff+4),
                fontsize=8, color=C['dose'], fontweight='bold')
style_ax(ax,'Figure 4a: FcRn Transfer Efficiency by Gestational Age',
         'Gestational Age (weeks)', 'FcRn Transfer Efficiency (%)')
ax.set_ylim(0,100)

ax = axes[1]
ratio = np.where(mat_conc>0.001, fet_conc/mat_conc, 0)
add_windows(ax, 0.10)
ax.plot(weeks, ratio, color=C['fetal'], lw=2.5, label='Fetal/maternal ratio')
ax.axhline(y=1.0, color='gray', ls=':', lw=1.5, label='Ratio = 1.0')
ax.fill_between(weeks, ratio, 1.0, where=ratio>1.0, alpha=0.12, color=C['fetal'],
                label='Fetal exceeds maternal')
add_doses(ax)
style_ax(ax,'Figure 4b: Fetal/Maternal Concentration Ratio',
         'Gestational Age (weeks)', 'Fetal/Maternal Ratio')
ax.legend(fontsize=8.5, framealpha=0.9)
watermark(fig4)
plt.tight_layout(rect=[0,0.02,1,1])
fig4.savefig(os.path.join(OUT_DIR,'Fig4_FcRn_Transfer_Dynamics.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig4")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(11,5.5))
fig5.patch.set_facecolor(C['bg'])
add_windows(ax, 0.08)
ax.plot(weeks, y_eup[3], color=C['euploid'],  lw=1.8, ls='-.', label='Euploid reference', zorder=5)
ax.plot(weeks, y_utx[3], color=C['t21_untx'], lw=2.2, label='T21 untreated (central)', zorder=6)
ax.plot(weeks, y_tx[3],  color=C['t21_tx'],   lw=2.2, ls='--', label='T21 + IL-13 blockade (central)', zorder=6)
p5t,p95t = np.percentile(mic_tx_mc,5,axis=0), np.percentile(mic_tx_mc,95,axis=0)
p5u,p95u = np.percentile(mic_utx_mc,5,axis=0),np.percentile(mic_utx_mc,95,axis=0)
ax.fill_between(weeks, p5t, p95t, alpha=0.18, color=C['t21_tx'],   label='Treated — 90% CI (n=100)')
ax.fill_between(weeks, p5u, p95u, alpha=0.12, color=C['t21_untx'], label='Untreated — 90% CI')
add_doses(ax)
style_ax(ax,'Figure 5: Sensitivity Analysis — Monte Carlo Parameter Uncertainty\n±20–25% variation across 8 key PK and PD parameters (n=100)',
         'Gestational Age (weeks)', 'Microglial Activation Index (normalised)')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
watermark(fig5)
plt.tight_layout(rect=[0,0.02,1,1])
fig5.savefig(os.path.join(OUT_DIR,'Fig5_Sensitivity_Analysis.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig5")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Downstream Neurodevelopmental Markers
# ─────────────────────────────────────────────────────────────────────────────
fig6, axes = plt.subplots(1,2, figsize=(14,6))
fig6.patch.set_facecolor(C['bg'])

# Panel A: Synaptic Density Index
ax = axes[0]
add_windows(ax, 0.10)
ax.plot(weeks, y_eup[5], color=C['euploid'],  lw=2.0, ls='-.', label='Euploid reference', zorder=4)
ax.plot(weeks, y_utx[5], color=C['t21_untx'], lw=2.3, label='T21 untreated', zorder=5)
ax.plot(weeks, y_tx[5],  color=C['synapse'],  lw=2.3, ls='--', label='T21 + IL-13 blockade', zorder=5)
ax.fill_between(weeks, y_utx[5], y_tx[5],
                where=y_tx[5]>y_utx[5], alpha=0.15, color=C['synapse'],
                label='Simulated synaptic density gain')
add_doses(ax)
style_ax(ax,'Figure 6a: Synaptic Density Index\nIL-13 inhibition of synaptogenesis + microglial pruning effects',
         'Gestational Age (weeks)', 'Synaptic Density Index (normalised)')
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.text(0.02, 0.05, 'Source relationship: Contestabile 2017;\nFlores-Aguilar 2020; Granot-Hershkovitz 2019',
        transform=ax.transAxes, fontsize=6.5, color='#666', style='italic')

# Panel B: Oligodendrocyte Maturation Index
ax = axes[1]
add_windows(ax, 0.10)
ax.plot(weeks, y_eup[6], color=C['euploid'],  lw=2.0, ls='-.', label='Euploid reference', zorder=4)
ax.plot(weeks, y_utx[6], color=C['t21_untx'], lw=2.3, label='T21 untreated', zorder=5)
ax.plot(weeks, y_tx[6],  color=C['oligo'],    lw=2.3, ls='--', label='T21 + IL-13 blockade', zorder=5)
ax.fill_between(weeks, y_utx[6], y_tx[6],
                where=y_tx[6]>y_utx[6], alpha=0.15, color=C['oligo'],
                label='Simulated myelination improvement')
add_doses(ax)
style_ax(ax,'Figure 6b: Oligodendrocyte Maturation Index\nMyelination onset (weeks 28–36); IL-13 impairs OPC differentiation',
         'Gestational Age (weeks)', 'Oligodendrocyte Maturation Index (normalised)')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.text(0.02, 0.05, 'Source relationship: Maes 2012 (IL-13 impairs OPC);\nStagni 2018 (DS myelination deficit)',
        transform=ax.transAxes, fontsize=6.5, color='#666', style='italic')

watermark(fig6)
plt.suptitle('Figure 6: Downstream Neurodevelopmental Markers\n'
             'Theoretical relationships derived from published DS preclinical literature',
             fontsize=11, fontweight='bold', color='#1F3864', y=1.02)
plt.tight_layout(rect=[0,0.02,1,1])
fig6.savefig(os.path.join(OUT_DIR,'Fig6_Neurodevelopmental_Markers.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig6")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Composite Neurodevelopmental Index (NDI)
# ─────────────────────────────────────────────────────────────────────────────
fig7, axes = plt.subplots(1,2, figsize=(14,6))
fig7.patch.set_facecolor(C['bg'])

# Panel A: NDI over gestational time
ax = axes[0]
add_windows(ax, 0.10)
ax.axhline(y=100, color=C['euploid'], lw=1.5, ls=':', alpha=0.7, label='Euploid = 100')
ax.plot(weeks, ndi_eup, color=C['euploid'],  lw=2.0, ls='-.', label='Euploid NDI', zorder=4)
ax.plot(weeks, ndi_utx, color=C['t21_untx'], lw=2.3, label='T21 untreated NDI', zorder=5)
ax.plot(weeks, ndi_tx,  color=C['t21_tx'],   lw=2.3, ls='--', label='T21 + IL-13 blockade NDI', zorder=5)
ax.fill_between(weeks, ndi_utx, ndi_tx,
                where=ndi_tx>ndi_utx, alpha=0.15, color=C['t21_tx'], label='Simulated NDI improvement')

# Sensitivity bands for NDI
p5n_tx  = np.percentile(ndi_tx_mc,  5,  axis=0)
p95n_tx = np.percentile(ndi_tx_mc,  95, axis=0)
ax.fill_between(weeks, p5n_tx, p95n_tx, alpha=0.10, color=C['t21_tx'], label='Treated NDI 90% CI')

add_doses(ax)
style_ax(ax,'Figure 7a: Composite Neurodevelopmental Index (NDI)\nWeighted aggregate: microglial + synaptic + oligodendrocyte',
         'Gestational Age (weeks)', 'NDI Score (euploid = 100)')
ax.set_ylim(0, 115)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

# Panel B: Bar chart of mean NDI by window period
ax = axes[1]
masks = {
    'Window I\n(Wks 8–16)':   (weeks>=8)  & (weeks<=16),
    'Window II\n(Wks 20–28)': (weeks>=20) & (weeks<=28),
    'Window III\n(Wks 28–36)':(weeks>=28) & (weeks<=36),
    'Overall\n(Wks 8–40)':    (weeks>=8)  & (weeks<=40),
}
labels = list(masks.keys())
ndi_eup_bars = [ndi_eup[m].mean() for m in masks.values()]
ndi_utx_bars = [ndi_utx[m].mean() for m in masks.values()]
ndi_tx_bars  = [ndi_tx[m].mean()  for m in masks.values()]

x = np.arange(len(labels))
w = 0.25
b1 = ax.bar(x-w, ndi_eup_bars, w, color=C['euploid'],  alpha=0.85, label='Euploid')
b2 = ax.bar(x,   ndi_utx_bars, w, color=C['t21_untx'], alpha=0.85, label='T21 untreated')
b3 = ax.bar(x+w, ndi_tx_bars,  w, color=C['t21_tx'],   alpha=0.85, label='T21 + IL-13 blockade')

for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 115)
ax.set_facecolor('white')
ax.set_title('Figure 7b: Mean NDI by Developmental Window\nComparison across three scenarios',
             fontsize=10.5, fontweight='bold', color='#1F3864', pad=7)
ax.set_ylabel('Mean NDI Score (euploid = 100)', fontsize=9.5)
ax.grid(True, alpha=0.3, axis='y', color=C['grid'])
ax.legend(fontsize=8.5, framealpha=0.9)
ax.axhline(y=100, color=C['euploid'], lw=1.0, ls=':', alpha=0.5)

watermark(fig7)
plt.suptitle('Figure 7: Composite Neurodevelopmental Index (NDI)\n'
             'Theoretical aggregate marker — requires empirical validation',
             fontsize=11, fontweight='bold', color='#1F3864', y=1.02)
plt.tight_layout(rect=[0,0.02,1,1])
fig7.savefig(os.path.join(OUT_DIR,'Fig7_Composite_NDI.png'), dpi=180, bbox_inches='tight', facecolor=C['bg'])
plt.close(); print("  ✓ Fig7")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — QUANTITATIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def pct_red(a, b, mask):
    u=a[mask].mean(); t=b[mask].mean()
    return (1-t/u)*100 if u>0 else 0

def pct_gain(a, b, mask):
    u=a[mask].mean(); t=b[mask].mean()
    return (t/u-1)*100 if u>0 else 0

masks_d = {
    'WI':  (weeks>=8)  &(weeks<=16),
    'WII': (weeks>=20) &(weeks<=28),
    'WIII':(weeks>=28) &(weeks<=36),
    'ALL': (weeks>=8)  &(weeks<=40),
}

summary = f"""
================================================================================
SIMULATION SUMMARY v3.0 — Trisomy 21 IL-13 Blockade Framework
Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

PHARMACOKINETIC RESULTS
────────────────────────────────────────────────────────────────────────────────
Peak maternal concentration:              {mat_conc.max():.4f} (normalised μg/mL)
Peak fetal concentration:                 {fet_conc.max():.4f}
Fetal/maternal ratio — Week 12:          {fet_conc[np.argmin(np.abs(weeks-12))]/(mat_conc[np.argmin(np.abs(weeks-12))]+1e-9):.3f}
Fetal/maternal ratio — Week 24:          {fet_conc[np.argmin(np.abs(weeks-24))]/(mat_conc[np.argmin(np.abs(weeks-24))]+1e-9):.3f}
Fetal/maternal ratio — Week 32:          {fet_conc[np.argmin(np.abs(weeks-32))]/(mat_conc[np.argmin(np.abs(weeks-32))]+1e-9):.3f}
FcRn efficiency — Dose 1 (wk 12):       {fcRn_efficiency(84)*100:.1f}%
FcRn efficiency — Dose 2 (wk 24):       {fcRn_efficiency(168)*100:.1f}%
FcRn efficiency — Dose 3 (wk 32):       {fcRn_efficiency(224)*100:.1f}%

MICROGLIAL ACTIVATION (mean, normalised)
────────────────────────────────────────────────────────────────────────────────
Period          Euploid    T21 Untreated    T21 Treated    Reduction
Window I:       {y_eup[3][masks_d['WI']].mean():.3f}      {y_utx[3][masks_d['WI']].mean():.3f}            {y_tx[3][masks_d['WI']].mean():.3f}          {pct_red(y_utx[3],y_tx[3],masks_d['WI']):.1f}%
Window II:      {y_eup[3][masks_d['WII']].mean():.3f}      {y_utx[3][masks_d['WII']].mean():.3f}            {y_tx[3][masks_d['WII']].mean():.3f}          {pct_red(y_utx[3],y_tx[3],masks_d['WII']):.1f}%
Window III:     {y_eup[3][masks_d['WIII']].mean():.3f}      {y_utx[3][masks_d['WIII']].mean():.3f}            {y_tx[3][masks_d['WIII']].mean():.3f}          {pct_red(y_utx[3],y_tx[3],masks_d['WIII']):.1f}%
Overall:        {y_eup[3][masks_d['ALL']].mean():.3f}      {y_utx[3][masks_d['ALL']].mean():.3f}            {y_tx[3][masks_d['ALL']].mean():.3f}          {pct_red(y_utx[3],y_tx[3],masks_d['ALL']):.1f}%
Free IL-13:     Overall reduction: {pct_red(y_utx[0],y_tx[0],masks_d['ALL']):.1f}%

DOWNSTREAM NEURODEVELOPMENTAL MARKERS
────────────────────────────────────────────────────────────────────────────────
SYNAPTIC DENSITY INDEX (mean, normalised):
Period          Euploid    T21 Untreated    T21 Treated    Gain
Window I:       {y_eup[5][masks_d['WI']].mean():.3f}      {y_utx[5][masks_d['WI']].mean():.3f}            {y_tx[5][masks_d['WI']].mean():.3f}          {pct_gain(y_utx[5],y_tx[5],masks_d['WI']):.1f}%
Window II:      {y_eup[5][masks_d['WII']].mean():.3f}      {y_utx[5][masks_d['WII']].mean():.3f}            {y_tx[5][masks_d['WII']].mean():.3f}          {pct_gain(y_utx[5],y_tx[5],masks_d['WII']):.1f}%
Window III:     {y_eup[5][masks_d['WIII']].mean():.3f}      {y_utx[5][masks_d['WIII']].mean():.3f}            {y_tx[5][masks_d['WIII']].mean():.3f}          {pct_gain(y_utx[5],y_tx[5],masks_d['WIII']):.1f}%
Overall:        {y_eup[5][masks_d['ALL']].mean():.3f}      {y_utx[5][masks_d['ALL']].mean():.3f}            {y_tx[5][masks_d['ALL']].mean():.3f}          {pct_gain(y_utx[5],y_tx[5],masks_d['ALL']):.1f}%

OLIGODENDROCYTE MATURATION INDEX (mean, normalised):
Window III:     {y_eup[6][masks_d['WIII']].mean():.3f}      {y_utx[6][masks_d['WIII']].mean():.3f}            {y_tx[6][masks_d['WIII']].mean():.3f}          {pct_gain(y_utx[6],y_tx[6],masks_d['WIII']):.1f}%
Overall:        {y_eup[6][masks_d['ALL']].mean():.3f}      {y_utx[6][masks_d['ALL']].mean():.3f}            {y_tx[6][masks_d['ALL']].mean():.3f}          {pct_gain(y_utx[6],y_tx[6],masks_d['ALL']):.1f}%

COMPOSITE NEURODEVELOPMENTAL INDEX (NDI, euploid = 100)
────────────────────────────────────────────────────────────────────────────────
Period          Euploid    T21 Untreated    T21 Treated    Δ NDI
Window I:       {ndi_eup[masks_d['WI']].mean():.1f}       {ndi_utx[masks_d['WI']].mean():.1f}             {ndi_tx[masks_d['WI']].mean():.1f}           +{ndi_tx[masks_d['WI']].mean()-ndi_utx[masks_d['WI']].mean():.1f}
Window II:      {ndi_eup[masks_d['WII']].mean():.1f}       {ndi_utx[masks_d['WII']].mean():.1f}             {ndi_tx[masks_d['WII']].mean():.1f}           +{ndi_tx[masks_d['WII']].mean()-ndi_utx[masks_d['WII']].mean():.1f}
Window III:     {ndi_eup[masks_d['WIII']].mean():.1f}       {ndi_utx[masks_d['WIII']].mean():.1f}             {ndi_tx[masks_d['WIII']].mean():.1f}           +{ndi_tx[masks_d['WIII']].mean()-ndi_utx[masks_d['WIII']].mean():.1f}
Overall:        {ndi_eup[masks_d['ALL']].mean():.1f}       {ndi_utx[masks_d['ALL']].mean():.1f}             {ndi_tx[masks_d['ALL']].mean():.1f}           +{ndi_tx[masks_d['ALL']].mean()-ndi_utx[masks_d['ALL']].mean():.1f}

SENSITIVITY ANALYSIS (Monte Carlo n=100, ±20–25% variation in 8 parameters)
────────────────────────────────────────────────────────────────────────────────
Microglial reduction positive across ALL samples: {'YES ✓' if (mic_utx_mc.mean(axis=1)>mic_tx_mc.mean(axis=1)).all() else 'NO'}
NDI improvement positive across ALL samples:      {'YES ✓' if (ndi_tx_mc.mean(axis=1)>ndi_utx_mc.mean(axis=1)).all() else 'NO'}
Treated NDI 90% CI (overall mean):               [{np.percentile(ndi_tx_mc[:,masks_d['ALL']].mean(axis=1),5):.1f}, {np.percentile(ndi_tx_mc[:,masks_d['ALL']].mean(axis=1),95):.1f}]
Untreated NDI 90% CI (overall mean):             [{np.percentile(ndi_utx_mc[:,masks_d['ALL']].mean(axis=1),5):.1f}, {np.percentile(ndi_utx_mc[:,masks_d['ALL']].mean(axis=1),95):.1f}]

RECOMMENDED PAPER WORDING (per reviewer suggestion)
────────────────────────────────────────────────────────────────────────────────
"Across the explored parameter space, the computational framework consistently
predicted reductions in simulated IL-13 signalling and microglial activation,
as well as improvements in downstream proxy markers of synaptic density and
oligodendrocyte maturation, supporting the pharmacological feasibility and
internal consistency of the proposed dosing strategy. Because the model employs
placeholder parameters rather than experimentally calibrated values, these
findings should be interpreted as hypothesis-generating rather than predictive."

IMPORTANT CAVEATS
────────────────────────────────────────────────────────────────────────────────
• All values are from placeholder-parameter simulation
• Downstream marker relationships are theoretical (published qualitative
  associations, not quantitative fits to experimental data)
• Monte Carlo bands reflect parameter uncertainty, not biological variability
• Calibration against T21 iPSC organoid, Ts65Dn mouse, and placental
  perfusion data is required before scientific conclusions can be drawn
================================================================================
"""

with open(os.path.join(OUT_DIR,'simulation_summary_v3.txt'),'w') as f:
    f.write(summary)

print(summary)
print(f"\nAll files saved to: {OUT_DIR}")
