# Trisomy 21 IL-13 Blockade — Computational Simulation

**Companion code to:**
> *"Neuroinflammatory Vulnerability Windows in Trisomy 21: A Theoretical Framework for Extended-Interval IL-13/TSLP Biologic Intervention During Fetal Neurodevelopment"*
> Leon Sandler (pen name: Leonid Sandler)
> Zenodo Preprint, June 2026
> DOI: *[assigned upon Zenodo upload]*

**Author contact:** leonsandler@alumni.swinburne.edu
**LinkedIn:** https://www.linkedin.com/in/-leon-sandler

---

## What This Is

This repository contains a fully executable Python simulation that models two linked biological processes:

1. **Pharmacokinetics (PBPK):** How a subcutaneous injection of a zumilokibart-class IL-13 antibody moves from maternal circulation through the placenta (via FcRn receptors) into the fetal compartment over gestational weeks 8–40.

2. **Pharmacodynamics (ODE system):** How fetal IL-13, TSLP, JAK/STAT, and microglial activation respond to that drug concentration in three scenarios: euploid (normal), trisomy 21 untreated, and trisomy 21 + IL-13 blockade.

The simulation produces **5 publication-quality figures** and a **quantitative summary table** directly referenced in the companion paper.

> ⚠️ **All kinetic parameters are literature-derived placeholders.** This is a hypothesis-testing scaffold, not a predictive clinical model. See the Disclaimer section below.

---

## Quick Start

### Requirements

- Python ≥ 3.8
- numpy
- scipy
- matplotlib

### Install dependencies

```bash
pip install numpy scipy matplotlib
```

### Run the simulation

```bash
python Trisomy21_PBPK_Simulation_v2.py
```

This will generate the following files in the same directory:

| File | Description |
|------|-------------|
| `Fig1_Placental_Permeation.png` | Maternal vs fetal drug concentration with FcRn efficiency |
| `Fig2_Window_Coverage.png` | Fetal antibody levels vs three vulnerability windows |
| `Fig3_Microglial_Suppression.png` | Microglial activation: T21 treated vs untreated vs euploid |
| `Fig4_FcRn_Transfer_Dynamics.png` | FcRn efficiency curve and fetal/maternal ratio over gestation |
| `Fig5_Sensitivity_Analysis.png` | Monte Carlo uncertainty bands (n=80, ±20–25% parameter variation) |
| `simulation_summary.txt` | Quantitative results table |

Runtime: approximately 30–60 seconds on a standard laptop.

---

## Simulation Architecture

```
Layer 1: PBPK Model
  ┌─────────────────────────────────────────────────────────────┐
  │  Subcut depot → Maternal central → [FcRn placenta] → Fetal  │
  │  Dosing: weeks 12, 24, 32  |  Half-life: ~42 days (IgG4)   │
  └─────────────────────────────────────────────────────────────┘
                          ↓ fetal drug concentration C(t)
Layer 2: Neuroinflammatory ODE System
  ┌─────────────────────────────────────────────────────────────┐
  │  IL-13 → [Antibody neutralisation] → JAK/STAT → Microglia  │
  │  TSLP  ────────────────────────────────────────────────────┘│
  │  T21 state: 1.65× IL-13, 1.45× TSLP production             │
  └─────────────────────────────────────────────────────────────┘
                          ↓ 3 scenarios × 5 state variables
Layer 3: Sensitivity Analysis
  ┌─────────────────────────────────────────────────────────────┐
  │  Monte Carlo: 80 samples, ±20–25% variation in 6 parameters│
  │  Output: 5th–95th percentile bands on microglial activation │
  └─────────────────────────────────────────────────────────────┘
```

---

## Key Equations

### FcRn Placental Transfer (Sigmoid Model)
```
E(t) = E_max / (1 + exp(-k × (t - t_mid)))
```
Where:
- `E_max` = 0.83 (maximum transfer efficiency, IgG4)
- `t_mid` = 196 days (week 28)
- `k` = 0.048

Source: Palmeira et al. (2012); Roopenian & Akilesh (2007)

### IL-13 Neutralisation (Competitive Binding)
```
d[IL-13]/dt = k_prod - k_cl·[IL-13] - k_on·[IL-13]·[Ab] + k_off·[Complex]
d[Complex]/dt = k_on·[IL-13]·[Ab] - k_off·[Complex] - k_deg·[Complex]
```

### Microglial Activation
```
d[JAK]/dt  = k_jak_il13·[IL-13] + k_jak_tslp·[TSLP] - k_cl_jak·[JAK]
d[Mic]/dt  = k_mic_jak·[JAK] - k_res·[Mic]
```
Where `k_res` is reduced by 40% in the T21 state vs euploid.

---

## Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Antibody molecular weight | ~150 kDa | IgG4 class |
| Subcut bioavailability | 72% | Zou et al. 2012 |
| Maternal volume of distribution | 8,500 mL | Dirks & Meibohm 2010 |
| Extended half-life (maternal) | 42 days | Robbie et al. 2013 |
| FcRn max transfer efficiency | 83% | Palmeira et al. 2012 |
| FcRn sigmoid midpoint | Week 28 | Roopenian & Akilesh 2007 |
| IL-13 production T21 multiplier | 1.65× | Rachubinski et al. 2021 |
| TSLP production T21 multiplier | 1.45× | Estimated |
| Microglial resolution reduction (T21) | -40% | Flores-Aguilar et al. 2020 |
| IL-13 receptor KD | ~0.5–5 nM | IL-13Rα1 literature |

---

## How to Calibrate with Real Data

This simulation is designed to be iteratively refined. When empirical data becomes available, replace parameters in these steps:

**Step 1 — Zumilokibart PK calibration**
When AbbVie publishes zumilokibart Phase 2 PK data, replace:
```python
T_HALF_MATERNAL = 42.0  # → replace with published value
K_ABSORB        = 0.045 # → replace with published value
BIOAVAILABILITY = 0.72  # → replace with published value
```

**Step 2 — FcRn transfer calibration**
Run ex vivo placental perfusion with zumilokibart and fit:
```python
FCRT_MAX       = 0.83   # → replace with measured transfer efficiency
FCRT_MIDPOINT  = 196.0  # → adjust if T21 placentae differ from euploid
```

**Step 3 — T21 inflammatory parameter calibration**
Quantify IL-13 and TSLP in T21 iPSC organoid supernatants:
```python
T21_IL13 = 1.65   # → replace with measured ratio vs euploid organoids
T21_TSLP = 1.45   # → replace with measured ratio
K_CL_MIC = 0.08   # → fit to microglial time-course data from Ts65Dn mice
```

**Step 4 — Rerun and compare**
```bash
python Trisomy21_PBPK_Simulation_v2.py
```
Compare output figures with experimental curves to validate model fit.

---

## Integration with Open Systems Pharmacology (OSP) Suite

For a more rigorous PBPK model, this simulation can be extended using the free Open Systems Pharmacology Suite:

1. Download: https://www.open-systems-pharmacology.org
2. Load the **Pregnancy PBPK** building block from the OSP library
3. Input zumilokibart compound properties (MW, lipophilicity, FcRn KD)
4. Export `C_fetal_brain(t)` as CSV
5. Load into this script:

```python
import pandas as pd
pk_data = pd.read_csv('osp_fetal_output.csv')
t_from_osp   = pk_data['time_days'].values
conc_from_osp = pk_data['fetal_conc'].values

# In run_neuro(), replace np.interp(t, t_days, fetal_conc)
# with np.interp(t, t_from_osp, conc_from_osp)
```

---

## Simulation Results Summary

From the current placeholder-parameter run:

| Metric | Value |
|--------|-------|
| FcRn efficiency at Dose 1 (wk 12) | ~0.4% |
| FcRn efficiency at Dose 2 (wk 24) | ~17% |
| FcRn efficiency at Dose 3 (wk 32) | ~66% |
| Fetal/maternal ratio at week 32 | ~1.1 |
| Microglial reduction — Window I | ~0% (indirect only) |
| Microglial reduction — Window II | ~5% |
| Microglial reduction — Window III | ~19% |
| Microglial reduction — Overall | ~11% |
| IL-13 free concentration reduction | ~17% |
| Reduction positive across all sensitivity samples | YES |

> The low Window I reduction is **expected and correct** — FcRn transfer is minimal at week 12, so dose 1 acts primarily via maternal cytokine reduction rather than direct fetal drug exposure. This is explicitly discussed in the paper.

---

## Disclaimer

**This simulation does not constitute evidence of clinical efficacy or safety.**

All parameters are placeholder estimates derived from published literature on IgG4 pharmacokinetics and IL-13 signalling biology. The model has not been validated against experimental T21-specific data. Outputs should be interpreted as demonstrating the **structural plausibility** of the dosing framework only.

The following are required before any scientific conclusions can be drawn:
- Calibration against T21 iPSC organoid IL-13 quantification
- Calibration against Ts65Dn / Dp16 mouse PK-PD experiments
- Ex vivo placental perfusion data for zumilokibart
- Published zumilokibart clinical PK parameters

---

## Citation

If you use this code, please cite:

```
Sandler, L. (2026). Neuroinflammatory Vulnerability Windows in Trisomy 21:
A Theoretical Framework for Extended-Interval IL-13/TSLP Biologic 
Intervention During Fetal Neurodevelopment.
Zenodo. DOI: [assigned upon upload]

Sandler, L. (2026). Trisomy21-IL13-Simulation [Software].
GitHub: https://github.com/leonsandler/trisomy21-il13-simulation
Zenodo. DOI: [assigned upon upload]
```

---

## Related Preprints by the Same Author

- GaaR: Goal-Aware Adaptive Regulation in Neural Architecture Search. DOI: 10.5281/zenodo.20737711
- SurfacePower: Graphene-Based Low-Voltage Electrical Distribution. DOI: 10.5281/zenodo.20576310
- Topological Cooper Scaffold. DOI: 10.5281/zenodo.20821704

---

## License

MIT License — free to use, modify, and distribute with attribution.

```
Copyright (c) 2026 Leon Sandler
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to deal in the Software without restriction, subject to the
condition that the above copyright notice and this permission notice appear
in all copies.
```
