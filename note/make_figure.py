#!/usr/bin/env python3
"""Figure for the note: for each distance d = 5..13, the number of stabilizer generators n - k needed at length n by
  - the quantum Singleton bound, n - k >= 2(d - 1);
  - a random code (first-moment / Gilbert-Varshamov level), n - k = log2 sum_{w=1}^{d-1} C(n,w) 3^w;
  - the best code in Grassl's table (largest k with lower bound >= d);
  - the first-moment prediction for random encoders + expurgation (predicted_gains.csv, from predict.py);
  - the 192 new codes.
Lower is better; at fixed (n, d) a vertical step of one is one logical qubit.
Reads ../codes/codes.csv, ../comparison/codetables-2026-11-04.csv and predicted_gains.csv; writes fig-codes.pdf.
Usage: python3 make_figure.py [k]   ('k' plots k instead of n - k, to fig-codes-k.pdf; the gaps are hard to see)
"""
import csv, math, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
PLOT_K = len(sys.argv) > 1 and sys.argv[1] == 'k'


def log2S(n, d):
    return math.log2(sum(math.comb(n, w) * 3 ** w for w in range(1, d)))


codes = [(int(r['n']), int(r['k']), int(r['d'])) for r in csv.DictReader(open(os.path.join(HERE, '..', 'codes', 'codes.csv')))]
lower = {(int(r['n']), int(r['k'])): int(r['lower'])
         for r in csv.DictReader(open(os.path.join(HERE, '..', 'comparison', 'codetables-2026-11-04.csv')))}
pred = {(int(r['n']), int(r['d'])): int(r['predicted_k'])
        for r in csv.DictReader(open(os.path.join(HERE, 'predicted_gains.csv'))) if r['predicted_k']}

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
y = (lambda n, k: k) if PLOT_K else (lambda n, k: n - k)
fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.0), sharex=True)
for ax, d in zip(axes.flat, range(5, 14)):
    ns = list(range(max(2 * d, 30), 257))
    ax.plot(ns, [y(n, n - 2 * (d - 1)) for n in ns], color=MUTED, lw=0.8, label='quantum Singleton bound')
    ax.plot(ns, [y(n, n - log2S(n, d)) for n in ns], color=MUTED, lw=0.9, ls=(0, (1, 1.5)),
            label='random code (GV level)')
    tn, tk = [], []
    for n in ns:
        k = max([k for k in range(1, n + 1) if lower[(n, k)] >= d], default=0)
        if k:
            tn.append(n); tk.append(y(n, k))
    ax.plot(tn, tk, color=BLUE, lw=0.8, label='codetables.de, best known')
    pn = [n for n in ns if (n, d) in pred and n >= 50]
    ax.plot(pn, [y(n, pred[(n, d)]) for n in pn], color=ORANGE, lw=0.9, ls=(0, (4, 2)),
            label='predicted for this method')
    ours = [(n, y(n, k)) for n, k, dd in codes if dd == d]
    ax.scatter([p[0] for p in ours], [p[1] for p in ours], s=13, color=ORANGE, edgecolor='white', linewidth=0.5,
               label='new codes (this work)', zorder=4)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.04 if PLOT_K else 0.04, 0.95, f'$d={d}$', transform=ax.transAxes, va='top', color=INK)
    ax.set_xlim(25, 260)
    if PLOT_K:
        ax.set_ylim(0, 256 - 2 * (d - 1))
    else:
        ax.set_ylim(0, max(tk + [y(n, n - log2S(n, d)) for n in ns]) * 1.05)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel('length $n$')
for ax in axes[:, 0]:
    ax.set_ylabel('$k$' if PLOT_K else 'generators $n-k$', fontsize=11 if PLOT_K else 8)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=2.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
name = 'fig-codes-k' if PLOT_K else 'fig-codes'
fig.savefig(os.path.join(HERE, name + '.pdf'))
fig.savefig(os.path.join(HERE, name + '.png'), dpi=160)
print('wrote', name + '.pdf')
