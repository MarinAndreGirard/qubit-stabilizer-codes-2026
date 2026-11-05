#!/usr/bin/env python3
"""Compare the codes with a snapshot of the codetables.de qubit table. Plain Python, no packages.

Input:  ../codes/codes.csv and codetables-2026-11-04.csv (all cells n=1..256, k=0..n, read 2026-11-04 from
        https://codetables.de/QECC/Tables_color.php?q=4&n0=..&n1=..&k0=0&k1=256; identical to a read on 2026-10-24).
Output: direct.csv          each code against the table cell at its own (n,k)
        raised_entries.csv  every table cell whose lower bound our codes raise
        summary.json        the counts quoted in the README

Propagation uses only two rules, both valid for any stabilizer code with k >= 1:
  [[n,k,d]] -> [[n+1,k,d]]   (add a qubit in a fixed state)
  [[n,k,d]] -> [[n,k-1,d]]   (make one logical operator a stabilizer)
so a code [[n0,k0,d0]] gives distance >= d0 at every cell n >= n0, 1 <= k <= k0. Cells with k = 0 are left
out (the table's k = 0 entries use a different distance convention).
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
codes = [dict(id=r['id'], n=int(r['n']), k=int(r['k']), d=int(r['d']))
         for r in csv.DictReader(open(os.path.join(HERE, '..', 'codes', 'codes.csv')))]
table = {(int(r['n']), int(r['k'])): (int(r['lower']), int(r['upper']))
         for r in csv.DictReader(open(os.path.join(HERE, 'codetables-2026-11-04.csv')))}
NMAX = 256


def table_kmax(n, d):
    """Largest k >= 1 at length n whose table lower bound is at least d (0 if none)."""
    return max([k for k in range(1, n + 1) if table[(n, k)][0] >= d], default=0)


# 1. each code at its own (n,k)
direct = []
for c in codes:
    lo, hi = table[(c['n'], c['k'])]
    kt = table_kmax(c['n'], c['d'])
    direct.append(dict(id=c['id'], n=c['n'], k=c['k'], d=c['d'], table_lower=lo, table_upper=hi,
                       gain_in_d=c['d'] - lo, table_kmax_at_d=kt, gain_in_k=c['k'] - kt,
                       within_upper_bound=c['d'] <= hi))

# 2. codes not implied by another of our codes through the two rules
def implies(a, b):
    return a is not b and a['n'] <= b['n'] and a['k'] >= b['k'] and a['d'] >= b['d']
nondominated = [c for c in codes if not any(implies(a, c) for a in codes)]

# 3. every table cell raised by propagation
raised = []
for n in range(1, NMAX + 1):
    for k in range(1, n + 1):
        best = max(((c['d'], c['id']) for c in codes if c['n'] <= n and c['k'] >= k), default=None)
        if best and best[0] > table[(n, k)][0]:
            raised.append(dict(n=n, k=k, table_lower=table[(n, k)][0], new_lower=best[0],
                               table_upper=table[(n, k)][1], gain_in_d=best[0] - table[(n, k)][0],
                               from_code=best[1], within_upper_bound=best[0] <= table[(n, k)][1]))

for name, rows in (('direct.csv', direct), ('raised_entries.csv', raised)):
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

gains = {}
for r in direct:
    gains[r['gain_in_d']] = gains.get(r['gain_in_d'], 0) + 1
rgains = {}
for r in raised:
    rgains[r['gain_in_d']] = rgains.get(r['gain_in_d'], 0) + 1
summary = dict(
    codes=len(codes), n_range=[min(c['n'] for c in codes), max(c['n'] for c in codes)],
    d_range=[min(c['d'] for c in codes), max(c['d'] for c in codes)],
    direct_gain_in_d=dict(sorted(gains.items())),
    direct_not_above_table=sum(r['gain_in_d'] <= 0 for r in direct),
    direct_above_upper_bound=sum(not r['within_upper_bound'] for r in direct),
    nondominated=len(nondominated),
    raised_entries=len(raised), raised_gain_in_d=dict(sorted(rgains.items())),
    raised_above_upper_bound=sum(not r['within_upper_bound'] for r in raised),
    table_snapshot='codetables-2026-11-04.csv')
json.dump(summary, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
