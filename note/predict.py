#!/usr/bin/env python3
"""Predict where random encoders + expurgation should beat the table, using the first-moment model of the note.

For a target distance D and parent dimension K, the expected number of logical operators of weight < D in a random
[[n,K]] code is m = E(n,K,D). The smallest q with 2Kq - 1.5 q^2 + m log2(1 - 2^-q) >= 0 is q_first; the model uses
q = q_first + 1 (at most 8) and predicts k = max_K (K - q) - 1. This calibration was fixed on earlier runs and, before
the d = 12-13 searches, predicted their k exactly; this script reproduces 52 of the 55 dated forecasts exactly and the
other 3 within one.

Output: predicted_gains.csv with, for every d = 5..13 and n = 40..256, the table's best k, the best k implied by our
192 codes, the predicted k, and the cost of certifying the predicted code (number of Pauli operators to enumerate).
Plain Python, no packages.
"""
import csv, os
from math import comb, log2

HERE = os.path.dirname(os.path.abspath(__file__))


def S(n, d):
    return sum(comb(n, w) * 3 ** w for w in range(1, d))


def q_first(m, K):
    for q in range(1, 30):
        if 2 * K * q - 1.5 * q * q + m * log2(1 - 2.0 ** -q) >= 0:
            return q
    return None


def predict(n, D, qmax=8):
    best, s = None, S(n, D)
    for K in range(1, n):
        q = q_first(s * 2.0 ** (K - n), K)
        if q is None or q + 1 > qmax:
            continue
        if best is None or K - (q + 1) > best[0]:
            best = (K - (q + 1), K, q + 1)
    return (best[0] - 1, best[1], best[2]) if best else (None, None, None)


def cert_ops(n, D):
    b = (D - 1) // 2
    a = D - 1 - b
    return sum(comb(n, w) * 3 ** w for w in range(b + 1)) + (comb(n, a) * 3 ** a if a > b else 0)


def main():
    lower = {}
    for r in csv.DictReader(open(os.path.join(HERE, '..', 'comparison', 'codetables-2026-11-04.csv'))):
        lower[(int(r['n']), int(r['k']))] = int(r['lower'])
    ours = [(int(r['n']), int(r['k']), int(r['d'])) for r in csv.DictReader(open(os.path.join(HERE, '..', 'codes', 'codes.csv')))]
    rows = []
    for d in range(5, 14):
        for n in range(40, 257):
            tk = max([k for k in range(1, n + 1) if lower[(n, k)] >= d], default=0)
            ok = max([k0 for n0, k0, d0 in ours if n0 <= n and d0 >= d], default=0)
            pk, K, q = predict(n, d)
            gain = (pk - max(tk, ok)) if pk is not None else None
            rows.append(dict(d=d, n=n, table_k=tk, ours_k=ok, predicted_k=pk if pk is not None else '',
                             parent_K=K or '', q=q or '', predicted_further_gain=gain if gain is not None else '',
                             certification_paulis=cert_ops(n, d)))
    with open(os.path.join(HERE, 'predicted_gains.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('d  cells_with_predicted_gain  n_range  max_gain  cells_cert<=1e14')
    for d in range(5, 14):
        g = [r for r in rows if r['d'] == d and r['predicted_further_gain'] not in ('', None) and r['predicted_further_gain'] >= 1]
        if g:
            print(d, len(g), f"{min(r['n'] for r in g)}-{max(r['n'] for r in g)}", max(r['predicted_further_gain'] for r in g),
                  sum(r['certification_paulis'] <= 1e14 for r in g))
        else:
            print(d, 0)
    allg = [r for r in rows if r['predicted_further_gain'] not in ('', None) and r['predicted_further_gain'] >= 1]
    print('total cells', len(allg), 'with cert <= 1e14:', sum(r['certification_paulis'] <= 1e14 for r in allg))


if __name__ == '__main__':
    main()
