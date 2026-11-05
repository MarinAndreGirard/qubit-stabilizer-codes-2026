#!/usr/bin/env python3
"""Run certify_distance.py on as many codes as fit in a time budget, cheapest first.

The cost of each code is estimated from the number of Paulis to enumerate (constants measured on an
Apple M3, 8 threads, 12 GB budget); codes estimated above --max-minutes are skipped. Results are appended
to verify/results/certify_local.jsonl, and codes already there are not rerun.

Usage: python3 verify/run_local.py [--budget-minutes 5] [--max-minutes 10] [--mem-gb 12]
"""
import argparse, csv, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certify_distance import certify  # noqa: E402


def estimate_seconds(n, d, mem_gb):
    b = (d - 1) // 2
    a = d - 1 - b
    ball = sum(math.comb(n, w) * 3 ** w for w in range(b + 1))
    shell = math.comb(n, a) * 3 ** a if a > b else 0
    P = 1
    while ball / P * (8 if a == b else 32) * 1.15 > mem_gb * 2 ** 30:
        P *= 2
    return ball * 1.15e-7 + (P - 1) * ball * 8e-9 + shell * 2.3e-8 + (P - 1) * shell * 5e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-minutes', type=float, default=5.0)
    ap.add_argument('--max-minutes', type=float, default=10.0)
    ap.add_argument('--mem-gb', type=float, default=12.0)
    args = ap.parse_args()
    base = os.path.join(HERE, '..', 'codes')
    out = os.path.join(HERE, 'results', 'certify_local.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)['file'] for l in open(out)} if os.path.exists(out) else set()
    rows = list(csv.DictReader(open(os.path.join(base, 'codes.csv'))))
    plan = sorted(((estimate_seconds(int(r['n']), int(r['d']), args.mem_gb), r) for r in rows), key=lambda p: p[0])
    plan = [(e, r) for e, r in plan if e <= args.max_minutes * 60 and os.path.basename(r['stabilizer_file']) not in done]
    print(f'{len(plan)} codes planned (estimate {sum(e for e, _ in plan) / 3600:.1f} h, budget {args.budget_minutes:g} min)', flush=True)
    t0 = time.time()
    for est, r in plan:
        if time.time() - t0 + est > args.budget_minutes * 60:
            print('time budget reached', flush=True)
            break
        res = certify(os.path.join(base, r['stabilizer_file']), int(r['d']), mem_gb=args.mem_gb, log=lambda s: None)
        res['id'] = r['id']
        res['estimated_seconds'] = round(est, 1)
        with open(out, 'a') as f:
            f.write(json.dumps(res) + '\n')
        print(f"{r['id']:16s} [[{r['n']},{r['k']},{r['d']}]] {res['verdict']:22s} {res['seconds']:8.1f}s (est {est:.0f}s)", flush=True)
        if not res['passed']:
            print('!!! certificate FAILED for', r['id'], flush=True)


if __name__ == '__main__':
    main()
