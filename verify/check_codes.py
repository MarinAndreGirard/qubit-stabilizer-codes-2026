#!/usr/bin/env python3
"""Check every code in codes/codes.csv. Plain Python 3.8+, no packages.

For each code it checks:
  1. the stabilizer file is unchanged (sha256 listed in codes.csv);
  2. there are n-k generators, each with n X bits and n Z bits;
  3. the generators are linearly independent over GF(2), so the code encodes exactly k qubits;
  4. the generators commute pairwise (symplectic inner product 0), so they define a stabilizer group;
  5. the listed logical operator (codes/witnesses.csv) has weight exactly d, commutes with every
     generator and is not in the stabilizer group, so the distance is AT MOST d.

The matching lower bound (no nontrivial logical of weight below d) is a separate, expensive computation:
see certify_distance.py.

Usage: python3 verify/check_codes.py [--codes codes/codes.csv]
Exit status 0 only if every check passes.
"""
import argparse, csv, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def popcount(v):
    return bin(v).count('1')


def read_stabilizers(path, n):
    """Return generators as pairs of ints (x, z); bit i of x is the X part on qubit i."""
    gens = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        xs, zs = line.split('|')
        assert len(xs) == n and len(zs) == n, f'{path}: row length'
        assert set(xs + zs) <= {'0', '1'}, f'{path}: non-binary entry'
        gens.append((int(xs[::-1], 2), int(zs[::-1], 2)))
    return gens


def parse_pauli(s, n):
    x = z = 0
    for tok in s.split():
        p, q = tok[0], int(tok[1:])
        assert p in 'XYZ' and 0 <= q < n and not ((x | z) >> q) & 1, f'bad or repeated Pauli {tok}'
        if p in 'XY':
            x |= 1 << q
        if p in 'ZY':
            z |= 1 << q
    return x, z


def commute(a, b):
    """Symplectic inner product of Paulis a=(ax,az), b=(bx,bz): 0 means they commute."""
    return (popcount(a[0] & b[1]) + popcount(a[1] & b[0])) & 1


class Span:
    """Row space over GF(2) of 2n-bit vectors (x | z << n), kept in echelon form keyed by leading bit."""

    def __init__(self, n):
        self.n, self.rows = n, {}

    def reduce(self, v):
        while v:
            top = v.bit_length() - 1
            if top not in self.rows:
                return v
            v ^= self.rows[top]
        return 0

    def add(self, p):
        v = self.reduce(p[0] | (p[1] << self.n))
        if v:
            self.rows[v.bit_length() - 1] = v
        return v != 0

    def contains(self, p):
        return self.reduce(p[0] | (p[1] << self.n)) == 0


def check(row, base, witnesses):
    n, k, d = int(row['n']), int(row['k']), int(row['d'])
    path = os.path.join(base, row['stabilizer_file'])
    errors = []
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != row['stabilizer_sha256']:
        errors.append('sha256 mismatch')
    gens = read_stabilizers(path, n)
    if len(gens) != n - k:
        errors.append(f'{len(gens)} generators, expected n-k={n - k}')
    span = Span(n)
    independent = all(span.add(g) for g in gens)
    if not independent:
        errors.append('generators are linearly dependent')
    bad_pairs = sum(commute(gens[i], gens[j]) for i in range(len(gens)) for j in range(i + 1, len(gens)))
    if bad_pairs:
        errors.append(f'{bad_pairs} anticommuting generator pairs')
    w = witnesses.get(row['id'])
    if w is None:
        errors.append('no witness')
    else:
        L = parse_pauli(w['logical'], n)
        weight = popcount(L[0] | L[1])
        if weight != d:
            errors.append(f'witness weight {weight} != d={d}')
        if any(commute(L, g) for g in gens):
            errors.append('witness anticommutes with a generator')
        if span.contains(L):
            errors.append('witness is a stabilizer')
    return dict(id=row['id'], n=n, k=k, d=d, ok=not errors, errors=errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default=os.path.join(HERE, '..', 'codes', 'codes.csv'))
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(args.codes))
    rows = list(csv.DictReader(open(args.codes)))
    witnesses = {r['id']: r for r in csv.DictReader(open(os.path.join(base, 'witnesses.csv')))}
    results = [check(r, base, witnesses) for r in rows]
    for r in results:
        if not r['ok']:
            print(json.dumps(r))
    good = sum(r['ok'] for r in results)
    print(f'{good}/{len(results)} codes pass: valid [[n,k]] stabilizer code and a weight-d logical (so distance <= d).')
    sys.exit(0 if good == len(results) else 1)


if __name__ == '__main__':
    main()
