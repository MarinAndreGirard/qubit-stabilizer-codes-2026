#!/usr/bin/env python3
"""Tests for certify_distance.py against an independent brute-force distance computation.

For every test code with known distance d:
  * the certifier must PASS at D = d, and
  * it must FAIL at D = d+1, reporting a logical of weight exactly d
    (this shows the certifier is not vacuous: it does find low-weight logicals when they exist).

Test codes:
  1. textbook codes: [[4,2,2]], [[6,4,2]], [[5,1,3]], Steane [[7,1,3]], Shor [[9,1,3]] (degenerate),
     rotated surface codes [[9,1,3]] and [[25,1,5]] (degenerate, many low-weight stabilizers);
  2. 60 random stabilizer codes with n = 8..14 (random Clifford circuits), distance by brute force,
     each also run with forced 2, 4 and 8 bucket passes;
  3. large-index / long-syndrome codes: a record code from codes/ placed next to a small known code
     (direct sum), with qubits shuffled and single-qubit Cliffords applied. The direct sum's distance is
     the smaller of the two, so a known low-weight logical sits among up to 255 qubits and, in one case,
     behind more than 64 syndrome bits.

Usage: python3 verify/test_certify.py
"""
import itertools, os, random, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certify_distance import certify          # noqa: E402
from check_codes import Span, commute         # noqa: E402

STAB = os.path.join(HERE, '..', 'codes', 'stabilizers')


# ---------------------------------------------------------------- codes as lists of (x, z) int pairs

def from_strings(n, rows):
    """rows like 'XZZXI' -> (x, z) ints."""
    out = []
    for s in rows:
        x = sum(1 << i for i, c in enumerate(s) if c in 'XY')
        z = sum(1 << i for i, c in enumerate(s) if c in 'ZY')
        out.append((x, z))
    return out


def surface(L):
    """Rotated surface code [[L*L, 1, L]]."""
    q = lambda r, c: r * L + c
    rows = []
    for r in range(-1, L):
        for c in range(-1, L):
            cells = [(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1) if 0 <= r + dr < L and 0 <= c + dc < L]
            if len(cells) < 2:
                continue
            kind = 'X' if (r + c) % 2 == 0 else 'Z'
            if len(cells) == 2:   # boundary checks: X on top/bottom, Z on left/right
                if (r in (-1, L - 1) and kind == 'Z') or (c in (-1, L - 1) and kind == 'X'):
                    continue
                if r in (-1, L - 1) and c in (-1, L - 1):
                    continue
            s = ['I'] * (L * L)
            for rr, cc in cells:
                s[q(rr, cc)] = kind
            rows.append(''.join(s))
    return L * L, from_strings(L * L, rows)


TEXTBOOK = {
    '[[4,2,2]]': (4, from_strings(4, ['XXXX', 'ZZZZ']), 2),
    '[[6,4,2]]': (6, from_strings(6, ['XXXXXX', 'ZZZZZZ']), 2),
    '[[5,1,3]]': (5, from_strings(5, ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']), 3),
    'Steane [[7,1,3]]': (7, from_strings(7, ['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']), 3),
    'Shor [[9,1,3]]': (9, from_strings(9, ['ZZIIIIIII', 'IZZIIIIII', 'IIIZZIIII', 'IIIIZZIII', 'IIIIIIZZI',
                                           'IIIIIIIZZ', 'XXXXXXIII', 'IIIXXXXXX']), 3),
    'surface [[9,1,3]]': surface(3) + (3,),
    'surface [[25,1,5]]': surface(5) + (5,),
}


def random_code(n, k, rng):
    gens = [(0, 1 << i) for i in range(n - k)]      # start from Z_0 .. Z_{n-k-1}, then scramble
    for _ in range(6 * n * n):
        g = rng.randrange(3)
        i, j = rng.sample(range(n), 2)
        new = []
        for x, z in gens:
            xi, zi = (x >> i) & 1, (z >> i) & 1
            if g == 0:                                # Hadamard on i
                x ^= (xi ^ zi) << i
                z ^= (xi ^ zi) << i
            elif g == 1:                              # phase on i
                z ^= xi << i
            else:                                     # CNOT i -> j
                x ^= xi << j
                z ^= ((z >> j) & 1) << i
            new.append((x, z))
        gens = new
    return gens


def brute_distance(n, gens, dmax=8):
    span = Span(n)
    for g in gens:
        assert span.add(g)
    for w in range(1, dmax + 1):
        for supp in itertools.combinations(range(n), w):
            for ps in itertools.product((1, 2, 3), repeat=w):
                x = sum(1 << q for q, p in zip(supp, ps) if p & 1)
                z = sum(1 << q for q, p in zip(supp, ps) if p & 2)
                if not any(commute((x, z), g) for g in gens) and not span.contains((x, z)):
                    return w
    return None


def direct_sum(code_a, code_b, rng):
    """[[na,ka,da]] (+) [[nb,kb,db]], qubits shuffled, random single-qubit Cliffords: distance min(da, db)."""
    (na, ga), (nb, gb) = code_a, code_b
    n = na + nb
    gens = ga + [(x << na, z << na) for x, z in gb]
    perm = list(range(n))
    rng.shuffle(perm)
    local = [rng.randrange(6) for _ in range(n)]     # the 6 single-qubit Cliffords up to phase

    def act(x, z):
        X = Z = 0
        for i in range(n):
            a, b = (x >> i) & 1, (z >> i) & 1
            for _ in range(local[i] % 3):           # cycle X -> Z -> Y -> X  (a,b) -> (a^b, a)... order 3
                a, b = b ^ a, a
            if local[i] >= 3:                        # then Hadamard
                a, b = b, a
            X |= a << perm[i]
            Z |= b << perm[i]
        return X, Z
    return n, [act(x, z) for x, z in gens]


def read_stab(name):
    rows = [l.strip() for l in open(os.path.join(STAB, name)) if l.strip() and not l.startswith('#')]
    n = len(rows[0].split('|')[0])
    return n, [(int(xs[::-1], 2), int(zs[::-1], 2)) for xs, zs in (r.split('|') for r in rows)]


def write_tmp(n, gens, d, k=None):
    f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    f.write(f'# [[{n},{k if k is not None else n - len(gens)},{d}]] test code\n')
    for x, z in gens:
        f.write(''.join(str((x >> i) & 1) for i in range(n)) + '|' + ''.join(str((z >> i) & 1) for i in range(n)) + '\n')
    f.close()
    return f.name


FAILS = []


def expect(name, n, gens, d, pbits_list=(None,)):
    path = write_tmp(n, gens, d)
    for pb in pbits_list:
        ok = certify(path, d, force_pbits=pb, log=lambda s: None)
        bad = certify(path, d + 1, force_pbits=pb, log=lambda s: None) if d + 1 <= 13 else None
        good = ok['passed'] and (bad is None or (not bad['passed'] and bad['logical_found']
                                                 and len(bad['logical_found'].split()) == d))
        tag = f'{name} n={n} d={d}' + (f' buckets=2^{pb}' if pb else '')
        print(('ok   ' if good else 'FAIL ') + tag, flush=True)
        if not good:
            FAILS.append((tag, ok, bad))
    os.unlink(path)


def main():
    rng = random.Random(2026)
    print('-- textbook codes (distance checked by brute force)')
    for name, (n, gens, d) in TEXTBOOK.items():
        bd = brute_distance(n, gens)
        assert bd == d, (name, bd)
        expect(name, n, gens, d, (None, 1, 3))

    print('-- random codes, n = 8..14')
    for t in range(60):
        n = rng.randrange(8, 15)
        k = rng.randrange(1, 4)
        d = 1
        while d < 2:                                   # distance-1 codes are outside the certifier's range
            gens = random_code(n, k, rng)
            d = brute_distance(n, gens)
        expect(f'random#{t} [[{n},{k}]]', n, gens, d, (None, 1, 2, 3))

    print('-- record code (+) small code: large qubit indices, long syndromes')
    big = read_stab('n250_k196_d09.txt')                  # 54 generators
    expect('[[250,196,9]] (+) [[5,1,3]]', *direct_sum(big, TEXTBOOK['[[5,1,3]]'][:2], rng), 3, (None, 2))
    big = read_stab('n227_k174_d09.txt')                  # 53 generators; + 24 -> 77 > 64 syndrome bits
    expect('[[227,174,9]] (+) surface [[25,1,5]]', *direct_sum(big, TEXTBOOK['surface [[25,1,5]]'][:2], rng), 5, (None, 2))

    print(f'\n{"ALL TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILURES"}')
    for f in FAILS:
        print(f)
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
