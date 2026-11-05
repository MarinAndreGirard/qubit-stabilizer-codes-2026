#!/usr/bin/env python3
"""Independent lower-bound certifier for binary stabilizer codes.

Claim checked: the code has NO nontrivial logical operator of weight 1..D-1, i.e. distance >= D.
Together with a logical of weight exactly D (check_codes.py) this gives distance = D.

Method (exhaustive meet in the middle, written from scratch for this release; it shares no code with the
search or certification programs that produced the codes):

  Let S be the stabilizer group, given by n-k generators g_1..g_r in [X|Z] form. The syndrome of a Pauli E
  is the bit vector (whether E anticommutes with g_j)_j. A Pauli is in the normalizer N(S) iff its syndrome
  is zero; it is a nontrivial logical iff it is in N(S) but not in S.

  Put b = floor((D-1)/2) and a = D-1-b (a = b or a = b+1). Any Pauli E of weight w <= D-1 can be written as
  E = A*B with disjoint supports, wt(B) <= b and wt(A) <= a (B = E on its first min(w,b) support qubits,
  A = E on the rest). If E is in N(S) then syndrome(A) = syndrome(B).

  So it suffices to
    (1) enumerate the "ball" of all Paulis of weight <= b (identity included) and find every pair with
        equal syndromes, and
    (2) if a = b+1, enumerate the "shell" of all Paulis of weight exactly a and find every one whose
        syndrome equals that of some Pauli in the ball,
  and then check each colliding pair (A,B) exactly: the product A*B is a nonidentity element of N(S) of
  weight <= D-1, and it is either a stabilizer (harmless, counted) or a nontrivial logical (claim refuted,
  the operator is printed). The certificate passes iff no pair gives a nontrivial logical.

  Syndromes are compared through a 64-bit linear projection (XOR of the syndrome words). Equal syndromes
  always give equal projections, so no collision is missed; the rare projection-only matches are discarded
  by the exact check. To fit in memory, the projections are split into 2^p buckets by a hash, and the
  enumeration is repeated once per bucket (pairs with equal syndrome always fall in the same bucket).

  Completeness check: the program counts every Pauli it visits and compares the totals with the closed
  forms sum_{w<=b} C(n,w) 3^w and C(n,a) 3^a.

Requires numpy and numba. Limits: n <= 256, n-k <= 128, D <= 13.
Usage:  python3 verify/certify_distance.py codes/stabilizers/n053_k030_d06.txt [--D 6] [--mem-gb 8]
"""
import argparse, json, math, os, platform, sys, time
os.environ.setdefault('KMP_WARNINGS', '0')
import numpy as np
import numba
from numba import njit, prange

U64 = np.uint64
C1, C2 = U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB)
COUNT, FILL, PROBE, COLLECT = 0, 1, 2, 3


@njit(cache=True, inline='always')
def mix(k):
    """splitmix64 finalizer: spreads key bits for bucket and hash-slot selection."""
    k = (k ^ (k >> U64(30))) * C1
    k = (k ^ (k >> U64(27))) * C2
    return k ^ (k >> U64(31))


@njit(cache=True)
def in_table(table, mask, has_zero, key):
    if key == U64(0):
        return has_zero
    i = mix(key) & mask
    while table[i] != U64(0):
        if table[i] == key:
            return True
        i = (i + U64(1)) & mask
    return False


@njit(cache=True)
def in_sorted(arr, key):
    lo, hi = 0, arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo < arr.shape[0] and arr[lo] == key


@njit(cache=True)
def walk(n, c0, c1, i0, p0, wmin, wmax, pbits, bucket, mode,
         keys_out, off, table, mask, has_zero, sus, rec_pk, rec_key, t):
    """Depth-first enumeration of all Paulis whose lowest-index support qubit is i0 with Pauli p0
    (0=X, 1=Z, 2=Y), up to weight wmax. Nodes of weight >= wmin are 'visited'.
    Returns (records, visited)."""
    pos = np.empty(wmax, np.int64)
    pc = np.empty(wmax, np.int64)
    s0 = np.empty(wmax, U64)
    s1 = np.empty(wmax, U64)
    pk = np.empty(wmax, U64)
    shift = U64(64 - pbits)
    cap = rec_pk.shape[1]
    depth = 0
    pos[0] = i0
    pc[0] = p0
    s0[0] = c0[i0, p0]
    s1[0] = c1[i0, p0]
    pk[0] = U64((i0 << 2) | (p0 + 1))
    nrec = 0
    nvis = 0
    while True:
        if depth + 1 >= wmin:
            nvis += 1
            key = s0[depth] ^ s1[depth]
            if pbits == 0 or (mix(key) >> shift) == U64(bucket):
                if mode == COUNT:
                    nrec += 1
                elif mode == FILL:
                    keys_out[off + nrec] = key
                    nrec += 1
                else:
                    if mode == PROBE:
                        hit = in_table(table, mask, has_zero, key)
                    else:
                        hit = in_sorted(sus, key)
                    if hit:
                        if nrec < cap:
                            rec_pk[t, nrec] = pk[depth]
                            rec_key[t, nrec] = key
                        nrec += 1
        # go one level deeper if possible
        if depth + 1 < wmax and pos[depth] + 1 < n:
            depth += 1
            pos[depth] = pos[depth - 1] + 1
            pc[depth] = 0
        else:
            # otherwise move to the next sibling, backtracking as needed
            while True:
                if depth == 0:
                    return nrec, nvis
                if pc[depth] < 2:
                    pc[depth] += 1
                    break
                if pos[depth] + 1 < n:
                    pos[depth] += 1
                    pc[depth] = 0
                    break
                depth -= 1
        q, p = pos[depth], pc[depth]
        s0[depth] = s0[depth - 1] ^ c0[q, p]
        s1[depth] = s1[depth - 1] ^ c1[q, p]
        pk[depth] = pk[depth - 1] | (U64((q << 2) | (p + 1)) << U64(10 * depth))


@njit(parallel=True, cache=True)
def run_tasks(n, c0, c1, ti, tp, wmin, wmax, pbits, bucket, mode,
              keys_out, offsets, table, mask, has_zero, sus, rec_pk, rec_key):
    T = ti.shape[0]
    recs = np.zeros(T, np.int64)
    vis = np.zeros(T, np.int64)
    for t in prange(T):
        r, v = walk(n, c0, c1, ti[t], tp[t], wmin, wmax, pbits, bucket, mode,
                    keys_out, offsets[t], table, mask, has_zero, sus, rec_pk, rec_key, t)
        recs[t] = r
        vis[t] = v
    return recs, vis


@njit(cache=True)
def duplicates(sorted_keys):
    out = []
    i, m = 0, sorted_keys.shape[0]
    while i < m:
        j = i + 1
        while j < m and sorted_keys[j] == sorted_keys[i]:
            j += 1
        if j - i > 1:
            out.append(sorted_keys[i])
        i = j
    return out


@njit(cache=True)
def build_table(sorted_keys, cap):
    table = np.zeros(cap, U64)
    mask = U64(cap - 1)
    has_zero = False
    prev = U64(0)
    for idx in range(sorted_keys.shape[0]):
        k = sorted_keys[idx]
        if idx > 0 and k == prev:
            continue
        prev = k
        if k == U64(0):
            has_zero = True
            continue
        i = mix(k) & mask
        while table[i] != U64(0):
            i = (i + U64(1)) & mask
        table[i] = k
    return table, has_zero


# ---------------------------------------------------------------- exact arithmetic on Paulis (Python ints)

def read_code(path):
    gens = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith('#'):
            xs, zs = line.split('|')
            gens.append((xs, zs))
    n = len(gens[0][0])
    assert all(len(x) == n and len(z) == n for x, z in gens)
    return n, gens


def unpack(pk):
    """Packed Pauli -> (x, z) ints. Each support element uses 10 bits: qubit (8 bits) and Pauli (1=X,2=Z,3=Y)."""
    x = z = 0
    pk = int(pk)
    while pk:
        e = pk & 0x3FF
        q, p = e >> 2, e & 3
        if p & 1:
            x |= 1 << q
        if p & 2:
            z |= 1 << q
        pk >>= 10
    return x, z


def pauli_str(x, z, n):
    return ' '.join(('Y' if (z >> i) & 1 else 'X') + str(i) if (x >> i) & 1 else 'Z' + str(i)
                    for i in range(n) if ((x | z) >> i) & 1)


class Exact:
    def __init__(self, n, gens):
        self.n = n
        self.g = [(int(xs[::-1], 2), int(zs[::-1], 2)) for xs, zs in gens]
        self.rows = {}
        for gx, gz in self.g:
            v = self._reduce(gx | (gz << n))
            assert v, 'generators are linearly dependent'
            self.rows[v.bit_length() - 1] = v

    def _reduce(self, v):
        while v:
            top = v.bit_length() - 1
            if top not in self.rows:
                return v
            v ^= self.rows[top]
        return 0

    def classify(self, x, z):
        if any((bin(x & gz).count('1') + bin(z & gx).count('1')) & 1 for gx, gz in self.g):
            return 'not_in_normalizer'
        return 'stabilizer' if self._reduce(x | (z << self.n)) == 0 else 'logical'


# ---------------------------------------------------------------- driver

def syndrome_columns(n, gens):
    r = len(gens)
    assert r <= 128, 'n-k > 128 not supported'
    c0 = np.zeros((n, 3), U64)
    c1 = np.zeros((n, 3), U64)
    for j, (xs, zs) in enumerate(gens):
        word, bit = (c0, j) if j < 64 else (c1, j - 64)
        for i in range(n):
            # X_i anticommutes with g_j iff g_j has Z on i; Z_i iff g_j has X on i; Y_i iff exactly one.
            for p, hit in ((0, zs[i] == '1'), (1, xs[i] == '1'), (2, (xs[i] == '1') != (zs[i] == '1'))):
                if hit:
                    word[i, p] |= U64(1) << U64(bit)
    return c0, c1


def run_recording(kernel, T, cap, mem_gb):
    """Run a recording kernel with per-task buffers of `cap` entries; if any task found more, enlarge the
    buffers and run it again, so that no record is ever dropped."""
    while True:
        rec_pk = np.zeros((T, cap), U64)
        rec_key = np.zeros((T, cap), U64)
        hits, vis = kernel(rec_pk, rec_key)
        if hits.max() <= cap:
            return hits, vis, rec_pk, rec_key
        cap = 1 << int(hits.max() - 1).bit_length()
        if T * cap * 16 > mem_gb * 2 ** 30:
            raise RuntimeError(f'{int(hits.max())} records in one task exceed the memory budget')


def certify(path, D, mem_gb=8.0, threads=None, force_pbits=None, cap=256, log=print):
    t_start = time.time()
    n, gens = read_code(path)
    r = len(gens)
    assert 2 <= D <= 13 and n <= 256
    exact = Exact(n, gens)
    b = (D - 1) // 2
    a = D - 1 - b
    c0, c1 = syndrome_columns(n, gens)
    if threads:
        numba.set_num_threads(threads)
    numba.set_parallel_chunksize(1)
    rng = np.random.default_rng(12345)
    order = rng.permutation(3 * n)                     # shuffle tasks for load balance
    ti = (order // 3).astype(np.int64)
    tp = (order % 3).astype(np.int64)
    T = 3 * n
    ball = sum(math.comb(n, w) * 3 ** w for w in range(b + 1))
    shell = math.comb(n, a) * 3 ** a if a > b else 0
    bytes_per_key = 8 if a == b else 32
    pbits = 0
    while ball / 2 ** pbits * bytes_per_key * 1.15 > mem_gb * 2 ** 30:
        pbits += 1
    if force_pbits is not None:
        pbits = force_pbits
    P = 2 ** pbits
    log(f'n={n} k={n - r} D={D}: ball weight<={b} has {ball:,} Paulis; '
        + (f'shell weight={a} has {shell:,}; ' if shell else '') + f'{P} bucket pass(es), {numba.get_num_threads()} threads')

    dummy_keys = np.zeros(1, U64)
    no_table = np.zeros(1, U64)
    no_rec = np.zeros((T, 1), U64)
    zeros_off = np.zeros(T, np.int64)
    suspicious, probe_hits = [], []
    visited_ball = visited_shell = 0
    stats = dict(bucket_max=0)
    for bucket in range(P):
        zero_here = pbits == 0 or (int(mix(U64(0))) >> (64 - pbits)) == bucket
        if b > 0:
            counts, vis = run_tasks(n, c0, c1, ti, tp, 1, b, pbits, bucket, COUNT,
                                    dummy_keys, zeros_off, no_table, U64(0), False, dummy_keys, no_rec, no_rec)
            offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
            m = int(counts.sum())
        else:
            counts, vis, offsets, m = np.zeros(T, np.int64), np.zeros(T, np.int64), zeros_off, 0
        keys = np.empty(m + (1 if zero_here else 0), U64)
        if b > 0:
            filled, vis2 = run_tasks(n, c0, c1, ti, tp, 1, b, pbits, bucket, FILL,
                                     keys, offsets, no_table, U64(0), False, dummy_keys, no_rec, no_rec)
            assert np.array_equal(filled, counts) and np.array_equal(vis, vis2)
        if zero_here:
            keys[m] = U64(0)                            # the identity, weight 0
        visited_ball += int(vis.sum()) + (1 if zero_here else 0)
        stats['bucket_max'] = max(stats['bucket_max'], keys.shape[0])
        keys.sort()
        suspicious.extend(int(k) for k in duplicates(keys))
        if a > b:
            cap_t = 1 << max(4, math.ceil(math.log2(1.5 * max(1, keys.shape[0]))))
            table, has_zero = build_table(keys, cap_t)
            del keys
            hits, vis, rec_pk, rec_key = run_recording(
                lambda pk, kk: run_tasks(n, c0, c1, ti, tp, a, a, pbits, bucket, PROBE, dummy_keys, zeros_off,
                                         table, U64(cap_t - 1), has_zero, dummy_keys, pk, kk), T, cap, mem_gb)
            del table
            visited_shell += int(vis.sum())
            for t in np.flatnonzero(hits):
                for j in range(hits[t]):
                    probe_hits.append((int(rec_pk[t, j]), int(rec_key[t, j])))
                    suspicious.append(int(rec_key[t, j]))
        else:
            del keys
        if P > 1:
            log(f'  bucket {bucket + 1}/{P} done, {time.time() - t_start:.0f}s')

    # every Pauli must be visited exactly once per pass: identity once overall, the rest once per bucket
    complete = visited_ball == 1 + (ball - 1) * P and visited_shell == shell * P

    # resolve: list every ball element whose key is suspicious, then check all candidate pairs exactly
    sus = np.unique(np.array(suspicious, dtype=U64))
    by_key = {}
    if sus.shape[0]:
        if 0 in set(sus.tolist()):
            by_key.setdefault(0, []).append(0)
        if b > 0:
            hits, _, rec_pk, rec_key = run_recording(
                lambda pk, kk: run_tasks(n, c0, c1, ti, tp, 1, b, 0, 0, COLLECT, dummy_keys, zeros_off,
                                         no_table, U64(0), False, sus, pk, kk), T, cap, mem_gb)
            for t in np.flatnonzero(hits):
                for j in range(hits[t]):
                    by_key.setdefault(int(rec_key[t, j]), []).append(int(rec_pk[t, j]))
    pairs = []
    for key in set(int(k) for k in suspicious):
        group = by_key.get(key, [])
        pairs += [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]
    pairs += [(pa, pb) for pa, key in probe_hits for pb in by_key.get(key, [])]
    outcome = dict(stabilizer=0, not_in_normalizer=0, logical=0)
    found = None
    for pa, pb in set(pairs):
        ax, az = unpack(pa)
        bx, bz = unpack(pb)
        x, z = ax ^ bx, az ^ bz
        kind = exact.classify(x, z)
        outcome[kind] += 1
        if kind == 'logical':
            w = bin(x | z).count('1')
            if found is None or w < found[0]:
                found = (w, pauli_str(x, z, n))
    passed = complete and found is None
    return dict(file=os.path.basename(path), n=n, k=n - r, D=D, passed=passed,
                verdict=('distance >= %d' % D) if passed else
                        ('FAILED: logical of weight %d found' % found[0] if found else 'FAILED: incomplete enumeration'),
                logical_found=found[1] if found else None,
                enumerated_ball=visited_ball, expected_ball=1 + (ball - 1) * P,
                enumerated_shell=visited_shell, expected_shell=shell * P, buckets=P,
                colliding_pairs_checked=len(set(pairs)), low_weight_stabilizers=outcome['stabilizer'],
                projection_only_matches=outcome['not_in_normalizer'],
                seconds=round(time.time() - t_start, 2), threads=numba.get_num_threads(),
                machine=platform.platform(), numba=numba.__version__, numpy=np.__version__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('stabilizer_file')
    ap.add_argument('--D', type=int, help='distance to certify (default: d from the file header)')
    ap.add_argument('--mem-gb', type=float, default=8.0)
    ap.add_argument('--threads', type=int)
    ap.add_argument('--out', help='append the JSON result to this file')
    args = ap.parse_args()
    D = args.D
    if D is None:
        head = open(args.stabilizer_file).readline()
        D = int(head.split(']]')[0].split(',')[-1])
    res = certify(args.stabilizer_file, D, args.mem_gb, args.threads, log=lambda s: print(s, file=sys.stderr, flush=True))
    line = json.dumps(res)
    print(line)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(line + '\n')
    sys.exit(0 if res['passed'] else 1)


if __name__ == '__main__':
    main()
