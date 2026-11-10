# 192 binary stabilizer codes, n = 53–254, d = 5–13

Explicit [[n,k,d]] qubit stabilizer codes. Each one has a larger minimum distance at its (n,k) than the lower
bound listed on [codetables.de](https://www.codetables.de) on 4 November 2026: 161 codes by +1 and 31 by +2.

Author: Marin Girard, Virginia Tech. A short write-up of the method and results is in [`note/note.pdf`](note/note.pdf).

## Contents

```
codes/
  codes.csv              one row per code: id, n, k, d, construction, file names, sha256 of the stabilizer file
  stabilizers/*.txt      n-k stabilizer generators per code, one per line, symplectic [X|Z] form
  witnesses.csv          for each code, one logical operator of weight exactly d
verify/
  check_codes.py         valid stabilizer code + weight-d logical; plain Python, no packages, about 1 s
  check_purity.py        every code is pure (no stabilizer of weight below d); plain Python, about 2 min
  certify_distance.py    independent proof of distance >= d by exhaustive search (numpy, numba)
  test_certify.py        tests of the certifier against brute force and planted low-weight logicals
  run_local.py           runs the certifier on the cheapest codes within a time budget
  results/               certificate records (see "Verification status")
comparison/
  compare.py             reproduces every number below from the two input files; plain Python, under 1 s
  codetables-2026-11-04.csv   all qubit-table bounds, n = 1..256, k = 0..n, read 4 November 2026
  direct.csv             each code against the table at its own (n,k)
  raised_entries.csv     every table cell our codes raise, and which code raises it
  summary.json
note/
  note.tex, note.pdf     short write-up: construction, where it wins, predictions, certification
  predict.py             first-moment predictor; writes predicted_gains.csv for all n <= 256, d = 5..13
  make_figure.py         makes fig-codes.pdf from codes/, comparison/ and predicted_gains.csv
provenance/
  recipes/*.json.gz      the tensor-network (LEGO) recipe and random seed that generated each code
```

Stabilizer file format: comment lines start with `#`; each other line is one generator,
`x_0 ... x_{n-1} | z_0 ... z_{n-1}`, where qubit i carries X if x_i=1, Z if z_i=1, and Y if both are 1.

## Checking the codes

```bash
python3 verify/check_codes.py
```

For every code this checks that the file matches its sha256, that the n-k generators are independent and
commute, and that the listed logical operator has weight d, commutes with every generator and is not a
stabilizer. So each code is a valid [[n,k]] code with **distance at most d**.

```bash
python3 verify/check_purity.py
```

This checks that every code is **pure**: its stabilizer group has no element of weight below d other than the
identity. It bounds the minimum weight of the stabilizer group from below with disjoint information sets (the
Brouwer–Zimmermann method) and enumerates the sums of a few generators that the bound does not cover. The
results are in `results/purity.csv`. With `--selftest` it compares itself with brute force on 400 small random
groups.

```bash
python3 verify/certify_distance.py codes/stabilizers/n097_k054_d09.txt
```

This proves **distance at least d**: no nontrivial logical operator has weight 1, ..., d-1. It enumerates every
Pauli of weight at most floor((d-1)/2), plus every Pauli of weight ceil((d-1)/2) when d is even, and finds
all pairs with equal syndromes. Each such pair multiplies to an element of the normalizer of weight
at most d-1. That element is checked exactly to be a stabilizer and not a logical. The program also counts
every Pauli it visits and compares the total with the closed form sum over w of C(n,w)·3^w. The method and the
argument for why it is complete are in the docstring. It was written for this release and shares no code with
the programs that found the codes.

`python3 verify/test_certify.py` runs 265 checks in about 25 s. For each test code of known distance d, the
certifier must pass at d and must fail at d+1, finding a logical of weight d. The test codes are textbook codes
(including the degenerate Shor and surface codes), 60 random codes whose distance is computed by brute force,
and 252–255-qubit codes with a planted low-weight logical.

## Verification status

| Check | Codes |
|---|---|
| Valid stabilizer code, weight-d logical (distance ≤ d) | 192 / 192 |
| Pure: no stabilizer of weight below d | 192 / 192, in `results/purity.csv` |
| Distance ≥ d, independent certifier on a laptop (Apple M3, 8 threads, about 5 min total) | 32 / 192, in `results/certify_local.jsonl` |
| Distance ≥ d, original certifiers on Aurora (ALCF), run on a code rebuilt from its recipe | 190 / 192, in `results/original_certifier_runs.csv` |
| Distance ≥ d by at least one of the two | **192 / 192** (30 by both) |

In every passing original run, the number of Paulis enumerated equals the closed-form size of the search space.
The file also lists 12 codes' earlier incomplete runs (each later completed) and, for 30 round-1 codes, a
control run at D = d+1 in which the certifier found a logical of weight d, as it should. The original runs used
two certifiers: `c-mitm` (round 1) and `c2` (later rounds). The cost of the independent laptop certifier grows
quickly with n and d. Codes with d ≤ 9 take from seconds to about an hour each, but the d = 12–13 codes need
10^12–10^14 Pauli operations and a cluster.

## Comparison with codetables.de

Table snapshot: all bounds for qubit codes, read on 4 November 2026 from the site's 16 full-table pages. It is
identical to a read made on 24 October 2026. `python3 comparison/compare.py` gives:

| | |
|---|---|
| Codes above the table's lower bound at their own (n,k) | 192 / 192: 161 by +1, 31 by +2 |
| Codes not implied by another of these codes | 162 |
| Table cells raised (k ≥ 1), using only [[n,k,d]] → [[n+1,k,d]] and [[n,k,d]] → [[n,k−1,d]] | 1,873: 1,748 by +1, 125 by +2 |
| Codes or raised cells above the table's upper bound | 0 |

At fixed distance the gain in k is up to +9. For example, the table's largest k at distance 12 is 98 for n = 172
and 113 for n = 188, while these codes give [[172,106,12]] and [[188,121,12]]. At d = 13 and n = 124 it is
51 versus [[124,58,13]].

## How the codes were made

190 codes: a random Clifford encoder (Bell pairs and random two-qubit gates, built as a quantum-LEGO tensor
network) gives a parent [[n,K]] code. All of its logical operators below the target weight are enumerated, and
q of its logical operators (combinations found by solving over GF(2)) are added as stabilizers, so that every
low-weight logical anticommutes with at least one of them. The result is [[n,K-q,d]].

2 codes ([[112,69,9]] and [[112,79,7]]): one qubit of a codetables code ([[113,73,9]] or [[113,85,7]]) is
punctured, and the code is repaired by the same expurgation step.

The search and certification programs were written and run by AI agents directed by the author. The files in
`verify/` were written and tested separately, so that the codes can be checked without trusting that software.

## Requirements

Python 3.8+. `check_codes.py` and `check_purity.py` need nothing else; the certifier needs `numpy` and `numba`.

## License

Code (`verify/`, `comparison/compare.py`): MIT. Codes and data: CC BY 4.0. The codetables.de snapshot
(`comparison/codetables-2026-11-04.csv`) is Markus Grassl's published data, included only for reproducibility;
it is not covered by these licenses. See `LICENSE`, and `CITATION.cff` for how to cite.
