# dqi-sim

A decoder-in-the-loop simulator for experimenting with variants of **Decoded Quantum
Interferometry** (Jordan et al., [arXiv:2408.08292](https://arxiv.org/abs/2408.08292)).
It has a CLI and a local web UI that drive the same pipeline. Every stage is a swappable
component, and every run is resource-guarded so experiments cannot take the machine down.

It does **not** simulate gates. For max-XORSAT, what DQI achieves depends on
the code and on how well a classical decoder recovers errors of each weight. You can compute
that exactly on small instances and estimate it by Monte Carlo on instances with
thousands of variables, which is the regime where variants differ. (The paper's
31k-variable result was obtained the same way.)

## The model in one paragraph

DQI prepares Σ_k w_k/√C(m,k) Σ_{|y|=k} (−1)^{v·y} |Bᵀy⟩, decodes y from the syndrome Bᵀy
in superposition, post-selects on a clean error register, and measures x after a Hadamard
transform. With f(x) = #sat − #unsat, every evaluator reduces this to two (ℓ+1)×(ℓ+1)
matrices:

```
P_succ = wᵀ G w          <f> = wᵀ F w / wᵀ G w          <s>/m = 1/2 + <f>/(2m)
```

Perfect decoding below half the dual distance gives G = I and F = A, the paper's tridiagonal
matrix, so the optimal w is its top eigenvector (the semicircle law). Real decoders and
syndrome collisions change F and G. Estimating them is the simulator's job. Once you have
them, choosing weights becomes a generalised eigenproblem (`optimal-aware`). That is one of
the variants included.

## Install

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/dqisim presets
.venv/bin/dqisim run quickstart
.venv/bin/dqisim ui            # http://127.0.0.1:8765
```

## Architecture: what you can swap

| stage | components (`dqisim list`) |
|---|---|
| problem | `xorsat-random`, `xorsat-regular` (Gallager k,D), `xorsat-irregular` (degree distributions), `xorsat-file`; RHS `v` = random / planted+noise / ones (MaxCut) |
| weights P(f) | `optimal-ideal` (paper), `optimal-aware` (maximises the measured F,G subject to P_succ ≥ `min_p_succ`, default 0.5), `product` (Gibbs-like Π(1+t f_i)), `single`, `uniform`, `custom` |
| decoder | `bp` (min-sum / sum-product), `bposd` (BP + OSD-0), `bitflip`, `ml` (exact coset leader up to t) |
| evaluator | `semicircle`, `ideal`, `mc`, `exact` (+ dense 2ⁿ distribution and sampling) |
| baseline | `sa`, `sa-degree`, `walksat`, `prange`, `random`, `exact` (all 2ⁿ by one FWHT) |
| DQI knobs | `dqi.ell` (int, fraction of m, or `auto` = decoder radius probe), `dqi.postselect`, `dqi.ell_auto_eps` |

An experiment is one spec (TOML or JSON). The CLI and the UI both edit it:

```bash
dqisim run small-exact --sweep dqi.decoder=ml,bp,bposd --seeds 0-4
dqisim run --problem xorsat-regular:n=400,k=3,D=6 --decoder bposd:iters=20 \
           --weights optimal-aware --ell 0.12 --evaluator mc:samples=300 \
           --baseline sa:sweeps=1000 --baseline walksat
dqisim run decoder-shootout --set dqi.postselect=false --sweep problem.n=200,400,800
dqisim spec quickstart --decoder bitflip > my.toml     # resolved spec: edit, then
dqisim run my.toml
dqisim report results/<run-dir> --metric p_succ
```

Sweeps take any dotted path and form a Cartesian product. Component values are written as
`name:key=val,...`; on the command line, separate them with `|` (`--sweep 'dqi.decoder=bp:iters=10|bposd'`).
Sweep points that share a problem and seed reuse the same instance, the same baseline results,
the same auto-ℓ probe and the same Monte Carlo draws (common random numbers), so
comparisons between variants are paired.

### Adding your own component

```python
# my_decoder.py
import numpy as np
from dqisim.registry import register, P
from dqisim.decoders import Decoder

@register("decoder", "my-decoder", "what it does", P("depth", int, 3, "help text"))
class MyDecoder(Decoder):
    def __init__(self, depth): self.depth = depth
    def decode(self, S):                     # S: (batch, n) syndromes -> (batch, m) errors
        return np.zeros((S.shape[0], self.inst.m), np.uint8)
```

`dqisim run ... --plugin my_decoder.py`, or `dqisim ui --plugin my_decoder.py`. The UI builds
its forms from the registry, so the new component and its parameters show up there with no
UI changes. The same pattern works for problems (`generate(rng) -> Instance`), weights
(`choose(m, ell, F, G)`), evaluators (`run(ctx) -> dict`) and baselines (`run(inst, rng) -> dict`).

## Evaluators and their assumptions

| evaluator | exact? | cost | notes |
|---|---|---|---|
| `semicircle` | asymptotic | O(1) | paper Eq. 6 at ℓ/m |
| `ideal` | exact if 2ℓ+1 < d⊥ and decoding is perfect | O(ℓ) | finite-m eigenvalue |
| `mc` | exact **in expectation over random v** | k_points × samples decodes | samples failure rate ε_k and the neighbour-pair success π_k; leaves out collision cross terms, whose mean over v is zero |
| `exact` | exact for the instance | Σ_{k≤ℓ} C(m,k) decodes | every error enumerated; F and G include all collisions; reports post-selected and non-post-selected; `dense` adds the full 2ⁿ output distribution, P(optimum) and best-of-shots |

A DQI decoder runs in superposition, so it cannot see |y|. Every decoder therefore uses one
fixed prior (default ℓ/m) for all syndromes.

## Resource safety

Every run, including each UI job (a separate process), gets the following:

- **CPU:** `nice 19` plus macOS background QoS, or Linux `ionice` idle class. BLAS and numba
  threads are capped (default 2, hard cap 4) before numpy loads. One worker by default, hard
  cap 2. New work pauses while the 1-minute load average is above 75% of the cores.
- **Memory:** each allocation-heavy step estimates its peak first and refuses if it would not
  fit the budget (default 4 GB RSS) or would push free system memory below a floor (8 GB).
  A watchdog thread samples RSS every second. On a breach it interrupts the run, and
  hard-exits if the run does not stop within 10 s.
- **Disk:** output goes only to the run directory, as small JSON (no state vectors). Writes
  are refused once the results directory reaches its cap (500 MB) or free disk would drop
  below 20 GB.
- **Time:** a wall-time budget per job (default 1 h).

You can change the limits under `[resources]` in a spec, with `--memory-gb/--walltime/--workers/--threads`,
or in the UI's Resources card. `dqisim status` shows what the guard sees.

## Validation (`pytest`, 53 tests)

- The semicircle law reproduces the paper's 0.7179 (OPI, n/p = 1/10). The finite-m
  eigenvalue converges to the semicircle for r/p ∈ {0.3, 0.5, 0.7}.
- With no collisions, `exact` equals the paper's formula to 1e-12. The dense output
  distribution equals P(f(x))², built directly from elementary symmetric polynomials (an
  independent check of the sign conventions).
- When collisions are present, the quadratic form matches the dense 2ⁿ distribution to 1e-10
  for every decoder.
- `mc` agrees with `exact` when averaged over instances (|Δ| < 0.01).
- Decoders: BP-OSD always returns a valid solution to Hy = s, and ML returns coset leaders.
- Safety: thread caps, nice 19 in children, refusal of oversized allocations, wall time, the
  disk cap, and the watchdog stopping a runaway allocation.
- The CLI end to end, plugins, and the UI end to end (job submission, Host and Content-Type
  guards).

## What the bundled presets show (first runs, Apple M-series, 1 core at nice 19)

| preset | runtime / peak RSS | result |
|---|---|---|
| `small-exact` (n=18, ℓ=3, 3 seeds) | 25 s / 170 MB | exact satisfied fraction: ML 0.689, BP-OSD 0.682, BP 0.671, bit flipping 0.628; `mc` within 0.002 of `exact` for every decoder; true optimum 0.870 |
| `ell-scan` (Gallager 3,6, m=600, BP, paper weights) | 13 min / 186 MB | peaks at 0.739 near ℓ/m ≈ 0.08 (P_succ 0.73), then collapses (P_succ 0.06 → 1e-4) while the semicircle keeps climbing to 0.90; SA 0.857 |
| `weights-shootout` (ℓ/m = 0.15) | 4.5 min / 187 MB | paper weights 0.667 at P_succ ≈ 0; decoder-aware 0.744 at P_succ 0.67 |
| `decoder-shootout` (m=800, ℓ/m = 0.12) | 5.3 min / 206 MB | BP ≈ BP-OSD ≈ 0.746 > bit flipping 0.682; SA 0.860, WalkSAT 0.805 |
| `irregular` (n=600, m=1000) | 24 s / 230 MB | DQI+BP 0.771 vs SA 0.869 and degree-aware SA 0.884 |

On these generic ensembles, DQI with classical decoders stays well below simulated annealing.
That matches the paper's own caution. The tool is for finding where that changes.

## Caveats

- Only F₂ (max-XORSAT). For OPI and general max-LINSAT over F_p there is just the analytic
  formula (`weights.ideal_fraction`, `semicircle`). Reed–Solomon decoding is not built.
- `sa-degree` follows the spirit of the paper's tailored "irregular annealing" but is not
  their exact rule. The `irregular` preset's degree mix is illustrative, not the paper's
  instance.
- `mc` assumes random v. For planted or structured v, use `exact` on small instances.
- No gate-level or noise simulation. For that, see DQI-Kit, the BCG/BMW Qiskit implementation,
  or the PennyLane and Classiq demos.
