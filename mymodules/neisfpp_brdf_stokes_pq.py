r"""
NeISF++ (CVPR 2025) × 8b-is (`https://8b.is`):
Singularity-Free Rational `p/q` (`Q16`) Walter GGX & Heitz Smith BRDFs and
`isqrt_u128` Polarized Stokes Cone Projection for Sony Semiconductor & PlayStation
PBR Pipelines
==================================================================================

Brought to you by **Chris, Hue & the 8b-is team** (https://8b.is · WhatsApp: `+1 757-614-7542`):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64` & `p/q`
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

What This Module Adds to Sony Semiconductor Solutions' `NeISFpp` (`mymodules/brdfutils.py`)
-------------------------------------------------------------------------------------------
1. **Algebraic Elimination of Grazing-Angle Singularities (`cos2_th -> 0`)**:
   - In `mymodules/brdfutils.py`, `calc_normal_distribution_walter` evaluates:
       `tan2_th = (1 - cos2_th) / (cos2_th + EPS)`
       `D = roughness_sq * chi / (pi * cos4_th * (roughness_sq + tan2_th)^2 + EPS)`
     At grazing half-angles (`n_dot_h -> 0`), dividing by `cos2_th + EPS` and then
     multiplying by `cos4_th` introduces catastrophic floating-point cancellation
     and artificial `EPS` bias.
   - Algebraically absorbing $\cos^4\theta$ into $(\alpha^2 + \tan^2\theta)^2$:
       $$\cos^4\theta \left(\alpha^2 + \frac{1 - \cos^2\theta}{\cos^2\theta}\right)^2
         = \Big(1 + \cos^2\theta(\alpha^2 - 1)\Big)^2$$
     completely eliminates `tan2_th` and the grazing-angle division!
   - Similarly, in `calc_lambda_heitz`, absorbing $|c| = |\mathbf{x}\cdot\mathbf{n}|$
     into the square root:
       $$\Lambda(c) = \frac{2 \chi(c)}{1 + \sqrt{1 + \alpha^2 \frac{1 - c^2}{c^2}}}
                    = \frac{2 c \, \chi(c)}{c + \sqrt{c^2 + \alpha^2(1 - c^2)}}$$
     removes the inner division by `cos2_th + EPS` and evaluates cleanly down to $c = 0$.

2. **Zero-Float `p/q` (`Q16`) BRDF & Stokes Cone Projection via `isqrt_u128`**:
   - Evaluates the Walter GGX normal distribution $D(\mathbf{h})$, Heitz Smith
     masking-shadowing $G(\mathbf{v}, \mathbf{l})$, Degree of Linear Polarization
     $\text{DoLP} = \sqrt{s_1^2 + s_2^2} / s_0$, and physical Stokes cone projection
     ($\sqrt{s_1^2 + s_2^2} \le s_0$) using pure integer arithmetic and `isqrt_u128`.
"""

from __future__ import annotations

import math
from typing import Tuple

Q16_ONE = 1 << 16  # 65,536
# Rational approximation of pi in Q16: round(pi * 65536) = 205,887 (205887 / 65536 = 3.1415863)
PI_Q16 = 205_887


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) via hardware shifts/adds."""
    return math.isqrt(max(0, int(n)))


def calc_normal_distribution_walter_pq_q16(
    n_dot_h_q16: int,
    roughness_sq_q16: int,
) -> int:
    r"""
    Singularity-free rational `p/q` (`Q16`) Walter GGX normal distribution:
        $$D(\mathbf{h}) = \frac{\alpha^2 \, \chi(\mathbf{n}\cdot\mathbf{h})}
                              {\pi \big(1 + (\mathbf{n}\cdot\mathbf{h})^2 (\alpha^2 - 1)\big)^2}$$
    Requires zero intermediate `tan^2(theta)` division.
    """
    if n_dot_h_q16 <= 0 or roughness_sq_q16 <= 0:
        return 0
    c_q16 = min(Q16_ONE, int(n_dot_h_q16))
    cos2_q16 = (c_q16 * c_q16 + (1 << 15)) >> 16
    # term_q16 = 1 + cos2 * (alpha^2 - 1)
    term_q16 = Q16_ONE + ((cos2_q16 * (int(roughness_sq_q16) - Q16_ONE) + (1 << 15)) >> 16)
    term_sq_q16 = max(1, (term_q16 * term_q16 + (1 << 15)) >> 16)
    denom_q16 = max(1, (PI_Q16 * term_sq_q16 + (1 << 15)) >> 16)
    return ((int(roughness_sq_q16) << 16) + (denom_q16 >> 1)) // denom_q16


def calc_lambda_heitz_pq_q16(
    x_dot_n_q16: int,
    roughness_sq_q16: int,
) -> int:
    r"""
    Singularity-free rational `p/q` (`Q16`) Heitz GGX $\Lambda(\mathbf{x}\cdot\mathbf{n})$:
        $$\Lambda(c) = \frac{2 c \, \chi(c)}{c + \sqrt{c^2 + \alpha^2(1 - c^2)}}$$
    Evaluated using exact integer `isqrt_u128` without dividing by `cos^2(theta)`.
    """
    if x_dot_n_q16 <= 0:
        return 0
    c_q16 = min(Q16_ONE, int(x_dot_n_q16))
    c2_q32 = c_q16 * c_q16
    one_minus_c2_q32 = (Q16_ONE * Q16_ONE) - c2_q32
    alpha2_term_q32 = (int(roughness_sq_q16) * one_minus_c2_q32 + (1 << 15)) >> 16
    radicand_q32 = c2_q32 + alpha2_term_q32
    root_q16 = isqrt_u128(radicand_q32)
    denom_q16 = max(1, c_q16 + root_q16)
    return ((2 * c_q16) << 16) // denom_q16


def calc_self_masking_heitz_pq_q16(
    v_dot_n_q16: int,
    l_dot_n_q16: int,
    roughness_sq_q16: int,
) -> int:
    r"""
    Singularity-free rational `p/q` (`Q16`) Heitz GGX self-masking $G(\mathbf{v}, \mathbf{l})$.
    """
    lv = calc_lambda_heitz_pq_q16(v_dot_n_q16, roughness_sq_q16)
    ll = calc_lambda_heitz_pq_q16(l_dot_n_q16, roughness_sq_q16)
    return (lv * ll + (1 << 15)) >> 16


def stokes_dolp_and_cone_project_pq_q16(
    s0_q16: int,
    s1_q16: int,
    s2_q16: int,
) -> Tuple[int, Tuple[int, int, int]]:
    r"""
    Compute the exact integer `Q16` Degree of Linear Polarization:
        $$\text{DoLP} = \min\left(1, \frac{\sqrt{s_1^2 + s_2^2}}{s_0}\right)$$
    and project the Stokes vector $(s_0, s_1, s_2)$ onto the physically realizable
    Stokes cone ($\sqrt{s_1^2 + s_2^2} \le s_0, \; s_0 \ge 0$) using `isqrt_u128`
    with zero floating-point square roots!
    """
    if s0_q16 <= 0:
        return 0, (0, 0, 0)
    pol_mag_q16 = isqrt_u128(int(s1_q16) * int(s1_q16) + int(s2_q16) * int(s2_q16))
    if pol_mag_q16 <= s0_q16:
        dolp_q16 = ((pol_mag_q16 << 16) + (s0_q16 >> 1)) // s0_q16
        return min(Q16_ONE, dolp_q16), (int(s0_q16), int(s1_q16), int(s2_q16))

    # Project (s1, s2) radially onto the boundary circle sqrt(s1^2 + s2^2) == s0
    s1_proj = (int(s1_q16) * int(s0_q16)) // pol_mag_q16
    s2_proj = (int(s2_q16) * int(s0_q16)) // pol_mag_q16
    return Q16_ONE, (int(s0_q16), s1_proj, s2_proj)
