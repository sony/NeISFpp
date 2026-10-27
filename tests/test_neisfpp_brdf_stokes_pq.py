import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mymodules")),
)

from neisfpp_brdf_stokes_pq import (
    Q16_ONE,
    calc_lambda_heitz_pq_q16,
    calc_normal_distribution_walter_pq_q16,
    calc_self_masking_heitz_pq_q16,
    isqrt_u128,
    stokes_dolp_and_cone_project_pq_q16,
)


class TestSonyNeISFppBrdfStokesPq(unittest.TestCase):
    def test_walter_ggx_pq_matches_exact_and_handles_grazing_angle(self):
        roughness_sq = 0.25
        r_sq_q16 = int(round(roughness_sq * Q16_ONE))

        for c in [0.95, 0.75, 0.5, 0.25, 0.05, 1e-4]:
            c_q16 = max(1, int(round(c * Q16_ONE)))
            d_pq = calc_normal_distribution_walter_pq_q16(c_q16, r_sq_q16) / float(Q16_ONE)
            c_real = c_q16 / float(Q16_ONE)
            denom = math.pi * ((1.0 + c_real * c_real * (roughness_sq - 1.0)) ** 2)
            d_exact = roughness_sq / denom
            self.assertAlmostEqual(d_pq, d_exact, delta=1.5e-3)

    def test_heitz_self_masking_pq_q16(self):
        roughness_sq = 0.36
        r_sq_q16 = int(round(roughness_sq * Q16_ONE))
        v_dot_n = 0.8
        l_dot_n = 0.6

        g_pq = (
            calc_self_masking_heitz_pq_q16(
                int(round(v_dot_n * Q16_ONE)),
                int(round(l_dot_n * Q16_ONE)),
                r_sq_q16,
            )
            / float(Q16_ONE)
        )

        def exact_lambda(c: float) -> float:
            return (2.0 * c) / (c + math.sqrt(c * c + roughness_sq * (1.0 - c * c)))

        g_exact = exact_lambda(v_dot_n) * exact_lambda(l_dot_n)
        self.assertAlmostEqual(g_pq, g_exact, delta=5e-4)

    def test_stokes_dolp_and_physical_cone_projection(self):
        # 1. Valid interior Stokes vector: s0=1.0, s1=0.6, s2=0.8 -> DoLP = 1.0
        s0 = Q16_ONE
        s1 = int(round(0.3 * Q16_ONE))
        s2 = int(round(0.4 * Q16_ONE))
        dolp_q16, (p0, p1, p2) = stokes_dolp_and_cone_project_pq_q16(s0, s1, s2)
        self.assertAlmostEqual(dolp_q16 / float(Q16_ONE), 0.5, delta=1e-4)
        self.assertEqual((p0, p1, p2), (s0, s1, s2))

        # 2. Unphysical over-polarized neural prediction: s0=0.5, s1=0.6, s2=0.8 (mag=1.0 > s0)
        s0_over = int(round(0.5 * Q16_ONE))
        dolp_over, (q0, q1, q2) = stokes_dolp_and_cone_project_pq_q16(
            s0_over,
            int(round(0.6 * Q16_ONE)),
            int(round(0.8 * Q16_ONE)),
        )
        self.assertEqual(dolp_over, Q16_ONE)
        self.assertLessEqual(isqrt_u128(q1 * q1 + q2 * q2), q0)


if __name__ == "__main__":
    unittest.main()
