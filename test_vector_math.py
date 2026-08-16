import unittest
import torch
import math

from mvp_rep_engineering.vector_math import generate_random_control_vectors

class TestVectorMath(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure we have a non-zero reference vector
        self.ref_shape = (4096,)
        self.reference_vector = torch.randn(self.ref_shape, device=self.device)
        self.reference_vector = (self.reference_vector / self.reference_vector.norm()) * 5.0
        self.seed = 42
        self.num_vectors = 3
        self.threshold = 0.2

    def test_number_of_vectors(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        self.assertEqual(len(vectors), self.num_vectors)

    def test_shape_and_device(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for v in vectors:
            self.assertEqual(v.shape, self.reference_vector.shape)
            self.assertEqual(v.device, self.reference_vector.device)
            self.assertEqual(v.dtype, self.reference_vector.dtype)

    def test_determinism(self):
        vectors1 = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        vectors2 = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for v1, v2 in zip(vectors1, vectors2):
            self.assertTrue(torch.allclose(v1, v2))

    def test_different_seeds_produce_different_vectors(self):
        vectors1 = generate_random_control_vectors(
            self.reference_vector, 1, self.seed, self.threshold
        )
        vectors2 = generate_random_control_vectors(
            self.reference_vector, 1, self.seed + 1, self.threshold
        )
        self.assertFalse(torch.allclose(vectors1[0], vectors2[0]))

    def test_norm_matching(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for v in vectors:
            self.assertTrue(
                torch.allclose(v.norm(p=2), torch.tensor(1.0, device=self.device), rtol=1e-5, atol=1e-6)
            )

    def test_finite_values(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for v in vectors:
            self.assertTrue(torch.isfinite(v).all())

    def test_candidate_similarity(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for v in vectors:
            cos_sim = torch.nn.functional.cosine_similarity(
                v.unsqueeze(0), self.reference_vector.unsqueeze(0)
            ).abs().item()
            self.assertLess(cos_sim, self.threshold)

    def test_pairwise_distinctness(self):
        vectors = generate_random_control_vectors(
            self.reference_vector, self.num_vectors, self.seed, self.threshold
        )
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                self.assertFalse(torch.allclose(vectors[i], vectors[j]))

    def test_degenerate_vector_raises_error(self):
        zero_ref = torch.zeros(self.ref_shape, device=self.device)
        with self.assertRaises(ValueError):
            generate_random_control_vectors(zero_ref, self.num_vectors, self.seed)

if __name__ == "__main__":
    unittest.main()
