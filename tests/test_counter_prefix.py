import unittest

import pandas as pd

from counter import restore_generated_prefix


class PrefixReconstructionTests(unittest.TestCase):
    def test_all_particles_restore_context_without_overwriting_generated_tokens(self):
        original = [5, 1, 19, 133, 117, 3, 20, 2]
        frame = pd.DataFrame([
            original,
            [0, 0, 0, 0, 0, 0, 21, 2],
            [0, 0, 0, 0, 0, 0, 22, 2],
        ])
        restored, start = restore_generated_prefix(frame)
        self.assertEqual(start, 6)
        self.assertEqual(restored.iloc[1].tolist(), original[:6] + [21, 2])
        self.assertEqual(restored.iloc[2].tolist(), original[:6] + [22, 2])
        self.assertEqual(restored.iloc[0].tolist(), original)
        self.assertEqual(frame.iloc[1, :6].tolist(), [0] * 6)

    def test_invalid_boundaries_are_rejected(self):
        for rows in ([[1, 2]], [[1, 2], [0, 0]], [[1, 2], [0, 2], [1, 2]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                restore_generated_prefix(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
