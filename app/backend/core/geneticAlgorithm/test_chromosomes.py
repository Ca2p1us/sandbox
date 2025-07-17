import unittest
from ..gene_repair.fm_params.repair import *


class Repair(unittest.TestCase):
    """
    """

    def test_round_off_to_digit(self):
        """四捨五入を行える 
        """
        self.assertEqual(round_off_to_digit(-0.5), round_off_to_digit(-1.0))
        self.assertEqual(round_off_to_digit(0.5), round_off_to_digit(1))
        self.assertEqual(round_off_to_digit(1), round_off_to_digit(1))
        self.assertEqual(round_off_to_digit(1.4), round_off_to_digit(1))
        self.assertEqual(round_off_to_digit(1.5), round_off_to_digit(2))


if __name__ == "__main__":
    unittest.main()
