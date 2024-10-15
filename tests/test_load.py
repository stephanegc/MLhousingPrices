import unittest   # The test framework
from MLhousingPrices import preprocessor

class Test_load_preprocessor(unittest.TestCase):
    def test_load_preprocessor(self):
        pp = preprocessor.load_preprocessor()
        