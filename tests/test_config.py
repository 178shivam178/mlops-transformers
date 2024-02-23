import unittest
from unittest.mock import patch, MagicMock
from prediction_service.prediction import sentence_prediction, NotInRange, lessInfo

class TestSentencePrediction(unittest.TestCase):

    def test_api_response_incorrect_range(self):
        with self.assertRaises(NotInRange) as context:
            sentence_prediction(" ".join(["test"] * 151))
        self.assertEqual(str(context.exception), "Input sentence should not exceed 150 words")

    def test_api_response_incorrect_col(self):
        with self.assertRaises(lessInfo) as context:
            sentence_prediction("word")
        self.assertEqual(str(context.exception), "Input sentence should be greater than 2 words")

if __name__ == '__main__':
    unittest.main()
