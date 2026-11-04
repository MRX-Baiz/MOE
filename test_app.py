"""Routing and artifact-handling tests; no trained checkpoints required."""
import unittest
from unittest.mock import Mock

from predictor import MoEPredictor


class PredictorTests(unittest.TestCase):
    def setUp(self):
        # Isolate routing from heavyweight model initialization and local artifacts.
        self.predictor = MoEPredictor.__new__(MoEPredictor)
        self.predictor.expert_status = {"fr": False, "ar": False}
        self.predictor.svm_available = False

    def test_routes_by_arabic_unicode(self):
        for text, expected in [("Bonjour", "fr"), ("مرحبا", "ar"),
                               ("Bonjour مرحبا", "ar"), ("Hello", "fr")]:
            with self.subTest(text=text):
                self.assertEqual(self.predictor.route(text), expected)

    def test_missing_models_return_error(self):
        self.assertIn("error", self.predictor.predict("Question", "Réponse"))

    def test_selected_expert_receives_reference(self):
        self.predictor.expert_status["ar"] = True
        self.predictor._predict_expert = Mock(return_value={"prediction": "correct"})
        self.predictor.predict("Question", "مرحبا", "Reference")
        self.predictor._predict_expert.assert_called_once_with(
            "ar", "Question", "مرحبا", "Reference")

    def test_missing_expert_uses_svm(self):
        self.predictor.svm_available = True
        self.predictor._predict_svm = Mock(return_value={"prediction": "incorrect"})
        self.predictor.predict("Question", "Réponse")
        self.predictor._predict_svm.assert_called_once_with("Question", "Réponse", "fr")

    def test_svm_does_not_invent_confidence(self):
        self.predictor.tfidf_vectorizer = Mock()
        self.predictor.svm_model = Mock()
        self.predictor.svm_model.predict.return_value = [0]
        result = self.predictor._predict_svm("Question", "Réponse", "fr")
        self.assertEqual(result["prediction"], "correct")
        self.assertIsNone(result["confidence"])


if __name__ == "__main__":
    unittest.main()
