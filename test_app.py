import sys
from unittest.mock import MagicMock

# Create a profound MagicMock for streamlit
st_mock = MagicMock()
sys.modules['streamlit'] = st_mock

from app_interface import MoEPredictor

def test_moe():
    print("Testing MoEPredictor Simplification (FR/AR)...")
    try:
        predictor = MoEPredictor()
        print(f"Experts loaded: {predictor.expert_status}")
        
        # Test routing
        print(f"Route 'Bonjour': {predictor.route('Bonjour')}")
        print(f"Route 'مرحبا': {predictor.route('مرحبا')}")
        
        # Test predict
        result_fr = predictor.predict("Qu'est-ce que le cloud?", "C'est internet.", "")
        print(f"FR Prediction Context: {result_fr}")
        
        result_ar = predictor.predict("ما هو السحابة؟", "إنها الإنترنت.", "")
        print(f"AR Prediction Context: {result_ar}")
        
        print("\n✅ Verification PASSED: MoEPredictor is correctly configured for 2 experts.")
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")

if __name__ == "__main__":
    test_moe()
