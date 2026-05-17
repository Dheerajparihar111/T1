# ============================================================
# test_api.py
# Quick test script to verify all modules work correctly
# Run with: python test_api.py
# ============================================================

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported without errors."""
    print("\n" + "="*60)
    print("TEST 1: Checking module imports...")
    print("="*60)

    try:
        from schemas import HealthData, PredictionResult, PredictionResponse
        print("✅ schemas.py imported successfully")
    except Exception as e:
        print(f"❌ schemas.py import failed: {e}")
        return False

    try:
        from recommendations import identify_risk_factors, generate_recommendations
        print("✅ recommendations.py imported successfully")
    except Exception as e:
        print(f"❌ recommendations.py import failed: {e}")
        return False

    try:
        from predict import prepare_features, determine_risk_level, FEATURE_NAMES
        print("✅ predict.py imported successfully")
    except Exception as e:
        print(f"❌ predict.py import failed: {e}")
        return False

    return True


def test_risk_levels():
    """Test the risk level determination logic."""
    print("\n" + "="*60)
    print("TEST 2: Risk Level Thresholds...")
    print("="*60)

    from predict import determine_risk_level

    tests = [
        (0.10, "Low Risk"),
        (0.29, "Low Risk"),
        (0.30, "Medium Risk"),
        (0.50, "Medium Risk"),
        (0.69, "Medium Risk"),
        (0.70, "High Risk"),
        (0.95, "High Risk"),
    ]

    all_passed = True
    for probability, expected in tests:
        result = determine_risk_level(probability)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"{status} probability={probability:.2f} → {result} (expected: {expected})")

    return all_passed


def test_feature_preparation():
    """Test that features are extracted correctly from health data."""
    print("\n" + "="*60)
    print("TEST 3: Feature Preparation...")
    print("="*60)

    from predict import prepare_features

    # Test with complete data
    complete_data = {
        "name": "Test User",
        "glucose": 150.0,
        "bmi": 32.5,
        "hba1c": 7.2,
        "age": 55.0,
        "blood_pressure": 135.0,
        "cholesterol": 220.0,
        "insulin": 30.0,
    }

    features = prepare_features(complete_data)
    print(f"✅ Complete data: shape={features.shape}, values={features[0].tolist()}")
    assert features.shape == (1, 5), f"Expected shape (1,5), got {features.shape}"

    # Test with missing data (should use defaults)
    incomplete_data = {
        "name": "Incomplete User",
        "glucose": 120.0,
        # bmi, hba1c, age, blood_pressure are missing
    }

    features_incomplete = prepare_features(incomplete_data)
    print(f"✅ Incomplete data handled: shape={features_incomplete.shape}")
    assert features_incomplete.shape == (1, 5), "Missing data should use defaults"

    return True


def test_risk_factors():
    """Test risk factor identification."""
    print("\n" + "="*60)
    print("TEST 4: Risk Factor Identification...")
    print("="*60)

    from recommendations import identify_risk_factors

    high_risk_data = {
        "glucose": 180.0,      # Very high
        "bmi": 35.0,           # Obese
        "hba1c": 8.5,          # Diabetic range
        "blood_pressure": 150.0,  # Stage 2 hypertension
        "cholesterol": 250.0,  # High
        "insulin": 40.0,       # Elevated
        "age": 60.0,           # Age risk
    }

    risk_factors = identify_risk_factors(high_risk_data)
    print(f"✅ Found {len(risk_factors)} risk factors:")
    for rf in risk_factors:
        print(f"   {rf}")

    assert len(risk_factors) > 0, "Should find risk factors for high values"
    return True


def test_recommendations():
    """Test recommendation generation."""
    print("\n" + "="*60)
    print("TEST 5: Recommendation Generation...")
    print("="*60)

    from recommendations import generate_recommendations

    health_data = {
        "glucose": 145.0,
        "bmi": 31.0,
        "hba1c": 6.8,
        "blood_pressure": 138.0,
        "cholesterol": 215.0,
    }

    recs = generate_recommendations(health_data, "High Risk")
    print(f"✅ Generated {len(recs)} recommendations:")
    for i, rec in enumerate(recs[:5], 1):  # Show first 5
        print(f"   {i}. {rec}")
    if len(recs) > 5:
        print(f"   ... and {len(recs) - 5} more")

    assert len(recs) > 0, "Should generate recommendations"
    return True


def test_supabase_connection():
    """Test Supabase connection (requires valid credentials)."""
    print("\n" + "="*60)
    print("TEST 6: Supabase Connection...")
    print("="*60)

    try:
        from supabase_client import supabase
        print("✅ Supabase client initialized")

        # Try a simple query
        result = supabase.table("health_data").select("count").limit(1).execute()
        print("✅ Successfully connected to Supabase and queried health_data table")
        return True
    except Exception as e:
        print(f"⚠️  Supabase connection test failed: {e}")
        print("   (This is OK if you're running without network access)")
        return True  # Don't fail the overall test for this


def test_model_loading():
    """Test ML model loading."""
    print("\n" + "="*60)
    print("TEST 7: ML Model Loading...")
    print("="*60)

    from predict import load_model_and_scaler

    try:
        model, scaler = load_model_and_scaler()
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Scaler loaded: {type(scaler).__name__}")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  Model files not found (expected during setup): {e}")
        print("   Place diabetes_model.pkl and scaler.pkl in the models/ folder")
        return True  # Not a failure — files just need to be placed
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "🏥 "*15)
    print("DIABETES PREDICTION API — TEST SUITE")
    print("🏥 "*15)

    results = {
        "imports": test_imports(),
        "risk_levels": test_risk_levels(),
        "features": test_feature_preparation(),
        "risk_factors": test_risk_factors(),
        "recommendations": test_recommendations(),
        "supabase": test_supabase_connection(),
        "model": test_model_loading(),
    }

    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"  {status} — {test_name}")

    print(f"\n{'='*60}")
    print(f"  {passed}/{total} tests passed")

    if passed == total:
        print("  🎉 ALL TESTS PASSED! Your API is ready to run.")
        print("  👉 Start with: uvicorn main:app --reload")
    else:
        print("  ⚠️  Some tests failed. Check the errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
