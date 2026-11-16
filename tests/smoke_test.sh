#!/usr/bin/env bash
# Smoke test for our ML API (Member 5, Ian).
# Usage: bash tests/smoke_test.sh https://localhost
#    or: bash tests/smoke_test.sh http://127.0.0.1:5000
#    or: bash tests/smoke_test.sh https://<our-service>.onrender.com
#
# When a member swaps in a new model, update their _URL and _BODY below
# from the sample cURL in their MEMBER block of api.py.
# NOTE (16 Nov): regression, classification and clustering still use the
# lecturer's sample inputs, because those blocks still serve his models.

BASE="${1:-https://localhost}"
PASS=0
FAIL=0

# check <label> <expected HTTP code> <method> <path> [json body]
check() {
  local label="$1" expected="$2" method="$3" path="$4" body="$5"
  echo "== $label"
  if [ -n "$body" ]; then
    response=$(curl -sk -w '\n%{http_code}' -X "$method" "$BASE$path" \
      -H "Content-Type: application/json" -d "$body")
  else
    response=$(curl -sk -w '\n%{http_code}' -X "$method" "$BASE$path")
  fi
  code=$(echo "$response" | tail -n1)
  echo "$response" | sed '$d'
  if [ "$code" = "$expected" ]; then
    echo "-> HTTP $code (OK)"; PASS=$((PASS + 1))
  else
    echo "-> HTTP $code (expected $expected)"; FAIL=$((FAIL + 1))
  fi
  echo
}

# ---- Fill these in as each member merges ----
REGRESSION_URL="/api/v1/models/sme-revenue-regressor/predictions"            # Lisa
REGRESSION_BODY='{"sector": "Technology", "county": "Nairobi", "quarter": "Q3", "business_age_years": 4.5, "num_employees": 12, "owner_education_level": "Secondary", "monthly_marketing_spend_kes": 45000, "avg_daily_foot_traffic": null, "online_presence_score": 78.0, "loan_amount_kes": 250000, "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5, "competitor_density": 5, "customer_satisfaction_score": 4.2}'
CLASSIFICATION_URL="/api/v1/models/sme-credit-risk-classifier/predictions"  # Stephanie
CLASSIFICATION_BODY='{"sector": "Retail", "county": "Nairobi", "quarter": "Q2", "loan_purpose": "Working Capital", "owner_education_level": "Secondary", "business_age_years": 2.1, "num_employees": 4, "monthly_revenue_kes": 850000, "loan_amount_kes": 150000, "loan_term_months": 12, "interest_rate_pct": 16.5, "collateral_value_kes": null, "debt_to_income_ratio": 0.62, "credit_score": 590, "num_previous_loans": 1, "missed_payments_count": 3, "online_presence_score": 40.0, "customer_satisfaction_score": 3.4, "competitor_density": 6}'
CLUSTERING_URL="/api/v1/models/mall-customer-segmenter/predictions"          # Amiir
CLUSTERING_BODY='{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
RECOMMENDER_URL="/api/v1/models/pharmacy-recommender/predictions"            # Felista
RECOMMENDER_BODY='{"cart": ["D05", "D04"]}'
# ---------------------------------------------

check "Health check"                  200 GET  "/api/v1/health"
check "Regression (Lisa)"             200 POST "$REGRESSION_URL"     "$REGRESSION_BODY"
check "Classification (Stephanie)"    200 POST "$CLASSIFICATION_URL" "$CLASSIFICATION_BODY"
check "Clustering (Amiir)"            200 POST "$CLUSTERING_URL"     "$CLUSTERING_BODY"
check "Recommender (Felista)"         200 POST "$RECOMMENDER_URL"    "$RECOMMENDER_BODY"
check "Missing input (should be 400)" 400 POST "$REGRESSION_URL"     '{"sector": "Technology"}'
check "Empty cart (should be 400)"    400 POST "$RECOMMENDER_URL"    '{"cart": []}'
check "Wrong method (should be 405)"  405 GET  "$REGRESSION_URL"
check "Unknown URL (should be 404)"   404 GET  "/api/v1/does-not-exist"

echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
