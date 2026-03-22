# LinearSHAP Expected Output

Run with: `go run main.go`

## Output

```
LinearSHAP Example: House Price Prediction
==========================================

Model: f(x) = 50000 + 100×sqft + 5000×bedrooms + 10000×bathrooms - 500×age

Background feature means:
  E[sqft] = 1800.0
  E[bedrooms] = 3.2
  E[bathrooms] = 2.0
  E[age] = 16.0

Base value (expected price): $258000.00

House to explain:
  sqft = 2200
  bedrooms = 4
  bathrooms = 3
  age = 8

Predicted price: $316000.00
Base value:      $258000.00
Difference:      $58000.00

Feature contributions (SHAP values):
  + sqft: $40000.00
  + bathrooms: $10000.00
  + age: $4000.00
  + bedrooms: $4000.00

Local accuracy check: true
  Sum of SHAP values: $58000.00
  Expected (pred - base): $58000.00

---

Explaining a smaller, older house:
  sqft = 1000
  bedrooms = 2
  bathrooms = 1
  age = 40

Predicted price: $150000.00

Feature contributions:
  - sqft: $-80000.00
  - age: $-12000.00
  - bathrooms: $-10000.00
  - bedrooms: $-6000.00
```

## Key Points Demonstrated

1. **Closed-Form Solution** - LinearSHAP computes SHAP values in O(n) time
2. **Exact Values** - Formula: SHAP[i] = weight[i] × (x[i] - E[x[i]])
3. **Interpretability** - Each feature's contribution is directly proportional to its deviation from the mean
4. **Local Accuracy** - SHAP values sum exactly to (prediction - base value)

## Understanding the Results

For the first house (2200 sqft, 4 bed, 3 bath, 8 years):

| Feature | Value | Mean | Deviation | Weight | SHAP |
|---------|-------|------|-----------|--------|------|
| sqft | 2200 | 1800 | +400 | 100 | +$40,000 |
| bedrooms | 4 | 3.2 | +0.8 | 5000 | +$4,000 |
| bathrooms | 3 | 2.0 | +1.0 | 10000 | +$10,000 |
| age | 8 | 16 | -8 | -500 | +$4,000 |

Note: Age has a negative weight (-500), so being newer (negative deviation) results in a positive SHAP contribution.
