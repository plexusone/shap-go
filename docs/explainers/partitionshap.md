# PartitionSHAP

PartitionSHAP organizes features into a hierarchical tree structure and computes SHAP values using Owen values. This approach is useful when features are naturally grouped or when you want hierarchical explanations.

## When to Use PartitionSHAP

- Features have natural groupings (demographics, financials, etc.)
- Feature correlations within groups are stronger than between groups
- You want group-level explanations before feature-level detail
- Interpretability benefits from domain-driven feature organization

## Basic Usage

```go
import (
    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/partition"
)

// Without hierarchy (flat mode - equivalent to standard SHAP)
exp, err := partition.New(model, background, nil,
    explainer.WithNumSamples(500),
    explainer.WithSeed(42),
)

result, err := exp.Explain(ctx, instance)
```

## With Feature Hierarchy

```go
// Define a meaningful feature hierarchy
hierarchy := &partition.Node{
    Name: "root",
    Children: []*partition.Node{
        {Name: "demographics", Children: []*partition.Node{
            {Name: "age", FeatureIdx: 0},
            {Name: "gender", FeatureIdx: 1},
            {Name: "education", FeatureIdx: 2},
        }},
        {Name: "financials", Children: []*partition.Node{
            {Name: "income", FeatureIdx: 3},
            {Name: "debt", FeatureIdx: 4},
            {Name: "savings", FeatureIdx: 5},
        }},
        {Name: "behavior", Children: []*partition.Node{
            {Name: "purchases", FeatureIdx: 6},
            {Name: "logins", FeatureIdx: 7},
        }},
    },
}

exp, err := partition.New(model, background, hierarchy,
    explainer.WithNumSamples(500),
    explainer.WithFeatureNames([]string{
        "age", "gender", "education",
        "income", "debt", "savings",
        "purchases", "logins",
    }),
)

result, err := exp.Explain(ctx, instance)
```

## How It Works

### Owen Values

PartitionSHAP uses Owen values, which extend Shapley values to hierarchical structures:

1. At each level of the hierarchy, children (groups or features) are treated as players
2. Marginal contributions are computed by sampling permutations
3. The contribution is recursively distributed down to individual features
4. The final attribution satisfies efficiency: sum(SHAP) = prediction - baseline

### Algorithm

1. Start at the root of the hierarchy
2. For each internal node with multiple children:
   - Sample permutations of children
   - Compute marginal contribution for each child
   - Scale contributions to sum to the parent's total contribution
3. For leaf nodes (individual features), assign the contribution directly
4. Result satisfies local accuracy property

## Configuration Options

| Option | Description |
|--------|-------------|
| `WithNumSamples(n)` | Number of permutation samples per node (default: 100) |
| `WithSeed(s)` | Random seed for reproducibility |
| `WithFeatureNames(names)` | Feature names for result labeling |

## Hierarchy Requirements

- The hierarchy must be a tree (no cycles)
- Every feature must appear exactly once as a leaf
- Feature indices must be valid (0 to numFeatures-1)
- No duplicate feature indices allowed

### Flat Mode

If no hierarchy is provided, PartitionSHAP creates a flat hierarchy where each feature is its own group. This is equivalent to permutation-based SHAP.

## Example: Credit Risk Model

```go
// Features: age, income, debt, credit_score, employment_years, num_accounts

// Group by category
hierarchy := &partition.Node{
    Name: "root",
    Children: []*partition.Node{
        {Name: "personal", Children: []*partition.Node{
            {Name: "age", FeatureIdx: 0},
            {Name: "employment_years", FeatureIdx: 4},
        }},
        {Name: "financial", Children: []*partition.Node{
            {Name: "income", FeatureIdx: 1},
            {Name: "debt", FeatureIdx: 2},
            {Name: "num_accounts", FeatureIdx: 5},
        }},
        {Name: "credit", Children: []*partition.Node{
            {Name: "credit_score", FeatureIdx: 3},
        }},
    },
}

exp, err := partition.New(model, background, hierarchy,
    explainer.WithNumSamples(500),
)

result, err := exp.Explain(ctx, applicant)

// Result shows how personal, financial, and credit
// factors contribute to the risk prediction
```

## Comparison with Other Methods

| Method | Hierarchy Support | Computation |
|--------|-------------------|-------------|
| PermutationSHAP | No | O(n! * samples) |
| KernelSHAP | No | O(2^n * samples) |
| PartitionSHAP | Yes | O(k! * samples * depth) |

Where k is the max children per node (usually much smaller than n).

## Limitations

- Requires defining a meaningful hierarchy (domain knowledge needed)
- More samples needed for deeper hierarchies
- Flat mode has similar complexity to permutation methods

## References

- Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees" (2020)
- Owen, G. "Values of Games with A Priori Unions" (1977)
