// Package partition provides a PartitionSHAP explainer for hierarchical feature grouping.
//
// PartitionSHAP organizes features into a hierarchical tree structure and computes
// SHAP values using Owen values. This approach is particularly useful when:
//
//   - Features are naturally grouped (e.g., age/gender/income in demographics)
//   - Feature correlations within groups are stronger than between groups
//   - A hierarchical explanation is desired (group-level then feature-level)
//
// # Algorithm Overview
//
// 1. Build a hierarchy tree where leaves are individual features
// 2. At each internal node, compute SHAP values treating children as single units
// 3. Recursively descend to attribute values to individual features
// 4. Uses Owen values to respect the hierarchical structure
//
// # Owen Values
//
// Owen values are a natural extension of Shapley values for hierarchical games.
// They ensure that:
//
//   - Features within the same group are treated symmetrically
//   - Groups are compared fairly at each level of the hierarchy
//   - The final attribution satisfies efficiency (sum equals prediction - baseline)
//
// # Usage
//
//	// Define a feature hierarchy
//	hierarchy := &partition.Node{
//		Name: "root",
//		Children: []*partition.Node{
//			{Name: "demographics", Children: []*partition.Node{
//				{Name: "age", FeatureIdx: 0},
//				{Name: "gender", FeatureIdx: 1},
//			}},
//			{Name: "financials", Children: []*partition.Node{
//				{Name: "income", FeatureIdx: 2},
//				{Name: "debt", FeatureIdx: 3},
//			}},
//		},
//	}
//
//	exp, err := partition.New(model, background, hierarchy, opts...)
//	result, err := exp.Explain(ctx, instance)
//
// # Flat Mode
//
// If no hierarchy is provided, PartitionSHAP falls back to treating each feature
// as its own group, equivalent to standard SHAP behavior.
//
// Reference: "From Local Explanations to Global Understanding with Explainable AI for Trees"
// (Lundberg et al., 2020)
package partition
