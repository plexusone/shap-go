package tree

import (
	"context"
	"fmt"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/permutation"
	"github.com/plexusone/shap-go/model"
)

// Benchmark TreeSHAP with varying number of trees.
// This tests how performance scales with ensemble size.

func BenchmarkTreeSHAP_Trees10(b *testing.B) {
	benchmarkTreeSHAPTrees(b, 10, 4, 10)
}

func BenchmarkTreeSHAP_Trees100(b *testing.B) {
	benchmarkTreeSHAPTrees(b, 100, 4, 10)
}

func BenchmarkTreeSHAP_Trees1000(b *testing.B) {
	benchmarkTreeSHAPTrees(b, 1000, 4, 10)
}

func benchmarkTreeSHAPTrees(b *testing.B, numTrees, depth, numFeatures int) {
	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark TreeSHAP with varying tree depth.
// Complexity is O(TLD²), so deeper trees increase compute time.

func BenchmarkTreeSHAP_Depth3(b *testing.B) {
	benchmarkTreeSHAPDepth(b, 50, 3, 10)
}

func BenchmarkTreeSHAP_Depth6(b *testing.B) {
	benchmarkTreeSHAPDepth(b, 50, 6, 10)
}

func BenchmarkTreeSHAP_Depth10(b *testing.B) {
	benchmarkTreeSHAPDepth(b, 50, 10, 10)
}

func benchmarkTreeSHAPDepth(b *testing.B, numTrees, depth, numFeatures int) {
	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark TreeSHAP with varying number of features.
// More features means larger SHAP value arrays and potentially more splits.

func BenchmarkTreeSHAP_Features5(b *testing.B) {
	benchmarkTreeSHAPFeatures(b, 50, 4, 5)
}

func BenchmarkTreeSHAP_Features20(b *testing.B) {
	benchmarkTreeSHAPFeatures(b, 50, 4, 20)
}

func BenchmarkTreeSHAP_Features50(b *testing.B) {
	benchmarkTreeSHAPFeatures(b, 50, 4, 50)
}

func benchmarkTreeSHAPFeatures(b *testing.B, numTrees, depth, numFeatures int) {
	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark batch processing with varying parallelism.

func BenchmarkTreeSHAP_Batch100_Workers1(b *testing.B) {
	benchmarkTreeSHAPBatch(b, 100, 1)
}

func BenchmarkTreeSHAP_Batch100_Workers4(b *testing.B) {
	benchmarkTreeSHAPBatch(b, 100, 4)
}

func BenchmarkTreeSHAP_Batch100_Workers8(b *testing.B) {
	benchmarkTreeSHAPBatch(b, 100, 8)
}

func benchmarkTreeSHAPBatch(b *testing.B, batchSize, numWorkers int) {
	numTrees, depth, numFeatures := 50, 4, 10

	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble, explainer.WithNumWorkers(numWorkers))
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	// Create batch of instances
	instances := make([][]float64, batchSize)
	for i := range instances {
		instances[i] = make([]float64, numFeatures)
		for j := range instances[i] {
			instances[i][j] = float64(i*numFeatures+j) / float64(batchSize*numFeatures)
		}
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.ExplainBatch(ctx, instances)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark comparison: TreeSHAP vs PermutationSHAP.
// This demonstrates the performance advantage of exact TreeSHAP over sampling.
//
// We use a small model (10 trees, depth 3, 5 features) because PermutationSHAP
// requires many model evaluations (2*(n_features+1)*n_samples per instance).

func BenchmarkComparison_TreeSHAP(b *testing.B) {
	numTrees, depth, numFeatures := 10, 3, 5

	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkComparison_PermutationSHAP_Samples10(b *testing.B) {
	benchmarkPermutationSHAP(b, 10)
}

func BenchmarkComparison_PermutationSHAP_Samples50(b *testing.B) {
	benchmarkPermutationSHAP(b, 50)
}

func BenchmarkComparison_PermutationSHAP_Samples100(b *testing.B) {
	benchmarkPermutationSHAP(b, 100)
}

func benchmarkPermutationSHAP(b *testing.B, numSamples int) {
	numTrees, depth, numFeatures := 10, 3, 5

	// Create ensemble and wrap it as a Model
	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	treeExp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create tree explainer: %v", err)
	}

	// Create a model wrapper for the tree ensemble
	predictFn := func(_ context.Context, input []float64) (float64, error) {
		return treeExp.predict(input), nil
	}
	m := model.NewFuncModel(predictFn, numFeatures)

	// Create background data
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, numFeatures)
		for j := range background[i] {
			background[i][j] = float64(i*numFeatures+j) / float64(10*numFeatures)
		}
	}

	// Create permutation explainer
	featureNames := make([]string, numFeatures)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("x%d", i)
	}

	exp, err := permutation.New(m, background,
		explainer.WithNumSamples(numSamples),
		explainer.WithFeatureNames(featureNames),
	)
	if err != nil {
		b.Fatalf("failed to create permutation explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark realistic model sizes typical for production XGBoost/LightGBM models.

func BenchmarkTreeSHAP_RealisticSmall(b *testing.B) {
	// Small model: 50 trees, depth 4, 10 features
	benchmarkTreeSHAPTrees(b, 50, 4, 10)
}

func BenchmarkTreeSHAP_RealisticMedium(b *testing.B) {
	// Medium model: 200 trees, depth 6, 30 features
	benchmarkTreeSHAPTrees(b, 200, 6, 30)
}

func BenchmarkTreeSHAP_RealisticLarge(b *testing.B) {
	// Large model: 500 trees, depth 8, 50 features
	benchmarkTreeSHAPTrees(b, 500, 8, 50)
}

// BenchmarkTreeSHAP_MemoryAllocs measures allocation patterns.
// Lower allocations mean less GC pressure in production.
func BenchmarkTreeSHAP_MemoryAllocs(b *testing.B) {
	numTrees, depth, numFeatures := 100, 5, 20

	ensemble := createBenchmarkEnsemble(numTrees, depth, numFeatures)
	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		result, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
		// Verify result is valid to prevent optimization
		if result.Prediction == 0 && len(result.Values) == 0 {
			b.Fatal("unexpected zero result")
		}
	}
}
