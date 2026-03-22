# API Reference

Complete API documentation for SHAP-Go packages.

## Package Overview

| Package | Description |
|---------|-------------|
| `explainer` | Core interfaces, types, and configuration options |
| `explainer/tree` | TreeSHAP for tree ensembles (XGBoost, LightGBM, CatBoost) |
| `explainer/linear` | LinearSHAP for linear models |
| `explainer/kernel` | KernelSHAP for model-agnostic explanations |
| `explainer/exact` | ExactSHAP for brute-force exact computation |
| `explainer/deepshap` | DeepSHAP for neural networks |
| `explainer/gradient` | GradientSHAP using expected gradients |
| `explainer/partition` | PartitionSHAP for hierarchical feature groupings |
| `explainer/additive` | AdditiveSHAP for Generalized Additive Models |
| `explainer/permutation` | PermutationSHAP for black-box models |
| `explainer/sampling` | SamplingSHAP (Monte Carlo approximation) |
| `explanation` | Explanation types and methods |
| `model` | Model interfaces and adapters |
| `model/onnx` | ONNX Runtime integration |
| `background` | Background data utilities |
| `masker` | Feature masking strategies |
| `render` | Visualization chart generation |

---

## explainer

Core interfaces, types, and configuration shared by all explainers.

### Explainer Interface

```go
type Explainer interface {
    // Explain computes SHAP values for a single instance
    Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

    // ExplainBatch computes SHAP values for multiple instances
    ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

    // BaseValue returns E[f(X)] - expected model output
    BaseValue() float64

    // FeatureNames returns the feature names
    FeatureNames() []string
}
```

### Config

```go
type Config struct {
    // NumSamples is the number of Monte Carlo samples (default: 100)
    NumSamples int

    // Seed is the random seed for reproducibility (nil uses current time)
    Seed *int64

    // NumWorkers is the number of parallel workers (0 = sequential)
    NumWorkers int

    // ModelID is an optional identifier for the model
    ModelID string

    // FeatureNames are the names of the input features
    FeatureNames []string

    // ConfidenceLevel for confidence intervals (0 = disabled, e.g., 0.95 for 95% CI)
    ConfidenceLevel float64

    // UseBatchedPredictions enables batched model predictions for efficiency
    UseBatchedPredictions bool
}
```

### Options

```go
// WithNumSamples sets the number of samples for sampling-based explainers
func WithNumSamples(n int) Option

// WithSeed sets random seed for reproducibility
func WithSeed(seed int64) Option

// WithNumWorkers sets parallel workers for computation
func WithNumWorkers(n int) Option

// WithModelID sets the model identifier
func WithModelID(id string) Option

// WithFeatureNames sets feature names for explanations
func WithFeatureNames(names []string) Option

// WithConfidenceLevel sets confidence level for intervals (e.g., 0.95 for 95% CI)
func WithConfidenceLevel(level float64) Option

// WithBatchedPredictions enables batched model predictions for efficiency
func WithBatchedPredictions(enabled bool) Option
```

### Batch Parallel API

```go
// ExplainBatchParallel explains multiple instances in parallel using any explainer
func ExplainBatchParallel[E Explainer](
    ctx context.Context,
    exp E,
    instances [][]float64,
    config BatchConfig,
) ([]*explanation.Explanation, error)

// ExplainBatchWithProgress explains with progress callback
func ExplainBatchWithProgress[E Explainer](
    ctx context.Context,
    exp E,
    instances [][]float64,
    config BatchConfig,
    progress func(completed, total int),
) ([]*explanation.Explanation, error)

type BatchConfig struct {
    Workers     int  // Number of parallel workers (0 = GOMAXPROCS)
    StopOnError bool // Stop all workers on first error
}
```

---

## explanation

Explanation types and methods.

### Explanation

```go
type Explanation struct {
    // Values maps feature name to SHAP value
    Values map[string]float64

    // FeatureNames in order
    FeatureNames []string

    // FeatureValues maps feature name to instance value
    FeatureValues map[string]float64

    // Prediction is the model output for this instance
    Prediction float64

    // BaseValue is the expected value E[f(x)]
    BaseValue float64

    // ModelID identifies the model (optional)
    ModelID string

    // Timestamp when explanation was computed
    Timestamp time.Time

    // Metadata contains algorithm-specific information
    Metadata ExplanationMetadata
}
```

### ExplanationMetadata

```go
type ExplanationMetadata struct {
    // Algorithm used (e.g., "tree", "kernel", "permutation")
    Algorithm string

    // NumSamples used for sampling-based methods
    NumSamples int

    // BackgroundSize is the number of background samples
    BackgroundSize int

    // ComputeTimeMS is the computation time in milliseconds
    ComputeTimeMS int64

    // ConfidenceIntervals if computed
    ConfidenceIntervals *ConfidenceIntervals
}
```

### ConfidenceIntervals

```go
type ConfidenceIntervals struct {
    // Level is the confidence level (e.g., 0.95 for 95%)
    Level float64

    // Lower bounds for each feature
    Lower map[string]float64

    // Upper bounds for each feature
    Upper map[string]float64

    // StandardErrors for each feature
    StandardErrors map[string]float64
}
```

### Explanation Methods

```go
// TopFeatures returns the n features with highest absolute SHAP values
func (e *Explanation) TopFeatures(n int) []FeatureContribution

// Verify checks local accuracy: sum(SHAP) ≈ prediction - baseValue
func (e *Explanation) Verify(tolerance float64) VerificationResult

// HasConfidenceIntervals returns true if confidence intervals are available
func (e *Explanation) HasConfidenceIntervals() bool

// GetConfidenceInterval returns the CI for a feature (lower, upper, ok)
func (e *Explanation) GetConfidenceInterval(feature string) (float64, float64, bool)

// ToJSON serializes to JSON
func (e *Explanation) ToJSON() ([]byte, error)

// ToJSONPretty serializes to formatted JSON
func (e *Explanation) ToJSONPretty() ([]byte, error)
```

### Supporting Types

```go
type FeatureContribution struct {
    Name      string
    SHAPValue float64
    Index     int
}

type VerificationResult struct {
    Valid      bool    // Whether within tolerance
    Expected   float64 // prediction - baseValue
    SumSHAP    float64 // sum of SHAP values
    Difference float64 // |Expected - SumSHAP|
}
```

---

## explainer/tree

TreeSHAP for tree-based models (XGBoost, LightGBM, CatBoost).

### Explainer

```go
// New creates a TreeSHAP explainer from a tree ensemble
func New(ensemble *TreeEnsemble, opts ...explainer.Option) (*Explainer, error)

// Explain computes exact SHAP values for an instance (O(TLD²))
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// ExplainInteractions computes SHAP interaction values
func (e *Explainer) ExplainInteractions(ctx context.Context, instance []float64) (*InteractionResult, error)
```

### InteractionResult

```go
type InteractionResult struct {
    // Interactions[i][j] is the interaction between features i and j
    // Diagonal elements are main effects
    Interactions [][]float64

    // FeatureNames for indexing
    FeatureNames []string

    // Prediction for this instance
    Prediction float64

    // BaseValue E[f(X)]
    BaseValue float64
}

// MainEffect returns the main effect for a feature
func (r *InteractionResult) MainEffect(feature int) float64

// Interaction returns the interaction between two features
func (r *InteractionResult) Interaction(i, j int) float64
```

### TreeEnsemble

```go
type TreeEnsemble struct {
    Trees        []*Tree
    NumTrees     int
    NumFeatures  int
    FeatureNames []string
    BaseScore    float64
    Objective    string
}

// Predict computes model output for an instance
func (e *TreeEnsemble) Predict(instance []float64) float64
```

### Model Loading

```go
// XGBoost
func LoadXGBoostModel(path string) (*TreeEnsemble, error)
func LoadXGBoostModelFromReader(r io.Reader) (*TreeEnsemble, error)
func ParseXGBoostJSON(data []byte) (*TreeEnsemble, error)

// LightGBM (JSON format)
func LoadLightGBMModel(path string) (*TreeEnsemble, error)
func LoadLightGBMModelFromReader(r io.Reader) (*TreeEnsemble, error)
func ParseLightGBMJSON(data []byte) (*TreeEnsemble, error)

// LightGBM (text format)
func LoadLightGBMTextModel(path string) (*TreeEnsemble, error)
func ParseLightGBMText(data []byte) (*TreeEnsemble, error)

// CatBoost
func LoadCatBoostModel(path string) (*TreeEnsemble, error)
func ParseCatBoostJSON(data []byte) (*TreeEnsemble, error)

// ONNX-ML TreeEnsemble
func ParseONNXTreeEnsemble(modelPath string) (*TreeEnsemble, error)
func ParseONNXTreeEnsembleFromBytes(data []byte) (*TreeEnsemble, error)
```

---

## explainer/linear

LinearSHAP for linear models with closed-form solution.

### Explainer

```go
// New creates a LinearSHAP explainer
// weights: model coefficients, intercept: bias term
func New(weights []float64, intercept float64, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes exact SHAP values (O(n) complexity)
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

---

## explainer/kernel

KernelSHAP for model-agnostic explanations using weighted linear regression.

### Explainer

```go
// New creates a KernelSHAP explainer
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using weighted linear regression
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

**Supports:** `WithNumSamples`, `WithSeed`, `WithNumWorkers`, `WithBatchedPredictions`

---

## explainer/exact

ExactSHAP for brute-force exact Shapley value computation.

### Explainer

```go
// New creates an ExactSHAP explainer (max 20 features due to O(2^n) complexity)
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes exact SHAP values by enumerating all 2^n coalitions
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

**Supports:** `WithBatchedPredictions`

**Limitations:** Maximum 20 features (configurable via `MaxFeatures` constant)

---

## explainer/deepshap

DeepSHAP for neural networks using DeepLIFT attribution rules.

### Explainer

```go
// New creates a DeepSHAP explainer from an ONNX activation session
func New(session *onnx.ActivationSession, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using DeepLIFT rescale rule
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

**Supported Layers:** Dense/Gemm, ReLU, Sigmoid, Tanh, Softmax, Add, Identity

---

## explainer/gradient

GradientSHAP using expected gradients with numerical differentiation.

### Explainer

```go
// New creates a GradientSHAP explainer
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using expected gradients
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

### GradientSHAP-Specific Options

```go
// WithEpsilon sets the finite difference step size (default: 1e-7)
func WithEpsilon(eps float64) Option

// WithNoiseStdev sets noise standard deviation for SmoothGrad (default: 0)
func WithNoiseStdev(stdev float64) Option

// WithLocalSmoothing sets the number of noise samples (default: 1)
func WithLocalSmoothing(n int) Option
```

**Supports:** `WithNumSamples`, `WithSeed`, `WithNumWorkers`, `WithConfidenceLevel`

---

## explainer/partition

PartitionSHAP for hierarchical Owen values with feature groupings.

### Explainer

```go
// New creates a PartitionSHAP explainer
// hierarchy: feature hierarchy tree (nil for flat mode)
func New(m model.Model, background [][]float64, hierarchy *Node, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using hierarchical Owen values
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// Hierarchy returns the feature hierarchy
func (e *Explainer) Hierarchy() *Node
```

### Node

```go
type Node struct {
    // Name of this node (feature or group name)
    Name string

    // FeatureIdx for leaf nodes (-1 for internal nodes)
    FeatureIdx int

    // Children for internal nodes (empty for leaves)
    Children []*Node
}

// IsLeaf returns true if this is a leaf node
func (n *Node) IsLeaf() bool

// GetFeatureIndices returns all feature indices under this node
func (n *Node) GetFeatureIndices() []int
```

**Supports:** `WithNumSamples`, `WithSeed`, `WithBatchedPredictions`

---

## explainer/additive

AdditiveSHAP for Generalized Additive Models (GAMs).

### Explainer

```go
// New creates an AdditiveSHAP explainer for additive models (no interactions)
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes exact SHAP values (O(n) complexity)
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// Reference returns the reference point (mean of background)
func (e *Explainer) Reference() []float64

// ExpectedEffects returns precomputed E[fᵢ(Xᵢ)] for each feature
func (e *Explainer) ExpectedEffects() []float64
```

---

## explainer/permutation

PermutationSHAP for black-box models with guaranteed local accuracy.

### Explainer

```go
// New creates a PermutationSHAP explainer with antithetic sampling
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using permutation sampling
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

**Supports:** `WithNumSamples`, `WithSeed`, `WithNumWorkers`, `WithConfidenceLevel`

---

## explainer/sampling

SamplingSHAP using Monte Carlo estimation.

### Explainer

```go
// New creates a SamplingSHAP explainer
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes approximate SHAP values
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)
```

**Supports:** `WithNumSamples`, `WithSeed`, `WithConfidenceLevel`

---

## model

Model interfaces and adapters.

### Model Interface

```go
type Model interface {
    // Predict returns model output for an input
    Predict(ctx context.Context, input []float64) (float64, error)

    // PredictBatch returns outputs for multiple inputs
    PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error)

    // NumFeatures returns the number of input features
    NumFeatures() int
}
```

### FuncModel

Wraps a prediction function as a Model:

```go
// NewFuncModel creates a Model from a prediction function
func NewFuncModel(
    predict func(ctx context.Context, input []float64) (float64, error),
    numFeatures int,
) Model
```

Example:

```go
predict := func(ctx context.Context, input []float64) (float64, error) {
    return input[0]*2 + input[1]*3, nil
}
m := model.NewFuncModel(predict, 2)
```

---

## model/onnx

ONNX Runtime integration for model inference.

### Runtime Management

```go
// InitializeRuntime loads the ONNX Runtime library
func InitializeRuntime(libraryPath string) error

// DestroyRuntime releases ONNX Runtime resources
func DestroyRuntime()
```

### Session

```go
// NewSession creates an ONNX inference session
func NewSession(config Config) (*Session, error)

// Predict runs inference for a single input
func (s *Session) Predict(ctx context.Context, input []float64) (float64, error)

// PredictBatch runs inference for multiple inputs
func (s *Session) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error)

// Close releases session resources
func (s *Session) Close() error

// NumFeatures returns the number of input features
func (s *Session) NumFeatures() int
```

### ActivationSession

For DeepSHAP with intermediate layer access:

```go
// NewActivationSession creates a session that captures intermediate activations
func NewActivationSession(config ActivationConfig) (*ActivationSession, error)

// PredictWithActivations returns prediction and layer activations
func (s *ActivationSession) PredictWithActivations(ctx context.Context, input []float64) (*ActivationResult, error)

type ActivationConfig struct {
    Config
    IntermediateOutputs []string // Layer outputs to capture
}

type ActivationResult struct {
    Prediction  float64
    Activations map[string][]float32
}
```

### Graph Parsing

```go
// ParseGraph parses ONNX model graph structure
func ParseGraph(modelPath string) (*GraphInfo, error)

// ParseGraphFromBytes parses from model bytes
func ParseGraphFromBytes(data []byte) (*GraphInfo, error)

type GraphInfo struct {
    Nodes            []NodeInfo
    TopologicalOrder []string
}

type NodeInfo struct {
    Name      string
    OpType    string
    LayerType LayerType
    Inputs    []string
    Outputs   []string
}
```

### Config

```go
type Config struct {
    ModelPath   string // Path to ONNX model file
    InputName   string // Input tensor name
    OutputName  string // Output tensor name
    NumFeatures int    // Number of input features
    OutputIndex int    // Output index for multi-output models
}
```

---

## background

Background data utilities.

### Dataset

```go
type Dataset struct {
    Data         [][]float64
    FeatureNames []string
}

// NewDataset creates a background dataset
func NewDataset(data [][]float64, featureNames []string) (*Dataset, error)

// Sample returns a random subset of the background data
func (d *Dataset) Sample(n int, rng *rand.Rand) [][]float64

// Mean returns the mean of each feature
func (d *Dataset) Mean() []float64

// Subset returns rows at the given indices
func (d *Dataset) Subset(indices []int) [][]float64
```

---

## masker

Feature masking strategies.

### IndependentMasker

```go
type IndependentMasker struct {
    Background [][]float64
}

// NewIndependentMasker creates a masker using independent feature assumption
func NewIndependentMasker(background [][]float64) *IndependentMasker

// Mask replaces masked features with background values
func (m *IndependentMasker) Mask(instance []float64, mask []bool) ([]float64, error)

// MaskWithBackground uses a specific background sample
func (m *IndependentMasker) MaskWithBackground(instance []float64, mask []bool, bgIndex int) ([]float64, error)
```

---

## render

Chart generation in ChartIR format.

### Waterfall

```go
// Waterfall creates a waterfall chart specification
func Waterfall(explanation *explanation.Explanation, opts WaterfallOptions) *chartir.Chart

type WaterfallOptions struct {
    Title       string
    MaxFeatures int
    ShowValues  bool
    Features    []string // Specific features to show
}
```

### FeatureImportance

```go
// FeatureImportance creates a bar chart of feature importance
func FeatureImportance(explanations []*explanation.Explanation, opts ImportanceOptions) *chartir.Chart

type ImportanceOptions struct {
    Title           string
    MaxFeatures     int
    SortBy          string   // "mean_abs", "max_abs", "variance"
    ExcludeFeatures []string
}
```

### Summary

```go
// Summary creates a beeswarm/summary plot
func Summary(
    explanations []*explanation.Explanation,
    featureValues [][]float64,
    opts SummaryOptions,
) *chartir.Chart

type SummaryOptions struct {
    Title       string
    MaxFeatures int
    ColorScale  string // "bluered", "viridis", "plasma"
}
```

### Dependence

```go
// Dependence creates a dependence scatter plot
func Dependence(
    explanations []*explanation.Explanation,
    featureValues [][]float64,
    opts DependenceOptions,
) *chartir.Chart

type DependenceOptions struct {
    Feature      string // Feature to analyze
    ColorFeature string // Feature for color coding
    Title        string
}
```

---

## Error Types

### Common Errors

```go
var (
    ErrNilModel                  = errors.New("model cannot be nil")
    ErrNoBackground              = errors.New("background data cannot be empty")
    ErrFeatureMismatch           = errors.New("feature count mismatch")
    ErrTooManyFeatures           = errors.New("too many features for exact computation")
    ErrInstanceFeatureMismatch   = errors.New("instance feature count mismatch")
    ErrMaskFeatureMismatch       = errors.New("mask feature count mismatch")
    ErrBackgroundFeatureMismatch = errors.New("background feature count mismatch")
)
```

### Checking Errors

```go
explanation, err := exp.Explain(ctx, instance)
if err != nil {
    if errors.Is(err, explainer.ErrFeatureMismatch) {
        // Handle feature mismatch
    }
    return err
}
```

---

## Context Support

All `Explain` methods accept `context.Context` for:

- **Cancellation**: Stop long computations
- **Timeouts**: Limit computation time
- **Tracing**: Integrate with observability tools

```go
// With timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

explanation, err := exp.Explain(ctx, instance)
if errors.Is(err, context.DeadlineExceeded) {
    log.Println("Explanation timed out")
}
```

---

## Thread Safety

- All explainers are **safe for concurrent use** after creation
- Create once, use from multiple goroutines
- Internal state is read-only after initialization

```go
exp, _ := tree.New(ensemble)

// Safe: concurrent explains
var wg sync.WaitGroup
for _, instance := range instances {
    wg.Add(1)
    go func(inst []float64) {
        defer wg.Done()
        exp.Explain(ctx, inst) // Safe
    }(instance)
}
wg.Wait()
```

---

## Explainer Comparison

| Explainer | Model Type | Complexity | Exact | Interactions |
|-----------|------------|------------|-------|--------------|
| TreeSHAP | Tree ensembles | O(TLD²) | ✅ | ✅ |
| LinearSHAP | Linear | O(n) | ✅ | ❌ |
| AdditiveSHAP | GAMs | O(n×b) | ✅ | ❌ |
| ExactSHAP | Any | O(n×2ⁿ×b) | ✅ | ❌ |
| KernelSHAP | Any | O(s×b) | ❌ | ❌ |
| PermutationSHAP | Any | O(s×n×b) | ❌ | ❌ |
| SamplingSHAP | Any | O(s×n×b) | ❌ | ❌ |
| GradientSHAP | Differentiable | O(s×n×b) | ❌ | ❌ |
| PartitionSHAP | Structured | O(s×g×b) | ❌ | ❌ |
| DeepSHAP | Neural nets | O(L×b) | ❌ | ❌ |

Where: n=features, b=background, s=samples, T=trees, L=layers, D=depth, g=groups
