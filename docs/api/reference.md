# API Reference

Complete API documentation for SHAP-Go packages.

## Package Overview

| Package | Description |
|---------|-------------|
| `explainer` | Core interfaces and types |
| `explainer/tree` | TreeSHAP for tree ensembles |
| `explainer/permutation` | PermutationSHAP for black-box models |
| `explainer/sampling` | SamplingSHAP (Monte Carlo approximation) |
| `model` | Model interfaces and adapters |
| `model/onnx` | ONNX Runtime integration |
| `render` | Visualization chart generation |

---

## explainer

Core interfaces and types shared by all explainers.

### Explainer Interface

```go
type Explainer interface {
    // Explain computes SHAP values for a single instance
    Explain(ctx context.Context, instance []float64) (*Explanation, error)

    // ExplainBatch computes SHAP values for multiple instances
    ExplainBatch(ctx context.Context, instances [][]float64) ([]*Explanation, error)
}
```

### Explanation

```go
type Explanation struct {
    // Values maps feature name to SHAP value
    Values map[string]float64

    // FeatureNames in order
    FeatureNames []string

    // Prediction is the model output for this instance
    Prediction float64

    // BaseValue is the expected value E[f(x)]
    BaseValue float64

    // ModelID identifies the model (optional)
    ModelID string

    // Timestamp when explanation was computed
    Timestamp time.Time
}
```

#### Methods

```go
// TopFeatures returns the n features with highest absolute SHAP values
func (e *Explanation) TopFeatures(n int) []FeatureContribution

// Verify checks local accuracy: sum(SHAP) ≈ prediction - baseValue
func (e *Explanation) Verify(tolerance float64) VerificationResult

// ToJSON serializes to JSON
func (e *Explanation) ToJSON() ([]byte, error)

// ToJSONPretty serializes to formatted JSON
func (e *Explanation) ToJSONPretty() ([]byte, error)
```

### FeatureContribution

```go
type FeatureContribution struct {
    Name      string
    SHAPValue float64
    Index     int
}
```

### VerificationResult

```go
type VerificationResult struct {
    Valid      bool    // Whether within tolerance
    Expected   float64 // prediction - baseValue
    Actual     float64 // sum of SHAP values
    Difference float64 // |Expected - Actual|
}
```

### Options

```go
// WithNumSamples sets the number of samples for sampling-based explainers
func WithNumSamples(n int) Option

// WithNumWorkers sets parallel workers for batch processing
func WithNumWorkers(n int) Option

// WithSeed sets random seed for reproducibility
func WithSeed(seed int64) Option

// WithFeatureNames sets feature names for explanations
func WithFeatureNames(names []string) Option

// WithModelID sets the model identifier
func WithModelID(id string) Option
```

---

## explainer/tree

TreeSHAP implementation for tree-based models.

### Explainer

```go
type Explainer struct {
    // contains filtered or unexported fields
}

// New creates a TreeSHAP explainer from a tree ensemble
func New(ensemble *TreeEnsemble, opts ...explainer.Option) (*Explainer, error)

// Explain computes exact SHAP values for an instance
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explainer.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances in parallel
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explainer.Explanation, error)
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

### Tree

```go
type Tree struct {
    Nodes []*Node
    Root  int
}
```

### Node

```go
type Node struct {
    Feature      int      // Feature index (-1 for leaf)
    Threshold    float64  // Split threshold
    Left         int      // Left child index
    Right        int      // Right child index
    Missing      int      // Missing value direction
    Value        float64  // Leaf value
    Cover        float64  // Number of samples
    IsLeaf       bool
    DecisionType string   // "<" or "<="
}
```

### Model Loading

```go
// LoadXGBoostModel loads from XGBoost JSON file
func LoadXGBoostModel(path string) (*TreeEnsemble, error)

// LoadXGBoostModelFromReader loads from io.Reader
func LoadXGBoostModelFromReader(r io.Reader) (*TreeEnsemble, error)

// ParseXGBoostJSON parses XGBoost JSON bytes
func ParseXGBoostJSON(data []byte) (*TreeEnsemble, error)

// LoadLightGBMModel loads from LightGBM JSON file
func LoadLightGBMModel(path string) (*TreeEnsemble, error)

// LoadLightGBMModelFromReader loads from io.Reader
func LoadLightGBMModelFromReader(r io.Reader) (*TreeEnsemble, error)

// ParseLightGBMJSON parses LightGBM JSON bytes
func ParseLightGBMJSON(data []byte) (*TreeEnsemble, error)
```

---

## explainer/permutation

PermutationSHAP for black-box models with guaranteed local accuracy.

### Explainer

```go
type Explainer struct {
    // contains filtered or unexported fields
}

// New creates a PermutationSHAP explainer
func New(model model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes SHAP values using antithetic sampling
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explainer.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explainer.Explanation, error)
```

---

## explainer/sampling

SamplingSHAP using simple Monte Carlo estimation.

### Explainer

```go
type Explainer struct {
    // contains filtered or unexported fields
}

// New creates a SamplingSHAP explainer
func New(model model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)

// Explain computes approximate SHAP values
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explainer.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explainer.Explanation, error)
```

---

## model

Model interfaces and adapters.

### Model Interface

```go
type Model interface {
    // Predict returns model output for an input
    Predict(ctx context.Context, input []float64) (float64, error)

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
type Session struct {
    // contains filtered or unexported fields
}

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

### Config

```go
type Config struct {
    // ModelPath is the path to the ONNX model file
    ModelPath string

    // InputName is the name of the input tensor
    InputName string

    // OutputName is the name of the output tensor
    OutputName string

    // NumFeatures is the number of input features
    NumFeatures int

    // OutputIndex selects which output to use (for multi-output models)
    OutputIndex int
}
```

---

## render

Chart generation in ChartIR format.

### Waterfall

```go
// Waterfall creates a waterfall chart specification
func Waterfall(explanation *explainer.Explanation, opts WaterfallOptions) *Chart

type WaterfallOptions struct {
    Title       string
    MaxFeatures int
    ShowValues  bool
    Features    []string  // Specific features to show
}
```

### FeatureImportance

```go
// FeatureImportance creates a bar chart of feature importance
func FeatureImportance(explanations []*explainer.Explanation, opts ImportanceOptions) *Chart

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
    explanations []*explainer.Explanation,
    featureValues [][]float64,
    opts SummaryOptions,
) *Chart

type SummaryOptions struct {
    Title       string
    MaxFeatures int
    ColorScale  string  // "bluered", "viridis", "plasma"
}
```

### Dependence

```go
// Dependence creates a dependence scatter plot
func Dependence(
    explanations []*explainer.Explanation,
    featureValues [][]float64,
    opts DependenceOptions,
) *Chart

type DependenceOptions struct {
    Feature      string  // Feature to analyze
    ColorFeature string  // Feature for color coding (optional)
    Title        string
}
```

### Chart

```go
type Chart struct {
    Type     string                 `json:"type"`
    Title    string                 `json:"title,omitempty"`
    Subtitle string                 `json:"subtitle,omitempty"`
    Data     map[string]interface{} `json:"data"`
}
```

---

## Error Types

### Common Errors

```go
var (
    ErrInvalidModel      = errors.New("invalid model")
    ErrInvalidInput      = errors.New("invalid input")
    ErrFeatureMismatch   = errors.New("feature count mismatch")
    ErrEmptyBackground   = errors.New("background data is empty")
    ErrNoTrees           = errors.New("model has no trees")
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
        exp.Explain(ctx, inst)  // Safe
    }(instance)
}
wg.Wait()
```
