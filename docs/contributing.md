# Contributing

We welcome contributions to SHAP-Go! This guide will help you get started.

## Getting Started

### Prerequisites

- Go 1.21 or later
- Git
- (Optional) Python 3.8+ for comparison tests

### Clone the Repository

```bash
git clone https://github.com/plexusone/shap-go.git
cd shap-go
```

### Run Tests

```bash
go test -v ./...
```

### Run Linter

```bash
golangci-lint run
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the code style guidelines below.

### 3. Write Tests

Add tests for new functionality:

```go
func TestYourFeature(t *testing.T) {
    // Test implementation
}
```

### 4. Run All Checks

```bash
# Tests
go test -v ./...

# Linting
golangci-lint run

# Coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### 5. Commit

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add support for XYZ"
git commit -m "fix: resolve issue with ABC"
git commit -m "docs: update installation guide"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## Code Style

### Go Conventions

- Use `gofmt` for formatting
- Follow [Effective Go](https://golang.org/doc/effective_go)
- Keep functions focused and small
- Prefer returning errors over panicking

### Naming

```go
// Good: Clear, descriptive names
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*Explanation, error)

// Avoid: Abbreviated or unclear names
func (e *Explainer) Exp(c context.Context, i []float64) (*Explanation, error)
```

### Error Handling

```go
// Good: Return errors to caller
func LoadModel(path string) (*Model, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("reading model file: %w", err)
    }
    // ...
}

// Avoid: Ignoring errors
func LoadModel(path string) *Model {
    data, _ := os.ReadFile(path)  // Bad!
    // ...
}
```

### Documentation

```go
// Explain computes SHAP values for a single instance.
//
// The returned Explanation contains feature contributions that sum to
// prediction - baseValue (local accuracy property).
//
// Returns an error if the instance has the wrong number of features.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*Explanation, error)
```

## Testing

### Unit Tests

Place tests in `*_test.go` files:

```go
func TestExplainer_Explain(t *testing.T) {
    // Setup
    ensemble := createTestEnsemble()
    exp, err := tree.New(ensemble)
    if err != nil {
        t.Fatal(err)
    }

    // Execute
    explanation, err := exp.Explain(context.Background(), []float64{1.0, 2.0})
    if err != nil {
        t.Fatal(err)
    }

    // Verify
    if explanation.Prediction == 0 {
        t.Error("expected non-zero prediction")
    }
}
```

### Table-Driven Tests

```go
func TestVerify(t *testing.T) {
    tests := []struct {
        name      string
        tolerance float64
        wantValid bool
    }{
        {"tight tolerance", 1e-10, true},
        {"loose tolerance", 1.0, true},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := explanation.Verify(tt.tolerance)
            if result.Valid != tt.wantValid {
                t.Errorf("got valid=%v, want %v", result.Valid, tt.wantValid)
            }
        })
    }
}
```

### Benchmarks

```go
func BenchmarkExplain(b *testing.B) {
    exp := setupExplainer()
    instance := []float64{1.0, 2.0, 3.0}
    ctx := context.Background()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        exp.Explain(ctx, instance)
    }
}
```

## Adding New Explainers

To add a new explainer type:

### 1. Create Package

```
explainer/
├── newtype/
│   ├── newtype.go
│   └── newtype_test.go
```

### 2. Implement Interface

```go
package newtype

import (
    "context"
    "github.com/plexusone/shap-go/explainer"
)

type Explainer struct {
    // fields
}

func New(/* params */, opts ...explainer.Option) (*Explainer, error) {
    // implementation
}

func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explainer.Explanation, error) {
    // implementation
}

func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explainer.Explanation, error) {
    // implementation
}
```

### 3. Add Tests

Verify local accuracy (sum of SHAP values = prediction - baseValue):

```go
func TestLocalAccuracy(t *testing.T) {
    exp := createExplainer()
    explanation, _ := exp.Explain(ctx, instance)

    result := explanation.Verify(1e-6)
    if !result.Valid {
        t.Errorf("local accuracy failed: diff=%v", result.Difference)
    }
}
```

### 4. Add Documentation

Create `docs/explainers/newtype.md` with:

- When to use
- Basic usage
- Configuration options
- Examples

### 5. Update Navigation

Add to `mkdocs.yml`:

```yaml
nav:
  - Explainers:
    - NewType: explainers/newtype.md
```

## Adding Model Parsers

To add support for a new model format:

### 1. Create Parser

```go
// explainer/tree/parse_newformat.go

func LoadNewFormatModel(path string) (*TreeEnsemble, error) {
    // implementation
}

func ParseNewFormatJSON(data []byte) (*TreeEnsemble, error) {
    // implementation
}
```

### 2. Add Tests

```go
func TestParseNewFormat(t *testing.T) {
    // Test with sample model
}
```

### 3. Add Documentation

Create `docs/models/newformat.md`.

## Pull Request Guidelines

### PR Title

Use conventional commit format:

```
feat: add KernelSHAP explainer
fix: correct SHAP value summation
docs: add LightGBM export guide
```

### PR Description

Include:

- What the change does
- Why it's needed
- How to test it
- Any breaking changes

### Checklist

Before submitting:

- [ ] Tests pass: `go test -v ./...`
- [ ] Linting passes: `golangci-lint run`
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional commits

## Issue Guidelines

### Bug Reports

Include:

- SHAP-Go version
- Go version
- Minimal reproduction case
- Expected vs actual behavior

### Feature Requests

Include:

- Use case description
- Proposed API (if applicable)
- Alternatives considered

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs for similar topics

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
