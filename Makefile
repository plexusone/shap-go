# shap-go Makefile

.PHONY: all test lint generate-testdata clean help

# Default target
all: lint test

# Run all tests
test:
	go test -v ./...

# Run tests with race detection
test-race:
	go test -race -v ./...

# Run TreeSHAP equivalence tests only
test-treeshap:
	go test -v ./explainer/tree/... -run TestAgainstPythonSHAP

# Run linter
lint:
	golangci-lint run ./...

# Format code
fmt:
	gofmt -w .

# Generate test data from Python SHAP
# This regenerates treeshap_test_cases.json using the Python SHAP library
# Note: On macOS, you may need to use conda. See testdata/python/README.md
generate-testdata:
	@echo "Setting up Python environment..."
	@cd testdata/python && \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip install -q -r requirements.txt && \
		echo "Generating test cases from Python SHAP..." && \
		python generate_test_cases.py > ../treeshap_test_cases.json
	@echo "Test cases generated: testdata/treeshap_test_cases.json"

# Generate test data using conda (for macOS)
generate-testdata-conda:
	@echo "Generating test cases using conda environment..."
	@cd testdata/python && \
		conda run -n shap-test python generate_test_cases.py > ../treeshap_test_cases.json
	@echo "Test cases generated: testdata/treeshap_test_cases.json"

# Verify Go implementation matches Python SHAP
# Regenerates test data and runs tests
verify-equivalence: generate-testdata test-treeshap

# Clean generated files
clean:
	rm -rf testdata/python/venv
	rm -rf testdata/python/__pycache__

# Show help
help:
	@echo "shap-go Makefile targets:"
	@echo ""
	@echo "  all                    - Run lint and test (default)"
	@echo "  test                   - Run all Go tests"
	@echo "  test-race              - Run tests with race detection"
	@echo "  test-treeshap          - Run TreeSHAP equivalence tests only"
	@echo "  lint                   - Run golangci-lint"
	@echo "  fmt                    - Format Go code"
	@echo "  generate-testdata      - Generate test cases from Python SHAP (pip)"
	@echo "  generate-testdata-conda- Generate test cases using conda (for macOS)"
	@echo "  verify-equivalence     - Regenerate test data and run equivalence tests"
	@echo "  clean                  - Remove generated files"
	@echo "  help                   - Show this help"
	@echo ""
	@echo "Note: On macOS, generate-testdata may fail due to llvmlite build issues."
	@echo "Use generate-testdata-conda or rely on CI for test data generation."
