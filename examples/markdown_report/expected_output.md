# Markdown Report Expected Output

Run with: `go run main.go > report.md`

## Description

This example generates a complete SHAP explanation report in Pandoc Markdown format. The output can be converted to PDF, HTML, or other formats.

## Converting the Report

```bash
# Generate the report
go run main.go > report.md

# Convert to PDF (requires pandoc and LaTeX)
pandoc report.md -o report.pdf

# Convert to HTML
pandoc report.md -o report.html --standalone
```

## Report Structure

The generated report includes:

1. **YAML Frontmatter** - Title, author, date for Pandoc
2. **Executive Summary** - Overview of SHAP analysis
3. **Individual Prediction Analysis**
   - Applicant profile table
   - Prediction breakdown
   - Feature contributions with visual bars
   - ASCII waterfall visualization
4. **Global Feature Importance** - Aggregated across all predictions
5. **Prediction Summary** - Top positive/negative features per prediction
6. **Local Accuracy Verification** - SHAP sum validation
7. **Methodology** - Algorithm details

## Sample Output Excerpt

```markdown
---
title: "SHAP Explanation Report"
subtitle: "Loan Approval Model Analysis"
author: "SHAP-Go"
date: "March 20, 2026"
---

# Executive Summary

This report provides SHAP analysis for a loan approval model...

# Individual Prediction Analysis

## Applicant Profile

| Feature | Value |
|---------|-------|
| income | 75000.00 |
| age | 35.00 |
| credit_score | 750.00 |
| debt_ratio | 0.15 |

## Feature Contributions

| Feature | Value | SHAP | Direction | Impact |
|---------|-------|------|-----------|--------|
| income | 75000.00 | +0.1234 | (+) increases | `#####` |
| credit_score | 750.00 | +0.0567 | (+) increases | `###` |
| age | 35.00 | +0.0123 | (+) increases | `#` |
| debt_ratio | 0.15 | -0.0345 | (-) decreases | `--` |

## Waterfall Visualization

```
Base Value: 0.6500
                              │
income        +0.1234  ████████│████████████
credit_score  +0.0567  ████████│██████
age           +0.0123  ████████│█
debt_ratio    -0.0345  ██████──│
                              │
Prediction: 0.8079
```
```

## Key Points Demonstrated

1. **Pandoc Compatibility** - YAML frontmatter for metadata
2. **Tables** - Markdown tables for data presentation
3. **ASCII Visualization** - Text-based waterfall charts
4. **Comprehensive Report** - All aspects of SHAP explanation
5. **Verification** - Local accuracy checks included

## Use Cases

- **Regulatory Compliance** - Document model decisions
- **Audit Trails** - Explain individual predictions
- **Stakeholder Communication** - Non-technical report format
- **Archival** - Version-controlled explanation documents
