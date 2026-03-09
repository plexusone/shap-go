package masker

import (
	"testing"
)

func TestNewIndependentMasker(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	m, err := NewIndependentMasker(background)
	if err != nil {
		t.Fatalf("NewIndependentMasker() error = %v", err)
	}

	if m.NumFeatures() != 3 {
		t.Errorf("NumFeatures() = %d, want 3", m.NumFeatures())
	}

	if m.BackgroundSize() != 3 {
		t.Errorf("BackgroundSize() = %d, want 3", m.BackgroundSize())
	}
}

func TestNewIndependentMasker_EmptyBackground(t *testing.T) {
	_, err := NewIndependentMasker([][]float64{})
	if err == nil {
		t.Error("NewIndependentMasker() should error with empty background")
	}
}

func TestNewIndependentMasker_InconsistentDimensions(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0}, // Wrong length
	}

	_, err := NewIndependentMasker(background)
	if err == nil {
		t.Error("NewIndependentMasker() should error with inconsistent dimensions")
	}
}

func TestIndependentMasker_MaskWithBackground(t *testing.T) {
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	m, err := NewIndependentMasker(background)
	if err != nil {
		t.Fatalf("NewIndependentMasker() error = %v", err)
	}

	instance := []float64{1.0, 2.0, 3.0}
	bgSample := []float64{0.0, 0.0, 0.0}

	tests := []struct {
		name     string
		mask     []bool
		expected []float64
	}{
		{
			name:     "no masking",
			mask:     []bool{false, false, false},
			expected: []float64{1.0, 2.0, 3.0},
		},
		{
			name:     "all masked",
			mask:     []bool{true, true, true},
			expected: []float64{0.0, 0.0, 0.0},
		},
		{
			name:     "partial masking",
			mask:     []bool{true, false, true},
			expected: []float64{0.0, 2.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := m.MaskWithBackground(instance, tt.mask, bgSample)

			for i, v := range result {
				if v != tt.expected[i] {
					t.Errorf("MaskWithBackground()[%d] = %f, want %f", i, v, tt.expected[i])
				}
			}
		})
	}
}

func TestIndependentMasker_Mask(t *testing.T) {
	background := [][]float64{
		{10.0, 20.0, 30.0},
		{10.0, 20.0, 30.0}, // Same values for deterministic test
	}

	m, err := NewIndependentMaskerWithSeed(background, 42)
	if err != nil {
		t.Fatalf("NewIndependentMaskerWithSeed() error = %v", err)
	}

	instance := []float64{1.0, 2.0, 3.0}
	mask := []bool{true, false, true}

	result := m.Mask(instance, mask)

	// Feature 1 should be unmasked (original value)
	if result[1] != 2.0 {
		t.Errorf("Mask()[1] = %f, want 2.0", result[1])
	}

	// Features 0 and 2 should be from background (10.0 and 30.0)
	if result[0] != 10.0 {
		t.Errorf("Mask()[0] = %f, want 10.0", result[0])
	}
	if result[2] != 30.0 {
		t.Errorf("Mask()[2] = %f, want 30.0", result[2])
	}
}

func TestIndependentMasker_MeanBackground(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	m, err := NewIndependentMasker(background)
	if err != nil {
		t.Fatalf("NewIndependentMasker() error = %v", err)
	}

	mean := m.MeanBackground()

	expected := []float64{3.0, 4.0} // (1+3+5)/3, (2+4+6)/3
	for i, v := range mean {
		if v != expected[i] {
			t.Errorf("MeanBackground()[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestIndependentMasker_SampleBackground(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	m, err := NewIndependentMaskerWithSeed(background, 42)
	if err != nil {
		t.Fatalf("NewIndependentMaskerWithSeed() error = %v", err)
	}

	sample := m.SampleBackground()

	if len(sample) != 2 {
		t.Errorf("SampleBackground() returned %d features, want 2", len(sample))
	}

	// Check that sample is one of the background samples
	isValid := (sample[0] == 1.0 && sample[1] == 2.0) ||
		(sample[0] == 3.0 && sample[1] == 4.0)

	if !isValid {
		t.Errorf("SampleBackground() = %v, expected one of the background samples", sample)
	}
}

func TestIndependentMasker_Background(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	m, err := NewIndependentMasker(background)
	if err != nil {
		t.Fatalf("NewIndependentMasker() error = %v", err)
	}

	bg := m.Background()

	if len(bg) != 2 {
		t.Errorf("Background() returned %d rows, want 2", len(bg))
	}
}

func TestIndependentMasker_ImplementsMasker(t *testing.T) {
	var _ Masker = (*IndependentMasker)(nil)
}
