#!/bin/bash
# Download pre-trained ONNX model from Hugging Face
#
# This downloads an Iris classification model that can be used
# as an alternative to generating one locally with generate_model.py
#
# Note: The Hugging Face model may have different input/output names
# than the locally generated model. Check the model structure with:
#   python -c "import onnx; m=onnx.load('iris.onnx'); print(m.graph)"

set -e

MODEL_URL="https://huggingface.co/Ritual-Net/iris-classification/resolve/main/iris.onnx"
OUTPUT_FILE="iris_hf.onnx"

echo "Downloading Iris ONNX model from Hugging Face..."
echo "Source: Ritual-Net/iris-classification"
echo ""

if command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$MODEL_URL"
elif command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$MODEL_URL"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

echo ""
echo "Downloaded: $OUTPUT_FILE"
echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "To use this model, you may need to update main.go with the correct"
echo "input/output names from this model. Inspect with:"
echo "  python -c \"import onnx; m=onnx.load('$OUTPUT_FILE'); print([i.name for i in m.graph.input])\""
