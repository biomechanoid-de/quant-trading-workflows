#!/usr/bin/env python3
"""Export and quantize sentiment model to ONNX INT8 for Pi 5 deployment.

One-time script. Run on a development machine (not the Pi cluster).
Outputs to models/distilbert-finance-sentiment/ which gets baked into Docker.

Usage:
    python scripts/export_sentiment_model.py

Requires (dev only, NOT runtime deps):
    pip install optimum[onnxruntime] transformers torch
"""

import os
import sys

MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "distilbert-finance-sentiment",
)


def main():
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    print(f"Exporting {MODEL_NAME} to ONNX...")

    # Step 1: Export to ONNX
    onnx_dir = OUTPUT_DIR + "-fp32"
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)
    print(f"FP32 ONNX model saved to {onnx_dir}")

    # Step 2: Quantize to INT8 (ARM64 optimized)
    print("Quantizing to INT8 for ARM64...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=OUTPUT_DIR, quantization_config=qconfig)

    # Step 3: Copy tokenizer files to quantized output
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"INT8 quantized model saved to {OUTPUT_DIR}")

    # Step 4: Report sizes
    fp32_size = sum(
        os.path.getsize(os.path.join(onnx_dir, f))
        for f in os.listdir(onnx_dir)
        if f.endswith(".onnx")
    )
    int8_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".onnx")
    )
    print(f"FP32 size: {fp32_size / 1024 / 1024:.1f} MB")
    print(f"INT8 size: {int8_size / 1024 / 1024:.1f} MB")
    print(f"Compression: {fp32_size / int8_size:.1f}x")

    # Clean up FP32
    import shutil
    shutil.rmtree(onnx_dir)
    print(f"Cleaned up {onnx_dir}")

    print("\nDone! Model is ready for Docker build.")
    print(f"Path inside container: /root/models/distilbert-finance-sentiment/")


if __name__ == "__main__":
    main()
