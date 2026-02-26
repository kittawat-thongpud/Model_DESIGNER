#!/bin/bash

# Script to generate PNG images from Mermaid diagrams
# Requires: npm install -g @mermaid-js/mermaid-cli

echo "🎨 Generating images from Mermaid diagrams..."

# Check if mmdc is installed
if ! command -v mmdc &> /dev/null
then
    echo "❌ Mermaid CLI not found!"
    echo "📦 Install with: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# Create output directory
mkdir -p images

# Generate images with transparent background
# Using --no-sandbox flag for Linux compatibility
MMDC_FLAGS="-b transparent --puppeteerConfigFile puppeteer-config.json"

echo "📊 Generating YOLO diagrams..."
mmdc -i yolo-pipeline.mmd -o images/yolo-pipeline.png $MMDC_FLAGS -w 1200 -H 400
mmdc -i yolo-grid-detection.mmd -o images/yolo-grid-detection.png $MMDC_FLAGS -w 1000 -H 600
mmdc -i yolo-duplication.mmd -o images/yolo-duplication.png $MMDC_FLAGS -w 800 -H 500

echo "📊 Generating RF-DETR diagrams..."
mmdc -i rfdetr-pipeline.mmd -o images/rfdetr-pipeline.png $MMDC_FLAGS -w 1200 -H 400
mmdc -i rfdetr-query-system.mmd -o images/rfdetr-query-system.png $MMDC_FLAGS -w 1000 -H 600
mmdc -i hungarian-matching.mmd -o images/hungarian-matching.png $MMDC_FLAGS -w 1000 -H 500

echo "📊 Generating comparison diagrams..."
mmdc -i tradeoff-decision.mmd -o images/tradeoff-decision.png $MMDC_FLAGS -w 1000 -H 600
mmdc -i paradigm-comparison.mmd -o images/paradigm-comparison.png $MMDC_FLAGS -w 1200 -H 500

echo "✅ All diagrams generated successfully!"
echo "📁 Images saved to: images/"
ls -lh images/
