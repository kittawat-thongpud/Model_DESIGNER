# Mermaid Diagrams for YOLO vs RF-DETR Presentation

This directory contains Mermaid diagram source files and scripts to generate PNG images for the presentation.

## 📁 Files

### Mermaid Source Files (.mmd)

1. **yolo-pipeline.mmd** - YOLO architecture pipeline (6 stages)
2. **yolo-grid-detection.mmd** - YOLO grid-based detection concept
3. **yolo-duplication.mmd** - YOLO duplication problem visualization
4. **rfdetr-pipeline.mmd** - RF-DETR architecture pipeline (8 stages)
5. **rfdetr-query-system.mmd** - RF-DETR object query system
6. **hungarian-matching.mmd** - Hungarian matching algorithm visualization
7. **tradeoff-decision.mmd** - Trade-off decision tree
8. **paradigm-comparison.mmd** - Two paradigms comparison

### Scripts

- **generate-images.sh** - Bash script to generate all PNG images

## 🚀 Quick Start

### Method 1: Using the Script (Recommended)

```bash
# Install Mermaid CLI (one-time setup)
npm install -g @mermaid-js/mermaid-cli

# Make script executable
chmod +x generate-images.sh

# Generate all images
./generate-images.sh
```

Images will be saved to `images/` directory.

### Method 2: Manual Generation

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate individual diagrams
mmdc -i yolo-pipeline.mmd -o images/yolo-pipeline.png -b transparent -w 1200 -H 400
mmdc -i rfdetr-pipeline.mmd -o images/rfdetr-pipeline.png -b transparent -w 1200 -H 400
# ... etc
```

### Method 3: Online Editor

1. Go to https://mermaid.live/
2. Copy content from any `.mmd` file
3. Paste into editor
4. Click "Download PNG" or "Download SVG"

### Method 4: VS Code Extension

1. Install "Markdown Preview Mermaid Support" extension
2. Create a markdown file with mermaid code blocks:
   ````markdown
   ```mermaid
   graph LR
       A --> B
   ```
   ````
3. Right-click diagram in preview → Export to PNG

## 🎨 Diagram Specifications

### Color Scheme

**YOLO (Blue):**
- Primary: `#1976d2`
- Light: `#e3f2fd`, `#bbdefb`, `#90caf9`

**RF-DETR (Green):**
- Primary: `#388e3c`
- Light: `#e8f5e9`, `#c8e6c9`, `#a5d6a7`, `#81c784`, `#66bb6a`, `#4caf50`

**Negative/Warning (Red):**
- `#d32f2f`, `#ffcdd2`

### Image Dimensions

- **Pipeline diagrams**: 1200×400px
- **Concept diagrams**: 1000×600px
- **Comparison diagrams**: 1000×500px
- **Background**: Transparent

## 📝 Usage in Marp Presentation

Once images are generated, update the Marp presentation:

```markdown
# YOLO Pipeline

![w:1000](diagrams/images/yolo-pipeline.png)

---

# RF-DETR Pipeline

![w:1000](diagrams/images/rfdetr-pipeline.png)
```

## 🔧 Customization

To modify diagrams:

1. Edit the `.mmd` file
2. Re-run `./generate-images.sh`
3. Images will be updated automatically

### Example: Change Colors

```mermaid
style A fill:#your-color,stroke:#your-border,stroke-width:2px
```

### Example: Adjust Layout

```mermaid
graph LR  # Left to Right
graph TB  # Top to Bottom
graph RL  # Right to Left
graph BT  # Bottom to Top
```

## 📚 Resources

- [Mermaid Documentation](https://mermaid.js.org/)
- [Mermaid Live Editor](https://mermaid.live/)
- [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli)
- [Mermaid Syntax](https://mermaid.js.org/intro/syntax-reference.html)

## 🐛 Troubleshooting

### Error: `mmdc: command not found`

```bash
npm install -g @mermaid-js/mermaid-cli
```

### Error: Permission denied

```bash
chmod +x generate-images.sh
```

### Error: Puppeteer issues

```bash
# Install Chromium dependencies (Linux)
sudo apt-get install -y libgbm1 libasound2

# Or use Docker
docker run --rm -v $(pwd):/data minlag/mermaid-cli -i input.mmd -o output.png
```

### Diagrams not rendering correctly

- Check Mermaid syntax
- Validate at https://mermaid.live/
- Ensure all nodes are properly connected
- Check for typos in style definitions

## ✅ Output

After running `./generate-images.sh`, you should have:

```
images/
├── yolo-pipeline.png
├── yolo-grid-detection.png
├── yolo-duplication.png
├── rfdetr-pipeline.png
├── rfdetr-query-system.png
├── hungarian-matching.png
├── tradeoff-decision.png
└── paradigm-comparison.png
```

All images are:
- ✅ Transparent background
- ✅ High resolution
- ✅ Consistent color scheme
- ✅ Ready for presentation use
