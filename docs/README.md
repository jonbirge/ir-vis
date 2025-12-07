# Publication Figure Generator

This tool generates high-quality diagrams using standard research tools.

## Prerequisites

1.  **Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **System Tools:**
    *   **LaTeX:** Required for Figures 1, 2, and 3.
        *   Mac: `brew install mactex`
        *   Linux: `sudo apt install texlive-full`
    *   **Graphviz:** Required for Figure 4.
        *   Mac: `brew install graphviz`
        *   Linux: `sudo apt install graphviz`

## Instructions

1.  **Setup PlotNeuralNet:**
    Run this to download the 3D diagramming library (no git required).
    ```bash
    python setup_pnn.py
    ```

2.  **Generate Figures:**
    Run the master script. This will generate source code and attempt to compile the PDFs and SVG.
    ```bash
    python generate_diagrams.py
    ```

## Outputs

| File | Tool | Description |
| :--- | :--- | :--- |
| `fig1_system.pdf` | **TikZ** | High-level schematic of the Dual-Encoder system. |
| `fig2_architecture.pdf` | **PlotNeuralNet** | 3D Isometric block diagram of the CNN with skip connections. |
| `fig3_attention.pdf` | **TikZ** | Detailed view of the Transformer/Attention module. |
| `fig4_automated_trace.svg` | **Graphviz** | Engineering-grade graph trace of the PyTorch code. |

*Note: If PDFs fail to generate locally due to missing TeX packages, upload the generated `.tex` files (and the `layers/` folder) to Overleaf.*
