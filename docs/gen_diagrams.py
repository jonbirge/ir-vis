import os
import sys
import subprocess
import torch

# ==========================================
# UTILS
# ==========================================
def compile_latex(filename):
    """Compiles a .tex file to PDF using system pdflatex."""
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found.")
        return

    print(f"⚙️  Compiling {filename}...")
    try:
        # Run twice to resolve references/layout
        cmd = ["pdflatex", "-interaction=nonstopmode", filename]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Generated {filename.replace('.tex', '.pdf')}")
    except FileNotFoundError:
        print(f"⚠️  'pdflatex' not found. Generated {filename} but could not compile.")
        print("   Upload the .tex file to Overleaf to view it.")

# ==========================================
# FIGURE 1: SYSTEM OVERVIEW (TikZ)
# ==========================================
def gen_fig1():
    print("\n--- Generating Figure 1 (System Overview) ---")
    code = r"""
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, fit, arrows.meta, shadows, shapes}

% Professional Color Palette
\definecolor{irblue}{RGB}{227, 242, 253}
\definecolor{irborder}{RGB}{21, 101, 192}
\definecolor{reforg}{RGB}{255, 243, 224}
\definecolor{refborder}{RGB}{239, 108, 0}
\definecolor{matchpurp}{RGB}{243, 229, 245}
\definecolor{matchborder}{RGB}{123, 31, 162}

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm,
    font=\sffamily,
    >=Stealth,
    block/.style={rectangle, rounded corners, draw=black!60, very thick, minimum width=2.8cm, minimum height=1.2cm, align=center, drop shadow},
    irblock/.style={block, fill=irblue, draw=irborder},
    refblock/.style={block, fill=reforg, draw=refborder},
    matchblock/.style={block, fill=matchpurp, draw=matchborder, minimum height=3.5cm, minimum width=3cm},
    img/.style={rectangle, draw=black!30, thick, minimum size=1.2cm, fill=white},
    line/.style={->, very thick, color=black!70}
]

% Nodes
\node[img, label=below:IR Input] (ir_img) {};
\node[img, label=below:Ref Input, below=2.5cm of ir_img] (ref_img) {};

\node[irblock, right=1.5cm of ir_img] (enc_cont) {\textbf{Content Encoder}\\(ResNet)};
\node[refblock, right=1.5cm of ref_img] (enc_ref) {\textbf{Reference Encoder}\\(ResNet)};

\node[matchblock, right=2cm of enc_cont, yshift=-1.75cm] (matcher) {\textbf{Feature Matching}\\(Cross-Attention)};

\node[block, fill=green!5, draw=green!60!black, right=2cm of matcher] (decoder) {\textbf{Decoder}\\(U-Net)};

\node[img, right=1.5cm of decoder, label=below:Color Output] (out_img) {};

% Connections
\draw[line] (ir_img) -- (enc_cont);
\draw[line] (ref_img) -- (enc_ref);

% Curved connections to Matcher
\draw[line] (enc_cont.east) to[out=0,in=160] node[above, font=\scriptsize, pos=0.3] {Content Feats} (matcher.160);
\draw[line] (enc_ref.east) to[out=0,in=200] node[below, font=\scriptsize, pos=0.3] {Ref Feats} (matcher.200);

\draw[line] (matcher) -- (decoder);
\draw[line] (decoder) -- (out_img);

% Skip connection
\draw[->, dashed, thick, color=gray] (enc_cont.north) to[out=45,in=135] node[midway, above, font=\small\itshape] {Spatial Skip Connections} (decoder.north);

\end{tikzpicture}
\end{document}
"""
    with open("fig1_system.tex", "w") as f: f.write(code)
    compile_latex("fig1_system.tex")

# ==========================================
# FIGURE 2: 3D ARCHITECTURE (PlotNeuralNet)
# ==========================================
def gen_fig2():
    print("\n--- Generating Figure 2 (3D Architecture) ---")
    if not os.path.exists("pycore"):
        print("❌ PlotNeuralNet not found. Please run 'python setup_pnn.py' first.")
        return

    # We generate a Python script that PNN will use to build the TeX
    pnn_script = r"""
import sys
sys.path.append('.')
from pycore.tikzeng import *
from pycore.blocks import *

def main():
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # --- 1. IR STREAM (Content) ---
        to_input('ir_img', label='IR Input'),
        *block_2d('conv1', bottom='ir_img', top='ir_img', width=2, height=30, depth=30, caption='Conv1'),
        *block_2d('pool1', bottom='conv1', top='conv1', xshift=2, width=1, height=24, depth=24, opacity=0.5),
        *block_Res('layer1', bottom='pool1', top='pool1', xshift=3, height=20, depth=20, width=4, caption='L1'),
        *block_Res('layer2', bottom='layer1', top='layer1', xshift=3, height=16, depth=16, width=6, caption='L2'),
        *block_Res('layer3', bottom='layer2', top='layer2', xshift=3, height=12, depth=12, width=8, caption='L3'),
        *block_Res('layer4', bottom='layer3', top='layer3', xshift=3, height=8, depth=8, width=12, caption='L4'),

        # --- 2. REF STREAM (Color) ---
        # Visual trick: Shift Z (depth) and Y to place it 'below'
        to_input('ref_img', label='Reference', xshift=0, yshift=-18),
        *block_2d('ref_conv1', bottom='ref_img', top='ref_img', width=2, height=30, depth=30, color="orange"),
        *block_2d('ref_pool1', bottom='ref_conv1', top='ref_conv1', xshift=2, width=1, height=24, depth=24, opacity=0.5, color="orange"),
        *block_Res('ref_layer1', bottom='ref_pool1', top='ref_pool1', xshift=3, height=20, depth=20, width=4, color="orange"),
        *block_Res('ref_layer2', bottom='ref_layer1', top='ref_layer1', xshift=3, height=16, depth=16, width=6, color="orange"),
        *block_Res('ref_layer3', bottom='ref_layer2', top='ref_layer2', xshift=3, height=12, depth=12, width=8, color="orange"),
        *block_Res('ref_layer4', bottom='ref_layer3', top='ref_layer3', xshift=3, height=8, depth=8, width=12, caption='Ref L4', color="orange"),

        # --- 3. ATTENTION MODULE ---
        *block_2d('attn', bottom='layer4', top='ref_layer4', xshift=6, yshift=9, width=4, height=30, depth=10, color="purple", caption="Cross-Attn", opacity=0.6),
        to_connection("layer4", "attn"),
        to_connection("ref_layer4", "attn"),

        # --- 4. DECODER (U-Net) ---
        *block_Unconv('up1', bottom='attn', top='attn', xshift=6, height=12, depth=12, width=8, caption="Up1"),
        to_skip(of='layer3', to='up1', pos=1.5), 
        
        *block_Unconv('up2', bottom='up1', top='up1', xshift=3, height=16, depth=16, width=6, caption="Up2"),
        to_skip(of='layer2', to='up2', pos=1.5),

        *block_Unconv('up3', bottom='up2', top='up2', xshift=3, height=20, depth=20, width=4, caption="Up3"),
        to_skip(of='layer1', to='up3', pos=1.5),
        
        *block_Unconv('up4', bottom='up3', top='up3', xshift=3, height=30, depth=30, width=2, caption="Up4"),
        *block_2d('out', bottom='up4', top='up4', xshift=3, width=1, height=32, depth=32, color="green", caption="RGB"),
        to_end()
    ]
    
    # Helper to define simple blocks
    def block_Res(name, bottom, top, xshift=0, height=10, depth=10, width=5, caption="", color="white"):
        return [to_ConvRes(name, height, depth, offset="(0,0,0)", to=f"({bottom}-east)", width=width, caption=caption, n_label="", color=color)]
        
    def block_Unconv(name, bottom, top, xshift=0, height=10, depth=10, width=5, caption=""):
        return [
            to_UnPool(name, offset="(1,0,0)", to=f"({bottom}-east)", width=1, height=height, depth=depth, opacity=0.5),
            to_Conv(name, height, depth, offset="(0,0,0)", to=f"({name}-east)", width=width, caption=caption, n_label="")
        ]
        
    def block_2d(name, bottom, top, xshift=0, yshift=0, width=5, height=10, depth=10, caption="", opacity=1.0, color="white"):
        return [to_Conv(name, height, depth, offset=f"({xshift},{yshift},0)", to=f"({bottom}-east)", width=width, caption=caption, opacity=opacity, color=color)]

    to_generate(arch, "fig2_architecture.tex")

if __name__ == "__main__":
    main()
"""
    with open("run_pnn.py", "w") as f: f.write(pnn_script)
    
    # Run the generator script
    subprocess.run([sys.executable, "run_pnn.py"])
    compile_latex("fig2_architecture.tex")

# ==========================================
# FIGURE 3: ATTENTION MODULE (TikZ)
# ==========================================
def gen_fig3():
    print("\n--- Generating Figure 3 (Attention Module) ---")
    code = r"""
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc, shapes}

\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Latex,
    op/.style={rectangle, draw, fill=white, thick, minimum width=2cm, minimum height=0.8cm, rounded corners=2pt},
    mul/.style={circle, draw, thick, fill=white, inner sep=0pt, minimum size=0.6cm},
    add/.style={circle, draw, thick, fill=white, inner sep=0pt, minimum size=0.6cm, path picture={\draw[thick] (path picture bounding box.south) -- (path picture bounding box.north) (path picture bounding box.west) -- (path picture bounding box.east);}}
]

% Inputs
\node (q) at (0,0) {\textbf{Content (Q)}};
\node (k) at (2.5,0) {\textbf{Ref (K)}};
\node (v) at (4.5,0) {\textbf{Ref (V)}};

% Projections
\node[op, below=0.5cm of q] (lin_q) {Linear};
\node[op, below=0.5cm of k] (lin_k) {Linear};
\node[op, below=0.5cm of v] (lin_v) {Linear};

\draw[->] (q) -- (lin_q);
\draw[->] (k) -- (lin_k);
\draw[->] (v) -- (lin_v);

% Attention Mechanism
\node[mul, below=1cm of lin_k, xshift=-1.25cm] (matmul1) {};
\node[right=0.1cm of matmul1] {\scriptsize MatMul};

\draw[->] (lin_q.south) to[out=270,in=135] (matmul1);
\draw[->] (lin_k.south) to[out=270,in=45] (matmul1);

\node[op, below=0.5cm of matmul1] (softmax) {Softmax};
\draw[->] (matmul1) -- (softmax);

\node[mul, below=0.5cm of softmax] (matmul2) {};
\node[right=0.1cm of matmul2] {\scriptsize MatMul};

\draw[->] (softmax) -- (matmul2);
\draw[->] (lin_v.south) to[out=270,in=45] (matmul2);

% Residual & Norm
\node[add, below=0.8cm of matmul2] (add1) {};
\draw[->] (matmul2) -- (add1);

% Skip connection from Q input
\draw[->, dashed] (lin_q.west) -- ++(-0.5,0) |- (add1.west);

\node[op, below=0.6cm of add1] (norm1) {Layer Norm};
\draw[->] (add1) -- (norm1);

\node[op, below=0.6cm of norm1] (ffn) {Feed Forward};
\draw[->] (norm1) -- (ffn);

\node[add, below=0.6cm of ffn] (add2) {};
\draw[->] (ffn) -- (add2);

% Skip connection for FFN
\draw[->, dashed] (norm1.east) -- ++(0.5,0) |- (add2.east);

\node[below=0.5cm of add2, font=\bfseries] (out) {Aligned Features};
\draw[->] (add2) -- (out);

\end{tikzpicture}
\end{document}
"""
    with open("fig3_attention.tex", "w") as f: f.write(code)
    compile_latex("fig3_attention.tex")

# ==========================================
# FIGURE 4: AUTOMATED GRAPH (SVG)
# ==========================================
def gen_fig4():
    print("\n--- Generating Figure 4 (Automated SVG Trace) ---")
    try:
        from torchview import draw_graph
        from model import create_model
        from config import get_config
        
        config = get_config()
        model = create_model(config.model)
        
        ir = torch.randn(1, 3, 256, 256)
        ref = torch.randn(1, 3, 256, 256)
        
        # Explicitly set SVG format
        graph = draw_graph(
            model, input_data=(ir, ref),
            expand_nested=True, depth=2,
            save_graph=False
        )
        graph.visual_graph.format = 'svg'
        graph.visual_graph.render(filename='fig4_automated_trace', cleanup=True)
        print("✅ Saved 'fig4_automated_trace.svg'")
        
    except ImportError:
        print("❌ Error: torchview or graphviz (python) not found.")
    except Exception as e:
        print(f"❌ Error generating graph: {e}")

if __name__ == "__main__":
    gen_fig1()
    gen_fig2()
    gen_fig3()
    gen_fig4()
    