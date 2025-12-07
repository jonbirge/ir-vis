import os
import sys
import subprocess
import torch

# ==========================================
# HELPER: COMPILE LATEX
# ==========================================
def compile_latex(filename):
    if not os.path.exists(filename): return
    print(f"⚙️  Compiling {filename}...")
    # Run twice for layout calculation
    cmd = ["pdflatex", "-interaction=nonstopmode", filename]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Generated {filename.replace('.tex', '.pdf')}")

# ==========================================
# FIG 1: SYSTEM OVERVIEW (NOW 3D / PNN)
# ==========================================
def gen_fig1_pnn():
    print("\n--- Generating Figure 1 (System Overview - 3D PNN) ---")
    if not os.path.exists("pycore"):
        print("❌ PlotNeuralNet not found. Run 'python setup_pnn.py' first.")
        return

    script = r"""
import sys
sys.path.append('.')
from pycore.tikzeng import *
from pycore.blocks import *

def main():
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # --- 1. IR Stream (Content) ---
        to_input('ir_img', label='IR Input'),
        
        # Content Encoder (Represented as a simplified monolithic stack)
        *block_2d('enc_c1', bottom='ir_img', top='ir_img', width=2, height=20, depth=20, caption='Content Enc', color='white'),
        *block_2d('enc_c2', bottom='enc_c1', top='enc_c1', xshift=2, width=3, height=15, depth=15),
        *block_2d('enc_c3', bottom='enc_c2', top='enc_c2', xshift=2, width=4, height=10, depth=10),

        # --- 2. Reference Stream (Color) ---
        # Shifted down in Y to separate the streams
        to_input('ref_img', label='Reference', xshift=0, yshift=-16),
        
        # Ref Encoder (Orange)
        *block_2d('enc_r1', bottom='ref_img', top='ref_img', width=2, height=20, depth=20, color='orange', caption='Ref Enc'),
        *block_2d('enc_r2', bottom='enc_r1', top='enc_r1', xshift=2, width=3, height=15, depth=15, color='orange'),
        *block_2d('enc_r3', bottom='enc_r2', top='enc_r2', xshift=2, width=4, height=10, depth=10, color='orange'),

        # --- 3. Feature Matcher (The Bridge) ---
        # Placed centrally
        *block_2d('matcher', bottom='enc_c3', top='enc_r3', xshift=6, yshift=-2, width=5, height=30, depth=8, color='purple', opacity=0.7, caption='Cross-Attention\\nModule'),
        
        # Connections
        to_connection('enc_c3', 'matcher'),
        to_connection('enc_r3', 'matcher'),

        # --- 4. Decoder ---
        # Growing blocks
        *block_2d('dec1', bottom='matcher', top='matcher', xshift=6, width=4, height=10, depth=10, color='green', caption='Decoder'),
        *block_2d('dec2', bottom='dec1', top='dec1', xshift=2, width=3, height=15, depth=15, color='green'),
        *block_2d('dec3', bottom='dec2', top='dec2', xshift=2, width=2, height=20, depth=20, color='green'),
        
        # Output
        to_input('out_img', label='Color Output', xshift=3),
        to_connection('dec3', 'out_img'),

        to_end()
    ]
    
    # Custom block helper
    def block_2d(name, bottom, top, xshift=0, yshift=0, width=5, height=10, depth=10, caption="", opacity=1.0, color="white"):
        return [to_Conv(name, height, depth, offset=f"({xshift},{yshift},0)", to=f"({bottom}-east)", width=width, caption=caption, opacity=opacity, color=color)]

    to_generate(arch, "fig1_system.tex")

if __name__ == "__main__":
    main()
"""
    with open("run_fig1.py", "w") as f: f.write(script)
    subprocess.run([sys.executable, "run_fig1.py"])
    compile_latex("fig1_system.tex")

# ==========================================
# FIG 2: DETAILED ARCHITECTURE (PNN)
# ==========================================
def gen_fig2_pnn():
    print("\n--- Generating Figure 2 (Detailed Architecture - 3D PNN) ---")
    if not os.path.exists("pycore"): return

    script = r"""
import sys
sys.path.append('.')
from pycore.tikzeng import *
from pycore.blocks import *

def main():
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # --- IR STREAM ---
        to_input('ir_img', label='IR Input'),
        *block_2d('conv1', bottom='ir_img', top='ir_img', width=2, height=30, depth=30, caption='Conv1'),
        *block_2d('pool1', bottom='conv1', top='conv1', xshift=2, width=1, height=24, depth=24, opacity=0.5),
        *block_Res('layer1', bottom='pool1', top='pool1', xshift=3, height=20, depth=20, width=4, caption='L1'),
        *block_Res('layer2', bottom='layer1', top='layer1', xshift=3, height=16, depth=16, width=6, caption='L2'),
        *block_Res('layer3', bottom='layer2', top='layer2', xshift=3, height=12, depth=12, width=8, caption='L3'),
        *block_Res('layer4', bottom='layer3', top='layer3', xshift=3, height=8, depth=8, width=12, caption='L4'),

        # --- REF STREAM ---
        to_input('ref_img', label='Reference', xshift=0, yshift=-18),
        *block_2d('ref_conv1', bottom='ref_img', top='ref_img', width=2, height=30, depth=30, color="orange"),
        *block_2d('ref_pool1', bottom='ref_conv1', top='ref_conv1', xshift=2, width=1, height=24, depth=24, opacity=0.5, color="orange"),
        *block_Res('ref_layer1', bottom='ref_pool1', top='ref_pool1', xshift=3, height=20, depth=20, width=4, color="orange"),
        *block_Res('ref_layer2', bottom='ref_layer1', top='ref_layer1', xshift=3, height=16, depth=16, width=6, color="orange"),
        *block_Res('ref_layer3', bottom='ref_layer2', top='ref_layer2', xshift=3, height=12, depth=12, width=8, color="orange"),
        *block_Res('ref_layer4', bottom='ref_layer3', top='ref_layer3', xshift=3, height=8, depth=8, width=12, caption='Ref L4', color="orange"),

        # --- MATCHING ---
        *block_2d('attn', bottom='layer4', top='ref_layer4', xshift=6, yshift=9, width=4, height=30, depth=10, color="purple", caption="Match", opacity=0.6),
        to_connection("layer4", "attn"),
        to_connection("ref_layer4", "attn"),

        # --- DECODER ---
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
    with open("run_fig2.py", "w") as f: f.write(script)
    subprocess.run([sys.executable, "run_fig2.py"])
    compile_latex("fig2_architecture.tex")

# ==========================================
# FIG 3: ATTENTION MODULE (MATCHED 2D)
# ==========================================
def gen_fig3_tikz():
    print("\n--- Generating Figure 3 (Attention Module - Matched 2D TikZ) ---")
    # Note: We now use the exact RGB colors from PlotNeuralNet (Orange, Purple) to match styles
    code = r"""
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc, shapes, shadows}

% Define Palette to Match PlotNeuralNet
\definecolor{pnn_orange}{RGB}{255,165,0}
\definecolor{pnn_purple}{RGB}{128,0,128}
\definecolor{pnn_green}{RGB}{0,128,0}
\definecolor{pnn_white}{RGB}{255,255,255}

\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Latex,
    % Styles
    block/.style={rectangle, draw=black!40, very thick, rounded corners=3pt, fill=white, drop shadow},
    op/.style={circle, draw=black!60, very thick, fill=white, inner sep=1pt, minimum size=0.8cm, drop shadow},
    input/.style={rectangle, draw=black!60, very thick, rounded corners=2pt, minimum width=2.5cm, minimum height=1cm, align=center},
    % Input Colors
    ir_in/.style={input, fill=pnn_white!90!gray},
    ref_in/.style={input, fill=pnn_orange!20},
    % Connection lines
    line/.style={->, very thick, color=black!70}
]

% --- Inputs ---
\node[ir_in] (q) at (0,0) {\textbf{Content (Q)}\\\scriptsize (from IR Enc)};
\node[ref_in] (k) at (3.5,0) {\textbf{Reference (K)}\\\scriptsize (from Ref Enc)};
\node[ref_in] (v) at (7,0) {\textbf{Reference (V)}\\\scriptsize (from Ref Enc)};

% --- Projections ---
\node[block, below=0.8cm of q, minimum width=2cm] (lin_q) {Linear Proj};
\node[block, below=0.8cm of k, minimum width=2cm] (lin_k) {Linear Proj};
\node[block, below=0.8cm of v, minimum width=2cm] (lin_v) {Linear Proj};

\draw[line] (q) -- (lin_q);
\draw[line] (k) -- (lin_k);
\draw[line] (v) -- (lin_v);

% --- MatMul 1 ---
\node[op, below=1.5cm of k, xshift=-1.75cm] (matmul1) {$\times$};
\node[left=0.1cm of matmul1, color=black!60] {\scriptsize MatMul};

% Curved paths to MatMul
\draw[line] (lin_q.south) to[out=270,in=135] (matmul1);
\draw[line] (lin_k.south) to[out=270,in=45] (matmul1);

% --- Softmax ---
\node[block, fill=pnn_purple!10, draw=pnn_purple, below=0.8cm of matmul1] (softmax) {Softmax};
\draw[line] (matmul1) -- (softmax);

% --- MatMul 2 ---
\node[op, below=0.8cm of softmax] (matmul2) {$\times$};
\draw[line] (softmax) -- (matmul2);
% Long path for V
\draw[line] (lin_v.south) to[out=270,in=45] (matmul2);

% --- Add & Norm ---
\node[op, below=1cm of matmul2] (add1) {$+$};
\draw[line] (matmul2) -- (add1);

% Residual skip (Q bypass)
\draw[line, dashed] (lin_q.west) -- ++(-0.6,0) |- (add1.west);

\node[block, below=0.6cm of add1, minimum width=3cm] (norm1) {Layer Norm};
\draw[line] (add1) -- (norm1);

% --- FFN ---
\node[block, below=0.8cm of norm1, minimum width=3cm, fill=pnn_purple!10, draw=pnn_purple] (ffn) {Feed Forward};
\draw[line] (norm1) -- (ffn);

\node[op, below=0.8cm of ffn] (add2) {$+$};
\draw[line] (ffn) -- (add2);

% Residual skip (Norm bypass)
\draw[line, dashed] (norm1.east) -- ++(0.6,0) |- (add2.east);

% --- Output ---
\node[block, below=0.8cm of add2, fill=pnn_green!20, draw=pnn_green, minimum width=4cm] (out) {\textbf{Aligned Features}};
\draw[line] (add2) -- (out);

\end{tikzpicture}
\end{document}
"""
    with open("fig3_attention.tex", "w") as f: f.write(code)
    compile_latex("fig3_attention.tex")

# ==========================================
# FIG 4: AUTOMATED GRAPH (SVG)
# ==========================================
def gen_fig4_svg():
    print("\n--- Generating Figure 4 (Automated SVG Trace) ---")
    try:
        from torchview import draw_graph
        from model import create_model
        from config import get_config
        
        config = get_config()
        model = create_model(config.model)
        ir = torch.randn(1, 3, 256, 256)
        ref = torch.randn(1, 3, 256, 256)
        
        graph = draw_graph(model, input_data=(ir, ref), expand_nested=True, depth=2, save_graph=False)
        graph.visual_graph.format = 'svg'
        graph.visual_graph.render(filename='fig4_automated_trace', cleanup=True)
        print("✅ Saved 'fig4_automated_trace.svg'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    gen_fig1_pnn()
    gen_fig2_pnn()
    gen_fig3_tikz()
    gen_fig4_svg()
    