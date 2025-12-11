import sys
import os
import subprocess
import torch # type: ignore
import re
import graphviz  # type: ignore # small helper for rendering modified DOT
import glob  # helper for cleaning LaTeX artifacts

# Add current directory to path to allow importing pycore
sys.path.append(os.path.abspath("."))

# Check for PlotNeuralNet libraries
try:
    from pycore.tikzeng import *
    from pycore.blocks import *
except ImportError:
    print("❌ Error: 'pycore' library not found.")
    print("   Please run the one-line setup command provided in the instructions.")
    sys.exit(1)

# ==========================================
# 1. PNN BLOCK HELPERS (Corrected)
# ==========================================
# We wrap the standard blocks to inject color commands and handle 
# dimensions correctly (mapping python args to visual size, not labels).

def block_Res(name, bottom, top, xshift=0, height=10, depth=10, width=5, caption="", color="white"):
    """Helper for a Residual Block visual."""
    return [
        f"\\renewcommand{{\\ConvColor}}{{{color}}}",
        to_ConvRes(
            name, 
            s_filer="", n_filer="",  # Suppress numeric labels
            height=height, depth=depth, width=width, # Visual dimensions
            offset="(0,0,0)", 
            to=f"({bottom}-east)", 
            caption=caption
        )
    ]

def block_Unconv(name, bottom, top, xshift=0, height=10, depth=10, width=5, caption="", color="white"):
    """Helper for an Upsampling/Deconv block."""
    return [
        f"\\renewcommand{{\\ConvColor}}{{{color}}}",
        to_UnPool(
            name, 
            offset="(1,0,0)", 
            to=f"({bottom}-east)", 
            width=1, 
            height=height, 
            depth=depth, 
            opacity=0.5,
            caption=caption
        ),
        to_Conv(
            name, 
            s_filer="", n_filer="",
            height=height, depth=depth, width=width,
            offset="(0,0,0)", 
            to=f"({name}-east)", 
            caption=""
        )
    ]

def block_Simple(name, bottom, top, xshift=0, yshift=0, width=5, height=10, depth=10, caption="", opacity=1.0, color="white"):
    """Helper for a standard 3D block."""
    return [
        f"\\renewcommand{{\\ConvColor}}{{{color}}}",
        # Note: do NOT forward an 'opacity' kwarg to to_Conv (unsupported by pycore).
        to_Conv(
            name, 
            s_filer="", n_filer="", 
            height=height, depth=depth, width=width,
            offset=f"({xshift},{yshift},0)", 
            to=f"({bottom}-east)", 
            caption=caption
        )
    ]

# ==========================================
# 2. COMPILATION UTILS
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
        
        pdf_name = filename.replace('.tex', '.pdf')
        if os.path.exists(pdf_name):
            print(f"✅ Generated {pdf_name}")
        else:
            print(f"⚠️  LaTeX failed to generate PDF. Check {filename.replace('.tex', '.log')}")
    except FileNotFoundError:
        print(f"⚠️  'pdflatex' not found. Generated .tex but cannot compile PDF.")

# ==========================================
# 3. FIGURE 1: SYSTEM OVERVIEW (3D PNN)
# ==========================================
def draw_fig1():
    print("\n--- Generating Figure 1 (System Overview) ---")
    
    # Define architecture as a Python list
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # --- 1. IR STREAM (Content) ---
        to_input('ir_img', label='IR Input'),
        
        # Simplified Content Encoder (White/Blueish)
        *block_Simple('enc_c1', 'ir_img', 'ir_img', width=2, height=40, depth=40, caption='Content Enc', color='white'),
        *block_Simple('enc_c2', 'enc_c1', 'enc_c1', xshift=2, width=4, height=30, depth=30, color='white'),
        *block_Simple('enc_c3', 'enc_c2', 'enc_c2', xshift=2, width=6, height=20, depth=20, color='white'),

        # --- 2. REFERENCE STREAM (Color) ---
        # Shifted Y (Vertical in 2D layout, side-by-side in 3D projection)
        to_input('ref_img', label='Reference', xshift=0, yshift=-16),
        
        # Reference Encoder (Orange)
        *block_Simple('enc_r1', 'ref_img', 'ref_img', width=2, height=40, depth=40, color='orange', caption='Ref Enc'),
        *block_Simple('enc_r2', 'enc_r1', 'enc_r1', xshift=2, width=4, height=30, depth=30, color='orange'),
        *block_Simple('enc_r3', 'enc_r2', 'enc_r2', xshift=2, width=6, height=20, depth=20, color='orange'),

        # --- 3. FEATURE MATCHING (Purple) ---
        # Centered between the streams
        *block_Simple('matcher', 'enc_c3', 'enc_r3', xshift=6, yshift=-5, width=8, height=45, depth=15, 
                      color='purple', caption='Feature Matching\\n(Cross-Attn)'),
        
        # Connections to Matcher
        to_connection('enc_c3', 'matcher'),
        to_connection('enc_r3', 'matcher'),

        # --- 4. DECODER (Green) ---
        *block_Simple('dec1', 'matcher', 'matcher', xshift=6, width=6, height=20, depth=20, color='green', caption='Decoder'),
        *block_Simple('dec2', 'dec1', 'dec1', xshift=2, width=4, height=30, depth=30, color='green'),
        *block_Simple('dec3', 'dec2', 'dec2', xshift=2, width=2, height=40, depth=40, color='green'),
        
        # Output
        to_input('out_img', label='Color Output', xshift=4),
        to_connection('dec3', 'out_img'),

        to_end()
    ]
    
    to_generate(arch, "fig1_system.tex")
    compile_latex("fig1_system.tex")

# ==========================================
# 4. FIGURE 2: DETAILED ARCHITECTURE (3D PNN)
# ==========================================
def draw_fig2():
    print("\n--- Generating Figure 2 (Detailed Architecture) ---")
    
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # --- 1. IR ENCODER (ResNet34) ---
        to_input('ir_img', label='IR Input'),
        
        # Initial Conv + Pool
        *block_Simple('conv1', 'ir_img', 'ir_img', width=2, height=40, depth=40, caption='Conv1'),
        *block_Simple('pool1', 'conv1', 'conv1', xshift=1, width=1, height=32, depth=32, caption='Pool'),
        
        # ResNet Layers (Width increases, Resolution decreases)
        # Visual Size: Height/Depth shrinks. Thickness (width) grows.
        *block_Res('layer1', 'pool1', 'pool1', xshift=2, height=32, depth=32, width=4, caption='L1'),
        *block_Res('layer2', 'layer1', 'layer1', xshift=2, height=24, depth=24, width=6, caption='L2'),
        *block_Res('layer3', 'layer2', 'layer2', xshift=2, height=16, depth=16, width=8, caption='L3'),
        *block_Res('layer4', 'layer3', 'layer3', xshift=2, height=10, depth=10, width=10, caption='L4'),

        # --- 2. REFERENCE ENCODER (ResNet34) ---
        # Shifted down
        to_input('ref_img', label='Reference', xshift=0, yshift=-20),
        
        *block_Simple('ref_conv1', 'ref_img', 'ref_img', width=2, height=40, depth=40, color="orange"),
        *block_Simple('ref_pool1', 'ref_conv1', 'ref_conv1', xshift=1, width=1, height=32, depth=32, color="orange"),
        
        *block_Res('ref_layer1', 'ref_pool1', 'ref_pool1', xshift=2, height=32, depth=32, width=4, color="orange"),
        *block_Res('ref_layer2', 'ref_layer1', 'ref_layer1', xshift=2, height=24, depth=24, width=6, color="orange"),
        *block_Res('ref_layer3', 'ref_layer2', 'ref_layer2', xshift=2, height=16, depth=16, width=8, color="orange"),
        *block_Res('ref_layer4', 'ref_layer3', 'layer3', xshift=2, height=10, depth=10, width=10, caption='Ref L4', color="orange"),

        # --- 3. ATTENTION MODULE ---
        *block_Simple('attn', 'layer4', 'ref_layer4', xshift=5, yshift=10, width=4, height=40, depth=10, 
                      color="purple", caption="Match", opacity=0.6),
        
        to_connection("layer4", "attn"),
        to_connection("ref_layer4", "attn"),

        # --- 4. DECODER (U-Net) ---
        # Up1 (receives skip from L3)
        *block_Unconv('up1', 'attn', 'attn', xshift=5, height=16, depth=16, width=8, caption="Up1", color="green"),
        to_skip(of='layer3', to='up1', pos=1.5), 
        
        # Up2 (receives skip from L2)
        *block_Unconv('up2', 'up1', 'up1', xshift=2, height=24, depth=24, width=6, caption="Up2", color="green"),
        to_skip(of='layer2', to='up2', pos=1.5),
        
        # Up3 (receives skip from L1)
        *block_Unconv('up3', 'up2', 'up2', xshift=2, height=32, depth=32, width=4, caption="Up3", color="green"),
        to_skip(of='layer1', to='up3', pos=1.5),
        
        # Up4
        *block_Unconv('up4', 'up3', 'up3', xshift=2, height=40, depth=40, width=2, caption="Up4", color="green"),
        
        # Output Block
        *block_Simple('out', 'up4', 'up4', xshift=2, width=1, height=40, depth=40, color="green", caption="RGB"),
        
        to_end()
    ]
    
    to_generate(arch, "fig2_architecture.tex")
    compile_latex("fig2_architecture.tex")

# ==========================================
# 5. FIGURE 3: ATTENTION MODULE (2D TikZ)
# ==========================================
def draw_fig3_tikz():
    print("\n--- Generating Figure 3 (Attention Module) ---")
    
    # 2D Logic diagrams must use raw TikZ (PNN is 3D only)
    code = r"""
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc, shapes, shadows}

\definecolor{pnn_orange}{RGB}{255,165,0}
\definecolor{pnn_purple}{RGB}{128,0,128}
\definecolor{pnn_green}{RGB}{0,128,0}
\definecolor{pnn_blue}{RGB}{0,0,255}

\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Latex,
    block/.style={rectangle, draw=black!50, thick, rounded corners=3pt, fill=white, drop shadow},
    op/.style={circle, draw=black!60, thick, fill=white, inner sep=1pt, minimum size=0.8cm, drop shadow},
    line/.style={->, thick, color=black!70}
]

% Contextual encoder features
\node[block, fill=blue!10] (ir_feat) at (-1,2) {\textbf{IR Encoder}\\ \scriptsize Spatial Features};
\node[block, fill=pnn_orange!10] (ref_feat) at (6,2) {\textbf{Reference Encoder}\\ \scriptsize Color Features};

% Inputs
\node[block, fill=blue!5] (q) at (0,0) {\textbf{Content (Q)}\\ \scriptsize From IR};
\node[block, fill=pnn_orange!20] (k) at (3.5,0) {\textbf{Ref (K)}\\ \scriptsize From Ref};
\node[block, fill=pnn_orange!20] (v) at (7,0) {\textbf{Ref (V)}\\ \scriptsize From Ref};

\draw[line] (ir_feat) -- (q.north);
\draw[line] (ref_feat.west) |- (k.north);
\draw[line] (ref_feat.east) |- (v.north);

% Linear Projections
\node[block, below=0.8cm of q] (lq) {Linear};
\node[block, below=0.8cm of k] (lk) {Linear};
\node[block, below=0.8cm of v] (lv) {Linear};

\draw[line] (q) -- (lq);
\draw[line] (k) -- (lk);
\draw[line] (v) -- (lv);

% Attention logic
\node[op, below=1.5cm of lk, xshift=-1.75cm] (mul1) {$\times$};
\node[left=0.1cm of mul1, color=gray] {\scriptsize MatMul};

\draw[line] (lq.south) to[out=270,in=135] (mul1);
\draw[line] (lk.south) to[out=270,in=45] (mul1);

\node[block, below=0.8cm of mul1, fill=pnn_purple!10, draw=pnn_purple] (soft) {Softmax};
\draw[line] (mul1) -- (soft);

\node[op, below=0.8cm of soft] (mul2) {$\times$};
\draw[line] (soft) -- (mul2);
\draw[line] (lv.south) to[out=270,in=45] (mul2);

% Add & Norm
\node[op, below=0.8cm of mul2] (add1) {$+$};
\draw[line] (mul2) -- (add1);
\draw[line, dashed] (lq.west) -- ++(-0.5,0) |- (add1.west); % Skip Connection

\node[block, below=0.8cm of add1] (ln1) {Layer Norm};
\draw[line] (add1) -- (ln1);

\node[block, below=0.8cm of ln1, minimum width=3cm] (ffn) {Feed Forward};
\draw[line] (ln1) -- (ffn);

\node[block, below=0.8cm of ffn, fill=pnn_green!20, draw=pnn_green] (out) {\textbf{Aligned Features}};
\draw[line] (ffn) -- (out);

\node[op, below=0.8cm of out] (add2) {$+$};
\draw[line] (out) -- (add2);
\draw[line, dashed] (ln1.east) -- ++(0.5,0) |- (add2.east); % Skip Connection

\end{tikzpicture}
\end{document}
"""
    with open("fig3_attention.tex", "w") as f: f.write(code)
    compile_latex("fig3_attention.tex")

# ==========================================
# 6. FIGURE 4: AUTOMATED GRAPH (PDF Trace)
# ==========================================
def draw_fig4_pdf():
    print("\n--- Generating Figure 4 (Automated Trace) ---")
    try:
        from torchview import draw_graph # type: ignore
        from model import create_model
        
        # Robust config import handling
        try:
            from config import get_config
        except ImportError:
            # Fallback for independence if config is missing or messy
            class MockConfig:
                model = type('obj', (object,), {
                    'encoder_backbone': 'resnet34', 'pretrained_encoder': False,
                    'attention_layer': 'layer4', 'num_attention_heads': 8,
                    'attention_dropout': 0.1, 'output_channels': 3,
                    'use_skip_connections': True, 'use_instance_norm': True
                })
            get_config = lambda: MockConfig()

        config = get_config()
        model = create_model(config.model)
        
        ir = torch.randn(1, 3, 256, 256)
        ref = torch.randn(1, 3, 256, 256)
        
        # Generate DOT via torchview
        graph = draw_graph(model, input_data=(ir, ref), expand_nested=True, depth=2, save_graph=False)
        src = graph.visual_graph.source  # raw DOT

        # Parse node labels and edges from DOT
        node_label_re = re.compile(r'(?m)^\s*("?[^"\s]+"?)\s*\[([^\]]*?label="([^"]*)"[^\]]*?)\];')
        edge_re = re.compile(r'(?m)^\s*("?[^"\s]+"?)\s*->\s*("?[^"\s]+"?)')

        nodes = {}       # node_id -> label text
        for m in node_label_re.finditer(src):
            node_id = m.group(1).strip('"')
            label = m.group(3)
            nodes[node_id] = label

        edges = []
        adj = {}
        for m in edge_re.finditer(src):
            s = m.group(1).strip('"'); d = m.group(2).strip('"')
            edges.append((s,d))
            adj.setdefault(s, []).append(d)

        # Identify input nodes (labels containing 'Input' / 'input')
        input_nodes = [n for n,l in nodes.items() if re.search(r'\binput\b|\bInput\b', l)]
        # if torchview names them differently, also fallback to nodes with 'Input' or small 'x' names
        if not input_nodes:
            input_nodes = [n for n,l in nodes.items() if re.search(r'\bIn\b|\bx\b', l)]

        # Candidate encoder nodes: look for ResNet/encoder/layer keywords
        candidate_encoders = [n for n,l in nodes.items() if re.search(r'resnet|resnet34|encoder|layer|Conv1|Conv', l, re.I)]
        
        # BFS helper: find first encoder reached from an input node
        def find_encoder_from(start):
            visited = set([start])
            q = [start]
            while q:
                cur = q.pop(0)
                if cur in candidate_encoders and cur != start:
                    return cur
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        visited.add(nb); q.append(nb)
            return None

        annotations = {}
        if input_nodes:
            # Map input 0 -> IR, input 1 -> Ref (best-effort)
            for idx, inode in enumerate(input_nodes[:2]):
                tag = "IR encoder" if idx == 0 else "Ref encoder"
                enc = find_encoder_from(inode)
                if enc:
                    annotations[enc] = tag

        # Heuristic: if still unannotated but exactly two candidate encoders, mark by label keywords
        if len(annotations) < 2 and len(candidate_encoders) >= 2:
            for c in candidate_encoders[:2]:
                lbl = nodes.get(c, "")
                if re.search(r'ref|vis|rgb', lbl, re.I):
                    annotations[c] = "Ref encoder"
                elif re.search(r'ir|infra', lbl, re.I):
                    annotations[c] = "IR encoder"

        # Apply annotations to DOT source (append a compact tag to node labels)
        modified_src = src
        for node_id, tag in annotations.items():
            orig_label = nodes.get(node_id, "")
            new_label = orig_label + r"\n(" + tag + r")"
            # Replace the first occurrence of the original label text in the node declaration
            pattern = re.compile(re.escape(orig_label))
            modified_src = pattern.sub(new_label, modified_src, count=1)

        # === Expand attention node(s) into a small detailed subgraph (Q,K,V,MatMul,Softmax,FFN,Out) ===
        attn_nodes = [n for n,l in nodes.items() if re.search(r'attn|attention|match', l, re.I)]
        for a in attn_nodes:
            # remove original node declaration (robust to quotes/unquoted)
            nod_pat = re.compile(r'(?ms)^\s*"?' + re.escape(a) + r'"?\s*\[.*?\];\s*')
            modified_src = nod_pat.sub('', modified_src)
            
            # remove any edges where this node is source or target
            edge_in_pat = re.compile(r'(?m)^[ \t]*"?.+?"?[ \t]*->[ \t]*"?' + re.escape(a) + r'"?.*\n')
            edge_out_pat = re.compile(r'(?m)^[ \t]*"?' + re.escape(a) + r'"?[ \t]*->[ \t]*".+?"?.*\n')
            modified_src = edge_in_pat.sub('', modified_src)
            modified_src = edge_out_pat.sub('', modified_src)
            
            # build clustered expanded attention subgraph
            cluster = f'''
  subgraph cluster_{a} {{
    label="Attention (expanded)";
    color=purple;
    "{a}_q" [label="Q", shape=box];
    "{a}_k" [label="K", shape=box];
    "{a}_v" [label="V", shape=box];
    "{a}_matmul" [label="MatMul(Q,K^T)", shape=box];
    "{a}_soft" [label="Softmax", shape=box];
    "{a}_mulv" [label="MatMul(soft,V)", shape=box];
    "{a}_ffn" [label="FFN", shape=box];
    "{a}_out" [label="Out", shape=box];
    "{a}_q" -> "{a}_matmul";
    "{a}_k" -> "{a}_matmul";
    "{a}_matmul" -> "{a}_soft";
    "{a}_soft" -> "{a}_mulv";
    "{a}_v" -> "{a}_mulv";
    "{a}_mulv" -> "{a}_ffn";
    "{a}_ffn" -> "{a}_out";
  }}
'''
            
            # reconnect predecessors -> Q/K/V and Out -> successors
            preds = [s for s,d in edges if d == a]
            succs = [d for s,d in edges if s == a]
            reconns = ''
            for p in preds:
                # connect predecessors to each Q,K,V (keeps semantics visible)
                reconns += f'  "{p}" -> "{a}_q";\n  "{p}" -> "{a}_k";\n  "{p}" -> "{a}_v";\n'
            for s in succs:
                reconns += f'  "{a}_out" -> "{s}";\n'
            
            # insert cluster+reconnections before final closing brace
            idx = modified_src.rfind('}')
            if idx != -1:
                modified_src = modified_src[:idx] + cluster + reconns + modified_src[idx:]
            else:
                modified_src = modified_src + '\n' + cluster + reconns

        # === end attention expansion ===

        # Insert a small legend node INSIDE the graph (before the final closing brace)
        legend_node = '  legend_node [label="Legend:\\n(IR encoder) = enc from IR input\\n(Ref encoder) = enc from reference input", shape=note, fontsize=10];\n'
        idx = modified_src.rfind('}')
        if idx != -1:
            # insert before the last closing brace to keep DOT syntax valid
            modified_src = modified_src[:idx] + legend_node + modified_src[idx:]
        else:
            # fallback: append if structure unexpected
            modified_src = modified_src + '\n' + legend_node

        # Render modified DOT to PDF
        out = graphviz.Source(modified_src)
        out.format = 'pdf'
        out.render(filename='fig4_automated_trace', cleanup=True)
        print("✅ Saved 'fig4_automated_trace.pdf' (annotated)")

    except Exception as e:
        print(f"⚠️  Skipping Fig 4: {e}")

# ==========================================
# 7. FIGURE 5: RESNET ENCODER (PNN)
# ==========================================
def draw_fig5_resnet_encoder():
    print("\n--- Generating Figure 5 (ResNet Encoder) ---")
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        to_input('ir_img', label='IR Input'),

        # Initial Conv + Pool
        *block_Simple('enc_conv1', 'ir_img', 'ir_img', width=2, height=40, depth=40, caption='Conv1'),
        *block_Simple('enc_pool1', 'enc_conv1', 'enc_conv1', xshift=1, width=1, height=32, depth=32, caption='Pool'),

        # ResNet Stages
        *block_Res('enc_l1', 'enc_pool1', 'enc_pool1', xshift=2, height=32, depth=32, width=4, caption='Layer 1'),
        *block_Res('enc_l2', 'enc_l1', 'enc_l1', xshift=2, height=24, depth=24, width=6, caption='Layer 2'),
        *block_Res('enc_l3', 'enc_l2', 'enc_l2', xshift=2, height=16, depth=16, width=8, caption='Layer 3'),
        *block_Res('enc_l4', 'enc_l3', 'enc_l3', xshift=2, height=10, depth=10, width=10, caption='Layer 4'),

        # Feature output label
        to_input('enc_feats', label='Encoded Features\\n(encoder output)', xshift=4),
        to_connection('enc_l4', 'enc_feats'),

        # Legend/annotation block
        *block_Simple('legend_enc', 'enc_feats', 'enc_feats', xshift=0, yshift=-18, width=12, height=8, depth=8,
                      caption='ResNet34 Encoder\\n(red/white = conv stages)', color='white'),

        to_end()
    ]
    to_generate(arch, "fig5_resnet_encoder.tex")
    compile_latex("fig5_resnet_encoder.tex")

# ==========================================
# 8. FIGURE 6: DECODER (PNN)
# ==========================================
def draw_fig6_decoder():
    print("\n--- Generating Figure 6 (Decoder / U-Net) ---")
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # Attention / bottleneck input
        to_input('attn_in', label='Attention Output', xshift=0, yshift=0),

        # Small encoder-skip placeholders (to show where skips come from)
        *block_Simple('src_l1', 'attn_in', 'attn_in', xshift=-8, yshift=10, width=2, height=32, depth=32, caption='L1 (skip)'),
        *block_Simple('src_l2', 'src_l1', 'src_l1', xshift=0, yshift=4, width=4, height=24, depth=24, caption='L2 (skip)'),
        *block_Simple('src_l3', 'src_l2', 'src_l2', xshift=0, yshift=4, width=6, height=16, depth=16, caption='L3 (skip)'),

        # Decoder upsamples
        *block_Unconv('d_up1', 'attn_in', 'attn_in', xshift=4, height=16, depth=16, width=8, caption='Up1'),
        to_skip(of='src_l3', to='d_up1', pos=1.5),

        *block_Unconv('d_up2', 'd_up1', 'd_up1', xshift=2, height=24, depth=24, width=6, caption='Up2'),
        to_skip(of='src_l2', to='d_up2', pos=1.5),

        *block_Unconv('d_up3', 'd_up2', 'd_up2', xshift=2, height=32, depth=32, width=4, caption='Up3'),
        to_skip(of='src_l1', to='d_up3', pos=1.5),

        *block_Unconv('d_up4', 'd_up3', 'd_up3', xshift=2, height=40, depth=40, width=2, caption='Up4'),
        *block_Simple('d_out', 'd_up4', 'd_up4', xshift=2, width=1, height=40, depth=40, caption='RGB Output', color='green'),

        # Annotate decoder role
        *block_Simple('legend_dec', 'd_out', 'd_out', xshift=0, yshift=-18, width=12, height=8, depth=8,
                      caption='Decoder / Upsampling Path\\n(green = deconv + conv)', color='white'),

        to_end()
    ]
    to_generate(arch, "fig6_decoder.tex")
    compile_latex("fig6_decoder.tex")

def cleanup_latex_cruft(basenames=None, exts=None):
	print("\n--- Cleaning generated LaTeX artifacts ---")
	basenames = basenames or [
		"fig1_system", "fig2_architecture", "fig3_attention",
		"fig4_automated_trace", "fig5_resnet_encoder", "fig6_decoder"
	]
	exts = exts or ['.tex', '.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz']
	removed = []
	for base in basenames:
		for ext in exts:
			for path in glob.glob(f"{base}{ext}"):
				try:
					os.remove(path)
					removed.append(path)
				except OSError:
					pass
	if removed:
		print("🧹 Removed temporary files:", ", ".join(removed))
	else:
		print("🧹 No temporary files found.")

if __name__ == "__main__":
    draw_fig3_tikz()
    draw_fig4_pdf()
    draw_fig5_resnet_encoder()
    cleanup_latex_cruft()
