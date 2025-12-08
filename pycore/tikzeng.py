
import os

def to_head( projectpath ):
    pathlayers = os.path.join( projectpath, 'layers/' ).replace('\\', '/')
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
"""

def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

# layers definition

def to_input(pathfile, to='(-3,0,0)', width=8, height=8, name=None, label=None, xshift=0, yshift=0):
    # allow using the path/file as the node name when no explicit name provided
    if name is None:
        name = pathfile

    # if shifts are provided and default `to` is used, compute position from shifts
    if (xshift != 0 or yshift != 0) and to == '(-3,0,0)':
        to = f"({xshift},{yshift},0)"

    tex = r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """ + to + r""" {\includegraphics[width=""" + str(width) + """cm,height=""" + str(height) + """cm]{""" + pathfile + """}};
"""
    if label:
        tex += r"""
\node[below=0.1cm of """ + name + """, font=\small] (""" + name + """_lbl) {""" + label + """};
"""
    return tex

# Conv
def to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(n_filer) + r""", }},
        zlabel=""" + str(s_filer) + r""",
        fill=\ConvColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu( name, s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={ """ + offset + r""" }] at """ + to + r""" 
    {RightBandedBox={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{ """ + str(n_filer[0]) + r""", """ + str(n_filer[1]) + r""" }},
        zlabel=""" + str(s_filer) + r""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height=""" + str(height) + r""",
        width={ """ + str(width[0]) + r""" , """ + str(width[1]) + r""" },
        depth=""" + str(depth) + r"""
        }
    };
"""



# Pool
def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """ + offset + r""" }] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        fill=\PoolColor,
        opacity=""" + str(opacity) + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """ + offset + r""" }] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        fill=\UnpoolColor,
        opacity=""" + str(opacity) + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""



def to_ConvRes( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ):
    return r"""
\pic[shift={ """ + offset + r""" }] at """ + to + r""" 
    {RightBandedBox={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{ """ + str(n_filer) + r""", }},
        zlabel=""" + str(s_filer) + r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity=""" + str(opacity) + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax( name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        zlabel=""" + str(s_filer) + r""",
        fill=\SoftmaxColor,
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

# SoftMax
def to_SoftMax( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{" ","dummy"}},
        zlabel=""" + str(s_filer) + r""",
        fill=\SoftmaxColor,
        opacity=""" + str(opacity) + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""

def to_Sum( name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r""" 
    {Ball={
        name=""" + name + r""",
        fill=\SumColor,
        opacity=""" + str(opacity) + r""",
        radius=""" + str(radius) + r""",
        logo=$+$
        }
    };
"""


def to_connection( of, to):
    return r"""
\draw [connection]  (""" + of + r"""-east)    -- node {\midarrow} (""" + to + r"""-west);
"""

def to_skip( of, to, pos=1.25):
    return r"""
\path (""" + of + r"""-southeast) -- (""" + of + r"""-northeast) coordinate[pos=""" + str(pos) + r"""] (""" + of + r"""-top) ;
\path (""" + to + r"""-south)  -- (""" + to + r"""-north)  coordinate[pos=""" + str(pos) + r"""] (""" + to + r"""-top) ;
\draw [copyconnection]  (""" + of + r"""-northeast)  
-- node {\copymidarrow}(""" + of + r"""-top)
-- node {\copymidarrow}(""" + to + r"""-top)
-- node {\copymidarrow} (""" + to + r"""-north);
"""

def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_generate( arch, pathname="file.tex" ):
    with open(pathname, "w") as f:
        for c in arch:
            f.write(c)
     


