import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import warnings
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

plt.rcParams["pdf.fonttype"]=42
warnings.filterwarnings ("ignore")

def plot_history(train_, valid_, task, fname: str):
    """Create a plot of train valid loss/acc"""
    fig, ax = plt.subplots(dpi=300)

    ax.plot(
        np.arange(len(train_)), train_, label="Train",
    )

    ax.plot(
        np.arange(len(valid_)), valid_, label="Valid",
    )

    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel=task,
    )

    fig.savefig(fname)
    plt.close(fig)


def plot_conf_mat(
            targ, 
            pred, 
            output, 
            idx2celltype=None, 
            cmap="Blues", 
            mask=0.1
):
    ConfMat = np.round(confusion_matrix(targ, pred, normalize="true"), 2)
    classes_y = list(np.unique(targ))
    classes_x = classes_y.copy()
    
    if idx2celltype != None:
        classes_x = list(idx2celltype.values())
        classes_y = list(idx2celltype.values())
    
    fig_size = int(len(classes_y)/2)
    fig = plt.figure(figsize=(fig_size, fig_size))
    
    ax = sns.heatmap(
            ConfMat, 
            fmt="g", 
            cmap=cmap,
            annot=True,
            cbar=True, 
            cbar_kws={"ticks":[0,1],"fraction":0.1},
            xticklabels=classes_x, 
            yticklabels=classes_y, 
            mask=ConfMat<mask,
            vmax=1, 
            vmin=0, 
            square=True
    )

    ax.tick_params(
            bottom=False,
            left=False,
            labelsize=14
    )

    plt.xticks(
            rotation=45,
            rotation_mode="anchor",
            verticalalignment="center",
            horizontalalignment="right"
    )

    plt.savefig(output, dpi=300, bbox_inches="tight")


def createCorlormap(color_input, value=None, name="custom_cmap"):  
    from matplotlib.colors import LinearSegmentedColormap
    """  
    创建colormap
    Parameters:  
    color_input (list or str): 颜色数组 16进制 / rgb数组 均可 (e.g., ['#FF0000', '#00FF00']).  or rgb (R, G, B)
    values (list): 默认请不要传入, 表示颜色对应映射到的值 0~1内, 长度和colors数组一致 
    name (string): colorbar name 无所谓的参数
    """  
    # Convert hex colors to RGB tuples
    if color_input == 'exp':
        # "#D3D3D3", "#C0ABF2", "#0000FF"
        colors = ["#D3D3D3", "#A685F1", '#7B4DE6', "#0303FF"]   #B39AF2   #9D7BED
        
    elif color_input == 'attn':
        # "#D3D3D3", "#E37373","#7919A5"
        colors = ["#D3D3D3", "#E25D5D","#781AA4"]   # #E25D5D  #781AA4
    else: 
        colors = color_input
    
    if (type(colors[0]) == str):
        rgb_colors = [tuple(int(color[1+i:1+i+2], 16) / 255.0 for i in (0, 2, 4)) for color in colors]  
    else:
        rgb_colors = colors
    # value
    if value is not None:
        dist = 1/(len(rgb_colors)-1)
        rgb_colors = list(zip(np.arange(start=0, stop=1+dist, step=dist), rgb_colors))
    # Create the colormap  
    cmap = LinearSegmentedColormap.from_list(name, rgb_colors)  
      
    return cmap  


def plot_tag_heatmap(
        adata,
        adata_cls, 
        attn_tensor, 
        TAGs, 
        topN: int=5, 
        range_scale: tuple=(-2, 2), 
        figsize: tuple=None, 
):
    r""" plot the heatmap of attention score or experssion signal (TAGs)
    :param adata: test dataset (exp)
    :param adata_cls: CellMemory output cls
    :param attn_tensor: CellMemory output attention score of cls
    :param TAGs: the top attention score genes for each cell type
    :param topN: plot top N TAGs
    :param range_scale: scale range
    :param figsize: if None, automatically set the fig size
    """

    # figsize
    figsize = figsize if figsize else (10, round(1.5*topN, 1))

    # order cell type
    order_celltype = adata_cls.obs["Pred"].value_counts().index.tolist()
    TAGs = TAGs[order_celltype]

    # topN TAGs
    dict_TAGs = {}
    for category in order_celltype:
        dict_TAGs[category] = TAGs[category][:topN].tolist()
    
    # colorbar
    def mappable(colormap: tuple):
        distance = 1/(len(colormap)-1)
        li_map = list(zip(np.arange(start=0, stop=1+distance, step=distance), colormap))
        return LinearSegmentedColormap.from_list("bar", li_map)
    
    # plot func
    def plot_heatmap(
                adata_mat, 
                dict_tags, 
                output, 
                cmap, 
                range_scale, 
                figsize
    ):
        cmap_seg = mappable(cmap)
        cmap_scalar = ScalarMappable(norm=Normalize(*range_scale), cmap=cmap_seg)
        ax = sc.pl.heatmap(adata_mat, dict_tags, groupby="Pred", cmap=cmap_seg, show_gene_labels=True,
                           swap_axes=True, use_raw=False, vmin=range_scale[0], vmax=range_scale[1],
                           figsize=figsize, show=False)
        
        fig = plt.figure(num=plt.get_fignums()[-1])
        axes_bar = fig.get_axes()[3]
        
        bBox = axes_bar.get_position()
        bBox.x1 = bBox.x0 + bBox.width * 1.5
        bBox.y1 = bBox.y0 + bBox.height * 0.5
        axes_bar.set_position(bBox)
        bar = plt.colorbar(mappable=cmap_scalar, cax=axes_bar, ticks=range_scale, pad=1)
         
        ax["groupby_ax"].set_xlabel("")
        ax["heatmap_ax"]._children[1].remove()
        # xticks 45°
        plt.sca(ax["groupby_ax"])
        plt.xticks(rotation=45, rotation_mode="anchor", verticalalignment="center",
                   horizontalalignment="right", fontsize=12)
        ax["gene_groups_ax"].remove()
        
        bar.outline.set_visible(False)
        ax["heatmap_ax"].spines[:].set_visible(False)
        
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close("all")
    
    # plot attention score heatmap
    print('plotting attention score ..')
    cmap = ("#66a9c9","#ffffff","#de1c31")
    adata_mat = ad.AnnData(
                        np.array(attn_tensor),
                        var=adata.var,
                        obs={"Pred": 
                                    pd.Categorical(adata_cls.obs["Pred"], 
                                    categories=order_celltype, 
                                    ordered=True
                                )
                            }
                        )

    sc.pp.scale(adata_mat)
    plot_heatmap(adata_mat, dict_TAGs, "TAGs_heatmap_atten.pdf", cmap, range_scale, figsize)

    # plot exp signal heatmap
    print('plotting experssion signal ..')
    cmap = ("#66a9c9","#ffffff","#c3338d")
    adata_mat = ad.AnnData(
                        adata.X,
                        var=adata.var,
                        obs={"Pred": 
                                    pd.Categorical(adata_cls.obs["Pred"], 
                                    categories=order_celltype, 
                                    ordered=True
                                )
                            }
                        )

    sc.pp.scale(adata_mat)
    plot_heatmap(adata_mat, dict_TAGs, "TAGs_heatmap_exp.pdf", cmap, range_scale, figsize)


