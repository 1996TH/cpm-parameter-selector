"""Visualization utilities for grid search results."""

from __future__ import annotations

import pandas as pd


def render_table_image(df: pd.DataFrame, ticker: str, save_path: str):
    """Render grid search table as a clean image with ticker name.

    Args:
        df: Raw grid search DataFrame (from grid_search()).
        ticker: Asset name for title.
        save_path: File path to save the image.
    """
    import matplotlib.pyplot as plt

    from cpm.param_selector import format_table

    table = format_table(df)
    n_rows, n_cols = table.shape

    fig_w = max(14, n_cols * 1.5 + 1.5)
    fig_h = n_rows * 0.5 + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    fig.text(0.5, 0.97, ticker, fontsize=16, fontweight="bold",
             ha="center", va="top", fontfamily="serif")

    col_labels = ["T \\ P"] + list(table.columns)
    cell_text = []
    for idx, row in table.iterrows():
        cell_text.append([str(idx)] + [str(v) for v in row.values])

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_text_props(fontweight="bold", fontfamily="serif", fontsize=13)
        cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#333333")
        cell.set_linewidth(1.5)

    for i in range(1, n_rows + 1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_text_props(fontfamily="serif", fontsize=12)
            cell.set_edgecolor("#666666")
            cell.set_linewidth(0.8)
            if j == 0:
                cell.set_text_props(fontweight="bold", fontfamily="serif", fontsize=13)
                cell.set_facecolor("#f0f0f0")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path
