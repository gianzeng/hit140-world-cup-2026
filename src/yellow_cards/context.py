from shared.context import load_context_data
from shared.paths import location
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ranking_cells(context):
    cells = (
        context.groupby(["qualification", "ranking_group"])
        .agg(n=("team", "size"), yellow_mean=("yellow_per_group_match", "mean"))
        .reset_index()
    )
    assert len(cells) == 4 and (cells.n > 0).all()
    cells.to_csv(location("yellow_rank_strata.csv", "outputs"), index=False)
    return cells


def standardise_ranking(context, cells):
    yellow = []
    for label, g in context.groupby("qualification"):
        cell = cells[cells.qualification == label]
        yellow.append(
            dict(
                group=label,
                n=len(g),
                higher_share=(g.ranking_group == "Higher").mean(),
                observed_mean=g.yellow_per_group_match.mean(),
                rank_standardised_mean=cell.yellow_mean.mean(),
            )
        )
    yellow = pd.DataFrame(yellow)
    yellow.to_csv(location("yellow_rank_standardisation.csv", "outputs"), index=False)
    return yellow


def plot_composition(yellow):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ix = np.arange(2)
    ax.bar(
        ix - 0.18,
        yellow.observed_mean,
        0.36,
        label="Observed composition",
        color="#0798B5",
    )
    ax.bar(
        ix + 0.18,
        yellow.rank_standardised_mean,
        0.36,
        label="50:50 ranking composition",
        color="#C51F35",
    )
    ax.set_xticks(ix, yellow.group)
    ax.set_ylabel("Yellow cards per group-stage match")
    ax.set_title("Qualification and ranking composition")
    ax.legend()
    fig.text(
        0.03,
        0.01,
        "48 teams. Higher/Eliminated: n=4, with 50% standardised weight. Descriptive only.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(location("yellow_composition.png", "outputs"), dpi=180)
    plt.close(fig)


def run(context=None):
    if context is None:
        context, _ = load_context_data()
    cells = ranking_cells(context)
    yellow = standardise_ranking(context, cells)
    plot_composition(yellow)
    return yellow


if __name__ == "__main__":
    print(run())
