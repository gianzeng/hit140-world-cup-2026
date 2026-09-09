from shared.context import load_context_data
from shared.paths import location
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derive(d):
    d = d.assign(scoreless=(d.goals == 0).astype(int))
    return d.groupby("team").agg(scoreless_match_share=("scoreless", "mean"))


def prepare_context():
    teams, matches = load_context_data()
    context = teams.merge(
        derive(matches).reset_index(), on="team", validate="one_to_one"
    )
    return context


def summarise(context, task, metrics):
    names = set(pd.read_csv(location(f"{task}_sample.csv", "data")).team)
    rows = []
    for scope, frame in [
        ("original_sample", context[context.team.isin(names)]),
        ("full_census", context),
    ]:
        for label, g in frame.groupby("ranking_group"):
            for metric in metrics:
                rows.append(
                    dict(
                        task=task,
                        scope=scope,
                        group=label,
                        metric=metric,
                        n=len(g),
                        mean=g[metric].mean(),
                    )
                )
    return rows


def plot_rates(context, metric, title, ylabel, name):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"Higher": "#C51F35", "Lower": "#0798B5"}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rng = np.random.default_rng(1402026)
    for index, (label, c) in enumerate(colors.items()):
        x = context[context.ranking_group == label][metric]
        ax.scatter(rng.normal(index, 0.04, len(x)), x, color=c, s=27, alpha=0.7)
        ax.scatter(index + 0.15, x.mean(), marker="D", color="#1C395B", s=50)
    ax.set_xticks([0, 1], ["Higher (24 teams)", "Lower (24 teams)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric in ("scoreless_match_share", "mean_match_shot_share"):
        ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.15)
    fig.text(
        0.03,
        0.01,
        "Full census. Dots: team metrics. Diamonds: equal-team means. Descriptive only.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(location(f"{name}.png", "outputs"), dpi=180)
    plt.close(fig)


def run(context=None):
    if context is None:
        context = prepare_context()
    plot_rates(
        context,
        "scoreless_match_share",
        "Scoring consistency across teams",
        "Share of matches without a goal",
        "goals_consistency",
    )
    return summarise(context, "01_goals", ["scoreless_match_share"])


if __name__ == "__main__":
    print(run())
