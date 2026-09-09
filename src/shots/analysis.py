from shared.paths import location
import json
import numpy as np
import pandas as pd
from scipy import stats
from shared.preparation import prepare, SEED
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = "03_shots"
OWNER = "Zhixuan Zhao"
METRIC = "shots_per_match"
GROUP = "ranking_group"
LABELS = ("Higher", "Lower")
SAMPLE_SIZES = (20, 20)


def select_sample(team):
    frame = team[["team", GROUP, METRIC]].sort_values("team")
    samples = [
        frame[frame[GROUP] == label].sample(n=n, random_state=SEED).sort_values("team")
        for label, n in zip(LABELS, SAMPLE_SIZES)
    ]
    selected = pd.concat(samples)
    values = [s[METRIC].to_numpy(float) for s in samples]
    return frame, selected, values


def describe_groups(frame, values):
    desc = []
    for label, x in zip(LABELS, values):
        N = int((frame[GROUP] == label).sum())
        n = len(x)
        sd = x.std(ddof=1)
        se = sd / np.sqrt(n)
        critical = stats.t.ppf(0.975, n - 1)
        # Sampling is without replacement.
        fpc = np.sqrt(1 - n / N)
        desc.append(
            dict(
                group=label,
                N=N,
                n=n,
                mean=x.mean(),
                sd=sd,
                median=np.median(x),
                min=x.min(),
                max=x.max(),
                ci_low=x.mean() - critical * se,
                ci_high=x.mean() + critical * se,
                finite_ci_low=x.mean() - critical * se * fpc,
                finite_ci_high=x.mean() + critical * se * fpc,
                census_mean=float(frame.loc[frame[GROUP] == label, METRIC].mean()),
            )
        )
    return desc


def welch_test(values, desc):
    x, y = values
    n1, n2 = (len(x), len(y))
    a = x.var(ddof=1) / n1
    b = y.var(ddof=1) / n2
    se = np.sqrt(a + b)
    df = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
    delta = x.mean() - y.mean()
    t, p = stats.ttest_ind(x, y, equal_var=False)
    crit = stats.t.ppf(0.975, df)
    assert np.isclose(t, delta / se)
    result = dict(
        task=KEY,
        owner=OWNER,
        metric=METRIC,
        labels=LABELS,
        descriptive=desc,
        difference=delta,
        t=float(t),
        df=float(df),
        p=float(p),
        difference_ci=[delta - crit * se, delta + crit * se],
        reject_at_005=bool(p < 0.05),
        seed=SEED,
        inference_note="Welch and ordinary t intervals are approximate model-based inference. Finite-population group intervals use FPC; no population sampling uncertainty remains for census means. Shared fixtures and tournament selection limit generalisation.",
    )
    return result


def save_results(selected, desc, result):
    selected.to_csv(location(f"{KEY}_sample.csv", "data"), index=False)
    location(f"{KEY}.json", "outputs").write_text(json.dumps(result, indent=2))
    pd.DataFrame(desc).to_csv(
        location(f"{KEY}_descriptive.csv", "outputs"), index=False
    )


def plot_results(values, desc):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["#C51F35", "#0798B5"]
    rng = np.random.default_rng(SEED)
    for i, (label, x, d, c) in enumerate(zip(LABELS, values, desc, colors)):
        ax.scatter(rng.normal(i, 0.045, len(x)), x, color=c, alpha=0.65, s=27)
        ax.errorbar(
            i + 0.16,
            d["mean"],
            yerr=[[d["mean"] - d["ci_low"]], [d["ci_high"] - d["mean"]]],
            fmt="D",
            color="#1C395B",
            capsize=6,
            linewidth=2,
        )
    ax.set_xticks([0, 1], [f"{l}\nn={n}" for l, n in zip(LABELS, SAMPLE_SIZES)])
    ax.set_ylabel("Shots per match")
    ax.set_title(OWNER + " — " + KEY[3:].replace("_", " ").title())
    ax.grid(axis="y", alpha=0.15)
    fig.text(
        0.02,
        0.01,
        "Dots: sampled team averages. Diamonds: means and ordinary 95% t intervals.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(location(f"{KEY}.png", "outputs"), dpi=180)
    plt.close(fig)


def run(team=None):
    if team is None:
        team = prepare()
    frame, selected, values = select_sample(team)
    desc = describe_groups(frame, values)
    result = welch_test(values, desc)
    save_results(selected, desc, result)
    plot_results(values, desc)
    return result


if __name__ == "__main__":
    print(run())
