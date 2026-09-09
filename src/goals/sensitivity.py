from shared.paths import location
from shared.preparation import ROOT
import numpy as np
import pandas as pd
from scipy.stats import skew


def check_distribution(sample, group, metric, r):
    diagnostics = []
    for label, part in sample.groupby(group):
        x = part[metric]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        flagged = part.loc[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr), "team"].tolist()
        diagnostics.append(
            dict(
                task=r["task"],
                group=label,
                n=len(x),
                skewness=float(skew(x, bias=False)),
                tukey_flagged_teams="; ".join(flagged),
                retained_all_teams=True,
            )
        )
    return diagnostics


def check_influence(sample, group, metric, r):
    influence = []
    deltas = []
    for name in sample.team:
        reduced = sample[sample.team != name]
        means = reduced.groupby(group)[metric].mean()
        deltas.append(float(means[r["labels"][0]] - means[r["labels"][1]]))
    influence.append(
        dict(
            task=r["task"],
            original_difference=r["difference"],
            leave_one_team_out_min=min(deltas),
            leave_one_team_out_max=max(deltas),
            all_directions_match=bool(
                all((np.sign(d) == np.sign(r["difference"]) for d in deltas))
            ),
        )
    )
    return influence


def check_group_stage(sample, tm, r):
    exposure = []
    variable = "goals"
    rates = tm[tm.stage == "GROUP STAGE"].groupby("team")[variable].mean()
    for scope, names in [
        ("original_sample", set(sample.team)),
        ("full_census", set(tm.team)),
    ]:
        base = tm[["team", "ranking_group"]].drop_duplicates()
        base = base[base.team.isin(names)].copy()
        base["group_rate"] = base.team.map(rates)
        full = tm.groupby("team")[variable].mean()
        base["full_rate"] = base.team.map(full)
        means = base.groupby("ranking_group")[["group_rate", "full_rate"]].mean()
        exposure.append(
            dict(
                task=r["task"],
                scope=scope,
                higher_group_mean=means.loc["Higher", "group_rate"],
                lower_group_mean=means.loc["Lower", "group_rate"],
                group_stage_difference=means.loc["Higher", "group_rate"]
                - means.loc["Lower", "group_rate"],
                full_tournament_difference=means.loc["Higher", "full_rate"]
                - means.loc["Lower", "full_rate"],
            )
        )
    return exposure


def run(result):
    sample = pd.read_csv(location(f"{result['task']}_sample.csv", "data"))
    group = sample.columns[1]
    metric = result["metric"]
    diagnostics = check_distribution(sample, group, metric, result)
    influence = check_influence(sample, group, metric, result)
    matches = pd.read_csv(ROOT / "data/processed/team_matches_208.csv")
    exposure = check_group_stage(sample, matches, result)
    return diagnostics, influence, exposure
