from shared.paths import location
import json
import numpy as np
import pandas as pd
from shared.preparation import prepare, ROOT
from src.goals import analysis as goals, context as goal_context
from src.yellow_cards import (
    analysis as cards,
    context as card_context,
    regression as match_model,
)
from src.shots import analysis as shots, context as shot_context
from src.saves import (
    analysis as saves,
    context as save_context,
    regression as team_model,
)
from shared.context import load_context_data
from src.goals import sensitivity as goals_checks
from src.yellow_cards import sensitivity as yellow_cards_checks
from src.shots import sensitivity as shots_checks
from src.saves import sensitivity as saves_checks


def save_comparisons(results):
    order = np.argsort([r["p"] for r in results])
    prev = 0
    for i, j in enumerate(order):
        prev = max(prev, min(1, (4 - i) * results[j]["p"]))
        results[j]["holm_p"] = prev
    location("all_results.json", "outputs").write_text(json.dumps(results, indent=2))
    pd.DataFrame(
        [
            {k: r[k] for k in ["task", "owner", "difference", "t", "df", "p", "holm_p"]}
            for r in results
        ]
    ).to_csv(location("test_summary.csv", "outputs"), index=False)


def save_sensitivity(results):
    diagnostics, influence, exposure = ([], [], [])
    tasks = (goals_checks, yellow_cards_checks, shots_checks, saves_checks)
    for task, result in zip(tasks, results):
        task_diagnostics, task_influence, task_exposure = task.run(result)
        diagnostics.extend(task_diagnostics)
        influence.extend(task_influence)
        exposure.extend(task_exposure)
    pd.DataFrame(diagnostics).to_csv(
        location("distribution_checks.csv", "outputs"), index=False
    )
    pd.DataFrame(influence).to_csv(
        location("leave_one_team_out.csv", "outputs"), index=False
    )
    pd.DataFrame(exposure).to_csv(
        location("group_stage_sensitivity.csv", "outputs"), index=False
    )
    print(pd.DataFrame(exposure).to_string(index=False))
    print(pd.DataFrame(influence).to_string(index=False))


def save_context_results():
    teams, matches = load_context_data()
    metrics = pd.concat(
        [
            goal_context.derive(matches),
            shot_context.derive(matches),
            save_context.derive(matches),
        ],
        axis=1,
    ).reset_index()
    context = teams.merge(metrics, on="team", validate="one_to_one")
    context = save_context.add_save_ratio(context)
    context.to_csv(ROOT / "data/processed/context_teams_48.csv", index=False)
    rows = (
        goal_context.run(context)
        + shot_context.run(context)
        + save_context.run(context)
    )
    pd.DataFrame(rows).to_csv(location("context_summary.csv", "outputs"), index=False)
    card_context.run(context)


def main():
    teams = prepare()
    results = [task.run(teams) for task in (goals, cards, shots, saves)]
    save_comparisons(results)
    save_sensitivity(results)
    save_context_results()
    match_model.run()
    team_model.run()


if __name__ == "__main__":
    main()
