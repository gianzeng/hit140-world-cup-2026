from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "01_goals": "goals",
    "02_yellow_cards": "yellow_cards",
    "03_shots": "shots",
    "04_saves": "saves",
}


def owner(name):
    for prefix, task in TASKS.items():
        if name.startswith(prefix):
            return task
    if name.startswith(("goals_consistency",)):
        return "goals"
    if name.startswith(("shots_share",)):
        return "shots"
    if name.startswith(("saves_exposure", "regression_team_match")):
        return "saves"
    if name.startswith(("yellow_", "regression_match")):
        return "yellow_cards"
    return None


def location(name, kind):
    task = owner(name)
    if task:
        base = ROOT / "src" / task / ("data" if kind == "data" else "outputs")
    else:
        base = ROOT / ("data/processed" if kind == "data" else "outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def task_data(key):
    return location(f"{key}_sample.csv", "data")
