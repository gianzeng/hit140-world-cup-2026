from shared.preparation import ROOT
import json
import pandas as pd


def load_context_data():
    if (
        json.loads((ROOT / "outputs/data_audit.json").read_text()).get("status")
        != "PASS"
    ):
        raise ValueError("Source audit must pass before contextual analysis")
    tm = pd.read_csv(ROOT / "data/processed/team_matches_208.csv")
    teams = pd.read_csv(ROOT / "data/processed/teams_48.csv")
    opponents = tm[["match_id", "team", "shots", "sot"]].rename(
        columns={"team": "opponent", "shots": "opponent_shots", "sot": "opponent_sot"}
    )
    d = tm.merge(opponents, on=["match_id", "opponent"], validate="one_to_one")
    assert len(d) == 208 and (d.shots + d.opponent_shots > 0).all()
    return teams, d
