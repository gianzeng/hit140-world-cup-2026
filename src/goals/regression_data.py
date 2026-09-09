from shared.preparation import ROOT
import pandas as pd


def prepare():
    m = pd.read_csv(ROOT / "data/processed/matches_104.csv")
    tm = pd.read_csv(ROOT / "data/processed/team_matches_208.csv")
    for field in ["goals", "shots", "yellow", "corners"]:
        m["total_" + field] = m[field + "1"] + m[field + "2"]
    ranks = tm.groupby("match_id").world_rank.agg(["min", "max"])
    m = m.merge(ranks, on="match_id", validate="one_to_one")
    m["rank_gap"] = m["max"] - m["min"]
    return m


if __name__ == "__main__":
    print(prepare().to_string(index=False))
