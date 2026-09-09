from shared.preparation import ROOT
import pandas as pd


def prepare():
    return pd.read_csv(ROOT / "data/processed/team_matches_208.csv")


if __name__ == "__main__":
    print(prepare().to_string(index=False))
