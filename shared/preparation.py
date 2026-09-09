from shared.paths import location
from pathlib import Path
import csv, json, re, hashlib, os

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".build/mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".build/cache")
)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN, OUT = (ROOT / "data/raw", ROOT / "data/processed", ROOT / "outputs")
ALIASES = {
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Turkey": "Türkiye",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Iran": "IR Iran",
    "D.R. Congo": "Congo DR",
    "Curacao": "Curaçao",
}
SEED = 1402026


def canonical(name):
    return ALIASES.get(name.strip(), name.strip())


def score(value):
    # The p marker identifies the shootout winner.
    if not re.fullmatch("p?\\d+p?", value.strip()):
        raise ValueError(f"Unexpected score format: {value!r}")
    return int(value.strip().replace("p", ""))


def require_clean_sources(issues):
    if issues:
        location("data_audit.json", "outputs").write_text(
            json.dumps({"status": "BLOCKED", "source_issues": issues}, indent=2)
        )
        raise ValueError(
            "Source validation blocked this run. See outputs/data_audit.json; previous outputs, if present, belong to an earlier run."
        )


def read_matches():
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    matches = []
    stage = None
    rows = list(csv.reader((RAW / "tsdl_matches.csv").open(encoding="utf-8-sig")))
    stages = {
        "GROUP STAGE",
        "ROUND OF 32",
        "ROUND OF 16",
        "QUARTER FINALS",
        "SEMI FINALS",
        "FINAL",
    }
    for line, r in enumerate(rows, 1):
        if r[0] in stages:
            stage = r[0]
            continue
        if not re.fullmatch("\\d{2}/\\d{2}/2026", r[0]):
            continue
        d = {
            "date": pd.to_datetime(r[0], dayfirst=True).strftime("%Y-%m-%d"),
            "stage": stage,
            "team1": canonical(r[1]),
            "team2": canonical(r[4]),
            "goals1": score(r[2]),
            "goals2": score(r[3]),
            "source": "TSDL",
            "source_row": line,
            "shootout": "p" in r[2] + r[3],
        }
        for metric, i in [
            ("yellow", 9),
            ("red", 11),
            ("corners", 17),
            ("xg", 20),
            ("shots", 22),
            ("sot", 24),
            ("fouls", 26),
        ]:
            for side in (1, 2):
                d[f"{metric}{side}"] = (
                    float(r[i + side - 1]) if r[i + side - 1].strip() else np.nan
                )
        matches.append(d)
    bronze = json.loads((RAW / "fifa_bronze_match.json").read_text())
    bronze.update(source="FIFA", source_row=np.nan, shootout=False)
    matches.append(
        {k: v for k, v in bronze.items() if k not in ("date_basis", "source_url")}
    )
    m = (
        pd.DataFrame(matches)
        .sort_values(["date", "team1", "team2"])
        .reset_index(drop=True)
    )
    m.insert(0, "match_id", [f"WC26_{i:03d}" for i in range(1, len(m) + 1)])
    assert len(m) == 104 and (not m.duplicated(["date", "team1", "team2"]).any())
    assert (m.stage == "GROUP STAGE").sum() == 72
    assert (m.team1 != m.team2).all()
    for metric in ["goals", "yellow", "red", "corners", "shots", "sot", "fouls"]:
        for side in (1, 2):
            s = m[f"{metric}{side}"]
            assert s.notna().all() and (s >= 0).all() and (s % 1 == 0).all(), (
                metric,
                side,
            )
            m[f"{metric}{side}"] = s.astype(int)
    issues = []
    for side in (1, 2):
        for _, r in m[m[f"sot{side}"] > m[f"shots{side}"]].iterrows():
            issues.append(
                {"match_id": r.match_id, "issue": f"SOT exceeds shots for side {side}"}
            )
    return m, issues


def make_team_matches(m):
    long = []
    for _, r in m.iterrows():
        for side, other in [(1, 2), (2, 1)]:
            d = {k: r[k] for k in ["match_id", "date", "stage", "source"]}
            d.update(
                team=r[f"team{side}"],
                opponent=r[f"team{other}"],
                goals_against=r[f"goals{other}"],
            )
            for metric in [
                "goals",
                "yellow",
                "red",
                "corners",
                "xg",
                "shots",
                "sot",
                "fouls",
            ]:
                d[metric] = r[f"{metric}{side}"]
            long.append(d)
    tm = pd.DataFrame(long)
    assert len(tm) == 208 and (not tm.duplicated(["match_id", "team"]).any())
    return tm


def assign_rankings(tm):
    ranking = pd.read_csv(RAW / "fifa_rankings_2026-06-11.csv")
    assert (
        len(ranking) == 48 and ranking.team.is_unique and ranking.world_rank.is_unique
    )
    assert set(tm.team) == set(ranking.team), set(tm.team) ^ set(ranking.team)
    ranking = ranking.sort_values("world_rank").reset_index(drop=True)
    ranking["entrant_rank"] = np.arange(1, 49)
    ranking["ranking_group"] = np.where(ranking.entrant_rank <= 24, "Higher", "Lower")
    tm = tm.merge(ranking, on="team", validate="many_to_one")
    opp = ranking[["team", "world_rank"]].rename(
        columns={"team": "opponent", "world_rank": "opponent_rank"}
    )
    tm = tm.merge(opp, on="opponent", validate="many_to_one")
    return tm, ranking


def assign_qualification(tm):
    qualified = set(tm.loc[tm.stage == "ROUND OF 32", "team"])
    assert len(qualified) == 32
    tm["qualification"] = np.where(tm.team.isin(qualified), "Qualified", "Eliminated")
    group = tm[tm.stage == "GROUP STAGE"]
    assert group.groupby("team").size().eq(3).all()
    return tm, group, qualified


def summarise_teams(tm, ranking, group, qualified):
    team = (
        tm.groupby("team")
        .agg(
            matches=("match_id", "size"),
            goals=("goals", "sum"),
            shots=("shots", "sum"),
            goals_against=("goals_against", "sum"),
        )
        .reset_index()
    )
    team = team.merge(ranking, on="team", validate="one_to_one")
    team = team.merge(
        group.groupby("team").yellow.mean().rename("yellow_per_group_match"),
        on="team",
        validate="one_to_one",
    )
    team = team.merge(
        pd.read_csv(RAW / "fifa_goalkeeping.csv"), on="team", validate="one_to_one"
    )
    assert team.saves.notna().all() and len(team) == 48
    team["goals_per_match"] = team.goals / team.matches
    team["shots_per_match"] = team.shots / team.matches
    team["saves_per_match"] = team.saves / team.matches
    team["qualification"] = np.where(
        team.team.isin(qualified), "Qualified", "Eliminated"
    )
    return team


def check_goalkeeping_totals(team, issues):
    for _, r in team[team.goals_against != team.goals_conceded].iterrows():
        issues.append(
            {
                "team": r.team,
                "issue": "Source disagreement: TSDL/FIFA match GA vs official team GA",
                "match_sum": int(r.goals_against),
                "official_total": int(r.goals_conceded),
            }
        )


def save_datasets(m, tm, team, ranking, qualified, issues):
    require_clean_sources(issues)
    for name, frame in [
        ("matches_104", m),
        ("team_matches_208", tm),
        ("teams_48", team),
        ("ranking_groups", ranking),
    ]:
        frame.to_csv(location(f"{name}.csv", "data"), index=False)
    audit = {
        "status": "PASS",
        "matches": len(m),
        "team_matches": len(tm),
        "teams": len(team),
        "group_matches": 72,
        "qualified": len(qualified),
        "missing_xg_matches": int(m[["xg1", "xg2"]].isna().any(axis=1).sum()),
        "source_issues": issues,
        "shootout_matches": int(m.shootout.sum()),
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RAW.iterdir()
            if p.is_file()
        },
    }
    location("data_audit.json", "outputs").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )


def prepare():
    matches, issues = read_matches()
    team_matches = make_team_matches(matches)
    team_matches, ranking = assign_rankings(team_matches)
    team_matches, group, qualified = assign_qualification(team_matches)
    teams = summarise_teams(team_matches, ranking, group, qualified)
    check_goalkeeping_totals(teams, issues)
    save_datasets(matches, team_matches, teams, ranking, qualified, issues)
    return teams
