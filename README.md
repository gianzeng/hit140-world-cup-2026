# HIT140 — FIFA World Cup 2026

| Task directory | Member | Analysis |
|---|---|---|
| `src/goals/` | Yi Xie | Goals and match-model data preparation |
| `src/yellow_cards/` | Jiean Zeng | Yellow cards and match model |
| `src/shots/` | Zhixuan Zhao | Shots and team-match data preparation |
| `src/saves/` | Jack Zhang | Goalkeeper saves and team-match model |

Each task contains its Python files, its sample/model datasets in `data/`, and its results and figures in `outputs/`.

`analysis.py` contains each task’s sampling, statistics, Welch test and chart. `context.py` and `sensitivity.py` contain its supplementary analyses.

`shared/` contains source cleaning, shared table loading and file paths. Root `data/` holds shared source datasets and combined team/match tables; root `outputs/` holds combined comparisons and the shared data audit.

## Run

Python 3.9–3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Individual primary analyses:

```bash
python -m src.goals.analysis
python -m src.yellow_cards.analysis
python -m src.shots.analysis
python -m src.saves.analysis
```

Use `python -m src.<task>.context` for a task's contextual comparison. Run `python run_all.py` to refresh all results, including the combined Holm adjustment.

Sources: [FIFA](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics) and [The Stats Don't Lie](https://www.thestatsdontlie.com/football/world-cup-2026/).
