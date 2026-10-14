"""Compute Phy-Q scores per template from agent results and compare to other agents."""
from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def load_results_from_log(log_path: Path) -> pd.DataFrame:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(
        r"\[CSV\] #(\d+) ([^\|]+) \| (Win|Lose) \| Score: (\d+) \| Plan: ([^|]+) \|"
        r"\s*(UNSOLVABLE)?\s*\| Mode: (train|test) \| Attempts: (\d+)(?: ABANDONED)?"
    )
    rows = []
    for m in pat.finditer(log):
        idx, level_name, result, score, plan, unsolv_flag, mode, attempts = m.groups()
        tmpl = int(re.search(r"_t(\d+)_", level_name).group(1))
        var = int(re.search(r"_(\d+)\.xml", level_name).group(1))
        scenario_match = re.match(r"(\w+)_t", level_name.strip())
        scenario = scenario_match.group(1) if scenario_match else "unknown"
        rows.append(
            {
                "level_index": int(idx),
                "level_name": level_name.strip(),
                "scenario": scenario,
                "template": tmpl,
                "variation": var,
                "mode": mode,
                "result": result,
                "score": int(score),
                "plan_source": plan.strip(),
                "unsolvable": bool(unsolv_flag),
                "attempts_used": int(attempts),
                "abandoned": "ABANDONED" in m.group(0),
            }
        )
    return pd.DataFrame(rows)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def outcome_counts(sub: pd.DataFrame) -> dict:
    """Mutually exclusive Win / Loss / Unsolved (planner unsolvable, lost)."""
    n = len(sub)
    if n == 0:
        return {
            "n": 0,
            "win": 0,
            "loss": 0,
            "unsolved": 0,
            "win_pct": 0.0,
            "loss_pct": 0.0,
            "unsolved_pct": 0.0,
        }
    is_win = sub["result"] == "Win"
    is_lose = sub["result"] == "Lose"
    unsolv = _as_bool(sub["unsolvable"])
    win = int(is_win.sum())
    unsolved = int((is_lose & unsolv).sum())
    loss = int((is_lose & ~unsolv).sum())
    return {
        "n": n,
        "win": win,
        "loss": loss,
        "unsolved": unsolved,
        "win_pct": win / n * 100,
        "loss_pct": loss / n * 100,
        "unsolved_pct": unsolved / n * 100,
    }


def _fmt_wlu(stats: dict) -> str:
    return (
        f"{stats['win']}/{stats['loss']}/{stats['unsolved']} "
        f"({stats['win_pct']:.0f}/{stats['loss_pct']:.0f}/{stats['unsolved_pct']:.0f}%)"
    )


def load_baseline_comparison(baseline_path: Path, scenario: str, templates: list[int]) -> tuple[dict, dict]:
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["scenario"] == scenario]

    baseline_by_template: dict[str, dict[int, float]] = {}
    baseline_overall: dict[str, float] = {}

    for agent in baseline["agent"].unique():
        agent_df = baseline[baseline["agent"] == agent]
        template_rates = {}
        for template in templates:
            t_data = agent_df[agent_df["template"] == template]["pass_rate"]
            template_rates[template] = t_data.mean() * 100 if len(t_data) else np.nan
        baseline_by_template[agent] = template_rates
        baseline_overall[agent] = agent_df["pass_rate"].mean() * 100

    return baseline_by_template, baseline_overall


def generate_report(
    df: pd.DataFrame,
    baseline_path: Path,
    source_label: str,
) -> str:
    out = StringIO()
    write = out.write

    df = df.copy()
    if "unsolvable" not in df.columns:
        df["unsolvable"] = False
    df["win"] = (df["result"] == "Win").astype(int)
    scenario = df["scenario"].mode().iloc[0] if len(df) else "unknown"
    templates = sorted(df["template"].unique())
    all_results = []

    write("=" * 80 + "\n")
    write(f"PHY-Q SCORES BY TEMPLATE - PDDL AGENT ({scenario})\n")
    write("=" * 80 + "\n")
    write(f"Source: {source_label} ({len(df)} levels)\n")
    write(
        "Outcomes: Win = level won; Loss = lost (planner solvable); "
        "Unsolved = lost with ENHSP unsolvable.\n\n"
    )

    total_train = df[df["mode"] == "train"]
    total_test = df[df["mode"] == "test"]
    overall_train = outcome_counts(total_train)
    overall_test = outcome_counts(total_test)
    overall_all = outcome_counts(df)

    write("## Outcome Breakdown (Win / Loss / Unsolved)\n")
    write("-" * 88 + "\n")
    write(
        f"{'Split':^10} | {'Win':^8} | {'Loss':^8} | {'Unsolved':^10} | "
        f"{'Win%':^8} | {'Loss%':^8} | {'Unsol%':^8} | {'N':^6}\n"
    )
    write("-" * 88 + "\n")
    for label, stats in [("Train", overall_train), ("Test", overall_test), ("All", overall_all)]:
        write(
            f"{label:^10} | {stats['win']:^8} | {stats['loss']:^8} | {stats['unsolved']:^10} | "
            f"{stats['win_pct']:^7.1f}% | {stats['loss_pct']:^7.1f}% | "
            f"{stats['unsolved_pct']:^7.1f}% | {stats['n']:^6}\n"
        )
    write("=" * 88 + "\n\n")

    write("## Per-Template Performance\n")
    write("-" * 120 + "\n")
    write(
        f"{'Tpl':^5} | {'Train W/L/U':^28} | {'Test W/L/U':^28} | "
        f"{'Tr Win%':^8} | {'Te Win%':^8} | {'Gap':^8} | {'TrN':^5} | {'TeN':^5}\n"
    )
    write("-" * 120 + "\n")

    for template in templates:
        t_df = df[df["template"] == template]
        train_df = t_df[t_df["mode"] == "train"]
        test_df = t_df[t_df["mode"] == "test"]
        tr = outcome_counts(train_df)
        te = outcome_counts(test_df)
        train_pct = tr["win_pct"]
        test_pct = te["win_pct"]
        gen_gap = train_pct - test_pct
        all_results.append(
            {
                "template": template,
                "train_pct": train_pct,
                "test_pct": test_pct,
                "gen_gap": gen_gap,
                "train_n": tr["n"],
                "test_n": te["n"],
                "train_win": tr["win"],
                "train_loss": tr["loss"],
                "train_unsolved": tr["unsolved"],
                "test_win": te["win"],
                "test_loss": te["loss"],
                "test_unsolved": te["unsolved"],
            }
        )
        write(
            f"{template:^5} | {_fmt_wlu(tr):^28} | {_fmt_wlu(te):^28} | "
            f"{train_pct:^7.1f}% | {test_pct:^7.1f}% | {gen_gap:^+7.1f}% | "
            f"{tr['n']:^5} | {te['n']:^5}\n"
        )

    write("-" * 120 + "\n")
    overall_train_pct = overall_train["win_pct"]
    overall_test_pct = overall_test["win_pct"]
    overall_gen_gap = overall_train_pct - overall_test_pct
    overall_accuracy = overall_all["win_pct"]
    write(
        f"{'ALL':^5} | {_fmt_wlu(overall_train):^28} | {_fmt_wlu(overall_test):^28} | "
        f"{overall_train_pct:^7.1f}% | {overall_test_pct:^7.1f}% | {overall_gen_gap:^+7.1f}% | "
        f"{overall_train['n']:^5} | {overall_test['n']:^5}\n"
    )
    write("=" * 120 + "\n")

    write("\n## PHY-Q SCORE SUMMARY (PDDL Agent)\n")
    write("=" * 80 + "\n")
    write(f"Overall Accuracy (all levels):     {overall_accuracy:.1f}%\n")
    write(
        f"Train:  Win {overall_train['win_pct']:.1f}% | "
        f"Loss {overall_train['loss_pct']:.1f}% | "
        f"Unsolved {overall_train['unsolved_pct']:.1f}% "
        f"({overall_train['n']} levels)\n"
    )
    write(
        f"Test:   Win {overall_test['win_pct']:.1f}% | "
        f"Loss {overall_test['loss_pct']:.1f}% | "
        f"Unsolved {overall_test['unsolved_pct']:.1f}% "
        f"({overall_test['n']} levels)\n"
    )
    write(f"Generalization Gap (train win% - test win%): {overall_gen_gap:+.1f}%\n")

    write("\n")
    write("=" * 80 + "\n")
    write("COMPARISON WITH OTHER AGENTS (Phy-Q Benchmark)\n")
    write("=" * 80 + "\n")

    if baseline_path.exists():
        baseline_by_template, baseline_overall = load_baseline_comparison(
            baseline_path, scenario, templates
        )

        write("\n## Pass Rate by Template (All Agents)\n")
        write("-" * 100 + "\n")
        header_parts = ["Agent".ljust(20)] + [f"T{t}".center(10) for t in templates] + ["Overall".center(10)]
        write(" | ".join(header_parts) + "\n")
        write("-" * 100 + "\n")

        pddl_row = ["PDDL Agent".ljust(20)]
        for r in all_results:
            combined = (r["train_pct"] * r["train_n"] + r["test_pct"] * r["test_n"]) / (
                r["train_n"] + r["test_n"]
            )
            pddl_row.append(f"{combined:.1f}%".center(10))
        pddl_row.append(f"{overall_accuracy:.1f}%".center(10))
        write(" | ".join(pddl_row) + "\n")

        for agent in sorted(baseline_by_template.keys()):
            row = [agent[:20].ljust(20)]
            for template in templates:
                rate = baseline_by_template[agent].get(template, np.nan)
                row.append(f"{rate:.1f}%".center(10) if pd.notna(rate) else "N/A".center(10))
            row.append(f"{baseline_overall[agent]:.1f}%".center(10))
            write(" | ".join(row) + "\n")
        write("-" * 100 + "\n")

        write(f"\n## Overall Performance Ranking ({scenario} scenario)\n")
        write("-" * 60 + "\n")
        all_agents_overall = {"PDDL Agent": overall_accuracy}
        all_agents_overall.update(baseline_overall)
        ranked = sorted(all_agents_overall.items(), key=lambda x: x[1], reverse=True)
        write(f"{'Rank':^6} | {'Agent':<25} | {'Pass Rate':^12}\n")
        write("-" * 60 + "\n")
        for i, (agent, rate) in enumerate(ranked, 1):
            marker = " <-- Our Agent" if agent == "PDDL Agent" else ""
            write(f"{i:^6} | {agent:<25} | {rate:^12.1f}%{marker}\n")
        write("-" * 60 + "\n")
        pddl_rank = next(i for i, (a, _) in enumerate(ranked, 1) if a == "PDDL Agent")
        write(f"\nPDDL Agent Rank: {pddl_rank}/{len(ranked)}\n")

        write("\n## Comparison vs Key Baselines\n")
        write("-" * 60 + "\n")
        for baseline_name in ["Human", "Datalab", "Eagle's Wing", "Bambirds", "Random Agent"]:
            if baseline_name in baseline_overall:
                diff = overall_accuracy - baseline_overall[baseline_name]
                status = "BETTER" if diff > 0 else ("SAME" if diff == 0 else "WORSE")
                write(f"vs {baseline_name:<20}: {diff:+.1f}% ({status})\n")
        write("=" * 80 + "\n")

        if "Human" in baseline_by_template:
            write("\n## Per-Template Comparison: PDDL Agent vs Human\n")
            write("-" * 80 + "\n")
            write(f"{'Template':^10} | {'PDDL':^12} | {'Human':^12} | {'Diff':^12} | {'Status':^10}\n")
            write("-" * 80 + "\n")
            for r in all_results:
                t = r["template"]
                pddl_combined = (r["train_pct"] * r["train_n"] + r["test_pct"] * r["test_n"]) / (
                    r["train_n"] + r["test_n"]
                )
                human_rate = baseline_by_template["Human"].get(t, np.nan)
                if pd.notna(human_rate):
                    diff = pddl_combined - human_rate
                    status = "BETTER" if diff > 0 else ("SAME" if diff == 0 else "WORSE")
                    write(
                        f"{t:^10} | {pddl_combined:^12.1f}% | {human_rate:^12.1f}% | "
                        f"{diff:^+12.1f}% | {status:^10}\n"
                    )
                else:
                    write(
                        f"{t:^10} | {pddl_combined:^12.1f}% | {'N/A':^12} | "
                        f"{'N/A':^12} | {'N/A':^10}\n"
                    )
            write("-" * 80 + "\n")
    else:
        write(f"\nWarning: baseline file not found: {baseline_path}. Skipping comparison.\n")

    write("\n")
    write("=" * 80 + "\n")
    write("SUMMARY EXPORT\n")
    write("=" * 80 + "\n")
    summary_data = []
    for r in all_results:
        summary_data.append(
            {
                "Template": r["template"],
                "Train Win": r["train_win"],
                "Train Loss": r["train_loss"],
                "Train Unsolved": r["train_unsolved"],
                "Train Win%": f"{r['train_pct']:.1f}%",
                "Test Win": r["test_win"],
                "Test Loss": r["test_loss"],
                "Test Unsolved": r["test_unsolved"],
                "Test Win%": f"{r['test_pct']:.1f}%",
                "Gen Gap": f"{r['gen_gap']:+.1f}%",
                "Train N": r["train_n"],
                "Test N": r["test_n"],
            }
        )
    summary_df = pd.DataFrame(summary_data)
    write(summary_df.to_string(index=False) + "\n")

    pd.DataFrame(all_results).to_csv("phyq_scores_by_template.csv", index=False)
    write("\nSaved detailed scores to: phyq_scores_by_template.csv\n")

    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Phy-Q scores and compare to baselines.")
    parser.add_argument(
        "--results",
        default="pddl_agent_results.csv",
        help="PDDL agent results CSV (default: pddl_agent_results.csv)",
    )
    parser.add_argument(
        "--log",
        help="Optional run log to extract results from (overrides --results when set)",
    )
    parser.add_argument(
        "--baseline",
        default="baseline_comparison.csv",
        help="Baseline pass rates CSV (default: baseline_comparison.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="phyq_scores_report.txt",
        help="Write full report to this file (default: phyq_scores_report.txt)",
    )
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            print(f"Error: log not found: {log_path}", file=sys.stderr)
            sys.exit(1)
        df = load_results_from_log(log_path)
        source_label = str(log_path)
    else:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"Error: results not found: {results_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(results_path)
        source_label = str(results_path)

    report = generate_report(df, Path(args.baseline), source_label)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
