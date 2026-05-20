from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_markdown_report(run_dir: Path, metrics: dict, warnings: list[str]) -> Path:
    lines = [
        '# Evidence Tester Report',
        '',
        f"Run ID: `{metrics.get('run_id', 'unknown')}`",
        '',
        '## Purpose',
        '',
        'This pack is evidence for a research hypothesis. It is not a trading signal, order-routing system, or financial advice.',
        '',
        '## Hypothesis',
        '',
        'U.S. net liquidity impulse, defined as Fed assets minus Treasury General Account minus ON RRP, is tested as a leading variable for BTC and Nasdaq/QQQ returns.',
        '',
        '## Headline metrics',
        '',
    ]
    for target, m in metrics.get('targets', {}).items():
        lines.append(f'### {target}')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---:|')
        for k, v in m.items():
            if isinstance(v, float):
                lines.append(f'| {k} | {v:.6f} |')
            else:
                lines.append(f'| {k} | {v} |')
        lines.append('')
    if warnings:
        lines.extend(['## Warnings', ''])
        for w in warnings:
            lines.append(f'- {w}')
        lines.append('')
    lines.extend([
        '## Interpretation discipline',
        '',
        '- Treat positive results as candidates for deeper validation, not permission to trade.',
        '- Reject results that rely on one short sub-period, one horizon, or unaligned release timing.',
        '- Promote a hypothesis only after out-of-sample and live-shadow evidence agree with the historical result.',
    ])
    path = run_dir / 'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
