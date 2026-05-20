from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _format_value(v) -> str:
    if isinstance(v, float):
        return f'{v:.6f}'
    if isinstance(v, dict):
        return '`' + json.dumps(v, sort_keys=True) + '`'
    return str(v)


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
        '## Validation summary',
        '',
        '| Target | Status | Main reasons |',
        '|---|---|---|',
    ]
    for target, m in metrics.get('targets', {}).items():
        validation = m.get('validation') or {}
        reasons = validation.get('reasons') or []
        reason_text = '<br>'.join(reasons) if reasons else 'Passed conservative evidence gate.'
        lines.append(f"| {target} | {validation.get('status', 'unknown')} | {reason_text} |")
    lines.extend(['', '## Headline metrics', ''])
    for target, m in metrics.get('targets', {}).items():
        lines.append(f'### {target}')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---:|')
        for k, v in m.items():
            lines.append(f'| {k} | {_format_value(v)} |')
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
        '- Reject OOS results that do not beat simple baselines such as always-long or always-short.',
        '- Promote a hypothesis only after out-of-sample and live-shadow evidence agree with the historical result.',
    ])
    path = run_dir / 'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
