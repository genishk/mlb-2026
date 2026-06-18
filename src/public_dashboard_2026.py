"""
MLB Analytics Pro — 2026 Season Public Performance Tracker

Responsive Streamlit dashboard for tracking moneyline + totals model
performance across the full season. Data is loaded from both the active
matched folder and backup folders to ensure complete season coverage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
MATCHED_DIR = PROJECT_ROOT / 'src' / 'odds' / 'data' / 'matched'
BACKUP_DATA_DIR = PROJECT_ROOT / 'src' / 'models' / 'backup_data'
TOTALS_BACKUP_DATA_DIR = PROJECT_ROOT / 'src' / 'models' / 'totals_backup_data'
RECORDS_DIR = PROJECT_ROOT / 'data' / 'records'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PublicDashboard2026')

st.set_page_config(
    page_title="MLB Analytics Pro — 2026 Performance",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════
# RESPONSIVE CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ---------- Fonts & Base ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---------- Metric cards ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1rem 0;
}
.m-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    transition: box-shadow .2s;
}
.m-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.08); }
.m-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.2;
    color: #0f172a;
}
.m-card .value.positive { color: #059669; }
.m-card .value.negative { color: #dc2626; }
.m-card .label {
    font-size: .78rem;
    font-weight: 500;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

/* ---------- Section heading ---------- */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin: 2rem 0 .5rem;
    padding-bottom: 6px;
    border-bottom: 2px solid #e5e7eb;
}

/* ---------- CTA row ---------- */
.cta-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: .8rem 0 1.5rem;
}
.cta-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: .85rem;
    font-weight: 600;
    text-decoration: none !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    box-shadow: 0 2px 8px rgba(37,99,235,.3);
    transition: all .2s;
}
.cta-btn:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    box-shadow: 0 4px 14px rgba(37,99,235,.4);
    transform: translateY(-1px);
    color: #ffffff !important;
}
.cta-btn.secondary {
    background: linear-gradient(135deg, #475569, #334155);
    box-shadow: 0 2px 8px rgba(71,85,105,.3);
}
.cta-btn.secondary:hover {
    background: linear-gradient(135deg, #334155, #1e293b);
    box-shadow: 0 4px 14px rgba(71,85,105,.4);
}

/* ---------- Podium cards ---------- */
.podium-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0 1.5rem;
}
.podium-card {
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
.podium-card.gold   { border-color: #fbbf24; background: linear-gradient(135deg, #fffbeb, #fef3c7); }
.podium-card.silver { border-color: #94a3b8; background: linear-gradient(135deg, #f8fafc, #e2e8f0); }
.podium-card.bronze { border-color: #d97706; background: linear-gradient(135deg, #fffbeb, #fed7aa); }
.podium-rank {
    font-size: 1.4rem;
    margin-bottom: 4px;
}
.podium-name {
    font-size: .9rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
}
.podium-stat {
    font-size: .78rem;
    color: #475569;
    line-height: 1.5;
}
.podium-roi {
    font-size: 1.3rem;
    font-weight: 800;
    color: #059669;
}

/* ---------- Highlight strip ---------- */
.highlight-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0;
}
.hl-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.hl-card .hl-val {
    font-size: 1.4rem;
    font-weight: 800;
    color: #1e40af;
}
.hl-card .hl-lbl {
    font-size: .72rem;
    color: #3b82f6;
    font-weight: 500;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: .03em;
}

/* ---------- Footer ---------- */
.footer-text {
    text-align: center;
    font-size: .75rem;
    color: #94a3b8;
    padding: 2rem 0 1rem;
}

/* ---------- Mobile (<=768px) ---------- */
@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .m-card { padding: 14px 10px; }
    .m-card .value { font-size: 1.35rem; }
    .m-card .label { font-size: .7rem; }
    .section-title { font-size: 1.15rem; }
    .cta-row { gap: 6px; }
    .cta-btn { padding: 8px 14px; font-size: .78rem; }
    .podium-grid { grid-template-columns: 1fr; gap: 8px; }
    .podium-roi { font-size: 1.1rem; }
    .highlight-strip { grid-template-columns: 1fr; gap: 8px; }
    .hl-card .hl-val { font-size: 1.15rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def _collect_active_files(pattern_prefix: str) -> list:
    """Gather active JSON files from matched dir + all backup_data sub-dirs."""
    files = []

    for f in MATCHED_DIR.glob(f'{pattern_prefix}_*_active.json'):
        files.append(f)

    for backup_dir in [BACKUP_DATA_DIR, TOTALS_BACKUP_DATA_DIR]:
        if backup_dir.exists():
            for sub in backup_dir.iterdir():
                if sub.is_dir():
                    for f in sub.glob(f'{pattern_prefix}_*_active.json'):
                        files.append(f)

    return files


def _extract_file_date(fname: str) -> str:
    """Extract YYYYMMDD from filename."""
    m = re.search(r'(\d{8})_\d{6}', fname)
    return m.group(1) if m else ''


def _load_predictions(pattern_prefix: str, exclude_today: bool = True) -> pd.DataFrame:
    """Load, deduplicate, and return predictions from all active files."""
    all_files = _collect_active_files(pattern_prefix)
    if not all_files:
        return pd.DataFrame()

    today = datetime.now().strftime('%Y%m%d')
    today_fmt = f"{today[:4]}-{today[4:6]}-{today[6:8]}"

    files_by_date: dict[str, Path] = {}
    for f in all_files:
        fd = _extract_file_date(f.name)
        if not fd:
            continue
        if exclude_today and fd == today:
            continue
        if fd not in files_by_date or f.stat().st_mtime > files_by_date[fd].stat().st_mtime:
            files_by_date[fd] = f

    covered_dates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in files_by_date}

    all_preds: list[dict] = []
    for fd, fpath in sorted(files_by_date.items()):
        fd_fmt = f"{fd[:4]}-{fd[4:6]}-{fd[6:8]}"
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            for rec in data:
                rd = rec.get('date', '')
                if exclude_today and rd == today_fmt:
                    continue
                if rd == fd_fmt:
                    all_preds.append(rec)
                elif rd not in covered_dates:
                    all_preds.append(rec)
        except Exception as e:
            logger.error(f"Failed to load {fpath.name}: {e}")

    if not all_preds:
        return pd.DataFrame()

    df = pd.DataFrame(all_preds)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
    return df


def _load_shadow_predictions(pattern_prefix: str) -> pd.DataFrame:
    """Load shadow predictions from matched dir only (no backups).
    Uses all available shadow files; if >7 unique dates, keep most recent 7."""
    files = list(MATCHED_DIR.glob(f'{pattern_prefix}_*_shadow.json'))
    if not files:
        return pd.DataFrame()

    today = datetime.now().strftime('%Y%m%d')
    today_fmt = f"{today[:4]}-{today[4:6]}-{today[6:8]}"

    files_by_date: dict[str, Path] = {}
    for f in files:
        fd = _extract_file_date(f.name)
        if not fd or fd == today:
            continue
        if fd not in files_by_date or f.stat().st_mtime > files_by_date[fd].stat().st_mtime:
            files_by_date[fd] = f

    if len(files_by_date) > 7:
        recent_dates = sorted(files_by_date.keys(), reverse=True)[:7]
        files_by_date = {d: files_by_date[d] for d in recent_dates}

    covered_dates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in files_by_date}

    all_preds: list[dict] = []
    for fd, fpath in sorted(files_by_date.items()):
        fd_fmt = f"{fd[:4]}-{fd[4:6]}-{fd[6:8]}"
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            for rec in data:
                rd = rec.get('date', '')
                if rd == today_fmt:
                    continue
                if rd == fd_fmt:
                    all_preds.append(rec)
                elif rd not in covered_dates:
                    all_preds.append(rec)
        except Exception as e:
            logger.error(f"Failed to load shadow {fpath.name}: {e}")

    if not all_preds:
        return pd.DataFrame()

    df = pd.DataFrame(all_preds)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
    return df


def _load_game_results() -> pd.DataFrame:
    record_files = sorted(RECORDS_DIR.glob('mlb_historical_records_*.json'))
    if not record_files:
        return pd.DataFrame()
    try:
        with open(record_files[-1], 'r', encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df[df['abstract_game_state'] == 'Final'].copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def _match_ml(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or results.empty:
        return pd.DataFrame()
    idx = {}
    for _, r in results.iterrows():
        idx[f"{r['date']}_{r['home_team_name']}_{r['away_team_name']}"] = r
    rows = []
    for _, p in preds.iterrows():
        key = f"{p['date'].date()}_{p['home_team']}_{p['away_team']}"
        if key in idx:
            res = idx[key]
            if pd.isna(res.get('home_win')):
                continue
            d = p.to_dict()
            d['actual_home_win'] = int(res['home_win'])
            d['actual_home_score'] = int(res.get('home_score', 0))
            d['actual_away_score'] = int(res.get('away_score', 0))
            rows.append(d)
    return pd.DataFrame(rows)


def _match_totals(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or results.empty:
        return pd.DataFrame()
    idx = {}
    for _, r in results.iterrows():
        idx[f"{r['date']}_{r['home_team_name']}_{r['away_team_name']}"] = r
    rows = []
    for _, p in preds.iterrows():
        key = f"{p['date'].date()}_{p['home_team']}_{p['away_team']}"
        if key in idx:
            res = idx[key]
            hs = int(res.get('home_score', 0))
            aws = int(res.get('away_score', 0))
            if hs == 0 and aws == 0:
                continue
            d = p.to_dict()
            d['actual_total'] = hs + aws
            rows.append(d)
    return pd.DataFrame(rows)


def _payout(stake: float, odds: float) -> float:
    return stake + (stake * odds / 100) if odds > 0 else stake + (stake * 100 / (-odds))


def _calc_ml_roi(matched: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in matched.columns if c.endswith('_probability')]
    excluded = {'win_probability', 'home_team_probability_odds',
                'away_team_probability_odds', 'predicted_winner_probability_odds'}
    models = sorted(c.replace('_probability', '') for c in prob_cols if c not in excluded)

    rows = []
    for model in models:
        pc = f'{model}_probability'
        if pc not in matched.columns:
            continue
        for _, r in matched.iterrows():
            hp = r[pc]
            ho = r.get('home_team_odds')
            ao = r.get('away_team_odds')
            hw = r['actual_home_win']
            if pd.isna(ho) or pd.isna(ao) or pd.isna(hp):
                continue
            home_bet = hp > 0.5
            if home_bet:
                profit = (_payout(100, ho) - 100) if hw == 1 else -100
                odds = ho
                prob = hp
            else:
                profit = (_payout(100, ao) - 100) if hw == 0 else -100
                odds = ao
                prob = 1 - hp
            rows.append({
                'model': model, 'date': r['date'],
                'home_team': r['home_team'], 'away_team': r['away_team'],
                'bet_odds': odds, 'bet_probability': prob,
                'actual_profit': profit, 'won': profit > 0,
            })
    return pd.DataFrame(rows)


def _calc_totals_roi(matched: pd.DataFrame) -> pd.DataFrame:
    excluded = {'predicted_total', 'ensemble_total', 'total_line', 'actual_total'}
    tcols = [c for c in matched.columns if c.endswith('_total') and c not in excluded]
    models = sorted(c.replace('_total', '') for c in tcols)

    rows = []
    for model in ['ensemble'] + models:
        pc = 'ensemble_total' if model == 'ensemble' else f'{model}_total'
        if pc not in matched.columns:
            continue
        for _, r in matched.iterrows():
            pt = r[pc]
            tl = r.get('total_line')
            oo = r.get('over_odds')
            uo = r.get('under_odds')
            at = r['actual_total']
            if pd.isna(tl) or pd.isna(oo) or pd.isna(uo) or pd.isna(pt):
                continue
            direction = 'OVER' if pt > tl else 'UNDER'
            odds = oo if direction == 'OVER' else uo
            if at == tl:
                continue
            won = (at > tl) if direction == 'OVER' else (at < tl)
            profit = (_payout(100, odds) - 100) if won else -100
            rows.append({
                'model': model, 'date': r['date'],
                'home_team': r['home_team'], 'away_team': r['away_team'],
                'direction': direction, 'total_line': tl,
                'predicted_total': pt, 'actual_total': at,
                'bet_odds': odds, 'actual_profit': profit, 'won': won,
                'margin_abs': abs(pt - tl),
            })
    return pd.DataFrame(rows)


def _summarise(betting: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model summary: WR, ROI, Profit."""
    rows = []
    for model in sorted(betting['model'].unique()):
        mb = betting[betting['model'] == model]
        n = len(mb)
        w = int(mb['won'].sum())
        profit = mb['actual_profit'].sum()
        rows.append({
            'Model': model.upper(),
            'Bets': n, 'Wins': w, 'Losses': n - w,
            'Win Rate (%)': round(w / n * 100, 2) if n else 0,
            'Profit ($)': round(profit, 2),
            'ROI (%)': round(profit / (n * 100) * 100, 2) if n else 0,
        })
    return pd.DataFrame(rows).sort_values('ROI (%)', ascending=False).reset_index(drop=True)


def _daily_breakdown(betting: pd.DataFrame, model: str) -> pd.DataFrame:
    mb = betting[betting['model'] == model].copy()
    if mb.empty:
        return pd.DataFrame()
    mb['date_str'] = mb['date'].dt.strftime('%Y-%m-%d')
    grp = mb.groupby('date_str').agg(
        bets=('won', 'count'),
        wins=('won', 'sum'),
        profit=('actual_profit', 'sum'),
    ).reset_index()
    grp['win_rate'] = (grp['wins'] / grp['bets'] * 100).round(1)
    grp['roi'] = (grp['profit'] / (grp['bets'] * 100) * 100).round(2)
    grp['cum_profit'] = grp['profit'].cumsum().round(2)
    grp.columns = ['Date', 'Bets', 'Wins', 'Profit ($)', 'Win Rate (%)', 'ROI (%)', 'Cum P/L ($)']
    return grp


# ═══════════════════════════════════════════════════════════════════════
# SEGMENT ANALYSIS (moneyline — same boundaries as internal dashboard)
# ═══════════════════════════════════════════════════════════════════════

def _ml_segment_analysis(betting: pd.DataFrame, model: str) -> dict:
    """Moneyline segment analysis — boundaries match mlb_model_performance_dashboard exactly."""
    mb = betting[betting['model'] == model].copy()
    if mb.empty:
        return {}

    mb['implied_prob'] = mb['bet_odds'].apply(
        lambda o: 100 / (o + 100) if o > 0 else (-o) / (-o + 100))
    mb['predicted_roi'] = (mb['bet_probability'] - mb['implied_prob']) * 100
    mb['confidence'] = (mb['bet_probability'] - 0.5).abs()
    mb['divergence'] = (mb['bet_probability'] - mb['implied_prob']) * 100

    def _kelly(row):
        try:
            o = row['bet_odds']
            dec = (o / 100) + 1 if o > 0 else (100 / abs(o)) + 1
            k = (row['bet_probability'] * dec - 1) / (dec - 1)
            return max(0, k * 100)
        except Exception:
            return 0
    mb['kelly'] = mb.apply(_kelly, axis=1)

    def _calc(sub):
        n = len(sub)
        if n == 0:
            return {'Bets': 0, 'Wins': 0, 'WR (%)': 0, 'ROI (%)': 0, 'Profit ($)': 0}
        w = int(sub['won'].sum())
        p = sub['actual_profit'].sum()
        return {
            'Bets': n, 'Wins': w,
            'WR (%)': round(w / n * 100, 1),
            'ROI (%)': round(p / (n * 100) * 100, 2),
            'Profit ($)': round(p, 1),
        }

    def _classify(series, classify_fn):
        groups: dict[str, list] = {}
        for idx, val in series.items():
            lbl = classify_fn(val)
            if lbl:
                groups.setdefault(lbl, []).append(idx)
        return {lbl: _calc(mb.loc[idxs]) for lbl, idxs in groups.items()}

    seg = {}

    def _pred_roi(v):
        if v < -20:   return 'Very Negative (<-20%)'
        if v < 0:     return 'Negative (-20% ~ 0%)'
        if v < 20:    return 'Positive (0% ~ 20%)'
        if v < 40:    return 'Very Positive A (20% ~ 40%)'
        if v < 60:    return 'Very Positive B (40% ~ 60%)'
        if v < 100:   return 'Extremely Positive A (60% ~ 100%)'
        return 'Extremely Positive B (>100%)'
    seg['Predicted ROI'] = _classify(mb['predicted_roi'], _pred_roi)

    def _odds(o):
        if o < -200:      return 'Heavy Favorite (< -200)'
        if o < -120:      return 'Favorite (-200 ~ -120)'
        if o <= 120:      return "Pick'em (-120 ~ +120)"
        if o <= 150:      return 'Underdog (+120 ~ +150)'
        if o <= 300:      return 'Strong Underdog (+150 ~ +300)'
        return 'Heavy Underdog (> +300)'
    seg['Odds'] = _classify(mb['bet_odds'], _odds)

    def _conf(c):
        if c < 0.05:  return 'Low (0 ~ 0.05)'
        if c < 0.15:  return 'Medium (0.05 ~ 0.15)'
        if c < 0.25:  return 'High (0.15 ~ 0.25)'
        return 'Very High (> 0.25)'
    seg['Confidence'] = _classify(mb['confidence'], _conf)

    def _div(d):
        if d < -10:   return 'Much Pessimistic (< -10%)'
        if d < -5:    return 'Slightly Pessimistic (-10% ~ -5%)'
        if d < 5:     return 'Market Aligned (-5% ~ +5%)'
        if d < 10:    return 'Slightly Optimistic (+5% ~ +10%)'
        return 'Much Optimistic (> +10%)'
    seg['Market Divergence'] = _classify(mb['divergence'], _div)

    def _kel(k):
        if k <= 0:    return 'No Bet (<=0%)'
        if k <= 5:    return 'Low (0% ~ 5%)'
        if k <= 15:   return 'Medium (5% ~ 15%)'
        if k <= 25:   return 'High (15% ~ 25%)'
        if k <= 60:   return 'V.High (25% ~ 60%)'
        return 'Extreme (> 60%)'
    seg['Kelly Criterion'] = _classify(mb['kelly'], _kel)

    return seg


def _totals_segment_analysis(betting: pd.DataFrame, model: str,
                             matched_raw: pd.DataFrame = None) -> dict:
    """Totals segment analysis — boundaries match mlb_model_performance_dashboard exactly."""
    mb = betting[betting['model'] == model].copy()
    if mb.empty:
        return {}

    excluded_cols = {'predicted_total', 'ensemble_total', 'total_line', 'actual_total'}

    consensus_map = {}
    if matched_raw is not None and not matched_raw.empty:
        tcols = [c for c in matched_raw.columns
                 if c.endswith('_total') and c not in excluded_cols]
        for _, row in matched_raw.iterrows():
            tl = row.get('total_line')
            if pd.isna(tl):
                continue
            over_count = 0
            total_count = 0
            for col in tcols:
                if not pd.isna(row.get(col)):
                    total_count += 1
                    if row[col] > tl:
                        over_count += 1
            if total_count > 0:
                key = (row['date'], row['home_team'], row['away_team'])
                consensus_map[key] = {
                    'over_pct': over_count / total_count * 100,
                    'under_pct': (total_count - over_count) / total_count * 100,
                }
    else:
        for (d, ht, at), grp in betting.groupby(['date', 'home_team', 'away_team']):
            over_n = (grp['direction'] == 'OVER').sum()
            total_n = len(grp)
            if total_n > 0:
                consensus_map[(d, ht, at)] = {
                    'over_pct': over_n / total_n * 100,
                    'under_pct': (total_n - over_n) / total_n * 100,
                }

    mb['consensus'] = mb.apply(
        lambda r: consensus_map.get(
            (r['date'], r['home_team'], r['away_team']),
            {'over_pct': 50, 'under_pct': 50}
        )['over_pct' if r['direction'] == 'OVER' else 'under_pct'],
        axis=1)

    def _calc(sub):
        n = len(sub)
        if n == 0:
            return {'Bets': 0, 'Wins': 0, 'WR (%)': 0, 'ROI (%)': 0, 'Profit ($)': 0}
        w = int(sub['won'].sum())
        p = sub['actual_profit'].sum()
        return {
            'Bets': n, 'Wins': w,
            'WR (%)': round(w / n * 100, 1),
            'ROI (%)': round(p / (n * 100) * 100, 2),
            'Profit ($)': round(p, 1),
        }

    def _classify(series, classify_fn):
        groups: dict[str, list] = {}
        for idx, val in series.items():
            lbl = classify_fn(val)
            if lbl:
                groups.setdefault(lbl, []).append(idx)
        return {lbl: _calc(mb.loc[idxs]) for lbl, idxs in groups.items()}

    seg = {}

    def _margin(m):
        if m < 0.5:  return '0 - 0.5'
        if m < 1.0:  return '0.5 - 1.0'
        if m < 2.0:  return '1.0 - 2.0'
        if m < 3.0:  return '2.0 - 3.0'
        return '3.0+'
    seg['Margin from Line'] = _classify(mb['margin_abs'], _margin)

    def _odds(o):
        if o < -130:      return 'Heavy Juice (< -130)'
        if o < -115:      return 'Moderate Juice (-130 ~ -115)'
        if o < -105:      return 'Standard (-115 ~ -105)'
        if o <= 100:      return 'Light Juice (-105 ~ +100)'
        return 'Plus Odds (+100+)'
    seg['Odds'] = _classify(mb['bet_odds'], _odds)

    def _cons(c):
        if c < 60:  return 'Low (< 60%)'
        if c < 75:  return 'Moderate (60% - 75%)'
        if c < 90:  return 'High (75% - 90%)'
        return 'Very High (90%+)'
    seg['Model Consensus'] = _classify(mb['consensus'], _cons)

    def _line(tl):
        if tl <= 7.0:  return 'Low (≤ 7.0)'
        if tl <= 8.0:  return 'Medium-Low (7.0 - 8.0)'
        if tl <= 9.0:  return 'Medium (8.0 - 9.0)'
        if tl <= 10.0: return 'Medium-High (9.0 - 10.0)'
        return 'High (10.0+)'
    seg['Line Level'] = _classify(mb['total_line'], _line)

    seg['Direction'] = {}
    for d in ['OVER', 'UNDER']:
        seg['Direction'][d] = _calc(mb[mb['direction'] == d])

    return seg


# ═══════════════════════════════════════════════════════════════════════
# SEGMENT RANKINGS
# ═══════════════════════════════════════════════════════════════════════

def _ml_segment_rankings(betting: pd.DataFrame, threshold_pct: float = 0.2) -> pd.DataFrame:
    models = sorted(betting['model'].unique())
    rows = []
    for model in models:
        model_total = len(betting[betting['model'] == model])
        min_bets = int(model_total * threshold_pct)
        seg = _ml_segment_analysis(betting, model)
        if not seg:
            continue
        for dim_name, buckets in seg.items():
            for seg_label, stats in buckets.items():
                if stats['Bets'] >= min_bets:
                    rows.append({
                        'Model': model.upper(),
                        'Dimension': dim_name,
                        'Segment': seg_label,
                        'Bets': stats['Bets'],
                        'WR (%)': stats['WR (%)'],
                        'ROI (%)': stats['ROI (%)'],
                        'Profit ($)': stats['Profit ($)'],
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('ROI (%)', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'
    return df


def _totals_segment_rankings(betting: pd.DataFrame, threshold_pct: float = 0.2,
                             matched_raw: pd.DataFrame = None) -> pd.DataFrame:
    models = sorted(betting['model'].unique())
    rows = []
    for model in models:
        model_total = len(betting[betting['model'] == model])
        min_bets = int(model_total * threshold_pct)
        seg = _totals_segment_analysis(betting, model, matched_raw=matched_raw)
        if not seg:
            continue
        for dim_name, buckets in seg.items():
            for seg_label, stats in buckets.items():
                if stats['Bets'] >= min_bets:
                    rows.append({
                        'Model': model.upper(),
                        'Dimension': dim_name,
                        'Segment': seg_label,
                        'Bets': stats['Bets'],
                        'WR (%)': stats['WR (%)'],
                        'ROI (%)': stats['ROI (%)'],
                        'Profit ($)': stats['Profit ($)'],
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('ROI (%)', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'
    return df


# ═══════════════════════════════════════════════════════════════════════
# CACHED DATA LOAD
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_all():
    results = _load_game_results()
    ml_preds = _load_predictions('mlb_predictions_with_odds')
    tot_preds = _load_predictions('mlb_totals_predictions_with_odds')
    ml_matched = _match_ml(ml_preds, results)
    tot_matched = _match_totals(tot_preds, results)
    ml_betting = _calc_ml_roi(ml_matched) if not ml_matched.empty else pd.DataFrame()
    tot_betting = _calc_totals_roi(tot_matched) if not tot_matched.empty else pd.DataFrame()
    return ml_betting, tot_betting, tot_matched


@st.cache_data(ttl=300)
def load_shadow():
    """Load shadow model data separately (used only for Segment Rankings)."""
    results = _load_game_results()
    ml_shadow = _load_shadow_predictions('mlb_predictions_with_odds')
    tot_shadow = _load_shadow_predictions('mlb_totals_predictions_with_odds')
    ml_s_matched = _match_ml(ml_shadow, results)
    tot_s_matched = _match_totals(tot_shadow, results)
    ml_s_betting = _calc_ml_roi(ml_s_matched) if not ml_s_matched.empty else pd.DataFrame()
    tot_s_betting = _calc_totals_roi(tot_s_matched) if not tot_s_matched.empty else pd.DataFrame()
    return ml_s_betting, tot_s_betting


# ═══════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _roi_color(val):
    if val > 5:
        return 'background-color:#d1fae5;color:#065f46;font-weight:600'
    if val > 0:
        return 'background-color:#ecfdf5;color:#047857;font-weight:500'
    if val > -5:
        return 'background-color:#fef3c7;color:#92400e;font-weight:500'
    return 'background-color:#fee2e2;color:#991b1b;font-weight:600'


def _render_segment_tables(seg: dict):
    for title, buckets in seg.items():
        st.markdown(f"**{title}**")
        rows = []
        for lbl, stats in buckets.items():
            if stats['Bets'] > 0:
                rows.append({'Segment': lbl, **stats})
        if rows:
            df = pd.DataFrame(rows)
            styled = df.style.map(_roi_color, subset=['ROI (%)']).format({
                'WR (%)': '{:.1f}%', 'ROI (%)': '{:.2f}%', 'Profit ($)': '${:.1f}',
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.caption("No data")


def _date_filter(betting: pd.DataFrame, prefix: str):
    if betting.empty:
        return betting
    mn = betting['date'].min().date()
    mx = betting['date'].max().date()
    c1, c2 = st.columns(2)
    with c1:
        s = st.date_input("Start", value=mn, min_value=mn, max_value=mx, key=f'{prefix}_s')
    with c2:
        e = st.date_input("End", value=mx, min_value=mn, max_value=mx, key=f'{prefix}_e')
    mask = (betting['date'].dt.date >= s) & (betting['date'].dt.date <= e)
    return betting[mask].copy()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── header ──
    st.markdown("# ⚾ MLB Analytics Pro")
    st.markdown("**2026 Season** — Real-time transparent performance tracking of ML prediction models.")

    ml_betting, tot_betting, tot_matched_raw = load_all()

    total_ml_games = len(ml_betting) // max(ml_betting['model'].nunique(), 1) if not ml_betting.empty else 0
    total_tot_games = len(tot_betting) // max(tot_betting['model'].nunique(), 1) if not tot_betting.empty else 0
    ml_models = ml_betting['model'].nunique() if not ml_betting.empty else 0
    tot_models = tot_betting['model'].nunique() if not tot_betting.empty else 0

    ml_summary = _summarise(ml_betting) if not ml_betting.empty else pd.DataFrame()
    tot_summary = _summarise(tot_betting) if not tot_betting.empty else pd.DataFrame()

    if not ml_betting.empty:
        seg_rank_header = _ml_segment_rankings(ml_betting, 0.2)
        best_roi = seg_rank_header.iloc[0]['ROI (%)'] if not seg_rank_header.empty else 0
        date_range = f"{ml_betting['date'].min().strftime('%b %d')} — {ml_betting['date'].max().strftime('%b %d, %Y')}"
    else:
        best_roi = 0
        date_range = "—"

    roi_cls = "positive" if best_roi > 0 else "negative"
    st.markdown(f"""
    <div class="metric-grid">
        <div class="m-card"><div class="value">{total_ml_games + total_tot_games}</div><div class="label">Games Tracked</div></div>
        <div class="m-card"><div class="value">{ml_models + tot_models}</div><div class="label">AI Models</div></div>
        <div class="m-card"><div class="value {roi_cls}">{best_roi:+.1f}%</div><div class="label">Best Segment ROI</div></div>
        <div class="m-card"><div class="value" style="font-size:1.1rem">{date_range}</div><div class="label">Period</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 3 Podium (Segment Rankings) ──
    if not ml_betting.empty:
        if len(seg_rank_header) >= 3:
            medals = [("🥇", "gold"), ("🥈", "silver"), ("🥉", "bronze")]
            podium_html = '<div class="podium-grid">'
            for i, (medal, cls) in enumerate(medals):
                r = seg_rank_header.iloc[i]
                podium_html += f"""
                <div class="podium-card {cls}">
                    <div class="podium-rank">{medal}</div>
                    <div class="podium-name">{r['Model']}</div>
                    <div class="podium-roi">{r['ROI (%)']:+.2f}%</div>
                    <div class="podium-stat">{r['Dimension']} · {r['Segment']}</div>
                    <div class="podium-stat">{r['WR (%)']:.1f}% WR · {r['Bets']} bets · ${r['Profit ($)']:+,.0f}</div>
                </div>"""
            podium_html += '</div>'
            st.markdown(podium_html, unsafe_allow_html=True)

    # # ── Highlight strip (disabled — too long on mobile) ──
    # if not ml_betting.empty:
    #     total_wins = int(ml_betting[ml_betting['model'] == ml_summary.iloc[0]['Model'].lower()]['won'].sum()) if not ml_summary.empty else 0
    #     total_bets_best = int(ml_summary.iloc[0]['Bets']) if not ml_summary.empty else 0
    #     avg_wr_all = ml_summary['Win Rate (%)'].mean() if not ml_summary.empty else 0
    #     profitable_models = int((ml_summary['ROI (%)'] > 0).sum()) if not ml_summary.empty else 0
    #     total_model_count = len(ml_summary) if not ml_summary.empty else 0
    #     st.markdown(f"""
    #     <div class="highlight-strip">
    #         <div class="hl-card"><div class="hl-val">{avg_wr_all:.1f}%</div><div class="hl-lbl">Avg Win Rate (All Models)</div></div>
    #         <div class="hl-card"><div class="hl-val">{profitable_models}/{total_model_count}</div><div class="hl-lbl">Models with Positive ROI</div></div>
    #         <div class="hl-card"><div class="hl-val">{total_wins}/{total_bets_best}</div><div class="hl-lbl">Best Model W/L Record</div></div>
    #     </div>
    #     """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-row">
        <a class="cta-btn" href="https://t.me/mlbanalytics_free" target="_blank">📊 Free Picks</a>
        <a class="cta-btn" href="https://www.winible.com/mlbanalyticspro" target="_blank">💎 Premium</a>
        <a class="cta-btn secondary" href="https://t.me/mlbanalytics_community" target="_blank">💬 Community</a>
    </div>
    """, unsafe_allow_html=True)

    # ── tabs ──
    tab_ml, tab_tot, tab_daily, tab_seg, tab_rank, tab_about = st.tabs([
        "📊 Moneyline Rankings",
        "⚾ Totals Rankings",
        "📅 Daily Trend",
        "🔬 Segment Analysis",
        "🏆 Segment Rankings",
        "ℹ️ About & Disclaimer",
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB: Moneyline Rankings
    # ──────────────────────────────────────────────────────────────────
    with tab_ml:
        st.markdown('<div class="section-title">Moneyline Model Performance</div>',
                    unsafe_allow_html=True)

        if ml_betting.empty:
            st.info("No moneyline data yet. Run the pipeline first.")
        else:
            filtered = _date_filter(ml_betting, 'ml')
            if filtered.empty:
                st.warning("No data in the selected range.")
            else:
                summary = _summarise(filtered)
                n_games = len(filtered) // max(filtered['model'].nunique(), 1)
                st.caption(f"{n_games} games · {filtered['model'].nunique()} models")

                styled = summary.style.map(_roi_color, subset=['ROI (%)']).format({
                    'Win Rate (%)': '{:.2f}%', 'ROI (%)': '{:.2f}%',
                    'Profit ($)': '${:.2f}',
                })
                st.dataframe(styled, use_container_width=True, hide_index=True)

                top = summary.head(10)
                fig = px.bar(
                    top, x='Model', y='ROI (%)',
                    color='ROI (%)', color_continuous_scale='RdYlGn',
                    text='ROI (%)', title='Top 10 Models — ROI (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  xaxis_tickangle=-30, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB: Totals Rankings
    # ──────────────────────────────────────────────────────────────────
    with tab_tot:
        st.markdown('<div class="section-title">Totals (Over/Under) Model Performance</div>',
                    unsafe_allow_html=True)

        if tot_betting.empty:
            st.info("No totals data yet. Run the totals pipeline first.")
        else:
            filtered = _date_filter(tot_betting, 'tot')
            if filtered.empty:
                st.warning("No data in the selected range.")
            else:
                summary = _summarise(filtered)
                n_games = len(filtered) // max(filtered['model'].nunique(), 1)
                st.caption(f"{n_games} games · {filtered['model'].nunique()} models")

                styled = summary.style.map(_roi_color, subset=['ROI (%)']).format({
                    'Win Rate (%)': '{:.2f}%', 'ROI (%)': '{:.2f}%',
                    'Profit ($)': '${:.2f}',
                })
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Over / Under summary
                if 'direction' in filtered.columns:
                    st.markdown("**Over vs Under Aggregate**")
                    dir_agg = filtered.groupby('direction').agg(
                        bets=('won', 'count'),
                        wins=('won', 'sum'),
                        profit=('actual_profit', 'sum'),
                    ).reset_index()
                    dir_agg['WR (%)'] = (dir_agg['wins'] / dir_agg['bets'] * 100).round(1)
                    dir_agg['ROI (%)'] = (dir_agg['profit'] / (dir_agg['bets'] * 100) * 100).round(2)
                    dir_agg.columns = ['Direction', 'Bets', 'Wins', 'Profit ($)', 'WR (%)', 'ROI (%)']

                    c1, c2 = st.columns(2)
                    for col_st, direction in [(c1, 'OVER'), (c2, 'UNDER')]:
                        with col_st:
                            row = dir_agg[dir_agg['Direction'] == direction]
                            if not row.empty:
                                r = row.iloc[0]
                                color = '#2196F3' if direction == 'OVER' else '#FF9800'
                                st.metric(
                                    f"{direction}",
                                    f"{int(r['Wins'])}/{int(r['Bets'])} ({r['WR (%)']:.1f}%)",
                                    f"ROI {r['ROI (%)']:+.2f}%")

                top = summary.head(10)
                fig = px.bar(
                    top, x='Model', y='ROI (%)',
                    color='ROI (%)', color_continuous_scale='RdYlGn',
                    text='ROI (%)', title='Top 10 Totals Models — ROI (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  xaxis_tickangle=-30, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB: Daily Trend
    # ──────────────────────────────────────────────────────────────────
    with tab_daily:
        st.markdown('<div class="section-title">Daily Performance Trend</div>',
                    unsafe_allow_html=True)

        dtype = st.radio("Type", ["Moneyline", "Totals"], horizontal=True,
                         key='daily_type')
        src = ml_betting if dtype == "Moneyline" else tot_betting

        if src.empty:
            st.info("No data available.")
        else:
            models = sorted(src['model'].unique())
            sel = st.selectbox("Model", [m.upper() for m in models],
                               key='daily_model')
            daily = _daily_breakdown(src, sel.lower())

            if daily.empty:
                st.warning("No daily data.")
            else:
                profitable = int((daily['ROI (%)'] > 0).sum())
                avg_roi = daily['ROI (%)'].mean()
                total_pl = daily['Cum P/L ($)'].iloc[-1]

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="m-card"><div class="value">{len(daily)}</div><div class="label">Days</div></div>
                    <div class="m-card"><div class="value">{profitable}/{len(daily)}</div><div class="label">Profitable Days</div></div>
                    <div class="m-card"><div class="value {'positive' if avg_roi > 0 else 'negative'}">{avg_roi:+.2f}%</div><div class="label">Avg Daily ROI</div></div>
                    <div class="m-card"><div class="value {'positive' if total_pl > 0 else 'negative'}">${total_pl:+,.0f}</div><div class="label">Cumulative P/L</div></div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily['Date'], y=daily['ROI (%)'],
                    mode='lines+markers', name='Daily ROI',
                    line=dict(color='#3b82f6', width=2.5),
                    marker=dict(size=5)))
                fig.add_hline(y=0, line_dash='dash', line_color='#dc2626',
                              line_width=1.5, opacity=.6)
                fig.update_layout(
                    height=380, title=f'{sel} — Daily ROI',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='ROI (%)', xaxis_title='')
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.area(
                    daily, x='Date', y='Cum P/L ($)',
                    title=f'{sel} — Cumulative Profit/Loss',
                    color_discrete_sequence=['#3b82f6'])
                fig2.update_layout(height=340,
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("**Daily Details**")
                st.dataframe(
                    daily.style.map(_roi_color, subset=['ROI (%)']).format({
                        'Win Rate (%)': '{:.1f}%', 'ROI (%)': '{:.2f}%',
                        'Profit ($)': '${:.1f}', 'Cum P/L ($)': '${:.1f}',
                    }),
                    use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB: Segment Analysis
    # ──────────────────────────────────────────────────────────────────
    with tab_seg:
        seg_ml_tab, seg_tot_tab = st.tabs(["📊 Moneyline", "⚾ Totals"])

        with seg_ml_tab:
            st.markdown('<div class="section-title">Segment Analysis (Moneyline)</div>',
                        unsafe_allow_html=True)
            if ml_betting.empty:
                st.info("No moneyline data yet.")
            else:
                seg_ml_filtered = _date_filter(ml_betting, 'seg_ml')
                models = sorted(seg_ml_filtered['model'].unique())
                sel = st.selectbox("Model", [m.upper() for m in models],
                                   key='seg_model')
                seg = _ml_segment_analysis(seg_ml_filtered, sel.lower())
                if seg:
                    _render_segment_tables(seg)
                else:
                    st.warning("No segment data for this model.")

        with seg_tot_tab:
            st.markdown('<div class="section-title">Segment Analysis (Totals)</div>',
                        unsafe_allow_html=True)
            if tot_betting.empty:
                st.info("No totals data yet.")
            else:
                seg_tot_filtered = _date_filter(tot_betting, 'seg_tot')
                tot_models = sorted(seg_tot_filtered['model'].unique())
                sel_t = st.selectbox("Model", [m.upper() for m in tot_models],
                                     key='seg_tot_model')
                seg_t = _totals_segment_analysis(seg_tot_filtered, sel_t.lower(),
                                                 matched_raw=tot_matched_raw)
                if seg_t:
                    _render_segment_tables(seg_t)
                else:
                    st.warning("No segment data for this model.")

    # ──────────────────────────────────────────────────────────────────
    # TAB: Segment Rankings
    # ──────────────────────────────────────────────────────────────────
    with tab_rank:
        rank_ml_tab, rank_tot_tab = st.tabs(["📊 Moneyline", "⚾ Totals"])

        def _render_ranked_table(betting_df, ranking_fn, prefix, shadow_betting_df=None,
                                 shadow_ranking_fn=None, matched_raw=None):
            """Render segment ranking table with All-Time + Recent + Shadow ROI."""
            c1, c2 = st.columns(2)
            with c1:
                threshold = st.slider(
                    "Min sample size (% of model total bets)", 10, 40, 20,
                    key=f'rank_{prefix}_thresh') / 100
            with c2:
                recent_days = st.selectbox(
                    "Recent period", [7, 14, 21, 30], index=0,
                    format_func=lambda d: f"Last {d} days",
                    key=f'rank_{prefix}_recent')

            extra = {'matched_raw': matched_raw} if matched_raw is not None else {}
            rank_all = ranking_fn(betting_df, threshold, **extra)
            if rank_all.empty:
                st.warning("No segments meet the sample size threshold.")
                return

            cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
            recent_df = betting_df[betting_df['date'] >= cutoff]

            if not recent_df.empty:
                rank_recent = ranking_fn(recent_df, 0.0, **extra)
            else:
                rank_recent = pd.DataFrame()

            merged = rank_all.copy()
            merged = merged.rename(columns={
                'ROI (%)': 'All-Time ROI (%)',
                'WR (%)': 'All-Time WR (%)',
            })

            if not rank_recent.empty:
                recent_lookup = rank_recent.set_index(['Model', 'Dimension', 'Segment'])
                r_roi, r_wr, r_bets = [], [], []
                for _, row in merged.iterrows():
                    key = (row['Model'], row['Dimension'], row['Segment'])
                    if key in recent_lookup.index:
                        rr = recent_lookup.loc[key]
                        r_roi.append(rr['ROI (%)'])
                        r_wr.append(rr['WR (%)'])
                        r_bets.append(int(rr['Bets']))
                    else:
                        r_roi.append(np.nan)
                        r_wr.append(np.nan)
                        r_bets.append(0)
                merged[f'Recent {recent_days}d ROI (%)'] = r_roi
                merged[f'Recent {recent_days}d WR (%)'] = r_wr
                merged[f'Recent {recent_days}d Bets'] = r_bets
            else:
                merged[f'Recent {recent_days}d ROI (%)'] = np.nan
                merged[f'Recent {recent_days}d WR (%)'] = np.nan
                merged[f'Recent {recent_days}d Bets'] = 0

            def _trend(row):
                alltime = row['All-Time ROI (%)']
                recent = row[f'Recent {recent_days}d ROI (%)']
                if pd.isna(recent):
                    return '—'
                diff = recent - alltime
                if diff > 2:
                    return '🔺'
                if diff < -2:
                    return '🔻'
                return '➡️'

            merged['Trend'] = merged.apply(_trend, axis=1)

            s_fn = shadow_ranking_fn or ranking_fn
            has_shadow = shadow_betting_df is not None and not shadow_betting_df.empty
            if has_shadow:
                shadow_rank = s_fn(shadow_betting_df, 0.0)
                if not shadow_rank.empty:
                    shadow_lookup = shadow_rank.set_index(['Model', 'Dimension', 'Segment'])
                    s_roi, s_wr, s_bets = [], [], []
                    for _, row in merged.iterrows():
                        key = (row['Model'], row['Dimension'], row['Segment'])
                        if key in shadow_lookup.index:
                            sr = shadow_lookup.loc[key]
                            s_roi.append(sr['ROI (%)'])
                            s_wr.append(sr['WR (%)'])
                            s_bets.append(int(sr['Bets']))
                        else:
                            s_roi.append(np.nan)
                            s_wr.append(np.nan)
                            s_bets.append(0)
                    merged['Shadow Bets'] = s_bets
                    merged['Shadow WR (%)'] = s_wr
                    merged['Shadow ROI (%)'] = s_roi
                    shadow_days = shadow_betting_df['date'].nunique()
                else:
                    has_shadow = False

            if not has_shadow:
                merged['Shadow Bets'] = 0
                merged['Shadow WR (%)'] = np.nan
                merged['Shadow ROI (%)'] = np.nan
                shadow_days = 0

            merged = merged.sort_values('All-Time ROI (%)', ascending=False).reset_index(drop=True)
            merged.index = merged.index + 1
            merged.index.name = 'Rank'

            col_order = ['Model', 'Dimension', 'Segment', 'Bets',
                         'All-Time WR (%)', 'All-Time ROI (%)', 'Profit ($)',
                         f'Recent {recent_days}d Bets', f'Recent {recent_days}d WR (%)',
                         f'Recent {recent_days}d ROI (%)', 'Trend',
                         'Shadow Bets', 'Shadow WR (%)', 'Shadow ROI (%)']
            merged = merged[col_order]

            shadow_label = f" · Shadow: {shadow_days}d data" if shadow_days > 0 else " · Shadow: no data"
            st.caption(f"Showing {len(merged)} segments · Trend: 🔺 improving (>+2%) · 🔻 declining (>-2%) · ➡️ stable{shadow_label}")

            roi_cols = ['All-Time ROI (%)', f'Recent {recent_days}d ROI (%)', 'Shadow ROI (%)']
            fmt = {
                'All-Time WR (%)': '{:.1f}%', 'All-Time ROI (%)': '{:.2f}%',
                'Profit ($)': '${:,.1f}',
                f'Recent {recent_days}d WR (%)': '{:.1f}%',
                f'Recent {recent_days}d ROI (%)': '{:.2f}%',
                'Shadow WR (%)': '{:.1f}%',
                'Shadow ROI (%)': '{:.2f}%',
            }
            styled = merged.style.map(_roi_color, subset=roi_cols).format(
                fmt, na_rep='—')
            st.dataframe(styled, use_container_width=True)

        ml_shadow_betting, tot_shadow_betting = load_shadow()

        with rank_ml_tab:
            st.markdown('<div class="section-title">Segment Rankings (Moneyline)</div>',
                        unsafe_allow_html=True)
            if ml_betting.empty:
                st.info("No moneyline data yet.")
            else:
                _render_ranked_table(ml_betting, _ml_segment_rankings, 'ml',
                                     shadow_betting_df=ml_shadow_betting)

        with rank_tot_tab:
            st.markdown('<div class="section-title">Segment Rankings (Totals)</div>',
                        unsafe_allow_html=True)
            if tot_betting.empty:
                st.info("No totals data yet.")
            else:
                _render_ranked_table(tot_betting, _totals_segment_rankings, 'tot',
                                     shadow_betting_df=tot_shadow_betting,
                                     shadow_ranking_fn=_totals_segment_rankings,
                                     matched_raw=tot_matched_raw)

    # ──────────────────────────────────────────────────────────────────
    # TAB: About & Disclaimer
    # ──────────────────────────────────────────────────────────────────
    with tab_about:
        st.markdown('<div class="section-title">About MLB Analytics Pro</div>',
                    unsafe_allow_html=True)

        col_text, col_img = st.columns([3, 1])
        with col_text:
            st.markdown("""
**MLB Analytics Pro** uses advanced machine learning to analyze MLB games
every day of the season.

- **24+ Models** — LightGBM, CatBoost, XGBoost, Neural Networks, SVM, Random Forest and ensemble variants
- **Moneyline + Totals** — Both game-winner and over/under predictions
- **100% Transparent** — Every prediction recorded and tracked, no cherry-picking
- **Updated Daily** — Automated pipeline runs before each game slate
            """)
        with col_img:
            try:
                st.image("images/KakaoTalk_20250727_162246740.jpg",
                         use_container_width=True)
            except Exception:
                pass

        st.markdown("---")
        st.markdown("#### ⚠️ Disclaimer")
        st.error("""
**NOT FINANCIAL OR GAMBLING ADVICE** — This service provides statistical
analysis and educational content only. Past performance does not guarantee
future results. Any betting or gambling activity carries significant risk
of financial loss. All decisions you make are entirely your own
responsibility. Comply with all applicable laws in your jurisdiction.
        """)

    # ── footer ──
    st.markdown("""
    <div class="footer-text">
        MLB Analytics Pro · 2026 Season · Updated continuously · 100% Transparent<br>
        <em>All results independently verified — no cherry-picking.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
