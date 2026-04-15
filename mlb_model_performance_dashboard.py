import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import logging


class MLBModelPerformanceAnalyzer:
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.predictions_dir = self.project_root / 'src' / 'odds' / 'data' / 'matched'
        self.records_dir = self.project_root / 'data' / 'records'
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MLBModelPerformanceAnalyzer')
    
    def load_matched_predictions(self, exclude_today: bool = True, model_tag: str = 'active') -> pd.DataFrame:
        """Load matched prediction files by tag.
        
        Each file contains 2 days of data (today + tomorrow). To avoid
        duplication while keeping data for dates that have no dedicated file:
        
        1. Collect all file_dates that exist (= dates with a dedicated file)
        2. From each file, always keep "today" records (file_date == record_date)
        3. Also keep "tomorrow" records ONLY if that date has no dedicated file
        4. When multiple files share the same file_date, use the latest one
        5. Final dedup by (date, home_team, away_team)
        """
        if model_tag == 'active':
            pattern = 'mlb_predictions_with_odds_*_active.json'
        elif model_tag == 'shadow':
            pattern = 'mlb_predictions_with_odds_*_shadow.json'
        else:
            pattern = 'mlb_predictions_with_odds_*.json'
        
        matched_files = sorted(self.predictions_dir.glob(pattern))
        
        if not matched_files:
            self.logger.error(f"No prediction files found (tag: {model_tag})")
            return pd.DataFrame()
        
        today = datetime.now().strftime('%Y%m%d')
        today_fmt = f"{today[:4]}-{today[4:6]}-{today[6:8]}"
        
        # Group files by file_date, keeping only the latest per date
        files_by_date = {}
        for file in matched_files:
            try:
                parts = file.stem.split('_')
                file_date = parts[4]  # YYYYMMDD
            except (IndexError, ValueError):
                self.logger.warning(f"Date extraction failed: {file.name}")
                continue
            
            if exclude_today and file_date == today:
                self.logger.info(f"Excluding today's file: {file.name}")
                continue
            
            files_by_date[file_date] = file
        
        # Set of all file_dates (dates that have a dedicated file)
        covered_dates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in files_by_date.keys()}
        
        all_predictions = []
        
        for file_date, file in sorted(files_by_date.items()):
            file_date_fmt = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}"
            self.logger.info(f"Loading: {file.name}")
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                kept = 0
                skipped = 0
                for record in data:
                    record_date = record.get('date', '')
                    
                    if exclude_today and record_date == today_fmt:
                        skipped += 1
                        continue
                    
                    is_same_day = (record_date == file_date_fmt)
                    
                    if is_same_day:
                        all_predictions.append(record)
                        kept += 1
                    elif record_date not in covered_dates:
                        # Next-day record with no dedicated file — keep it
                        all_predictions.append(record)
                        kept += 1
                    else:
                        skipped += 1
                
                if skipped > 0:
                    self.logger.info(f"  Kept {kept}/{kept + skipped} (skipped {skipped} duplicate next-day records)")
                
            except Exception as e:
                self.logger.error(f"Failed to load {file.name}: {e}")
                continue
        
        if not all_predictions:
            self.logger.error("No prediction data loaded.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_predictions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Final dedup by (date, home_team, away_team) — keep last occurrence
        before = len(df)
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
        if before != len(df):
            self.logger.info(f"Deduplicated: {before} -> {len(df)}")
        
        self.logger.info(f"Loaded {len(df)} predictions (tag: {model_tag})")
        return df
    
    def load_game_results(self) -> pd.DataFrame:
        """Load actual game results from historical records."""
        record_files = sorted(self.records_dir.glob('mlb_historical_records_*.json'))
        
        if not record_files:
            self.logger.error("No historical record files found.")
            return pd.DataFrame()
        
        latest_file = record_files[-1]
        self.logger.info(f"Loading game results: {latest_file.name}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # Only completed games
        df = df[df['abstract_game_state'] == 'Final'].copy()
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        self.logger.info(f"Loaded {len(df)} completed games")
        return df
    
    def match_predictions_with_results(self, predictions_df: pd.DataFrame,
                                       results_df: pd.DataFrame) -> pd.DataFrame:
        """Match predictions with actual game results."""
        if predictions_df.empty or results_df.empty:
            self.logger.error("Empty data.")
            return pd.DataFrame()
        
        # Build index for fast lookup
        results_index = {}
        for _, row in results_df.iterrows():
            key = f"{row['date']}_{row['home_team_name']}_{row['away_team_name']}"
            results_index[key] = row
        
        matched_data = []
        unmatched_count = 0
        
        for _, pred in predictions_df.iterrows():
            pred_date = pred['date'].date()
            key = f"{pred_date}_{pred['home_team']}_{pred['away_team']}"
            
            if key in results_index:
                result = results_index[key]
                
                if 'home_win' not in result or pd.isna(result.get('home_win')):
                    unmatched_count += 1
                    continue
                
                matched_game = pred.to_dict()
                matched_game['actual_home_win'] = int(result['home_win'])
                matched_game['actual_home_score'] = int(result.get('home_score', 0))
                matched_game['actual_away_score'] = int(result.get('away_score', 0))
                
                matched_data.append(matched_game)
            else:
                unmatched_count += 1
        
        matched_df = pd.DataFrame(matched_data)
        self.logger.info(f"Matched: {len(matched_df)}, Unmatched: {unmatched_count}")
        return matched_df
    
    def detect_models(self, matched_df: pd.DataFrame) -> List[str]:
        """Auto-detect model names from probability columns."""
        prob_cols = [col for col in matched_df.columns if col.endswith('_probability')]
        
        excluded = {
            'win_probability', 'home_team_probability_odds',
            'away_team_probability_odds', 'predicted_winner_probability_odds'
        }
        
        models = []
        for col in sorted(prob_cols):
            if col in excluded:
                continue
            model_name = col.replace('_probability', '')
            models.append(model_name)
        
        self.logger.info(f"Detected models: {models}")
        return models
    
    def calculate_betting_roi(self, matched_df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """Calculate betting ROI for each model and each game."""
        results = []
        
        for model in models:
            prob_col = f'{model}_probability'
            
            if prob_col not in matched_df.columns:
                continue
            
            for _, row in matched_df.iterrows():
                home_prob = row[prob_col]
                home_odds = row.get('home_team_odds')
                away_odds = row.get('away_team_odds')
                actual_home_win = row['actual_home_win']
                
                if pd.isna(home_odds) or pd.isna(away_odds) or pd.isna(home_prob):
                    continue
                
                bet_on_home = home_prob > 0.5
                
                if bet_on_home:
                    if actual_home_win == 1:
                        payout = self._calculate_payout(100, home_odds)
                        profit = payout - 100
                    else:
                        profit = -100
                    bet_team = 'home'
                    bet_odds = home_odds
                    bet_prob = home_prob
                else:
                    if actual_home_win == 0:
                        payout = self._calculate_payout(100, away_odds)
                        profit = payout - 100
                    else:
                        profit = -100
                    bet_team = 'away'
                    bet_odds = away_odds
                    bet_prob = 1 - home_prob
                
                if bet_odds > 0:
                    implied_prob = 100 / (bet_odds + 100)
                else:
                    implied_prob = (-bet_odds) / (-bet_odds + 100)
                
                predicted_roi = (bet_prob - implied_prob) * 100
                
                results.append({
                    'model': model,
                    'date': row['date'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'bet_team': bet_team,
                    'bet_odds': bet_odds,
                    'bet_probability': bet_prob,
                    'implied_probability': implied_prob,
                    'predicted_roi_pct': predicted_roi,
                    'actual_profit': profit,
                    'actual_roi_pct': profit,
                    'won': profit > 0,
                    'confidence_level': self._get_confidence_level(bet_prob),
                    'predicted_roi_bucket': self._get_roi_bucket(predicted_roi),
                    'odds_bucket': self._get_odds_bucket(bet_odds)
                })
        
        return pd.DataFrame(results)
    
    def _calculate_payout(self, stake: float, american_odds: float) -> float:
        if american_odds > 0:
            return stake + (stake * american_odds / 100)
        else:
            return stake + (stake * 100 / (-american_odds))
    
    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.8:
            return '80%+'
        elif probability >= 0.7:
            return '70-80%'
        elif probability >= 0.6:
            return '60-70%'
        else:
            return '50-60%'
    
    def _get_roi_bucket(self, predicted_roi: float) -> str:
        if predicted_roi >= 20:
            return '20%+'
        elif predicted_roi >= 10:
            return '10-20%'
        elif predicted_roi >= 0:
            return '0-10%'
        else:
            return 'Negative'
    
    def _get_odds_bucket(self, american_odds: float) -> str:
        if american_odds >= 300:
            return '+300+ (Heavy Underdog)'
        elif american_odds >= 200:
            return '+200 ~ +299 (Underdog)'
        elif american_odds >= 150:
            return '+150 ~ +199'
        elif american_odds >= 100:
            return '+100 ~ +149'
        elif american_odds >= -110:
            return "-110 ~ +99 (Pick'em)"
        elif american_odds >= -150:
            return '-150 ~ -111'
        elif american_odds >= -200:
            return '-200 ~ -151 (Favorite)'
        elif american_odds >= -300:
            return '-300 ~ -201'
        else:
            return '-300- (Heavy Favorite)'
    
    def analyze_model_performance(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        summary = []
        
        for model in betting_results['model'].unique():
            model_bets = betting_results[betting_results['model'] == model]
            
            if len(model_bets) == 0:
                continue
            
            total_bets = len(model_bets)
            wins = model_bets['won'].sum()
            losses = total_bets - wins
            win_rate = wins / total_bets * 100
            
            total_profit = model_bets['actual_profit'].sum()
            total_staked = total_bets * 100
            roi = (total_profit / total_staked) * 100
            
            avg_odds = model_bets['bet_odds'].mean()
            avg_probability = model_bets['bet_probability'].mean()
            
            summary.append({
                'Model': model.upper(),
                'Total Bets': total_bets,
                'Wins': int(wins),
                'Losses': int(losses),
                'Win Rate (%)': round(win_rate, 2),
                'Total Profit ($)': round(total_profit, 2),
                'ROI (%)': round(roi, 2),
                'Avg Odds': round(avg_odds, 0),
                'Avg Confidence': round(avg_probability * 100, 2)
            })
        
        return pd.DataFrame(summary).sort_values('ROI (%)', ascending=False)
    
    def analyze_by_confidence(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        confidence_levels = ['50-60%', '60-70%', '70-80%', '80%+']
        summary = []
        
        for model in betting_results['model'].unique():
            for conf_level in confidence_levels:
                mask = (betting_results['model'] == model) & \
                       (betting_results['confidence_level'] == conf_level)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                roi = (total_profit / total_staked) * 100
                
                summary.append({
                    'Model': model.upper(),
                    'Confidence': conf_level,
                    'Bets': total_bets,
                    'Wins': int(wins),
                    'Win Rate (%)': round(win_rate, 2),
                    'ROI (%)': round(roi, 2)
                })
        
        return pd.DataFrame(summary)
    
    def analyze_by_predicted_roi(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        roi_buckets = ['Negative', '0-10%', '10-20%', '20%+']
        summary = []
        
        for model in betting_results['model'].unique():
            for roi_bucket in roi_buckets:
                mask = (betting_results['model'] == model) & \
                       (betting_results['predicted_roi_bucket'] == roi_bucket)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                actual_roi = (total_profit / total_staked) * 100
                
                avg_predicted_roi = model_bets['predicted_roi_pct'].mean()
                
                summary.append({
                    'Model': model.upper(),
                    'Predicted ROI': roi_bucket,
                    'Bets': total_bets,
                    'Wins': int(wins),
                    'Win Rate (%)': round(win_rate, 2),
                    'Avg Pred ROI (%)': round(avg_predicted_roi, 2),
                    'Actual ROI (%)': round(actual_roi, 2)
                })
        
        return pd.DataFrame(summary)
    
    def segment_analysis(self, betting_results: pd.DataFrame, model: str) -> dict:
        """Run 5-type segment analysis for a single model.

        Mirrors the telegram dashboard logic exactly (same formulas,
        same segment boundaries, same labels).
        """
        model_bets = betting_results[betting_results['model'] == model.lower()].copy()
        if model_bets.empty:
            return {}

        def _decimal_odds(american):
            if american > 0:
                return (american / 100) + 1
            return (100 / abs(american)) + 1

        analysis_data = []
        for _, r in model_bets.iterrows():
            prob = r['bet_probability']
            selection_odds = r['bet_odds']
            dec = _decimal_odds(selection_odds)
            ev_roi = (prob * dec - 1) * 100

            if selection_odds > 0:
                market_implied = 100 / (selection_odds + 100)
            else:
                market_implied = abs(selection_odds) / (abs(selection_odds) + 100)

            divergence = prob - market_implied

            home_prob = prob if r['bet_team'] == 'home' else (1 - prob)
            confidence = abs(home_prob - 0.5)

            b = dec - 1
            kelly_frac = (prob * b - (1 - prob)) / b if b > 0 else 0
            kelly_pct = max(0, kelly_frac) * 100

            actual_roi = r['actual_profit']

            analysis_data.append({
                'ev_predicted_roi': ev_roi,
                'actual_roi': actual_roi,
                'selection_odds': selection_odds,
                'confidence': confidence,
                'divergence': divergence,
                'kelly_pct': kelly_pct,
                'predicted_team': r['bet_team'],
                'actual_correct': r['won'],
            })

        def _calc_segment(items):
            if not items:
                return {'games': 0, 'predicted_roi': 0, 'actual_roi': 0,
                        'roi_difference': 0, 'win_rate': 0, 'accuracy': 0}
            n = len(items)
            pred = sum(d['ev_predicted_roi'] for d in items) / n
            act = sum(d['actual_roi'] for d in items) / n
            wins = sum(1 for d in items if d['actual_roi'] > 0)
            correct = sum(1 for d in items if d['actual_correct'])
            return {
                'games': n,
                'predicted_roi': pred,
                'actual_roi': act,
                'roi_difference': act - pred,
                'win_rate': (wins / n) * 100,
                'accuracy': (correct / n) * 100,
            }

        def _classify(data, classify_fn, ordered_labels):
            buckets = {lbl: [] for lbl in ordered_labels}
            for d in data:
                lbl = classify_fn(d)
                if lbl in buckets:
                    buckets[lbl].append(d)
            return {lbl: _calc_segment(buckets[lbl]) for lbl in ordered_labels}

        seg = {}

        # 1. Predicted ROI
        def _pred_roi_label(d):
            v = d['ev_predicted_roi']
            if v < -20: return 'Very Negative (<-20%)'
            if v < 0:   return 'Negative (-20% ~ 0%)'
            if v < 20:  return 'Positive (0% ~ 20%)'
            if v < 40:  return 'Very Positive A (20% ~ 40%)'
            if v < 60:  return 'Very Positive B (40% ~ 60%)'
            if v < 100: return 'Extremely Positive A (60% ~ 100%)'
            return 'Extremely Positive B (>100%)'

        seg['predicted_roi'] = _classify(analysis_data, _pred_roi_label, [
            'Very Negative (<-20%)', 'Negative (-20% ~ 0%)',
            'Positive (0% ~ 20%)', 'Very Positive A (20% ~ 40%)',
            'Very Positive B (40% ~ 60%)', 'Extremely Positive A (60% ~ 100%)',
            'Extremely Positive B (>100%)'])

        # 2. Odds Ranges
        def _odds_label(d):
            o = d['selection_odds']
            if o < -200:      return 'Heavy Favorite (< -200)'
            if o < -120:      return 'Favorite (-200 ~ -120)'
            if -120 <= o <= 120: return 'Pick Em (-120 ~ +120)'
            if o <= 150:      return 'Underdog (+120 ~ +150)'
            if o <= 300:      return 'Strong Underdog (+150 ~ +300)'
            return 'Heavy Underdog (> +300)'

        seg['odds'] = _classify(analysis_data, _odds_label, [
            'Heavy Favorite (< -200)', 'Favorite (-200 ~ -120)',
            'Pick Em (-120 ~ +120)', 'Underdog (+120 ~ +150)',
            'Strong Underdog (+150 ~ +300)', 'Heavy Underdog (> +300)'])

        # 3. Confidence Levels
        def _conf_label(d):
            c = d['confidence']
            if c < 0.05: return 'Low Confidence (0-0.05)'
            if c < 0.15: return 'Medium Confidence (0.05-0.15)'
            if c < 0.25: return 'High Confidence (0.15-0.25)'
            return 'Very High Confidence (>0.25)'

        seg['confidence'] = _classify(analysis_data, _conf_label, [
            'Low Confidence (0-0.05)', 'Medium Confidence (0.05-0.15)',
            'High Confidence (0.15-0.25)', 'Very High Confidence (>0.25)'])

        # 4. Market vs Model Divergence
        def _div_label(d):
            v = d['divergence']
            if v >= 0.10:  return 'Model Much More Optimistic (+10%+)'
            if v >= 0.05:  return 'Model Slightly Optimistic (+5% ~ +10%)'
            if v >= -0.05: return 'Market Aligned (-5% ~ +5%)'
            if v >= -0.10: return 'Model Slightly Pessimistic (-10% ~ -5%)'
            return 'Model Much More Pessimistic (<-10%)'

        seg['market_divergence'] = _classify(analysis_data, _div_label, [
            'Model Much More Optimistic (+10%+)',
            'Model Slightly Optimistic (+5% ~ +10%)',
            'Market Aligned (-5% ~ +5%)',
            'Model Slightly Pessimistic (-10% ~ -5%)',
            'Model Much More Pessimistic (<-10%)'])

        # 5. Kelly Criterion
        def _kelly_label(d):
            k = d['kelly_pct']
            if k <= 0:  return 'No Selection (Kelly \u2264 0%)'
            if k <= 5:  return 'Low Confidence (0% < Kelly \u2264 5%)'
            if k <= 15: return 'Medium Confidence (5% < Kelly \u2264 15%)'
            if k <= 25: return 'High Confidence (15% < Kelly \u2264 25%)'
            if k <= 60: return 'Very High Confidence (25% ~ 60%)'
            return 'Extremely High Confidence (Kelly > 60%)'

        seg['kelly'] = _classify(analysis_data, _kelly_label, [
            'No Selection (Kelly \u2264 0%)',
            'Low Confidence (0% < Kelly \u2264 5%)',
            'Medium Confidence (5% < Kelly \u2264 15%)',
            'High Confidence (15% < Kelly \u2264 25%)',
            'Very High Confidence (25% ~ 60%)',
            'Extremely High Confidence (Kelly > 60%)'])

        return seg

    # ------------------------------------------------------------------
    # Totals (Over/Under) helpers
    # ------------------------------------------------------------------

    def load_matched_totals_predictions(self, exclude_today: bool = True,
                                        model_tag: str = 'active') -> pd.DataFrame:
        """Load totals matched prediction files with 2-day dedup logic."""
        if model_tag == 'active':
            pattern = 'mlb_totals_predictions_with_odds_*_active.json'
        elif model_tag == 'shadow':
            pattern = 'mlb_totals_predictions_with_odds_*_shadow.json'
        else:
            pattern = 'mlb_totals_predictions_with_odds_*.json'

        matched_files = sorted(self.predictions_dir.glob(pattern))

        if not matched_files:
            self.logger.info(f"No totals prediction files found (tag: {model_tag})")
            return pd.DataFrame()

        today = datetime.now().strftime('%Y%m%d')
        today_fmt = f"{today[:4]}-{today[4:6]}-{today[6:8]}"

        files_by_date = {}
        for file in matched_files:
            try:
                parts = file.stem.split('_')
                file_date = parts[5]
            except (IndexError, ValueError):
                self.logger.warning(f"Totals date extraction failed: {file.name}")
                continue

            if exclude_today and file_date == today:
                self.logger.info(f"Excluding today's totals file: {file.name}")
                continue

            files_by_date[file_date] = file

        covered_dates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in files_by_date.keys()}

        all_predictions = []

        for file_date, file in sorted(files_by_date.items()):
            file_date_fmt = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}"
            self.logger.info(f"Loading totals: {file.name}")

            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                kept = 0
                skipped = 0
                for record in data:
                    record_date = record.get('date', '')

                    if exclude_today and record_date == today_fmt:
                        skipped += 1
                        continue

                    is_same_day = (record_date == file_date_fmt)

                    if is_same_day:
                        all_predictions.append(record)
                        kept += 1
                    elif record_date not in covered_dates:
                        all_predictions.append(record)
                        kept += 1
                    else:
                        skipped += 1

                if skipped > 0:
                    self.logger.info(
                        f"  Kept {kept}/{kept + skipped} "
                        f"(skipped {skipped} duplicate next-day records)")

            except Exception as e:
                self.logger.error(f"Failed to load {file.name}: {e}")
                continue

        if not all_predictions:
            self.logger.info("No totals prediction data loaded.")
            return pd.DataFrame()

        df = pd.DataFrame(all_predictions)
        df['date'] = pd.to_datetime(df['date'])

        before = len(df)
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
        if before != len(df):
            self.logger.info(f"Totals deduplicated: {before} -> {len(df)}")

        self.logger.info(f"Loaded {len(df)} totals predictions (tag: {model_tag})")
        return df

    def match_totals_with_results(self, predictions_df: pd.DataFrame,
                                  results_df: pd.DataFrame) -> pd.DataFrame:
        """Match totals predictions with actual scores."""
        if predictions_df.empty or results_df.empty:
            return pd.DataFrame()

        results_index = {}
        for _, row in results_df.iterrows():
            key = f"{row['date']}_{row['home_team_name']}_{row['away_team_name']}"
            results_index[key] = row

        matched_data = []
        for _, pred in predictions_df.iterrows():
            pred_date = pred['date'].date()
            key = f"{pred_date}_{pred['home_team']}_{pred['away_team']}"

            if key in results_index:
                result = results_index[key]
                home_score = int(result.get('home_score', 0))
                away_score = int(result.get('away_score', 0))

                if home_score == 0 and away_score == 0:
                    continue

                matched_game = pred.to_dict()
                matched_game['actual_home_score'] = home_score
                matched_game['actual_away_score'] = away_score
                matched_game['actual_total'] = home_score + away_score
                matched_data.append(matched_game)

        matched_df = pd.DataFrame(matched_data)
        self.logger.info(f"Totals matched: {len(matched_df)} games")
        return matched_df

    def detect_totals_models(self, matched_df: pd.DataFrame) -> List[str]:
        """Auto-detect totals model names from *_total columns."""
        excluded = {'predicted_total', 'ensemble_total', 'total_line', 'actual_total'}
        total_cols = [c for c in matched_df.columns
                      if c.endswith('_total') and c not in excluded]
        models = sorted([c.replace('_total', '') for c in total_cols])
        self.logger.info(f"Detected totals models: {len(models)}")
        return models

    def calculate_totals_betting_roi(self, matched_df: pd.DataFrame,
                                     models: List[str]) -> pd.DataFrame:
        """Calculate Over/Under betting ROI for each totals model."""
        results = []
        all_models = ['ensemble'] + models

        for model in all_models:
            if model == 'ensemble':
                pred_col = 'ensemble_total'
            else:
                pred_col = f'{model}_total'

            if pred_col not in matched_df.columns:
                continue

            for _, row in matched_df.iterrows():
                pred_total = row[pred_col]
                total_line = row.get('total_line')
                over_odds = row.get('over_odds')
                under_odds = row.get('under_odds')
                actual_total = row['actual_total']

                if pd.isna(total_line) or pd.isna(over_odds) or pd.isna(under_odds):
                    continue
                if pd.isna(pred_total):
                    continue

                difference = pred_total - total_line
                direction = 'OVER' if difference > 0 else 'UNDER'

                if direction == 'OVER':
                    bet_odds = over_odds
                else:
                    bet_odds = under_odds

                if actual_total == total_line:
                    profit = 0
                    won = None
                elif direction == 'OVER':
                    won = actual_total > total_line
                    profit = (self._calculate_payout(100, over_odds) - 100) if won else -100
                else:
                    won = actual_total < total_line
                    profit = (self._calculate_payout(100, under_odds) - 100) if won else -100

                margin_abs = abs(difference)

                results.append({
                    'model': model,
                    'date': row['date'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'predicted_total': pred_total,
                    'total_line': total_line,
                    'actual_total': actual_total,
                    'direction': direction,
                    'bet_odds': bet_odds,
                    'actual_profit': profit,
                    'won': won,
                    'margin': difference,
                    'margin_abs': margin_abs,
                    'margin_bucket': self._get_margin_bucket(margin_abs),
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.dropna(subset=['won'])
            df['won'] = df['won'].astype(bool)
        return df

    @staticmethod
    def _get_margin_bucket(margin_abs: float) -> str:
        if margin_abs < 0.5:
            return '0 - 0.5'
        elif margin_abs < 1.0:
            return '0.5 - 1.0'
        elif margin_abs < 2.0:
            return '1.0 - 2.0'
        elif margin_abs < 3.0:
            return '2.0 - 3.0'
        return '3.0+'

    def analyze_totals_performance(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate totals performance per model."""
        summary = []
        for model in betting_results['model'].unique():
            mb = betting_results[betting_results['model'] == model]
            if mb.empty:
                continue
            n = len(mb)
            wins = int(mb['won'].sum())
            losses = n - wins
            wr = wins / n * 100
            profit = mb['actual_profit'].sum()
            roi = (profit / (n * 100)) * 100
            avg_margin = mb['margin_abs'].mean()

            summary.append({
                'Model': model.upper(),
                'Bets': n,
                'Wins': wins,
                'Losses': losses,
                'Win Rate (%)': round(wr, 2),
                'Total Profit ($)': round(profit, 2),
                'ROI (%)': round(roi, 2),
                'Avg |Margin|': round(avg_margin, 2),
            })
        return pd.DataFrame(summary).sort_values('ROI (%)', ascending=False)

    def analyze_totals_by_direction(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """Performance split by Over vs Under."""
        summary = []
        for model in betting_results['model'].unique():
            for direction in ['OVER', 'UNDER']:
                mb = betting_results[
                    (betting_results['model'] == model)
                    & (betting_results['direction'] == direction)]
                if mb.empty:
                    continue
                n = len(mb)
                wins = int(mb['won'].sum())
                wr = wins / n * 100
                profit = mb['actual_profit'].sum()
                roi = (profit / (n * 100)) * 100

                summary.append({
                    'Model': model.upper(),
                    'Direction': direction,
                    'Bets': n,
                    'Wins': wins,
                    'Win Rate (%)': round(wr, 2),
                    'ROI (%)': round(roi, 2),
                })
        return pd.DataFrame(summary)

    def analyze_totals_by_margin(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """Performance by margin bucket (|predicted - line|)."""
        margin_order = ['0 - 0.5', '0.5 - 1.0', '1.0 - 2.0', '2.0 - 3.0', '3.0+']
        summary = []
        for model in betting_results['model'].unique():
            for bucket in margin_order:
                mb = betting_results[
                    (betting_results['model'] == model)
                    & (betting_results['margin_bucket'] == bucket)]
                if mb.empty:
                    continue
                n = len(mb)
                wins = int(mb['won'].sum())
                wr = wins / n * 100
                profit = mb['actual_profit'].sum()
                roi = (profit / (n * 100)) * 100

                summary.append({
                    'Model': model.upper(),
                    'Margin': bucket,
                    'Bets': n,
                    'Wins': wins,
                    'Win Rate (%)': round(wr, 2),
                    'ROI (%)': round(roi, 2),
                })
        return pd.DataFrame(summary)

    # -----------------------------------------------------------------
    # Pick generation (mirrors telegram_dashboard logic exactly)
    # -----------------------------------------------------------------

    ALL_MODELS = [
        'model1', 'model2', 'model3', 'model4', 'model5', 'model6',
        'model7', 'model8', 'model9', 'model_rf', 'model_nn', 'model_svm',
        'model_advanced_catboost_basic', 'model_advanced_catboost',
        'model_advanced_lgbm_basic', 'model_advanced_lgbm',
        'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
        'model_advanced_xgboost_basic', 'model_advanced_xgboost',
        'model1_extended_lgbm', 'model2_extended_catboost',
        'model3_extended_xgboost',
    ]

    ZONE_OPTIONS = {
        'predicted_roi': [
            'Very Negative (<-20%)', 'Negative (-20% ~ 0%)',
            'Positive (0% ~ 20%)', 'Very Positive A (20% ~ 40%)',
            'Very Positive B (40% ~ 60%)',
            'Extremely Positive A (60% ~ 100%)',
            'Extremely Positive B (>100%)',
        ],
        'odds': [
            'Heavy Favorite (< -200)', 'Favorite (-200 ~ -120)',
            'Pick Em (-120 ~ +120)', 'Underdog (+120 ~ +150)',
            'Strong Underdog (+150 ~ +300)', 'Heavy Underdog (> +300)',
        ],
        'confidence': [
            'Low Confidence (0-0.05)', 'Medium Confidence (0.05-0.15)',
            'High Confidence (0.15-0.25)', 'Very High Confidence (>0.25)',
        ],
        'odds_probability_divergence': [
            'Model Much More Pessimistic (<-10%)',
            'Model Slightly Pessimistic (-10% ~ -5%)',
            'Market Aligned (-5% ~ +5%)',
            'Model Slightly Optimistic (+5% ~ +10%)',
            'Model Much More Optimistic (+10%+)',
        ],
        'kelly_criterion': [
            'No Selection (Kelly \u2264 0%)',
            'Low Confidence (0% < Kelly \u2264 5%)',
            'Medium Confidence (5% < Kelly \u2264 15%)',
            'High Confidence (15% < Kelly \u2264 25%)',
            'Very High Confidence (25% ~ 60%)',
            'Extremely High Confidence (Kelly > 60%)',
        ],
    }

    def load_latest_predictions_raw(self, model_tag: str = 'active'):
        """Load latest raw prediction file (today's games)."""
        pattern = f'mlb_predictions_with_odds_*_{model_tag}.json'
        files = sorted(self.predictions_dir.glob(pattern))
        if not files:
            return None
        import re
        def _ts(f):
            m = re.search(r'(\d{8}_\d{6})', f.name)
            return m.group(1) if m else ''
        latest = max(files, key=_ts)
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            filtered = []
            for g in data:
                if g.get('home_team_odds') is not None and g.get('away_team_odds') is not None:
                    g['home_odds'] = g['home_team_odds']
                    g['away_odds'] = g['away_team_odds']
                    filtered.append(g)
            return filtered, latest.name
        except Exception:
            return None

    @staticmethod
    def _odds_to_prob(odds):
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def _kelly(win_prob, odds):
        try:
            dec = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
            k = (win_prob * dec - 1) / (dec - 1)
            return max(0, k * 100)
        except Exception:
            return 0

    def process_game_pick(self, game, weights):
        """Compute pick metrics for a game (same logic as telegram)."""
        ensemble_prob = 0
        total_w = 0
        for model, w in weights.items():
            pk = f'{model}_probability'
            if pk in game and game[pk] is not None and w > 0:
                ensemble_prob += float(game[pk]) * w
                total_w += w
        if total_w == 0:
            return None
        ensemble_prob /= total_w

        if ensemble_prob > 0.5:
            side = 'home'
            sel_odds = game.get('home_odds')
            sel_team = game.get('home_team')
        else:
            side = 'away'
            sel_odds = game.get('away_odds')
            sel_team = game.get('away_team')

        if sel_odds is None:
            return None
        sel_odds = float(sel_odds)

        win_prob = ensemble_prob if side == 'home' else 1 - ensemble_prob
        win_payout = (sel_odds / 100) * 100 if sel_odds > 0 else (100 / abs(sel_odds)) * 100
        pred_roi = win_prob * win_payout + (1 - win_prob) * (-100)
        confidence = abs(ensemble_prob - 0.5)
        market_prob = self._odds_to_prob(sel_odds)
        divergence = (win_prob - market_prob) * 100
        kelly = self._kelly(win_prob, sel_odds)

        return {
            'game': f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}",
            'pick': sel_team,
            'side': side,
            'odds': sel_odds,
            'win_prob': win_prob,
            'predicted_roi': pred_roi,
            'confidence': confidence,
            'divergence': divergence,
            'kelly': kelly,
            'date': game.get('date', ''),
            'start_time': game.get('start_time_et', ''),
        }

    @staticmethod
    def _check_zone(processed, dimension, segment):
        if dimension == 'predicted_roi':
            v = processed['predicted_roi']
            if segment == 'Very Negative (<-20%)':          return v < -20
            if segment == 'Negative (-20% ~ 0%)':           return -20 <= v < 0
            if segment == 'Positive (0% ~ 20%)':            return 0 <= v < 20
            if segment == 'Very Positive A (20% ~ 40%)':    return 20 <= v < 40
            if segment == 'Very Positive B (40% ~ 60%)':    return 40 <= v < 60
            if segment == 'Extremely Positive A (60% ~ 100%)': return 60 <= v < 100
            if segment == 'Extremely Positive B (>100%)':   return v >= 100
        elif dimension == 'odds':
            o = processed['odds']
            if segment == 'Heavy Favorite (< -200)':        return o < -200
            if segment == 'Favorite (-200 ~ -120)':         return -200 <= o < -120
            if segment == 'Pick Em (-120 ~ +120)':          return -120 <= o <= 120
            if segment == 'Underdog (+120 ~ +150)':         return 120 < o <= 150
            if segment == 'Strong Underdog (+150 ~ +300)':  return 150 < o <= 300
            if segment == 'Heavy Underdog (> +300)':        return o > 300
        elif dimension == 'confidence':
            c = processed['confidence']
            if segment == 'Low Confidence (0-0.05)':        return c < 0.05
            if segment == 'Medium Confidence (0.05-0.15)':  return 0.05 <= c < 0.15
            if segment == 'High Confidence (0.15-0.25)':    return 0.15 <= c < 0.25
            if segment == 'Very High Confidence (>0.25)':   return c >= 0.25
        elif dimension == 'odds_probability_divergence':
            d = processed['divergence']
            if segment == 'Model Much More Pessimistic (<-10%)':       return d < -10
            if segment == 'Model Slightly Pessimistic (-10% ~ -5%)':   return -10 <= d < -5
            if segment == 'Market Aligned (-5% ~ +5%)':               return -5 <= d < 5
            if segment == 'Model Slightly Optimistic (+5% ~ +10%)':    return 5 <= d < 10
            if segment == 'Model Much More Optimistic (+10%+)':        return d >= 10
        elif dimension == 'kelly_criterion':
            k = processed['kelly']
            if segment == 'No Selection (Kelly \u2264 0%)':                return k <= 0
            if segment == 'Low Confidence (0% < Kelly \u2264 5%)':         return 0 < k <= 5
            if segment == 'Medium Confidence (5% < Kelly \u2264 15%)':     return 5 < k <= 15
            if segment == 'High Confidence (15% < Kelly \u2264 25%)':      return 15 < k <= 25
            if segment == 'Very High Confidence (25% ~ 60%)':             return 25 < k <= 60
            if segment == 'Extremely High Confidence (Kelly > 60%)':      return k > 60
        return False

    def find_picks(self, games, selected_zones, weights):
        """Find games matching all selected zones (AND across dimensions, OR within)."""
        picks = []
        if not games or not selected_zones or not weights:
            return picks
        for g in games:
            processed = self.process_game_pick(g, weights)
            if processed is None:
                continue
            match = True
            for dim, segs in selected_zones.items():
                if not any(self._check_zone(processed, dim, s) for s in segs):
                    match = False
                    break
            if match:
                picks.append(processed)
        return picks

    def totals_segment_analysis(self, betting_results: pd.DataFrame,
                                model: str, matched_df: pd.DataFrame,
                                totals_models: List[str]) -> dict:
        """Run 5-dimension segment analysis for a totals model."""
        model_bets = betting_results[betting_results['model'] == model.lower()].copy()
        if model_bets.empty:
            return {}

        consensus_map = {}
        for _, row in matched_df.iterrows():
            game_key = (
                row['date'].date() if hasattr(row['date'], 'date') else row['date'],
                row['home_team'], row['away_team'])
            total_line = row.get('total_line')
            if pd.isna(total_line):
                continue
            over_count = 0
            total_count = 0
            for m in totals_models:
                col = f"{m}_total"
                if col in row.index and not pd.isna(row[col]):
                    total_count += 1
                    if row[col] > total_line:
                        over_count += 1
            if total_count > 0:
                consensus_map[game_key] = {
                    'over_pct': over_count / total_count * 100,
                    'under_pct': (total_count - over_count) / total_count * 100,
                }

        analysis_data = []
        for _, r in model_bets.iterrows():
            game_key = (
                r['date'].date() if hasattr(r['date'], 'date') else r['date'],
                r['home_team'], r['away_team'])
            cons = consensus_map.get(game_key, {'over_pct': 50, 'under_pct': 50})
            direction_consensus = (cons['over_pct'] if r['direction'] == 'OVER'
                                   else cons['under_pct'])

            analysis_data.append({
                'margin_abs': r['margin_abs'],
                'bet_odds': r['bet_odds'],
                'total_line': r['total_line'],
                'predicted_total': r['predicted_total'],
                'direction': r['direction'],
                'actual_profit': r['actual_profit'],
                'won': r['won'],
                'consensus': direction_consensus,
            })

        def _calc(items):
            if not items:
                return {'games': 0, 'wins': 0, 'win_rate': 0,
                        'roi': 0, 'avg_margin': 0}
            n = len(items)
            wins = sum(1 for d in items if d['won'])
            profit = sum(d['actual_profit'] for d in items)
            avg_margin = sum(d['margin_abs'] for d in items) / n
            return {
                'games': n,
                'wins': wins,
                'win_rate': (wins / n) * 100,
                'roi': (profit / (n * 100)) * 100,
                'avg_margin': avg_margin,
            }

        def _classify(data, fn, labels):
            buckets = {lbl: [] for lbl in labels}
            for d in data:
                lbl = fn(d)
                if lbl in buckets:
                    buckets[lbl].append(d)
            return {lbl: _calc(buckets[lbl]) for lbl in labels}

        seg = {}

        def _margin_label(d):
            m = d['margin_abs']
            if m < 0.5: return '0 - 0.5'
            if m < 1.0: return '0.5 - 1.0'
            if m < 2.0: return '1.0 - 2.0'
            if m < 3.0: return '2.0 - 3.0'
            return '3.0+'

        seg['margin'] = _classify(analysis_data, _margin_label,
            ['0 - 0.5', '0.5 - 1.0', '1.0 - 2.0', '2.0 - 3.0', '3.0+'])

        def _odds_label(d):
            o = d['bet_odds']
            if o < -130:         return 'Heavy Juice (< -130)'
            if o < -115:         return 'Moderate Juice (-130 ~ -115)'
            if o < -105:         return 'Standard (-115 ~ -105)'
            if o <= 100:         return 'Light Juice (-105 ~ +100)'
            return 'Plus Odds (+100+)'

        seg['odds'] = _classify(analysis_data, _odds_label,
            ['Heavy Juice (< -130)', 'Moderate Juice (-130 ~ -115)',
             'Standard (-115 ~ -105)', 'Light Juice (-105 ~ +100)',
             'Plus Odds (+100+)'])

        def _consensus_label(d):
            c = d['consensus']
            if c < 60:  return 'Low (< 60%)'
            if c < 75:  return 'Moderate (60% - 75%)'
            if c < 90:  return 'High (75% - 90%)'
            return 'Very High (90%+)'

        seg['consensus'] = _classify(analysis_data, _consensus_label,
            ['Low (< 60%)', 'Moderate (60% - 75%)',
             'High (75% - 90%)', 'Very High (90%+)'])

        def _line_label(d):
            tl = d['total_line']
            if tl <= 7.0:  return 'Low (≤ 7.0)'
            if tl <= 8.0:  return 'Medium-Low (7.0 - 8.0)'
            if tl <= 9.0:  return 'Medium (8.0 - 9.0)'
            if tl <= 10.0: return 'Medium-High (9.0 - 10.0)'
            return 'High (10.0+)'

        seg['line_level'] = _classify(analysis_data, _line_label,
            ['Low (\u2264 7.0)', 'Medium-Low (7.0 - 8.0)',
             'Medium (8.0 - 9.0)', 'Medium-High (9.0 - 10.0)',
             'High (10.0+)'])

        def _dir_label(d):
            return d['direction']

        seg['direction'] = _classify(analysis_data, _dir_label,
            ['OVER', 'UNDER'])

        return seg

    # ------------------------------------------------------------------
    # Totals Picks helpers
    # ------------------------------------------------------------------

    TOTALS_ALL_MODELS = [
        'model1', 'model2', 'model3', 'model4', 'model5', 'model6',
        'model7', 'model8', 'model9', 'model_rf', 'model_nn', 'model_svm',
        'model_advanced_catboost_basic', 'model_advanced_catboost',
        'model_advanced_lgbm_basic', 'model_advanced_lgbm',
        'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
        'model_advanced_xgboost_basic', 'model_advanced_xgboost',
        'model1_extended_lgbm', 'model2_extended_catboost',
        'model3_extended_xgboost',
    ]

    TOTALS_ZONE_OPTIONS = {
        'margin': [
            '0 - 0.5', '0.5 - 1.0', '1.0 - 2.0', '2.0 - 3.0', '3.0+',
        ],
        'odds': [
            'Heavy Juice (< -130)', 'Moderate Juice (-130 ~ -115)',
            'Standard (-115 ~ -105)', 'Light Juice (-105 ~ +100)',
            'Plus Odds (+100+)',
        ],
        'consensus': [
            'Low (< 60%)', 'Moderate (60% - 75%)',
            'High (75% - 90%)', 'Very High (90%+)',
        ],
        'line_level': [
            'Low (\u2264 7.0)', 'Medium-Low (7.0 - 8.0)',
            'Medium (8.0 - 9.0)', 'Medium-High (9.0 - 10.0)',
            'High (10.0+)',
        ],
        'direction': [
            'OVER', 'UNDER',
        ],
    }

    def load_latest_totals_predictions_raw(self, model_tag: str = 'active'):
        """Load latest raw totals prediction file (today's games)."""
        pattern = f'mlb_totals_predictions_with_odds_*_{model_tag}.json'
        files = sorted(self.predictions_dir.glob(pattern))
        if not files:
            return None
        import re
        def _ts(f):
            m = re.search(r'(\d{8}_\d{6})', f.name)
            return m.group(1) if m else ''
        latest = max(files, key=_ts)
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            filtered = []
            for g in data:
                if (g.get('total_line') is not None
                        and g.get('over_odds') is not None
                        and g.get('under_odds') is not None):
                    filtered.append(g)
            return filtered, latest.name
        except Exception:
            return None

    def process_totals_game_pick(self, game, weights):
        """Compute pick metrics for a totals game using the same segment dimensions."""
        ensemble_total = 0
        total_w = 0
        for model, w in weights.items():
            col = f'{model}_total'
            if col in game and game[col] is not None and w > 0:
                ensemble_total += float(game[col]) * w
                total_w += w
        if total_w == 0:
            return None
        ensemble_total /= total_w

        total_line = float(game['total_line'])
        margin = ensemble_total - total_line
        margin_abs = abs(margin)

        if margin > 0:
            direction = 'OVER'
            sel_odds = float(game['over_odds'])
        else:
            direction = 'UNDER'
            sel_odds = float(game['under_odds'])

        over_count = 0
        total_model_count = 0
        for m in self.TOTALS_ALL_MODELS:
            col = f'{m}_total'
            if col in game and game[col] is not None:
                total_model_count += 1
                if float(game[col]) > total_line:
                    over_count += 1
        if total_model_count > 0:
            over_pct = over_count / total_model_count * 100
            under_pct = 100 - over_pct
            consensus = over_pct if direction == 'OVER' else under_pct
        else:
            consensus = 50.0

        return {
            'game': f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}",
            'direction': direction,
            'total_line': total_line,
            'predicted_total': round(ensemble_total, 2),
            'margin_abs': round(margin_abs, 2),
            'odds': sel_odds,
            'consensus': round(consensus, 1),
            'date': game.get('date', ''),
            'start_time': game.get('start_time_et', ''),
        }

    @staticmethod
    def _check_totals_zone(processed, dimension, segment):
        if dimension == 'margin':
            m = processed['margin_abs']
            if segment == '0 - 0.5':    return m < 0.5
            if segment == '0.5 - 1.0':  return 0.5 <= m < 1.0
            if segment == '1.0 - 2.0':  return 1.0 <= m < 2.0
            if segment == '2.0 - 3.0':  return 2.0 <= m < 3.0
            if segment == '3.0+':       return m >= 3.0
        elif dimension == 'odds':
            o = processed['odds']
            if segment == 'Heavy Juice (< -130)':           return o < -130
            if segment == 'Moderate Juice (-130 ~ -115)':   return -130 <= o < -115
            if segment == 'Standard (-115 ~ -105)':         return -115 <= o < -105
            if segment == 'Light Juice (-105 ~ +100)':      return -105 <= o <= 100
            if segment == 'Plus Odds (+100+)':              return o > 100
        elif dimension == 'consensus':
            c = processed['consensus']
            if segment == 'Low (< 60%)':          return c < 60
            if segment == 'Moderate (60% - 75%)': return 60 <= c < 75
            if segment == 'High (75% - 90%)':     return 75 <= c < 90
            if segment == 'Very High (90%+)':     return c >= 90
        elif dimension == 'line_level':
            tl = processed['total_line']
            if segment == 'Low (\u2264 7.0)':              return tl <= 7.0
            if segment == 'Medium-Low (7.0 - 8.0)':   return 7.0 < tl <= 8.0
            if segment == 'Medium (8.0 - 9.0)':       return 8.0 < tl <= 9.0
            if segment == 'Medium-High (9.0 - 10.0)': return 9.0 < tl <= 10.0
            if segment == 'High (10.0+)':              return tl > 10.0
        elif dimension == 'direction':
            return processed['direction'] == segment
        return False

    def find_totals_picks(self, games, selected_zones, weights):
        """Find totals games matching all selected zones."""
        picks = []
        if not games or not selected_zones or not weights:
            return picks
        for g in games:
            processed = self.process_totals_game_pick(g, weights)
            if processed is None:
                continue
            match = True
            for dim, segs in selected_zones.items():
                if not any(self._check_totals_zone(processed, dim, s) for s in segs):
                    match = False
                    break
            if match:
                picks.append(processed)
        return picks

    def analyze_by_odds(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        odds_buckets = [
            '-300- (Heavy Favorite)',
            '-300 ~ -201',
            '-200 ~ -151 (Favorite)',
            '-150 ~ -111',
            "-110 ~ +99 (Pick'em)",
            '+100 ~ +149',
            '+150 ~ +199',
            '+200 ~ +299 (Underdog)',
            '+300+ (Heavy Underdog)'
        ]
        
        summary = []
        
        for model in betting_results['model'].unique():
            for odds_bucket in odds_buckets:
                mask = (betting_results['model'] == model) & \
                       (betting_results['odds_bucket'] == odds_bucket)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                actual_roi = (total_profit / total_staked) * 100
                
                avg_odds = model_bets['bet_odds'].mean()
                avg_confidence = model_bets['bet_probability'].mean() * 100
                
                summary.append({
                    'Model': model.upper(),
                    'Odds Range': odds_bucket,
                    'Bets': total_bets,
                    'Wins': int(wins),
                    'Win Rate (%)': round(win_rate, 2),
                    'Avg Odds': round(avg_odds, 0),
                    'Avg Confidence (%)': round(avg_confidence, 2),
                    'Actual ROI (%)': round(actual_roi, 2)
                })
        
        return pd.DataFrame(summary)


# =========================================================================
# Odds Range tab helper (used by both Active and Shadow tabs)
# =========================================================================

ODDS_ORDER = [
    '-300- (Heavy Favorite)',
    '-300 ~ -201',
    '-200 ~ -151 (Favorite)',
    '-150 ~ -111',
    "-110 ~ +99 (Pick'em)",
    '+100 ~ +149',
    '+150 ~ +199',
    '+200 ~ +299 (Underdog)',
    '+300+ (Heavy Underdog)'
]


def _simple_odds_label(odds_range: str) -> str:
    if 'Heavy Favorite' in str(odds_range):
        return 'Heavy Fav'
    elif 'Heavy Underdog' in str(odds_range):
        return 'Heavy Dog'
    elif 'Favorite' in str(odds_range):
        return 'Favorite'
    elif 'Underdog' in str(odds_range):
        return 'Underdog'
    elif "Pick'em" in str(odds_range):
        return "Pick'em"
    return str(odds_range)


def render_odds_range_tab(odds_analysis: pd.DataFrame, model_list: List[str],
                          selectbox_key: str, tag_label: str = ''):
    """Reusable renderer for odds-range analysis tabs."""
    if odds_analysis.empty:
        st.warning("No odds analysis data.")
        return
    
    selected_model = st.selectbox(
        "Select Model",
        options=[m.upper() for m in model_list],
        key=selectbox_key
    )
    
    if not selected_model:
        return
    
    model_data = odds_analysis[odds_analysis['Model'] == selected_model].copy()
    
    if model_data.empty:
        st.info(f"No data for {selected_model}.")
        return
    
    model_data['Odds Range'] = pd.Categorical(
        model_data['Odds Range'], categories=ODDS_ORDER, ordered=True
    )
    model_data = model_data.sort_values('Odds Range')
    
    suffix = f" ({tag_label})" if tag_label else ''
    st.subheader(f"📊 {selected_model} Performance by Odds Range{suffix}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_roi = model_data.loc[model_data['Actual ROI (%)'].idxmax()]
        st.metric("Best ROI Range", best_roi['Odds Range'], f"{best_roi['Actual ROI (%)']:.2f}%")
    
    with col2:
        total_bets = model_data['Bets'].sum()
        total_wins = model_data['Wins'].sum()
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        st.metric("Overall Win Rate", f"{win_rate:.2f}%")
    
    with col3:
        most_bets = model_data.loc[model_data['Bets'].idxmax()]
        st.metric("Most Bets Range", most_bets['Odds Range'], f"{int(most_bets['Bets'])} bets")
    
    with col4:
        weighted_roi = (model_data['Actual ROI (%)'] * model_data['Bets']).sum() / total_bets if total_bets > 0 else 0
        st.metric("Weighted Avg ROI", f"{weighted_roi:.2f}%")
    
    st.markdown("---")
    
    simple_labels = [_simple_odds_label(r) for r in model_data['Odds Range']]
    
    # ROI chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=simple_labels,
        y=model_data['Actual ROI (%)'],
        marker_color=['green' if x > 0 else 'red' for x in model_data['Actual ROI (%)']],
        text=model_data['Actual ROI (%)'].round(2),
        textposition='outside'
    ))
    fig.update_layout(
        title=f"{selected_model}{suffix} - ROI by Odds Range",
        xaxis_title="Odds Range (Favorite <-> Underdog)",
        yaxis_title="ROI (%)",
        showlegend=False, height=500, xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Win rate & bet count
    c1, c2 = st.columns(2)
    
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=simple_labels, y=model_data['Win Rate (%)'],
            marker_color='lightblue',
            text=model_data['Win Rate (%)'].round(1), textposition='outside'
        ))
        fig.update_layout(
            title=f"{selected_model}{suffix} - Win Rate",
            xaxis_title="Odds Range", yaxis_title="Win Rate (%)",
            showlegend=False, height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=simple_labels, y=model_data['Bets'],
            marker_color='lightcoral',
            text=model_data['Bets'], textposition='outside'
        ))
        fig.update_layout(
            title=f"{selected_model}{suffix} - Bet Distribution",
            xaxis_title="Odds Range", yaxis_title="Number of Bets",
            showlegend=False, height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detail table
    st.subheader(f"📋 Detailed Odds Range Statistics{suffix}")
    display = model_data.copy()
    display['_roi_num'] = display['Actual ROI (%)']
    display['Bets'] = display['Bets'].astype(int)
    display['Wins'] = display['Wins'].astype(int)
    display['Win Rate (%)'] = display['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
    display['Avg Odds'] = display['Avg Odds'].astype(int)
    display['Avg Confidence (%)'] = display['Avg Confidence (%)'].apply(lambda x: f"{x:.2f}")
    display['Actual ROI (%)'] = display['Actual ROI (%)'].apply(lambda x: f"{x:.2f}")
    
    def _style(row):
        styles = [''] * len(row)
        roi_idx = display.columns.get_loc('Actual ROI (%)')
        if row['_roi_num'] > 0:
            styles[roi_idx] = 'color: green; font-weight: bold'
        else:
            styles[roi_idx] = 'color: red; font-weight: bold'
        return styles
    
    st.dataframe(
        display.style.apply(_style, axis=1),
        use_container_width=True,
        column_config={'_roi_num': None}
    )
    
    # Heatmap (all models)
    st.markdown("---")
    st.subheader(f"🔄 Compare All Models{suffix}")
    
    pivot = odds_analysis.pivot(index='Model', columns='Odds Range', values='Actual ROI (%)')
    pivot = pivot.reindex(columns=ODDS_ORDER)
    pivot.columns = [_simple_odds_label(c) for c in pivot.columns]
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Odds Range", y="Model", color="ROI (%)"),
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title=f"ROI (%) Heatmap{suffix}"
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Segment Analysis tab helper (used by Active & Shadow tabs)
# =========================================================================

_SEG_TAB_CONFIG = [
    ('predicted_roi',     '\U0001F4CA Predicted ROI',  'Performance by Predicted ROI Ranges',    'Predicted ROI Range'),
    ('odds',              '\U0001F4B0 Odds Ranges',    'Performance by Betting Odds Ranges',     'Odds Range'),
    ('confidence',        '\U0001F3AF Confidence Levels', 'Performance by Confidence Levels',    'Confidence Level'),
    ('market_divergence', '\U0001F4C8 Market vs Model', 'Performance by Market vs Model Divergence', 'Market Divergence'),
    ('kelly',             '\U0001F3B0 Kelly Criterion', 'Performance by Kelly Criterion',        'Kelly Criterion Range'),
]


def _render_segment_table(seg_data: dict, segment_type: str):
    """Display a single segment analysis table with colour coding.

    seg_data is a dict {label: {games, predicted_roi, actual_roi,
    roi_difference, win_rate, accuracy}} matching the telegram format.
    """
    if not seg_data:
        st.info("No data in this segment.")
        return

    table_rows = []
    for label, stats in seg_data.items():
        if stats['games'] > 0:
            table_rows.append({
                segment_type: label,
                'Games': stats['games'],
                'Predicted ROI (%)': stats['predicted_roi'],
                'Actual ROI (%)': stats['actual_roi'],
                'ROI Difference (%)': stats['roi_difference'],
                'Win Rate (%)': stats['win_rate'],
                'Accuracy (%)': stats['accuracy'],
            })

    if not table_rows:
        st.info("No data with games available.")
        return

    df = pd.DataFrame(table_rows)

    def _style_roi(val):
        if val > 5:
            return 'background-color: #d4edda; color: #155724'
        elif val < -5:
            return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    styled = df.style.map(
        _style_roi, subset=['Actual ROI (%)']
    ).format({
        'Predicted ROI (%)': '{:.2f}%',
        'Actual ROI (%)': '{:.2f}%',
        'ROI Difference (%)': '{:.2f}%',
        'Win Rate (%)': '{:.1f}%',
        'Accuracy (%)': '{:.1f}%',
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if len(df) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df[segment_type], y=df['Actual ROI (%)'],
            marker_color=['green' if v > 0 else 'red' for v in df['Actual ROI (%)']],
            text=df['Actual ROI (%)'].apply(lambda v: f'{v:.1f}%'),
            textposition='outside', name='Actual ROI'
        ))
        fig.add_trace(go.Scatter(
            x=df[segment_type], y=df['Predicted ROI (%)'],
            mode='lines+markers', name='Predicted ROI',
            line=dict(color='royalblue', dash='dot', width=2)
        ))
        fig.update_layout(height=400, showlegend=True,
                          xaxis_tickangle=-30, yaxis_title='ROI (%)')
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_segment_analysis_tab(analyzer, betting_results: pd.DataFrame,
                                 model_list: list, key_prefix: str,
                                 tag_label: str = 'Active'):
    """Full segment analysis UI: model selector + 5 sub-tabs."""
    st.header(f"\U0001F52C Segment Analysis ({tag_label})")
    st.markdown(
        "Detailed analysis of model performance across different prediction "
        "confidence levels, odds ranges, and market scenarios."
    )

    selected = st.selectbox(
        "\U0001F3AF Select model for segment analysis:",
        options=[m.upper() for m in model_list],
        key=f'{key_prefix}_seg_model'
    )

    if not selected:
        return

    seg = analyzer.segment_analysis(betting_results, selected)

    if not seg:
        st.warning(f"No segment data for {selected}.")
        return

    sub_tabs = st.tabs([cfg[1] for cfg in _SEG_TAB_CONFIG])

    for (seg_key, _, subtitle, col_label), stab in zip(_SEG_TAB_CONFIG, sub_tabs):
        with stab:
            st.subheader(subtitle)
            data = seg.get(seg_key, {})
            _render_segment_table(data, col_label)


# =========================================================================
# Totals (O/U) tab helper
# =========================================================================

# =========================================================================
# Picks tab helper
# =========================================================================

_ZONE_DIM_LABELS = {
    'predicted_roi':               '\U0001F4CA Predicted ROI',
    'odds':                        '\U0001F4B0 Odds',
    'confidence':                  '\U0001F3AF Confidence',
    'odds_probability_divergence': '\U0001F4C8 Market vs Model',
    'kelly_criterion':             '\U0001F3B0 Kelly Criterion',
}


@st.fragment
def render_picks_tab(analyzer, key_prefix: str, tag_label: str = 'Active'):
    """Full picks UI: model selector + zone filters + results table."""
    st.header(f"\U0001F3AF Today's Picks ({tag_label})")

    result = analyzer.load_latest_predictions_raw(model_tag=tag_label.lower())
    if result is None:
        st.warning(
            f"No {tag_label} prediction data found. "
            "Run the pipeline first."
        )
        return
    games, fname = result
    st.success(f"Loaded {len(games)} games from `{fname}`")

    col_left, col_right = st.columns([1, 2])

    # --- Left column: configuration ---
    with col_left:
        st.markdown("#### \U0001F916 Model Selection")
        mode = st.radio(
            "Mode", ["Single Model", "Custom Weights"],
            key=f'{key_prefix}_pick_mode', horizontal=True)

        weights = {}
        if mode == "Single Model":
            options = ['ENSEMBLE'] + [m.upper() for m in analyzer.ALL_MODELS]
            sel = st.selectbox("Model", options, key=f'{key_prefix}_pick_model')
            if sel == 'ENSEMBLE':
                weights = {m: 1.0 for m in analyzer.ALL_MODELS}
            else:
                weights = {sel.lower(): 1.0}
        else:
            for m in analyzer.ALL_MODELS:
                label = m.replace('model_', '').replace('_', ' ').title()
                w = st.slider(label, 0.0, 1.0, 0.0, 0.05,
                              key=f'{key_prefix}_w_{m}')
                if w > 0:
                    weights[m] = w

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        active_w = {k: v for k, v in weights.items() if v > 0}
        if active_w:
            names = ', '.join(
                k.replace('model_', '').replace('_', ' ').title()
                for k in list(active_w)[:5])
            extra = f' +{len(active_w)-5}' if len(active_w) > 5 else ''
            st.caption(f"Active: {names}{extra}")

        st.markdown("---")
        st.markdown("#### \U0001F50D Zone Filters")
        st.caption("Same dimension = OR, across dimensions = AND")

        selected_zones: dict = {}
        for dim, dim_label in _ZONE_DIM_LABELS.items():
            with st.expander(dim_label, expanded=False):
                chosen = []
                for option in analyzer.ZONE_OPTIONS[dim]:
                    if st.checkbox(option, key=f'{key_prefix}_z_{dim}_{option}'):
                        chosen.append(option)
                if chosen:
                    selected_zones[dim] = chosen

    # --- Right column: results ---
    with col_right:
        st.markdown("#### \U0001F4CB Matching Picks")

        if not active_w:
            st.info("Select at least one model with non-zero weight.")
            return

        if not selected_zones:
            st.info("Select at least one zone filter to generate picks.")
            return

        picks = analyzer.find_picks(games, selected_zones, weights)

        if not picks:
            st.warning("No games match the selected zones.")
            return

        avg_roi = sum(p['predicted_roi'] for p in picks) / len(picks)
        avg_conf = sum(p['confidence'] for p in picks) / len(picks)
        c1, c2, c3 = st.columns(3)
        c1.metric("Matching Games", len(picks))
        c2.metric("Avg Predicted ROI", f"{avg_roi:+.1f}%")
        c3.metric("Avg Confidence", f"{avg_conf:.3f}")

        rows = []
        for i, p in enumerate(picks, 1):
            rows.append({
                '#': i,
                'Game': p['game'],
                'Pick': p['pick'],
                'Side': p['side'].upper(),
                'Odds': int(p['odds']),
                'Win Prob': f"{p['win_prob']:.1%}",
                'Pred ROI': f"{p['predicted_roi']:+.1f}%",
                'Confidence': f"{p['confidence']:.3f}",
                'Divergence': f"{p['divergence']:+.1f}%",
                'Kelly': f"{p['kelly']:.1f}%",
                'Time (ET)': p.get('start_time', ''),
            })

        df = pd.DataFrame(rows)

        def _style_pick(row):
            styles = [''] * len(row)
            roi_idx = df.columns.get_loc('Pred ROI')
            val = float(row['Pred ROI'].replace('%', '').replace('+', ''))
            if val > 0:
                styles[roi_idx] = 'color: green; font-weight: bold'
            elif val < 0:
                styles[roi_idx] = 'color: red; font-weight: bold'
            return styles

        st.dataframe(
            df.style.apply(_style_pick, axis=1),
            use_container_width=True, hide_index=True, height=min(600, 50 + 35 * len(rows)))

        active_zones_str = ' | '.join(
            f"{_ZONE_DIM_LABELS[d]}: {', '.join(s)}"
            for d, s in selected_zones.items())
        st.caption(f"Filters: {active_zones_str}")


# =========================================================================
# Totals Picks tab
# =========================================================================

_TOTALS_ZONE_DIM_LABELS = {
    'margin':     '\U0001F3AF Margin from Line',
    'odds':       '\U0001F4B0 Bet Odds',
    'consensus':  '\U0001F91D Model Consensus',
    'line_level': '\U0001F4CA Total Line Level',
    'direction':  '\u2195\ufe0f Direction',
}


@st.fragment
def render_totals_picks_tab(analyzer, key_prefix: str,
                            tag_label: str = 'Active'):
    """Totals picks UI: model selector + zone filters (same dims as segment analysis)."""
    st.header(f"\u26BE Today's Totals Picks ({tag_label})")

    result = analyzer.load_latest_totals_predictions_raw(
        model_tag=tag_label.lower())
    if result is None:
        st.warning(
            f"No {tag_label} totals prediction data found. "
            "Run the totals pipeline first.")
        return
    games, fname = result
    st.success(f"Loaded {len(games)} games from `{fname}`")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### \U0001F916 Model Selection")
        mode = st.radio(
            "Mode", ["Single Model", "Custom Weights"],
            key=f'{key_prefix}_tpick_mode', horizontal=True)

        weights = {}
        if mode == "Single Model":
            options = (['ENSEMBLE']
                       + [m.upper() for m in analyzer.TOTALS_ALL_MODELS])
            sel = st.selectbox("Model", options,
                               key=f'{key_prefix}_tpick_model')
            if sel == 'ENSEMBLE':
                weights = {m: 1.0 for m in analyzer.TOTALS_ALL_MODELS}
            else:
                weights = {sel.lower(): 1.0}
        else:
            for m in analyzer.TOTALS_ALL_MODELS:
                label = m.replace('model_', '').replace('_', ' ').title()
                w = st.slider(label, 0.0, 1.0, 0.0, 0.05,
                              key=f'{key_prefix}_tw_{m}')
                if w > 0:
                    weights[m] = w

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        active_w = {k: v for k, v in weights.items() if v > 0}
        if active_w:
            names = ', '.join(
                k.replace('model_', '').replace('_', ' ').title()
                for k in list(active_w)[:5])
            extra = f' +{len(active_w)-5}' if len(active_w) > 5 else ''
            st.caption(f"Active: {names}{extra}")

        st.markdown("---")
        st.markdown("#### \U0001F50D Zone Filters")
        st.caption("Same dimension = OR, across dimensions = AND")

        selected_zones: dict = {}
        for dim, dim_label in _TOTALS_ZONE_DIM_LABELS.items():
            with st.expander(dim_label, expanded=False):
                chosen = []
                for option in analyzer.TOTALS_ZONE_OPTIONS[dim]:
                    if st.checkbox(option,
                                   key=f'{key_prefix}_tz_{dim}_{option}'):
                        chosen.append(option)
                if chosen:
                    selected_zones[dim] = chosen

    with col_right:
        st.markdown("#### \U0001F4CB Matching Totals Picks")

        if not active_w:
            st.info("Select at least one model with non-zero weight.")
            return

        if not selected_zones:
            st.info("Select at least one zone filter to generate picks.")
            return

        picks = analyzer.find_totals_picks(games, selected_zones, weights)

        if not picks:
            st.warning("No games match the selected zones.")
            return

        avg_margin = sum(p['margin_abs'] for p in picks) / len(picks)
        avg_cons = sum(p['consensus'] for p in picks) / len(picks)
        c1, c2, c3 = st.columns(3)
        c1.metric("Matching Games", len(picks))
        c2.metric("Avg |Margin|", f"{avg_margin:.2f}")
        c3.metric("Avg Consensus", f"{avg_cons:.1f}%")

        rows = []
        for i, p in enumerate(picks, 1):
            rows.append({
                '#': i,
                'Game': p['game'],
                'Direction': p['direction'],
                'Line': p['total_line'],
                'Predicted': p['predicted_total'],
                '|Margin|': p['margin_abs'],
                'Odds': int(p['odds']),
                'Consensus': f"{p['consensus']:.1f}%",
                'Time (ET)': p.get('start_time', ''),
            })

        df = pd.DataFrame(rows)

        def _style_totals_pick(row):
            styles = [''] * len(row)
            dir_idx = df.columns.get_loc('Direction')
            if row['Direction'] == 'OVER':
                styles[dir_idx] = 'color: #2196F3; font-weight: bold'
            else:
                styles[dir_idx] = 'color: #FF9800; font-weight: bold'
            margin_idx = df.columns.get_loc('|Margin|')
            if row['|Margin|'] >= 2.0:
                styles[margin_idx] = 'color: green; font-weight: bold'
            return styles

        st.dataframe(
            df.style.apply(_style_totals_pick, axis=1),
            use_container_width=True, hide_index=True,
            height=min(600, 50 + 35 * len(rows)))

        active_zones_str = ' | '.join(
            f"{_TOTALS_ZONE_DIM_LABELS[d]}: {', '.join(s)}"
            for d, s in selected_zones.items())
        st.caption(f"Filters: {active_zones_str}")


# =========================================================================
# Totals (O/U) tab helpers
# =========================================================================

MARGIN_ORDER = ['0 - 0.5', '0.5 - 1.0', '1.0 - 2.0', '2.0 - 3.0', '3.0+']

_TOTALS_SEG_TAB_CONFIG = [
    ('margin',     '\U0001F3AF Margin from Line',    'Performance by |Predicted - Line| Margin', 'Margin'),
    ('odds',       '\U0001F4B0 Bet Odds',            'Performance by Bet Odds Ranges',           'Odds Range'),
    ('consensus',  '\U0001F91D Model Consensus',      'Performance by Model Consensus Level',     'Consensus'),
    ('line_level', '\U0001F4CA Total Line Level',     'Performance by Total Line Level',           'Line Level'),
    ('direction',  '\u2195\ufe0f Direction',          'Performance by Bet Direction (Over/Under)', 'Direction'),
]


def _render_totals_segment_table(seg_data: dict, segment_type: str):
    """Display a totals segment analysis table with colour coding."""
    if not seg_data:
        st.info("No data in this segment.")
        return

    table_rows = []
    for label, stats in seg_data.items():
        if stats['games'] > 0:
            table_rows.append({
                segment_type: label,
                'Games': stats['games'],
                'Wins': stats['wins'],
                'Win Rate (%)': stats['win_rate'],
                'ROI (%)': stats['roi'],
                'Avg |Margin|': stats['avg_margin'],
            })

    if not table_rows:
        st.info("No data with games available.")
        return

    df = pd.DataFrame(table_rows)

    def _style_roi(val):
        if val > 5:
            return 'background-color: #d4edda; color: #155724'
        elif val < -5:
            return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    def _style_wr(val):
        if val >= 55:
            return 'background-color: #d4edda; color: #155724'
        elif val < 45:
            return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    styled = df.style.map(
        _style_roi, subset=['ROI (%)']
    ).map(
        _style_wr, subset=['Win Rate (%)']
    ).format({
        'Win Rate (%)': '{:.1f}%',
        'ROI (%)': '{:.2f}%',
        'Avg |Margin|': '{:.2f}',
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if len(df) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df[segment_type], y=df['ROI (%)'],
            marker_color=['green' if v > 0 else 'red' for v in df['ROI (%)']],
            text=df['ROI (%)'].apply(lambda v: f'{v:.1f}%'),
            textposition='outside', name='ROI'
        ))
        fig.add_trace(go.Scatter(
            x=df[segment_type], y=df['Win Rate (%)'],
            mode='lines+markers', name='Win Rate',
            yaxis='y2', line=dict(color='royalblue', width=2)
        ))
        fig.update_layout(
            height=400, showlegend=True,
            xaxis_tickangle=-30,
            yaxis=dict(title='ROI (%)'),
            yaxis2=dict(title='Win Rate (%)', overlaying='y',
                        side='right', range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_totals_tab(analyzer, totals_predictions: pd.DataFrame,
                      results_df: pd.DataFrame,
                      key_prefix: str, tag_label: str = 'Active'):
    """Full Totals O/U analysis UI."""
    st.header(f"Totals Over/Under Performance ({tag_label})")

    if totals_predictions.empty:
        st.warning(
            f"No {tag_label} totals prediction data available. "
            "Run `python mlb_totals_pipeline.py` first."
        )
        return

    matched_totals = analyzer.match_totals_with_results(totals_predictions, results_df)
    if matched_totals.empty:
        st.warning("No totals predictions matched with completed game results yet.")
        return

    totals_models = analyzer.detect_totals_models(matched_totals)
    if not totals_models:
        st.warning("No totals model columns detected.")
        return

    totals_betting = analyzer.calculate_totals_betting_roi(matched_totals, totals_models)
    if totals_betting.empty:
        st.warning("No totals betting results (games may lack odds data).")
        return

    t_min = totals_betting['date'].min().date()
    t_max = totals_betting['date'].max().date()
    st.markdown("##### \U0001F4C5 Date Range")
    c1, c2 = st.columns(2)
    with c1:
        t_start = st.date_input("Start", value=t_min,
            min_value=t_min, max_value=t_max, key=f'{key_prefix}_date_start')
    with c2:
        t_end = st.date_input("End", value=t_max,
            min_value=t_min, max_value=t_max, key=f'{key_prefix}_date_end')

    date_mask = (
        (totals_betting['date'].dt.date >= t_start)
        & (totals_betting['date'].dt.date <= t_end)
    )
    tb = totals_betting[date_mask].copy()

    if tb.empty:
        st.warning("No totals data in the selected date range.")
        return

    n_models = tb['model'].nunique()
    n_games = len(tb) // max(n_models, 1)
    st.info(f"Totals bets: {len(tb)} ({n_games} games x {n_models} models)")

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Overall Performance",
        "Over vs Under",
        "By Margin from Line",
        "Segment Analysis",
        "Detailed Results",
    ])

    def _style_roi(val):
        if val > 5:
            return 'background-color: #d4edda; color: #155724'
        elif val < -5:
            return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    # --- Sub-tab 1: Overall Performance ---
    with sub1:
        st.subheader("Overall Totals Model Performance")
        perf = analyzer.analyze_totals_performance(tb)
        if not perf.empty:
            styled = perf.style.map(_style_roi, subset=['ROI (%)']).format({
                'Win Rate (%)': '{:.2f}%', 'ROI (%)': '{:.2f}%',
                'Total Profit ($)': '${:.2f}', 'Avg |Margin|': '{:.2f}',
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

            fig = px.bar(
                perf, x='Model', y='ROI (%)',
                color='ROI (%)',
                color_continuous_scale='RdYlGn',
                title="ROI by Model (Totals O/U)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                perf, x='Model', y='Win Rate (%)',
                color='Win Rate (%)',
                color_continuous_scale='Blues',
                title="Win Rate by Model (Totals O/U)"
            )
            fig2.add_hline(y=50, line_dash="dash", line_color="red",
                           annotation_text="50% Break-even")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # --- Sub-tab 2: Over vs Under ---
    with sub2:
        st.subheader("Over vs Under Performance")
        dir_perf = analyzer.analyze_totals_by_direction(tb)
        if not dir_perf.empty:
            agg = dir_perf.groupby('Direction').agg({
                'Bets': 'sum', 'Wins': 'sum',
            }).reset_index()
            agg['Win Rate (%)'] = agg['Wins'] / agg['Bets'] * 100

            col_o, col_u = st.columns(2)
            for col_st, direction, color in [
                (col_o, 'OVER', '#2196F3'), (col_u, 'UNDER', '#FF9800')
            ]:
                with col_st:
                    row = agg[agg['Direction'] == direction]
                    if not row.empty:
                        r = row.iloc[0]
                        st.metric(f"{direction} Total",
                                  f"{int(r['Wins'])}/{int(r['Bets'])}",
                                  f"{r['Win Rate (%)']:.1f}%")

            fig = px.bar(
                agg, x='Direction', y='Win Rate (%)',
                color='Direction',
                color_discrete_map={'OVER': '#2196F3', 'UNDER': '#FF9800'},
                title="Aggregate Win Rate: Over vs Under",
                text='Win Rate (%)'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.add_hline(y=50, line_dash="dash", line_color="red")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Per-Model Breakdown")
            pivot_rows = []
            for model in dir_perf['Model'].unique():
                m = dir_perf[dir_perf['Model'] == model]
                row_data = {'Model': model}
                for d in ['OVER', 'UNDER']:
                    dm = m[m['Direction'] == d]
                    if not dm.empty:
                        r = dm.iloc[0]
                        row_data[f'{d} Bets'] = int(r['Bets'])
                        row_data[f'{d} W/L'] = f"{int(r['Wins'])}/{int(r['Bets'] - r['Wins'])}"
                        row_data[f'{d} WR%'] = r['Win Rate (%)']
                        row_data[f'{d} ROI%'] = r['ROI (%)']
                    else:
                        row_data[f'{d} Bets'] = 0
                        row_data[f'{d} W/L'] = '-'
                        row_data[f'{d} WR%'] = 0.0
                        row_data[f'{d} ROI%'] = 0.0
                pivot_rows.append(row_data)

            pivot_df = pd.DataFrame(pivot_rows)
            roi_cols = [c for c in pivot_df.columns if 'ROI%' in c]
            styled = pivot_df.style.map(
                _style_roi, subset=roi_cols
            ).format({c: '{:.2f}%' for c in pivot_df.columns
                       if 'WR%' in c or 'ROI%' in c})
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Sub-tab 3: By Margin from Line ---
    with sub3:
        st.subheader("Performance by |Predicted - Line| Margin")
        margin_perf = analyzer.analyze_totals_by_margin(tb)
        if not margin_perf.empty:
            if len(margin_perf['Model'].unique()) > 1:
                st.markdown("#### ROI Heatmap (All Models)")
                heatmap_roi = margin_perf.pivot_table(
                    index='Model', columns='Margin', values='ROI (%)')
                ordered_cols = [c for c in MARGIN_ORDER if c in heatmap_roi.columns]
                if ordered_cols:
                    heatmap_roi = heatmap_roi[ordered_cols]
                    fig = px.imshow(
                        heatmap_roi,
                        labels=dict(x="Margin", y="Model", color="ROI (%)"),
                        color_continuous_scale='RdYlGn', aspect="auto",
                        title=f"ROI (%) by Model x Margin ({tag_label})"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Win Rate Heatmap (All Models)")
                heatmap_wr = margin_perf.pivot_table(
                    index='Model', columns='Margin', values='Win Rate (%)')
                ordered_wr = [c for c in MARGIN_ORDER if c in heatmap_wr.columns]
                if ordered_wr:
                    heatmap_wr = heatmap_wr[ordered_wr]
                    fig2 = px.imshow(
                        heatmap_wr,
                        labels=dict(x="Margin", y="Model", color="WR (%)"),
                        color_continuous_scale='Blues', aspect="auto",
                        title=f"Win Rate (%) by Model x Margin ({tag_label})"
                    )
                    fig2.update_layout(height=500)
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Pivot Table (WR% / ROI% per Margin)")
            pivot_rows = []
            for model in sorted(margin_perf['Model'].unique()):
                m = margin_perf[margin_perf['Model'] == model]
                row_data = {'Model': model}
                for bucket in MARGIN_ORDER:
                    bm = m[m['Margin'] == bucket]
                    if not bm.empty:
                        r = bm.iloc[0]
                        row_data[f'{bucket} Bets'] = int(r['Bets'])
                        row_data[f'{bucket} WR%'] = r['Win Rate (%)']
                        row_data[f'{bucket} ROI%'] = r['ROI (%)']
                    else:
                        row_data[f'{bucket} Bets'] = 0
                        row_data[f'{bucket} WR%'] = 0.0
                        row_data[f'{bucket} ROI%'] = 0.0
                pivot_rows.append(row_data)

            pivot_df = pd.DataFrame(pivot_rows)
            roi_cols = [c for c in pivot_df.columns if 'ROI%' in c]
            styled = pivot_df.style.map(
                _style_roi, subset=roi_cols
            ).format({c: '{:.1f}%' for c in pivot_df.columns
                       if 'WR%' in c or 'ROI%' in c})
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Single Model Detail")
            sel_model = st.selectbox(
                "Select model:",
                options=sorted(tb['model'].unique()),
                format_func=str.upper,
                key=f'{key_prefix}_margin_model'
            )
            model_margin = margin_perf[margin_perf['Model'] == sel_model.upper()]
            if not model_margin.empty:
                model_margin = model_margin.copy()
                model_margin['Margin'] = pd.Categorical(
                    model_margin['Margin'], categories=MARGIN_ORDER, ordered=True)
                model_margin = model_margin.sort_values('Margin')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=model_margin['Margin'], y=model_margin['ROI (%)'],
                    marker_color=['green' if v > 0 else 'red'
                                  for v in model_margin['ROI (%)']],
                    text=model_margin['ROI (%)'].apply(lambda v: f'{v:.1f}%'),
                    textposition='outside', name='ROI'
                ))
                fig.add_trace(go.Scatter(
                    x=model_margin['Margin'], y=model_margin['Win Rate (%)'],
                    mode='lines+markers', name='Win Rate',
                    yaxis='y2', line=dict(color='royalblue', width=2)
                ))
                fig.update_layout(
                    height=400,
                    title=f"{sel_model.upper()} - ROI & Win Rate by Margin",
                    yaxis=dict(title='ROI (%)'),
                    yaxis2=dict(title='Win Rate (%)', overlaying='y',
                                side='right', range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Sub-tab 4: Segment Analysis ---
    with sub4:
        st.header(f"Segment Analysis ({tag_label})")
        st.markdown(
            "Detailed analysis of totals model performance across different "
            "margin levels, odds ranges, model consensus, and line levels."
        )

        date_mask_matched = (
            (matched_totals['date'].dt.date >= t_start)
            & (matched_totals['date'].dt.date <= t_end)
        )
        matched_totals_filtered = matched_totals[date_mask_matched].copy()

        all_model_options = ['ENSEMBLE'] + [m.upper() for m in totals_models]
        selected_seg_model = st.selectbox(
            "\U0001F3AF Select model for segment analysis:",
            options=all_model_options,
            key=f'{key_prefix}_seg_model'
        )

        if selected_seg_model:
            seg = analyzer.totals_segment_analysis(
                tb, selected_seg_model, matched_totals_filtered, totals_models)

            if not seg:
                st.warning(f"No segment data for {selected_seg_model}.")
            else:
                seg_subs = st.tabs(
                    [cfg[1] for cfg in _TOTALS_SEG_TAB_CONFIG])

                for (seg_key, _, subtitle, col_label), stab in zip(
                        _TOTALS_SEG_TAB_CONFIG, seg_subs):
                    with stab:
                        st.subheader(subtitle)
                        data = seg.get(seg_key, {})
                        _render_totals_segment_table(data, col_label)

    # --- Sub-tab 5: Detailed Results ---
    with sub5:
        st.subheader("Detailed Totals Betting Results")
        model_filter = st.selectbox(
            "Filter by model:",
            options=['ALL'] + sorted(tb['model'].unique()),
            format_func=lambda x: x.upper(),
            key=f'{key_prefix}_detail_model'
        )
        dir_filter = st.selectbox(
            "Filter by direction:",
            options=['ALL', 'OVER', 'UNDER'],
            key=f'{key_prefix}_detail_dir'
        )

        detail = tb.copy()
        if model_filter != 'ALL':
            detail = detail[detail['model'] == model_filter]
        if dir_filter != 'ALL':
            detail = detail[detail['direction'] == dir_filter]

        display_cols = ['date', 'model', 'home_team', 'away_team',
                        'predicted_total', 'total_line', 'actual_total',
                        'direction', 'bet_odds', 'won', 'actual_profit',
                        'margin']
        existing = [c for c in display_cols if c in detail.columns]
        detail_show = detail[existing].copy()
        detail_show['date'] = detail_show['date'].dt.strftime('%Y-%m-%d')
        for c in ['predicted_total', 'total_line', 'margin']:
            if c in detail_show.columns:
                detail_show[c] = detail_show[c].round(2)
        if 'actual_profit' in detail_show.columns:
            detail_show['actual_profit'] = detail_show['actual_profit'].round(2)
        detail_show.columns = [c.replace('_', ' ').title() for c in detail_show.columns]

        st.dataframe(detail_show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            detail_show.to_csv(index=False).encode('utf-8'),
            file_name=f'totals_results_{tag_label.lower()}.csv',
            mime='text/csv',
            key=f'{key_prefix}_csv'
        )


# =========================================================================
# Cached data loading (all heavy I/O + computation done once)
# =========================================================================

@st.cache_data(ttl=300)
def _load_all_data():
    _a = MLBModelPerformanceAnalyzer()

    results_df = _a.load_game_results()
    if results_df.empty:
        return None

    active_preds = _a.load_matched_predictions(exclude_today=True, model_tag='active')
    active_matched = pd.DataFrame()
    active_models: list = []
    active_betting = pd.DataFrame()
    if not active_preds.empty:
        active_matched = _a.match_predictions_with_results(active_preds, results_df)
        if not active_matched.empty:
            active_models = _a.detect_models(active_matched)
            if active_models:
                active_betting = _a.calculate_betting_roi(active_matched, active_models)

    shadow_preds = _a.load_matched_predictions(exclude_today=True, model_tag='shadow')
    shadow_matched = pd.DataFrame()
    shadow_models: list = []
    shadow_betting = pd.DataFrame()
    if not shadow_preds.empty:
        shadow_matched = _a.match_predictions_with_results(shadow_preds, results_df)
        if not shadow_matched.empty:
            shadow_models = _a.detect_models(shadow_matched)
            if shadow_models:
                shadow_betting = _a.calculate_betting_roi(shadow_matched, shadow_models)

    totals_active = _a.load_matched_totals_predictions(exclude_today=True, model_tag='active')
    totals_shadow = _a.load_matched_totals_predictions(exclude_today=True, model_tag='shadow')

    return {
        'results_df': results_df,
        'active_preds': active_preds,
        'active_matched': active_matched,
        'active_models': active_models,
        'active_betting': active_betting,
        'shadow_preds': shadow_preds,
        'shadow_matched': shadow_matched,
        'shadow_models': shadow_models,
        'shadow_betting': shadow_betting,
        'totals_active': totals_active,
        'totals_shadow': totals_shadow,
    }


# =========================================================================
# Main dashboard
# =========================================================================

def main():
    st.set_page_config(
        page_title="MLB Model Performance Dashboard",
        page_icon="⚾",
        layout="wide"
    )
    
    st.title("⚾ MLB Model Performance Analysis Dashboard")
    st.markdown("---")
    
    analyzer = MLBModelPerformanceAnalyzer()
    
    # --- Data loading (cached) ----------------------------------------------
    with st.spinner("Loading data..."):
        data = _load_all_data()

        if data is None:
            st.error("Cannot load game results. Run mlb_match_analyzer.py first.")
            return

        results_df = data['results_df']
        active_preds = data['active_preds']
        matched_df = data['active_matched']
        active_matched = matched_df
        models = data['active_models']
        betting_results = data['active_betting']
        active_betting = betting_results
        shadow_preds = data['shadow_preds']
        shadow_matched = data['shadow_matched']
        shadow_models = data['shadow_models']
        shadow_betting = data['shadow_betting']
        totals_active_preds = data['totals_active']
        totals_shadow_preds = data['totals_shadow']

        if active_preds.empty:
            st.error("Cannot load active prediction data. Run the pipeline first.")
            return
        if matched_df.empty:
            st.error("Cannot match predictions with results. Predictions may be for today's upcoming games.")
            return
        if not models:
            st.error("No models detected.")
            return

        st.info(f"✅ Detected models: {', '.join([m.upper() for m in models])}")

    st.success(f"✅ Analysis complete: {len(matched_df)} games matched!")
    
    # --- Date filter ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📅 Date Range Filter")
    
    min_date = betting_results['date'].min().date()
    max_date = betting_results['date'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date,
                                    min_value=min_date, max_value=max_date, key='start_date')
    with col2:
        end_date = st.date_input("End Date", value=max_date,
                                  min_value=min_date, max_value=max_date, key='end_date')
    
    date_mask = (betting_results['date'].dt.date >= start_date) & \
                (betting_results['date'].dt.date <= end_date)
    betting_results_filtered = betting_results[date_mask].copy()
    
    total_games = len(betting_results_filtered)
    st.info(f"📊 **Filtered Period:** {start_date} ~ {end_date} | **Total Bets:** {total_games}")
    
    if total_games == 0:
        st.warning("No data in the selected date range.")
        return
    
    st.markdown("---")
    
    # --- Tabs ----------------------------------------------------------------
    (tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9,
     tab10, tab11, tab12, tab13, tab14, tab15) = st.tabs([
        "\U0001F4CA Overall Performance",
        "\U0001F3AF Confidence Analysis",
        "\U0001F4B0 Predicted vs Actual ROI",
        "\U0001F3B2 Odds Range (Active)",
        "\U0001F319 Odds Range (Shadow)",
        "\U0001F4CB Detailed Results",
        "\U0001F504 Active vs Shadow",
        "\U0001F52C Segment Analysis (Active)",
        "\U0001F52C Segment Analysis (Shadow)",
        "\u26BE Totals O/U (Active)",
        "\u26BE Totals O/U (Shadow)",
        "\U0001F3AF Picks (Active)",
        "\U0001F3AF Picks (Shadow)",
        "\u26BE Totals Picks (Active)",
        "\u26BE Totals Picks (Shadow)",
    ])
    
    # =====================================================================
    # Tab 1: Overall Performance
    # =====================================================================
    with tab1:
        st.header("Overall Model Performance")
        
        overall_perf = analyzer.analyze_model_performance(betting_results_filtered)
        
        if overall_perf.empty:
            st.warning("No performance data.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            best = overall_perf.iloc[0]
            with col1:
                st.metric("Best ROI Model", best['Model'])
            with col2:
                st.metric("Best ROI", f"{best['ROI (%)']}%")
            with col3:
                st.metric("Win Rate", f"{best['Win Rate (%)']}%")
            with col4:
                st.metric("Total Profit", f"${best['Total Profit ($)']}")
            
            st.markdown("---")
            st.subheader("📈 Model Performance Summary")
            
            display_perf = overall_perf.copy()
            display_perf['_roi_num'] = display_perf['ROI (%)']
            display_perf['_profit_num'] = display_perf['Total Profit ($)']
            
            display_perf['Win Rate (%)'] = display_perf['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
            display_perf['Total Profit ($)'] = display_perf['Total Profit ($)'].apply(lambda x: f"{x:.2f}")
            display_perf['ROI (%)'] = display_perf['ROI (%)'].apply(lambda x: f"{x:.2f}")
            display_perf['Avg Odds'] = display_perf['Avg Odds'].round(0).astype(int)
            display_perf['Avg Confidence'] = display_perf['Avg Confidence'].apply(lambda x: f"{x:.2f}")
            
            def style_perf(row):
                styles = [''] * len(row)
                roi_idx = display_perf.columns.get_loc('ROI (%)')
                profit_idx = display_perf.columns.get_loc('Total Profit ($)')
                if row['_roi_num'] > 0:
                    styles[roi_idx] = 'color: green; font-weight: bold'
                else:
                    styles[roi_idx] = 'color: red; font-weight: bold'
                if row['_profit_num'] > 0:
                    styles[profit_idx] = 'color: green; font-weight: bold'
                else:
                    styles[profit_idx] = 'color: red; font-weight: bold'
                return styles
            
            st.dataframe(
                display_perf.style.apply(style_perf, axis=1),
                use_container_width=True, height=600,
                column_config={'_roi_num': None, '_profit_num': None}
            )
            
            # ROI comparison chart
            st.subheader("📊 ROI Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=overall_perf['Model'],
                y=overall_perf['ROI (%)'],
                marker_color=['green' if x > 0 else 'red' for x in overall_perf['ROI (%)']],
                text=overall_perf['ROI (%)'].round(2),
                textposition='outside'
            ))
            fig.update_layout(
                title="Model ROI Comparison (%)",
                xaxis_title="Model", yaxis_title="ROI (%)",
                showlegend=False, height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(overall_perf, x='Model', y='Win Rate (%)',
                             title='Win Rate by Model',
                             color='Win Rate (%)', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(overall_perf, x='Win Rate (%)', y='ROI (%)',
                                 size='Total Bets', color='Model',
                                 title='Win Rate vs ROI', hover_data=['Total Bets'])
                st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================================
    # Tab 2: Confidence Analysis
    # =====================================================================
    with tab2:
        st.header("Performance by Confidence Level")
        
        confidence_perf = analyzer.analyze_by_confidence(betting_results_filtered)
        
        if confidence_perf.empty:
            st.warning("No confidence analysis data.")
        else:
            selected_model = st.selectbox(
                "Select Model to Analyze",
                options=[m.upper() for m in models],
                key='confidence_model_select'
            )
            
            if selected_model:
                model_conf = confidence_perf[confidence_perf['Model'] == selected_model].copy()
                
                if not model_conf.empty:
                    conf_order = ['50-60%', '60-70%', '70-80%', '80%+']
                    model_conf['Confidence'] = pd.Categorical(
                        model_conf['Confidence'], categories=conf_order, ordered=True
                    )
                    model_conf = model_conf.sort_values('Confidence')
                    
                    st.subheader(f"📊 {selected_model} Performance by Confidence")
                    cols = st.columns(len(model_conf))
                    for idx, (_, row) in enumerate(model_conf.iterrows()):
                        with cols[idx]:
                            st.metric(
                                label=row['Confidence'],
                                value=f"{row['ROI (%)']:.2f}%",
                                delta=f"{row['Win Rate (%)']:.1f}% Win Rate"
                            )
                            st.caption(f"{int(row['Bets'])} bets, {int(row['Wins'])} wins")
                    
                    st.markdown("---")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=model_conf['Confidence'], y=model_conf['ROI (%)'],
                            marker_color=['green' if x > 0 else 'red' for x in model_conf['ROI (%)']],
                            text=model_conf['ROI (%)'].round(2), textposition='outside'
                        ))
                        fig.update_layout(
                            title=f"{selected_model} - ROI by Confidence",
                            xaxis_title="Confidence Level", yaxis_title="ROI (%)",
                            showlegend=False, height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=model_conf['Confidence'], y=model_conf['Win Rate (%)'],
                            marker_color='lightblue',
                            text=model_conf['Win Rate (%)'].round(1), textposition='outside'
                        ))
                        fig.update_layout(
                            title=f"{selected_model} - Win Rate by Confidence",
                            xaxis_title="Confidence Level", yaxis_title="Win Rate (%)",
                            showlegend=False, height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail table
                    st.subheader("📋 Detailed Statistics")
                    display_conf = model_conf.copy()
                    display_conf['_roi_num'] = display_conf['ROI (%)']
                    display_conf['Bets'] = display_conf['Bets'].astype(int)
                    display_conf['Wins'] = display_conf['Wins'].astype(int)
                    display_conf['Win Rate (%)'] = display_conf['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                    display_conf['ROI (%)'] = display_conf['ROI (%)'].apply(lambda x: f"{x:.2f}")
                    
                    def style_conf(row):
                        styles = [''] * len(row)
                        roi_idx = display_conf.columns.get_loc('ROI (%)')
                        if row['_roi_num'] > 0:
                            styles[roi_idx] = 'color: green; font-weight: bold'
                        else:
                            styles[roi_idx] = 'color: red; font-weight: bold'
                        return styles
                    
                    st.dataframe(
                        display_conf.style.apply(style_conf, axis=1),
                        use_container_width=True,
                        column_config={'_roi_num': None}
                    )
                else:
                    st.info(f"No confidence data for {selected_model}.")
        
        # All model heatmap
        if not confidence_perf.empty:
            st.markdown("---")
            st.subheader("🔄 Compare All Models")
            
            pivot_roi = confidence_perf.pivot(index='Model', columns='Confidence', values='ROI (%)')
            pivot_roi = pivot_roi.reindex(columns=['50-60%', '60-70%', '70-80%', '80%+'])
            
            fig = px.imshow(
                pivot_roi,
                labels=dict(x="Confidence Level", y="Model", color="ROI (%)"),
                color_continuous_scale='RdYlGn', aspect="auto",
                title="ROI (%) Heatmap - All Models by Confidence Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================================
    # Tab 3: Predicted vs Actual ROI
    # =====================================================================
    with tab3:
        st.header("Predicted ROI vs Actual ROI Analysis")
        
        roi_analysis = analyzer.analyze_by_predicted_roi(betting_results_filtered)
        
        if roi_analysis.empty:
            st.warning("No ROI analysis data.")
        else:
            selected_model_roi = st.selectbox(
                "Select Model", options=[m.upper() for m in models],
                key='roi_model_select'
            )
            
            if selected_model_roi:
                model_roi = roi_analysis[roi_analysis['Model'] == selected_model_roi].copy()
                
                if not model_roi.empty:
                    roi_order = ['Negative', '0-10%', '10-20%', '20%+']
                    model_roi['Predicted ROI'] = pd.Categorical(
                        model_roi['Predicted ROI'], categories=roi_order, ordered=True
                    )
                    model_roi = model_roi.sort_values('Predicted ROI')
                    
                    st.subheader(f"📊 {selected_model_roi} Performance by Predicted ROI Bucket")
                    cols = st.columns(len(model_roi))
                    for idx, (_, row) in enumerate(model_roi.iterrows()):
                        with cols[idx]:
                            st.metric(
                                label=row['Predicted ROI'],
                                value=f"{row['Actual ROI (%)']:.2f}%",
                                delta=f"Pred: {row['Avg Pred ROI (%)']:.2f}%"
                            )
                            st.caption(f"{int(row['Bets'])} bets, {int(row['Wins'])} wins")
                    
                    st.markdown("---")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Predicted ROI (Avg)',
                        x=model_roi['Predicted ROI'],
                        y=model_roi['Avg Pred ROI (%)'],
                        marker_color='lightblue',
                        text=model_roi['Avg Pred ROI (%)'].round(2), textposition='outside'
                    ))
                    fig.add_trace(go.Bar(
                        name='Actual ROI',
                        x=model_roi['Predicted ROI'],
                        y=model_roi['Actual ROI (%)'],
                        marker_color=['green' if x > 0 else 'red' for x in model_roi['Actual ROI (%)']],
                        text=model_roi['Actual ROI (%)'].round(2), textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"{selected_model_roi} - Predicted vs Actual ROI",
                        xaxis_title="Predicted ROI Bucket", yaxis_title="ROI (%)",
                        barmode='group', height=500, showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail table
                    st.subheader("📋 Detailed ROI Bucket Statistics")
                    display_roi = model_roi.copy()
                    display_roi['_actual_roi_num'] = display_roi['Actual ROI (%)']
                    display_roi['_pred_roi_num'] = display_roi['Avg Pred ROI (%)']
                    display_roi['Bets'] = display_roi['Bets'].astype(int)
                    display_roi['Wins'] = display_roi['Wins'].astype(int)
                    display_roi['Win Rate (%)'] = display_roi['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                    display_roi['Avg Pred ROI (%)'] = display_roi['Avg Pred ROI (%)'].apply(lambda x: f"{x:.2f}")
                    display_roi['Actual ROI (%)'] = display_roi['Actual ROI (%)'].apply(lambda x: f"{x:.2f}")
                    
                    def style_roi(row):
                        styles = [''] * len(row)
                        a_idx = display_roi.columns.get_loc('Actual ROI (%)')
                        p_idx = display_roi.columns.get_loc('Avg Pred ROI (%)')
                        if row['_actual_roi_num'] > 0:
                            styles[a_idx] = 'color: green; font-weight: bold'
                        else:
                            styles[a_idx] = 'color: red; font-weight: bold'
                        if row['_pred_roi_num'] > 0:
                            styles[p_idx] = 'color: blue; font-weight: bold'
                        elif row['_pred_roi_num'] < 0:
                            styles[p_idx] = 'color: orange; font-weight: bold'
                        return styles
                    
                    st.dataframe(
                        display_roi.style.apply(style_roi, axis=1),
                        use_container_width=True,
                        column_config={'_actual_roi_num': None, '_pred_roi_num': None}
                    )
                else:
                    st.info(f"No ROI data for {selected_model_roi}.")
        
        if not roi_analysis.empty:
            st.markdown("---")
            st.subheader("🔄 Compare All Models")
            
            pivot_actual = roi_analysis.pivot(index='Model', columns='Predicted ROI', values='Actual ROI (%)')
            pivot_actual = pivot_actual.reindex(columns=['Negative', '0-10%', '10-20%', '20%+'])
            
            fig = px.imshow(
                pivot_actual,
                labels=dict(x="Predicted ROI Bucket", y="Model", color="Actual ROI (%)"),
                color_continuous_scale='RdYlGn', aspect="auto",
                title="Actual ROI Heatmap by Predicted ROI Bucket"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================================
    # Tab 4: Odds Range (Active)
    # =====================================================================
    with tab4:
        st.header("🎲 Performance by Odds Range (Active)")
        
        odds_analysis = analyzer.analyze_by_odds(betting_results_filtered)
        render_odds_range_tab(odds_analysis, models, 'odds_active_model', 'Active')
    
    # =====================================================================
    # Tab 5: Odds Range (Shadow)
    # =====================================================================
    with tab5:
        st.header("🌙 Performance by Odds Range (Shadow)")

        if shadow_preds.empty:
            st.warning("⚠️ No Shadow prediction data yet. Data will appear after running the pipeline with shadow models.")
            st.info("Shadow data is generated by running `python mlb_betting_pipeline.py --model-tag both`")
        elif shadow_matched.empty:
            st.warning("No Shadow predictions match completed game results yet.")
        elif shadow_betting.empty:
            st.warning("No Shadow betting data available.")
        else:
            shadow_odds = analyzer.analyze_by_odds(shadow_betting)
            render_odds_range_tab(shadow_odds, shadow_models, 'odds_shadow_model', 'Shadow')
    
    # =====================================================================
    # Tab 6: Detailed Results
    # =====================================================================
    with tab6:
        st.header("📋 Detailed Betting Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                options=['All'] + [m.upper() for m in models],
                key='detail_model_filter'
            )
        
        with col2:
            filter_result = st.selectbox(
                "Filter by Result",
                options=['All', 'Won', 'Lost']
            )
        
        with col3:
            filter_confidence = st.selectbox(
                "Filter by Confidence",
                options=['All', '50-60%', '60-70%', '70-80%', '80%+']
            )
        
        col4, _ = st.columns(2)
        with col4:
            filter_odds = st.selectbox(
                "Filter by Odds Range",
                options=['All'] + ODDS_ORDER
            )
        
        filtered = betting_results_filtered.copy()
        
        if filter_model != 'All':
            filtered = filtered[filtered['model'] == filter_model.lower()]
        if filter_result == 'Won':
            filtered = filtered[filtered['won'] == True]
        elif filter_result == 'Lost':
            filtered = filtered[filtered['won'] == False]
        if filter_confidence != 'All':
            filtered = filtered[filtered['confidence_level'] == filter_confidence]
        if filter_odds != 'All':
            filtered = filtered[filtered['odds_bucket'] == filter_odds]
        
        st.subheader(f"📋 Showing {len(filtered)} bets")
        
        if not filtered.empty:
            display_cols = [
                'date', 'model', 'home_team', 'away_team', 'bet_team',
                'bet_probability', 'bet_odds', 'odds_bucket', 'predicted_roi_pct',
                'actual_profit', 'actual_roi_pct', 'won', 'confidence_level'
            ]
            
            disp = filtered[display_cols].copy()
            disp['_won'] = disp['won']
            disp['_profit_num'] = disp['actual_profit']
            disp['_roi_num'] = disp['actual_roi_pct']
            
            disp['date'] = disp['date'].dt.strftime('%Y-%m-%d')
            disp['model'] = disp['model'].str.upper()
            disp['bet_probability'] = (disp['bet_probability'] * 100).apply(lambda x: f"{x:.2f}")
            disp['bet_odds'] = disp['bet_odds'].round(0).astype(int)
            disp['predicted_roi_pct'] = disp['predicted_roi_pct'].apply(lambda x: f"{x:.2f}")
            disp['actual_profit'] = disp['actual_profit'].apply(lambda x: f"{x:.2f}")
            disp['actual_roi_pct'] = disp['actual_roi_pct'].apply(lambda x: f"{x:.2f}")
            
            disp.columns = [
                'Date', 'Model', 'Home Team', 'Away Team', 'Bet On',
                'Confidence (%)', 'Odds', 'Odds Range', 'Pred ROI (%)',
                'Profit ($)', 'Actual ROI (%)', 'Won', 'Confidence Level',
                '_won', '_profit_num', '_roi_num'
            ]
            
            def style_results(row):
                bg = 'background-color: #d4edda' if row['_won'] else 'background-color: #f8d7da'
                styles = [bg] * len(row)
                profit_idx = disp.columns.get_loc('Profit ($)')
                roi_idx = disp.columns.get_loc('Actual ROI (%)')
                if row['_profit_num'] > 0:
                    styles[profit_idx] = f'{bg}; color: green; font-weight: bold'
                else:
                    styles[profit_idx] = f'{bg}; color: red; font-weight: bold'
                if row['_roi_num'] > 0:
                    styles[roi_idx] = f'{bg}; color: green; font-weight: bold'
                else:
                    styles[roi_idx] = f'{bg}; color: red; font-weight: bold'
                return styles
            
            st.dataframe(
                disp.style.apply(style_results, axis=1),
                use_container_width=True, height=600,
                column_config={'_won': None, '_profit_num': None, '_roi_num': None}
            )
            
            csv = disp.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"mlb_model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data matches the current filters.")
    
    # =====================================================================
    # Tab 7: Active vs Shadow
    # =====================================================================
    with tab7:
        st.header("🔄 Active vs Shadow Model Comparison")
        st.markdown("""
        **Active**: Currently deployed models (used for actual betting)  
        **Shadow**: New models being tested in parallel (pending promotion)
        """)
        
        has_active = not active_preds.empty
        has_shadow = not shadow_preds.empty
        
        if not has_active and not has_shadow:
            st.warning("No Active/Shadow tagged data found.")
            st.info("""
            **How to set up Active/Shadow:**
            1. Run `python mlb_betting_pipeline.py --model-tag both`
            2. Or `python mlb_switch_and_train.py --action train` to train shadow models
            """)
        else:
            c1, c2 = st.columns(2)
            with c1:
                if has_active:
                    st.success(f"Active data: {len(active_preds)} predictions")
                else:
                    st.warning("No Active data")
            with c2:
                if has_shadow:
                    st.success(f"Shadow data: {len(shadow_preds)} predictions")
                else:
                    st.warning("No Shadow data")
            
            if has_active and has_shadow:
                st.markdown("---")
                st.subheader("📊 Performance Comparison")
                
                active_matched_t7 = active_matched.copy()
                shadow_matched_t7 = shadow_matched.copy()
                
                # Date filter for A/S comparison
                if not active_matched_t7.empty or not shadow_matched_t7.empty:
                    st.markdown("---")
                    st.subheader("📅 Date Range Filter (A/S)")
                    
                    all_dates = []
                    if not active_matched_t7.empty:
                        active_matched_t7['date'] = pd.to_datetime(active_matched_t7['date'])
                        all_dates.extend(active_matched_t7['date'].tolist())
                    if not shadow_matched_t7.empty:
                        shadow_matched_t7['date'] = pd.to_datetime(shadow_matched_t7['date'])
                        all_dates.extend(shadow_matched_t7['date'].tolist())
                    
                    if all_dates:
                        as_min = min(all_dates).date()
                        as_max = max(all_dates).date()
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            as_start = st.date_input("Start Date (A/S)", value=as_min,
                                                      min_value=as_min, max_value=as_max, key='as_start')
                        with c2:
                            as_end = st.date_input("End Date (A/S)", value=as_max,
                                                    min_value=as_min, max_value=as_max, key='as_end')
                        
                        if not active_matched_t7.empty:
                            mask = (active_matched_t7['date'].dt.date >= as_start) & \
                                   (active_matched_t7['date'].dt.date <= as_end)
                            active_matched_t7 = active_matched_t7[mask]
                        
                        if not shadow_matched_t7.empty:
                            mask = (shadow_matched_t7['date'].dt.date >= as_start) & \
                                   (shadow_matched_t7['date'].dt.date <= as_end)
                            shadow_matched_t7 = shadow_matched_t7[mask]
                        
                        st.info(f"Filter: {as_start} ~ {as_end} | Active: {len(active_matched_t7)} games, Shadow: {len(shadow_matched_t7)} games")
                    
                    st.markdown("---")
                
                # Use pre-computed betting data with date filter
                if not active_matched_t7.empty and not active_betting.empty:
                    active_betting_t7 = active_betting[
                        (active_betting['date'].dt.date >= as_start) &
                        (active_betting['date'].dt.date <= as_end)
                    ].copy()
                    active_perf = analyzer.analyze_model_performance(active_betting_t7)
                    active_perf['Type'] = 'Active'
                else:
                    active_perf = pd.DataFrame()
                
                if not shadow_matched_t7.empty and not shadow_betting.empty:
                    shadow_betting_t7 = shadow_betting[
                        (shadow_betting['date'].dt.date >= as_start) &
                        (shadow_betting['date'].dt.date <= as_end)
                    ].copy()
                    shadow_perf = analyzer.analyze_model_performance(shadow_betting_t7)
                    shadow_perf['Type'] = 'Shadow'
                else:
                    shadow_perf = pd.DataFrame()
                
                if not active_perf.empty and not shadow_perf.empty:
                    comparison = pd.concat([active_perf, shadow_perf], ignore_index=True)
                    
                    st.subheader("📈 ROI Comparison by Model")
                    
                    pivot_cmp = comparison.pivot(
                        index='Model', columns='Type', values='ROI (%)'
                    ).reset_index()
                    
                    if 'Active' in pivot_cmp.columns and 'Shadow' in pivot_cmp.columns:
                        pivot_cmp['Difference'] = pivot_cmp['Shadow'] - pivot_cmp['Active']
                        pivot_cmp['Better'] = pivot_cmp['Difference'].apply(
                            lambda x: 'Shadow' if x > 0 else ('Active' if x < 0 else 'Same')
                        )
                        
                        st.dataframe(pivot_cmp, use_container_width=True)
                        
                        # Bar chart comparison
                        fig = go.Figure()
                        models_list = pivot_cmp['Model'].tolist()
                        active_roi = pivot_cmp['Active'].tolist()
                        shadow_roi = pivot_cmp['Shadow'].tolist()
                        
                        fig.add_trace(go.Bar(
                            name='Active', x=models_list, y=active_roi,
                            marker_color='royalblue',
                            text=[f"{x:.2f}%" for x in active_roi], textposition='outside'
                        ))
                        fig.add_trace(go.Bar(
                            name='Shadow', x=models_list, y=shadow_roi,
                            marker_color='darkorange',
                            text=[f"{x:.2f}%" for x in shadow_roi], textposition='outside'
                        ))
                        fig.update_layout(
                            title="Active vs Shadow ROI Comparison",
                            xaxis_title="Model", yaxis_title="ROI (%)",
                            barmode='group', height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Win rate comparison
                        st.subheader("📊 Win Rate Comparison")
                        win_pivot = comparison.pivot(
                            index='Model', columns='Type', values='Win Rate (%)'
                        ).reset_index()
                        
                        if 'Active' in win_pivot.columns and 'Shadow' in win_pivot.columns:
                            fig_win = go.Figure()
                            fig_win.add_trace(go.Bar(
                                name='Active', x=win_pivot['Model'].tolist(),
                                y=win_pivot['Active'].tolist(), marker_color='royalblue'
                            ))
                            fig_win.add_trace(go.Bar(
                                name='Shadow', x=win_pivot['Model'].tolist(),
                                y=win_pivot['Shadow'].tolist(), marker_color='darkorange'
                            ))
                            fig_win.update_layout(
                                title="Active vs Shadow Win Rate Comparison",
                                xaxis_title="Model", yaxis_title="Win Rate (%)",
                                barmode='group', height=400
                            )
                            st.plotly_chart(fig_win, use_container_width=True)
                        
                        # Recommendation
                        st.markdown("---")
                        st.subheader("💡 Recommendation")
                        
                        active_avg = active_perf['ROI (%)'].mean() if not active_perf.empty else 0
                        shadow_avg = shadow_perf['ROI (%)'].mean() if not shadow_perf.empty else 0
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Active Avg ROI", f"{active_avg:.2f}%")
                        with c2:
                            st.metric("Shadow Avg ROI", f"{shadow_avg:.2f}%")
                        with c3:
                            diff = shadow_avg - active_avg
                            if diff > 2:
                                st.success(f"Shadow is {diff:.2f}% better\n\n**Recommend switching!**")
                            elif diff < -2:
                                st.error(f"Active is {-diff:.2f}% better\n\n**Keep current models**")
                            else:
                                st.warning(f"Difference: {abs(diff):.2f}%\n\n**Need more data**")
                    else:
                        st.info("Not enough data for comparison.")
                else:
                    if active_perf.empty:
                        st.warning("Active matched data is empty.")
                    if shadow_perf.empty:
                        st.warning("Shadow matched data is empty.")
            
            elif has_active:
                st.info("Only Active data available. Run shadow predictions for comparison.")
            elif has_shadow:
                st.info("Only Shadow data available. Run active predictions for comparison.")

    # =====================================================================
    # Tab 8: Segment Analysis (Active)
    # =====================================================================
    with tab8:
        if not active_betting.empty:
            seg_a_min = active_betting['date'].min().date()
            seg_a_max = active_betting['date'].max().date()
            st.markdown("##### \U0001F4C5 Date Range")
            c1, c2 = st.columns(2)
            with c1:
                seg_a_start = st.date_input("Start", value=seg_a_min,
                    min_value=seg_a_min, max_value=seg_a_max, key='seg_a_start')
            with c2:
                seg_a_end = st.date_input("End", value=seg_a_max,
                    min_value=seg_a_min, max_value=seg_a_max, key='seg_a_end')
            seg_a_mask = (
                (active_betting['date'].dt.date >= seg_a_start)
                & (active_betting['date'].dt.date <= seg_a_end)
            )
            seg_a_filtered = active_betting[seg_a_mask].copy()
            if seg_a_filtered.empty:
                st.warning("No data in the selected date range.")
            else:
                render_segment_analysis_tab(
                    analyzer, seg_a_filtered, models,
                    key_prefix='active', tag_label='Active')
        else:
            st.warning("No Active betting data available.")

    # =====================================================================
    # Tab 9: Segment Analysis (Shadow)
    # =====================================================================
    with tab9:
        if shadow_preds.empty:
            st.header("\U0001F52C Segment Analysis (Shadow)")
            st.warning(
                "\u26A0\uFE0F No Shadow prediction data yet. "
                "Data will appear after running the pipeline with shadow models."
            )
            st.info(
                "Run `python mlb_betting_pipeline.py --model-tag both` "
                "or `python mlb_switch_and_train.py --action train`"
            )
        elif shadow_matched.empty:
            st.header("\U0001F52C Segment Analysis (Shadow)")
            st.warning("No Shadow predictions match completed game results yet.")
        elif shadow_betting.empty:
            st.header("\U0001F52C Segment Analysis (Shadow)")
            st.warning("No Shadow betting data available.")
        else:
            seg_s_min = shadow_betting['date'].min().date()
            seg_s_max = shadow_betting['date'].max().date()
            st.markdown("##### \U0001F4C5 Date Range")
            c1, c2 = st.columns(2)
            with c1:
                seg_s_start = st.date_input("Start", value=seg_s_min,
                    min_value=seg_s_min, max_value=seg_s_max, key='seg_s_start')
            with c2:
                seg_s_end = st.date_input("End", value=seg_s_max,
                    min_value=seg_s_min, max_value=seg_s_max, key='seg_s_end')
            seg_s_mask = (
                (shadow_betting['date'].dt.date >= seg_s_start)
                & (shadow_betting['date'].dt.date <= seg_s_end)
            )
            shadow_filtered_t9 = shadow_betting[seg_s_mask].copy()

            if shadow_filtered_t9.empty:
                st.header("\U0001F52C Segment Analysis (Shadow)")
                st.warning("No Shadow data in the selected date range.")
            else:
                render_segment_analysis_tab(
                    analyzer, shadow_filtered_t9, shadow_models,
                    key_prefix='shadow', tag_label='Shadow')


    # =====================================================================
    # Tab 10: Totals O/U (Active)
    # =====================================================================
    with tab10:
        render_totals_tab(
            analyzer, totals_active_preds, results_df,
            key_prefix='totals_active', tag_label='Active')

    # =====================================================================
    # Tab 11: Totals O/U (Shadow)
    # =====================================================================
    with tab11:
        render_totals_tab(
            analyzer, totals_shadow_preds, results_df,
            key_prefix='totals_shadow', tag_label='Shadow')

    # =====================================================================
    # Tab 12: Picks (Active)
    # =====================================================================
    with tab12:
        render_picks_tab(analyzer, key_prefix='picks_active', tag_label='Active')

    # =====================================================================
    # Tab 13: Picks (Shadow)
    # =====================================================================
    with tab13:
        render_picks_tab(analyzer, key_prefix='picks_shadow', tag_label='Shadow')

    # =====================================================================
    # Tab 14: Totals Picks (Active)
    # =====================================================================
    with tab14:
        render_totals_picks_tab(
            analyzer, key_prefix='totals_picks_active', tag_label='Active')

    # =====================================================================
    # Tab 15: Totals Picks (Shadow)
    # =====================================================================
    with tab15:
        render_totals_picks_tab(
            analyzer, key_prefix='totals_picks_shadow', tag_label='Shadow')


if __name__ == '__main__':
    main()
