import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import argparse


class TotalsOddsMatcher:
    """MLB 토탈 예측 결과와 Over/Under 배당률 매칭"""
    
    def __init__(self):
        self.MLB_TEAM_ABBREV = {
            'Arizona Diamondbacks': 'ARI',
            'Atlanta Braves': 'ATL',
            'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC',
            'Chicago White Sox': 'CWS',
            'Cincinnati Reds': 'CIN',
            'Cleveland Guardians': 'CLE',
            'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU',
            'Kansas City Royals': 'KC',
            'Los Angeles Angels': 'LAA',
            'Los Angeles Dodgers': 'LAD',
            'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL',
            'Minnesota Twins': 'MIN',
            'New York Mets': 'NYM',
            'New York Yankees': 'NYY',
            'Oakland Athletics': 'OAK',
            'Athletics': 'OAK',
            'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT',
            'San Diego Padres': 'SD',
            'San Francisco Giants': 'SF',
            'Seattle Mariners': 'SEA',
            'St. Louis Cardinals': 'STL',
            'Tampa Bay Rays': 'TB',
            'Texas Rangers': 'TEX',
            'Toronto Blue Jays': 'TOR',
            'Washington Nationals': 'WSH'
        }
        self.MLB_TEAM_FULL = {v: k for k, v in self.MLB_TEAM_ABBREV.items()}
    
    def match_odds(self, prediction_file, odds_file):
        """토탈 예측 결과와 Over/Under 배당률 매칭"""
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        
        with open(odds_file, 'r') as f:
            odds = json.load(f)
            
        odds_df = pd.DataFrame(odds)
        
        matched_predictions = []
        
        for pred in predictions:
            date = pred['date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            home_team_abbrev = self.MLB_TEAM_ABBREV.get(home_team, home_team)
            away_team_abbrev = self.MLB_TEAM_ABBREV.get(away_team, away_team)
            
            game_odds = odds_df[
                (odds_df['date'] == date) & 
                (odds_df['home_team'] == home_team_abbrev) & 
                (odds_df['away_team'] == away_team_abbrev)
            ]
            
            matched_pred = pred.copy()
            
            if not game_odds.empty:
                odds_row = game_odds.iloc[0]
                total_line = float(odds_row['total_line']) if pd.notna(odds_row.get('total_line')) else None
                over_odds = float(odds_row['over_odds']) if pd.notna(odds_row.get('over_odds')) else None
                under_odds = float(odds_row['under_odds']) if pd.notna(odds_row.get('under_odds')) else None
                
                matched_pred['total_line'] = total_line
                matched_pred['over_odds'] = over_odds
                matched_pred['under_odds'] = under_odds
                matched_pred['bookmaker'] = odds_row.get('bookmaker')
                matched_pred['bookmaker_title'] = odds_row.get('bookmaker_title')
                
                predicted_total = pred.get('predicted_total') or pred.get('ensemble_total')
                if predicted_total is not None and total_line is not None:
                    if predicted_total > total_line:
                        matched_pred['predicted_direction'] = 'Over'
                        matched_pred['predicted_direction_odds'] = over_odds
                    elif predicted_total < total_line:
                        matched_pred['predicted_direction'] = 'Under'
                        matched_pred['predicted_direction_odds'] = under_odds
                    else:
                        matched_pred['predicted_direction'] = 'Push'
                        matched_pred['predicted_direction_odds'] = None
                    
                    matched_pred['predicted_margin'] = round(predicted_total - total_line, 2)
                else:
                    matched_pred['predicted_direction'] = None
                    matched_pred['predicted_direction_odds'] = None
                    matched_pred['predicted_margin'] = None
            else:
                matched_pred['total_line'] = None
                matched_pred['over_odds'] = None
                matched_pred['under_odds'] = None
                matched_pred['bookmaker'] = None
                matched_pred['bookmaker_title'] = None
                matched_pred['predicted_direction'] = None
                matched_pred['predicted_direction_odds'] = None
                matched_pred['predicted_margin'] = None
            
            matched_predictions.append(matched_pred)
        
        def get_sort_key(pred):
            try:
                if 'start_time' in pred and pred['start_time']:
                    return pd.to_datetime(pred['start_time'])
                else:
                    return pd.to_datetime(pred['date'])
            except:
                return pd.to_datetime(pred['date'])
        
        matched_predictions.sort(key=get_sort_key)
        
        return matched_predictions
    
    def save_matched_predictions(self, matched_predictions, output_dir=None, model_tag=None):
        """매칭된 토탈 예측 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'data' / 'matched'
        else:
            output_dir = Path(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if model_tag:
            filename = output_dir / f'mlb_totals_predictions_with_odds_{timestamp}_{model_tag}.json'
        else:
            filename = output_dir / f'mlb_totals_predictions_with_odds_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(matched_predictions, f, indent=2)
            
        print(f"\nTotals matched predictions saved to: {filename}")
        return filename


def run_for_tag(model_tag=None):
    """특정 태그의 토탈 예측 파일을 오즈와 매칭"""
    pred_dir = Path(__file__).parent.parent / 'predictions'
    odds_dir = Path(__file__).parent / 'data' / 'odds'
    
    if model_tag:
        prediction_files = list(pred_dir.glob(f'mlb_totals_ensemble_predictions_*_{model_tag}.json'))
        if not prediction_files:
            prediction_files = list(pred_dir.glob('mlb_totals_ensemble_predictions_*.json'))
            prediction_files = [f for f in prediction_files if 'active' not in f.name and 'shadow' not in f.name]
        print(f"\n=== [{model_tag.upper()}] 토탈 오즈 매칭 ===")
    else:
        prediction_files = list(pred_dir.glob('mlb_totals_ensemble_predictions_*.json'))
    
    odds_files = list(odds_dir.glob('processed_mlb_totals_odds_*.json'))
    
    if not prediction_files or not odds_files:
        print(f"No totals prediction or odds files found (tag={model_tag})")
        return
    
    latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
    latest_odds = max(odds_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using prediction file: {latest_prediction.name}")
    print(f"Using odds file: {latest_odds.name}")
    
    matcher = TotalsOddsMatcher()
    matched_predictions = matcher.match_odds(latest_prediction, latest_odds)
    output_file = matcher.save_matched_predictions(matched_predictions, model_tag=model_tag)
    
    if matched_predictions:
        print("\nSample matched totals prediction:")
        print(json.dumps(matched_predictions[0], indent=2))
    
    return output_file


def main():
    """토탈 오즈 매칭 실행"""
    parser = argparse.ArgumentParser(description='MLB 토탈 예측-오즈 매칭')
    parser.add_argument('--model-tag', type=str, default=None,
                       choices=['active', 'shadow', 'both'],
                       help='모델 태그 (active, shadow, both)')
    args = parser.parse_args()
    
    if args.model_tag == 'both':
        for tag in ['active', 'shadow']:
            try:
                run_for_tag(tag)
            except Exception as e:
                print(f"\n[{tag.upper()}] 토탈 오즈 매칭 오류: {e}")
    else:
        run_for_tag(args.model_tag)


if __name__ == "__main__":
    main()
