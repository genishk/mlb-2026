import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

class OddsMatcher:
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
        # 팀명 약자 -> 전체 이름 매핑 생성
        self.MLB_TEAM_FULL = {v: k for k, v in self.MLB_TEAM_ABBREV.items()}
    
    def match_odds(self, prediction_file, odds_file):
        """예측 결과와 배당률 매칭"""
        # 파일 로드
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        
        with open(odds_file, 'r') as f:
            odds = json.load(f)
            
        # odds를 DataFrame으로 변환
        odds_df = pd.DataFrame(odds)
        
        # 결과를 저장할 리스트
        matched_predictions = []
        
        for pred in predictions:
            date = pred['date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            home_team_abbrev = self.MLB_TEAM_ABBREV.get(home_team)
            away_team_abbrev = self.MLB_TEAM_ABBREV.get(away_team)
            
            # 해당 경기의 odds 찾기
            game_odds = odds_df[
                (odds_df['date'] == date) & 
                (odds_df['home_team'] == home_team_abbrev) & 
                (odds_df['away_team'] == away_team_abbrev)
            ]
            
            # odds 정보 추출
            home_odds_info = game_odds[game_odds['is_home'] == True].iloc[0] if not game_odds[game_odds['is_home'] == True].empty else None
            away_odds_info = game_odds[game_odds['is_home'] == False].iloc[0] if not game_odds[game_odds['is_home'] == False].empty else None
            
            # 예측 결과에 odds 정보 추가
            matched_pred = pred.copy()
            
            # 홈팀 odds 정보 추가
            if home_odds_info is not None:
                matched_pred.update({
                    'home_team_odds': float(home_odds_info['odds']),
                    'home_team_probability_odds': float(home_odds_info['probability']),
                })
            else:
                matched_pred.update({
                    'home_team_odds': None,
                    'home_team_probability_odds': None,
                })
            
            # 원정팀 odds 정보 추가
            if away_odds_info is not None:
                matched_pred.update({
                    'away_team_odds': float(away_odds_info['odds']),
                    'away_team_probability_odds': float(away_odds_info['probability']),
                })
            else:
                matched_pred.update({
                    'away_team_odds': None,
                    'away_team_probability_odds': None,
                })
            
            # 예측된 승자의 odds 정보 추가
            if matched_pred['predicted_winner'] == home_team and home_odds_info is not None:
                matched_pred['predicted_winner_odds'] = float(home_odds_info['odds'])
                matched_pred['predicted_winner_probability_odds'] = float(home_odds_info['probability'])
            elif matched_pred['predicted_winner'] == away_team and away_odds_info is not None:
                matched_pred['predicted_winner_odds'] = float(away_odds_info['odds'])
                matched_pred['predicted_winner_probability_odds'] = float(away_odds_info['probability'])
            else:
                matched_pred['predicted_winner_odds'] = None
                matched_pred['predicted_winner_probability_odds'] = None
            
            # 배당사 정보 추가
            if not game_odds.empty:
                matched_pred['bookmaker'] = game_odds.iloc[0]['bookmaker']
                matched_pred['bookmaker_title'] = game_odds.iloc[0]['bookmaker_title']
            else:
                matched_pred['bookmaker'] = None
                matched_pred['bookmaker_title'] = None
            
            matched_predictions.append(matched_pred)
        
        # 시작 시간 순으로 정렬
        def get_sort_key(pred):
            """정렬 키 생성: 날짜 + 시작시간"""
            try:
                # start_time이 있으면 사용
                if 'start_time' in pred and pred['start_time']:
                    return pd.to_datetime(pred['start_time'])
                # start_time이 없으면 날짜만 사용
                else:
                    return pd.to_datetime(pred['date'])
            except:
                # 파싱 실패 시 날짜만 사용
                return pd.to_datetime(pred['date'])
        
        matched_predictions.sort(key=get_sort_key)
        
        return matched_predictions
    
    def save_matched_predictions(self, matched_predictions, output_dir=None):
        """매칭된 예측 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'data' / 'matched'
        else:
            output_dir = Path(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'mlb_predictions_with_odds_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(matched_predictions, f, indent=2)
            
        print(f"\nMatched predictions saved to: {filename}")
        return filename

def main():
    """테스트 실행"""
    # 가장 최신 파일 찾기
    pred_dir = Path(__file__).parent.parent / 'predictions'
    odds_dir = Path(__file__).parent / 'data' / 'odds'
    
    prediction_files = list(pred_dir.glob('mlb_ensemble_predictions_*.json'))
    odds_files = list(odds_dir.glob('processed_mlb_odds_*.json'))
    
    if not prediction_files or not odds_files:
        print("No prediction or odds files found")
        return
        
    latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
    latest_odds = max(odds_files, key=lambda x: x.stat().st_mtime)
    
    print(f"\nUsing prediction file: {latest_prediction}")
    print(f"Using odds file: {latest_odds}")
    
    # odds 매칭
    matcher = OddsMatcher()
    matched_predictions = matcher.match_odds(latest_prediction, latest_odds)
    
    # 결과 저장
    matcher.save_matched_predictions(matched_predictions)
    
    # 샘플 출력
    print("\nSample matched prediction:")
    print(json.dumps(matched_predictions[0], indent=2))

if __name__ == "__main__":
    main() 