import requests
import pandas as pd
from pathlib import Path
import json
import os
from datetime import datetime
import time
from datetime import datetime, timedelta
import pytz

class MLBOddsFetcher:
    # MLB 팀명 매핑 딕셔너리 (전체 이름 -> 약자)
    MLB_TEAM_ABBREV = {
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
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "baseball_mlb"
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone('US/Eastern')
        
    def _convert_to_et(self, utc_time_str):
        """UTC 시간을 미국 동부 시간으로 변환"""
        # UTC 시간 문자열을 datetime 객체로 변환 (이미 UTC timezone이 포함되어 있음)
        utc_dt = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
        
        # ET로 변환
        et_dt = utc_dt.astimezone(self.et_tz)
        return et_dt

    def fetch_moneyline_odds(self):
        """MLB 경기 머니라인(승패) 배당률 가져오기"""
        # 1. 먼저 경기 목록 가져오기
        events_url = f"{self.base_url}/{self.sport}/events"
        events_params = {
            'apiKey': self.api_key,
        }
        
        try:
            # 경기 목록 가져오기
            events_response = requests.get(events_url, params=events_params)
            events_response.raise_for_status()
            events_data = events_response.json()
            
            print(f"\nFound {len(events_data)} MLB games")
            
            all_odds = []
            # 2. 각 경기별로 배당률 가져오기
            for event in events_data:
                event_id = event['id']
                odds_url = f"{self.base_url}/{self.sport}/events/{event_id}/odds"
                odds_params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'h2h',
                    'oddsFormat': 'american',
                    'bookmakers': 'fanduel'  # FanDuel만 사용
                }
                
                print(f"\nFetching odds for {event['home_team']} vs {event['away_team']}")
                odds_response = requests.get(odds_url, params=odds_params)
                odds_response.raise_for_status()
                
                odds_data = odds_response.json()
                all_odds.append(odds_data)
                
                # API 호출 간격 조절 (rate limiting 방지)
                time.sleep(1)
            
            # 응답 저장
            self._save_raw_response(all_odds)
            
            return self._process_odds(all_odds)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching odds: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
                print(f"URL: {e.response.url}")
            return None
    
    def _get_game_status(self, game):
        """경기 상태 판단 (예정/진행중/완료)"""
        from datetime import datetime
        import pytz
        
        # 현재 시간 (UTC)
        current_utc = datetime.now(pytz.UTC)
        
        # 경기 시작 시간
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        
        # 경기 완료 여부
        completed = game.get('completed', False)
        
        # 점수 정보
        scores = game.get('scores', {})
        home_score = scores.get('home')
        away_score = scores.get('away')
        
        if completed:
            return "completed"
        elif home_score is not None or away_score is not None:
            return "in_progress"
        elif current_utc >= commence_time:
            # 시작 시간이 지났지만 점수가 없으면 진행중일 가능성
            return "likely_in_progress"
        else:
            return "scheduled"
    
    def _process_odds(self, raw_data):
        """API 응답을 처리하여 DataFrame으로 변환"""
        processed_odds = []
        
        try:
            for game in raw_data:
                # 경기 상태 판단
                game_status = self._get_game_status(game)
                
                # 경기 관련 추가 정보
                game_id = game['id']
                home_team = self.MLB_TEAM_ABBREV.get(game['home_team'], game['home_team'])
                away_team = self.MLB_TEAM_ABBREV.get(game['away_team'], game['away_team'])
                commence_time = game['commence_time']
                
                # UTC를 ET로 변환
                et_time = self._convert_to_et(commence_time)
                
                # 추가 가능한 경기 정보들
                sport_title = game.get('sport_title')
                sport_key = game.get('sport_key')
                completed = game.get('completed', False)
                home_score = game.get('scores', {}).get('home')
                away_score = game.get('scores', {}).get('away')
                last_update = game.get('last_update')
                
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker['key'] != 'fanduel':  # FanDuel만 처리
                        continue
                        
                    bookmaker_key = bookmaker['key']
                    bookmaker_title = bookmaker['title']
                    bookmaker_last_update = bookmaker.get('last_update')
                    
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            market_key = market['key']
                            market_last_update = market.get('last_update')
                            
                            # Process home team odds
                            home_outcome = next((outcome for outcome in market['outcomes'] 
                                               if outcome['name'] == game['home_team']), None)
                            # Process away team odds
                            away_outcome = next((outcome for outcome in market['outcomes']
                                               if outcome['name'] == game['away_team']), None)
                            
                            if home_outcome and away_outcome:
                                # Add home team odds
                                processed_odds.append({
                                    'game_id': game_id,
                                    'date': et_time.strftime('%Y-%m-%d'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'team': home_team,
                                    'is_home': True,
                                    'market_key': market_key,
                                    'odds': home_outcome['price'],
                                    'bookmaker': bookmaker_key,
                                    'probability': self._convert_odds_to_probability(home_outcome['price']),
                                    'timestamp': datetime.now(self.et_tz).strftime('%Y%m%d_%H%M%S'),
                                    'sport_title': sport_title,
                                    'commence_datetime': et_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'bookmaker_title': bookmaker_title,
                                    'market_last_update': self._convert_to_et(market_last_update).strftime('%Y-%m-%d %H:%M:%S') if market_last_update else None,
                                    'completed': completed,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'game_last_update': self._convert_to_et(last_update).strftime('%Y-%m-%d %H:%M:%S') if last_update else None,
                                    'bookmaker_last_update': self._convert_to_et(bookmaker_last_update).strftime('%Y-%m-%d %H:%M:%S') if bookmaker_last_update else None,
                                    'game_status': game_status
                                })
                                
                                # Add away team odds
                                processed_odds.append({
                                    'game_id': game_id,
                                    'date': et_time.strftime('%Y-%m-%d'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'team': away_team,
                                    'is_home': False,
                                    'market_key': market_key,
                                    'odds': away_outcome['price'],
                                    'bookmaker': bookmaker_key,
                                    'probability': self._convert_odds_to_probability(away_outcome['price']),
                                    'timestamp': datetime.now(self.et_tz).strftime('%Y%m%d_%H%M%S'),
                                    'sport_title': sport_title,
                                    'commence_datetime': et_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'bookmaker_title': bookmaker_title,
                                    'market_last_update': self._convert_to_et(market_last_update).strftime('%Y-%m-%d %H:%M:%S') if market_last_update else None,
                                    'completed': completed,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'game_last_update': self._convert_to_et(last_update).strftime('%Y-%m-%d %H:%M:%S') if last_update else None,
                                    'bookmaker_last_update': self._convert_to_et(bookmaker_last_update).strftime('%Y-%m-%d %H:%M:%S') if bookmaker_last_update else None,
                                    'game_status': game_status
                                })
                            
            df = pd.DataFrame(processed_odds)
            if not df.empty:
                # 시간 정보 처리 - 이미 모든 시간이 ET로 변환되어 문자열로 저장되어 있음
                df['commence_datetime'] = pd.to_datetime(df['commence_datetime'])
                
                print("\nComplete data structure:")
                print("\nColumns:", df.columns.tolist())
                print("\nSample row:")
                if len(df) > 0:
                    print(df.iloc[0].to_dict())
                else:
                    print("No data available")
                
            return df
            
        except Exception as e:
            print(f"Error processing odds: {e}")
            import traceback
            print(traceback.format_exc())
            print("Raw data sample:", json.dumps(raw_data[:1] if raw_data else [], indent=2))
            return pd.DataFrame()
        
    def _convert_odds_to_probability(self, american_odds):
        """미국식 배당률을 확률로 변환"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return (-american_odds) / (-american_odds + 100)
    
    def _save_raw_response(self, data):
        """API 응답 원본 저장"""
        odds_dir = Path(__file__).parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = odds_dir / f'mlb_odds_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_odds(self, scheduled_only=True):
        """배당률 가져오기 (FanDuel만)
        
        Args:
            scheduled_only (bool): True이면 예정된 경기만 반환, False이면 모든 경기 반환
        """
        odds_df = self.fetch_moneyline_odds()
        if odds_df is None or odds_df.empty:
            return None
        
        # 예정된 경기만 필터링
        if scheduled_only:
            odds_df = odds_df[odds_df['game_status'] == 'scheduled']
            print(f"\n예정된 경기만 필터링: {len(odds_df)} 레코드")
            
        return odds_df.sort_values('probability', ascending=False)
        
    def compare_with_predictions(self, prediction_file=None):
        """예측 결과와 배당률 비교"""
        # 예측 파일이 지정되지 않은 경우 가장 최신 파일 사용
        if prediction_file is None:
            pred_dir = Path(__file__).parent.parent / 'predictions'
            prediction_files = list(pred_dir.glob('mlb_ensemble_predictions_*.json'))
            if not prediction_files:
                print("No prediction files found")
                return None
            prediction_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
        
        print(f"\nUsing prediction file: {prediction_file}")
        
        # 예측 데이터 로드
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        
        # 배당률 데이터 가져오기
        odds_df = self.get_odds()
        if odds_df is None or odds_df.empty:
            print("No odds data available")
            return None
        
        # 결과 저장을 위한 리스트
        comparison_results = []
        
        # 각 예측별로 비교
        for pred in predictions:
            game_id = pred.get('game_id')
            home_team = pred.get('home_team')
            away_team = pred.get('away_team')
            predicted_winner = pred.get('predicted_winner')
            win_probability = pred.get('win_probability')
            
            # 이 경기의 배당률 필터링
            game_odds = odds_df[
                (odds_df['home_team'] == self.MLB_TEAM_ABBREV.get(home_team, home_team)) & 
                (odds_df['away_team'] == self.MLB_TEAM_ABBREV.get(away_team, away_team))
            ]
            
            if not game_odds.empty:
                # 홈팀 배당률
                home_odds = game_odds[game_odds['is_home'] == True]
                home_best_odds = home_odds.iloc[0] if not home_odds.empty else None
                
                # 원정팀 배당률
                away_odds = game_odds[game_odds['is_home'] == False]
                away_best_odds = away_odds.iloc[0] if not away_odds.empty else None
                
                # 예측된 승자의 배당률
                if predicted_winner == home_team and home_best_odds is not None:
                    predicted_winner_odds = home_best_odds['odds']
                    predicted_winner_prob = home_best_odds['probability']
                    predicted_winner_bookmaker = home_best_odds['bookmaker']
                elif predicted_winner == away_team and away_best_odds is not None:
                    predicted_winner_odds = away_best_odds['odds']
                    predicted_winner_prob = away_best_odds['probability']
                    predicted_winner_bookmaker = away_best_odds['bookmaker']
                else:
                    predicted_winner_odds = None
                    predicted_winner_prob = None
                    predicted_winner_bookmaker = None
                
                # 배당률과 예측 확률의 차이 계산
                if predicted_winner_prob is not None:
                    prob_difference = win_probability - predicted_winner_prob
                else:
                    prob_difference = None
                
                # 베팅 가치 계산 (양수이면 가치 있는 베팅)
                if prob_difference is not None:
                    betting_value = prob_difference
                else:
                    betting_value = None
                
                # 결과 추가
                comparison_results.append({
                    'game_id': game_id,
                    'date': pred.get('date'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_winner': predicted_winner,
                    'model_win_probability': win_probability,
                    'bookmaker_probability': predicted_winner_prob,
                    'probability_difference': prob_difference,
                    'betting_value': betting_value,
                    'winner_odds': predicted_winner_odds,
                    'bookmaker': predicted_winner_bookmaker
                })
            else:
                print(f"No odds found for game: {home_team} vs {away_team}")
        
        # 결과를 DataFrame으로 변환
        comparison_df = pd.DataFrame(comparison_results)
        
        # 베팅 가치가 높은 순서대로 정렬
        if not comparison_df.empty and 'betting_value' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('betting_value', ascending=False)
            
            # 결과 저장
            self._save_comparison_results(comparison_df)
            
        return comparison_df
    
    def _save_comparison_results(self, df):
        """비교 결과 저장"""
        if df.empty:
            print("No comparison results to save")
            return
            
        odds_dir = Path(__file__).parent / 'data' / 'analysis'
        os.makedirs(odds_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = odds_dir / f'mlb_betting_analysis_{timestamp}.json'
        
        df.to_json(filename, orient='records', indent=2)
        print(f"\nComparison results saved to: {filename}")

def main():
    """테스트 실행"""
    api_key = "3e76069f78f461b5348b7dfdf1ff5535"  # API 키
    
    fetcher = MLBOddsFetcher(api_key)
    
    # 배당률 데이터 가져오기
    odds = fetcher.get_odds()
    
    if odds is not None and not odds.empty:
        print("\nFanDuel MLB Odds:")
        print(odds[['team', 'odds', 'probability', 'bookmaker']].head(10))
        
        # 파일로 저장
        odds_dir = Path(__file__).parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        odds_file = odds_dir / f'processed_mlb_odds_{timestamp}.json'
        odds.to_json(odds_file, orient='records', indent=2)
        print(f"\nOdds saved to: {odds_file}")
    else:
        print("Failed to fetch odds data")

if __name__ == "__main__":
    main() 