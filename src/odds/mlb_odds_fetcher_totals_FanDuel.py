import requests
import pandas as pd
from pathlib import Path
import json
import os
from datetime import datetime, timedelta
import time
import pytz


class MLBTotalsOddsFetcher:
    """MLB Total Over/Under 배당 수집기 (FanDuel)"""
    
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
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "baseball_mlb"
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone('US/Eastern')
        
    def _convert_to_et(self, utc_time_str):
        """UTC 시간을 미국 동부 시간으로 변환"""
        utc_dt = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
        et_dt = utc_dt.astimezone(self.et_tz)
        return et_dt

    def fetch_totals_odds(self):
        """MLB 경기 totals (Over/Under) 배당률 가져오기"""
        events_url = f"{self.base_url}/{self.sport}/events"
        events_params = {
            'apiKey': self.api_key,
        }
        
        try:
            print("\n=== MLB Totals 배당 수집 시작 (FanDuel) ===")
            
            events_response = requests.get(events_url, params=events_params)
            events_response.raise_for_status()
            events_data = events_response.json()
            
            print(f"Found {len(events_data)} MLB games")
            
            all_odds = []
            for event in events_data:
                event_id = event['id']
                odds_url = f"{self.base_url}/{self.sport}/events/{event_id}/odds"
                odds_params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'totals',
                    'oddsFormat': 'american',
                    'bookmakers': 'fanduel'
                }
                
                print(f"  Fetching totals for {event['home_team']} vs {event['away_team']}")
                odds_response = requests.get(odds_url, params=odds_params)
                odds_response.raise_for_status()
                
                odds_data = odds_response.json()
                all_odds.append(odds_data)
                
                time.sleep(1)
            
            self._save_raw_response(all_odds)
            return self._process_totals_odds(all_odds)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching totals odds: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return None
    
    def _process_totals_odds(self, raw_data):
        """API 응답을 처리하여 totals 배당 DataFrame으로 변환"""
        processed_odds = []
        
        try:
            for game in raw_data:
                game_id = game['id']
                home_team_full = game['home_team']
                away_team_full = game['away_team']
                home_team = self.MLB_TEAM_ABBREV.get(home_team_full, home_team_full)
                away_team = self.MLB_TEAM_ABBREV.get(away_team_full, away_team_full)
                commence_time = game['commence_time']
                
                et_time = self._convert_to_et(commence_time)
                
                completed = game.get('completed', False)
                game_status = self._get_game_status(game)
                
                for bookmaker in game.get('bookmakers', []):
                    if bookmaker['key'] != 'fanduel':
                        continue
                        
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'totals':
                            over_outcome = next((o for o in market['outcomes'] if o['name'] == 'Over'), None)
                            under_outcome = next((o for o in market['outcomes'] if o['name'] == 'Under'), None)
                            
                            if over_outcome and under_outcome:
                                processed_odds.append({
                                    'game_id': game_id,
                                    'date': et_time.strftime('%Y-%m-%d'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_team_full': home_team_full,
                                    'away_team_full': away_team_full,
                                    'total_line': over_outcome.get('point'),
                                    'over_odds': over_outcome.get('price'),
                                    'under_odds': under_outcome.get('price'),
                                    'bookmaker': bookmaker['key'],
                                    'bookmaker_title': bookmaker['title'],
                                    'commence_datetime': et_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'completed': completed,
                                    'game_status': game_status,
                                    'timestamp': datetime.now(self.et_tz).strftime('%Y%m%d_%H%M%S'),
                                    'market_last_update': market.get('last_update'),
                                })
            
            df = pd.DataFrame(processed_odds)
            if not df.empty:
                print(f"\nTotals 배당 수집 완료: {len(df)}개 경기")
                print(f"  Line 범위: {df['total_line'].min()} ~ {df['total_line'].max()}")
                
            return df
            
        except Exception as e:
            print(f"Error processing totals odds: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def _get_game_status(self, game):
        """경기 상태 판단"""
        current_utc = datetime.now(pytz.UTC)
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        completed = game.get('completed', False)
        
        if completed:
            return "completed"
        elif current_utc >= commence_time:
            return "likely_in_progress"
        else:
            return "scheduled"
    
    def _save_raw_response(self, data):
        """API 응답 원본 저장"""
        odds_dir = Path(__file__).parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = odds_dir / f'mlb_totals_odds_raw_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_odds(self, scheduled_only=True):
        """배당률 가져오기 (FanDuel만)"""
        odds_df = self.fetch_totals_odds()
        if odds_df is None or odds_df.empty:
            return None
        
        if scheduled_only:
            odds_df = odds_df[odds_df['game_status'] == 'scheduled']
            print(f"\n예정된 경기만 필터링: {len(odds_df)}개 경기")
            
        return odds_df


def main():
    """테스트 실행"""
    api_key = "3e76069f78f461b5348b7dfdf1ff5535"
    
    fetcher = MLBTotalsOddsFetcher(api_key)
    odds = fetcher.get_odds()
    
    if odds is not None and not odds.empty:
        print("\nFanDuel MLB Totals Odds:")
        print(odds[['home_team', 'away_team', 'total_line', 'over_odds', 'under_odds']].to_string(index=False))
        
        odds_dir = Path(__file__).parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        odds_file = odds_dir / f'processed_mlb_totals_odds_{timestamp}.json'
        odds.to_json(odds_file, orient='records', indent=2)
        print(f"\nTotals odds saved to: {odds_file}")
    else:
        print("Failed to fetch totals odds data")


if __name__ == "__main__":
    main()
