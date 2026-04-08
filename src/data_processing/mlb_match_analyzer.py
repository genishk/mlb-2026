import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import requests
import pytz

class MLBMatchAnalyzer:
    """
    Class for analyzing MLB match data specifically for money line prediction.
    Focuses on team-level statistics and match outcomes rather than player-specific data.
    """
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.current_date = datetime.now()
        self.eastern_tz = pytz.timezone('US/Eastern')  # MLB primarily uses Eastern Time
        
        # Set up project paths
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.data_dir = os.path.join(self.project_root, 'data', 'match_data')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("MLBMatchAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create formatters and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(console_handler)
        
    def _make_request(self, url, params=None):
        """Make a GET request to the specified URL and return JSON response."""
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making request to {url}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {url}: {str(e)}")
            return None
    
    def collect_match_data(self, days_back=2, save_output=True):
        """
        Collect MLB match data for historical analysis.
        Time is converted to Eastern Time (ET) from UTC.
        """
        self.logger.info(f"Collecting MLB match data for the past {days_back} days...")

        # Calculate start date (days_back days ago)
        start_date = self.current_date - timedelta(days=days_back)
        end_date = self.current_date - timedelta(days=1)  # Yesterday
        
        self.logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Initialize empty list to store game data
        all_games = []
        
        # Track processed game IDs to avoid duplicates
        processed_game_ids = set()
        
        # Process each day in the date range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            self.logger.info(f"Processing date: {date_str}")
            
            # Get schedule data from the API
            # MLB API provides schedule data per day unlike NHL's weekly data
            schedule_url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,  # MLB
                'date': date_str,
                'hydrate': 'team,linescore,flags,liveLookin,person,stats,probablePitcher,game(content(summary,media(epg))),game(tickets)',
                'useLatestGames': 'true',
                'language': 'en'
            }
            
            schedule_data = self._make_request(schedule_url, params)
            
            if not schedule_data or 'dates' not in schedule_data or not schedule_data['dates']:
                self.logger.warning(f"No schedule data available for {date_str}")
                current_date += timedelta(days=1)
                continue
            
            # Process games for the day
            date_data = schedule_data['dates'][0]
            if 'games' not in date_data:
                self.logger.warning(f"No games found for {date_str}")
                current_date += timedelta(days=1)
                continue
                
            games = date_data['games']
            self.logger.info(f"Found {len(games)} games for {date_str}")
            
            for game in games:
                # Only completed games
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                
                game_id = game.get('gamePk')
                
                # Skip if game was already processed
                if game_id in processed_game_ids:
                    continue
                
                # Mark game as processed
                processed_game_ids.add(game_id)
                
                # Convert game time to Eastern Time
                if 'gameDate' in game:
                    try:
                        utc_time = datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00'))
                        utc_time = utc_time.replace(tzinfo=pytz.UTC)
                        eastern_time = utc_time.astimezone(self.eastern_tz)
                        game['gameDateET'] = eastern_time.strftime('%Y-%m-%dT%H:%M:%S%z')
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error converting time for game {game_id}: {e}")
                
                # Get additional data for the game
                boxscore_data = self._make_request(f"{self.base_url}/game/{game_id}/boxscore")
                linescore_data = self._make_request(f"{self.base_url}/game/{game_id}/linescore")
                play_by_play_data = self._make_request(f"{self.base_url}/game/{game_id}/playByPlay")
                
                # Store all raw data
                game_data = {
                    'id': game_id,
                    'date': date_str,
                    'schedule': game,
                    'boxscore': boxscore_data,
                    'linescore': linescore_data,
                    'play_by_play': play_by_play_data
                }
                
                all_games.append(game_data)
            
            # Move to the next day
            current_date += timedelta(days=1)
        
        # Log about unique games vs total processed
        self.logger.info(f"Collected {len(all_games)} unique games from {len(processed_game_ids)} game IDs processed")
        
        # Save collected data
        if all_games and save_output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.data_dir, f'mlb_historical_data_{timestamp}.json')
            
            full_data = {
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'days_back': days_back,
                'game_count': len(all_games),
                'games': all_games
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(all_games)} games to {output_file}")
        
        return all_games

    def collect_upcoming_games(self, days_ahead=1):
        """
        Collect upcoming MLB games data for prediction.
        Time is converted to Eastern Time (ET) from UTC.
        Only includes games scheduled to start within the specified days ahead.
        """
        self.logger.info(f"Collecting upcoming games for next {days_ahead} days...")

        upcoming_games = []

        # Calculate dates range
        start_date = self.current_date
        end_date = self.current_date + timedelta(days=days_ahead - 1)
        
        # Get current date in Eastern Time for filtering
        now_et = datetime.now(self.eastern_tz)
        today_et = now_et.strftime('%Y-%m-%d')
        self.logger.info(f"Current date in Eastern Time: {today_et}")
        
        date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get schedule data from the API
        schedule_url = f"{self.base_url}/schedule"
        params = {
            'sportId': 1,  # MLB
            'startDate': date_str,
            'endDate': end_date_str,
            'hydrate': 'team,linescore,flags,liveLookin,person,stats,probablePitcher,game(content(summary,media(epg))),game(tickets)',
            'useLatestGames': 'true',
            'language': 'en'
        }
        
        schedule_data = self._make_request(schedule_url, params)
        
        if not schedule_data or 'dates' not in schedule_data or not schedule_data['dates']:
            self.logger.warning(f"No schedule data available for {date_str} to {end_date_str}")
            return []
        
        # Process each date in the response
        for date_data in schedule_data['dates']:
            if 'games' not in date_data:
                continue
            
            date = date_data.get('date')
            self.logger.info(f"Processing games for {date}")
            
            for game in date_data['games']:
                # Skip completed games - only include scheduled games
                if game.get('status', {}).get('abstractGameState') != 'Preview':
                    continue
                
                # Convert game time to Eastern Time
                game_start_time_et = None
                if 'gameDate' in game:
                    try:
                        utc_time = datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00'))
                        utc_time = utc_time.replace(tzinfo=pytz.UTC)
                        eastern_time = utc_time.astimezone(self.eastern_tz)
                        game['gameDateET'] = eastern_time.strftime('%Y-%m-%dT%H:%M:%S%z')
                        game_start_time_et = eastern_time.strftime('%Y-%m-%d')
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error converting time for game {game.get('gamePk')}: {e}")
                
                # Skip games outside our date range
                end_date_et = end_date.strftime('%Y-%m-%d')
                if game_start_time_et < today_et or game_start_time_et > end_date_et:
                    self.logger.debug(f"Skipping game {game.get('gamePk')} scheduled for {game_start_time_et} as it's outside range {today_et} to {end_date_et}")
                    continue
                
                game_id = game.get('gamePk')
                
                # Get additional data for the upcoming game
                boxscore_data = self._make_request(f"{self.base_url}/game/{game_id}/boxscore")
                linescore_data = self._make_request(f"{self.base_url}/game/{game_id}/linescore")
                
                # Store all raw data
                game_data = {
                    'id': game_id,
                    'date': date,
                    'schedule': game,
                    'boxscore': boxscore_data,
                    'linescore': linescore_data
                }
                
                upcoming_games.append(game_data)
        
        # Save upcoming games
        if upcoming_games:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.data_dir, f'mlb_upcoming_games_{timestamp}.json')
            
            full_data = {
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'days_ahead': days_ahead,
                'game_count': len(upcoming_games),
                'games': upcoming_games
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(upcoming_games)} upcoming games to {output_file}")
            return upcoming_games
            
        self.logger.warning("No upcoming games found for the specified range")
        return []
    
    def analyze_api_structure(self, game_id):
        """
        Analyze and display the structure of API responses for a specific game.
        This helps in understanding what data is available for match prediction.
        """
        self.logger.info(f"Analyzing API structure for game ID: {game_id}")
        
        # List of endpoints to analyze
        endpoints = [
            f"{self.base_url}/game/{game_id}/boxscore",
            f"{self.base_url}/game/{game_id}/linescore",
            f"{self.base_url}/game/{game_id}/playByPlay"
        ]
        
        results = {}
        raw_data = {}
        
        for endpoint in endpoints:
            endpoint_name = endpoint.split('/')[-1]
            response = self._make_request(endpoint)
            
            if not response:
                self.logger.warning(f"No data available for endpoint: {endpoint_name}")
                continue
            
            # Save the entire original response
            raw_data[endpoint_name] = response
            
            # Extract top-level keys and their data types
            top_level_structure = {}
            for key, value in response.items():
                data_type = type(value).__name__
                
                if isinstance(value, dict):
                    # For dictionaries, also show first-level keys
                    subkeys = list(value.keys())
                    top_level_structure[key] = {
                        'type': data_type,
                        'keys': subkeys
                    }
                elif isinstance(value, list) and value:
                    # For lists, show the number of items and type of first item
                    list_type = type(value[0]).__name__ if value else "empty"
                    sample_keys = list(value[0].keys()) if list_type == "dict" and value else []
                    top_level_structure[key] = {
                        'type': f"{data_type}[{list_type}]",
                        'count': len(value),
                        'sample_keys': sample_keys
                    }
                else:
                    top_level_structure[key] = {
                        'type': data_type
                    }
            
            results[endpoint_name] = top_level_structure
        
        # Print results in a structured format
        for endpoint, structure in results.items():
            self.logger.info(f"\n===== {endpoint.upper()} STRUCTURE =====")
            for key, info in structure.items():
                self.logger.info(f"- {key} ({info['type']})")
                
                if 'keys' in info:
                    self.logger.info(f"  Subkeys: {', '.join(info['keys'])}")
                
                if 'count' in info:
                    self.logger.info(f"  Count: {info['count']}")
                    if 'sample_keys' in info and info['sample_keys']:
                        self.logger.info(f"  Sample keys: {', '.join(info['sample_keys'])}")
        
        # Save analysis results to a single consolidated file
        structure_file = os.path.join(self.data_dir, 'mlb_api_structure.json')
        
        # Save API structure and raw data
        full_structure_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'game_id': game_id,
            'api_structure': results,
            'raw_data': raw_data
        }
        
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(full_structure_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Saved API structure and raw data to: {structure_file}")
        
        return results

if __name__ == "__main__":
    analyzer = MLBMatchAnalyzer()
    
    # Find a recent MLB game to analyze API structure
    # This is an example game ID, you should replace with an actual MLB game ID
    analyzer.analyze_api_structure("718211")
    
    # Collect historical data for the past 10 days  #105
    analyzer.collect_match_data(days_back=110)
    
    # Collect upcoming games for the next 3 days
    analyzer.collect_upcoming_games(days_ahead=2) 