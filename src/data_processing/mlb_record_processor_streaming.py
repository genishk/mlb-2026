import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pytz

# ijson 라이브러리 (설치 필요: pip install ijson)
try:
    import ijson
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    print("ijson 라이브러리가 설치되지 않았습니다. 'pip install ijson'을 실행하세요.")

# MLB 팀 ID와 이름 매핑
MLB_TEAMS = {
    "108": "Los Angeles Angels",
    "109": "Arizona Diamondbacks",
    "110": "Baltimore Orioles",
    "111": "Boston Red Sox",
    "112": "Chicago Cubs",
    "113": "Cincinnati Reds",
    "114": "Cleveland Guardians",
    "115": "Colorado Rockies",
    "116": "Detroit Tigers",
    "117": "Houston Astros",
    "118": "Kansas City Royals",
    "119": "Los Angeles Dodgers",
    "120": "Washington Nationals",
    "121": "New York Mets",
    "133": "Oakland Athletics",
    "134": "Pittsburgh Pirates",
    "135": "San Diego Padres",
    "136": "Seattle Mariners",
    "137": "San Francisco Giants",
    "138": "St. Louis Cardinals",
    "139": "Tampa Bay Rays",
    "140": "Texas Rangers",
    "141": "Toronto Blue Jays",
    "142": "Minnesota Twins",
    "143": "Philadelphia Phillies",
    "144": "Atlanta Braves",
    "145": "Chicago White Sox",
    "146": "Miami Marlins",
    "147": "New York Yankees",
    "158": "Milwaukee Brewers",
}

# 팀 약어와 ID 매핑
MLB_TEAM_ABBREVS = {
    "LAA": "108", "ARI": "109", "BAL": "110", "BOS": "111", 
    "CHC": "112", "CIN": "113", "CLE": "114", "COL": "115", 
    "DET": "116", "HOU": "117", "KC": "118", "LAD": "119", 
    "WSH": "120", "NYM": "121", "OAK": "133", "PIT": "134", 
    "SD": "135", "SEA": "136", "SF": "137", "STL": "138", 
    "TB": "139", "TEX": "140", "TOR": "141", "MIN": "142", 
    "PHI": "143", "ATL": "144", "CWS": "145", "MIA": "146", 
    "NYY": "147", "MIL": "158"
}


class MLBRecordProcessorStreaming:
    """
    MLB 경기 데이터를 매치 단위 레코드로 처리하여 JSON으로 저장하는 클래스 (스트리밍 버전)
    
    이 클래스는 큰 JSON 파일을 메모리 효율적으로 처리할 수 있습니다.
    ijson 라이브러리를 사용해서 JSON을 스트리밍 방식으로 파싱합니다.
    """
    
    def __init__(self, debug_mode=True, chunk_size=100):
        # 프로젝트 디렉토리 설정
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'match_data'
        self.records_dir = self.project_root / 'data' / 'records'
        
        # 디렉토리가 없으면 생성
        self.records_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger("MLBRecordProcessorStreaming")
        self.logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러 생성
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 팀 정보 매핑 저장
        self.team_map = MLB_TEAMS
        self.team_abbrev_map = MLB_TEAM_ABBREVS
        
        # 디버그 모드 설정
        self.debug_mode = debug_mode
        
        # 청크 크기 설정 (한 번에 처리할 게임 수)
        self.chunk_size = chunk_size
        
        # 처리된 경기 목록 (중복 방지)
        self.processed_game_ids = set()
        
        # 시간대 설정
        self.eastern_tz = pytz.timezone('US/Eastern')  # 미국 동부 시간대
    
    def find_latest_data_file(self, file_prefix="mlb_historical_data_", file_suffix=".json"):
        """가장 최근에 생성된 데이터 파일 찾기"""
        files = list(self.data_dir.glob(f"{file_prefix}*{file_suffix}"))
        if not files:
            self.logger.error(f"데이터 파일을 찾을 수 없습니다: {file_prefix}*{file_suffix}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        self.logger.info(f"최신 데이터 파일: {latest_file}")
        return latest_file
    
    def get_file_size_mb(self, file_path):
        """파일 크기를 MB 단위로 반환"""
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def load_and_process_in_chunks(self, file_path=None, data_type='historical'):
        """
        대용량 파일을 청크 단위로 처리하여 메모리 사용량 최적화
        ijson을 사용할 수 없는 경우 기본 방식 사용
        """
        if file_path is None:
            if data_type == 'historical':
                file_path = self.find_latest_data_file(file_prefix="mlb_historical_data_")
            else:
                file_path = self.find_latest_data_file(file_prefix="mlb_upcoming_games_")
                
        if file_path is None:
            return None
        
        # 파일 크기 확인
        file_size_mb = self.get_file_size_mb(file_path)
        self.logger.info(f"청크 단위 처리 시작: {file_path} ({file_size_mb:.2f} MB)")
        
        # ijson이 사용 가능한 경우 스트리밍 처리
        if STREAMING_AVAILABLE and file_size_mb > 100:
            return self._process_with_streaming(file_path, data_type)
        else:
            # 기본 처리 방식
            return self._process_with_fallback(file_path, data_type, file_size_mb)
    
    def _process_with_streaming(self, file_path, data_type):
        """ijson을 사용한 스트리밍 처리"""
        try:
            is_historical = (data_type == 'historical')
            processed_records = []
            total_processed = 0
            chunk_buffer = []
            
            with open(file_path, 'rb') as f:
                # 메타데이터 먼저 읽기
                f.seek(0)
                parser = ijson.parse(f)
                metadata = {}
                
                for prefix, event, value in parser:
                    if prefix == 'collection_date' and event == 'string':
                        metadata['collection_date'] = value
                    elif prefix == 'days_back' and event == 'number':
                        metadata['days_back'] = value
                    elif prefix == 'game_count' and event == 'number':
                        metadata['game_count'] = value
                        self.logger.info(f"총 처리할 경기 수: {value}")
                        break
                
                # 게임 데이터를 청크 단위로 처리
                f.seek(0)
                games_parser = ijson.items(f, 'games.item')
                
                for game in games_parser:
                    chunk_buffer.append(game)
                    
                    # 청크가 가득 찼으면 처리
                    if len(chunk_buffer) >= self.chunk_size:
                        chunk_records = self._process_game_chunk(chunk_buffer, is_historical)
                        processed_records.extend(chunk_records)
                        total_processed += len(chunk_records)
                        
                        self.logger.info(f"청크 처리 완료: {total_processed}개 레코드 누적")
                        
                        # 메모리 정리
                        chunk_buffer.clear()
                
                # 마지막 청크 처리
                if chunk_buffer:
                    chunk_records = self._process_game_chunk(chunk_buffer, is_historical)
                    processed_records.extend(chunk_records)
                    total_processed += len(chunk_records)
                    
                    self.logger.info(f"마지막 청크 처리 완료: {total_processed}개 레코드 총 처리")
            
            self.logger.info(f"스트리밍 처리 완료: 총 {len(processed_records)}개 레코드")
            return processed_records
            
        except Exception as e:
            self.logger.error(f"스트리밍 처리 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _process_with_fallback(self, file_path, data_type, file_size_mb):
        """기본 처리 방식 (ijson 없이)"""
        try:
            # 파일이 너무 크면 경고
            if file_size_mb > 500:
                self.logger.warning(f"파일이 매우 큽니다 ({file_size_mb:.2f} MB). 메모리 부족이 발생할 수 있습니다.")
                response = input("계속 진행하시겠습니까? (y/N): ")
                if response.lower() != 'y':
                    return None
            
            self.logger.info("기본 처리 방식으로 데이터 로드 중...")
            
            # 전체 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data or 'games' not in data:
                self.logger.error("유효하지 않은 데이터 형식")
                return None
            
            games = data['games']
            self.logger.info(f"로드된 경기 수: {len(games)}")
            
            # 청크 단위로 처리
            is_historical = (data_type == 'historical')
            processed_records = []
            
            for i in range(0, len(games), self.chunk_size):
                chunk = games[i:i + self.chunk_size]
                chunk_records = self._process_game_chunk(chunk, is_historical)
                processed_records.extend(chunk_records)
                
                self.logger.info(f"청크 {i//self.chunk_size + 1} 처리 완료: {len(processed_records)}개 레코드 누적")
            
            self.logger.info(f"기본 처리 완료: 총 {len(processed_records)}개 레코드")
            return processed_records
            
        except MemoryError:
            self.logger.error("메모리 부족으로 처리할 수 없습니다. ijson 설치를 권장합니다: pip install ijson")
            return None
        except Exception as e:
            self.logger.error(f"기본 처리 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _process_game_chunk(self, games_chunk, is_historical=True):
        """게임 청크를 처리하여 레코드로 변환"""
        chunk_records = []
        skipped_games = 0
        
        for game in games_chunk:
            try:
                # 게임 ID 확인 (중복 방지)
                game_id = game.get('id')
                if not game_id:
                    skipped_games += 1
                    continue
                
                # 중복 처리 방지
                if game_id in self.processed_game_ids:
                    skipped_games += 1
                    continue
                
                # 과거 경기이면서 완료되지 않은 경기 건너뛰기
                if is_historical and not self._is_game_completed(game):
                    skipped_games += 1
                    continue
                
                # 현재 처리 중인 경기 ID 기록
                self.processed_game_ids.add(game_id)
                
                # 게임 데이터 평탄화하여 단일 레벨 레코드로 변환
                game_record = self._extract_and_flatten_game_data(game, is_historical=is_historical)
                
                # 기록 저장
                chunk_records.append(game_record)
                    
            except Exception as e:
                game_id = game.get('id', '알 수 없음')
                self.logger.error(f"경기 {game_id} 처리 중 오류 발생: {str(e)}")
                skipped_games += 1
        
        if skipped_games > 0:
            self.logger.debug(f"청크에서 {skipped_games}개 경기 건너뜀")
        
        return chunk_records
    
    def _is_game_completed(self, game) -> bool:
        """경기가 완료되었는지 확인"""
        # 스케줄의 게임 상태 확인
        schedule = game.get('schedule', {})
        status = schedule.get('status', {})
        game_state = status.get('abstractGameState', '').lower()
        
        # Final 또는 Complete 상태면 완료된 경기
        if game_state in ['final', 'complete']:
            return True
        
        # 박스스코어의 게임 정보 확인
        boxscore = game.get('boxscore', {})
        if boxscore:
            info = boxscore.get('info', [])
            for item in info:
                if item.get('label', {}).get('value', '') == 'Status':
                    if 'final' in item.get('value', '').lower():
                        return True
        
        return False
    
    def _extract_and_flatten_game_data(self, game, is_historical=True) -> Dict[str, Any]:
        """게임 데이터를 추출하고 평탄화하여 단일 레벨의 레코드로 변환"""
        # 빈 레코드로 시작
        record = {}
        
        # 기본 게임 정보 추출
        record['game_id'] = game.get('id')
        record['date'] = game.get('date', '')
        
        # 스케줄 정보 처리
        schedule = game.get('schedule', {})
        
        # 날짜/시간 정보
        game_date = schedule.get('gameDate', '')
        if game_date:
            try:
                date_obj = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                # 이미 date가 없는 경우에만 설정
                if not record['date']:
                    record['date'] = date_obj.strftime('%Y-%m-%d')
                
                # UTC 시간을 동부 시간으로 변환
                utc_time = date_obj.replace(tzinfo=pytz.UTC)
                eastern_time = utc_time.astimezone(self.eastern_tz)
                record['start_time'] = eastern_time.isoformat()  # ISO 형식으로 저장
                record['start_time_et'] = eastern_time.strftime('%Y-%m-%d %H:%M:%S %Z')  # 가독성 있는 형식
            except ValueError:
                self.logger.warning(f"경기 {record['game_id']}의 날짜 형식이 잘못되었습니다: {game_date}")
                record['start_time'] = game_date  # 원본 값 유지
                record['start_time_et'] = ''
        
        # 시즌, 경기장, 경기 상태 등 기본 정보
        record['season'] = schedule.get('season', '')
        record['venue_id'] = schedule.get('venue', {}).get('id', '')
        record['venue_name'] = schedule.get('venue', {}).get('name', '')
        record['game_type'] = schedule.get('gameType', '')
        record['day_night'] = schedule.get('dayNight', '')
        
        # 경기 상태 정보
        status = schedule.get('status', {})
        record['abstract_game_state'] = status.get('abstractGameState', '')
        record['detailed_state'] = status.get('detailedState', '')
        record['status_code'] = status.get('statusCode', '')
        
        # 홈팀/원정팀 기본 정보
        if 'teams' in schedule:
            # 홈팀 정보
            home_team = schedule['teams'].get('home', {})
            if home_team:
                home_team_data = home_team.get('team', {})
                home_team_id = str(home_team_data.get('id', ''))
                record['home_team_id'] = home_team_id
                record['home_team_name'] = self.team_map.get(home_team_id, home_team_data.get('name', ''))
                
                # 홈팀 기록
                record['home_wins'] = home_team.get('leagueRecord', {}).get('wins', 0)
                record['home_losses'] = home_team.get('leagueRecord', {}).get('losses', 0)
                record['home_win_pct'] = home_team.get('leagueRecord', {}).get('pct', '0')
                
                # 홈팀 기본 점수 (과거 경기인 경우에만)
                if is_historical and 'score' in home_team:
                    record['home_score'] = home_team['score']
            
            # 원정팀 정보
            away_team = schedule['teams'].get('away', {})
            if away_team:
                away_team_data = away_team.get('team', {})
                away_team_id = str(away_team_data.get('id', ''))
                record['away_team_id'] = away_team_id
                record['away_team_name'] = self.team_map.get(away_team_id, away_team_data.get('name', ''))
                
                # 원정팀 기록
                record['away_wins'] = away_team.get('leagueRecord', {}).get('wins', 0)
                record['away_losses'] = away_team.get('leagueRecord', {}).get('losses', 0)
                record['away_win_pct'] = away_team.get('leagueRecord', {}).get('pct', '0')
                
                # 원정팀 기본 점수 (과거 경기인 경우에만)
                if is_historical and 'score' in away_team:
                    record['away_score'] = away_team['score']
        
        # 선발 투수 정보
        self._extract_probable_pitchers(record, schedule)
        
        # 날씨 정보
        weather = schedule.get('weather', {})
        if weather:
            record['weather_temp'] = weather.get('temp')
            record['weather_condition'] = weather.get('condition')
            record['wind'] = weather.get('wind')
        
        # 과거 경기인 경우 상세 통계 추출
        if is_historical:
            # 박스스코어 정보 처리 (통계 데이터)
            boxscore = game.get('boxscore', {})
            if boxscore:
                self._extract_boxscore_data(record, boxscore)
            
            # 라인스코어 정보 처리 (이닝별 득점)
            linescore = game.get('linescore', {})
            if linescore:
                self._extract_linescore_data(record, linescore)
            
            # 플레이바이플레이 데이터 처리
            play_by_play = game.get('play_by_play', {})
            if play_by_play:
                self._extract_play_by_play_data(record, play_by_play)
            
            # 승패 여부 추가 (점수 정보가 있는 경우)
            if 'home_score' in record and 'away_score' in record:
                record['home_win'] = 1 if record['home_score'] > record['away_score'] else 0
        else:
            # 예정된 경기인 경우 팀 통계 요약 정보만 추가
            boxscore = game.get('boxscore', {})
            if boxscore and 'teams' in boxscore:
                # 홈팀 주요 통계
                if 'home' in boxscore['teams'] and 'teamStats' in boxscore['teams']['home']:
                    home_stats = boxscore['teams']['home']['teamStats']
                    if 'batting' in home_stats:
                        record['home_batting_avg'] = home_stats['batting'].get('avg', '.000')
                        record['home_batting_obp'] = home_stats['batting'].get('obp', '.000')
                        record['home_batting_slg'] = home_stats['batting'].get('slg', '.000')
                        record['home_batting_ops'] = home_stats['batting'].get('ops', '.000')
                    if 'pitching' in home_stats:
                        record['home_pitching_era'] = home_stats['pitching'].get('era', '0.00')
                        record['home_pitching_whip'] = home_stats['pitching'].get('whip', '0.00')
                
                # 원정팀 주요 통계
                if 'away' in boxscore['teams'] and 'teamStats' in boxscore['teams']['away']:
                    away_stats = boxscore['teams']['away']['teamStats']
                    if 'batting' in away_stats:
                        record['away_batting_avg'] = away_stats['batting'].get('avg', '.000')
                        record['away_batting_obp'] = away_stats['batting'].get('obp', '.000')
                        record['away_batting_slg'] = away_stats['batting'].get('slg', '.000')
                        record['away_batting_ops'] = away_stats['batting'].get('ops', '.000')
                    if 'pitching' in away_stats:
                        record['away_pitching_era'] = away_stats['pitching'].get('era', '0.00')
                        record['away_pitching_whip'] = away_stats['pitching'].get('whip', '0.00')
        
        return record
    
    def _extract_probable_pitchers(self, record, schedule):
        """선발 투수 정보 추출"""
        if 'teams' not in schedule:
            return
        
        # 홈팀 선발 투수
        home_team = schedule['teams'].get('home', {})
        if 'probablePitcher' in home_team:
            pitcher = home_team['probablePitcher']
            record['home_probable_pitcher_id'] = pitcher.get('id', '')
            record['home_probable_pitcher_name'] = pitcher.get('fullName', '')
            
            # 투수 기록이 있으면 추가
            if 'stats' in pitcher and pitcher['stats'] and len(pitcher['stats']) > 0:
                stats = pitcher['stats'][0]
                record['home_pitcher_wins'] = stats.get('wins', 0)
                record['home_pitcher_losses'] = stats.get('losses', 0)
                record['home_pitcher_era'] = stats.get('era', 0.0)
        
        # 원정팀 선발 투수
        away_team = schedule['teams'].get('away', {})
        if 'probablePitcher' in away_team:
            pitcher = away_team['probablePitcher']
            record['away_probable_pitcher_id'] = pitcher.get('id', '')
            record['away_probable_pitcher_name'] = pitcher.get('fullName', '')
            
            # 투수 기록이 있으면 추가
            if 'stats' in pitcher and pitcher['stats'] and len(pitcher['stats']) > 0:
                stats = pitcher['stats'][0]
                record['away_pitcher_wins'] = stats.get('wins', 0)
                record['away_pitcher_losses'] = stats.get('losses', 0)
                record['away_pitcher_era'] = stats.get('era', 0.0)
    
    def _extract_boxscore_data(self, record, boxscore):
        """박스스코어 데이터에서 팀 및 선수 통계 추출 (간소화 버전)"""
        if 'teams' not in boxscore:
            return
        
        teams = boxscore['teams']
        
        # 홈팀 박스스코어
        if 'home' in teams:
            home_boxscore = teams['home']
            self._extract_team_boxscore(record, home_boxscore, 'home')
        
        # 원정팀 박스스코어
        if 'away' in teams:
            away_boxscore = teams['away']
            self._extract_team_boxscore(record, away_boxscore, 'away')
    
    def _extract_team_boxscore(self, record, team_boxscore, team_prefix):
        """팀별 박스스코어 통계 추출"""
        # 팀 타격 통계
        if 'teamStats' in team_boxscore and 'batting' in team_boxscore['teamStats']:
            batting_stats = team_boxscore['teamStats']['batting']
            for stat_name, value in batting_stats.items():
                record[f'{team_prefix}_batting_{stat_name}'] = value
        
        # 팀 투구 통계
        if 'teamStats' in team_boxscore and 'pitching' in team_boxscore['teamStats']:
            pitching_stats = team_boxscore['teamStats']['pitching']
            for stat_name, value in pitching_stats.items():
                record[f'{team_prefix}_pitching_{stat_name}'] = value
        
        # 타자 통계 - 선발 선수들 (top batters)
        if 'players' in team_boxscore:
            batters = []
            pitchers = []
            
            for player_id, player_data in team_boxscore['players'].items():
                if 'stats' in player_data:
                    # 타자 통계
                    if 'batting' in player_data['stats']:
                        player_info = {
                            'id': player_id,
                            'name': player_data.get('person', {}).get('fullName', ''),
                            'position': player_data.get('position', {}).get('abbreviation', ''),
                            **player_data['stats']['batting']
                        }
                        batters.append(player_info)
                    
                    # 투수 통계
                    if 'pitching' in player_data['stats']:
                        player_info = {
                            'id': player_id,
                            'name': player_data.get('person', {}).get('fullName', ''),
                            'position': 'P',
                            **player_data['stats']['pitching']
                        }
                        pitchers.append(player_info)
            
            # 최고 타자 정보 (안타 기준)
            if batters:
                # 안타 기준으로 정렬
                top_batters = sorted(batters, key=lambda x: x.get('hits', 0), reverse=True)
                if top_batters:
                    top_batter = top_batters[0]
                    record[f'{team_prefix}_top_batter_name'] = top_batter.get('name', '')
                    record[f'{team_prefix}_top_batter_hits'] = top_batter.get('hits', 0)
                    record[f'{team_prefix}_top_batter_ab'] = top_batter.get('atBats', 0)
                    record[f'{team_prefix}_top_batter_rbi'] = top_batter.get('rbi', 0)
                    record[f'{team_prefix}_top_batter_avg'] = top_batter.get('avg', '.000')
            
            # 선발 투수 정보
            if pitchers:
                # 이닝 피치 기준으로 정렬 (선발 투수가 보통 가장 많이 던짐)
                starting_pitchers = sorted(pitchers, key=lambda x: float(x.get('inningsPitched', '0')), reverse=True)
                if starting_pitchers:
                    starter = starting_pitchers[0]
                    record[f'{team_prefix}_starter_name'] = starter.get('name', '')
                    record[f'{team_prefix}_starter_ip'] = starter.get('inningsPitched', '0')
                    record[f'{team_prefix}_starter_h'] = starter.get('hits', 0)
                    record[f'{team_prefix}_starter_r'] = starter.get('runs', 0)
                    record[f'{team_prefix}_starter_er'] = starter.get('earnedRuns', 0)
                    record[f'{team_prefix}_starter_bb'] = starter.get('baseOnBalls', 0)
                    record[f'{team_prefix}_starter_k'] = starter.get('strikeOuts', 0)
                    record[f'{team_prefix}_starter_hr'] = starter.get('homeRuns', 0)
                    record[f'{team_prefix}_starter_era'] = starter.get('era', '0.00')
    
    def _extract_linescore_data(self, record, linescore):
        """라인스코어 데이터에서 이닝별 득점 및 합계 추출"""
        # 이닝별 득점
        innings = linescore.get('innings', [])
        for i, inning in enumerate(innings, 1):
            if 'home' in inning and 'runs' in inning['home']:
                record[f'home_inning_{i}_runs'] = inning['home']['runs']
            
            if 'away' in inning and 'runs' in inning['away']:
                record[f'away_inning_{i}_runs'] = inning['away']['runs']
        
        # 총 R/H/E 정보
        if 'teams' in linescore:
            # 홈팀 합계
            if 'home' in linescore['teams']:
                home_totals = linescore['teams']['home']
                record['home_runs'] = home_totals.get('runs', 0)
                record['home_hits'] = home_totals.get('hits', 0)
                record['home_errors'] = home_totals.get('errors', 0)
                record['home_left_on_base'] = home_totals.get('leftOnBase', 0)
            
            # 원정팀 합계
            if 'away' in linescore['teams']:
                away_totals = linescore['teams']['away']
                record['away_runs'] = away_totals.get('runs', 0)
                record['away_hits'] = away_totals.get('hits', 0)
                record['away_errors'] = away_totals.get('errors', 0)
                record['away_left_on_base'] = away_totals.get('leftOnBase', 0)
    
    def _extract_play_by_play_data(self, record, play_by_play):
        """플레이 바이 플레이 데이터에서 주요 이벤트 추출"""
        # 주요 이벤트 카운트
        all_plays = play_by_play.get('allPlays', [])
        
        if not all_plays:
            return
        
        # 이벤트 유형 카운트
        event_counts = {}
        
        # 홈/원정팀 이벤트 카운트
        home_events = {}
        away_events = {}
        
        home_team_id = record.get('home_team_id', '')
        away_team_id = record.get('away_team_id', '')
        
        for play in all_plays:
            # 플레이 결과
            result = play.get('result', {})
            event_type = result.get('eventType', '')
            
            # 이벤트 집계
            if event_type:
                if event_type not in event_counts:
                    event_counts[event_type] = 0
                event_counts[event_type] += 1
                
                # 팀별 이벤트 집계
                team_id = str(play.get('about', {}).get('team', {}).get('id', ''))
                
                if team_id == home_team_id:
                    if event_type not in home_events:
                        home_events[event_type] = 0
                    home_events[event_type] += 1
                elif team_id == away_team_id:
                    if event_type not in away_events:
                        away_events[event_type] = 0
                    away_events[event_type] += 1
        
        # 주요 이벤트 통계 추가
        important_events = ['single', 'double', 'triple', 'home_run', 'walk', 'strikeout', 'field_error', 'grounded_into_double_play']
        
        for event in important_events:
            # 이벤트 카운트 변환 (API 이벤트명이 다를 수 있음)
            api_event = event.upper()
            
            # 전체 이벤트 카운트
            if api_event in event_counts:
                record[f'total_{event}'] = event_counts[api_event]
            
            # 홈팀 이벤트 카운트
            if api_event in home_events:
                record[f'home_{event}'] = home_events[api_event]
            
            # 원정팀 이벤트 카운트
            if api_event in away_events:
                record[f'away_{event}'] = away_events[api_event]
        
        # 중요 플레이 추출 (홈런, 득점 등)
        scoring_plays = play_by_play.get('scoringPlays', [])
        if scoring_plays:
            # 득점 플레이 수
            record['scoring_plays_count'] = len(scoring_plays)
            
            # 첫 번째 득점 플레이 정보
            if scoring_plays:
                first_scoring_play = all_plays[scoring_plays[0]]
                result = first_scoring_play.get('result', {})
                record['first_scoring_play'] = result.get('description', '')
    
    def save_records(self, records, file_prefix="mlb_records"):
        """처리된 레코드를 JSON 파일로 저장"""
        if not records:
            self.logger.error("저장할 레코드가 없습니다.")
            return None
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_prefix}_{timestamp}.json"
        output_path = self.records_dir / filename
        
        self.logger.info(f"처리된 레코드를 JSON으로 저장 중: {output_path}")
        
        # JSON으로 변환 및 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        
        self.logger.info(f"데이터 저장 완료: {output_path}")
        return output_path
    
    def process_and_save_streaming(self, data_type='historical'):
        """스트리밍 방식으로 데이터 처리 및 저장 파이프라인"""
        self.logger.info(f"스트리밍 방식으로 {data_type} 데이터 처리 시작")
        
        # 청크 단위로 처리
        records = self.load_and_process_in_chunks(data_type=data_type)
        if not records:
            self.logger.error(f"{data_type} 데이터 처리 실패")
            return None
        
        # 데이터 샘플 확인
        self.logger.info("\n=== 처리된 레코드 샘플 ===")
        if records and len(records) > 0:
            sample_record = records[0]
            # 레코드의 필드들 중 일부만 표시
            basic_fields = ['game_id', 'date', 'venue_name', 'home_team_name', 'away_team_name', 
                          'home_score', 'away_score', 'home_hits', 'away_hits']
            
            self.logger.info("\n기본 정보:")
            for field in basic_fields:
                if field in sample_record:
                    self.logger.info(f"  {field}: {sample_record[field]}")
            
            # 총 필드 수 표시
            self.logger.info(f"\n레코드 필드 수: {len(sample_record)}")
        
        # 데이터 저장
        is_historical = (data_type == 'historical')
        file_prefix = "mlb_historical_records" if is_historical else "mlb_upcoming_records"
        output_path = self.save_records(records, file_prefix=file_prefix)
        
        return output_path


# 스크립트로 실행 시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = MLBRecordProcessorStreaming(debug_mode=True, chunk_size=50)
    
    print("=== MLB 레코드 처리기 (스트리밍 버전) ===")
    print("대용량 JSON 파일을 메모리 효율적으로 처리합니다.")
    
    if not STREAMING_AVAILABLE:
        print("\n경고: ijson 라이브러리가 설치되지 않았습니다.")
        print("최적의 성능을 위해 'pip install ijson'을 실행하세요.")
    
    # 과거 경기 데이터 처리 (스트리밍 방식)
    historical_output = processor.process_and_save_streaming(data_type='historical')
    print(f"\n과거 경기 처리 결과: {historical_output}")
    
    # 예정된 경기 데이터 처리
    upcoming_output = processor.process_and_save_streaming(data_type='upcoming')
    print(f"\n예정된 경기 처리 결과: {upcoming_output}") 