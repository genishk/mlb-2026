import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from collections import defaultdict

class MLBPredictionDataProcessor:
    """
    MLB 예정 경기 데이터를 머신러닝 예측 데이터로 변환하는 프로세서
    
    이 클래스는 예정된 MLB 경기 정보를 가져와서 예측에 필요한 피처를 생성합니다.
    경기 기본 정보는 upcoming_records에서 가져오고, 
    피처 구성은 historical_records의 과거 경기 데이터를 사용합니다.
    """
    
    def __init__(self, debug_mode=True):
        """프로세서 초기화"""
        # 프로젝트 디렉토리 설정
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.records_dir = self.project_root / 'data' / 'records'
        self.prediction_dir = self.project_root / 'data' / 'prediction'
        self.training_dir = self.project_root / 'data' / 'training'
        
        # 디렉토리가 없으면 생성
        self.prediction_dir.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger("MLBPredictionDataProcessor")
        self.logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러 생성
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 디버그 모드 설정
        self.debug_mode = debug_mode
        
        # 데이터 저장용 변수
        self.historical_data = None
        self.upcoming_data = None
        self.prediction_data = None
        
        self.logger.info("MLB 예측 데이터 프로세서 초기화 완료")
    
    def find_latest_records_file(self, file_prefix="mlb_historical_records_", file_suffix=".json"):
        """가장 최근에 생성된 레코드 파일 찾기"""
        files = list(self.records_dir.glob(f"{file_prefix}*{file_suffix}"))
        if not files:
            self.logger.error(f"레코드 파일을 찾을 수 없습니다: {file_prefix}*{file_suffix}")
            return None
        
        # 파일명에서 타임스탬프를 추출하여 최신 파일 찾기
        def extract_timestamp(file_path):
            """파일명에서 타임스탬프 추출 (YYYYMMDD_HHMMSS 형식)"""
            try:
                filename = file_path.stem  # 확장자 제거
                # 파일명에서 타임스탬프 부분 추출 (마지막 _ 이후)
                timestamp_part = filename.split('_')[-2] + '_' + filename.split('_')[-1]
                return timestamp_part
            except:
                # 타임스탬프 추출 실패 시 파일 수정 시간 사용
                return file_path.stat().st_mtime
        
        # 타임스탬프 기준으로 정렬하여 최신 파일 선택
        try:
            latest_file = max(files, key=extract_timestamp)
            self.logger.info(f"최신 레코드 파일: {latest_file}")
            return latest_file
        except Exception as e:
            # 타임스탬프 추출 실패 시 기존 방식 사용
            self.logger.warning(f"타임스탬프 추출 실패, 파일 수정 시간 사용: {str(e)}")
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"최신 레코드 파일: {latest_file}")
            return latest_file
    
    def load_records(self, historical_file_path=None, upcoming_file_path=None):
        """MLB 레코드 데이터 로드하기
        
        과거 경기 데이터와 예정 경기 데이터를 모두 로드합니다.
        이미 완료된 경기 기록은 과거 데이터로, 예정된 경기는 별도 파일로 로드합니다.
        
        Args:
            historical_file_path: 과거 경기 기록 파일 경로 (기본값: 가장 최근 파일)
            upcoming_file_path: 예정 경기 기록 파일 경로 (기본값: 가장 최근 파일)
            
        Returns:
            bool: 성공 여부
        """
        # 과거 경기 데이터 로드
        if historical_file_path is None:
            historical_file_path = self.find_latest_records_file("mlb_historical_records_")
                
        if historical_file_path is None:
            self.logger.error("과거 경기 레코드 파일을 찾을 수 없습니다. 파일이 data/records 디렉토리에 존재하는지 확인하세요.")
            return False
        
        # 예정 경기 데이터 로드
        if upcoming_file_path is None:
            upcoming_file_path = self.find_latest_records_file("mlb_upcoming_records_")
                
        if upcoming_file_path is None:
            self.logger.error("예정 경기 레코드 파일을 찾을 수 없습니다. 파일이 data/records 디렉토리에 존재하는지 확인하세요.")
            return False
        
        try:
            # 1. 과거 경기 데이터 로드
            self.logger.info(f"MLB 과거 경기 데이터 로드 중: {historical_file_path}")
            with open(historical_file_path, 'r', encoding='utf-8') as f:
                historical_records = json.load(f)
            
            self.logger.info(f"로드된 과거 경기 수: {len(historical_records)}")
            
            # 2. 예정 경기 데이터 로드
            self.logger.info(f"MLB 예정 경기 데이터 로드 중: {upcoming_file_path}")
            with open(upcoming_file_path, 'r', encoding='utf-8') as f:
                upcoming_records = json.load(f)
            
            self.logger.info(f"로드된 예정 경기 수: {len(upcoming_records)}")
            
            # 과거 경기 데이터프레임 변환
            historical_df = pd.DataFrame(historical_records)
            
            # 예정 경기 데이터프레임 변환
            upcoming_df = pd.DataFrame(upcoming_records)
            
            # 날짜 열을 datetime 형식으로 변환
            try:
                historical_df['date'] = pd.to_datetime(historical_df['date'])
                upcoming_df['date'] = pd.to_datetime(upcoming_df['date'])
            except Exception as e:
                self.logger.error(f"날짜 변환 중 오류 발생: {str(e)}. 'date' 열이 올바른 형식인지 확인하세요.")
                return False
            
            # 날짜순으로 정렬 (시간 순서가 매우 중요)
            historical_df = historical_df.sort_values('date')
            upcoming_df = upcoming_df.sort_values('date')
            
            # team_id를 문자열로 통일
            for df in [historical_df, upcoming_df]:
                if 'home_team_id' in df.columns:
                    df['home_team_id'] = df['home_team_id'].astype(str)
                else:
                    self.logger.error("데이터에 'home_team_id' 열이 없습니다. 데이터 형식을 확인하세요.")
                    return False
                    
                if 'away_team_id' in df.columns:
                    df['away_team_id'] = df['away_team_id'].astype(str)
                else:
                    self.logger.error("데이터에 'away_team_id' 열이 없습니다. 데이터 형식을 확인하세요.")
                    return False
            
            self.historical_data = historical_df
            self.upcoming_data = upcoming_df
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"파일을 찾을 수 없습니다: {e.filename}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"잘못된 JSON 형식입니다: {e.doc}")
            return False
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def collect_team_games(self):
        """팀별 경기 데이터 수집
        
        시간 순서대로 팀별 경기 결과와 통계를 수집합니다.
        과거 데이터에서 팀별 성적 및 통계를 추출합니다.
        
        Returns:
            Dict: 팀별 경기 결과 및 날짜 정보
        """
        self.logger.info("팀별 과거 경기 데이터 수집 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("과거 데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
        
        df = self.historical_data
        
        # 팀별 경기 결과 및 통계 저장
        team_games = defaultdict(list)
        team_games_dates = defaultdict(list)
        
        # 주요 타격 통계 필드 정의
        batting_stat_fields = [
            'batting_avg', 'batting_obp', 'batting_slg', 'batting_ops',
            'batting_hits', 'batting_homeRuns', 'batting_runs', 'batting_rbi', 
            'batting_baseOnBalls', 'batting_strikeOuts'
        ]
        
        # 주요 투구 통계 필드 정의
        pitching_stat_fields = [
            'pitching_era', 'pitching_whip', 'pitching_inningsPitched',
            'pitching_hits', 'pitching_runs', 'pitching_earnedRuns',
            'pitching_baseOnBalls', 'pitching_strikeOuts', 'pitching_homeRuns'
        ]
        
        # 정렬된 데이터프레임 순회
        for idx, row in df.iterrows():
            game_date = row['date']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            
            # 승패 여부 확인 (home_win 필드 사용)
            if 'home_win' in row and not pd.isna(row['home_win']):
                home_won = row['home_win'] == 1
                away_won = not home_won
            # 점수로 승패 계산
            elif 'home_score' in row and 'away_score' in row and not pd.isna(row['home_score']) and not pd.isna(row['away_score']):
                home_won = row['home_score'] > row['away_score']
                away_won = row['away_score'] > row['home_score']
            else:
                # 승패를 알 수 없는 경우 (미래 경기 등)
                continue
            
            # 홈팀 데이터 수집
            home_batting_stats = {}
            home_pitching_stats = {}
            
            # 타격 통계 수집
            for field in batting_stat_fields:
                field_name = f'home_{field}'
                if field_name in row and not pd.isna(row[field_name]):
                    home_batting_stats[field] = row[field_name]
            
            # 투구 통계 수집
            for field in pitching_stat_fields:
                field_name = f'home_{field}'
                if field_name in row and not pd.isna(row[field_name]):
                    home_pitching_stats[field] = row[field_name]
            
            # 홈팀 결과 저장
            team_games[home_team_id].append({
                'game_id': row['game_id'],
                'date': game_date,
                'is_home': True,
                'opponent_id': away_team_id,
                'won': home_won,
                'score': row.get('home_score', 0),
                'opponent_score': row.get('away_score', 0),
                'batting_stats': home_batting_stats,
                'pitching_stats': home_pitching_stats
            })
            team_games_dates[home_team_id].append(game_date)
            
            # 원정팀 데이터 수집
            away_batting_stats = {}
            away_pitching_stats = {}
            
            # 타격 통계 수집
            for field in batting_stat_fields:
                field_name = f'away_{field}'
                if field_name in row and not pd.isna(row[field_name]):
                    away_batting_stats[field] = row[field_name]
            
            # 투구 통계 수집
            for field in pitching_stat_fields:
                field_name = f'away_{field}'
                if field_name in row and not pd.isna(row[field_name]):
                    away_pitching_stats[field] = row[field_name]
            
            # 원정팀 결과 저장
            team_games[away_team_id].append({
                'game_id': row['game_id'],
                'date': game_date,
                'is_home': False,
                'opponent_id': home_team_id,
                'won': away_won,
                'score': row.get('away_score', 0),
                'opponent_score': row.get('home_score', 0),
                'batting_stats': away_batting_stats,
                'pitching_stats': away_pitching_stats
            })
            team_games_dates[away_team_id].append(game_date)
        
        # 각 팀별 경기 날짜순으로 정렬
        for team_id in team_games:
            # 날짜 기준으로 정렬
            sorted_games = sorted(zip(team_games_dates[team_id], team_games[team_id]), key=lambda x: x[0])
            team_games_dates[team_id] = [date for date, _ in sorted_games]
            team_games[team_id] = [game for _, game in sorted_games]
        
        # 결과 반환
        result = {
            'team_games': team_games,
            'team_games_dates': team_games_dates
        }
        
        self.logger.info(f"팀별 과거 경기 데이터 수집 완료: {len(team_games)}개 팀")
        return result
    
    def create_prediction_features(self, n_recent_games=10):
        """예정된 경기에 대한 예측 피처 생성
        
        과거 경기 데이터를 사용하여 예정된 경기에 대한 예측 피처를 생성합니다.
        
        Args:
            n_recent_games: 최근 몇 경기를 고려할지 (기본값: 10)
            
        Returns:
            pd.DataFrame: 예측 피처가 포함된 데이터프레임
        """
        self.logger.info("예정된 경기에 대한 예측 피처 생성 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None or self.upcoming_data is None:
            self.logger.error("과거 데이터 또는 예정 데이터가 로드되지 않았습니다.")
            return None
        
        # 팀별 경기 데이터 수집
        team_data = self.collect_team_games()
        if team_data is None:
            return None
        
        team_games = team_data['team_games']
        
        # 결과 데이터프레임 초기화 (예정된 경기 데이터 복사)
        result_df = self.upcoming_data.copy()
        
        # 주요 타격 통계 필드 정의
        batting_stat_fields = [
            'batting_avg', 'batting_obp', 'batting_slg', 'batting_ops',
            'batting_hits', 'batting_homeRuns', 'batting_runs', 'batting_rbi', 
            'batting_baseOnBalls', 'batting_strikeOuts'
        ]
        
        # 주요 투구 통계 필드 정의
        pitching_stat_fields = [
            'pitching_era', 'pitching_whip', 'pitching_inningsPitched',
            'pitching_hits', 'pitching_runs', 'pitching_earnedRuns',
            'pitching_baseOnBalls', 'pitching_strikeOuts', 'pitching_homeRuns'
        ]
        
        # 각 예정 경기에 대해 피처 생성
        for idx, row in result_df.iterrows():
            game_date = row['date']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            
            # 날짜 형식 확인 및 변환
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            
            # 홈팀 통계 및 시즌 통계 추가
            if home_team_id in team_games:
                # 현재 경기 이전의 홈팀 경기만 필터링
                home_previous_games = [g for g in team_games[home_team_id] if pd.to_datetime(g['date']) < game_date]
                
                if home_previous_games:
                    # 시즌 전체 홈팀 통계
                    home_total_games = len(home_previous_games)
                    home_wins = sum(1 for g in home_previous_games if g['won'])
                    home_losses = home_total_games - home_wins
                    home_home_games = sum(1 for g in home_previous_games if g['is_home'])
                    home_home_wins = sum(1 for g in home_previous_games if g['is_home'] and g['won'])
                    home_home_losses = home_home_games - home_home_wins
                    home_away_games = home_total_games - home_home_games
                    home_away_wins = home_wins - home_home_wins
                    home_away_losses = home_away_games - home_away_wins
                    
                    # 홈팀 승률 계산
                    home_win_rate = home_wins / home_total_games if home_total_games > 0 else 0.5
                    home_home_win_rate = home_home_wins / home_home_games if home_home_games > 0 else 0.5
                    home_away_win_rate = home_away_wins / home_away_games if home_away_games > 0 else 0.5
                    
                    # 홈팀 시즌 득점 및 실점 통계
                    home_total_runs_for = sum(g['score'] for g in home_previous_games)
                    home_total_runs_against = sum(g['opponent_score'] for g in home_previous_games)
                    home_avg_runs_for = home_total_runs_for / home_total_games if home_total_games > 0 else 0
                    home_avg_runs_against = home_total_runs_against / home_total_games if home_total_games > 0 else 0
                    
                    # 홈팀 시즌 타격 통계 평균 계산
                    home_batting_avgs = {}
                    for stat in batting_stat_fields:
                        values = [g['batting_stats'].get(stat, 0) for g in home_previous_games 
                                 if 'batting_stats' in g and stat in g['batting_stats']]
                        if values:
                            try:
                                # 문자열을 실수로 변환
                                numeric_values = []
                                for v in values:
                                    try:
                                        numeric_values.append(float(v) if isinstance(v, str) else v)
                                    except (ValueError, TypeError):
                                        # 변환할 수 없는 값은 건너뜀
                                        self.logger.warning(f"홈팀 {stat} 값을 변환할 수 없음: {v}")
                                
                                if numeric_values:
                                    home_batting_avgs[stat] = sum(numeric_values) / len(numeric_values)
                                else:
                                    home_batting_avgs[stat] = 0
                            except Exception as e:
                                self.logger.warning(f"홈팀 {stat} 평균 계산 중 오류: {str(e)}")
                                home_batting_avgs[stat] = 0
                        else:
                            home_batting_avgs[stat] = 0
                    
                    # 홈팀 시즌 투구 통계 평균 계산
                    home_pitching_avgs = {}
                    for stat in pitching_stat_fields:
                        values = [g['pitching_stats'].get(stat, 0) for g in home_previous_games 
                                 if 'pitching_stats' in g and stat in g['pitching_stats']]
                        if values:
                            try:
                                # 문자열을 실수로 변환
                                numeric_values = []
                                for v in values:
                                    try:
                                        numeric_values.append(float(v) if isinstance(v, str) else v)
                                    except (ValueError, TypeError):
                                        # 변환할 수 없는 값은 건너뜀
                                        self.logger.warning(f"홈팀 {stat} 값을 변환할 수 없음: {v}")
                                
                                if numeric_values:
                                    home_pitching_avgs[stat] = sum(numeric_values) / len(numeric_values)
                                else:
                                    home_pitching_avgs[stat] = 0
                            except Exception as e:
                                self.logger.warning(f"홈팀 {stat} 평균 계산 중 오류: {str(e)}")
                                home_pitching_avgs[stat] = 0
                        else:
                            home_pitching_avgs[stat] = 0
                    
                    # 홈팀 전적 문자열 생성 (승-패 형식)
                    home_overall_record = f"{home_wins}-{home_losses}"
                    home_home_record = f"{home_home_wins}-{home_home_losses}"
                    home_road_record = f"{home_away_wins}-{home_away_losses}"
                    
                    # 홈팀 시즌 통계 저장 (훈련 데이터와 일치하는 이름으로)
                    result_df.loc[idx, 'home_season_games'] = home_total_games  # 추가 정보로 유지
                    result_df.loc[idx, 'home_overall_record'] = home_overall_record
                    result_df.loc[idx, 'home_overall_record_win_rate'] = home_win_rate
                    result_df.loc[idx, 'home_home_record'] = home_home_record
                    result_df.loc[idx, 'home_road_record'] = home_road_record
                    result_df.loc[idx, 'home_home_record_win_rate'] = home_home_win_rate
                    result_df.loc[idx, 'home_road_record_win_rate'] = home_away_win_rate
                    result_df.loc[idx, 'home_avg_runs_for'] = home_avg_runs_for
                    result_df.loc[idx, 'home_avg_runs_against'] = home_avg_runs_against
                    
                    # 타격 통계 저장
                    for stat, value in home_batting_avgs.items():
                        result_df.loc[idx, f'home_avg_{stat}'] = value
                    
                    # 투구 통계 저장
                    for stat, value in home_pitching_avgs.items():
                        result_df.loc[idx, f'home_avg_{stat}'] = value
                    
                    # 홈팀 최근 N경기 통계
                    home_recent_games = home_previous_games[-n_recent_games:] if len(home_previous_games) >= n_recent_games else home_previous_games
                    
                    if home_recent_games:
                        # 최근 N경기 승률
                        home_recent_wins = sum(1 for g in home_recent_games if g['won'])
                        home_recent_win_rate = home_recent_wins / len(home_recent_games)
                        
                        # 최근 N경기 홈/원정 세분화 승률
                        home_recent_home_games = sum(1 for g in home_recent_games if g['is_home'])
                        home_recent_home_wins = sum(1 for g in home_recent_games if g['is_home'] and g['won'])
                        home_recent_away_games = len(home_recent_games) - home_recent_home_games
                        home_recent_away_wins = home_recent_wins - home_recent_home_wins
                        
                        home_recent_home_win_rate = home_recent_home_wins / home_recent_home_games if home_recent_home_games > 0 else 0.5
                        home_recent_away_win_rate = home_recent_away_wins / home_recent_away_games if home_recent_away_games > 0 else 0.5
                        
                        # 최근 N경기 평균 득점 및 실점
                        home_recent_avg_score = sum(g['score'] for g in home_recent_games) / len(home_recent_games)
                        home_recent_avg_allowed = sum(g['opponent_score'] for g in home_recent_games) / len(home_recent_games)
                        
                        # 최근 N경기 통계 저장
                        result_df.loc[idx, 'home_recent_win_rate'] = home_recent_win_rate
                        result_df.loc[idx, 'home_recent_home_win_rate'] = home_recent_home_win_rate
                        result_df.loc[idx, 'home_recent_away_win_rate'] = home_recent_away_win_rate
                        result_df.loc[idx, 'home_recent_avg_score'] = home_recent_avg_score
                        result_df.loc[idx, 'home_recent_avg_allowed'] = home_recent_avg_allowed
                        
                        # 타격 통계 평균 계산
                        for stat in batting_stat_fields:
                            values = [g['batting_stats'].get(stat, 0) for g in home_recent_games 
                                     if 'batting_stats' in g and stat in g['batting_stats']]
                            if values:
                                try:
                                    # 문자열을 실수로 변환
                                    numeric_values = []
                                    for v in values:
                                        try:
                                            numeric_values.append(float(v) if isinstance(v, str) else v)
                                        except (ValueError, TypeError):
                                            # 변환할 수 없는 값은 건너뜀
                                            self.logger.warning(f"홈팀 최근 {stat} 값을 변환할 수 없음: {v}")
                                    
                                    if numeric_values:
                                        result_df.loc[idx, f'home_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                                    else:
                                        result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
                                except Exception as e:
                                    self.logger.warning(f"홈팀 최근 {stat} 평균 계산 중 오류: {str(e)}")
                                    result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
                            else:
                                result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
            
            # 원정팀 통계 및 시즌 통계 추가 (홈팀과 동일한 로직)
            if away_team_id in team_games:
                # 현재 경기 이전의 원정팀 경기만 필터링
                away_previous_games = [g for g in team_games[away_team_id] if pd.to_datetime(g['date']) < game_date]
                
                if away_previous_games:
                    # 시즌 전체 원정팀 통계
                    away_total_games = len(away_previous_games)
                    away_wins = sum(1 for g in away_previous_games if g['won'])
                    away_losses = away_total_games - away_wins
                    away_home_games = sum(1 for g in away_previous_games if g['is_home'])
                    away_home_wins = sum(1 for g in away_previous_games if g['is_home'] and g['won'])
                    away_home_losses = away_home_games - away_home_wins
                    away_away_games = away_total_games - away_home_games
                    away_away_wins = away_wins - away_home_wins
                    away_away_losses = away_away_games - away_away_wins
                    
                    # 원정팀 승률 계산
                    away_win_rate = away_wins / away_total_games if away_total_games > 0 else 0.5
                    away_home_win_rate = away_home_wins / away_home_games if away_home_games > 0 else 0.5
                    away_away_win_rate = away_away_wins / away_away_games if away_away_games > 0 else 0.5
                    
                    # 원정팀 시즌 득점 및 실점 통계
                    away_total_runs_for = sum(g['score'] for g in away_previous_games)
                    away_total_runs_against = sum(g['opponent_score'] for g in away_previous_games)
                    away_avg_runs_for = away_total_runs_for / away_total_games if away_total_games > 0 else 0
                    away_avg_runs_against = away_total_runs_against / away_total_games if away_total_games > 0 else 0
                    
                    # 원정팀 시즌 타격 통계 평균 계산
                    away_batting_avgs = {}
                    for stat in batting_stat_fields:
                        values = [g['batting_stats'].get(stat, 0) for g in away_previous_games 
                                 if 'batting_stats' in g and stat in g['batting_stats']]
                        if values:
                            try:
                                # 문자열을 실수로 변환
                                numeric_values = []
                                for v in values:
                                    try:
                                        numeric_values.append(float(v) if isinstance(v, str) else v)
                                    except (ValueError, TypeError):
                                        # 변환할 수 없는 값은 건너뜀
                                        self.logger.warning(f"원정팀 {stat} 값을 변환할 수 없음: {v}")
                                
                                if numeric_values:
                                    away_batting_avgs[stat] = sum(numeric_values) / len(numeric_values)
                                else:
                                    away_batting_avgs[stat] = 0
                            except Exception as e:
                                self.logger.warning(f"원정팀 {stat} 평균 계산 중 오류: {str(e)}")
                                away_batting_avgs[stat] = 0
                        else:
                            away_batting_avgs[stat] = 0
                    
                    # 원정팀 시즌 투구 통계 평균 계산
                    away_pitching_avgs = {}
                    for stat in pitching_stat_fields:
                        values = [g['pitching_stats'].get(stat, 0) for g in away_previous_games 
                                 if 'pitching_stats' in g and stat in g['pitching_stats']]
                        if values:
                            try:
                                # 문자열을 실수로 변환
                                numeric_values = []
                                for v in values:
                                    try:
                                        numeric_values.append(float(v) if isinstance(v, str) else v)
                                    except (ValueError, TypeError):
                                        # 변환할 수 없는 값은 건너뜀
                                        self.logger.warning(f"원정팀 {stat} 값을 변환할 수 없음: {v}")
                                
                                if numeric_values:
                                    away_pitching_avgs[stat] = sum(numeric_values) / len(numeric_values)
                                else:
                                    away_pitching_avgs[stat] = 0
                            except Exception as e:
                                self.logger.warning(f"원정팀 {stat} 평균 계산 중 오류: {str(e)}")
                                away_pitching_avgs[stat] = 0
                        else:
                            away_pitching_avgs[stat] = 0
                    
                    # 원정팀 전적 문자열 생성 (승-패 형식)
                    away_overall_record = f"{away_wins}-{away_losses}"
                    away_home_record = f"{away_home_wins}-{away_home_losses}"
                    away_road_record = f"{away_away_wins}-{away_away_losses}"
                    
                    # 원정팀 시즌 통계 저장 (훈련 데이터와 일치하는 이름으로)
                    result_df.loc[idx, 'away_season_games'] = away_total_games  # 추가 정보로 유지
                    result_df.loc[idx, 'away_overall_record'] = away_overall_record
                    result_df.loc[idx, 'away_overall_record_win_rate'] = away_win_rate
                    result_df.loc[idx, 'away_home_record'] = away_home_record
                    result_df.loc[idx, 'away_road_record'] = away_road_record
                    result_df.loc[idx, 'away_home_record_win_rate'] = away_home_win_rate
                    result_df.loc[idx, 'away_road_record_win_rate'] = away_away_win_rate
                    result_df.loc[idx, 'away_avg_runs_for'] = away_avg_runs_for
                    result_df.loc[idx, 'away_avg_runs_against'] = away_avg_runs_against
                    
                    # 타격 통계 저장
                    for stat, value in away_batting_avgs.items():
                        result_df.loc[idx, f'away_avg_{stat}'] = value
                    
                    # 투구 통계 저장
                    for stat, value in away_pitching_avgs.items():
                        result_df.loc[idx, f'away_avg_{stat}'] = value
                    
                    # 원정팀 최근 N경기 통계
                    away_recent_games = away_previous_games[-n_recent_games:] if len(away_previous_games) >= n_recent_games else away_previous_games
                    
                    if away_recent_games:
                        # 최근 N경기 승률
                        away_recent_wins = sum(1 for g in away_recent_games if g['won'])
                        away_recent_win_rate = away_recent_wins / len(away_recent_games)
                        
                        # 최근 N경기 홈/원정 세분화 승률
                        away_recent_home_games = sum(1 for g in away_recent_games if g['is_home'])
                        away_recent_home_wins = sum(1 for g in away_recent_games if g['is_home'] and g['won'])
                        away_recent_away_games = len(away_recent_games) - away_recent_home_games
                        away_recent_away_wins = away_recent_wins - away_recent_home_wins
                        
                        away_recent_home_win_rate = away_recent_home_wins / away_recent_home_games if away_recent_home_games > 0 else 0.5
                        away_recent_away_win_rate = away_recent_away_wins / away_recent_away_games if away_recent_away_games > 0 else 0.5
                        
                        # 최근 N경기 평균 득점 및 실점
                        away_recent_avg_score = sum(g['score'] for g in away_recent_games) / len(away_recent_games)
                        away_recent_avg_allowed = sum(g['opponent_score'] for g in away_recent_games) / len(away_recent_games)
                        
                        # 최근 N경기 통계 저장
                        result_df.loc[idx, 'away_recent_win_rate'] = away_recent_win_rate
                        result_df.loc[idx, 'away_recent_home_win_rate'] = away_recent_home_win_rate
                        result_df.loc[idx, 'away_recent_away_win_rate'] = away_recent_away_win_rate
                        result_df.loc[idx, 'away_recent_avg_score'] = away_recent_avg_score
                        result_df.loc[idx, 'away_recent_avg_allowed'] = away_recent_avg_allowed
                        
                        # 타격 통계 평균 계산
                        for stat in batting_stat_fields:
                            values = [g['batting_stats'].get(stat, 0) for g in away_recent_games 
                                     if 'batting_stats' in g and stat in g['batting_stats']]
                            if values:
                                try:
                                    # 문자열을 실수로 변환
                                    numeric_values = []
                                    for v in values:
                                        try:
                                            numeric_values.append(float(v) if isinstance(v, str) else v)
                                        except (ValueError, TypeError):
                                            # 변환할 수 없는 값은 건너뜀
                                            self.logger.warning(f"원정팀 최근 {stat} 값을 변환할 수 없음: {v}")
                                    
                                    if numeric_values:
                                        result_df.loc[idx, f'away_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                                    else:
                                        result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
                                except Exception as e:
                                    self.logger.warning(f"원정팀 최근 {stat} 평균 계산 중 오류: {str(e)}")
                                    result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
                            else:
                                result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
            
            # 팀 간 상대 전적 추가
            self._add_head_to_head_stats(result_df, idx, home_team_id, away_team_id, team_games, game_date)
            
            # 팀 휴식일 수 추가
            self._add_rest_days(result_df, idx, home_team_id, away_team_id, team_games, game_date)
        
        # 파생 변수 추가
        self._add_derived_features(result_df)
        
        # 누락된 값 처리
        processed_df = self._handle_missing_values(result_df)
        if processed_df is not None:
            result_df = processed_df
        
        # 파편화 문제 해결을 위해 최종 복사본 생성
        result_df = result_df.copy()
        
        self.logger.info("예측 피처 생성 완료")
        self.prediction_data = result_df
        return result_df
    
    def _add_head_to_head_stats(self, df, idx, home_team_id, away_team_id, team_games, game_date):
        """양 팀 간의 상대 전적 추가"""
        # 홈팀의 경기에서 원정팀과의 대결 필터링
        home_vs_away = []
        if home_team_id in team_games:
            # 날짜 형식 통일 (문자열을 datetime으로 변환)
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            
            home_vs_away = [g for g in team_games[home_team_id] 
                            if g['opponent_id'] == away_team_id and pd.to_datetime(g['date']) < game_date]
        
        # 모든 h2h 경기 합치기
        h2h_games = len(home_vs_away)
        
        # 각 통계의 기본값 설정
        values = {
            'h2h_games': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_home_win_rate': 0.5,
            'home_vs_away_wins': 0,
            'home_vs_away_losses': 0,
            'home_vs_away_win_rate': 0.5,
            'home_vs_away_avg_score': 0,
            'home_vs_away_avg_allowed': 0,
            'away_vs_home_wins': 0,
            'away_vs_home_losses': 0,
            'away_vs_home_win_rate': 0.5,
            'away_vs_home_avg_score': 0,
            'away_vs_home_avg_allowed': 0
        }
        
        if h2h_games > 0:
            # 홈팀이 홈 경기에서 이긴 횟수 (홈팀 관점)
            home_wins = sum(1 for g in home_vs_away if g['is_home'] and g['won'])
            # 전체 홈 경기 (홈팀 관점)
            home_games = sum(1 for g in home_vs_away if g['is_home'])
            
            # 홈팀 상대 전적
            home_vs_away_wins = sum(1 for g in home_vs_away if g['won'])
            home_vs_away_losses = len(home_vs_away) - home_vs_away_wins
            
            # h2h 전적 통계
            values['h2h_games'] = h2h_games
            values['h2h_home_wins'] = home_wins
            values['h2h_away_wins'] = h2h_games - home_wins
            values['h2h_home_win_rate'] = home_wins / max(1, home_games)
            
            values['home_vs_away_wins'] = home_vs_away_wins
            values['home_vs_away_losses'] = home_vs_away_losses
            values['away_vs_home_wins'] = home_vs_away_losses
            values['away_vs_home_losses'] = home_vs_away_wins
            
            # 승률 통계
            values['home_vs_away_win_rate'] = home_vs_away_wins / max(1, len(home_vs_away))
            values['away_vs_home_win_rate'] = home_vs_away_losses / max(1, len(home_vs_away))
            
            # 득점 통계
            if home_vs_away:
                home_scores = []
                away_scores = []
                
                for g in home_vs_away:
                    try:
                        if g['is_home']:
                            # 문자열을 실수로 변환
                            home_score = float(g['score']) if isinstance(g['score'], str) else g['score']
                            away_score = float(g['opponent_score']) if isinstance(g['opponent_score'], str) else g['opponent_score']
                            home_scores.append(home_score)
                            away_scores.append(away_score)
                        else:
                            # 문자열을 실수로 변환
                            away_score = float(g['score']) if isinstance(g['score'], str) else g['score']
                            home_score = float(g['opponent_score']) if isinstance(g['opponent_score'], str) else g['opponent_score']
                            away_scores.append(away_score)
                            home_scores.append(home_score)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"상대전적 득점 변환 오류: {str(e)}")
                        continue
                
                # 홈/어웨이 팀 평균 득점
                if home_scores:
                    values['home_vs_away_avg_score'] = sum(home_scores) / len(home_scores)
                    values['home_vs_away_avg_allowed'] = sum(away_scores) / len(away_scores)
                
                if away_scores:
                    values['away_vs_home_avg_score'] = sum(away_scores) / len(away_scores)
                    values['away_vs_home_avg_allowed'] = sum(home_scores) / len(home_scores)
        
        # 한 번에 모든 값 할당
        for key, value in values.items():
            df.loc[idx, key] = value
    
    def _add_rest_days(self, df, idx, home_team_id, away_team_id, team_games, game_date):
        """팀별 휴식일 수 추가"""
        # 날짜 형식 통일 (문자열을 datetime으로 변환)
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # 초기 값 설정
        home_rest_days = 3  # 기본값
        away_rest_days = 3  # 기본값
        
        # 홈팀 이전 경기 날짜
        if home_team_id in team_games:
            home_previous_games = [g for g in team_games[home_team_id] if pd.to_datetime(g['date']) < game_date]
            if home_previous_games:
                home_prev_date = max(pd.to_datetime(g['date']) for g in home_previous_games)
                home_rest_days = (game_date - home_prev_date).days
        
        # 원정팀 이전 경기 날짜
        if away_team_id in team_games:
            away_previous_games = [g for g in team_games[away_team_id] if pd.to_datetime(g['date']) < game_date]
            if away_previous_games:
                away_prev_date = max(pd.to_datetime(g['date']) for g in away_previous_games)
                away_rest_days = (game_date - away_prev_date).days
        
        # 모든 값 한 번에 할당
        values = {
            'home_rest_days': home_rest_days,
            'away_rest_days': away_rest_days,
            'rest_advantage': home_rest_days - away_rest_days,
            'both_well_rested': 1 if home_rest_days >= 2 and away_rest_days >= 2 else 0,
            'both_tired': 1 if home_rest_days == 0 and away_rest_days == 0 else 0
        }
        
        for key, value in values.items():
            df.loc[idx, key] = value
    
    def _add_derived_features(self, df):
        """파생 변수 추가"""
        # 팀 간 기본 비교 특성
        compare_metrics = [
            # 최근 성적 비교
            'recent_win_rate', 'recent_avg_score', 'recent_avg_allowed',
            # 타격 지표 비교 
            'recent_avg_batting_avg', 'recent_avg_batting_ops', 'recent_avg_batting_homeRuns',
            # 투구 지표 비교
            'recent_avg_pitching_era', 'recent_avg_pitching_whip', 'rest_days',
            # 전체 시즌 성적 비교
            'overall_record_win_rate', 'home_record_win_rate', 'road_record_win_rate',
            'avg_runs_for', 'avg_runs_against',
            # 시즌 타격/투구 통계 비교
            'avg_batting_avg', 'avg_batting_ops', 'avg_batting_homeRuns',
            'avg_pitching_era', 'avg_pitching_whip',
            # 상대전적 비교
            'vs_away_win_rate', 'vs_home_win_rate'
        ]
        
        for metric in compare_metrics:
            home_col = f'home_{metric}'
            away_col = f'away_{metric}'
            
            if home_col in df.columns and away_col in df.columns:
                # 팀 간 차이 계산
                df[f'diff_{metric}'] = df[home_col] - df[away_col]
                
                # 팀 간 비율 계산 (0 나누기 방지)
                if metric.endswith('_rate') or metric.startswith('avg_'):
                    # ERA, WHIP은 낮을수록 좋으므로 반대로 계산 (away / home)
                    if 'era' in metric or 'whip' in metric:
                        safe_home = df[home_col].replace(0, 0.001)  # 0 나누기 방지
                        df[f'ratio_{metric}'] = df[away_col] / safe_home
                    else:
                        safe_away = df[away_col].replace(0, 0.001)  # 0 나누기 방지
                        df[f'ratio_{metric}'] = df[home_col] / safe_away
        
        # 공격력 지표 (최근 득점 + 시즌 득점 + 타격 지표)
        if 'home_recent_avg_score' in df.columns and 'home_avg_runs_for' in df.columns:
            # ops 컬럼이 없으면 대체값 사용
            if 'home_avg_batting_ops' in df.columns:
                # 홈팀 공격력
                df['home_offense_power'] = (
                    df['home_recent_avg_score'] + 
                    df['home_avg_runs_for'] + 
                    df['home_avg_batting_ops'] * 10
                ) / 3
            else:
                # ops 없이 계산
                df['home_offense_power'] = (
                    df['home_recent_avg_score'] + 
                    df['home_avg_runs_for']
                ) / 2
            
            # ops 컬럼이 없으면 대체값 사용
            if 'away_avg_batting_ops' in df.columns:
                # 원정팀 공격력 
                df['away_offense_power'] = (
                    df['away_recent_avg_score'] + 
                    df['away_avg_runs_for'] + 
                    df['away_avg_batting_ops'] * 10
                ) / 3
            else:
                # ops 없이 계산
                df['away_offense_power'] = (
                    df['away_recent_avg_score'] + 
                    df['away_avg_runs_for']
                ) / 2
            
            # 공격력 차이
            if 'home_offense_power' in df.columns and 'away_offense_power' in df.columns:
                df['diff_offense_power'] = df['home_offense_power'] - df['away_offense_power']
        
        # 수비력 지표 (최근 실점 + 시즌 실점 + 투수 지표) - 낮을수록 좋음
        if 'home_recent_avg_allowed' in df.columns and 'home_avg_runs_against' in df.columns:
            # 홈팀 수비력 (투수력)
            df['home_defense_power'] = (
                df['home_recent_avg_allowed'] + 
                df['home_avg_runs_against'] + 
                df['home_avg_pitching_era'] / 2
            ) / 3
            
            # 원정팀 수비력 (투수력)
            df['away_defense_power'] = (
                df['away_recent_avg_allowed'] + 
                df['away_avg_runs_against'] + 
                df['away_avg_pitching_era'] / 2
            ) / 3
            
            # 수비력은 낮을수록 좋으므로 차이를 계산할 때 부호 반전
            df['diff_defense_power'] = df['away_defense_power'] - df['home_defense_power']
        
        # 전체 팀 강도 지표 (공격력 - 수비력)
        if 'home_offense_power' in df.columns and 'home_defense_power' in df.columns:
            df['home_team_strength'] = df['home_offense_power'] - df['home_defense_power']
            df['away_team_strength'] = df['away_offense_power'] - df['away_defense_power']
            df['diff_team_strength'] = df['home_team_strength'] - df['away_team_strength']
        
        # 홈/원정 이점 지표
        if 'home_home_record_win_rate' in df.columns and 'away_road_record_win_rate' in df.columns:
            df['home_advantage'] = df['home_home_record_win_rate'] - df['home_road_record_win_rate']
            df['away_disadvantage'] = df['away_home_record_win_rate'] - df['away_road_record_win_rate']
            df['venue_factor'] = df['home_advantage'] + df['away_disadvantage']
            
        # 날짜 기반 특성
        if 'date' in df.columns:
            # 요일 (0=월요일, 6=일요일)
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            # 주말 여부 (토/일)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            # 월 (1=1월, 12=12월)
            df['month'] = pd.to_datetime(df['date']).dt.month
            # 시즌 초반 (3월-5월)
            df['early_season'] = df['month'].isin([3, 4, 5]).astype(int)
            # 시즌 중반 (6월-7월)
            df['mid_season'] = df['month'].isin([6, 7]).astype(int)
            # 시즌 후반 (8월-10월)
            df['late_season'] = df['month'].isin([8, 9, 10]).astype(int)
        
        return df
    
    def _handle_missing_values(self, df):
        """결측치 처리
        
        데이터셋의 결측치를 적절한 값으로 대체합니다.
        
        - 승률/비율 관련 컬럼은 적정 기본값으로 대체 (예: 승률 0.5)
        - 팀별 통계는 해당 팀의 평균값으로 대체
        - 남은 결측치는 평균값으로 대체
        - 그래도 남은 결측치는 0으로 대체
        
        Args:
            df: 결측치를 처리할 DataFrame
            
        Returns:
            pd.DataFrame: 결측치가 처리된 데이터프레임
        """
        self.logger.info("예측 데이터 결측치 처리 중...")
        
        # 입력 DataFrame이 없으면 오류
        if df is None or df.empty:
            self.logger.error("처리할 DataFrame이 없거나 비어 있습니다.")
            return None
        
        # 복사본 생성 (원본 보존)
        result_df = df.copy()
        
        # 결측치가 있는 컬럼 찾기
        missing_cols = [col for col in result_df.columns if result_df[col].isna().any()]
        self.logger.info(f"결측치가 있는 컬럼 수: {len(missing_cols)}")
        
        # MLB 관련 기본값 정의
        mlb_defaults = {
            'batting_avg': 0.250,  # MLB 평균 타율
            'batting_obp': 0.320,  # MLB 평균 출루율
            'batting_slg': 0.400,  # MLB 평균 장타율
            'batting_ops': 0.720,  # MLB 평균 OPS
            'pitching_era': 4.50,  # MLB 평균 ERA
            'pitching_whip': 1.30,  # MLB 평균 WHIP
            'win_rate': 0.500      # 기본 승률
        }
        
        # 결측치 처리 로직 (컬럼별로 적용)
        for col in missing_cols:
            missing = result_df[col].isna().sum()
            if missing > 0:
                self.logger.info(f"  - {col}: {missing}개 결측치 발견")
                
                # 1. 승률/비율 관련 컬럼은 적정 기본값으로 대체
                if any(metric in col for metric in ['win_rate', 'pct', 'avg', 'ops', 'era', 'whip']):
                    # 컬럼 유형에 따라 적절한 기본값 선택
                    if 'win_rate' in col:
                        default_value = mlb_defaults['win_rate']
                    elif 'batting_avg' in col:
                        default_value = mlb_defaults['batting_avg']
                    elif 'batting_obp' in col:
                        default_value = mlb_defaults['batting_obp']
                    elif 'batting_slg' in col:
                        default_value = mlb_defaults['batting_slg']
                    elif 'batting_ops' in col:
                        default_value = mlb_defaults['batting_ops']
                    elif 'era' in col:
                        default_value = mlb_defaults['pitching_era']
                    elif 'whip' in col:
                        default_value = mlb_defaults['pitching_whip']
                    else:
                        # 기타 비율 지표는 중앙값 사용
                        default_value = result_df[col].median()
                    
                    result_df[col].fillna(default_value, inplace=True)
                
                # 2. 팀별 통계는 해당 팀의 평균으로 대체
                elif col.startswith(('home_', 'away_')) and not col.endswith(('_id', '_score')):
                    team_type = col.split('_')[0]  # 'home' or 'away'
                    team_col = f"{team_type}_team_id"
                    
                    try:
                        # 숫자형 컬럼에만 적용
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            # 팀별 평균 계산
                            if team_col in result_df.columns:
                                # 모든 NaN값을 한번에 처리
                                team_means = result_df.groupby(team_col)[col].transform('mean')
                                result_df[col] = result_df[col].fillna(team_means)
                        else:
                            # 숫자형이 아닌 경우 mode(최빈값) 사용
                            if team_col in result_df.columns:
                                for team_id in result_df[team_col].unique():
                                    mask = result_df[team_col] == team_id
                                    mode_val = result_df.loc[mask, col].mode().iloc[0] if not result_df.loc[mask, col].dropna().empty else None
                                    if mode_val is not None:
                                        result_df.loc[mask & result_df[col].isna(), col] = mode_val
                    except Exception as e:
                        self.logger.warning(f"팀별 통계 처리 중 오류 발생 ({col}): {str(e)}")
                        # 오류 발생 시 중앙값으로 대체 시도
                        try:
                            if pd.api.types.is_numeric_dtype(result_df[col]):
                                result_df[col] = result_df[col].fillna(result_df[col].median())
                        except:
                            pass
                
                # 3. 남은 결측치는 컬럼 평균으로 대체
                if result_df[col].isna().any():
                    try:
                        # 숫자형 컬럼에만 평균 적용
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            result_df[col].fillna(result_df[col].mean(), inplace=True)
                        # 문자열 컬럼은 최빈값 적용
                        elif pd.api.types.is_object_dtype(result_df[col]) or pd.api.types.is_string_dtype(result_df[col]):
                            mode_val = result_df[col].mode().iloc[0] if not result_df[col].dropna().empty else ""
                            result_df[col].fillna(mode_val, inplace=True)
                    except Exception as e:
                        self.logger.warning(f"결측치 평균 대체 중 오류 발생 ({col}): {str(e)}")
                
                # 4. 그래도 남은 결측치는 0 또는 빈 문자열로 대체
                if result_df[col].isna().any():
                    try:
                        # 데이터 타입에 따라 적절한 기본값 사용
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            result_df[col].fillna(0, inplace=True)
                        elif pd.api.types.is_object_dtype(result_df[col]) or pd.api.types.is_string_dtype(result_df[col]):
                            result_df[col].fillna("", inplace=True)
                        else:
                            result_df[col].fillna(0, inplace=True)
                    except Exception as e:
                        self.logger.warning(f"결측치 최종 대체 중 오류 발생 ({col}): {str(e)}")
                        # 어떤 방법으로도 처리할 수 없는 경우 해당 컬럼 제거 고려
                        if result_df[col].isna().all():
                            self.logger.warning(f"컬럼 {col}의 모든 값이 결측치이므로 제거를 고려하세요.")
        
        self.logger.info("결측치 처리 완료")
        return result_df

    def process(self, n_recent_games=10):
        """전체 예측 데이터 처리 파이프라인 실행
        
        Args:
            n_recent_games: 최근 몇 경기를 고려할지 (기본값: 10)
            
        Returns:
            str: 저장된 파일 경로 또는 None (실패 시)
        """
        self.logger.info("예측 데이터 처리 파이프라인 시작")
        
        # 1. 데이터 로드
        if not self.load_records():
            self.logger.error("데이터 로드 실패")
            return None
        
        self.logger.info("데이터 로드 성공")
        
        # 2. 예측 피처 생성
        self.logger.info("예측 피처 생성 시작")
        prediction_data = self.create_prediction_features(n_recent_games=n_recent_games)
        
        if prediction_data is None:
            self.logger.error("예측 피처 생성 실패")
            return None
        
        self.logger.info(f"예측 피처 생성 완료: {len(prediction_data)} 행, {len(prediction_data.columns)} 열")
        self.prediction_data = prediction_data
        
        # 3. 데이터 저장
        self.logger.info("데이터 저장 시작")
        output_path = self.save_prediction_data()
        self.logger.info(f"데이터 저장 결과: {output_path}")
        
        return output_path

    def save_prediction_data(self):
        """예측 데이터 저장
        
        예측 피처가 추가된 데이터를 JSON 파일로 저장합니다.
        훈련 데이터와 동일한 피처 순서를 사용하여 비교를 용이하게 합니다.
        
        Returns:
            str: 저장된 파일 경로
        """
        self.logger.info("예측 데이터 저장 중...")
        
        # 예측 데이터 확인
        if not hasattr(self, 'prediction_data') or self.prediction_data is None or self.prediction_data.empty:
            self.logger.error("저장할 예측 데이터가 없거나 비어 있습니다.")
            return None
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlb_prediction_data_{timestamp}.json"
        output_path = self.prediction_dir / filename
        
        # 데이터프레임 복사
        df_copy = self.prediction_data.copy()
        
        # 디버깅 정보 출력
        self.logger.info(f"예측 데이터 크기: {df_copy.shape}, 컬럼: {list(df_copy.columns)}")
        
        # 중복된 컬럼 확인 및 제거
        duplicated_columns = df_copy.columns[df_copy.columns.duplicated()].tolist()
        if duplicated_columns:
            self.logger.warning(f"중복된 컬럼 발견: {duplicated_columns}")
            df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
        
        # 1. 기본 경기 정보 필드
        basic_info_fields = [
            'game_id', 'date', 'start_time', 'season', 'venue', 'venue_location',
            'game_type', 'game_state', 
            'home_team_id', 'home_team_name', 'home_team_abbrev', 
            'away_team_id', 'away_team_name', 'away_team_abbrev'
        ]
        
        # 2. 시즌 통계 필드
        season_stat_fields = [
            'home_overall_record', 'home_overall_record_win_rate', 
            'home_home_record', 'home_road_record',
            'home_home_record_win_rate', 'home_road_record_win_rate',
            'home_avg_runs_for', 'home_avg_runs_against',
            'away_overall_record', 'away_overall_record_win_rate', 
            'away_home_record', 'away_road_record',
            'away_home_record_win_rate', 'away_road_record_win_rate',
            'away_avg_runs_for', 'away_avg_runs_against'
        ]
        
        # 3. 최근 통계 필드
        recent_stat_fields = [
            'home_recent_win_rate', 'home_recent_home_win_rate', 'home_recent_away_win_rate', 
            'home_recent_avg_score', 'home_recent_avg_allowed',
            'away_recent_win_rate', 'away_recent_home_win_rate', 'away_recent_away_win_rate', 
            'away_recent_avg_score', 'away_recent_avg_allowed'
        ]
        
        # 4. 상대전적 필드
        h2h_fields = [col for col in df_copy.columns if 'h2h_' in col or '_vs_' in col]
        
        # 5. 휴식일 관련 필드
        rest_fields = [col for col in df_copy.columns if 'rest' in col]
        
        # 6. 팀 간 비교 필드
        diff_fields = [col for col in df_copy.columns if 'diff_' in col]
        ratio_fields = [col for col in df_copy.columns if 'ratio_' in col]
        
        # 7. 날짜/경기장 관련 파생 변수
        time_fields = [
            'day_of_week', 'is_weekend', 'month', 'early_season', 'mid_season', 'late_season'
        ]
        
        # 8. 추가 통계 필드 (타격/투구)
        batting_fields = [col for col in df_copy.columns if 'batting' in col and col not in basic_info_fields]
        pitching_fields = [col for col in df_copy.columns if 'pitching' in col and col not in basic_info_fields]
        
        # 9. 팀 강약 관련 필드
        strength_fields = []
        for field in ['home_offense_power', 'home_defense_power', 'home_team_strength',
                      'away_offense_power', 'away_defense_power', 'away_team_strength',
                      'home_advantage', 'away_disadvantage', 'venue_factor']:
            if field in df_copy.columns:
                strength_fields.append(field)
        
        # 모든 필드 모으기 (순서 지정)
        ordered_fields = []
        
        # 실제로 존재하는 필드만 필터링 (각 카테고리별로 확인)
        for field_list in [basic_info_fields, season_stat_fields, recent_stat_fields, h2h_fields, 
                          rest_fields, diff_fields, ratio_fields, time_fields, 
                          batting_fields, pitching_fields, strength_fields]:
            for field in field_list:
                if field in df_copy.columns and field not in ordered_fields:
                    ordered_fields.append(field)
        
        # 모든 컬럼이 포함됐는지 확인
        missing_columns = [col for col in df_copy.columns if col not in ordered_fields]
        if missing_columns:
            self.logger.warning(f"누락된 컬럼: {missing_columns}")
            # 누락된 컬럼도 추가
            ordered_fields.extend(missing_columns)
        
        self.logger.info(f"정렬할 컬럼 수: {len(ordered_fields)}, 데이터프레임 컬럼 수: {len(df_copy.columns)}")
        
        # 열 순서 재정렬 (안전하게 처리)
        try:
            df_copy = df_copy[ordered_fields]
        except Exception as e:
            self.logger.error(f"열 순서 재정렬 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 데이터프레임 유지
            self.logger.warning("원본 데이터프레임 순서를 유지합니다.")
        
        # 날짜/시간 열 변환 (JSON 직렬화 오류 방지)
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
        
        # JSON으로 변환/저장
        df_copy.to_json(output_path, orient='records', indent=2)
        
        self.logger.info(f"예측 데이터 저장 완료: {output_path}")
        return output_path


# 스크립트로 실행 시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = MLBPredictionDataProcessor(debug_mode=True)
    output_path = processor.process(n_recent_games=10)
    print(f"생성된 예측 데이터: {output_path}") 