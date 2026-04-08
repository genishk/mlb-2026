import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from collections import defaultdict
import numpy as np

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
    
    # *** collect_team_games 메서드 제거: 이제 create_prediction_features에서 직접 처리 ***
    
    def create_prediction_features(self, n_recent_games=10):
        """예정된 경기에 대한 예측 피처 생성
        
        과거 경기 데이터를 사용하여 예정된 경기에 대한 예측 피처를 생성합니다.
        *** 훈련데이터와 완전히 동일한 로직 사용 ***
        
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
        
        # *** 훈련데이터와 동일한 방식: 과거 데이터를 시간순으로 처리하여 팀별 누적 통계 구축 ***
        historical_df = self.historical_data.copy().sort_values('date')
        
        # 팀별 모든 경기 기록 (시간순으로 구축)
        team_games = defaultdict(list)
        
        # 팀별 시즌 통계 (훈련데이터와 완전히 동일한 구조)
        team_season_stats = defaultdict(lambda: {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'home_games': 0,
            'home_wins': 0,
            'home_losses': 0,
            'away_games': 0,
            'away_wins': 0,
            'away_losses': 0,
            'total_runs_for': 0,
            'total_runs_against': 0,
            # 평균 타격 통계 (누적값 저장)
            'total_batting_avg': 0,
            'total_batting_obp': 0,
            'total_batting_slg': 0,
            'total_batting_ops': 0,
            'total_batting_hits': 0,
            'total_batting_homeRuns': 0,
            'total_batting_runs': 0,
            'total_batting_rbi': 0,
            'total_batting_baseOnBalls': 0,
            'total_batting_strikeOuts': 0,
            # 평균 투구 통계 (누적값 저장)
            'total_pitching_era': 0,
            'total_pitching_whip': 0,
            'total_pitching_inningsPitched': 0,
            'total_pitching_hits': 0,
            'total_pitching_runs': 0,
            'total_pitching_earnedRuns': 0,
            'total_pitching_baseOnBalls': 0,
            'total_pitching_strikeOuts': 0,
            'total_pitching_homeRuns': 0
        })
        
        # *** 1단계: 과거 경기를 시간순으로 처리하여 팀별 누적 통계 구축 ***
        for idx, row in historical_df.iterrows():
            game_date = row['date']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            
            # 완료된 경기만 처리 (점수가 있는 경기)
            if 'home_score' in row and 'away_score' in row and \
               not pd.isna(row['home_score']) and not pd.isna(row['away_score']):
                
                home_score = float(row['home_score'])
                away_score = float(row['away_score'])
                
                # 승패 여부
                home_won = home_score > away_score
                
                # 홈팀 타격/투구 통계 수집
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
                
                # 원정팀 타격/투구 통계 수집
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
                
                # 홈팀 경기 기록 추가
                team_games[home_team_id].append({
                    'game_id': row['game_id'],
                    'date': game_date,
                    'opponent_id': away_team_id,
                    'is_home': True,
                    'won': home_won,
                    'score': home_score,
                    'opponent_score': away_score,
                    'batting_stats': home_batting_stats,
                    'pitching_stats': home_pitching_stats
                })
                
                # 홈팀 시즌 통계 업데이트 (훈련데이터와 동일한 방식)
                team_season_stats[home_team_id]['total_games'] += 1
                team_season_stats[home_team_id]['home_games'] += 1
                
                if home_won:
                    team_season_stats[home_team_id]['wins'] += 1
                    team_season_stats[home_team_id]['home_wins'] += 1
                else:
                    team_season_stats[home_team_id]['losses'] += 1
                    team_season_stats[home_team_id]['home_losses'] += 1
                
                team_season_stats[home_team_id]['total_runs_for'] += home_score
                team_season_stats[home_team_id]['total_runs_against'] += away_score
                
                # 홈팀 타격 통계 누적 업데이트
                for stat, value in home_batting_stats.items():
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[home_team_id][f'total_{stat}'] += numeric_value
                
                # 홈팀 투구 통계 누적 업데이트
                for stat, value in home_pitching_stats.items():
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[home_team_id][f'total_{stat}'] += numeric_value
                
                # 원정팀 경기 기록 추가
                team_games[away_team_id].append({
                    'game_id': row['game_id'],
                    'date': game_date,
                    'opponent_id': home_team_id,
                    'is_home': False,
                    'won': not home_won,
                    'score': away_score,
                    'opponent_score': home_score,
                    'batting_stats': away_batting_stats,
                    'pitching_stats': away_pitching_stats
                })
                
                # 원정팀 시즌 통계 업데이트 (훈련데이터와 동일한 방식)
                team_season_stats[away_team_id]['total_games'] += 1
                team_season_stats[away_team_id]['away_games'] += 1
                
                if not home_won:
                    team_season_stats[away_team_id]['wins'] += 1
                    team_season_stats[away_team_id]['away_wins'] += 1
                else:
                    team_season_stats[away_team_id]['losses'] += 1
                    team_season_stats[away_team_id]['away_losses'] += 1
                
                team_season_stats[away_team_id]['total_runs_for'] += away_score
                team_season_stats[away_team_id]['total_runs_against'] += home_score
                
                # 원정팀 타격 통계 누적 업데이트
                for stat, value in away_batting_stats.items():
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[away_team_id][f'total_{stat}'] += numeric_value
                
                # 원정팀 투구 통계 누적 업데이트
                for stat, value in away_pitching_stats.items():
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[away_team_id][f'total_{stat}'] += numeric_value
        
        # *** 2단계: 예정된 각 경기에 대해 피처 생성 (훈련데이터와 동일한 로직) ***
        for idx, row in result_df.iterrows():
            game_date = row['date']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            
            # 날짜 형식 확인 및 변환
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            
            # 홈팀 시즌 통계 추가 (훈련데이터와 완전히 동일한 로직)
            if team_season_stats[home_team_id]['total_games'] > 0:
                # 전체 승률 및 기록
                home_wins = team_season_stats[home_team_id]['wins']
                home_losses = team_season_stats[home_team_id]['losses']
                home_total_games = team_season_stats[home_team_id]['total_games']
                
                # 승패 레코드 문자열 (예: "15-10")
                result_df.loc[idx, 'home_overall_record'] = f"{home_wins}-{home_losses}"
                result_df.loc[idx, 'home_overall_record_win_rate'] = home_wins / home_total_games
                
                # 홈/원정 승률 및 기록
                home_home_wins = team_season_stats[home_team_id]['home_wins']
                home_home_losses = team_season_stats[home_team_id]['home_losses']
                home_away_wins = team_season_stats[home_team_id]['away_wins']
                home_away_losses = team_season_stats[home_team_id]['away_losses']
                
                # 홈/원정 레코드 문자열
                result_df.loc[idx, 'home_home_record'] = f"{home_home_wins}-{home_home_losses}"
                result_df.loc[idx, 'home_road_record'] = f"{home_away_wins}-{home_away_losses}"
                
                # 홈/원정 승률
                home_home_games = max(1, team_season_stats[home_team_id]['home_games'])
                home_away_games = max(1, team_season_stats[home_team_id]['away_games'])
                
                result_df.loc[idx, 'home_home_record_win_rate'] = home_home_wins / home_home_games
                result_df.loc[idx, 'home_road_record_win_rate'] = home_away_wins / home_away_games
                
                # 평균 득점/실점
                result_df.loc[idx, 'home_avg_runs_for'] = team_season_stats[home_team_id]['total_runs_for'] / home_total_games
                result_df.loc[idx, 'home_avg_runs_against'] = team_season_stats[home_team_id]['total_runs_against'] / home_total_games
                
                # 주요 타격 통계 평균 (훈련데이터와 동일: 누적값/경기수)
                for stat in batting_stat_fields:
                    if team_season_stats[home_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'home_avg_{stat}'] = team_season_stats[home_team_id][f'total_{stat}'] / home_total_games
                
                # 주요 투구 통계 평균 (훈련데이터와 동일: 누적값/경기수)
                for stat in pitching_stat_fields:
                    if team_season_stats[home_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'home_avg_{stat}'] = team_season_stats[home_team_id][f'total_{stat}'] / home_total_games
            else:
                # 팀 데이터가 없는 경우 기본값 (훈련데이터와 동일)
                result_df.loc[idx, 'home_overall_record'] = "0-0"
                result_df.loc[idx, 'home_overall_record_win_rate'] = 0.5
                result_df.loc[idx, 'home_home_record'] = "0-0"
                result_df.loc[idx, 'home_road_record'] = "0-0"
                result_df.loc[idx, 'home_home_record_win_rate'] = 0.5
                result_df.loc[idx, 'home_road_record_win_rate'] = 0.5
                result_df.loc[idx, 'home_avg_runs_for'] = 0
                result_df.loc[idx, 'home_avg_runs_against'] = 0
                
                # 타격 통계 기본값
                for stat in batting_stat_fields:
                    result_df.loc[idx, f'home_avg_{stat}'] = 0
                
                # 투구 통계 기본값
                for stat in pitching_stat_fields:
                    result_df.loc[idx, f'home_avg_{stat}'] = 0
            
            # 원정팀 시즌 통계 추가 (홈팀과 동일한 로직)
            if team_season_stats[away_team_id]['total_games'] > 0:
                # 전체 승률 및 기록
                away_wins = team_season_stats[away_team_id]['wins']
                away_losses = team_season_stats[away_team_id]['losses']
                away_total_games = team_season_stats[away_team_id]['total_games']
                
                result_df.loc[idx, 'away_overall_record'] = f"{away_wins}-{away_losses}"
                result_df.loc[idx, 'away_overall_record_win_rate'] = away_wins / away_total_games
                
                # 홈/원정 승률 및 기록
                away_home_wins = team_season_stats[away_team_id]['home_wins']
                away_home_losses = team_season_stats[away_team_id]['home_losses']
                away_away_wins = team_season_stats[away_team_id]['away_wins']
                away_away_losses = team_season_stats[away_team_id]['away_losses']
                
                result_df.loc[idx, 'away_home_record'] = f"{away_home_wins}-{away_home_losses}"
                result_df.loc[idx, 'away_road_record'] = f"{away_away_wins}-{away_away_losses}"
                
                away_home_games = max(1, team_season_stats[away_team_id]['home_games'])
                away_away_games = max(1, team_season_stats[away_team_id]['away_games'])
                
                result_df.loc[idx, 'away_home_record_win_rate'] = away_home_wins / away_home_games
                result_df.loc[idx, 'away_road_record_win_rate'] = away_away_wins / away_away_games
                
                # 평균 득점/실점
                result_df.loc[idx, 'away_avg_runs_for'] = team_season_stats[away_team_id]['total_runs_for'] / away_total_games
                result_df.loc[idx, 'away_avg_runs_against'] = team_season_stats[away_team_id]['total_runs_against'] / away_total_games
                
                # 주요 타격 통계 평균 (훈련데이터와 동일: 누적값/경기수)
                for stat in batting_stat_fields:
                    if team_season_stats[away_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'away_avg_{stat}'] = team_season_stats[away_team_id][f'total_{stat}'] / away_total_games
                
                # 주요 투구 통계 평균 (훈련데이터와 동일: 누적값/경기수)
                for stat in pitching_stat_fields:
                    if team_season_stats[away_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'away_avg_{stat}'] = team_season_stats[away_team_id][f'total_{stat}'] / away_total_games
            else:
                # 팀 데이터가 없는 경우 기본값 (훈련데이터와 동일)
                result_df.loc[idx, 'away_overall_record'] = "0-0"
                result_df.loc[idx, 'away_overall_record_win_rate'] = 0.5
                result_df.loc[idx, 'away_home_record'] = "0-0"
                result_df.loc[idx, 'away_road_record'] = "0-0"
                result_df.loc[idx, 'away_home_record_win_rate'] = 0.5
                result_df.loc[idx, 'away_road_record_win_rate'] = 0.5
                result_df.loc[idx, 'away_avg_runs_for'] = 0
                result_df.loc[idx, 'away_avg_runs_against'] = 0
                
                # 타격 통계 기본값
                for stat in batting_stat_fields:
                    result_df.loc[idx, f'away_avg_{stat}'] = 0
                
                # 투구 통계 기본값
                for stat in pitching_stat_fields:
                    result_df.loc[idx, f'away_avg_{stat}'] = 0
            
            # 홈팀 최근 N경기 통계 계산 (훈련데이터와 동일한 로직)
            home_previous_games = [g for g in team_games[home_team_id] if pd.to_datetime(g['date']) < game_date]
            
            if len(home_previous_games) > 0:
                # 최근 N경기만 선택
                recent_games = home_previous_games[-n_recent_games:] if len(home_previous_games) >= n_recent_games else home_previous_games
                
                # 최근 N경기 승률
                result_df.loc[idx, 'home_recent_win_rate'] = sum(1 for g in recent_games if g['won']) / len(recent_games)
                
                # 최근 N경기 평균 득점/실점
                result_df.loc[idx, 'home_recent_avg_score'] = sum(g['score'] for g in recent_games) / len(recent_games)
                result_df.loc[idx, 'home_recent_avg_allowed'] = sum(g['opponent_score'] for g in recent_games) / len(recent_games)
                
                # 최근 N경기 홈/원정 승률
                home_games = [g for g in recent_games if g['is_home']]
                away_games = [g for g in recent_games if not g['is_home']]
                
                if home_games:
                    result_df.loc[idx, 'home_recent_home_win_rate'] = sum(1 for g in home_games if g['won']) / len(home_games)
                else:
                    result_df.loc[idx, 'home_recent_home_win_rate'] = 0.5
                    
                if away_games:
                    result_df.loc[idx, 'home_recent_away_win_rate'] = sum(1 for g in away_games if g['won']) / len(away_games)
                else:
                    result_df.loc[idx, 'home_recent_away_win_rate'] = 0.5
                
                # 최근 N경기 타격 통계 평균 (훈련데이터와 동일한 방식)
                for stat in batting_stat_fields:
                    values = [g['batting_stats'].get(stat, 0) for g in recent_games if 'batting_stats' in g and stat in g['batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
                
                # 최근 N경기 투구 통계 평균 (훈련데이터와 동일한 방식)
                for stat in pitching_stat_fields:
                    values = [g['pitching_stats'].get(stat, 0) for g in recent_games if 'pitching_stats' in g and stat in g['pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
            else:
                # 이전 경기가 없는 경우 기본값 (훈련데이터와 동일)
                result_df.loc[idx, 'home_recent_win_rate'] = 0.5
                result_df.loc[idx, 'home_recent_avg_score'] = 0
                result_df.loc[idx, 'home_recent_avg_allowed'] = 0
                result_df.loc[idx, 'home_recent_home_win_rate'] = 0.5
                result_df.loc[idx, 'home_recent_away_win_rate'] = 0.5
                
                # 타격 통계 기본값
                for stat in batting_stat_fields:
                    result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
                
                # 투구 통계 기본값
                for stat in pitching_stat_fields:
                    result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
            
            # 원정팀 최근 N경기 통계 계산 (홈팀과 동일한 로직)
            away_previous_games = [g for g in team_games[away_team_id] if pd.to_datetime(g['date']) < game_date]
            
            if len(away_previous_games) > 0:
                # 최근 N경기만 선택
                recent_games = away_previous_games[-n_recent_games:] if len(away_previous_games) >= n_recent_games else away_previous_games
                
                # 최근 N경기 승률
                result_df.loc[idx, 'away_recent_win_rate'] = sum(1 for g in recent_games if g['won']) / len(recent_games)
                
                # 최근 N경기 평균 득점/실점
                result_df.loc[idx, 'away_recent_avg_score'] = sum(g['score'] for g in recent_games) / len(recent_games)
                result_df.loc[idx, 'away_recent_avg_allowed'] = sum(g['opponent_score'] for g in recent_games) / len(recent_games)
                
                # 최근 N경기 홈/원정 승률
                home_games = [g for g in recent_games if g['is_home']]
                away_games = [g for g in recent_games if not g['is_home']]
                
                if home_games:
                    result_df.loc[idx, 'away_recent_home_win_rate'] = sum(1 for g in home_games if g['won']) / len(home_games)
                else:
                    result_df.loc[idx, 'away_recent_home_win_rate'] = 0.5
                    
                if away_games:
                    result_df.loc[idx, 'away_recent_away_win_rate'] = sum(1 for g in away_games if g['won']) / len(away_games)
                else:
                    result_df.loc[idx, 'away_recent_away_win_rate'] = 0.5
                
                # 최근 N경기 타격 통계 평균 (훈련데이터와 동일한 방식)
                for stat in batting_stat_fields:
                    values = [g['batting_stats'].get(stat, 0) for g in recent_games if 'batting_stats' in g and stat in g['batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
                
                # 최근 N경기 투구 통계 평균 (훈련데이터와 동일한 방식)
                for stat in pitching_stat_fields:
                    values = [g['pitching_stats'].get(stat, 0) for g in recent_games if 'pitching_stats' in g and stat in g['pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
            else:
                # 이전 경기가 없는 경우 기본값 (훈련데이터와 동일)
                result_df.loc[idx, 'away_recent_win_rate'] = 0.5
                result_df.loc[idx, 'away_recent_avg_score'] = 0
                result_df.loc[idx, 'away_recent_avg_allowed'] = 0
                result_df.loc[idx, 'away_recent_home_win_rate'] = 0.5
                result_df.loc[idx, 'away_recent_away_win_rate'] = 0.5
                
                # 타격 통계 기본값
                for stat in batting_stat_fields:
                    result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
                
                # 투구 통계 기본값
                for stat in pitching_stat_fields:
                    result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
            
            # *** 상대전적과 휴식일 처리 (훈련데이터와 동일한 순서) ***
            # 상대전적 정보 추가
            self._add_head_to_head_stats(result_df, idx, home_team_id, away_team_id, team_games, game_date)
            
            # 휴식일 수 추가
            self._add_rest_days(result_df, idx, home_team_id, away_team_id, team_games, game_date)
        
        # 파생 변수 추가
        self._add_derived_features(result_df)
        
        # 누락된 값 처리
        processed_df = self._handle_missing_values(result_df)
        if processed_df is not None:
            result_df = processed_df
        
        # 파편화 문제 해결을 위해 최종 복사본 생성
        result_df = result_df.copy()
        
        self.logger.info("예측 피처 생성 완료 (훈련데이터와 동일한 로직 적용)")
        self.prediction_data = result_df
        return result_df
    
    def _add_head_to_head_stats(self, df, idx, home_team_id, away_team_id, team_games, game_date):
        """양 팀 간의 상대 전적 추가 (훈련데이터와 완전히 동일한 로직)"""
        
        # *** 훈련데이터와 완전히 동일한 방식: 팀 ID 순서 그대로 사용 (방향성 유지) ***
        team_key = (home_team_id, away_team_id)
        
        # 날짜 형식 통일 (문자열을 datetime으로 변환)
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # *** 훈련데이터와 완전히 동일한 방식: h2h_records 딕셔너리 사용 ***
        # 클래스 변수로 h2h_records가 없으면 생성
        if not hasattr(self, 'h2h_records'):
            self.h2h_records = {}
        
        # 현재 경기 이전의 상대전적만 계산 (훈련데이터와 동일한 방식)
        if team_key not in self.h2h_records:
            # 과거 경기에서 상대전적 구축
            home_wins = 0
            away_wins = 0
            total_games = 0
            recent_games = []
            
            # 홈팀 관점에서 과거 경기 찾기
            if home_team_id in team_games:
                past_h2h_games = [g for g in team_games[home_team_id] 
                                if g['opponent_id'] == away_team_id and pd.to_datetime(g['date']) < game_date]
                
                for g in past_h2h_games:
                    total_games += 1
                    if g['won']:
                        home_wins += 1
                    else:
                        away_wins += 1
                    
                    # 최근 5경기만 저장 (훈련데이터와 동일)
                    if len(recent_games) < 5:
                        recent_games.append({
                            'date': g['date'],
                            'home_score': g['score'] if g['is_home'] else g['opponent_score'],
                            'away_score': g['opponent_score'] if g['is_home'] else g['score'],
                            'home_batting_stats': g['batting_stats'] if g['is_home'] else {},
                            'home_pitching_stats': g['pitching_stats'] if g['is_home'] else {},
                            'away_batting_stats': {} if g['is_home'] else g['batting_stats'],
                            'away_pitching_stats': {} if g['is_home'] else g['pitching_stats']
                        })
            
            self.h2h_records[team_key] = {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'total_games': total_games,
                'recent_games': recent_games
            }
        
        # 현재 시점까지의 상대 전적 추출 (훈련데이터와 동일)
        home_wins = self.h2h_records[team_key]['home_wins']
        away_wins = self.h2h_records[team_key]['away_wins']
        total_games = self.h2h_records[team_key]['total_games']
        recent_games = self.h2h_records[team_key]['recent_games']
        
        # 주요 타격 통계 필드 정의 (훈련데이터와 동일)
        batting_stat_fields = [
            'batting_avg', 'batting_ops', 'batting_hits', 'batting_homeRuns'
        ]
        
        # 주요 투구 통계 필드 정의 (훈련데이터와 동일)
        pitching_stat_fields = [
            'pitching_era', 'pitching_whip', 'pitching_strikeOuts'
        ]
        
        # 상대 전적 승패 기록 (훈련데이터와 동일)
        df.loc[idx, 'home_vs_away_wins'] = home_wins
        df.loc[idx, 'home_vs_away_losses'] = away_wins
        df.loc[idx, 'away_vs_home_wins'] = away_wins
        df.loc[idx, 'away_vs_home_losses'] = home_wins
        
        # 승률 계산 (0 나누기 방지, 훈련데이터와 동일)
        if total_games > 0:
            df.loc[idx, 'home_vs_away_win_rate'] = home_wins / total_games
            df.loc[idx, 'away_vs_home_win_rate'] = away_wins / total_games
        else:
            # 이전 상대전적이 없는 경우 0.5로 기본값 설정 (훈련데이터와 동일)
            df.loc[idx, 'home_vs_away_win_rate'] = 0.5
            df.loc[idx, 'away_vs_home_win_rate'] = 0.5
        
        # 최근 상대 전적 기반 통계 (평균 득점, 실점 등, 훈련데이터와 동일)
        if recent_games:
            # 홈팀 상대전 평균 득점/실점 (훈련데이터와 동일)
            df.loc[idx, 'home_vs_away_avg_score'] = sum(game['home_score'] for game in recent_games) / len(recent_games)
            df.loc[idx, 'home_vs_away_avg_allowed'] = sum(game['away_score'] for game in recent_games) / len(recent_games)
            
            # 원정팀 상대전 평균 득점/실점 (훈련데이터와 동일)
            df.loc[idx, 'away_vs_home_avg_score'] = sum(game['away_score'] for game in recent_games) / len(recent_games)
            df.loc[idx, 'away_vs_home_avg_allowed'] = sum(game['home_score'] for game in recent_games) / len(recent_games)
            
            # 홈팀 주요 타격 통계 평균 (훈련데이터와 동일)
            for stat in batting_stat_fields:
                values = [game['home_batting_stats'].get(stat, 0) for game in recent_games if 'home_batting_stats' in game and stat in game['home_batting_stats']]
                if values:
                    # 문자열을 실수로 변환 (훈련데이터와 동일)
                    numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                    df.loc[idx, f'home_vs_away_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                else:
                    df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
            
            # 홈팀 주요 투구 통계 평균 (훈련데이터와 동일)
            for stat in pitching_stat_fields:
                values = [game['home_pitching_stats'].get(stat, 0) for game in recent_games if 'home_pitching_stats' in game and stat in game['home_pitching_stats']]
                if values:
                    # 문자열을 실수로 변환 (훈련데이터와 동일)
                    numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                    df.loc[idx, f'home_vs_away_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                else:
                    df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
            
            # 원정팀 주요 타격 통계 평균 (훈련데이터와 동일)
            for stat in batting_stat_fields:
                values = [game['away_batting_stats'].get(stat, 0) for game in recent_games if 'away_batting_stats' in game and stat in game['away_batting_stats']]
                if values:
                    # 문자열을 실수로 변환 (훈련데이터와 동일)
                    numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                    df.loc[idx, f'away_vs_home_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                else:
                    df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
            
            # 원정팀 주요 투구 통계 평균 (훈련데이터와 동일)
            for stat in pitching_stat_fields:
                values = [game['away_pitching_stats'].get(stat, 0) for game in recent_games if 'away_pitching_stats' in game and stat in game['away_pitching_stats']]
                if values:
                    # 문자열을 실수로 변환 (훈련데이터와 동일)
                    numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                    df.loc[idx, f'away_vs_home_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                else:
                    df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
        else:
            # 이전 상대전적이 없는 경우 기본값 설정 (훈련데이터와 동일)
            df.loc[idx, 'home_vs_away_avg_score'] = 0
            df.loc[idx, 'home_vs_away_avg_allowed'] = 0
            df.loc[idx, 'away_vs_home_avg_score'] = 0
            df.loc[idx, 'away_vs_home_avg_allowed'] = 0
            
            # 타격 통계 기본값 (훈련데이터와 동일)
            for stat in batting_stat_fields:
                df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
            
            # 투구 통계 기본값 (훈련데이터와 동일)
            for stat in pitching_stat_fields:
                df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
    
    def _add_rest_days(self, df, idx, home_team_id, away_team_id, team_games, game_date):
        """팀별 휴식일 수 추가 (훈련데이터와 완전히 동일한 로직)"""
        # 날짜 형식 통일 (문자열을 datetime으로 변환)
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # 초기 값 설정 (훈련데이터와 동일)
        home_rest_days = 5  # 기본값 (시즌 첫 경기 가정)
        away_rest_days = 5  # 기본값 (시즌 첫 경기 가정)
        
        # 홈팀 이전 경기 날짜 (훈련데이터와 동일한 방식)
        if home_team_id in team_games:
            home_previous_games = [g for g in team_games[home_team_id] if pd.to_datetime(g['date']) < game_date]
            if home_previous_games:
                home_prev_date = max(pd.to_datetime(g['date']) for g in home_previous_games)
                
                # 날짜 차이 계산 (훈련데이터와 동일)
                if isinstance(game_date, pd.Timestamp) and isinstance(home_prev_date, pd.Timestamp):
                    home_rest_days = (game_date.date() - home_prev_date.date()).days
                else:
                    # 문자열을 datetime으로 변환
                    current_dt = pd.to_datetime(game_date).date() if not isinstance(game_date, pd.Timestamp) else game_date.date()
                    last_dt = pd.to_datetime(home_prev_date).date() if not isinstance(home_prev_date, pd.Timestamp) else home_prev_date.date()
                    home_rest_days = (current_dt - last_dt).days
                
                home_rest_days = max(0, home_rest_days)
        
        # 원정팀 이전 경기 날짜 (훈련데이터와 동일한 방식)
        if away_team_id in team_games:
            away_previous_games = [g for g in team_games[away_team_id] if pd.to_datetime(g['date']) < game_date]
            if away_previous_games:
                away_prev_date = max(pd.to_datetime(g['date']) for g in away_previous_games)
                
                # 날짜 차이 계산 (훈련데이터와 동일)
                if isinstance(game_date, pd.Timestamp) and isinstance(away_prev_date, pd.Timestamp):
                    away_rest_days = (game_date.date() - away_prev_date.date()).days
                else:
                    # 문자열을 datetime으로 변환
                    current_dt = pd.to_datetime(game_date).date() if not isinstance(game_date, pd.Timestamp) else game_date.date()
                    last_dt = pd.to_datetime(away_prev_date).date() if not isinstance(away_prev_date, pd.Timestamp) else away_prev_date.date()
                    away_rest_days = (current_dt - last_dt).days
                
                away_rest_days = max(0, away_rest_days)
        
        # 모든 값 한 번에 할당 (훈련데이터와 동일)
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
        self.logger.info("예측 데이터 처리 파이프라인 시작 (훈련데이터와 동일한 순서)")
        
        # 1. 데이터 로드
        if not self.load_records():
            self.logger.error("데이터 로드 실패")
            return None
        
        self.logger.info("데이터 로드 성공")
        
        # 2. 예측 피처 생성 (시즌 통계 + 최근 통계)
        self.logger.info("예측 피처 생성 시작")
        prediction_data = self.create_prediction_features(n_recent_games=n_recent_games)
        
        if prediction_data is None:
            self.logger.error("예측 피처 생성 실패")
            return None
        
        self.logger.info(f"예측 피처 생성 완료: {len(prediction_data)} 행, {len(prediction_data.columns)} 열")
        self.prediction_data = prediction_data
        
        # 3. 상대 전적 정보 추가 (훈련데이터와 동일한 순서)
        self.logger.info("상대 전적 정보 추가 시작")
        self.add_head_to_head_stats()
        
        # 4. 휴식일 수 추가 (훈련데이터와 동일한 순서)
        self.logger.info("휴식일 수 정보 추가 시작")
        self.add_rest_days()
        
        # 5. 결측치 처리 (훈련데이터와 동일한 순서)
        self.logger.info("결측치 처리 시작")
        self.handle_missing_values()
        
        # 6. 특성 생성 (파생 변수 추가, 훈련데이터와 동일한 순서)
        self.logger.info("파생 변수 생성 시작")
        self.create_features()
        
        # 7. 고급 특성 추가 (훈련데이터와 동일한 순서)
        self.logger.info("고급 특성 생성 시작")
        self.prediction_data = self._add_advanced_features(self.prediction_data)
        
        # 8. 데이터 저장
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
    
    def add_head_to_head_stats(self):
        """상대 전적 정보 추가 (훈련데이터와 동일한 인터페이스)"""
        self.logger.info("상대 전적 정보 추가 중...")
        
        if self.prediction_data is None:
            self.logger.error("예측 데이터가 없습니다. create_prediction_features()를 먼저 호출하세요.")
            return None
        
        # 이미 create_prediction_features에서 처리되었으므로 별도 처리 불필요
        # 하지만 인터페이스 일관성을 위해 메서드 제공
        self.logger.info("상대 전적 정보는 이미 create_prediction_features에서 처리되었습니다.")
        return self.prediction_data
    
    def add_rest_days(self):
        """휴식일 수 정보 추가 (훈련데이터와 동일한 인터페이스)"""
        self.logger.info("휴식일 수 정보 추가 중...")
        
        if self.prediction_data is None:
            self.logger.error("예측 데이터가 없습니다. create_prediction_features()를 먼저 호출하세요.")
            return None
        
        # 이미 create_prediction_features에서 처리되었으므로 별도 처리 불필요
        # 하지만 인터페이스 일관성을 위해 메서드 제공
        self.logger.info("휴식일 수 정보는 이미 create_prediction_features에서 처리되었습니다.")
        return self.prediction_data
    
    def handle_missing_values(self):
        """결측치 처리 (훈련데이터와 동일한 인터페이스)"""
        self.logger.info("결측치 처리 중...")
        
        if self.prediction_data is None:
            self.logger.error("예측 데이터가 없습니다. create_prediction_features()를 먼저 호출하세요.")
            return None
        
        # 이미 create_prediction_features에서 처리되었으므로 별도 처리 불필요
        # 하지만 인터페이스 일관성을 위해 메서드 제공
        self.logger.info("결측치 처리는 이미 create_prediction_features에서 처리되었습니다.")
        return self.prediction_data
    
    def create_features(self):
        """파생 변수 생성 (훈련데이터와 동일한 인터페이스)"""
        self.logger.info("파생 변수 생성 중...")
        
        if self.prediction_data is None:
            self.logger.error("예측 데이터가 없습니다. create_prediction_features()를 먼저 호출하세요.")
            return None
        
        # 이미 create_prediction_features에서 처리되었으므로 별도 처리 불필요
        # 하지만 인터페이스 일관성을 위해 메서드 제공
        self.logger.info("파생 변수 생성은 이미 create_prediction_features에서 처리되었습니다.")
        return self.prediction_data
    
    def _add_advanced_features(self, df):
        """고급 특성 생성 (훈련 데이터와 100% 동일한 로직)
        
        특성 상호작용, 모멘텀 지표, 상대적 순위, 비선형 변환 등의 
        고급 특성을 생성합니다.
        
        Args:
            df: 특성을 추가할 DataFrame
            
        Returns:
            pd.DataFrame: 고급 특성이 추가된 데이터프레임
        """
        self.logger.info("고급 특성 생성 중...")
        
        # 입력 DataFrame이 없으면 오류
        if df is None or df.empty:
            self.logger.error("처리할 DataFrame이 없거나 비어 있습니다.")
            return None
        
        # 복사본 생성 (원본 보존)
        result_df = df.copy()
        
        # 1. 특성 상호작용 (Feature Interactions)
        self.logger.info("특성 상호작용 생성 중...")
        
        # 타격력 × 투수력 상호작용
        if all(col in result_df.columns for col in ['home_avg_batting_ops', 'away_avg_pitching_era']):
            result_df['home_batting_vs_away_pitching'] = result_df['home_avg_batting_ops'] / (result_df['away_avg_pitching_era'] + 0.01)
            result_df['away_batting_vs_home_pitching'] = result_df['away_avg_batting_ops'] / (result_df['home_avg_pitching_era'] + 0.01)
            result_df['batting_pitching_advantage'] = result_df['home_batting_vs_away_pitching'] - result_df['away_batting_vs_home_pitching']
        
        # 홈 어드밴티지 × 팀 강도 상호작용
        if all(col in result_df.columns for col in ['home_advantage', 'home_recent_win_rate']):
            result_df['home_advantage_strength'] = result_df['home_advantage'] * result_df['home_recent_win_rate']
            result_df['away_disadvantage_strength'] = result_df['away_disadvantage'] * result_df['away_recent_win_rate']
            result_df['venue_strength_factor'] = result_df['home_advantage_strength'] - result_df['away_disadvantage_strength']
        
        # 휴식일 × 최근 성적 상호작용
        if all(col in result_df.columns for col in ['home_rest_days', 'home_recent_win_rate']):
            result_df['home_rest_performance'] = result_df['home_rest_days'] * result_df['home_recent_win_rate']
            result_df['away_rest_performance'] = result_df['away_rest_days'] * result_df['away_recent_win_rate']
            result_df['rest_performance_diff'] = result_df['home_rest_performance'] - result_df['away_rest_performance']
        
        # 2. 모멘텀 지표 (Momentum Indicators)
        self.logger.info("모멘텀 지표 생성 중...")
        
        # 예측 데이터의 경우 historical_data를 사용하여 모멘텀 계산
        if hasattr(self, 'historical_data') and self.historical_data is not None:
            # 팀별 모멘텀 계산을 위한 데이터 구조
            team_momentum = defaultdict(lambda: {
                'games': [],
                'recent_3_wins': 0,
                'recent_5_wins': 0,
                'recent_7_wins': 0,
                'recent_3_scores': [],
                'recent_5_scores': [],
                'recent_7_scores': []
            })
            
            # 과거 데이터로 팀별 모멘텀 구축
            historical_sorted = self.historical_data.sort_values('date')
            
            for _, hist_row in historical_sorted.iterrows():
                if 'home_score' in hist_row and 'away_score' in hist_row and \
                   not pd.isna(hist_row['home_score']) and not pd.isna(hist_row['away_score']):
                    
                    home_id = hist_row['home_team_id']
                    away_id = hist_row['away_team_id']
                    home_score = float(hist_row['home_score'])
                    away_score = float(hist_row['away_score'])
                    home_won = home_score > away_score
                    game_date = hist_row['date']
                    
                    # 홈팀 기록 업데이트
                    team_momentum[home_id]['games'].append({
                        'date': game_date,
                        'won': home_won,
                        'score': home_score,
                        'opponent_score': away_score
                    })
                    
                    # 원정팀 기록 업데이트
                    team_momentum[away_id]['games'].append({
                        'date': game_date,
                        'won': not home_won,
                        'score': away_score,
                        'opponent_score': home_score
                    })
            
            # 예측 데이터에 모멘텀 지표 추가
            for idx, row in result_df.iterrows():
                home_id = row['home_team_id']
                away_id = row['away_team_id']
                
                # 홈팀 모멘텀
                home_games = team_momentum[home_id]['games']
                if len(home_games) >= 3:
                    recent_3 = home_games[-3:]
                    result_df.loc[idx, 'home_momentum_3'] = sum(1 for g in recent_3 if g['won']) / 3
                    result_df.loc[idx, 'home_scoring_trend_3'] = sum(g['score'] for g in recent_3) / 3
                else:
                    result_df.loc[idx, 'home_momentum_3'] = 0.5
                    result_df.loc[idx, 'home_scoring_trend_3'] = 0
                    
                if len(home_games) >= 5:
                    recent_5 = home_games[-5:]
                    result_df.loc[idx, 'home_momentum_5'] = sum(1 for g in recent_5 if g['won']) / 5
                    result_df.loc[idx, 'home_scoring_trend_5'] = sum(g['score'] for g in recent_5) / 5
                else:
                    result_df.loc[idx, 'home_momentum_5'] = 0.5
                    result_df.loc[idx, 'home_scoring_trend_5'] = 0
                    
                if len(home_games) >= 7:
                    recent_7 = home_games[-7:]
                    result_df.loc[idx, 'home_momentum_7'] = sum(1 for g in recent_7 if g['won']) / 7
                    result_df.loc[idx, 'home_scoring_trend_7'] = sum(g['score'] for g in recent_7) / 7
                else:
                    result_df.loc[idx, 'home_momentum_7'] = 0.5
                    result_df.loc[idx, 'home_scoring_trend_7'] = 0
                
                # 원정팀 모멘텀
                away_games = team_momentum[away_id]['games']
                if len(away_games) >= 3:
                    recent_3 = away_games[-3:]
                    result_df.loc[idx, 'away_momentum_3'] = sum(1 for g in recent_3 if g['won']) / 3
                    result_df.loc[idx, 'away_scoring_trend_3'] = sum(g['score'] for g in recent_3) / 3
                else:
                    result_df.loc[idx, 'away_momentum_3'] = 0.5
                    result_df.loc[idx, 'away_scoring_trend_3'] = 0
                    
                if len(away_games) >= 5:
                    recent_5 = away_games[-5:]
                    result_df.loc[idx, 'away_momentum_5'] = sum(1 for g in recent_5 if g['won']) / 5
                    result_df.loc[idx, 'away_scoring_trend_5'] = sum(g['score'] for g in recent_5) / 5
                else:
                    result_df.loc[idx, 'away_momentum_5'] = 0.5
                    result_df.loc[idx, 'away_scoring_trend_5'] = 0
                    
                if len(away_games) >= 7:
                    recent_7 = away_games[-7:]
                    result_df.loc[idx, 'away_momentum_7'] = sum(1 for g in recent_7 if g['won']) / 7
                    result_df.loc[idx, 'away_scoring_trend_7'] = sum(g['score'] for g in recent_7) / 7
                else:
                    result_df.loc[idx, 'away_momentum_7'] = 0.5
                    result_df.loc[idx, 'away_scoring_trend_7'] = 0
                
                # 모멘텀 차이 계산
                result_df.loc[idx, 'momentum_diff_3'] = result_df.loc[idx, 'home_momentum_3'] - result_df.loc[idx, 'away_momentum_3']
                result_df.loc[idx, 'momentum_diff_5'] = result_df.loc[idx, 'home_momentum_5'] - result_df.loc[idx, 'away_momentum_5']
                result_df.loc[idx, 'momentum_diff_7'] = result_df.loc[idx, 'home_momentum_7'] - result_df.loc[idx, 'away_momentum_7']
                
                result_df.loc[idx, 'scoring_trend_diff_3'] = result_df.loc[idx, 'home_scoring_trend_3'] - result_df.loc[idx, 'away_scoring_trend_3']
                result_df.loc[idx, 'scoring_trend_diff_5'] = result_df.loc[idx, 'home_scoring_trend_5'] - result_df.loc[idx, 'away_scoring_trend_5']
                result_df.loc[idx, 'scoring_trend_diff_7'] = result_df.loc[idx, 'home_scoring_trend_7'] - result_df.loc[idx, 'away_scoring_trend_7']
        else:
            # historical_data가 없는 경우 기본값 설정
            self.logger.warning("과거 데이터가 없어 모멘텀 지표를 기본값으로 설정합니다.")
            momentum_cols = [
                'home_momentum_3', 'home_momentum_5', 'home_momentum_7',
                'away_momentum_3', 'away_momentum_5', 'away_momentum_7',
                'home_scoring_trend_3', 'home_scoring_trend_5', 'home_scoring_trend_7',
                'away_scoring_trend_3', 'away_scoring_trend_5', 'away_scoring_trend_7',
                'momentum_diff_3', 'momentum_diff_5', 'momentum_diff_7',
                'scoring_trend_diff_3', 'scoring_trend_diff_5', 'scoring_trend_diff_7'
            ]
            for col in momentum_cols:
                if 'diff' in col:
                    result_df[col] = 0
                elif 'momentum' in col:
                    result_df[col] = 0.5
                else:
                    result_df[col] = 0
        
        # 3. 상대적 순위 지표 (Relative Rankings)
        self.logger.info("상대적 순위 지표 생성 중...")
        
        # 현재 시점의 모든 팀 승률로 순위 계산
        if hasattr(self, 'historical_data') and self.historical_data is not None:
            # 최신 데이터에서 팀별 승률 추출
            latest_data = self.historical_data.sort_values('date').groupby(['home_team_id', 'away_team_id']).last().reset_index()
            
            # 홈팀과 원정팀의 승률 수집
            all_win_rates = {}
            
            # 홈팀 승률 수집
            if 'home_overall_record_win_rate' in latest_data.columns:
                for _, row in latest_data.iterrows():
                    if not pd.isna(row['home_overall_record_win_rate']):
                        all_win_rates[row['home_team_id']] = row['home_overall_record_win_rate']
            
            # 원정팀 승률 수집
            if 'away_overall_record_win_rate' in latest_data.columns:
                for _, row in latest_data.iterrows():
                    if not pd.isna(row['away_overall_record_win_rate']):
                        all_win_rates[row['away_team_id']] = row['away_overall_record_win_rate']
            
            if len(all_win_rates) > 1:
                # 승률 기준 순위 계산 (높은 승률이 1위)
                sorted_teams = sorted(all_win_rates.items(), key=lambda x: x[1], reverse=True)
                team_rankings = {team_id: rank + 1 for rank, (team_id, _) in enumerate(sorted_teams)}
                
                # 예측 데이터에 순위 정보 추가
                for idx, row in result_df.iterrows():
                    home_rank = team_rankings.get(row['home_team_id'], len(team_rankings) // 2)
                    away_rank = team_rankings.get(row['away_team_id'], len(team_rankings) // 2)
                    
                    result_df.loc[idx, 'home_league_rank'] = home_rank
                    result_df.loc[idx, 'away_league_rank'] = away_rank
                    result_df.loc[idx, 'rank_difference'] = away_rank - home_rank  # 양수면 홈팀이 상위
                    
                    # 정규화된 순위 (0-1 스케일)
                    total_teams = len(team_rankings)
                    result_df.loc[idx, 'home_rank_normalized'] = 1 - (home_rank - 1) / (total_teams - 1)
                    result_df.loc[idx, 'away_rank_normalized'] = 1 - (away_rank - 1) / (total_teams - 1)
                    result_df.loc[idx, 'rank_advantage'] = result_df.loc[idx, 'home_rank_normalized'] - result_df.loc[idx, 'away_rank_normalized']
            else:
                # 데이터가 부족한 경우 기본값
                for idx, row in result_df.iterrows():
                    result_df.loc[idx, 'home_league_rank'] = 15
                    result_df.loc[idx, 'away_league_rank'] = 15
                    result_df.loc[idx, 'rank_difference'] = 0
                    result_df.loc[idx, 'home_rank_normalized'] = 0.5
                    result_df.loc[idx, 'away_rank_normalized'] = 0.5
                    result_df.loc[idx, 'rank_advantage'] = 0
        else:
            # historical_data가 없는 경우 기본값
            self.logger.warning("과거 데이터가 없어 순위 지표를 기본값으로 설정합니다.")
            for idx, row in result_df.iterrows():
                result_df.loc[idx, 'home_league_rank'] = 15
                result_df.loc[idx, 'away_league_rank'] = 15
                result_df.loc[idx, 'rank_difference'] = 0
                result_df.loc[idx, 'home_rank_normalized'] = 0.5
                result_df.loc[idx, 'away_rank_normalized'] = 0.5
                result_df.loc[idx, 'rank_advantage'] = 0
        
        # 4. 비선형 변환 (Non-linear Transformations)
        self.logger.info("비선형 변환 생성 중...")
        
        # 로그 변환 (양수 값에만 적용)
        log_transform_cols = [
            'home_avg_runs_for', 'away_avg_runs_for',
            'home_recent_avg_score', 'away_recent_avg_score'
        ]
        
        for col in log_transform_cols:
            if col in result_df.columns:
                # 0보다 큰 값에만 로그 적용
                positive_mask = result_df[col] > 0
                result_df.loc[positive_mask, f'{col}_log'] = np.log1p(result_df.loc[positive_mask, col])
                result_df.loc[~positive_mask, f'{col}_log'] = 0
        
        # 제곱근 변환
        sqrt_transform_cols = [
            'home_avg_batting_homeRuns', 'away_avg_batting_homeRuns',
            'home_recent_avg_batting_homeRuns', 'away_recent_avg_batting_homeRuns'
        ]
        
        for col in sqrt_transform_cols:
            if col in result_df.columns:
                # 음수가 아닌 값에만 제곱근 적용
                non_negative_mask = result_df[col] >= 0
                result_df.loc[non_negative_mask, f'{col}_sqrt'] = np.sqrt(result_df.loc[non_negative_mask, col])
                result_df.loc[~non_negative_mask, f'{col}_sqrt'] = 0
        
        # 지수 변환 (작은 값들에 적용)
        exp_transform_cols = [
            'diff_recent_win_rate', 'diff_overall_record_win_rate'
        ]
        
        for col in exp_transform_cols:
            if col in result_df.columns:
                # 절댓값이 작은 차이값들에 지수 변환 적용
                result_df[f'{col}_exp'] = np.sign(result_df[col]) * (np.exp(np.abs(result_df[col])) - 1)
        
        # 5. 복합 지표 (Composite Indicators)
        self.logger.info("복합 지표 생성 중...")
        
        # 종합 팀 파워 지수
        power_components = [
            'home_recent_win_rate', 'home_avg_batting_ops', 'home_rank_normalized'
        ]
        
        if all(col in result_df.columns for col in power_components):
            result_df['home_power_index'] = (
                result_df['home_recent_win_rate'] * 0.4 +
                result_df['home_avg_batting_ops'] * 0.3 +
                result_df['home_rank_normalized'] * 0.3
            )
            
            result_df['away_power_index'] = (
                result_df['away_recent_win_rate'] * 0.4 +
                result_df['away_avg_batting_ops'] * 0.3 +
                result_df['away_rank_normalized'] * 0.3
            )
            
            result_df['power_index_diff'] = result_df['home_power_index'] - result_df['away_power_index']
        
        # 경기 예측 신뢰도 지수
        confidence_components = [
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_vs_away_win_rate', 'away_vs_home_win_rate'
        ]
        
        if all(col in result_df.columns for col in confidence_components):
            # 팀 간 성적 차이가 클수록, 상대전적이 명확할수록 신뢰도 높음
            win_rate_diff = np.abs(result_df['home_recent_win_rate'] - result_df['away_recent_win_rate'])
            h2h_clarity = np.abs(result_df['home_vs_away_win_rate'] - 0.5) * 2  # 0.5에서 멀수록 명확
            
            result_df['prediction_confidence'] = (win_rate_diff + h2h_clarity) / 2
        
        self.logger.info("고급 특성 생성 완료")
        return result_df


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