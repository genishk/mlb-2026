import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from collections import defaultdict
import numpy as np

class MLBTrainingDataProcessor:
    """
    MLB 매치 레코드를 머신러닝 훈련 데이터로 변환하는 프로세서
    
    이 클래스는 평탄화된 MLB 매치 레코드를 가져와서
    시간적 데이터 리키지 없이 훈련 데이터셋으로 변환합니다.
    각 경기의 피처는 해당 경기 시점 이전의 데이터만 사용하여 계산됩니다.
    """
    
    def __init__(self, debug_mode=True):
        """프로세서 초기화"""
        # 프로젝트 디렉토리 설정
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.records_dir = self.project_root / 'data' / 'records'
        self.training_dir = self.project_root / 'data' / 'training'
        
        # 디렉토리가 없으면 생성
        self.training_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger("MLBTrainingDataProcessor")
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
        
        self.logger.info("MLB 훈련 데이터 프로세서 초기화 완료")
    
    def find_latest_records_file(self, file_prefix="mlb_historical_records_", file_suffix=".json"):
        """가장 최근에 생성된 레코드 파일 찾기"""
        files = list(self.records_dir.glob(f"{file_prefix}*{file_suffix}"))
        if not files:
            self.logger.error(f"레코드 파일을 찾을 수 없습니다: {file_prefix}*{file_suffix}")
            return None
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"최신 레코드 파일: {latest_file}")
        return latest_file
    
    def load_records(self, file_path=None):
        """MLB 레코드 데이터 로드하기"""
        if file_path is None:
            file_path = self.find_latest_records_file()
                
        if file_path is None:
            self.logger.error("레코드 파일을 찾을 수 없습니다. 파일이 data/records 디렉토리에 존재하는지 확인하세요.")
            return False
        
        try:
            self.logger.info(f"MLB 레코드 데이터 로드 중: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            self.logger.info(f"로드된 경기 수: {len(records)}")
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(records)
            
            # 날짜 열을 datetime 형식으로 변환
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                self.logger.error(f"날짜 변환 중 오류 발생: {str(e)}. 'date' 열이 올바른 형식인지 확인하세요.")
                return False
            
            # 날짜순으로 정렬 (시간 순서가 매우 중요)
            df = df.sort_values('date')
            
            # team_id를 문자열로 통일
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
            
            self.historical_data = df
            return True
            
        except FileNotFoundError:
            self.logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return False
        except json.JSONDecodeError:
            self.logger.error(f"잘못된 JSON 형식입니다: {file_path}")
            return False
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def collect_team_games(self):
        """팀별 경기 데이터 수집
        
        시간 순서대로 팀별 경기 결과와 통계를 수집합니다.
        데이터 리키지를 방지하기 위해 사용됩니다.
        
        Returns:
            Dict: 팀별 경기 결과 및 날짜 정보
        """
        self.logger.info("팀별 경기 데이터 수집 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
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
        
        self.logger.info(f"팀별 경기 데이터 수집 완료: {len(team_games)}개 팀")
        return result
    
    def add_recent_stats(self, n_games=10):
        """최근 N경기 통계 및 시즌 전체 통계 추가
        
        각 경기에 대해 양 팀의 최근 N경기 통계 및 시즌 전체 통계를 추가합니다.
        중요: 각 경기에 대해 해당 경기 시점 이전의 데이터만 사용합니다.
        *** 예측데이터와 완전히 동일한 로직 사용 ***
        
        Args:
            n_games: 최근 몇 경기를 고려할지 (기본값: 10)
            
        Returns:
            pd.DataFrame: 특성이 추가된 데이터프레임
        """
        self.logger.info(f"최근 {n_games}경기 통계 및 시즌 전체 통계 추가 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
        
        # 결과 데이터프레임 복사
        result_df = self.historical_data.copy()
        
        # 날짜 순으로 정렬 (시간 순서가 매우 중요)
        result_df = result_df.sort_values('date')
        
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
        
        # 팀별 모든 경기 기록 (동적으로 업데이트됨)
        team_games = defaultdict(list)
        
        # *** 예측데이터와 완전히 동일한 구조: 팀별 시즌 통계 (경기마다 누적 업데이트됨) ***
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
        
        # 각 경기를 시간 순서대로 처리 (중요: 통계 업데이트가 연대기적으로 이루어짐)
        for idx, row in result_df.iterrows():
            game_date = row['date']
            home_team_id = row['home_team_id']
            away_team_id = row['away_team_id']
            
            # 1. 현재 경기 시점까지의 팀 통계를 계산하여 피처 추가
            
            # *** 예측데이터와 완전히 동일한 로직: 홈팀 시즌 통계 추가 ***
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
                
                # *** 핵심: 누적값을 경기수로 나누어 평균 계산 (예측데이터와 동일) ***
                # 주요 타격 통계 평균
                for stat in batting_stat_fields:
                    if team_season_stats[home_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'home_avg_{stat}'] = team_season_stats[home_team_id][f'total_{stat}'] / home_total_games
                
                # 주요 투구 통계 평균
                for stat in pitching_stat_fields:
                    if team_season_stats[home_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'home_avg_{stat}'] = team_season_stats[home_team_id][f'total_{stat}'] / home_total_games
            else:
                # 팀 데이터가 없는 경우 기본값 (예측데이터와 동일)
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
            
            # *** 예측데이터와 완전히 동일한 로직: 원정팀 시즌 통계 추가 ***
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
                
                # *** 핵심: 누적값을 경기수로 나누어 평균 계산 (예측데이터와 동일) ***
                # 주요 타격 통계 평균
                for stat in batting_stat_fields:
                    if team_season_stats[away_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'away_avg_{stat}'] = team_season_stats[away_team_id][f'total_{stat}'] / away_total_games
                
                # 주요 투구 통계 평균
                for stat in pitching_stat_fields:
                    if team_season_stats[away_team_id][f'total_{stat}'] != 0:
                        result_df.loc[idx, f'away_avg_{stat}'] = team_season_stats[away_team_id][f'total_{stat}'] / away_total_games
            else:
                # 팀 데이터가 없는 경우 기본값 (예측데이터와 동일)
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
            
            # *** 예측데이터와 완전히 동일한 로직: 홈팀 최근 N경기 통계 계산 ***
            home_previous_games = team_games[home_team_id]
            
            if len(home_previous_games) > 0:
                # 최근 N경기만 선택
                recent_games = home_previous_games[-n_games:] if len(home_previous_games) >= n_games else home_previous_games
                
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
                
                # *** 예측데이터와 동일한 방식: 최근 N경기 타격 통계 평균 ***
                for stat in batting_stat_fields:
                    values = [g['batting_stats'].get(stat, 0) for g in recent_games if 'batting_stats' in g and stat in g['batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
                
                # *** 예측데이터와 동일한 방식: 최근 N경기 투구 통계 평균 ***
                for stat in pitching_stat_fields:
                    values = [g['pitching_stats'].get(stat, 0) for g in recent_games if 'pitching_stats' in g and stat in g['pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_recent_avg_{stat}'] = 0
            else:
                # 이전 경기가 없는 경우 기본값 (예측데이터와 동일)
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
            
            # *** 예측데이터와 완전히 동일한 로직: 원정팀 최근 N경기 통계 계산 ***
            away_previous_games = team_games[away_team_id]
            
            if len(away_previous_games) > 0:
                # 최근 N경기만 선택
                recent_games = away_previous_games[-n_games:] if len(away_previous_games) >= n_games else away_previous_games
                
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
                
                # *** 예측데이터와 동일한 방식: 최근 N경기 타격 통계 평균 ***
                for stat in batting_stat_fields:
                    values = [g['batting_stats'].get(stat, 0) for g in recent_games if 'batting_stats' in g and stat in g['batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
                
                # *** 예측데이터와 동일한 방식: 최근 N경기 투구 통계 평균 ***
                for stat in pitching_stat_fields:
                    values = [g['pitching_stats'].get(stat, 0) for g in recent_games if 'pitching_stats' in g and stat in g['pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_recent_avg_{stat}'] = 0
            else:
                # 이전 경기가 없는 경우 기본값 (예측데이터와 동일)
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
            
            # 2. 현재 경기 결과를 기반으로 팀 통계 업데이트 (다음 경기 피처 계산용)
            
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
                
                # 홈팀 경기 기록 업데이트
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
                
                # *** 예측데이터와 완전히 동일한 방식: 홈팀 시즌 통계 누적 업데이트 ***
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
                    # 문자열을 실수로 변환
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[home_team_id][f'total_{stat}'] += numeric_value
                
                # 홈팀 투구 통계 누적 업데이트
                for stat, value in home_pitching_stats.items():
                    # 문자열을 실수로 변환
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[home_team_id][f'total_{stat}'] += numeric_value
                
                # 원정팀 경기 기록 업데이트
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
                
                # *** 예측데이터와 완전히 동일한 방식: 원정팀 시즌 통계 누적 업데이트 ***
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
                    # 문자열을 실수로 변환
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[away_team_id][f'total_{stat}'] += numeric_value
                
                # 원정팀 투구 통계 누적 업데이트
                for stat, value in away_pitching_stats.items():
                    # 문자열을 실수로 변환
                    numeric_value = float(value) if isinstance(value, str) else value
                    team_season_stats[away_team_id][f'total_{stat}'] += numeric_value
        
        self.logger.info("팀 통계 추가 완료 (예측데이터와 동일한 로직 적용)")
        self.historical_data = result_df
        return result_df
    
    def add_rest_days(self):
        """각 팀의 이전 경기와의 휴식일 수 계산
        
        각 경기에 대해 양 팀의 이전 경기와의 일수 차이(휴식일)를 계산합니다.
        중요: 각 경기에 대해 해당 경기 시점 이전의 데이터만 사용합니다.
        
        Returns:
            pd.DataFrame: 휴식일 정보가 추가된 데이터프레임
        """
        self.logger.info("휴식일 수 정보 추가 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
        
        # 결과 데이터프레임 복사
        result_df = self.historical_data.copy()
        
        # 날짜 순으로 정렬 (시간 순서가 매우 중요)
        result_df = result_df.sort_values('date')
        
        # 팀별 가장 최근 경기 날짜를 저장할 딕셔너리
        team_last_game_dates = {}
        
        # 각 경기를 시간 순서대로 처리
        for idx, row in result_df.iterrows():
            current_date = row['date']
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            
            # 홈팀 휴식일 계산
            if home_id in team_last_game_dates:
                last_game_date = team_last_game_dates[home_id]
                
                # 날짜 차이 계산 (datetime 객체 처리)
                if isinstance(current_date, pd.Timestamp) and isinstance(last_game_date, pd.Timestamp):
                    rest_days = (current_date.date() - last_game_date.date()).days
                else:
                    # 문자열을 datetime으로 변환
                    current_dt = pd.to_datetime(current_date).date() if not isinstance(current_date, pd.Timestamp) else current_date.date()
                    last_dt = pd.to_datetime(last_game_date).date() if not isinstance(last_game_date, pd.Timestamp) else last_game_date.date()
                    rest_days = (current_dt - last_dt).days
                
                # 결과에 기록
                result_df.loc[idx, 'home_rest_days'] = max(0, rest_days)
            else:
                # 팀의 첫 경기인 경우 (시즌 시작 또는 데이터 부족)
                result_df.loc[idx, 'home_rest_days'] = 5  # 기본값 (시즌 첫 경기 가정)
            
            # 원정팀 휴식일 계산
            if away_id in team_last_game_dates:
                last_game_date = team_last_game_dates[away_id]
                
                # 날짜 차이 계산 (datetime 객체 처리)
                if isinstance(current_date, pd.Timestamp) and isinstance(last_game_date, pd.Timestamp):
                    rest_days = (current_date.date() - last_game_date.date()).days
                else:
                    # 문자열을 datetime으로 변환
                    current_dt = pd.to_datetime(current_date).date() if not isinstance(current_date, pd.Timestamp) else current_date.date()
                    last_dt = pd.to_datetime(last_game_date).date() if not isinstance(last_game_date, pd.Timestamp) else last_game_date.date()
                    rest_days = (current_dt - last_dt).days
                
                # 결과에 기록
                result_df.loc[idx, 'away_rest_days'] = max(0, rest_days)
            else:
                # 팀의 첫 경기인 경우 (시즌 시작 또는 데이터 부족)
                result_df.loc[idx, 'away_rest_days'] = 5  # 기본값 (시즌 첫 경기 가정)
            
            # 현재 경기 날짜를 각 팀의 최근 경기로 업데이트
            team_last_game_dates[home_id] = current_date
            team_last_game_dates[away_id] = current_date
        
        self.logger.info("휴식일 수 정보 추가 완료")
        self.historical_data = result_df
        return result_df
    
    def add_head_to_head_stats(self):
        """상대 전적 정보 추가
        
        각 경기에 대해 양 팀 간의 이전 상대 전적 정보를 추가합니다.
        중요: 각 경기에 대해 해당 경기 시점 이전의 데이터만 사용합니다.
        
        Returns:
            pd.DataFrame: 상대 전적 정보가 추가된 데이터프레임
        """
        self.logger.info("상대 전적 정보 추가 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
        
        # 결과 데이터프레임 복사
        result_df = self.historical_data.copy()
        
        # 날짜 순으로 정렬 (시간 순서가 매우 중요)
        result_df = result_df.sort_values('date')
        
        # 각 팀 간의 상대전적을 저장할 딕셔너리 (경기 진행에 따라 업데이트됨)
        h2h_records = {}  # {(team1_id, team2_id): [team1_wins, team2_wins, total_games, [recent_games]]}
        
        # 주요 타격 통계 필드 정의
        batting_stat_fields = [
            'batting_avg', 'batting_ops', 'batting_hits', 'batting_homeRuns'
        ]
        
        # 주요 투구 통계 필드 정의
        pitching_stat_fields = [
            'pitching_era', 'pitching_whip', 'pitching_strikeOuts'
        ]
        
        # 모든 경기를 시간 순서대로 처리
        for idx, row in result_df.iterrows():
            game_date = row['date']
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            
            # 팀 ID를 정렬하지 않고 홈팀, 원정팀 순서 그대로 사용 (방향성 유지)
            team_key = (home_id, away_id)
            
            # 1. 현재 경기 시점까지의 상대전적 통계 추출
            if team_key not in h2h_records:
                h2h_records[team_key] = {
                    'home_wins': 0,
                    'away_wins': 0,
                    'total_games': 0,
                    'recent_games': []
                }
            
            # 현재 시점까지의 상대 전적 추출
            home_wins = h2h_records[team_key]['home_wins']
            away_wins = h2h_records[team_key]['away_wins']
            total_games = h2h_records[team_key]['total_games']
            recent_games = h2h_records[team_key]['recent_games']
            
            # 상대 전적 승패 기록
            result_df.loc[idx, 'home_vs_away_wins'] = home_wins
            result_df.loc[idx, 'home_vs_away_losses'] = away_wins
            result_df.loc[idx, 'away_vs_home_wins'] = away_wins
            result_df.loc[idx, 'away_vs_home_losses'] = home_wins
            
            # 승률 계산 (0 나누기 방지)
            if total_games > 0:
                result_df.loc[idx, 'home_vs_away_win_rate'] = home_wins / total_games
                result_df.loc[idx, 'away_vs_home_win_rate'] = away_wins / total_games
            else:
                # 이전 상대전적이 없는 경우 0.5로 기본값 설정
                result_df.loc[idx, 'home_vs_away_win_rate'] = 0.5
                result_df.loc[idx, 'away_vs_home_win_rate'] = 0.5
            
            # 최근 상대 전적 기반 통계 (평균 득점, 실점 등)
            if recent_games:
                # 홈팀 상대전 평균 득점/실점
                result_df.loc[idx, 'home_vs_away_avg_score'] = sum(game['home_score'] for game in recent_games) / len(recent_games)
                result_df.loc[idx, 'home_vs_away_avg_allowed'] = sum(game['away_score'] for game in recent_games) / len(recent_games)
                
                # 원정팀 상대전 평균 득점/실점
                result_df.loc[idx, 'away_vs_home_avg_score'] = sum(game['away_score'] for game in recent_games) / len(recent_games)
                result_df.loc[idx, 'away_vs_home_avg_allowed'] = sum(game['home_score'] for game in recent_games) / len(recent_games)
                
                # 홈팀 주요 타격 통계 평균
                for stat in batting_stat_fields:
                    values = [game['home_batting_stats'].get(stat, 0) for game in recent_games if 'home_batting_stats' in game and stat in game['home_batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_vs_away_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                
                # 홈팀 주요 투구 통계 평균
                for stat in pitching_stat_fields:
                    values = [game['home_pitching_stats'].get(stat, 0) for game in recent_games if 'home_pitching_stats' in game and stat in game['home_pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'home_vs_away_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                
                # 원정팀 주요 타격 통계 평균
                for stat in batting_stat_fields:
                    values = [game['away_batting_stats'].get(stat, 0) for game in recent_games if 'away_batting_stats' in game and stat in game['away_batting_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_vs_home_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
                
                # 원정팀 주요 투구 통계 평균
                for stat in pitching_stat_fields:
                    values = [game['away_pitching_stats'].get(stat, 0) for game in recent_games if 'away_pitching_stats' in game and stat in game['away_pitching_stats']]
                    if values:
                        # 문자열을 실수로 변환
                        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
                        result_df.loc[idx, f'away_vs_home_avg_{stat}'] = sum(numeric_values) / len(numeric_values)
                    else:
                        result_df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
            else:
                # 이전 상대전적이 없는 경우 기본값 설정
                result_df.loc[idx, 'home_vs_away_avg_score'] = 0
                result_df.loc[idx, 'home_vs_away_avg_allowed'] = 0
                result_df.loc[idx, 'away_vs_home_avg_score'] = 0
                result_df.loc[idx, 'away_vs_home_avg_allowed'] = 0
                
                # 타격 통계 기본값
                for stat in batting_stat_fields:
                    result_df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                    result_df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
                
                # 투구 통계 기본값
                for stat in pitching_stat_fields:
                    result_df.loc[idx, f'home_vs_away_avg_{stat}'] = 0
                    result_df.loc[idx, f'away_vs_home_avg_{stat}'] = 0
            
            # 2. 현재 경기 결과를 이용해 상대전적 업데이트 (다음 경기 피처 계산용)
            # 완료된 경기만 추가 (점수가 있는 경기)
            if 'home_score' in row and 'away_score' in row and \
               not pd.isna(row['home_score']) and not pd.isna(row['away_score']):
                
                home_score = float(row['home_score'])
                away_score = float(row['away_score'])
                
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
                
                # 승자 결정 및 상대전적 업데이트
                if home_score > away_score:
                    h2h_records[team_key]['home_wins'] += 1
                else:
                    h2h_records[team_key]['away_wins'] += 1
                
                # 총 경기 수 증가
                h2h_records[team_key]['total_games'] += 1
                
                # 최근 경기 목록 업데이트 (최대 5개 유지)
                h2h_records[team_key]['recent_games'].append({
                    'date': game_date,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_batting_stats': home_batting_stats,
                    'home_pitching_stats': home_pitching_stats,
                    'away_batting_stats': away_batting_stats,
                    'away_pitching_stats': away_pitching_stats
                })
                
                # 최대 5개의 최근 경기만 유지
                if len(h2h_records[team_key]['recent_games']) > 5:
                    h2h_records[team_key]['recent_games'] = h2h_records[team_key]['recent_games'][-5:]
        
        self.logger.info("상대 전적 정보 추가 완료")
        self.historical_data = result_df
        return result_df
    
    def handle_missing_values(self):
        """결측치 처리
        
        데이터셋의 결측치를 적절한 값으로 대체합니다.
        
        - 승률/비율 관련 컬럼은 적정 기본값으로 대체 (예: 승률 0.5)
        - 팀별 통계는 해당 팀의 평균값으로 대체
        - 나머지 결측치는 평균값으로 대체
        - 그래도 남은 결측치는 0으로 대체
        
        Returns:
            pd.DataFrame: 결측치가 처리된 데이터프레임
        """
        self.logger.info("결측치 처리 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
        
        result_df = self.historical_data.copy()
        
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
                    
                    result_df[col] = result_df[col].fillna(default_value)
                
                # 2. 팀별 통계는 해당 팀의 평균으로 대체
                elif col.startswith(('home_', 'away_')) and not col.endswith(('_id', '_score')):
                    team_type = col.split('_')[0]  # 'home' or 'away'
                    team_col = f"{team_type}_team_id"
                    
                    # 팀별 평균 계산 (숫자형 데이터만)
                    if team_col in result_df.columns:
                        # 숫자형 데이터인 경우에만 평균 계산
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            team_means = result_df.groupby(team_col)[col].transform(lambda x: x.mean())
                            result_df[col] = result_df[col].fillna(team_means)
                        else:
                            # 문자열인 경우 가장 빈번한 값으로 대체
                            team_modes = result_df.groupby(team_col)[col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                            result_df[col] = result_df[col].fillna(team_modes)
                
                # 3. 남은 결측치는 컬럼 평균으로 대체 (숫자형 데이터만)
                if result_df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                
                # 4. 그래도 남은 결측치는 0으로 대체 (숫자형) 또는 'Unknown'으로 대체 (문자열)
                if result_df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        result_df[col] = result_df[col].fillna('Unknown')
        
        self.logger.info("결측치 처리 완료")
        self.historical_data = result_df
        return result_df
    
    def create_features(self):
        """모델 훈련용 특성 생성
        
        팀 간 비교 특성 및 파생 변수를 생성합니다.
        
        Returns:
            pd.DataFrame: 특성이 추가된 데이터프레임
        """
        self.logger.info("모델 훈련용 특성 생성 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다. load_records()를 먼저 호출하세요.")
            return None
            
        result_df = self.historical_data.copy()
        
        # 1. 팀 간 기본 비교 특성
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
            
            if home_col in result_df.columns and away_col in result_df.columns:
                # 팀 간 차이 계산
                result_df[f'diff_{metric}'] = result_df[home_col] - result_df[away_col]
                
                # 팀 간 비율 계산 (0 나누기 방지)
                if metric.endswith('_rate') or metric.startswith('avg_'):
                    # ERA, WHIP은 낮을수록 좋으므로 반대로 계산 (away / home)
                    if 'era' in metric or 'whip' in metric:
                        safe_home = result_df[home_col].replace(0, 0.001)  # 0 나누기 방지
                        result_df[f'ratio_{metric}'] = result_df[away_col] / safe_home
                    else:
                        safe_away = result_df[away_col].replace(0, 0.001)  # 0 나누기 방지
                        result_df[f'ratio_{metric}'] = result_df[home_col] / safe_away
        
        # 2. 추가 복합 피처
        
        # 공격력 지표 (최근 득점 + 시즌 득점 + 타격 지표)
        if 'home_recent_avg_score' in result_df.columns and 'home_avg_runs_for' in result_df.columns:
            # 홈팀 공격력
            result_df['home_offense_power'] = (
                result_df['home_recent_avg_score'] + 
                result_df['home_avg_runs_for'] + 
                result_df['home_avg_batting_ops'] * 10
            ) / 3
            
            # 원정팀 공격력
            result_df['away_offense_power'] = (
                result_df['away_recent_avg_score'] + 
                result_df['away_avg_runs_for'] + 
                result_df['away_avg_batting_ops'] * 10
            ) / 3
            
            # 공격력 차이
            result_df['diff_offense_power'] = result_df['home_offense_power'] - result_df['away_offense_power']
        
        # 수비력 지표 (최근 실점 + 시즌 실점 + 투수 지표) - 낮을수록 좋음
        if 'home_recent_avg_allowed' in result_df.columns and 'home_avg_runs_against' in result_df.columns:
            # 홈팀 수비력 (투수력)
            result_df['home_defense_power'] = (
                result_df['home_recent_avg_allowed'] + 
                result_df['home_avg_runs_against'] + 
                result_df['home_avg_pitching_era'] / 2
            ) / 3
            
            # 원정팀 수비력 (투수력)
            result_df['away_defense_power'] = (
                result_df['away_recent_avg_allowed'] + 
                result_df['away_avg_runs_against'] + 
                result_df['away_avg_pitching_era'] / 2
            ) / 3
            
            # 수비력은 낮을수록 좋으므로 차이를 계산할 때 부호 반전
            result_df['diff_defense_power'] = result_df['away_defense_power'] - result_df['home_defense_power']
        
        # 전체 팀 강도 지표 (공격력 - 수비력)
        if 'home_offense_power' in result_df.columns and 'home_defense_power' in result_df.columns:
            result_df['home_team_strength'] = result_df['home_offense_power'] - result_df['home_defense_power']
            result_df['away_team_strength'] = result_df['away_offense_power'] - result_df['away_defense_power']
            result_df['diff_team_strength'] = result_df['home_team_strength'] - result_df['away_team_strength']
        
        # 홈/원정 이점 지표
        if 'home_home_record_win_rate' in result_df.columns and 'away_road_record_win_rate' in result_df.columns:
            result_df['home_advantage'] = result_df['home_home_record_win_rate'] - result_df['home_road_record_win_rate']
            result_df['away_disadvantage'] = result_df['away_home_record_win_rate'] - result_df['away_road_record_win_rate']
            result_df['venue_factor'] = result_df['home_advantage'] + result_df['away_disadvantage']
        
        # 팀 휴식 이점
        if 'home_rest_days' in result_df.columns and 'away_rest_days' in result_df.columns:
            # 휴식일 차이 (양수면 홈팀 유리)
            result_df['rest_advantage'] = result_df['home_rest_days'] - result_df['away_rest_days']
            # 양팀 모두 충분한 휴식을 취했는지 (2일 이상)
            result_df['both_well_rested'] = ((result_df['home_rest_days'] >= 2) & (result_df['away_rest_days'] >= 2)).astype(int)
            # 양팀 모두 휴식이 부족한지 (0일)
            result_df['both_tired'] = ((result_df['home_rest_days'] == 0) & (result_df['away_rest_days'] == 0)).astype(int)
        
        # 3. 홈팀/원정팀 승리 예측을 위한 목표 변수 (완료된 경기만)
        if 'home_score' in result_df.columns and 'away_score' in result_df.columns:
            mask = (~result_df['home_score'].isna()) & (~result_df['away_score'].isna())
            
            # 승패 여부
            result_df.loc[mask, 'home_win'] = (result_df.loc[mask, 'home_score'] > result_df.loc[mask, 'away_score']).astype(int)
            
            # 점수차 계산
            result_df.loc[mask, 'score_diff'] = result_df.loc[mask, 'home_score'] - result_df.loc[mask, 'away_score']
            
            # 총 득점 계산
            result_df.loc[mask, 'total_score'] = result_df.loc[mask, 'home_score'] + result_df.loc[mask, 'away_score']
            
            # 추가 득점 관련 파생 변수
            # 고득점 경기 여부 (총점 10점 초과)
            result_df.loc[mask, 'high_scoring_game'] = (result_df.loc[mask, 'total_score'] > 10).astype(int)
            # 저득점 경기 여부 (총점 5점 이하)
            result_df.loc[mask, 'low_scoring_game'] = (result_df.loc[mask, 'total_score'] <= 5).astype(int)
            # 접전 여부 (1점차)
            result_df.loc[mask, 'close_game'] = (result_df.loc[mask, 'score_diff'].abs() <= 1).astype(int)
        
        # 4. 날짜 기반 특성
        if 'date' in result_df.columns:
            # 요일 (0=월요일, 6=일요일)
            result_df['day_of_week'] = pd.to_datetime(result_df['date']).dt.dayofweek
            # 주말 여부 (토/일)
            result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
            # 월 (1=1월, 12=12월)
            result_df['month'] = pd.to_datetime(result_df['date']).dt.month
            # 시즌 초반 (3월-5월)
            result_df['early_season'] = result_df['month'].isin([3, 4, 5]).astype(int)
            # 시즌 중반 (6월-7월)
            result_df['mid_season'] = result_df['month'].isin([6, 7]).astype(int)
            # 시즌 후반 (8월-10월)
            result_df['late_season'] = result_df['month'].isin([8, 9, 10]).astype(int)
        
        self.logger.info("특성 생성 완료")
        self.historical_data = result_df
        return result_df
    
    def add_advanced_features(self):
        """고급 특성 생성 (기존 특성에 추가)
        
        특성 상호작용, 모멘텀 지표, 상대적 순위, 비선형 변환 등의 
        고급 특성을 생성합니다.
        
        Returns:
            pd.DataFrame: 고급 특성이 추가된 데이터프레임
        """
        self.logger.info("고급 특성 생성 중...")
        
        # 데이터가 로드되지 않았으면 오류
        if self.historical_data is None:
            self.logger.error("데이터가 로드되지 않았습니다.")
            return None
            
        result_df = self.historical_data.copy()
        
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
        
        # 날짜 순으로 정렬하여 모멘텀 계산
        sorted_df = result_df.sort_values('date')
        
        for idx, row in sorted_df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            game_date = row['date']
            
            # 현재 시점까지의 모멘텀 계산
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
            
            # 현재 경기 결과를 팀 기록에 추가 (다음 경기 계산용)
            if 'home_score' in row and 'away_score' in row and \
               not pd.isna(row['home_score']) and not pd.isna(row['away_score']):
                
                home_score = float(row['home_score'])
                away_score = float(row['away_score'])
                home_won = home_score > away_score
                
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
        
        # 3. 상대적 순위 지표 (Relative Rankings)
        self.logger.info("상대적 순위 지표 생성 중...")
        
        # 각 날짜별로 팀들의 승률 기준 순위 계산
        for idx, row in result_df.iterrows():
            current_date = row['date']
            
            # 현재 날짜까지의 모든 팀 승률 계산
            date_mask = result_df['date'] <= current_date
            current_data = result_df[date_mask]
            
            if len(current_data) > 0:
                # 홈팀들의 승률 계산
                home_win_rates = current_data.groupby('home_team_id')['home_overall_record_win_rate'].last()
                away_win_rates = current_data.groupby('away_team_id')['away_overall_record_win_rate'].last()
                
                # 모든 팀의 승률 통합
                all_win_rates = {}
                for team_id, win_rate in home_win_rates.items():
                    if not pd.isna(win_rate):
                        all_win_rates[team_id] = win_rate
                for team_id, win_rate in away_win_rates.items():
                    if not pd.isna(win_rate):
                        all_win_rates[team_id] = win_rate
                
                if len(all_win_rates) > 1:
                    # 승률 기준 순위 계산 (높은 승률이 1위)
                    sorted_teams = sorted(all_win_rates.items(), key=lambda x: x[1], reverse=True)
                    team_rankings = {team_id: rank + 1 for rank, (team_id, _) in enumerate(sorted_teams)}
                    
                    # 홈팀과 원정팀의 순위
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
        self.historical_data = result_df
        return result_df
    
    def save_training_data(self, remove_match_results=True, keep_target_vars=True, recent_days=40):
        """훈련 데이터를 JSON으로 저장
        
        Args:
            remove_match_results: 해당 매치의 결과 피처를 제거할지 여부 (기본값: True)
            keep_target_vars: 타겟 변수는 유지할지 여부 (기본값: True)
            recent_days: 저장할 최근 N일치 데이터 (기본값: 40)
        """
        self.logger.info("훈련 데이터 저장 중...")
        
        if self.historical_data is None:
            self.logger.error("저장할 훈련 데이터가 없습니다.")
            return None
        
        # 데이터프레임 복사
        df_copy = self.historical_data.copy()
        
        # 중복된 컬럼 확인
        duplicated_columns = df_copy.columns[df_copy.columns.duplicated()].tolist()
        if duplicated_columns:
            self.logger.warning(f"중복된 컬럼 발견: {duplicated_columns}")
            # 중복된 컬럼을 제거 (첫 번째 발견된 컬럼만 유지)
            df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
            self.logger.info(f"중복된 컬럼 제거 후 컬럼 수: {len(df_copy.columns)}")
        
        # 최근 N일치 데이터만 필터링 (저장 직전에 수행)
        if recent_days > 0 and 'date' in df_copy.columns:
            # 'date' 열이 문자열인 경우 datetime으로 변환
            if df_copy['date'].dtype == 'object':
                df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # 기준 날짜 설정 (현재 날짜)
            reference_date = datetime.now()
            # 필터링 날짜 계산 (기준 날짜로부터 recent_days일 전)
            filter_date = reference_date - pd.Timedelta(days=recent_days)
            
            # 필터링 전 레코드 수
            total_records = len(df_copy)
            
            # 최근 N일치 데이터만 필터링
            df_copy = df_copy[df_copy['date'] >= filter_date]
            
            # 필터링 후 레코드 수
            filtered_records = len(df_copy)
            
            self.logger.info(f"날짜 필터링: 최근 {recent_days}일치 데이터만 유지 ({filter_date.strftime('%Y-%m-%d')} 이후)")
            self.logger.info(f"필터링 전: {total_records}개 레코드, 필터링 후: {filtered_records}개 레코드")
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlb_training_data_{timestamp}.json"
        output_path = self.training_dir / filename
        
        # 1. 기본 경기 정보 필드
        basic_info_fields = [
            'game_id', 'date', 'start_time', 'season', 'venue', 'venue_location',
            'game_type', 'game_state', 
            'home_team_id', 'home_team_name', 'home_team_abbrev', 
            'away_team_id', 'away_team_name', 'away_team_abbrev',
            'home_score', 'away_score', 'both_tired'
        ]
        
        # 2. 경기 결과 정보
        result_fields = [
            'home_win', 'score_diff', 'total_score', 
            'home_runs', 'away_runs', 'high_scoring_game', 'low_scoring_game', 'close_game'
        ]
        
        # 3. 날짜/경기장 관련 파생 변수
        time_fields = [
            'day_of_week', 'is_weekend', 'month', 'early_season', 'mid_season', 'late_season'
        ]
        
        # 제외할 실제 경기 스탯 필드
        excluded_stat_fields = [
            'home_sog', 'home_batting_hits', 'home_batting_baseOnBalls', 'home_batting_strikeOuts',
            'home_pitching_hits', 'home_pitching_runs', 'home_pitching_earnedRuns',
            'home_pitching_baseOnBalls', 'home_pitching_strikeOuts', 'home_pitching_homeRuns',
            'away_sog', 'away_batting_hits', 'away_batting_baseOnBalls', 'away_batting_strikeOuts',
            'away_pitching_hits', 'away_pitching_runs', 'away_pitching_earnedRuns',
            'away_pitching_baseOnBalls', 'away_pitching_strikeOuts', 'away_pitching_homeRuns',
            # 날짜 및 위치 관련 필드는 이미 time_fields에 포함됨
            # 비율 관련 필드 제외
            'ratio_recent_win_rate', 'ratio_overall_record_win_rate', 'ratio_home_record_win_rate', 
            'ratio_road_record_win_rate', 'ratio_avg_runs_for', 'ratio_avg_runs_against',
            # 팀 강약 관련 필드 제외
            'home_offense_power', 'home_defense_power', 'home_team_strength',
            'away_offense_power', 'away_defense_power', 'away_team_strength',
            'diff_avg_runs_for', 'diff_avg_runs_against', 'diff_offense_power', 'diff_defense_power', 'diff_team_strength'
        ]
        
        # 4. 홈팀 통계 필드 (접두어: home_)
        home_fields = [col for col in df_copy.columns if col.startswith('home_') 
                       and col not in basic_info_fields and col not in result_fields
                       and col not in excluded_stat_fields]
        
        # 5. 원정팀 통계 필드 (접두어: away_)
        away_fields = [col for col in df_copy.columns if col.startswith('away_') 
                       and col not in basic_info_fields and col not in result_fields
                       and col not in excluded_stat_fields]
        
        # 6. 상대전적 필드 (접두어: h2h_, *_vs_*)
        h2h_fields = [col for col in df_copy.columns if 'h2h_' in col or '_vs_' in col]
        
        # 7. 휴식 및 차이 필드
        rest_fields = [col for col in df_copy.columns if 'rest' in col and col != 'both_tired']
        diff_fields = [col for col in df_copy.columns if 'diff_' in col]
        ratio_fields = [col for col in df_copy.columns if 'ratio_' in col]
        other_comparison_fields = ['home_advantage', 'away_disadvantage', 'venue_factor']
        
        # 모든 필드 목록 구성 - 일관된 순서 유지를 위해 예측 데이터와 동일한 순서 사용
        all_fields = basic_info_fields + result_fields + time_fields + home_fields + away_fields + h2h_fields + rest_fields + diff_fields + ratio_fields + other_comparison_fields
        
        # 결과 필드 처리
        if remove_match_results:
            # 타겟 변수는 선택적으로 유지
            if keep_target_vars:
                result_fields_to_remove = []  # home_score와 away_score는 이미 basic_info_fields에 있으므로 여기서 제거하지 않음
                
                # 특정 필드 제거
                for field in result_fields:
                    if field not in ['home_win', 'score_diff', 'total_score'] and field in df_copy.columns:
                        result_fields_to_remove.append(field)
                
                # 제거할 필드들을 all_fields에서 제외
                all_fields = [col for col in all_fields if col not in result_fields_to_remove]
            else:
                # 모든 결과 필드 제거
                all_fields = [col for col in all_fields if col not in result_fields]
        
        # 실제로 존재하는 필드만 필터링 (중복 제거) 및 제외된 실제 경기 스탯 필드 제거
        ordered_fields = []
        for col in all_fields:
            if col in df_copy.columns and col not in ordered_fields and col not in excluded_stat_fields:
                ordered_fields.append(col)
        
        # 누락된 필드 추가 (순서는 상관없이 마지막에 추가, excluded_stat_fields에 없는 필드만)
        for col in df_copy.columns:
            if col not in ordered_fields and col not in excluded_stat_fields:
                ordered_fields.append(col)
        
        # 열 순서 재정렬
        df_copy = df_copy[ordered_fields]
        
        # 날짜/시간 열 변환 (JSON 직렬화 오류 방지)
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
        
        # JSON으로 변환/저장
        df_copy.to_json(output_path, orient='records', indent=2)
        
        self.logger.info(f"데이터 저장 완료: {output_path}")
        return output_path
    
    def process_and_save(self, remove_match_results=True, keep_target_vars=True, sort_by_date=True, reverse_order=True, recent_days=40):
        """데이터 처리 및 저장 파이프라인
        
        Args:
            remove_match_results: 해당 매치의 결과 피처를 제거할지 여부 (기본값: True)
            keep_target_vars: 타겟 변수는 유지할지 여부 (기본값: True)
            sort_by_date: 날짜순으로 정렬할지 여부 (기본값: True)
            reverse_order: 날짜 역순으로 정렬할지 여부 (최근 데이터가 위로, 기본값: True)
            recent_days: 저장할 최근 N일치 데이터 (기본값: 40)
        
        Returns:
            Path: 저장된 파일 경로
        """
        # 1. 데이터 로드
        if not self.load_records():
            self.logger.error("데이터 로드 실패")
            return None
        
        # 2. 최근 통계 추가
        self.add_recent_stats()
        
        # 3. 휴식일 수 추가
        self.add_rest_days()
        
        # 4. 상대 전적 정보 추가
        self.add_head_to_head_stats()
        
        # 5. 결측치 처리
        self.handle_missing_values()
        
        # 6. 특성 생성
        self.create_features()
        
        # 7. 고급 특성 추가
        self.add_advanced_features()
        
        # 8. 날짜순 정렬 (선택적)
        if sort_by_date and 'date' in self.historical_data.columns:
            self.logger.info("데이터를 날짜순으로 정렬합니다.")
            if reverse_order:
                self.logger.info("역순 정렬 적용: 최근 데이터가 위로 오도록")
                self.historical_data = self.historical_data.sort_values('date', ascending=False)
            else:
                self.historical_data = self.historical_data.sort_values('date')
        
        # 9. 데이터 샘플 확인
        self.logger.info("\n=== 처리된 훈련 데이터 샘플 ===")
        if self.historical_data is not None and not self.historical_data.empty:
            sample_record = self.historical_data.iloc[0].to_dict()
            
            # 기본 필드들 표시
            basic_fields = ['game_id', 'date', 'home_team_name', 'away_team_name', 
                          'home_score', 'away_score', 'home_win']
            
            self.logger.info("\n기본 정보:")
            for field in basic_fields:
                if field in sample_record:
                    self.logger.info(f"  {field}: {sample_record[field]}")
            
            # 생성된 특성 필드 표시
            feature_fields = [key for key in sample_record.keys() if any(prefix in key for prefix in 
                                                                 ['recent_', 'diff_', 'vs_', '_rest_days'])]
            
            if feature_fields:
                self.logger.info("\n생성된 특성:")
                for field in sorted(feature_fields)[:10]:  # 특성 필드 중 일부만 표시
                    self.logger.info(f"  {field}: {sample_record[field]}")
            
            # 총 필드 수 표시
            self.logger.info(f"\n총 필드 수: {len(sample_record)}")
        
        # 10. 데이터 저장
        return self.save_training_data(
            remove_match_results=remove_match_results, 
            keep_target_vars=keep_target_vars,
            recent_days=recent_days
        )
            
# 스크립트로 실행 시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = MLBTrainingDataProcessor(debug_mode=True)
    
    # 기본 설정: 매치 결과 제거, 타겟 변수 유지, 날짜 역순 정렬(최근 데이터가 위로)
    output_path = processor.process_and_save(
        remove_match_results=True,  # 경기 결과 정보 제거
        keep_target_vars=True,      # 타겟 변수(home_win, score_diff, total_score)는 유지
        sort_by_date=True,          # 날짜순 정렬 적용 (일관성 유지)
        reverse_order=True,         # 역순 정렬 적용 (최근 데이터가 위로 오도록)
        recent_days=50              # 최근 90일치 데이터만 저장
    )
    
    print(f"\nMLB 훈련 데이터 처리 결과: {output_path}")