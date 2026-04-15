import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any
import logging

class SimpleModelAnalyzer:
    """
    간단한 모델 성과 분석기
    
    기능:
    1. 예측 파일과 히스토리컬 레코드 매칭
    2. 날짜별 필터링
    3. 25개 모델의 ROI 계산
    """
    
    def __init__(self, data_prefix=None):
        """
        초기화
        Args:
            data_prefix: 데이터 구분자 (None이면 구분자 없는 파일, 그 외는 해당 구분자 파일들)
        """
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "odds" / "data" / "matched"
        self.records_dir = self.project_root / "data" / "records"
        self.data_prefix = data_prefix if data_prefix is not None else "None"
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_available_dates(self) -> List[str]:
        """사용 가능한 날짜 목록 반환 (active 태그 파일만)"""
        if self.data_prefix == "None":
            pattern = "mlb_predictions_with_odds_*_active.json"
        else:
            pattern = f"{self.data_prefix}mlb_predictions_with_odds_*_active.json"
            
        prediction_files = list(self.predictions_dir.glob(pattern))
        dates = []
        
        for file_path in prediction_files:
            filename = file_path.stem
            parts = filename.split('_')
            
            if self.data_prefix == "None":
                if len(parts) >= 5:
                    date_part = parts[4]  # YYYYMMDD 형식
                    try:
                        # YYYY-MM-DD로 변환
                        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                        dates.append(date_str)
                    except (ValueError, IndexError):
                        continue
            else:
                # 구분자가 있는 경우: 55_105_mlb_predictions_with_odds_20250722_091712
                if len(parts) >= 7:
                    date_part = parts[6]  # YYYYMMDD 형식
                    try:
                        # YYYY-MM-DD로 변환
                        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                        dates.append(date_str)
                    except (ValueError, IndexError):
                        continue
        
        return sorted(list(set(dates)))
    
    def load_data(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """데이터 로드 및 날짜 필터링"""
        self.logger.info("📂 데이터 로딩 시작...")
        
        # 예측 파일들 로드 (active 태그 파일만)
        if self.data_prefix == "None":
            pattern = "mlb_predictions_with_odds_*_active.json"
        else:
            pattern = f"{self.data_prefix}mlb_predictions_with_odds_*_active.json"
        prediction_files = list(self.predictions_dir.glob(pattern))
        
        # 레코드 파일 로드
        record_files = list(self.records_dir.glob("mlb_historical_records_*.json"))
        
        if not record_files:
            raise ValueError("히스토리컬 레코드 파일을 찾을 수 없습니다")
        
        latest_record_file = max(record_files, key=lambda x: x.stat().st_mtime)
        
        # 🆕 히스토리컬 레코드 파일 정보 로깅
        self.logger.info(f"📋 사용할 히스토리컬 레코드 파일: {latest_record_file.name}")
        self.logger.info(f"   파일 크기: {latest_record_file.stat().st_size / 1024:.1f} KB")
        
        with open(latest_record_file, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)
        
        self.logger.info(f"   레코드 수: {len(historical_data)}개")
        
        # 예측 데이터 로드 및 날짜 필터링
        filtered_predictions = {}
        used_prediction_files = []
        skipped_files = []
        
        for file_path in prediction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 파일명에서 날짜 추출
                filename = file_path.stem
                parts = filename.split('_')
                
                if self.data_prefix == "None":
                    if len(parts) >= 5:
                        date_part = parts[4]  # YYYYMMDD 형식
                    else:
                        continue
                else:
                    # 구분자가 있는 경우: 55_105_mlb_predictions_with_odds_20250722_091712
                    if len(parts) >= 7:
                        date_part = parts[6]  # YYYYMMDD 형식
                    else:
                        continue
                
                try:
                    file_date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    
                    # 날짜 범위 필터링
                    should_include = False
                    if start_date and end_date:
                        if start_date <= file_date_str <= end_date:
                            should_include = True
                    else:
                        should_include = True
                    
                    if should_include:
                        filtered_predictions[date_part] = data
                        used_prediction_files.append({
                            'filename': file_path.name,
                            'date': file_date_str,
                            'games': len(data),
                            'size_kb': file_path.stat().st_size / 1024
                        })
                    else:
                        skipped_files.append({
                            'filename': file_path.name,
                            'date': file_date_str,
                            'reason': f'날짜 범위 외부 (필터: {start_date} ~ {end_date})'
                        })
                        
                except (ValueError, IndexError):
                        skipped_files.append({
                            'filename': file_path.name,
                            'date': 'unknown',
                            'reason': '날짜 파싱 실패'
                        })
                        continue
                        
            except Exception as e:
                skipped_files.append({
                    'filename': file_path.name,
                    'date': 'unknown',
                    'reason': f'파일 로드 실패: {e}'
                })
                self.logger.warning(f"파일 로드 실패: {file_path.name} - {e}")
                continue
        
        # 🆕 상세한 파일 사용 현황 로깅
        self.logger.info(f"\n📁 예측 파일 사용 현황:")
        self.logger.info(f"   전체 예측 파일: {len(prediction_files)}개")
        self.logger.info(f"   사용된 파일: {len(used_prediction_files)}개")
        self.logger.info(f"   제외된 파일: {len(skipped_files)}개")
        
        if start_date and end_date:
            self.logger.info(f"   날짜 필터: {start_date} ~ {end_date}")
        else:
            self.logger.info(f"   날짜 필터: 모든 날짜")
        
        if used_prediction_files:
            self.logger.info(f"\n✅ 사용된 예측 파일들:")
            total_games = 0
            for file_info in sorted(used_prediction_files, key=lambda x: x['date']):
                self.logger.info(f"   • {file_info['filename']} - {file_info['date']} - {file_info['games']}경기 - {file_info['size_kb']:.1f}KB")
                total_games += file_info['games']
            self.logger.info(f"   총 예측 게임: {total_games}개")
        
        if skipped_files:
            self.logger.info(f"\n❌ 제외된 파일들:")
            for file_info in skipped_files[:5]:  # 최대 5개만 표시
                self.logger.info(f"   • {file_info['filename']} - {file_info['date']} - {file_info['reason']}")
            if len(skipped_files) > 5:
                self.logger.info(f"   • ... 및 {len(skipped_files) - 5}개 추가 파일")
        
        self.logger.info(f"\n✅ 로드 완료: 예측 파일 {len(filtered_predictions)}개, 히스토리컬 레코드 {len(historical_data)}개")
        
        return {
            'predictions': filtered_predictions,
            'historical_records': historical_data,
            'file_info': {
                'historical_record_file': latest_record_file.name,
                'used_prediction_files': used_prediction_files,
                'skipped_files': skipped_files,
                'total_prediction_files': len(prediction_files)
            }
        }
    
    def match_predictions_with_results(self, predictions: Dict, historical_data: List) -> List[Dict]:
        """예측과 실제 결과 매칭"""
        self.logger.info("🔗 예측 결과 매칭 중...")
        
        # 히스토리컬 데이터 인덱싱
        historical_index = {}
        for record in historical_data:
            if all(key in record for key in ['date', 'home_team_name', 'away_team_name']):
                key = f"{record['date']}_{record['home_team_name']}_{record['away_team_name']}"
                historical_index[key] = record
        
        self.logger.info(f"📋 히스토리컬 데이터: {len(historical_index)}개 경기 인덱싱됨")
        
        matched_results = []
        total_predictions = 0
        matched_count = 0
        unmatched_count = 0
        no_home_win_count = 0
        
        # 🆕 날짜별 매칭 통계
        date_stats = {}
        
        for date_str, pred_data in predictions.items():
            date_matched = 0
            date_total = len(pred_data)
            total_predictions += date_total
            
            for prediction in pred_data:
                key = f"{prediction['date']}_{prediction['home_team']}_{prediction['away_team']}"
                
                if key in historical_index:
                    historical_record = historical_index[key]
                    
                    if 'home_win' in historical_record:
                        # 기본 정보
                        matched_result = {
                            'date': prediction['date'],
                            'home_team': prediction['home_team'],
                            'away_team': prediction['away_team'],
                            'actual_home_win': historical_record['home_win'],
                            'home_odds': prediction.get('home_team_odds'),
                            'away_odds': prediction.get('away_team_odds'),
                        }
                        
                        # 제외할 필드들 (모델이 아닌 것들)
                        excluded_probability_fields = {
                            'win_probability',  # 최종 앙상블 확률
                            'home_win_probability',  # 개별 모델의 홈팀 승리 확률
                            'away_win_probability',  # 원정팀 승리 확률
                            'predicted_winner_probability',  # 예측 승자 확률
                            'predicted_winner_probability_odds'  # 배당 기반 확률
                        }
                        
                        # 모든 모델 확률 추가 (제외 필드 제외)
                        for key, value in prediction.items():
                            if (key.endswith('_probability') and 
                                isinstance(value, (int, float)) and 
                                value > 0 and 
                                key not in excluded_probability_fields):
                                matched_result[key] = value
                        
                        matched_results.append(matched_result)
                        matched_count += 1
                        date_matched += 1
                    else:
                        no_home_win_count += 1
                else:
                    unmatched_count += 1
            
            # 날짜별 통계 저장
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            date_stats[date_formatted] = {
                'total': date_total,
                'matched': date_matched,
                'match_rate': (date_matched / date_total * 100) if date_total > 0 else 0
            }
        
        # 🆕 상세한 매칭 결과 로깅
        self.logger.info(f"\n📊 매칭 결과 상세:")
        self.logger.info(f"   총 예측 게임: {total_predictions}개")
        self.logger.info(f"   성공적 매칭: {matched_count}개")
        self.logger.info(f"   매칭 실패: {unmatched_count}개 (히스토리컬 데이터에 없음)")
        self.logger.info(f"   home_win 없음: {no_home_win_count}개")
        self.logger.info(f"   매칭률: {(matched_count / total_predictions * 100):.1f}%")
        
        self.logger.info(f"\n📅 날짜별 매칭 현황:")
        for date, stats in sorted(date_stats.items()):
            self.logger.info(f"   • {date}: {stats['matched']}/{stats['total']} ({stats['match_rate']:.1f}%)")
        
        self.logger.info(f"\n✅ 매칭 완료: {len(matched_results)}개 경기")
        return matched_results
    
    def calculate_expected_roi(self, predictions: Dict) -> Dict[str, Dict]:
        """Expected ROI 계산 (확률과 배당 기반)"""
        self.logger.info("🎯 모델별 Expected ROI 계산 중...")
        
        # 모든 모델 찾기
        all_models = set()
        for date_str, pred_data in predictions.items():
            for prediction in pred_data:
                for key in prediction.keys():
                    if key.endswith('_probability') and prediction.get(key, 0) > 0:
                        # 제외할 필드들 확인
                        excluded_probability_fields = {
                            'win_probability',
                            'home_win_probability', 
                            'away_win_probability',
                            'predicted_winner_probability',
                            'predicted_winner_probability_odds'
                        }
                        if key not in excluded_probability_fields:
                            model_name = key.replace('_probability', '')
                            all_models.add(model_name)
        
        model_results = {}
        
        for model in sorted(all_models):
            prob_key = f'{model}_probability'
            
            total_expected_return = 0.0
            total_invested = 0.0
            total_predictions = 0
            
            for date_str, pred_data in predictions.items():
                for prediction in pred_data:
                    prob = prediction.get(prob_key)
                    home_odds = prediction.get('home_team_odds')
                    away_odds = prediction.get('away_team_odds')
                    
                    if prob is not None and prob > 0 and home_odds is not None and away_odds is not None:
                        total_predictions += 1
                        bet_amount = 10  # 기본 베팅 금액
                        
                        # 홈팀 승률 기준 예측
                        predicted_home_win = 1 if prob > 0.5 else 0
                        
                        if predicted_home_win:
                            # 홈팀에 베팅
                            if home_odds > 0:
                                win_profit = bet_amount * (home_odds / 100)
                            else:
                                win_profit = bet_amount * (100 / abs(home_odds))
                            
                            # Expected return = 홈팀 승률 × 승리시 수익 + 홈팀 패배율 × 손실
                            expected_return = prob * win_profit + (1 - prob) * (-bet_amount)
                        else:
                            # 원정팀에 베팅
                            if away_odds > 0:
                                win_profit = bet_amount * (away_odds / 100)
                            else:
                                win_profit = bet_amount * (100 / abs(away_odds))
                            
                            # Expected return = 원정팀 승률 × 승리시 수익 + 원정팀 패배율 × 손실
                            away_prob = 1 - prob  # 원정팀 승률
                            expected_return = away_prob * win_profit + prob * (-bet_amount)
                        
                        total_expected_return += expected_return
                        total_invested += bet_amount
            
            if total_predictions > 0:
                expected_roi = (total_expected_return / total_invested * 100) if total_invested > 0 else 0.0
                
                model_results[model] = {
                    'model_name': model,
                    'expected_roi': expected_roi,
                    'total_bets': total_predictions,
                    'expected_profit_loss': total_expected_return,
                    'total_invested': total_invested
                }
        
        self.logger.info(f"✅ Expected ROI 계산 완료: {len(model_results)}개 모델")
        return model_results
    
    def calculate_daily_performance(self, matched_results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
        """모델별 일일 성과 계산"""
        self.logger.info("📅 모델별 일일 성과 계산 중...")
        
        # 모든 모델 찾기
        all_models = set()
        for result in matched_results:
            for key in result.keys():
                if key.endswith('_probability') and result[key] > 0:
                    model_name = key.replace('_probability', '')
                    all_models.add(model_name)
        
        daily_performances = {}
        
        for model in sorted(all_models):
            prob_key = f'{model}_probability'
            model_data = [r for r in matched_results if r.get(prob_key, 0) > 0]
            
            if len(model_data) == 0:
                continue
            
            # 날짜별로 그룹화
            daily_data = {}
            for result in model_data:
                date = result['date']
                if date not in daily_data:
                    daily_data[date] = []
                daily_data[date].append(result)
            
            # 각 날짜별 성과 계산
            model_daily_performance = {}
            for date, day_results in daily_data.items():
                total_return = 0.0
                total_invested = 0.0
                correct_predictions = 0
                total_predictions = 0
                games_with_odds = 0
                
                daily_games = []
                
                for result in day_results:
                    prob = result[prob_key]
                    actual_home_win = result['actual_home_win']
                    predicted_home_win = 1 if prob > 0.5 else 0
                    is_correct = predicted_home_win == actual_home_win
                    
                    home_odds = result.get('home_odds')
                    away_odds = result.get('away_odds')
                    
                    game_info = {
                        'away_team': result['away_team'],
                        'home_team': result['home_team'],
                        'predicted_home_win': predicted_home_win,
                        'actual_home_win': actual_home_win,
                        'home_probability': prob,
                        'is_correct': is_correct,
                        'has_odds': home_odds is not None and away_odds is not None
                    }
                    
                    if home_odds is not None and away_odds is not None:
                        games_with_odds += 1
                        total_predictions += 1
                        
                        if is_correct:
                            correct_predictions += 1
                        
                        predicted_odds = home_odds if predicted_home_win else away_odds
                        bet_amount = 10
                        
                        if is_correct and predicted_odds:
                            if predicted_odds > 0:
                                profit = bet_amount * (predicted_odds / 100)
                            else:
                                profit = bet_amount * (100 / abs(predicted_odds))
                            total_return += profit
                        else:
                            total_return -= bet_amount
                        
                        total_invested += bet_amount
                        
                        game_info.update({
                            'home_odds': home_odds,
                            'away_odds': away_odds,
                            'predicted_odds': predicted_odds,
                            'bet_amount': bet_amount,
                            'profit': profit if is_correct and predicted_odds else -bet_amount
                        })
                    
                    daily_games.append(game_info)
                
                # 해당 날짜 성과 계산
                daily_roi = (total_return / total_invested * 100) if total_invested > 0 else 0.0
                daily_win_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
                
                model_daily_performance[date] = {
                    'roi': daily_roi,
                    'win_rate': daily_win_rate,
                    'total_games': len(day_results),
                    'games_with_odds': games_with_odds,
                    'total_bets': total_predictions,
                    'correct_predictions': correct_predictions,
                    'total_invested': total_invested,
                    'profit_loss': total_return,
                    'games': daily_games
                }
            
            daily_performances[model] = model_daily_performance
        
        self.logger.info(f"✅ 일일 성과 계산 완료: {len(daily_performances)}개 모델")
        return daily_performances
    
    def calculate_model_roi(self, matched_results: List[Dict]) -> Dict[str, Dict]:
        """각 모델의 Actual ROI 계산 (홈팀 승률 기준)"""
        self.logger.info("💰 모델별 Actual ROI 계산 중...")
        
        # 모든 모델 찾기
        all_models = set()
        for result in matched_results:
            for key in result.keys():
                if key.endswith('_probability') and result[key] > 0:
                    model_name = key.replace('_probability', '')
                    all_models.add(model_name)
        
        self.logger.info(f"📊 발견된 모델: {len(all_models)}개")
        self.logger.info(f"   모델 목록: {sorted(list(all_models))}")
        
        model_results = {}
        
        # 🆕 전체 배당률 현황 분석
        total_games = len(matched_results)
        games_with_odds = sum(1 for r in matched_results if r.get('home_odds') is not None and r.get('away_odds') is not None)
        games_without_odds = total_games - games_with_odds
        
        self.logger.info(f"\n💰 배당률 현황:")
        self.logger.info(f"   전체 매칭된 게임: {total_games}개")
        self.logger.info(f"   배당률 있는 게임: {games_with_odds}개 ({games_with_odds/total_games*100:.1f}%)")
        self.logger.info(f"   배당률 없는 게임: {games_without_odds}개 ({games_without_odds/total_games*100:.1f}%)")
        
        for model in sorted(all_models):
            prob_key = f'{model}_probability'
            
            # 해당 모델의 데이터만 필터링 (배당률이 있는 경우만)
            model_data = [r for r in matched_results if r.get(prob_key, 0) > 0]
            
            if len(model_data) == 0:
                continue
            
            total_return = 0.0
            total_invested = 0.0
            correct_predictions = 0
            total_predictions = 0  # 배당률이 있는 경기만 카운트
            
            # 🆕 모델별 배당률 현황
            model_total_games = len(model_data)
            model_games_with_odds = 0
            
            for result in model_data:
                prob = result[prob_key]  # 홈팀 승률
                actual_home_win = result['actual_home_win']
                predicted_home_win = 1 if prob > 0.5 else 0  # 홈팀 승률 > 50%면 홈팀 승리 예측
                is_correct = predicted_home_win == actual_home_win
                
                # ROI 계산 (배당 정보가 있는 경우만)
                home_odds = result.get('home_odds')
                away_odds = result.get('away_odds')
                
                if home_odds is not None and away_odds is not None:
                    # 배당률이 있는 경우만 계산에 포함
                    total_predictions += 1
                    model_games_with_odds += 1
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    predicted_odds = home_odds if predicted_home_win else away_odds
                    bet_amount = 10  # 기본 베팅 금액
                    
                    if is_correct and predicted_odds:
                        if predicted_odds > 0:
                            profit = bet_amount * (predicted_odds / 100)
                        else:
                            profit = bet_amount * (100 / abs(predicted_odds))
                        total_return += profit
                    else:
                        total_return -= bet_amount
                    
                    total_invested += bet_amount
            
            # 최종 계산 (배당률이 있는 경기만으로)
            actual_roi = (total_return / total_invested * 100) if total_invested > 0 else 0.0
            win_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
            profit_loss = total_return
            
            model_results[model] = {
                'model_name': model,
                'actual_roi': actual_roi,
                'win_rate': win_rate,
                'total_bets': total_predictions,  # 배당률이 있는 경기 수만 표시
                'correct_predictions': correct_predictions,
                'profit_loss': profit_loss,
                'total_invested': total_invested
            }
        
            # 🆕 모델별 상세 로깅 (주요 모델들만)
            if model in ['model1', 'model2', 'model3', 'model6', 'model_advanced_xgboost']:
                excluded_games = model_total_games - model_games_with_odds
                self.logger.info(f"   🎯 {model}: {model_games_with_odds}/{model_total_games}경기 사용 (배당률 없어서 {excluded_games}경기 제외)")
        
        self.logger.info(f"\n✅ Actual ROI 계산 완료: {len(model_results)}개 모델")
        return model_results
    
    def analyze(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """전체 분석 실행 (Expected ROI와 Actual ROI 모두 계산)"""
        self.logger.info("🚀 모델 성과 분석 시작")
        
        # 1. 데이터 로드
        data = self.load_data(start_date, end_date)
        
        # 2. Expected ROI 계산 (확률과 배당 기반)
        expected_performances = self.calculate_expected_roi(data['predictions'])
        
        # 3. 매칭
        matched_results = self.match_predictions_with_results(
            data['predictions'], 
            data['historical_records']
        )
        
        # 4. Actual ROI 계산
        actual_performances = self.calculate_model_roi(matched_results)
        
        # 5. Expected와 Actual 결합
        combined_performances = {}
        
        # 모든 모델 목록 수집
        all_model_names = set(expected_performances.keys()) | set(actual_performances.keys())
        
        for model_name in all_model_names:
            expected = expected_performances.get(model_name, {})
            actual = actual_performances.get(model_name, {})
            
            combined_performances[model_name] = {
                'model_name': model_name,
                # Expected ROI 데이터
                'expected_roi': expected.get('expected_roi', 0.0),
                'expected_profit_loss': expected.get('expected_profit_loss', 0.0),
                # Actual ROI 데이터  
                'actual_roi': actual.get('actual_roi', 0.0),
                'win_rate': actual.get('win_rate', 0.0),
                'correct_predictions': actual.get('correct_predictions', 0),
                'profit_loss': actual.get('profit_loss', 0.0),
                # 공통 데이터
                'total_bets': actual.get('total_bets', expected.get('total_bets', 0)),
                'total_invested': actual.get('total_invested', expected.get('total_invested', 0.0)),
                # 차이 계산
                'roi_difference': actual.get('actual_roi', 0.0) - expected.get('expected_roi', 0.0)
            }
        
        # 6. 결과 요약
        analysis_summary = {
            'total_models': len(combined_performances),
            'total_games': len(matched_results),
            'date_range': {
                'start': start_date if start_date else 'All',
                'end': end_date if end_date else 'All'
            },
            'files_analyzed': len(data['predictions'])
        }
        
        self.logger.info("✅ 분석 완료!")
        
        # 7. 일일 성과 계산
        daily_performances = self.calculate_daily_performance(matched_results)
        
        return {
            'model_performances': combined_performances,
            'daily_performances': daily_performances,
            'analysis_summary': analysis_summary,
            'matched_data': matched_results,
            'file_info': data['file_info']
        }

def main():
    """테스트 실행"""
    analyzer = SimpleModelAnalyzer()
    
    try:
        # 사용 가능한 날짜 확인
        available_dates = analyzer.get_available_dates()
        print(f"사용 가능한 날짜: {available_dates}")
        
        # 분석 실행
        results = analyzer.analyze()
        
        print("\n" + "="*60)
        print("📊 모델 성과 분석 결과")
        print("="*60)
        
        summary = results['analysis_summary']
        print(f"\n📈 분석 요약:")
        print(f"   • 분석된 모델: {summary['total_models']}개")
        print(f"   • 총 경기: {summary['total_games']}개")
        print(f"   • 파일 수: {summary['files_analyzed']}개")
        
        print(f"\n🏆 모델별 성과 (ROI 순위):")
        
        # ROI 순으로 정렬
        sorted_models = sorted(
            results['model_performances'].items(), 
            key=lambda x: x[1]['actual_roi'], 
            reverse=True
        )
        
        for i, (model_name, performance) in enumerate(sorted_models, 1):
            expected_roi = performance['expected_roi']
            actual_roi = performance['actual_roi']
            roi_diff = performance['roi_difference']
            win_rate = performance['win_rate']
            total_bets = performance['total_bets']
            profit_loss = performance['profit_loss']
            
            status = "🟢" if actual_roi > 0 else "🔴" if actual_roi < -5 else "🟡"
            
            print(f"   {i:2d}. {status} {model_name:<30} | Expected: {expected_roi:6.2f}% | Actual: {actual_roi:6.2f}% | Diff: {roi_diff:6.2f}% | 승률: {win_rate:5.1f}% | 경기: {total_bets:3d} | 손익: ${profit_loss:6.1f}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 