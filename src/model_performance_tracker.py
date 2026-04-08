import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import streamlit as st


class ModelPerformanceTracker:
    """MLB 모델들의 성과를 추적하고 분석하는 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "predictions"
        self.records_dir = self.project_root / "data" / "records"
    
    def get_latest_prediction_files_by_date(self) -> Dict[str, str]:
        """날짜별로 가장 최신 앙상블 예측 파일들을 가져옵니다"""
        
        # 먼저 odds 디렉토리에서 mlb_predictions_with_odds 파일 찾기
        odds_dir = self.project_root / "src" / "odds" / "data" / "matched"
        prediction_files = []
        
        if odds_dir.exists():
            prediction_files = list(odds_dir.glob("mlb_predictions_with_odds_*.json"))
        
        # odds 파일이 없으면 기존 ensemble 파일 찾기
        if not prediction_files and self.predictions_dir.exists():
            prediction_files = list(self.predictions_dir.glob("mlb_ensemble_predictions_*.json"))
        
        if not prediction_files:
            st.error("예측 파일을 찾을 수 없습니다")
            return {}
        
        # 날짜별로 파일들 그룹화
        files_by_date = {}
        for file_path in prediction_files:
            try:
                # 파일명에서 날짜 추출
                filename = file_path.stem
                if "mlb_predictions_with_odds_" in filename:
                    # mlb_predictions_with_odds_YYYYMMDD_HHMMSS.json
                    date_part = filename.split('_')[4]  # YYYYMMDD 부분
                else:
                    # mlb_ensemble_predictions_YYYYMMDD_HHMMSS.json  
                    date_part = filename.split('_')[3]  # YYYYMMDD 부분
                    
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                
                if date_str not in files_by_date:
                    files_by_date[date_str] = []
                files_by_date[date_str].append(file_path)
            except (IndexError, ValueError):
                continue
        
        # 각 날짜별로 가장 최신 파일 선택
        latest_files = {}
        for date_str, files in files_by_date.items():
            # 파일 수정 시간으로 정렬하여 가장 최신 파일 선택
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            latest_files[date_str] = str(latest_file)
        
        return latest_files
    
    def get_latest_historical_records(self) -> Optional[str]:
        """가장 최신 히스토리컬 레코드 파일을 가져옵니다"""
        
        if not self.records_dir.exists():
            st.error(f"레코드 파일 디렉토리를 찾을 수 없습니다: {self.records_dir}")
            return None
        
        # mlb_historical_records_*.json 파일들 찾기
        record_files = list(self.records_dir.glob("mlb_historical_records_*.json"))
        
        if not record_files:
            st.error("히스토리컬 레코드 파일을 찾을 수 없습니다")
            return None
        
        # 가장 최신 파일 선택
        latest_file = max(record_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def load_prediction_data(self, file_path: str) -> List[Dict]:
        """예측 데이터 파일을 로드합니다"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"예측 데이터 로드 실패: {e}")
            return []
    
    def load_historical_data(self, file_path: str) -> List[Dict]:
        """히스토리컬 레코드 데이터를 로드합니다"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"히스토리컬 데이터 로드 실패: {e}")
            return []
    
    def match_predictions_with_results(self, predictions: List[Dict], historical_data: List[Dict]) -> List[Dict]:
        """예측 데이터와 실제 결과를 매칭합니다"""
        
        # 히스토리컬 데이터를 날짜-팀 조합으로 인덱싱
        historical_index = {}
        for record in historical_data:
            if 'date' in record and 'home_team_name' in record and 'away_team_name' in record:
                date = record['date']
                home_team = record['home_team_name']
                away_team = record['away_team_name']
                
                # 날짜-홈팀-원정팀 조합으로 키 생성
                key = f"{date}_{home_team}_{away_team}"
                historical_index[key] = record
        
        matched_data = []
        
        for prediction in predictions:
            pred_date = prediction.get('date', '')
            home_team = prediction.get('home_team', '')
            away_team = prediction.get('away_team', '')
            
            # 매칭 키 생성
            key = f"{pred_date}_{home_team}_{away_team}"
            
            if key in historical_index:
                historical_record = historical_index[key]
                
                # 실제 결과 확인 (home_win 필드 사용)
                if 'home_win' in historical_record:
                    actual_home_win = historical_record['home_win']
                    
                    # 매칭된 데이터 생성
                    matched_record = {
                        'date': pred_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'predicted_winner': prediction.get('predicted_winner', ''),
                        'win_probability': prediction.get('win_probability', 0),
                        'model1_probability': prediction.get('model1_probability', 0),
                        'model2_probability': prediction.get('model2_probability', 0),
                        'model3_probability': prediction.get('model3_probability', 0),
                        'model4_probability': prediction.get('model4_probability', 0),
                        'model5_probability': prediction.get('model5_probability', 0),
                        'model6_probability': prediction.get('model6_probability', 0),
                        'model7_probability': prediction.get('model7_probability', 0),
                        'model8_probability': prediction.get('model8_probability', 0),
                        'model9_probability': prediction.get('model9_probability', 0),
                        'model_rf_probability': prediction.get('model_rf_probability', 0),
                        'model_nn_probability': prediction.get('model_nn_probability', 0),
                        'model_svm_probability': prediction.get('model_svm_probability', 0),
                        'model_advanced_catboost_basic_probability': prediction.get('model_advanced_catboost_basic_probability', 0),
                        'model_advanced_catboost_probability': prediction.get('model_advanced_catboost_probability', 0),
                        'model_advanced_lgbm_basic_probability': prediction.get('model_advanced_lgbm_basic_probability', 0),
                        'model_advanced_lgbm_probability': prediction.get('model_advanced_lgbm_probability', 0),
                        'model_advanced_nn_probability': prediction.get('model_advanced_nn_probability', 0),
                        'model_advanced_rf_probability': prediction.get('model_advanced_rf_probability', 0),
                        'model_advanced_svm_probability': prediction.get('model_advanced_svm_probability', 0),
                        'model_advanced_xgboost_basic_probability': prediction.get('model_advanced_xgboost_basic_probability', 0),
                        'model_advanced_xgboost_probability': prediction.get('model_advanced_xgboost_probability', 0),
                        'model1_extended_lgbm_probability': prediction.get('model1_extended_lgbm_probability', 0),
                        'model2_extended_catboost_probability': prediction.get('model2_extended_catboost_probability', 0),
                        'model3_extended_xgboost_probability': prediction.get('model3_extended_xgboost_probability', 0),
                        'ensemble_probability': prediction.get('ensemble_probability', 0),
                        'actual_home_win': actual_home_win,
                        'actual_winner': home_team if actual_home_win == 1 else away_team
                    }
                    
                    matched_data.append(matched_record)
        
        return matched_data
    
    def calculate_model_performance(self, matched_data: List[Dict]) -> Dict:
        """모델별 성과를 계산합니다"""
        
        if not matched_data:
            return {}
        
        models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9', 
                 'model_rf', 'model_nn', 'model_svm', 
                 'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic', 
                 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                 'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 
                 'model2_extended_catboost', 'model3_extended_xgboost', 'ensemble']
        performance = {}
        
        for model in models:
            prob_key = f'{model}_probability'
            
            correct_predictions = 0
            total_predictions = 0  # Only count records with actual predictions
            
            probabilities = []
            actual_outcomes = []
            predicted_outcomes = []
            
            for record in matched_data:
                # Skip if no prediction data for this model
                if prob_key not in record or record[prob_key] is None:
                    continue
                    
                prob = record[prob_key]
                actual_home_win = record.get('actual_home_win', 0)
                
                # Skip if probability is 0 for new models (indicates no actual prediction)
                advanced_models = ['model_rf', 'model_nn', 'model_svm', 
                                 'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic',
                                 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                                 'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 
                                 'model2_extended_catboost', 'model3_extended_xgboost']
                if prob == 0 and model in advanced_models:
                    continue
                
                # Only count this record if we have a valid prediction
                total_predictions += 1
                
                # 모델의 예측 (홈팀 승리 확률 > 0.5이면 홈팀 승리 예측)
                predicted_home_win = 1 if prob > 0.5 else 0
                
                # 정확도 계산
                if predicted_home_win == actual_home_win:
                    correct_predictions += 1
                
                # 메트릭 계산을 위한 데이터 수집
                probabilities.append(prob)
                actual_outcomes.append(actual_home_win)
                predicted_outcomes.append(predicted_home_win)
            
            # Only include model in results if it has predictions
            if total_predictions > 0:
                # 기본 성과 메트릭 계산
                accuracy = correct_predictions / total_predictions
                
                # Brier Score 계산 (확률 예측의 정확도)
                brier_score = np.mean([(prob - actual) ** 2 for prob, actual in zip(probabilities, actual_outcomes)])
                
                # Log Loss 계산
                epsilon = 1e-15  # log(0) 방지
                log_loss = -np.mean([
                    actual * np.log(max(prob, epsilon)) + (1 - actual) * np.log(max(1 - prob, epsilon))
                    for prob, actual in zip(probabilities, actual_outcomes)
                ])
                
                # 평균 신뢰도 계산 (예측 확률의 절댓값)
                avg_confidence = np.mean([max(prob, 1-prob) for prob in probabilities])
                
                # Calibration 분석
                calibration = self._calculate_calibration(probabilities, actual_outcomes)
                
                performance[model] = {
                    'accuracy': accuracy,
                    'correct_predictions': correct_predictions,
                    'total_predictions': total_predictions,
                    'brier_score': brier_score,
                    'log_loss': log_loss,
                    'avg_confidence': avg_confidence,
                    'calibration': calibration
                }
        
        return performance
    
    def _calculate_calibration(self, probabilities: List[float], actual_outcomes: List[int]) -> Dict:
        """확률 구간별 calibration 계산"""
        
        # 확률을 10개 구간으로 나누어 calibration 계산
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        calibration_data = []
        
        for i in range(len(bins) - 1):
            bin_mask = (np.array(probabilities) >= bins[i]) & (np.array(probabilities) < bins[i+1])
            
            if np.sum(bin_mask) > 0:
                bin_probs = np.array(probabilities)[bin_mask]
                bin_outcomes = np.array(actual_outcomes)[bin_mask]
                
                avg_prob = np.mean(bin_probs)
                actual_freq = np.mean(bin_outcomes)
                count = np.sum(bin_mask)
                
                calibration_data.append({
                    'bin_center': bin_centers[i],
                    'avg_probability': avg_prob,
                    'actual_frequency': actual_freq,
                    'count': count,
                    'calibration_error': abs(avg_prob - actual_freq)
                })
        
        return calibration_data
    
    def analyze_prediction_confidence(self, matched_data: List[Dict]) -> Dict:
        """예측 신뢰도별 성과 분석"""
        
        confidence_ranges = {
            'Low (50-60%)': (0.5, 0.6),
            'Medium (60-70%)': (0.6, 0.7),
            'High (70-80%)': (0.7, 0.8),
            'Very High (80%+)': (0.8, 1.01)  # 1.0 포함하도록 수정
        }
        
        models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9', 
                 'model_rf', 'model_nn', 'model_svm',
                 'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic',
                 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                 'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 
                 'model2_extended_catboost', 'model3_extended_xgboost', 'ensemble']
        confidence_analysis = {}
        
        for model in models:
            prob_key = f'{model}_probability'
            model_analysis = {}
            
            for range_name, (min_conf, max_conf) in confidence_ranges.items():
                correct = 0
                total = 0
                confidence_values = []
                
                for record in matched_data:
                    # Skip if no prediction data for this model
                    if prob_key not in record or record[prob_key] is None:
                        continue
                        
                    prob = record[prob_key]
                    
                    # Skip if probability is 0 for new models (indicates no actual prediction)
                    advanced_models = ['model_rf', 'model_nn', 'model_svm',
                                     'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic',
                                     'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                                     'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 
                                     'model2_extended_catboost', 'model3_extended_xgboost']
                    if prob == 0 and model in advanced_models:
                        continue
                        
                    confidence = max(prob, 1 - prob)  # 승리 확률의 신뢰도
                    
                    # Very High 범위는 >= 조건, 나머지는 < 조건 사용
                    if range_name == 'Very High (80%+)':
                        in_range = confidence >= min_conf
                    else:
                        in_range = min_conf <= confidence < max_conf
                    
                    if in_range:
                        total += 1
                        confidence_values.append(confidence)
                        
                        predicted_home_win = 1 if prob > 0.5 else 0
                        actual_home_win = record.get('actual_home_win', 0)
                        
                        if predicted_home_win == actual_home_win:
                            correct += 1
                
                accuracy = correct / total if total > 0 else 0
                avg_confidence = np.mean(confidence_values) if confidence_values else 0
                
                model_analysis[range_name] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'avg_confidence': avg_confidence
                }
            
            # Only include model in analysis if it has any predictions
            if any(model_analysis[range_name]['total'] > 0 for range_name in confidence_ranges):
                confidence_analysis[model] = model_analysis
        
        return confidence_analysis
    
    def get_performance_summary(self, start_date: str = None, end_date: str = None) -> Tuple[Dict, List[Dict], Dict]:
        """전체 성과 요약을 가져옵니다"""
        
        # 예측 파일들 가져오기
        prediction_files = self.get_latest_prediction_files_by_date()
        
        if not prediction_files:
            return {}, [], {}
        
        # 날짜 필터링이 지정된 경우 해당 날짜 범위의 파일만 선택
        if start_date and end_date:
            filtered_files = {}
            for date, file_path in prediction_files.items():
                if start_date <= date <= end_date:
                    filtered_files[date] = file_path
            prediction_files = filtered_files
        
        if not prediction_files:
            return {}, [], {}
        
        # 히스토리컬 레코드 파일 가져오기
        historical_file = self.get_latest_historical_records()
        
        if not historical_file:
            return {}, [], {}
        
        # 히스토리컬 데이터 로드
        historical_data = self.load_historical_data(historical_file)
        
        # 선택된 날짜 범위의 예측 데이터만 수집 및 매칭
        all_matched_data = []
        
        for date, pred_file in prediction_files.items():
            predictions = self.load_prediction_data(pred_file)
            
            if predictions:
                matched_data = self.match_predictions_with_results(predictions, historical_data)
                all_matched_data.extend(matched_data)
        
        if not all_matched_data:
            return {}, [], {}
        
        # 성과 계산
        performance = self.calculate_model_performance(all_matched_data)
        confidence_analysis = self.analyze_prediction_confidence(all_matched_data)
        
        return performance, all_matched_data, confidence_analysis
    
    def identify_underdog_picks(self, prediction_data: List[Dict]) -> List[Dict]:
        """배당 기준으로 언더독 픽을 식별합니다"""
        underdog_picks = []
        
        for game in prediction_data:
            predicted_winner = game.get('predicted_winner', '')
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            home_odds = game.get('home_team_odds')
            away_odds = game.get('away_team_odds')
            
            # 배당이 없으면 스킵
            if home_odds is None or away_odds is None:
                continue
            
            # 언더독 픽 여부 판단 (배당이 높은 팀을 예측한 경우)
            is_underdog_pick = False
            predicted_odds = None
            
            if predicted_winner == home_team:
                # 홈팀 예측: 홈팀 배당이 더 높거나 양수이면서 원정팀이 음수인 경우
                if (home_odds > 0 and away_odds < 0) or (home_odds > away_odds):
                    is_underdog_pick = True
                    predicted_odds = home_odds
            elif predicted_winner == away_team:
                # 원정팀 예측: 원정팀 배당이 더 높거나 양수이면서 홈팀이 음수인 경우
                if (away_odds > 0 and home_odds < 0) or (away_odds > home_odds):
                    is_underdog_pick = True
                    predicted_odds = away_odds
            
            if is_underdog_pick:
                underdog_pick = game.copy()
                underdog_pick['predicted_odds'] = predicted_odds
                underdog_pick['underdog_team'] = predicted_winner
                underdog_picks.append(underdog_pick)
        
        return underdog_picks
    
    def analyze_underdog_performance(self, matched_data: List[Dict], start_date: str = None, end_date: str = None) -> Dict:
        """언더독 픽의 성과를 분석합니다"""
        
        # 예측 파일들 가져오기
        prediction_files = self.get_latest_prediction_files_by_date()
        
        # 날짜 필터링
        if start_date and end_date:
            filtered_files = {}
            for date, file_path in prediction_files.items():
                if start_date <= date <= end_date:
                    filtered_files[date] = file_path
            prediction_files = filtered_files
        
        if not prediction_files:
            return {}
        
        # 히스토리컬 레코드 파일 가져오기
        historical_file = self.get_latest_historical_records()
        if not historical_file:
            return {}
        
        # 히스토리컬 데이터 로드
        historical_data = self.load_historical_data(historical_file)
        
        # 파일별로 개별 처리하여 언더독 픽 분석
        underdog_results = []
        
        for date, pred_file in prediction_files.items():
            # 해당 파일의 예측 데이터 로드
            predictions = self.load_prediction_data(pred_file)
            if not predictions:
                continue
            
            # 해당 파일의 언더독 픽 식별
            underdog_picks = self.identify_underdog_picks(predictions)
            if not underdog_picks:
                continue
            
            # 해당 파일의 예측 데이터를 히스토리컬 데이터와 매칭
            file_matched_data = self.match_predictions_with_results(predictions, historical_data)
            
            # 해당 파일의 매칭된 데이터에서 언더독 픽만 필터링
            for matched_record in file_matched_data:
                # 같은 파일 내의 언더독 픽과 매칭
                for underdog_pick in underdog_picks:
                    if (matched_record['date'] == underdog_pick.get('date') and
                        matched_record['home_team'] == underdog_pick.get('home_team') and
                        matched_record['away_team'] == underdog_pick.get('away_team') and
                        matched_record['predicted_winner'] == underdog_pick.get('predicted_winner')):
                        
                        # 언더독 픽 결과 생성
                        underdog_result = matched_record.copy()
                        underdog_result['predicted_odds'] = underdog_pick['predicted_odds']
                        underdog_result['underdog_team'] = underdog_pick['underdog_team']
                        underdog_results.append(underdog_result)
                        break
        
        if not underdog_results:
            return {
                'total_picks': 0,
                'correct_picks': 0,
                'accuracy': 0,
                'details': [],
                'roi_analysis': {},
                'performance_by_odds_range': {}
            }
        
        # 성과 분석
        correct_picks = 0
        total_picks = len(underdog_results)
        
        # ROI 분석을 위한 데이터
        bet_results = []
        
        for result in underdog_results:
            is_correct = result['predicted_winner'] == result['actual_winner']
            if is_correct:
                correct_picks += 1
            
            # 베팅 결과 계산 (고정 $10 베팅 가정)
            bet_amount = 10
            if is_correct:
                # 배당에 따른 이익 계산
                odds = result['predicted_odds']
                if odds > 0:
                    profit = bet_amount * (odds / 100)
                else:
                    profit = bet_amount * (100 / abs(odds))
            else:
                profit = -bet_amount
            
            bet_results.append({
                'game': f"{result['away_team']} @ {result['home_team']}",
                'date': result['date'],
                'predicted_winner': result['predicted_winner'],
                'actual_winner': result['actual_winner'],
                'odds': result['predicted_odds'],
                'ensemble_probability': result['ensemble_probability'],
                'bet_amount': bet_amount,
                'profit': profit,
                'is_correct': is_correct
            })
        
        accuracy = correct_picks / total_picks if total_picks > 0 else 0
        
        # ROI 계산
        total_invested = sum(bet['bet_amount'] for bet in bet_results)
        total_profit = sum(bet['profit'] for bet in bet_results)
        net_profit = total_profit
        roi = (net_profit / total_invested * 100) if total_invested > 0 else 0
        
        # 배당 범위별 성과 분석
        odds_ranges = {
            'Small Underdog (+100 to +200)': (100, 200),
            'Medium Underdog (+200 to +400)': (200, 400),
            'Big Underdog (+400+)': (400, float('inf'))
        }
        
        performance_by_odds = {}
        for range_name, (min_odds, max_odds) in odds_ranges.items():
            range_results = [r for r in underdog_results if min_odds <= r['predicted_odds'] <= max_odds]
            
            if range_results:
                range_correct = sum(1 for r in range_results if r['predicted_winner'] == r['actual_winner'])
                range_total = len(range_results)
                range_accuracy = range_correct / range_total
                
                # 해당 범위의 ROI 계산
                range_bets = [b for b in bet_results if min_odds <= b['odds'] <= max_odds]
                range_invested = sum(b['bet_amount'] for b in range_bets)
                range_profit = sum(b['profit'] for b in range_bets)
                range_roi = (range_profit / range_invested * 100) if range_invested > 0 else 0
                
                performance_by_odds[range_name] = {
                    'total_picks': range_total,
                    'correct_picks': range_correct,
                    'accuracy': range_accuracy,
                    'total_invested': range_invested,
                    'net_profit': range_profit,
                    'roi': range_roi
                }
        
        return {
            'total_picks': total_picks,
            'correct_picks': correct_picks,
            'accuracy': accuracy,
            'total_invested': total_invested,
            'net_profit': net_profit,
            'roi': roi,
            'details': bet_results,
            'performance_by_odds_range': performance_by_odds,
            'avg_odds': np.mean([r['predicted_odds'] for r in underdog_results]) if underdog_results else 0,
            'avg_ensemble_prob': np.mean([r['ensemble_probability'] for r in underdog_results]) if underdog_results else 0
        } 