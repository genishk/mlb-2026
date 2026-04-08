import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import itertools
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class StableEnsembleOptimizer:
    """
    안정성 우선 앙상블 최적화 도구
    
    목표:
    - 높은 저점 (낮은 최대 낙폭)
    - 일관된 성과 (낮은 변동성)
    - 양호한 전체 ROI
    - 확실한 근거와 검증
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "odds" / "data" / "matched"
        self.records_dir = self.project_root / "data" / "records"
        
        # 최적화 설정 (더 관대하게)
        self.min_games_required = 20  # 최소 게임 수 (기존 50에서 20으로 낮춤)
        self.outlier_threshold = 3.0  # 아웃라이어 제거 기준 (표준편차)
        
    def load_historical_performance_data(self, start_date: str = None, end_date: str = None, days_back: int = 60) -> Dict:
        """히스토리컬 성과 데이터 로드 및 전처리 (날짜 범위 지원)"""
        
        # 예측 파일들 로드
        prediction_files = list(self.predictions_dir.glob("mlb_predictions_with_odds_*.json"))
        if not prediction_files:
            raise ValueError("예측 파일을 찾을 수 없습니다")
        
        # 레코드 파일 로드
        record_files = list(self.records_dir.glob("mlb_historical_records_*.json"))
        if not record_files:
            raise ValueError("히스토리컬 레코드 파일을 찾을 수 없습니다")
        
        latest_record_file = max(record_files, key=lambda x: x.stat().st_mtime)
        with open(latest_record_file, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)
        
        # 날짜 필터링 기준 설정
        if start_date and end_date:
            # 특정 날짜 범위
            print(f"날짜 범위 필터링: {start_date} ~ {end_date}")
        else:
            # 기존 방식 (최근 N일)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            print(f"최근 {days_back}일 데이터 사용")
        
        all_predictions = {}
        for file_path in prediction_files:
            try:
                # 파일명에서 날짜 추출
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) >= 5:
                    date_part = parts[4]  # YYYYMMDD
                    file_date = datetime.strptime(date_part, '%Y%m%d')
                    file_date_str = file_date.strftime('%Y-%m-%d')
                    
                    # 날짜 범위 확인
                    if start_date and end_date:
                        if start_date <= file_date_str <= end_date:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_predictions[date_part] = data
                    else:
                        if file_date >= cutoff_date:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_predictions[date_part] = data
            except Exception as e:
                print(f"파일 처리 중 오류 ({file_path}): {e}")
                continue
        
        print(f"로드된 예측 파일 수: {len(all_predictions)}")
        
        return {
            'predictions': all_predictions,
            'historical_records': historical_data
        }
    
    def match_predictions_with_results(self, predictions: Dict, historical_data: List) -> List[Dict]:
        """예측과 실제 결과 매칭 (simple_model_analyzer와 동일한 방식)"""
        
        # 히스토리컬 데이터 인덱싱
        historical_index = {}
        for record in historical_data:
            if all(field in record for field in ['date', 'home_team_name', 'away_team_name']):
                key = f"{record['date']}_{record['home_team_name']}_{record['away_team_name']}"
                historical_index[key] = record
        
        matched_results = []
        
        for date_str, pred_data in predictions.items():
            for prediction in pred_data:
                pred_date = prediction.get('date', '')
                home_team = prediction.get('home_team', '')
                away_team = prediction.get('away_team', '')
                
                key = f"{pred_date}_{home_team}_{away_team}"
                
                if key in historical_index:
                    historical_record = historical_index[key]
                    
                    if 'home_win' in historical_record:
                        # 기본 정보
                        matched_record = {
                            'date': pred_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'actual_home_win': historical_record.get('home_win', 0),
                            'home_score': historical_record.get('home_score', 0),
                            'away_score': historical_record.get('away_score', 0),
                            # 배당률 (simple_model_analyzer와 동일한 필드명 사용)
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
                        for field_key, value in prediction.items():
                            if (field_key.endswith('_probability') and 
                                isinstance(value, (int, float)) and 
                                value > 0 and 
                                field_key not in excluded_probability_fields):
                                matched_record[field_key] = value
                        
                        matched_results.append(matched_record)
        
        print(f"매칭 결과: {len(matched_results)}개 경기")
        return matched_results
    
    def remove_outliers(self, data: List[Dict]) -> List[Dict]:
        """아웃라이어 제거로 안정성 향상 (배당률 필수)"""
        
        if len(data) < 10:
            return data
        
        # 🚨 배당률이 있는 게임만 필터링 먼저 수행
        odds_available_data = []
        for record in data:
            home_odds = record.get('home_odds')
            away_odds = record.get('away_odds')
            
            if home_odds is not None and away_odds is not None:
                try:
                    float(home_odds)
                    float(away_odds)
                    odds_available_data.append(record)
                except (ValueError, TypeError):
                    continue
        
        print(f"아웃라이어 제거: 배당률 사용 가능 게임 {len(odds_available_data)}/{len(data)}")
        
        if len(odds_available_data) < 10:
            print(f"아웃라이어 제거: 배당률 사용 가능 게임 부족 ({len(odds_available_data)}개), 원본 데이터 반환")
            return odds_available_data  # 배당률 있는 게임들만 반환
        
        # 날짜별 수익률 계산 (배당률 기반)
        daily_returns = {}
        for record in odds_available_data:
            date = record['date']
            if date not in daily_returns:
                daily_returns[date] = []
            
            home_odds = float(record['home_odds'])
            away_odds = float(record['away_odds'])
            
            # 간단한 수익률 계산 (앙상블 확률 없이 50% 기준 사용)
            home_prob = 0.5  # 임시로 50% 사용 (아웃라이어 제거용)
            actual_win = record.get('actual_home_win', 0)
            
            if home_prob > 0.5:
                # 홈팀에 베팅
                if actual_win == 1:
                    if home_odds > 0:
                        return_rate = (home_odds / 100) * 100  # 양수 배당률
                    else:
                        return_rate = (100 / abs(home_odds)) * 100  # 음수 배당률
                else:
                    return_rate = -100  # 손실
            else:
                # 원정팀에 베팅
                if actual_win == 0:
                    if away_odds > 0:
                        return_rate = (away_odds / 100) * 100  # 양수 배당률
                    else:
                        return_rate = (100 / abs(away_odds)) * 100  # 음수 배당률
                else:
                    return_rate = -100  # 손실
            
            daily_returns[date].append(return_rate)
        
        # 빈 날짜는 제거
        daily_returns = {date: returns for date, returns in daily_returns.items() if returns}
        
        if len(daily_returns) < 3:  # 최소 3일의 데이터가 필요
            print(f"아웃라이어 제거: 데이터 부족으로 스킵, {len(odds_available_data)} 데이터 유지")
            return odds_available_data
        
        # 일일 평균 수익률 계산
        daily_avg_returns = [np.mean(returns) for returns in daily_returns.values()]
        
        # 아웃라이어 감지 및 제거
        mean_return = np.mean(daily_avg_returns)
        std_return = np.std(daily_avg_returns)
        
        if std_return == 0:  # 표준편차가 0이면 아웃라이어 없음
            print(f"아웃라이어 제거: 표준편차 0으로 스킵, {len(odds_available_data)} 데이터 유지")
            return odds_available_data
        
        outlier_dates = set()
        for date, avg_return in zip(daily_returns.keys(), daily_avg_returns):
            if abs(avg_return - mean_return) > self.outlier_threshold * std_return:
                outlier_dates.add(date)
        
        # 아웃라이어 날짜 제거
        filtered_data = [record for record in odds_available_data if record['date'] not in outlier_dates]
        
        print(f"아웃라이어 제거: {len(outlier_dates)}일 제거, {len(filtered_data)}/{len(odds_available_data)} 데이터 유지")
        
        return filtered_data
    
    def calculate_ensemble_performance(self, matched_data: List[Dict], weights: Dict[str, float]) -> Dict:
        """앙상블 성과 계산 (안정성 지표 포함)"""
        
        if not matched_data:
            return self._empty_performance()
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self._empty_performance()
        
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        daily_returns = []
        total_profit = 0
        total_invested = 0
        correct_predictions = 0
        
        # 날짜별로 그룹화하여 계산
        data_by_date = {}
        for record in matched_data:
            date = record['date']
            if date not in data_by_date:
                data_by_date[date] = []
            data_by_date[date].append(record)
        
        for date, day_records in data_by_date.items():
            day_profit = 0
            day_invested = 0
            day_correct = 0
            
            for record in day_records:
                # 앙상블 확률 계산
                ensemble_prob = 0
                total_weight_used = 0
                
                for model, weight in normalized_weights.items():
                    prob_key = f"{model}_probability"
                    if prob_key in record and record[prob_key] is not None:
                        ensemble_prob += record[prob_key] * weight
                        total_weight_used += weight
                
                if total_weight_used == 0:
                    continue
                
                ensemble_prob /= total_weight_used
                
                # 베팅 결정 및 수익 계산
                bet_amount = 100  # 단위 베팅
                actual_win = record.get('actual_home_win', 0)
                home_odds = record.get('home_odds')
                away_odds = record.get('away_odds')
                
                # 🚨 배당률이 필수! 없으면 스킵
                if home_odds is None or away_odds is None:
                    continue  # 배당률 없으면 의미있는 ROI 계산 불가
                
                try:
                    home_odds = float(home_odds)
                    away_odds = float(away_odds)
                except (ValueError, TypeError):
                    continue  # 잘못된 배당률 형식이면 스킵
                
                # 배당률 기반 ROI 계산
                if ensemble_prob > 0.5:
                    # 홈팀에 베팅
                    day_invested += bet_amount
                    if actual_win == 1:
                        if home_odds > 0:
                            profit = bet_amount * (home_odds / 100)  # 양수 배당률
                        else:
                            profit = bet_amount * (100 / abs(home_odds))  # 음수 배당률
                        day_profit += profit
                        day_correct += 1
                    else:
                        day_profit -= bet_amount
                else:
                    # 원정팀에 베팅
                    day_invested += bet_amount
                    if actual_win == 0:
                        if away_odds > 0:
                            profit = bet_amount * (away_odds / 100)  # 양수 배당률
                        else:
                            profit = bet_amount * (100 / abs(away_odds))  # 음수 배당률
                        day_profit += profit
                        day_correct += 1
                    else:
                        day_profit -= bet_amount
            
            if day_invested > 0:
                day_return = (day_profit / day_invested) * 100
                daily_returns.append(day_return)
                total_profit += day_profit
                total_invested += day_invested
                correct_predictions += day_correct
        
        # 성과 지표 계산
        performance = self._calculate_risk_metrics(daily_returns, total_profit, total_invested, correct_predictions, len(matched_data))
        
        return performance
    
    def _calculate_risk_metrics(self, daily_returns: List[float], total_profit: float, 
                               total_invested: float, correct_predictions: int, total_games: int) -> Dict:
        """리스크 조정 성과 지표 계산"""
        
        if not daily_returns or total_invested == 0:
            return self._empty_performance()
        
        returns_array = np.array(daily_returns)
        
        # 기본 지표
        roi = (total_profit / total_invested) * 100 if total_invested > 0 else 0
        win_rate = (correct_predictions / total_games) * 100 if total_games > 0 else 0
        
        # 리스크 지표
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        
        # Sharpe Ratio (무위험 수익률 0 가정)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (하방 위험만 고려)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = avg_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum Drawdown 계산
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # 안정성 점수 (여러 지표 종합)
        stability_score = self._calculate_stability_score(
            sharpe_ratio, sortino_ratio, max_drawdown, volatility
        )
        
        # 손실 연속일 계산
        consecutive_losses = self._calculate_consecutive_losses(returns_array)
        
        return {
            'roi': roi,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_invested': total_invested,
            'total_games': total_games,
            'correct_predictions': correct_predictions,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'stability_score': stability_score,
            'avg_daily_return': avg_return,
            'consecutive_losses_max': consecutive_losses,
            'daily_returns': daily_returns,
            'profit_days': len([r for r in daily_returns if r > 0]),
            'loss_days': len([r for r in daily_returns if r < 0]),
            'total_days': len(daily_returns)
        }
    
    def _calculate_stability_score(self, sharpe: float, sortino: float, 
                                  max_dd: float, volatility: float) -> float:
        """안정성 점수 계산 (0-100, 높을수록 안정적)"""
        
        # 각 지표를 0-100 스케일로 정규화
        sharpe_score = min(max(sharpe * 20 + 50, 0), 100)  # Sharpe 0이면 50점
        sortino_score = min(max(sortino * 20 + 50, 0), 100)
        drawdown_score = max(100 - max_dd, 0)  # 낙폭이 클수록 점수 낮음
        volatility_score = max(100 - volatility * 2, 0)  # 변동성이 클수록 점수 낮음
        
        # 가중 평균 (낙폭과 변동성에 더 큰 가중치)
        stability_score = (
            sharpe_score * 0.2 +
            sortino_score * 0.2 +
            drawdown_score * 0.35 +
            volatility_score * 0.25
        )
        
        return stability_score
    
    def _calculate_consecutive_losses(self, returns: np.ndarray) -> int:
        """최대 연속 손실 일수 계산"""
        
        if len(returns) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for return_val in returns:
            if return_val < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _empty_performance(self) -> Dict:
        """빈 성과 딕셔너리 반환"""
        return {
            'roi': 0, 'win_rate': 0, 'total_profit': 0, 'total_invested': 0,
            'total_games': 0, 'correct_predictions': 0, 'sharpe_ratio': 0,
            'sortino_ratio': 0, 'volatility': 0, 'max_drawdown': 0,
            'stability_score': 0, 'avg_daily_return': 0, 'consecutive_losses_max': 0,
            'daily_returns': [], 'profit_days': 0, 'loss_days': 0, 'total_days': 0
        }
    
    def grid_search_optimization(self, matched_data: List[Dict], 
                               available_models: List[str], 
                               target_metric: str = "stability_score",
                               return_top_n: int = 1) -> Dict:
        """그리드 서치로 최적 가중치 탐색 (상위 N개 반환 옵션 추가)"""
        
        print(f"그리드 서치 시작: {len(available_models)}개 모델, 목표지표: {target_metric}")
        
        # 가중치 후보값 (기존 7개 + 2개 추가)
        weight_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        print(f"가중치 후보값: {weight_options}")
        
        # 상위 N개를 저장할 리스트
        top_results = []
        
        # 모든 가중치 조합 생성 (합이 1.0이 되는 조합)
        total_combinations = 0
        tested_combinations = 0
        
        for weights_tuple in itertools.product(weight_options, repeat=len(available_models)):
            total_combinations += 1
            
            # 가중치 합이 0.95~1.05 범위인 조합만 테스트 (반올림 오차 고려)
            if 0.95 <= sum(weights_tuple) <= 1.05:
                weights_dict = {model: weight for model, weight in zip(available_models, weights_tuple)}
                
                # 성과 계산
                performance = self.calculate_ensemble_performance(matched_data, weights_dict)
                
                if performance['total_games'] >= self.min_games_required:
                    tested_combinations += 1
                    
                    # 목표 지표에 따라 점수 계산
                    if target_metric == "stability_score":
                        score = performance['stability_score']
                    elif target_metric == "roi":
                        score = performance['roi']
                    elif target_metric == "sharpe_ratio":
                        score = performance['sharpe_ratio']
                    elif target_metric == "risk_adjusted_roi":
                        # ROI를 최대 낙폭으로 나눈 값 (위험 조정 수익률)
                        score = performance['roi'] / max(performance['max_drawdown'], 1)
                    else:
                        score = performance['stability_score']
                    
                    # 결과를 리스트에 추가
                    top_results.append({
                        'weights': weights_dict.copy(),
                        'performance': performance.copy(),
                        'score': score
                    })
        
        print(f"그리드 서치 완료: {tested_combinations}/{total_combinations} 조합 테스트")
        
        # 테스트된 조합이 없으면 균등 가중치 사용
        if tested_combinations == 0:
            print("⚠️  그리드 서치에서 유효한 조합을 찾지 못했습니다. 균등 가중치를 사용합니다.")
            equal_weight = 1.0 / len(available_models)
            best_weights = {model: equal_weight for model in available_models}
            best_performance = self.calculate_ensemble_performance(matched_data, best_weights)
            
            return {
                'optimal_weights': best_weights,
                'best_performance': best_performance,
                'optimization_target': target_metric,
                'tested_combinations': tested_combinations,
                'total_combinations': total_combinations,
                'top_results': [{'weights': best_weights, 'performance': best_performance, 'score': 0}]
            }
        
        # 점수 기준으로 정렬하여 상위 N개 선택
        top_results.sort(key=lambda x: x['score'], reverse=True)
        top_n_results = top_results[:min(return_top_n, len(top_results))]
        
        # 최고 성과 (기존 호환성을 위해)
        best_result = top_n_results[0]
        
        print(f"상위 {len(top_n_results)}개 조합 선정:")
        for i, result in enumerate(top_n_results[:5]):  # 상위 5개만 출력
            print(f"{i+1}. {target_metric}={result['score']:.3f}, 가중치={result['weights']}")
        
        return {
            'optimal_weights': best_result['weights'],
            'best_performance': best_result['performance'],
            'optimization_target': target_metric,
            'tested_combinations': tested_combinations,
            'total_combinations': total_combinations,
            'top_results': top_n_results  # 🎯 상위 N개 결과 추가
        }
    
    def validate_weights_stability(self, matched_data: List[Dict], 
                                 weights: Dict[str, float], 
                                 validation_periods: int = 5) -> Dict:
        """가중치 안정성 검증 (여러 기간으로 분할하여 일관성 확인)"""
        
        if len(matched_data) < validation_periods * 10:
            return {'is_stable': False, 'reason': '검증용 데이터 부족'}
        
        # 데이터를 날짜순으로 정렬
        sorted_data = sorted(matched_data, key=lambda x: x['date'])
        
        # 검증 기간별로 분할
        period_size = len(sorted_data) // validation_periods
        period_performances = []
        
        for i in range(validation_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < validation_periods - 1 else len(sorted_data)
            
            period_data = sorted_data[start_idx:end_idx]
            performance = self.calculate_ensemble_performance(period_data, weights)
            period_performances.append(performance)
        
        # 안정성 지표 계산
        rois = [p['roi'] for p in period_performances]
        stability_scores = [p['stability_score'] for p in period_performances]
        max_drawdowns = [p['max_drawdown'] for p in period_performances]
        
        roi_consistency = 1 - (np.std(rois) / (np.mean(np.abs(rois)) + 1e-6))
        stability_consistency = 1 - (np.std(stability_scores) / (np.mean(stability_scores) + 1e-6))
        
        # 전체 안정성 점수
        overall_stability = (roi_consistency + stability_consistency) / 2
        
        # 안정성 판정 기준
        is_stable = (
            overall_stability > 0.7 and  # 70% 이상 일관성
            np.mean(max_drawdowns) < 20 and  # 평균 낙폭 20% 미만
            min(rois) > -10  # 최악 기간에도 -10% 이상
        )
        
        return {
            'is_stable': is_stable,
            'overall_stability': overall_stability,
            'roi_consistency': roi_consistency,
            'stability_consistency': stability_consistency,
            'period_performances': period_performances,
            'period_rois': rois,
            'period_stability_scores': stability_scores,
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_period_roi': min(rois)
        }
    
    def optimize_ensemble(self, start_date: str = None, end_date: str = None, 
                         days_back: int = 60, 
                         target_metric: str = "stability_score",
                         max_models: int = 6,
                         auto_select_models: bool = True,
                         stability_weight: float = 0.5,
                         roi_weight: float = 0.5,
                         min_roi: float = -15.0,
                         min_stability: float = 30.0,
                         max_drawdown: float = 75.0,
                         max_consecutive_losses: int = 10,
                         min_games: int = 10) -> Dict:
        """메인 최적화 함수 (개선된 버전)"""
        
        print(f"=== 앙상블 최적화 시작 ===")
        if start_date and end_date:
            print(f"분석 기간: {start_date} ~ {end_date}")
        else:
            print(f"분석 기간: 최근 {days_back}일")
        print(f"목표 지표: {target_metric}")
        print(f"자동 모델 선별: {'예' if auto_select_models else '아니오'}")
        
        # 1. 데이터 로드
        data = self.load_historical_performance_data(start_date, end_date, days_back)
        matched_data = self.match_predictions_with_results(
            data['predictions'], data['historical_records']
        )
        
        print(f"매칭된 게임 수: {len(matched_data)}")
        
        if len(matched_data) < self.min_games_required:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.min_games_required}게임 필요, 현재 {len(matched_data)}게임")
        
        # 2. 아웃라이어 제거
        cleaned_data = self.remove_outliers(matched_data)
        
        # 3. 사용 가능한 모델 자동 발견
        available_models = self.discover_available_models(cleaned_data)
        
        if len(available_models) == 0:
            raise ValueError("사용 가능한 모델을 찾을 수 없습니다")
        
        print(f"사용 가능한 모델: {len(available_models)}개")
        
        # 4. 개별 모델 성과 평가 및 선별
        if auto_select_models:
            model_performances = self.evaluate_individual_models(cleaned_data, available_models)
            selected_models = self.select_top_models(
                model_performances, 
                max_models=max_models,
                min_roi=min_roi,
                min_stability=min_stability,
                stability_weight=stability_weight,
                roi_weight=roi_weight,
                max_drawdown=max_drawdown,
                max_consecutive_losses=max_consecutive_losses,
                min_games=min_games
            )
        else:
            selected_models = available_models[:max_models]  # 처음 몇 개만 선택
            model_performances = self.evaluate_individual_models(cleaned_data, selected_models)
        
        if len(selected_models) < 1:
            raise ValueError(f"선별된 모델이 부족합니다 ({len(selected_models)}개). 최소 1개 필요")
        
        # 5. 앙상블 최적화 실행 (Train-Validation Split)
        print(f"\n=== Train-Validation Split 최적화 (파일 단위) ===")
        
        # 🆕 파일 단위로 분할하기 위해 날짜별로 그룹화
        date_groups = {}
        for record in cleaned_data:
            date = record['date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(record)
        
        # 날짜순으로 정렬
        sorted_dates = sorted(date_groups.keys())
        
        # 파일 단위 7:3 분할
        total_files = len(sorted_dates)
        train_file_count = round(total_files * 0.7)  # 🆕 int() 대신 round() 사용하여 반올림
        if train_file_count == 0:  # 파일이 1개뿐인 경우
            train_file_count = 1
        elif train_file_count == total_files:  # 모든 파일이 training에 할당되는 경우
            train_file_count = max(1, total_files - 1)  # 최소 1개는 validation용으로
        
        train_dates = sorted_dates[:train_file_count]
        validation_dates = sorted_dates[train_file_count:]
        
        # 분할된 데이터 구성
        train_data = []
        validation_data = []
        
        for date in train_dates:
            train_data.extend(date_groups[date])
        
        for date in validation_dates:
            validation_data.extend(date_groups[date])
        
        print(f"파일 단위 분할:")
        print(f"  총 파일 수: {total_files}개")
        print(f"  분석용 파일: {len(train_dates)}개 ({train_dates})")
        print(f"  검증용 파일: {len(validation_dates)}개 ({validation_dates})")
        print(f"  분석용 데이터: {len(train_data)}게임")
        print(f"  검증용 데이터: {len(validation_data)}게임")
        
        if len(train_data) < self.min_games_required:
            raise ValueError(f"분석용 데이터가 부족합니다. 최소 {self.min_games_required}게임 필요, 현재 {len(train_data)}게임")
        
        if len(validation_data) < 5:  # 검증용 데이터 최소 5게임
            print("⚠️  검증용 데이터가 부족합니다. 전체 데이터로 최적화를 진행합니다.")
            train_data = cleaned_data
            validation_data = None
            train_dates = sorted_dates
            validation_dates = []
        
        # Step 1: 분석용 데이터로 상위 10개 조합 탐색
        print("\n=== Step 1: 분석용 데이터로 상위 10개 조합 탐색 ===")
        train_optimization = self.grid_search_optimization(
            train_data, selected_models, target_metric, return_top_n=10
        )
        
        if train_optimization['optimal_weights'] is None:
            raise ValueError("분석용 데이터에서 유효한 가중치를 찾지 못했습니다")
        
        # Step 2: 검증용 데이터로 최종 조합 선정
        if validation_data is not None:
            print("\n=== Step 2: 검증용 데이터로 최종 조합 선정 ===")
            
            validation_results = []
            for i, candidate in enumerate(train_optimization['top_results']):
                val_performance = self.calculate_ensemble_performance(validation_data, candidate['weights'])
                
                # 목표 지표에 따라 검증 점수 계산
                if target_metric == "stability_score":
                    val_score = val_performance['stability_score']
                elif target_metric == "roi":
                    val_score = val_performance['roi']
                elif target_metric == "sharpe_ratio":
                    val_score = val_performance['sharpe_ratio']
                elif target_metric == "risk_adjusted_roi":
                    val_score = val_performance['roi'] / max(val_performance['max_drawdown'], 1)
                else:
                    val_score = val_performance['stability_score']
                
                validation_results.append({
                    'weights': candidate['weights'],
                    'train_performance': candidate['performance'],
                    'train_score': candidate['score'],
                    'validation_performance': val_performance,
                    'validation_score': val_score
                })
                
                print(f"후보 {i+1}: Train {target_metric}={candidate['score']:.3f}, Val {target_metric}={val_score:.3f}")
            
            # 검증 점수 기준으로 최종 선정
            best_candidate = max(validation_results, key=lambda x: x['validation_score'])
            
            print(f"\n최종 선정: Train {target_metric}={best_candidate['train_score']:.3f}, Val {target_metric}={best_candidate['validation_score']:.3f}")
            
            optimization_result = {
                'optimal_weights': best_candidate['weights'],
                'train_performance': best_candidate['train_performance'],
                'validation_performance': best_candidate['validation_performance'],
                'optimization_target': target_metric,
                'tested_combinations': train_optimization['tested_combinations'],
                'total_combinations': train_optimization['total_combinations'],
                'validation_results': validation_results,
                'split_used': True,
                # 🆕 사용된 파일 정보 추가
                'train_files': train_dates,
                'validation_files': validation_dates
            }
        else:
            # 검증용 데이터가 없으면 기존 방식
            optimization_result = train_optimization
            optimization_result['split_used'] = False
            optimization_result['train_files'] = train_dates
            optimization_result['validation_files'] = []
        
        # 6. 안정성 검증
        stability_validation = self.validate_weights_stability(
            cleaned_data, optimization_result['optimal_weights']
        )
        
        # 7. 전체 데이터로 최종 성과 계산 (리포팅용)
        final_performance = self.calculate_ensemble_performance(
            cleaned_data, optimization_result['optimal_weights']
        )
        
        # 🆕 8. 앙상블 구간별 성과 분석 - 동일한 분할 데이터 사용으로 효율성 개선
        print(f"\n=== 구간별 성과 분석 (동일한 분할 데이터 사용) ===")
        
        # ✅ 이미 분할된 train_data와 validation_data 사용 (중복 분할 방지)
        if 'train_data' in locals() and len(train_data) > 0:
            validation_data_len = len(validation_data) if validation_data is not None else 0
            print(f"📊 구간분석용 데이터: {len(train_data)}개 (최적화와 동일한 분할)")
            print(f"🔍 검증용 데이터: {validation_data_len}개 (최적화와 동일한 분할)")
            
            # 동일한 분할 데이터로 구간분석 수행
            segments_analysis = self.analyze_ensemble_segments(train_data, optimization_result['optimal_weights'])
            validation_data_for_zones = validation_data if validation_data is not None else []
            
            # 분할 정보를 segments_analysis에 추가 (기존과 호환성 유지)
            if segments_analysis:
                segments_analysis['split_info'] = {
                    'total_games': len(cleaned_data),
                    'analysis_games': len(train_data),
                    'validation_games': validation_data_len,
                    'split_date': train_dates[-1] if train_dates else '',
                    'split_method': 'file_based_same_as_optimization'
                }
        else:
            # 백업: 분할된 데이터가 없으면 기존 방식 사용
            print(f"⚠️ 분할된 데이터가 없어 기존 방식으로 구간분석 수행")
            segments_analysis_result = self.analyze_ensemble_segments_with_split(
                cleaned_data, optimization_result['optimal_weights']
            )
            segments_analysis = segments_analysis_result['segments_analysis']
            validation_data_for_zones = segments_analysis_result['validation_data']
        
        # 🆕 전체 데이터 구간분석 (참고용)
        print(f"\n=== 참고용 전체 데이터 구간분석 ===")
        full_segments_analysis = self.analyze_ensemble_segments(cleaned_data, optimization_result['optimal_weights'])
        
        # 9. 결과 종합
        result = {
            'optimal_weights': optimization_result['optimal_weights'],
            'final_performance': final_performance,
            'stability_validation': stability_validation,
            'optimization_details': optimization_result,
            'model_discovery': {
                'available_models': available_models,
                'selected_models': selected_models,
                'individual_performances': model_performances,
                'auto_selected': auto_select_models
            },
            'data_summary': {
                'total_games': len(matched_data),
                'cleaned_games': len(cleaned_data),
                'train_games': len(train_data) if 'train_data' in locals() else len(cleaned_data),
                'validation_games': len(validation_data) if validation_data is not None else 0,
                'analysis_period': f"{start_date} ~ {end_date}" if start_date and end_date else f"{days_back} days",
                'models_discovered': len(available_models),
                'models_selected': len(selected_models),
                'target_metric': target_metric,
                'split_used': optimization_result.get('split_used', False)
            },
            'segments_analysis': segments_analysis,
            'full_segments_analysis': full_segments_analysis,  # 🆕 전체 데이터 구간분석 추가
            # ✅ 수정: 조합 분석에도 동일한 분할 데이터 사용 (일관성 확보)
            'matched_data': train_data if 'train_data' in locals() and len(train_data) > 0 else cleaned_data,  # 구간분석과 동일한 데이터
            'validation_data_for_zones': validation_data_for_zones  # 검증용 데이터
        }
        
        # Train-Validation Split이 사용된 경우 추가 정보 포함
        if optimization_result.get('split_used', False):
            result['train_performance'] = optimization_result['train_performance']
            result['validation_performance'] = optimization_result['validation_performance']
            result['validation_results'] = optimization_result.get('validation_results', [])
        
        print(f"\n=== 최적화 완료 ===")
        print(f"최적 가중치: {optimization_result['optimal_weights']}")
        if optimization_result.get('split_used', False):
            print(f"분석용 성과: ROI={optimization_result['train_performance']['roi']:.2f}%, 안정성={optimization_result['train_performance']['stability_score']:.1f}")
            print(f"검증용 성과: ROI={optimization_result['validation_performance']['roi']:.2f}%, 안정성={optimization_result['validation_performance']['stability_score']:.1f}")
        print(f"전체 데이터 성과: ROI={final_performance['roi']:.2f}%, 안정성={final_performance['stability_score']:.1f}")
        
        return result
    
    def discover_available_models(self, matched_data: List[Dict]) -> List[str]:
        """매칭된 데이터에서 사용 가능한 모델들을 자동 발견"""
        
        if not matched_data:
            return []
        
        # 모든 가능한 모델 키들 (_probability 형태)
        possible_model_keys = [
            'model1_probability', 'model2_probability', 'model3_probability', 'model4_probability', 
            'model5_probability', 'model6_probability', 'model7_probability', 'model8_probability', 'model9_probability',
            'model_rf_probability', 'model_nn_probability', 'model_svm_probability',
            'model1_extended_lgbm_probability', 'model2_extended_catboost_probability', 'model3_extended_xgboost_probability',
            'model_advanced_catboost_basic_probability', 'model_advanced_catboost_probability',
            'model_advanced_lgbm_basic_probability', 'model_advanced_lgbm_probability',
            'model_advanced_nn_probability', 'model_advanced_rf_probability', 'model_advanced_svm_probability',
            'model_advanced_xgboost_basic_probability', 'model_advanced_xgboost_probability'
        ]
        
        # 각 모델의 데이터 availability 확인
        model_availability = {}
        
        for model_key in possible_model_keys:
            model_name = model_key.replace('_probability', '')
            valid_predictions = 0
            total_records = 0
            
            for record in matched_data:
                total_records += 1
                if model_key in record and record[model_key] is not None:
                    try:
                        prob_value = float(record[model_key])
                        if 0 <= prob_value <= 1:  # 유효한 확률값
                            valid_predictions += 1
                    except (ValueError, TypeError):
                        continue
            
            if total_records > 0:
                availability_rate = valid_predictions / total_records
                model_availability[model_name] = {
                    'valid_predictions': valid_predictions,
                    'total_records': total_records,
                    'availability_rate': availability_rate
                }
        
        # 충분한 데이터가 있는 모델들만 선택 (50% 이상 데이터 사용 가능)
        available_models = []
        for model_name, stats in model_availability.items():
            if stats['availability_rate'] >= 0.5 and stats['valid_predictions'] >= 10:
                available_models.append(model_name)
        
        print(f"발견된 모델들:")
        for model_name, stats in model_availability.items():
            status = "✅" if model_name in available_models else "❌"
            print(f"  {status} {model_name}: {stats['valid_predictions']}/{stats['total_records']} ({stats['availability_rate']:.1%})")
        
        return available_models
    
    def evaluate_individual_models(self, matched_data: List[Dict], available_models: List[str]) -> Dict[str, Dict]:
        """개별 모델들의 성과를 평가하여 우수한 모델들 선별"""
        
        model_performances = {}
        
        for model in available_models:
            # 개별 모델 성과 계산 (가중치 1.0으로 설정)
            weights = {model: 1.0}
            performance = self.calculate_ensemble_performance(matched_data, weights)
            
            if performance['total_games'] >= 10:  # 최소 게임 수 확인
                model_performances[model] = performance
        
        # 성과 기준으로 정렬
        sorted_models = sorted(
            model_performances.items(),
            key=lambda x: x[1]['stability_score'],  # 안정성 점수 기준
            reverse=True
        )
        
        print(f"\n=== 개별 모델 성과 평가 ===")
        for i, (model, perf) in enumerate(sorted_models[:10]):  # 상위 10개만 출력
            print(f"{i+1:2d}. {model:25s} | ROI: {perf['roi']:6.2f}% | Stability: {perf['stability_score']:5.1f} | Win Rate: {perf['win_rate']:5.1f}%")
        
        return dict(sorted_models)
    
    def select_top_models(self, model_performances: Dict[str, Dict], 
                         max_models: int = 6, 
                         min_roi: float = -15.0,
                         min_stability: float = 30.0,
                         stability_weight: float = 0.5,
                         roi_weight: float = 0.5,
                         max_drawdown: float = 75.0,
                         max_consecutive_losses: int = 10,
                         min_games: int = 10,
                         _recursion_depth: int = 0) -> List[str]:
        """우수한 모델들을 선별하여 앙상블에 포함할 모델 결정 (안정성 + ROI 복합 기준)"""
        
        # 재귀 깊이 제한 (최대 10번 시도)
        if _recursion_depth >= 10:
            print("⚠️  재귀 깊이 제한 도달. 사용 가능한 모든 모델을 사용합니다.")
            # 모든 모델을 ROI 기준으로 정렬하여 상위 모델들 선택
            all_models_sorted = sorted(
                model_performances.items(),
                key=lambda x: x[1]['roi'],
                reverse=True
            )
            selected_models = [model for model, _ in all_models_sorted[:max_models]]
            print(f"강제 선택된 모델: {len(selected_models)}개")
            for i, (model, perf) in enumerate(all_models_sorted[:max_models]):
                print(f"{i+1}. {model:25s} | ROI: {perf['roi']:6.2f}% | Stability: {perf['stability_score']:5.1f}")
            return selected_models
        
        # 🆕 기간별 Max Drawdown 기준 조정
        # 분석 기간이 길수록 Max Drawdown 기준을 완화
        total_games = sum(perf['total_games'] for perf in model_performances.values()) / len(model_performances)
        period_adjustment = 0
        
        if total_games > 200:  # 200게임 이상 (약 8일 이상)
            period_adjustment = 15  # +15% 완화
            print(f"📊 긴 분석 기간 감지 ({total_games:.0f}게임): Max Drawdown 기준 {max_drawdown}% → {max_drawdown + period_adjustment}%")
        elif total_games > 100:  # 100게임 이상 (약 4일 이상)
            period_adjustment = 10  # +10% 완화
            print(f"📊 중간 분석 기간 감지 ({total_games:.0f}게임): Max Drawdown 기준 {max_drawdown}% → {max_drawdown + period_adjustment}%")
        elif total_games > 50:   # 50게임 이상 (약 2일 이상)
            period_adjustment = 5   # +5% 완화
            print(f"📊 표준 분석 기간 감지 ({total_games:.0f}게임): Max Drawdown 기준 {max_drawdown}% → {max_drawdown + period_adjustment}%")
        
        # 필터링 기준 적용
        filtered_models = []
        
        for model, perf in model_performances.items():
            # 기본 성과 조건
            roi_ok = perf['roi'] >= min_roi
            stability_ok = perf['stability_score'] >= min_stability
            games_ok = perf['total_games'] >= min_games
            
            # 추가 안정성 조건 (재귀 깊이에 따라 완화)
            max_drawdown_threshold = max_drawdown + period_adjustment + (_recursion_depth * 10)  # 기간 조정 + 재귀 완화
            consecutive_losses_threshold = max_consecutive_losses + _recursion_depth  # 기본 10일로 완화
            
            max_drawdown_ok = perf['max_drawdown'] <= max_drawdown_threshold
            consecutive_losses_ok = perf['consecutive_losses_max'] <= consecutive_losses_threshold
            
            if roi_ok and stability_ok and games_ok and max_drawdown_ok and consecutive_losses_ok:
                filtered_models.append((model, perf))
        
        # 🔧 초기화로 오류 방지
        selected_models = []
        
        if not filtered_models:
            print("⚠️  조건을 만족하는 모델이 없습니다.")
            print(f"📊 총 {len(model_performances)}개 모델 중 0개 선별됨")
            
            # 각 기준별로 몇 개 모델이 탈락했는지 분석
            print("\n=== 필터링 기준별 모델 분석 ===")
            for criterion in ['roi', 'stability', 'games', 'max_drawdown', 'consecutive_losses']:
                passed_count = 0
                for model, perf in model_performances.items():
                    if criterion == 'roi' and perf['roi'] >= min_roi:
                        passed_count += 1
                    elif criterion == 'stability' and perf['stability_score'] >= min_stability:
                        passed_count += 1
                    elif criterion == 'games' and perf['total_games'] >= min_games:
                        passed_count += 1
                    elif criterion == 'max_drawdown' and perf['max_drawdown'] <= max_drawdown_threshold:
                        passed_count += 1
                    elif criterion == 'consecutive_losses' and perf['consecutive_losses_max'] <= consecutive_losses_threshold:
                        passed_count += 1
                
                criterion_names = {
                    'roi': f'ROI >= {min_roi}%',
                    'stability': f'Stability >= {min_stability}',
                    'games': f'Games >= {min_games}',
                    'max_drawdown': f'Max Drawdown <= {max_drawdown_threshold:.1f}%',
                    'consecutive_losses': f'Consecutive Losses <= {consecutive_losses_threshold}'
                }
                print(f"  {criterion_names[criterion]}: {passed_count}/{len(model_performances)} 모델 통과")
        else:
            # 🎯 복합 점수 계산을 위한 정규화
            rois = [perf['roi'] for _, perf in filtered_models]
            stability_scores = [perf['stability_score'] for _, perf in filtered_models]
            
            # 정규화를 위한 min-max 값
            roi_min, roi_max = min(rois), max(rois)
            stability_min, stability_max = min(stability_scores), max(stability_scores)
            
            # 복합 점수 계산 (안정성 50% + ROI 50%)
            models_with_composite_score = []
            for model, perf in filtered_models:
                # ROI 정규화 (0-100 범위로)
                if roi_max != roi_min:
                    roi_normalized = (perf['roi'] - roi_min) / (roi_max - roi_min) * 100
                else:
                    roi_normalized = 50  # 모든 ROI가 같으면 중간값
                
                # 안정성 점수는 이미 0-100 범위이므로 그대로 사용
                stability_normalized = perf['stability_score']
                
                # 복합 점수 = 안정성 50% + ROI 50%
                composite_score = (stability_normalized * stability_weight) + (roi_normalized * roi_weight)
                
                models_with_composite_score.append((model, perf, composite_score))
            
            # 복합 점수 기준으로 정렬
            top_models = sorted(models_with_composite_score, 
                               key=lambda x: x[2],  # 복합 점수 기준
                               reverse=True)[:max_models]
            
            selected_models = [model for model, _, _ in top_models]
            
            print(f"\n=== 앙상블 선별 결과 (복합 점수 기준, 시도 {_recursion_depth + 1}) ===")
            print(f"필터링 기준: ROI >= {min_roi}%, Stability >= {min_stability}, Max DD <= {max_drawdown_threshold:.1f}%, Consecutive Losses <= {consecutive_losses_threshold}")
            print(f"선별 기준: 안정성 {stability_weight*100}% + ROI {roi_weight*100}% 복합 점수")
            print(f"선별된 모델: {len(selected_models)}개 (최대 {max_models}개)")
            print(f"{'순위':<4} {'모델':<25} {'복합점수':<8} {'ROI':<8} {'안정성':<6}")
            print("-" * 55)
            
            for i, (model, perf, composite_score) in enumerate(top_models):
                print(f"{i+1:2d}.  {model:<25} {composite_score:6.1f}   {perf['roi']:6.2f}%  {perf['stability_score']:5.1f}")
        
        if len(selected_models) < 1:
            if _recursion_depth < 5:  # 5번까지만 기준 완화 시도
                print("⚠️  선별된 모델이 부족합니다. 기준을 완화하여 재시도...")
                # 기준 완화하여 재시도
                return self.select_top_models(
                    model_performances, 
                    max_models, 
                    min_roi=min_roi-5,  # 더 크게 완화
                    min_stability=max(0, min_stability-20),  # 더 크게 완화하되 0 이하로는 내리지 않음
                    stability_weight=stability_weight,
                    roi_weight=roi_weight,
                    max_drawdown=max_drawdown,
                    max_consecutive_losses=max_consecutive_losses,
                    min_games=min_games,
                    _recursion_depth=_recursion_depth + 1
                )
            else:
                print("⚠️  기준 완화 시도 횟수 초과. 최소 1개 모델을 강제 선택합니다.")
                # 강제로 상위 1개 모델 선택 (ROI 기준)
                all_models_sorted = sorted(
                    model_performances.items(),
                    key=lambda x: x[1]['roi'],
                    reverse=True
                )
                forced_selection = [model for model, _ in all_models_sorted[:max(1, max_models)]]
                print(f"강제 선택된 모델: {len(forced_selection)}개")
                for i, (model, perf) in enumerate(all_models_sorted[:max(1, max_models)]):
                    print(f"{i+1}. {model:25s} | ROI: {perf['roi']:6.2f}% | Stability: {perf['stability_score']:5.1f}")
                return forced_selection
        
        return selected_models 
    
    def analyze_ensemble_segments_with_split(self, matched_data: List[Dict], optimal_weights: Dict[str, float]) -> Dict:
        """7:3 분할하여 70% 데이터로만 구간분석 수행, 30% 데이터는 검증용으로 반환"""
        
        if not matched_data or not optimal_weights:
            return {
                'segments_analysis': {},
                'validation_data': [],
                'split_info': {
                    'total_games': 0,
                    'analysis_games': 0,
                    'validation_games': 0,
                    'split_date': ''
                }
            }
        
        print(f"\n=== 구간분석용 데이터 7:3 분할 ===")
        
        # 날짜순 정렬
        sorted_data = sorted(matched_data, key=lambda x: x.get('date', ''))
        
        # 7:3 분할 (70% 구간분석용, 30% 검증용)
        split_index = int(len(sorted_data) * 0.7)
        analysis_data = sorted_data[:split_index]
        validation_data = sorted_data[split_index:]
        
        split_date = analysis_data[-1]['date'] if analysis_data else ''
        
        print(f"📊 전체 게임: {len(sorted_data)}")
        print(f"📈 구간분석용 (70%): {len(analysis_data)} 게임 (~ {split_date})")
        print(f"🔍 검증용 (30%): {len(validation_data)} 게임 ({split_date} 이후)")
        
        # 70% 데이터로만 구간분석 수행
        segments_analysis = self.analyze_ensemble_segments(analysis_data, optimal_weights)
        
        # 분할 정보를 segments_analysis에 추가
        if segments_analysis:
            segments_analysis['split_info'] = {
                'total_games': len(sorted_data),
                'analysis_games': len(analysis_data),
                'validation_games': len(validation_data),
                'split_date': split_date
            }
        
        return {
            'segments_analysis': segments_analysis,
            'validation_data': validation_data,  # 30% 검증 데이터
            'split_info': {
                'total_games': len(sorted_data),
                'analysis_games': len(analysis_data),
                'validation_games': len(validation_data),
                'split_date': split_date
            }
        }

    def analyze_ensemble_segments(self, matched_data: List[Dict], optimal_weights: Dict[str, float]) -> Dict:
        """최적화된 앙상블의 구간별 성과 분석"""
        
        if not matched_data or not optimal_weights:
            return {}
        
        print(f"\n=== 앙상블 구간별 성과 분석 ===")
        
        # 🆕 디버깅을 위한 제외 이유 추적
        exclusion_reasons = {
            'total_input': 0,
            'no_ensemble_prob': 0,
            'no_odds': 0,
            'invalid_odds_format': 0,
            'successful': 0
        }
        
        # 앙상블 확률 및 성과 계산
        analysis_data = []
        
        for record in matched_data:
            exclusion_reasons['total_input'] += 1
            
            # 앙상블 확률 계산
            ensemble_prob = 0
            total_weight_used = 0
            
            for model, weight in optimal_weights.items():
                prob_key = f"{model}_probability"
                if prob_key in record and record[prob_key] is not None:
                    ensemble_prob += record[prob_key] * weight
                    total_weight_used += weight
            
            if total_weight_used == 0:
                exclusion_reasons['no_ensemble_prob'] += 1
                continue
            
            ensemble_prob /= total_weight_used
            
            # 베팅 정보
            actual_win = record.get('actual_home_win', 0)
            home_odds = record.get('home_odds')
            away_odds = record.get('away_odds')
            
            if home_odds is None or away_odds is None:
                exclusion_reasons['no_odds'] += 1
                continue
                
            try:
                home_odds = float(home_odds)
                away_odds = float(away_odds)
            except (ValueError, TypeError):
                exclusion_reasons['invalid_odds_format'] += 1
                continue
            
            # 성공적으로 처리된 게임
            exclusion_reasons['successful'] += 1
            
            # 예측 기반 베팅 결정 및 ROI 계산
            bet_amount = 100
            
            if ensemble_prob > 0.5:
                # 홈팀에 베팅
                predicted_team = "home"
                if actual_win == 1:
                    if home_odds > 0:
                        actual_roi = (home_odds / 100) * 100
                    else:
                        actual_roi = (100 / abs(home_odds)) * 100
                else:
                    actual_roi = -100
                
                # 예측 ROI 계산
                if home_odds > 0:
                    predicted_roi = ensemble_prob * (home_odds / 100) * 100 + (1 - ensemble_prob) * (-100)
                else:
                    predicted_roi = ensemble_prob * (100 / abs(home_odds)) * 100 + (1 - ensemble_prob) * (-100)
                    
                bet_odds = home_odds
            else:
                # 원정팀에 베팅
                predicted_team = "away"
                if actual_win == 0:
                    if away_odds > 0:
                        actual_roi = (away_odds / 100) * 100
                    else:
                        actual_roi = (100 / abs(away_odds)) * 100
                else:
                    actual_roi = -100
                
                # 예측 ROI 계산
                if away_odds > 0:
                    predicted_roi = (1 - ensemble_prob) * (away_odds / 100) * 100 + ensemble_prob * (-100)
                else:
                    predicted_roi = (1 - ensemble_prob) * (100 / abs(away_odds)) * 100 + ensemble_prob * (-100)
                    
                bet_odds = away_odds
            
            # 신뢰도 계산 (0.5에서 얼마나 멀리 있는지)
            confidence = abs(ensemble_prob - 0.5)
            
            analysis_data.append({
                'ensemble_prob': ensemble_prob,
                'predicted_roi': predicted_roi,
                'actual_roi': actual_roi,
                'confidence': confidence,
                'bet_odds': bet_odds,
                'predicted_team': predicted_team,
                'actual_win': actual_win,
                'date': record.get('date', ''),
                'home_team': record.get('home_team', ''),
                'away_team': record.get('away_team', '')
            })
        
        # 🆕 제외 이유 출력
        print(f"\n=== 구간별 분석 데이터 처리 결과 ===")
        print(f"📊 입력 게임 수: {exclusion_reasons['total_input']}")
        print(f"✅ 성공 처리: {exclusion_reasons['successful']}")
        print(f"❌ 제외된 게임:")
        print(f"  - 앙상블 확률 계산 실패: {exclusion_reasons['no_ensemble_prob']}")
        print(f"  - 배당률 누락: {exclusion_reasons['no_odds']}")
        print(f"  - 배당률 형식 오류: {exclusion_reasons['invalid_odds_format']}")
        print(f"📈 처리 성공률: {exclusion_reasons['successful'] / exclusion_reasons['total_input'] * 100:.1f}%")
        
        if not analysis_data:
            return {}
        
        # 구간별 분석 수행
        segments_analysis = {}
        
        # 1. 예측 ROI 구간별 분석
        roi_segments = self._analyze_by_predicted_roi(analysis_data)
        segments_analysis['predicted_roi'] = roi_segments
        
        # 2. 확률 구간별 분석 - 제거됨 (confidence와 동일)
        
        # 3. 배당률 구간별 분석
        odds_segments = self._analyze_by_odds(analysis_data)
        segments_analysis['odds'] = odds_segments
        
        # 4. 신뢰도 구간별 분석
        confidence_segments = self._analyze_by_confidence(analysis_data)
        segments_analysis['confidence'] = confidence_segments
        
        # 5. 배당률 vs 확률 괴리도 분석
        odds_probability_divergence = self._analyze_by_odds_probability_divergence(analysis_data)
        segments_analysis['odds_probability_divergence'] = odds_probability_divergence
        
        # 6. Kelly Criterion 분석
        kelly_criterion = self._analyze_by_kelly_criterion(analysis_data)
        segments_analysis['kelly_criterion'] = kelly_criterion
        
        # 7. 모델 합의도 분석
        model_consensus = self._analyze_by_model_consensus(analysis_data, matched_data, optimal_weights)
        segments_analysis['model_consensus'] = model_consensus
        
        # 전체 성과
        total_predicted_roi = np.mean([d['predicted_roi'] for d in analysis_data])
        total_actual_roi = np.mean([d['actual_roi'] for d in analysis_data])
        total_games = len(analysis_data)
        
        segments_analysis['overall'] = {
            'total_games': total_games,
            'predicted_roi': total_predicted_roi,
            'actual_roi': total_actual_roi,
            'roi_difference': total_actual_roi - total_predicted_roi
        }
        
        # 🆕 제외 이유도 결과에 포함
        segments_analysis['data_quality'] = exclusion_reasons
        
        return segments_analysis
    
    def _analyze_by_predicted_roi(self, data: List[Dict]) -> Dict:
        """예측 ROI 구간별 분석"""
        segments = {
            'Very Negative (<-20%)': [],
            'Negative (-20% ~ 0%)': [],
            'Positive (0% ~ 20%)': [],
            'Very Positive (>20%)': []
        }
        
        for d in data:
            pred_roi = d['predicted_roi']
            if pred_roi < -20:
                segments['Very Negative (<-20%)'].append(d)
            elif pred_roi < 0:
                segments['Negative (-20% ~ 0%)'].append(d)
            elif pred_roi < 20:
                segments['Positive (0% ~ 20%)'].append(d)
            else:
                segments['Very Positive (>20%)'].append(d)
        
        return self._calculate_segment_performance(segments)
    
    # _analyze_by_probability 제거됨 (confidence와 동일한 분석)
    
    def _analyze_by_odds(self, data: List[Dict]) -> Dict:
        """배당률 구간별 분석 (미국식 배당률: 음수=Favorite, 양수=Underdog)"""
        segments = {
            'Heavy Favorite (< -200)': [],
            'Favorite (-200 ~ -120)': [],
            'Pick Em (-120 ~ +120)': [],
            'Underdog (+120 ~ +300)': [],
            'Heavy Underdog (> +300)': []
        }
        
        for d in data:
            odds = d['bet_odds']
            if odds < -200:
                segments['Heavy Favorite (< -200)'].append(d)
            elif odds < -120:
                segments['Favorite (-200 ~ -120)'].append(d)
            elif -120 <= odds <= 120:
                segments['Pick Em (-120 ~ +120)'].append(d)
            elif odds <= 300:
                segments['Underdog (+120 ~ +300)'].append(d)
            else:
                segments['Heavy Underdog (> +300)'].append(d)
        
        return self._calculate_segment_performance(segments)
    
    def _analyze_by_confidence(self, data: List[Dict]) -> Dict:
        """신뢰도 구간별 분석"""
        segments = {
            'Low Confidence (0-0.05)': [],
            'Medium Confidence (0.05-0.15)': [],
            'High Confidence (0.15-0.25)': [],
            'Very High Confidence (>0.25)': []
        }
        
        for d in data:
            confidence = d['confidence']
            if confidence < 0.05:
                segments['Low Confidence (0-0.05)'].append(d)
            elif confidence < 0.15:
                segments['Medium Confidence (0.05-0.15)'].append(d)
            elif confidence < 0.25:
                segments['High Confidence (0.15-0.25)'].append(d)
            else:
                segments['Very High Confidence (>0.25)'].append(d)
        
        return self._calculate_segment_performance(segments)
    
    def _analyze_by_odds_probability_divergence(self, data: List[Dict]) -> Dict:
        """배당률 vs 확률 괴리도 분석 (시장 vs 모델 의견 차이)"""
        segments = {
            'Model Much More Optimistic (+10%+)': [],
            'Model Slightly Optimistic (+5% ~ +10%)': [],
            'Market Aligned (-5% ~ +5%)': [],
            'Model Slightly Pessimistic (-10% ~ -5%)': [],
            'Model Much More Pessimistic (-10%--)': []
        }
        
        for d in data:
            bet_odds = d['bet_odds']
            ensemble_prob = d['ensemble_prob']
            
            # 시장 배당률을 확률로 변환
            if bet_odds > 0:
                market_implied_prob = 100 / (bet_odds + 100)
            else:
                market_implied_prob = abs(bet_odds) / (abs(bet_odds) + 100)
            
            # 모델 확률 (베팅 팀 기준으로 조정)
            if d['predicted_team'] == 'home':
                model_prob = ensemble_prob
            else:
                model_prob = 1 - ensemble_prob
            
            # 괴리도 계산 (모델 확률 - 시장 확률)
            divergence = model_prob - market_implied_prob
            
            # 구간 분류
            if divergence >= 0.10:
                segments['Model Much More Optimistic (+10%+)'].append(d)
            elif divergence >= 0.05:
                segments['Model Slightly Optimistic (+5% ~ +10%)'].append(d)
            elif -0.05 <= divergence < 0.05:
                segments['Market Aligned (-5% ~ +5%)'].append(d)
            elif divergence >= -0.10:
                segments['Model Slightly Pessimistic (-10% ~ -5%)'].append(d)
            else:
                segments['Model Much More Pessimistic (-10%--)'].append(d)
        
        return self._calculate_segment_performance(segments)
    
    def _analyze_by_kelly_criterion(self, data: List[Dict]) -> Dict:
        """Kelly Criterion 분석 (최적 베팅 사이즈별 성과)"""
        segments = {
            'No Bet (Kelly ≤ 0%)': [],
            'Small Bet (0% < Kelly ≤ 5%)': [],
            'Medium Bet (5% < Kelly ≤ 15%)': [],
            'Large Bet (15% < Kelly ≤ 25%)': [],
            'Extreme Bet (Kelly > 25%)': []
        }
        
        for d in data:
            bet_odds = d['bet_odds']
            ensemble_prob = d['ensemble_prob']
            
            # 베팅 팀에 맞는 승률 계산
            if d['predicted_team'] == 'home':
                win_prob = ensemble_prob
            else:
                win_prob = 1 - ensemble_prob
            
            # 배당률을 decimal odds로 변환
            if bet_odds > 0:
                decimal_odds = (bet_odds / 100) + 1
            else:
                decimal_odds = (100 / abs(bet_odds)) + 1
            
            # Kelly Criterion 계산: f* = (p * b - q) / b
            # p = 승률, q = 패배율, b = decimal_odds - 1
            p = win_prob
            q = 1 - win_prob
            b = decimal_odds - 1
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            kelly_percentage = max(0, kelly_fraction) * 100  # 음수면 0으로 처리
            
            # 구간 분류
            if kelly_percentage <= 0:
                segments['No Bet (Kelly ≤ 0%)'].append(d)
            elif kelly_percentage <= 5:
                segments['Small Bet (0% < Kelly ≤ 5%)'].append(d)
            elif kelly_percentage <= 15:
                segments['Medium Bet (5% < Kelly ≤ 15%)'].append(d)
            elif kelly_percentage <= 25:
                segments['Large Bet (15% < Kelly ≤ 25%)'].append(d)
            else:
                segments['Extreme Bet (Kelly > 25%)'].append(d)
        
        return self._calculate_segment_performance(segments)
    
    def _calculate_segment_performance(self, segments: Dict[str, List]) -> Dict:
        """구간별 성과 계산"""
        results = {}
        
        for segment_name, segment_data in segments.items():
            if not segment_data:
                results[segment_name] = {
                    'games': 0,
                    'predicted_roi': 0,
                    'actual_roi': 0,
                    'roi_difference': 0,
                    'win_rate': 0,
                    'accuracy': 0
                }
                continue
            
            games = len(segment_data)
            predicted_roi = np.mean([d['predicted_roi'] for d in segment_data])
            actual_roi = np.mean([d['actual_roi'] for d in segment_data])
            roi_difference = actual_roi - predicted_roi
            
            # 승률 계산 (실제 수익이 난 경우)
            wins = sum(1 for d in segment_data if d['actual_roi'] > 0)
            win_rate = (wins / games) * 100 if games > 0 else 0
            
            # 정확도 계산 (예측한 팀이 실제로 이긴 경우)
            correct_predictions = 0
            for d in segment_data:
                if d['predicted_team'] == 'home' and d['actual_win'] == 1:
                    correct_predictions += 1
                elif d['predicted_team'] == 'away' and d['actual_win'] == 0:
                    correct_predictions += 1
            
            accuracy = (correct_predictions / games) * 100 if games > 0 else 0
            
            results[segment_name] = {
                'games': games,
                'predicted_roi': predicted_roi,
                'actual_roi': actual_roi,
                'roi_difference': roi_difference,
                'win_rate': win_rate,
                'accuracy': accuracy
            }
        
        return results 
    
    def _analyze_by_model_consensus(self, analysis_data: List[Dict], matched_data: List[Dict], optimal_weights: Dict[str, float]) -> Dict:
        """모델 합의도 분석 (개별 모델들의 의견 일치 정도별 성과)"""
        segments = {
            'Strong Consensus (80%+)': [],
            'Moderate Consensus (60-80%)': [],
            'Weak Consensus (50-60%)': [],
            'No Consensus (<50%)': []
        }
        
        # analysis_data와 matched_data를 매칭하여 개별 모델 확률 추출
        for analysis_record in analysis_data:
            # 해당하는 matched_data 레코드 찾기
            matching_record = None
            for matched_record in matched_data:
                if (analysis_record['date'] == matched_record.get('date', '') and
                    analysis_record['home_team'] == matched_record.get('home_team', '') and
                    analysis_record['away_team'] == matched_record.get('away_team', '')):
                    matching_record = matched_record
                    break
            
            if not matching_record:
                continue
            
            # 개별 모델 확률들 수집
            model_probs = []
            for model_name in optimal_weights.keys():
                prob_key = f"{model_name}_probability"
                if prob_key in matching_record and matching_record[prob_key] is not None:
                    try:
                        prob = float(matching_record[prob_key])
                        if 0 <= prob <= 1:
                            model_probs.append(prob)
                    except (ValueError, TypeError):
                        continue
            
            if len(model_probs) < 2:  # 최소 2개 모델 필요
                continue
            
            # 합의도 계산: 같은 팀을 지지하는 모델 비율
            home_supporters = sum(1 for p in model_probs if p > 0.5)
            away_supporters = len(model_probs) - home_supporters
            
            # 다수 의견의 비율
            majority_count = max(home_supporters, away_supporters)
            consensus_rate = majority_count / len(model_probs)
            
            # 구간 분류
            if consensus_rate >= 0.8:
                segments['Strong Consensus (80%+)'].append(analysis_record)
            elif consensus_rate >= 0.6:
                segments['Moderate Consensus (60-80%)'].append(analysis_record)
            elif consensus_rate >= 0.5:
                segments['Weak Consensus (50-60%)'].append(analysis_record)
            else:
                segments['No Consensus (<50%)'].append(analysis_record)
        
        return self._calculate_segment_performance(segments)