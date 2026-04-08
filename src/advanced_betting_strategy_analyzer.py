import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelStrengths:
    """모델별 강점 분석"""
    strong_odds_ranges: List[str]
    weak_odds_ranges: List[str] 
    strong_situations: List[str]
    weak_situations: List[str]
    optimal_confidence_threshold: float
    best_team_types: List[str]
    roi_by_odds_range: Dict[str, float]
    consistency_score: float

@dataclass
class SituationalPerformance:
    """상황별 성과"""
    favorites_roi: float  # 선호팀 ROI
    underdogs_roi: float  # 언더독 ROI
    close_games_roi: float  # 박빙 게임 ROI
    blowout_predictions_roi: float  # 일방적 예상 게임 ROI
    high_total_games_roi: float  # 고득점 게임 ROI
    low_total_games_roi: float  # 저득점 게임 ROI

@dataclass
class TemporalStability:
    """시간적 안정성"""
    weekly_rois: List[float]
    stability_score: float  # 변동성의 역수
    trend_direction: str  # 'improving', 'declining', 'stable'
    recent_vs_past_performance: float

@dataclass
class AdvancedModelMetrics:
    """고급 모델 지표"""
    model_name: str
    overall_roi: float
    win_rate: float
    total_bets: int
    strengths: ModelStrengths
    situational_performance: SituationalPerformance
    temporal_stability: TemporalStability
    calibration_score: float  # 예측 확률의 정확성
    edge_detection_ability: float  # 가치 베팅 찾는 능력

@dataclass
class OptimalCombination:
    """최적 조합"""
    situation: str
    recommended_models: List[str]
    weights: Dict[str, float]
    expected_roi: float
    confidence_level: float
    reasoning: str

class AdvancedBettingStrategyAnalyzer:
    """
    고차원적 베팅 전략 분석기
    
    주요 기능:
    1. 진짜 과적합 방지 (시계열 분할 검증)
    2. 모델별 강점/약점 세부 분석
    3. 상황별 최적 조합 탐지
    4. 지능적 포트폴리오 구성
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "odds" / "data" / "matched"
        self.records_dir = self.project_root / "data" / "records"
        
        # 분석 설정
        self.odds_ranges = {
            "heavy_favorites": (-300, -151),
            "favorites": (-150, -121),
            "slight_favorites": (-120, -101),
            "pick_em": (-100, 100),
            "slight_underdogs": (101, 120),
            "underdogs": (121, 200),
            "heavy_underdogs": (201, 500)
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_historical_performance_data(self, days_back: int = 30, start_date: str = None, end_date: str = None) -> Dict:
        """히스토리컬 성과 데이터 로드 (날짜 범위 지원)"""
        # 예측 파일들 로드
        prediction_files = list(self.predictions_dir.glob("mlb_predictions_with_odds_*.json"))
        
        # 레코드 파일 로드
        record_files = list(self.records_dir.glob("mlb_historical_records_*.json"))
        
        if not record_files:
            raise ValueError("히스토리컬 레코드 파일을 찾을 수 없습니다")
        
        latest_record_file = max(record_files, key=lambda x: x.stat().st_mtime)
        with open(latest_record_file, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)
        
        # 예측 데이터 결합 (날짜 필터링 적용)
        all_predictions = {}
        
        for file_path in prediction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 파일명에서 날짜 추출
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) >= 5:
                    date_part = parts[4]  # YYYYMMDD 형식
                    
                    # 날짜 범위 필터링 (start_date, end_date가 지정된 경우)
                    if start_date and end_date:
                        # YYYYMMDD를 YYYY-MM-DD로 변환
                        try:
                            file_date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            if start_date <= file_date_str <= end_date:
                                all_predictions[date_part] = data
                        except (ValueError, IndexError):
                            continue
                    else:
                        # 날짜 범위가 지정되지 않은 경우 기존 방식 사용
                    all_predictions[date_part] = data
                        
            except Exception:
                continue
        
        return {
            'predictions': all_predictions,
            'historical_records': historical_data
        }
    
    def load_data_with_temporal_split(self, total_days: int = 365) -> Dict[str, Any]:
        """
        시계열 분할로 데이터 로드
        - 과거 70% : 분석용
        - 최근 30% : 검증용 (과적합 방지)
        """
        # 기존 방식으로 모든 데이터 로드
        data = self.load_historical_performance_data(100)  # 충분한 데이터 로드
        
        # 날짜별로 분할
        split_date = datetime.now() - timedelta(days=15)  # 최근 15일은 검증용
        
        train_predictions = {}
        validation_predictions = {}
        
        for date_str, pred_data in data['predictions'].items():
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if file_date < split_date:
                    train_predictions[date_str] = pred_data
                else:
                    validation_predictions[date_str] = pred_data
            except ValueError:
                continue
        
        return {
            'train_predictions': train_predictions,
            'validation_predictions': validation_predictions,
            'historical_records': data['historical_records'],
            'split_date': split_date.strftime('%Y-%m-%d')
        }
    
    def analyze_model_strengths_and_weaknesses(self, matched_results: List[Dict], model: str) -> ModelStrengths:
        """모델별 강점/약점 세부 분석"""
        prob_key = f'{model}_probability'
        model_results = [r for r in matched_results if r.get(prob_key, 0) > 0]
        
        if len(model_results) < 10:
            return None
        
        # 배당 구간별 성과 분석
        roi_by_odds = {}
        strong_odds_ranges = []
        weak_odds_ranges = []
        
        for range_name, (min_odds, max_odds) in self.odds_ranges.items():
            range_results = []
            for result in model_results:
                home_odds = result.get('home_odds', 0)
                away_odds = result.get('away_odds', 0)
                
                # 예측한 팀의 배당 확인
                prob = result[prob_key]
                predicted_home = prob > 0.5
                predicted_odds = home_odds if predicted_home else away_odds
                
                if predicted_odds and min_odds <= predicted_odds <= max_odds:
                    range_results.append(result)
            
            if len(range_results) >= 5:  # 최소 5경기
                roi = self._calculate_roi_for_results(range_results, model)
                roi_by_odds[range_name] = roi
                
                if roi > 3:  # 3% 이상이면 강점
                    strong_odds_ranges.append(range_name)
                elif roi < -5:  # -5% 이하면 약점
                    weak_odds_ranges.append(range_name)
        
        # 상황별 분석
        strong_situations = []
        weak_situations = []
        
        # 확신도별 분석
        optimal_threshold = self._find_optimal_confidence_threshold(model_results, model)
        
        # 일관성 점수
        consistency = self._calculate_consistency_score(model_results, model)
        
        return ModelStrengths(
            strong_odds_ranges=strong_odds_ranges,
            weak_odds_ranges=weak_odds_ranges,
            strong_situations=strong_situations,
            weak_situations=weak_situations,
            optimal_confidence_threshold=optimal_threshold,
            best_team_types=[],  # 추후 구현
            roi_by_odds_range=roi_by_odds,
            consistency_score=consistency
        )
    
    def analyze_situational_performance(self, matched_results: List[Dict], model: str) -> SituationalPerformance:
        """상황별 성과 분석"""
        prob_key = f'{model}_probability'
        model_results = [r for r in matched_results if r.get(prob_key, 0) > 0]
        
        # 선호팀 vs 언더독
        favorites_results = []
        underdogs_results = []
        
        for result in model_results:
            home_odds = result.get('home_odds', 0)
            away_odds = result.get('away_odds', 0)
            prob = result[prob_key]
            predicted_home = prob > 0.5
            
            predicted_odds = home_odds if predicted_home else away_odds
            
            if predicted_odds:
                if predicted_odds < 0:  # 선호팀
                    favorites_results.append(result)
                else:  # 언더독
                    underdogs_results.append(result)
        
        # 박빙 게임 vs 일방적 게임
        close_games = []
        blowout_games = []
        
        for result in model_results:
            prob = result[prob_key]
            confidence = abs(prob - 0.5)
            
            if confidence < 0.1:  # 확률이 0.4-0.6 사이 (박빙)
                close_games.append(result)
            elif confidence > 0.2:  # 확률이 0.3 이하 또는 0.7 이상 (일방적)
                blowout_games.append(result)
        
        return SituationalPerformance(
            favorites_roi=self._calculate_roi_for_results(favorites_results, model),
            underdogs_roi=self._calculate_roi_for_results(underdogs_results, model),
            close_games_roi=self._calculate_roi_for_results(close_games, model),
            blowout_predictions_roi=self._calculate_roi_for_results(blowout_games, model),
            high_total_games_roi=0.0,  # 추후 구현
            low_total_games_roi=0.0   # 추후 구현
        )
    
    def analyze_temporal_stability(self, matched_results: List[Dict], model: str) -> TemporalStability:
        """시간적 안정성 분석"""
        prob_key = f'{model}_probability'
        model_results = [r for r in matched_results if r.get(prob_key, 0) > 0]
        
        # 날짜별로 그룹핑
        results_by_date = {}
        for result in model_results:
            date = result['date']
            if date not in results_by_date:
                results_by_date[date] = []
            results_by_date[date].append(result)
        
        # 주별 ROI 계산
        weekly_rois = []
        dates = sorted(results_by_date.keys())
        
        for i in range(0, len(dates), 7):  # 7일씩 묶어서
            week_results = []
            for j in range(i, min(i+7, len(dates))):
                week_results.extend(results_by_date[dates[j]])
            
            if len(week_results) >= 3:  # 최소 3경기
                weekly_roi = self._calculate_roi_for_results(week_results, model)
                weekly_rois.append(weekly_roi)
        
        # 안정성 점수 (변동성의 역수)
        stability_score = 1 / (np.std(weekly_rois) + 0.01) if len(weekly_rois) > 1 else 1.0
        
        # 트렌드 분석
        if len(weekly_rois) >= 3:
            recent_avg = np.mean(weekly_rois[-2:])
            past_avg = np.mean(weekly_rois[:-2])
            
            if recent_avg > past_avg + 1:
                trend = 'improving'
            elif recent_avg < past_avg - 1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            recent_avg = past_avg = 0
        
        return TemporalStability(
            weekly_rois=weekly_rois,
            stability_score=stability_score,
            trend_direction=trend,
            recent_vs_past_performance=recent_avg - past_avg
        )
    
    def find_optimal_combinations(self, model_metrics: Dict[str, AdvancedModelMetrics]) -> List[OptimalCombination]:
        """상황별 최적 모델 조합 찾기"""
        combinations = []
        
        # 1. 선호팀 베팅에 최적인 조합
        favorite_specialists = []
        for model_name, metrics in model_metrics.items():
            if metrics.situational_performance.favorites_roi > 2:
                favorite_specialists.append((model_name, metrics.situational_performance.favorites_roi))
        
        if favorite_specialists:
            favorite_specialists.sort(key=lambda x: x[1], reverse=True)
            top_favorite_models = [x[0] for x in favorite_specialists[:3]]
            
            combinations.append(OptimalCombination(
                situation="favorites_betting",
                recommended_models=top_favorite_models,
                weights=self._calculate_performance_weights([model_metrics[m] for m in top_favorite_models]),
                expected_roi=np.mean([x[1] for x in favorite_specialists[:3]]),
                confidence_level=85.0,
                reasoning="선호팀 베팅에서 일관되게 좋은 성과를 보인 모델들의 조합"
            ))
        
        # 2. 언더독 베팅에 최적인 조합
        underdog_specialists = []
        for model_name, metrics in model_metrics.items():
            if metrics.situational_performance.underdogs_roi > 5:
                underdog_specialists.append((model_name, metrics.situational_performance.underdogs_roi))
        
        if underdog_specialists:
            underdog_specialists.sort(key=lambda x: x[1], reverse=True)
            top_underdog_models = [x[0] for x in underdog_specialists[:2]]
            
            combinations.append(OptimalCombination(
                situation="underdog_value_betting",
                recommended_models=top_underdog_models,
                weights=self._calculate_performance_weights([model_metrics[m] for m in top_underdog_models]),
                expected_roi=np.mean([x[1] for x in underdog_specialists[:2]]),
                confidence_level=75.0,
                reasoning="언더독에서 높은 가치를 찾아내는 전문 모델들"
            ))
        
        # 3. 박빙 게임 전문 조합
        close_game_specialists = []
        for model_name, metrics in model_metrics.items():
            if metrics.situational_performance.close_games_roi > 1:
                close_game_specialists.append((model_name, metrics.situational_performance.close_games_roi))
        
        if close_game_specialists:
            close_game_specialists.sort(key=lambda x: x[1], reverse=True)
            top_close_models = [x[0] for x in close_game_specialists[:3]]
            
            combinations.append(OptimalCombination(
                situation="close_games",
                recommended_models=top_close_models,
                weights=self._calculate_performance_weights([model_metrics[m] for m in top_close_models]),
                expected_roi=np.mean([x[1] for x in close_game_specialists[:3]]),
                confidence_level=70.0,
                reasoning="예측하기 어려운 박빙 게임에서 우수한 성과를 보인 모델들"
            ))
        
        # 4. 안정성 중심 조합 (위험 회피)
        stable_models = []
        for model_name, metrics in model_metrics.items():
            if metrics.temporal_stability.stability_score > 0.8 and metrics.overall_roi > 0:
                stable_models.append((model_name, metrics.temporal_stability.stability_score))
        
        if stable_models:
            stable_models.sort(key=lambda x: x[1], reverse=True)
            top_stable_models = [x[0] for x in stable_models[:3]]
            
            combinations.append(OptimalCombination(
                situation="conservative_strategy",
                recommended_models=top_stable_models,
                weights=self._calculate_performance_weights([model_metrics[m] for m in top_stable_models]),
                expected_roi=np.mean([model_metrics[m].overall_roi for m in top_stable_models]),
                confidence_level=90.0,
                reasoning="시간적으로 안정된 성과를 보이는 보수적 전략 모델들"
            ))
        
        return combinations
    
    def _calculate_roi_for_results(self, results: List[Dict], model: str) -> float:
        """특정 결과 리스트에 대한 ROI 계산"""
        if not results:
            return 0.0
        
        prob_key = f'{model}_probability'
        total_return = 0.0
        total_invested = 0.0
        
        for result in results:
            prob = result[prob_key]
            actual_home_win = result['actual_home_win']
            predicted_home_win = 1 if prob > 0.5 else 0
            is_correct = predicted_home_win == actual_home_win
            
            home_odds = result.get('home_odds')
            away_odds = result.get('away_odds')
            
            if home_odds is not None and away_odds is not None:
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
        
        return (total_return / total_invested * 100) if total_invested > 0 else 0.0
    
    def _find_optimal_confidence_threshold(self, results: List[Dict], model: str) -> float:
        """최적 확신도 임계값 찾기"""
        prob_key = f'{model}_probability'
        thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        best_threshold = 0.55
        best_roi = -100
        
        for threshold in thresholds:
            confident_results = []
            for result in results:
                prob = result[prob_key]
                confidence = max(prob, 1-prob)
                if confidence >= threshold:
                    confident_results.append(result)
            
            if len(confident_results) >= 5:
                roi = self._calculate_roi_for_results(confident_results, model)
                if roi > best_roi:
                    best_roi = roi
                    best_threshold = threshold
        
        return best_threshold
    
    def _calculate_consistency_score(self, results: List[Dict], model: str) -> float:
        """일관성 점수 계산"""
        if len(results) < 10:
            return 0.0
        
        # 10경기씩 묶어서 배치별 ROI 계산
        batch_size = 10
        batch_rois = []
        
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            if len(batch) >= 5:
                roi = self._calculate_roi_for_results(batch, model)
                batch_rois.append(roi)
        
        if len(batch_rois) < 2:
            return 0.0
        
        # 변동성의 역수로 일관성 측정
        return 1 / (np.std(batch_rois) + 0.1)
    
    def _calculate_performance_weights(self, metrics_list: List[AdvancedModelMetrics]) -> Dict[str, float]:
        """성과 기반 가중치 계산"""
        if not metrics_list:
            return {}
        
        weights = {}
        total_score = 0
        
        for metrics in metrics_list:
            # 복합 점수: ROI + 안정성 + 일관성
            score = (
                metrics.overall_roi * 0.5 +
                metrics.temporal_stability.stability_score * 0.3 +
                metrics.strengths.consistency_score * 0.2
            )
            weights[metrics.model_name] = max(score, 0.1)
            total_score += weights[metrics.model_name]
        
        # 정규화
        for model in weights:
            weights[model] /= total_score
        
        return weights
    
    def run_comprehensive_analysis(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """종합적 분석 실행 (날짜 범위 지원)"""
        if start_date and end_date:
            self.logger.info(f"🚀 고차원적 베팅 전략 분석 시작 (날짜 범위: {start_date} ~ {end_date})")
        else:
        self.logger.info("🚀 고차원적 베팅 전략 분석 시작")
        
        # 1. 날짜 범위를 적용한 데이터 로드
        historical_data = self.load_historical_performance_data(100, start_date, end_date)
        self.logger.info(f"✅ 데이터 로드 완료")
        self.logger.info(f"   예측 파일: {len(historical_data['predictions'])}개")
        self.logger.info(f"   히스토리컬 레코드: {len(historical_data['historical_records'])}개")
        
        # 2. 매칭된 결과 생성
        matched_results = self._match_predictions_with_results(
            historical_data['predictions'], 
            historical_data['historical_records']
        )
        
        self.logger.info(f"📈 매칭된 데이터: {len(matched_results)}개")
        
        # 3. 모델 발견 (기존 방식과 동일)
        discovered_models = set()
        for result in matched_results:
            for key in result.keys():
                if key.endswith('_probability') and result[key] > 0:
                    model_name = key.replace('_probability', '')
                    discovered_models.add(model_name)
        
        self.logger.info(f"📊 발견된 모델: {len(discovered_models)}개")
        if discovered_models:
            self.logger.info(f"   모델 리스트: {sorted(list(discovered_models))}")
        
        # 4. 시계열 분할 (매칭된 데이터 기반)
        split_date = datetime.now() - timedelta(days=5)  # 최근 5일만 검증용으로 변경
        train_matched = []
        validation_matched = []
        
        for result in matched_results:
            try:
                result_date = datetime.strptime(result['date'], '%Y-%m-%d')
                if result_date < split_date:
                    train_matched.append(result)
                else:
                    validation_matched.append(result)
            except ValueError:
                train_matched.append(result)  # 기본적으로 훈련 데이터에 포함
        
        # 만약 훈련 데이터가 너무 적으면 더 많이 할당
        if len(train_matched) < 50:
            split_index = max(50, int(len(matched_results) * 0.7))  # 최소 50개 또는 70%
            train_matched = matched_results[:split_index]
            validation_matched = matched_results[split_index:]
        
        self.logger.info(f"   훈련 데이터: {len(train_matched)}개")
        self.logger.info(f"   검증 데이터: {len(validation_matched)}개")
        
        model_metrics = {}
        
        # 5. 각 모델별 고급 분석
        for model in discovered_models:
            try:
                strengths = self.analyze_model_strengths_and_weaknesses(train_matched, model)
                if strengths is None:
                    continue
                
                situational = self.analyze_situational_performance(train_matched, model)
                temporal = self.analyze_temporal_stability(train_matched, model)
                
                overall_roi = self._calculate_roi_for_results(
                    [r for r in train_matched if r.get(f'{model}_probability', 0) > 0], 
                    model
                )
                
                win_rate = self._calculate_win_rate(train_matched, model)
                total_bets = len([r for r in train_matched if r.get(f'{model}_probability', 0) > 0])
                
                model_metrics[model] = AdvancedModelMetrics(
                    model_name=model,
                    overall_roi=overall_roi,
                    win_rate=win_rate,
                    total_bets=total_bets,
                    strengths=strengths,
                    situational_performance=situational,
                    temporal_stability=temporal,
                    calibration_score=0.0,  # 추후 구현
                    edge_detection_ability=0.0  # 추후 구현
                )
                
                self.logger.info(f"   ✅ {model}: ROI {overall_roi:.2f}%, 승률 {win_rate:.1f}%, {total_bets}경기")
                
            except Exception as e:
                self.logger.warning(f"   ❌ {model}: 분석 실패 - {e}")
                continue
        
        # 6. 최적 조합 탐지
        optimal_combinations = self.find_optimal_combinations(model_metrics)
        self.logger.info(f"🎯 발견된 최적 조합: {len(optimal_combinations)}개")
        
        # 7. 검증 데이터로 과적합 확인
        validation_results = self._validate_with_validation_data(
            validation_matched,
            optimal_combinations,
            model_metrics
        )
        
        return {
            'model_metrics': {k: asdict(v) for k, v in model_metrics.items()},
            'optimal_combinations': [asdict(combo) for combo in optimal_combinations],
            'validation_results': validation_results,
            'analysis_summary': {
                'total_models_analyzed': len(model_metrics),
                'profitable_models': len([m for m in model_metrics.values() if m.overall_roi > 0]),
                'combinations_found': len(optimal_combinations),
                'split_date': split_date.strftime('%Y-%m-%d'),
                'train_samples': len(train_matched),
                'validation_samples': len(validation_matched),
                'date_range_filtered': bool(start_date and end_date),
                'start_date': start_date if start_date else 'All',
                'end_date': end_date if end_date else 'All'
            }
        }
    
    def _match_predictions_with_results(self, predictions: Dict, historical_data: List) -> List[Dict]:
        """예측과 실제 결과 매칭 (기존 검증된 로직)"""
        # 히스토리컬 데이터 인덱싱
        historical_index = {}
        for record in historical_data:
            if all(key in record for key in ['date', 'home_team_name', 'away_team_name']):
                key = f"{record['date']}_{record['home_team_name']}_{record['away_team_name']}"
                historical_index[key] = record
        
        matched_results = []
        
        for date_str, pred_data in predictions.items():
            for prediction in pred_data:
                key = f"{prediction['date']}_{prediction['home_team']}_{prediction['away_team']}"
                
                if key in historical_index:
                    historical_record = historical_index[key]
                    
                    if 'home_win' in historical_record:
                        # 기본 매칭 정보
                        matched_result = {
                            'date': prediction['date'],
                            'home_team': prediction['home_team'],
                            'away_team': prediction['away_team'],
                            'predicted_winner': prediction['predicted_winner'],
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
                        
                        # 모든 *_probability 필드를 동적으로 추가 (제외 필드 제외)
                        for key, value in prediction.items():
                            if (key.endswith('_probability') and 
                                isinstance(value, (int, float)) and 
                                value > 0 and 
                                key not in excluded_probability_fields):
                                matched_result[key] = value
                        
                        matched_results.append(matched_result)
        
        return matched_results
    
    def _calculate_win_rate(self, matched_results: List[Dict], model: str) -> float:
        """승률 계산"""
        prob_key = f'{model}_probability'
        model_results = [r for r in matched_results if r.get(prob_key, 0) > 0]
        
        if not model_results:
            return 0.0
        
        wins = 0
        for result in model_results:
            prob = result[prob_key]
            actual_home_win = result['actual_home_win']
            predicted_home_win = 1 if prob > 0.5 else 0
            if predicted_home_win == actual_home_win:
                wins += 1
        
        return (wins / len(model_results)) * 100
    
    def _validate_with_validation_data(self, validation_matched: List[Dict], 
                                     optimal_combinations: List[OptimalCombination], 
                                     model_metrics: Dict[str, AdvancedModelMetrics]) -> Dict[str, Any]:
        """검증 데이터로 전략 검증 (과적합 방지)"""
        validation_results = {}
        
        for combo in optimal_combinations:
            situation = combo.situation
            models = combo.recommended_models
            weights = combo.weights
            
            # 각 모델별 검증 성과
            model_validation_rois = {}
            for model in models:
                if model in model_metrics:
                    val_roi = self._calculate_roi_for_results(
                        [r for r in validation_matched if r.get(f'{model}_probability', 0) > 0],
                        model
                    )
                    model_validation_rois[model] = val_roi
            
            # 가중 평균 검증 ROI
            if model_validation_rois:
                weighted_validation_roi = sum(
                    model_validation_rois.get(model, 0) * weight 
                    for model, weight in weights.items()
                )
            else:
                weighted_validation_roi = 0.0
            
            validation_results[situation] = {
                'expected_roi': combo.expected_roi,
                'validation_roi': weighted_validation_roi,
                'overfitting_risk': combo.expected_roi - weighted_validation_roi,
                'model_validation_rois': model_validation_rois,
                'reliable': abs(combo.expected_roi - weighted_validation_roi) < 3  # 3% 이내면 신뢰
            }
        
        return validation_results

def main():
    """메인 실행 함수"""
    analyzer = AdvancedBettingStrategyAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis()
        
        print("\n" + "="*80)
        print("🎯 고차원적 베팅 전략 분석 결과")
        print("="*80)
        
        summary = results['analysis_summary']
        print(f"\n📊 분석 요약:")
        print(f"   • 분석된 모델 수: {summary['total_models_analyzed']}개")
        print(f"   • 수익 모델 수: {summary['profitable_models']}개")
        print(f"   • 발견된 최적 조합: {summary['combinations_found']}개")
        print(f"   • 시계열 분할 기준: {summary['split_date']}")
        
        print(f"\n🏆 최적 전략 조합:")
        for combo in results['optimal_combinations']:
            print(f"\n   📈 {combo['situation'].replace('_', ' ').title()}")
            print(f"      모델: {', '.join(combo['recommended_models'])}")
            print(f"      예상 ROI: {combo['expected_roi']:.2f}%")
            print(f"      신뢰도: {combo['confidence_level']:.1f}%")
            print(f"      이유: {combo['reasoning']}")
        
        print(f"\n🔍 과적합 검증 결과:")
        for situation, validation in results['validation_results'].items():
            status = "✅ 신뢰" if validation['reliable'] else "⚠️ 과적합 위험"
            print(f"   {situation}: {status}")
            print(f"      훈련 ROI: {validation['expected_roi']:.2f}% → 검증 ROI: {validation['validation_roi']:.2f}%")
        
        # 결과 저장
        results_dir = Path("data/advanced_analysis")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = results_dir / f"advanced_strategy_analysis_{timestamp}.json"
        
        # numpy 타입을 Python 네이티브 타입으로 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        results_converted = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 상세 결과 저장: {output_file}")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 