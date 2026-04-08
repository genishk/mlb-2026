import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from datetime import datetime
import matplotlib.pyplot as plt

class MLBAdvancedXGBoostBasicModel:
    """
    MLB 경기 승패 예측을 위한 고급 특성 + 기본 파라미터 XGBoost 모델
    
    고급 특성 엔지니어링과 특성 선택을 포함하지만 기본 모델과 동일한 파라미터를 사용하는
    XGBoost 분류기를 사용하여 홈팀 승리 여부를 예측합니다.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.feature_selector = None
        self.model_dir = Path(__file__).parent.parent / "models" / "saved_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None
        self.scaler = RobustScaler()  # 이상치에 강한 스케일러 사용
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출 (고급 특성 포함)"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']
        
        # 승패 레이블 생성 (홈팀 기준)
        if 'home_win' in df.columns:
            y = df['home_win'].astype(int)
        else:
            y = None
        
        # 기본 특성 (기존 모델과 동일)
        base_features = [
            # 팀 승률
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            
            # 기본 타/투 지표
            'home_avg_runs_for', 'away_avg_runs_for',
            'home_avg_runs_against', 'away_avg_runs_against',
            'home_avg_batting_avg', 'away_avg_batting_avg',
            'home_avg_batting_ops', 'away_avg_batting_ops',
            'home_avg_batting_homeRuns', 'away_avg_batting_homeRuns',
            'home_avg_pitching_era', 'away_avg_pitching_era',
            'home_avg_pitching_whip', 'away_avg_pitching_whip',
            
            # 최근 트렌드
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_avg_allowed', 'away_recent_avg_allowed',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            'home_recent_avg_batting_avg', 'away_recent_avg_batting_avg',
            'home_recent_avg_batting_ops', 'away_recent_avg_batting_ops',
            'home_recent_avg_batting_homeRuns', 'away_recent_avg_batting_homeRuns',
            
            # 상대전적
            'home_vs_away_win_rate', 'away_vs_home_win_rate',
            'home_vs_away_avg_score', 'away_vs_home_avg_score',
            'home_vs_away_avg_allowed', 'away_vs_home_avg_allowed',
            
            # 홈/원정 어드밴티지
            'home_advantage', 'away_disadvantage',
            'venue_factor',
            
            # 컨디션
            'home_rest_days', 'away_rest_days',
            'rest_advantage', 'both_well_rested', 'both_tired',
            
            # 팀 간 차이 지표
            'diff_recent_win_rate',
            'diff_recent_avg_score',
            'diff_recent_avg_allowed',
            'diff_recent_avg_batting_avg',
            'diff_recent_avg_batting_ops',
            'diff_recent_avg_batting_homeRuns',
            'diff_overall_record_win_rate',
            'diff_home_record_win_rate',
            'diff_road_record_win_rate',
            'diff_avg_batting_avg',
            'diff_avg_batting_ops',
            'diff_avg_batting_homeRuns',
            'diff_avg_pitching_era',
            'diff_avg_pitching_whip',
            
            # 시즌 시점
            'day_of_week', 'is_weekend', 'month',
            'early_season', 'mid_season', 'late_season'
        ]
        
        # 고급 특성 추가
        advanced_features = [
            # 특성 상호작용
            'home_batting_vs_away_pitching', 'away_batting_vs_home_pitching',
            'batting_pitching_advantage',
            'home_advantage_strength', 'away_disadvantage_strength',
            'venue_strength_factor',
            'home_rest_performance', 'away_rest_performance',
            'rest_performance_diff',
            
            # 모멘텀 지표
            'home_momentum_3', 'home_momentum_5', 'home_momentum_7',
            'away_momentum_3', 'away_momentum_5', 'away_momentum_7',
            'momentum_diff_3', 'momentum_diff_5', 'momentum_diff_7',
            'home_scoring_trend_3', 'home_scoring_trend_5', 'home_scoring_trend_7',
            'away_scoring_trend_3', 'away_scoring_trend_5', 'away_scoring_trend_7',
            'scoring_trend_diff_3', 'scoring_trend_diff_5', 'scoring_trend_diff_7',
            
            # 상대적 순위
            'home_league_rank', 'away_league_rank', 'rank_difference',
            'home_rank_normalized', 'away_rank_normalized', 'rank_advantage',
            
            # 비선형 변환
            'home_avg_runs_for_log', 'away_avg_runs_for_log',
            'home_recent_avg_score_log', 'away_recent_avg_score_log',
            'home_avg_batting_homeRuns_sqrt', 'away_avg_batting_homeRuns_sqrt',
            'home_recent_avg_batting_homeRuns_sqrt', 'away_recent_avg_batting_homeRuns_sqrt',
            'diff_recent_win_rate_exp', 'diff_overall_record_win_rate_exp',
            
            # 복합 지표
            'home_power_index', 'away_power_index', 'power_index_diff',
            'prediction_confidence'
        ]
        
        # 모든 특성 결합
        all_features = base_features + advanced_features
        
        # 존재하는 특성만 선택
        available_features = [feat for feat in all_features if feat in df.columns]
        X = df[available_features].copy()
        
        # 누락된 특성 확인
        missing_features = set(all_features) - set(available_features)
        if missing_features:
            print(f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: {list(missing_features)[:10]}...")
        
        # 결측치 처리
        X = X.fillna(0)
        
        # 추가 특성 엔지니어링
        X = self._additional_feature_engineering(X)
        
        self.feature_names = X.columns.tolist()
        print(f"총 사용 특성 수: {len(self.feature_names)}")
        
        return X, y
    
    def _additional_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """추가 특성 엔지니어링"""
        X_new = X.copy()
        
        # 1. 특성 간 비율 계산
        if 'home_recent_win_rate' in X_new.columns and 'away_recent_win_rate' in X_new.columns:
            # 안전한 나눗셈을 위한 작은 값 추가
            safe_away = X_new['away_recent_win_rate'] + 1e-8
            X_new['win_rate_ratio'] = X_new['home_recent_win_rate'] / safe_away
        
        # 2. 모멘텀 가중 평균
        momentum_cols = [col for col in X_new.columns if 'momentum_diff' in col]
        if len(momentum_cols) >= 3:
            # 최근 모멘텀에 더 높은 가중치
            weights = [0.5, 0.3, 0.2]  # 3경기, 5경기, 7경기 순
            X_new['weighted_momentum'] = sum(
                X_new[col] * weight for col, weight in zip(momentum_cols, weights)
            )
        
        # 3. 종합 우위 지수
        advantage_features = [
            'batting_pitching_advantage', 'rank_advantage', 'power_index_diff'
        ]
        available_advantage = [feat for feat in advantage_features if feat in X_new.columns]
        if len(available_advantage) >= 2:
            X_new['overall_advantage'] = X_new[available_advantage].mean(axis=1)
        
        # 4. 변동성 지표 (최근 성적의 일관성)
        if all(col in X_new.columns for col in ['home_momentum_3', 'home_momentum_5', 'home_momentum_7']):
            X_new['home_momentum_stability'] = X_new[['home_momentum_3', 'home_momentum_5', 'home_momentum_7']].std(axis=1)
            X_new['away_momentum_stability'] = X_new[['away_momentum_3', 'away_momentum_5', 'away_momentum_7']].std(axis=1)
            X_new['momentum_stability_diff'] = X_new['home_momentum_stability'] - X_new['away_momentum_stability']
        
        return X_new
    
    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        """
        훈련 데이터를 사용하여 고급 특성 + 기본 파라미터 XGBoost 모델을 훈련합니다.
        
        Args:
            training_data: 훈련 데이터 리스트
            verbose: 상세 출력 여부
        """
        print("\n=== 고급 특성 + 기본 파라미터 XGBoost 모델 학습 시작 ===")
        
        # 특성과 레이블 추출
        X, y = self.prepare_features(training_data)
        
        # 특성 선택 (상위 K개 특성 선택)
        if len(X.columns) > 50:  # 특성이 50개 이상인 경우 선택 적용
            print("특성 선택 수행 중...")
            
            # 상수 특성 제거 (워닝 방지)
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:  # 고유값이 1개 이하인 특성
                    constant_features.append(col)
            
            if constant_features:
                print(f"상수 특성 {len(constant_features)}개 제거: {constant_features[:5]}...")
                X = X.drop(columns=constant_features)
            
            # 데이터 수 대비 적절한 특성 수 계산 (기본 모델과 유사하게)
            optimal_features = min(len(training_data) // 8, 60)  # XGBoost 최적 특성 수
            k_features = min(optimal_features, len(X.columns))
            
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # 선택된 특성 이름 저장
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]
            
            # DataFrame으로 변환
            X = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            print(f"특성 선택 완료: {len(X.columns)}개 특성 선택됨 (데이터 {len(training_data)}개 대비)")
        else:
            self.selected_features = X.columns.tolist()
        
        # 특성 스케일링 (XGBoost는 스케일링이 덜 중요하지만 일관성을 위해 적용)
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 시간 가중치 (mlb_model3.py와 동일)
        n_samples = len(X)
        sample_weights = np.linspace(1, 2, n_samples)
        
        # mlb_model3.py와 동일한 XGBoost 파라미터 사용
        self.model = XGBClassifier(
            n_estimators=350,             # 트리 개수 감소 (400 -> 350)
            learning_rate=0.015,          # 학습률 감소 (0.02 -> 0.015)
            max_depth=5,                  # 트리 깊이 유지
            min_child_weight=2,           # 과적합 방지 유지
            subsample=0.8,                # 데이터 샘플링 감소 (0.85 -> 0.8)
            colsample_bytree=0.8,         # 특성 샘플링 감소 (0.85 -> 0.8)
            reg_alpha=0.4,                # L1 규제 증가 (0.3 -> 0.4)
            reg_lambda=1.8,               # L2 규제 증가 (1.5 -> 1.8)
            random_state=42,
            objective='binary:logistic',  # 이진 분류
            eval_metric='auc',            # AUC 측정 지표
            use_label_encoder=False,
            verbosity=0
        )
        
        # 교차 검증으로 모델 성능 평가 (시계열 데이터 고려)
        if verbose and len(X) > 100:  # 충분한 데이터가 있을 때만
            print("교차 검증 수행 중...")
            
            # 시계열 교차 검증 (미래 데이터 리키지 방지)
            tscv = TimeSeriesSplit(n_splits=3)
            # fit_params 제거하여 호환성 문제 해결
            cv_scores = cross_val_score(
                self.model, X, y, 
                cv=tscv, 
                scoring='roc_auc'
            )
            print(f"교차 검증 AUC 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # 최종 모델 훈련
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 중요도 출력
        if verbose and hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
            
        # 검증 성능 평가
        if verbose:
            self.evaluate_model(X, y, n_games=50)
    
    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        """예측을 위해 게임 데이터를 준비합니다."""
        if not self.selected_features:
            raise ValueError("선택된 특성이 없습니다. 먼저 모델을 훈련해주세요.")
        
        # 특성 추출
        X, _ = self.prepare_features(game_data)
        
        # 특성 선택 적용
        if self.feature_selector is not None:
            # 선택된 특성만 사용
            X = X[self.selected_features]
        
        # 스케일링 적용
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X
    
    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """새로운 경기 데이터를 사용하여 승패 예측을 수행합니다."""
        print("\n=== 고급 특성 + 기본 파라미터 XGBoost 모델 예측 수행 ===")
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            # 예측용 데이터 준비
            X_model = self.prepare_predictions(game_data)
            
            # 예측 수행
            probabilities = self.model.predict_proba(X_model)[:, 1]
            predictions = (probabilities > 0.5).astype(int)
            
            # 결과 구성
            results = []
            for i, game in enumerate(game_data):
                result = game.copy()
                result['home_win_probability'] = float(probabilities[i])
                result['home_win_predicted'] = int(predictions[i])
                result['model_confidence'] = float(abs(probabilities[i] - 0.5) * 2)  # 0-1 스케일 신뢰도
                results.append(result)
            
            print(f"고급 특성 + 기본 파라미터 XGBoost 모델 예측 완료: {len(results)}개 경기")
            return results
            
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, timestamp: str = None) -> str:
        """학습된 모델과 관련 객체들을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 파일 경로들
        model_path = self.model_dir / f"mlb_advanced_xgboost_basic_{timestamp}.joblib"
        scaler_path = self.model_dir / f"mlb_advanced_xgboost_basic_scaler_{timestamp}.joblib"
        selector_path = self.model_dir / f"mlb_advanced_xgboost_basic_selector_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_advanced_xgboost_basic_features_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, selector_path)
        
        # 특성 정보 저장
        feature_info = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'model_info': {
                'type': 'xgboost_advanced_basic',
                'params': self.model.get_params(),
                'feature_selection': self.feature_selector is not None,
                'scaling': 'robust'
            }
        }
        
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\n=== 고급 특성 + 기본 파라미터 XGBoost 모델 저장 완료 ===")
        print(f"모델: {model_path}")
        print(f"스케일러: {scaler_path}")
        if self.feature_selector is not None:
            print(f"특성 선택기: {selector_path}")
        print(f"특성 정보: {feature_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, scaler_path: str = None, 
                   selector_path: str = None, feature_path: str = None) -> None:
        """저장된 모델과 관련 객체들을 로드"""
        # 모델 로드
        self.model = joblib.load(model_path)
        
        # 스케일러 로드
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        
        # 특성 선택기 로드
        if selector_path and Path(selector_path).exists():
            self.feature_selector = joblib.load(selector_path)
        
        # 특성 정보 로드
        if feature_path and Path(feature_path).exists():
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
                self.selected_features = feature_data.get('selected_features', [])
        
        print(f"\n=== 고급 특성 + 기본 파라미터 XGBoost 모델 로드 완료 ===")
        print(f"모델: {model_path}")
    
    def _plot_feature_importance(self) -> None:
        """모델의 특성 중요도를 출력"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return
            
        try:
            importances = self.model.feature_importances_
            importances = 100.0 * (importances / importances.sum())
            
            # 상위 25개 중요 특성 출력
            print("\n=== 상위 25개 중요 특성 (%) ===")
            feature_names = self.selected_features if self.selected_features else self.feature_names
            feature_importance = dict(zip(feature_names, importances))
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:25]
            
            for feature, importance in sorted_features:
                print(f"{feature}: {importance:.2f}%")
                
        except Exception as e:
            print(f"특성 중요도 출력 중 오류: {e}")

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """모델 성능 평가"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        # 최근 n_games 선택
        if len(X) <= n_games:
            X_recent = X
            y_recent = y
        else:
            X_recent = X[-n_games:]
            y_recent = y[-n_games:]
            
        # 예측
        y_pred = self.model.predict(X_recent)
        y_prob = self.model.predict_proba(X_recent)[:, 1]
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_recent, y_pred)
        try:
            roc_auc = roc_auc_score(y_recent, y_prob)
        except:
            roc_auc = 0.5
            
        # 혼동 행렬
        cm = confusion_matrix(y_recent, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 정밀도, 재현율, F1 점수
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 결과 출력
        print(f"\n=== 고급 특성 + 기본 파라미터 XGBoost 모델 성능 (최근 {len(X_recent)}경기) ===")
        print(f"정확도: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"정밀도: {precision:.3f}")
        print(f"재현율: {recall:.3f}")
        print(f"F1 점수: {f1:.3f}")
        
        print("\n혼동 행렬:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }


def get_latest_training_data() -> List[Dict]:
    """가장 최근의 학습 데이터 파일을 가져옵니다."""
    project_root = Path(__file__).parent.parent.parent
    training_dir = project_root / "data" / "training"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"학습 데이터 디렉토리가 존재하지 않습니다: {training_dir.absolute()}")
    
    json_files = list(training_dir.glob('mlb_training_data_*.json'))
    
    if not json_files:
        raise FileNotFoundError("학습 데이터 파일을 찾을 수 없습니다.")
    
    # 파일명으로 정렬하여 가장 최근 파일 선택
    latest_file = max(json_files, key=lambda x: x.name)
    print(f"선택된 학습 데이터 파일: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    return training_data

def get_latest_prediction_data() -> List[Dict]:
    """가장 최근의 예측 데이터 파일을 가져옵니다."""
    project_root = Path(__file__).parent.parent.parent
    prediction_dir = project_root / "data" / "prediction"
    
    if not prediction_dir.exists():
        raise FileNotFoundError(f"예측 데이터 디렉토리가 존재하지 않습니다: {prediction_dir.absolute()}")
    
    json_files = list(prediction_dir.glob('mlb_prediction_data_*.json'))
    
    if not json_files:
        print("예측 파일이 없습니다.")
        return []
    
    # 파일명으로 정렬하여 가장 최근 파일 선택
    latest_file = max(json_files, key=lambda x: x.name)
    print(f"선택된 예측 데이터 파일: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
    
    return prediction_data

def save_predictions(predictions: List[Dict], model_name: str) -> str:
    """예측 결과를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_dir = Path(__file__).parent.parent / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = pred_dir / f"mlb_game_predictions_{model_name}_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"예측 결과 저장 완료: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    # MLB 고급 특성 + 기본 파라미터 XGBoost 모델 초기화
    model = MLBAdvancedXGBoostBasicModel()
    
    try:
        # 최신 학습 데이터 로드
        training_data = get_latest_training_data()
        
        # 고급 모델 학습
        model.train_model(training_data)
        
        # 모델 저장
        model.save_model()
        
        # 예측 데이터가 있으면 예측 수행
        prediction_data = get_latest_prediction_data()
        
        if prediction_data:
            # 새 경기 예측
            predictions = model.predict(prediction_data)
            
            # 예측 결과 출력
            print("\n=== 예측 결과 ===")
            for pred in predictions:
                home_team = pred['home_team_name']
                away_team = pred['away_team_name']
                home_win_prob = pred['home_win_probability']
                is_home_win = pred['home_win_predicted'] == 1
                confidence = pred['model_confidence']
                
                winner = home_team if is_home_win else away_team
                loser = away_team if is_home_win else home_team
                win_prob = home_win_prob if is_home_win else 1 - home_win_prob
                
                print(f"{pred['date']} - {home_team} (홈) vs {away_team} (원정)")
                print(f"  예측 승자: {winner} (승리확률: {win_prob:.3f})")
                print(f"  예측 패자: {loser}")
                print(f"  홈팀 승리확률: {home_win_prob:.3f}")
                print(f"  모델 신뢰도: {confidence:.3f}")
                print()
                
            # 예측 결과 저장
            save_predictions(predictions, "advanced_xgboost_basic")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc() 