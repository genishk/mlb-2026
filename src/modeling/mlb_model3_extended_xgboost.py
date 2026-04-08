import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from datetime import datetime


class MLBBettingModel3ExtendedXGBoost:
    """
    MLB 경기 승패 예측을 위한 Model3 확장 버전 (XGBoost)
    
    mlb_model3와 동일한 기본 구조를 사용하되, 123개의 확장된 특성을 사용합니다.
    고급 기법 없이 단순한 XGBoost 분류기로 홈팀 승리 여부를 예측합니다.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent.parent / "models" / "saved_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None  # 날짜 정보 저장을 위한 변수 추가
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출 (123개 확장 특성 사용)"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']  # 날짜 정보 저장
        
        # 승패 레이블 생성 (홈팀 기준)
        if 'home_win' in df.columns:
            y = df['home_win'].astype(int)
        else:
            # 예측 데이터에는 home_win이 없을 수 있음
            y = None
        
        # 기본 특성 (mlb_model3와 동일)
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
        
        # 고급 특성 추가 (mlb_model_advanced_lgbm.py에서 가져옴)
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
        
        # 모든 특성 결합 (총 123개)
        all_features = base_features + advanced_features
        
        # 기본 특성으로 DataFrame 생성 (존재하는 컬럼만 사용)
        available_features = [feat for feat in all_features if feat in df.columns]
        X = df[available_features].copy()
        
        # 누락된 특성이 있는지 확인하고 로그 출력
        missing_features = set(all_features) - set(available_features)
        if missing_features:
            print(f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: {list(missing_features)[:10]}...")
        
        # 결측치 처리 (mlb_model3와 동일하게 단순 처리)
        X = X.fillna(0)
        
        # 추가 특성 엔지니어링
        X = self._additional_feature_engineering(X)
        
        # 팀 강도 지표 추가 (존재하는 경우)
        team_strength_features = [
            'home_offense_power', 'away_offense_power',
            'home_defense_power', 'away_defense_power',
            'home_team_strength', 'away_team_strength',
            'diff_offense_power', 'diff_defense_power', 'diff_team_strength'
        ]
        
        for feat in team_strength_features:
            if feat in df.columns:
                X[feat] = df[feat]
        
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """XGBoost 모델 학습 (mlb_model3와 동일한 파라미터)"""
        n_samples = len(X)
        # 선형 증가 가중치 (완만한 증가)
        sample_weights = np.linspace(1, 2, n_samples)
        
        # 하이퍼파라미터 설정 (XGBoost 특화, mlb_model3와 동일)
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
        
        print("\n=== Model3 확장 XGBoost 모델 학습 시작 ===")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 이름 저장 (모델 학습 후 실제 사용된 특성 집합)
        self.feature_names = list(X.columns)
        print(f"모델이 학습한 특성 수: {len(self.feature_names)}")
        
        # 특성 중요도 계산 및 정규화
        importances = self.model.feature_importances_
        importances = 100.0 * (importances / importances.sum())  # 퍼센트로 변환
        
        # 특성 중요도 시각화 저장
        self._plot_feature_importance(importances)
        
        metrics = {
            'feature_importance': dict(zip(
                self.feature_names,
                importances
            ))
        }
        
        # 상위 30개 중요 특성 출력 (특성이 많으므로 더 많이 출력)
        print("\n=== 상위 30개 중요 특성 (%) ===")
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:30]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.2f}%")
        
        return metrics
    
    def _plot_feature_importance(self, importances):
        """특성 중요도 시각화 및 저장"""
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[-20:]  # 상위 20개 특성만
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance (%)')
        plt.title('Top 20 Most Important Features (Model3 Extended XGBoost)')
        plt.tight_layout()
        
        # 저장할 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_dir = self.model_dir.parent / "plots"
        img_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(img_dir / f"feature_importance_model3_extended_xgboost_{timestamp}.png")
        plt.close()
    
    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """학습된 모델로 최근 N경기 예측 성능 평가"""
        # 최근 n_games 선택 (또는 전체 데이터를 사용 가능한 경우)
        if len(X) <= n_games:
            X_recent = X
            y_recent = y
            dates_recent = self.dates
        else:
            X_recent = X[-n_games:]
            y_recent = y[-n_games:]
            dates_recent = self.dates[-n_games:]
        
        # 예측 수행
        y_pred = self.model.predict(X_recent)
        y_pred_proba = self.model.predict_proba(X_recent)[:, 1]
        
        # 혼동 행렬 계산
        conf_matrix = confusion_matrix(y_recent, y_pred)
        
        # 추가 평가 지표 계산
        precision = precision_score(y_recent, y_pred)
        recall = recall_score(y_recent, y_pred)
        f1 = f1_score(y_recent, y_pred)
        
        # 결과 저장
        results = {
            'accuracy': accuracy_score(y_recent, y_pred),
            'roc_auc': roc_auc_score(y_recent, y_pred_proba),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': list(zip(
                dates_recent.dt.strftime('%Y-%m-%d').tolist(),
                y_recent.tolist(),
                y_pred.tolist(),
                y_pred_proba.tolist()
            ))
        }
        
        # 결과 출력
        print(f"\n=== 최근 {len(X_recent)}경기 예측 성능 ===")
        print(f"정확도: {results['accuracy']:.3f}")
        print(f"ROC-AUC: {results['roc_auc']:.3f}")
        print(f"정밀도: {results['precision']:.3f}")
        print(f"재현율: {results['recall']:.3f}")
        print(f"F1 점수: {results['f1_score']:.3f}")
        
        print("\n혼동 행렬:")
        print(f"TN: {conf_matrix[0,0]}, FP: {conf_matrix[0,1]}")
        print(f"FN: {conf_matrix[1,0]}, TP: {conf_matrix[1,1]}")
        
        # 홈팀 승리 예측 정확도
        if conf_matrix[1,0] + conf_matrix[1,1] > 0:  # 분모가 0이 아닌 경우
            home_win_accuracy = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
            print(f"홈팀 승리 예측 정확도: {home_win_accuracy:.3f}")  # 실제 홈팀 승리를 얼마나 잘 맞추는지
        
        # 원정팀 승리 예측 정확도
        if conf_matrix[0,0] + conf_matrix[0,1] > 0:  # 분모가 0이 아닌 경우
            away_win_accuracy = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print(f"원정팀 승리 예측 정확도: {away_win_accuracy:.3f}")  # 실제 원정팀 승리를 얼마나 잘 맞추는지
        
        print("\n=== 최근 10경기 예측 상세 ===")
        for date, true, pred, prob in results['predictions'][-10:]:
            print(f"날짜: {date}, 실제: {true}, 예측: {pred}, 홈팀 승리확률: {prob:.3f}")
        
        return results
    
    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        """예측을 위해 게임 데이터를 준비합니다."""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특성 추출 (prepare_features와 동일한 방식)
        X, _ = self.prepare_features(game_data)
        
        print(f"prepare_features 후 특성 수: {len(X.columns)}")
        
        # 모델이 실제로 학습한 특성 수 확인
        if hasattr(self.model, 'n_features_in_'):
            model_feature_count = self.model.n_features_in_
        elif hasattr(self.model, 'feature_importances_'):
            model_feature_count = len(self.model.feature_importances_)
        else:
            model_feature_count = len(self.feature_names) if self.feature_names else len(X.columns)
        
        print(f"모델이 학습한 특성 수: {model_feature_count}")
        
        # 정확히 모델이 학습한 특성 수만큼만 선택 (첫 N개 특성)
        feature_columns = list(X.columns)[:model_feature_count]
        X = X[feature_columns].copy()
        
        print(f"최종 모델 입력 특성 수: {len(X.columns)}")
        
        return X
    
    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """새로운 경기 데이터를 사용하여 승패 예측을 수행합니다."""
        print("\n=== Model3 확장 XGBoost 모델 예측 수행 ===")
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train_model을 호출하거나 load_model을 사용하여 저장된 모델을 로드하세요.")
        
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        
        print(f"모델 특성 수: {len(self.feature_names)}")
        
        try:
            # 예측용 데이터 준비
            X_model = self.prepare_predictions(game_data)
            
            # 예측 수행
            print("=== 예측 수행 ===")
            probabilities = self.model.predict_proba(X_model)[:, 1]  # 홈팀 승리 확률
            predictions = (probabilities > 0.5).astype(int)  # 확률 > 0.5이면 홈팀 승리 예측
            
            # 결과 구성
            results = []
            for i, game in enumerate(game_data):
                result = game.copy()
                result['home_win_probability'] = float(probabilities[i])
                result['home_win_predicted'] = int(predictions[i])
                results.append(result)
            
            print(f"Model3 확장 XGBoost 모델 예측 완료: {len(results)}개 경기")
            return results
            
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, timestamp: str = None) -> str:
        """학습된 모델과 특성 이름을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로
        model_path = self.model_dir / f"mlb_model3_extended_xgboost_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_model3_extended_xgboost_features_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'xgboost_extended',
                    'params': self.model.get_params(),
                    'feature_count': len(self.feature_names)
                }
            }, f, indent=2)
        
        print(f"\n=== Model3 확장 XGBoost 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, feature_path: str = None) -> None:
        """저장된 모델과 특성 이름 로드"""
        # 모델 로드
        self.model = joblib.load(model_path)
        
        # 특성 이름 로드 (제공된 경우)
        if feature_path:
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
        
        print(f"\n=== Model3 확장 XGBoost 모델 로드 완료 ===")
        print(f"모델 로드 경로: {model_path}")
        if feature_path:
            print(f"특성 정보 로드 경로: {feature_path}")


def get_latest_training_data() -> List[Dict]:
    """가장 최근의 학습 데이터 파일을 가져옵니다."""
    project_root = Path(__file__).parent.parent.parent
    training_dir = project_root / "data" / "training"
    
    # 디버깅용 메시지
    print("=== 경로 디버깅 ===")
    print(f"스크립트 경로: {Path(__file__).absolute()}")
    print(f"프로젝트 루트: {project_root.absolute()}")
    print(f"학습 데이터 경로: {training_dir.absolute()}")
    
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
    
    # 디버깅용 메시지
    print(f"예측 데이터 경로: {prediction_dir.absolute()}")
    print("=== 예측 데이터 파일 확인 ===")
    
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
    # 로깅 설정
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 경로 디버깅
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    training_path = project_root / "data" / "training"
    prediction_path = project_root / "data" / "prediction"
    
    print(f"\n=== 경로 디버깅 ===")
    print(f"스크립트 경로: {script_path}")
    print(f"프로젝트 루트: {project_root}")
    print(f"학습 데이터 경로: {training_path}")
    print(f"예측 데이터 경로: {prediction_path}")
    
    try:
        # 최신 학습 데이터 로드
        training_data = get_latest_training_data()
        
        # 모델 초기화 및 특성 준비
        model = MLBBettingModel3ExtendedXGBoost()
        X, y = model.prepare_features(training_data)
        
        print("\n=== 데이터 준비 완료 ===")
        print(f"특성 수: {len(model.feature_names)}")
        print(f"샘플 수: {len(X)}")
        
        # 전체 데이터로 모델 학습
        metrics = model.train_model(X, y)
        
        # 최근 50경기 성능 평가
        eval_results = model.evaluate_recent_games(X, y, n_games=50)
        
        # 모델 저장
        model_path = model.save_model()
        
        # 최신 예측 데이터 로드
        try:
            prediction_data = get_latest_prediction_data()
            
            print("\n=== 새 경기 예측 수행 ===")
            predictions = model.predict(prediction_data)
            
            # 예측 결과 출력
            print("\n=== Model3 확장 XGBoost 모델 예측 결과 ===")
            for pred in predictions:
                home_team = pred['home_team_name']
                away_team = pred['away_team_name']
                home_win_prob = pred['home_win_probability']
                is_home_win = pred['home_win_predicted'] == 1
                
                winner = home_team if is_home_win else away_team
                loser = away_team if is_home_win else home_team
                win_prob = home_win_prob if is_home_win else 1 - home_win_prob
                
                print(f"{pred['date']} - {home_team} (홈) vs {away_team} (원정)")
                print(f"  예측 승자: {winner} (승리확률: {win_prob:.3f})")
                print(f"  예측 패자: {loser}")
                print(f"  홈팀 승리확률: {home_win_prob:.3f}")
                print()
            
            # 예측 결과 저장
            save_predictions(predictions, "model3_extended_xgboost")
            
        except FileNotFoundError as e:
            print(f"\n경고: {e}")
            print("예측 데이터 파일이 없어 예측을 수행하지 않습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(traceback.format_exc()) 