import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from datetime import datetime
import matplotlib.pyplot as plt

class MLBTotalsAdvancedLGBMBasicModel:
    """
    MLB 경기 총 득점 예측을 위한 고급 특성 + 기본 파라미터 LightGBM 모델
    
    고급 특성 엔지니어링과 특성 선택을 포함하지만 기본 모델과 동일한 파라미터를 사용하는
    LightGBM 회귀기를 사용하여 경기 총 득점을 예측합니다.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.feature_selector = None
        self.model_dir = Path(__file__).parent.parent / "models" / "totals_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None
        self.scaler = RobustScaler()
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출 (고급 특성 포함)"""
        df = pd.DataFrame(data)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']
        
        if 'home_score' in df.columns and 'away_score' in df.columns:
            y = (df['home_score'] + df['away_score']).astype(float)
        else:
            y = None
        
        base_features = [
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            'home_avg_runs_for', 'away_avg_runs_for',
            'home_avg_runs_against', 'away_avg_runs_against',
            'home_avg_batting_avg', 'away_avg_batting_avg',
            'home_avg_batting_ops', 'away_avg_batting_ops',
            'home_avg_batting_homeRuns', 'away_avg_batting_homeRuns',
            'home_avg_pitching_era', 'away_avg_pitching_era',
            'home_avg_pitching_whip', 'away_avg_pitching_whip',
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_avg_allowed', 'away_recent_avg_allowed',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            'home_recent_avg_batting_avg', 'away_recent_avg_batting_avg',
            'home_recent_avg_batting_ops', 'away_recent_avg_batting_ops',
            'home_recent_avg_batting_homeRuns', 'away_recent_avg_batting_homeRuns',
            'home_vs_away_win_rate', 'away_vs_home_win_rate',
            'home_vs_away_avg_score', 'away_vs_home_avg_score',
            'home_vs_away_avg_allowed', 'away_vs_home_avg_allowed',
            'home_advantage', 'away_disadvantage',
            'venue_factor',
            'home_rest_days', 'away_rest_days',
            'rest_advantage', 'both_well_rested', 'both_tired',
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
            'day_of_week', 'is_weekend', 'month',
            'early_season', 'mid_season', 'late_season'
        ]
        
        advanced_features = [
            'home_batting_vs_away_pitching', 'away_batting_vs_home_pitching',
            'batting_pitching_advantage',
            'home_advantage_strength', 'away_disadvantage_strength',
            'venue_strength_factor',
            'home_rest_performance', 'away_rest_performance',
            'rest_performance_diff',
            'home_momentum_3', 'home_momentum_5', 'home_momentum_7',
            'away_momentum_3', 'away_momentum_5', 'away_momentum_7',
            'momentum_diff_3', 'momentum_diff_5', 'momentum_diff_7',
            'home_scoring_trend_3', 'home_scoring_trend_5', 'home_scoring_trend_7',
            'away_scoring_trend_3', 'away_scoring_trend_5', 'away_scoring_trend_7',
            'scoring_trend_diff_3', 'scoring_trend_diff_5', 'scoring_trend_diff_7',
            'home_league_rank', 'away_league_rank', 'rank_difference',
            'home_rank_normalized', 'away_rank_normalized', 'rank_advantage',
            'home_avg_runs_for_log', 'away_avg_runs_for_log',
            'home_recent_avg_score_log', 'away_recent_avg_score_log',
            'home_avg_batting_homeRuns_sqrt', 'away_avg_batting_homeRuns_sqrt',
            'home_recent_avg_batting_homeRuns_sqrt', 'away_recent_avg_batting_homeRuns_sqrt',
            'diff_recent_win_rate_exp', 'diff_overall_record_win_rate_exp',
            'home_power_index', 'away_power_index', 'power_index_diff',
            'prediction_confidence'
        ]
        
        all_features = base_features + advanced_features
        available_features = [feat for feat in all_features if feat in df.columns]
        X = df[available_features].copy()
        
        missing_features = set(all_features) - set(available_features)
        if missing_features:
            print(f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: {list(missing_features)[:10]}...")
        
        X = X.fillna(0)
        X = self._additional_feature_engineering(X)
        
        self.feature_names = X.columns.tolist()
        print(f"총 사용 특성 수: {len(self.feature_names)}")
        
        return X, y
    
    def _additional_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """추가 특성 엔지니어링"""
        X_new = X.copy()
        
        if 'home_recent_win_rate' in X_new.columns and 'away_recent_win_rate' in X_new.columns:
            safe_away = X_new['away_recent_win_rate'] + 1e-8
            X_new['win_rate_ratio'] = X_new['home_recent_win_rate'] / safe_away
        
        momentum_cols = [col for col in X_new.columns if 'momentum_diff' in col]
        if len(momentum_cols) >= 3:
            weights = [0.5, 0.3, 0.2]
            X_new['weighted_momentum'] = sum(
                X_new[col] * weight for col, weight in zip(momentum_cols, weights)
            )
        
        advantage_features = [
            'batting_pitching_advantage', 'rank_advantage', 'power_index_diff'
        ]
        available_advantage = [feat for feat in advantage_features if feat in X_new.columns]
        if len(available_advantage) >= 2:
            X_new['overall_advantage'] = X_new[available_advantage].mean(axis=1)
        
        if all(col in X_new.columns for col in ['home_momentum_3', 'home_momentum_5', 'home_momentum_7']):
            X_new['home_momentum_stability'] = X_new[['home_momentum_3', 'home_momentum_5', 'home_momentum_7']].std(axis=1)
            X_new['away_momentum_stability'] = X_new[['away_momentum_3', 'away_momentum_5', 'away_momentum_7']].std(axis=1)
            X_new['momentum_stability_diff'] = X_new['home_momentum_stability'] - X_new['away_momentum_stability']
        
        return X_new
    
    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        """훈련 데이터를 사용하여 고급 특성 + 기본 파라미터 LightGBM 총 득점 예측 모델을 훈련합니다."""
        print("\n=== 고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 학습 시작 ===")
        
        X, y = self.prepare_features(training_data)
        
        if len(X.columns) > 50:
            print("특성 선택 수행 중...")
            
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                print(f"상수 특성 {len(constant_features)}개 제거: {constant_features[:5]}...")
                X = X.drop(columns=constant_features)
            
            optimal_features = min(len(training_data) // 8, 60)
            k_features = min(optimal_features, len(X.columns))
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]
            
            X = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            print(f"특성 선택 완료: {len(X.columns)}개 특성 선택됨 (데이터 {len(training_data)}개 대비)")
        else:
            self.selected_features = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        n_samples = len(X)
        sample_weights = np.exp(np.linspace(0, 1, n_samples)) - 1
        sample_weights = sample_weights / sample_weights.mean()
        
        self.model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=8,
            min_child_samples=30,
            min_child_weight=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            boosting_type='gbdt',
            objective='regression',
            metric='rmse',
            verbose=-1
        )
        
        if verbose and len(X) > 100:
            print("교차 검증 수행 중...")
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                fit_params={'sample_weight': sample_weights}
            )
            print(f"교차 검증 MAE 점수: {-cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        if verbose and hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
            
        if verbose:
            self.evaluate_model(X, y, n_games=50)
    
    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        """예측을 위해 게임 데이터를 준비합니다."""
        if not self.selected_features:
            raise ValueError("선택된 특성이 없습니다. 먼저 모델을 훈련해주세요.")
        
        X, _ = self.prepare_features(game_data)
        
        if self.feature_selector is not None:
            X = X[self.selected_features]
        
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X
    
    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """새로운 경기 데이터를 사용하여 총 득점 예측을 수행합니다."""
        print("\n=== 고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 예측 수행 ===")
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_model = self.prepare_predictions(game_data)
            predicted_totals = self.model.predict(X_model)
            
            results = []
            for i, game in enumerate(game_data):
                result = game.copy()
                result['predicted_total'] = float(predicted_totals[i])
                results.append(result)
            
            print(f"고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 예측 완료: {len(results)}개 경기")
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
        
        model_path = self.model_dir / f"mlb_totals_advanced_lgbm_basic_{timestamp}.joblib"
        scaler_path = self.model_dir / f"mlb_totals_advanced_lgbm_basic_scaler_{timestamp}.joblib"
        selector_path = self.model_dir / f"mlb_totals_advanced_lgbm_basic_selector_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_totals_advanced_lgbm_basic_features_{timestamp}.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, selector_path)
        
        feature_info = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'model_info': {
                'type': 'lightgbm_totals_advanced_basic',
                'params': self.model.get_params(),
                'feature_selection': self.feature_selector is not None,
                'scaling': 'robust'
            }
        }
        
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\n=== 고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 저장 완료 ===")
        print(f"모델: {model_path}")
        print(f"스케일러: {scaler_path}")
        if self.feature_selector is not None:
            print(f"특성 선택기: {selector_path}")
        print(f"특성 정보: {feature_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, scaler_path: str = None,
                   selector_path: str = None, feature_path: str = None) -> None:
        """저장된 모델과 관련 객체들을 로드"""
        self.model = joblib.load(model_path)
        
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        
        if selector_path and Path(selector_path).exists():
            self.feature_selector = joblib.load(selector_path)
        
        if feature_path and Path(feature_path).exists():
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
                self.selected_features = feature_data.get('selected_features', [])
        
        print(f"\n=== 고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 로드 완료 ===")
        print(f"모델: {model_path}")
    
    def _plot_feature_importance(self) -> None:
        """모델의 특성 중요도를 출력"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return
            
        try:
            importances = self.model.feature_importances_
            importances = 100.0 * (importances / importances.sum())
            
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
        """모델 성능 평가 (회귀 지표)"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        if len(X) <= n_games:
            X_recent = X
            y_recent = y
        else:
            X_recent = X[-n_games:]
            y_recent = y[-n_games:]
            
        y_pred = self.model.predict(X_recent)
        
        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)
        
        errors = np.abs(y_recent.values - y_pred)
        within_1 = np.mean(errors <= 1) * 100
        within_2 = np.mean(errors <= 2) * 100
        within_3 = np.mean(errors <= 3) * 100
        
        print(f"\n=== 고급 특성 + 기본 파라미터 LightGBM 총 득점 모델 성능 (최근 {len(X_recent)}경기) ===")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"±1점 이내: {within_1:.1f}%")
        print(f"±2점 이내: {within_2:.1f}%")
        print(f"±3점 이내: {within_3:.1f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'within_1': within_1,
            'within_2': within_2,
            'within_3': within_3
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
    
    output_path = pred_dir / f"mlb_totals_predictions_{model_name}_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"예측 결과 저장 완료: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    model = MLBTotalsAdvancedLGBMBasicModel()
    
    try:
        training_data = get_latest_training_data()
        
        training_data = [
            game for game in training_data
            if game.get('home_score') is not None and game.get('away_score') is not None
        ]
        print(f"총 득점 데이터가 있는 경기 수: {len(training_data)}")
        
        model.train_model(training_data)
        model.save_model()
        
        prediction_data = get_latest_prediction_data()
        
        if prediction_data:
            predictions = model.predict(prediction_data)
            
            print("\n=== 총 득점 예측 결과 ===")
            for pred in predictions:
                home_team = pred['home_team_name']
                away_team = pred['away_team_name']
                predicted_total = pred['predicted_total']
                
                print(f"{pred['date']} - {home_team} (홈) vs {away_team} (원정)")
                print(f"  예측 총 득점: {predicted_total:.1f}")
                print()
                
            save_predictions(predictions, "totals_advanced_lgbm_basic")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
