import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from datetime import datetime
import matplotlib.pyplot as plt

class MLBTotalsAdvancedRandomForestModel:
    """
    MLB 경기 총 득점 예측을 위한 고급 Random Forest 모델
    
    고급 특성 엔지니어링과 특성 선택을 포함한 Random Forest + Extra Trees 앙상블을 사용하여
    경기 총 득점을 예측합니다.
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
            'home_advantage', 'away_disadvantage', 'venue_factor',
            'home_rest_days', 'away_rest_days',
            'rest_advantage', 'both_well_rested', 'both_tired',
            'diff_recent_win_rate', 'diff_recent_avg_score', 'diff_recent_avg_allowed',
            'diff_recent_avg_batting_avg', 'diff_recent_avg_batting_ops',
            'diff_recent_avg_batting_homeRuns', 'diff_overall_record_win_rate',
            'diff_home_record_win_rate', 'diff_road_record_win_rate',
            'diff_avg_batting_avg', 'diff_avg_batting_ops',
            'diff_avg_batting_homeRuns', 'diff_avg_pitching_era', 'diff_avg_pitching_whip',
            'day_of_week', 'is_weekend', 'month',
            'early_season', 'mid_season', 'late_season'
        ]
        advanced_features = [
            'home_batting_vs_away_pitching', 'away_batting_vs_home_pitching',
            'batting_pitching_advantage',
            'home_advantage_strength', 'away_disadvantage_strength',
            'venue_strength_factor',
            'home_rest_performance', 'away_rest_performance', 'rest_performance_diff',
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
        X_new = X.copy()
        if 'home_recent_win_rate' in X_new.columns and 'away_recent_win_rate' in X_new.columns:
            safe_away = X_new['away_recent_win_rate'] + 1e-8
            X_new['win_rate_ratio'] = X_new['home_recent_win_rate'] / safe_away
        momentum_cols = [col for col in X_new.columns if 'momentum_diff' in col]
        if len(momentum_cols) >= 3:
            weights = [0.5, 0.3, 0.2]
            X_new['weighted_momentum'] = sum(X_new[col] * w for col, w in zip(momentum_cols, weights))
        advantage_features = ['batting_pitching_advantage', 'rank_advantage', 'power_index_diff']
        available_advantage = [f for f in advantage_features if f in X_new.columns]
        if len(available_advantage) >= 2:
            X_new['overall_advantage'] = X_new[available_advantage].mean(axis=1)
        if all(col in X_new.columns for col in ['home_momentum_3', 'home_momentum_5', 'home_momentum_7']):
            X_new['home_momentum_stability'] = X_new[['home_momentum_3', 'home_momentum_5', 'home_momentum_7']].std(axis=1)
            X_new['away_momentum_stability'] = X_new[['away_momentum_3', 'away_momentum_5', 'away_momentum_7']].std(axis=1)
            X_new['momentum_stability_diff'] = X_new['home_momentum_stability'] - X_new['away_momentum_stability']
        return X_new
    
    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        print("\n=== 고급 Random Forest 총 득점 모델 학습 시작 ===")
        X, y = self.prepare_features(training_data)
        
        if len(X.columns) > 50:
            print("특성 선택 수행 중...")
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_features:
                print(f"상수 특성 {len(constant_features)}개 제거: {constant_features[:5]}...")
                X = X.drop(columns=constant_features)
            optimal_features = min(len(training_data) // 5, 70)
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
        sample_weights = np.linspace(0.6, 1.4, n_samples)
        sample_weights = sample_weights / sample_weights.mean()
        
        rf_model = RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            max_samples=0.85,
            random_state=42,
            n_jobs=-1,
            criterion='squared_error',
            oob_score=True
        )
        
        et_model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=15,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
            criterion='squared_error'
        )
        
        self.model = VotingRegressor(
            estimators=[
                ('rf', rf_model),
                ('et', et_model)
            ],
            n_jobs=-1
        )
        
        if verbose and len(X) > 100:
            print("교차 검증 수행 중...")
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            print(f"교차 검증 MAE 점수: {-cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        if hasattr(self.model.named_estimators_['rf'], 'oob_score_'):
            print(f"Random Forest OOB R² 점수: {self.model.named_estimators_['rf'].oob_score_:.3f}")
        
        if verbose:
            self._plot_feature_importance()
        if verbose:
            self.evaluate_model(X, y, n_games=50)
    
    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        if not self.selected_features:
            raise ValueError("선택된 특성이 없습니다. 먼저 모델을 훈련해주세요.")
        X, _ = self.prepare_features(game_data)
        if self.feature_selector is not None:
            X = X[self.selected_features]
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X
    
    def predict(self, game_data: List[Dict]) -> List[Dict]:
        print("\n=== 고급 Random Forest 총 득점 모델 예측 수행 ===")
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
            print(f"고급 Random Forest 총 득점 모델 예측 완료: {len(results)}개 경기")
            return results
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, timestamp: str = None) -> str:
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"mlb_totals_advanced_rf_{timestamp}.joblib"
        scaler_path = self.model_dir / f"mlb_totals_advanced_rf_scaler_{timestamp}.joblib"
        selector_path = self.model_dir / f"mlb_totals_advanced_rf_selector_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_totals_advanced_rf_features_{timestamp}.json"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, selector_path)
        feature_info = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'model_info': {
                'type': 'random_forest_totals_advanced',
                'rf_params': self.model.named_estimators_['rf'].get_params(),
                'et_params': self.model.named_estimators_['et'].get_params(),
                'feature_selection': self.feature_selector is not None,
                'scaling': 'robust'
            }
        }
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"\n=== 고급 Random Forest 총 득점 모델 저장 완료 ===")
        print(f"모델: {model_path}")
        print(f"스케일러: {scaler_path}")
        if self.feature_selector is not None:
            print(f"특성 선택기: {selector_path}")
        print(f"특성 정보: {feature_path}")
        return str(model_path)
    
    def load_model(self, model_path: str, scaler_path: str = None,
                   selector_path: str = None, feature_path: str = None) -> None:
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
        print(f"\n=== 고급 Random Forest 총 득점 모델 로드 완료 ===")
    
    def _plot_feature_importance(self) -> None:
        if self.model is None:
            return
        try:
            rf_importances = self.model.named_estimators_['rf'].feature_importances_
            et_importances = self.model.named_estimators_['et'].feature_importances_
            avg_importances = (rf_importances + et_importances) / 2
            avg_importances = 100.0 * (avg_importances / avg_importances.sum())
            print("\n=== 상위 25개 중요 특성 (RF + ET 평균 %) ===")
            feature_names = self.selected_features if self.selected_features else self.feature_names
            feature_importance_dict = dict(zip(feature_names, avg_importances))
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:25]
            for feature, importance in sorted_features:
                print(f"{feature}: {importance:.2f}%")
        except Exception as e:
            print(f"특성 중요도 출력 중 오류: {e}")

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        if len(X) <= n_games:
            X_recent, y_recent = X, y
        else:
            X_recent, y_recent = X[-n_games:], y[-n_games:]
        y_pred = self.model.predict(X_recent)
        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)
        errors = np.abs(y_recent.values - y_pred)
        within_1 = np.mean(errors <= 1) * 100
        within_2 = np.mean(errors <= 2) * 100
        within_3 = np.mean(errors <= 3) * 100
        print(f"\n=== 고급 Random Forest 총 득점 모델 성능 (최근 {len(X_recent)}경기) ===")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"±1점 이내: {within_1:.1f}%")
        print(f"±2점 이내: {within_2:.1f}%")
        print(f"±3점 이내: {within_3:.1f}%")
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'within_1': within_1, 'within_2': within_2, 'within_3': within_3}


def get_latest_training_data() -> List[Dict]:
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
        return json.load(f)

def get_latest_prediction_data() -> List[Dict]:
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
        return json.load(f)

def save_predictions(predictions: List[Dict], model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_dir = Path(__file__).parent.parent / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    output_path = pred_dir / f"mlb_totals_predictions_{model_name}_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"예측 결과 저장 완료: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    model = MLBTotalsAdvancedRandomForestModel()
    try:
        training_data = get_latest_training_data()
        training_data = [g for g in training_data if g.get('home_score') is not None and g.get('away_score') is not None]
        print(f"총 득점 데이터가 있는 경기 수: {len(training_data)}")
        model.train_model(training_data)
        model.save_model()
        prediction_data = get_latest_prediction_data()
        if prediction_data:
            predictions = model.predict(prediction_data)
            print("\n=== 총 득점 예측 결과 ===")
            for pred in predictions:
                print(f"{pred['date']} - {pred['home_team_name']} (홈) vs {pred['away_team_name']} (원정)")
                print(f"  예측 총 득점: {pred['predicted_total']:.1f}\n")
            save_predictions(predictions, "totals_advanced_rf")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
