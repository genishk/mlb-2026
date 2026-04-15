"""
MLB 총점(Over/Under) 예측 모델 3 - XGBoost Regressor
기존 mlb_model3.py 기반, 타겟만 total_score로 변경
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime


class MLBTotalsModel3:
    """XGBoost 회귀 모델 - MLB 총점 예측"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent.parent / "models" / "totals_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None

    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 타겟(총점) 추출"""
        df = pd.DataFrame(data)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']

        if 'total_score' in df.columns:
            y = df['total_score'].astype(float)
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            y = df['home_score'].astype(float) + df['away_score'].astype(float)
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

        available_features = [f for f in base_features if f in df.columns]
        X = df[available_features].copy()

        missing_features = set(base_features) - set(available_features)
        if missing_features:
            print(f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: {missing_features}")

        X = X.fillna(0)

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
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """XGBoost 회귀 모델 학습"""
        n_samples = len(X)
        sample_weights = np.linspace(1, 2, n_samples)

        self.model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.015,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.4,
            reg_lambda=1.8,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='mae',
            base_score=8.5,
            verbosity=0
        )

        print("\n=== 총점 예측 모델 학습 시작 (XGBoost Regressor) ===")
        self.model.fit(X, y, sample_weight=sample_weights)

        importances = self.model.feature_importances_
        importances = 100.0 * (importances / importances.sum())

        metrics = {
            'feature_importance': dict(zip(self.feature_names, importances))
        }

        print("\n=== 상위 15개 중요 특성 (%) ===")
        sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:15]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.2f}%")

        return metrics

    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """최근 N경기 총점 예측 성능 평가"""
        if len(X) <= n_games:
            X_recent, y_recent, dates_recent = X, y, self.dates
        else:
            X_recent = X[-n_games:]
            y_recent = y[-n_games:]
            dates_recent = self.dates[-n_games:]

        y_pred = self.model.predict(X_recent)

        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)

        print(f"\n=== 최근 {len(X_recent)}경기 총점 예측 성능 ===")
        print(f"MAE (평균 절대 오차): {mae:.2f}점")
        print(f"RMSE (평균 제곱근 오차): {rmse:.2f}점")
        print(f"R² (결정 계수): {r2:.3f}")

        errors = y_pred - y_recent.values
        within_1 = np.sum(np.abs(errors) <= 1) / len(errors) * 100
        within_2 = np.sum(np.abs(errors) <= 2) / len(errors) * 100
        within_3 = np.sum(np.abs(errors) <= 3) / len(errors) * 100

        print(f"\n=== 오차 범위별 적중률 ===")
        print(f"±1점 이내: {within_1:.1f}%")
        print(f"±2점 이내: {within_2:.1f}%")
        print(f"±3점 이내: {within_3:.1f}%")

        print(f"\n=== 최근 10경기 예측 상세 ===")
        print(f"{'날짜':<12} {'실제 총점':>10} {'예측 총점':>10} {'오차':>8}")
        print("-" * 45)
        for date, actual, pred in list(zip(
            dates_recent.dt.strftime('%Y-%m-%d').tolist(),
            y_recent.tolist(),
            y_pred.tolist()
        ))[-10:]:
            print(f"{date:<12} {actual:>10.0f} {pred:>10.1f} {pred - actual:>+8.1f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """새로운 경기 데이터로 총점 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")

        X_model = self.prepare_predictions(game_data)
        predicted_totals = self.model.predict(X_model)

        results = []
        for i, game in enumerate(game_data):
            result = game.copy()
            result['predicted_total'] = float(predicted_totals[i])
            results.append(result)

        print(f"총점 예측 완료: {len(results)}개 경기")
        return results

    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        """예측을 위해 게임 데이터 준비"""
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")

        df = pd.DataFrame(game_data)
        features_to_use = [f for f in self.feature_names if f in df.columns]

        missing = [f for f in features_to_use if f not in df.columns]
        if missing:
            raise ValueError(f"예측 데이터에 다음 특성이 없습니다: {missing}")

        X = df[features_to_use].copy()
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"예측 데이터에 결측치가 있는 특성: {null_cols}")

        return X

    def save_model(self, timestamp: str = None) -> str:
        """학습된 모델과 특성 이름 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.model_dir / f"mlb_totals_model3_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_totals_features3_{timestamp}.json"

        joblib.dump(self.model, model_path)

        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'xgboost_regressor',
                    'target': 'total_score',
                    'params': self.model.get_params()
                }
            }, f, indent=2)

        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")
        return str(model_path)

    def load_model(self, model_path: str, feature_path: str = None) -> None:
        """저장된 모델과 특성 이름 로드"""
        self.model = joblib.load(model_path)
        if feature_path:
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
        print(f"모델 로드 완료: {model_path}")


def get_latest_training_data() -> List[Dict]:
    """가장 최근의 학습 데이터 파일 로드"""
    project_root = Path(__file__).parent.parent.parent
    training_dir = project_root / "data" / "training"

    if not training_dir.exists():
        raise FileNotFoundError(f"학습 데이터 디렉토리가 없습니다: {training_dir}")

    json_files = list(training_dir.glob('mlb_training_data_*.json'))
    if not json_files:
        raise FileNotFoundError("학습 데이터 파일을 찾을 수 없습니다.")

    latest_file = max(json_files, key=lambda x: x.name)
    print(f"학습 데이터 파일 로드: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    data = get_latest_training_data()

    data = [g for g in data if g.get('home_score') is not None and g.get('away_score') is not None]
    print(f"완료된 경기 수: {len(data)}")

    model = MLBTotalsModel3()
    X, y = model.prepare_features(data)

    print(f"\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    print(f"타겟(총점) 평균: {y.mean():.1f}점")
    print(f"타겟(총점) 범위: {y.min():.0f} ~ {y.max():.0f}점")

    metrics = model.train_model(X, y)
    eval_results = model.evaluate_recent_games(X, y, n_games=50)
    model.save_model()
