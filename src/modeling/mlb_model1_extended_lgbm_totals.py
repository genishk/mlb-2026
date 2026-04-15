"""
MLB 총점(Over/Under) 예측 — Model1 확장 (LightGBM Regressor)
123개 특성(기본+고급) 및 추가 피처 엔지니어링, 타겟은 total_score
"""
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MLBTotalsModel1ExtendedLGBM:
    """LightGBM 회귀 — 확장 특성 기반 MLB 총점 예측"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent.parent / "models" / "totals_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습/예측 공통: 원시 DataFrame에서 특성 행렬 구성 (feature_names는 설정하지 않음)"""
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
            'diff_avg_batting_avg', 'diff_avg_batting_ops', 'diff_avg_batting_homeRuns',
            'diff_avg_pitching_era', 'diff_avg_pitching_whip',
            'day_of_week', 'is_weekend', 'month',
            'early_season', 'mid_season', 'late_season',
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
            'prediction_confidence',
        ]

        all_features = base_features + advanced_features
        available_features = [f for f in all_features if f in df.columns]
        X = df[available_features].copy()

        missing_features = set(all_features) - set(available_features)
        if missing_features:
            print(
                f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: "
                f"{list(missing_features)[:10]}..."
            )

        X = X.fillna(0)
        X = self._additional_feature_engineering(X)

        team_strength_features = [
            'home_offense_power', 'away_offense_power',
            'home_defense_power', 'away_defense_power',
            'home_team_strength', 'away_team_strength',
            'diff_offense_power', 'diff_defense_power', 'diff_team_strength',
        ]
        for feat in team_strength_features:
            if feat in df.columns:
                X[feat] = df[feat]

        return X

    def _additional_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        if 'home_recent_win_rate' in X_new.columns and 'away_recent_win_rate' in X_new.columns:
            safe_away = X_new['away_recent_win_rate'] + 1e-8
            X_new['win_rate_ratio'] = X_new['home_recent_win_rate'] / safe_away
        momentum_cols = [col for col in X_new.columns if 'momentum_diff' in col]
        if len(momentum_cols) >= 3:
            weights = [0.5, 0.3, 0.2]
            X_new['weighted_momentum'] = sum(
                X_new[col] * w for col, w in zip(momentum_cols, weights)
            )
        advantage_features = ['batting_pitching_advantage', 'rank_advantage', 'power_index_diff']
        available_advantage = [f for f in advantage_features if f in X_new.columns]
        if len(available_advantage) >= 2:
            X_new['overall_advantage'] = X_new[available_advantage].mean(axis=1)
        if all(col in X_new.columns for col in ['home_momentum_3', 'home_momentum_5', 'home_momentum_7']):
            X_new['home_momentum_stability'] = X_new[
                ['home_momentum_3', 'home_momentum_5', 'home_momentum_7']
            ].std(axis=1)
            X_new['away_momentum_stability'] = X_new[
                ['away_momentum_3', 'away_momentum_5', 'away_momentum_7']
            ].std(axis=1)
            X_new['momentum_stability_diff'] = (
                X_new['home_momentum_stability'] - X_new['away_momentum_stability']
            )
        return X_new

    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
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

        X = self._transform_features(df)
        self.feature_names = X.columns.tolist()
        print(f"총 사용 특성 수: {len(self.feature_names)}")
        return X, y

    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        print("\n=== 총점 예측 모델 학습 시작 (Model1 확장 LightGBM Regressor) ===")
        X, y = self.prepare_features(training_data)
        if y is None:
            raise ValueError("타겟(total_score 또는 home_score/away_score)을 만들 수 없습니다.")

        self.feature_names = list(X.columns)
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
            verbose=-1,
        )
        self.model.fit(X, y, sample_weight=sample_weights)

        if verbose and hasattr(self.model, 'feature_importances_'):
            self._print_feature_importance()
        if verbose:
            self.evaluate_model(X, y, n_games=50)

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        if len(X) <= n_games:
            X_recent, y_recent = X, y
        else:
            X_recent, y_recent = X.iloc[-n_games:], y.iloc[-n_games:]

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

        print("\n=== 오차 범위별 적중률 ===")
        print(f"±1점 이내: {within_1:.1f}%")
        print(f"±2점 이내: {within_2:.1f}%")
        print(f"±3점 이내: {within_3:.1f}%")

        if hasattr(self, 'dates') and self.dates is not None:
            dates_recent = self.dates.iloc[-len(X_recent):]
            print("\n=== 최근 10경기 예측 상세 ===")
            print(f"{'날짜':<12} {'실제 총점':>10} {'예측 총점':>10} {'오차':>8}")
            print("-" * 45)
            indices = list(range(max(0, len(X_recent) - 10), len(X_recent)))
            for i in indices:
                date_str = dates_recent.iloc[i].strftime("%Y-%m-%d")
                actual = y_recent.iloc[i]
                pred = y_pred[i]
                err = pred - actual
                print(f"{date_str:<12} {actual:>10.0f} {pred:>10.1f} {err:>+8.1f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")

        df = pd.DataFrame(game_data)
        X = self._transform_features(df)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    def predict(self, game_data: List[Dict]) -> List[Dict]:
        print("\n=== 새 경기 총점 예측 수행 (Model1 확장 LightGBM) ===")
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        X_model = self.prepare_predictions(game_data)
        predicted_totals = self.model.predict(X_model)

        results = []
        for i, game in enumerate(game_data):
            result = game.copy()
            result['predicted_total'] = float(predicted_totals[i])
            results.append(result)

        print(f"예측 완료: {len(results)}개 경기")
        return results

    def save_model(self, timestamp: str = None) -> str:
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.model_dir / f"mlb_totals_model1_extended_lgbm_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_totals_features1_extended_lgbm_{timestamp}.json"

        joblib.dump(self.model, model_path)
        with open(feature_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'feature_names': self.feature_names,
                    'model_info': {
                        'type': 'lightgbm_regressor_extended',
                        'target': 'total_score',
                        'params': self.model.get_params(),
                    },
                },
                f,
                indent=2,
            )

        print("\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")
        return str(model_path)

    def load_model(self, model_path: str, feature_path: str = None) -> None:
        self.model = joblib.load(model_path)
        if feature_path:
            with open(feature_path, 'r', encoding='utf-8') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
        print(f"모델 로드 완료: {model_path}")

    def _print_feature_importance(self) -> None:
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return
        try:
            if hasattr(self.model, 'booster_'):
                importances = self.model.booster_.feature_importance(importance_type='gain')
            else:
                importances = self.model.feature_importances_
            importances = 100.0 * (importances / importances.sum())
            print("\n=== 상위 15개 중요 특성 (%) ===")
            feature_imp = sorted(
                zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
            )
            for feat, imp in feature_imp[:15]:
                print(f"{feat}: {imp:.2f}%")
        except Exception as e:
            print(f"특성 중요도 출력 중 오류: {e}")


def get_latest_training_data() -> List[Dict]:
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
    data = [
        g for g in data
        if g.get('home_score') is not None and g.get('away_score') is not None
    ]
    print(f"완료된 경기 수: {len(data)}")

    model = MLBTotalsModel1ExtendedLGBM()
    model.train_model(data)
    model.save_model()
