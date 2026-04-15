"""
MLB 총점(Over/Under) 예측 모델 - SVR + StandardScaler
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class MLBTotalsModelSVM:
    """SVR + StandardScaler - MLB 총점 예측"""

    def __init__(self) -> None:
        self.model: Optional[SVR] = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.model_dir = Path(__file__).parent.parent / "models" / "totals_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates: Optional[pd.Series] = None

    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        self.dates = df["date"]

        if "total_score" in df.columns:
            y = df["total_score"].astype(float)
        elif "home_score" in df.columns and "away_score" in df.columns:
            y = df["home_score"].astype(float) + df["away_score"].astype(float)
        else:
            y = pd.Series(dtype=float)

        base_features = [
            "home_overall_record_win_rate",
            "away_overall_record_win_rate",
            "home_home_record_win_rate",
            "away_home_record_win_rate",
            "home_road_record_win_rate",
            "away_road_record_win_rate",
            "home_avg_runs_for",
            "away_avg_runs_for",
            "home_avg_runs_against",
            "away_avg_runs_against",
            "home_avg_batting_avg",
            "away_avg_batting_avg",
            "home_avg_batting_ops",
            "away_avg_batting_ops",
            "home_avg_batting_homeRuns",
            "away_avg_batting_homeRuns",
            "home_avg_pitching_era",
            "away_avg_pitching_era",
            "home_avg_pitching_whip",
            "away_avg_pitching_whip",
            "home_recent_win_rate",
            "away_recent_win_rate",
            "home_recent_avg_score",
            "away_recent_avg_score",
            "home_recent_avg_allowed",
            "away_recent_avg_allowed",
            "home_recent_home_win_rate",
            "away_recent_home_win_rate",
            "home_recent_away_win_rate",
            "away_recent_away_win_rate",
            "home_recent_avg_batting_avg",
            "away_recent_avg_batting_avg",
            "home_recent_avg_batting_ops",
            "away_recent_avg_batting_ops",
            "home_recent_avg_batting_homeRuns",
            "away_recent_avg_batting_homeRuns",
            "home_vs_away_win_rate",
            "away_vs_home_win_rate",
            "home_vs_away_avg_score",
            "away_vs_home_avg_score",
            "home_vs_away_avg_allowed",
            "away_vs_home_avg_allowed",
            "home_advantage",
            "away_disadvantage",
            "venue_factor",
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "both_well_rested",
            "both_tired",
            "diff_recent_win_rate",
            "diff_recent_avg_score",
            "diff_recent_avg_allowed",
            "diff_recent_avg_batting_avg",
            "diff_recent_avg_batting_ops",
            "diff_recent_avg_batting_homeRuns",
            "diff_overall_record_win_rate",
            "diff_home_record_win_rate",
            "diff_road_record_win_rate",
            "diff_avg_batting_avg",
            "diff_avg_batting_ops",
            "diff_avg_batting_homeRuns",
            "diff_avg_pitching_era",
            "diff_avg_pitching_whip",
            "day_of_week",
            "is_weekend",
            "month",
            "early_season",
            "mid_season",
            "late_season",
        ]

        available_features = [f for f in base_features if f in df.columns]
        X = df[available_features].copy()
        missing = set(base_features) - set(available_features)
        if missing:
            print(f"경고: {len(missing)}개의 특성이 데이터에 없습니다: {missing}")
        X = X.fillna(0)

        team_strength_features = [
            "home_offense_power",
            "away_offense_power",
            "home_defense_power",
            "away_defense_power",
            "home_team_strength",
            "away_team_strength",
            "diff_offense_power",
            "diff_defense_power",
            "diff_team_strength",
        ]
        for feat in team_strength_features:
            if feat in df.columns:
                X[feat] = df[feat]

        self.feature_names = X.columns.tolist()
        return X, y

    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        print("\n=== 총점 예측 모델 학습 시작 (SVR + StandardScaler) ===")
        X, y = self.prepare_features(training_data)
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)

        self.model = SVR(
            C=3.5,
            kernel="rbf",
            gamma=0.02,
            epsilon=0.1,
            shrinking=True,
            tol=5e-3,
            cache_size=300,
            verbose=False,
            max_iter=1500,
        )
        self.model.fit(X_scaled, y)

        if verbose:
            self.evaluate_model(X, y, n_games=50)

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        if len(X) <= n_games:
            X_recent, y_recent = X, y
        else:
            X_recent, y_recent = X.iloc[-n_games:], y.iloc[-n_games:]

        X_recent_scaled = self.scaler.transform(X_recent)
        y_pred = self.model.predict(X_recent_scaled)

        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)

        print(f"\n=== 최근 {len(X_recent)}경기 총점 예측 성능 ===")
        print(f"MAE (평균 절대 오차): {mae:.2f}점")
        print(f"RMSE (평균 제곱근 오차): {rmse:.2f}점")
        print(f"R² (결정 계수): {r2:.3f}")

        errors = y_pred - y_recent.values
        print(f"\n=== 오차 범위별 적중률 ===")
        print(f"±1점 이내: {np.sum(np.abs(errors) <= 1) / len(errors) * 100:.1f}%")
        print(f"±2점 이내: {np.sum(np.abs(errors) <= 2) / len(errors) * 100:.1f}%")
        print(f"±3점 이내: {np.sum(np.abs(errors) <= 3) / len(errors) * 100:.1f}%")

        if self.dates is not None:
            dates_recent = self.dates[-len(X_recent) :]
            print(f"\n=== 최근 10경기 예측 상세 ===")
            print(f"{'날짜':<12} {'실제 총점':>10} {'예측 총점':>10} {'오차':>8}")
            print("-" * 45)
            for i in range(max(0, len(X_recent) - 10), len(X_recent)):
                print(
                    f"{dates_recent.iloc[i].strftime('%Y-%m-%d'):<12} "
                    f"{y_recent.iloc[i]:>10.0f} {y_pred[i]:>10.1f} {y_pred[i] - y_recent.iloc[i]:>+8.1f}"
                )

        return {"mae": mae, "rmse": rmse, "r2": r2}

    def prepare_predictions(self, game_data: List[Dict]) -> np.ndarray:
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        df = pd.DataFrame(game_data)
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"예측 데이터에 다음 특성이 없습니다: {missing}")
        X = df[self.feature_names].copy()
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"예측 데이터에 결측치가 있는 특성: {null_cols}")
        return self.scaler.transform(X)

    def predict(self, game_data: List[Dict]) -> List[Dict]:
        print("\n=== 새 경기 총점 예측 수행 (SVR) ===")
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        X_scaled = self.prepare_predictions(game_data)
        predicted_totals = self.model.predict(X_scaled)
        results = []
        for i, game in enumerate(game_data):
            result = game.copy()
            result["predicted_total"] = float(predicted_totals[i])
            results.append(result)
        print(f"예측 완료: {len(results)}개 경기")
        return results

    def save_model(self, timestamp: Optional[str] = None) -> str:
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"mlb_totals_model_svm_{timestamp}.joblib"
        scaler_path = self.model_dir / f"mlb_totals_scaler_svm_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_totals_features_svm_{timestamp}.json"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "model_info": {
                        "type": "svr_totals",
                        "target": "total_score",
                        "params": self.model.get_params(),
                    },
                },
                f,
                indent=2,
            )
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델: {model_path}")
        print(f"스케일러: {scaler_path}")
        print(f"특성: {feature_path}")
        return str(model_path)

    def _infer_paths_from_model_path(self, model_path: str) -> Tuple[Optional[Path], Optional[Path]]:
        p = Path(model_path)
        stem = p.stem
        if "model_svm_" in stem:
            ts = stem.split("model_svm_", 1)[-1]
            return (
                p.parent / f"mlb_totals_scaler_svm_{ts}.joblib",
                p.parent / f"mlb_totals_features_svm_{ts}.json",
            )
        return None, None

    def load_model(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        feature_path: Optional[str] = None,
    ) -> None:
        self.model = joblib.load(model_path)
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        else:
            sp, _ = self._infer_paths_from_model_path(model_path)
            if sp and sp.is_file():
                self.scaler = joblib.load(sp)
        if feature_path:
            with open(feature_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f).get("feature_names", [])
        else:
            _, fp = self._infer_paths_from_model_path(model_path)
            if fp and fp.is_file():
                with open(fp, "r", encoding="utf-8") as f:
                    self.feature_names = json.load(f).get("feature_names", [])
        print(f"모델 로드 완료: {model_path}")


def get_latest_training_data() -> List[Dict]:
    project_root = Path(__file__).parent.parent.parent
    training_dir = project_root / "data" / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"학습 데이터 디렉토리가 없습니다: {training_dir}")
    json_files = list(training_dir.glob("mlb_training_data_*.json"))
    if not json_files:
        raise FileNotFoundError("학습 데이터 파일을 찾을 수 없습니다.")
    latest_file = max(json_files, key=lambda x: x.name)
    print(f"학습 데이터 파일 로드: {latest_file.name}")
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = get_latest_training_data()
    data = [g for g in data if g.get("home_score") is not None and g.get("away_score") is not None]
    print(f"완료된 경기 수: {len(data)}")
    model = MLBTotalsModelSVM()
    model.train_model(data)
    model.save_model()
