import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import matplotlib.pyplot as plt

class MLBBettingModel1:
    """
    MLB 경기 승패 예측을 위한 첫 번째 모델
    
    LightGBM 분류기를 사용하여 홈팀 승리 여부를 예측합니다.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent.parent / "models" / "saved_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None  # 날짜 정보 저장을 위한 변수 추가
        self.scaler = StandardScaler()  # 특성 스케일링을 위한 스케일러 추가
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출"""
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
        
        # MLB 데이터에 맞는 기본 특성 선택
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
        
        # 기본 특성으로 DataFrame 생성 (존재하는 컬럼만 사용)
        available_features = [feat for feat in base_features if feat in df.columns]
        X = df[available_features].copy()
        
        # 누락된 특성이 있는지 확인하고 로그 출력
        missing_features = set(base_features) - set(available_features)
        if missing_features:
            print(f"경고: {len(missing_features)}개의 특성이 데이터에 없습니다: {missing_features}")
        
        # 결측치 처리
        X = X.fillna(0)
        
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
        
        return X, y
    
    def train_model(self, training_data: List[Dict], verbose: bool = True) -> None:
        """
        훈련 데이터를 사용하여 승패 예측 모델을 훈련합니다.
        
        Args:
            training_data: 훈련 데이터 리스트
            verbose: 상세 출력 여부
        """
        print("\n=== 모델 학습 시작 ===")
        
        # 특성과 레이블 추출
        X, y = self.prepare_features(training_data)
        
        # 특성 이름 저장 (모델에서 사용하는 정확한 특성 집합)
        # 디버그를 위해 모델에 사용하는 실제 특성 이름들만 정확히 저장
        self.feature_names = list(X.columns)
        
        # 지수적 증가 가중치 (최근 데이터에 더 급격한 가중치)
        n_samples = len(X)
        sample_weights = np.exp(np.linspace(0, 1, n_samples)) - 1
        sample_weights = sample_weights / sample_weights.mean()  # 정규화
        
        # 모델 학습
        self.model = LGBMClassifier(
            n_estimators=500,              # 트리 개수 감소 (1200 -> 500)
            learning_rate=0.01,            # 학습률 감소 (0.05 -> 0.01)
            max_depth=3,                   # 트리 깊이 감소 (4 -> 3)
            num_leaves=8,                  # 리프 노드 수 감소 (70 -> 8)
            min_child_samples=30,          # 최소 샘플 수 증가 (15 -> 30)
            min_child_weight=0.1,          # 최소 가중치 증가 (0.01 -> 0.1)
            subsample=0.7,                 # 데이터 샘플링 비율 감소 (0.85 -> 0.7)
            colsample_bytree=0.7,          # 특성 샘플링 비율 감소 (0.85 -> 0.7)
            reg_alpha=0.5,                 # L1 규제 증가 (0.1 -> 0.5)
            reg_lambda=2.0,                # L2 규제 증가 (1.0 -> 2.0)
            random_state=42,
            boosting_type='gbdt',
            objective='binary',
            metric='auc',
            verbose=-1
        )
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 중요도 계산
        if verbose and hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
            
        # 검증 데이터에 대한 성능 평가
        if verbose:
            self.evaluate_model(X, y, n_games=50)
    
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
        """
        예측을 위해 게임 데이터를 준비합니다.
        모델 학습에 사용된 특성과 동일한 특성만 사용합니다.
        """
        # 특성 이름이 로드되지 않은 경우 에러
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(game_data)
        
        # 원본 base_features 리스트 가져오기 (prepare_features 메서드에 정의되어 있음)
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
        
        # 모델이 학습한 특성들만 선택
        features_to_use = [feat for feat in base_features if feat in self.feature_names]
        
        # 피처 존재 여부 확인
        missing_features = [feat for feat in features_to_use if feat not in df.columns]
        if missing_features:
            raise ValueError(f"예측 데이터에 다음 특성이 없습니다: {missing_features}. "
                           f"데이터 처리 단계에서 해당 특성이 생성되었는지 확인하세요.")
            
        # 필요한 특성만 추출
        X = df[features_to_use].copy()
        
        # 결측치 확인 (완전성을 위해)
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"예측 데이터에 결측치가 있는 특성이 있습니다: {null_cols}. "
                           f"데이터 처리 단계에서 결측치를 처리해주세요.")
        
        print(f"원본 예측 데이터 특성 수: {len(df.columns)}")
        print(f"모델 입력 특성 수: {len(features_to_use)}")
        
        return X

    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """
        새로운 경기 데이터를 사용하여 승패 예측을 수행합니다.
        
        Args:
            game_data: 예측할 경기 데이터 리스트
            
        Returns:
            예측 결과가 포함된 딕셔너리 리스트
        """
        print("\n=== 새 경기 예측 수행 ===")
        
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
            
            print(f"예측 완료: {len(results)}개 경기")
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
        model_path = self.model_dir / f"mlb_betting_model1_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_features1_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'lightgbm',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
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
        
        print(f"\n=== 모델 로드 완료 ===")
        print(f"모델 로드 경로: {model_path}")
        if feature_path:
            print(f"특성 정보 로드 경로: {feature_path}")
    
    def _plot_feature_importance(self) -> None:
        """모델의 특성 중요도를 시각화"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("모델이 학습되지 않았거나 특성 중요도를 지원하지 않습니다.")
            return
            
        try:
            # 특성 중요도 계산 (gain 기준으로 정규화)
            if hasattr(self.model, 'booster_'):
                importances = self.model.booster_.feature_importance(importance_type='gain')
            else:
                importances = self.model.feature_importances_
                
            importances = 100.0 * (importances / importances.sum())  # 퍼센트로 변환
            
            # 상위 20개 중요 특성 출력
            print("\n=== 상위 20개 중요 특성 (%) ===")
            feature_importance = dict(zip(self.feature_names, importances))
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            for feature, importance in sorted_features:
                print(f"{feature}: {importance:.2f}%")
                
            # 시각화 저장
            # 현재 구현 생략
            
        except Exception as e:
            print(f"특성 중요도 시각화 중 오류: {e}")

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
            roc_auc = 0.5  # 클래스가 하나만 있는 경우
            
        # 혼동 행렬
        cm = confusion_matrix(y_recent, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 정밀도, 재현율, F1 점수
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 홈팀/원정팀 승리 예측 정확도
        home_win_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
        away_win_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 결과 출력
        print(f"\n=== 최근 {len(X_recent)}경기 예측 성능 ===")
        print(f"정확도: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print(f"정밀도: {precision:.3f}")
        print(f"재현율: {recall:.3f}")
        print(f"F1 점수: {precision:.3f}")
        
        print("\n혼동 행렬:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        print(f"홈팀 승리 예측 정확도: {home_win_accuracy:.3f}")
        print(f"원정팀 승리 예측 정확도: {away_win_accuracy:.3f}")
        
        # 최근 10경기 예측 상세 결과
        self._show_recent_predictions(X_recent, y_recent, 10)
        
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

    def _show_recent_predictions(self, X: pd.DataFrame, y: pd.Series, n_games: int = 10) -> None:
        """최근 n경기의 예측 결과 상세 출력"""
        if not hasattr(self, 'dates') or self.dates is None:
            print("날짜 정보가 없습니다.")
            return
            
        if self.model is None:
            print("모델이 학습되지 않았습니다.")
            return
            
        # 날짜 순으로 정렬하여 최근 n_games 가져오기
        recent_indices = list(range(max(0, len(X) - n_games), len(X)))
        
        # 예측
        y_pred = self.model.predict(X.iloc[recent_indices])
        y_prob = self.model.predict_proba(X.iloc[recent_indices])[:, 1]
        
        # 결과 출력
        print(f"\n=== 최근 {len(recent_indices)}경기 예측 상세 ===")
        for i, idx in enumerate(recent_indices):
            # 날짜가 있는 경우 출력, 없으면 인덱스만 출력
            if hasattr(self, 'dates') and len(self.dates) > idx:
                date_info = self.dates.iloc[idx].strftime("%Y-%m-%d")
            else:
                date_info = f"Game {idx}"
                
            print(f"날짜: {date_info}, 실제: {y.iloc[idx]}, 예측: {y_pred[i]}, 홈팀 승리확률: {y_prob[i]:.3f}")


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
    # MLB 베팅 모델 초기화
    model = MLBBettingModel1()
    
    # 학습 데이터 로드
    print("=== 경로 디버깅 ===")
    print(f"스크립트 경로: {Path(__file__).absolute()}")
    print(f"프로젝트 루트: {Path(__file__).parent.parent.parent}")
    
    try:
        # 최신 학습 데이터 로드
        training_data = get_latest_training_data()
        
        # 전체 데이터로 모델 학습
        model.train_model(training_data)
        
        # 모델 저장
        model.save_model()
        
        # 예측 데이터가 있으면 예측 수행
        print("예측 데이터 경로:", Path(__file__).parent.parent.parent / "data" / "prediction")
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
                
                winner = home_team if is_home_win else away_team
                loser = away_team if is_home_win else home_team
                win_prob = home_win_prob if is_home_win else 1 - home_win_prob
                
                print(f"{pred['date']} - {home_team} (홈) vs {away_team} (원정)")
                print(f"  예측 승자: {winner} (승리확률: {win_prob:.3f})")
                print(f"  예측 패자: {loser}")
                print(f"  홈팀 승리확률: {home_win_prob:.3f}")
                print()
                
            # 예측 결과 저장
            save_predictions(predictions, "model1")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc() 