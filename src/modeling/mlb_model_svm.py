import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import matplotlib.pyplot as plt


class MLBBettingModelSVM:
    """
    MLB 경기 승패 예측을 위한 Support Vector Machine 모델
    
    SVM 분류기를 사용하여 홈팀 승리 여부를 예측합니다.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_dir = Path(__file__).parent.parent / "models" / "saved_models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.dates = None
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출"""
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """SVM 모델 학습"""
        n_samples = len(X)
        # 시간 가중치 (최근 데이터에 적당한 가중치 적용)
        sample_weights = np.linspace(0.6, 1.4, n_samples)  # 선형 증가 (적당한 범위)
        
        # 특성 스케일링 (SVM에 매우 중요)
        X_scaled = self.scaler.fit_transform(X)
        
        # SVM 모델 설정 (90%대 성능 목표)
        # 더 세밀한 조정으로 90%대 달성
        self.model = SVC(
            C=3.5,                         # 규제 매개변수 미세 조정 (8.0 -> 3.5)
            kernel='rbf',                  # RBF 커널 유지 (안정적인 성능)
            gamma=0.02,                    # 감마 값 더 보수적으로 (0.05 -> 0.02)
            degree=3,                      # 다항식 차수 기본값 유지
            coef0=0.0,                     # 커널 함수 독립항 기본값
            shrinking=True,                # 휴리스틱 사용 여부
            probability=True,              # 확률 추정 활성화 (중요!)
            tol=5e-3,                      # 허용 오차 더 크게 (1e-3 -> 5e-3)
            cache_size=300,                # 캐시 크기 적당히
            class_weight='balanced',       # 클래스 불균형 처리
            verbose=False,                 # 상세 출력 비활성화
            max_iter=1500,                 # 최대 반복 횟수 조정 (2000 -> 1500)
            decision_function_shape='ovr', # 다중 클래스 결정 함수 형태
            break_ties=False,              # 동점 처리 방식
            random_state=42
        )
        
        print("\n=== Support Vector Machine 모델 학습 시작 ===")
        print(f"커널: {self.model.kernel.upper()}, C: {self.model.C}, 감마: {self.model.gamma}")
        print(f"허용 오차: {self.model.tol}, 최대 반복: {self.model.max_iter}")
        print("세밀한 조정으로 90%대 성능 목표")
        
        # 모델 학습
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        
        print(f"학습 완료 - 서포트 벡터 수: {self.model.n_support_}")
        print(f"총 서포트 벡터 비율: {sum(self.model.n_support_) / len(X):.3f}")
        
        # SVM은 특성 중요도를 직접 제공하지 않으므로
        # 선형 SVM으로 근사하여 특성 중요도 계산
        feature_importance = self._calculate_feature_importance(X_scaled, y, sample_weights)
        
        metrics = {
            'feature_importance': dict(zip(self.feature_names, feature_importance)),
            'n_support_vectors': self.model.n_support_.tolist(),
            'support_vector_ratio': sum(self.model.n_support_) / len(X),
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma
        }
        
        # 상위 15개 중요 특성 출력
        print("\n=== 상위 15개 중요 특성 (선형 근사 %) ===")
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:15]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.2f}%")
        
        return metrics
    
    def _calculate_feature_importance(self, X_scaled: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
        """선형 SVM을 사용하여 특성 중요도 근사 계산"""
        from sklearn.svm import LinearSVC
        
        # 선형 SVM으로 근사
        linear_svm = LinearSVC(
            C=self.model.C,
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        
        try:
            linear_svm.fit(X_scaled, y, sample_weight=sample_weights)
            # 계수의 절댓값을 특성 중요도로 사용
            feature_importance = np.abs(linear_svm.coef_[0])
        except:
            # 선형 SVM이 실패하면 균등 분포로 설정
            feature_importance = np.ones(X_scaled.shape[1])
        
        # 백분율로 정규화
        feature_importance = 100.0 * (feature_importance / feature_importance.sum())
        
        return feature_importance
    
    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """학습된 모델로 최근 N경기 예측 성능 평가"""
        # 최근 n_games 선택
        if len(X) <= n_games:
            X_recent = X
            y_recent = y
            dates_recent = self.dates
        else:
            X_recent = X[-n_games:]
            y_recent = y[-n_games:]
            dates_recent = self.dates[-n_games:]
        
        # 특성 스케일링
        X_recent_scaled = self.scaler.transform(X_recent)
        
        # 예측 수행
        y_pred = self.model.predict(X_recent_scaled)
        y_pred_proba = self.model.predict_proba(X_recent_scaled)[:, 1]
        
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
        if conf_matrix[1,0] + conf_matrix[1,1] > 0:
            home_win_accuracy = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
            print(f"홈팀 승리 예측 정확도: {home_win_accuracy:.3f}")
        
        # 원정팀 승리 예측 정확도
        if conf_matrix[0,0] + conf_matrix[0,1] > 0:
            away_win_accuracy = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print(f"원정팀 승리 예측 정확도: {away_win_accuracy:.3f}")
        
        print("\n=== 최근 10경기 예측 상세 ===")
        for date, true, pred, prob in results['predictions'][-10:]:
            print(f"날짜: {date}, 실제: {true}, 예측: {pred}, 홈팀 승리확률: {prob:.3f}")
        
        return results
    
    def prepare_predictions(self, game_data: List[Dict]) -> pd.DataFrame:
        """예측을 위해 게임 데이터를 준비합니다."""
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        
        df = pd.DataFrame(game_data)
        
        # 기본 특성 리스트
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
            'diff_recent_avg_batting_avg', 'diff_recent_avg_batting_ops', 'diff_recent_avg_batting_homeRuns',
            'diff_overall_record_win_rate', 'diff_home_record_win_rate', 'diff_road_record_win_rate',
            'diff_avg_batting_avg', 'diff_avg_batting_ops', 'diff_avg_batting_homeRuns',
            'diff_avg_pitching_era', 'diff_avg_pitching_whip',
            'day_of_week', 'is_weekend', 'month',
            'early_season', 'mid_season', 'late_season'
        ]
        
        # 모델이 학습한 특성들만 선택
        features_to_use = [feat for feat in base_features if feat in self.feature_names]
        
        # 피처 존재 여부 확인
        missing_features = [feat for feat in features_to_use if feat not in df.columns]
        if missing_features:
            raise ValueError(f"예측 데이터에 다음 특성이 없습니다: {missing_features}")
            
        # 필요한 특성만 추출
        X = df[features_to_use].copy()
        
        # 결측치 확인
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            raise ValueError(f"예측 데이터에 결측치가 있는 특성이 있습니다: {null_cols}")
        
        print(f"원본 예측 데이터 특성 수: {len(df.columns)}")
        print(f"모델 입력 특성 수: {len(features_to_use)}")
        
        return X

    def predict(self, game_data: List[Dict]) -> List[Dict]:
        """새로운 경기 데이터를 사용하여 승패 예측을 수행합니다."""
        print("\n=== 새 경기 예측 수행 (SVM) ===")
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        if not self.feature_names:
            raise ValueError("특성 이름이 로드되지 않았습니다.")
        
        print(f"모델 특성 수: {len(self.feature_names)}")
        
        try:
            # 예측용 데이터 준비
            X_model = self.prepare_predictions(game_data)
            
            # 특성 스케일링
            X_model_scaled = self.scaler.transform(X_model)
            
            # 예측 수행
            print("=== 예측 수행 ===")
            probabilities = self.model.predict_proba(X_model_scaled)[:, 1]  # 홈팀 승리 확률
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
            raise ValueError("저장할 모델이 없습니다.")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로
        model_path = self.model_dir / f"mlb_betting_model_svm_{timestamp}.joblib"
        scaler_path = self.model_dir / f"mlb_scaler_svm_{timestamp}.joblib"
        feature_path = self.model_dir / f"mlb_features_svm_{timestamp}.json"
        
        # 모델과 스케일러 저장
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'support_vector_machine',
                    'params': self.model.get_params(),
                    'kernel': self.model.kernel,
                    'C': self.model.C,
                    'gamma': self.model.gamma,
                    'n_support_vectors': self.model.n_support_.tolist()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"스케일러 저장 경로: {scaler_path}")
        print(f"특성 정보 저장 경로: {feature_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, scaler_path: str = None, feature_path: str = None) -> None:
        """저장된 모델과 특성 이름 로드"""
        # 모델 로드
        self.model = joblib.load(model_path)
        
        # 스케일러 로드
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        # 특성 이름 로드
        if feature_path:
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get('feature_names', [])
        
        print(f"\n=== 모델 로드 완료 ===")
        print(f"모델 로드 경로: {model_path}")
        if scaler_path:
            print(f"스케일러 로드 경로: {scaler_path}")
        if feature_path:
            print(f"특성 정보 로드 경로: {feature_path}")


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


if __name__ == "__main__":
    try:
        # 최신 학습 데이터 로드
        training_data = get_latest_training_data()
        
        # 모델 초기화 및 특성 준비
        model = MLBBettingModelSVM()
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
            
            if prediction_data:
                print("\n=== 새 경기 예측 수행 ===")
                predictions = model.predict(prediction_data)
                
                # 예측 결과 출력
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pred_dir = Path(__file__).parent.parent / "predictions"
                pred_dir.mkdir(exist_ok=True, parents=True)
                
                with open(pred_dir / f"mlb_game_predictions_model_svm_{timestamp}.json", 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                print(f"예측 결과 저장 완료: {pred_dir / f'mlb_game_predictions_model_svm_{timestamp}.json'}")
            
        except FileNotFoundError as e:
            print(f"\n경고: {e}")
            print("예측 데이터 파일이 없어 예측을 수행하지 않습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(traceback.format_exc()) 