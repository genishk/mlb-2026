import sys
from pathlib import Path

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler

from src.modeling.mlb_model1 import MLBBettingModel1
from src.modeling.mlb_model2 import MLBBettingModel2
from src.modeling.mlb_model3 import MLBBettingModel3
from src.modeling.mlb_model4 import MLBBettingModel4
from src.modeling.mlb_model5 import MLBBettingModel5
from src.modeling.mlb_model6 import MLBBettingModel6
from src.modeling.mlb_model7 import MLBBettingModel7
from src.modeling.mlb_model8 import MLBBettingModel8
from src.modeling.mlb_model9 import MLBBettingModel9
from src.modeling.mlb_model_rf import MLBBettingModelRF
from src.modeling.mlb_model_nn import MLBBettingModelNN
from src.modeling.mlb_model_svm import MLBBettingModelSVM
from src.modeling.mlb_model_advanced_catboost_basic import MLBAdvancedCatBoostBasicModel
from src.modeling.mlb_model_advanced_catboost import MLBAdvancedCatBoostModel
from src.modeling.mlb_model_advanced_lgbm_basic import MLBAdvancedLGBMBasicModel
from src.modeling.mlb_model_advanced_lgbm import MLBAdvancedBettingModel as MLBAdvancedLGBMModel
from src.modeling.mlb_model_advanced_nn import MLBAdvancedNeuralNetworkModel
from src.modeling.mlb_model_advanced_rf import MLBAdvancedRandomForestModel
from src.modeling.mlb_model_advanced_svm import MLBAdvancedSVMModel
from src.modeling.mlb_model_advanced_xgboost_basic import MLBAdvancedXGBoostBasicModel
from src.modeling.mlb_model_advanced_xgboost import MLBAdvancedXGBoostModel
from src.modeling.mlb_model1_extended_lgbm import MLBBettingModel1ExtendedLGBM
from src.modeling.mlb_model2_extended_catboost import MLBBettingModel2ExtendedCatBoost
from src.modeling.mlb_model3_extended_xgboost import MLBBettingModel3ExtendedXGBoost

class MLBEnsemblePredictor:
    """MLB 경기 예측을 위한 앙상블 예측기
    
    세 가지 모델(LightGBM, CatBoost, XGBoost)의 예측을 결합하여
    더 안정적인 예측을 수행합니다.
    추가로 모델 4~9의 개별 예측도 제공합니다.
    """
    
    def __init__(self):
        # 앙상블에 사용되는 메인 모델들 (1~3)
        self.models = {
            'model1': MLBBettingModel1(),  # LightGBM
            'model2': MLBBettingModel2(),  # CatBoost
            'model3': MLBBettingModel3()   # XGBoost
        }
        
        # 테스트용 추가 모델들 (4~9) - 앙상블에 포함되지 않음
        self.additional_models = {
            'model4': MLBBettingModel4(),
            'model5': MLBBettingModel5(),
            'model6': MLBBettingModel6(),
            'model7': MLBBettingModel7(),
            'model8': MLBBettingModel8(),
            'model9': MLBBettingModel9()
        }
        
        # 새로 추가한 고성능 모델들
        self.new_models = {
            'model_rf': MLBBettingModelRF(),   # Random Forest
            'model_nn': MLBBettingModelNN(),   # Neural Network
            'model_svm': MLBBettingModelSVM(), # SVM
            'model_advanced_catboost_basic': MLBAdvancedCatBoostBasicModel(),  # 고급 CatBoost Basic
            'model_advanced_catboost': MLBAdvancedCatBoostModel(),  # 고급 CatBoost
            'model_advanced_lgbm_basic': MLBAdvancedLGBMBasicModel(),  # 고급 LightGBM Basic
            'model_advanced_lgbm': MLBAdvancedLGBMModel(),  # 고급 LightGBM
            'model_advanced_nn': MLBAdvancedNeuralNetworkModel(),  # 고급 Neural Network
            'model_advanced_rf': MLBAdvancedRandomForestModel(),  # 고급 Random Forest
            'model_advanced_svm': MLBAdvancedSVMModel(),  # 고급 SVM
            'model_advanced_xgboost_basic': MLBAdvancedXGBoostBasicModel(),  # 고급 XGBoost Basic
            'model_advanced_xgboost': MLBAdvancedXGBoostModel(),  # 고급 XGBoost
            'model1_extended_lgbm': MLBBettingModel1ExtendedLGBM(),  # LightGBM 확장 (123개 특성)
            'model2_extended_catboost': MLBBettingModel2ExtendedCatBoost(),  # CatBoost 확장 (123개 특성)
            'model3_extended_xgboost': MLBBettingModel3ExtendedXGBoost()  # XGBoost 확장 (123개 특성)
        }
        
        self.model_paths = {}
        self.project_root = Path(__file__).parent.parent.parent
        
    def load_latest_models(self) -> Dict[str, str]:
        """각 모델의 가장 최근 저장 파일을 로드"""
        models_dir = self.project_root / "src" / "models" / "saved_models"
        
        if not models_dir.exists():
            raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        
        loaded_models = {}
        
        # 메인 앙상블 모델들 (1~3) 로드
        for model_type in ['model1', 'model2', 'model3']:
            model_files = list(models_dir.glob(f'mlb_betting_{model_type}_*.joblib'))
            feature_files = list(models_dir.glob(f'mlb_features{model_type[-1]}_*.json'))
            
            if model_files and feature_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
                
                # 모델과 특성 로드
                self.models[model_type].load_model(str(latest_model), str(latest_features))
                loaded_models[model_type] = str(latest_model)
                print(f"{model_type} 로드 완료: {latest_model.name}")
            else:
                print(f"경고: {model_type}의 모델 또는 특성 파일을 찾을 수 없습니다.")
        
        # 추가 테스트 모델들 (4~9) 로드
        for model_type in ['model4', 'model5', 'model6', 'model7', 'model8', 'model9']:
            model_files = list(models_dir.glob(f'mlb_betting_{model_type}_*.joblib'))
            feature_files = list(models_dir.glob(f'mlb_features{model_type[-1]}_*.json'))
            
            if model_files and feature_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
                
                # 모델과 특성 로드
                self.additional_models[model_type].load_model(str(latest_model), str(latest_features))
                loaded_models[model_type] = str(latest_model)
                print(f"{model_type} 로드 완료: {latest_model.name}")
            else:
                print(f"경고: {model_type}의 모델 또는 특성 파일을 찾을 수 없습니다.")
        
        # 새로 추가한 고성능 모델들 로드
        new_model_patterns = {
            'model_rf': 'mlb_betting_model_rf_*.joblib',
            'model_nn': 'mlb_betting_model_nn_*.joblib', 
            'model_svm': 'mlb_betting_model_svm_*.joblib',
            'model_advanced_catboost_basic': 'mlb_advanced_catboost_basic_*.joblib',
            'model_advanced_catboost': 'mlb_advanced_catboost_*.joblib',
            'model_advanced_lgbm_basic': 'mlb_advanced_lgbm_basic_*.joblib',
            'model_advanced_lgbm': 'mlb_advanced_lgbm_*.joblib',
            'model_advanced_nn': 'mlb_advanced_nn_*.joblib',
            'model_advanced_rf': 'mlb_advanced_rf_*.joblib',
            'model_advanced_svm': 'mlb_advanced_svm_*.joblib',
            'model_advanced_xgboost_basic': 'mlb_advanced_xgboost_basic_*.joblib',
            'model_advanced_xgboost': 'mlb_advanced_xgboost_*.joblib',
            'model1_extended_lgbm': 'mlb_model1_extended_lgbm_*.joblib',
            'model2_extended_catboost': 'mlb_model2_extended_catboost_*.joblib',
            'model3_extended_xgboost': 'mlb_model3_extended_xgboost_*.joblib'
        }
        
        new_feature_patterns = {
            'model_rf': 'mlb_features_rf_*.json',
            'model_nn': 'mlb_features_nn_*.json',
            'model_svm': 'mlb_features_svm_*.json',
            'model_advanced_catboost_basic': 'mlb_advanced_catboost_basic_features_*.json',
            'model_advanced_catboost': 'mlb_advanced_catboost_features_*.json',
            'model_advanced_lgbm_basic': 'mlb_advanced_lgbm_basic_features_*.json',
            'model_advanced_lgbm': 'mlb_advanced_lgbm_features_*.json',
            'model_advanced_nn': 'mlb_advanced_nn_features_*.json',
            'model_advanced_rf': 'mlb_advanced_rf_features_*.json',
            'model_advanced_svm': 'mlb_advanced_svm_features_*.json',
            'model_advanced_xgboost_basic': 'mlb_advanced_xgboost_basic_features_*.json',
            'model_advanced_xgboost': 'mlb_advanced_xgboost_features_*.json',
            'model1_extended_lgbm': 'mlb_model1_extended_lgbm_features_*.json',
            'model2_extended_catboost': 'mlb_model2_extended_catboost_features_*.json',
            'model3_extended_xgboost': 'mlb_model3_extended_xgboost_features_*.json'
        }
        
        new_scaler_patterns = {
            'model_rf': None,  # Random Forest는 스케일러 없음
            'model_nn': 'mlb_scaler_nn_*.joblib',
            'model_svm': 'mlb_scaler_svm_*.joblib',
            'model_advanced_catboost_basic': 'mlb_advanced_catboost_basic_scaler_*.joblib',
            'model_advanced_catboost': 'mlb_advanced_catboost_scaler_*.joblib',
            'model_advanced_lgbm_basic': 'mlb_advanced_lgbm_basic_scaler_*.joblib',
            'model_advanced_lgbm': 'mlb_advanced_lgbm_scaler_*.joblib',
            'model_advanced_nn': 'mlb_advanced_nn_scaler_*.joblib',
            'model_advanced_rf': 'mlb_advanced_rf_scaler_*.joblib',
            'model_advanced_svm': 'mlb_advanced_svm_scaler_*.joblib',
            'model_advanced_xgboost_basic': 'mlb_advanced_xgboost_basic_scaler_*.joblib',
            'model_advanced_xgboost': 'mlb_advanced_xgboost_scaler_*.joblib',
            'model1_extended_lgbm': None,  # 확장 모델은 스케일러 없음
            'model2_extended_catboost': None,  # 확장 모델은 스케일러 없음
            'model3_extended_xgboost': None  # 확장 모델은 스케일러 없음
        }
        
        # 고급 모델들용 특성 선택기 패턴 추가
        new_selector_patterns = {
            'model_advanced_catboost_basic': 'mlb_advanced_catboost_basic_selector_*.joblib',
            'model_advanced_catboost': 'mlb_advanced_catboost_selector_*.joblib',
            'model_advanced_lgbm_basic': 'mlb_advanced_lgbm_basic_selector_*.joblib',
            'model_advanced_lgbm': 'mlb_advanced_lgbm_selector_*.joblib',
            'model_advanced_nn': 'mlb_advanced_nn_selector_*.joblib',
            'model_advanced_rf': 'mlb_advanced_rf_selector_*.joblib',
            'model_advanced_svm': 'mlb_advanced_svm_selector_*.joblib',
            'model_advanced_xgboost_basic': 'mlb_advanced_xgboost_basic_selector_*.joblib',
            'model_advanced_xgboost': 'mlb_advanced_xgboost_selector_*.joblib'
        }
        
        for model_type in ['model_rf', 'model_nn', 'model_svm', 'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic', 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm', 'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost']:
            if model_type in ['model_advanced_catboost_basic', 'model_advanced_catboost']:
                # 고급 CatBoost 모델들의 경우 더 정확한 필터링
                if model_type == 'model_advanced_catboost_basic':
                    model_files = [f for f in models_dir.glob('mlb_advanced_catboost_basic_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name]
                else:  # model_advanced_catboost
                    model_files = [f for f in models_dir.glob('mlb_advanced_catboost_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name 
                                  and 'basic' not in f.name]  # basic 버전과 구분
            elif model_type in ['model_advanced_lgbm_basic', 'model_advanced_lgbm']:
                # 고급 LightGBM 모델들의 경우 더 정확한 필터링
                if model_type == 'model_advanced_lgbm_basic':
                    model_files = [f for f in models_dir.glob('mlb_advanced_lgbm_basic_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name]
                else:  # model_advanced_lgbm
                    model_files = [f for f in models_dir.glob('mlb_advanced_lgbm_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name 
                                  and 'basic' not in f.name]  # basic 버전과 구분
            elif model_type in ['model_advanced_xgboost_basic', 'model_advanced_xgboost']:
                # 고급 XGBoost 모델들의 경우 더 정확한 필터링
                if model_type == 'model_advanced_xgboost_basic':
                    model_files = [f for f in models_dir.glob('mlb_advanced_xgboost_basic_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name]
                else:  # model_advanced_xgboost
                    model_files = [f for f in models_dir.glob('mlb_advanced_xgboost_*.joblib') 
                                  if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name 
                                  and 'basic' not in f.name]  # basic 버전과 구분
            elif model_type in ['model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm']:
                # 새로운 고급 모델들의 경우 정확한 필터링
                model_files = [f for f in models_dir.glob(new_model_patterns[model_type]) 
                              if 'selector' not in f.name and 'scaler' not in f.name and 'features' not in f.name]
            elif model_type in ['model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost']:
                # 확장 모델들은 스케일러/선택기 없이 단순 로드
                model_files = list(models_dir.glob(new_model_patterns[model_type]))
            else:
                model_files = list(models_dir.glob(new_model_patterns[model_type]))
            
            feature_files = list(models_dir.glob(new_feature_patterns[model_type]))
            
            if model_files and feature_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
                
                # 스케일러 파일 확인
                scaler_path = None
                if new_scaler_patterns[model_type]:
                    scaler_files = list(models_dir.glob(new_scaler_patterns[model_type]))
                    if scaler_files:
                        scaler_path = str(max(scaler_files, key=lambda x: x.stat().st_mtime))
                
                # 특성 선택기 파일 확인 (고급 모델들만)
                selector_path = None
                if model_type in new_selector_patterns:
                    selector_files = list(models_dir.glob(new_selector_patterns[model_type]))
                    if selector_files:
                        selector_path = str(max(selector_files, key=lambda x: x.stat().st_mtime))
                
                # 모델과 특성 로드
                if model_type == 'model_rf':
                    # Random Forest는 스케일러 없음
                    self.new_models[model_type].load_model(str(latest_model), str(latest_features))
                elif model_type in ['model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost']:
                    # 확장 모델들은 단순 로드 (스케일러/선택기 없음)
                    self.new_models[model_type].load_model(str(latest_model), str(latest_features))
                elif model_type in ['model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic', 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm', 'model_advanced_xgboost_basic', 'model_advanced_xgboost']:
                    # 모든 고급 모델들은 4개 파일 모두 필요
                    self.new_models[model_type].load_model(
                        str(latest_model), 
                        scaler_path, 
                        selector_path, 
                        str(latest_features)
                    )
                else:
                    # Neural Network, SVM은 스케일러 포함
                    self.new_models[model_type].load_model(str(latest_model), scaler_path, str(latest_features))
                
                loaded_models[model_type] = str(latest_model)
                print(f"{model_type} 로드 완료: {latest_model.name}")
            else:
                print(f"경고: {model_type}의 모델 또는 특성 파일을 찾을 수 없습니다.")
        
        return loaded_models
    
    def load_prediction_data(self) -> List[Dict]:
        """가장 최근의 예측 데이터를 로드합니다 (고급 CatBoost와 동일한 방식)."""
        data_dir = self.project_root / "data" / "prediction"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"예측 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        
        # 예측 데이터 파일 찾기 (고급 CatBoost와 동일한 방식)
        prediction_files = list(data_dir.glob("mlb_prediction_data_*.json"))
        if not prediction_files:
            raise FileNotFoundError("예측 데이터 파일을 찾을 수 없습니다.")
        
        latest_prediction_file = max(prediction_files, key=lambda x: x.name)
        print(f"예측 데이터 파일 로드: {latest_prediction_file.name}")
        
        # 예측 데이터 로드 (고급 CatBoost와 정확히 동일한 방식)
        with open(latest_prediction_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
        
        print(f"예측 데이터 로드 완료: {len(prediction_data)}개 경기")
        
        # 순서 변경 없이 원본 그대로 반환 (고급 CatBoost와 동일)
        return prediction_data
    
    def analyze_probability_scales(self, predictions_df: pd.DataFrame) -> Dict:
        """각 모델의 예측 확률 스케일 분석"""
        model_columns = ['model1_prob', 'model2_prob', 'model3_prob']
        scale_analysis = {}
        
        for col in model_columns:
            probs = predictions_df[col]
            scale_analysis[col] = {
                'min': probs.min(),
                'max': probs.max(),
                'mean': probs.mean(),
                'std': probs.std(),
                'median': probs.median(),
                'skew': probs.skew(),
                'ranges': {
                    '0.5-0.6': ((probs >= 0.5) & (probs < 0.6)).sum(),
                    '0.6-0.7': ((probs >= 0.6) & (probs < 0.7)).sum(),
                    '0.7-0.8': ((probs >= 0.7) & (probs < 0.8)).sum(),
                    '0.8-0.9': ((probs >= 0.8) & (probs < 0.9)).sum(),
                    '0.9+': (probs >= 0.9).sum()
                }
            }
        
        return scale_analysis
    
    def analyze_model_agreement(self, predictions_df: pd.DataFrame) -> Dict:
        """모델 간 예측 일치도 분석"""
        model_columns = ['model1_prob', 'model2_prob', 'model3_prob']
        
        # 각 모델의 예측 (0 또는 1)
        predictions = pd.DataFrame()
        for col in model_columns:
            predictions[col] = (predictions_df[col] > 0.5).astype(int)
        
        # 모델 간 일치도 분석
        agreement_analysis = {
            'full_agreement': (predictions.sum(axis=1) == 3).sum() + (predictions.sum(axis=1) == 0).sum(),
            'partial_agreement': (predictions.sum(axis=1) == 2).sum() + (predictions.sum(axis=1) == 1).sum(),
            'model_correlations': {}
        }
        
        # 모델 간 상관관계
        for i, col1 in enumerate(model_columns):
            for col2 in model_columns[i+1:]:
                corr = predictions_df[col1].corr(predictions_df[col2])
                agreement_analysis['model_correlations'][f'{col1}_vs_{col2}'] = corr
        
        return agreement_analysis
    
    def predict_games(self, game_data: List[Dict], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """앙상블 예측 수행"""
        if not weights:
            weights = {'model1': 0.33, 'model2': 0.33, 'model3': 0.34}
            
        all_predictions = []
        
        # 메인 앙상블 모델들 (1~3) 예측 수행
        for model_name, model in self.models.items():
            predictions = model.predict(game_data)
            for i, pred in enumerate(predictions):
                if len(all_predictions) <= i:
                    all_predictions.append({})
                all_predictions[i].update({
                    f'{model_name}_prob': pred['home_win_probability'],
                    'date': pred['date'],
                    'home_team': pred['home_team_name'],
                    'away_team': pred['away_team_name']
                })
        
                # 시간 정보 추가 (원본 게임 데이터에서)
                if i < len(game_data):
                    game = game_data[i]
                    if 'start_time_et' in game:
                        all_predictions[i]['start_time_et'] = game['start_time_et']
                    if 'start_time' in game:
                        all_predictions[i]['start_time'] = game['start_time']
                    if 'venue_name' in game:
                        all_predictions[i]['venue_name'] = game['venue_name']
                    if 'day_night' in game:
                        all_predictions[i]['day_night'] = game['day_night']
        
        # 추가 모델들 (4~9) 예측 수행
        for model_name, model in self.additional_models.items():
            predictions = model.predict(game_data)
            for i, pred in enumerate(predictions):
                if i < len(all_predictions):
                    all_predictions[i][f'{model_name}_prob'] = pred['home_win_probability']
        
        # 새로 추가한 고성능 모델들 예측 수행
        for model_name, model in self.new_models.items():
            try:
                # 모든 모델을 동일하게 처리 (고급 CatBoost도 포함)
                predictions = model.predict(game_data)
                
                # 인덱스 기반 매칭 (모든 모델 동일)
                for i, pred in enumerate(predictions):
                    if i < len(all_predictions):
                        all_predictions[i][f'{model_name}_prob'] = pred['home_win_probability']
                
                print(f"{model_name} 예측 완료")
            except Exception as e:
                print(f"경고: {model_name} 예측 중 오류 발생: {e}")
                # 오류 발생 시 기본값으로 0.5 설정
                for i in range(len(all_predictions)):
                    all_predictions[i][f'{model_name}_prob'] = 0.5
        
        # DataFrame 생성
        predictions_df = pd.DataFrame(all_predictions)
        
        # 앙상블 확률 계산 (모든 활성화된 모델 사용)
        ensemble_probs = []
        for _, row in predictions_df.iterrows():
            weighted_sum = 0
            for model_name, weight in weights.items():
                if f'{model_name}_prob' in row:
                    weighted_sum += row[f'{model_name}_prob'] * weight
                else:
                    print(f"경고: {model_name}_prob 컬럼을 찾을 수 없습니다.")
            ensemble_probs.append(weighted_sum)
        
        predictions_df['ensemble_prob'] = ensemble_probs
        predictions_df['win_probability'] = ensemble_probs
        
        # 승자 예측
        predictions_df['predicted_winner'] = predictions_df.apply(
            lambda row: row['home_team'] if row['win_probability'] > 0.5 else row['away_team'],
            axis=1
        )
        
        # 시간순으로 정렬
        def get_sort_time(row):
            start_time = row.get('start_time_et', 'Unknown')
            if pd.isna(start_time) or start_time == 'Unknown':
                return '9999-12-31 23:59:59'  # 시간 정보가 없는 경우 마지막에 정렬
            try:
                # start_time_et 형식: "2025-05-23 19:10:00 EDT"
                time_part = str(start_time).split(' EDT')[0].split(' EST')[0]
                return time_part
            except:
                return '9999-12-31 23:59:59'
        
        predictions_df['sort_time'] = predictions_df.apply(get_sort_time, axis=1)
        predictions_df = predictions_df.sort_values('sort_time').drop('sort_time', axis=1)
        
        return predictions_df
    
    def save_predictions(self, predictions: pd.DataFrame) -> str:
        """예측 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_dir = self.project_root / "src" / "predictions"
        pred_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = pred_dir / f"mlb_ensemble_predictions_{timestamp}.json"
        
        # 예측 결과를 딕셔너리 리스트로 변환
        predictions_list = []
        for _, row in predictions.iterrows():
            pred_dict = {
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'predicted_winner': row['predicted_winner'],
                'win_probability': float(row['win_probability']),
                'model1_probability': float(row['model1_prob']),
                'model2_probability': float(row['model2_prob']),
                'model3_probability': float(row['model3_prob']),
                'ensemble_probability': float(row['ensemble_prob'])
            }
            
            # 시간 정보 추가 (있는 경우)
            if 'start_time_et' in row and pd.notna(row['start_time_et']):
                pred_dict['start_time_et'] = row['start_time_et']
            
            if 'start_time' in row and pd.notna(row['start_time']):
                pred_dict['start_time'] = row['start_time']
                
            if 'venue_name' in row and pd.notna(row['venue_name']):
                pred_dict['venue_name'] = row['venue_name']
                
            if 'day_night' in row and pd.notna(row['day_night']):
                pred_dict['day_night'] = row['day_night']
            
            # 추가 모델들 (4~9) 예측 확률 추가
            for model_num in range(4, 10):
                model_col = f'model{model_num}_prob'
                if model_col in row and pd.notna(row[model_col]):
                    pred_dict[f'model{model_num}_probability'] = float(row[model_col])
            
            # 새로 추가한 고성능 모델들 예측 확률 추가
            for model_name in ['model_rf', 'model_nn', 'model_svm', 'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic', 'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm', 'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost']:
                model_col = f'{model_name}_prob'
                if model_col in row and pd.notna(row[model_col]):
                    pred_dict[f'{model_name}_probability'] = float(row[model_col])
            
            predictions_list.append(pred_dict)
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_list, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        print(f"총 예측 경기 수: {len(predictions_list)}")
        
        return str(output_path) 