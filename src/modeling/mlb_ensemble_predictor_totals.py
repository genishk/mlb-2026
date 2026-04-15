import sys
from pathlib import Path

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

from src.modeling.mlb_model1_totals import MLBTotalsModel1
from src.modeling.mlb_model2_totals import MLBTotalsModel2
from src.modeling.mlb_model3_totals import MLBTotalsModel3
from src.modeling.mlb_model4_totals import MLBTotalsModel4
from src.modeling.mlb_model5_totals import MLBTotalsModel5
from src.modeling.mlb_model6_totals import MLBTotalsModel6
from src.modeling.mlb_model7_totals import MLBTotalsModel7
from src.modeling.mlb_model8_totals import MLBTotalsModel8
from src.modeling.mlb_model9_totals import MLBTotalsModel9
from src.modeling.mlb_model_rf_totals import MLBTotalsModelRF
from src.modeling.mlb_model_nn_totals import MLBTotalsModelNN
from src.modeling.mlb_model_svm_totals import MLBTotalsModelSVM
from src.modeling.mlb_model_advanced_catboost_basic_totals import MLBTotalsAdvancedCatBoostBasicModel
from src.modeling.mlb_model_advanced_catboost_totals import MLBTotalsAdvancedCatBoostModel
from src.modeling.mlb_model_advanced_lgbm_basic_totals import MLBTotalsAdvancedLGBMBasicModel
from src.modeling.mlb_model_advanced_lgbm_totals import MLBTotalsAdvancedBettingModel as MLBTotalsAdvancedLGBMModel
from src.modeling.mlb_model_advanced_nn_totals import MLBTotalsAdvancedNeuralNetworkModel
from src.modeling.mlb_model_advanced_rf_totals import MLBTotalsAdvancedRandomForestModel
from src.modeling.mlb_model_advanced_svm_totals import MLBTotalsAdvancedSVMModel
from src.modeling.mlb_model_advanced_xgboost_basic_totals import MLBTotalsAdvancedXGBoostBasicModel
from src.modeling.mlb_model_advanced_xgboost_totals import MLBTotalsAdvancedXGBoostModel
from src.modeling.mlb_model1_extended_lgbm_totals import MLBTotalsModel1ExtendedLGBM
from src.modeling.mlb_model2_extended_catboost_totals import MLBTotalsModel2ExtendedCatBoost
from src.modeling.mlb_model3_extended_xgboost_totals import MLBTotalsModel3ExtendedXGBoost


class MLBEnsemblePredictorTotals:
    """MLB 총 득점 예측을 위한 앙상블 예측기 (회귀)
    
    24개 토탈 모델의 예측을 결합하여 총 득점을 예측합니다.
    메인 앙상블은 모델 1~3의 가중 평균을 사용합니다.
    """
    
    def __init__(self):
        self.models = {
            'model1': MLBTotalsModel1(),
            'model2': MLBTotalsModel2(),
            'model3': MLBTotalsModel3()
        }
        
        self.additional_models = {
            'model4': MLBTotalsModel4(),
            'model5': MLBTotalsModel5(),
            'model6': MLBTotalsModel6(),
            'model7': MLBTotalsModel7(),
            'model8': MLBTotalsModel8(),
            'model9': MLBTotalsModel9()
        }
        
        self.new_models = {
            'model_rf': MLBTotalsModelRF(),
            'model_nn': MLBTotalsModelNN(),
            'model_svm': MLBTotalsModelSVM(),
            'model_advanced_catboost_basic': MLBTotalsAdvancedCatBoostBasicModel(),
            'model_advanced_catboost': MLBTotalsAdvancedCatBoostModel(),
            'model_advanced_lgbm_basic': MLBTotalsAdvancedLGBMBasicModel(),
            'model_advanced_lgbm': MLBTotalsAdvancedLGBMModel(),
            'model_advanced_nn': MLBTotalsAdvancedNeuralNetworkModel(),
            'model_advanced_rf': MLBTotalsAdvancedRandomForestModel(),
            'model_advanced_svm': MLBTotalsAdvancedSVMModel(),
            'model_advanced_xgboost_basic': MLBTotalsAdvancedXGBoostBasicModel(),
            'model_advanced_xgboost': MLBTotalsAdvancedXGBoostModel(),
            'model1_extended_lgbm': MLBTotalsModel1ExtendedLGBM(),
            'model2_extended_catboost': MLBTotalsModel2ExtendedCatBoost(),
            'model3_extended_xgboost': MLBTotalsModel3ExtendedXGBoost()
        }
        
        self.model_paths = {}
        self.project_root = Path(__file__).parent.parent.parent
        self.current_tag = None
        
    def _find_tagged_file(self, models_dir: Path, pattern_base: str, model_tag: str, 
                         ext: str = '.joblib', exclude_keywords: Optional[List[str]] = None) -> Optional[Path]:
        """태그 기반 파일 찾기 (fallback: 타임스탬프 버전)"""
        tagged = models_dir / f"{pattern_base}_{model_tag}{ext}"
        if tagged.exists():
            return tagged
        ts_files = list(models_dir.glob(f"{pattern_base}_*{ext}"))
        ts_files = [f for f in ts_files if 'active' not in f.name and 'shadow' not in f.name]
        if exclude_keywords:
            for kw in exclude_keywords:
                ts_files = [f for f in ts_files if kw not in f.name]
        if ts_files:
            return max(ts_files, key=lambda x: x.stat().st_mtime)
        return None
        
    def load_latest_models(self, model_tag: str = 'active') -> Dict[str, str]:
        """각 토탈 모델의 가장 최근 저장 파일을 로드"""
        models_dir = self.project_root / "src" / "models" / "totals_models"
        self.current_tag = model_tag
        
        if not models_dir.exists():
            raise FileNotFoundError(f"토탈 모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        
        loaded_models = {}
        
        print(f"\n=== [{model_tag.upper()}] 토탈 모델 로드 시작 ===")
        
        for model_type in ['model1', 'model2', 'model3']:
            model_num = model_type[-1]
            model_file = self._find_tagged_file(models_dir, f'mlb_totals_model{model_num}', model_tag)
            feature_file = self._find_tagged_file(models_dir, f'mlb_totals_features{model_num}', model_tag, ext='.json')
            
            if model_file and feature_file:
                self.models[model_type].load_model(str(model_file), str(feature_file))
                loaded_models[model_type] = str(model_file)
                print(f"{model_type} 로드 완료: {model_file.name}")
            else:
                print(f"경고: {model_type}의 토탈 모델 또는 특성 파일을 찾을 수 없습니다.")
        
        for model_type in ['model4', 'model5', 'model6', 'model7', 'model8', 'model9']:
            model_num = model_type[-1]
            model_file = self._find_tagged_file(models_dir, f'mlb_totals_model{model_num}', model_tag)
            feature_file = self._find_tagged_file(models_dir, f'mlb_totals_features{model_num}', model_tag, ext='.json')
            
            if model_file and feature_file:
                self.additional_models[model_type].load_model(str(model_file), str(feature_file))
                loaded_models[model_type] = str(model_file)
                print(f"{model_type} 로드 완료: {model_file.name}")
            else:
                print(f"경고: {model_type}의 토탈 모델 또는 특성 파일을 찾을 수 없습니다.")
        
        new_model_configs = {
            'model_rf': {
                'model_base': 'mlb_totals_model_rf', 'feature_base': 'mlb_totals_features_rf',
                'scaler_base': None, 'selector_base': None, 'load_type': 'basic'
            },
            'model_nn': {
                'model_base': 'mlb_totals_model_nn', 'feature_base': 'mlb_totals_features_nn',
                'scaler_base': 'mlb_totals_scaler_nn', 'selector_base': None, 'load_type': 'scaler'
            },
            'model_svm': {
                'model_base': 'mlb_totals_model_svm', 'feature_base': 'mlb_totals_features_svm',
                'scaler_base': 'mlb_totals_scaler_svm', 'selector_base': None, 'load_type': 'scaler'
            },
            'model_advanced_catboost_basic': {
                'model_base': 'mlb_totals_advanced_catboost_basic', 'feature_base': 'mlb_totals_advanced_catboost_basic_features',
                'scaler_base': 'mlb_totals_advanced_catboost_basic_scaler', 'selector_base': 'mlb_totals_advanced_catboost_basic_selector',
                'load_type': 'advanced'
            },
            'model_advanced_catboost': {
                'model_base': 'mlb_totals_advanced_catboost', 'feature_base': 'mlb_totals_advanced_catboost_features',
                'scaler_base': 'mlb_totals_advanced_catboost_scaler', 'selector_base': 'mlb_totals_advanced_catboost_selector',
                'load_type': 'advanced', 'exclude_keywords': ['basic']
            },
            'model_advanced_lgbm_basic': {
                'model_base': 'mlb_totals_advanced_lgbm_basic', 'feature_base': 'mlb_totals_advanced_lgbm_basic_features',
                'scaler_base': 'mlb_totals_advanced_lgbm_basic_scaler', 'selector_base': 'mlb_totals_advanced_lgbm_basic_selector',
                'load_type': 'advanced'
            },
            'model_advanced_lgbm': {
                'model_base': 'mlb_totals_advanced_lgbm', 'feature_base': 'mlb_totals_advanced_lgbm_features',
                'scaler_base': 'mlb_totals_advanced_lgbm_scaler', 'selector_base': 'mlb_totals_advanced_lgbm_selector',
                'load_type': 'advanced', 'exclude_keywords': ['basic']
            },
            'model_advanced_nn': {
                'model_base': 'mlb_totals_advanced_nn', 'feature_base': 'mlb_totals_advanced_nn_features',
                'scaler_base': 'mlb_totals_advanced_nn_scaler', 'selector_base': 'mlb_totals_advanced_nn_selector',
                'load_type': 'advanced'
            },
            'model_advanced_rf': {
                'model_base': 'mlb_totals_advanced_rf', 'feature_base': 'mlb_totals_advanced_rf_features',
                'scaler_base': 'mlb_totals_advanced_rf_scaler', 'selector_base': 'mlb_totals_advanced_rf_selector',
                'load_type': 'advanced'
            },
            'model_advanced_svm': {
                'model_base': 'mlb_totals_advanced_svm', 'feature_base': 'mlb_totals_advanced_svm_features',
                'scaler_base': 'mlb_totals_advanced_svm_scaler', 'selector_base': 'mlb_totals_advanced_svm_selector',
                'load_type': 'advanced'
            },
            'model_advanced_xgboost_basic': {
                'model_base': 'mlb_totals_advanced_xgboost_basic', 'feature_base': 'mlb_totals_advanced_xgboost_basic_features',
                'scaler_base': 'mlb_totals_advanced_xgboost_basic_scaler', 'selector_base': 'mlb_totals_advanced_xgboost_basic_selector',
                'load_type': 'advanced'
            },
            'model_advanced_xgboost': {
                'model_base': 'mlb_totals_advanced_xgboost', 'feature_base': 'mlb_totals_advanced_xgboost_features',
                'scaler_base': 'mlb_totals_advanced_xgboost_scaler', 'selector_base': 'mlb_totals_advanced_xgboost_selector',
                'load_type': 'advanced', 'exclude_keywords': ['basic']
            },
            'model1_extended_lgbm': {
                'model_base': 'mlb_totals_model1_extended_lgbm', 'feature_base': 'mlb_totals_features1_extended_lgbm',
                'scaler_base': None, 'selector_base': None, 'load_type': 'basic'
            },
            'model2_extended_catboost': {
                'model_base': 'mlb_totals_model2_extended_catboost', 'feature_base': 'mlb_totals_features2_extended_catboost',
                'scaler_base': None, 'selector_base': None, 'load_type': 'basic'
            },
            'model3_extended_xgboost': {
                'model_base': 'mlb_totals_model3_extended_xgboost', 'feature_base': 'mlb_totals_features3_extended_xgboost',
                'scaler_base': None, 'selector_base': None, 'load_type': 'basic'
            },
        }
        
        for model_type, cfg in new_model_configs.items():
            excl = cfg.get('exclude_keywords')
            model_file = self._find_tagged_file(models_dir, cfg['model_base'], model_tag, exclude_keywords=excl)
            feature_file = self._find_tagged_file(models_dir, cfg['feature_base'], model_tag, ext='.json', exclude_keywords=excl)
            
            if not model_file or not feature_file:
                print(f"경고: {model_type}의 토탈 모델 또는 특성 파일을 찾을 수 없습니다.")
                continue
            
            scaler_path = None
            if cfg['scaler_base']:
                scaler_file = self._find_tagged_file(models_dir, cfg['scaler_base'], model_tag, exclude_keywords=excl)
                if scaler_file:
                    scaler_path = str(scaler_file)
            
            selector_path = None
            if cfg['selector_base']:
                selector_file = self._find_tagged_file(models_dir, cfg['selector_base'], model_tag, exclude_keywords=excl)
                if selector_file:
                    selector_path = str(selector_file)
            
            if cfg['load_type'] == 'basic':
                self.new_models[model_type].load_model(str(model_file), str(feature_file))
            elif cfg['load_type'] == 'scaler':
                self.new_models[model_type].load_model(str(model_file), scaler_path, str(feature_file))
            elif cfg['load_type'] == 'advanced':
                self.new_models[model_type].load_model(str(model_file), scaler_path, selector_path, str(feature_file))
            
            loaded_models[model_type] = str(model_file)
            print(f"{model_type} 로드 완료: {model_file.name}")
        
        return loaded_models
    
    def load_prediction_data(self) -> List[Dict]:
        """가장 최근의 예측 데이터를 로드합니다."""
        data_dir = self.project_root / "data" / "prediction"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"예측 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        
        prediction_files = list(data_dir.glob("mlb_prediction_data_*.json"))
        if not prediction_files:
            raise FileNotFoundError("예측 데이터 파일을 찾을 수 없습니다.")
        
        latest_prediction_file = max(prediction_files, key=lambda x: x.name)
        print(f"예측 데이터 파일 로드: {latest_prediction_file.name}")
        
        with open(latest_prediction_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
        
        print(f"예측 데이터 로드 완료: {len(prediction_data)}개 경기")
        return prediction_data
    
    def predict_games(self, game_data: List[Dict], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """앙상블 총 득점 예측 수행"""
        if not weights:
            weights = {'model1': 0.33, 'model2': 0.33, 'model3': 0.34}
            
        all_predictions = []
        
        for model_name, model in self.models.items():
            predictions = model.predict(game_data)
            for i, pred in enumerate(predictions):
                if len(all_predictions) <= i:
                    all_predictions.append({})
                all_predictions[i].update({
                    f'{model_name}_total': pred['predicted_total'],
                    'date': pred['date'],
                    'home_team': pred['home_team_name'],
                    'away_team': pred['away_team_name']
                })
        
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
        
        for model_name, model in self.additional_models.items():
            predictions = model.predict(game_data)
            for i, pred in enumerate(predictions):
                if i < len(all_predictions):
                    all_predictions[i][f'{model_name}_total'] = pred['predicted_total']
        
        for model_name, model in self.new_models.items():
            try:
                predictions = model.predict(game_data)
                for i, pred in enumerate(predictions):
                    if i < len(all_predictions):
                        all_predictions[i][f'{model_name}_total'] = pred['predicted_total']
                print(f"{model_name} 예측 완료")
            except Exception as e:
                print(f"경고: {model_name} 예측 중 오류 발생: {e}")
                for i in range(len(all_predictions)):
                    all_predictions[i][f'{model_name}_total'] = np.nan
        
        predictions_df = pd.DataFrame(all_predictions)
        
        ensemble_totals = []
        for _, row in predictions_df.iterrows():
            weighted_sum = 0
            for model_name, weight in weights.items():
                col = f'{model_name}_total'
                if col in row and pd.notna(row[col]):
                    weighted_sum += row[col] * weight
            ensemble_totals.append(weighted_sum)
        
        predictions_df['ensemble_total'] = ensemble_totals
        predictions_df['predicted_total'] = ensemble_totals
        
        def get_sort_time(row):
            start_time = row.get('start_time_et', 'Unknown')
            if pd.isna(start_time) or start_time == 'Unknown':
                return '9999-12-31 23:59:59'
            try:
                time_part = str(start_time).split(' EDT')[0].split(' EST')[0]
                return time_part
            except:
                return '9999-12-31 23:59:59'
        
        predictions_df['sort_time'] = predictions_df.apply(get_sort_time, axis=1)
        predictions_df = predictions_df.sort_values('sort_time').drop('sort_time', axis=1)
        
        return predictions_df
    
    def save_predictions(self, predictions: pd.DataFrame, model_tag: Optional[str] = None) -> str:
        """예측 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = model_tag or self.current_tag
        pred_dir = self.project_root / "src" / "predictions"
        pred_dir.mkdir(exist_ok=True, parents=True)
        
        if tag:
            output_path = pred_dir / f"mlb_totals_ensemble_predictions_{timestamp}_{tag}.json"
        else:
            output_path = pred_dir / f"mlb_totals_ensemble_predictions_{timestamp}.json"
        
        predictions_list = []
        for _, row in predictions.iterrows():
            pred_dict = {
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'predicted_total': float(row['predicted_total']),
                'model1_total': float(row['model1_total']),
                'model2_total': float(row['model2_total']),
                'model3_total': float(row['model3_total']),
                'ensemble_total': float(row['ensemble_total'])
            }
            
            if 'start_time_et' in row and pd.notna(row['start_time_et']):
                pred_dict['start_time_et'] = row['start_time_et']
            if 'start_time' in row and pd.notna(row['start_time']):
                pred_dict['start_time'] = row['start_time']
            if 'venue_name' in row and pd.notna(row['venue_name']):
                pred_dict['venue_name'] = row['venue_name']
            if 'day_night' in row and pd.notna(row['day_night']):
                pred_dict['day_night'] = row['day_night']
            
            for model_num in range(4, 10):
                model_col = f'model{model_num}_total'
                if model_col in row and pd.notna(row[model_col]):
                    pred_dict[f'model{model_num}_total'] = float(row[model_col])
            
            all_new_models = [
                'model_rf', 'model_nn', 'model_svm',
                'model_advanced_catboost_basic', 'model_advanced_catboost',
                'model_advanced_lgbm_basic', 'model_advanced_lgbm',
                'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                'model_advanced_xgboost_basic', 'model_advanced_xgboost',
                'model1_extended_lgbm', 'model2_extended_catboost', 'model3_extended_xgboost'
            ]
            for model_name in all_new_models:
                model_col = f'{model_name}_total'
                if model_col in row and pd.notna(row[model_col]):
                    pred_dict[f'{model_name}_total'] = float(row[model_col])
            
            predictions_list.append(pred_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_list, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 토탈 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        print(f"총 예측 경기 수: {len(predictions_list)}")
        
        return str(output_path)
