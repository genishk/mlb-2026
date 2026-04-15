import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.mlb_ensemble_predictor_totals import MLBEnsemblePredictorTotals
import json
import pandas as pd
from datetime import datetime
import logging
import argparse

ENSEMBLE_CONFIG = {
    'model1': 0.33,
    'model2': 0.33,
    'model3': 0.34,
}


def validate_and_normalize_weights(weights):
    """가중치 검증 및 정규화"""
    active_weights = {k: v for k, v in weights.items() if v > 0}
    
    if not active_weights:
        raise ValueError("최소 하나의 모델에는 0보다 큰 가중치를 설정해야 합니다.")
    
    total_weight = sum(active_weights.values())
    normalized_weights = {k: v/total_weight for k, v in active_weights.items()}
    
    print(f"\n=== 토탈 앙상블 가중치 설정 ===")
    print(f"사용할 모델: {list(normalized_weights.keys())}")
    print("정규화된 가중치:")
    for model, weight in normalized_weights.items():
        print(f"  {model}: {weight:.3f}")
    
    return normalized_weights


def run_for_tag(model_tag: str):
    """특정 태그의 토탈 모델로 앙상블 예측 실행"""
    print(f"\n{'='*60}")
    print(f"  [{model_tag.upper()}] 토탈 앙상블 예측 시작")
    print(f"{'='*60}")
    
    predictor = MLBEnsemblePredictorTotals()
    
    loaded_models = predictor.load_latest_models(model_tag=model_tag)
    print(f"\n총 {len(loaded_models)}개 토탈 모델 로드 완료")
    
    data = predictor.load_prediction_data()
    print(f"예측할 경기 수: {len(data)}")
    
    weights = validate_and_normalize_weights(ENSEMBLE_CONFIG)
    
    predictions_df = predictor.predict_games(data, weights=weights)
    output_path = predictor.save_predictions(predictions_df, model_tag=model_tag)
    
    print("\n=== 상세 토탈 예측 결과 (시간순 정렬) ===")
    for _, row in predictions_df.iterrows():
        time_info = ""
        if 'start_time_et' in row and pd.notna(row['start_time_et']) and row['start_time_et'] != 'Unknown':
            time_info = f" ({row['start_time_et']})"
        
        print(f"\n{row['date']}{time_info} - {row['home_team']} vs {row['away_team']}")
        print(f"  앙상블 예측 총 득점: {row['predicted_total']:.1f}")
        
        print("  메인 모델 (1~3):")
        for i in range(1, 4):
            col = f'model{i}_total'
            if col in row and pd.notna(row[col]):
                print(f"    Model {i}: {row[col]:.1f}")
        
        print("  추가 모델 (4~9):")
        for i in range(4, 10):
            col = f'model{i}_total'
            if col in row and pd.notna(row[col]):
                print(f"    Model {i}: {row[col]:.1f}")
        
        all_new = [
            ('model_rf', 'Random Forest'), ('model_nn', 'Neural Network'), ('model_svm', 'SVM'),
            ('model_advanced_lgbm_basic', 'Adv. LGBM Basic'), ('model_advanced_lgbm', 'Adv. LGBM'),
            ('model_advanced_catboost_basic', 'Adv. CatBoost Basic'), ('model_advanced_catboost', 'Adv. CatBoost'),
            ('model_advanced_xgboost_basic', 'Adv. XGBoost Basic'), ('model_advanced_xgboost', 'Adv. XGBoost'),
            ('model_advanced_nn', 'Adv. NN'), ('model_advanced_rf', 'Adv. RF'), ('model_advanced_svm', 'Adv. SVM'),
            ('model1_extended_lgbm', 'Ext. LGBM'), ('model2_extended_catboost', 'Ext. CatBoost'),
            ('model3_extended_xgboost', 'Ext. XGBoost'),
        ]
        print("  확장/고급 모델:")
        for model_key, model_name in all_new:
            col = f'{model_key}_total'
            if col in row and pd.notna(row[col]):
                print(f"    {model_name}: {row[col]:.1f}")
    
    totals = predictions_df['predicted_total']
    print(f"\n=== 토탈 예측 통계 ===")
    print(f"  예측 범위: {totals.min():.1f} ~ {totals.max():.1f}")
    print(f"  평균: {totals.mean():.1f}")
    print(f"  중앙값: {totals.median():.1f}")
    
    print(f"\n=== [{model_tag.upper()}] 토탈 예측 저장 완료: {output_path} ===")
    return output_path


def main():
    """MLB 토탈 앙상블 예측 모델 실행"""
    parser = argparse.ArgumentParser(description='MLB 토탈 앙상블 예측 실행')
    parser.add_argument('--model-tag', type=str, default='active',
                       choices=['active', 'shadow', 'both'],
                       help='모델 태그 (active, shadow, both)')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.model_tag == 'both':
            for tag in ['active', 'shadow']:
                try:
                    run_for_tag(tag)
                except Exception as e:
                    print(f"\n[{tag.upper()}] 토탈 예측 중 오류 발생: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
        else:
            run_for_tag(args.model_tag)
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
