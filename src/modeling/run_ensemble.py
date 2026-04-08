import sys
from pathlib import Path

# Add project root to system path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.mlb_ensemble_predictor import MLBEnsemblePredictor
import json
import pandas as pd
from datetime import datetime
import logging

# ========================================
# 앙상블 설정 - 여기서 쉽게 조절 가능!
# ========================================

# 앙상블에 사용할 모델과 가중치 설정

ENSEMBLE_CONFIG = {
    'model7': 1.000,
}


# ENSEMBLE_CONFIG = {
#     # 기존 4개 모델 (각각 0.125)
#     'model1': 0.25,   # LightGBM
#     'model2': 0.25,   # CatBoost  
#     'model3': 0.25,   # XGBoost
#     'model9': 0.25   # 추가 모델
    
#     # 새로 추가된 3개 확장 모델 (각각 0.15 - 더 높은 가중치)
#     # 'model1_extended_lgbm': 0.15,      # LightGBM 확장 (123개 특성)
#     # 'model2_extended_catboost': 0.15,  # CatBoost 확장 (123개 특성)
#     # 'model3_extended_xgboost': 0.15,   # XGBoost 확장 (123개 특성)
    
#     # 다른 모델들은 주석 처리
#     # 'model4': 0.08,   # 추가 모델들
#     # 'model5': 0.08,
#     # 'model6': 0.08,
#     # 'model7': 0.08,
#     # 'model8': 0.08,
#     # 'model_advanced_catboost_basic': 0.125,  # 고급 CatBoost Basic
#     # 'model_advanced_catboost': 0.125,  # 고급 CatBoost
#     # 'model_advanced_lgbm_basic': 0.125,  # 고급 LightGBM Basic
#     # 'model_advanced_lgbm': 0.125  # 고급 LightGBM
#     # 'model_advanced_nn': 0.125,  # 고급 Neural Network
#     # 'model_advanced_rf': 0.125,  # 고급 Random Forest
#     # 'model_advanced_svm': 0.125,  # 고급 SVM
#     # 'model_advanced_xgboost_basic': 0.125,  # 고급 XGBoost Basic
#     # 'model_advanced_xgboost': 0.125  # 고급 XGBoost
    
#     # # 새로 추가한 3개 고성능 모델 (더 높은 가중치)
#     # 'model_rf': 0.15,   # Random Forest (88% 정확도)
#     # 'model_nn': 0.15,   # Neural Network (94% 정확도)
#     # 'model_svm': 0.20,  # SVM (96% 정확도) - 최고 성능으로 가장 높은 가중치
# }

# 가중치 합이 1.0인지 자동 검증 및 정규화
def validate_and_normalize_weights(weights):
    """가중치 검증 및 정규화"""
    # 0이 아닌 가중치만 필터링
    active_weights = {k: v for k, v in weights.items() if v > 0}
    
    if not active_weights:
        raise ValueError("최소 하나의 모델에는 0보다 큰 가중치를 설정해야 합니다.")
    
    # 가중치 합 계산
    total_weight = sum(active_weights.values())
    
    # 정규화 (합이 1.0이 되도록)
    normalized_weights = {k: v/total_weight for k, v in active_weights.items()}
    
    print(f"\n=== 앙상블 가중치 설정 ===")
    print(f"사용할 모델: {list(normalized_weights.keys())}")
    print("정규화된 가중치:")
    for model, weight in normalized_weights.items():
        print(f"  {model}: {weight:.3f}")
    print(f"가중치 합: {sum(normalized_weights.values()):.3f}")
    
    return normalized_weights

def main():
    """MLB 앙상블 예측 모델 실행 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 경로 디버깅
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        
        print(f"\n=== 경로 디버깅 ===")
        print(f"스크립트 경로: {script_path}")
        print(f"프로젝트 루트: {project_root}")
        
        # 예측기 초기화
        predictor = MLBEnsemblePredictor()
        
        # 모든 모델 로드
        loaded_models = predictor.load_latest_models()
        print(f"\n총 {len(loaded_models)}개 모델 로드 완료")
        
        # 예측 데이터 로드
        data = predictor.load_prediction_data()
        print(f"예측할 경기 수: {len(data)}")
        
        # 앙상블 가중치 설정 및 검증
        weights = validate_and_normalize_weights(ENSEMBLE_CONFIG)
        
        # 앙상블 예측 수행
        try:
            predictions_df = predictor.predict_games(data, weights=weights)
            
            # 예측 결과 저장
            output_path = predictor.save_predictions(predictions_df)
            
            # 모델별 예측 확률 스케일 분석
            scale_analysis = predictor.analyze_probability_scales(predictions_df)
            print("\n=== 모델별 예측 확률 스케일 분석 ===")
            for model, stats in scale_analysis.items():
                print(f"\n{model}:")
                print(f"  범위: {stats['min']:.3f} ~ {stats['max']:.3f}")
                print(f"  평균: {stats['mean']:.3f} (표준편차: {stats['std']:.3f})")
                print(f"  중앙값: {stats['median']:.3f}")
                print(f"  왜도: {stats['skew']:.3f}")
                print("  확률 구간별 경기 수:")
                for range_name, count in stats['ranges'].items():
                    print(f"    {range_name}: {count}경기")
            
            # 예측 신뢰도 통계
            high_confidence = (predictions_df['win_probability'] >= 0.7).sum()
            medium_confidence = ((predictions_df['win_probability'] >= 0.6) & 
                               (predictions_df['win_probability'] < 0.7)).sum()
            low_confidence = (predictions_df['win_probability'] < 0.6).sum()
            
            print("\n=== 앙상블 예측 신뢰도 분석 ===")
            print(f"높은 신뢰도 (70% 이상): {high_confidence}경기")
            print(f"중간 신뢰도 (60-70%): {medium_confidence}경기")
            print(f"낮은 신뢰도 (60% 미만): {low_confidence}경기")
            
            # 모델 간 예측 일치도 분석
            agreement_analysis = predictor.analyze_model_agreement(predictions_df)
            print("\n=== 모델 간 예측 일치도 ===")
            print(f"완전 일치 (3개 모델 동일): {agreement_analysis['full_agreement']}경기")
            print(f"부분 일치 (2개 모델 동일): {agreement_analysis['partial_agreement']}경기")
            print("\n모델 간 상관관계:")
            for pair, corr in agreement_analysis['model_correlations'].items():
                print(f"  {pair}: {corr:.3f}")
            
            # 상세 예측 결과 출력
            print("\n=== 상세 예측 결과 (시간순 정렬) ===")
            for _, row in predictions_df.iterrows():
                # 시간 정보 표시
                time_info = ""
                if 'start_time_et' in row and pd.notna(row['start_time_et']) and row['start_time_et'] != 'Unknown':
                    time_info = f" ({row['start_time_et']})"
                
                venue_info = ""
                if 'venue_name' in row and pd.notna(row['venue_name']):
                    venue_info = f" @ {row['venue_name']}"
                
                day_night_info = ""
                if 'day_night' in row and pd.notna(row['day_night']):
                    day_night_info = f" [{row['day_night']}]"
                
                print(f"\n{row['date']}{time_info} - {row['home_team']} vs {row['away_team']}{venue_info}{day_night_info}")
                print(f"예측 승자: {row['predicted_winner']}")
                print(f"승리 확률: {row['win_probability']:.3f}")
                
                print("기존 모델 예측 (홈팀 승리 확률):")
                print(f"  Model 1 (LightGBM): {row['model1_prob']:.3f}")
                print(f"  Model 2 (CatBoost): {row['model2_prob']:.3f}")
                print(f"  Model 3 (XGBoost): {row['model3_prob']:.3f}")
                
                # 추가 모델들 (4~9) 예측 확률 출력
                additional_models = []
                for model_num in range(4, 10):
                    model_col = f'model{model_num}_prob'
                    if model_col in row and pd.notna(row[model_col]):
                        additional_models.append(f"  Model {model_num}: {row[model_col]:.3f}")
                
                if additional_models:
                    print("기존 추가 모델들:")
                    for model_info in additional_models:
                        print(model_info)
                
                # 새로 추가한 고성능 모델들
                print("새로 추가한 고성능 모델들:")
                if 'model_rf_prob' in row and pd.notna(row['model_rf_prob']):
                    print(f"  Random Forest (88%): {row['model_rf_prob']:.3f}")
                if 'model_nn_prob' in row and pd.notna(row['model_nn_prob']):
                    print(f"  Neural Network (94%): {row['model_nn_prob']:.3f}")
                if 'model_svm_prob' in row and pd.notna(row['model_svm_prob']):
                    print(f"  SVM (96%): {row['model_svm_prob']:.3f}")
                
                # 새로 추가된 확장 모델들 (123개 특성)
                print("확장 모델들 (123개 특성):")
                if 'model1_extended_lgbm_prob' in row and pd.notna(row['model1_extended_lgbm_prob']):
                    print(f"  LightGBM 확장 (98%): {row['model1_extended_lgbm_prob']:.3f}")
                if 'model2_extended_catboost_prob' in row and pd.notna(row['model2_extended_catboost_prob']):
                    print(f"  CatBoost 확장 (100%): {row['model2_extended_catboost_prob']:.3f}")
                if 'model3_extended_xgboost_prob' in row and pd.notna(row['model3_extended_xgboost_prob']):
                    print(f"  XGBoost 확장 (100%): {row['model3_extended_xgboost_prob']:.3f}")
                
                # 고급 모델들
                print("고급 모델들:")
                if 'model_advanced_catboost_basic_prob' in row and pd.notna(row['model_advanced_catboost_basic_prob']):
                    print(f"  Advanced CatBoost Basic: {row['model_advanced_catboost_basic_prob']:.3f}")
                if 'model_advanced_catboost_prob' in row and pd.notna(row['model_advanced_catboost_prob']):
                    print(f"  Advanced CatBoost: {row['model_advanced_catboost_prob']:.3f}")
                if 'model_advanced_lgbm_basic_prob' in row and pd.notna(row['model_advanced_lgbm_basic_prob']):
                    print(f"  Advanced LightGBM Basic: {row['model_advanced_lgbm_basic_prob']:.3f}")
                if 'model_advanced_lgbm_prob' in row and pd.notna(row['model_advanced_lgbm_prob']):
                    print(f"  Advanced LightGBM: {row['model_advanced_lgbm_prob']:.3f}")
                if 'model_advanced_nn_prob' in row and pd.notna(row['model_advanced_nn_prob']):
                    print(f"  Advanced Neural Network: {row['model_advanced_nn_prob']:.3f}")
                if 'model_advanced_rf_prob' in row and pd.notna(row['model_advanced_rf_prob']):
                    print(f"  Advanced Random Forest: {row['model_advanced_rf_prob']:.3f}")
                if 'model_advanced_svm_prob' in row and pd.notna(row['model_advanced_svm_prob']):
                    print(f"  Advanced SVM: {row['model_advanced_svm_prob']:.3f}")
                if 'model_advanced_xgboost_basic_prob' in row and pd.notna(row['model_advanced_xgboost_basic_prob']):
                    print(f"  Advanced XGBoost Basic: {row['model_advanced_xgboost_basic_prob']:.3f}")
                if 'model_advanced_xgboost_prob' in row and pd.notna(row['model_advanced_xgboost_prob']):
                    print(f"  Advanced XGBoost: {row['model_advanced_xgboost_prob']:.3f}")
                
                print(f"최종 앙상블: {row['ensemble_prob']:.3f}")
            
        except Exception as e:
            print(f"\n예측 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 