import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List
import subprocess
import time
import json
import argparse


class MLBTotalsPipeline:
    """MLB 토탈(Over/Under) 파이프라인
    
    기존 머니라인 파이프라인과 동일한 훈련/예측 데이터를 공유합니다.
    데이터 수집/처리 단계는 기본적으로 주석 처리되어 있으며,
    독립 실행이 필요하면 주석 해제하면 됩니다.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_logging()
        
        self.dirs = {
            'data': self.project_root / 'data',
            'match_data': self.project_root / 'data' / 'match_data',
            'records': self.project_root / 'data' / 'records',
            'training': self.project_root / 'data' / 'training',
            'prediction': self.project_root / 'data' / 'prediction',
            'totals_models': self.project_root / 'src' / 'models' / 'totals_models',
            'predictions': self.project_root / 'src' / 'predictions',
            'odds': self.project_root / 'src' / 'odds' / 'data' / 'odds',
            'matched': self.project_root / 'src' / 'odds' / 'data' / 'matched',
            'logs': self.project_root / 'logs',
            'src_data_processing': self.project_root / 'src' / 'data_processing',
            'src_modeling': self.project_root / 'src' / 'modeling',
            'src_odds': self.project_root / 'src' / 'odds'
        }
        
        for dir_path in self.dirs.values():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Error creating directory {dir_path}: {str(e)}")
                raise
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'mlb_totals_pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('MLBTotalsPipeline')
    
    def check_dependencies(self) -> bool:
        """필요한 라이브러리 체크"""
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'lightgbm': 'lightgbm',
            'catboost': 'catboost',
            'xgboost': 'xgboost',
            'sklearn': 'scikit-learn',
            'requests': 'requests',
            'pytz': 'pytz'
        }
        
        missing_packages = []
        for package, pip_name in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(pip_name)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            return False
        return True
    
    def validate_file_structure(self) -> bool:
        """필요한 토탈 파일들의 존재 여부 확인"""
        required_files = [
            'src/modeling/run_ensemble_totals.py',
            'src/modeling/mlb_ensemble_predictor_totals.py',
            'src/odds/mlb_odds_fetcher_totals_FanDuel.py',
            'src/odds/odds_matcher_totals.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        return True
    
    def check_data_exists(self) -> bool:
        """훈련/예측 데이터가 존재하는지 확인"""
        training_files = list(self.dirs['training'].glob("mlb_training_data_*.json"))
        prediction_files = list(self.dirs['prediction'].glob("mlb_prediction_data_*.json"))
        
        if not training_files:
            self.logger.error("훈련 데이터가 없습니다. mlb_betting_pipeline.py를 먼저 실행하거나 데이터 수집 단계 주석을 해제하세요.")
            return False
        if not prediction_files:
            self.logger.error("예측 데이터가 없습니다. mlb_betting_pipeline.py를 먼저 실행하거나 데이터 수집 단계 주석을 해제하세요.")
            return False
        
        latest_training = max(training_files, key=lambda x: x.name)
        latest_prediction = max(prediction_files, key=lambda x: x.name)
        self.logger.info(f"훈련 데이터: {latest_training.name}")
        self.logger.info(f"예측 데이터: {latest_prediction.name}")
        return True
    
    def run_script(self, script_name: str, description: str, extra_args: Optional[List[str]] = None) -> bool:
        """Python 스크립트 실행"""
        try:
            self.logger.info(f"Starting: {description}")
            
            if script_name in ['mlb_match_analyzer.py', 'mlb_record_processor_streaming.py',
                              'mlb_training_data_processor_v2.py', 'mlb_prediction_data_processor_v2.py']:
                script_path = self.project_root / 'src' / 'data_processing' / script_name
            elif script_name in ['mlb_odds_fetcher_totals_FanDuel.py', 'odds_matcher_totals.py']:
                script_path = self.project_root / 'src' / 'odds' / script_name
            elif script_name.startswith('mlb_model') or script_name.startswith('run_ensemble'):
                script_path = self.project_root / 'src' / 'modeling' / script_name
            else:
                script_path = self.project_root / script_name
            
            if not script_path.exists():
                self.logger.error(f"Script not found: {script_path}")
                return False
            
            env = os.environ.copy()
            env['PROJECT_ROOT'] = str(self.project_root)
            env['PYTHONPATH'] = str(self.project_root)
            env['DATA_DIR'] = str(self.project_root / 'data')
            
            cmd = [sys.executable, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(self.project_root)
            )
            
            if result.stdout:
                self.logger.info(f"Output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Stderr: {result.stderr}")
            
            self.logger.info(f"Completed: {description}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in {description}: {str(e)}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error in {description}: {str(e)}")
            return False


def is_jupyter():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def main():
    if is_jupyter():
        class Args:
            model_tag = 'both'
            skip_data_check = False
        args = Args()
    else:
        parser = argparse.ArgumentParser(description='MLB Totals Pipeline')
        parser.add_argument('--model-tag', type=str, default='both',
                           choices=['active', 'shadow', 'both'],
                           help='Model tag for prediction (default: both)')
        parser.add_argument('--skip-data-check', action='store_true',
                           help='Skip data existence check')
        args = parser.parse_args()
    
    pipeline = MLBTotalsPipeline()
    
    try:
        if not pipeline.check_dependencies():
            print("Missing dependencies. Check log for details.")
            return False
        if not pipeline.validate_file_structure():
            print("Missing required files. Check log for details.")
            return False
        if not args.skip_data_check:
            if not pipeline.check_data_exists():
                print("데이터가 없습니다. mlb_betting_pipeline.py를 먼저 실행하거나 데이터 수집 단계 주석을 해제하세요.")
                return False
        
        model_tag = args.model_tag
        print(f"\n{'='*60}")
        print(f"  MLB Totals Pipeline - Model Tag: {model_tag.upper()}")
        print(f"{'='*60}")
        
        # steps: (스크립트, 설명, 추가인자, 공통단계여부)
        # 공통단계(True): 데이터 수집/처리 등 모델 태그와 무관한 작업 (1회만 실행)
        # 모델별단계(False): 예측/매칭 등 모델 태그에 따라 다르게 실행되는 작업
        steps = [
            # === Step 1~4: 데이터 수집/처리 (머니라인과 공유, 독립 실행 시 주석 해제) ===
            # ('mlb_match_analyzer.py', 'Collecting MLB match data', None, True),
            # ('mlb_record_processor_streaming.py', 'Processing MLB match records with streaming', None, True),
            # ('mlb_training_data_processor_v2.py', 'Preparing MLB training data', None, True),
            # ('mlb_prediction_data_processor_v2.py', 'Preparing MLB prediction data', None, True),
            
            # === Step 5: 토탈 모델 학습 (필요 시 주석 해제, 개별 선택 가능) ===
            # ('mlb_model1_totals.py', 'Training Totals Model 1 (LightGBM)', None, True),
            # ('mlb_model2_totals.py', 'Training Totals Model 2 (CatBoost)', None, True),
            # ('mlb_model3_totals.py', 'Training Totals Model 3 (XGBoost)', None, True),
            # ('mlb_model4_totals.py', 'Training Totals Model 4 (LightGBM)', None, True),
            # ('mlb_model5_totals.py', 'Training Totals Model 5 (CatBoost)', None, True),
            # ('mlb_model6_totals.py', 'Training Totals Model 6 (XGBoost)', None, True),
            # ('mlb_model7_totals.py', 'Training Totals Model 7 (LightGBM)', None, True),
            # ('mlb_model8_totals.py', 'Training Totals Model 8 (CatBoost)', None, True),
            # ('mlb_model9_totals.py', 'Training Totals Model 9 (XGBoost)', None, True),
            # ('mlb_model_rf_totals.py', 'Training Totals Random Forest model', None, True),
            # ('mlb_model_nn_totals.py', 'Training Totals Neural Network model', None, True),
            # ('mlb_model_svm_totals.py', 'Training Totals SVM model', None, True),
            # ('mlb_model_advanced_lgbm_basic_totals.py', 'Training Totals Advanced LightGBM Basic', None, True),
            # ('mlb_model_advanced_lgbm_totals.py', 'Training Totals Advanced LightGBM', None, True),
            # ('mlb_model_advanced_catboost_basic_totals.py', 'Training Totals Advanced CatBoost Basic', None, True),
            # ('mlb_model_advanced_catboost_totals.py', 'Training Totals Advanced CatBoost', None, True),
            # ('mlb_model_advanced_xgboost_basic_totals.py', 'Training Totals Advanced XGBoost Basic', None, True),
            # ('mlb_model_advanced_xgboost_totals.py', 'Training Totals Advanced XGBoost', None, True),
            # ('mlb_model_advanced_nn_totals.py', 'Training Totals Advanced Neural Network', None, True),
            # ('mlb_model_advanced_rf_totals.py', 'Training Totals Advanced Random Forest', None, True),
            # ('mlb_model_advanced_svm_totals.py', 'Training Totals Advanced SVM', None, True),
            # ('mlb_model1_extended_lgbm_totals.py', 'Training Totals Extended LightGBM (123 features)', None, True),
            # ('mlb_model2_extended_catboost_totals.py', 'Training Totals Extended CatBoost (123 features)', None, True),
            # ('mlb_model3_extended_xgboost_totals.py', 'Training Totals Extended XGBoost (123 features)', None, True),
            
            # === Step 6: 토탈 앙상블 예측 (모델 태그별 실행) ===
            ('run_ensemble_totals.py', 'Running totals ensemble predictions', ['--model-tag', model_tag], False),
            
            # === Step 7: 토탈 배당률 수집 ===
            ('mlb_odds_fetcher_totals_FanDuel.py', 'Fetching MLB totals odds from FanDuel', None, True),
            
            # === Step 8: 토탈 배당률 매칭 (모델 태그별 실행) ===
            ('odds_matcher_totals.py', 'Matching totals odds with predictions', ['--model-tag', model_tag], False),
        ]
        
        critical_scripts = [
            'mlb_match_analyzer.py',
            'mlb_record_processor_streaming.py',
            'mlb_training_data_processor_v2.py',
            'mlb_prediction_data_processor_v2.py'
        ]
        
        success_count = 0
        failed_steps = []
        step_num = 0
        
        if model_tag == 'both':
            common = [(s, d, a, c) for s, d, a, c in steps if c]
            for script, description, extra_args, _ in common:
                step_num += 1
                print(f"\n=== Step {step_num}: {description} ===")
                
                if pipeline.run_script(script, description, extra_args):
                    success_count += 1
                    print(f"Successfully completed: {description}")
                else:
                    failed_steps.append(f"Step {step_num}: {description}")
                    print(f"Failed: {description}")
                    if script in critical_scripts:
                        print(f"Critical step failed. Stopping pipeline.")
                        break
                    print(f"Non-critical step failed. Continuing...")
                time.sleep(2)
            
            tagged = [(s, d, a, c) for s, d, a, c in steps if not c]
            for tag in ['active', 'shadow']:
                print(f"\n--- [{tag.upper()}] Totals Pipeline ---")
                for script, description, _, _ in tagged:
                    step_num += 1
                    tag_desc = f'[{tag.upper()}] {description}'
                    tag_args = ['--model-tag', tag]
                    print(f"\n=== Step {step_num}: {tag_desc} ===")
                    
                    if pipeline.run_script(script, tag_desc, extra_args=tag_args):
                        success_count += 1
                        print(f"Successfully completed: {tag_desc}")
                    else:
                        failed_steps.append(f"Step {step_num}: {tag_desc}")
                        print(f"Failed: {tag_desc}")
                    time.sleep(2)
        else:
            for script, description, extra_args, _ in steps:
                step_num += 1
                print(f"\n=== Step {step_num}/{len(steps)}: {description} ===")
                
                if pipeline.run_script(script, description, extra_args):
                    success_count += 1
                    print(f"Successfully completed: {description}")
                else:
                    failed_steps.append(f"Step {step_num}: {description}")
                    print(f"Failed: {description}")
                    if script in critical_scripts:
                        print(f"Critical step failed. Stopping pipeline.")
                        break
                    print(f"Non-critical step failed. Continuing...")
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"  Totals Pipeline Summary (tag: {model_tag.upper()})")
        print(f"{'='*60}")
        print(f"Total steps: {step_num}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(failed_steps)}")
        
        if failed_steps:
            print(f"\nFailed steps:")
            for step in failed_steps:
                print(f"  - {step}")
        
        if success_count == step_num:
            print(f"\nTotals Pipeline completed successfully! All {step_num} steps executed without errors.")
        elif success_count > 0:
            print(f"\nTotals Pipeline completed with {len(failed_steps)} failed steps. Check logs for details.")
        else:
            print(f"\nTotals Pipeline failed completely. Check logs for details.")
            return False
        
        return True
            
    except KeyboardInterrupt:
        print("\n=== Totals Pipeline interrupted by user ===")
        sys.exit(1)
    except Exception as e:
        print(f"\n=== Unexpected error: {str(e)} ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
