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

class MLBBettingPipeline:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_logging()
        
        # 필요한 디렉토리 구조 정의 및 생성
        self.dirs = {
            'data': self.project_root / 'data',
            'match_data': self.project_root / 'data' / 'match_data',
            'records': self.project_root / 'data' / 'records',
            'training': self.project_root / 'data' / 'training',
            'prediction': self.project_root / 'data' / 'prediction',
            'odds': self.project_root / 'data' / 'odds',
            'models': self.project_root / 'models' / 'saved_models',
            'predictions': self.project_root / 'predictions',
            'plots': self.project_root / 'models' / 'plots',
            'logs': self.project_root / 'logs',
            'src_data_processing': self.project_root / 'src' / 'data_processing',
            'src_modeling': self.project_root / 'src' / 'modeling',
            'src_odds': self.project_root / 'src' / 'odds'
        }
        
        # 디렉토리 생성 및 존재 확인
        for dir_path in self.dirs.values():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    raise Exception(f"Failed to create directory: {dir_path}")
            except Exception as e:
                self.logger.error(f"Error creating directory {dir_path}: {str(e)}")
                raise
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'mlb_pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('MLBBettingPipeline')
    
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
            'matplotlib': 'matplotlib',
            'plotly': 'plotly',
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
            self.logger.info("Install missing packages using: pip install " + " ".join(missing_packages))
            return False
        return True
    
    def validate_file_structure(self) -> bool:
        """필요한 파일들의 존재 여부 확인"""
        required_files = [
            'src/data_processing/mlb_match_analyzer.py',
            'src/data_processing/mlb_record_processor_streaming.py',
            'src/data_processing/mlb_training_data_processor_v2.py',
            'src/data_processing/mlb_prediction_data_processor_v2.py',
            'src/modeling/run_ensemble.py',
            'src/odds/mlb_odds_fetcher_FanDuel.py',
            'src/odds/odds_matcher.py'
        ]
        
        # 모델 파일들 추가
        model_files = [
            'mlb_model_advanced_catboost_basic.py', 'mlb_model_advanced_catboost.py',
            'mlb_model_advanced_lgbm_basic.py', 'mlb_model_advanced_lgbm.py',
            'mlb_model_advanced_nn.py', 'mlb_model_advanced_rf.py',
            'mlb_model_advanced_svm.py', 'mlb_model_advanced_xgboost_basic.py',
            'mlb_model_advanced_xgboost.py', 'mlb_model_nn.py', 'mlb_model_rf.py',
            'mlb_model_svm.py', 'mlb_model1_extended_lgbm.py', 'mlb_model1.py',
            'mlb_model2_extended_catboost.py', 'mlb_model2.py',
            'mlb_model3_extended_xgboost.py', 'mlb_model3.py',
            'mlb_model4.py', 'mlb_model5.py', 'mlb_model6.py',
            'mlb_model7.py', 'mlb_model8.py', 'mlb_model9.py'
        ]
        
        for model_file in model_files:
            required_files.append(f'src/modeling/{model_file}')
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        return True
    
    def run_script(self, script_name: str, description: str, extra_args: Optional[List[str]] = None) -> bool:
        """Python 스크립트 실행"""
        try:
            self.logger.info(f"Starting: {description}")
            
            if script_name in ['mlb_match_analyzer.py', 'mlb_record_processor.py', 'mlb_record_processor_streaming.py',
                              'mlb_training_data_processor_v2.py', 'mlb_prediction_data_processor_v2.py']:
                script_path = self.project_root / 'src' / 'data_processing' / script_name
            elif script_name in ['mlb_odds_fetcher_FanDuel.py', 'odds_matcher.py']:
                script_path = self.project_root / 'src' / 'odds' / script_name
            elif script_name.startswith('mlb_model') or script_name == 'run_ensemble.py':
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
            
            stream_scripts = {'mlb_match_analyzer.py'}
            if script_name in stream_scripts:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env, cwd=str(self.project_root), bufsize=1)
                output_lines = []
                for line in process.stdout:
                    line = line.rstrip()
                    print(f"  {line}")
                    output_lines.append(line)
                process.wait()
                if output_lines:
                    self.logger.info(f"Output: {''.join(output_lines[-20:])}")
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
            else:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True,
                    env=env, cwd=str(self.project_root))
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout}")
                if result.stderr:
                    self.logger.warning(f"Stderr: {result.stderr}")
            
            self.logger.info(f"Completed: {description}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in {description}: {str(e)}")
            if hasattr(e, 'stderr') and e.stderr:
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
        args = Args()
    else:
        parser = argparse.ArgumentParser(description='MLB Betting Pipeline')
        parser.add_argument('--model-tag', type=str, default='both',
                           choices=['active', 'shadow', 'both'],
                           help='Model tag for prediction (default: both)')
        args = parser.parse_args()
    
    pipeline = MLBBettingPipeline()
    
    try:
        if not pipeline.check_dependencies():
            print("Missing dependencies. Check log for details.")
            return False
        if not pipeline.validate_file_structure():
            print("Missing required files. Check log for details.")
            return False
        
        model_tag = args.model_tag
        print(f"\n{'='*60}")
        print(f"  MLB Betting Pipeline - Model Tag: {model_tag.upper()}")
        print(f"{'='*60}")
        
        # steps: (스크립트, 설명, 추가인자, 공통단계여부)
        # 공통단계(True): 데이터 수집/처리 등 모델 태그와 무관한 작업 (1회만 실행)
        # 모델별단계(False): 예측/매칭 등 모델 태그에 따라 다르게 실행되는 작업
        steps = [
            # Step 1: 데이터 수집
            ('mlb_match_analyzer.py', 'Collecting MLB match data', None, True),
            
            # Step 2: 레코드 처리 (스트리밍 방식)
            ('mlb_record_processor_streaming.py', 'Processing MLB match records with streaming', None, True),
            
            # Step 3: 훈련 및 예측 데이터 준비
            ('mlb_training_data_processor_v2.py', 'Preparing MLB training data', None, True),
            ('mlb_prediction_data_processor_v2.py', 'Preparing MLB prediction data', None, True),
            
            # Step 4: 모든 모델 훈련 - 각 모델은 개별적으로 주석 처리 가능
            # ('mlb_model_advanced_catboost_basic.py', 'Training Advanced CatBoost Basic model', None, True),
            # ('mlb_model_advanced_catboost.py', 'Training Advanced CatBoost model', None, True),
            # ('mlb_model_advanced_lgbm_basic.py', 'Training Advanced LightGBM Basic model', None, True),
            # ('mlb_model_advanced_lgbm.py', 'Training Advanced LightGBM model', None, True),
            # ('mlb_model_advanced_nn.py', 'Training Advanced Neural Network model', None, True),
            # ('mlb_model_advanced_rf.py', 'Training Advanced Random Forest model', None, True),
            # ('mlb_model_advanced_svm.py', 'Training Advanced SVM model', None, True),
            # ('mlb_model_advanced_xgboost_basic.py', 'Training Advanced XGBoost Basic model', None, True),
            # ('mlb_model_advanced_xgboost.py', 'Training Advanced XGBoost model', None, True),
            # ('mlb_model_nn.py', 'Training Neural Network model', None, True),
            # ('mlb_model_rf.py', 'Training Random Forest model', None, True),
            # ('mlb_model_svm.py', 'Training SVM model', None, True),
            # ('mlb_model1_extended_lgbm.py', 'Training Model 1 Extended LightGBM', None, True),
            # ('mlb_model1.py', 'Training Model 1 LightGBM', None, True),
            # ('mlb_model2_extended_catboost.py', 'Training Model 2 Extended CatBoost', None, True),
            # ('mlb_model2.py', 'Training Model 2 CatBoost', None, True),
            # ('mlb_model3_extended_xgboost.py', 'Training Model 3 Extended XGBoost', None, True),
            # ('mlb_model3.py', 'Training Model 3 XGBoost', None, True),
            # ('mlb_model4.py', 'Training Model 4', None, True),
            # ('mlb_model5.py', 'Training Model 5', None, True),
            # ('mlb_model6.py', 'Training Model 6', None, True),
            # ('mlb_model7.py', 'Training Model 7', None, True),
            # ('mlb_model8.py', 'Training Model 8', None, True),
            # ('mlb_model9.py', 'Training Model 9', None, True),
            
            # Step 5: 앙상블 예측 (모델 태그별 실행)
            ('run_ensemble.py', 'Running ensemble predictions', ['--model-tag', model_tag], False),
            
            # Step 6: 배당률 수집
            ('mlb_odds_fetcher_FanDuel.py', 'Fetching current MLB odds from FanDuel', None, True),
            
            # Step 7: 배당률 매칭 (모델 태그별 실행)
            ('odds_matcher.py', 'Matching odds with predictions', ['--model-tag', model_tag], False),
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
        
        # both 모드: 공통 1회 + 모델별 태그 순차
        if model_tag == 'both':
            # 공통 단계 실행
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
            
            # critical failure 체크
            if any(s in critical_scripts for s, d, a, c in steps 
                   if c and f"Step {steps.index((s,d,a,c))+1}: {d}" in [fs.split(': ', 1)[-1] if ': ' in fs else '' for fs in failed_steps]):
                pass
            
            # 태그별 단계 실행 (active → shadow)
            tagged = [(s, d, a, c) for s, d, a, c in steps if not c]
            for tag in ['active', 'shadow']:
                print(f"\n--- [{tag.upper()}] Model Pipeline ---")
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
            # 단일 태그 모드: 전체 순차 실행
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
        
        # ─── 결과 요약 ───
        print(f"\n{'='*60}")
        print(f"  Pipeline Summary (tag: {model_tag.upper()})")
        print(f"{'='*60}")
        print(f"Total steps: {step_num}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(failed_steps)}")
        
        if failed_steps:
            print(f"\nFailed steps:")
            for step in failed_steps:
                print(f"  - {step}")
        
        if success_count == step_num:
            print(f"\nPipeline completed successfully! All {step_num} steps executed without errors.")
        elif success_count > 0:
            print(f"\nPipeline completed with {len(failed_steps)} failed steps. Check logs for details.")
        else:
            print(f"\nPipeline failed completely. Check logs for details.")
            return False
        
        return True
            
    except KeyboardInterrupt:
        print("\n=== Pipeline interrupted by user ===")
        sys.exit(1)
    except Exception as e:
        print(f"\n=== Unexpected error: {str(e)} ===")
        sys.exit(1)

if __name__ == "__main__":
    main() 