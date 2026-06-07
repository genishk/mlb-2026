"""
MLB Active/Shadow 모델 전환 및 학습 스크립트

사용법:
    python mlb_switch_and_train.py --action status        # 현재 모델 상태 확인
    python mlb_switch_and_train.py --action train         # 새 Shadow 모델 학습
    python mlb_switch_and_train.py --action switch        # Shadow → Active 전환
    python mlb_switch_and_train.py --action full          # 전환 + 학습 (전체 사이클)
    python mlb_switch_and_train.py --action clean-shadow  # Shadow 모델 파일만 삭제

워크플로우:
    1. active가 shadow보다 계속 성적이 좋을 때:
       - --action clean-shadow → shadow 삭제, 그 후 --action train → 새 shadow 학습
    2. shadow가 active보다 성적이 좋을 때:
       - --action switch → shadow를 active로 승격
    3. 전체 사이클 (주간):
       - --action full → 기존 shadow→active 승격 후 새 shadow 학습
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import logging


MODEL_CONFIGS = [
    # Basic models (1-9): model + features
    *[{
        'key': f'model{n}',
        'name': f'Model {n}',
        'script': f'src/modeling/mlb_model{n}.py',
        'artifacts': [
            {'glob_base': f'mlb_betting_model{n}', 'ext': '.joblib', 'type': 'model'},
            {'glob_base': f'mlb_features{n}', 'ext': '.json', 'type': 'features'},
        ]
    } for n in range(1, 10)],

    # RF: model + features (no scaler)
    {
        'key': 'model_rf',
        'name': 'Random Forest',
        'script': 'src/modeling/mlb_model_rf.py',
        'artifacts': [
            {'glob_base': 'mlb_betting_model_rf', 'ext': '.joblib', 'type': 'model'},
            {'glob_base': 'mlb_features_rf', 'ext': '.json', 'type': 'features'},
        ]
    },

    # NN: model + scaler + features
    {
        'key': 'model_nn',
        'name': 'Neural Network',
        'script': 'src/modeling/mlb_model_nn.py',
        'artifacts': [
            {'glob_base': 'mlb_betting_model_nn', 'ext': '.joblib', 'type': 'model'},
            {'glob_base': 'mlb_scaler_nn', 'ext': '.joblib', 'type': 'scaler'},
            {'glob_base': 'mlb_features_nn', 'ext': '.json', 'type': 'features'},
        ]
    },

    # SVM: model + scaler + features
    {
        'key': 'model_svm',
        'name': 'SVM',
        'script': 'src/modeling/mlb_model_svm.py',
        'artifacts': [
            {'glob_base': 'mlb_betting_model_svm', 'ext': '.joblib', 'type': 'model'},
            {'glob_base': 'mlb_scaler_svm', 'ext': '.joblib', 'type': 'scaler'},
            {'glob_base': 'mlb_features_svm', 'ext': '.json', 'type': 'features'},
        ]
    },

    # Advanced models: model + scaler + selector + features
    *[{
        'key': f'advanced_{algo}',
        'name': f'Advanced {algo.replace("_", " ").title()}',
        'script': f'src/modeling/mlb_model_advanced_{algo}.py',
        'artifacts': [
            {'glob_base': f'mlb_advanced_{algo}', 'ext': '.joblib', 'type': 'model',
             'exclude': ['scaler', 'selector', 'features'] + (['basic'] if '_' not in algo and algo not in ('nn', 'rf', 'svm') else [])},
            {'glob_base': f'mlb_advanced_{algo}_scaler', 'ext': '.joblib', 'type': 'scaler'},
            {'glob_base': f'mlb_advanced_{algo}_selector', 'ext': '.joblib', 'type': 'selector'},
            {'glob_base': f'mlb_advanced_{algo}_features', 'ext': '.json', 'type': 'features'},
        ]
    } for algo in [
        'catboost_basic', 'catboost', 'lgbm_basic', 'lgbm',
        'nn', 'rf', 'svm', 'xgboost_basic', 'xgboost'
    ]],

    # Extended models: model + features
    {
        'key': 'model1_extended_lgbm',
        'name': 'Extended LightGBM (123 features)',
        'script': 'src/modeling/mlb_model1_extended_lgbm.py',
        'artifacts': [
            {'glob_base': 'mlb_model1_extended_lgbm', 'ext': '.joblib', 'type': 'model',
             'exclude': ['features']},
            {'glob_base': 'mlb_model1_extended_lgbm_features', 'ext': '.json', 'type': 'features'},
        ]
    },
    {
        'key': 'model2_extended_catboost',
        'name': 'Extended CatBoost (123 features)',
        'script': 'src/modeling/mlb_model2_extended_catboost.py',
        'artifacts': [
            {'glob_base': 'mlb_model2_extended_catboost', 'ext': '.joblib', 'type': 'model',
             'exclude': ['features']},
            {'glob_base': 'mlb_model2_extended_catboost_features', 'ext': '.json', 'type': 'features'},
        ]
    },
    {
        'key': 'model3_extended_xgboost',
        'name': 'Extended XGBoost (123 features)',
        'script': 'src/modeling/mlb_model3_extended_xgboost.py',
        'artifacts': [
            {'glob_base': 'mlb_model3_extended_xgboost', 'ext': '.joblib', 'type': 'model',
             'exclude': ['features']},
            {'glob_base': 'mlb_model3_extended_xgboost_features', 'ext': '.json', 'type': 'features'},
        ]
    },
]


class MLBModelManager:
    """MLB Active/Shadow 모델 관리자"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.model_dir = self.project_root / 'src' / 'models' / 'saved_models'
        self.backup_dir = self.project_root / 'src' / 'models' / 'backup_models'
        self.predictions_dir = self.project_root / 'src' / 'predictions'
        self.matched_dir = self.project_root / 'src' / 'odds' / 'data' / 'matched'
        self.data_backup_dir = self.project_root / 'src' / 'models' / 'backup_data'

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.data_backup_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logging()

    def setup_logging(self):
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'model_switch_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('MLBModelManager')

    def _find_tagged_files(self, tag: str) -> dict:
        """특정 태그의 모든 파일 찾기"""
        result = {}
        for cfg in MODEL_CONFIGS:
            key = cfg['key']
            files = []
            for art in cfg['artifacts']:
                path = self.model_dir / f"{art['glob_base']}_{tag}{art['ext']}"
                files.append({'path': path, 'exists': path.exists(), 'type': art['type']})
            result[key] = {'name': cfg['name'], 'files': files}
        return result

    def _find_timestamp_files(self, glob_base: str, ext: str, exclude: list = None, after: datetime = None):
        """타임스탬프 기반 파일 찾기 (tagged 파일 제외)"""
        pattern = f"{glob_base}_*{ext}"
        candidates = list(self.model_dir.glob(pattern))
        candidates = [f for f in candidates if '_active' not in f.name and '_shadow' not in f.name]
        if exclude:
            for kw in exclude:
                candidates = [f for f in candidates if kw not in f.name.replace(glob_base, '')]
        if after:
            candidates = [f for f in candidates if datetime.fromtimestamp(f.stat().st_mtime) >= after]
        return candidates

    # ─── STATUS ───

    def display_status(self):
        active = self._find_tagged_files('active')
        shadow = self._find_tagged_files('shadow')

        print(f"\n{'='*70}")
        print(f"  MLB MODEL STATUS")
        print(f"{'='*70}")

        active_count = sum(1 for v in active.values() if v['files'][0]['exists'])
        shadow_count = sum(1 for v in shadow.values() if v['files'][0]['exists'])

        print(f"\n  ACTIVE MODELS ({active_count}/{len(MODEL_CONFIGS)})")
        print(f"  {'-'*50}")
        for key, info in active.items():
            model_file = info['files'][0]
            if model_file['exists']:
                mtime = datetime.fromtimestamp(model_file['path'].stat().st_mtime)
                arts = sum(1 for f in info['files'] if f['exists'])
                print(f"    {info['name']:40s} [{arts} files] {mtime:%Y-%m-%d %H:%M}")
            else:
                print(f"    {info['name']:40s} [MISSING]")

        print(f"\n  SHADOW MODELS ({shadow_count}/{len(MODEL_CONFIGS)})")
        print(f"  {'-'*50}")
        if shadow_count == 0:
            print(f"    Shadow models not found.")
        else:
            for key, info in shadow.items():
                model_file = info['files'][0]
                if model_file['exists']:
                    mtime = datetime.fromtimestamp(model_file['path'].stat().st_mtime)
                    arts = sum(1 for f in info['files'] if f['exists'])
                    print(f"    {info['name']:40s} [{arts} files] {mtime:%Y-%m-%d %H:%M}")

        # 백업 목록
        backups = sorted(
            [d for d in self.backup_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime, reverse=True
        ) if self.backup_dir.exists() else []

        print(f"\n  BACKUPS ({len(backups)})")
        print(f"  {'-'*50}")
        if backups:
            for b in backups[:5]:
                count = len(list(b.iterdir()))
                print(f"    {b.name}  ({count} files)")
            if len(backups) > 5:
                print(f"    ... and {len(backups)-5} more")
        else:
            print(f"    No backups.")

        print(f"\n{'='*70}")

    # ─── BACKUP ───

    def backup_active_models(self) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_folder = self.backup_dir / f'active_backup_{timestamp}'
        backup_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Active models backup: {backup_folder.name}")
        count = 0
        for cfg in MODEL_CONFIGS:
            for art in cfg['artifacts']:
                src = self.model_dir / f"{art['glob_base']}_active{art['ext']}"
                if src.exists():
                    shutil.copy2(src, backup_folder / src.name)
                    count += 1
        self.logger.info(f"Backed up {count} files")
        return backup_folder

    # ─── SWITCH ───

    def switch_shadow_to_active(self) -> bool:
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Shadow -> Active switch starting")
        self.logger.info(f"{'='*50}")

        shadow_info = self._find_tagged_files('shadow')
        shadow_model_count = sum(1 for v in shadow_info.values() if v['files'][0]['exists'])

        if shadow_model_count == 0:
            self.logger.error("No shadow models found. Run --action train first.")
            return False

        self.logger.info(f"Found {shadow_model_count} shadow models")

        # 1. Backup active
        active_info = self._find_tagged_files('active')
        active_count = sum(1 for v in active_info.values() if v['files'][0]['exists'])
        if active_count > 0:
            self.logger.info(f"\n1) Backing up {active_count} active models...")
            self.backup_active_models()

        # 2. Delete active
        self.logger.info(f"\n2) Removing current active models...")
        removed = 0
        for cfg in MODEL_CONFIGS:
            for art in cfg['artifacts']:
                path = self.model_dir / f"{art['glob_base']}_active{art['ext']}"
                if path.exists():
                    path.unlink()
                    removed += 1
        self.logger.info(f"Removed {removed} active files")

        # 3. Rename shadow → active
        self.logger.info(f"\n3) Promoting shadow -> active...")
        switched = 0
        for cfg in MODEL_CONFIGS:
            promoted = False
            for art in cfg['artifacts']:
                shadow_path = self.model_dir / f"{art['glob_base']}_shadow{art['ext']}"
                active_path = self.model_dir / f"{art['glob_base']}_active{art['ext']}"
                if shadow_path.exists():
                    shadow_path.rename(active_path)
                    promoted = True
            if promoted:
                switched += 1
                self.logger.info(f"  Promoted: {cfg['name']}")

        self.logger.info(f"\nSwitch complete: {switched} models promoted to active")

        # 4. Switch prediction/matched files
        self.logger.info(f"\n4) Switching prediction/matched files...")
        self._switch_prediction_files()

        return True

    def _switch_prediction_files(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_folder = self.data_backup_dir / f'data_backup_{timestamp}'
        backup_folder.mkdir(parents=True, exist_ok=True)

        backed = 0
        renamed = 0

        # predictions folder
        for pattern_active, pattern_shadow in [
            ('mlb_ensemble_predictions_*_active.json', 'mlb_ensemble_predictions_*_shadow.json'),
        ]:
            for f in self.predictions_dir.glob(pattern_active):
                shutil.move(str(f), str(backup_folder / f.name))
                backed += 1
            for f in self.predictions_dir.glob(pattern_shadow):
                new_name = f.name.replace('_shadow.json', '_active.json')
                f.rename(self.predictions_dir / new_name)
                renamed += 1

        # matched folder
        for pattern_active, pattern_shadow in [
            ('mlb_predictions_with_odds_*_active.json', 'mlb_predictions_with_odds_*_shadow.json'),
        ]:
            for f in self.matched_dir.glob(pattern_active):
                shutil.move(str(f), str(backup_folder / f.name))
                backed += 1
            for f in self.matched_dir.glob(pattern_shadow):
                new_name = f.name.replace('_shadow.json', '_active.json')
                f.rename(self.matched_dir / new_name)
                renamed += 1

        self.logger.info(f"  Backed up: {backed} files, Renamed: {renamed} files")

    # ─── TRAIN ───

    def train_shadow_models(self) -> bool:
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"New shadow model training starting")
        self.logger.info(f"{'='*50}")

        env = os.environ.copy()
        env['PROJECT_ROOT'] = str(self.project_root)
        env['PYTHONPATH'] = str(self.project_root)

        trained = 0
        failed = []

        for cfg in MODEL_CONFIGS:
            script_path = self.project_root / cfg['script']
            if not script_path.exists():
                self.logger.warning(f"  Script not found: {cfg['script']}")
                failed.append(cfg['key'])
                continue

            self.logger.info(f"\n  Training: {cfg['name']}...")
            train_start = datetime.now()

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    cwd=str(self.project_root),
                    timeout=900  # 15 min
                )

                # Find newly created timestamp files and rename to _shadow
                renamed_count = 0
                for art in cfg['artifacts']:
                    exclude = art.get('exclude')
                    ts_files = self._find_timestamp_files(
                        art['glob_base'], art['ext'],
                        exclude=exclude, after=train_start
                    )
                    if ts_files:
                        latest = max(ts_files, key=lambda x: x.stat().st_mtime)
                        shadow_path = self.model_dir / f"{art['glob_base']}_shadow{art['ext']}"

                        if shadow_path.exists():
                            shadow_path.unlink()

                        latest.rename(shadow_path)
                        renamed_count += 1
                        self.logger.info(f"    Saved: {shadow_path.name}")

                        # Clean up any other timestamp files from this training
                        for leftover in ts_files:
                            if leftover.exists() and leftover != latest:
                                leftover.unlink()

                if renamed_count > 0:
                    trained += 1
                    self.logger.info(f"  {cfg['name']} training complete ({renamed_count} files)")
                else:
                    self.logger.warning(f"  {cfg['name']} trained but no output files found")
                    failed.append(cfg['key'])

            except subprocess.TimeoutExpired:
                self.logger.error(f"  {cfg['name']} timeout (15min)")
                failed.append(cfg['key'])
            except subprocess.CalledProcessError as e:
                stderr_preview = e.stderr[:300] if e.stderr else 'Unknown error'
                self.logger.error(f"  {cfg['name']} failed: {stderr_preview}")
                failed.append(cfg['key'])
            except Exception as e:
                self.logger.error(f"  {cfg['name']} error: {str(e)}")
                failed.append(cfg['key'])

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Training results: {trained} succeeded, {len(failed)} failed")
        if failed:
            self.logger.info(f"  Failed: {failed}")
        self.logger.info(f"{'='*50}")

        return trained > 0

    # ─── CLEANUP ───

    def clean_shadow_models(self):
        """Delete all shadow model files from saved_models AND shadow analysis files (active untouched)."""
        deleted = 0
        for cfg in MODEL_CONFIGS:
            for art in cfg['artifacts']:
                path = self.model_dir / f"{art['glob_base']}_shadow{art['ext']}"
                if path.exists():
                    path.unlink()
                    self.logger.info(f"Deleted: {path.name}")
                    deleted += 1
        self.logger.info(f"Cleaned {deleted} shadow model files from {self.model_dir}")

        analysis_deleted = 0
        for f in self.matched_dir.glob('mlb_predictions_with_odds_*_shadow.json'):
            f.unlink()
            self.logger.info(f"Deleted analysis: {f.name}")
            analysis_deleted += 1
        self.logger.info(f"Cleaned {analysis_deleted} shadow analysis files from matched/")
        return deleted + analysis_deleted

    def cleanup_old_backups(self, keep_count: int = 5):
        if not self.backup_dir.exists():
            return

        backups = sorted(
            [d for d in self.backup_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if len(backups) > keep_count:
            for old in backups[keep_count:]:
                self.logger.info(f"Removing old backup: {old.name}")
                shutil.rmtree(old)


def main():
    parser = argparse.ArgumentParser(
        description='MLB Active/Shadow Model Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlb_switch_and_train.py                         # full cycle (switch + train)
  python mlb_switch_and_train.py --action status         # check current model status
  python mlb_switch_and_train.py --action train          # train new shadow models only
  python mlb_switch_and_train.py --action switch         # promote shadow -> active
  python mlb_switch_and_train.py --action full           # switch then train new shadow
  python mlb_switch_and_train.py --action clean-shadow   # delete shadow files only
        """
    )
    parser.add_argument(
        '--action', '-a', type=str,
        choices=['status', 'train', 'switch', 'full', 'clean-shadow'],
        default='full',
        help='Action to perform (default: full = switch + train)'
    )
    parser.add_argument(
        '--keep-backups', '-k', type=int, default=5,
        help='Number of backups to keep (default: 5)'
    )
    args = parser.parse_args()

    manager = MLBModelManager()

    try:
        if args.action == 'status':
            manager.display_status()

        elif args.action == 'train':
            manager.train_shadow_models()
            manager.display_status()

        elif args.action == 'switch':
            if manager.switch_shadow_to_active():
                manager.cleanup_old_backups(args.keep_backups)
            manager.display_status()

        elif args.action == 'clean-shadow':
            print(f"\nCleaning shadow model files (active models untouched)...")
            manager.clean_shadow_models()
            manager.display_status()

        elif args.action == 'full':
            print(f"\nFull cycle: Shadow -> Active, then train new Shadow")

            if manager.switch_shadow_to_active():
                manager.cleanup_old_backups(args.keep_backups)
                manager.train_shadow_models()
            else:
                print(f"\nSwitch failed. If no shadow models exist, run --action train first.")

            manager.display_status()

    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
