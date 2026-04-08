import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
import glob
import re
import json
from typing import Dict, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simple_model_analyzer import SimpleModelAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="🎯 MLB Model Selection Analyzer",
    page_icon="🎯", 
    layout="wide"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #007bff;
        margin-bottom: 2rem;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .pattern-box {
        background: linear-gradient(90deg, #f0f8ff, #ffffff);
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .condition-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .reasoning-item {
        background: #f8f9fa;
        border-left: 3px solid #17a2b8;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class ModelSelectionAnalyzer:
    """
    모델 선택 분석기
    
    기능:
    1. model7과 model_svm의 일별 성과 비교
    2. 경기 조건에 따른 모델 우위 패턴 학습
    3. 오늘의 최적 모델 추천
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "src" / "odds" / "data" / "matched"
        self.records_dir = self.project_root / "data" / "records"
        
        # 비교할 주요 모델들
        self.target_models = ['model7', 'model_svm']
        
        # 조건 분석을 위한 임계값들
        self.thresholds = {
            'high_consensus': 0.8,
            'low_consensus': 0.6,
            'high_confidence': 0.15,
            'low_confidence': 0.05,
            'underdog_ratio': 0.4,
            'favorite_ratio': 0.6,
            'wide_odds_spread': 200,
            'narrow_odds_spread': 100
        }
    
    def get_available_prefixes(self) -> List[str]:
        """사용 가능한 구분자들을 탐지 (telegram_dashboard와 동일)"""
        try:
            all_files = list(self.predictions_dir.glob("*mlb_predictions_with_odds_*.json"))
            prefixes = set()
            
            for file_path in all_files:
                filename = file_path.name
                if filename.startswith("mlb_predictions_with_odds_"):
                    prefixes.add("None")
                else:
                    parts = filename.split("_")
                    if len(parts) >= 7 and parts[-1].endswith(".json"):
                        prefix_parts = parts[:2]
                        if all(part.isdigit() for part in prefix_parts):
                            prefix = "_".join(prefix_parts) + "_"
                            prefixes.add(prefix)
            
            sorted_prefixes = sorted([p for p in prefixes if p != "None"])
            return ["None"] + sorted_prefixes
            
        except Exception as e:
            st.error(f"구분자 탐지 중 오류: {e}")
            return ["None"]
    
    def get_available_dates_for_prefix(self, prefix: str) -> List[str]:
        """특정 구분자에 대한 사용 가능한 날짜들을 반환"""
        try:
            if prefix == "None":
                pattern = "mlb_predictions_with_odds_*.json"
            else:
                pattern = f"{prefix}mlb_predictions_with_odds_*.json"
            
            files = list(self.predictions_dir.glob(pattern))
            dates = []
            
            for file_path in files:
                filename = file_path.name
                if prefix == "None":
                    parts = filename.replace(".json", "").split("_")
                    if len(parts) >= 5:
                        date_part = parts[4]
                        if len(date_part) == 8 and date_part.isdigit():
                            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            dates.append(formatted_date)
                else:
                    parts = filename.replace(".json", "").split("_")
                    if len(parts) >= 7:
                        date_part = parts[6]
                        if len(date_part) == 8 and date_part.isdigit():
                            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            dates.append(formatted_date)
            
            return sorted(list(set(dates)))
            
        except Exception as e:
            st.error(f"날짜 탐지 중 오류: {e}")
            return []
    
    def extract_daily_conditions(self, games_data: List[Dict]) -> Dict[str, float]:
        """일별 경기 조건 특성 추출"""
        if not games_data:
            return {}
        
        conditions = {
            'avg_confidence': 0.0,
            'consensus_strength': 0.0,
            'underdog_ratio': 0.0
        }
        
        valid_games = []
        consensus_scores = []
        confidence_scores = []
        underdog_count = 0
        
        for game in games_data:
            home_odds = game.get('home_odds') or game.get('home_team_odds')
            away_odds = game.get('away_odds') or game.get('away_team_odds')
            
            if home_odds is None or away_odds is None:
                continue
            
            try:
                home_odds = float(home_odds)
                away_odds = float(away_odds)
            except (ValueError, TypeError):
                continue
            
            valid_games.append(game)
            
            # 언더독 여부 (배당률이 +120 이상)
            if home_odds > 120 or away_odds > 120:
                underdog_count += 1
            
            # 모델 예측 확률들 수집
            model_probs = []
            for model in self.target_models:
                prob_key = f'{model}_probability'
                if prob_key in game and game[prob_key] is not None:
                    try:
                        prob = float(game[prob_key])
                        if 0 <= prob <= 1:
                            model_probs.append(prob)
                    except (ValueError, TypeError):
                        continue
            
            if len(model_probs) >= 2:  # 최소 2개 모델의 예측이 있어야 consensus 계산 가능
                # Consensus 계산 (예측 일치도)
                home_votes = sum(1 for p in model_probs if p > 0.5)
                away_votes = len(model_probs) - home_votes
                consensus = max(home_votes, away_votes) / len(model_probs)
                consensus_scores.append(consensus)
                
                # 평균 신뢰도 계산 (0.5에서 얼마나 멀리 떨어져 있는가)
                avg_confidence = np.mean([abs(p - 0.5) for p in model_probs])
                confidence_scores.append(avg_confidence)
        
        # 🎯 핵심 3개 조건만 계산
        if valid_games:
            conditions['avg_confidence'] = np.mean(confidence_scores) if confidence_scores else 0
            conditions['consensus_strength'] = np.mean(consensus_scores) if consensus_scores else 0
            conditions['underdog_ratio'] = underdog_count / len(valid_games)
        
        return conditions
    
    def analyze_daily_performance_comparison(self, start_date: str, end_date: str, data_prefix: str = "None") -> Dict[str, Any]:
        """일별 성과 비교 분석"""
        analyzer = SimpleModelAnalyzer(data_prefix=data_prefix)
        results = analyzer.analyze(start_date, end_date)
        
        if not results or 'daily_performances' not in results:
            return None
        
        daily_data = results['daily_performances']
        comparison_data = []
        
        # 공통 날짜 찾기
        model7_dates = set(daily_data.get('model7', {}).keys())
        svm_dates = set(daily_data.get('model_svm', {}).keys())
        common_dates = model7_dates.intersection(svm_dates)
        
        for date in sorted(common_dates):
            model7_perf = daily_data['model7'][date]
            svm_perf = daily_data['model_svm'][date]
            
            # 해당 날짜의 게임 데이터를 로드하여 조건 분석
            date_str = date.replace('-', '')
            game_conditions = self.load_and_analyze_daily_conditions(date_str, data_prefix)
            
            comparison_data.append({
                'date': date,
                'model7_roi': model7_perf['roi'],
                'svm_roi': svm_perf['roi'],
                'model7_win_rate': model7_perf['win_rate'],
                'svm_win_rate': svm_perf['win_rate'],
                'winner': 'model7' if model7_perf['roi'] > svm_perf['roi'] else 'model_svm',
                'performance_gap': abs(model7_perf['roi'] - svm_perf['roi']),
                'conditions': game_conditions
            })
        
        return {
            'comparison_data': comparison_data,
            'total_days': len(comparison_data),
            'model7_wins': sum(1 for d in comparison_data if d['winner'] == 'model7'),
            'svm_wins': sum(1 for d in comparison_data if d['winner'] == 'model_svm')
        }
    
    def load_and_analyze_daily_conditions(self, date_str: str, data_prefix: str) -> Dict[str, float]:
        """특정 날짜의 게임 조건 분석"""
        try:
            if data_prefix == "None":
                pattern = f"mlb_predictions_with_odds_{date_str}_*.json"
            else:
                pattern = f"{data_prefix}mlb_predictions_with_odds_{date_str}_*.json"
            
            files = list(self.predictions_dir.glob(pattern))
            
            if not files:
                return {}
            
            # 가장 최근 파일 선택
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                games_data = json.load(f)
            
            return self.extract_daily_conditions(games_data)
            
        except Exception as e:
            st.warning(f"날짜 {date_str} 조건 분석 실패: {e}")
            return {}
    
    def learn_selection_patterns(self, comparison_data: List[Dict]) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """패턴 학습을 통한 모델 선택 규칙 생성"""
        if not comparison_data:
            return None, {}
        
        # 🎯 핵심 특성만 선택 (과적합 방지)
        features = []
        labels = []
        feature_names = ['consensus_strength', 'underdog_ratio', 'avg_confidence']
        
        for data in comparison_data:
            conditions = data['conditions']
            if conditions and all(name in conditions for name in feature_names):
                feature_vector = [conditions[name] for name in feature_names]
                features.append(feature_vector)
                labels.append(data['winner'])
        
        if len(features) < 5:  # 최소 데이터 요구사항 (원래대로)
            return None, {'error': '학습 데이터가 부족합니다 (최소 5일 필요)'}
        
        # Random Forest 모델 학습 (원래 방식)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        X = np.array(features)
        y = np.array(labels)
        
        rf_model.fit(X, y)
        
        # 특성 중요도 분석
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
        
        # 모델 성능 평가 (학습 데이터에서)
        y_pred = rf_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # 패턴 분석 결과
        pattern_analysis = {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'total_samples': len(features),
            'model7_wins': sum(1 for label in y if label == 'model7'),
            'svm_wins': sum(1 for label in y if label == 'model_svm')
        }
        
        return rf_model, pattern_analysis
    
    def find_latest_prediction_file(self, data_prefix: str = "None") -> str:
        """최신 예측 파일 찾기"""
        if data_prefix == "None":
            pattern = str(self.predictions_dir / "mlb_predictions_with_odds_*.json")
        else:
            pattern = str(self.predictions_dir / f"{data_prefix}mlb_predictions_with_odds_*.json")
        
        try:
            files = glob.glob(pattern)
            if not files:
                return None
                
            def extract_datetime(filename):
                if data_prefix == "None":
                    match = re.search(r'mlb_predictions_with_odds_(\d{8}_\d{6})\.json', filename)
                else:
                    escaped_prefix = data_prefix.replace("_", r"\_")
                    match = re.search(rf'{escaped_prefix}mlb_predictions_with_odds_(\d{{8}}_\d{{6}})\.json', filename)
                
                if match:
                    return match.group(1)
                return "00000000_000000"
            
            latest_file = max(files, key=extract_datetime)
            return latest_file
            
        except Exception as e:
            st.error(f"파일 검색 중 오류: {e}")
            return None
    
    def recommend_today_model(self, rf_model: RandomForestClassifier, data_prefix: str = "None") -> Dict[str, Any]:
        """오늘의 최적 모델 추천"""
        latest_file = self.find_latest_prediction_file(data_prefix)
        
        if not latest_file:
            return {'error': '최신 예측 파일을 찾을 수 없습니다'}
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                today_games = json.load(f)
            
            today_conditions = self.extract_daily_conditions(today_games)
            
            if not today_conditions:
                return {'error': '오늘의 조건을 분석할 수 없습니다'}
            
            # 🎯 핵심 특성만 사용 (일관성 유지)
            feature_names = ['consensus_strength', 'underdog_ratio', 'avg_confidence']
            
            feature_vector = [[today_conditions.get(name, 0) for name in feature_names]]
            
            # 예측
            predicted_model = rf_model.predict(feature_vector)[0]
            prediction_proba = rf_model.predict_proba(feature_vector)[0]
            
            # 신뢰도 계산
            confidence = max(prediction_proba)
            
            # 추천 이유 생성 (comparison_data 전달)
            comparison_data = None
            if 'comparison_data' in st.session_state:
                comparison_data = st.session_state['comparison_data']
            reasoning = self.generate_reasoning(today_conditions, predicted_model, comparison_data)
            
            return {
                'recommended_model': predicted_model,
                'confidence': confidence,
                'reasoning': reasoning,
                'today_conditions': today_conditions,
                'latest_file': Path(latest_file).name
            }
            
        except Exception as e:
            return {'error': f'분석 중 오류 발생: {e}'}
    
    def generate_reasoning(self, conditions: Dict[str, float], recommended_model: str, comparison_data: List[Dict] = None) -> List[str]:
        """추천 이유 생성 (동적 기준 사용)"""
        reasoning = []
        
        # 동적 기준 계산 (comparison_data가 있을 때)
        if comparison_data:
            consensus_values = [d['conditions'].get('consensus_strength', 0) for d in comparison_data]
            underdog_values = [d['conditions'].get('underdog_ratio', 0) for d in comparison_data]
            confidence_values = [d['conditions'].get('avg_confidence', 0) for d in comparison_data]
            
            consensus_median = np.median(consensus_values)
            underdog_median = np.median(underdog_values)
            confidence_median = np.median(confidence_values)
            
            # 동적 기준으로 분석
            if conditions.get('consensus_strength', 0) > consensus_median:
                reasoning.append(f"✅ 높은 모델 합의도 ({conditions['consensus_strength']:.3f} vs median {consensus_median:.3f})")
            else:
                reasoning.append(f"⚠️ 낮은 모델 합의도 ({conditions['consensus_strength']:.3f} vs median {consensus_median:.3f})")
            
            if conditions.get('underdog_ratio', 0) > underdog_median:
                reasoning.append(f"📈 높은 언더독 비율 ({conditions['underdog_ratio']:.1%} vs median {underdog_median:.1%})")
            else:
                reasoning.append(f"⭐ 낮은 언더독 비율 ({conditions['underdog_ratio']:.1%} vs median {underdog_median:.1%})")
            
            if conditions.get('avg_confidence', 0) > confidence_median:
                reasoning.append(f"🎯 높은 예측 신뢰도 ({conditions['avg_confidence']:.3f} vs median {confidence_median:.3f})")
            else:
                reasoning.append(f"❓ 낮은 예측 신뢰도 ({conditions['avg_confidence']:.3f} vs median {confidence_median:.3f})")
        
        else:
            # 기존 고정 기준 사용 (fallback)
            if conditions.get('consensus_strength', 0) > self.thresholds['high_consensus']:
                reasoning.append(f"✅ 높은 모델 합의도 ({conditions['consensus_strength']:.2f} vs 임계값 {self.thresholds['high_consensus']})")
            elif conditions.get('consensus_strength', 0) < self.thresholds['low_consensus']:
                reasoning.append(f"⚠️ 낮은 모델 합의도 ({conditions['consensus_strength']:.2f} vs 임계값 {self.thresholds['low_consensus']})")
            
            if conditions.get('underdog_ratio', 0) > self.thresholds['underdog_ratio']:
                reasoning.append(f"📈 높은 언더독 비율 ({conditions['underdog_ratio']:.1%} vs 평균 {self.thresholds['underdog_ratio']:.1%})")
            
            if conditions.get('avg_confidence', 0) > self.thresholds['high_confidence']:
                reasoning.append(f"🎯 높은 예측 신뢰도 ({conditions['avg_confidence']:.3f})")
            elif conditions.get('avg_confidence', 0) < self.thresholds['low_confidence']:
                reasoning.append(f"❓ 낮은 예측 신뢰도 ({conditions['avg_confidence']:.3f})")
        
        # 🚫 배당률 스프레드와 시장 괴리도는 모델에서 사용하지 않으므로 제거
        # (3개 핵심 변수만 사용하기로 결정함)
        
        return reasoning

@st.cache_data
def load_model_selection_analyzer():
    """모델 선택 분석기 로드 (캐시됨)"""
    return ModelSelectionAnalyzer()

def main():
    # 헤더
    st.markdown('<div class="main-header">🎯 MLB Model Selection Analyzer</div>', unsafe_allow_html=True)
    
    # 면책조항
    st.error("""
    ⚠️ **LEGAL DISCLAIMER**: This service provides **statistical analysis for educational purposes only**. 
    NOT betting advice. High risk of financial loss. Past performance ≠ future results. 
    You are **solely responsible** for your decisions and must comply with local laws.
    """)
    
    # 분석기 초기화
    analyzer = load_model_selection_analyzer()
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 Pattern Analysis", "🎯 Today's Recommendation", "📈 Historical Comparison"])
    
    # === 탭 1: 패턴 분석 ===
    with tab1:
        st.header("📊 Model Selection Pattern Analysis")
        st.info("Analyze which model performs better under different game conditions")
        
        # 구분자 선택
        st.markdown("### 📂 Data Source Selection")
        available_prefixes = analyzer.get_available_prefixes()
        
        prefix_display_names = {}
        for prefix in available_prefixes:
            if prefix == "None":
                prefix_display_names[prefix] = "None (기본 - 구분자 없는 파일)"
            else:
                prefix_display_names[prefix] = f"{prefix.rstrip('_')} (구분자 파일)"
        
        selected_prefix = st.selectbox(
            "📂 Select Data Source:",
            available_prefixes,
            format_func=lambda x: prefix_display_names[x],
            help="구분자 없는 파일은 기본 분석, 구분자 파일은 특별 테스트 분석입니다.",
            key="pattern_prefix"
        )
        
        # 날짜 선택
        col1, col2 = st.columns(2)
        
        with col1:
            available_dates = analyzer.get_available_dates_for_prefix(selected_prefix)
            if available_dates:
                min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
                max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="pattern_start_date"
                )
            else:
                st.error("데이터가 없습니다.")
                return
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="pattern_end_date"
            )
        
        if start_date <= end_date:
            with st.spinner("🔍 Analyzing model selection patterns..."):
                # 성과 비교 분석
                comparison_results = analyzer.analyze_daily_performance_comparison(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    selected_prefix
                )
                
                if comparison_results and comparison_results['comparison_data']:
                    comparison_data = comparison_results['comparison_data']
                    
                    # 패턴 학습
                    rf_model, pattern_analysis = analyzer.learn_selection_patterns(comparison_data)
                    
                    if rf_model is not None:
                        # 패턴 분석 결과 표시
                        st.markdown("## 📈 Pattern Learning Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Days", comparison_results['total_days'])
                        with col2:
                            st.metric("Model7 Wins", comparison_results['model7_wins'])
                        with col3:
                            st.metric("SVM Wins", comparison_results['svm_wins'])
                        with col4:
                            accuracy = pattern_analysis['accuracy']
                            st.metric("Pattern Accuracy", f"{accuracy:.1%}")
                            if accuracy >= 1.0:
                                st.warning("⚠️ 100% 정확도는 과적합 가능성이 있습니다")
                        
                        # 특성 중요도 차트
                        st.markdown("### 🎯 Feature Importance")
                        importance_df = pd.DataFrame([
                            {'Feature': k, 'Importance': v} 
                            for k, v in pattern_analysis['feature_importance'].items()
                        ]).sort_values('Importance', ascending=True)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance for Model Selection',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # 일별 비교 차트
                        st.markdown("### 📅 Daily Performance Comparison")
                        daily_df = pd.DataFrame(comparison_data)
                        
                        fig_daily = go.Figure()
                        fig_daily.add_trace(go.Scatter(
                            x=daily_df['date'],
                            y=daily_df['model7_roi'],
                            mode='lines+markers',
                            name='Model7 ROI',
                            line=dict(color='blue')
                        ))
                        fig_daily.add_trace(go.Scatter(
                            x=daily_df['date'],
                            y=daily_df['svm_roi'],
                            mode='lines+markers',
                            name='SVM ROI',
                            line=dict(color='red')
                        ))
                        fig_daily.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_daily.update_layout(
                            title='Daily ROI Comparison: Model7 vs SVM',
                            xaxis_title='Date',
                            yaxis_title='ROI (%)',
                            height=500
                        )
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # 세션 상태에 모델 저장 (다른 탭에서 사용)
                        st.session_state['rf_model'] = rf_model
                        st.session_state['selected_prefix'] = selected_prefix
                        st.session_state['comparison_data'] = comparison_data  # 디버깅용
                        
                        # 조건별 성과 분석
                        st.markdown("### 🔍 Condition-based Performance Analysis")
                        st.info("패턴 분석: 어떤 조건에서 어떤 모델이 우수한지 확인")
                        
                        # 🎯 핵심 3가지 조건별 분석
                        consensus_values = [d['conditions'].get('consensus_strength', 0) for d in comparison_data]
                        underdog_values = [d['conditions'].get('underdog_ratio', 0) for d in comparison_data]
                        confidence_values = [d['conditions'].get('avg_confidence', 0) for d in comparison_data]
                        
                        # 동적 임계값 (중간값 기준)
                        consensus_median = np.median(consensus_values)
                        underdog_median = np.median(underdog_values)
                        confidence_median = np.median(confidence_values)
                        
                        st.markdown(f"**분석 기준**: Consensus {consensus_median:.2f}, Underdog {underdog_median:.1%}, Confidence {confidence_median:.3f}")
                        
                        # 1. Consensus 기준 분석
                        high_consensus_days = [d for d in comparison_data if d['conditions'].get('consensus_strength', 0) > consensus_median]
                        low_consensus_days = [d for d in comparison_data if d['conditions'].get('consensus_strength', 0) <= consensus_median]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if high_consensus_days:
                                model7_wins_high = sum(1 for d in high_consensus_days if d['winner'] == 'model7')
                                high_consensus_winner = "Model7" if model7_wins_high > len(high_consensus_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>🤝 High Consensus Days ({len(high_consensus_days)} days)</h4>
                                    <p><strong>Winner:</strong> {high_consensus_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_high}/{len(high_consensus_days)} ({model7_wins_high/len(high_consensus_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in high_consensus_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(high_consensus_days)-model7_wins_high}/{len(high_consensus_days)} ({(1-model7_wins_high/len(high_consensus_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in high_consensus_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            if low_consensus_days:
                                model7_wins_low = sum(1 for d in low_consensus_days if d['winner'] == 'model7')
                                low_consensus_winner = "Model7" if model7_wins_low > len(low_consensus_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>🤔 Low Consensus Days ({len(low_consensus_days)} days)</h4>
                                    <p><strong>Winner:</strong> {low_consensus_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_low}/{len(low_consensus_days)} ({model7_wins_low/len(low_consensus_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in low_consensus_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(low_consensus_days)-model7_wins_low}/{len(low_consensus_days)} ({(1-model7_wins_low/len(low_consensus_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in low_consensus_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 2. Underdog 기준 분석
                        st.markdown("---")
                        high_underdog_days = [d for d in comparison_data if d['conditions'].get('underdog_ratio', 0) > underdog_median]
                        low_underdog_days = [d for d in comparison_data if d['conditions'].get('underdog_ratio', 0) <= underdog_median]
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            if high_underdog_days:
                                model7_wins_underdog = sum(1 for d in high_underdog_days if d['winner'] == 'model7')
                                underdog_winner = "Model7" if model7_wins_underdog > len(high_underdog_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>📈 High Underdog Days ({len(high_underdog_days)} days)</h4>
                                    <p><strong>Winner:</strong> {underdog_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_underdog}/{len(high_underdog_days)} ({model7_wins_underdog/len(high_underdog_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in high_underdog_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(high_underdog_days)-model7_wins_underdog}/{len(high_underdog_days)} ({(1-model7_wins_underdog/len(high_underdog_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in high_underdog_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col4:
                            if low_underdog_days:
                                model7_wins_favorite = sum(1 for d in low_underdog_days if d['winner'] == 'model7')
                                favorite_winner = "Model7" if model7_wins_favorite > len(low_underdog_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>⭐ Low Underdog Days ({len(low_underdog_days)} days)</h4>
                                    <p><strong>Winner:</strong> {favorite_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_favorite}/{len(low_underdog_days)} ({model7_wins_favorite/len(low_underdog_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in low_underdog_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(low_underdog_days)-model7_wins_favorite}/{len(low_underdog_days)} ({(1-model7_wins_favorite/len(low_underdog_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in low_underdog_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 3. Confidence 기준 분석
                        st.markdown("---")
                        high_confidence_days = [d for d in comparison_data if d['conditions'].get('avg_confidence', 0) > confidence_median]
                        low_confidence_days = [d for d in comparison_data if d['conditions'].get('avg_confidence', 0) <= confidence_median]
                        
                        col5, col6 = st.columns(2)
                        
                        with col5:
                            if high_confidence_days:
                                model7_wins_high_conf = sum(1 for d in high_confidence_days if d['winner'] == 'model7')
                                high_conf_winner = "Model7" if model7_wins_high_conf > len(high_confidence_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>🎯 High Confidence Days ({len(high_confidence_days)} days)</h4>
                                    <p><strong>Winner:</strong> {high_conf_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_high_conf}/{len(high_confidence_days)} ({model7_wins_high_conf/len(high_confidence_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in high_confidence_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(high_confidence_days)-model7_wins_high_conf}/{len(high_confidence_days)} ({(1-model7_wins_high_conf/len(high_confidence_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in high_confidence_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col6:
                            if low_confidence_days:
                                model7_wins_low_conf = sum(1 for d in low_confidence_days if d['winner'] == 'model7')
                                low_conf_winner = "Model7" if model7_wins_low_conf > len(low_confidence_days)/2 else "SVM"
                                st.markdown(f"""
                                <div class="pattern-box">
                                    <h4>❓ Low Confidence Days ({len(low_confidence_days)} days)</h4>
                                    <p><strong>Winner:</strong> {low_conf_winner} 우세</p>
                                    <p><strong>Model7:</strong> {model7_wins_low_conf}/{len(low_confidence_days)} ({model7_wins_low_conf/len(low_confidence_days)*100:.1f}%) | ROI: {np.mean([d['model7_roi'] for d in low_confidence_days]):.1f}%</p>
                                    <p><strong>SVM:</strong> {len(low_confidence_days)-model7_wins_low_conf}/{len(low_confidence_days)} ({(1-model7_wins_low_conf/len(low_confidence_days))*100:.1f}%) | ROI: {np.mean([d['svm_roi'] for d in low_confidence_days]):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    else:
                        st.warning("패턴 학습을 위한 데이터가 부족합니다.")
                else:
                    st.warning("비교할 수 있는 데이터가 없습니다.")
    
    # === 탭 2: 오늘의 추천 ===
    with tab2:
        st.header("🎯 Today's Model Recommendation")
        st.info("Get AI-powered recommendation for today's optimal model based on game conditions")
        
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **이 추천 시스템 사용법:**
            1. **Pattern Analysis 탭**에서 먼저 과거 패턴 학습
            2. **높은 신뢰도 (70%+)**: 추천 신뢰 가능
            3. **낮은 신뢰도 (60% 미만)**: 고급 분석 정보 확인 필요
            4. **100% 정확도**: 과적합 가능성, 신중하게 판단
            
            **주의사항:**
            - 교육 목적으로만 사용
            - 실제 베팅 결정은 본인 책임
            - 과거 성과 ≠ 미래 결과
            """)
        
        if 'rf_model' in st.session_state:
            rf_model = st.session_state['rf_model']
            selected_prefix = st.session_state.get('selected_prefix', 'None')
            
            with st.spinner("🤖 Analyzing today's conditions..."):
                recommendation = analyzer.recommend_today_model(rf_model, selected_prefix)
                
                if 'error' not in recommendation:
                    # 추천 결과 표시
                    confidence_level = recommendation['confidence']
                    confidence_color = "#28a745" if confidence_level > 0.7 else "#ffc107" if confidence_level > 0.6 else "#dc3545"
                    confidence_text = "High" if confidence_level > 0.7 else "Medium" if confidence_level > 0.6 else "Low"
                    
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h2>🎯 Today's Recommendation</h2>
                        <h1>{recommendation['recommended_model'].upper()}</h1>
                        <h3>Confidence: {recommendation['confidence']:.1%} ({confidence_text})</h3>
                        <p>Based on: {recommendation['latest_file']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence_level < 0.6:
                        st.warning("⚠️ **Low Confidence Warning**: 모델 간 성과 차이가 작습니다. 고급 분석 정보를 확인해보세요.")
                    
                    # 분석 근거
                    st.markdown("### 📋 Analysis Reasoning")
                    for reason in recommendation['reasoning']:
                        st.markdown(f"""
                        <div class="reasoning-item">
                            {reason}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 🔍 디버깅 정보 추가
                    st.markdown("### 🔍 Debug Information")
                    with st.expander("🔍 Advanced Analysis Details", expanded=False):
                        conditions = recommendation['today_conditions']
                        
                        # 중간값과 비교
                        if 'comparison_data' in st.session_state:
                            comparison_data = st.session_state['comparison_data']
                            consensus_values = [d['conditions'].get('consensus_strength', 0) for d in comparison_data]
                            underdog_values = [d['conditions'].get('underdog_ratio', 0) for d in comparison_data]
                            confidence_values = [d['conditions'].get('avg_confidence', 0) for d in comparison_data]
                            
                            consensus_median = np.median(consensus_values)
                            underdog_median = np.median(underdog_values)
                            confidence_median = np.median(confidence_values)
                            
                            st.markdown(f"""
                            **오늘의 조건 분류:**
                            - Consensus: {conditions.get('consensus_strength', 0):.3f} → {'HIGH' if conditions.get('consensus_strength', 0) > consensus_median else 'LOW'} (median: {consensus_median:.3f})
                            - Underdog: {conditions.get('underdog_ratio', 0):.1%} → {'HIGH' if conditions.get('underdog_ratio', 0) > underdog_median else 'LOW'} (median: {underdog_median:.1%})
                            - Confidence: {conditions.get('avg_confidence', 0):.3f} → {'HIGH' if conditions.get('avg_confidence', 0) > confidence_median else 'LOW'} (median: {confidence_median:.3f})
                            
                            **예상 패턴 기반 추천:** 위 조건들에서 과거에 어떤 모델이 더 우세했는지 확인 필요
                            """)
                            
                            # Random Forest 입력값 표시
                            feature_vector = [conditions.get('consensus_strength', 0), 
                                            conditions.get('underdog_ratio', 0), 
                                            conditions.get('avg_confidence', 0)]
                            st.markdown(f"**Random Forest 입력값:** {feature_vector}")
                            
                            # 🚨 Random Forest 디버깅
                            st.markdown("### 🚨 Random Forest Debug")
                            if 'rf_model' in st.session_state:
                                debug_rf = st.session_state['rf_model']
                                debug_prediction = debug_rf.predict([feature_vector])
                                debug_proba = debug_rf.predict_proba([feature_vector])
                                
                                st.markdown(f"""
                                **Random Forest 상세 정보:**
                                - 예측 결과: {debug_prediction[0]}
                                - 확률: model7={debug_proba[0][0]:.3f}, svm={debug_proba[0][1]:.3f}
                                - 클래스 순서: {debug_rf.classes_}
                                
                                                                 **분석:** Random Forest는 복잡한 패턴 조합을 고려합니다.
                                """)
                            
                            # 🎯 정확한 조합 분석
                            st.markdown("### 🎯 Exact Combination Analysis")
                            today_high_consensus = conditions.get('consensus_strength', 0) > consensus_median
                            today_low_underdog = conditions.get('underdog_ratio', 0) <= underdog_median
                            today_low_confidence = conditions.get('avg_confidence', 0) <= confidence_median
                            
                            # 오늘과 같은 조합의 과거 데이터 찾기
                            matching_days = []
                            for d in comparison_data:
                                day_high_consensus = d['conditions'].get('consensus_strength', 0) > consensus_median
                                day_low_underdog = d['conditions'].get('underdog_ratio', 0) <= underdog_median  
                                day_low_confidence = d['conditions'].get('avg_confidence', 0) <= confidence_median
                                
                                if (day_high_consensus == today_high_consensus and 
                                    day_low_underdog == today_low_underdog and 
                                    day_low_confidence == today_low_confidence):
                                    matching_days.append(d)
                            
                            if matching_days:
                                model7_wins_exact = sum(1 for d in matching_days if d['winner'] == 'model7')
                                svm_wins_exact = len(matching_days) - model7_wins_exact
                                model7_roi_exact = np.mean([d['model7_roi'] for d in matching_days])
                                svm_roi_exact = np.mean([d['svm_roi'] for d in matching_days])
                                
                                exact_winner = "Model7" if model7_wins_exact > svm_wins_exact else "SVM"
                                
                                st.markdown(f"""
                                **오늘과 정확히 같은 조합 (High Consensus + Low Underdog + Low Confidence):**
                                - 총 {len(matching_days)}일
                                - Model7: {model7_wins_exact}승 ({model7_wins_exact/len(matching_days)*100:.1f}%) | 평균 ROI: {model7_roi_exact:.1f}%
                                - SVM: {svm_wins_exact}승 ({svm_wins_exact/len(matching_days)*100:.1f}%) | 평균 ROI: {svm_roi_exact:.1f}%
                                - **과거 패턴 기준 우세:** {exact_winner}
                                
                                ✅ **이것이 Random Forest가 학습한 정확한 패턴입니다!**
                                """)
                            else:
                                st.warning("⚠️ 오늘과 정확히 같은 조합의 과거 데이터가 없습니다.")
                    
                    # 오늘의 조건 상세
                    st.markdown("### 📊 Today's Game Conditions")
                    conditions = recommendation['today_conditions']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Consensus Strength", f"{conditions.get('consensus_strength', 0):.2f}")
                    with col2:
                        st.metric("Underdog Ratio", f"{conditions.get('underdog_ratio', 0):.1%}")
                    with col3:
                        st.metric("Avg Confidence", f"{conditions.get('avg_confidence', 0):.3f}")
                    
                    # 🎯 핵심 3개 조건만 차트로 표시
                    condition_df = pd.DataFrame([
                        {'Condition': 'Consensus Strength', 'Value': conditions.get('consensus_strength', 0), 'Threshold': analyzer.thresholds['high_consensus']},
                        {'Condition': 'Underdog Ratio', 'Value': conditions.get('underdog_ratio', 0), 'Threshold': analyzer.thresholds['underdog_ratio']},
                        {'Condition': 'Avg Confidence', 'Value': conditions.get('avg_confidence', 0), 'Threshold': analyzer.thresholds['high_confidence']}
                    ])
                    
                    fig_conditions = go.Figure()
                    fig_conditions.add_trace(go.Bar(
                        x=condition_df['Condition'],
                        y=condition_df['Value'],
                        name='Today\'s Value',
                        marker_color='lightblue'
                    ))
                    fig_conditions.add_trace(go.Scatter(
                        x=condition_df['Condition'],
                        y=condition_df['Threshold'],
                        mode='markers',
                        name='Threshold',
                        marker=dict(color='red', size=10, symbol='diamond')
                    ))
                    fig_conditions.update_layout(
                        title='Today\'s Conditions vs Thresholds',
                        yaxis_title='Value',
                        height=400
                    )
                    st.plotly_chart(fig_conditions, use_container_width=True)
                    
                else:
                    st.error(recommendation['error'])
        else:
            st.warning("먼저 'Pattern Analysis' 탭에서 모델을 학습시켜주세요.")
    
    # === 탭 3: 히스토리컬 비교 ===
    with tab3:
        st.header("📈 Historical Model Comparison")
        st.info("Detailed comparison of Model7 vs SVM performance over time")
        
        # 구분자 선택 (독립적)
        selected_prefix_hist = st.selectbox(
            "📂 Select Data Source:",
            analyzer.get_available_prefixes(),
            format_func=lambda x: "None (기본)" if x == "None" else f"{x.rstrip('_')} (구분자)",
            key="hist_prefix"
        )
        
        # 날짜 선택 (독립적)
        available_dates_hist = analyzer.get_available_dates_for_prefix(selected_prefix_hist)
        if available_dates_hist:
            col1, col2 = st.columns(2)
            with col1:
                start_date_hist = st.date_input(
                    "Start Date",
                    value=datetime.strptime(available_dates_hist[0], '%Y-%m-%d').date(),
                    key="hist_start"
                )
            with col2:
                end_date_hist = st.date_input(
                    "End Date",
                    value=datetime.strptime(available_dates_hist[-1], '%Y-%m-%d').date(),
                    key="hist_end"
                )
            
            if st.button("📊 Analyze Historical Comparison"):
                with st.spinner("📈 Analyzing historical performance..."):
                    comparison_results = analyzer.analyze_daily_performance_comparison(
                        start_date_hist.strftime('%Y-%m-%d'),
                        end_date_hist.strftime('%Y-%m-%d'),
                        selected_prefix_hist
                    )
                    
                    if comparison_results and comparison_results['comparison_data']:
                        comparison_data = comparison_results['comparison_data']
                        
                        # 전체 통계
                        st.markdown("### 📊 Overall Comparison Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        model7_total_roi = sum(d['model7_roi'] for d in comparison_data) / len(comparison_data)
                        svm_total_roi = sum(d['svm_roi'] for d in comparison_data) / len(comparison_data)
                        
                        with col1:
                            st.metric("Total Days", len(comparison_data))
                        with col2:
                            st.metric("Model7 Avg ROI", f"{model7_total_roi:.2f}%")
                        with col3:
                            st.metric("SVM Avg ROI", f"{svm_total_roi:.2f}%")
                        with col4:
                            winner = "Model7" if model7_total_roi > svm_total_roi else "SVM"
                            st.metric("Overall Winner", winner)
                        
                        # 상세 테이블
                        st.markdown("### 📋 Daily Comparison Table")
                        table_df = pd.DataFrame([
                            {
                                'Date': d['date'],
                                'Model7 ROI (%)': d['model7_roi'],
                                'SVM ROI (%)': d['svm_roi'],
                                'Winner': d['winner'],
                                'Performance Gap': d['performance_gap'],
                                'Consensus': d['conditions'].get('consensus_strength', 0),
                                'Underdog Ratio': d['conditions'].get('underdog_ratio', 0)
                            }
                            for d in comparison_data
                        ])
                        
                        # 스타일링
                        def highlight_winner(row):
                            if row['Winner'] == 'model7':
                                return ['background-color: #d4edda'] * len(row)
                            else:
                                return ['background-color: #f8d7da'] * len(row)
                        
                        styled_table = table_df.style.apply(highlight_winner, axis=1).format({
                            'Model7 ROI (%)': '{:.2f}%',
                            'SVM ROI (%)': '{:.2f}%',
                            'Performance Gap': '{:.2f}%',
                            'Consensus': '{:.2f}',
                            'Underdog Ratio': '{:.1%}'
                        })
                        
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    
                    else:
                        st.warning("비교할 수 있는 데이터가 없습니다.")

if __name__ == "__main__":
    main()