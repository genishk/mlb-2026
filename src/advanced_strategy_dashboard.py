import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.advanced_betting_strategy_analyzer import AdvancedBettingStrategyAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="🎯 Advanced MLB Strategy Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 고급 스타일 설정
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .success-card {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #e8f5e8 0%, #ffffff 100%);
    }
    
    .warning-card {
        border-left: 5px solid #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffffff 100%);
    }
    
    .danger-card {
        border-left: 5px solid #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #ffffff 100%);
    }
    
    .info-card {
        border-left: 5px solid #17a2b8;
        background: linear-gradient(135deg, #d1ecf1 0%, #ffffff 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #2e7d32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2e7d32, #1f77b4);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .analysis-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .model-performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_analysis_results():
    """저장된 고급 분석 결과 로드"""
    results_dir = Path("data/advanced_analysis")
    
    if not results_dir.exists():
        return None
    
    # 가장 최근 분석 결과 찾기
    analysis_files = list(results_dir.glob("advanced_strategy_analysis_*.json"))
    
    if not analysis_files:
        return None
    
    # 여러 파일 시도 (손상된 파일 대비)
    for latest_file in sorted(analysis_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            st.warning(f"손상된 분석 파일 건너뜀: {latest_file.name}")
            continue
        except Exception as e:
            st.error(f"분석 결과 로드 실패: {e}")
            continue
    
    # 모든 파일이 손상된 경우
    st.error("모든 분석 결과 파일이 손상되었습니다. 새로운 분석을 실행해주세요.")
    return None

def convert_numpy_types(obj):
    """numpy 타입을 Python 네이티브 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def display_analysis_summary(results: Dict[str, Any]):
    """분석 요약 표시"""
    summary = results['analysis_summary']
    
    st.markdown(f"""
    <div class="analysis-summary">
        <h2>🎯 Analysis Intelligence Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-top: 1rem;">
            <div>
                <h3>{summary['total_models_analyzed']}</h3>
                <p>Models Analyzed</p>
            </div>
            <div>
                <h3>{summary['profitable_models']}</h3>
                <p>Profitable Models</p>
            </div>
            <div>
                <h3>{summary['combinations_found']}</h3>
                <p>Optimal Combinations</p>
            </div>
            <div>
                <h3>{summary['train_samples']}</h3>
                <p>Training Samples</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_model_performance_matrix(model_metrics: Dict[str, Any]):
    """모델 성과 매트릭스 시각화"""
    st.markdown('<div class="section-header">📊 Model Performance Intelligence Matrix</div>', unsafe_allow_html=True)
    
    # 데이터 준비
    performance_data = []
    for model_name, metrics in model_metrics.items():
        performance_data.append({
            'Model': model_name,
            'ROI (%)': metrics['overall_roi'],
            'Win Rate (%)': metrics['win_rate'],
            'Total Bets': metrics['total_bets'],
            'Consistency': metrics['strengths']['consistency_score'],
            'Stability': metrics['temporal_stability']['stability_score'],
            'Risk Score': abs(metrics['overall_roi']) / max(metrics['win_rate'], 1) * 100
        })
    
    df = pd.DataFrame(performance_data)
    
    if df.empty:
        st.warning("모델 성과 데이터가 없습니다.")
        return
    
    # 성과 히트맵
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # ROI vs Win Rate 스캐터 플롯
        fig_scatter = px.scatter(
            df, x='Win Rate (%)', y='ROI (%)', 
            size='Total Bets', color='Consistency',
            hover_name='Model',
            title="🎯 Model Performance Landscape",
            color_continuous_scale='viridis',
            size_max=20
        )
        
        # 4분면 구분선 추가
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even ROI")
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50% Win Rate")
        
        # 우수 모델 영역 표시
        fig_scatter.add_shape(
            type="rect", x0=50, y0=0, x1=100, y1=df['ROI (%)'].max() * 1.1,
            fillcolor="lightgreen", opacity=0.1,
            line=dict(color="green", width=1, dash="dot"),
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # 상위 모델 랭킹
        st.markdown("### 🏆 Top Performers")
        
        # 복합 점수 계산
        df['Composite Score'] = (
            df['ROI (%)'] * 0.4 +
            (df['Win Rate (%)'] - 50) * 0.3 +
            df['Consistency'] * 20 * 0.2 +
            df['Stability'] * 10 * 0.1
        )
        
        top_models = df.nlargest(5, 'Composite Score')
        
        for idx, row in top_models.iterrows():
            card_class = "success-card" if row['ROI (%)'] > 3 else "info-card" if row['ROI (%)'] > 0 else "warning-card"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h4>{row['Model']}</h4>
                <p><strong>ROI:</strong> {row['ROI (%)']:.2f}%</p>
                <p><strong>Win Rate:</strong> {row['Win Rate (%)']:.1f}%</p>
                <p><strong>Score:</strong> {row['Composite Score']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

def display_situational_analysis(model_metrics: Dict[str, Any]):
    """상황별 분석 시각화"""
    st.markdown('<div class="section-header">🎲 Situational Performance Analysis</div>', unsafe_allow_html=True)
    
    # 상황별 데이터 준비
    situational_data = []
    
    for model_name, metrics in model_metrics.items():
        sit_perf = metrics['situational_performance']
        situational_data.append({
            'Model': model_name,
            'Favorites ROI': sit_perf['favorites_roi'],
            'Underdogs ROI': sit_perf['underdogs_roi'],
            'Close Games ROI': sit_perf['close_games_roi'],
            'Blowout Games ROI': sit_perf['blowout_predictions_roi']
        })
    
    df_situational = pd.DataFrame(situational_data)
    
    if df_situational.empty:
        st.warning("상황별 성과 데이터가 없습니다.")
        return
    
    # 상황별 히트맵
    col1, col2 = st.columns(2)
    
    with col1:
        # 상황별 ROI 히트맵
        situation_columns = ['Favorites ROI', 'Underdogs ROI', 'Close Games ROI', 'Blowout Games ROI']
        heatmap_data = df_situational.set_index('Model')[situation_columns]
        
        fig_heatmap = px.imshow(
            heatmap_data.values,
            x=situation_columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn',
            title="🌡️ Situational ROI Heatmap",
            aspect="auto"
        )
        
        fig_heatmap.update_layout(
            xaxis_title="Situation Type",
            yaxis_title="Models"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # 최고 상황별 전문가
        st.markdown("### 🎯 Situation Specialists")
        
        specialists = {}
        for situation in situation_columns:
            best_model = df_situational.loc[df_situational[situation].idxmax()]
            specialists[situation] = {
                'model': best_model['Model'],
                'roi': best_model[situation]
            }
        
        for situation, spec in specialists.items():
            card_class = "success-card" if spec['roi'] > 5 else "info-card" if spec['roi'] > 0 else "warning-card"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h5>{situation.replace(' ROI', '')}</h5>
                <p><strong>Best Model:</strong> {spec['model']}</p>
                <p><strong>ROI:</strong> {spec['roi']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

def display_temporal_stability(model_metrics: Dict[str, Any]):
    """시간적 안정성 분석"""
    st.markdown('<div class="section-header">⏱️ Temporal Stability Analysis</div>', unsafe_allow_html=True)
    
    # 안정성 데이터 준비
    stability_data = []
    weekly_roi_data = []
    
    for model_name, metrics in model_metrics.items():
        temp_stability = metrics['temporal_stability']
        stability_data.append({
            'Model': model_name,
            'Stability Score': temp_stability['stability_score'],
            'Trend': temp_stability['trend_direction'],
            'Recent vs Past': temp_stability['recent_vs_past_performance']
        })
        
        # 주별 ROI 데이터
        weekly_rois = temp_stability.get('weekly_rois', [])
        for week, roi in enumerate(weekly_rois):
            weekly_roi_data.append({
                'Model': model_name,
                'Week': week + 1,
                'ROI': roi
            })
    
    df_stability = pd.DataFrame(stability_data)
    df_weekly = pd.DataFrame(weekly_roi_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_weekly.empty:
            # 주별 ROI 트렌드
            fig_trends = px.line(
                df_weekly, x='Week', y='ROI', color='Model',
                title="📈 Weekly ROI Trends",
                markers=True
            )
            
            fig_trends.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        if not df_stability.empty:
            # 안정성 점수 막대 차트
            fig_stability = px.bar(
                df_stability, x='Model', y='Stability Score',
                color='Stability Score',
                title="🎯 Model Stability Scores",
                color_continuous_scale='viridis'
            )
            
            fig_stability.update_xaxes(tickangle=45)
            st.plotly_chart(fig_stability, use_container_width=True)

def display_optimal_combinations(optimal_combinations: List[Dict]):
    """최적 조합 표시"""
    st.markdown('<div class="section-header">💎 Optimal Strategy Combinations</div>', unsafe_allow_html=True)
    
    if not optimal_combinations:
        st.info("발견된 최적 조합이 없습니다.")
        return
    
    for combo in optimal_combinations:
        situation = combo['situation'].replace('_', ' ').title()
        
        # 예상 ROI에 따른 카드 스타일 결정
        if combo['expected_roi'] > 10:
            card_class = "success-card"
            emoji = "🚀"
        elif combo['expected_roi'] > 5:
            card_class = "info-card"
            emoji = "📈"
        elif combo['expected_roi'] > 0:
            card_class = "warning-card"
            emoji = "⚡"
        else:
            card_class = "danger-card"
            emoji = "⚠️"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h3>{emoji} {situation}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p><strong>Expected ROI:</strong> {combo['expected_roi']:.2f}%</p>
                    <p><strong>Confidence:</strong> {combo['confidence_level']:.1f}%</p>
                </div>
                <div>
                    <p><strong>Models:</strong> {', '.join(combo['recommended_models'][:2])}{'...' if len(combo['recommended_models']) > 2 else ''}</p>
                </div>
            </div>
            <p style="margin-top: 1rem; font-style: italic; color: #666;">
                💡 {combo['reasoning']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 가중치 시각화
        if combo['weights']:
            weights_df = pd.DataFrame([
                {'Model': model, 'Weight': weight} 
                for model, weight in combo['weights'].items()
            ])
            
            fig_weights = px.pie(
                weights_df, values='Weight', names='Model',
                title=f"Model Weights for {situation}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)

def display_validation_results(validation_results: Dict[str, Any]):
    """검증 결과 표시 (과적합 방지)"""
    st.markdown('<div class="section-header">🛡️ Overfitting Prevention Validation</div>', unsafe_allow_html=True)
    
    if not validation_results:
        st.info("검증 결과가 없습니다.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Strategy Validation Results")
        
        for situation, validation in validation_results.items():
            reliability = "✅ Reliable" if validation['reliable'] else "⚠️ Overfitting Risk"
            card_class = "success-card" if validation['reliable'] else "warning-card"
            
            overfitting_risk = validation['overfitting_risk']
            risk_level = "Low" if abs(overfitting_risk) < 2 else "Medium" if abs(overfitting_risk) < 5 else "High"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h4>{situation.replace('_', ' ').title()}</h4>
                <p><strong>Status:</strong> {reliability}</p>
                <p><strong>Expected ROI:</strong> {validation['expected_roi']:.2f}%</p>
                <p><strong>Validation ROI:</strong> {validation['validation_roi']:.2f}%</p>
                <p><strong>Overfitting Risk:</strong> {risk_level} ({overfitting_risk:.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # 검증 성과 차트
        validation_data = []
        for situation, validation in validation_results.items():
            validation_data.append({
                'Strategy': situation.replace('_', ' ').title(),
                'Expected ROI': validation['expected_roi'],
                'Validation ROI': validation['validation_roi'],
                'Reliable': validation['reliable']
            })
        
        if validation_data:
            df_validation = pd.DataFrame(validation_data)
            
            fig_validation = go.Figure()
            
            fig_validation.add_trace(go.Bar(
                name='Expected ROI',
                x=df_validation['Strategy'],
                y=df_validation['Expected ROI'],
                marker_color='lightblue'
            ))
            
            fig_validation.add_trace(go.Bar(
                name='Validation ROI',
                x=df_validation['Strategy'],
                y=df_validation['Validation ROI'],
                marker_color='darkblue'
            ))
            
            fig_validation.update_layout(
                title="Expected vs Validation ROI",
                xaxis_title="Strategy",
                yaxis_title="ROI (%)",
                barmode='group'
            )
            
            st.plotly_chart(fig_validation, use_container_width=True)

def run_live_analysis():
    """실시간 분석 실행"""
    st.markdown('<div class="section-header">🔬 Live Strategy Analysis</div>', unsafe_allow_html=True)
    
    # ModelPerformanceTracker 임포트 (날짜별 파일 검색용)
    from model_performance_tracker import ModelPerformanceTracker
    
    # 사용 가능한 날짜 범위 확인
    tracker = ModelPerformanceTracker()
    prediction_files = tracker.get_latest_prediction_files_by_date()
    available_file_dates = sorted(prediction_files.keys())
    
    if not available_file_dates:
        st.error("No prediction files found. Please ensure prediction files exist.")
        return
    
    # 날짜 범위 설정 섹션
    st.markdown("### 📅 Date Range Selection")
    st.markdown("Select the date range for analysis. Only prediction files within this range will be included.")
    
    # Convert string dates to datetime objects for date picker
    min_date = datetime.strptime(available_file_dates[0], '%Y-%m-%d').date()
    max_date = datetime.strptime(available_file_dates[-1], '%Y-%m-%d').date()
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="analysis_start_date"
        )
    
    with col_date2:
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="analysis_end_date"
        )
    
    # 날짜 검증
    if start_date > end_date:
        st.error("⚠️ Start date cannot be later than end date. Please adjust your date selection.")
        return
    
    # Convert back to string format for filtering
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Filter files based on date range
    filtered_files = {
        date: path for date, path in prediction_files.items()
        if start_date_str <= date <= end_date_str
    }
    
    if not filtered_files:
        st.warning(f"⚠️ No prediction files found in the selected date range ({start_date_str} to {end_date_str}). Please adjust your date selection.")
        return
    
    # Display filtering info
    st.info(f"📊 Analysis will include {len(filtered_files)} prediction files from {start_date_str} to {end_date_str} (Total available: {len(prediction_files)} files)")
    
    # 데이터 소스 정보 표시
    with st.expander("View prediction files to be analyzed"):
        st.markdown("**Prediction Files in Selected Date Range:**")
        for date in sorted(filtered_files.keys()):
            file_name = Path(filtered_files[date]).name
            st.markdown(f"- {date}: `{file_name}`")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### ⚙️ Analysis Settings")
        # days_back은 이제 날짜 선택으로 대체되므로 제거하거나 비활성화
        st.info(f"Analysis period: {(end_date - start_date).days + 1} days")
        
    with col2:
        st.markdown("### 🎯 Advanced Options")
        min_confidence = st.slider("🎚️ Minimum Confidence", 0.5, 0.9, 0.55, 0.05, help="낮을수록 더 많은 베팅 기회")
        
    with col3:
        st.markdown("### 🚀 Execute Analysis")
        st.write("")  # 공간 확보
        analyze_button = st.button("🧠 Run Advanced Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("🔍 Advanced analysis in progress... This may take a few minutes."):
            try:
                # 진행률 표시
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # 분석기 초기화
                status_text.text("🔧 Initializing advanced analyzer...")
                progress_bar.progress(10)
                
                analyzer = AdvancedBettingStrategyAnalyzer()
                
                # 분석 실행 (날짜 범위 정보 전달)
                status_text.text("📊 Running comprehensive analysis...")
                progress_bar.progress(30)
                
                # 선택된 날짜 범위를 분석기에 전달
                results = analyzer.run_comprehensive_analysis(start_date_str, end_date_str)
                
                # 분석 결과에 사용된 날짜 범위 정보 추가
                results['date_range_used'] = {
                    'start_date': start_date_str,
                    'end_date': end_date_str,
                    'files_analyzed': len(filtered_files),
                    'total_files_available': len(prediction_files)
                }
                
                status_text.text("💾 Saving analysis results...")
                progress_bar.progress(80)
                
                # numpy 타입 변환
                results_converted = convert_numpy_types(results)
                
                # 결과 저장
                results_dir = Path("data/advanced_analysis")
                results_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = results_dir / f"advanced_strategy_analysis_{timestamp}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_converted, f, indent=2, ensure_ascii=False)
                
                status_text.text("✅ Analysis completed successfully!")
                progress_bar.progress(100)
                
                # 성공 메시지
                st.success("🎉 Advanced analysis completed successfully!")
                st.info(f"📁 Results saved to: {output_file}")
                st.info(f"📊 Analyzed {len(filtered_files)} files from {start_date_str} to {end_date_str}")
                
                # 자동 새로고침 안내
                st.warning("🔄 Please refresh the page to view the new analysis results.")
                
                # 새로고침 버튼
                if st.button("🔄 Refresh Dashboard"):
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

def main():
    # 메인 헤더
    st.markdown('<div class="main-header">🎯 Advanced MLB Strategy Intelligence</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666; font-size: 1.1em;">
        🚀 <strong>Next-Generation Strategy Analysis</strong> 🚀<br>
        Deep Learning • Overfitting Prevention • Risk-Adjusted Optimization • Real-Time Intelligence
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 네비게이션
    st.sidebar.markdown("## 🎮 Navigation Center")
    page = st.sidebar.radio(
        "Analysis Modules:",
        [
            "🏠 Dashboard Overview",
            "📊 Model Performance Matrix", 
            "🎲 Situational Analysis",
            "⏱️ Temporal Stability",
            "💎 Optimal Combinations",
            "🛡️ Validation Results",
            "🔬 Live Analysis"
        ],
        index=0
    )
    
    # 분석 결과 로드
    results = load_analysis_results()
    
    if page == "🏠 Dashboard Overview":
        if results:
            display_analysis_summary(results)
            
            # 주요 지표 요약
            col1, col2, col3 = st.columns(3)
            
            model_metrics = results.get('model_metrics', {})
            profitable_models = [m for m in model_metrics.values() if m['overall_roi'] > 0]
            
            with col1:
                st.metric(
                    "💰 Profitable Models",
                    f"{len(profitable_models)}/{len(model_metrics)}",
                    delta=f"{len(profitable_models)/len(model_metrics)*100:.0f}%" if model_metrics else "0%"
                )
            
            with col2:
                avg_roi = np.mean([m['overall_roi'] for m in profitable_models]) if profitable_models else 0
                st.metric(
                    "📈 Average ROI", 
                    f"{avg_roi:.2f}%",
                    delta="Profitable models only"
                )
            
            with col3:
                total_combinations = len(results.get('optimal_combinations', []))
                st.metric(
                    "🎯 Strategy Combinations",
                    total_combinations,
                    delta="Ready to deploy"
                )
                
            # 빠른 인사이트
            st.markdown('<div class="section-header">⚡ Quick Insights</div>', unsafe_allow_html=True)
            
            if profitable_models:
                best_model = max(profitable_models, key=lambda x: x['overall_roi'])
                worst_profitable = min(profitable_models, key=lambda x: x['overall_roi'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card success-card">
                        <h4>🏆 Top Performer</h4>
                        <p><strong>Model:</strong> {best_model.get('model_name', 'Unknown')}</p>
                        <p><strong>ROI:</strong> {best_model['overall_roi']:.2f}%</p>
                        <p><strong>Win Rate:</strong> {best_model['win_rate']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card info-card">
                        <h4>📊 Portfolio Diversity</h4>
                        <p><strong>Model Range:</strong> {worst_profitable['overall_roi']:.1f}% ~ {best_model['overall_roi']:.1f}%</p>
                        <p><strong>Strategy Count:</strong> {total_combinations}</p>
                        <p><strong>Risk Level:</strong> Diversified</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🔍 No analysis results found. Please run a live analysis first!")
            run_live_analysis()
    
    elif page == "📊 Model Performance Matrix":
        if results and 'model_metrics' in results:
            display_model_performance_matrix(results['model_metrics'])
        else:
            st.warning("Model performance data not available. Please run analysis first.")
    
    elif page == "🎲 Situational Analysis":
        if results and 'model_metrics' in results:
            display_situational_analysis(results['model_metrics'])
        else:
            st.warning("Situational analysis data not available. Please run analysis first.")
    
    elif page == "⏱️ Temporal Stability":
        if results and 'model_metrics' in results:
            display_temporal_stability(results['model_metrics'])
        else:
            st.warning("Temporal stability data not available. Please run analysis first.")
    
    elif page == "💎 Optimal Combinations":
        if results and 'optimal_combinations' in results:
            display_optimal_combinations(results['optimal_combinations'])
        else:
            st.warning("Optimal combinations data not available. Please run analysis first.")
    
    elif page == "🛡️ Validation Results":
        if results and 'validation_results' in results:
            display_validation_results(results['validation_results'])
        else:
            st.warning("Validation results not available. Please run analysis first.")
    
    elif page == "🔬 Live Analysis":
        run_live_analysis()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
        <p>🎯 <strong>Advanced MLB Strategy Intelligence</strong> | Powered by Machine Learning & Statistical Analysis</p>
        <p>⚠️ <em>All strategies are based on historical data. Past performance does not guarantee future results.</em></p>
        <p>🔬 <em>Overfitting prevention mechanisms are actively applied to all analysis results.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 