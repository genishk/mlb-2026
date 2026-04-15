import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from pathlib import Path
import os
from models.mlb_betting_optimizer import MLBBettingOptimizer
from model_performance_tracker import ModelPerformanceTracker
from betting_performance_tracker import BettingPerformanceTracker

st.set_page_config(
    page_title="MLB Betting Optimization System",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color theme settings (MLB colors)
MLB_COLORS = {
    'primary': '#002D72',    # MLB blue
    'secondary': '#E31937',  # MLB red
    'accent': '#FDB827',     # Gold accent
    'background': '#F5F5F5', # Background color
    'text': '#333333',       # Text color
}

# Apply styles
st.markdown(f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {MLB_COLORS['primary']};
        font-weight: bold;
    }}
    .stButton>button {{
        background-color: {MLB_COLORS['primary']};
        color: white;
    }}
    .stProgress .st-bo {{
        background-color: {MLB_COLORS['secondary']};
    }}
    .highlight {{
        background-color: {MLB_COLORS['accent']};
        color: {MLB_COLORS['primary']};
        padding: 0.3rem;
        border-radius: 0.3rem;
    }}
    .good-value {{
        color: green;
        font-weight: bold;
    }}
    .bad-value {{
        color: red;
        font-weight: bold;
    }}
    .info-box {{
        background-color: {MLB_COLORS['secondary']};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #F0F0F0;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {MLB_COLORS['primary']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Main title and description
st.title("MLB Betting Optimization System ⚾")
st.markdown("""
This system analyzes MLB game predictions and current odds to identify the most valuable betting opportunities.
It optimizes capital allocation through the Kelly Criterion and portfolio optimization techniques, with a focus on 3-5 team parlays.
""")

# MLB team abbreviations and full names mapping
TEAM_ABBREV = {
    'ARI': 'Arizona Diamondbacks',
    'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CWS': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',
    'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',
    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets',
    'NYY': 'New York Yankees',
    'OAK': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates',
    'SD': 'San Diego Padres',
    'SF': 'San Francisco Giants',
    'SEA': 'Seattle Mariners',
    'STL': 'St. Louis Cardinals',
    'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',
    'WSH': 'Washington Nationals'
}

# Reverse mapping from full names to abbreviations
TEAM_FULLNAME = {v: k for k, v in TEAM_ABBREV.items()}

# File loading functions
def load_predictions_with_odds():
    """Load the latest predictions with odds file"""
    data_dir = Path(__file__).parent / "odds" / "data" / "matched"
    pred_files = list(data_dir.glob("mlb_predictions_with_odds_*_active.json"))
    if not pred_files:
        st.error("No predictions with odds file found")
        st.stop()
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading predictions and odds from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

# Load predictions with odds data
predictions_data = load_predictions_with_odds()

# Sidebar settings
st.sidebar.header("Navigation")

# Page selection
page_selection = st.sidebar.radio(
    "Select Page",
    ["Betting Analysis", "Model Performance Tracking", "Betting Performance Tracking"],
    index=0
)

st.sidebar.header("Settings")

# Bankroll settings (only for betting analysis)
if page_selection == "Betting Analysis":
    bankroll = st.sidebar.slider("Betting Bankroll ($)", 100, 5000, 500, 50)
    # Minimum bet amount
    min_bet = st.sidebar.slider("Minimum Bet Amount ($)", 1, 20, 5, 1)
    # ROI 임계값 설정 - 분석 전에 설정할 수 있도록 추가
    roi_threshold = st.sidebar.slider("Minimum ROI Threshold (%)", 0.0, 30.0, 2.0, 1.0)
else:
    # Default values for model performance page
    bankroll = 500
    min_bet = 5
    roi_threshold = 2.0

# Main content area
if page_selection == "Betting Analysis":
    st.markdown(f"## MLB Games Schedule ({len(predictions_data)} Games)")

    # Add visual game cards
    st.markdown("### Game Cards")

    # Calculate number of columns based on number of games
    cols_per_row = 3

    # Group games by date
    games_by_date = {}
    for game in predictions_data:
        date = game['date'].split('T')[0] if 'T' in game['date'] else game['date']
        if date not in games_by_date:
            games_by_date[date] = []
        games_by_date[date].append(game)

    # Sort dates
    sorted_dates = sorted(games_by_date.keys())

    # Display games by date
    for date in sorted_dates:
        st.markdown(f"#### {date}")
        
        # Create rows of games
        games = games_by_date[date]
        for i in range(0, len(games), cols_per_row):
            cols = st.columns(cols_per_row)
            
            # Add games to columns
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(games):
                    game = games[idx]
                    
                    # Determine the winner and styling
                    home_team = game['home_team']
                    away_team = game['away_team']
                    predicted_winner = game['predicted_winner']
                    win_prob = game['win_probability'] * 100
                    
                    # Calculate both teams' win probabilities
                    # win_probability는 항상 홈팀의 승리 확률
                    home_win_prob = win_prob  # 홈팀 승리 확률
                    away_win_prob = 100 - win_prob  # 원정팀 승리 확률
                    
                    home_odds = game['home_team_odds']
                    away_odds = game['away_team_odds']
                    
                    home_style = "font-weight: bold; color: green;" if predicted_winner == home_team else ""
                    away_style = "font-weight: bold; color: green;" if predicted_winner == away_team else ""
                    
                    # Initialize all HTML badge variables
                    underdog_badge = ""
                    
                    # Determine if this is an underdog pick
                    is_underdog_pick = False
                    
                    if home_odds is not None and away_odds is not None:
                        # In American odds, positive is underdog, negative is favorite
                        if predicted_winner == home_team and ((home_odds > 0 and away_odds < 0) or (home_odds > away_odds)):
                            is_underdog_pick = True
                        elif predicted_winner == away_team and ((away_odds > 0 and home_odds < 0) or (away_odds > home_odds)):
                            is_underdog_pick = True
                    
                    if is_underdog_pick:
                        underdog_badge = f"""<span style="background-color: {MLB_COLORS['secondary']}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; display: inline-block; margin-left: 5px;">UNDERDOG PICK</span>"""
                    
                    # Create card HTML
                    with cols[j]:
                        card_html = f"""
                        <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; background-color: #f8f9fa;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div style="{away_style}">
                                    {away_team}
                                    <div style="font-size: 0.8em; color: #666;">
                                        {f"{away_odds:+g}" if away_odds is not None else "N/A"}
                                    </div>
                                    <div style="font-size: 0.7em; color: #333; margin-top: 2px;">
                                        {away_win_prob:.1f}%
                                    </div>
                                </div>
                                <div>@</div>
                                <div style="{home_style}">
                                    {home_team}
                                    <div style="font-size: 0.8em; color: #666;">
                                        {f"{home_odds:+g}" if home_odds is not None else "N/A"}
                                    </div>
                                    <div style="font-size: 0.7em; color: #333; margin-top: 2px;">
                                        {home_win_prob:.1f}%
                                    </div>
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 8px;">
                                <div style="font-weight: bold;">Model Pick: {predicted_winner} {underdog_badge}</div>
                                <div>Win Probability: {away_win_prob if predicted_winner == away_team else home_win_prob:.1f}%</div>
                                <div style="margin-top: 5px; height: 8px; background-color: #e9ecef; border-radius: 4px;">
                                    <div style="height: 100%; width: {away_win_prob if predicted_winner == away_team else home_win_prob}%; background-color: {MLB_COLORS['primary']}; border-radius: 4px;"></div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

    # Add space between card view and table view
    st.markdown("---")
    st.markdown("### Detailed Game Table")

    # Create match display list
    match_display_list = []
    for game in predictions_data:
        # odds가 있는 경기만 표시 (N/A는 표시하되, 배팅 분석에서는 제외)
        match_display_list.append({
            "date": game['date'],
            "home_team": game['home_team'],
            "away_team": game['away_team'],
            "predicted_winner": game['predicted_winner'],
            "win_probability": game['win_probability'],
            "home_odds": game['home_team_odds'],
            "away_odds": game['away_team_odds']
        })

    # Prepare odds data for optimizer
    team_date_odds = {}
    for game in predictions_data:
        date = game['date'].split('T')[0] if 'T' in game['date'] else game['date']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Store odds with date in key
        team_date_odds[f"{home_team}_{date}"] = str(game['home_team_odds']) if game['home_team_odds'] is not None else "0"
        team_date_odds[f"{away_team}_{date}"] = str(game['away_team_odds']) if game['away_team_odds'] is not None else "0"

    # Display games in a table
    games_df = pd.DataFrame(match_display_list)
    if not games_df.empty:
        # Format probabilities and odds - predicted_winner의 실제 확률 계산
        def get_winner_probability(row):
            home_prob = row['win_probability']  # 홈팀 승리 확률
            if row['predicted_winner'] == row['home_team']:
                return f"{home_prob*100:.1f}%"  # 홈팀이 predicted_winner
            else:
                return f"{(1-home_prob)*100:.1f}%"  # 원정팀이 predicted_winner
        
        games_df['Win Probability'] = games_df.apply(get_winner_probability, axis=1)
        games_df['Home Odds'] = games_df['home_odds'].apply(lambda x: f"{x:+g}" if x is not None else "N/A")
        games_df['Away Odds'] = games_df['away_odds'].apply(lambda x: f"{x:+g}" if x is not None else "N/A")
        
        # Reorder and rename columns
        display_df = games_df[[
            'date', 
            'away_team', 
            'Away Odds',
            'home_team', 
            'Home Odds',
            'predicted_winner',
            'Win Probability'
        ]].rename(columns={
            'date': 'Date',
            'away_team': 'Away Team',
            'home_team': 'Home Team',
            'predicted_winner': 'Model Pick'
        })
        
        # Add row index to the DataFrame
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1  # Start from 1 instead of 0
        
        st.dataframe(
            display_df,
            hide_index=False,  # Show the index column
            use_container_width=True
        )

    # Run betting analysis button
    if st.sidebar.button("Run Betting Analysis", key="run_analysis"):
        with st.spinner("Analyzing betting opportunities..."):
            optimizer = MLBBettingOptimizer()
            optimizer.min_bet_amount = min_bet
            optimizer.min_roi_threshold = roi_threshold
            
            # Pass the odds data to optimizer
            portfolio = optimizer.analyze_and_save(team_date_odds, bankroll)
            
            # Cache analysis results
            st.session_state['portfolio'] = portfolio
            st.session_state['bankroll'] = bankroll
            st.session_state['roi_threshold'] = roi_threshold
            st.success("Analysis completed!")

    # Display analysis results if available
    if 'portfolio' in st.session_state:
        portfolio = st.session_state['portfolio']
        bankroll = st.session_state['bankroll']
        
        # Portfolio summary
        st.markdown("## Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Investment",
                f"${portfolio['total_investment']:.2f}",
                f"{(portfolio['total_investment']/bankroll)*100:.1f}% of bankroll"
            )
        
        with col2:
            if portfolio['total_investment'] > 0:
                roi = (portfolio['expected_profit']/portfolio['total_investment'])*100
            else:
                roi = 0.0
                
            st.metric(
                "Expected Profit",
                f"${portfolio['expected_profit']:.2f}",
                f"{roi:.1f}% ROI"
            )
        
        with col3:
            st.metric(
                "Maximum Loss",
                f"${portfolio['max_loss']:.2f}",
                f"{(portfolio['max_loss']/bankroll)*100:.1f}% of bankroll"
            )
        
        with col4:
            total_bets = len(portfolio['singles']) + len(portfolio['parlays'])
            st.metric("Total Bets", total_bets)
        
        # Create tabs for different bet types
        tabs = st.tabs(["Single Bets", "Parlay Bets"])
        
        # Single bets tab
        with tabs[0]:
            if portfolio.get('all_single_bets'):
                # 모든 싱글 베팅 데이터 (ROI 필터링 전)
                singles_data = []
                for bet in portfolio['all_single_bets']:
                    # ROI 값 가져오기
                    roi_num = bet['roi']
                    
                    singles_data.append({
                        'Team': bet['team'],
                        'Match': bet['match'],
                        'Date': bet['date'],
                        'Win Prob': bet['probability']*100,  # 실제 숫자 값
                        'Odds': f"+{bet['odds']:.0f}" if bet['odds'] > 0 else f"{bet['odds']:.0f}",
                        'Amount': bet['amount'],  # 실제 숫자 값
                        'Potential Profit': bet['potential_profit'],  # 실제 숫자 값
                        'Expected Value': bet['expected_profit'],  # 실제 숫자 값
                        'ROI': roi_num,  # 실제 숫자 값
                        # Hidden numeric columns for sorting
                        'prob_num': float(bet['probability']),
                        'amount_num': float(bet['amount']),
                        'potential_profit_num': float(bet['potential_profit']),
                        'expected_profit_num': float(bet['expected_profit']),
                        'roi_num': roi_num
                    })
                
                singles_df = pd.DataFrame(singles_data)
                
                # 필터링 전 싱글 베팅 수
                total_singles = len(singles_df)
                
                # ROI 임계값 가져오기 (기본값 5.0)
                roi_threshold = st.session_state.get('roi_threshold', 5.0)
                
                # Filter bets with ROI >= threshold
                singles_df = singles_df[singles_df['roi_num'] >= roi_threshold]
                
                # 필터링 후 싱글 베팅 수
                filtered_singles = len(singles_df)
                
                # Sort by ROI descending
                singles_df = singles_df.sort_values('roi_num', ascending=False)
                singles_df = singles_df.reset_index(drop=True)
                singles_df.index = singles_df.index + 1  # Start from 1 instead of 0
                
                # 필터링 결과 표시
                st.markdown(f"### Single Bets ({filtered_singles} of {total_singles} with ROI ≥ {roi_threshold}%)")
                
                # Hide numeric sorting columns
                display_cols = [col for col in singles_df.columns if not col.endswith('_num')]
                
                st.data_editor(
                    singles_df[display_cols],
                    column_config={
                        "ROI": st.column_config.NumberColumn(
                            "ROI",
                            format="%.1f%%",
                        ),
                        "Win Prob": st.column_config.NumberColumn(
                            "Win Prob",
                            format="%.1f%%",
                        ),
                        "Amount": st.column_config.NumberColumn(
                            "Amount",
                            format="$%.2f",
                        ),
                        "Potential Profit": st.column_config.NumberColumn(
                            "Potential Profit", 
                            format="$%.2f",
                        ),
                        "Expected Value": st.column_config.NumberColumn(
                            "Expected Value",
                            format="$%.2f",
                        ),
                    },
                    hide_index=False,
                    use_container_width=True,
                    disabled=True
                )
                
                # Add selection criteria explanation
                with st.expander("Single Bet Selection Criteria"):
                    st.markdown("""
                    **Single Bet Selection Criteria:**
                    
                    The following criteria are used to filter bets:
                    
                    1. **Odds Check**: Bets with no odds are excluded.
                    
                    2. **Win Probability Check**: 
                       - All teams with a 40% or higher win probability are considered, not just the predicted winners
                       - This allows for underdog bets with positive expected value
                    
                    3. **Expected Value (EV) Check**:
                       - EV = (Win Probability × Potential Profit) - ((1-Win Probability) × Bet Amount)
                       - **Only bets with positive EV (EV > 0) are selected.**
                    
                    4. **Same Game Filter**:
                       - When both teams in the same game have positive EV, only the team with higher EV is selected
                       - This avoids betting on both sides of the same game, which could limit overall returns
                    
                    **Bet Size Determination Method:**
                    
                    * **Kelly Criterion Calculation**:
                      - Optimal Bet = Bankroll × ((Win Probability × Odds - 1) / (Odds - 1)) / 4
                      - We use 1/4 Kelly for a conservative approach.
                      - The Kelly Criterion is only used for bet size determination, not for filtering.
                    
                    * **Bet Limit**:
                      - Minimum Bet: $5 (can be changed in settings)
                      - Maximum Bet: 5% of total funds
                      
                    * **Profit Calculation**:
                      - For positive odds (+100): Bet Amount × (Odds/100)
                      - For negative odds (-100): Bet Amount × (100/|Odds|)
                    """)
            else:
                st.info("No profitable single bets found")
        
        # Parlay bets tab
        with tabs[1]:
            if portfolio['parlays']:
                parlays_data = []
                for parlay in portfolio['parlays']:
                    teams = " + ".join(parlay['teams'])
                    
                    # 각 팀에 대한 상세 정보
                    teams_with_details = []
                    for pick in parlay['picks']:
                        match_parts = pick['match'].split(' vs ')
                        opponent = match_parts[0] if match_parts[1] == pick['team'] else match_parts[1]
                        
                        # Use team abbreviations instead of full names
                        team_abbr = TEAM_FULLNAME.get(pick['team'], pick['team'])
                        opponent_abbr = TEAM_FULLNAME.get(opponent, opponent)
                        
                        # Format date more compactly (if date exists)
                        if 'date' in pick:
                            # Convert to date object and format as MM/DD
                            try:
                                date_obj = datetime.strptime(pick['date'], '%Y-%m-%d')
                                short_date = date_obj.strftime('%m/%d')
                                teams_with_details.append(f"{team_abbr} ({short_date} vs {opponent_abbr})")
                            except:
                                # Fallback if date parsing fails
                                teams_with_details.append(f"{team_abbr} (vs {opponent_abbr})")
                        else:
                            teams_with_details.append(f"{team_abbr} (vs {opponent_abbr})")
                    
                    teams_detail = " + ".join(teams_with_details)
                    
                    # Calculate ROI as a numeric value for sorting
                    roi_num = (parlay['expected_profit'] / parlay['amount']) * 100 if parlay['amount'] > 0 else 0
                    
                    parlays_data.append({
                        'Teams': teams,
                        'Teams_Detail': teams_detail,
                        'Type': parlay['type'],
                        'Win Prob': parlay['probability']*100,  # 실제 숫자 값
                        'Base Odds': parlay['odds'],  # 실제 숫자 값
                        'Boosted Odds': parlay['boosted_odds'],  # 실제 숫자 값
                        'Amount': parlay['amount'],  # 실제 숫자 값
                        'Potential Profit': parlay['potential_profit'],  # 실제 숫자 값
                        'Expected Value': parlay['expected_profit'],  # 실제 숫자 값
                        'ROI': roi_num,  # 실제 숫자 값
                        # Hidden numeric columns for sorting
                        'prob_num': float(parlay['probability']),
                        'base_odds_num': float(parlay['odds']),
                        'boosted_odds_num': float(parlay['boosted_odds']),
                        'amount_num': float(parlay['amount']),
                        'potential_profit_num': float(parlay['potential_profit']),
                        'expected_profit_num': float(parlay['expected_profit']),
                        'roi_num': roi_num,
                        'parlay_size': int(parlay['type'].split('_')[0])
                    })
                
                parlays_df = pd.DataFrame(parlays_data)
                
                # 필터링 전 파레이 베팅 수
                total_parlays = len(parlays_df)
                
                # ROI 임계값 가져오기 (기본값 5.0%)
                roi_threshold = st.session_state.get('roi_threshold', 5.0)
                
                # Filter parlays with ROI >= threshold
                parlays_df = parlays_df[parlays_df['roi_num'] >= roi_threshold]
                
                # 필터링 후 파레이 베팅 수
                filtered_parlays = len(parlays_df)
                
                # First sort by ROI descending
                parlays_df = parlays_df.sort_values('roi_num', ascending=False)
                parlays_df = parlays_df.reset_index(drop=True)
                parlays_df.index = parlays_df.index + 1
                
                # 필터링 결과 표시
                st.markdown(f"### Parlay Bets ({filtered_parlays} of {total_parlays} with ROI ≥ {roi_threshold}%)")
                
                # 팀/경기별 추천 횟수 계산
                pick_counts = {}
                for parlay in portfolio['parlays']:
                    for pick in parlay['picks']:
                        pick_key = f"{pick['team']} ({pick['date']} vs {pick['match']})"
                        if pick_key not in pick_counts:
                            pick_counts[pick_key] = 0
                        pick_counts[pick_key] += 1
                
                # 추천 횟수가 많은 순으로 정렬
                sorted_pick_counts = dict(sorted(pick_counts.items(), key=lambda item: item[1], reverse=True))
                
                # 3팀, 4팀, 5팀 파라레이로 분리하여 탭으로 표시
                parlay_sizes = sorted(parlays_df['parlay_size'].unique())
                parlay_tabs = st.tabs([f"{size}-Team Parlays" for size in parlay_sizes])
                
                for i, size in enumerate(parlay_sizes):
                    with parlay_tabs[i]:
                        size_parlays = parlays_df[parlays_df['parlay_size'] == size]
                        size_count = len(size_parlays)
                        
                        if not size_parlays.empty:
                            st.markdown(f"#### {size}-Team Parlays ({size_count} recommendations)")
                            
                            # 동일 금액 계산 (모든 팔레이의 amount 합계 / 팔레이 수)
                            total_amount = size_parlays['amount_num'].sum()
                            avg_amount = total_amount / size_count if size_count > 0 else 0
                            
                            # adjusted_amount 열 추가 (copy를 만들어서 warning 방지)
                            size_parlays = size_parlays.copy()
                            size_parlays['adjusted_amount'] = avg_amount
                            
                            # 표시할 열 선택 (Teams_Detail은 팀+날짜+상대팀 정보가 포함된 상세 정보)
                            display_cols = ['Teams_Detail', 'Win Prob', 'Boosted Odds', 'Amount', 'adjusted_amount', 'Potential Profit', 'Expected Value', 'ROI']
                            
                            # 테이블 표시
                            st.dataframe(
                                size_parlays[display_cols].rename(columns={'Teams_Detail': 'Teams', 'adjusted_amount': 'Adjusted Amount'}),
                                column_config={
                                    "Teams": st.column_config.TextColumn(
                                        "Teams",
                                        width="medium",
                                        help="Team abbreviations with date (MM/DD) and opponent"
                                    ),
                                    "ROI": st.column_config.NumberColumn(
                                        "ROI",
                                        format="%.1f%%",
                                    ),
                                    "Win Prob": st.column_config.NumberColumn(
                                        "Win Prob",
                                        format="%.1f%%",
                                    ),
                                    "Boosted Odds": st.column_config.NumberColumn(
                                        "Boosted Odds",
                                        format="%.2f",
                                    ),
                                    "Amount": st.column_config.NumberColumn(
                                        "Amount",
                                        format="$%.2f",
                                    ),
                                    "Adjusted Amount": st.column_config.NumberColumn(
                                        "Adjusted Amount",
                                        format="$%.2f",
                                    ),
                                    "Potential Profit": st.column_config.NumberColumn(
                                        "Potential Profit", 
                                        format="$%.2f",
                                    ),
                                    "Expected Value": st.column_config.NumberColumn(
                                        "Expected Value",
                                        format="$%.2f",
                                    ),
                                },
                                hide_index=False,
                                use_container_width=True
                            )
                            
                            # Add expander to view full team details
                            with st.expander("View full team details"):
                                for idx, row in size_parlays.iterrows():
                                    st.markdown(f"**Parlay #{idx}**: {row['Teams']}")
                                    
                                    # Get the original picks for this parlay
                                    for parlay in portfolio['parlays']:
                                        if parlay['type'] == f"{size}_team" and parlay['probability']*100 == row['Win Prob']:
                                            for j, pick in enumerate(parlay['picks'], 1):
                                                match_parts = pick['match'].split(' vs ')
                                                opponent = match_parts[0] if match_parts[1] == pick['team'] else match_parts[1]
                                                st.markdown(f"- **{pick['team']}** vs {opponent} ({pick['date']})")
                                            st.markdown("---")
                                            break
                        
                            # Pick usage section
                            if f'{size}_team' in portfolio['pick_usage']:
                                st.markdown(f"##### Pick Usage Frequency in {size}-Team Parlays")
                                usage_data = portfolio['pick_usage'][f'{size}_team']
                                
                                if usage_data:
                                    # 테이블 형태로 표시
                                    usage_df = pd.DataFrame(usage_data)
                                    st.dataframe(
                                        usage_df,
                                        column_config={
                                            "team": st.column_config.TextColumn("Team"),
                                            "match": st.column_config.TextColumn("Match"),
                                            "date": st.column_config.TextColumn("Date"),
                                            "count": st.column_config.NumberColumn("Usage Count")
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                    
                                    # 총 픽 사용 수 표시
                                    total_picks = sum(item['count'] for item in usage_data)
                                    st.info(f"Total picks used: {total_picks} (Expected: {size * size_count})")
                                    
                                    # 최소 사용 횟수를 충족하지 못한 픽이 있는지 확인
                                    min_pick_usage = 3 if size == 3 else (3 if size == 4 else 4)
                                    picks_below_min = [item for item in usage_data if item['count'] < min_pick_usage]
                                    
                                    if picks_below_min:
                                        st.warning(f"{len(picks_below_min)} picks do not meet minimum usage requirement ({min_pick_usage})")
                                        with st.expander("View picks below minimum usage"):
                                            min_usage_df = pd.DataFrame(picks_below_min)
                                            st.dataframe(
                                                min_usage_df,
                                                column_config={
                                                    "team": st.column_config.TextColumn("Team"),
                                                    "match": st.column_config.TextColumn("Match"),
                                                    "date": st.column_config.TextColumn("Date"),
                                                    "count": st.column_config.NumberColumn("Usage Count")
                                                },
                                                hide_index=True,
                                                use_container_width=True
                                            )
                                    else:
                                        st.success(f"All picks meet the minimum usage requirement ({min_pick_usage})")
                                        
                        else:
                            st.info(f"No {size}-team parlays with ROI ≥ {roi_threshold}%")
                
                # Detailed parlay information
                st.markdown("### Detailed Parlay Information")
                for i, parlay in enumerate(portfolio['parlays'], 1):
                    roi = (parlay['expected_profit'] / parlay['amount']) * 100 if parlay['amount'] > 0 else 0
                    if roi >= roi_threshold:
                        with st.expander(f"Parlay #{i} - {parlay['type']} (ROI: {roi:.1f}%)"):
                            st.markdown("#### Picks")
                            for j, pick in enumerate(parlay['picks'], 1):
                                match_parts = pick['match'].split(' vs ')
                                opponent = match_parts[0] if match_parts[1] == pick['team'] else match_parts[1]
                                st.markdown(f"""
                                **Pick {j}**: **{pick['team']}** vs {opponent}
                                - Date: {pick['date']}
                                - Win Probability: {pick['probability']*100:.1f}%
                                - Odds: {format(pick['odds'], "+g") if pick['odds'] is not None else "N/A"}
                                """)
                            
                            # Find the adjusted amount for this parlay size
                            parlay_size = int(parlay['type'].split('_')[0])
                            size_parlays = parlays_df[parlays_df['parlay_size'] == parlay_size]
                            if not size_parlays.empty:
                                total_amount = size_parlays['amount_num'].sum()
                                adjusted_amount = total_amount / len(size_parlays) if len(size_parlays) > 0 else 0
                            else:
                                adjusted_amount = 0
                            
                            st.markdown("#### Parlay Details")
                            st.markdown(f"""
                            - Combined Probability: {parlay['probability']*100:.1f}%
                            - Base Odds: {parlay['odds']:.2f}
                            - Boosted Odds: {parlay['boosted_odds']:.2f}
                            - Bet Amount: ${parlay['amount']:.2f}
                            - Adjusted Amount: ${adjusted_amount:.2f}
                            - Potential Profit: ${parlay['potential_profit']:.2f}
                            - Expected Value: ${parlay['expected_profit']:.2f}
                            - ROI: {roi:.1f}%
                            """)
            else:
                st.info("No profitable parlay bets found")
    else:
        st.info("Click 'Run Betting Analysis' to see betting recommendations")

# Model Performance Tracking Page
elif page_selection == "Model Performance Tracking":
    st.markdown("## Model Performance Tracking 📊")
    st.markdown("""
    This section analyzes the historical performance of our three individual models (LightGBM, CatBoost, XGBoost) 
    and the ensemble model by comparing predictions with actual game results.
    """)
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker()
    
    # Load and analyze performance
    with st.spinner("Loading and analyzing model performance..."):
        try:
            # 초기 로드는 모든 데이터로 수행하여 날짜 범위 확인
            performance, matched_data, confidence_analysis = tracker.get_performance_summary()
            
            if not performance or not matched_data:
                st.error("No performance data available. Please ensure prediction and historical record files exist.")
                st.stop()
            
            # Date filtering section
            st.markdown("### Date Filtering")
            
            # Get available date range from prediction files instead of game dates
            prediction_files = tracker.get_latest_prediction_files_by_date()
            available_file_dates = sorted(prediction_files.keys())
            
            if available_file_dates:
                # Convert string dates to datetime objects for date picker
                min_date = datetime.strptime(available_file_dates[0], '%Y-%m-%d').date()
                max_date = datetime.strptime(available_file_dates[-1], '%Y-%m-%d').date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date", 
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                # Convert back to string format for filtering
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # 날짜 필터링이 전체 범위가 아닌 경우 해당 날짜 범위의 데이터만 다시 로드
                if start_date_str != available_file_dates[0] or end_date_str != available_file_dates[-1]:
                    # 선택된 날짜 범위의 데이터만 로드
                    filtered_performance, filtered_data, filtered_confidence_analysis = tracker.get_performance_summary(start_date_str, end_date_str)
                    
                    # Display filtering info
                    selected_dates = [date for date in available_file_dates if start_date_str <= date <= end_date_str]
                    st.info(f"Showing {len(filtered_data)} games from prediction files dated {start_date_str} to {end_date_str} (Total available: {len(matched_data)} games)")
                    
                    # 데이터 소스 정보 표시
                    st.markdown("#### Data Sources")
                    st.markdown("**View data source files**")
                    st.markdown("**Prediction Files Used:**")
                    for date in selected_dates:
                        if date in prediction_files:
                            file_name = Path(prediction_files[date]).name
                            st.markdown(f"- {date}: {file_name}")
                else:
                    # 전체 범위인 경우 기존 데이터 사용
                    filtered_performance = performance
                    filtered_data = matched_data
                    filtered_confidence_analysis = confidence_analysis
                    
                    # Display filtering info
                    st.info(f"Showing {len(filtered_data)} games from prediction files dated {start_date_str} to {end_date_str} (Total available: {len(matched_data)} games)")
                    
                    # 데이터 소스 정보 표시
                    st.markdown("#### Data Sources")
                    st.markdown("**View data source files**")
                    st.markdown("**Prediction Files Used:**")
                    for date in available_file_dates:
                        if date in prediction_files:
                            file_name = Path(prediction_files[date]).name
                            st.markdown(f"- {date}: {file_name}")
                
            else:
                st.error("No date information available in the data")
                filtered_data = matched_data
                filtered_performance = performance
                filtered_confidence_analysis = confidence_analysis
            
            # Overall Performance Summary
            st.markdown("### Overall Performance Summary")
            
            # Create performance metrics table
            performance_data = []
            models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9', 
                     'model_rf', 'model_nn', 'model_svm',
                     'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic',
                     'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                     'model_advanced_xgboost_basic', 'model_advanced_xgboost', 'model1_extended_lgbm', 
                     'model2_extended_catboost', 'model3_extended_xgboost', 'ensemble']
            model_names = ['Model 1 (LightGBM)', 'Model 2 (CatBoost)', 'Model 3 (XGBoost)', 
                          'Model 4', 'Model 5', 'Model 6', 'Model 7', 'Model 8', 'Model 9',
                          'Random Forest (88%)', 'Neural Network (94%)', 'SVM (96%)',
                          'Advanced CatBoost Basic', 'Advanced CatBoost', 'Advanced LightGBM Basic',
                          'Advanced LightGBM', 'Advanced Neural Network', 'Advanced Random Forest', 'Advanced SVM',
                          'Advanced XGBoost Basic', 'Advanced XGBoost', 'Model1 Extended LightGBM', 
                          'Model2 Extended CatBoost', 'Model3 Extended XGBoost', 'Ensemble Model']
            
            for i, model in enumerate(models):
                if model in filtered_performance:
                    perf = filtered_performance[model]
                    performance_data.append({
                        'Model': model_names[i],
                        'Accuracy': f"{perf['accuracy']:.1%}",
                        'Correct Predictions': f"{perf['correct_predictions']}/{perf['total_predictions']}",
                        'Brier Score': f"{perf['brier_score']:.4f}",
                        'Log Loss': f"{perf['log_loss']:.4f}",
                        'Avg Confidence': f"{perf['avg_confidence']:.1%}"
                    })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(
                performance_df,
                column_config={
                    "Model": st.column_config.TextColumn("Model"),
                    "Accuracy": st.column_config.TextColumn("Accuracy"),
                    "Correct Predictions": st.column_config.TextColumn("Correct/Total"),
                    "Brier Score": st.column_config.TextColumn("Brier Score", help="Lower is better (perfect = 0)"),
                    "Log Loss": st.column_config.TextColumn("Log Loss", help="Lower is better"),
                    "Avg Confidence": st.column_config.TextColumn("Avg Confidence")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Model comparison chart
            st.markdown("### Model Accuracy Comparison")
            
            # Create accuracy comparison chart
            if filtered_performance:
                # Only include models that have predictions
                chart_models = []
                chart_names = []
                chart_accuracies = []
                
                for i, model in enumerate(models):
                    if model in filtered_performance:
                        chart_models.append(model)
                        chart_names.append(model_names[i])
                        chart_accuracies.append(filtered_performance[model]['accuracy'] * 100)
                
                if chart_accuracies:
                    accuracy_data = {
                        'Model': chart_names,
                        'Accuracy': chart_accuracies
                    }
                    
                    fig_accuracy = px.bar(
                        accuracy_data, 
                        x='Model', 
                        y='Accuracy',
                        title='Model Accuracy Comparison (Filtered Data)',
                        color='Accuracy',
                        color_continuous_scale='viridis'
                    )
                    fig_accuracy.update_layout(
                        yaxis_title="Accuracy (%)",
                        xaxis_title="Model",
                        showlegend=False
                    )
                    fig_accuracy.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                else:
                    st.info("No accuracy data available for the selected date range")
            else:
                st.info("No performance data available for the selected date range")
            
            # Confidence Analysis
            st.markdown("### Performance by Prediction Confidence")
            
            # Only create tabs for models that have confidence analysis data
            available_models = []
            available_model_names = []
            
            for i, model in enumerate(models):
                if model in filtered_confidence_analysis:
                    available_models.append(model)
                    available_model_names.append(model_names[i])
            
            if available_models:
                confidence_tabs = st.tabs(available_model_names)
                
                for i, model in enumerate(available_models):
                    with confidence_tabs[i]:
                        conf_data = []
                        for range_name, stats in filtered_confidence_analysis[model].items():
                            if stats['total'] > 0:
                                conf_data.append({
                                    'Confidence Range': range_name,
                                    'Accuracy': f"{stats['accuracy']:.1%}",
                                    'Correct': stats['correct'],
                                    'Total': stats['total'],
                                    'Sample Size': f"{stats['correct']}/{stats['total']}"
                                })
                        
                        if conf_data:
                            conf_df = pd.DataFrame(conf_data)
                            st.dataframe(
                                conf_df,
                                column_config={
                                    "Confidence Range": st.column_config.TextColumn("Confidence Range"),
                                    "Accuracy": st.column_config.TextColumn("Accuracy"),
                                    "Sample Size": st.column_config.TextColumn("Correct/Total")
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Create confidence vs accuracy chart
                            chart_data = {
                                'Confidence Range': [item['Confidence Range'] for item in conf_data],
                                'Accuracy': [float(item['Accuracy'].strip('%')) for item in conf_data],
                                'Sample Size': [item['Total'] for item in conf_data]
                            }
                            
                            fig_conf = px.bar(
                                chart_data,
                                x='Confidence Range',
                                y='Accuracy',
                                title=f'{available_model_names[i]} - Accuracy by Confidence Level (Filtered Data)',
                                color='Sample Size',
                                color_continuous_scale='blues'
                            )
                            fig_conf.update_layout(
                                yaxis_title="Accuracy (%)",
                                xaxis_title="Confidence Range"
                            )
                            fig_conf.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                            st.plotly_chart(fig_conf, use_container_width=True)
                        else:
                            st.info("No confidence data available for this model in the selected date range")
            else:
                st.info("No models have confidence analysis data for the selected date range")
            
            # Detailed Predictions Analysis
            st.markdown("### Recent Predictions Analysis")
            
            # Add option to limit the number of displayed games
            options = [10, 20, 50, len(filtered_data)]
            default_value = min(20, len(filtered_data))
            default_index = options.index(default_value) if default_value in options else 1
            
            num_games_to_show = st.selectbox(
                "Number of games to display:",
                options=options,
                index=default_index,
                key="num_games_display"
            )
            
            # Create detailed predictions table
            detailed_data = []
            display_data = filtered_data[-num_games_to_show:] if len(filtered_data) > num_games_to_show else filtered_data
            
            for record in display_data:
                # 예측된 승자의 승률 계산
                home_win_prob = record['win_probability']
                away_win_prob = 1 - home_win_prob
                
                # 예측된 승자가 홈팀이면 홈팀 승률, 원정팀이면 원정팀 승률 표시
                if record['predicted_winner'] == record['home_team']:
                    predicted_winner_prob = home_win_prob
                else:
                    predicted_winner_prob = away_win_prob
                
                # 실제 홈팀 승리 여부
                actual_home_win = record['actual_home_win']
                
                # 각 모델별 correct 여부 계산
                def get_model_result(prob, model_name=None):
                    # 새로운 모델들의 경우 데이터가 없으면 N/A 표시
                    advanced_models = ['model_rf', 'model_nn', 'model_svm',
                                     'model_advanced_catboost_basic', 'model_advanced_catboost', 'model_advanced_lgbm_basic',
                                     'model_advanced_lgbm', 'model_advanced_nn', 'model_advanced_rf', 'model_advanced_svm',
                                     'model_advanced_xgboost_basic', 'model_advanced_xgboost']
                    if model_name in advanced_models and prob == 0:
                        return "N/A"
                    predicted_home_win = 1 if prob > 0.5 else 0
                    is_correct = predicted_home_win == actual_home_win
                    icon = '✅' if is_correct else '❌'
                    return f"{prob:.3f} {icon}"
                
                detailed_data.append({
                    'Date': record['date'],
                    'Match': f"{record['away_team']} @ {record['home_team']}",
                    'Predicted Winner': record['predicted_winner'],
                    'Actual Winner': record['actual_winner'],
                    'Correct': '✅' if record['predicted_winner'] == record['actual_winner'] else '❌',
                    'Model 1 Prob': get_model_result(record['model1_probability']),
                    'Model 2 Prob': get_model_result(record['model2_probability']),
                    'Model 3 Prob': get_model_result(record['model3_probability']),
                    'Model 4 Prob': get_model_result(record['model4_probability']),
                    'Model 5 Prob': get_model_result(record['model5_probability']),
                    'Model 6 Prob': get_model_result(record['model6_probability']),
                    'Model 7 Prob': get_model_result(record['model7_probability']),
                    'Model 8 Prob': get_model_result(record['model8_probability']),
                    'Model 9 Prob': get_model_result(record['model9_probability']),
                    'RF Prob': get_model_result(record.get('model_rf_probability', 0), 'model_rf'),
                    'NN Prob': get_model_result(record.get('model_nn_probability', 0), 'model_nn'),
                    'SVM Prob': get_model_result(record.get('model_svm_probability', 0), 'model_svm'),
                    'Advanced CB Basic': get_model_result(record.get('model_advanced_catboost_basic_probability', 0), 'model_advanced_catboost_basic'),
                    'Advanced CB': get_model_result(record.get('model_advanced_catboost_probability', 0), 'model_advanced_catboost'),
                    'Advanced LGB Basic': get_model_result(record.get('model_advanced_lgbm_basic_probability', 0), 'model_advanced_lgbm_basic'),
                    'Advanced LGB': get_model_result(record.get('model_advanced_lgbm_probability', 0), 'model_advanced_lgbm'),
                    'Advanced NN': get_model_result(record.get('model_advanced_nn_probability', 0), 'model_advanced_nn'),
                    'Advanced RF': get_model_result(record.get('model_advanced_rf_probability', 0), 'model_advanced_rf'),
                    'Advanced SVM': get_model_result(record.get('model_advanced_svm_probability', 0), 'model_advanced_svm'),
                    'Advanced XGB Basic': get_model_result(record.get('model_advanced_xgboost_basic_probability', 0), 'model_advanced_xgboost_basic'),
                    'Advanced XGB': get_model_result(record.get('model_advanced_xgboost_probability', 0), 'model_advanced_xgboost'),
                    'Extended LGB': get_model_result(record.get('model1_extended_lgbm_probability', 0), 'model1_extended_lgbm'),
                    'Extended CB': get_model_result(record.get('model2_extended_catboost_probability', 0), 'model2_extended_catboost'),
                    'Extended XGB': get_model_result(record.get('model3_extended_xgboost_probability', 0), 'model3_extended_xgboost'),
                    'Ensemble Prob': get_model_result(record['ensemble_probability']),
                    'Win Probability': f"{predicted_winner_prob:.1%}"
                })
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                st.dataframe(
                    detailed_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Match": st.column_config.TextColumn("Match"),
                        "Predicted Winner": st.column_config.TextColumn("Predicted Winner"),
                        "Actual Winner": st.column_config.TextColumn("Actual Winner"),
                        "Correct": st.column_config.TextColumn("Correct"),
                        "Model 1 Prob": st.column_config.TextColumn("Model 1", help="LightGBM home win probability with ✅/❌ for correctness"),
                        "Model 2 Prob": st.column_config.TextColumn("Model 2", help="CatBoost home win probability with ✅/❌ for correctness"),
                        "Model 3 Prob": st.column_config.TextColumn("Model 3", help="XGBoost home win probability with ✅/❌ for correctness"),
                        "Model 4 Prob": st.column_config.TextColumn("Model 4", help="Model 4 home win probability with ✅/❌ for correctness"),
                        "Model 5 Prob": st.column_config.TextColumn("Model 5", help="Model 5 home win probability with ✅/❌ for correctness"),
                        "Model 6 Prob": st.column_config.TextColumn("Model 6", help="Model 6 home win probability with ✅/❌ for correctness"),
                        "Model 7 Prob": st.column_config.TextColumn("Model 7", help="Model 7 home win probability with ✅/❌ for correctness"),
                        "Model 8 Prob": st.column_config.TextColumn("Model 8", help="Model 8 home win probability with ✅/❌ for correctness"),
                        "Model 9 Prob": st.column_config.TextColumn("Model 9", help="Model 9 home win probability with ✅/❌ for correctness"),
                        "RF Prob": st.column_config.TextColumn("RF", help="Random Forest home win probability with ✅/❌ for correctness"),
                        "NN Prob": st.column_config.TextColumn("NN", help="Neural Network home win probability with ✅/❌ for correctness"),
                        "SVM Prob": st.column_config.TextColumn("SVM", help="SVM home win probability with ✅/❌ for correctness"),
                        "Advanced CB Basic": st.column_config.TextColumn("Advanced CB Basic", help="Advanced CatBoost Basic home win probability with ✅/❌ for correctness"),
                        "Advanced CB": st.column_config.TextColumn("Advanced CB", help="Advanced CatBoost home win probability with ✅/❌ for correctness"),
                        "Advanced LGB Basic": st.column_config.TextColumn("Advanced LGB Basic", help="Advanced LightGBM Basic home win probability with ✅/❌ for correctness"),
                        "Advanced LGB": st.column_config.TextColumn("Advanced LGB", help="Advanced LightGBM home win probability with ✅/❌ for correctness"),
                        "Advanced NN": st.column_config.TextColumn("Advanced NN", help="Advanced Neural Network home win probability with ✅/❌ for correctness"),
                        "Advanced RF": st.column_config.TextColumn("Advanced RF", help="Advanced Random Forest home win probability with ✅/❌ for correctness"),
                        "Advanced SVM": st.column_config.TextColumn("Advanced SVM", help="Advanced SVM home win probability with ✅/❌ for correctness"),
                        "Advanced XGB Basic": st.column_config.TextColumn("Advanced XGB Basic", help="Advanced XGBoost Basic home win probability with ✅/❌ for correctness"),
                        "Advanced XGB": st.column_config.TextColumn("Advanced XGB", help="Advanced XGBoost home win probability with ✅/❌ for correctness"),
                        "Extended LGB": st.column_config.TextColumn("Extended LGB", help="Extended LightGBM home win probability with ✅/❌ for correctness"),
                        "Extended CB": st.column_config.TextColumn("Extended CB", help="Extended CatBoost home win probability with ✅/❌ for correctness"),
                        "Extended XGB": st.column_config.TextColumn("Extended XGB", help="Extended XGBoost home win probability with ✅/❌ for correctness"),
                        "Ensemble Prob": st.column_config.TextColumn("Ensemble", help="Ensemble home win probability with ✅/❌ for correctness"),
                        "Win Probability": st.column_config.TextColumn("Predicted Winner Prob", help="Predicted winner's win probability")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            # Summary Statistics
            st.markdown("### Summary Statistics")
            
            total_predictions = len(filtered_data)
            correct_ensemble = sum(1 for record in filtered_data if record['predicted_winner'] == record['actual_winner'])
            ensemble_accuracy = correct_ensemble / total_predictions if total_predictions > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyzed Games", total_predictions)
            
            with col2:
                st.metric("Ensemble Accuracy", f"{ensemble_accuracy:.1%}")
            
            with col3:
                if filtered_performance:
                    best_model = max(filtered_performance.keys(), key=lambda x: filtered_performance[x]['accuracy'])
                    best_accuracy = filtered_performance[best_model]['accuracy']
                    model_display_name = model_names[models.index(best_model)] if best_model in models else best_model.title()
                    st.metric("Best Individual Model", f"{model_display_name} ({best_accuracy:.1%})")
                else:
                    st.metric("Best Individual Model", "N/A")
            
            with col4:
                if filtered_performance:
                    avg_brier = np.mean([filtered_performance[model]['brier_score'] for model in models if model in filtered_performance])
                    st.metric("Avg Brier Score", f"{avg_brier:.4f}")
                else:
                    st.metric("Avg Brier Score", "N/A")
            
            # File information
            st.markdown("### Data Sources")
            prediction_files = tracker.get_latest_prediction_files_by_date()
            historical_file = tracker.get_latest_historical_records()
            
            with st.expander("View data source files"):
                st.markdown("**Prediction Files Used:**")
                for date, file_path in prediction_files.items():
                    filename = Path(file_path).name
                    st.markdown(f"- {date}: `{filename}`")
                
                st.markdown("**Historical Records File:**")
                if historical_file:
                    hist_filename = Path(historical_file).name
                    st.markdown(f"- `{hist_filename}`")
                else:
                    st.markdown("- No historical file found")
                    
                st.markdown(f"**Total Available Games:** {len(matched_data)}")
                st.markdown(f"**Filtered Games:** {len(filtered_data)}")
            
            # Underdog Picks Analysis
            st.markdown("---")
            st.markdown("### Underdog Picks Analysis 🎯")
            st.markdown("""
            This section analyzes the performance of underdog picks - games where our ensemble model 
            predicted the team with worse odds to win. These picks often have higher potential returns 
            but lower win probabilities.
            """)
            
            # Get underdog analysis
            with st.spinner("Analyzing underdog picks..."):
                start_date_str = start_date.strftime('%Y-%m-%d') if 'start_date' in locals() else None
                end_date_str = end_date.strftime('%Y-%m-%d') if 'end_date' in locals() else None
                
                underdog_analysis = tracker.analyze_underdog_performance(
                    filtered_data, 
                    start_date_str, 
                    end_date_str
                )
            
            if underdog_analysis and underdog_analysis.get('total_picks', 0) > 0:
                # Overall underdog performance
                st.markdown("#### Overall Underdog Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Underdog Picks", underdog_analysis['total_picks'])
                
                with col2:
                    st.metric("Accuracy", f"{underdog_analysis['accuracy']:.1%}")
                
                with col3:
                    st.metric("ROI", f"{underdog_analysis['roi']:.1f}%")
                
                with col4:
                    st.metric("Net Profit ($10/bet)", f"${underdog_analysis['net_profit']:.2f}")
                
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("Avg Odds", f"+{underdog_analysis['avg_odds']:.0f}")
                
                with col6:
                    st.metric("Avg Ensemble Prob", f"{underdog_analysis['avg_ensemble_prob']:.1%}")
                
                with col7:
                    st.metric("Total Invested", f"${underdog_analysis['total_invested']:.2f}")
                
                # Performance by odds range
                if underdog_analysis.get('performance_by_odds_range'):
                    st.markdown("#### Performance by Odds Range")
                    
                    odds_performance_data = []
                    for range_name, stats in underdog_analysis['performance_by_odds_range'].items():
                        odds_performance_data.append({
                            'Odds Range': range_name,
                            'Total Picks': stats['total_picks'],
                            'Correct Picks': stats['correct_picks'],
                            'Accuracy': f"{stats['accuracy']:.1%}",
                            'ROI': f"{stats['roi']:.1f}%",
                            'Net Profit': f"${stats['net_profit']:.2f}",
                            'Total Invested': f"${stats['total_invested']:.2f}"
                        })
                    
                    if odds_performance_data:
                        odds_df = pd.DataFrame(odds_performance_data)
                        st.dataframe(
                            odds_df,
                            column_config={
                                "Odds Range": st.column_config.TextColumn("Odds Range"),
                                "Total Picks": st.column_config.NumberColumn("Total Picks"),
                                "Correct Picks": st.column_config.NumberColumn("Correct Picks"),
                                "Accuracy": st.column_config.TextColumn("Accuracy"),
                                "ROI": st.column_config.TextColumn("ROI"),
                                "Net Profit": st.column_config.TextColumn("Net Profit"),
                                "Total Invested": st.column_config.TextColumn("Total Invested")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Create ROI chart by odds range
                        chart_data = {
                            'Odds Range': [item['Odds Range'] for item in odds_performance_data],
                            'ROI': [float(item['ROI'].strip('%')) for item in odds_performance_data],
                            'Accuracy': [float(item['Accuracy'].strip('%')) for item in odds_performance_data]
                        }
                        
                        fig_underdog_roi = px.bar(
                            chart_data,
                            x='Odds Range',
                            y='ROI',
                            title='Underdog Picks ROI by Odds Range',
                            color='Accuracy',
                            color_continuous_scale='viridis'
                        )
                        fig_underdog_roi.update_layout(yaxis_title="ROI (%)")
                        fig_underdog_roi.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                        st.plotly_chart(fig_underdog_roi, use_container_width=True)
                
                # Detailed underdog picks
                st.markdown("#### Detailed Underdog Picks")
                
                # Add option to limit the number of displayed underdog picks
                num_underdog_options = [10, 20, 50, len(underdog_analysis['details'])]
                default_underdog_value = min(20, len(underdog_analysis['details']))
                default_underdog_index = num_underdog_options.index(default_underdog_value) if default_underdog_value in num_underdog_options else 1
                
                num_underdog_to_show = st.selectbox(
                    "Number of underdog picks to display:",
                    options=num_underdog_options,
                    index=default_underdog_index,
                    key="num_underdog_display"
                )
                
                # Display detailed underdog picks
                display_underdog_data = underdog_analysis['details'][-num_underdog_to_show:] if len(underdog_analysis['details']) > num_underdog_to_show else underdog_analysis['details']
                
                if display_underdog_data:
                    underdog_details = []
                    for bet in display_underdog_data:
                        underdog_details.append({
                            'Date': bet['date'],
                            'Game': bet['game'],
                            'Predicted Winner': bet['predicted_winner'],
                            'Actual Winner': bet['actual_winner'],
                            'Result': '✅ Won' if bet['is_correct'] else '❌ Lost',
                            'Odds': f"+{bet['odds']:.0f}" if bet['odds'] > 0 else f"{bet['odds']:.0f}",
                            'Ensemble Prob': f"{bet['ensemble_probability']:.1%}",
                            'Bet Amount': f"${bet['bet_amount']:.2f}",
                            'Profit/Loss': f"${bet['profit']:.2f}"
                        })
                    
                    underdog_details_df = pd.DataFrame(underdog_details)
                    st.dataframe(
                        underdog_details_df,
                        column_config={
                            "Date": st.column_config.TextColumn("Date"),
                            "Game": st.column_config.TextColumn("Game"),
                            "Predicted Winner": st.column_config.TextColumn("Predicted Winner"),
                            "Actual Winner": st.column_config.TextColumn("Actual Winner"),
                            "Result": st.column_config.TextColumn("Result"),
                            "Odds": st.column_config.TextColumn("Odds"),
                            "Ensemble Prob": st.column_config.TextColumn("Ensemble Prob"),
                            "Bet Amount": st.column_config.TextColumn("Bet Amount"),
                            "Profit/Loss": st.column_config.TextColumn("Profit/Loss")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Underdog picks insights
                with st.expander("Underdog Picks Insights"):
                    st.markdown("""
                    **What are Underdog Picks?**
                    
                    Underdog picks are games where our ensemble model predicts the team with worse odds (higher potential payout) to win.
                    These picks are identified by comparing the predicted winner with the betting odds:
                    
                    - If we predict the home team to win, but the home team has higher odds (or positive odds while away team has negative odds)
                    - If we predict the away team to win, but the away team has higher odds (or positive odds while home team has negative odds)
                    
                    **ROI Calculation:**
                    - Fixed bet amount of $10 per pick
                    - Profit calculation based on American odds format
                    - For positive odds (+150): Profit = $10 × (150/100) = $15
                    - For negative odds (-120): Profit = $10 × (100/120) = $8.33
                    
                    **Key Metrics:**
                    - **Accuracy**: Percentage of underdog picks that were correct
                    - **ROI**: Return on investment percentage
                    - **Average Odds**: Average odds of underdog picks
                    - **Average Ensemble Probability**: Average confidence of our model in underdog picks
                    """)
            else:
                st.info("No underdog picks found in the selected date range or no odds data available.")
                
        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")
            st.exception(e)

elif page_selection == "Betting Performance Tracking":
    st.markdown("## Betting Performance Tracking 📊")
    st.markdown("Track the actual performance of our betting recommendations.")
    
    try:
        # Initialize betting performance tracker
        betting_tracker = BettingPerformanceTracker()
        
        # Get betting analysis files
        betting_files = betting_tracker.get_betting_analysis_files()
        
        if not betting_files:
            st.error("No betting analysis files found in data/analysis directory")
            st.stop()
        
        # Date filtering section
        st.markdown("### Date Filtering")
        
        # Get available date range from betting files
        available_dates = sorted(betting_files.keys())
        
        if available_dates:
            min_date = datetime.strptime(available_dates[0], '%Y-%m-%d').date()
            max_date = datetime.strptime(available_dates[-1], '%Y-%m-%d').date()
            
            # Create date range selector
            col1, col2 = st.columns([1, 1])
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="betting_start_date_input"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="betting_end_date_input"
                )
            
            # Validate date range
            if start_date > end_date:
                st.error("⚠️ Start date cannot be later than end date. Please adjust your date selection.")
                st.stop()
            
            # Convert dates to strings for filtering
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Filter files based on date range
            filtered_files = {
                date: path for date, path in betting_files.items()
                if start_date_str <= date <= end_date_str
            }
            
            if not filtered_files:
                st.warning(f"⚠️ No betting analysis files found in the selected date range ({start_date_str} to {end_date_str}). Please adjust your date selection.")
                st.stop()
            
            # Display filtering info
            st.info(f"Showing {len(filtered_files)} betting analysis files from {start_date_str} to {end_date_str} (Total available: {len(betting_files)} files)")
            
        else:
            st.error("No betting analysis files available")
            st.stop()
            start_date_str = None
            end_date_str = None
        
        # Get performance data with date filtering
        with st.spinner("Loading betting performance data..."):
            daily_performance, total_performance = betting_tracker.get_performance_summary(start_date_str, end_date_str)
        
        if not daily_performance and not total_performance:
            st.error("No historical records file found or no completed bets available")
            st.stop()
        
        # Display overview metrics
        st.markdown("### Performance Overview")
        
        # Create tabs for different bet types
        bet_type_tabs = st.tabs(['Singles', '3-Team Parlays', '4-Team Parlays', '5-Team Parlays', 'Daily Performance'])
        
        # Singles Performance
        with bet_type_tabs[0]:
            st.markdown("#### Single Bets Performance")
            
            if 'single_bets' in total_performance and total_performance['single_bets']['amount']['performance']['total_bets'] > 0:
                perf = total_performance['single_bets']['amount']['performance']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Bets", perf['total_bets'])
                with col2:
                    st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                with col3:
                    st.metric("Net Profit", f"${perf['net_profit']:.2f}")
                with col4:
                    st.metric("ROI", f"{perf['roi']:.2f}%")
                
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("Total Invested", f"${perf['total_invested']:.2f}")
                with col6:
                    st.metric("Total Returned", f"${perf['total_returned']:.2f}")
                with col7:
                    st.metric("Won/Lost", f"{perf['won_bets']}/{perf['lost_bets']}")
                
                # Show bet details
                with st.expander("View Individual Bet Details", expanded=False):
                    bet_details = []
                    for detail in total_performance['single_bets']['amount']['details']:
                        bet = detail['bet']
                        result = detail['result']
                        bet_details.append({
                            'Date': bet['date'],
                            'Match': bet['match'],
                            'Team': bet['team'],
                            'Odds': bet['odds'],
                            'Amount': f"${bet['amount']:.2f}",
                            'Result': '✅ Won' if detail['bet_won'] else '❌ Lost',
                            'Profit/Loss': f"${(bet['amount'] * (100/abs(bet['odds'])) if bet['odds'] < 0 else bet['amount'] * (bet['odds']/100)) if detail['bet_won'] else -bet['amount']:.2f}"
                        })
                    
                    if bet_details:
                        # 🚀 Pagination 추가
                        page_size = 20  # 싱글 베팅은 조금 더 많이 표시
                        total_pages = (len(bet_details) + page_size - 1) // page_size
                        
                        if total_pages > 1:
                            col_page, col_info = st.columns([1, 3])
                            with col_page:
                                page = st.selectbox(
                                    "Page", 
                                    range(1, total_pages + 1),
                                    key="singles_page"
                                )
                            with col_info:
                                st.info(f"Showing {len(bet_details)} single bets ({page_size} per page)")
                            
                            start_idx = (page - 1) * page_size
                            end_idx = start_idx + page_size
                            current_page_details = bet_details[start_idx:end_idx]
                        else:
                            current_page_details = bet_details
                            st.info(f"Showing all {len(bet_details)} single bets")
                        
                        details_df = pd.DataFrame(current_page_details)
                        st.dataframe(details_df, use_container_width=True, hide_index=True)
            else:
                st.info("No completed single bets available")
        
        # 3-Team Parlays Performance
        with bet_type_tabs[1]:
            st.markdown("#### 3-Team Parlay Performance")
            
            if '3_team_parlay' in [parlay.get('type') for parlay in betting_tracker.load_betting_analysis(list(betting_files.values())[0]).get('parlays', [])]:
                amount_tabs = st.tabs(['Regular Amount', 'Adjusted Amount'])
                
                for i, amount_type in enumerate(['amount', 'adjusted_amount']):
                    with amount_tabs[i]:
                        # Boosted vs Regular Odds 탭 추가
                        odds_tabs = st.tabs(['Boosted Odds', 'Regular Odds'])
                        
                        for j, odds_type in enumerate(['boosted', 'regular']):
                            with odds_tabs[j]:
                                if (total_performance['parlay_3_team'][amount_type][odds_type]['performance']['total_bets'] > 0):
                                    perf = total_performance['parlay_3_team'][amount_type][odds_type]['performance']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Total Parlays", perf['total_bets'])
                                    with col2:
                                        st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                                    with col3:
                                        st.metric("Net Profit", f"${perf['net_profit']:.2f}")
                                    with col4:
                                        st.metric("ROI", f"{perf['roi']:.2f}%")
                                    
                                    col5, col6, col7 = st.columns(3)
                                    
                                    with col5:
                                        st.metric("Total Invested", f"${perf['total_invested']:.2f}")
                                    with col6:
                                        st.metric("Total Returned", f"${perf['total_returned']:.2f}")
                                    with col7:
                                        st.metric("Won/Lost", f"{perf['won_bets']}/{perf['lost_bets']}")
                                    
                                    # Show parlay details
                                    with st.expander("View 3-Team Parlay Details", expanded=False):
                                        parlay_details = betting_tracker.get_parlay_details(
                                            total_performance['parlay_3_team'][amount_type][odds_type]['details']
                                        )
                                        
                                        # 🚀 Pagination 추가
                                        if parlay_details:
                                            page_size = 10
                                            total_pages = (len(parlay_details) + page_size - 1) // page_size
                                            
                                            if total_pages > 1:
                                                col_page, col_info = st.columns([1, 3])
                                                with col_page:
                                                    page = st.selectbox(
                                                        "Page", 
                                                        range(1, total_pages + 1),
                                                        key=f"3team_page_{amount_type}_{odds_type}"
                                                    )
                                                with col_info:
                                                    st.info(f"Showing {len(parlay_details)} parlays ({page_size} per page)")
                                                
                                                start_idx = (page - 1) * page_size
                                                end_idx = start_idx + page_size
                                                current_page_details = parlay_details[start_idx:end_idx]
                                            else:
                                                current_page_details = parlay_details
                                                st.info(f"Showing all {len(parlay_details)} parlays")
                                        else:
                                            current_page_details = []
                                        
                                        for k, parlay in enumerate(current_page_details):
                                            # 전체 인덱스 계산 (페이지 고려)
                                            if 'page' in locals() and len(parlay_details) > page_size:
                                                global_k = ((page - 1) * page_size) + k + 1
                                            else:
                                                global_k = k + 1
                                                
                                            st.markdown(f"**Parlay {global_k}** - {'✅ Won' if parlay['parlay_won'] else '❌ Lost'}")
                                            
                                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                                            with col_info1:
                                                investment = parlay['adjusted_amount'] if amount_type == 'adjusted_amount' else parlay['amount']
                                                st.write(f"Investment: ${investment:.2f}")
                                            with col_info2:
                                                odds_display = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                                st.write(f"{odds_type.title()} Odds: {odds_display:.2f}")
                                            with col_info3:
                                                if parlay['parlay_won']:
                                                    odds_for_calc = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                                    profit = investment * (odds_for_calc - 1)
                                                    st.write(f"Profit: ${profit:.2f}")
                                                else:
                                                    st.write(f"Loss: ${investment:.2f}")
                                            with col_info4:
                                                win_prob = parlay['probability'] * 100
                                                st.write(f"Win Prob: {win_prob:.1f}%")
                                            
                                            st.markdown("**Individual Picks:**")
                                            for pick in parlay['picks_details']:
                                                pick_status = '✅' if pick['pick_won'] else '❌'
                                                date_str = f" [{pick['date']}]" if pick.get('date') else ""
                                                st.write(f"{pick_status} {pick['team']} ({pick['match']}){date_str} - Winner: {pick['actual_winner']}")
                                            
                                            st.markdown("---")
                                else:
                                    st.info(f"No completed 3-team parlays available for {amount_type} with {odds_type} odds")
            else:
                st.info("No 3-team parlays found in betting analysis")
        
        # 4-Team Parlays Performance
        with bet_type_tabs[2]:
            st.markdown("#### 4-Team Parlay Performance")
            
            amount_tabs = st.tabs(['Regular Amount', 'Adjusted Amount'])
            
            for i, amount_type in enumerate(['amount', 'adjusted_amount']):
                with amount_tabs[i]:
                    # Boosted vs Regular Odds 탭 추가
                    odds_tabs = st.tabs(['Boosted Odds', 'Regular Odds'])
                    
                    for j, odds_type in enumerate(['boosted', 'regular']):
                        with odds_tabs[j]:
                            if (total_performance['parlay_4_team'][amount_type][odds_type]['performance']['total_bets'] > 0):
                                perf = total_performance['parlay_4_team'][amount_type][odds_type]['performance']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Parlays", perf['total_bets'])
                                with col2:
                                    st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                                with col3:
                                    st.metric("Net Profit", f"${perf['net_profit']:.2f}")
                                with col4:
                                    st.metric("ROI", f"{perf['roi']:.2f}%")
                                
                                col5, col6, col7 = st.columns(3)
                                
                                with col5:
                                    st.metric("Total Invested", f"${perf['total_invested']:.2f}")
                                with col6:
                                    st.metric("Total Returned", f"${perf['total_returned']:.2f}")
                                with col7:
                                    st.metric("Won/Lost", f"{perf['won_bets']}/{perf['lost_bets']}")
                                
                                # Show parlay details
                                with st.expander("View 4-Team Parlay Details", expanded=False):
                                    parlay_details = betting_tracker.get_parlay_details(
                                        total_performance['parlay_4_team'][amount_type][odds_type]['details']
                                    )
                                    
                                    # 🚀 Pagination 추가
                                    if parlay_details:
                                        page_size = 10
                                        total_pages = (len(parlay_details) + page_size - 1) // page_size
                                        
                                        if total_pages > 1:
                                            col_page, col_info = st.columns([1, 3])
                                            with col_page:
                                                page = st.selectbox(
                                                    "Page", 
                                                    range(1, total_pages + 1),
                                                    key=f"4team_page_{amount_type}_{odds_type}"
                                                )
                                            with col_info:
                                                st.info(f"Showing {len(parlay_details)} parlays ({page_size} per page)")
                                            
                                            start_idx = (page - 1) * page_size
                                            end_idx = start_idx + page_size
                                            current_page_details = parlay_details[start_idx:end_idx]
                                        else:
                                            current_page_details = parlay_details
                                            st.info(f"Showing all {len(parlay_details)} parlays")
                                    else:
                                        current_page_details = []
                                    
                                    for k, parlay in enumerate(current_page_details):
                                        # 전체 인덱스 계산 (페이지 고려)
                                        if 'page' in locals() and len(parlay_details) > page_size:
                                            global_k = ((page - 1) * page_size) + k + 1
                                        else:
                                            global_k = k + 1
                                            
                                        st.markdown(f"**Parlay {global_k}** - {'✅ Won' if parlay['parlay_won'] else '❌ Lost'}")
                                        
                                        col_info1, col_info2, col_info3 = st.columns(3)
                                        with col_info1:
                                            investment = parlay['adjusted_amount'] if amount_type == 'adjusted_amount' else parlay['amount']
                                            st.write(f"Investment: ${investment:.2f}")
                                        with col_info2:
                                            odds_display = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                            st.write(f"{odds_type.title()} Odds: {odds_display:.2f}")
                                        with col_info3:
                                            if parlay['parlay_won']:
                                                odds_for_calc = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                                profit = investment * (odds_for_calc - 1)
                                                st.write(f"Profit: ${profit:.2f}")
                                            else:
                                                st.write(f"Loss: ${investment:.2f}")
                                        
                                        st.markdown("**Individual Picks:**")
                                        for pick in parlay['picks_details']:
                                            pick_status = '✅' if pick['pick_won'] else '❌'
                                            date_str = f" [{pick['date']}]" if pick.get('date') else ""
                                            st.write(f"{pick_status} {pick['team']} ({pick['match']}){date_str} - Winner: {pick['actual_winner']}")
                                        
                                        st.markdown("---")
                            else:
                                st.info(f"No completed 4-team parlays available for {amount_type} with {odds_type} odds")
        
        # 5-Team Parlays Performance
        with bet_type_tabs[3]:
            st.markdown("#### 5-Team Parlay Performance")
            
            amount_tabs = st.tabs(['Regular Amount', 'Adjusted Amount'])
            
            for i, amount_type in enumerate(['amount', 'adjusted_amount']):
                with amount_tabs[i]:
                    # Boosted vs Regular Odds 탭 추가
                    odds_tabs = st.tabs(['Boosted Odds', 'Regular Odds'])
                    
                    for j, odds_type in enumerate(['boosted', 'regular']):
                        with odds_tabs[j]:
                            if (total_performance['parlay_5_team'][amount_type][odds_type]['performance']['total_bets'] > 0):
                                perf = total_performance['parlay_5_team'][amount_type][odds_type]['performance']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Parlays", perf['total_bets'])
                                with col2:
                                    st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                                with col3:
                                    st.metric("Net Profit", f"${perf['net_profit']:.2f}")
                                with col4:
                                    st.metric("ROI", f"{perf['roi']:.2f}%")
                                
                                col5, col6, col7 = st.columns(3)
                                
                                with col5:
                                    st.metric("Total Invested", f"${perf['total_invested']:.2f}")
                                with col6:
                                    st.metric("Total Returned", f"${perf['total_returned']:.2f}")
                                with col7:
                                    st.metric("Won/Lost", f"{perf['won_bets']}/{perf['lost_bets']}")
                                
                                # Show parlay details
                                with st.expander("View 5-Team Parlay Details", expanded=False):
                                    parlay_details = betting_tracker.get_parlay_details(
                                        total_performance['parlay_5_team'][amount_type][odds_type]['details']
                                    )
                                    
                                    # 🚀 Pagination 추가
                                    if parlay_details:
                                        page_size = 10
                                        total_pages = (len(parlay_details) + page_size - 1) // page_size
                                        
                                        if total_pages > 1:
                                            col_page, col_info = st.columns([1, 3])
                                            with col_page:
                                                page = st.selectbox(
                                                    "Page", 
                                                    range(1, total_pages + 1),
                                                    key=f"5team_page_{amount_type}_{odds_type}"
                                                )
                                            with col_info:
                                                st.info(f"Showing {len(parlay_details)} parlays ({page_size} per page)")
                                            
                                            start_idx = (page - 1) * page_size
                                            end_idx = start_idx + page_size
                                            current_page_details = parlay_details[start_idx:end_idx]
                                        else:
                                            current_page_details = parlay_details
                                            st.info(f"Showing all {len(parlay_details)} parlays")
                                    else:
                                        current_page_details = []
                                    
                                    for k, parlay in enumerate(current_page_details):
                                        # 전체 인덱스 계산 (페이지 고려)
                                        if 'page' in locals() and len(parlay_details) > page_size:
                                            global_k = ((page - 1) * page_size) + k + 1
                                        else:
                                            global_k = k + 1
                                            
                                        st.markdown(f"**Parlay {global_k}** - {'✅ Won' if parlay['parlay_won'] else '❌ Lost'}")
                                        
                                        col_info1, col_info2, col_info3 = st.columns(3)
                                        with col_info1:
                                            investment = parlay['adjusted_amount'] if amount_type == 'adjusted_amount' else parlay['amount']
                                            st.write(f"Investment: ${investment:.2f}")
                                        with col_info2:
                                            odds_display = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                            st.write(f"{odds_type.title()} Odds: {odds_display:.2f}")
                                        with col_info3:
                                            if parlay['parlay_won']:
                                                odds_for_calc = parlay['boosted_odds'] if odds_type == 'boosted' else parlay['odds']
                                                profit = investment * (odds_for_calc - 1)
                                                st.write(f"Profit: ${profit:.2f}")
                                            else:
                                                st.write(f"Loss: ${investment:.2f}")
                                        
                                        st.markdown("**Individual Picks:**")
                                        for pick in parlay['picks_details']:
                                            pick_status = '✅' if pick['pick_won'] else '❌'
                                            date_str = f" [{pick['date']}]" if pick.get('date') else ""
                                            st.write(f"{pick_status} {pick['team']} ({pick['match']}){date_str} - Winner: {pick['actual_winner']}")
                                        
                                        st.markdown("---")
                            else:
                                st.info(f"No completed 5-team parlays available for {amount_type} with {odds_type} odds")
        
        # Daily Performance
        with bet_type_tabs[4]:
            st.markdown("#### Daily Performance Breakdown")
            
            if daily_performance:
                # Create summary table for each day
                daily_summary = []
                for date, day_results in daily_performance.items():
                    day_summary = {'Date': date}
                    
                    # Singles
                    if 'single_bets' in day_results:
                        single_perf = day_results['single_bets']['amount']['performance']
                        day_summary['Singles ROI'] = f"{single_perf['roi']:.1f}%"
                        day_summary['Singles P/L'] = f"${single_perf['net_profit']:.2f}"
                    else:
                        day_summary['Singles ROI'] = "N/A"
                        day_summary['Singles P/L'] = "N/A"
                    
                    # 3-Team Parlays (Boosted Odds)
                    if 'parlay_3_team' in day_results and 'amount' in day_results['parlay_3_team'] and 'boosted' in day_results['parlay_3_team']['amount']:
                        parlay3_perf = day_results['parlay_3_team']['amount']['boosted']['performance']
                        day_summary['3-Team ROI'] = f"{parlay3_perf['roi']:.1f}%"
                        day_summary['3-Team P/L'] = f"${parlay3_perf['net_profit']:.2f}"
                    else:
                        day_summary['3-Team ROI'] = "N/A"
                        day_summary['3-Team P/L'] = "N/A"
                    
                    # 4-Team Parlays (Boosted Odds)
                    if 'parlay_4_team' in day_results and 'amount' in day_results['parlay_4_team'] and 'boosted' in day_results['parlay_4_team']['amount']:
                        parlay4_perf = day_results['parlay_4_team']['amount']['boosted']['performance']
                        day_summary['4-Team ROI'] = f"{parlay4_perf['roi']:.1f}%"
                        day_summary['4-Team P/L'] = f"${parlay4_perf['net_profit']:.2f}"
                    else:
                        day_summary['4-Team ROI'] = "N/A"
                        day_summary['4-Team P/L'] = "N/A"
                    
                    # 5-Team Parlays (Boosted Odds)
                    if 'parlay_5_team' in day_results and 'amount' in day_results['parlay_5_team'] and 'boosted' in day_results['parlay_5_team']['amount']:
                        parlay5_perf = day_results['parlay_5_team']['amount']['boosted']['performance']
                        day_summary['5-Team ROI'] = f"{parlay5_perf['roi']:.1f}%"
                        day_summary['5-Team P/L'] = f"${parlay5_perf['net_profit']:.2f}"
                    else:
                        day_summary['5-Team ROI'] = "N/A"
                        day_summary['5-Team P/L'] = "N/A"
                    
                    daily_summary.append(day_summary)
                
                if daily_summary:
                    daily_df = pd.DataFrame(daily_summary)
                    st.dataframe(daily_df, use_container_width=True, hide_index=True)
                
                # Create performance chart
                st.markdown("#### Performance Trends")
                
                chart_data = []
                for date, day_results in daily_performance.items():
                    for bet_type in ['single_bets', 'parlay_3_team', 'parlay_4_team', 'parlay_5_team']:
                        if bet_type in day_results:
                            if bet_type == 'single_bets':
                                # Singles는 기존 구조 유지
                                perf = day_results[bet_type]['amount']['performance']
                                if perf['total_bets'] > 0:
                                    chart_data.append({
                                        'Date': date,
                                        'Bet Type': bet_type.replace('_', ' ').title(),
                                        'ROI': perf['roi'],
                                        'Net Profit': perf['net_profit']
                                    })
                            else:
                                # Parlays는 boosted odds 사용
                                if 'amount' in day_results[bet_type] and 'boosted' in day_results[bet_type]['amount']:
                                    perf = day_results[bet_type]['amount']['boosted']['performance']
                                    if perf['total_bets'] > 0:
                                        chart_data.append({
                                            'Date': date,
                                            'Bet Type': bet_type.replace('_', ' ').title() + ' (Boosted)',
                                            'ROI': perf['roi'],
                                            'Net Profit': perf['net_profit']
                                        })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    
                    # ROI Chart
                    fig_roi = px.line(
                        chart_df,
                        x='Date',
                        y='ROI',
                        color='Bet Type',
                        title='ROI Trends by Bet Type',
                        markers=True
                    )
                    fig_roi.update_layout(yaxis_title="ROI (%)")
                    st.plotly_chart(fig_roi, use_container_width=True)
                    
                    # Profit/Loss Chart
                    fig_pl = px.bar(
                        chart_df,
                        x='Date',
                        y='Net Profit',
                        color='Bet Type',
                        title='Daily Profit/Loss by Bet Type',
                        barmode='group'
                    )
                    fig_pl.update_layout(yaxis_title="Net Profit ($)")
                    st.plotly_chart(fig_pl, use_container_width=True)
            else:
                st.info("No daily performance data available")
        
        # Data Sources
        st.markdown("### Data Sources")
        with st.expander("View betting analysis files"):
            st.markdown("**Betting Analysis Files (Filtered):**")
            for date, file_path in filtered_files.items():
                filename = Path(file_path).name
                st.markdown(f"- {date}: `{filename}`")
            
            if len(filtered_files) < len(betting_files):
                st.markdown("**Available Files (All):**")
                for date, file_path in betting_files.items():
                    filename = Path(file_path).name
                    if date in filtered_files:
                        st.markdown(f"- {date}: `{filename}` ✅")
                    else:
                        st.markdown(f"- {date}: `{filename}` ⏸️ (filtered out)")
            
            historical_file = betting_tracker.get_latest_historical_records()
            if historical_file:
                hist_filename = Path(historical_file).name
                st.markdown(f"**Historical Records:** `{hist_filename}`")
    
    except Exception as e:
        st.error(f"Error loading betting performance data: {str(e)}")
        st.exception(e)