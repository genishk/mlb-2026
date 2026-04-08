import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

class BettingPerformanceTracker:
    """
    베팅 추천의 실제 성과를 추적하는 클래스
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data" / "analysis"
        self.historical_dir = Path(__file__).parent.parent / "data" / "records"
        
    def get_betting_analysis_files(self) -> Dict[str, str]:
        """베팅 분석 파일들을 날짜별로 가져오기"""
        files_by_date = {}
        
        if not self.data_dir.exists():
            return files_by_date
            
        for file_path in self.data_dir.glob("mlb_betting_analysis_*.json"):
            # 파일명에서 날짜 추출 (예: mlb_betting_analysis_20250524_131956.json)
            filename = file_path.stem
            try:
                date_part = filename.split('_')[3]  # 20250524
                date_obj = datetime.strptime(date_part, '%Y%m%d')
                date_str = date_obj.strftime('%Y-%m-%d')
                files_by_date[date_str] = str(file_path)
            except (ValueError, IndexError):
                continue
                
        return files_by_date
    
    def get_latest_historical_records(self) -> Optional[str]:
        """가장 최근의 historical records 파일 경로 반환"""
        if not self.historical_dir.exists():
            return None
            
        historical_files = list(self.historical_dir.glob("mlb_historical_records_*.json"))
        if not historical_files:
            return None
            
        # 파일명에서 날짜 추출하여 정렬
        dated_files = []
        for file_path in historical_files:
            try:
                filename = file_path.stem
                date_part = filename.split('_')[3]  # mlb_historical_records_20250525_123436.json
                date_obj = datetime.strptime(date_part, '%Y%m%d')
                dated_files.append((date_obj, str(file_path)))
            except (ValueError, IndexError):
                continue
                
        if dated_files:
            dated_files.sort(key=lambda x: x[0], reverse=True)
            return dated_files[0][1]
        
        return None
    
    def load_betting_analysis(self, file_path: str) -> Dict:
        """베팅 분석 JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading betting analysis file {file_path}: {e}")
            return {}
    
    def load_historical_data(self, file_path: str) -> List[Dict]:
        """Historical records JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading historical data file {file_path}: {e}")
            return []
    
    def match_bet_with_results(self, bet: Dict, historical_data: List[Dict]) -> Optional[Dict]:
        """개별 베팅을 historical records와 매칭"""
        try:
            if 'picks' in bet:  # 팔레이 베팅
                return self._match_parlay_with_results(bet, historical_data)
            else:  # 싱글 베팅
                return self._match_single_with_results(bet, historical_data)
        except Exception as e:
            print(f"Error matching bet with results: {e}")
            return None
    
    def _match_single_with_results(self, bet: Dict, historical_data: List[Dict]) -> Optional[Dict]:
        """싱글 베팅을 결과와 매칭"""
        team = bet.get('team')
        match = bet.get('match')
        date = bet.get('date')
        
        if not all([team, match, date]):
            return None
            
        # match에서 홈팀과 원정팀 추출 (예: "Chicago Cubs vs Cincinnati Reds")
        if ' vs ' not in match:
            return None
            
        match_parts = match.split(' vs ')
        if len(match_parts) != 2:
            return None
            
        away_team = match_parts[0].strip()
        home_team = match_parts[1].strip()
        
        # Historical data에서 해당 경기 찾기
        for record in historical_data:
            if (record.get('date') == date and 
                record.get('away_team_name') == away_team and
                record.get('home_team_name') == home_team):
                
                # 승자 결정
                home_score = record.get('home_score', 0)
                away_score = record.get('away_score', 0)
                
                if home_score > away_score:
                    winner = home_team
                elif away_score > home_score:
                    winner = away_team
                else:
                    continue  # 무승부인 경우 스킵
                
                # 베팅한 팀이 이겼는지 확인
                bet_won = (winner == team)
                
                return {
                    'bet': bet,
                    'result': record,
                    'bet_won': bet_won,
                    'completed': True
                }
        
        # 경기를 찾지 못한 경우 (아직 완료되지 않음)
        return {
            'bet': bet,
            'result': None,
            'bet_won': None,
            'completed': False
        }
    
    def _match_parlay_with_results(self, parlay: Dict, historical_data: List[Dict]) -> Optional[Dict]:
        """팔레이 베팅을 결과와 매칭"""
        picks = parlay.get('picks', [])
        if not picks:
            return None
        
        matched_picks = []
        all_completed = True
        all_won = True
        
        for pick in picks:
            team = pick.get('team')
            match = pick.get('match')
            date = pick.get('date')
            
            if not all([team, match, date]):
                all_completed = False
                continue
            
            # match에서 홈팀과 원정팀 추출
            if ' vs ' not in match:
                all_completed = False
                continue
                
            match_parts = match.split(' vs ')
            if len(match_parts) != 2:
                all_completed = False
                continue
                
            away_team = match_parts[0].strip()
            home_team = match_parts[1].strip()
            
            # Historical data에서 해당 경기 찾기
            pick_result = None
            for record in historical_data:
                if (record.get('date') == date and 
                    record.get('away_team_name') == away_team and
                    record.get('home_team_name') == home_team):
                    
                    # 승자 결정
                    home_score = record.get('home_score', 0)
                    away_score = record.get('away_score', 0)
                    
                    if home_score > away_score:
                        winner = home_team
                        actual_winner = home_team
                    elif away_score > home_score:
                        winner = away_team
                        actual_winner = away_team
                    else:
                        continue  # 무승부인 경우 스킵
                    
                    pick_won = (winner == team)
                    pick_result = {
                        'pick': pick,
                        'result': record,
                        'pick_won': pick_won,
                        'team': team,
                        'match': match,
                        'date': date,
                        'actual_winner': actual_winner
                    }
                    break
            
            if pick_result is None:
                # 경기가 아직 완료되지 않음
                all_completed = False
                pick_result = {
                    'pick': pick,
                    'result': None,
                    'pick_won': None,
                    'team': team,
                    'match': match,
                    'date': date,
                    'actual_winner': None
                }
            else:
                if not pick_result['pick_won']:
                    all_won = False
            
            matched_picks.append(pick_result)
        
        # 팔레이는 모든 picks가 이겨야 성공
        parlay_won = all_completed and all_won
        
        return {
            'bet': parlay,
            'picks_results': matched_picks,
            'bet_won': parlay_won if all_completed else None,
            'completed': all_completed
        }
    
    def calculate_betting_performance(self, betting_data: Dict, historical_data: List[Dict]) -> Dict:
        """베팅 성과 계산"""
        results = {
            'single_bets': {'amount': {'performance': {}, 'details': []}},
            'parlay_3_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            },
            'parlay_4_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            },
            'parlay_5_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            }
        }
        
        # 싱글 베팅 성과 계산
        if 'singles' in betting_data:
            single_results = []
            for bet in betting_data['singles']:
                matched = self.match_bet_with_results(bet, historical_data)
                if matched and matched['completed']:
                    single_results.append(matched)
            
            results['single_bets']['amount']['performance'] = self._calculate_performance_metrics(
                single_results, 'amount', False
            )
            results['single_bets']['amount']['details'] = single_results
        
        # 팔레이 베팅 성과 계산
        if 'parlays' in betting_data:
            for parlay in betting_data['parlays']:
                parlay_type = parlay.get('type', '')
                matched = self.match_bet_with_results(parlay, historical_data)
                
                if matched and matched['completed']:
                    if parlay_type == '3_team_parlay':
                        results['parlay_3_team']['amount']['boosted']['details'].append(matched)
                        results['parlay_3_team']['amount']['regular']['details'].append(matched)
                        results['parlay_3_team']['adjusted_amount']['boosted']['details'].append(matched)
                        results['parlay_3_team']['adjusted_amount']['regular']['details'].append(matched)
                    elif parlay_type == '4_team_parlay':
                        results['parlay_4_team']['amount']['boosted']['details'].append(matched)
                        results['parlay_4_team']['amount']['regular']['details'].append(matched)
                        results['parlay_4_team']['adjusted_amount']['boosted']['details'].append(matched)
                        results['parlay_4_team']['adjusted_amount']['regular']['details'].append(matched)
                    elif parlay_type == '5_team_parlay':
                        results['parlay_5_team']['amount']['boosted']['details'].append(matched)
                        results['parlay_5_team']['amount']['regular']['details'].append(matched)
                        results['parlay_5_team']['adjusted_amount']['boosted']['details'].append(matched)
                        results['parlay_5_team']['adjusted_amount']['regular']['details'].append(matched)
        
        # 팔레이 성과 계산
        for parlay_type in ['parlay_3_team', 'parlay_4_team', 'parlay_5_team']:
            for amount_type in ['amount', 'adjusted_amount']:
                for odds_type in ['boosted', 'regular']:
                    details = results[parlay_type][amount_type][odds_type]['details']
                    use_boosted = (odds_type == 'boosted')
                    results[parlay_type][amount_type][odds_type]['performance'] = self._calculate_performance_metrics(
                        details, amount_type, True, use_boosted
                    )
        
        return results
    
    def _calculate_performance_metrics(self, matched_results: List[Dict], amount_type: str, is_parlay: bool, use_boosted_odds: bool = True) -> Dict:
        """성과 지표 계산"""
        if not matched_results:
            return {
                'total_bets': 0,
                'won_bets': 0,
                'lost_bets': 0,
                'win_rate': 0.0,
                'total_invested': 0.0,
                'total_returned': 0.0,
                'net_profit': 0.0,
                'roi': 0.0
            }
        
        total_bets = len(matched_results)
        won_bets = sum(1 for result in matched_results if result['bet_won'])
        lost_bets = total_bets - won_bets
        win_rate = won_bets / total_bets if total_bets > 0 else 0.0
        
        total_invested = 0.0
        total_returned = 0.0
        
        for result in matched_results:
            bet = result['bet']
            bet_won = result['bet_won']
            
            # 투자 금액
            if amount_type == 'adjusted_amount' and 'adjusted_amount' in bet:
                investment = bet['adjusted_amount']
            else:
                investment = bet.get('amount', 0.0)
            
            total_invested += investment
            
            # 수익 계산
            if bet_won:
                if is_parlay:
                    # 팔레이의 경우 boosted_odds 또는 일반 odds 사용 (decimal odds)
                    if use_boosted_odds:
                        decimal_odds = bet.get('boosted_odds', bet.get('odds', 1.0))
                    else:
                        decimal_odds = bet.get('odds', 1.0)
                    # decimal odds에서 수익 = 투자금 * (decimal_odds - 1)
                    profit = investment * (decimal_odds - 1)
                else:
                    # 싱글의 경우 일반 odds 사용 (미국식 odds)
                    odds = bet.get('odds', 0.0)
                    if odds > 0:
                        # 양수 odds (예: +150)
                        profit = investment * (odds / 100)
                    else:
                        # 음수 odds (예: -125)
                        profit = investment * (100 / abs(odds))
                
                total_returned += investment + profit
            # 베팅을 잃으면 투자금액만큼 손실 (total_returned에 추가하지 않음)
        
        net_profit = total_returned - total_invested
        roi = (net_profit / total_invested * 100) if total_invested > 0 else 0.0
        
        return {
            'total_bets': total_bets,
            'won_bets': won_bets,
            'lost_bets': lost_bets,
            'win_rate': win_rate * 100,  # 백분율로 변환
            'total_invested': total_invested,
            'total_returned': total_returned,
            'net_profit': net_profit,
            'roi': roi
        }
    
    def get_daily_performance(self, betting_files: Dict[str, str]) -> Dict[str, Dict]:
        """일별 베팅 성과 분석"""
        daily_performance = {}
        
        # 최신 historical records 로드
        historical_file = self.get_latest_historical_records()
        if not historical_file:
            return daily_performance
        
        historical_data = self.load_historical_data(historical_file)
        
        for date, file_path in betting_files.items():
            betting_data = self.load_betting_analysis(file_path)
            if betting_data:
                daily_performance[date] = self.calculate_betting_performance(betting_data, historical_data)
        
        return daily_performance
    
    def get_total_performance(self, betting_files: Dict[str, str]) -> Dict:
        """전체 베팅 성과 분석"""
        daily_performance = self.get_daily_performance(betting_files)
        
        if not daily_performance:
            return {}
        
        # 모든 일별 성과를 합계
        total_results = {
            'single_bets': {'amount': {'performance': {}, 'details': []}},
            'parlay_3_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            },
            'parlay_4_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            },
            'parlay_5_team': {
                'amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                },
                'adjusted_amount': {
                    'boosted': {'performance': {}, 'details': []},
                    'regular': {'performance': {}, 'details': []}
                }
            }
        }
        
        # 모든 일별 데이터 누적
        for date, day_results in daily_performance.items():
            for bet_type in ['single_bets', 'parlay_3_team', 'parlay_4_team', 'parlay_5_team']:
                if bet_type in day_results:
                    if bet_type == 'single_bets':
                        amount_types = ['amount']
                        for amount_type in amount_types:
                            if amount_type in day_results[bet_type]:
                                details = day_results[bet_type][amount_type].get('details', [])
                                total_results[bet_type][amount_type]['details'].extend(details)
                    else:
                        amount_types = ['amount', 'adjusted_amount']
                        for amount_type in amount_types:
                            if amount_type in day_results[bet_type]:
                                for odds_type in ['boosted', 'regular']:
                                    if odds_type in day_results[bet_type][amount_type]:
                                        details = day_results[bet_type][amount_type][odds_type].get('details', [])
                                        total_results[bet_type][amount_type][odds_type]['details'].extend(details)
        
        # 전체 성과 계산
        for bet_type in ['single_bets', 'parlay_3_team', 'parlay_4_team', 'parlay_5_team']:
            if bet_type == 'single_bets':
                amount_types = ['amount']
                for amount_type in amount_types:
                    details = total_results[bet_type][amount_type]['details']
                    is_parlay = bet_type != 'single_bets'
                    total_results[bet_type][amount_type]['performance'] = self._calculate_performance_metrics(
                        details, amount_type, is_parlay
                    )
            else:
                amount_types = ['amount', 'adjusted_amount']
                for amount_type in amount_types:
                    for odds_type in ['boosted', 'regular']:
                        details = total_results[bet_type][amount_type][odds_type]['details']
                        is_parlay = bet_type != 'single_bets'
                        use_boosted = (odds_type == 'boosted')
                        total_results[bet_type][amount_type][odds_type]['performance'] = self._calculate_performance_metrics(
                            details, amount_type, is_parlay, use_boosted
                        )
        
        return total_results
    
    def get_performance_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[Dict, Dict]:
        """베팅 성과 요약 반환"""
        betting_files = self.get_betting_analysis_files()
        
        # 날짜 필터링
        if start_date or end_date:
            betting_files = self._filter_files_by_date(betting_files, start_date, end_date)
        
        daily_performance = self.get_daily_performance(betting_files)
        total_performance = self.get_total_performance(betting_files)
        
        return daily_performance, total_performance
    
    def _filter_files_by_date(self, betting_files: Dict[str, str], start_date: Optional[str], end_date: Optional[str]) -> Dict[str, str]:
        """날짜 범위로 베팅 파일 필터링"""
        filtered_files = {}
        
        for date_str, file_path in betting_files.items():
            # 날짜 비교
            include_file = True
            
            if start_date:
                if date_str < start_date:
                    include_file = False
            
            if end_date:
                if date_str > end_date:
                    include_file = False
            
            if include_file:
                filtered_files[date_str] = file_path
        
        return filtered_files
    
    def get_parlay_details(self, parlay_results: List[Dict]) -> List[Dict]:
        """팔레이 베팅의 상세 정보 생성"""
        detailed_parlays = []
        
        for result in parlay_results:
            bet = result['bet']
            parlay_detail = {
                'teams': bet.get('teams', []),
                'bet_type': bet.get('type', ''),
                'probability': bet.get('probability', 0.0),
                'odds': bet.get('odds', 0.0),
                'boosted_odds': bet.get('boosted_odds', 0.0),
                'amount': bet.get('amount', 0.0),
                'adjusted_amount': bet.get('adjusted_amount', 0.0),
                'potential_profit': bet.get('potential_profit', 0.0),
                'expected_profit': bet.get('expected_profit', 0.0),
                'parlay_won': result['bet_won'],
                'picks_details': []
            }
            
            # 개별 픽 상세 정보
            if 'picks_results' in result:
                for pick_result in result['picks_results']:
                    pick = pick_result['pick']
                    pick_detail = {
                        'team': pick.get('team'),
                        'match': pick.get('match'),
                        'date': pick.get('date'),
                        'probability': pick.get('probability', 0.0),
                        'odds': pick.get('odds', 0.0),
                        'pick_won': pick_result.get('pick_won'),
                        'actual_winner': pick_result.get('actual_winner')
                    }
                    parlay_detail['picks_details'].append(pick_detail)
            
            detailed_parlays.append(parlay_detail)
        
        return detailed_parlays
    
    def calculate_expected_vs_actual(self, performance_data: Dict) -> Dict:
        """예상 수익 vs 실제 수익 비교"""
        comparison = {}
        
        for bet_type in ['single_bets', 'parlay_3_team', 'parlay_4_team', 'parlay_5_team']:
            if bet_type in performance_data:
                if bet_type == 'single_bets':
                    amount_types = ['amount']
                else:
                    amount_types = ['amount', 'adjusted_amount']
                
                comparison[bet_type] = {}
                
                for amount_type in amount_types:
                    if amount_type in performance_data[bet_type]:
                        details = performance_data[bet_type][amount_type]['details']
                        
                        total_expected = sum(detail['bet'].get('expected_profit', 0.0) for detail in details)
                        total_actual = performance_data[bet_type][amount_type]['performance']['net_profit']
                        
                        comparison[bet_type][amount_type] = {
                            'expected_profit': total_expected,
                            'actual_profit': total_actual,
                            'difference': total_actual - total_expected,
                            'accuracy': (total_actual / total_expected * 100) if total_expected != 0 else 0
                        } 