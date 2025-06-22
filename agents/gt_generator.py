import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd # machine_info.csv를 처리하기 위해 pandas 사용

# 당신의 GA 코드를 임포트 (경로에 맞게 수정 필요)
# 예: from agents.gt_generator.ga_core import GeneticAlgorithm
# 예: from agents.gt_generator.ga_utils import calculate_schedule_cost, check_deadlines

class GTGeneratorAgent:
    def __init__(self, machine_item_capacity_data: pd.DataFrame):
        self.machine_item_capacity_data = machine_item_capacity_data
        
        # 여기서 당신의 GA 클래스 인스턴스화 또는 함수 로드
        # self.ga_solver = GeneticAlgorithm(machine_item_capacity_data=machine_item_capacity_data)

    def _calculate_fitness(self, schedule: Dict[str, Any], order_info: Dict[str, Any], current_date_str: str) -> float:
        """
        주어진 스케줄의 적합도를 계산합니다. (GA의 핵심)
        납기, 비용, 긴급 여부 등을 종합적으로 고려합니다.
        이 함수는 당신의 ga_core.py/ga_utils.py에 있는 적합도 함수를 호출하도록 구현해야 합니다.
        """
        # --- 여기에 당신의 기존 GA 적합도 계산 로직을 통합 ---
        # 예시:
        fitness = 0.0
        
        # 1. 납기 지연 패널티
        try:
            due_date = datetime.strptime(order_info['time'], '%Y-%m-%d').date()
            completion_date = datetime.strptime(schedule['estimated_completion_date'], '%Y-%m-%d').date()
            if completion_date > due_date:
                delay_days = (completion_date - due_date).days
                fitness += delay_days * 100 # 기본 패널티
                if order_info['urgent']:
                    fitness += delay_days * 300 # 긴급 주문은 더 큰 패널티
            elif completion_date < due_date:
                # 조기 완료에 대한 보너스 (선택 사항)
                pass
        except ValueError:
            fitness += 1000 # 날짜 파싱 오류

        # 2. 총 생산 비용 패널티 (기본 비용 대비 증가분)
        # 실제 GA에서는 자원 사용량에 따른 비용 모델이 필요
        # 예시: 총 생산 시간이 길수록 비용 증가
        if schedule['total_production_hours'] is not None:
             fitness += schedule['total_production_hours'] * 5 # 시간당 가중치
        
        # 3. 비현실적인 스케줄 패널티 (시작일이 현재보다 이전, 완료일이 시작일보다 빠름 등)
        try:
            start_date = datetime.strptime(schedule['estimated_start_date'], '%Y-%m-%d').date()
            current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()
            if start_date < current_date:
                fitness += 500 # 과거 시작일 패널티
            if datetime.strptime(schedule['estimated_completion_date'], '%Y-%m-%d').date() < start_date:
                fitness += 500 # 완료일이 시작일보다 빠름 패널티
        except ValueError:
            fitness += 500 # 날짜 파싱 오류

        return fitness

    def generate_gt_schedule(self, order_info: Dict[str, Any], current_date: str) -> str:
        print(f"  GTGeneratorAgent: 유전 알고리즘으로 GT 스케줄 생성 요청 - 품목: {order_info.get('item')}, 긴급: {order_info.get('urgent')}")

        # --- 이 부분에서 당신의 GA 코드를 호출하여 최적 스케줄을 생성합니다 ---
        # 예: best_schedule = self.ga_solver.run(order_info, current_date)
        
        # 임시 더미 GA 결과 (실제 GA 결과로 대체되어야 함)
        # 당신의 GA 코드가 생성하는 스케줄 결과와 포맷을 맞춰야 합니다.
        best_schedule_result = self._run_dummy_ga(order_info, current_date)

        if best_schedule_result:
            return (
                f"GT 스케줄 (GA 기반):\n"
                f"  추천 기계: {best_schedule_result.get('recommended_machine')}\n"
                f"  예상 생산 시작일: {best_schedule_result.get('estimated_start_date')}\n"
                f"  예상 생산 완료일: {best_schedule_result.get('estimated_completion_date')}\n"
                f"  총 필요 시간: {best_schedule_result.get('total_production_hours')} 시간\n"
                f"  총 생산 비용: {best_schedule_result.get('total_production_cost')} 원\n"
                f"  고려사항: {best_schedule_result.get('notes')}"
            )
        else:
            return "GT 스케줄 생성 실패 (GA 실행 오류 또는 최적 해를 찾지 못함)"

    def _run_dummy_ga(self, order_info: Dict[str, Any], current_date_str: str) -> Dict[str, Any]:
        """
        임시 GA 시뮬레이션: 가장 적합한 기계 하나를 무작위로 선택하고, 대략적인 스케줄 생성.
        이 함수는 당신의 ga_core.py의 핵심 GA 실행 로직으로 대체되어야 합니다.
        """
        POPULATION_SIZE = 50
        NUM_GENERATIONS = 100

        best_chromosome_data = None
        min_fitness = float('inf')

        # 주문 품목을 생산할 수 있는 기계 목록 필터링
        eligible_machines = self.machine_item_capacity_data[self.machine_item_capacity_data['item'] == order_info['item']]
        
        if eligible_machines.empty:
            print(f"  GTGeneratorAgent 경고: 품목 '{order_info['item']}'을 생산할 수 있는 기계가 없습니다.")
            return {}

        for _ in range(NUM_GENERATIONS): # 각 세대 반복
            # 간단한 초기화 (실제 GA에서는 더 정교함)
            temp_machine = eligible_machines.sample(1).iloc[0] # 적합한 기계 중 하나 무작위 선택
            
            # 예상 시간 계산
            capacity = temp_machine['capacity']
            estimated_hours = order_info['qty'] / capacity if capacity > 0 else order_info['qty'] * 10
            estimated_hours = max(1.0, round(estimated_hours, 2)) # 최소 1시간
            
            # 날짜 계산 (납기일과 현재 날짜 고려)
            try:
                due_date_dt = datetime.strptime(order_info['time'], '%Y-%m-%d').date()
            except ValueError:
                due_date_dt = datetime.now().date() + timedelta(days=30)
            
            current_date_dt = datetime.strptime(current_date_str, '%Y-%m-%d').date()

            # 시작일은 현재 날짜 이후, 완료일은 납기일 이전 또는 최소한의 지연
            start_date = current_date_dt + timedelta(days=random.randint(0, 3)) # 0~3일 후 시작
            
            # 완료일은 시작일 + 생산 시간 (일 단위 변환)
            completion_date = start_date + timedelta(days=max(1, int(estimated_hours / 8))) # 8시간 근무일 기준

            # 납기일 초과 시 강제 조정 (임시)
            if completion_date > due_date_dt:
                completion_date = due_date_dt
                start_date = completion_date - timedelta(days=max(1, int(estimated_hours / 8))) # 역산

            current_schedule = {
                "recommended_machine_id": temp_machine['machine'], # 여기서는 기계 이름이 ID 역할
                "recommended_machine": temp_machine['machine'],
                "estimated_start_date": start_date.strftime('%Y-%m-%d'),
                "estimated_completion_date": completion_date.strftime('%Y-%m-%d'),
                "total_production_hours": estimated_hours,
                "total_production_cost": order_info['qty'] * order_info['cost'], # 임시 비용
                "notes": "GA 더미 시뮬레이션 결과"
            }
            
            current_fitness = self._calculate_fitness(current_schedule, order_info, current_date_str)

            if current_fitness < min_fitness:
                min_fitness = current_fitness
                best_chromosome_data = current_schedule.copy()
        
        return best_chromosome_data