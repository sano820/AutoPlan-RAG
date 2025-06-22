# agents/gt_generator.py (ga_util.py를 수정할 수 없을 때)

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import json

# 당신의 GA 코드를 임포트 (경로에 맞게 수정)
from agents.gt_generator.ga_core import GeneticAlgorithm
# ga_util.py는 직접 수정하지 않으므로, 그 안의 함수를 직접 호출하지 않습니다.
# import agents.gt_generator.ga_util as ga_util # 이 줄은 더 이상 필요 없을 수 있습니다.


class GTGeneratorAgent:
    """
    유전 알고리즘(GA)을 사용하여 최적의 스케줄(Ground Truth)을 생성하는 에이전트.
    """
    def __init__(self, machine_item_capacity_data: pd.DataFrame):
        self.machine_item_capacity_data = machine_item_capacity_data
        print("  GTGeneratorAgent: 유전 알고리즘 기반 GT 스케줄러 초기화 완료.")

        # GA 클래스 초기화에 필요한 데이터셋을 저장할 내부 변수 (GA 출력 변환 시 필요)
        self.dit_data_for_conversion = {}
        self.mijt_data_for_conversion = {}

    def _prepare_ga_inputs(self, order_info: Dict[str, Any], current_date_str: str) -> Dict[str, Any]:
        """
        주문 정보와 기계 데이터를 바탕으로 GA에 필요한 입력 데이터를 준비합니다.
        (T_set, I_set, J_set, cit_data, pit_data, dit_data, mijt_data 등)
        ga_util.py를 수정할 수 없으므로, 여기서 모든 전처리 로직을 직접 구현합니다.
        """
        item = order_info['item']
        qty = order_info['qty']
        due_date_str = order_info['time'] # '2025-07-15' (예시)
        cost_per_item = order_info['cost']
        urgent = order_info['urgent']

        # 날짜 관련 처리 (T_set 생성)
        current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
        due_date = datetime.strptime(due_date_str, '%Y-%m-%d')

        T_set = []
        delta = timedelta(days=1)
        temp_date = current_date # 현재 날짜부터 시작
        while temp_date <= due_date:
            T_set.append(temp_date.strftime('%Y-%m-%d'))
            temp_date += delta
        
        # 품목 종류 집합 (I_set): 현재는 주문 품목 하나만 고려
        I_set = [item]

        # 기계 종류 집합 (J_set): machine_item_capacity_data에서 유일한 기계 목록 추출
        J_set = self.machine_item_capacity_data['machine'].unique().tolist()
        J_set.sort() # 순서 일관성을 위해 정렬


        # dit_data: (item, time)을 키로 하는 요구량 딕셔너리
        # 납기일에만 총 요구량이 있는 것으로 가정
        dit_data = {}
        for t in T_set:
            if t == due_date_str:
                dit_data[(item, t)] = qty
            else:
                dit_data[(item, t)] = 0

        # cit_data: (item, time)을 키로 하는 미생산 시 비용 딕셔너리 (페널티)
        # 납기일에만 미생산 페널티를 부과
        cit_data = {}
        for t in T_set:
            if t == due_date_str:
                # 미생산 시 개당 비용 * 갯수를 페널티로 (GA가 최소화하는 값)
                cit_data[(item, t)] = cost_per_item * qty # 예시, 실제는 더 복잡할 수 있음
            else:
                cit_data[(item, t)] = 0

        # pit_data: (item, time)을 키로 하는 긴급생산 필요 여부 (우선순위 가중치) 딕셔너리
        # GA의 _calculate_objective_value에서 사용됨 (priority_it)
        pit_data = {}
        for t in T_set:
            if t == due_date_str:
                # 긴급하면 높은 우선순위 가중치 (더 큰 페널티를 받도록)
                pit_data[(item, t)] = 2 if urgent else 1
            else:
                pit_data[(item, t)] = 1 # 다른 날짜에는 기본 우선순위

        # mijt_data: (item, machine, time)을 키로 하는 생산 능력 딕셔너리
        # GA의 _check_constraints에서는 mijt가 "단위 생산당 필요한 시간(분)"으로 해석됨.
        # machine_item_capacity_data['capacity']는 '시간당 생산량'이므로 변환이 필요합니다.
        
        mijt_data = {}
        for _, row in self.machine_item_capacity_data.iterrows():
            machine_name = row['machine']
            prod_item = row['item']
            capacity_per_hour = row['capacity'] # machine_info.csv의 capacity (예: 시간당 5개)

            # 주문 품목과 관련된 기계-품목 조합만 고려 (혹은 모든 조합 고려 가능)
            if prod_item == item: # 현재 주문 품목만 필터링
                for t in T_set:
                    if capacity_per_hour > 0:
                        # 1개 생산에 필요한 시간 (분) = 60분 / 시간당 생산량
                        time_per_unit_minutes = 60 / capacity_per_hour 
                        mijt_data[(prod_item, machine_name, t)] = time_per_unit_minutes
                    else:
                        # 생산 능력이 0이거나 음수이면 생산 불가능 (무한한 시간 소요)
                        mijt_data[(prod_item, machine_name, t)] = float('inf')
            # else: # 주문 품목이 아닌 경우 해당 mijt 키는 생성하지 않음 (GA가 get 시 0으로 처리)
            #     pass

        # xijt_keys_list: GA 염색체(xijt 딕셔너리)의 키 순서를 정의
        # 모든 가능한 (i,j,t) 조합을 생성합니다.
        xijt_keys_list = []
        for i in I_set:
            for j in J_set:
                for t in T_set:
                    # mijt_data에 존재하는 유효한 생산 조합만 키에 포함 (선택 사항)
                    # GA 내부에서 mijt=0인 경우 0으로 처리하므로 모든 조합을 넣어도 무방
                    xijt_keys_list.append((i, j, t))


        return {
            'T_set': T_set,
            'I_set': I_set,
            'J_set': J_set,
            'cit_data': cit_data,
            'pit_data': pit_data,
            'dit_data': dit_data,
            'mijt_data': mijt_data,
            'xijt_keys_list': xijt_keys_list
        }

    def _convert_ga_output_to_schedule_format(self, best_xijt_solution: Dict[tuple, float], order_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        GA의 xijt 딕셔너리 형태의 최적해를 AI 스케줄과 비교 가능한 형식으로 변환합니다.
        _prepare_ga_inputs에서 저장된 dit_data_for_conversion 및 mijt_data_for_conversion 사용.
        """
        item = order_info['item']
        order_qty = order_info['qty']
        cost_per_item = order_info['cost']

        recommended_machines_info = {} # {machine_name: assigned_qty}
        total_production_minutes_ga = 0
        
        # GA의 xijt를 기반으로 각 기계별 할당 수량 계산 및 총 시간 계산
        # dit_data_for_conversion과 mijt_data_for_conversion 사용
        for (i, j, t), ratio in best_xijt_solution.items():
            if i == item and ratio > 0: # 현재 주문 품목에 대한 유효한 생산 계획만
                # 해당 (i,t)에 대한 요구량 (dit_data_for_conversion에서 가져옴)
                demand_at_t = self.dit_data_for_conversion.get((i, t), 0)
                
                # GA가 할당한 비율을 기반으로 실제 할당 수량 계산
                assigned_qty = round(ratio * demand_at_t) 
                
                if assigned_qty > 0: # 0보다 큰 할당 수량만 기록
                    if j not in recommended_machines_info:
                        recommended_machines_info[j] = 0
                    recommended_machines_info[j] += assigned_qty
                    
                    # 해당 기계-품목-시간 조합의 단위 생산당 시간 (mijt_data_for_conversion에서 가져옴)
                    time_per_unit = self.mijt_data_for_conversion.get((i, j, t), float('inf'))
                    if time_per_unit != float('inf'):
                        total_production_minutes_ga += assigned_qty * time_per_unit
        
        # recommended_machines 리스트 포맷으로 변환
        final_recommended_machines = [{"machine_name": m, "assigned_qty": q} for m, q in recommended_machines_info.items()]

        # 총 생산 시간 (시)
        total_production_hours = total_production_minutes_ga / 60 if total_production_minutes_ga > 0 else 0

        # 예상 시작/완료일 추론
        # xijt_solution에서 생산량이 0보다 큰 모든 (item, machine, time) 키의 time 부분을 추출
        relevant_dates_for_prod = sorted(list(set([k[2] for k, val in best_xijt_solution.items() if val > 0 and k[0] == item])))
        
        estimated_start_date = relevant_dates_for_prod[0] if relevant_dates_for_prod else order_info.get('current_date_str', datetime.now().strftime('%Y-%m-%d'))
        estimated_completion_date = relevant_dates_for_prod[-1] if relevant_dates_for_prod else order_info['time'] # 납기일로 기본값 설정
        
        # 총 생산 비용 계산
        # GA는 페널티를 최소화하지만, 여기서 실제 생산 비용을 추정할 수 있습니다.
        # 여기서는 단순히 주문량 * 개당 비용으로 가정합니다.
        # GA가 미생산 페널티를 줄였다면, 주문량에 근접하게 생산했다고 가정할 수 있습니다.
        total_production_cost = order_qty * cost_per_item 

        notes = "유전 알고리즘 기반 스케줄링 결과."
        
        return {
            "recommended_machines": final_recommended_machines,
            "estimated_start_date": estimated_start_date,
            "estimated_completion_date": estimated_completion_date,
            "total_production_hours": round(total_production_hours, 2), # 소수점 둘째 자리까지
            "total_production_cost": round(total_production_cost, 2),
            "notes": notes
        }

    def generate_gt_schedule(self, order_info: Dict[str, Any], current_date: str) -> str:
        print(f"  GTGeneratorAgent: 유전 알고리즘으로 GT 스케줄 생성 요청 - 품목: {order_info.get('item')}, 긴급: {order_info.get('urgent')}")

        # 1. GA 입력 데이터 준비 (이제 _prepare_ga_inputs가 모든 것을 처리)
        ga_inputs = self._prepare_ga_inputs(order_info, current_date)
        
        # GTGeneratorAgent 클래스 내부에 GA 입력 데이터 저장 (GA 출력 변환 함수에서 필요)
        self.dit_data_for_conversion = ga_inputs['dit_data']
        self.mijt_data_for_conversion = ga_inputs['mijt_data']

        # 2. GeneticAlgorithm 인스턴스 생성 및 실행
        ga_solver = GeneticAlgorithm(
            T_set=ga_inputs['T_set'],
            I_set=ga_inputs['I_set'],
            J_set=ga_inputs['J_set'],
            cit_data=ga_inputs['cit_data'],
            pit_data=ga_inputs['pit_data'],
            dit_data=ga_inputs['dit_data'],
            mijt_data=ga_inputs['mijt_data'],
            n_pop=50,      # 적절한 값으로 조정
            n_iter=200,    # 적절한 값으로 조정
            r_cross=0.9,
            r_mut=0.01,
            xijt_keys_list=ga_inputs['xijt_keys_list']
            # max_machine_work_time, overproduction_penalty_factor, gene_swap_prob 등은 ga_core의 기본값 사용
        )
        
        best_xijt_solution, best_objective_score, ga_log, ga_log_detail = ga_solver.solve()

        if best_xijt_solution:
            # 3. GA 결과(xijt)를 스케줄 형식으로 변환
            gt_schedule_data = self._convert_ga_output_to_schedule_format(best_xijt_solution, order_info)
            
            gt_schedule_data['ga_objective_score'] = round(best_objective_score, 2)
            gt_schedule_data['notes'] += f" (GA 점수: {gt_schedule_data['ga_objective_score']})"

            return json.dumps(gt_schedule_data, indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "error": "GT 스케줄 생성 실패 (GA 최적화 문제 또는 해를 찾지 못함)",
                "details": "Genetic Algorithm did not return a valid solution."
            }, indent=2, ensure_ascii=False)