import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class PlannerAgent:
    def __init__(self, llm_model: str):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)
        self.prompt_template = PromptTemplate(
            input_variables=["item", "qty", "due_date", "cost_per_item", "is_urgent", "machine_item_capacity_info", "current_date"],
            template="""
            당신은 최고의 생산 스케줄을 생성하는 전문 플래너입니다.
            **현재 날짜**: {current_date}

            **주문 정보:**
            - 품목: {item}
            - 수량: {qty}개
            - 납기: {due_date}
            - 개당 가격: {cost_per_item}원
            - 긴급 여부: {is_urgent}

            **사용 가능한 기계-품목별 생산 능력 정보:**
            {machine_item_capacity_info}

            위 정보를 바탕으로, 다음 조건을 만족하는 가장 효율적이고 현실적인 생산 스케줄을 JSON 형태로 제안해주세요:
            1.  **추천 기계**: 이 작업을 수행할 가장 적합한 기계 (기계 이름). 여러 기계 사용 시, 각 기계별 할당량을 명시.
            2.  **예상 생산 시작일**: YYYY-MM-DD 형식. (현재 날짜 이후여야 함)
            3.  **예상 생산 완료일**: YYYY-MM-DD 형식. (납기일을 가능한 한 준수)
            4.  **총 예상 생산 시간**: (시간 단위, 소수점 포함 가능)
            5.  **예상 총 생산 비용**: (수량 * 개당 가격)을 기준으로 예상되는 총 생산 비용.
            6.  **고려사항**: 스케줄에 영향을 미칠 수 있는 추가 제약 조건, 긴급 주문 처리 전략, 비용 효율화 방안 등.

            **최적화 목표:**
            - 납기일 준수 최우선 (특히 긴급 주문)
            - 총 생산 비용 최소화
            - 기계 자원 효율적 활용

            JSON 출력은 반드시 다음 키를 포함해야 합니다: 'recommended_machines', 'estimated_start_date', 'estimated_completion_date', 'total_production_hours', 'total_production_cost', 'notes'.
            'recommended_machines'는 배열 형태여야 하며, 각 요소는 '{{"machine_name": "...", "assigned_qty": ...}}' 형태입니다.

            예시 출력:
            {{
              "recommended_machines": [
                {{"machine_name": "CNC 밀링 머신", "assigned_qty": 100}}
              ],
              "estimated_start_date": "2025-06-25",
              "estimated_completion_date": "2025-06-29",
              "total_production_hours": 20.0,
              "total_production_cost": 500000,
              "notes": "긴급 주문이므로 우선 처리. 기계 M001의 가용성 확인."
            }}

            스케줄 생성:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_schedule(self, order_info: Dict[str, Any], machine_item_capacity_info: List[str], current_date: str) -> str:
        print(f"  PlannerAgent: 스케줄 생성 요청 - 주문: {order_info.get('item')}, 긴급: {order_info.get('urgent')}")
        machine_info_str = "\n".join(machine_item_capacity_info)
        try:
            schedule_output = self.chain.run(
                item=order_info["item"],
                qty=order_info["qty"],
                due_date=order_info["time"],
                cost_per_item=order_info["cost"],
                is_urgent=order_info["urgent"],
                machine_item_capacity_info=machine_info_str,
                current_date=current_date
            )
            import re
            import json
            match = re.search(r'\{.*\}', schedule_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_schedule = json.loads(json_str)
                    return json.dumps(parsed_schedule, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    print(f"  PlannerAgent 경고: LLM이 유효하지 않은 JSON을 반환했습니다. 원시 출력: {schedule_output}")
                    return schedule_output
            else:
                print(f"  PlannerAgent 경고: LLM 출력에서 JSON 형식을 찾을 수 없습니다. 원시 출력: {schedule_output}")
                return schedule_output
        except Exception as e:
            print(f"  PlannerAgent 오류: 스케줄 생성 중 예외 발생 - {e}")
            return "스케줄 생성 실패"