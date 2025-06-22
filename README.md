# AutoPlan-RAG
## Multi-Agent RAG 기반 스케줄링 시스템 - Diffusion LLM 활용해보기
이 프로젝트는 Diffusion LLM과 유전 알고리즘(GA)을 활용하여 주문 정보를 바탕으로 최적의 생산 스케줄을 생성하고 평가하는 다중 에이전트 시스템입니다.  
-현재는 Google Gemini LLM 활용 중  

### 프로젝트 목표
LLM 기반 스케줄링: 자연어 주문을 파싱하고, LLM(Gemini)이 기계 정보를 기반으로 초기 스케줄을 제안합니다.  
유전 알고리즘(GA) 기반 최적화: LLM 스케줄의 Ground Truth(GT) 역할을 할 최적의 스케줄을 유전 알고리즘으로 생성합니다.  
스케줄 평가: LLM과 GA가 생성한 스케줄을 비교하고 평가하여 LLM 기반 스케줄링의 성능을 검증하고 개선 방향을 모색합니다.  
확장성: 현재 LLM API를 사용하지만, 향후 Diffusion LLM으로의 확장을 염두에 둡니다.  
실시간 스케줄링: 매일 한 번 스케줄링을 진행하여 당일의 최적 스케줄을 제공하는 것을 목표로 합니다.  

### 프로젝트 구조
```
AutoPlan-RAG/
├── agents/                           # 각 에이전트의 로직을 담는 디렉토리
│   ├── gt_generator/                 # 유전 알고리즘 관련 파일 (ga_core.py, ga_utils.py)
│   │   ├── __init__.py
│   │   ├── ga_core.py                # 당신의 유전 알고리즘 핵심 로직
│   │   └── ga_utils.py               # 당신의 유전 알고리즘 유틸리티 함수들
│   ├── __init__.py
│   ├── evaluator.py                  # AI 스케줄과 GT 스케줄을 비교 평가하는 에이전트
│   ├── order_parser.py               # 자연어 주문을 파싱하는 에이전트
│   ├── planner.py                    # LLM을 사용하여 스케줄을 계획하는 에이전트
│   └── retriever.py                  # 기계 및 품목 생산 능력 정보를 검색하는 에이전트
├── data/                             # 데이터 관련 파일
│   ├── processed/                    # (선택 사항) 전처리된 데이터 또는 중간 결과
│   └── raw/                          # 원본 데이터
│       └── machine_info.csv          # 기계 정보 및 품목별 생산 능력 데이터
├── prompts/                          # LLM 프롬프트 템플릿 파일
│   ├── evaluator_prompt.txt
│   ├── order_parser_prompt.txt
│   └── planner_prompt.txt
├── utils/                            # 유틸리티 함수
│   └── data_loader.py                # 데이터 로드 및 FAISS 벡터스토어 초기화
├── .env                              # 환경 변수 (API 키 등) - Git 추적 제외
├── .gitignore                        # Git 추적에서 제외할 파일 및 폴더 지정
├── main.py                           # 시스템의 메인 실행 파일 (전체 파이프라인 실행)
├── README.md                         # 프로젝트 설명 파일
└── requirements.txt                  # 프로젝트에 필요한 Python 라이브러리 목록
```
### 해야 되는 것들
1. 🧬 유전 알고리즘 (GA) 코드 통합 및 구현 (가장 중요!)  
이 프로젝트의 핵심인 GT 스케줄 생성 로직을 완성하는 단계입니다. 현재 GTGeneratorAgent의 GA 로직은 더미(dummy) 코드이므로, 당신이 보유한 실제 GA 코드로 대체해야 합니다.  

GA 파일들 배치: 당신의 ga_core.py와 ga_utils.py 파일을 agents/gt_generator/ 디렉토리 아래에 넣어주세요.  

agents/gt_generator.py 완성:  
ga_core.py 및 ga_utils.py 모듈을 임포트하도록 수정합니다.  
GTGeneratorAgent 클래스 내부에서 당신의 GA 솔버 클래스를 인스턴스화하고, machine_info.csv에서 로드된 machine_item_capacity_data를 솔버에 전달해야 합니다.  
핵심: 적합도(Fitness) 함수 구현: 당신의 GA 코드 내에서 납기 준수, 총 비용 최소화, 긴급 주문 처리 우선순위, 기계 자원 효율성 등 스케줄링 문제의 모든 제약 조건과 목적 함수를 반영하는 적합도 함수를 정교하게 설계하고 구현합니다. 이것이 GA 성능의 핵심입니다.  
generate_gt_schedule 함수가 실제 GA 솔버를 실행하고, GA가 반환하는 최적의 스케줄 결과를 main.py에서 기대하는 형식(LLM 스케줄과 비교 가능한 형식)으로 가공하여 반환하도록 구현합니다.   
기존의 _run_dummy_ga 함수는 삭제하거나 실제 GA 로직으로 대체합니다.  

2. 📝 프롬프트 최적화
LLM의 응답 품질을 향상시키기 위해 지속적인 프롬프트 엔지니어링이 필요합니다.  

반복적인 테스트 및 개선: prompts/order_parser_prompt.txt, prompts/planner_prompt.txt, prompts/evaluator_prompt.txt 파일의 내용을 main.py 실행 결과를 보면서 계속해서 다듬고 개선해야 합니다.  
구체적인 지시 추가: LLM이 스케줄링 및 평가 시 고려해야 할 추가적인 제약 조건, 우선순위, 예외 처리 규칙 등을 프롬프트에 명확하게 명시합니다.  
예시 다양화: LLM이 다양한 시나리오를 이해하고 적절히 대응할 수 있도록 프롬프트 내의 예시를 더 다양하고 복잡하게 구성할 수 있습니다.  
