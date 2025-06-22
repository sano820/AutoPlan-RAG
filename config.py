# config.py
import os
# from dotenv import load_dotenv # 여기서 호출하지 않고, 애플리케이션 진입점에서 호출

# .env 파일에서 환경 변수 로드 (main.py 또는 앱 시작 시)
# load_dotenv()

# Google Gemini API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일에 설정하거나 직접 할당해주세요.")

# LangChain LLM 모델 이름
LLM_MODEL_NAME = "gemini-2.0-flash-lite"

# FAISS 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retriever 에이전트가 검색할 문서의 수
RETRIEVER_K = 5 # 더 많은 관련 기계를 찾도록 k값 증가

# 기타 상수 (예: 기본 납기 일수 등)
DEFAULT_DUE_DATE_OFFSET_DAYS = 7

# 데이터 파일 경로
MACHINE_INFO_CSV = "data/raw/machine_info.csv"
# FAISS 인덱스 저장/로드 경로 (선택 사항)
FAISS_INDEX_PATH = "data/faiss_index"
FAISS_INDEX_NAME = "machines_items_capacity"