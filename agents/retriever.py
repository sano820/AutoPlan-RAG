from typing import List
from langchain_community.vectorstores import FAISS

class RetrieverAgent:
    def __init__(self, vector_store: FAISS, k: int = 5): # k값 증가
        self.vector_store = vector_store
        self.k = k

    def retrieve_machine_item_capacity_info(self, item_query: str) -> List[str]:
        print(f"  RetrieverAgent: 품목 '{item_query}'에 대한 기계-품목-생산량 정보 검색 요청...")
        try:
            # item_query를 사용하여 FAISS에서 관련 기계-품목-생산량 정보 검색
            # FAISS 인덱스 생성 시 "기계_이름 (생산_아이템) - 시간당_생산량" 형태로 임베딩되어야 함
            docs = self.vector_store.similarity_search(item_query, k=self.k)
            retrieved_info = [doc.page_content for doc in docs]
            if not retrieved_info:
                print(f"  RetrieverAgent: '{item_query}'와 관련된 기계 정보를 찾지 못했습니다.")
            return retrieved_info
        except Exception as e:
            print(f"  RetrieverAgent 오류: 기계-품목-생산량 정보 검색 중 오류 발생 - {e}")
            return []