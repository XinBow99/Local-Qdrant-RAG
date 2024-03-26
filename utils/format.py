from pydantic import BaseModel, Field
from typing import List, Optional
class Query(BaseModel):
    """
    主要用於接收使用者的查詢
    :param query: 使用者的查詢
    :param similarity_top_k: 查詢的相似度前幾名
    """
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    """
    主要用於回傳給使用者的回應
    :param search_result: 搜尋結果
    :param source: 搜尋結果的來源
    """
    search_result: str 
    source: str