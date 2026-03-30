from pydantic import BaseModel
from typing import List, Optional

class AgentState(BaseModel):
    user_query: str
    plan: Optional[List[str]] = None
    research: Optional[str] = None
    result: Optional[str] = None
