from pydantic import BaseModel 

class SummarizationRequest(BaseModel): 
    article: str 

class SummarizationResponse(BaseModel): 
    summary: str