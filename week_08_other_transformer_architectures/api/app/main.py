from fastapi import FastAPI
from app.schemas import SummarizationRequest, SummarizationResponse
from app.pipeline import SummarizerPipeline

app = FastAPI(title = 'Pegasus Summarization API', version = '1.0')

# Load once at startup
pipeline = SummarizerPipeline()

@app.post('/summarize', response_model = SummarizationResponse)
def summarize(request: SummarizationRequest):
    summary = pipeline.summarize(request.article)
    return SummarizationResponse(summary = summary)