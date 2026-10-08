from fastapi import APIRouter, HTTPException
from backend.models.schemas import NLQueryRequest, NLQueryResponse
from backend.services.dataset_store import dataset_store
from backend.services.query_engine import QueryEngineService

router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=NLQueryResponse)
def execute_query(req: NLQueryRequest):
    entry = dataset_store.get_dataset(req.dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        response = QueryEngineService.process_query(
            df=entry.processed_df,
            profile=entry.profile,
            query=req.query
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")
