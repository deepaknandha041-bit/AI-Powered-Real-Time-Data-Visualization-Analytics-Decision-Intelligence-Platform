import os
import shutil
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Response
from typing import Optional, List, Dict, Any
from backend.config import settings
from backend.models.schemas import (
    DatasetOverview, DatasetProfile, DataQualityReport, 
    PreprocessingSummary, VisualizationRecommendation, 
    AIInsight, CustomChartRequest
)
from backend.services.data_loader import DataLoaderService
from backend.services.dataset_store import dataset_store
from backend.services.recommender import RecommenderService
from backend.services.insights import InsightsEngineService

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("", response_model=List[DatasetOverview])
def list_datasets():
    return dataset_store.list_datasets()

@router.post("/upload", response_model=DatasetOverview)
async def upload_dataset(
    file: UploadFile = File(...),
    impute_missing: bool = Form(True),
    remove_duplicates: bool = Form(True)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported data format. Please upload CSV, XLSX, or JSON. Received: '{ext}'"
        )

    # Save to upload directory safely
    temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest and validate tabular structure
        try:
            df, metadata = DataLoaderService.load_from_file(temp_path, file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"We couldn't identify a usable tabular structure in this file: {str(e)}"
            )

        if len(df) == 0:
            raise HTTPException(status_code=422, detail="The uploaded dataset contains 0 rows.")

        # Register and process
        entry = dataset_store.register_dataset(
            name=file.filename,
            original_df=df,
            source_type="upload",
            file_size_kb=metadata["file_size_kb"],
            impute_missing=impute_missing,
            remove_duplicates=remove_duplicates
        )

        return DatasetOverview(
            id=entry.dataset_id,
            name=entry.name,
            source_type=entry.source_type,
            row_count=len(entry.processed_df),
            column_count=len(entry.processed_df.columns),
            quality_score=entry.profile.quality.overall_quality_score,
            quality_status=entry.profile.quality.status,
            created_at=entry.created_at,
            file_size_kb=entry.file_size_kb,
            is_realtime_active=False
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/load-demo", response_model=DatasetOverview)
def load_demo_dataset(demo_key: str = Query("saas_sales")):
    try:
        df, metadata = DataLoaderService.load_demo(demo_key, settings.DEMO_DIR)
        display_name = f"[DEMO] {metadata['display_name']}"
        entry = dataset_store.register_dataset(
            name=display_name,
            original_df=df,
            source_type="demo",
            file_size_kb=metadata["file_size_kb"],
            impute_missing=True,
            remove_duplicates=True
        )
        return DatasetOverview(
            id=entry.dataset_id,
            name=entry.name,
            source_type=entry.source_type,
            row_count=len(entry.processed_df),
            column_count=len(entry.processed_df.columns),
            quality_score=entry.profile.quality.overall_quality_score,
            quality_status=entry.profile.quality.status,
            created_at=entry.created_at,
            file_size_kb=entry.file_size_kb,
            is_realtime_active=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}", response_model=DatasetOverview)
def get_dataset(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return DatasetOverview(
        id=entry.dataset_id,
        name=entry.name,
        source_type=entry.source_type,
        row_count=len(entry.processed_df),
        column_count=len(entry.processed_df.columns),
        quality_score=entry.profile.quality.overall_quality_score,
        quality_status=entry.profile.quality.status,
        created_at=entry.created_at,
        file_size_kb=entry.file_size_kb,
        is_realtime_active=entry.is_realtime_active
    )

@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return entry.profile

@router.get("/{dataset_id}/quality", response_model=DataQualityReport)
def get_dataset_quality(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return entry.profile.quality

@router.get("/{dataset_id}/preprocessing", response_model=PreprocessingSummary)
def get_dataset_preprocessing(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return entry.profile.preprocessing_summary

@router.get("/{dataset_id}/recommendations", response_model=List[VisualizationRecommendation])
def get_recommendations(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return entry.recommendations

@router.get("/{dataset_id}/insights", response_model=List[AIInsight])
def get_insights(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return entry.insights

@router.post("/{dataset_id}/visualize", response_model=VisualizationRecommendation)
def create_custom_chart(dataset_id: str, req: CustomChartRequest):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        return RecommenderService.generate_custom_chart(entry.processed_df, req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}/table")
def get_table_view(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100),
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = False,
    use_original: bool = False
):
    try:
        return dataset_store.get_table_data(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
            search=search,
            sort_by=sort_by,
            sort_desc=sort_desc,
            use_original=use_original
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{dataset_id}/export-csv")
def export_processed_csv(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    csv_bytes = entry.processed_df.to_csv(index=False).encode('utf-8')
    filename = f"datamind_processed_{entry.name.replace('[DEMO] ', '').replace(' ', '_').lower()}"
    if not filename.endswith('.csv'):
        filename += '.csv'
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    dataset_store.delete_dataset(dataset_id)
    return {"message": "Dataset deleted successfully."}
