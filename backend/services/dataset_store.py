import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from backend.models.schemas import (
    DatasetOverview, DatasetProfile, VisualizationRecommendation, 
    AIInsight, DataQualityReport, PreprocessingSummary
)
from backend.services.data_loader import DataLoaderService
from backend.services.preprocessor import PreprocessorService
from backend.services.profiler import ProfilerService
from backend.services.recommender import RecommenderService
from backend.services.insights import InsightsEngineService

class DatasetEntry:
    def __init__(
        self,
        dataset_id: str,
        name: str,
        source_type: str,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        recommendations: List[VisualizationRecommendation],
        insights: List[AIInsight],
        file_size_kb: float,
        created_at: str
    ):
        self.dataset_id = dataset_id
        self.name = name
        self.source_type = source_type
        self.original_df = original_df
        self.processed_df = processed_df
        self.profile = profile
        self.recommendations = recommendations
        self.insights = insights
        self.file_size_kb = file_size_kb
        self.created_at = created_at
        self.is_realtime_active = False

class DatasetStore:
    def __init__(self):
        self.datasets: Dict[str, DatasetEntry] = {}

    def register_dataset(
        self,
        name: str,
        original_df: pd.DataFrame,
        source_type: str = "upload",
        file_size_kb: float = 0.0,
        impute_missing: bool = True,
        remove_duplicates: bool = True
    ) -> DatasetEntry:
        dataset_id = str(uuid.uuid4())[:12]
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Preprocess
        processed_df, summary, quality, type_map = PreprocessorService.process_dataset(
            original_df,
            impute_missing=impute_missing,
            remove_duplicates=remove_duplicates
        )

        # 2. Build Profile
        profile = ProfilerService.build_profile(
            dataset_id=dataset_id,
            name=name,
            df=processed_df,
            column_type_map=type_map,
            quality=quality,
            summary=summary
        )

        # 3. Generate Recommendations
        recommendations = RecommenderService.recommend_visualizations(processed_df, profile)

        # 4. Generate Insights
        insights = InsightsEngineService.generate_deterministic_insights(processed_df, profile)

        entry = DatasetEntry(
            dataset_id=dataset_id,
            name=name,
            source_type=source_type,
            original_df=original_df,
            processed_df=processed_df,
            profile=profile,
            recommendations=recommendations,
            insights=insights,
            file_size_kb=file_size_kb,
            created_at=now_str
        )
        self.datasets[dataset_id] = entry
        return entry

    def get_dataset(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> List[DatasetOverview]:
        result = []
        for d in self.datasets.values():
            result.append(DatasetOverview(
                id=d.dataset_id,
                name=d.name,
                source_type=d.source_type,
                row_count=len(d.processed_df),
                column_count=len(d.processed_df.columns),
                quality_score=d.profile.quality.overall_quality_score,
                quality_status=d.profile.quality.status,
                created_at=d.created_at,
                file_size_kb=d.file_size_kb,
                is_realtime_active=d.is_realtime_active
            ))
        # Sort newest first
        result.reverse()
        return result

    def get_table_data(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 25,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False,
        use_original: bool = False
    ) -> Dict[str, Any]:
        entry = self.get_dataset(dataset_id)
        if not entry:
            raise ValueError(f"Dataset {dataset_id} not found.")

        df = entry.original_df if use_original else entry.processed_df
        table_df = df.copy()

        # Global search filter across all string representation of values
        if search and search.strip():
            s = search.strip().lower()
            mask = table_df.astype(str).apply(lambda row: row.str.lower().str.contains(s, regex=False)).any(axis=1)
            table_df = table_df[mask]

        # Sorting
        if sort_by and sort_by in table_df.columns:
            table_df = table_df.sort_values(by=sort_by, ascending=not sort_desc)

        total_rows = len(table_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        sliced = table_df.iloc[start_idx:end_idx].copy()
        
        # Format timestamps / NaN to JSON serializable
        clean_rows = []
        for idx, row in sliced.iterrows():
            row_dict = {"_row_id": int(idx)}
            for col in table_df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (int, float, bool, str)):
                    row_dict[col] = val
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = str(val)
            clean_rows.append(row_dict)

        # Detect columns with missing values in current view
        col_meta = []
        for col in table_df.columns:
            inferred = entry.profile.columns
            m = next((c for c in inferred if c.name == col), None)
            col_meta.append({
                "name": col,
                "type": m.inferred_type if m else "text",
                "missing_pct": m.missing_pct if m else 0.0
            })

        return {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "columns": col_meta,
            "rows": clean_rows
        }

    def delete_dataset(self, dataset_id: str):
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]

dataset_store = DatasetStore()
