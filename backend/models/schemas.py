from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ColumnProfile(BaseModel):
    name: str
    inferred_type: str  # "numerical", "categorical", "datetime", "boolean", "text"
    raw_type: str
    non_null_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_ratio: float
    sample_values: List[Any] = []
    is_identifier: bool = False
    is_constant: bool = False
    # Numerical
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    skewness: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    iqr: Optional[float] = None
    outlier_count: Optional[int] = None
    # Categorical
    top_categories: Optional[List[Dict[str, Any]]] = None
    # Datetime
    min_date: Optional[str] = None
    max_date: Optional[str] = None

class DataQualityReport(BaseModel):
    total_rows: int
    total_columns: int
    completeness_score: float
    validity_score: float
    uniqueness_score: float
    overall_quality_score: float
    status: str  # "GOOD", "WARNING", "CRITICAL"
    missing_values_count: int
    missing_values_pct: float
    duplicate_rows_count: int
    duplicate_rows_pct: float
    outlier_cells_count: int
    outlier_rows_count: int
    constant_columns: List[str] = []
    empty_columns: List[str] = []
    issues: List[Dict[str, Any]] = []

class PreprocessingSummary(BaseModel):
    original_rows: int
    original_cols: int
    processed_rows: int
    processed_cols: int
    actions_performed: List[str] = []
    missing_values_handled: int = 0
    duplicates_removed: int = 0
    columns_type_cast: Dict[str, str] = {}
    outliers_detected: int = 0

class CorrelationPair(BaseModel):
    col1: str
    col2: str
    coefficient: float
    abs_coefficient: float
    significance: str  # "Strong Positive", "Moderate Positive", "Strong Negative", etc.

class DatasetProfile(BaseModel):
    dataset_id: str
    name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    numerical_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    text_columns: List[str]
    correlations: List[CorrelationPair] = []
    time_range: Optional[Dict[str, str]] = None
    quality: DataQualityReport
    preprocessing_summary: PreprocessingSummary

class VisualizationRecommendation(BaseModel):
    id: str
    title: str
    chart_type: str  # line, bar, grouped_bar, stacked_bar, area, scatter, histogram, pie, donut, heatmap, box_plot
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    secondary_y: Optional[List[str]] = None
    group_by: Optional[str] = None
    aggregation: Optional[str] = None  # sum, avg, count, min, max, none
    reason: str
    relevance_score: float  # Genuinely calculated 0-100
    data: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class AIInsight(BaseModel):
    id: str
    category: str  # trend, anomaly, correlation, dominance, distribution, summary
    title: str
    description: str
    severity: str  # info, warning, critical, positive
    confidence_metric: str  # e.g., "R² = 0.92", "Z = 3.6 (p < 0.001)", "Delta = +34.2%"
    impact_score: float  # 0 to 100
    related_columns: List[str] = []
    source: str = "Deterministic Statistical Engine"

class DatasetOverview(BaseModel):
    id: str
    name: str
    source_type: str  # "upload", "demo", "stream"
    row_count: int
    column_count: int
    quality_score: float
    quality_status: str
    created_at: str
    file_size_kb: float
    is_realtime_active: bool = False

class NLQueryRequest(BaseModel):
    query: str
    dataset_id: str

class NLQueryResponse(BaseModel):
    query: str
    interpreted_intent: str
    answer: str
    statistical_evidence: Dict[str, Any] = {}
    chart_spec: Optional[VisualizationRecommendation] = None
    table_slice: Optional[List[Dict[str, Any]]] = None

class CustomChartRequest(BaseModel):
    dataset_id: str
    chart_type: str
    x_axis: str
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    aggregation: Optional[str] = "sum"  # sum, avg, count, min, max, none
    limit: Optional[int] = 50
    sort_by: Optional[str] = "y_desc"  # x_asc, x_desc, y_asc, y_desc
