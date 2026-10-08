import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from backend.models.schemas import (
    ColumnProfile, CorrelationPair, DatasetProfile, 
    DataQualityReport, PreprocessingSummary
)

class ProfilerService:
    @staticmethod
    def profile_column(series: pd.Series, col_name: str, col_type: str) -> ColumnProfile:
        total = len(series)
        non_null = int(series.notna().sum())
        missing = total - non_null
        missing_pct = round((missing / max(1, total)) * 100.0, 2)
        unique_cnt = int(series.nunique(dropna=True))
        cardinality_ratio = round(unique_cnt / max(1, non_null), 4)
        sample_vals = [x if pd.notna(x) else None for x in series.dropna().head(5).tolist()]

        # Convert sample values to serializable types
        clean_samples = []
        for v in sample_vals:
            if isinstance(v, (np.integer, int)):
                clean_samples.append(int(v))
            elif isinstance(v, (np.floating, float)):
                clean_samples.append(round(float(v), 2))
            elif isinstance(v, pd.Timestamp):
                clean_samples.append(v.isoformat())
            else:
                clean_samples.append(str(v))

        is_id = (unique_cnt == non_null and non_null > 20 and col_type in ["numerical", "categorical", "text"])
        is_const = (unique_cnt <= 1)

        profile = ColumnProfile(
            name=col_name,
            inferred_type=col_type,
            raw_type=str(series.dtype),
            non_null_count=non_null,
            missing_count=missing,
            missing_pct=missing_pct,
            unique_count=unique_cnt,
            cardinality_ratio=cardinality_ratio,
            sample_values=clean_samples,
            is_identifier=is_id,
            is_constant=is_const
        )

        if col_type == "numerical" and non_null > 0:
            nums = series.dropna().astype(float)
            profile.min_val = round(float(nums.min()), 2)
            profile.max_val = round(float(nums.max()), 2)
            profile.mean_val = round(float(nums.mean()), 2)
            profile.median_val = round(float(nums.median()), 2)
            profile.std_val = round(float(nums.std()), 2) if len(nums) > 1 else 0.0
            profile.skewness = round(float(nums.skew()), 2) if len(nums) > 2 else 0.0
            
            q25 = float(nums.quantile(0.25))
            q75 = float(nums.quantile(0.75))
            iqr = q75 - q25
            profile.q25 = round(q25, 2)
            profile.q75 = round(q75, 2)
            profile.iqr = round(iqr, 2)
            
            if iqr > 0:
                low = q25 - 1.5 * iqr
                high = q75 + 1.5 * iqr
                profile.outlier_count = int(((nums < low) | (nums > high)).sum())
            else:
                profile.outlier_count = 0

        elif col_type == "categorical":
            val_counts = series.dropna().value_counts().head(8)
            top_cats = []
            for cat, count in val_counts.items():
                top_cats.append({
                    "category": str(cat),
                    "count": int(count),
                    "percentage": round((count / max(1, non_null)) * 100.0, 1)
                })
            profile.top_categories = top_cats

        elif col_type == "datetime" and non_null > 0:
            dts = pd.to_datetime(series.dropna(), errors='coerce')
            dts_valid = dts.dropna()
            if len(dts_valid) > 0:
                profile.min_date = dts_valid.min().strftime("%Y-%m-%d")
                profile.max_date = dts_valid.max().strftime("%Y-%m-%d")

        return profile

    @classmethod
    def compute_correlations(cls, df: pd.DataFrame, num_cols: List[str]) -> List[CorrelationPair]:
        if len(num_cols) < 2:
            return []
            
        corr_matrix = df[num_cols].corr(method='pearson')
        pairs = []
        seen = set()

        for c1 in num_cols:
            for c2 in num_cols:
                if c1 != c2 and (c2, c1) not in seen:
                    seen.add((c1, c2))
                    val = corr_matrix.loc[c1, c2]
                    if pd.notna(val):
                        coeff = round(float(val), 3)
                        abs_c = abs(coeff)
                        if abs_c >= 0.7:
                            sig = "Strong Positive" if coeff > 0 else "Strong Negative"
                        elif abs_c >= 0.4:
                            sig = "Moderate Positive" if coeff > 0 else "Moderate Negative"
                        elif abs_c >= 0.2:
                            sig = "Weak Positive" if coeff > 0 else "Weak Negative"
                        else:
                            sig = "Negligible Correlation"
                            
                        pairs.append(CorrelationPair(
                            col1=c1,
                            col2=c2,
                            coefficient=coeff,
                            abs_coefficient=abs_c,
                            significance=sig
                        ))

        # Sort descending by absolute correlation strength
        pairs.sort(key=lambda x: x.abs_coefficient, reverse=True)
        return pairs

    @classmethod
    def build_profile(
        cls,
        dataset_id: str,
        name: str,
        df: pd.DataFrame,
        column_type_map: Dict[str, str],
        quality: DataQualityReport,
        summary: PreprocessingSummary
    ) -> DatasetProfile:
        columns_profile = []
        num_cols = []
        cat_cols = []
        date_cols = []
        bool_cols = []
        text_cols = []

        for col, col_type in column_type_map.items():
            col_prof = cls.profile_column(df[col], col, col_type)
            columns_profile.append(col_prof)
            if col_type == "numerical":
                num_cols.append(col)
            elif col_type == "categorical":
                cat_cols.append(col)
            elif col_type == "datetime":
                date_cols.append(col)
            elif col_type == "boolean":
                bool_cols.append(col)
            elif col_type == "text":
                text_cols.append(col)

        correlations = cls.compute_correlations(df, num_cols)

        time_range = None
        if date_cols:
            primary_date = date_cols[0]
            dts = pd.to_datetime(df[primary_date].dropna(), errors='coerce').dropna()
            if len(dts) > 0:
                time_range = {
                    "column": primary_date,
                    "start": dts.min().strftime("%Y-%m-%d"),
                    "end": dts.max().strftime("%Y-%m-%d"),
                    "days_span": str((dts.max() - dts.min()).days)
                }

        return DatasetProfile(
            dataset_id=dataset_id,
            name=name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns_profile,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            datetime_columns=date_cols,
            boolean_columns=bool_cols,
            text_columns=text_cols,
            correlations=correlations,
            time_range=time_range,
            quality=quality,
            preprocessing_summary=summary
        )
