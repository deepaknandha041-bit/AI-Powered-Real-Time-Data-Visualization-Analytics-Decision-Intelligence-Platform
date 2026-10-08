import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from backend.models.schemas import PreprocessingSummary, DataQualityReport

class PreprocessorService:
    @staticmethod
    def infer_and_clean_column(series: pd.Series) -> Tuple[pd.Series, str]:
        """
        Attempts to infer and clean a column's data type.
        Returns cleaned series and inferred type: 'numerical', 'datetime', 'categorical', 'boolean', 'text'
        """
        # If boolean already
        if series.dtype == bool or set(series.dropna().unique()).issubset({True, False, 1, 0, 'true', 'false', 'True', 'False'}):
            if set(series.dropna().unique()).issubset({True, False, 'true', 'false', 'True', 'False'}):
                bool_map = {'true': True, 'false': False, 'True': True, 'False': False, True: True, False: False}
                return series.map(bool_map), "boolean"
        
        # If numerical already
        if pd.api.types.is_numeric_dtype(series):
            return series, "numerical"
            
        # Try string cleaning if object dtype
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            # Drop na for inspection
            sample = series.dropna().astype(str).str.strip()
            if len(sample) == 0:
                return series, "categorical"
                
            # Check if strings are numeric with currency symbols or commas (e.g. "$1,234.50" or "12.5%")
            cleaned_num_sample = sample.str.replace(r'[\$,€,£,%]', '', regex=True).str.replace(',', '', regex=False).str.strip()
            num_converted = pd.to_numeric(cleaned_num_sample, errors='coerce')
            if num_converted.notna().sum() / max(1, len(sample)) > 0.85:
                # Successfully numeric
                full_cleaned = series.astype(str).str.replace(r'[\$,€,£,%]', '', regex=True).str.replace(',', '', regex=False).str.strip()
                return pd.to_numeric(full_cleaned, errors='coerce'), "numerical"

            # Check if strings are datetime
            # Look for common date patterns
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',
                r'^\d{2}/\d{2}/\d{4}',
                r'^\d{4}/\d{2}/\d{2}',
                r'^\d{2}-\d{2}-\d{4}'
            ]
            has_date_format = any(sample.str.contains(pat, regex=True).mean() > 0.6 for pat in date_patterns)
            if has_date_format:
                try:
                    dt_series = pd.to_datetime(series, errors='coerce')
                    if dt_series.notna().sum() / max(1, len(sample)) > 0.70:
                        return dt_series, "datetime"
                except Exception:
                    pass

            # Check cardinality for Categorical vs Text
            unique_count = series.nunique(dropna=True)
            total_count = len(sample)
            cardinality_ratio = unique_count / max(1, total_count)
            avg_char_length = sample.str.len().mean()

            if unique_count <= 50 or cardinality_ratio < 0.20 or avg_char_length < 35:
                return series.astype(str).str.strip(), "categorical"
            else:
                return series.astype(str).str.strip(), "text"

        return series, "categorical"

    @classmethod
    def process_dataset(
        cls, 
        df_original: pd.DataFrame, 
        impute_missing: bool = True, 
        remove_duplicates: bool = True,
        remove_empty_cols: bool = True
    ) -> Tuple[pd.DataFrame, PreprocessingSummary, DataQualityReport, Dict[str, str]]:
        """
        Executes complete preprocessing pipeline while preserving df_original untouched.
        Returns:
            processed_df, PreprocessingSummary, DataQualityReport, column_type_map
        """
        df = df_original.copy()
        actions = []
        original_rows, original_cols = len(df), len(df.columns)
        
        # 1. Whitespace strip on column names
        df.columns = [str(c).strip() for c in df.columns]
        actions.append("Trimmed column header whitespace")

        # 2. Identify empty and constant columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        
        if remove_empty_cols and empty_cols:
            df.drop(columns=empty_cols, inplace=True)
            actions.append(f"Removed {len(empty_cols)} completely empty columns: {empty_cols}")

        # 3. Detect duplicate rows
        duplicate_count = int(df.duplicated().sum())
        duplicates_removed = 0
        if remove_duplicates and duplicate_count > 0:
            df.drop_duplicates(inplace=True)
            duplicates_removed = duplicate_count
            actions.append(f"Removed {duplicate_count} duplicate rows")

        # 4. Type Inference and Type Casting
        column_type_map: Dict[str, str] = {}
        for col in df.columns:
            cleaned_series, inferred_type = cls.infer_and_clean_column(df[col])
            df[col] = cleaned_series
            column_type_map[col] = inferred_type
        actions.append("Performed automated data type inference and normalization")

        # 5. Missing value detection & handling
        missing_count_total = int(df.isna().sum().sum())
        missing_handled = 0
        if impute_missing and missing_count_total > 0:
            for col in df.columns:
                if df[col].isna().any():
                    t = column_type_map[col]
                    if t == "numerical":
                        med = df[col].median()
                        if pd.notna(med):
                            df[col] = df[col].fillna(med)
                            missing_handled += int(df_original[col].isna().sum())
                    elif t == "categorical":
                        mode = df[col].mode()
                        fill_val = mode[0] if len(mode) > 0 else "Unknown"
                        df[col] = df[col].fillna(fill_val)
                        missing_handled += int(df_original[col].isna().sum())
            actions.append(f"Imputed missing numerical values with median and categorical with mode")

        # 6. Outlier detection across numerical columns
        outlier_cells_total = 0
        outlier_rows_set = set()
        for col, t in column_type_map.items():
            if t == "numerical":
                nums = df[col].dropna()
                if len(nums) > 10:
                    q25 = nums.quantile(0.25)
                    q75 = nums.quantile(0.75)
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                        outlier_cells_total += len(col_outliers)
                        outlier_rows_set.update(col_outliers)

        actions.append(f"Scanned {len(column_type_map)} columns for anomalous outliers using IQR methodology")

        # 7. Quality Calculation
        total_cells = original_rows * original_cols if original_rows * original_cols > 0 else 1
        missing_original = int(df_original.isna().sum().sum())
        completeness_pct = max(0.0, round((1.0 - (missing_original / total_cells)) * 100.0, 1))
        
        dup_pct = round((duplicate_count / max(1, original_rows)) * 100.0, 1)
        uniqueness_pct = max(0.0, round(100.0 - dup_pct, 1))
        
        outlier_ratio = outlier_cells_total / max(1, original_rows)
        validity_pct = max(0.0, round(min(100.0, 100.0 - (outlier_ratio * 25.0)), 1))

        # Overall composite quality score
        overall_score = round(completeness_pct * 0.45 + uniqueness_pct * 0.25 + validity_pct * 0.30, 1)
        if overall_score >= 85.0:
            status = "GOOD"
        elif overall_score >= 65.0:
            status = "WARNING"
        else:
            status = "CRITICAL"

        issues = []
        if missing_original > 0:
            issues.append({"type": "missing_values", "level": "warning" if completeness_pct > 90 else "critical", "message": f"{missing_original} missing values ({round((missing_original/total_cells)*100, 2)}% of data)."})
        if duplicate_count > 0:
            issues.append({"type": "duplicates", "level": "info", "message": f"{duplicate_count} duplicate rows ({dup_pct}%) detected."})
        if outlier_cells_total > 0:
            issues.append({"type": "outliers", "level": "warning", "message": f"{outlier_cells_total} statistical outlier values detected across numerical metrics."})
        if empty_cols:
            issues.append({"type": "empty_columns", "level": "critical", "message": f"Empty columns identified: {', '.join(empty_cols)}."})
        if constant_cols:
            issues.append({"type": "constant_columns", "level": "info", "message": f"Zero-variance constant columns identified: {', '.join(constant_cols)}."})

        quality_report = DataQualityReport(
            total_rows=original_rows,
            total_columns=original_cols,
            completeness_score=completeness_pct,
            validity_score=validity_pct,
            uniqueness_score=uniqueness_pct,
            overall_quality_score=overall_score,
            status=status,
            missing_values_count=missing_original,
            missing_values_pct=round((missing_original / total_cells) * 100.0, 2),
            duplicate_rows_count=duplicate_count,
            duplicate_rows_pct=dup_pct,
            outlier_cells_count=outlier_cells_total,
            outlier_rows_count=len(outlier_rows_set),
            constant_columns=constant_cols,
            empty_columns=empty_cols,
            issues=issues
        )

        summary = PreprocessingSummary(
            original_rows=original_rows,
            original_cols=original_cols,
            processed_rows=len(df),
            processed_cols=len(df.columns),
            actions_performed=actions,
            missing_values_handled=missing_handled,
            duplicates_removed=duplicates_removed,
            columns_type_cast={c: column_type_map[c] for c in df.columns},
            outliers_detected=outlier_cells_total
        )

        return df, summary, quality_report, column_type_map
