import re
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.models.schemas import (
    DatasetProfile, NLQueryResponse, VisualizationRecommendation
)

class QueryEngineService:
    @classmethod
    def _find_best_column(cls, query_text: str, candidate_cols: List[str]) -> Optional[str]:
        q = query_text.lower()
        # Direct exact match
        for col in candidate_cols:
            if col.lower() in q or col.lower().replace("_", " ") in q:
                return col
        # Semantic keyword match
        synonyms = {
            "sale": ["sales", "revenue", "amount", "charge", "income", "turnover"],
            "region": ["region", "country", "territory", "location", "facility"],
            "product": ["product", "plan", "item", "service", "tier", "sku"],
            "customer": ["customer", "client", "user", "segment", "churn"],
            "date": ["date", "time", "day", "month", "year", "timestamp"],
            "cost": ["cost", "spend", "expense", "charges", "marketing"],
            "rating": ["rating", "csat", "score", "satisfaction", "review"],
            "ticket": ["ticket", "support", "issue", "complaint"],
            "price": ["price", "close", "closing", "cost", "value"]
        }
        for sem_key, keywords in synonyms.items():
            if any(k in q for k in keywords):
                for col in candidate_cols:
                    if any(k in col.lower() for k in keywords):
                        return col
        return candidate_cols[0] if candidate_cols else None

    @classmethod
    def process_query(
        cls, 
        df: pd.DataFrame, 
        profile: DatasetProfile, 
        query: str
    ) -> NLQueryResponse:
        q = query.strip().lower()
        num_cols = profile.numerical_columns
        cat_cols = profile.categorical_columns
        date_cols = profile.datetime_columns
        
        # 1. Check for Highest / Max / Top queries
        if any(w in q for w in ["highest", "top", "best", "maximum", "max", "most"]):
            target_cat = cls._find_best_column(q, cat_cols)
            target_num = cls._find_best_column(q, num_cols)

            if target_cat and target_num:
                agg = "sum" if any(k in target_num.lower() for k in ["sale", "revenue", "spend", "unit", "charge", "ticket"]) else "mean"
                grouped = df.groupby(target_cat)[target_num].agg(agg).reset_index()
                grouped = grouped.sort_values(target_num, ascending=False)
                top_row = grouped.iloc[0]
                top_name = str(top_row[target_cat])
                top_val = round(float(top_row[target_num]), 2)
                
                chart_records = grouped.head(8).to_dict(orient='records')
                chart_data = [{"x": str(r[target_cat]), "y": round(float(r[target_num]), 2)} for r in chart_records]
                
                chart = VisualizationRecommendation(
                    id=str(uuid.uuid4())[:8],
                    title=f"Top {target_cat} by {target_num}",
                    chart_type="bar",
                    x_axis=target_cat,
                    y_axis=target_num,
                    aggregation=agg,
                    reason=f"Bar chart ranking {target_cat} by cumulative {target_num}.",
                    relevance_score=95.0,
                    data=chart_data
                )

                return NLQueryResponse(
                    query=query,
                    interpreted_intent=f"Top ranking calculation for '{target_cat}' by '{target_num}' ({agg})",
                    answer=f"The highest-performing {target_cat} is '{top_name}' with a total {target_num} of {top_val:,}.",
                    statistical_evidence={
                        "category": target_cat,
                        "leader": top_name,
                        "metric": target_num,
                        "value": top_val,
                        "rank_1_share": f"{round((top_val / max(1, grouped[target_num].sum())) * 100, 1)}%"
                    },
                    chart_spec=chart,
                    table_slice=grouped.head(10).to_dict(orient='records')
                )

        # 2. Check for Lowest / Minimum queries
        if any(w in q for w in ["lowest", "bottom", "worst", "minimum", "min", "least"]):
            target_cat = cls._find_best_column(q, cat_cols)
            target_num = cls._find_best_column(q, num_cols)

            if target_cat and target_num:
                grouped = df.groupby(target_cat)[target_num].mean().reset_index()
                grouped = grouped.sort_values(target_num, ascending=True)
                low_row = grouped.iloc[0]
                low_name = str(low_row[target_cat])
                low_val = round(float(low_row[target_num]), 2)

                chart_records = grouped.head(8).to_dict(orient='records')
                chart_data = [{"x": str(r[target_cat]), "y": round(float(r[target_num]), 2)} for r in chart_records]

                chart = VisualizationRecommendation(
                    id=str(uuid.uuid4())[:8],
                    title=f"Lowest {target_cat} by {target_num}",
                    chart_type="bar",
                    x_axis=target_cat,
                    y_axis=target_num,
                    aggregation="mean",
                    reason=f"Bar chart ranking lowest {target_cat} by {target_num}.",
                    relevance_score=92.0,
                    data=chart_data
                )

                return NLQueryResponse(
                    query=query,
                    interpreted_intent=f"Bottom ranking calculation for '{target_cat}' by '{target_num}' (mean)",
                    answer=f"The lowest {target_cat} is '{low_name}' with an average {target_num} of {low_val:,}.",
                    statistical_evidence={"category": target_cat, "lowest": low_name, "value": low_val},
                    chart_spec=chart,
                    table_slice=grouped.head(10).to_dict(orient='records')
                )

        # 3. Check for Trend / Time queries
        if any(w in q for w in ["trend", "time", "history", "trajectory", "over time", "longitudinal"]):
            if date_cols and num_cols:
                primary_date = date_cols[0]
                target_num = cls._find_best_column(q, num_cols) or num_cols[0]
                temp_df = df[[primary_date, target_num]].dropna().copy()
                temp_df['dt'] = pd.to_datetime(temp_df[primary_date], errors='coerce')
                temp_df = temp_df.dropna(subset=['dt']).sort_values('dt')
                
                # Resample if lots of points
                if temp_df['dt'].nunique() > 50:
                    temp_df['p'] = temp_df['dt'].dt.strftime('%Y-%m')
                    chart_df = temp_df.groupby('p')[target_num].sum().reset_index()
                else:
                    chart_df = temp_df.groupby(temp_df['dt'].dt.strftime('%Y-%m-%d'))[target_num].sum().reset_index()

                chart_df.columns = ['x', 'y']
                chart_df['y'] = chart_df['y'].round(2)
                records = chart_df.to_dict(orient='records')

                delta_pct = 0.0
                if len(records) > 2:
                    delta_pct = round(((records[-1]['y'] - records[0]['y']) / max(1.0, records[0]['y'])) * 100.0, 1)

                chart = VisualizationRecommendation(
                    id=str(uuid.uuid4())[:8],
                    title=f"{target_num} Chronological Trajectory",
                    chart_type="line",
                    x_axis=primary_date,
                    y_axis=target_num,
                    reason=f"Temporal trend line tracking {target_num} over time.",
                    relevance_score=94.0,
                    data=records
                )

                return NLQueryResponse(
                    query=query,
                    interpreted_intent=f"Temporal longitudinal trajectory of '{target_num}' over '{primary_date}'",
                    answer=f"{target_num} spans {len(records)} periods from {records[0]['x']} to {records[-1]['x']}. Net movement is {'up by ' + str(delta_pct) + '%' if delta_pct > 0 else 'down by ' + str(abs(delta_pct)) + '%'}.",
                    statistical_evidence={"periods": len(records), "net_change": f"{delta_pct}%", "start_val": records[0]['y'], "end_val": records[-1]['y']},
                    chart_spec=chart,
                    table_slice=records[:15]
                )

        # 4. Check for Outliers / Anomalies
        if any(w in q for w in ["outlier", "anomaly", "anomalies", "unusual", "abnormal", "extreme"]):
            target_num = cls._find_best_column(q, num_cols) or (num_cols[0] if num_cols else None)
            if target_num:
                s = pd.to_numeric(df[target_num], errors='coerce').dropna()
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr = q75 - q25
                low_b, high_b = q25 - 1.5 * iqr, q75 + 1.5 * iqr
                outliers_df = df[(df[target_num] < low_b) | (df[target_num] > high_b)]
                outlier_cnt = len(outliers_df)

                # Return boxplot or histogram
                chart_records = [{"x": f"Row {idx}", "y": round(float(v), 2)} for idx, v in outliers_df[target_num].head(10).items()]
                chart = VisualizationRecommendation(
                    id=str(uuid.uuid4())[:8],
                    title=f"Detected Outliers in {target_num}",
                    chart_type="bar",
                    x_axis="Row",
                    y_axis=target_num,
                    reason=f"Values deviating beyond 1.5 IQR boundary ({round(low_b, 1)} to {round(high_b, 1)}).",
                    relevance_score=91.0,
                    data=chart_records
                )

                return NLQueryResponse(
                    query=query,
                    interpreted_intent=f"Outlier detection using IQR bounds for '{target_num}'",
                    answer=f"Identified {outlier_cnt} statistical outliers in '{target_num}' outside the standard range of [{round(low_b, 2)}, {round(high_b, 2)}]. The highest detected value is {round(float(s.max()), 2)}.",
                    statistical_evidence={
                        "outlier_count": outlier_cnt,
                        "lower_boundary": round(low_b, 2),
                        "upper_boundary": round(high_b, 2),
                        "median": round(float(s.median()), 2)
                    },
                    chart_spec=chart,
                    table_slice=outliers_df.head(10).to_dict(orient='records')
                )

        # 5. Check for Correlation / Relationship
        if any(w in q for w in ["correlat", "relation", "versus", "vs", "scatter", "depend"]):
            if len(num_cols) >= 2:
                pair = profile.correlations[0] if profile.correlations else None
                c1 = pair.col1 if pair else num_cols[0]
                c2 = pair.col2 if pair else num_cols[1]
                coeff = pair.coefficient if pair else round(float(df[[c1, c2]].corr().iloc[0, 1]), 3)

                sub = df[[c1, c2]].dropna()
                sampled = sub.sample(n=min(100, len(sub)), random_state=42)
                scatter_data = [{"x": round(float(r[c1]), 2), "y": round(float(r[c2]), 2)} for _, r in sampled.iterrows()]

                chart = VisualizationRecommendation(
                    id=str(uuid.uuid4())[:8],
                    title=f"{c1} vs {c2} Correlation",
                    chart_type="scatter",
                    x_axis=c1,
                    y_axis=c2,
                    reason=f"Scatter plot displaying relationship with Pearson r = {coeff}.",
                    relevance_score=93.0,
                    data=scatter_data
                )

                return NLQueryResponse(
                    query=query,
                    interpreted_intent=f"Bivariate correlation evaluation between '{c1}' and '{c2}'",
                    answer=f"The correlation between '{c1}' and '{c2}' is r = {coeff} ({'Positive' if coeff > 0 else 'Negative'} relationship). Approximately {round(coeff**2 * 100, 1)}% of variance in {c2} is explained by {c1}.",
                    statistical_evidence={"variable_1": c1, "variable_2": c2, "pearson_r": coeff, "r_squared": round(coeff**2, 3)},
                    chart_spec=chart,
                    table_slice=sampled.head(10).to_dict(orient='records')
                )

        # 6. Default / General Breakdown by Category
        target_cat = cls._find_best_column(q, cat_cols) or (cat_cols[0] if cat_cols else None)
        target_num = cls._find_best_column(q, num_cols) or (num_cols[0] if num_cols else None)

        if target_cat and target_num:
            grouped = df.groupby(target_cat)[target_num].sum().reset_index().sort_values(target_num, ascending=False).head(10)
            chart_records = [{"x": str(r[target_cat]), "y": round(float(r[target_num]), 2)} for _, r in grouped.iterrows()]
            
            chart = VisualizationRecommendation(
                id=str(uuid.uuid4())[:8],
                title=f"{target_num} by {target_cat}",
                chart_type="bar",
                x_axis=target_cat,
                y_axis=target_num,
                aggregation="sum",
                reason=f"Breakdown of {target_num} across distinct {target_cat} categories.",
                relevance_score=88.0,
                data=chart_records
            )

            return NLQueryResponse(
                query=query,
                interpreted_intent=f"Segmented breakdown of '{target_num}' by '{target_cat}'",
                answer=f"Analyzed {target_num} aggregated across {len(grouped)} distinct categories of {target_cat}. Leading category is '{grouped.iloc[0][target_cat]}' with {round(float(grouped.iloc[0][target_num]), 2):,}.",
                statistical_evidence={"dimension": target_cat, "measure": target_num, "top_category": str(grouped.iloc[0][target_cat])},
                chart_spec=chart,
                table_slice=grouped.to_dict(orient='records')
            )

        # Fallback summary response
        return NLQueryResponse(
            query=query,
            interpreted_intent="General dataset summary request",
            answer=f"Dataset '{profile.name}' contains {len(df):,} rows and {len(df.columns)} columns ({len(num_cols)} numerical, {len(cat_cols)} categorical, {len(date_cols)} temporal). Data health score is {profile.quality.overall_quality_score}%.",
            statistical_evidence={"rows": len(df), "columns": len(df.columns), "quality_score": profile.quality.overall_quality_score},
            chart_spec=None,
            table_slice=df.head(10).to_dict(orient='records')
        )
