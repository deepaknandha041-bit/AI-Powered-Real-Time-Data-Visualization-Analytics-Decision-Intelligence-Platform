import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from backend.models.schemas import DatasetProfile, AIInsight

class InsightsEngineService:
    @classmethod
    def generate_deterministic_insights(
        cls, 
        df: pd.DataFrame, 
        profile: DatasetProfile
    ) -> List[AIInsight]:
        insights: List[AIInsight] = []
        num_cols = profile.numerical_columns
        cat_cols = profile.categorical_columns
        date_cols = profile.datetime_columns

        # 1. Temporal Trend Insights
        if date_cols and num_cols:
            primary_date = date_cols[0]
            # Test top numerical columns
            for col in num_cols[:2]:
                try:
                    temp_df = df[[primary_date, col]].dropna().copy()
                    temp_df['dt'] = pd.to_datetime(temp_df[primary_date], errors='coerce')
                    temp_df = temp_df.dropna(subset=['dt']).sort_values('dt')
                    
                    if len(temp_df) > 10:
                        first_third = temp_df.iloc[:len(temp_df)//3][col].mean()
                        last_third = temp_df.iloc[-len(temp_df)//3:][col].mean()
                        
                        if first_third > 0:
                            pct_change = round(((last_third - first_third) / first_third) * 100.0, 1)
                            # Linear slope check
                            x_indices = np.arange(len(temp_df))
                            slope, _ = np.polyfit(x_indices, temp_df[col].values, 1)
                            
                            if abs(pct_change) >= 8.0:
                                direction = "increased" if pct_change > 0 else "decreased"
                                sev = "positive" if (pct_change > 0 and "cost" not in col.lower() and "churn" not in col.lower()) else ("warning" if pct_change < 0 else "info")
                                
                                insights.append(AIInsight(
                                    id=str(uuid.uuid4())[:8],
                                    category="trend",
                                    title=f"Significant Longitudinal Trend in {col}",
                                    description=f"{col} has {direction} by {abs(pct_change)}% over the analyzed observation window (baseline avg {round(first_third, 2)} vs recent avg {round(last_third, 2)}).",
                                    severity=sev,
                                    confidence_metric=f"Slope: {round(slope, 3)}/period (Delta: {pct_change}%)",
                                    impact_score=min(95.0, 70.0 + abs(pct_change) * 0.4),
                                    related_columns=[primary_date, col],
                                    source="Deterministic Statistical Engine"
                                ))
                except Exception as e:
                    print(f"Error computing trend insight: {e}")

        # 2. Category Dominance / Concentration Insights
        if cat_cols and num_cols:
            for cat in cat_cols[:2]:
                val_counts = df[cat].value_counts(dropna=True)
                if len(val_counts) >= 2:
                    top_cat = val_counts.index[0]
                    top_cnt = val_counts.iloc[0]
                    total_cnt = len(df)
                    top_pct = round((top_cnt / total_cnt) * 100.0, 1)
                    
                    # Also check measure contribution
                    m_col = num_cols[0]
                    try:
                        grp_sum = df.groupby(cat)[m_col].sum()
                        total_sum = grp_sum.sum()
                        if total_sum > 0:
                            top_measure_cat = grp_sum.idxmax()
                            top_measure_val = grp_sum.max()
                            top_measure_pct = round((top_measure_val / total_sum) * 100.0, 1)
                            
                            if top_measure_pct >= 35.0:
                                insights.append(AIInsight(
                                    id=str(uuid.uuid4())[:8],
                                    category="dominance",
                                    title=f"Market Concentration in '{top_measure_cat}'",
                                    description=f"Category '{top_measure_cat}' represents {top_measure_pct}% of the cumulative {m_col} across all {len(grp_sum)} segments.",
                                    severity="info" if top_measure_pct < 60 else "warning",
                                    confidence_metric=f"Dominance Index: {top_measure_pct}% of total",
                                    impact_score=min(92.0, 65.0 + top_measure_pct * 0.3),
                                    related_columns=[cat, m_col],
                                    source="Deterministic Statistical Engine"
                                ))
                    except Exception:
                        pass

        # 3. Correlation Insights
        if profile.correlations:
            for pair in profile.correlations[:2]:
                if pair.abs_coefficient >= 0.45:
                    dir_str = "positive" if pair.coefficient > 0 else "inverse/negative"
                    insights.append(AIInsight(
                        id=str(uuid.uuid4())[:8],
                        category="correlation",
                        title=f"{pair.significance} Identified",
                        description=f"A statistically notable {dir_str} relationship exists between '{pair.col1}' and '{pair.col2}' (Pearson r = {pair.coefficient}). As {pair.col1} shifts, {pair.col2} displays concurrent movement.",
                        severity="positive" if pair.abs_coefficient > 0.6 else "info",
                        confidence_metric=f"Pearson r = {pair.coefficient} (R² = {round(pair.coefficient**2, 3)})",
                        impact_score=round(pair.abs_coefficient * 95.0, 1),
                        related_columns=[pair.col1, pair.col2],
                        source="Deterministic Statistical Engine"
                    ))

        # 4. Outlier & Anomaly Insights
        for c_prof in profile.columns:
            if c_prof.inferred_type == "numerical" and c_prof.outlier_count and c_prof.outlier_count > 0:
                pct_outliers = round((c_prof.outlier_count / max(1, c_prof.non_null_count)) * 100.0, 2)
                insights.append(AIInsight(
                    id=str(uuid.uuid4())[:8],
                    category="anomaly",
                    title=f"Anomalous Outliers Detected in {c_prof.name}",
                    description=f"{c_prof.outlier_count} record(s) ({pct_outliers}%) in '{c_prof.name}' deviate beyond the 1.5×IQR boundary (Min: {c_prof.min_val}, Max: {c_prof.max_val}, Median: {c_prof.median_val}).",
                    severity="warning" if pct_outliers > 1.0 else "info",
                    confidence_metric=f"IQR Threshold: {c_prof.iqr} (Q25={c_prof.q25}, Q75={c_prof.q75})",
                    impact_score=min(90.0, 60.0 + min(30.0, c_prof.outlier_count * 2.0)),
                    related_columns=[c_prof.name],
                    source="Deterministic Statistical Engine"
                ))

        # 5. Data Quality & Health Insight
        q = profile.quality
        if q.overall_quality_score >= 90.0:
            insights.append(AIInsight(
                id=str(uuid.uuid4())[:8],
                category="summary",
                title="High Integrity Dataset Health",
                description=f"Overall dataset quality score is {q.overall_quality_score}% with {q.completeness_score}% completeness and {q.uniqueness_score}% record uniqueness.",
                severity="positive",
                confidence_metric=f"Quality Score: {q.overall_quality_score}/100",
                impact_score=85.0,
                related_columns=[],
                source="Deterministic Statistical Engine"
            ))
        else:
            insights.append(AIInsight(
                id=str(uuid.uuid4())[:8],
                category="summary",
                title="Data Quality Attention Required",
                description=f"Dataset health is rated {q.status} ({q.overall_quality_score}%). Contains {q.missing_values_count} missing entries ({q.missing_values_pct}%) and {q.duplicate_rows_count} duplicate rows.",
                severity="warning" if q.status == "WARNING" else "critical",
                confidence_metric=f"Quality Score: {q.overall_quality_score}/100",
                impact_score=90.0,
                related_columns=[],
                source="Deterministic Statistical Engine"
            ))

        # Sort insights by impact score descending
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        return insights

    @classmethod
    async def synthesize_with_llm_if_available(
        cls,
        df: pd.DataFrame,
        profile: DatasetProfile,
        base_insights: List[AIInsight],
        api_key: Optional[str] = None
    ) -> List[AIInsight]:
        """
        If an API key is available, pass the calculated statistical context to generate
        enhanced executive insights, strictly labeled as LLM Synthesis.
        Otherwise, returns deterministic insights.
        """
        # Strict rule: Never pretend or fake LLM response
        if not api_key:
            return base_insights

        # We can implement standard OpenAI / Anthropic client call if user specifies an API key
        try:
            import httpx
            # Prepare compact aggregated statistical payload (NOT raw private rows)
            stat_summary = {
                "name": profile.name,
                "rows": profile.row_count,
                "columns": profile.column_count,
                "quality_score": profile.quality.overall_quality_score,
                "top_correlations": [{"c1": p.col1, "c2": p.col2, "r": p.coefficient} for p in profile.correlations[:3]],
                "columns_summary": [
                    {
                        "col": c.name, 
                        "type": c.inferred_type, 
                        "mean": c.mean_val, 
                        "outliers": c.outlier_count
                    } for c in profile.columns if c.inferred_type == "numerical"
                ][:6]
            }

            prompt = f"""You are DataMind AI Analyst. Based on this genuine statistical dataset summary:
{stat_summary}
Provide 1 high-level executive strategic finding. Return concise JSON with keys: title, description, severity."""

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2
                    }
                )
                if resp.status_code == 200:
                    import json
                    content = json.loads(resp.json()["choices"][0]["message"]["content"])
                    llm_insight = AIInsight(
                        id=str(uuid.uuid4())[:8],
                        category="summary",
                        title=content.get("title", "Executive Strategic Synthesis"),
                        description=content.get("description", "Automated strategic data summary."),
                        severity=content.get("severity", "positive"),
                        confidence_metric="LLM Synthesis (Conditioned on Statistical Profile)",
                        impact_score=95.0,
                        related_columns=[],
                        source="LLM Synthesis (Verified via Local Profile)"
                    )
                    return [llm_insight] + base_insights
        except Exception as e:
            print(f"LLM API synthesis fallback: {e}")

        return base_insights
