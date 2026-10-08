import os
import json
import io
import pandas as pd
from typing import Tuple, Dict, Any

class DataLoaderService:
    @staticmethod
    def load_from_file(file_path: str, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        file_size_kb = round(os.path.getsize(file_path) / 1024.0, 2)
        
        try:
            if ext == ".csv":
                # Try UTF-8 first with auto-separator sniffing
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
                except Exception:
                    df = pd.read_csv(file_path, sep=',', encoding='latin1')
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    # Check for records/data key
                    for key in ["data", "records", "items", "results"]:
                        if key in data and isinstance(data[key], list):
                            df = pd.json_normalize(data[key])
                            break
                    else:
                        df = pd.json_normalize([data])
                else:
                    raise ValueError("JSON must contain an array or object of records.")
            else:
                raise ValueError(f"Unsupported file format: {ext}. Please upload CSV, XLSX, or JSON.")
                
            # Clean initial column names (strip trailing/leading spaces)
            df.columns = [str(c).strip() for c in df.columns]
            
            metadata = {
                "filename": filename,
                "file_size_kb": file_size_kb,
                "row_count": len(df),
                "column_count": len(df.columns),
                "format": ext[1:].upper()
            }
            return df, metadata
        except Exception as e:
            raise ValueError(f"Failed to parse dataset '{filename}': {str(e)}")

    @staticmethod
    def load_demo(demo_key: str, demo_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        mapping = {
            "saas_sales": ("demo_saas_sales.csv", "Global SaaS Revenue & Marketing Performance"),
            "customer_churn": ("demo_customer_churn.csv", "Customer Retention & Churn Intelligence"),
            "financial_market": ("demo_financial_market.csv", "Financial Market Asset Dynamics"),
            "iot_telemetry": ("demo_iot_telemetry.csv", "IoT Smart Grid Telemetry")
        }
        if demo_key not in mapping:
            raise ValueError(f"Unknown demo dataset key '{demo_key}'. Available: {list(mapping.keys())}")
            
        fname, display_name = mapping[demo_key]
        path = os.path.join(demo_dir, fname)
        if not os.path.exists(path):
            from backend.demo_data.generate_samples import generate_demo_datasets
            generate_demo_datasets(demo_dir)
            
        df = pd.read_csv(path)
        metadata = {
            "filename": fname,
            "display_name": display_name,
            "file_size_kb": round(os.path.getsize(path) / 1024.0, 2),
            "row_count": len(df),
            "column_count": len(df.columns),
            "format": "CSV (DEMO)"
        }
        return df, metadata
