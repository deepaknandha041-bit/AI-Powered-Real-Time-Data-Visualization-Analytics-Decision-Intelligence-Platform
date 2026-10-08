import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import numpy as np
import pandas as pd
from starlette.websockets import WebSocket

class StreamManagerService:
    def __init__(self):
        # dataset_id -> set of WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # dataset_id -> asyncio.Task
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # dataset_id -> rolling records
        self.rolling_data: Dict[str, List[Dict[str, Any]]] = {}
        # dataset_id -> metric baseline stats (mean, std)
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
        # dataset_id -> total packets sent
        self.packet_counts: Dict[str, int] = {}
        self.is_demo_streaming: Dict[str, bool] = {}

    async def connect(self, dataset_id: str, websocket: WebSocket):
        await websocket.accept()
        if dataset_id not in self.active_connections:
            self.active_connections[dataset_id] = set()
        self.active_connections[dataset_id].add(websocket)
        print(f"WebSocket client connected to dataset '{dataset_id}'. Total clients: {len(self.active_connections[dataset_id])}")

    def disconnect(self, dataset_id: str, websocket: WebSocket):
        if dataset_id in self.active_connections:
            self.active_connections[dataset_id].discard(websocket)
            if len(self.active_connections[dataset_id]) == 0:
                del self.active_connections[dataset_id]
        print(f"WebSocket client disconnected from dataset '{dataset_id}'.")

    async def broadcast(self, dataset_id: str, message: Dict[str, Any]):
        if dataset_id in self.active_connections:
            dead_sockets = set()
            for connection in self.active_connections[dataset_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_sockets.add(connection)
            for dead in dead_sockets:
                self.active_connections[dataset_id].discard(dead)

    def initialize_stream(self, dataset_id: str, initial_df: pd.DataFrame):
        self.rolling_data[dataset_id] = initial_df.tail(100).to_dict(orient='records')
        self.packet_counts[dataset_id] = 0
        
        # Calculate baseline statistics for numerical columns
        num_cols = initial_df.select_dtypes(include=[np.number]).columns.tolist()
        baselines = {}
        for c in num_cols:
            s = initial_df[c].dropna()
            if len(s) > 2:
                baselines[c] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()) if s.std() > 0 else 1.0,
                    "min": float(s.min()),
                    "max": float(s.max())
                }
        self.baseline_stats[dataset_id] = baselines

    async def ingest_record(self, dataset_id: str, record: Dict[str, Any], source_label: str = "Live Feed"):
        """
        Validates, checks anomalies, updates metrics, and broadcasts.
        """
        now_iso = datetime.utcnow().isoformat()
        
        # Check anomaly using Z-score against baseline
        anomalies = []
        baselines = self.baseline_stats.get(dataset_id, {})
        for col, val in record.items():
            if col in baselines and isinstance(val, (int, float)):
                base = baselines[col]
                z_score = (float(val) - base["mean"]) / (base["std"] if base["std"] > 0 else 1.0)
                if abs(z_score) >= 2.8:
                    anomalies.append({
                        "column": col,
                        "value": round(float(val), 2),
                        "z_score": round(float(z_score), 2),
                        "baseline_mean": round(base["mean"], 2),
                        "severity": "CRITICAL" if abs(z_score) > 3.5 else "WARNING",
                        "message": f"Anomalous spike in {col} (Z={round(z_score, 2)})"
                    })

        # Append to rolling buffer
        if dataset_id not in self.rolling_data:
            self.rolling_data[dataset_id] = []
        self.rolling_data[dataset_id].append(record)
        if len(self.rolling_data[dataset_id]) > 300:
            self.rolling_data[dataset_id].pop(0)

        self.packet_counts[dataset_id] = self.packet_counts.get(dataset_id, 0) + 1

        # Calculate current rolling stats for top numerical metric
        rolling_metrics = {}
        if self.rolling_data[dataset_id]:
            recent_df = pd.DataFrame(self.rolling_data[dataset_id])
            for col in list(baselines.keys())[:3]:
                if col in recent_df.columns:
                    s = pd.to_numeric(recent_df[col], errors='coerce').dropna()
                    if len(s) > 0:
                        rolling_metrics[col] = {
                            "current": round(float(record.get(col, s.iloc[-1])), 2),
                            "rolling_mean": round(float(s.mean()), 2),
                            "rolling_max": round(float(s.max()), 2),
                            "rolling_min": round(float(s.min()), 2),
                            "count": len(s)
                        }

        packet = {
            "type": "live_data_update",
            "dataset_id": dataset_id,
            "timestamp": now_iso,
            "record": record,
            "anomalies": anomalies,
            "rolling_metrics": rolling_metrics,
            "packet_number": self.packet_counts[dataset_id],
            "stream_source": source_label
        }

        await self.broadcast(dataset_id, packet)
        return packet

    async def start_demo_stream(self, dataset_id: str, df: pd.DataFrame, interval_sec: float = 1.5):
        """
        Runs a deterministic demo stream generating new data points based on dataset schema.
        """
        if self.is_demo_streaming.get(dataset_id, False):
            return

        self.is_demo_streaming[dataset_id] = True
        self.initialize_stream(dataset_id, df)

        async def stream_loop():
            step = 0
            while self.is_demo_streaming.get(dataset_id, False):
                await asyncio.sleep(interval_sec)
                step += 1
                
                # Generate new synthetic point that mimics distribution
                new_record = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if "time" in col_lower or "date" in col_lower:
                        new_record[col] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        base_mean = float(df[col].mean())
                        base_std = float(df[col].std()) if df[col].std() > 0 else 1.0
                        
                        # Occasionally introduce a realistic anomaly burst (every ~18 steps)
                        is_spike = (step % 18 == 0)
                        if is_spike:
                            val = base_mean + base_std * random.choice([3.2, 3.8, -3.1])
                        else:
                            val = np.random.normal(base_mean, base_std * 0.3)
                        
                        if pd.api.types.is_integer_dtype(df[col]):
                            new_record[col] = int(max(0, round(val)))
                        else:
                            new_record[col] = round(float(val), 2)
                    else:
                        # Categorical: pick random existing choice
                        vals = df[col].dropna().unique().tolist()
                        new_record[col] = str(random.choice(vals)) if vals else "Sample"

                await self.ingest_record(dataset_id, new_record, source_label="Demo Data Stream")

        task = asyncio.create_task(stream_loop())
        self.running_tasks[dataset_id] = task

    def stop_demo_stream(self, dataset_id: str):
        self.is_demo_streaming[dataset_id] = False
        if dataset_id in self.running_tasks:
            self.running_tasks[dataset_id].cancel()
            del self.running_tasks[dataset_id]

stream_manager = StreamManagerService()
