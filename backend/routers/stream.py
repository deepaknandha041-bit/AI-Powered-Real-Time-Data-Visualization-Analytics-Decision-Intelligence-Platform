from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any
from backend.services.dataset_store import dataset_store
from backend.services.stream_manager import stream_manager

router = APIRouter(tags=["stream"])

@router.websocket("/ws/datasets/{dataset_id}")
async def websocket_endpoint(websocket: WebSocket, dataset_id: str):
    await stream_manager.connect(dataset_id, websocket)
    try:
        # Send initial status handshake
        await websocket.send_json({
            "type": "connection_established",
            "dataset_id": dataset_id,
            "status": "LIVE",
            "message": f"Connected to DataMind real-time telemetry stream for dataset {dataset_id}"
        })
        while True:
            # Keep socket alive and receive client commands (e.g. ping/filter)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        stream_manager.disconnect(dataset_id, websocket)
    except Exception:
        stream_manager.disconnect(dataset_id, websocket)

@router.post("/api/stream/{dataset_id}/start-demo")
async def start_demo_stream(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    entry.is_realtime_active = True
    await stream_manager.start_demo_stream(dataset_id, entry.processed_df, interval_sec=1.5)
    return {"message": "Demo data stream started", "dataset_id": dataset_id, "status": "LIVE"}

@router.post("/api/stream/{dataset_id}/stop-demo")
def stop_demo_stream(dataset_id: str):
    entry = dataset_store.get_dataset(dataset_id)
    if entry:
        entry.is_realtime_active = False
    stream_manager.stop_demo_stream(dataset_id)
    return {"message": "Demo data stream stopped", "dataset_id": dataset_id, "status": "OFFLINE"}

@router.post("/api/stream/{dataset_id}/ingest")
async def ingest_record(dataset_id: str, payload: Dict[str, Any]):
    entry = dataset_store.get_dataset(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    packet = await stream_manager.ingest_record(dataset_id, payload, source_label="HTTP Ingest")
    return {"status": "success", "packet": packet}
