import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import datasets, query, stream
from backend.services.data_loader import DataLoaderService
from backend.services.dataset_store import dataset_store
from backend.demo_data.generate_samples import generate_demo_datasets

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure demo datasets exist and pre-load default demo dataset
    os.makedirs(settings.DEMO_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    saas_path = os.path.join(settings.DEMO_DIR, "demo_saas_sales.csv")
    if not os.path.exists(saas_path):
        print("Generating demo datasets on startup...")
        generate_demo_datasets(settings.DEMO_DIR)

    # Preload the SaaS dataset into memory for immediate exploration
    try:
        df_demo, meta_demo = DataLoaderService.load_demo("saas_sales", settings.DEMO_DIR)
        dataset_store.register_dataset(
            name="[DEMO] Global SaaS Revenue & Marketing Performance",
            original_df=df_demo,
            source_type="demo",
            file_size_kb=meta_demo["file_size_kb"],
            impute_missing=True,
            remove_duplicates=True
        )
        print("Successfully preloaded default SaaS demo dataset into DataVista AI.")
    except Exception as e:
        print(f"Warning: Failed to preload demo dataset: {e}")

    yield
    # Shutdown logic if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-Powered Real-Time Data Visualization & Insight Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix=settings.API_PREFIX)
app.include_router(query.router, prefix=settings.API_PREFIX)
app.include_router(stream.router)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "DataVista AI Engine",
        "datasets_loaded": len(dataset_store.datasets)
    }

# Mount built React/PowerBI UI static files so http://localhost:8000 serves the full web app
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "name": settings.PROJECT_NAME,
            "status": "online",
            "docs": "/docs",
            "health": "/api/health",
            "datasets_loaded": len(dataset_store.datasets)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
