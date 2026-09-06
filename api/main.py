import logging
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ingestion.pipeline import IngestionConfig, run_ingestion
from ingestion.storage.object_store import MinioObjectStore
from ingestion.storage.store import IngestionStore, MetadataStore, VectorStore
from retrieval.pipeline import search_and_generate
from retrieval.models import QueryRequest
from shared.config import settings

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

app = FastAPI(title="RAG Project API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file: {e}")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rỗng")

    if len(pdf_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File quá lớn. Tối đa {MAX_UPLOAD_SIZE // (1024*1024)}MB")

    try:
        config = IngestionConfig(
            chunk_strategy=settings.CHUNK_STRATEGY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embed_provider=settings.EMBED_PROVIDER,
            embed_batch_size=settings.EMBED_BATCH_SIZE,
            chroma_path=settings.CHROMA_PATH,
            chroma_collection=settings.CHROMA_COLLECTION,
            mongo_uri=settings.MONGO_URI,
            mongo_db=settings.MONGO_DB,
            minio_enabled=settings.MINIO_ENABLED,
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_access_key=settings.MINIO_ACCESS_KEY,
            minio_secret_key=settings.MINIO_SECRET_KEY,
            minio_bucket=settings.MINIO_BUCKET,
            minio_secure=settings.MINIO_SECURE,
        )
        result = run_ingestion(pdf_bytes, file.filename, config)
        return {
            "status": "success",
            "file_id": result.file_record.file_id,
            "filename": result.file_record.filename,
            "total_chunks": result.total_chunks,
            "duration_seconds": result.duration_seconds,
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/files")
async def list_files():
    """Liệt kê danh sách tất cả file_id và thông tin tài liệu PDF đang có trong MinIO."""
    try:
        if not settings.MINIO_ENABLED:
            return {"total_files": 0, "file_ids": [], "files": [], "message": "MinIO is disabled"}

        object_store = MinioObjectStore(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            secure=settings.MINIO_SECURE,
        )
        files = object_store.list_files()
        file_ids = [f["file_id"] for f in files]
        return {
            "total_files": len(files),
            "file_ids": file_ids,
            "files": files,
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách files từ MinIO: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    vector_store = VectorStore(settings.CHROMA_PATH, settings.CHROMA_COLLECTION)
    metadata_store = MetadataStore(settings.MONGO_URI, settings.MONGO_DB)
    object_store = MinioObjectStore(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_BUCKET,
        secure=settings.MINIO_SECURE,
    ) if settings.MINIO_ENABLED else None

    try:
        vector_store.delete_by_file(file_id)
    except Exception as e:
        logger.warning(f"Chroma delete failed for {file_id}: {e}")

    try:
        deleted = metadata_store.delete_by_file(file_id)
    except Exception as e:
        logger.error(f"MongoDB delete failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB delete failed: {e}")

    if object_store:
        try:
            # Các file ingest từ trước khi có MinIO sẽ trả về 0 object đã xóa.
            object_store.delete_file(file_id)
        except Exception as e:
            logger.warning(f"MinIO delete skipped for {file_id}: {e}")

    return {"status": "deleted", "file_id": file_id, "deleted_chunks": deleted}


@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = search_and_generate(
            query=request.query,
            file_id=request.file_id,
            top_k=request.top_k,
        )
        return {
            "answer": result.answer,
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "file_id": s.file_id,
                    "text": s.text[:500],
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "score": round(s.score, 4),
                    "filename": s.filename,
                }
                for s in result.sources
            ],
            "intent": result.intent,
            "query_time_ms": round(result.query_time_ms, 2),
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# === API-FOOTBALL / API-SPORTS ENDPOINTS ===

@app.get("/widgets", response_class=HTMLResponse)
async def football_widgets():
    """Trang giao diện trực quan nhúng API-Sports Widgets hiển thị giải đấu, tỷ số và lịch thi đấu."""
    api_key = settings.APISPORTS_KEY
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API-Sports Football Widgets - Live Scores & Leagues</title>
    <!-- API-Sports Widget Scripts -->
    <script type="module" src="https://widgets.api-sports.io/2.0.3/widgets.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
        }}
        h1 {{
            color: #38bdf8;
            margin-bottom: 8px;
        }}
        .badge {{
            display: inline-block;
            background: #1e293b;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 14px;
            color: #94a3b8;
            border: 1px solid #334155;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            color: #1e293b;
        }}
        .footer {{
            text-align: center;
            margin-top: 24px;
            color: #64748b;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ API-Sports Football Widgets Dashboard</h1>
        <div class="badge">Live Scores, Leagues & Standings</div>
    </div>

    <div class="container">
        <!-- API-Sports Widgets Requested by User -->
        <api-sports-widget data-type="leagues"></api-sports-widget>

        <!-- Configuration Widget -->
        <api-sports-widget data-type="config"
            data-key="{api_key}"
            data-sport="football"
            data-lang="en"
            data-theme="white"
            data-show-errors="true">
        </api-sports-widget>
    </div>

    <div class="footer">
        RAG Project - Football Assistant & Live Data Integration &bull; Powered by API-Sports
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/football/status")
async def football_status():
    """Kiểm tra hạn mức API-Sports và thông tin tài khoản."""
    from services.football_api import FootballApiClient
    client = FootballApiClient()
    return client.get_status()


@app.get("/football/live")
async def football_live():
    """Lấy danh sách các trận đấu đang diễn ra trực tiếp (Live Scores)."""
    from services.football_api import FootballApiClient
    client = FootballApiClient()
    matches = client.get_live_fixtures()
    return {"count": len(matches), "matches": matches}


@app.get("/football/teams")
async def football_teams(search: str = Query(..., min_length=2, description="Tên câu lạc bộ cần tìm (vd: Arsenal, Real Madrid)")):
    """Tìm kiếm thông tin câu lạc bộ / đội bóng theo tên."""
    from services.football_api import FootballApiClient
    client = FootballApiClient()
    teams = client.search_teams(search)
    return {"results": len(teams), "teams": teams}


@app.get("/football/players")
async def football_players(search: str = Query(..., min_length=2, description="Tên cầu thủ cần tìm (vd: Messi, Ronaldo, Haaland)")):
    """Tìm kiếm hồ sơ cầu thủ theo tên."""
    from services.football_api import FootballApiClient
    client = FootballApiClient()
    players = client.search_players(search)
    return {"results": len(players), "players": players}

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
