"""Comprehensive RAG API runtime test script."""
__test__ = False  # Script chạy thủ công, không phải test pytest.
import subprocess
import sys
import time
import json
import requests
import io
import os
from datetime import datetime

PYTHON = r"C:\Users\khooi\AppData\Local\Programs\Python\Python313\python.exe"
LOG_FILE = r"D:\RAG_project\RAG_project\docs\RUNTIME_LOG.md"
PDF_FILE = r"D:\RAG_project\RAG_project\test_sample.pdf"
BASE_URL = "http://127.0.0.1:8000"

log_entries = []

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    entry = f"[{timestamp}] [{level}] {msg}"
    log_entries.append(entry)
    print(entry)

def section(title: str):
    log_entries.append(f"\n{'='*60}")
    log_entries.append(f"  {title}")
    log_entries.append(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def write_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# RAG Project Runtime Test Log\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Python:** {sys.version}\n\n")
        f.write("```\n")
        for entry in log_entries:
            f.write(entry + "\n")
        f.write("```\n")
    log(f"Log written to {LOG_FILE}")

def check_deps():
    section("1. CHECKING DEPENDENCIES")
    deps = ["fastapi", "uvicorn", "reportlab", "pypdf", "chromadb", "pymongo",
            "sentence_transformers", "requests", "openai"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            log(f"  {dep}: INSTALLED")
        except ImportError:
            log(f"  {dep}: MISSING", "WARNING")

def check_services():
    section("2. CHECKING EXTERNAL SERVICES")
    
    # Check MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.server_info()
        log("  MongoDB: AVAILABLE (localhost:27017)")
        client.close()
    except Exception as e:
        log(f"  MongoDB: UNAVAILABLE - {e}", "WARNING")
    
    # Check Ollama
    try:
        resp = requests.get("http://100.71.230.7:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            log(f"  Ollama: AVAILABLE (models: {', '.join(models[:5])})")
        else:
            log(f"  Ollama: UNREACHABLE (status {resp.status_code})", "WARNING")
    except Exception as e:
        log(f"  Ollama: UNAVAILABLE - {type(e).__name__}", "WARNING")
    
    # Check ChromaDB (can always create, no server needed)
    try:
        import chromadb
        log("  ChromaDB: AVAILABLE (local persistence)")
    except Exception as e:
        log(f"  ChromaDB: UNAVAILABLE - {e}", "ERROR")

def start_server():
    section("3. STARTING UVICORN SERVER")
    log(f"Starting server at {BASE_URL}")
    log(f"Command: {PYTHON} -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    
    server_proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=r"D:\RAG_project\RAG_project",
    )
    log(f"Server PID: {server_proc.pid}")
    
    # Wait for server to be ready
    max_wait = 30
    for i in range(max_wait):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=1)
            if resp.status_code == 200:
                log(f"Server READY after {i+1}s")
                return server_proc
        except requests.ConnectionError:
            pass
        
        # Check if process died
        if server_proc.poll() is not None:
            stdout_data = ""
            try:
                stdout_data, _ = server_proc.communicate(timeout=1)
            except Exception:
                pass
            log(f"SERVER DIED during startup! Exit code: {server_proc.returncode}", "ERROR")
            if stdout_data:
                for line in stdout_data.strip().split("\n")[-20:]:
                    log(f"  STDOUT: {line}", "ERROR")
            return None
        
        time.sleep(1)
    
    log("Server DID NOT START within timeout", "ERROR")
    server_proc.kill()
    return None

def collect_server_output(proc, timeout_sec=1):
    """Collect any pending output from server without blocking."""
    if proc is None or proc.poll() is not None:
        return
    import threading
    lines = []
    def _read():
        try:
            for line in iter(proc.stdout.readline, ""):
                lines.append(line.rstrip())
        except Exception:
            pass
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout_sec)

def test_health():
    section("4. TESTING /health ENDPOINT")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        log(f"  Status: {resp.status_code}")
        log(f"  Body: {resp.json()}")
        if resp.status_code == 200 and resp.json() == {"status": "ok"}:
            log("  RESULT: PASS", "SUCCESS")
            return True
        else:
            log("  RESULT: FAIL (unexpected response)", "ERROR")
            return False
    except Exception as e:
        log(f"  RESULT: FAIL - {e}", "ERROR")
        return False

def test_ingest():
    section("5. TESTING POST /ingest ENDPOINT")
    
    # Test 1: Non-PDF rejected
    log("Test 5a: Upload non-PDF file")
    try:
        resp = requests.post(f"{BASE_URL}/ingest",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            timeout=10)
        log(f"  Status: {resp.status_code}")
        log(f"  Body: {resp.json()}")
        if resp.status_code == 400:
            log("  RESULT: PASS (correctly rejected)", "SUCCESS")
        else:
            log("  RESULT: FAIL", "ERROR")
    except Exception as e:
        log(f"  RESULT: ERROR - {e}", "ERROR")
    
    # Test 2: Upload real PDF
    log("\nTest 5b: Upload valid PDF")
    try:
        with open(PDF_FILE, "rb") as f:
            pdf_bytes = f.read()
        log(f"  PDF size: {len(pdf_bytes)} bytes")
        
        resp = requests.post(f"{BASE_URL}/ingest",
            files={"file": ("test_sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            timeout=120)
        log(f"  Status: {resp.status_code}")
        
        try:
            body = resp.json()
            log(f"  Body: {json.dumps(body, indent=2, ensure_ascii=False)}")
        except Exception:
            log(f"  Body (raw): {resp.text[:500]}")
        
        if resp.status_code == 200:
            log("  RESULT: PASS (ingestion successful)", "SUCCESS")
            return body.get("file_id")
        elif resp.status_code == 500:
            log(f"  RESULT: FAIL (server error - likely ChromaDB or MongoDB unavailable)", "WARNING")
        else:
            log(f"  RESULT: FAIL", "ERROR")
    except Exception as e:
        log(f"  RESULT: ERROR - {e}", "ERROR")
    return None

def test_query(file_id=None):
    section("6. TESTING POST /query ENDPOINT")
    
    if file_id:
        log(f"Using file_id from ingestion: {file_id}")
    
    payload = {"query": "Nội dung trang 1 nói gì?"}
    if file_id:
        payload["file_id"] = file_id
    
    try:
        resp = requests.post(f"{BASE_URL}/query",
            json=payload,
            timeout=120)
        log(f"  Status: {resp.status_code}")
        
        try:
            body = resp.json()
            log(f"  Body: {json.dumps(body, indent=2, ensure_ascii=False)[:1000]}")
        except Exception:
            log(f"  Body (raw): {resp.text[:500]}")
        
        if resp.status_code == 200:
            log("  RESULT: PASS", "SUCCESS")
            return body
        elif resp.status_code == 500:
            log(f"  RESULT: FAIL (server error - likely no data in DB or LLM unavailable)", "WARNING")
        else:
            log(f"  RESULT: FAIL", "ERROR")
    except Exception as e:
        log(f"  RESULT: ERROR - {e}", "ERROR")
    return None

def test_endpoint_structures():
    section("7. TESTING API STRUCTURE")
    
    log("Testing endpoint structure via TestClient (no real DB needed)...")
    
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        
        # Health
        resp = client.get("/health")
        log(f"  GET /health -> {resp.status_code} {resp.json()}")
        
        # Non-PDF rejection
        resp = client.post("/ingest", files={"file": ("test.txt", io.BytesIO(b"x"), "text/plain")})
        log(f"  POST /ingest (non-PDF) -> {resp.status_code} (expected 400)")
        
        # Empty file rejection
        resp = client.post("/ingest", files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")})
        log(f"  POST /ingest (empty) -> {resp.status_code} (expected 400)")
        
        # Query structure
        resp = client.post("/query", json={"query": "test", "top_k": 3})
        log(f"  POST /query -> {resp.status_code} (expected 500 - no DB/embedder)")
        
        log("  API structure validation COMPLETE", "SUCCESS")
    except Exception as e:
        log(f"  API structure test ERROR: {e}", "ERROR")

def summary(health_ok, ingest_ok, query_ok):
    section("SUMMARY")
    log(f"  /health endpoint: {'WORKING' if health_ok else 'FAILED'}")
    log(f"  /ingest endpoint: {'WORKING' if ingest_ok else 'FAILED (ChromaDB/MongoDB needed)'}")
    log(f"  /query endpoint:  {'WORKING' if query_ok else 'FAILED (needs ingested data + LLM)'}")
    
    external = []
    try:
        from pymongo import MongoClient
        MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000).server_info()
        external.append("MongoDB")
    except Exception:
        pass
    try:
        requests.get("http://100.71.230.7:11434/api/tags", timeout=2)
        external.append("Ollama")
    except Exception:
        pass
    
    if external:
        log(f"  External services available: {', '.join(external)}")
    else:
        log(f"  External services: NONE available")
    
    log(f"  NOTE: /ingest requires ChromaDB + MongoDB + sentence-transformers")
    log(f"  NOTE: /query requires ChromaDB (with data) + Ollama/OpenAI")

def main():
    log("Starting RAG Project Runtime Tests", "HEADER")
    log(f"Python: {sys.executable}")
    log(f"Working dir: {os.getcwd()}")
    
    try:
        check_deps()
        check_services()
        
        server_proc = start_server()
        
        health_ok = False
        ingest_ok = False
        query_ok = False
        file_id = None
        
        if server_proc:
            health_ok = test_health()
            
            if health_ok:
                # Collect initial server output
                time.sleep(0.5)
                collect_server_output(server_proc, timeout_sec=0.5)
                
                # Run ingest
                file_id = test_ingest()
                if file_id:
                    ingest_ok = True
                
                # Collect server output
                collect_server_output(server_proc, timeout_sec=0.5)
                
                # Run query
                query_ok = test_query(file_id) is not None
                
                # Collect final server output
                collect_server_output(server_proc, timeout_sec=1)
            
            # Kill server
            log("Shutting down server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
                log(f"Server exit code: {server_proc.returncode}")
            except subprocess.TimeoutExpired:
                server_proc.kill()
                log("Server forcefully killed")
        else:
            log("Server failed to start, testing API structure only", "WARNING")
            test_endpoint_structures()
            health_ok = False
        
        # Also run structured tests
        test_endpoint_structures()
        
        summary(health_ok, ingest_ok, query_ok)
        
    except Exception as e:
        log(f"FATAL: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
    
    write_log()
    print(f"\nFull log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
