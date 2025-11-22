# main.py
import io
import time
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from numba import cuda
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("gpu_service_requests_total", "Total requests to /add")
REQUEST_LATENCY = Histogram("gpu_service_add_latency_seconds", "Latency for /add")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

async def load_npz_array(upload: UploadFile):
    content = await upload.read()
    f = io.BytesIO(content)
    try:
        npz = np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad NPZ file: {e}")
    return npz[npz.files[0]].astype(np.float32)

@cuda.jit
def add_kernel(A, B, C, n):
    i = cuda.grid(1)
    if i < n:
        C[i] = A[i] + B[i]

@app.post("/add")
async def add_matrices(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start_all = time.perf_counter()

    a = await load_npz_array(file_a)
    b = await load_npz_array(file_b)

    if a.shape != b.shape:
        raise HTTPException(status_code=400, detail="Matrices must have the same shape")

    a_flat = a.ravel()
    b_flat = b.ravel()
    n = a_flat.size

    d_a = cuda.to_device(a_flat)
    d_b = cuda.to_device(b_flat)
    d_c = cuda.device_array_like(d_a)

    threads = 256
    blocks = (n + threads - 1) // threads

    t0 = time.perf_counter()
    add_kernel[blocks, threads](d_a, d_b, d_c, n)
    cuda.synchronize()
    t1 = time.perf_counter()

    REQUEST_LATENCY.observe(t1 - t0)

    return {
        "matrix_shape": list(a.shape),
        "elapsed_time": float(t1 - t0),
        "device": "GPU",
        "total_time": float(time.perf_counter() - start_all)
    }

@app.get("/gpu-info")
def gpu_info():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        result = []
        for line in p.stdout.strip().splitlines():
            idx, used, total = [x.strip() for x in line.split(",")]
            result.append({
                "gpu": idx,
                "memory_used_MB": int(used),
                "memory_total_MB": int(total)
            })
        return {"gpus": result}
    except Exception as e:
        return {"error": str(e)}
