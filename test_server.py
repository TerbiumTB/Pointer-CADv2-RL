import os
import json
import yaml
import time
import uvicorn
import threading
import argparse
from tqdm import tqdm
from loguru import logger
from itertools import cycle
from statistics import median
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

# ==================== 滚动TQDM定义 ====================

class TimeRotatingTqdm:
    def __init__(
        self,
        *tqdm_args,
        rotate_interval=5,
        item_limit=5,
        **tqdm_kwargs
    ):
        """
        Args:
            rotate_interval (float): postfix 轮换时间间隔（秒）
            *tqdm_args / **tqdm_kwargs: 原样传给 tqdm
        """
        self._tqdm = tqdm(*tqdm_args, **tqdm_kwargs)
        self._rotate_interval = rotate_interval
        self._item_limit = item_limit

        self._postfix_dict = {}
        self._postfix_current_keys = []
        self._postfix_keys_cycle = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._rotate_loop,
            daemon=True
        )
        self._thread.start()

    def __getattr__(self, name):
        with self._lock:
            return getattr(self._tqdm, name)
    
    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tqdm(*args, **kwargs)

    def set_postfix(self, d: dict):
        with self._lock:
            self._postfix_dict = d
            if self._postfix_keys_cycle is None and len(d) > 0:
                self._postfix_keys_cycle = cycle(d.keys())
                self._item_limit = min(len(d), self._item_limit)

        self._update()

    def close(self):
        self._stop_event.set()
        self._thread.join()
        with self._lock:
            self._tqdm.close()

    def _update(self):
        with self._lock:
            items = {}
            for k in self._postfix_current_keys:
                items[k] = self._postfix_dict.get(k, "N/A")
            
            self._tqdm.set_postfix(items, refresh=True)

    def _rotate_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self._rotate_interval if self._postfix_keys_cycle is not None else 0.1)

            with self._lock:
                if self._postfix_keys_cycle is None or len(self._postfix_dict) == 0:
                    continue
                
                self._postfix_current_keys = [next(self._postfix_keys_cycle) for _ in range(self._item_limit)]
            
            self._update()

# ==================== 请求体定义 ====================
class ConnectRequest(BaseModel):
    gpu_info: str
    dataset_size: int
    log_dir: str
    config: dict

class ModelIDRequest(BaseModel):
    model_id: str

class ReportRequest(BaseModel):
    model_id: str
    result: dict

class FinishRequest(BaseModel):
    gpu_info: str

# ==================== 全局状态变量 ====================
clients = []
claimed_model_ids = {}
invalid_model_ids = []
cds = []
f1 = {"line": [], "circle": [], "arc": [], "extrude": [], "chamfer": [], "fillet": []}
watertightness = []
global_dataset_size = None
progress_bar = None
log_dir = None
finished_clients = 0
expected_clients = None

# ==================== 日志设置 ====================
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")

# ==================== API 实现 ====================
@app.post("/connect")
async def connect(req: ConnectRequest):
    global global_dataset_size, progress_bar, log_dir, expected_clients

    if expected_clients is not None and len(clients) >= expected_clients:
        raise HTTPException(status_code=409, detail="All expected evaluation clients are already connected")

    if global_dataset_size is None:
        global_dataset_size = req.dataset_size
        progress_bar = TimeRotatingTqdm(total=global_dataset_size, dynamic_ncols=True, desc="Test ✨")
        logger.info(f"Initial dataset_size set to {global_dataset_size}")
    elif req.dataset_size != global_dataset_size:
        raise HTTPException(status_code=400, detail="Dataset size mismatch with server")

    if log_dir is None:
        log_dir = req.log_dir
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"job_{len(clients)}.yaml"), "w+") as f:
        yaml.dump(req.config, f, default_flow_style=False)

    clients.append(req.gpu_info)
    logger.info(f"Client connected: GPU={req.gpu_info}, Total clients={len(clients)}")
    return {"client_count": len(clients), "status": "connected", "log_dir": log_dir}


@app.post("/apply_model_id")
async def apply_model_id(req: ModelIDRequest):
    already_claimed = req.model_id in claimed_model_ids
    allowed = not already_claimed
    if allowed:
        claimed_model_ids[req.model_id] = None
        logger.info(f"Model ID '{req.model_id}' approved. Total claimed: {len(claimed_model_ids)}")

    return {
        "allowed": allowed,
        "model_id": req.model_id,
        "claimed": len(claimed_model_ids),
        "total": global_dataset_size,
        "progress": round(len(claimed_model_ids) / global_dataset_size, 4)
    }


@app.post("/report")
async def report_result(req: ReportRequest):
    global progress_bar, claimed_model_ids, invalid_model_ids, cds, f1, watertightness

    if req.model_id not in claimed_model_ids:
        raise HTTPException(status_code=400, detail=f"Model ID '{req.model_id}' not found or not approved.")

    if progress_bar:
        progress_bar.update(1)

    claimed_model_ids[req.model_id] = req.result

    if not req.result.get("status", True):
        invalid_model_ids.append(req.model_id)
    else:
        if "chamfer distance" in req.result and req.result["chamfer distance"] is not None:
            cds.append(req.result["chamfer distance"])
        if isinstance(req.result.get("f1"), dict):
            for key in f1.keys():
                if key in req.result["f1"] and req.result["f1"][key] is not None:
                    f1[key].append(req.result["f1"][key])
        if "is watertight" in req.result and req.result["is watertight"] is not None:
            watertightness.append(int(req.result["is watertight"]))

    progress_bar.set_postfix({
        "CDmean": f"{(sum(cds) / len(cds)):.2f}" if cds else "N/A",
        "CDmedian": f"{median(cds):.2f}" if cds else "N/A",
        "LineF1": f"{(sum(f1['line']) / len(f1['line'])):.2f}%" if f1['line'] else "N/A",
        "ArcF1": f"{(sum(f1['arc']) / len(f1['arc'])):.2f}%" if f1['arc'] else "N/A",
        "CircleF1": f"{(sum(f1['circle']) / len(f1['circle'])):.2f}%" if f1['circle'] else "N/A",
        "ExtrudeF1": f"{(sum(f1['extrude']) / len(f1['extrude'])):.2f}%" if f1['extrude'] else "N/A",
        "ChamferF1": f"{(sum(f1['chamfer']) / len(f1['chamfer'])):.2f}%" if f1['chamfer'] else "N/A",
        "FilletF1": f"{(sum(f1['fillet']) / len(f1['fillet'])):.2f}%" if f1['fillet'] else "N/A",
        "WT": f"{(sum(watertightness) / len(watertightness)) * 100:.2f}%" if watertightness else "N/A",
        "IR": f"{len(invalid_model_ids) / progress_bar.n * 100:.2f}%" if progress_bar.n > 0 else "N/A",
    })

    logger.info(f"Received report for Model ID '{req.model_id}': {req.result}")
    return {"status": "report received", "model_id": req.model_id}


@app.post("/finish")
async def finish(req: FinishRequest):
    global finished_clients, log_dir, claimed_model_ids, expected_clients

    finished_clients += 1
    logger.info(f"Client finished: {req.gpu_info}. Total finished: {finished_clients}/{len(clients)}")

    output_path = os.path.join(log_dir, "results.json")
    with open(output_path, 'w') as f:
        json.dump(claimed_model_ids, f, indent=4)
        logger.success(f"Results saved to {output_path}")

    clients_to_finish = expected_clients if expected_clients is not None else len(clients)
    if finished_clients >= clients_to_finish:
        logger.info("All clients finished. Shutting down server...")

        def shutdown():
            time.sleep(0.5)
            os._exit(0)
        threading.Thread(target=shutdown, daemon=True).start()

    return {"status": "acknowledged", "finished": finished_clients, "total_clients": len(clients)}


@app.get("/ping")
async def ping():
    return {
        "status": "pong",
        "clients": len(clients),
        "expected_clients": expected_clients,
        "finished": finished_clients,
    }


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server with tqdm and loguru logging.")
    parser.add_argument("-p", "--port", type=int, default=32500, help="Port to run the FastAPI server on")
    parser.add_argument(
        "--expected-clients",
        type=int,
        default=None,
        help="Wait for this many clients before shutting down (default: connected clients)",
    )
    args = parser.parse_args()

    if args.expected_clients is not None and args.expected_clients <= 0:
        parser.error("--expected-clients must be greater than zero")
    expected_clients = args.expected_clients

    logger.info(
        f"Starting FastAPI server on port {args.port}; "
        f"expected clients: {expected_clients or 'dynamic'}"
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=None, access_log=False)
