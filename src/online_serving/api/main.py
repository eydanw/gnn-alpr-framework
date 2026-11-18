"""
Module: src/online_serving/api/main.py
FastAPI Serving Service cho Adaptive Learning Path Planner (A* + GNN Enriched Graph).
Tuân thủ FR-04: Phản hồi thời gian thực dưới 6s kết hợp Redis Cache.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
from online_serving.core.a_star import AdaptiveAStarPlanner

app = FastAPI(
    title="Adaptive Learning Path Recommendation API",
    description=(
        "Dịch vụ gợi ý lộ trình học tập cá nhân hóa sử dụng Thuật toán A* trên"
        " Đồ thị Tri thức làm giàu bởi GNN (GraphSAGE)."
    ),
    version="1.0.0",
)

# Khởi tạo singleton planner trong bộ nhớ
planner: Optional[AdaptiveAStarPlanner] = None


@app.on_event("startup")
def startup_event():
    global planner
    print("🚀 Đang khởi động FastAPI Service & Nạp Knowledge Graph...")
    planner = AdaptiveAStarPlanner()
    print("✅ Đã sẵn sàng phục vụ API!")


@app.on_event("shutdown")
def shutdown_event():
    global planner
    if planner:
        planner.close()
    print("🛑 Đã đóng kết nối cơ sở dữ liệu.")


# --- Pydantic Schemas ---
class PathRequest(BaseModel):
    learner_id: str = Field(..., example="learner_001")
    start_kp_id: str = Field(..., example="6378799e-afca-4669-be9b-9aef4a54fdb1")
    target_kp_id: str = Field(..., example="ac3efcd9-a52e-4e8f-ab18-47217b02c2ee")
    learner_mastery: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Độ thành thạo hiện tại của người học trên các KPs [0.0 - 1.0]",
        example={"6378799e-afca-4669-be9b-9aef4a54fdb1": 0.8},
    )
    use_cache: bool = Field(default=True, description="Kích hoạt Redis Caching")


class StepDetail(BaseModel):
    step: int
    kp_id: str
    title: str
    difficulty: float
    skill_level: str
    edge_transition: Optional[Dict[str, Any]] = None


class PathResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    learner_id: str
    start_kp_id: str
    target_kp_id: str
    total_steps: int
    path: List[StepDetail]
    latency_ms: float
    cached: bool


# --- API Endpoints ---
@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "nodes_loaded": planner.graph.number_of_nodes() if planner else 0,
        "edges_loaded": planner.graph.number_of_edges() if planner else 0,
    }


@app.post(
    "/api/v1/learning-path/plan",
    response_model=PathResponse,
    tags=["Pathfinding"],
)
def plan_learning_path(req: PathRequest):
    if not planner:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Learning Path Planner chưa sẵn sàng.",
        )
    result = planner.find_learning_path(
        learner_id=req.learner_id,
        start_kp_id=req.start_kp_id,
        target_kp_id=req.target_kp_id,
        learner_mastery=req.learner_mastery,
        use_cache=req.use_cache,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get(
                "message", "Không tìm thấy lộ trình học tập phù hợp."
            ),
        )
    return result


if __name__ == "__main__":
    uvicorn.run("online_serving.api.main:app", host="0.0.0.0", port=8000, reload=False)