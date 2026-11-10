
from fastapi import APIRouter, HTTPException, Path

from app.core.execution_trace import (
    get_completed_execution_trace,
)


router = APIRouter(
    prefix="/traces",
    tags=["执行记录"],
)


@router.get("/{request_id}")
def query_execution_trace(
    request_id: str = Path(
        min_length=12,
        max_length=12,
        pattern="^[a-f0-9]{12}$",
        description="Agent 请求 ID",
    ),
):
    """根据 request_id 查询 Agent 完整执行记录。"""

    trace = get_completed_execution_trace(
        request_id
    )

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="执行记录不存在或已过期",
        )

    return trace
