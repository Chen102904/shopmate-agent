
from fastapi import APIRouter, Depends, HTTPException, Path

from app.api.deps import require_user_id
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
    user_id: str = Depends(require_user_id),
):
    """根据 request_id 查询 Agent 完整执行记录。

    执行轨迹包含工具参数（例如订单号）、检索来源和最终回答，
    属于用户私有数据，因此必须校验归属。
    不属于当前用户的轨迹统一返回 404，不透露该轨迹是否存在。
    """

    trace = get_completed_execution_trace(
        request_id
    )

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="执行记录不存在或已过期",
        )

    if trace.get("user_id") != user_id:
        raise HTTPException(
            status_code=404,
            detail="执行记录不存在或已过期",
        )

    return trace
