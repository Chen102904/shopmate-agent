"""本地演示辅助接口：模拟客服审核。

为什么需要它：

    正常的审核入口是 `POST /after-sales/{id}/review`，需要 X-Admin-Token。
    这是刻意设计的——审核权限和用户身份必须分开，
    否则用户就能审核通过自己的退款申请，等于自批退款。

    但本地自己用的时候，我希望点一下就能看到
    「待审核 → 已通过 → 订单变已退款」这条链路真的能跑通，
    而不是去开一个终端敲 curl 带管理员令牌。

所以这里单独开一个模拟入口，并且：

    1. **默认关闭**：只有设了 SHOPMATE_DEMO_REVIEW=1 才启用，
       没设的时候整个路由返回 404，对外等于不存在。
    2. **只能动自己的工单**：仍然按 X-User-Id 校验归属，
       不能拿它去改别人的售后单。
    3. **和生产审核入口互不影响**：管理员接口的鉴权一行没改。

面试时如果有人问「你为什么留了自批入口」，
答案是：这个路由在未配置环境变量时不存在，
生产部署不会打开它；它存在的唯一目的是本地演示可交互。
"""

import os

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_user_id
from app.schemas.after_sales import AfterSalesResponse, ReviewRequest
from app.services.after_sales_service import (
    get_after_sales_ticket,
    review_after_sales_ticket,
)
from app.services.ticket_store import InvalidStatusTransition

router = APIRouter(prefix="/demo", tags=["演示辅助"])

DEMO_REVIEW_ENV = "SHOPMATE_DEMO_REVIEW"


def demo_review_enabled() -> bool:
    """是否启用模拟审核。

    每次请求现读环境变量，不在模块导入时缓存：
        .env 是靠 llm_client 导入时才加载的，
        如果这里在导入阶段取值，很可能拿到的是 None。
    """

    return os.getenv(DEMO_REVIEW_ENV, "").strip() == "1"


def _ensure_enabled() -> None:
    if not demo_review_enabled():
        # 和未配置管理员令牌时一致：不暴露「这里有个开关」，
        # 直接用 404 表示这个接口不存在。
        raise HTTPException(status_code=404, detail="接口不存在")


@router.get("/status")
def demo_status():
    """前端据此决定要不要显示「模拟审核」按钮。"""

    return {"review_enabled": demo_review_enabled()}


@router.post("/review/{ticket_id}", response_model=AfterSalesResponse)
def simulate_review(
    ticket_id: UUID,
    payload: ReviewRequest,
    user_id: str = Depends(require_user_id),
):
    """模拟客服审核自己的一张工单。

    只允许 approved / rejected，且只能审自己的单。
    """

    _ensure_enabled()

    ticket = get_after_sales_ticket(str(ticket_id), user_id)

    if ticket is None:
        # 不区分「不存在」和「不是你的」，和业务接口口径一致。
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在或无权访问",
        )

    try:
        updated, _ = review_after_sales_ticket(
            str(ticket_id),
            payload.decision,
            payload.review_note,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except InvalidStatusTransition as exc:
        # 已经审核过、或用户已撤销的单不能再审。
        raise HTTPException(status_code=409, detail=str(exc))

    if updated is None:
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在",
        )

    return updated
