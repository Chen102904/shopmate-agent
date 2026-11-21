import os
import secrets

from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from app.api.deps import require_user_id
from app.models.after_sales import TicketStatus
from app.schemas.after_sales import (
    AfterSalesCreateRequest,
    AfterSalesResponse,
    ReviewRequest,
)
from app.services.after_sales_service import (
    IdempotencyConflict,
    RequestInProgress,
    cancel_after_sales_ticket,
    create_after_sales_ticket,
    get_after_sales_ticket,
    review_after_sales_ticket,
)
from app.services.ticket_store import (
    InvalidStatusTransition,
    list_recent_tickets,
)

router = APIRouter(prefix="/after-sales", tags=["售后工单"])

# 幂等键请求头。
#
# 用标准名字 Idempotency-Key（Stripe、IETF draft 都这么叫），
# 不自己发明 X- 前缀，客户端接入时少一层沟通成本。
IDEMPOTENCY_HEADER = "Idempotency-Key"

# 幂等键长度上限。
# 太长的键没有实际意义，只会浪费索引空间，
# 也留出被拿来塞垃圾数据的口子。
MAX_IDEMPOTENCY_KEY_LENGTH = 255


# 管理员凭证。
#
# 为什么单独一套，不复用 X-User-Id：
#     X-User-Id 是演示级身份，任何人可以声称自己是任何人。
#     如果审核接口也用它，用户就能审核通过自己的退款申请——
#     等于自批退款。这是这个项目里最不能出现的漏洞。
#
# 不配置就等于不开启：
#     默认没有值，审核接口返回 404，避免误上线一个谁都能调的审核入口。
ADMIN_TOKEN = os.getenv("SHOPMATE_ADMIN_TOKEN", "").strip()

ADMIN_TOKEN_HEADER = "X-Admin-Token"


def require_admin(
    x_admin_token: str | None = Header(
        default=None,
        alias=ADMIN_TOKEN_HEADER,
        description="管理员凭证。未配置 SHOPMATE_ADMIN_TOKEN 时该接口不可用。",
    ),
) -> None:
    """校验管理员身份。

    没配置 ADMIN_TOKEN 时返回 404 而不是 403：
        403 等于告诉外界「这里有个管理员接口，只是你没权限」，
        404 让这个接口在未启用时对外界完全不存在。

    比较用 secrets.compare_digest：
        普通字符串比较会在第一个不同字符处提前返回，
        攻击者能靠响应时间差逐位猜出 token。
    """

    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=404,
            detail="接口不存在",
        )

    provided = (x_admin_token or "").encode("utf-8")

    if not provided or not secrets.compare_digest(
        provided,
        ADMIN_TOKEN.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=403,
            detail="无权限执行该操作",
        )


@router.post("", response_model=AfterSalesResponse)
def create_ticket(
    request: AfterSalesCreateRequest,
    response: Response,
    idempotency_key: str | None = Header(
        default=None,
        alias=IDEMPOTENCY_HEADER,
        description=(
            "可选。客户端为每次「创建工单」生成一个唯一值（如 UUID）。"
            "同一个键重复提交同一份内容，只会建一张工单，"
            "后续请求返回第一次的结果（HTTP 200）。"
            "不传则按「同一用户 + 同一订单 + 同一类型」做业务去重。"
        ),
    ),
    user_id: str = Depends(require_user_id),
):
    """创建售后工单。

    订单归属由服务端身份决定，请求体里的 user_id 会被忽略，
    防止用户替别人的订单申请售后。

    幂等语义：
        - 新建成功 → 201，响应头 Idempotent-Replayed: false
        - 重复请求 → 200，响应头 Idempotent-Replayed: true
          （返回的是第一次那张工单，不会产生第二张）
        - 同一个键换内容 → 409
        - 同一个键上一笔还没跑完 → 409
    """

    if idempotency_key is not None:

        idempotency_key = idempotency_key.strip()

        if not idempotency_key:
            # 传了空字符串等于没传，按无键处理，不要让人用空键绕过
            idempotency_key = None

        elif len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Idempotency-Key 长度不能超过 "
                    f"{MAX_IDEMPOTENCY_KEY_LENGTH} 个字符"
                ),
            )

    try:
        ticket, replayed = create_after_sales_ticket(
            order_id=request.order_id,
            user_id=user_id,
            request_type=request.request_type,
            reason=request.reason,
            idempotency_key=idempotency_key,
            with_meta=True,
        )

    except IdempotencyConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except RequestInProgress as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    response.status_code = 200 if replayed else 201
    response.headers["Idempotent-Replayed"] = "true" if replayed else "false"

    return ticket


@router.get("", response_model=list[AfterSalesResponse])
def list_tickets(
    status: TicketStatus | None = Query(
        default=None,
        description="按状态筛选，例如 pending、cancelled。不传返回全部。",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    user_id: str = Depends(require_user_id),
):
    """列出当前用户的售后工单，最新的在前。

    只返回本人的工单：user_id 来自请求头，不接受查询参数指定，
    否则改个 URL 就能翻别人的售后记录。
    """

    return list_recent_tickets(user_id, limit=limit, status=status)


@router.get("/{ticket_id}", response_model=AfterSalesResponse)
def query_ticket(
    ticket_id: UUID,
    user_id: str = Depends(require_user_id),
):
    """根据工单编号查询售后工单。只能查询自己的工单。"""

    ticket = get_after_sales_ticket(
        str(ticket_id),
        user_id,
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在或无权访问",
        )

    return ticket


@router.post(
    "/{ticket_id}/cancel",
    response_model=AfterSalesResponse,
)
def cancel_ticket(
    ticket_id: UUID,
    response: Response,
    user_id: str = Depends(require_user_id),
):
    """撤销自己的售后申请。

    语义：
        - 只有「待人工审核」的工单可以撤销
        - 撤销后状态变成 cancelled，不再进入审核
        - 重复撤销返回 200 和当前状态（幂等），不报错
        - 别人的工单返回 404，不区分「不存在」和「不是你的」

    为什么是「撤销」不是「删除」：
        工单是审计凭证。撤销后审核员仍应看得到
        「这张单提交过、后来被用户撤回了」。
    """

    try:
        ticket, changed = cancel_after_sales_ticket(
            str(ticket_id),
            user_id,
        )

    except InvalidStatusTransition as exc:
        # 状态不允许撤销（比如已经被审核通过或驳回）。
        # 这不是「找不到」，而是「不能这么做」，所以是 409 不是 404。
        raise HTTPException(status_code=409, detail=str(exc))

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在或无权访问",
        )

    if not changed:
        # 已经是取消状态：幂等返回，让客户端可以安全重试
        response.status_code = 200
        response.headers["Idempotent-Replayed"] = "true"
    else:
        response.status_code = 200
        response.headers["Idempotent-Replayed"] = "false"

    return ticket


@router.post(
    "/{ticket_id}/review",
    response_model=AfterSalesResponse,
)
def review_ticket(
    ticket_id: UUID,
    payload: ReviewRequest,
    _: None = Depends(require_admin),
):
    """人工审核：通过或驳回。**管理员接口**。

    权限：
        需要请求头 X-Admin-Token，且与服务端 SHOPMATE_ADMIN_TOKEN 一致。
        没配置这个环境变量时接口直接 404（功能未启用）。

    为什么必须单独一套凭证：
        现有的 X-User-Id 是演示级身份，任何人都能声称自己是任何人。
        如果审核也用它，用户就能审核通过自己的退款申请——
        等于自批退款。这是这个项目里最不能出现的漏洞。
    """

    try:
        ticket, changed = review_after_sales_ticket(
            str(ticket_id),
            payload.decision,
            payload.review_note,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except InvalidStatusTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="售后工单不存在",
        )

    return ticket
