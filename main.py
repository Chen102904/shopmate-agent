import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging_config import configure_logging
from app.core.request_context import get_request_id
from app.core.api_security import api_security_middleware


# 初始化统一日志系统
configure_logging()

logger = logging.getLogger("app.api")


from app.api.order import router as order_router
from app.api.logistics import router as logistics_router
from app.api.refund import router as refund_router
from app.api.after_sales import router as after_sales_router
from app.api.chat import router as chat_router
from app.api.traces import router as traces_router
from app.api.health import router as health_router
from app.api.auth import router as auth_router


def env_bool(name: str, default: bool) -> bool:
    """读取布尔类型环境变量。"""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# 开发环境默认开放 Swagger 文档。
# 生产环境可以通过 SHOPMATE_ENABLE_DOCS=false 关闭。
enable_docs = env_bool(
    "SHOPMATE_ENABLE_DOCS",
    default=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。

    启动时清理过期草稿。

    为什么挂在启动而不是定时任务：
        这个项目没有引入调度器，加一个 APScheduler
        只为了删几张过期记录，依赖成本不划算。
        草稿 10 分钟就过期，真正的堆积速度取决于进程重启频率，
        而重启频率远低于草稿产生速度。

    只删「已过期 + 未消费」的：
        已消费的草稿要留着做幂等回放（用户重复说确认时
        要能返回原工单号），不能一起删掉。

    失败不影响启动：
        清理是维护动作，不该因为它让整个服务起不来。
    """

    try:
        from app.agent.ticket_draft import cleanup_expired_drafts

        removed = cleanup_expired_drafts()

        if removed:
            logger.info("启动清理过期草稿 | removed=%s", removed)

    except Exception:
        logger.warning("启动清理过期草稿失败，忽略", exc_info=True)

    yield


app = FastAPI(
    title="ShopMate Agent",
    description="智能电商售后 Agent 后端服务",
    version="1.2.0",
    docs_url="/docs" if enable_docs else None,
    redoc_url="/redoc" if enable_docs else None,
    openapi_url="/openapi.json" if enable_docs else None,
    lifespan=lifespan,
)


# 注册 API 安全中间件
@app.middleware("http")
async def security_middleware(
    request: Request,
    call_next,
):
    return await api_security_middleware(
        request,
        call_next,
    )


# 注册原有业务路由
app.include_router(order_router)
app.include_router(logistics_router)
app.include_router(refund_router)
app.include_router(after_sales_router)

# 注册新增路由
app.include_router(chat_router)
app.include_router(traces_router)

# 健康检查：放在最前面注册，探活请求不经过业务路由
app.include_router(health_router)

# 令牌签发
app.include_router(auth_router)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    统一处理未捕获的服务器异常。

    保留 FastAPI 对 HTTPException 和参数校验错误
    的原有处理方式。
    """

    request_id = get_request_id()

    logger.exception(
        "API 未处理异常 | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误，请稍后重试",
        },
    )
