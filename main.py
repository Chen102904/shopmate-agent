import logging
import os

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


app = FastAPI(
    title="ShopMate Agent",
    description="智能电商售后 Agent 后端服务",
    version="1.2.0",
    docs_url="/docs" if enable_docs else None,
    redoc_url="/redoc" if enable_docs else None,
    openapi_url="/openapi.json" if enable_docs else None,
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

# 令牌签发：客户端拿用户名密码换一个带签名的令牌，
# 后续请求用 Authorization 头，比自报 X-User-Id 可信
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