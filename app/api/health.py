"""健康检查接口。

用途：
    部署后用 /health 快速判断「服务到底能不能干活」，
    而不是等用户发消息才发现某个依赖挂了。
    容器编排（Docker healthcheck）和负载均衡探活也用它。

设计原则：
    1. 只做轻量检查，不调用大模型、不加载 Embedding 模型。
       健康检查被频繁轮询，不能每次都跑几秒钟。
    2. 不返回任何密钥、路径等敏感信息，只返回「是否正常」。
    3. 有任一必需项异常时返回 503，让探活能正确判定为不健康。
"""

import os

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse


router = APIRouter(tags=["健康检查"])


BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"


# 需要至少配置一个的模型密钥
LLM_KEY_NAMES = (
    "DEEPSEEK_API_KEY",
    "ZHIPU_API_KEY",
    "QWEN_API_KEY",
)


def _check_database() -> dict:
    """检查会话数据库是否可读。"""

    try:
        from app.agent.conversation_store import _get_connection

        with _get_connection() as connection:
            connection.execute("SELECT 1").fetchone()

        return {"ok": True}

    except Exception as exc:
        # 只返回异常类型，不返回可能包含路径的完整信息
        return {"ok": False, "error": type(exc).__name__}


def _check_knowledge_base() -> dict:
    """检查知识库文件是否存在、是否为空，并报告当前版本。

    为什么要报版本：
        「知识库正常」这句话在排查问题时不够用。
        用户说「昨天还能退，今天说不能」，
        第一件事就是确认这两次回答是不是基于同一版知识。
        版本号（内容指纹）让这件事可以直接对上。
    """

    try:
        if not KNOWLEDGE_DIR.exists():
            return {"ok": False, "error": "知识库目录不存在"}

        documents = sorted(KNOWLEDGE_DIR.glob("*.md"))

        if not documents:
            return {"ok": False, "error": "知识库没有文档"}

        sections = 0

        for document in documents:
            text = document.read_text(encoding="utf-8")
            sections += text.count("\n## ")

        if sections == 0:
            return {"ok": False, "error": "知识库没有可检索章节"}

        # 版本指纹。算不出来时降级为 unknown，
        # 不能因为这一项让整个健康检查返回 503——
        # 检索能力还在，只是追溯信息缺失。
        try:
            from app.rag.knowledge_version import compute_knowledge_version

            version = compute_knowledge_version()
        except Exception:
            version = "unknown"

        return {
            "ok": True,
            "documents": len(documents),
            "sections": sections,
            "version": version,
        }

    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _check_llm_config() -> dict:
    """检查是否配置了至少一个模型密钥。

    注意：只检查「有没有配」，不检查「配得对不对」。
    验证密钥有效性需要真实调用，那属于业务请求该做的事，
    不应该放在健康检查里（会让探活变慢且产生费用）。
    """

    configured = {
        name: bool(os.getenv(name))
        for name in LLM_KEY_NAMES
    }

    return {
        "ok": any(configured.values()),
        "configured": configured,
        "note": "只检查是否配置，不校验密钥有效性",
    }


def _check_ticket_store() -> dict:
    """检查工单表、幂等键表与草稿表是否可用。

    为什么单独查：
        建工单是有副作用的操作。如果这几张表不可写，
        用户能查到订单却提交不了售后，
        这种「半可用」状态必须能提前发现，
        不能等到用户点了提交才 500。

    草稿表也在这里：
        它是写操作链路的第一环。草稿写不进去，
        draft 工具会直接失败，用户连申请内容都看不到。
    """

    try:
        from app.core.db import get_connection

        with get_connection() as connection:
            ticket_count = connection.execute(
                "SELECT COUNT(*) AS c FROM after_sales_tickets"
            ).fetchone()["c"]

            key_count = connection.execute(
                "SELECT COUNT(*) AS c FROM idempotency_keys"
            ).fetchone()["c"]

            draft_count = connection.execute(
                """
                SELECT COUNT(*) AS c
                FROM ticket_drafts
                WHERE status IN ('open', 'consuming')
                """
            ).fetchone()["c"]

        return {
            "ok": True,
            "tickets": int(ticket_count),
            "idempotency_keys": int(key_count),
            "pending_drafts": int(draft_count),
        }

    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _check_reranker() -> dict:
    """检查重排模型文件是否存在。

    只检查文件，不加载模型：
        reranker 是 1.1 GB 的 cross-encoder，
        加载一次要一两秒，不能放进被高频轮询的探活里。

    模型缺失不算不健康：
        它是可选优化，代码里会自动降级为纯 RRF 排序，
        服务仍然可用。所以这一项不进 healthy 判定。
    """

    from app.rag.reranker import DISABLE_RERANK, RERANK_MODEL_DIR

    if DISABLE_RERANK:
        return {"ok": True, "enabled": False, "reason": "环境变量已关闭"}

    if not RERANK_MODEL_DIR.exists():
        return {
            "ok": False,
            "enabled": True,
            "reason": "模型文件不存在，检索会自动降级为不重排",
            "optional": True,
        }

    return {"ok": True, "enabled": True}


def _check_models() -> dict:
    """报告各模型供应商的熔断状态。

    为什么要把这个放进健康检查：
        「模型还能不能用」是排查问题的第一分叉。
        用户说「一直提示服务不可用」，
        运维需要一眼看出是「供应商挂了」还是「我们自己的熔断卡住了」。

    为什么不参与 healthy 判定：
        熔断打开后 30 秒会自动试探恢复。
        如果这时候让 /health 返回 503，
        编排系统（K8s、负载均衡）会把实例摘掉甚至重启容器。
        重启并不能让上游供应商更快恢复，却会丢掉熔断状态、
        打断正在处理的请求——比等 30 秒更糟。

        所以这里只报状态，不判死。全部熔断时给一个
        degraded_reason，让人看得懂发生了什么。
    """

    from app.core.circuit_breaker import all_breakers_snapshot

    try:
        providers = all_breakers_snapshot()
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}

    if not providers:
        # 还没发生过模型调用，熔断器尚未创建。
        # 这不是异常：服务刚启动就是这个状态。
        return {"ok": True, "providers": [], "note": "尚无调用记录"}

    open_providers = [
        item["provider"]
        for item in providers
        if item["state"] == "open"
    ]

    all_open = len(open_providers) == len(providers)

    return {
        "ok": not all_open,
        "providers": providers,
        "open_count": len(open_providers),
        "all_open": all_open,
        "degraded_reason": (
            "所有模型均处于熔断状态，请求会快速失败而不是等待"
            if all_open
            else None
        ),
    }


@router.get("/health")
def health():
    """返回服务健康状态。

    必需项（数据库、工单表、知识库、模型密钥）全部正常返回 200，
    任一项异常返回 503。

    reranker 是可选优化，单独报告但不参与健康判定：
    模型缺失时检索会自动降级为不重排，服务仍然可用。
    """

    checks = {
        "database": _check_database(),
        "ticket_store": _check_ticket_store(),
        "knowledge_base": _check_knowledge_base(),
        "llm_config": _check_llm_config(),
        "reranker": _check_reranker(),
        "models": _check_models(),
    }

    required = (
        "database",
        "ticket_store",
        "knowledge_base",
        "llm_config",
    )

    healthy = all(checks[name]["ok"] for name in required)

    payload = {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    }

    return JSONResponse(
        status_code=200 if healthy else 503,
        content=payload,
    )
