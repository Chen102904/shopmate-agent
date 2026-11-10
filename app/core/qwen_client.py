
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 加载项目根目录下的 .env
load_dotenv(BASE_DIR / ".env")

# 千问模型配置
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.8-max")


def get_qwen_client() -> OpenAI:
    """
    创建阿里云百炼千问客户端。
    """

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")

    if not api_key:
        raise ValueError("未配置 QWEN_API_KEY，请检查 .env 文件。")

    if not base_url:
        raise ValueError("未配置 QWEN_BASE_URL，请检查 .env 文件。")

    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=30,
        max_retries=0,
    )


def ask_qwen(prompt: str) -> str:
    """
    调用千问模型，返回文本回答。
    """

    client = get_qwen_client()

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content or ""
