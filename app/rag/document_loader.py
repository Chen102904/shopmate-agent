
from pathlib import Path


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 售后政策知识库目录
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"


def load_markdown_file(file_path: Path) -> str:
    """
    读取指定的 Markdown 文件。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{file_path}"
        )

    if file_path.suffix.lower() != ".md":
        raise ValueError("目前只支持 Markdown 文件")

    return file_path.read_text(encoding="utf-8")


def load_knowledge_documents() -> list[dict]:
    """
    加载知识库目录下的所有 Markdown 文件。

    Returns:
        文档列表，每个元素包含文件名和文本内容。
    """

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        content = load_markdown_file(file_path)

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents
