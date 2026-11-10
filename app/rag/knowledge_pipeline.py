
from app.rag.document_loader import load_knowledge_documents
from app.rag.chunker import chunk_markdown


def build_knowledge_chunks(
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """
    加载知识库文档，并按照 Markdown 标题切分文本。

    每个 Chunk 保留：
    - chunk_id：文本块编号
    - source：来源文件
    - content：文本块内容
    """

    documents = load_knowledge_documents()

    all_chunks = []

    for document in documents:
        source = document["source"]
        content = document["content"]

        chunks = chunk_markdown(
            content,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{source}_{index}",
                    "source": source,
                    "content": chunk,
                }
            )

    return all_chunks
