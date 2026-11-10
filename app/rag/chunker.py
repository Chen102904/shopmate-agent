
import re


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    将长文本切分为多个文本块。
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap 必须大于等于 0 且小于 chunk_size"
        )

    text = text.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def split_markdown_sections(text: str) -> list[str]:
    """
    根据 Markdown 二级标题切分文档。

    例如：
    ## 一、退款政策
    ## 二、退货政策
    ## 三、换货政策

    每个二级标题及其下面的内容组成一个独立章节。
    """

    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()

    sections = []
    prefix_lines = []
    current_lines = []
    found_section = False

    for line in lines:

        # 遇到 Markdown 二级标题
        if re.match(r"^##\s+", line):

            # 保存上一章节
            if current_lines:
                section = "\n".join(current_lines).strip()

                if section:
                    sections.append(section)

            # 第一章节保留文档标题等前置内容
            if not found_section:
                current_lines = prefix_lines + [line]
                found_section = True
            else:
                current_lines = [line]

        else:

            # 第一个二级标题之前的内容
            if not found_section:
                prefix_lines.append(line)
            else:
                current_lines.append(line)

    # 保存最后一个章节
    if current_lines:
        section = "\n".join(current_lines).strip()

        if section:
            sections.append(section)

    # 如果文档没有二级标题，就将全文作为一个章节
    if not sections and prefix_lines:
        sections.append("\n".join(prefix_lines).strip())

    return sections


def chunk_markdown(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    先根据 Markdown 标题切分章节，
    再对过长的章节进行普通文本切分。
    """

    sections = split_markdown_sections(text)

    all_chunks = []

    for section in sections:

        section_chunks = chunk_text(
            section,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(section_chunks)

    return all_chunks
