"""知识库版本与热重载测试。

测两类东西：

    1. 指纹与快照（纯文件操作）
       确定性、换行归一、变更检测、快照/对比/回滚

    2. 检索器热重载（要构造 KnowledgeRetriever）
       用假模型替代 SentenceTransformer，避免每次跑测试加载 100MB 模型；
       逻辑（指纹比对、重建时机、失败降级、冷却）都是真的。

所有用例都在 tmp_path 里造知识库，不动项目的 data/knowledge。
"""

import numpy as np
import pytest

from app.rag import document_loader
from app.rag import knowledge_version as kv


# ============================================================
# 夹具
# ============================================================


@pytest.fixture
def kb_dir(tmp_path, monkeypatch):
    """把知识库和快照目录都重定向到临时目录。"""

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()

    snapshots = tmp_path / "snapshots"

    monkeypatch.setattr(kv, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(kv, "SNAPSHOT_DIR", snapshots)

    # 建索引走的是 document_loader 的目录常量，也要一起改，
    # 否则索引读到的是真项目知识库，版本号却按临时目录算。
    monkeypatch.setattr(document_loader, "KNOWLEDGE_DIR", knowledge)

    kv.invalidate_version_cache()

    yield knowledge

    kv.invalidate_version_cache()


def _write(directory, name: str, text: str, newline: str = "\n") -> None:
    content = text.replace("\n", newline)
    (directory / name).write_text(content, encoding="utf-8", newline="")


# ============================================================
# 1. 指纹
# ============================================================


def test_same_content_same_version(kb_dir):
    _write(kb_dir, "a.md", "# 标题\n\n## 一章\n内容\n")

    first = kv.compute_knowledge_version(force=True)
    second = kv.compute_knowledge_version(force=True)

    assert first == second
    assert len(first) == kv.VERSION_PREFIX_LENGTH


def test_crlf_and_lf_produce_same_version(kb_dir):
    """同一份内容，Windows 的 CRLF 和 Linux 的 LF 必须算出同一个版本。

    这是跨平台最容易翻车的地方：
        git 在 Windows 上可能 checkout 成 CRLF，
        Docker 卷挂载进去是 LF。
        不做归一的话，同一份知识库在两边版本号不同，
        「我们用的是同一版吗」这个问题就永远答不上来。
    """

    _write(kb_dir, "a.md", "# 标题\n\n## 一章\n内容\n", newline="\n")
    lf_version = kv.compute_knowledge_version(force=True)

    _write(kb_dir, "a.md", "# 标题\n\n## 一章\n内容\n", newline="\r\n")
    crlf_version = kv.compute_knowledge_version(force=True)

    assert lf_version == crlf_version


def test_content_change_changes_version(kb_dir):
    _write(kb_dir, "a.md", "## 退款\n签收后 7 天内可退\n")
    before = kv.compute_knowledge_version(force=True)

    # 只改一个词，版本必须变
    _write(kb_dir, "a.md", "## 退款\n签收后 15 天内可退\n")
    after = kv.compute_knowledge_version(force=True)

    assert before != after


def test_file_order_does_not_matter(kb_dir):
    """文件名排序保证结果确定，不受目录遍历顺序影响。"""

    _write(kb_dir, "b.md", "B 内容\n")
    _write(kb_dir, "a.md", "A 内容\n")
    first = kv.compute_knowledge_version(force=True)

    # 删掉重写，改变文件创建顺序
    (kb_dir / "a.md").unlink()
    (kb_dir / "b.md").unlink()
    _write(kb_dir, "a.md", "A 内容\n")
    _write(kb_dir, "b.md", "B 内容\n")

    assert kv.compute_knowledge_version(force=True) == first


def test_empty_dir_returns_empty(kb_dir):
    assert kv.compute_knowledge_version(force=True) == "empty"


def test_cache_reused_then_force_recomputes(kb_dir, monkeypatch):
    _write(kb_dir, "a.md", "原始内容\n")

    first = kv.compute_knowledge_version(force=True)

    # 不改变文件特征时，第二次走缓存
    assert kv.compute_knowledge_version() == first

    # 绕过缓存强制重算，结果应一致
    assert kv.compute_knowledge_version(force=True) == first


def test_describe_lists_files(kb_dir):
    _write(kb_dir, "a.md", "内容 A\n")
    _write(kb_dir, "b.md", "内容 B\n")

    info = kv.describe_knowledge()

    assert info["file_count"] == 2
    assert {item["name"] for item in info["files"]} == {"a.md", "b.md"}
    assert info["total_bytes"] > 0


# ============================================================
# 2. 快照
# ============================================================


def test_ensure_snapshot_copies_files(kb_dir):
    _write(kb_dir, "a.md", "内容\n")

    version = kv.compute_knowledge_version(force=True)
    path = kv.ensure_snapshot(version)

    assert path is not None
    assert path.exists()
    assert (path / "a.md").exists()
    assert (path / "manifest.json").exists()


def test_ensure_snapshot_is_idempotent(kb_dir):
    _write(kb_dir, "a.md", "内容\n")

    version = kv.compute_knowledge_version(force=True)

    first = kv.ensure_snapshot(version)
    second = kv.ensure_snapshot(version)

    assert first == second


def test_snapshot_empty_knowledge_returns_none(kb_dir):
    assert kv.ensure_snapshot("empty") is None


def test_prune_keeps_only_max_snapshots(kb_dir, monkeypatch):
    monkeypatch.setattr(kv, "MAX_SNAPSHOTS", 3)

    for index in range(5):
        _write(kb_dir, "a.md", f"内容 {index}\n")
        version = kv.compute_knowledge_version(force=True)
        kv.ensure_snapshot(version)

    assert len(kv.list_snapshots()) <= 3


# ============================================================
# 3. 对比
# ============================================================


def test_diff_detects_changed_file(kb_dir):
    _write(kb_dir, "a.md", "## 一章\n内容\n")
    old = kv.compute_knowledge_version(force=True)
    kv.ensure_snapshot(old)

    _write(kb_dir, "a.md", "## 一章\n内容\n## 二章\n新内容\n")
    kv.invalidate_version_cache()

    result = kv.diff_with_snapshot(old)

    assert len(result["changed"]) == 1
    assert result["changed"][0]["name"] == "a.md"
    assert result["changed"][0]["line_delta"] > 0


def test_diff_detects_added_and_removed(kb_dir):
    _write(kb_dir, "a.md", "A\n")
    _write(kb_dir, "b.md", "B\n")
    old = kv.compute_knowledge_version(force=True)
    kv.ensure_snapshot(old)

    (kb_dir / "b.md").unlink()
    _write(kb_dir, "c.md", "C\n")
    kv.invalidate_version_cache()

    result = kv.diff_with_snapshot(old)

    assert result["added"] == ["c.md"]
    assert result["removed"] == ["b.md"]
    assert result["unchanged"] == ["a.md"]


def test_diff_unknown_version_raises(kb_dir):
    _write(kb_dir, "a.md", "A\n")

    with pytest.raises(FileNotFoundError):
        kv.diff_with_snapshot("not_exist")


# ============================================================
# 4. 回滚
# ============================================================


def test_restore_overwrites_files(kb_dir):
    _write(kb_dir, "a.md", "原始内容\n")
    original = kv.compute_knowledge_version(force=True)
    kv.ensure_snapshot(original)

    # 改坏
    _write(kb_dir, "a.md", "改坏的内容\n")
    kv.invalidate_version_cache()
    assert kv.compute_knowledge_version(force=True) != original

    # 回滚
    restored = kv.restore_snapshot(original)

    assert "a.md" in restored
    assert kv.compute_knowledge_version(force=True) == original
    assert "原始内容" in (kb_dir / "a.md").read_text(encoding="utf-8")


def test_restore_unknown_version_raises(kb_dir):
    with pytest.raises(FileNotFoundError):
        kv.restore_snapshot("not_exist")


# ============================================================
# 5. 检索器热重载
# ============================================================


class _FakeModel:
    """假 Embedding 模型。

    只保证「同样的输入得到同样的向量」，
    向量的语义没有意义——这里测的是重载时机，不是检索效果。
    """

    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)

        items = [texts] if single else list(texts)

        vectors = []

        for text in items:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.random(16)
            vectors.append(vector / (np.linalg.norm(vector) + 1e-9))

        matrix = np.array(vectors)

        return matrix[0] if single else matrix


@pytest.fixture
def retriever(kb_dir, monkeypatch):
    """构造一个用假模型、不加载 reranker 的检索器。"""

    from app.rag import retriever as retriever_module

    monkeypatch.setattr(
        retriever_module,
        "SentenceTransformer",
        lambda *args, **kwargs: _FakeModel(),
    )
    # 冷却设为 0，测试里不需要等
    monkeypatch.setattr(retriever_module, "RELOAD_CHECK_INTERVAL", 0.0)
    monkeypatch.setattr(retriever_module, "HOT_RELOAD_ENABLED", True)

    _write(kb_dir, "a.md", "## 退款政策\n签收后 7 天内可以申请退款\n")

    instance = retriever_module.KnowledgeRetriever(
        use_hybrid=False,
        use_rerank=False,
    )
    instance.build_index()

    return instance


def test_build_index_records_version(retriever, kb_dir):
    assert retriever.version == kv.compute_knowledge_version(force=True)
    assert len(retriever.chunks) > 0


def test_hot_reload_rebuilds_on_change(retriever, kb_dir):
    old_version = retriever.version
    old_chunks = len(retriever.chunks)

    _write(
        kb_dir,
        "a.md",
        "## 退款政策\n签收后 7 天内可以申请退款\n\n"
        "## 换货政策\n尺码不合适可以申请换货\n\n"
        "## 运费\n质量问题运费由商家承担\n",
    )
    kv.invalidate_version_cache()

    assert retriever.maybe_reload() is True
    assert retriever.version != old_version
    assert len(retriever.chunks) > old_chunks


def test_hot_reload_noop_when_unchanged(retriever):
    assert retriever.maybe_reload() is False
    assert retriever.version == retriever.version


def test_hot_reload_keeps_old_index_on_failure(retriever, kb_dir, monkeypatch):
    """知识库被改坏时，检索不能跟着挂。

    这是热重载最危险的地方：
        如果先清空 self.chunks 再重建，重建失败就等于把服务废了。
        正确做法是「先建好新的，成功了再替换」。
    """

    from app.rag import retriever as retriever_module

    old_version = retriever.version
    old_chunks = list(retriever.chunks)

    # 改文件让指纹变化，同时让重建必然失败
    _write(kb_dir, "a.md", "改坏的内容\n")
    kv.invalidate_version_cache()

    monkeypatch.setattr(
        retriever_module,
        "build_knowledge_chunks",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟解析失败")),
    )

    assert retriever.maybe_reload() is False

    # 旧索引必须还在
    assert retriever.version == old_version
    assert retriever.chunks == old_chunks
    assert retriever.embeddings is not None

    # 而且还能正常检索
    results = retriever.search("退款", top_k=1)
    assert isinstance(results, list)


def test_hot_reload_respects_cooldown(retriever, kb_dir, monkeypatch):
    """冷却期内不重复检查指纹，避免每次请求都读文件。"""

    from app.rag import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "RELOAD_CHECK_INTERVAL", 60.0)

    _write(kb_dir, "a.md", "## 新章节\n新内容\n")
    kv.invalidate_version_cache()

    # 第一次检查刷新了时间戳，同一请求内的后续调用被挡住
    retriever._last_check_time = 0.0
    first = retriever.maybe_reload()

    _write(kb_dir, "a.md", "## 又改了\n又一段\n")
    kv.invalidate_version_cache()

    second = retriever.maybe_reload()

    assert first is True
    assert second is False


def test_hot_reload_can_be_disabled(retriever, kb_dir, monkeypatch):
    from app.rag import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "HOT_RELOAD_ENABLED", False)

    old_version = retriever.version
    _write(kb_dir, "a.md", "## 新章节\n新内容\n")
    kv.invalidate_version_cache()

    assert retriever.maybe_reload() is False
    assert retriever.version == old_version


def test_build_index_creates_snapshot(retriever):
    """建索引时顺手存快照，下次变更后才有「上一版」可比。"""

    snapshots = kv.list_snapshots()

    assert any(item["version"] == retriever.version for item in snapshots)
