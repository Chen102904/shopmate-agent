"""知识库版本：指纹、快照与变更检测。

为什么需要这个模块：

    在加版本之前，知识库是一个「隐式全局状态」：
        - 改了 md 文件，必须重启进程才生效；
        - chunk_id 是 `文件名_序号`，内容一变序号就漂移，
          事后看到 `after_sales_policy.md_12` 根本不知道是哪一版；
        - 评测报告只记录数据集版本，不记录知识库版本，
          所以「这次 Hit@3 掉了 4 个点」无法区分
          是模型波动还是知识库改坏了。

    这个坑是真踩过的：
        把「尚未付款」改写成「未付款」，Hit@3 从 0.68 掉到 0.44。
        改动的人当时以为只是措辞优化，没有任何机制提示他
        「你正在改动检索命中的关键锚点」。

设计取舍（三层，不做第四层）：

    第 1 层 指纹：内容哈希，用来确认「这条回答基于哪一版知识」。
    第 2 层 热重载：文件变了自动重建索引，构建失败保留旧索引。
    第 3 层 快照：每个出现过的版本留一份副本，支持对比与回滚。

    不做：向量索引持久化、单 chunk 增量更新。
    15 个文本块重建不到 0.2 秒，做增量更新是纯负担。

指纹为什么必须规范化换行：
    同一份内容在 Windows 上是 CRLF、在 git 里可能被转成 LF、
    在 Docker 挂载卷里又可能是 LF。
    不统一的话，同一份知识库在不同机器上算出不同指纹，
    「版本一致」这件事就没法验证了。
"""

import hashlib
import json
import shutil

from datetime import datetime
from pathlib import Path


# 知识库目录（与 document_loader 保持一致，避免两处定义漂移）
BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

# 版本快照目录
SNAPSHOT_DIR = BASE_DIR / "data" / "knowledge_versions"

# 快照保留数量。
# 每份只有几十 KB，留 10 个足够回溯近期变更；
# 不做无限保留，否则目录会随时间无限增长。
MAX_SNAPSHOTS = 10

# 指纹取前多少位。
# 12 位十六进制 = 48 bit，对「同一项目内的知识库版本」足够，
# 且比完整的 64 位短一半，日志和报告里更好读。
VERSION_PREFIX_LENGTH = 12


def _normalize_text(text: str) -> str:
    """统一换行符并去掉尾部空白。

    CRLF / CR / LF 三种都要归一到 \\n，
    否则跨平台算出的指纹不一致。
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    return normalized.strip() + "\n"


def _knowledge_files() -> list[Path]:
    """知识库文件列表，按文件名排序。

    排序是必须的：
        glob 的顺序依赖文件系统，同一台机器上也可能变。
        指纹算法必须是确定性的，否则同样的内容会算出不同结果。
    """

    if not KNOWLEDGE_DIR.exists():
        return []

    return sorted(
        KNOWLEDGE_DIR.glob("*.md"),
        key=lambda path: path.name,
    )


# 按「文件名 + mtime + 大小」缓存的版本号。
# 避免每次检索都读一遍全部文件，同时保证文件一变就会重算。
_version_cache: dict | None = None


def _files_signature(files: list[Path]) -> tuple:
    """文件的外部特征，用来判断「有没有可能变了」。"""

    signature = []

    for path in files:
        try:
            stat = path.stat()
            signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            # 刚好被删掉或没有权限：把名字记上，让指纹重新计算
            signature.append((path.name, -1, -1))

    return tuple(signature)


def compute_knowledge_version(force: bool = False) -> str:
    """计算知识库内容指纹。

    参数：
        force: True 时忽略缓存强制重算。

    返回：
        12 位十六进制字符串；知识库为空时返回 "empty"。
    """

    global _version_cache

    files = _knowledge_files()

    if not files:
        return "empty"

    signature = _files_signature(files)

    if (
        not force
        and _version_cache is not None
        and _version_cache.get("signature") == signature
    ):
        return _version_cache["version"]

    digest = hashlib.sha256()

    for path in files:

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            # 读不到就当空文件，不能因为一个坏文件让整个版本号算不出来
            raw = ""

        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_normalize_text(raw).encode("utf-8"))
        digest.update(b"\x00")

    version = digest.hexdigest()[:VERSION_PREFIX_LENGTH]

    _version_cache = {
        "version": version,
        "signature": signature,
    }

    return version


def invalidate_version_cache() -> None:
    """清掉版本缓存，强制下一次重算。

    测试和「手动触发重载」会用到。
    """

    global _version_cache
    _version_cache = None


def describe_knowledge() -> dict:
    """知识库的当前状态，供健康检查和 CLI 展示。

    返回：
        {
            "version": 指纹,
            "file_count": 文件数,
            "files": [{"name","size","sha256"}],
            "total_bytes": 总字节数
        }
    """

    files = _knowledge_files()

    details = []

    for path in files:

        try:
            raw = path.read_text(encoding="utf-8")
            size = len(raw.encode("utf-8"))
        except OSError:
            raw = ""
            size = 0

        details.append(
            {
                "name": path.name,
                "size": size,
                "sha256": hashlib.sha256(
                    _normalize_text(raw).encode("utf-8")
                ).hexdigest()[:16],
            }
        )

    return {
        "version": compute_knowledge_version(),
        "file_count": len(files),
        "files": details,
        "total_bytes": sum(item["size"] for item in details),
    }


# ============================================================
# 快照：保留每个出现过的版本，支持对比与回滚
# ============================================================


def snapshot_path(version: str) -> Path:
    return SNAPSHOT_DIR / version


def ensure_snapshot(version: str | None = None) -> Path | None:
    """把当前知识库复制一份到快照目录。

    为什么在「构建索引时」调用而不是在「检测到变更时」：
        变更发生时，旧内容已经被覆盖了，再想备份已经来不及。
        所以在每次构建索引时顺手存一份当前版本——
        下次变更后，这一份就是「上一版」的快照。

    已经存在就跳过，不重复写。
    """

    version = version or compute_knowledge_version()

    if version in ("", "empty"):
        return None

    target = snapshot_path(version)

    if target.exists():
        return target

    files = _knowledge_files()

    if not files:
        return None

    try:
        target.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": [],
        }

        for path in files:

            raw = path.read_text(encoding="utf-8")

            (target / path.name).write_text(
                _normalize_text(raw),
                encoding="utf-8",
            )

            manifest["files"].append(
                {
                    "name": path.name,
                    "size": len(raw.encode("utf-8")),
                }
            )

        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _prune_snapshots()

        return target

    except OSError:
        # 快照是辅助能力，失败不能影响索引构建和服务启动
        return None


def _prune_snapshots() -> None:
    """只保留最近 MAX_SNAPSHOTS 个快照，超出的按时间删除。"""

    if not SNAPSHOT_DIR.exists():
        return

    entries = []

    for path in SNAPSHOT_DIR.iterdir():

        if not path.is_dir():
            continue

        try:
            entries.append((path.stat().st_mtime, path))
        except OSError:
            continue

    entries.sort(key=lambda item: item[0], reverse=True)

    for _, path in entries[MAX_SNAPSHOTS:]:
        shutil.rmtree(path, ignore_errors=True)


def list_snapshots() -> list[dict]:
    """列出所有快照，最新的在前。"""

    if not SNAPSHOT_DIR.exists():
        return []

    items = []

    for path in SNAPSHOT_DIR.iterdir():

        if not path.is_dir():
            continue

        manifest_file = path / "manifest.json"

        manifest: dict = {}

        if manifest_file.exists():
            try:
                manifest = json.loads(
                    manifest_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                manifest = {}

        items.append(
            {
                "version": path.name,
                "created_at": manifest.get("created_at", "unknown"),
                "file_count": len(manifest.get("files", [])),
                "path": str(path),
            }
        )

    items.sort(key=lambda item: item["created_at"], reverse=True)

    return items


def read_snapshot(version: str) -> dict[str, str]:
    """读取某个快照的全部文件内容，返回 {文件名: 文本}。"""

    target = snapshot_path(version)

    if not target.exists():
        raise FileNotFoundError(f"快照不存在：{version}")

    contents: dict[str, str] = {}

    for path in sorted(target.glob("*.md"), key=lambda p: p.name):
        contents[path.name] = path.read_text(encoding="utf-8")

    return contents


def restore_snapshot(version: str) -> list[str]:
    """把知识库恢复到指定快照。

    这是破坏性操作：会覆盖 data/knowledge/ 下的同名文件。
    调用方（CLI 脚本）必须要求用户显式确认。

    返回：被覆盖的文件名列表。
    """

    contents = read_snapshot(version)

    if not contents:
        raise ValueError(f"快照里没有 markdown 文件：{version}")

    restored = []

    for name, text in contents.items():
        (KNOWLEDGE_DIR / name).write_text(text, encoding="utf-8")
        restored.append(name)

    invalidate_version_cache()

    return restored


def diff_with_snapshot(version: str) -> dict:
    """把当前知识库与某个快照做对比。

    只做「文件级 + 行级摘要」，不做完整 diff：
        这个工具是给「改完想知道动了多少」用的，
        真要逐行看，git diff 更合适。
    """

    snapshot = read_snapshot(version)

    current = {
        path.name: _normalize_text(path.read_text(encoding="utf-8"))
        for path in _knowledge_files()
    }

    result = {"added": [], "removed": [], "changed": [], "unchanged": []}

    for name, text in current.items():

        if name not in snapshot:
            result["added"].append(name)
            continue

        if snapshot[name] == text:
            result["unchanged"].append(name)
        else:
            old_lines = snapshot[name].splitlines()
            new_lines = text.splitlines()
            result["changed"].append(
                {
                    "name": name,
                    "old_lines": len(old_lines),
                    "new_lines": len(new_lines),
                    "line_delta": len(new_lines) - len(old_lines),
                }
            )

    for name in snapshot:
        if name not in current:
            result["removed"].append(name)

    return result
