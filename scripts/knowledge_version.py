"""知识库版本管理命令行工具。

用法：
    python scripts/knowledge_version.py show           # 当前版本与文件清单
    python scripts/knowledge_version.py list           # 历史快照
    python scripts/knowledge_version.py diff <版本>     # 与某个快照对比
    python scripts/knowledge_version.py restore <版本>  # 回滚（需 --yes）

为什么需要命令行工具而不是只留接口：

    改知识库的人不一定是程序员。
    「我改了三章政策，想知道跟上一版差多少」
    这件事应该有一条命令能回答，
    而不是让人去翻 data/knowledge_versions/ 目录。

回滚为什么要 --yes：

    它会直接覆盖 data/knowledge/ 下的文件。
    知识库是检索的唯一数据源，覆盖错了服务就开始答错，
    而且没有「撤销」——只能再 restore 回来。
    所以必须显式确认。
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag.knowledge_version import (  # noqa: E402
    compute_knowledge_version,
    describe_knowledge,
    diff_with_snapshot,
    list_snapshots,
    restore_snapshot,
)


def _cmd_show() -> None:
    info = describe_knowledge()

    print("=" * 60)
    print(f"当前知识库版本：{info['version']}")
    print(f"文件数：{info['file_count']}   总字节：{info['total_bytes']}")
    print("=" * 60)

    for item in info["files"]:
        print(f"  {item['name']:<32} {item['size']:>7} B   sha256:{item['sha256']}")

    print()
    print("说明：版本号是内容指纹，只要有一个字符变化就会变。")
    print("      回答、日志、评测结果都会带上它，用来确认「基于哪一版知识」。")


def _cmd_list() -> None:
    items = list_snapshots()

    if not items:
        print("还没有任何快照。")
        print("快照会在构建索引时自动生成（启动服务或跑一次检索即可）。")
        return

    current = compute_knowledge_version()

    print(f"当前版本：{current}")
    print()
    print(f"{'版本':<16}{'创建时间':<22}{'文件数':<8}")
    print("-" * 48)

    for item in items:
        mark = "  ← 当前" if item["version"] == current else ""
        print(
            f"{item['version']:<16}"
            f"{item['created_at']:<22}"
            f"{item['file_count']:<8}{mark}"
        )


def _cmd_diff(version: str) -> None:
    try:
        result = diff_with_snapshot(version)
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        print("用 `list` 查看有哪些版本。")
        return

    current = compute_knowledge_version()

    print(f"对比：{version}（旧） → {current}（当前）")
    print()

    if result["added"]:
        print("新增文件：")
        for name in result["added"]:
            print(f"  + {name}")

    if result["removed"]:
        print("删除文件：")
        for name in result["removed"]:
            print(f"  - {name}")

    if result["changed"]:
        print("内容有变化：")
        for item in result["changed"]:
            delta = item["line_delta"]
            arrow = f"+{delta}" if delta > 0 else str(delta)
            print(
                f"  ~ {item['name']:<30} "
                f"{item['old_lines']} → {item['new_lines']} 行（{arrow}）"
            )

    if result["unchanged"]:
        print(f"未变化：{len(result['unchanged'])} 个文件")

    print()
    print("提示：行数变化只是摘要。要看具体改了什么，用：")
    print(f"  git diff -- data/knowledge/")
    print("  （前提是改动还没被 git 提交）")


def _cmd_restore(version: str, confirmed: bool) -> None:
    items = {item["version"] for item in list_snapshots()}

    if version not in items:
        print(f"错误：没有这个快照：{version}")
        print("用 `list` 查看可用版本。")
        return

    if not confirmed:
        print("⚠️  这个操作会覆盖 data/knowledge/ 下的同名文件，且无法撤销。")
        print(f"   目标版本：{version}")
        print()
        print("确认无误后重新执行，并加上 --yes：")
        print(f"   python scripts/knowledge_version.py restore {version} --yes")
        return

    restored = restore_snapshot(version)

    print(f"已恢复到 {version}，覆盖 {len(restored)} 个文件：")
    for name in restored:
        print(f"  {name}")

    print()
    print(f"新版本号：{compute_knowledge_version()}")
    print("提示：如果服务正在运行且开启了热重载，下一次检索会自动重建索引；")
    print("      否则需要重启服务。")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print(__doc__)
        return

    command = args[0]

    if command == "show":
        _cmd_show()
        return

    if command == "list":
        _cmd_list()
        return

    if command == "diff":
        if len(args) < 2:
            print("用法：python scripts/knowledge_version.py diff <版本>")
            return
        _cmd_diff(args[1])
        return

    if command == "restore":
        if len(args) < 2:
            print("用法：python scripts/knowledge_version.py restore <版本> --yes")
            return
        _cmd_restore(args[1], confirmed="--yes" in sys.argv)
        return

    print(f"未知命令：{command}")
    print(__doc__)


if __name__ == "__main__":
    main()
