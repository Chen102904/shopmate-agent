"""ShopMate 统一评测基础设施。

模块：
    common   运行 ID、环境快照、结果落盘
    dataset  JSONL 数据集加载与版本哈希
    metrics  指标计算（分类 / 检索 / 集合 / 序列）
    report   Markdown 报告片段

用法示例：
    from evaluation.common import new_run_id, build_payload, save_result
    from evaluation.dataset import load_dataset, dataset_version
    from evaluation.metrics import per_class_prf, macro_average
"""

__all__ = ["common", "dataset", "metrics", "report"]
