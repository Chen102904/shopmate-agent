"""确认意图判定测试。

这个模块看起来只是几个关键词匹配，但它守的是「写操作能不能落库」这道门，
所以每一条边界都要有对应用例：
    判松了 → 模型误建工单
    判严了 → 用户说了「确认」还是建不了，体验崩

用例分组：
    1. 正常确认（必须通过）
    2. 否定（必须拦）
    3. 被否定的确认词（「不确定」这类，最容易被关键词匹配漏掉）
    4. 疑问句（看着像确认，其实在问问题）
    5. 无关消息
"""

import pytest

from app.agent.confirmation import (
    is_explicit_confirmation,
    _normalize,
    _effective_length,
)


# ============================================================
# 1. 正常确认
# ============================================================


@pytest.mark.parametrize(
    "message",
    [
        "确认",
        "确认提交",
        "可以",
        "可以，帮我提交吧",
        "好的",
        "好",
        "行",
        "嗯",
        "是",
        "同意",
        "就这样吧",
        "没问题",
        "麻烦你了",
        "ok",
        "OK",
        "yes",
    ],
)
def test_explicit_confirmation_passes(message):
    """这些都应该被判为明确确认。"""

    confirmed, reason = is_explicit_confirmation(message)

    assert confirmed is True, f"{message!r} 应判确认，实际 {reason}"


# ============================================================
# 2. 否定
# ============================================================


@pytest.mark.parametrize(
    "message",
    [
        "先别建",
        "算了",
        "不用了",
        "不要",
        "取消吧",
        "我再想想",
        "先不提交",
        "暂不申请",
        "撤销",
    ],
)
def test_denial_blocks(message):
    """用户明确拒绝时必须拦下，即使句子里含确认词。"""

    confirmed, reason = is_explicit_confirmation(message)

    assert confirmed is False, f"{message!r} 应判拒绝，实际 {reason}"
    assert reason.startswith("denied:")


def test_denial_wins_over_confirmation_word():
    """「不要确认」同时含否定词和确认词，否定优先。

    这是最容易写错的一条：
    如果先匹配「确认」再匹配「不要」，
    用户说「不要确认」会变成「确认」——后果是直接建单。
    """

    confirmed, reason = is_explicit_confirmation("不要确认")

    assert confirmed is False
    assert reason.startswith("denied:")


# ============================================================
# 3. 被否定的确认词
# ============================================================


@pytest.mark.parametrize(
    "message",
    [
        "不确定",
        "不太确定",
        "没确认",
        "不好",
        "不确认",
        "我还不确定要不要",
    ],
)
def test_negated_confirmation_words_are_blocked(message):
    """「不确定」含子串「确定」，不能因此判成确认。

    这个问题在矛盾检测里遇到过一模一样的：
    「不支持退款」包含「支持退款」。
    解法一致——看否定前缀，不看子串。
    """

    confirmed, reason = is_explicit_confirmation(message)

    assert confirmed is False, f"{message!r} 应判拒绝，实际 {reason}"


def test_negative_phrase_without_negation_is_confirmation():
    """「没问题」是确认，不能因为含「没」就被当否定。

    否定词判定要落在「确认词的前缀」上，
    而不是「整句里出现过否定字」。
    """

    confirmed, _ = is_explicit_confirmation("没问题")

    assert confirmed is True


# ============================================================
# 4. 疑问句
# ============================================================


@pytest.mark.parametrize(
    "message",
    [
        "建个工单要多久",
        "好，那退货要多久",
        "可以吗",
        "确认吗？",
        "这样就行吗",
        "能帮我看看要多少钱吗",
        "是不是提交了就退钱",
    ],
)
def test_questions_are_not_confirmation(message):
    """问句不是同意。

    「好，那退货要多久」以「好」开头，
    只看前缀会误判成确认——这条是最典型的坑。
    """

    confirmed, reason = is_explicit_confirmation(message)

    assert confirmed is False, f"{message!r} 应判非确认，实际 {reason}"
    assert reason.startswith("question:")


# ============================================================
# 5. 无确认表达
# ============================================================


@pytest.mark.parametrize(
    "message",
    [
        "ORD001 帮我退款",
        "商品破损了",
        "",
        "   ",
        "那个订单怎么样了",
    ],
)
def test_no_marker_blocks(message):
    """没有确认表达就要求再问一遍。

    宁可多问一句，也不要赌模型的理解。
    """

    confirmed, _ = is_explicit_confirmation(message)

    assert confirmed is False


# ============================================================
# 归一化工具
# ============================================================


def test_normalize_strips_zero_width_characters():
    """零宽字符会让「确认」两个字匹配不上，必须先清掉。"""

    assert _normalize("确\u200b认") == "确认"


def test_effective_length_ignores_punctuation():
    assert _effective_length("好的，") == 2
    assert _effective_length("确认。") == 2
