"""用户确认意图判定。

为什么需要这个模块：

    光有「草稿 + 确认」两步还不够。
    两步之间的那一步——模型判断「用户是不是同意了」——
    仍然完全由模型做主。

    现实里会翻车的情况：
        用户：「建个工单要多久？」     ← 问耗时，不是要建
        模型：调 confirm            ← 理解错，直接建了一张单

        用户：「先别建，我再看看」     ← 明确拒绝
        模型：看到「建」字就调 confirm

    这类错误的代价是真实的：工单会进人工审核队列，
    撤销要走线下流程。

    所以再加一道确定性闸门：
        confirm 执行前，检查「本会话最近一条用户消息」
        是否含明确的确认表达，且不含否定表达。

    为什么闸门放在服务端而不是只靠提示词：
        提示词是建议，模型有权不遵守。
        代码里的 if 是规则，模型绕不过去。

设计取舍：
    - 否定优先：一条消息里同时出现「不要」和「确认」，
      比如「不要确认」，必须判为拒绝。
    - 宁可漏判为「未确认」（让模型再问一遍），
      也不误判为「已确认」（建错单）。
      所以词表偏保守，且要求消息足够短才认单字确认。
    - 单字确认（「好」「是」「行」）只在消息很短时才算数，
      避免「好的，那退货要多久」被当成确认。
"""

import re


# 明确确认表达。
# 覆盖口头同意、书面同意、行动指令三类。
CONFIRM_MARKERS = (
    "确认",
    "确定",
    "同意",
    "提交",
    "就这样",
    "没问题",
    "可以的",
    "可以",
    "麻烦你",
    "帮我建",
    "帮我提交",
    "帮我申请",
    "ok",
    "okay",
    "yes",
    "sure",
)

# 短消息里才算数的单字/短词确认。
# 长消息里出现「好」大概率是语气词，不是同意。
SHORT_CONFIRM_MARKERS = (
    "好",
    "行",
    "是",
    "对",
    "嗯",
    "建吧",
    "退吧",
    "办吧",
)

# 否定表达：出现即判拒绝，优先级高于一切确认词。
# 注意这里不包含单字「不」：
#   「不错，就这样」里的「不」不是否定，
#   用「不要 / 不用 / 不建」这类多字词才能避免误伤。
#
# 「别」保留单字形式：
#   它会把「别人」「特别」也一起判成否定，
#   但那两种情况的后果只是「多问用户一遍」，
#   比漏判成确认、直接建单要轻得多。
DENY_MARKERS = (
    "不要",
    "不用",
    "别",
    "取消",
    "算了",
    "先不",
    "暂不",
    "不建",
    "不申请",
    "不确认",
    "不同意",
    "还没",
    "没想好",
    "撤销",
    "再想想",
    "再看看",
    "等一下",
    "先看看",
)

# 「短消息」的门槛：去掉标点后不超过这个字数，才认单字确认。
# 6 个字大约覆盖「好的」「嗯嗯知道了」「行吧麻烦你」这类真确认，
# 又能挡住「好，那退货要多久」这种其实是提问的长句。
SHORT_MESSAGE_MAX_CHARS = 6

# 疑问标记：出现即判「不是确认」。
#
# 为什么需要这一条：
#   「好，那退货要多久」以「好」开头，只看前缀会误判为确认，
#   但它明明是在问问题。
#   只要句子里有疑问语气，就不可能是同意。
QUESTION_MARKERS = (
    "?",
    "？",
    "吗",
    "多久",
    "多长时间",
    "怎么",
    "多少",
    "能不能",
    "可不可以",
    "是不是",
    "要不要",
    "为什么",
    "什么",
    "哪",
)

# 标点集合，用于计算「有效字数」。
_PUNCTUATION = r"，。,.!！、；;：:~～…—-_\t "

# 否定前缀字符。
#
# 为什么不能只靠 DENY_MARKERS 列举：
#   「不确定」「不太确定」「没确认」「不好」这些词的否定字
#   紧贴在确认词前面，组合几乎是无穷的，列举不完。
#   所以改成规则：确认词往前看两个字，出现否定字就判否决。
#
#   这和做矛盾检测时「不支持退款」包含「支持退款」是同一类问题，
#   解法也一样——看前缀，不看子串。
NEGATION_PREFIX_CHARS = "不没别非勿莫"


def _normalize(message: str) -> str:
    """归一化：去空白、转小写，便于匹配中英混写。"""

    text = (message or "").strip().lower()

    # 全角空格和零宽字符会让「确认」两个字匹配不上
    text = re.sub(r"[\s\u200b-\u200f\ufeff]+", "", text)

    return text


def _effective_length(text: str) -> int:
    """去掉标点后的字数。"""

    return len([ch for ch in text if ch not in _PUNCTUATION])


def _negated_marker(text: str) -> str | None:
    """找出「被否定的确认词」。

    逐个定位每个确认词，看它前面两个字里有没有否定字。
    命中就返回那个确认词，用于日志。

    例：
        「不确定」   → 确认词「确定」前有「不」 → 命中
        「不太确定」 → 确认词「确定」前有「不太」 → 命中（只看前两位）
        「不好」     → 确认词「好」前有「不」   → 命中
        「没问题」   → 确认词「没问题」前无字   → 不命中（这是真确认）
    """

    for marker in CONFIRM_MARKERS + SHORT_CONFIRM_MARKERS:

        start = 0

        while True:

            index = text.find(marker, start)

            if index == -1:
                break

            window = text[max(0, index - 2):index]

            if any(ch in window for ch in NEGATION_PREFIX_CHARS):
                return marker

            start = index + 1

    return None


def is_explicit_confirmation(
    message: str,
    ignore_deny: tuple[str, ...] = (),
    extra_confirm: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """判断用户这条消息是否构成明确确认。

    参数：
        ignore_deny: 本次判定里**不算否定**的词。
        extra_confirm: 本次判定里**额外算确认**的词。

    为什么要这两个参数：

        「撤销」这个词在不同语境下语义相反：
            建单语境：用户说「撤销」= 我反悔了，别建（否定）
            撤销语境：用户说「撤销吧」= 同意撤销（肯定）

        同一套词表直接复用会拦错：
            cancel 工具里，用户明说「确认撤销」，
            却因为命中 DENY_MARKERS 里的「撤销」被判成拒绝。

        所以由调用方声明本次语境的例外，
        而不是把词表拆成两套——拆开之后两边会逐渐漂移。

    返回：
        (True,  "matched:确认")  → 可以执行写操作
        (False, "denied:不要")   → 用户明确拒绝
        (False, "negated:确定")  → 「不确定」这类被否定的确认词
        (False, "question")      → 这是个问句，不是同意
        (False, "no_marker")     → 没有确认表达，需要再问

    第二个返回值只用于日志和调试，不返回给模型，
    避免模型（或用户）看到内部判定逻辑后针对性绕过。

    判定顺序（顺序本身就是设计）：
        否定词 > 疑问 > 被否定的确认词 > 明确确认词 > 短消息单字确认

    为什么疑问要排在「被否定的确认词」前面：
        「是不是提交了就退钱」里的「不是」会命中否定前缀检查，
        但这句话本身是疑问句，不是否定表态。
        先按疑问排除，判定原因才准确（也才好排查）。
    """

    text = _normalize(message)

    if not text:
        return False, "empty"

    # 1. 否定词优先。
    #    「不要确认」同时含否定词和确认词，必须判拒绝。
    for marker in DENY_MARKERS:
        if marker in ignore_deny:
            continue
        if marker in text:
            return False, f"denied:{marker}"

    # 2. 疑问其次。
    #    「确认吗？」「好的，那要多久」都是提问，不是同意。
    for marker in QUESTION_MARKERS:
        if marker in text:
            return False, f"question:{marker}"

    # 3. 被否定的确认词。
    #    「不确定」「不太确定」「没确认」都落在这一条。
    negated = _negated_marker(text)

    if negated:
        return False, f"negated:{negated}"

    # 4. 明确确认词（含本次语境额外允许的）
    for marker in CONFIRM_MARKERS + tuple(extra_confirm):
        if marker in text:
            return True, f"matched:{marker}"

    # 5. 短消息里的单字确认。
    #    只在消息本身就短的时候认，避免长句开头的「好」被误判。
    if _effective_length(text) <= SHORT_MESSAGE_MAX_CHARS:
        for marker in SHORT_CONFIRM_MARKERS:
            if text.startswith(marker):
                return True, f"matched_short:{marker}"

    return False, "no_marker"
