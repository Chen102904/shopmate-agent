
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": (
                "根据订单编号查询订单基础信息：订单状态、金额、下单时间。"
                "仅当用户明确询问订单状态或金额时调用。"
                "检查退款资格不需要先调用本工具，"
                "check_refund_eligibility 内部已包含订单状态查询。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logistics",
            "description": (
                "根据订单编号查询物流信息：承运公司、运单号、运输状态、签收时间。"
                "仅当用户询问快递进度、物流状态、签收情况时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD002",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": (
                "根据订单编号检查该订单是否符合退款条件，"
                "返回是否可退、原因和机器可读状态。"
                "本工具内部会自动读取订单状态和实际签收时间，"
                "因此不需要先调用 get_order，也不需要先调用 get_logistics。"
                "判断某个具体订单能不能退时，只调用本工具即可。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_after_sales_ticket",
            "description": (
                "为售后申请创建一张待确认草稿。本工具不会真的建单，"
                "只是把参数冻结下来并返回一个 draft_id 供后续确认。\n"
                "什么时候用：用户明确要求申请退款、退货、换货，"
                "或者你判断需要为用户提交售后申请时。\n"
                "调用后你必须把草稿内容（订单号、申请类型、理由）"
                "完整复述给用户，并明确询问是否确认提交。\n"
                "严禁在同一轮里紧接着调用 confirm_after_sales_ticket——"
                "必须先让用户看到内容并亲口同意。\n"
                "如果用户还没说清楚要退还是要换、或者没给理由，"
                "先用文字追问，不要猜。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD001",
                    },
                    "request_type": {
                        "type": "string",
                        "enum": ["refund", "return", "exchange"],
                        "description": (
                            "申请类型：refund=退款，return=退货，exchange=换货"
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "申请理由，必须来自用户的原话或确认过的事实，"
                            "例如：商品破损、尺码不合适、少发一件。"
                            "不要自行编造理由。"
                        ),
                    },
                },
                "required": ["order_id", "request_type", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_after_sales_ticket",
            "description": (
                "确认并正式提交之前生成的售后申请草稿。"
                "这是一个写操作，会在系统里真正创建工单并进入人工审核。\n"
                "严格前置条件：用户已经在上一轮看到了草稿内容，"
                "并且用明确的语言表示同意（例如「确认」「可以」「提交吧」）。\n"
                "以下情况绝对不能调用：\n"
                "- 用户只是在询问流程、耗时、条件；\n"
                "- 用户说「先不」「算了」「再想想」；\n"
                "- 你还没有把草稿内容展示给用户。\n"
                "draft_id 可以省略：省略时系统自动使用本会话最近一张"
                "待确认的草稿。不要自己编造 draft_id，也不要为了拿编号"
                "重新生成草稿。订单号和申请类型已冻结在草稿里，"
                "你不要也不能在这一步修改它们。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": (
                            "草稿编号，可省略。"
                            "只有你确实记得草稿编号时才传；"
                            "省略时会自动使用最近一张待确认草稿。"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_after_sales_tickets",
            "description": (
                "查询用户的售后工单进度。提交申请之后，用户会追问"
                "「我那个退款怎么样了」「工单审核到哪一步了」，用这个工具回答。\n"
                "参数 keyword 可以填订单号（ORD001）或工单号（UUID），"
                "也可以不填——不填时返回最近的工单。\n"
                "返回值里还会带上 pending_draft：如果用户有一张生成了但"
                "还没确认提交的草稿，这里能看到。\n"
                "注意：查不到不等于用户没申请过——可能是草稿没提交，"
                "要看 pending_draft 再回答。绝对不要编造工单号或审核结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "订单号或工单号。不确定时可以不传，"
                            "系统会返回该用户最近的工单。"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_after_sales_ticket",
            "description": (
                "撤销用户已经提交的售后申请。这是有副作用的操作："
                "工单会变成「已取消」，之后不再进入审核。\n"
                "什么时候用：用户明确表示不想申请了、要撤回。\n"
                "什么时候不能用：\n"
                "- 用户只是催进度、问状态、抱怨太慢——那是查询，不是撤销；\n"
                "- 用户没说清楚要撤销哪一张。\n"
                "调用后系统会校验用户是否明确确认过。如果返回 "
                "confirmation_required，说明用户还没确认："
                "把 hint 里给出的「要撤销哪张单、撤销后什么后果」告诉用户，"
                "等用户回复「确认」「撤销吧」之后再调用一次。\n"
                "不要自己编造工单号——不传参数时会自动用最近一张待审核的申请。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "工单编号（UUID），可省略",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "订单编号例如 ORD001，可省略",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_after_sales_policy",
            "description": (
                "检索售后政策知识库，回答退款政策、退货条件、换货规则、"
                "物流异常处理、退款到账时间、运费承担、特殊商品规则、"
                "人工客服审核规则等问题。"
                "问的是「规则本身」时用它；"
                "问的是「我这个订单能不能退」时用 check_refund_eligibility。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户想查询的售后政策问题，例如：商品签收超过7天还能退款吗",
                    }
                },
                "required": ["query"],
            },
        },
    },
]
