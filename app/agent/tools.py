
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "根据订单编号查询订单信息",
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
            "description": "根据订单编号查询物流信息",
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
            "description": "根据订单编号检查订单是否符合退款条件",
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
            "name": "search_after_sales_policy",
            "description": "检索售后政策知识库，回答退款政策、退货条件、换货规则、物流异常处理和人工客服审核规则等问题",
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
