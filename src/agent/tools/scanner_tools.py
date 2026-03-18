# -*- coding: utf-8 -*-
"""
全市场扫描工具 - 注册为 Agent 可调用的工具
"""

import logging
from .registry import tool, ToolParameter

logger = logging.getLogger(__name__)


@tool(
    name="scan_strong_stocks",
    description=(
        "全市场扫描A股，支持三种策略："
        "1) trend - 趋势股：股价在MA20上方且MA5>=MA10；"
        "2) dip - 跌停低吸：近10天内有2-3个跌停后企稳反弹，适合低吸回调；"
        "3) oversold - 超跌反弹：RSI超卖+大幅回撤+缩量企稳；"
        "4) all - 同时运行以上三个策略。"
        "当用户要求推荐股票、选股、扫描市场、找强势股、找跌停低吸、找超跌反弹时使用此工具。"
    ),
    category="analysis",
    parameters=[
        ToolParameter(
            name="mode",
            type="string",
            description="扫描模式：all(全部策略)/trend(趋势)/dip(跌停低吸)/oversold(超跌反弹)，默认all",
            required=False,
            default="all",
        ),
        ToolParameter(
            name="top_n",
            type="integer",
            description="返回前N只股票（默认15）",
            required=False,
            default=15,
        ),
        ToolParameter(
            name="max_cap",
            type="number",
            description="最大流通市值（亿元，默认300）",
            required=False,
            default=300,
        ),
        ToolParameter(
            name="max_bias",
            type="number",
            description="最大乖离率（%，默认8）",
            required=False,
            default=8.0,
        ),
        ToolParameter(
            name="min_turnover",
            type="number",
            description="最小换手率（%，默认1）",
            required=False,
            default=1.0,
        ),
    ],
)
def scan_strong_stocks(
    mode: str = "all",
    top_n: int = 15,
    max_cap: float = 300,
    max_bias: float = 8.0,
    min_turnover: float = 1.0,
) -> dict:
    """全市场多策略扫描"""
    try:
        import sys
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        from scanner import scan_market

        logger.info(f"开始全市场扫描: mode={mode}, top={top_n}, 市值<{max_cap}亿")

        candidates = scan_market(
            max_cap=max_cap,
            min_turnover=min_turnover,
            max_bias=max_bias,
            top_n=top_n,
            mode=mode,
        )

        if not candidates:
            return {
                "success": True,
                "count": 0,
                "message": f"当前市场未找到符合条件的股票（模式：{mode}）",
                "stocks": [],
            }

        stocks = []
        codes = []
        for s in candidates:
            stock_info = {
                "code": s["代码"],
                "name": s["名称"],
                "price": s["现价"],
                "change_pct": s["涨跌幅"],
                "turnover": s["换手率"],
                "market_cap": s["流通市值(亿)"],
                "strategy": s.get("策略", ""),
                "bias_pct": s.get("乖离率", 0),
            }
            # 跌停低吸特有字段
            if "跌停次数" in s:
                stock_info["limit_down_count"] = s["跌停次数"]
                stock_info["rsi6"] = s.get("RSI6", 0)
                stock_info["bounce_pct"] = s.get("反弹幅度", 0)
                stock_info["vol_shrink"] = s.get("缩量企稳", "否")
                stock_info["score"] = s.get("综合评分", 0)
            # 超跌反弹特有字段
            if "20日跌幅" in s:
                stock_info["rsi6"] = s.get("RSI6", 0)
                stock_info["drop_20d"] = s.get("20日跌幅", 0)
                stock_info["score"] = s.get("综合评分", 0)

            stocks.append(stock_info)
            codes.append(s["代码"])

        logger.info(f"扫描完成，找到 {len(stocks)} 只: {codes}")

        return {
            "success": True,
            "count": len(stocks),
            "message": f"找到 {len(stocks)} 只符合条件的股票（模式：{mode}）",
            "codes_list": ",".join(codes),
            "stocks": stocks,
        }

    except Exception as e:
        logger.error(f"全市场扫描失败: {e}")
        return {
            "success": False,
            "count": 0,
            "message": f"扫描失败: {str(e)}",
            "stocks": [],
        }


ALL_SCANNER_TOOLS = [scan_strong_stocks._tool_definition]