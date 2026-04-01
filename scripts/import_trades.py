"""
一次性脚本：从 E:\\table.xls 导入历史交易记录，然后 FIFO 同步持仓。

用法:
    cd D:\\华尔街之狼
    python scripts/import_trades.py
"""
import sys, os

# 确保项目根目录在 path 里
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trade_journal import import_trades_from_file
from portfolio_manager import load_portfolio, sync_portfolio_from_trades, save_portfolio

XLS_PATH = r"E:\table.xls"


def main():
    print(f"=== 导入交易记录: {XLS_PATH} ===")
    result = import_trades_from_file(XLS_PATH, source="broker_export")
    print(f"导入: {result['imported']} 条")
    print(f"跳过: {result['skipped']} 条")
    if result["errors"]:
        print(f"错误 ({len(result['errors'])}):")
        for e in result["errors"][:10]:
            print(f"  - {e}")

    if result["imported"] > 0:
        print("\n=== FIFO 同步持仓 ===")
        portfolio = load_portfolio()
        portfolio = sync_portfolio_from_trades(portfolio)
        save_portfolio(portfolio)
        print(f"持仓更新完成，共 {len(portfolio.get('holdings', []))} 只")
        for h in portfolio.get("holdings", []):
            print(f"  {h['name']}({h['code']}): {h['shares']}股 "
                  f"成本{h['cost_price']} 买入日{h.get('buy_date', '?')}")


if __name__ == "__main__":
    main()
