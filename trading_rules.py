"""A 股现货日频交易的板块申报数量规则。

规则集中在这里，资金分配、现金缩放和账本成交校验都调用同一套函数。
卖出允许清仓时一次卖掉全部余额，因此公司行动形成的零股不会被卡住。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from backtest.errors import ExecutionError

RULE_VERSION = "cn_a_board_v1"
# 订单状态机口径：V57 起停牌开关关闭时只静默顺延，不再提前记为成交。
ORDER_RULE_VERSION = "cn_a_execution_v2"
BOARD_MAIN = "主板"
BOARD_CHINEXT = "创业板"
BOARD_STAR = "科创板"
BOARD_BSE = "北交所"
VALID_BOARDS = (BOARD_MAIN, BOARD_CHINEXT, BOARD_STAR, BOARD_BSE)
BSE_TRADING_START = "2021-11-15"

EPSILON = 1e-9


@dataclass(frozen=True)
class BoardRule:
    """一个板块的买入最低量、递增单位和卖出申报限制。"""

    board: str
    min_buy_shares: int
    buy_increment: int
    min_sell_shares: int
    sell_increment: int

    def _whole(self, value: float) -> int:
        """把股数转成整数，非整数或负数直接报错。"""
        number = float(value)
        rounded = int(round(number))
        if number < -EPSILON or abs(number - rounded) > EPSILON:
            raise ExecutionError(f"{self.board} 的股数必须是整股：{value}")
        return max(rounded, 0)

    def _floored(self, value: float) -> int:
        """把金额或预算换算出的理想股数向下取整。"""
        number = float(value)
        if number < -EPSILON:
            raise ExecutionError(f"{self.board} 的股数不能为负：{value}")
        return max(int(math.floor(number + EPSILON)), 0)

    def is_valid_buy(self, quantity: float) -> bool:
        """判断买入申报量是否符合板块规则。"""
        shares = self._whole(quantity)
        return (
            shares >= self.min_buy_shares
            and (shares - self.min_buy_shares) % self.buy_increment == 0
        )

    def is_valid_sell(self, quantity: float, held: float) -> bool:
        """判断卖出申报量是否符合板块规则；清仓允许卖出全部实际余额。"""
        shares = self._whole(quantity)
        position = self._whole(held)
        if shares <= 0 or shares > position:
            return False
        if shares == position:
            return True
        return (
            shares >= self.min_sell_shares
            and (shares - self.min_sell_shares) % self.sell_increment == 0
        )

    def floor_buy_quantity(self, quantity: float) -> int:
        """把买入申报量向下取整到合法单位。"""
        shares = self._floored(quantity)
        if shares < self.min_buy_shares:
            return 0
        return self.min_buy_shares + (
            (shares - self.min_buy_shares) // self.buy_increment
        ) * self.buy_increment

    def floor_sell_quantity(self, quantity: float, held: float) -> int:
        """把卖出申报量向下取整到合法单位；超过持仓时按清仓处理。"""
        shares = self._floored(quantity)
        position = self._whole(held)
        if shares <= 0 or position <= 0:
            return 0
        if shares >= position:
            return position
        if shares < self.min_sell_shares:
            return 0
        return self.min_sell_shares + (
            (shares - self.min_sell_shares) // self.sell_increment
        ) * self.sell_increment

    def normalize_target(self, held: float, desired: float) -> int:
        """把理想目标持仓改成当天可以下单达到的合法目标持仓。"""
        position = self._whole(held)
        target = self._floored(desired)
        if target > position:
            quantity = self.floor_buy_quantity(target - position)
            return position + quantity
        if target < position:
            if target == 0:
                return 0
            quantity = self.floor_sell_quantity(position - target, position)
            return position - quantity
        return position

    def cap_target(self, held: float, target: float, max_shares: float) -> int:
        """把目标持仓限制在最大可成交股数以内，并保持当前持仓的零股。"""
        position = self._whole(held)
        normalized = self.normalize_target(position, target)
        limit = self._floored(max_shares)
        delta = normalized - position
        if abs(delta) <= limit:
            return normalized
        if delta > 0:
            return position + self.floor_buy_quantity(limit)
        quantity = self.floor_sell_quantity(limit, position)
        return position - quantity

    def previous_buy_target(self, held: float, target: float) -> int:
        """返回比当前买单低一档的合法目标持仓。"""
        position = self._whole(held)
        normalized = self.normalize_target(position, target)
        quantity = normalized - position
        if quantity <= 0:
            return position
        next_quantity = quantity - self.buy_increment
        if next_quantity < self.min_buy_shares:
            return position
        return position + next_quantity

    def is_valid_order(self, held: float, signed_quantity: float) -> bool:
        """判断一笔有符号订单是否合法；零股订单不算成交。"""
        quantity = float(signed_quantity)
        if abs(quantity) <= EPSILON:
            return True
        if quantity > 0:
            return self.is_valid_buy(quantity)
        return self.is_valid_sell(-quantity, held)


BOARD_RULES = {
    BOARD_MAIN: BoardRule(BOARD_MAIN, 100, 100, 100, 100),
    BOARD_CHINEXT: BoardRule(BOARD_CHINEXT, 100, 100, 100, 100),
    BOARD_STAR: BoardRule(BOARD_STAR, 200, 1, 200, 1),
    BOARD_BSE: BoardRule(BOARD_BSE, 100, 1, 100, 1),
}


def rule_for(board: str) -> BoardRule:
    """按板块名称返回交易规则，未知板块直接报错。"""
    try:
        return BOARD_RULES[str(board)]
    except KeyError as error:
        raise ExecutionError(
            f"未知板块 {board!r}；可用板块为 {list(VALID_BOARDS)}"
        ) from error


def board_from_stock_code(stock_code: str) -> str:
    """按标准股票代码推断板块，仅供测试夹具和无基础表的老工具使用。"""
    code = str(stock_code).upper()
    if code.startswith("BJ"):
        return BOARD_BSE
    if code.startswith(("SH688", "SH689")):
        return BOARD_STAR
    if code.startswith(("SZ300", "SZ301", "SZ302")):
        return BOARD_CHINEXT
    if code.startswith(("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")):
        return BOARD_MAIN
    raise ExecutionError(f"无法从股票代码推断板块：{stock_code}")


def rule_summary() -> dict[str, dict[str, int]]:
    """返回写入运行元数据的稳定规则摘要。"""
    return {
        board: {
            "min_buy_shares": rule.min_buy_shares,
            "buy_increment": rule.buy_increment,
            "min_sell_shares": rule.min_sell_shares,
            "sell_increment": rule.sell_increment,
        }
        for board, rule in BOARD_RULES.items()
    }
