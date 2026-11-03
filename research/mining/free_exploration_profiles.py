"""自由探索的固定机制类别与字段依赖模板。"""

from __future__ import annotations

from dataclasses import dataclass

from mining.contract_types import sha256_json

FREE_EXPLORATION_PROFILE_VERSION = "mining_free_exploration_profiles_v1"


@dataclass(frozen=True)
class MechanismProfile:
    """一种可解释的初始研究机制。"""

    category: str
    fields: tuple[str, ...]
    statement_template: str
    mechanism: str
    expected_direction: str
    falsification_criteria: tuple[str, ...]

    def material(self) -> dict[str, object]:
        """返回固定配置哈希输入。"""

        return {
            "category": self.category,
            "fields": list(self.fields),
            "statement_template": self.statement_template,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
        }


MECHANISM_PROFILES = (
    MechanismProfile(
        category="price_trend",
        fields=("close", "ret", "vwap"),
        statement_template="{label}的趋势延续可能带来未来超额收益",
        mechanism="投资者对新信息反应不足，价格可能需要逐步吸收信息。",
        expected_direction="positive",
        falsification_criteria=(
            "开发期年度同号率未达到冻结门槛",
            "连续多轮修订后开发期质量仍未改善",
        ),
    ),
    MechanismProfile(
        category="short_reversal",
        fields=("close", "ret"),
        statement_template="{label}的短期反转可能带来未来超额收益",
        mechanism="短期交易冲击可能造成价格偏离，随后存在修复。",
        expected_direction="negative",
        falsification_criteria=(
            "开发期方向与预期相反且没有稳定改善",
            "连续多轮修订后开发期质量仍未改善",
        ),
    ),
    MechanismProfile(
        category="volatility",
        fields=("close", "high", "low", "ret"),
        statement_template="{label}波动上升可能对应更低的风险调整收益",
        mechanism="高波动可能反映更高的不确定性并压低风险补偿。",
        expected_direction="negative",
        falsification_criteria=(
            "开发期波动关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
    MechanismProfile(
        category="liquidity",
        fields=("amount_cny", "circ_mv", "turnover_rate", "volume_share"),
        statement_template="{label}变化可能通过流动性溢价影响未来收益",
        mechanism="流动性状态变化可能改变交易成本、需求冲击和风险补偿。",
        expected_direction="negative",
        falsification_criteria=(
            "开发期流动性关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
    MechanismProfile(
        category="valuation",
        fields=("dv_ttm", "pb", "pe_ttm", "ps_ttm"),
        statement_template="{label}估值水平可能通过估值修复影响未来收益",
        mechanism="估值偏离可能反映市场预期差异，并在后续逐步修复。",
        expected_direction="positive",
        falsification_criteria=(
            "开发期估值关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
    MechanismProfile(
        category="profitability",
        fields=("fi_grossprofit_margin", "fi_roa", "fi_roe", "fi_roic"),
        statement_template="{label}所反映的盈利能力可能带来未来超额收益",
        mechanism="持续盈利能力可能反映竞争壁垒和更稳定的现金创造能力。",
        expected_direction="positive",
        falsification_criteria=(
            "开发期盈利关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
    MechanismProfile(
        category="growth",
        fields=(
            "fi_netprofit_yoy",
            "fi_op_yoy",
            "fi_or_yoy",
            "fi_tr_yoy",
        ),
        statement_template="{label}所反映的增长变化可能带来未来超额收益",
        mechanism="增长预期调整可能尚未完全反映到价格中。",
        expected_direction="positive",
        falsification_criteria=(
            "开发期增长关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
    MechanismProfile(
        category="cashflow_quality",
        fields=(
            "cf_free_cashflow",
            "cf_n_cash_flows_fnc_act",
            "cf_n_cashflow_act",
        ),
        statement_template="{label}所反映的现金流质量可能带来未来超额收益",
        mechanism="现金流质量可能反映盈利兑现能力和财务稳健程度。",
        expected_direction="positive",
        falsification_criteria=(
            "开发期现金流关系方向不稳定",
            "开发期质量未达到冻结门槛",
        ),
    ),
)


def profiles_material() -> dict[str, object]:
    """返回全部机制配置的版本化内容。"""

    return {
        "schema_version": FREE_EXPLORATION_PROFILE_VERSION,
        "profiles": [profile.material() for profile in MECHANISM_PROFILES],
    }


FREE_EXPLORATION_PROFILE_HASH = sha256_json(profiles_material())


__all__ = [
    "FREE_EXPLORATION_PROFILE_HASH",
    "FREE_EXPLORATION_PROFILE_VERSION",
    "MECHANISM_PROFILES",
    "MechanismProfile",
    "profiles_material",
]
