"""Cost savings measurement and Q1 reporting."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import cfg


@dataclass
class CostComparison:
    """Before/after cost model for a compliance review automation."""
    documents_per_month: int = 400
    manual_hours_per_doc: float = 5.0
    analyst_hourly_rate: float = 85.0       # USD, fully loaded cost
    avg_llm_cost_per_doc: float = 3.50      # USD, LLM + compute
    human_review_rate: float = 0.15         # 15% of docs still need human review
    human_review_fraction: float = 0.30     # HITL review = 30% of full manual time

    @property
    def manual_monthly_cost(self) -> float:
        return self.documents_per_month * self.manual_hours_per_doc * self.analyst_hourly_rate

    @property
    def automated_llm_cost(self) -> float:
        return self.documents_per_month * self.avg_llm_cost_per_doc

    @property
    def automated_human_cost(self) -> float:
        return (
            self.documents_per_month
            * self.human_review_rate
            * self.manual_hours_per_doc
            * self.human_review_fraction
            * self.analyst_hourly_rate
        )

    @property
    def automated_monthly_cost(self) -> float:
        return self.automated_llm_cost + self.automated_human_cost

    @property
    def monthly_savings(self) -> float:
        return self.manual_monthly_cost - self.automated_monthly_cost

    @property
    def savings_pct(self) -> float:
        return (self.monthly_savings / self.manual_monthly_cost) * 100

    @property
    def annual_savings(self) -> float:
        return self.monthly_savings * 12

    @property
    def payback_months(self) -> float:
        implementation_cost = 50_000  # rough estimate
        return implementation_cost / self.monthly_savings

    def print_report(self) -> None:
        bar = "═" * 56
        print(f"\n╔{bar}╗")
        print(f"║{'Q1 COMPLIANCE AUTOMATION — COST REDUCTION REPORT':^56}║")
        print(f"╠{bar}╣")
        print(f"║  Documents/month:            {self.documents_per_month:>8,}              ║")
        print(f"║  Manual cost/doc:            ${self.manual_hours_per_doc * self.analyst_hourly_rate:>8,.0f}              ║")
        print(f"║  Automated cost/doc:         ${self.avg_llm_cost_per_doc:>8.2f}              ║")
        print(f"╠{bar}╣")
        print(f"║  Manual monthly cost:        ${self.manual_monthly_cost:>10,.0f}            ║")
        print(f"║  Automated monthly cost:     ${self.automated_monthly_cost:>10,.0f}            ║")
        print(f"║  Monthly savings:            ${self.monthly_savings:>10,.0f}            ║")
        print(f"║  Savings %:                  {self.savings_pct:>9.1f}%            ║")
        print(f"╠{bar}╣")
        print(f"║  Annual savings:             ${self.annual_savings:>10,.0f}            ║")
        print(f"║  Implementation payback:     {self.payback_months:>7.1f} months            ║")
        print(f"╚{bar}╝\n")

    def to_dict(self) -> dict:
        return {
            "documents_per_month": self.documents_per_month,
            "manual_monthly_cost_usd": round(self.manual_monthly_cost, 2),
            "automated_monthly_cost_usd": round(self.automated_monthly_cost, 2),
            "monthly_savings_usd": round(self.monthly_savings, 2),
            "savings_pct": round(self.savings_pct, 1),
            "annual_savings_usd": round(self.annual_savings, 2),
            "payback_months": round(self.payback_months, 1),
        }
