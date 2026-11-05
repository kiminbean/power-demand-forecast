"""정산 응답 모의(simulated) 표시 회귀 테스트.

KPX 실측 연동 전까지 정산 API 산출물은 모의 값이므로,
응답 스키마가 is_simulated=True / data_source="simulated"를
기본으로 담고 있어야 모바일·웹 정산 화면이 실운영 결과와 구분할 수 있다.
"""

from api.v6_routes import (
    SETTLEMENT_DATA_SOURCE,
    SettlementRecord,
    SettlementStats,
)


class TestSettlementSimulationFlag:
    """정산 스키마의 모의 결과 표시 기본값 검증."""

    def test_data_source_constant_is_simulated(self):
        """KPX 실측 연동 전까지 data_source 상수는 simulated여야 한다."""
        assert SETTLEMENT_DATA_SOURCE == "simulated"

    def test_record_defaults_to_simulated(self):
        """SettlementRecord는 기본적으로 모의 결과로 표시된다."""
        record = SettlementRecord(
            date="2026-11-05",
            cleared_mwh=180.0,
            generation_mwh=175.0,
            imbalance_mwh=5.0,
            revenue_million=16.6,
            imbalance_million=-1.2,
            net_revenue_million=15.4,
            accuracy_pct=93.0,
            avg_deviation=3.0,
            hours_no_penalty=20,
            hours_over_generation=2,
            hours_under_generation=2,
        )
        assert record.is_simulated is True
        assert record.data_source == "simulated"

    def test_stats_defaults_to_simulated(self):
        """SettlementStats는 기본적으로 모의 결과로 표시된다."""
        stats = SettlementStats(
            generation_revenue_million=1251.0,
            generation_change_pct=5.2,
            imbalance_charges_million=-45.3,
            imbalance_change_pct=-12.1,
            net_revenue_million=1255.7,
            net_change_pct=4.8,
            forecast_accuracy_pct=94.5,
            accuracy_change_pct=1.2,
        )
        assert stats.is_simulated is True
        assert stats.data_source == "simulated"

    def test_real_source_can_be_explicit(self):
        """실운영 전환 시 is_simulated=False로 명시할 수 있어야 한다."""
        record = SettlementRecord(
            date="2026-11-05",
            cleared_mwh=180.0,
            generation_mwh=175.0,
            imbalance_mwh=5.0,
            revenue_million=16.6,
            imbalance_million=-1.2,
            net_revenue_million=15.4,
            accuracy_pct=93.0,
            avg_deviation=3.0,
            hours_no_penalty=20,
            hours_over_generation=2,
            hours_under_generation=2,
            is_simulated=False,
            data_source="real",
        )
        assert record.is_simulated is False
        assert record.data_source == "real"
