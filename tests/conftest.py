"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_jeju_smp_last_price(monkeypatch, tmp_path):
    """테스트가 실제 data/smp/jeju_rt_smp_last.json에 가짜 가격을 쓰거나 읽지 않게 한다.

    실서버 API는 이 파일을 API 실패 시 마지막 검증값으로 쓰므로, 테스트 값이 섞이면 공개 화면에 나온다.
    """
    try:
        from api import realtime_api
    except Exception:  # API 의존성이 없는 테스트 환경에서는 격리할 대상도 없다
        return
    monkeypatch.setattr(realtime_api, "JEJU_SMP_LAST_PATH", tmp_path / "jeju_rt_smp_last.json")
