"""Regression tests for chronological weather interpolation."""

import pandas as pd

from src.data.preprocessing import preprocess_weather_data


def test_weather_interpolation_uses_chronological_not_input_order() -> None:
    """Unsorted records must not distort values between observations."""
    weather = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"]
            ),
            "기온": [10.0, 0.0, None],
        }
    )

    result, stats = preprocess_weather_data(weather)
    temperatures = result.set_index("datetime")["기온"]

    assert temperatures.loc[pd.Timestamp("2024-01-01 01:00")] == 5.0
    assert stats["final_missing"] == {}
