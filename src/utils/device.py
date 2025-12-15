"""
디바이스 관리 유틸리티
M1 MacBook Pro MPS 우선 사용
"""

import torch


def get_device() -> torch.device:
    """
    최적 디바이스 자동 선택

    우선순위: MPS > CUDA > CPU

    Returns:
        torch.device: 사용할 디바이스
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[Device] Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] Using CPU")

    return device


def get_device_info() -> dict:
    """
    디바이스 정보 반환

    Returns:
        dict: 디바이스 정보
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    return info


def to_device(data, device=None):
    """
    데이터를 디바이스로 이동

    Args:
        data: 텐서 또는 모델
        device: 대상 디바이스 (None이면 자동 선택)

    Returns:
        디바이스로 이동된 데이터
    """
    if device is None:
        device = get_device()

    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    if hasattr(data, 'to'):
        return data.to(device)

    return data


if __name__ == "__main__":
    # 디바이스 정보 출력
    print("=" * 50)
    print("Device Information")
    print("=" * 50)

    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("=" * 50)
