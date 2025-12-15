#!/usr/bin/env python
"""
Power Demand Forecast API Runner
================================

API 서버 실행 스크립트

Usage:
------
# 개발 모드 (hot reload)
python run_api.py --dev

# 프로덕션 모드
python run_api.py

# 커스텀 포트
python run_api.py --port 8080
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Power Demand Forecast API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--dev', action='store_true', help='Development mode with hot reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')

    args = parser.parse_args()

    # Set environment variables
    os.environ.setdefault('HOST', args.host)
    os.environ.setdefault('PORT', str(args.port))

    if args.dev:
        os.environ['DEBUG'] = 'true'
        os.environ['RELOAD'] = 'true'
        os.environ['LOG_LEVEL'] = 'DEBUG'

    # Import and run
    import uvicorn

    print("=" * 60)
    print("Power Demand Forecast API")
    print("=" * 60)
    print(f"Mode: {'Development' if args.dev else 'Production'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    print()
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print()
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        workers=1 if args.dev else args.workers,
        log_level="debug" if args.dev else "info"
    )


if __name__ == '__main__':
    main()
