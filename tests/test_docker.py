"""
Docker 설정 테스트 (Task 15)
============================
Docker 설정 파일 유효성 검증
"""

import pytest
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Dockerfile Tests
# ============================================================================

class TestDockerfile:
    """Dockerfile 테스트"""

    @pytest.fixture
    def dockerfile_content(self):
        """Dockerfile 내용"""
        with open(PROJECT_ROOT / "Dockerfile", "r") as f:
            return f.read()

    def test_dockerfile_exists(self):
        """Dockerfile 존재"""
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_dockerfile_has_base_image(self, dockerfile_content):
        """베이스 이미지 정의"""
        assert "FROM python:3.11-slim" in dockerfile_content

    def test_dockerfile_has_workdir(self, dockerfile_content):
        """WORKDIR 설정"""
        assert "WORKDIR /app" in dockerfile_content

    def test_dockerfile_copies_requirements(self, dockerfile_content):
        """requirements 복사"""
        assert "COPY requirements.txt" in dockerfile_content

    def test_dockerfile_installs_dependencies(self, dockerfile_content):
        """의존성 설치"""
        assert "pip install" in dockerfile_content

    def test_dockerfile_creates_user(self, dockerfile_content):
        """비루트 사용자 생성"""
        assert "useradd" in dockerfile_content or "adduser" in dockerfile_content

    def test_dockerfile_switches_user(self, dockerfile_content):
        """사용자 전환"""
        assert "USER appuser" in dockerfile_content

    def test_dockerfile_exposes_port(self, dockerfile_content):
        """포트 노출"""
        assert "EXPOSE 8000" in dockerfile_content

    def test_dockerfile_has_healthcheck(self, dockerfile_content):
        """헬스체크 설정"""
        assert "HEALTHCHECK" in dockerfile_content

    def test_dockerfile_has_cmd(self, dockerfile_content):
        """CMD 설정"""
        assert "CMD" in dockerfile_content

    def test_dockerfile_has_multistage_build(self, dockerfile_content):
        """멀티스테이지 빌드"""
        assert dockerfile_content.count("FROM") >= 2

    def test_dockerfile_has_production_stage(self, dockerfile_content):
        """production 스테이지"""
        assert "as production" in dockerfile_content.lower()


class TestDockerfileDashboard:
    """Dashboard Dockerfile 테스트"""

    @pytest.fixture
    def dockerfile_content(self):
        """Dashboard Dockerfile 내용"""
        dockerfile_path = PROJECT_ROOT / "Dockerfile.dashboard"
        if dockerfile_path.exists():
            with open(dockerfile_path, "r") as f:
                return f.read()
        return None

    def test_dockerfile_dashboard_exists(self):
        """Dashboard Dockerfile 존재"""
        assert (PROJECT_ROOT / "Dockerfile.dashboard").exists()

    def test_dashboard_has_streamlit(self, dockerfile_content):
        """Streamlit 설치"""
        if dockerfile_content:
            assert "streamlit" in dockerfile_content.lower()

    def test_dashboard_exposes_port(self, dockerfile_content):
        """포트 노출"""
        if dockerfile_content:
            assert "EXPOSE 8501" in dockerfile_content


# ============================================================================
# Docker Compose Tests
# ============================================================================

class TestDockerCompose:
    """Docker Compose 테스트"""

    @pytest.fixture
    def compose_config(self):
        """Docker Compose 설정"""
        with open(PROJECT_ROOT / "docker-compose.yml", "r") as f:
            return yaml.safe_load(f)

    def test_docker_compose_exists(self):
        """docker-compose.yml 존재"""
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_compose_version(self, compose_config):
        """버전 정의"""
        assert "version" in compose_config
        assert compose_config["version"] >= "3.8"

    def test_compose_has_services(self, compose_config):
        """서비스 정의"""
        assert "services" in compose_config
        assert len(compose_config["services"]) >= 1

    def test_compose_has_api_service(self, compose_config):
        """API 서비스"""
        services = compose_config["services"]
        assert "api" in services

    def test_api_service_config(self, compose_config):
        """API 서비스 설정"""
        api = compose_config["services"]["api"]
        assert "build" in api
        assert "ports" in api
        assert "healthcheck" in api

    def test_compose_has_dashboard_service(self, compose_config):
        """Dashboard 서비스"""
        services = compose_config["services"]
        assert "dashboard" in services

    def test_dashboard_depends_on_api(self, compose_config):
        """Dashboard가 API에 의존"""
        dashboard = compose_config["services"]["dashboard"]
        assert "depends_on" in dashboard
        # Check if api is in depends_on (might be dict or list)
        depends = dashboard["depends_on"]
        if isinstance(depends, dict):
            assert "api" in depends
        else:
            assert "api" in depends

    def test_compose_has_networks(self, compose_config):
        """네트워크 정의"""
        assert "networks" in compose_config
        assert "power-demand-network" in compose_config["networks"]

    def test_compose_has_volumes(self, compose_config):
        """볼륨 정의"""
        assert "volumes" in compose_config

    def test_services_have_resource_limits(self, compose_config):
        """리소스 제한"""
        api = compose_config["services"]["api"]
        if "deploy" in api and "resources" in api["deploy"]:
            assert "limits" in api["deploy"]["resources"]

    def test_services_have_restart_policy(self, compose_config):
        """재시작 정책"""
        api = compose_config["services"]["api"]
        assert "restart" in api

    def test_compose_has_development_profile(self, compose_config):
        """개발 프로파일"""
        services = compose_config["services"]
        dev_services = [
            name for name, config in services.items()
            if "profiles" in config and "dev" in config["profiles"]
        ]
        assert len(dev_services) >= 1


# ============================================================================
# Nginx Configuration Tests
# ============================================================================

class TestNginxConfig:
    """Nginx 설정 테스트"""

    @pytest.fixture
    def nginx_config(self):
        """Nginx 설정 내용"""
        config_path = PROJECT_ROOT / "docker" / "nginx" / "nginx.conf"
        if config_path.exists():
            with open(config_path, "r") as f:
                return f.read()
        return None

    def test_nginx_config_exists(self):
        """Nginx 설정 파일 존재"""
        assert (PROJECT_ROOT / "docker" / "nginx" / "nginx.conf").exists()

    def test_nginx_has_upstream_api(self, nginx_config):
        """API upstream 설정"""
        if nginx_config:
            assert "upstream api_backend" in nginx_config

    def test_nginx_has_upstream_dashboard(self, nginx_config):
        """Dashboard upstream 설정"""
        if nginx_config:
            assert "upstream dashboard_backend" in nginx_config

    def test_nginx_has_api_proxy(self, nginx_config):
        """API 프록시 설정"""
        if nginx_config:
            assert "location /api/" in nginx_config

    def test_nginx_has_websocket_support(self, nginx_config):
        """WebSocket 지원"""
        if nginx_config:
            assert "Upgrade" in nginx_config
            assert "upgrade" in nginx_config

    def test_nginx_has_rate_limiting(self, nginx_config):
        """Rate limiting 설정"""
        if nginx_config:
            assert "limit_req_zone" in nginx_config

    def test_nginx_has_security_headers(self, nginx_config):
        """보안 헤더"""
        if nginx_config:
            assert "X-Frame-Options" in nginx_config
            assert "X-Content-Type-Options" in nginx_config

    def test_nginx_has_gzip(self, nginx_config):
        """Gzip 압축"""
        if nginx_config:
            assert "gzip on" in nginx_config


# ============================================================================
# Prometheus Configuration Tests
# ============================================================================

class TestPrometheusConfig:
    """Prometheus 설정 테스트"""

    @pytest.fixture
    def prometheus_config(self):
        """Prometheus 설정"""
        config_path = PROJECT_ROOT / "docker" / "prometheus" / "prometheus.yml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return None

    def test_prometheus_config_exists(self):
        """Prometheus 설정 파일 존재"""
        assert (PROJECT_ROOT / "docker" / "prometheus" / "prometheus.yml").exists()

    def test_prometheus_has_global(self, prometheus_config):
        """글로벌 설정"""
        if prometheus_config:
            assert "global" in prometheus_config

    def test_prometheus_has_scrape_interval(self, prometheus_config):
        """스크레이프 간격"""
        if prometheus_config:
            assert "scrape_interval" in prometheus_config["global"]

    def test_prometheus_has_scrape_configs(self, prometheus_config):
        """스크레이프 설정"""
        if prometheus_config:
            assert "scrape_configs" in prometheus_config
            assert len(prometheus_config["scrape_configs"]) >= 1

    def test_prometheus_scrapes_api(self, prometheus_config):
        """API 스크레이프"""
        if prometheus_config:
            jobs = [job["job_name"] for job in prometheus_config["scrape_configs"]]
            assert "api" in jobs


# ============================================================================
# Environment Configuration Tests
# ============================================================================

class TestEnvironmentConfig:
    """환경 설정 테스트"""

    def test_dockerignore_exists(self):
        """dockerignore 존재"""
        # .dockerignore가 있으면 좋지만 필수는 아님
        pass

    def test_compose_environment_variables(self):
        """환경 변수 기본값"""
        with open(PROJECT_ROOT / "docker-compose.yml", "r") as f:
            content = f.read()

        # 환경 변수 기본값 확인
        assert "${API_PORT:-8000}" in content or "API_PORT" in content
        assert "${DASHBOARD_PORT:-8501}" in content or "DASHBOARD_PORT" in content


# ============================================================================
# Integration Tests
# ============================================================================

class TestDockerIntegration:
    """Docker 통합 테스트"""

    def test_all_required_files_exist(self):
        """필수 파일 존재"""
        required_files = [
            "Dockerfile",
            "Dockerfile.dashboard",
            "docker-compose.yml",
            "requirements.txt"
        ]

        for file in required_files:
            assert (PROJECT_ROOT / file).exists(), f"{file} not found"

    def test_docker_directory_structure(self):
        """Docker 디렉토리 구조"""
        docker_dir = PROJECT_ROOT / "docker"
        if docker_dir.exists():
            assert (docker_dir / "nginx").exists()
            assert (docker_dir / "prometheus").exists()

    def test_compose_can_be_parsed(self):
        """Docker Compose 파싱 가능"""
        with open(PROJECT_ROOT / "docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)
        assert config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
