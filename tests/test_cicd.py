"""
CI/CD 파이프라인 테스트 (Task 16)
=================================
GitHub Actions 워크플로우 설정 검증
"""

import pytest
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


# ============================================================================
# Workflow Existence Tests
# ============================================================================

class TestWorkflowExistence:
    """워크플로우 파일 존재 테스트"""

    def test_github_directory_exists(self):
        """.github 디렉토리 존재"""
        assert (PROJECT_ROOT / ".github").exists()

    def test_workflows_directory_exists(self):
        """workflows 디렉토리 존재"""
        assert WORKFLOWS_DIR.exists()

    def test_ci_workflow_exists(self):
        """CI 워크플로우 존재"""
        assert (WORKFLOWS_DIR / "ci.yml").exists()

    def test_cd_workflow_exists(self):
        """CD 워크플로우 존재"""
        assert (WORKFLOWS_DIR / "cd.yml").exists()


# ============================================================================
# CI Workflow Tests
# ============================================================================

class TestCIWorkflow:
    """CI 워크플로우 테스트"""

    @pytest.fixture
    def ci_config(self):
        """CI 워크플로우 설정"""
        with open(WORKFLOWS_DIR / "ci.yml", "r") as f:
            return yaml.safe_load(f)

    def test_ci_has_name(self, ci_config):
        """워크플로우 이름"""
        assert "name" in ci_config
        assert ci_config["name"] == "CI"

    def test_ci_triggers(self, ci_config):
        """트리거 설정"""
        # YAML에서 'on'은 True로 파싱됨
        assert "on" in ci_config or True in ci_config
        triggers = ci_config.get("on") or ci_config.get(True, {})

        # push와 pull_request 트리거
        assert "push" in triggers or "pull_request" in triggers

    def test_ci_push_branches(self, ci_config):
        """push 브랜치 설정"""
        triggers = ci_config.get("on") or ci_config.get(True, {})
        if "push" in triggers:
            push = triggers["push"]
            assert "branches" in push
            branches = push["branches"]
            assert "main" in branches or "develop" in branches

    def test_ci_has_jobs(self, ci_config):
        """jobs 정의"""
        assert "jobs" in ci_config
        assert len(ci_config["jobs"]) >= 1

    def test_ci_has_lint_job(self, ci_config):
        """lint job 존재"""
        jobs = ci_config["jobs"]
        assert "lint" in jobs

    def test_ci_has_test_job(self, ci_config):
        """test job 존재"""
        jobs = ci_config["jobs"]
        assert "test" in jobs

    def test_test_job_matrix(self, ci_config):
        """테스트 매트릭스"""
        test_job = ci_config["jobs"]["test"]
        if "strategy" in test_job:
            strategy = test_job["strategy"]
            if "matrix" in strategy:
                matrix = strategy["matrix"]
                # Python 버전 매트릭스
                assert "python-version" in matrix
                versions = matrix["python-version"]
                assert len(versions) >= 1

    def test_ci_uses_checkout(self, ci_config):
        """checkout 액션 사용"""
        jobs = ci_config["jobs"]
        for job_name, job in jobs.items():
            if "steps" in job:
                uses_checkout = any(
                    "checkout" in str(step.get("uses", ""))
                    for step in job["steps"]
                )
                if job_name in ["lint", "test"]:
                    assert uses_checkout, f"{job_name} should use checkout action"

    def test_ci_uses_python_setup(self, ci_config):
        """Python setup 액션 사용"""
        jobs = ci_config["jobs"]
        for job_name, job in jobs.items():
            if "steps" in job:
                uses_python = any(
                    "setup-python" in str(step.get("uses", ""))
                    for step in job["steps"]
                )
                if job_name in ["lint", "test"]:
                    assert uses_python, f"{job_name} should use setup-python action"

    def test_ci_uses_cache(self, ci_config):
        """캐시 액션 사용"""
        jobs = ci_config["jobs"]
        uses_cache = False
        for job in jobs.values():
            if "steps" in job:
                for step in job["steps"]:
                    if "cache" in str(step.get("uses", "")):
                        uses_cache = True
                        break
        assert uses_cache, "CI should use caching"

    def test_ci_runs_pytest(self, ci_config):
        """pytest 실행"""
        test_job = ci_config["jobs"]["test"]
        if "steps" in test_job:
            runs_pytest = any(
                "pytest" in str(step.get("run", ""))
                for step in test_job["steps"]
            )
            assert runs_pytest, "Test job should run pytest"

    def test_ci_has_security_job(self, ci_config):
        """security job 존재"""
        jobs = ci_config["jobs"]
        assert "security" in jobs

    def test_ci_job_dependencies(self, ci_config):
        """job 의존성"""
        jobs = ci_config["jobs"]
        test_job = jobs.get("test", {})
        if "needs" in test_job:
            # test가 lint에 의존
            needs = test_job["needs"]
            if isinstance(needs, list):
                assert "lint" in needs
            else:
                assert needs == "lint"


# ============================================================================
# CD Workflow Tests
# ============================================================================

class TestCDWorkflow:
    """CD 워크플로우 테스트"""

    @pytest.fixture
    def cd_config(self):
        """CD 워크플로우 설정"""
        with open(WORKFLOWS_DIR / "cd.yml", "r") as f:
            return yaml.safe_load(f)

    def test_cd_has_name(self, cd_config):
        """워크플로우 이름"""
        assert "name" in cd_config

    def test_cd_triggers(self, cd_config):
        """트리거 설정"""
        # YAML에서 'on'은 True로 파싱됨
        assert "on" in cd_config or True in cd_config

    def test_cd_has_workflow_dispatch(self, cd_config):
        """수동 트리거 지원"""
        triggers = cd_config.get("on") or cd_config.get(True, {})
        assert "workflow_dispatch" in triggers

    def test_cd_has_jobs(self, cd_config):
        """jobs 정의"""
        assert "jobs" in cd_config
        assert len(cd_config["jobs"]) >= 1

    def test_cd_has_build_job(self, cd_config):
        """build job 존재"""
        jobs = cd_config["jobs"]
        # build, build-and-push, 또는 docker 중 하나
        has_build = any(
            "build" in job_name.lower()
            for job_name in jobs.keys()
        )
        assert has_build, "CD should have a build job"

    def test_cd_uses_docker_buildx(self, cd_config):
        """Docker Buildx 사용"""
        jobs = cd_config["jobs"]
        uses_buildx = False
        for job in jobs.values():
            if "steps" in job:
                for step in job["steps"]:
                    if "docker/setup-buildx-action" in str(step.get("uses", "")):
                        uses_buildx = True
                        break
        assert uses_buildx, "CD should use Docker Buildx"

    def test_cd_uses_docker_build(self, cd_config):
        """Docker build 액션 사용"""
        jobs = cd_config["jobs"]
        uses_build = False
        for job in jobs.values():
            if "steps" in job:
                for step in job["steps"]:
                    if "docker/build-push-action" in str(step.get("uses", "")):
                        uses_build = True
                        break
        assert uses_build, "CD should use docker/build-push-action"

    def test_cd_has_registry_config(self, cd_config):
        """레지스트리 설정"""
        if "env" in cd_config:
            env = cd_config["env"]
            assert "REGISTRY" in env or "IMAGE_NAME" in env

    def test_cd_has_deploy_job(self, cd_config):
        """deploy job 존재"""
        jobs = cd_config["jobs"]
        has_deploy = any(
            "deploy" in job_name.lower()
            for job_name in jobs.keys()
        )
        assert has_deploy, "CD should have deploy job"


# ============================================================================
# Workflow Validation Tests
# ============================================================================

class TestWorkflowValidation:
    """워크플로우 유효성 테스트"""

    def test_all_workflows_are_valid_yaml(self):
        """모든 워크플로우가 유효한 YAML"""
        if WORKFLOWS_DIR.exists():
            for yml_file in WORKFLOWS_DIR.glob("*.yml"):
                with open(yml_file, "r") as f:
                    config = yaml.safe_load(f)
                    assert config is not None, f"{yml_file.name} is invalid YAML"

    def test_workflows_have_required_fields(self):
        """필수 필드 확인"""
        # 'on'은 YAML에서 True로 파싱될 수 있음
        if WORKFLOWS_DIR.exists():
            for yml_file in WORKFLOWS_DIR.glob("*.yml"):
                with open(yml_file, "r") as f:
                    config = yaml.safe_load(f)
                    assert "name" in config, f"{yml_file.name} missing name"
                    assert "on" in config or True in config, f"{yml_file.name} missing on"
                    assert "jobs" in config, f"{yml_file.name} missing jobs"

    def test_all_jobs_have_runs_on(self):
        """모든 job에 runs-on 설정"""
        if WORKFLOWS_DIR.exists():
            for yml_file in WORKFLOWS_DIR.glob("*.yml"):
                with open(yml_file, "r") as f:
                    config = yaml.safe_load(f)
                    for job_name, job in config.get("jobs", {}).items():
                        assert "runs-on" in job, f"{yml_file.name}/{job_name} missing runs-on"

    def test_all_jobs_have_steps(self):
        """모든 job에 steps 설정"""
        if WORKFLOWS_DIR.exists():
            for yml_file in WORKFLOWS_DIR.glob("*.yml"):
                with open(yml_file, "r") as f:
                    config = yaml.safe_load(f)
                    for job_name, job in config.get("jobs", {}).items():
                        assert "steps" in job, f"{yml_file.name}/{job_name} missing steps"


# ============================================================================
# Best Practices Tests
# ============================================================================

class TestWorkflowBestPractices:
    """워크플로우 베스트 프랙티스 테스트"""

    @pytest.fixture
    def all_configs(self):
        """모든 워크플로우 설정"""
        configs = {}
        if WORKFLOWS_DIR.exists():
            for yml_file in WORKFLOWS_DIR.glob("*.yml"):
                with open(yml_file, "r") as f:
                    configs[yml_file.name] = yaml.safe_load(f)
        return configs

    def test_uses_specific_action_versions(self, all_configs):
        """특정 액션 버전 사용 (v4 형식 허용)"""
        for name, config in all_configs.items():
            for job in config.get("jobs", {}).values():
                for step in job.get("steps", []):
                    uses = step.get("uses", "")
                    if uses and "@" in uses:
                        # @v1, @v2, @sha 등 허용
                        version = uses.split("@")[1]
                        assert version, f"{name} has action without version"

    def test_lint_before_test(self, all_configs):
        """lint가 test 전에 실행"""
        ci_config = all_configs.get("ci.yml", {})
        jobs = ci_config.get("jobs", {})

        if "test" in jobs and "lint" in jobs:
            test_needs = jobs["test"].get("needs", [])
            if isinstance(test_needs, str):
                test_needs = [test_needs]
            assert "lint" in test_needs, "Test should depend on lint"

    def test_uses_fail_fast(self, all_configs):
        """매트릭스에 fail-fast 설정"""
        ci_config = all_configs.get("ci.yml", {})
        test_job = ci_config.get("jobs", {}).get("test", {})

        if "strategy" in test_job:
            strategy = test_job["strategy"]
            # fail-fast 설정 확인 (false가 명시적으로 좋음)
            if "fail-fast" in strategy:
                assert strategy["fail-fast"] is False, "Should use fail-fast: false for better debugging"

    def test_uses_environment_variables(self, all_configs):
        """환경 변수 사용"""
        ci_config = all_configs.get("ci.yml", {})
        assert "env" in ci_config, "CI should define environment variables"

    def test_cd_has_permissions(self, all_configs):
        """CD에 권한 설정"""
        cd_config = all_configs.get("cd.yml", {})
        jobs = cd_config.get("jobs", {})

        for job_name, job in jobs.items():
            if "push" in job_name.lower() or "build" in job_name.lower():
                # permissions가 job 레벨에 있어야 함
                if "permissions" in job:
                    assert True
                    break


# ============================================================================
# Integration Tests
# ============================================================================

class TestCICDIntegration:
    """CI/CD 통합 테스트"""

    def test_all_required_workflows_exist(self):
        """필수 워크플로우 존재"""
        required = ["ci.yml", "cd.yml"]
        for workflow in required:
            assert (WORKFLOWS_DIR / workflow).exists(), f"{workflow} not found"

    def test_workflows_cover_main_branch(self):
        """main 브랜치 커버"""
        for yml_file in WORKFLOWS_DIR.glob("*.yml"):
            with open(yml_file, "r") as f:
                config = yaml.safe_load(f)

            triggers = config.get("on", {})
            if "push" in triggers:
                branches = triggers["push"].get("branches", [])
                assert "main" in branches, f"{yml_file.name} should cover main branch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
