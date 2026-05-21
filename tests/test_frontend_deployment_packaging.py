"""Static checks for the OCR operator-console deployment package."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_frontend_dockerfile_builds_next_app():
    dockerfile = (ROOT / "Dockerfile.frontend").read_text(encoding="utf-8")
    assert "FROM node:20-alpine AS deps" in dockerfile
    assert "npm ci" in dockerfile
    assert "npm run build" in dockerfile
    assert 'CMD ["npm", "run", "start", "--", "-H", "0.0.0.0"]' in dockerfile


def test_helm_values_define_frontend_image_and_runtime():
    values = yaml.safe_load((ROOT / "helm" / "ocr-local" / "values.yaml").read_text(encoding="utf-8"))
    assert values["image"]["frontend"]["repository"] == "ocr-local/frontend"
    assert values["frontend"]["enabled"] is True
    assert values["frontend"]["apiBaseUrl"] == "/"
    assert values["frontend"]["service"]["port"] == 3000


def test_frontend_helm_templates_exist_and_target_frontend_component():
    deployment = (
        ROOT / "helm" / "ocr-local" / "templates" / "frontend-deployment.yaml"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "helm" / "ocr-local" / "templates" / "frontend-service.yaml"
    ).read_text(encoding="utf-8")
    helpers = (
        ROOT / "helm" / "ocr-local" / "templates" / "_helpers.tpl"
    ).read_text(encoding="utf-8")

    assert 'define "ocr-local.frontendImage"' in helpers
    assert "app.kubernetes.io/component: frontend" in deployment
    assert "NEXT_PUBLIC_API_BASE_URL" in deployment
    assert "readinessProbe:" in deployment
    assert "app.kubernetes.io/component: frontend" in service


def test_ingress_routes_api_to_coordinator_and_root_to_frontend_when_enabled():
    ingress = (
        ROOT / "helm" / "ocr-local" / "templates" / "ingress.yaml"
    ).read_text(encoding="utf-8")
    assert "if .Values.frontend.enabled" in ingress
    assert "path: /api" in ingress
    assert "name: {{ .Release.Name }}-coordinator" in ingress
    assert "name: {{ .Release.Name }}-frontend" in ingress
