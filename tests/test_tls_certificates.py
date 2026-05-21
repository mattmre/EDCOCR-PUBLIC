"""Tests for inter-service TLS via cert-manager (Wave 5).

Validates Helm template rendering for:
- Certificate and Issuer resources when tls.enabled=true
- Correct secretName, DNS names, and duration/renewBefore on each Certificate
- No TLS resources when tls.enabled=false (default)
- TLS volume mounts appear in StatefulSets when TLS is enabled
- TLS env vars appear in ConfigMap when TLS is enabled
- TLS ports appear in NetworkPolicy when TLS is enabled

Requires: helm CLI available on PATH.

Run with: python -m pytest tests/test_tls_certificates.py -v
"""

import os
import shutil
import subprocess
import unittest

import yaml

# ---------------------------------------------------------------------------
# Helm chart path and common flags
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
CHART_PATH = os.path.join(_PROJECT_ROOT, "helm", "ocr-local")

_REQUIRED_SETS = [
    "--set", "secrets.djangoSecretKey=test",
    "--set", "secrets.postgresPassword=test",
    "--set", "secrets.rabbitmqPassword=test",
    "--set", "secrets.redisPassword=test",
    "--set", "secrets.flowerPassword=test-flower-password",
]

HELM_AVAILABLE = shutil.which("helm") is not None


def _helm_template(*extra_sets):
    """Run helm template and return parsed YAML documents as a list of dicts."""
    cmd = [
        "helm", "template", "tls-test", CHART_PATH,
    ] + list(_REQUIRED_SETS) + list(extra_sets)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    docs = []
    for doc in yaml.safe_load_all(result.stdout):
        if doc is not None:
            docs.append(doc)
    return docs


def _find_resources(docs, kind, name_contains=None):
    """Filter rendered documents by kind and optional name substring."""
    results = []
    for doc in docs:
        if doc.get("kind") == kind:
            if name_contains is None:
                results.append(doc)
            elif name_contains in doc.get("metadata", {}).get("name", ""):
                results.append(doc)
    return results


# ============================================================================
# TestTLSDisabledByDefault
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSDisabledByDefault(unittest.TestCase):
    """Verify that no TLS resources are generated with default values."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template()

    def test_no_certificate_resources(self):
        """No Certificate resources should be rendered when tls.enabled=false."""
        certs = _find_resources(self.docs, "Certificate")
        self.assertEqual(
            len(certs), 0,
            "No Certificate resources should be rendered with default values"
        )

    def test_no_issuer_resources(self):
        """No Issuer resources should be rendered when tls.enabled=false."""
        issuers = _find_resources(self.docs, "Issuer")
        self.assertEqual(
            len(issuers), 0,
            "No Issuer resources should be rendered with default values"
        )

    def test_default_secret_omits_metrics_api_key(self):
        """Default render should not inject a placeholder metrics API key."""
        secrets = _find_resources(self.docs, "Secret", "tls-test-secret")
        self.assertEqual(len(secrets), 1)
        self.assertNotIn("METRICS_API_KEY", secrets[0].get("data", {}))

    def test_default_probes_omit_metrics_header(self):
        """Coordinator probes should not send auth headers when no API key is set."""
        deployments = _find_resources(self.docs, "Deployment", "tls-test-coordinator")
        self.assertEqual(len(deployments), 1)
        container = deployments[0]["spec"]["template"]["spec"]["containers"][0]
        for probe_name in ("livenessProbe", "readinessProbe", "startupProbe"):
            probe = container[probe_name]["httpGet"]
            self.assertNotIn(
                "httpHeaders",
                probe,
                f"{probe_name} should omit auth headers when metricsApiKey is unset",
            )

    def test_no_tls_volume_mounts_in_postgres(self):
        """PostgreSQL StatefulSet should have no TLS volume mounts by default."""
        sts_list = _find_resources(self.docs, "StatefulSet", "postgres")
        self.assertGreaterEqual(len(sts_list), 1)
        containers = sts_list[0]["spec"]["template"]["spec"]["containers"]
        for container in containers:
            mounts = container.get("volumeMounts", [])
            tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
            self.assertEqual(
                len(tls_mounts), 0,
                "No tls-certs volumeMount should appear in PostgreSQL by default"
            )

    def test_no_tls_volume_mounts_in_rabbitmq(self):
        """RabbitMQ StatefulSet should have no TLS volume mounts by default."""
        sts_list = _find_resources(self.docs, "StatefulSet", "rabbitmq")
        self.assertGreaterEqual(len(sts_list), 1)
        containers = sts_list[0]["spec"]["template"]["spec"]["containers"]
        for container in containers:
            mounts = container.get("volumeMounts", [])
            tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
            self.assertEqual(
                len(tls_mounts), 0,
                "No tls-certs volumeMount should appear in RabbitMQ by default"
            )

    def test_no_tls_volume_mounts_in_redis(self):
        """Redis StatefulSet should have no TLS volume mounts by default."""
        sts_list = _find_resources(self.docs, "StatefulSet", "redis")
        # Filter out sentinel/replica StatefulSets
        redis_sts = [
            s for s in sts_list
            if s["metadata"]["name"] == "tls-test-redis"
        ]
        self.assertEqual(len(redis_sts), 1)
        containers = redis_sts[0]["spec"]["template"]["spec"]["containers"]
        for container in containers:
            mounts = container.get("volumeMounts", [])
            tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
            self.assertEqual(
                len(tls_mounts), 0,
                "No tls-certs volumeMount should appear in Redis by default"
            )

    def test_no_tls_env_vars_in_configmap(self):
        """ConfigMap should have no TLS env vars when tls.enabled=false."""
        configmaps = _find_resources(self.docs, "ConfigMap", "config")
        self.assertGreaterEqual(len(configmaps), 1)
        data = configmaps[0].get("data", {})
        self.assertNotIn(
            "DATABASE_SSLMODE", data,
            "DATABASE_SSLMODE should not appear in ConfigMap by default"
        )
        self.assertNotIn(
            "CELERY_BROKER_USE_SSL", data,
            "CELERY_BROKER_USE_SSL should not appear in ConfigMap by default"
        )
        self.assertNotIn(
            "REDIS_SSL", data,
            "REDIS_SSL should not appear in ConfigMap by default"
        )


# ============================================================================
# TestTLSEnabled
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSEnabled(unittest.TestCase):
    """Verify TLS resources are correctly generated when tls.enabled=true."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template("--set", "tls.enabled=true")

    # ------------------------------------------------------------------
    # Issuer resources
    # ------------------------------------------------------------------
    def test_selfsigned_root_issuer_created(self):
        """A self-signed root Issuer should be created."""
        issuers = _find_resources(self.docs, "Issuer", "selfsigned-root")
        self.assertEqual(
            len(issuers), 1,
            "Exactly one selfsigned-root Issuer should be rendered"
        )
        self.assertIn(
            "selfSigned", issuers[0]["spec"],
            "Root Issuer must have selfSigned spec"
        )

    def test_ca_issuer_created(self):
        """A CA Issuer backed by the CA certificate should be created."""
        issuers = _find_resources(self.docs, "Issuer", "ca-issuer")
        self.assertEqual(
            len(issuers), 1,
            "Exactly one ca-issuer Issuer should be rendered"
        )
        self.assertIn(
            "ca", issuers[0]["spec"],
            "CA Issuer must have ca spec"
        )
        self.assertEqual(
            issuers[0]["spec"]["ca"]["secretName"],
            "tls-test-ocr-local-ca-secret",
            "CA Issuer must reference the CA certificate secret"
        )

    # ------------------------------------------------------------------
    # CA Certificate
    # ------------------------------------------------------------------
    def test_ca_certificate_created(self):
        """A CA Certificate should be created with isCA=true."""
        certs = _find_resources(self.docs, "Certificate", "ca-cert")
        self.assertEqual(
            len(certs), 1,
            "Exactly one CA Certificate should be rendered"
        )
        self.assertTrue(
            certs[0]["spec"]["isCA"],
            "CA Certificate must have isCA=true"
        )
        self.assertEqual(
            certs[0]["spec"]["secretName"],
            "tls-test-ocr-local-ca-secret",
        )

    # ------------------------------------------------------------------
    # Per-service Certificates
    # ------------------------------------------------------------------
    def test_postgresql_certificate_created(self):
        """A PostgreSQL Certificate should be created with correct properties."""
        certs = _find_resources(self.docs, "Certificate", "postgresql-cert")
        self.assertEqual(len(certs), 1)
        spec = certs[0]["spec"]
        self.assertEqual(spec["secretName"], "tls-test-ocr-local-postgresql-tls")
        self.assertEqual(spec["duration"], "8760h")
        self.assertEqual(spec["renewBefore"], "720h")

    def test_postgresql_certificate_dns_names(self):
        """PostgreSQL Certificate should have correct DNS names."""
        certs = _find_resources(self.docs, "Certificate", "postgresql-cert")
        self.assertEqual(len(certs), 1)
        dns_names = certs[0]["spec"]["dnsNames"]
        self.assertIn("tls-test-postgres", dns_names)
        # Should include namespace-qualified names
        has_svc = any(".svc" in name for name in dns_names)
        self.assertTrue(has_svc, f"DNS names should include .svc entries: {dns_names}")

    def test_rabbitmq_certificate_created(self):
        """A RabbitMQ Certificate should be created with correct properties."""
        certs = _find_resources(self.docs, "Certificate", "rabbitmq-cert")
        self.assertEqual(len(certs), 1)
        spec = certs[0]["spec"]
        self.assertEqual(spec["secretName"], "tls-test-ocr-local-rabbitmq-tls")
        self.assertEqual(spec["duration"], "8760h")
        self.assertEqual(spec["renewBefore"], "720h")

    def test_rabbitmq_certificate_dns_names(self):
        """RabbitMQ Certificate should have correct DNS names."""
        certs = _find_resources(self.docs, "Certificate", "rabbitmq-cert")
        self.assertEqual(len(certs), 1)
        dns_names = certs[0]["spec"]["dnsNames"]
        self.assertIn("tls-test-rabbitmq", dns_names)
        has_svc = any(".svc" in name for name in dns_names)
        self.assertTrue(has_svc, f"DNS names should include .svc entries: {dns_names}")

    def test_redis_certificate_created(self):
        """A Redis Certificate should be created with correct properties."""
        certs = _find_resources(self.docs, "Certificate", "redis-cert")
        self.assertEqual(len(certs), 1)
        spec = certs[0]["spec"]
        self.assertEqual(spec["secretName"], "tls-test-ocr-local-redis-tls")
        self.assertEqual(spec["duration"], "8760h")
        self.assertEqual(spec["renewBefore"], "720h")

    def test_redis_certificate_dns_names(self):
        """Redis Certificate should have correct DNS names."""
        certs = _find_resources(self.docs, "Certificate", "redis-cert")
        self.assertEqual(len(certs), 1)
        dns_names = certs[0]["spec"]["dnsNames"]
        self.assertIn("tls-test-redis", dns_names)
        has_svc = any(".svc" in name for name in dns_names)
        self.assertTrue(has_svc, f"DNS names should include .svc entries: {dns_names}")

    def test_all_certificates_use_ecdsa(self):
        """All Certificates should use ECDSA private keys."""
        certs = _find_resources(self.docs, "Certificate")
        self.assertGreaterEqual(len(certs), 4)  # CA + 3 services
        for cert in certs:
            pk = cert["spec"].get("privateKey", {})
            self.assertEqual(
                pk.get("algorithm"), "ECDSA",
                f"Certificate {cert['metadata']['name']} should use ECDSA"
            )

    def test_all_service_certificates_reference_ca_issuer(self):
        """All service Certificates should reference the CA issuer."""
        service_certs = [
            c for c in _find_resources(self.docs, "Certificate")
            if not c["spec"].get("isCA", False)
        ]
        self.assertEqual(len(service_certs), 3)
        for cert in service_certs:
            issuer_ref = cert["spec"]["issuerRef"]
            self.assertIn(
                "ca-issuer", issuer_ref["name"],
                f"Certificate {cert['metadata']['name']} should reference the CA issuer"
            )

    # ------------------------------------------------------------------
    # Total resource count
    # ------------------------------------------------------------------
    def test_total_certificate_count(self):
        """Should have exactly 4 Certificate resources (1 CA + 3 services)."""
        certs = _find_resources(self.docs, "Certificate")
        self.assertEqual(
            len(certs), 4,
            f"Expected 4 Certificates (CA + postgresql + rabbitmq + redis), "
            f"got {len(certs)}"
        )

    def test_total_issuer_count(self):
        """Should have exactly 2 Issuer resources (root + CA)."""
        issuers = _find_resources(self.docs, "Issuer")
        self.assertEqual(
            len(issuers), 2,
            f"Expected 2 Issuers (selfsigned-root + ca-issuer), got {len(issuers)}"
        )


# ============================================================================
# TestTLSVolumeMounts
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSVolumeMounts(unittest.TestCase):
    """Verify TLS volume mounts appear in StatefulSets when tls.enabled=true."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template("--set", "tls.enabled=true")

    def _get_statefulset(self, name_contains):
        """Get a single StatefulSet by name substring, excluding sentinel/replica."""
        sts_list = _find_resources(self.docs, "StatefulSet", name_contains)
        main_sts = [
            s for s in sts_list
            if "sentinel" not in s["metadata"]["name"]
            and "replica" not in s["metadata"]["name"]
        ]
        self.assertEqual(
            len(main_sts), 1,
            f"Expected exactly 1 {name_contains} StatefulSet, got {len(main_sts)}"
        )
        return main_sts[0]

    def test_postgres_has_tls_volume_mount(self):
        """PostgreSQL container should have a tls-certs volumeMount at /etc/tls."""
        sts = self._get_statefulset("postgres")
        containers = sts["spec"]["template"]["spec"]["containers"]
        mounts = containers[0].get("volumeMounts", [])
        tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
        self.assertEqual(len(tls_mounts), 1)
        self.assertEqual(tls_mounts[0]["mountPath"], "/etc/tls")
        self.assertTrue(tls_mounts[0]["readOnly"])

    def test_postgres_has_tls_volume(self):
        """PostgreSQL pod should have a tls-certs volume from the correct secret."""
        sts = self._get_statefulset("postgres")
        volumes = sts["spec"]["template"]["spec"].get("volumes", [])
        tls_volumes = [v for v in volumes if v["name"] == "tls-certs"]
        self.assertEqual(len(tls_volumes), 1)
        self.assertEqual(
            tls_volumes[0]["secret"]["secretName"],
            "tls-test-ocr-local-postgresql-tls"
        )

    def test_rabbitmq_has_tls_volume_mount(self):
        """RabbitMQ container should have a tls-certs volumeMount at /etc/tls."""
        sts = self._get_statefulset("rabbitmq")
        containers = sts["spec"]["template"]["spec"]["containers"]
        mounts = containers[0].get("volumeMounts", [])
        tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
        self.assertEqual(len(tls_mounts), 1)
        self.assertEqual(tls_mounts[0]["mountPath"], "/etc/tls")
        self.assertTrue(tls_mounts[0]["readOnly"])

    def test_rabbitmq_has_tls_volume(self):
        """RabbitMQ pod should have a tls-certs volume from the correct secret."""
        sts = self._get_statefulset("rabbitmq")
        volumes = sts["spec"]["template"]["spec"].get("volumes", [])
        tls_volumes = [v for v in volumes if v["name"] == "tls-certs"]
        self.assertEqual(len(tls_volumes), 1)
        self.assertEqual(
            tls_volumes[0]["secret"]["secretName"],
            "tls-test-ocr-local-rabbitmq-tls"
        )

    def test_redis_has_tls_volume_mount(self):
        """Redis container should have a tls-certs volumeMount at /etc/tls."""
        sts = self._get_statefulset("redis")
        containers = sts["spec"]["template"]["spec"]["containers"]
        mounts = containers[0].get("volumeMounts", [])
        tls_mounts = [m for m in mounts if m["name"] == "tls-certs"]
        self.assertEqual(len(tls_mounts), 1)
        self.assertEqual(tls_mounts[0]["mountPath"], "/etc/tls")
        self.assertTrue(tls_mounts[0]["readOnly"])

    def test_redis_has_tls_volume(self):
        """Redis pod should have a tls-certs volume from the correct secret."""
        sts = self._get_statefulset("redis")
        volumes = sts["spec"]["template"]["spec"].get("volumes", [])
        tls_volumes = [v for v in volumes if v["name"] == "tls-certs"]
        self.assertEqual(len(tls_volumes), 1)
        self.assertEqual(
            tls_volumes[0]["secret"]["secretName"],
            "tls-test-ocr-local-redis-tls"
        )


# ============================================================================
# TestTLSServiceConfiguration
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSServiceConfiguration(unittest.TestCase):
    """Verify TLS-specific configuration in services when tls.enabled=true."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template("--set", "tls.enabled=true")

    def test_postgres_ssl_args(self):
        """PostgreSQL container args should include SSL configuration."""
        sts_list = _find_resources(self.docs, "StatefulSet", "postgres")
        self.assertGreaterEqual(len(sts_list), 1)
        args = sts_list[0]["spec"]["template"]["spec"]["containers"][0]["args"]
        args_str = " ".join(str(a) for a in args)
        self.assertIn("ssl=on", args_str)
        self.assertIn("ssl_cert_file=/etc/tls/tls.crt", args_str)
        self.assertIn("ssl_key_file=/etc/tls/tls.key", args_str)
        self.assertIn("ssl_ca_file=/etc/tls/ca.crt", args_str)

    def test_rabbitmq_ssl_env_vars(self):
        """RabbitMQ container should have SSL env vars when TLS is enabled."""
        sts_list = _find_resources(self.docs, "StatefulSet", "rabbitmq")
        self.assertGreaterEqual(len(sts_list), 1)
        env_list = sts_list[0]["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e.get("value", "") for e in env_list}
        self.assertEqual(env_dict.get("RABBITMQ_SSL_CERTFILE"), "/etc/tls/tls.crt")
        self.assertEqual(env_dict.get("RABBITMQ_SSL_KEYFILE"), "/etc/tls/tls.key")
        self.assertEqual(env_dict.get("RABBITMQ_SSL_CACERTFILE"), "/etc/tls/ca.crt")

    def test_rabbitmq_amqps_port(self):
        """RabbitMQ container should expose AMQPS port 5671 when TLS is enabled."""
        sts_list = _find_resources(self.docs, "StatefulSet", "rabbitmq")
        self.assertGreaterEqual(len(sts_list), 1)
        ports = sts_list[0]["spec"]["template"]["spec"]["containers"][0]["ports"]
        port_names = [p["name"] for p in ports]
        self.assertIn("amqps", port_names)
        amqps_port = [p for p in ports if p["name"] == "amqps"][0]
        self.assertEqual(amqps_port["containerPort"], 5671)

    def test_redis_tls_args(self):
        """Redis command should include --tls-* arguments when TLS is enabled."""
        sts_list = _find_resources(self.docs, "StatefulSet", "redis")
        redis_sts = [
            s for s in sts_list
            if s["metadata"]["name"] == "tls-test-redis"
        ]
        self.assertEqual(len(redis_sts), 1)
        command = redis_sts[0]["spec"]["template"]["spec"]["containers"][0]["command"]
        command_str = " ".join(str(c) for c in command)
        self.assertIn("--tls-port 6380", command_str)
        self.assertIn("--tls-cert-file /etc/tls/tls.crt", command_str)
        self.assertIn("--tls-key-file /etc/tls/tls.key", command_str)
        self.assertIn("--tls-ca-cert-file /etc/tls/ca.crt", command_str)

    def test_redis_tls_port_exposed(self):
        """Redis container should expose TLS port 6380 when TLS is enabled."""
        sts_list = _find_resources(self.docs, "StatefulSet", "redis")
        redis_sts = [
            s for s in sts_list
            if s["metadata"]["name"] == "tls-test-redis"
        ]
        self.assertEqual(len(redis_sts), 1)
        ports = redis_sts[0]["spec"]["template"]["spec"]["containers"][0]["ports"]
        port_numbers = [p["containerPort"] for p in ports]
        self.assertIn(6380, port_numbers)

    def test_configmap_has_tls_env_vars(self):
        """ConfigMap should include TLS-related env vars when TLS is enabled."""
        cms = _find_resources(self.docs, "ConfigMap", "config")
        self.assertGreaterEqual(len(cms), 1)
        data = cms[0].get("data", {})
        self.assertEqual(data.get("DATABASE_SSLMODE"), "verify-full")
        self.assertEqual(data.get("DATABASE_SSLROOTCERT"), "/etc/tls/ca.crt")
        self.assertEqual(data.get("CELERY_BROKER_USE_SSL"), "true")
        self.assertEqual(data.get("REDIS_SSL"), "true")


# ============================================================================
# TestTLSNetworkPolicy
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSNetworkPolicy(unittest.TestCase):
    """Verify NetworkPolicy includes TLS ports when both features are enabled."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template(
            "--set", "tls.enabled=true",
            "--set", "networkPolicy.enabled=true",
        )

    def _get_netpol_egress_ports(self, netpol_name_contains, target_component):
        """Extract egress port list for a specific target component."""
        netpols = _find_resources(self.docs, "NetworkPolicy", netpol_name_contains)
        self.assertGreaterEqual(len(netpols), 1)
        for rule in netpols[0]["spec"]["egress"]:
            to_list = rule.get("to", [])
            for to_item in to_list:
                selector = to_item.get("podSelector", {}).get("matchLabels", {})
                if selector.get("app.kubernetes.io/component") == target_component:
                    return [p["port"] for p in rule.get("ports", [])]
        return []

    def test_worker_netpol_allows_rabbitmq_tls_port(self):
        """Worker NetworkPolicy should allow egress to RabbitMQ AMQPS port 5671."""
        ports = self._get_netpol_egress_ports("worker-netpol", "rabbitmq")
        self.assertIn(5671, ports, f"Port 5671 should be allowed. Found: {ports}")

    def test_worker_netpol_allows_redis_tls_port(self):
        """Worker NetworkPolicy should allow egress to Redis TLS port 6380."""
        ports = self._get_netpol_egress_ports("worker-netpol", "redis")
        self.assertIn(6380, ports, f"Port 6380 should be allowed. Found: {ports}")

    def test_coordinator_netpol_allows_rabbitmq_tls_port(self):
        """Coordinator NetworkPolicy should allow egress to RabbitMQ AMQPS port 5671."""
        ports = self._get_netpol_egress_ports("coordinator-netpol", "rabbitmq")
        self.assertIn(5671, ports, f"Port 5671 should be allowed. Found: {ports}")

    def test_coordinator_netpol_allows_redis_tls_port(self):
        """Coordinator NetworkPolicy should allow egress to Redis TLS port 6380."""
        ports = self._get_netpol_egress_ports("coordinator-netpol", "redis")
        self.assertIn(6380, ports, f"Port 6380 should be allowed. Found: {ports}")


# ============================================================================
# TestTLSWithClusterIssuer
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSWithClusterIssuer(unittest.TestCase):
    """Verify TLS works with ClusterIssuer instead of Issuer."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template(
            "--set", "tls.enabled=true",
            "--set", "tls.certManager.issuerKind=ClusterIssuer",
        )

    def test_root_is_cluster_issuer(self):
        """Root issuer should be a ClusterIssuer when configured."""
        cluster_issuers = _find_resources(
            self.docs, "ClusterIssuer", "selfsigned-root"
        )
        self.assertEqual(
            len(cluster_issuers), 1,
            "Exactly one selfsigned-root ClusterIssuer should be rendered"
        )

    def test_ca_is_cluster_issuer(self):
        """CA issuer should be a ClusterIssuer when configured."""
        cluster_issuers = _find_resources(
            self.docs, "ClusterIssuer", "ca-issuer"
        )
        self.assertEqual(
            len(cluster_issuers), 1,
            "Exactly one ca-issuer ClusterIssuer should be rendered"
        )

    def test_certificates_reference_cluster_issuer(self):
        """All Certificates should reference ClusterIssuer kind."""
        certs = _find_resources(self.docs, "Certificate")
        self.assertGreaterEqual(len(certs), 4)
        for cert in certs:
            issuer_ref = cert["spec"]["issuerRef"]
            self.assertEqual(
                issuer_ref["kind"], "ClusterIssuer",
                f"Certificate {cert['metadata']['name']} should reference ClusterIssuer"
            )

    def test_no_namespace_scoped_issuers(self):
        """No namespace-scoped Issuer resources when using ClusterIssuer."""
        issuers = _find_resources(self.docs, "Issuer")
        self.assertEqual(
            len(issuers), 0,
            "No namespace-scoped Issuer should exist when using ClusterIssuer"
        )


# ============================================================================
# TestTLSVolumeDefaultMode
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestTLSVolumeDefaultMode(unittest.TestCase):
    """Verify TLS volume secrets have restrictive defaultMode."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _helm_template("--set", "tls.enabled=true")

    def _get_tls_volume_default_mode(self, sts_name_contains):
        """Extract defaultMode from the tls-certs volume in a StatefulSet."""
        sts_list = _find_resources(self.docs, "StatefulSet", sts_name_contains)
        main_sts = [
            s for s in sts_list
            if "sentinel" not in s["metadata"]["name"]
            and "replica" not in s["metadata"]["name"]
        ]
        self.assertEqual(len(main_sts), 1)
        volumes = main_sts[0]["spec"]["template"]["spec"].get("volumes", [])
        tls_volumes = [v for v in volumes if v["name"] == "tls-certs"]
        self.assertEqual(len(tls_volumes), 1)
        return tls_volumes[0]["secret"].get("defaultMode")

    def test_postgres_tls_volume_mode(self):
        """PostgreSQL TLS volume should have defaultMode 0440."""
        mode = self._get_tls_volume_default_mode("postgres")
        # YAML parses octal 0440 as decimal 288
        self.assertEqual(mode, 288, f"Expected defaultMode 0440 (288), got {mode}")

    def test_rabbitmq_tls_volume_mode(self):
        """RabbitMQ TLS volume should have defaultMode 0440."""
        mode = self._get_tls_volume_default_mode("rabbitmq")
        self.assertEqual(mode, 288, f"Expected defaultMode 0440 (288), got {mode}")

    def test_redis_tls_volume_mode(self):
        """Redis TLS volume should have defaultMode 0440."""
        mode = self._get_tls_volume_default_mode("redis")
        self.assertEqual(mode, 288, f"Expected defaultMode 0440 (288), got {mode}")
