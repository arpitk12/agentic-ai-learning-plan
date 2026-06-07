"""
SOLUTION — Exercise 5: Kubernetes Deployment Manifests

Key concepts demonstrated:
- Namespace:   logical isolation boundary for all resources
- Secret:      base64-encoded sensitive config (API keys, passwords)
- ConfigMap:   non-sensitive config (model names, log level, URLs)
- PVC:         durable storage that survives pod restarts (for ChromaDB/Qdrant)
- Deployment:  manages replicas, rolling updates, health probes, resource limits
- Service:     stable DNS name + load balancing across pods
- HPA:         auto-scale 2-10 replicas based on CPU or custom Prometheus metrics
- Celery:      same image as the API, different CMD — processes background tasks

Critical settings and WHY they matter:
  resources.limits     → without limits, one pod can OOM-kill other pods
  livenessProbe        → restarts truly stuck/deadlocked pods
  readinessProbe       → removes not-yet-ready pods from load balancer (different purpose!)
  podAntiAffinity      → schedules pods on different nodes for true HA
  RollingUpdate        → maxUnavailable=0 means zero-downtime deployments

pip install pyyaml
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import base64
import yaml
from pathlib import Path


def b64(s: str) -> str:
    """K8s Secret data values must be base64-encoded."""
    return base64.b64encode(s.encode()).decode()


# ─── Resource builders ────────────────────────────────────────────────────────

def namespace(name: str = "agent-prod") -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": name, "labels": {"app": "agent", "env": "production"}},
    }


def secret(ns: str = "agent-prod") -> dict:
    """
    Stores API keys and passwords.

    IMPORTANT: base64 is encoding, NOT encryption.
    In production: use `kubectl create secret generic` in your CI/CD pipeline,
    or integrate HashiCorp Vault / AWS Secrets Manager / GCP Secret Manager.
    Never commit real secret values to git — even in encrypted form without rotation.
    """
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "agent-api-secrets", "namespace": ns},
        "type": "Opaque",
        "data": {
            "GEMINI_API_KEY": b64("REPLACE_WITH_REAL_KEY"),
            "API_KEY":        b64("REPLACE_WITH_API_KEY"),
            "DATABASE_URL":   b64("postgresql://agent:agent@postgres-service:5432/agentdb"),
        },
    }


def configmap(ns: str = "agent-prod") -> dict:
    """
    Non-sensitive config injected as environment variables.
    Changes to ConfigMap do NOT restart pods automatically.
    Use stakater/Reloader if you need automatic restarts on config changes.
    """
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "agent-api-config", "namespace": ns},
        "data": {
            "MODEL":                  "gemini/gemini-2.0-flash",
            "REDIS_URL":              "redis://redis-service:6379/0",
            "MAX_AGENT_STEPS":        "15",
            "LOG_LEVEL":              "INFO",
            "PYTHONUNBUFFERED":       "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }


def pvc(ns: str = "agent-prod", storage_gb: int = 20) -> dict:
    """
    PersistentVolumeClaim: requests durable storage from the cluster.

    Without a PVC, ChromaDB data is lost when a pod restarts.
    Cloud providers auto-provision the actual disk (EBS on AWS, PD on GCP).

    ReadWriteOnce (RWO): one pod at a time. Use for single-instance DBs.
    ReadWriteMany (RWX): multiple pods simultaneously. Needs NFS/GlusterFS.
    """
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "chroma-pvc", "namespace": ns},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": "standard",
            "resources": {"requests": {"storage": f"{storage_gb}Gi"}},
        },
    }


def deployment_api(
    ns: str = "agent-prod",
    image: str = "ghcr.io/yourorg/agent-api:latest",
    replicas: int = 3,
) -> dict:
    """
    Deployment: the heart of production.

    RollingUpdate:
      maxSurge=1 / maxUnavailable=0 → add one new pod before removing any old pod.
      This guarantees traffic is always served during updates (zero downtime).

    livenessProbe vs readinessProbe:
      liveness  → 'is the pod ALIVE?' — failure triggers a restart
      readiness → 'is the pod READY to receive traffic?' — failure removes from LB
      Both check /health, but with DIFFERENT timing and thresholds.

    initialDelaySeconds=90 for liveness:
      sentence-transformers loads ~80MB model at startup. Pod is alive but not
      serving until that completes. We give it 90s before the first liveness check.

    podAntiAffinity:
      Without this, Kubernetes may schedule all 3 replicas on the SAME node.
      One node failure → total outage. Anti-affinity distributes pods across nodes.
    """
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "agent-api", "namespace": ns, "labels": {"app": "agent-api"}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "agent-api"}},
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0},
            },
            "template": {
                "metadata": {"labels": {"app": "agent-api"}},
                "spec": {
                    "containers": [{
                        "name": "agent-api",
                        "image": image,
                        "imagePullPolicy": "Always",
                        "ports": [{"containerPort": 8000, "name": "http"}],
                        "resources": {
                            "requests": {"cpu": "500m",  "memory": "512Mi"},
                            "limits":   {"cpu": "2000m", "memory": "2Gi"},
                        },
                        "envFrom": [
                            {"configMapRef": {"name": "agent-api-config"}},
                            {"secretRef":    {"name": "agent-api-secrets"}},
                        ],
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 90,   # wait for model load
                            "periodSeconds": 30,
                            "failureThreshold": 3,
                            "timeoutSeconds": 10,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                            "failureThreshold": 2,
                            "timeoutSeconds": 5,
                        },
                        "volumeMounts": [{"name": "chroma-storage", "mountPath": "/app/chroma_db"}],
                    }],
                    "volumes": [{"name": "chroma-storage", "persistentVolumeClaim": {"claimName": "chroma-pvc"}}],
                    "affinity": {
                        "podAntiAffinity": {
                            "preferredDuringSchedulingIgnoredDuringExecution": [{
                                "weight": 100,
                                "podAffinityTerm": {
                                    "labelSelector": {
                                        "matchExpressions": [{"key": "app", "operator": "In", "values": ["agent-api"]}]
                                    },
                                    "topologyKey": "kubernetes.io/hostname",
                                },
                            }],
                        },
                    },
                },
            },
        },
    }


def deployment_celery(
    ns: str = "agent-prod",
    image: str = "ghcr.io/yourorg/agent-api:latest",
    replicas: int = 2,
) -> dict:
    """
    Celery worker deployment: same image, different command.
    Workers need more memory because embedding models are loaded in-process.
    No readiness/liveness probes needed (no HTTP port), but you can add
    a custom exec probe that checks the Celery ping.
    """
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "celery-worker", "namespace": ns, "labels": {"app": "celery-worker"}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "celery-worker"}},
            "template": {
                "metadata": {"labels": {"app": "celery-worker"}},
                "spec": {
                    "containers": [{
                        "name": "celery-worker",
                        "image": image,
                        "command": ["celery", "-A", "celery_app", "worker",
                                    "--loglevel=info", "--concurrency=4"],
                        "resources": {
                            "requests": {"cpu": "500m",  "memory": "1Gi"},
                            "limits":   {"cpu": "2000m", "memory": "4Gi"},
                        },
                        "envFrom": [
                            {"configMapRef": {"name": "agent-api-config"}},
                            {"secretRef":    {"name": "agent-api-secrets"}},
                        ],
                    }],
                },
            },
        },
    }


def service(ns: str = "agent-prod") -> dict:
    """
    Service: stable network endpoint for a dynamic set of pods.

    Pods get new IPs on every restart. Service provides a stable
    DNS name (agent-api-service.agent-prod.svc.cluster.local)
    and load-balances across all matching pods.

    LoadBalancer type → cloud provider creates an external LB with a public IP.
    For internal-only access, use ClusterIP.
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "agent-api-service", "namespace": ns,
            "annotations": {"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
        },
        "spec": {
            "selector": {"app": "agent-api"},
            "ports": [{"port": 80, "targetPort": 8000, "protocol": "TCP", "name": "http"}],
            "type": "LoadBalancer",
        },
    }


def hpa(ns: str = "agent-prod") -> dict:
    """
    HPA: auto-scale based on CPU or custom Prometheus metrics.

    minReplicas=2: always HA — never drop to a single pod.
    maxReplicas=10: cost cap.

    scaleUp stabilization=60s:   react quickly to traffic spikes
    scaleDown stabilization=300s: wait 5 min before removing pods (prevents thrashing)

    Custom metric 'agent_runs_active' requires prometheus-adapter or KEDA
    installed in the cluster to bridge Prometheus → Kubernetes metrics API.
    """
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": "agent-api-hpa", "namespace": ns},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "agent-api"},
            "minReplicas": 2,
            "maxReplicas": 10,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 70},
                    },
                },
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {"name": "agent_runs_active"},
                        "target": {"type": "AverageValue", "averageValue": "10"},
                    },
                },
            ],
            "behavior": {
                "scaleUp":   {"stabilizationWindowSeconds": 60},
                "scaleDown": {"stabilizationWindowSeconds": 300},
            },
        },
    }


# ─── Generate all manifests ────────────────────────────────────────────────────

MANIFESTS = [
    ("00-namespace.yaml",        namespace),
    ("01-secret.yaml",           secret),
    ("02-configmap.yaml",        configmap),
    ("03-pvc.yaml",              pvc),
    ("04-deployment-api.yaml",   deployment_api),
    ("05-deployment-celery.yaml", deployment_celery),
    ("06-service.yaml",          service),
    ("07-hpa.yaml",              hpa),
]


def generate(output_dir: str = "./k8s") -> list[str]:
    Path(output_dir).mkdir(exist_ok=True)
    paths = []
    for filename, factory in MANIFESTS:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            yaml.dump(factory(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        paths.append(path)
    return paths


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== K8s Deployment Solution ===\n")

    files = generate("./k8s")
    for f in files:
        print(f"  ✓ {f}")
    print(f"\n{len(files)} manifest files written to ./k8s/")

    # Walkthrough: deployment
    dep = deployment_api()
    c = dep["spec"]["template"]["spec"]["containers"][0]
    print(f"""
DEPLOYMENT WALKTHROUGH
{'='*55}
replicas:         {dep['spec']['replicas']}   (3 pods always running for HA)
strategy:         RollingUpdate
  maxSurge:       1   (add new pod BEFORE removing old → zero downtime)
  maxUnavailable: 0   (never drop below {dep['spec']['replicas']} pods)

resources:
  cpu:    {c['resources']['requests']['cpu']} req → {c['resources']['limits']['cpu']} limit
  memory: {c['resources']['requests']['memory']} req → {c['resources']['limits']['memory']} limit
  (limits are MANDATORY — without them one pod can OOM the entire node)

livenessProbe:
  initialDelay: {c['livenessProbe']['initialDelaySeconds']}s  ← wait for model load (80MB embedding model)
  period: {c['livenessProbe']['periodSeconds']}s, failThreshold: {c['livenessProbe']['failureThreshold']}  → pod RESTARTED on failure

readinessProbe:
  initialDelay: {c['readinessProbe']['initialDelaySeconds']}s
  period: {c['readinessProbe']['periodSeconds']}s, failThreshold: {c['readinessProbe']['failureThreshold']}  → pod REMOVED from load balancer on failure

podAntiAffinity: preferred (weight=100)
  → K8s tries to schedule pods on DIFFERENT nodes
  → if one node fails, only 1/3 pods die
""")

    # HPA walkthrough
    h = hpa()
    print(f"""HPA WALKTHROUGH
{'='*55}
min: {h['spec']['minReplicas']}  max: {h['spec']['maxReplicas']}
scale-up  stabilisation: {h['spec']['behavior']['scaleUp']['stabilizationWindowSeconds']}s   (fast response to spikes)
scale-down stabilisation: {h['spec']['behavior']['scaleDown']['stabilizationWindowSeconds']}s  (prevent thrashing)

Triggers:
  CPU > 70%                       → add pods
  agent_runs_active > 10 per pod  → add pods
""")

    print("""DEPLOY COMMANDS
='='*55
  minikube start --cpus=4 --memory=8192
  kubectl apply -f k8s/
  kubectl get pods -n agent-prod -w
  kubectl get svc  -n agent-prod
  kubectl logs -f <pod-name> -n agent-prod

UPDATE (rolling)
  kubectl set image deployment/agent-api agent-api=image:v2 -n agent-prod
  kubectl rollout status deployment/agent-api -n agent-prod

ROLLBACK
  kubectl rollout undo deployment/agent-api -n agent-prod
""")
