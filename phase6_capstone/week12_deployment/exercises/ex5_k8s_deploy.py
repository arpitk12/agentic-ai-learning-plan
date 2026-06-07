"""
Exercise 5: Kubernetes Deployment for Production Agents
Guide Section: §10.2 — Kubernetes Deployment

Goal: Understand every K8s resource needed to deploy a production agent API
      and generate the complete set of manifest files.

Resources covered:
  Namespace         — logical isolation for all agent resources
  Secret            — API keys and passwords (base64-encoded)
  ConfigMap         — non-sensitive configuration (model names, URLs)
  PersistentVolumeClaim — persistent storage for vector DB
  Deployment        — manages replicas, rolling updates, health probes
  Service           — stable network endpoint + load balancer
  HorizontalPodAutoscaler — auto-scale based on CPU / custom metrics

No running K8s cluster needed — this exercise generates YAML and explains it.
To actually deploy: minikube start, then kubectl apply -f k8s/

pip install pyyaml
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import base64
import yaml
from pathlib import Path


# ─── Helper ───────────────────────────────────────────────────────────────────

def b64(s: str) -> str:
    """Base64-encode a string (required format for K8s Secret data values)."""
    return base64.b64encode(s.encode()).decode()


# ─── Resource Factories ────────────────────────────────────────────────────────

def make_namespace(name: str = "agent-prod") -> dict:
    """
    Namespace: logical isolation for all resources.
    All your K8s objects (Deployment, Service, etc.) live inside a namespace.
    Namespaces allow multiple teams or environments to share one cluster.
    """
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": name,
            "labels": {"app": "agent", "env": "production"},
        },
    }


def make_secret(namespace: str = "agent-prod") -> dict:
    """
    Secret: stores sensitive configuration (API keys, passwords, tokens).

    Important:
      - K8s Secrets are base64-encoded, NOT encrypted by default.
      - For real secrets: use Vault (HashiCorp), AWS Secrets Manager, or K8s SOPS.
      - Never commit actual API keys in Secret YAML to git.
      - In production: use `kubectl create secret` or CI/CD secret injection.

    This file uses placeholder values — replace before deploying.
    """
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "agent-api-secrets", "namespace": namespace},
        "type": "Opaque",
        "data": {
            # base64-encoded placeholder values — replace in production
            "GEMINI_API_KEY": b64("REPLACE_WITH_REAL_KEY"),
            "API_KEY":        b64("REPLACE_WITH_API_KEY"),
            "DATABASE_URL":   b64("postgresql://agent:agent@postgres-service:5432/agentdb"),
        },
        # Production approach: use `kubectl create secret generic agent-api-secrets`
        # `  --from-literal=GEMINI_API_KEY=$GEMINI_API_KEY`
        # `  --from-literal=API_KEY=$API_KEY`
    }


def make_configmap(namespace: str = "agent-prod") -> dict:
    """
    ConfigMap: non-sensitive configuration.

    Difference from Secret:
      ConfigMap → public config (model names, log levels, feature flags)
      Secret    → sensitive config (API keys, passwords, tokens)

    Changes to ConfigMap don't restart pods automatically.
    Use Reloader (stakater/Reloader) if you need automatic restarts on config changes.
    """
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "agent-api-config", "namespace": namespace},
        "data": {
            "MODEL":              "gemini/gemini-2.0-flash",
            "REDIS_URL":          "redis://redis-service:6379/0",
            "MAX_AGENT_STEPS":    "15",
            "LOG_LEVEL":          "INFO",
            "PYTHONUNBUFFERED":   "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }


def make_pvc(namespace: str = "agent-prod", storage_gb: int = 20) -> dict:
    """
    PersistentVolumeClaim: request for durable storage.

    Why needed for agents:
      - ChromaDB or Qdrant vector store must survive pod restarts
      - Without PVC, data is lost when a pod dies
      - Cloud providers (AWS, GCP, Azure) auto-provision the actual disk

    Access modes:
      ReadWriteOnce (RWO)  — one pod at a time (default, for most DBs)
      ReadWriteMany (RWX)  — multiple pods simultaneously (needs NFS or GlusterFS)
    """
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "chroma-pvc", "namespace": namespace},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": "standard",  # use "gp2" on EKS, "premium-rwo" on GKE
            "resources": {"requests": {"storage": f"{storage_gb}Gi"}},
        },
    }


def make_deployment(
    namespace: str = "agent-prod",
    image: str = "ghcr.io/yourorg/agent-api:latest",
    replicas: int = 3,
) -> dict:
    """
    Deployment: the heart of your production agent.

    Key settings and WHY they matter:

    replicas=3: three pods running at once
      → if one pod dies/restarts, two others still serve traffic
      → rolling update deploys one pod at a time → zero downtime

    RollingUpdate strategy:
      maxSurge=1:        add one extra pod before removing old ones
      maxUnavailable=0:  never drop below `replicas` pods during update

    resources.limits:
      CRITICAL — without limits, one pod can consume all node memory
      → OOM killer evicts other pods → cascading failures
      Set: requests (guaranteed) < limits (maximum allowed)

    livenessProbe: "Is the pod ALIVE?"
      → K8s restarts the pod if health check fails failureThreshold times
      → Use for: deadlocks, hung threads, infinite loops

    readinessProbe: "Is the pod READY to accept traffic?"
      → K8s removes the pod from the load balancer if not ready
      → Use for: startup time, dependency checks (DB not yet connected)
      → Different from liveness: a pod can be alive but not ready

    podAntiAffinity:
      → Schedules pods on DIFFERENT nodes
      → Without this: all 3 pods could land on one node → node failure = outage
    """
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "agent-api",
            "namespace": namespace,
            "labels": {"app": "agent-api"},
        },
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
                            # Requests: minimum guaranteed (used for scheduling)
                            # Limits: maximum allowed (pod killed if exceeded)
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                            "limits":   {"cpu": "2000m", "memory": "2Gi"},
                        },
                        # Load config + secrets from K8s objects (not hardcoded)
                        "envFrom": [
                            {"configMapRef": {"name": "agent-api-config"}},
                            {"secretRef":    {"name": "agent-api-secrets"}},
                        ],
                        # Liveness: restart pod if truly stuck
                        # initialDelaySeconds=90 → wait for sentence-transformer loading
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 90,
                            "periodSeconds": 30,
                            "failureThreshold": 3,
                            "timeoutSeconds": 10,
                        },
                        # Readiness: remove from load balancer if not ready
                        # initialDelaySeconds=30 → wait for FastAPI startup
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                            "failureThreshold": 2,
                            "timeoutSeconds": 5,
                        },
                        "volumeMounts": [{
                            "name": "chroma-storage",
                            "mountPath": "/app/chroma_db",
                        }],
                    }],
                    "volumes": [{
                        "name": "chroma-storage",
                        "persistentVolumeClaim": {"claimName": "chroma-pvc"},
                    }],
                    # Spread pods across different nodes for high availability
                    "affinity": {
                        "podAntiAffinity": {
                            "preferredDuringSchedulingIgnoredDuringExecution": [{
                                "weight": 100,
                                "podAffinityTerm": {
                                    "labelSelector": {
                                        "matchExpressions": [{
                                            "key": "app",
                                            "operator": "In",
                                            "values": ["agent-api"],
                                        }],
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


def make_service(namespace: str = "agent-prod") -> dict:
    """
    Service: stable network endpoint for a set of pods.

    Problem it solves:
      Pods have ephemeral IPs that change every restart.
      Service provides a stable DNS name (agent-api-service:80)
      and load balances traffic across all healthy pods.

    Service types:
      ClusterIP    — only accessible inside the cluster (default)
      NodePort     — accessible on each node's IP:port (for local testing)
      LoadBalancer — creates a cloud load balancer (AWS ELB, GCP LB, etc.)
                     This is what you use for public internet access.
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "agent-api-service",
            "namespace": namespace,
            "annotations": {
                # AWS: use NLB instead of classic ELB for better performance
                "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
            },
        },
        "spec": {
            "selector": {"app": "agent-api"},
            "ports": [{"port": 80, "targetPort": 8000, "protocol": "TCP", "name": "http"}],
            "type": "LoadBalancer",
        },
    }


def make_hpa(namespace: str = "agent-prod") -> dict:
    """
    HorizontalPodAutoscaler: auto-scale based on metrics.

    Scale up when:  CPU > 70% OR active_agents > 10 per pod
    Scale down when: metrics below threshold for 5 minutes (stabilization window)

    minReplicas=2: always run at least 2 pods (HA even at idle)
    maxReplicas=10: cap to control costs

    Custom metrics (agent_runs_active) require:
      1. prometheus-adapter or KEDA installed in cluster
      2. The metric exported by your /metrics endpoint

    Behavior settings:
      scaleUp stabilizationWindow=60s   → don't wait long to scale up (capacity now!)
      scaleDown stabilizationWindow=300s → wait 5 min before scaling down (avoid thrashing)
    """
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": "agent-api-hpa", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": "agent-api",
            },
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
                # Custom metric: scale when each pod handles too many concurrent agents
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


def make_celery_deployment(
    namespace: str = "agent-prod",
    image: str = "ghcr.io/yourorg/agent-api:latest",
    replicas: int = 2,
) -> dict:
    """
    Celery worker Deployment: same image, different command.
    Workers process long-running agent tasks from the Redis queue.
    """
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "celery-worker",
            "namespace": namespace,
            "labels": {"app": "celery-worker"},
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "celery-worker"}},
            "template": {
                "metadata": {"labels": {"app": "celery-worker"}},
                "spec": {
                    "containers": [{
                        "name": "celery-worker",
                        "image": image,
                        # Different CMD from the API deployment
                        "command": ["celery", "-A", "celery_app", "worker",
                                    "--loglevel=info", "--concurrency=4"],
                        "resources": {
                            # Workers need more memory (embedding models loaded in process)
                            "requests": {"cpu": "500m", "memory": "1Gi"},
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


# ─── Generate All Manifests ────────────────────────────────────────────────────

MANIFESTS = [
    ("00-namespace.yaml",       make_namespace),
    ("01-secret.yaml",          make_secret),
    ("02-configmap.yaml",       make_configmap),
    ("03-pvc.yaml",             make_pvc),
    ("04-deployment-api.yaml",  make_deployment),
    ("05-deployment-celery.yaml", make_celery_deployment),
    ("06-service.yaml",         make_service),
    ("07-hpa.yaml",             make_hpa),
]


def generate_manifests(output_dir: str = "./k8s") -> list[str]:
    """Generate all Kubernetes manifest files into output_dir."""
    Path(output_dir).mkdir(exist_ok=True)
    created = []
    for filename, factory in MANIFESTS:
        manifest = factory()
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        created.append(path)
    return created


# ─── Explain Each Resource ────────────────────────────────────────────────────

def print_architecture_overview():
    print("""
┌────────────────────────────────────────────────────────────────────────┐
│                    K8s AGENT DEPLOYMENT ARCHITECTURE                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Internet → LoadBalancer Service → [agent-api Pod × 3]               │
│                                           │                            │
│                                      Redis Service                     │
│                                           │                            │
│                               [celery-worker Pod × 2]                 │
│                                           │                            │
│                              PostgreSQL / ChromaDB PVC                 │
│                                                                        │
│   HPA watches CPU + agent_runs_active → scales 2-10 pods automatically│
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  MANIFESTS                                                             │
│  00-namespace.yaml        → logical isolation boundary                 │
│  01-secret.yaml           → API keys (base64, use Vault in prod)      │
│  02-configmap.yaml        → model names, URLs, log level              │
│  03-pvc.yaml              → persistent storage for ChromaDB           │
│  04-deployment-api.yaml   → FastAPI: 3 replicas, probes, limits       │
│  05-deployment-celery.yaml→ Celery: 2 replicas, same image, diff CMD  │
│  06-service.yaml          → LoadBalancer (public IP via cloud LB)     │
│  07-hpa.yaml              → auto-scale 2-10 on CPU or agent load      │
└────────────────────────────────────────────────────────────────────────┘
""")


def print_deployment_commands():
    print("""
┌────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT COMMANDS                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  SETUP (one time)                                                      │
│    minikube start --cpus=4 --memory=8192    # local cluster            │
│    kubectl apply -f k8s/                    # apply all manifests      │
│                                                                        │
│  VERIFY                                                                │
│    kubectl get pods -n agent-prod           # list pods + status       │
│    kubectl get svc  -n agent-prod           # get LoadBalancer IP      │
│    kubectl logs -f <pod-name> -n agent-prod # stream logs              │
│                                                                        │
│  DEBUG                                                                 │
│    kubectl describe pod <pod> -n agent-prod # events + conditions      │
│    kubectl exec -it <pod> -n agent-prod -- bash  # shell into pod      │
│    kubectl top pods -n agent-prod           # CPU / memory usage       │
│                                                                        │
│  UPDATE (rolling deploy)                                               │
│    kubectl set image deployment/agent-api \\                            │
│      agent-api=ghcr.io/yourorg/agent-api:v2 -n agent-prod             │
│    kubectl rollout status deployment/agent-api -n agent-prod           │
│                                                                        │
│  ROLLBACK                                                              │
│    kubectl rollout undo deployment/agent-api -n agent-prod             │
│                                                                        │
│  SCALE MANUALLY                                                        │
│    kubectl scale deployment agent-api --replicas=5 -n agent-prod       │
│    kubectl get hpa -n agent-prod            # check autoscaler         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Kubernetes Deployment Exercise ===\n")

    print_architecture_overview()

    print("Generating manifest files…")
    files = generate_manifests("./k8s")
    for f in files:
        print(f"  ✓ {f}")
    print(f"\n✅ {len(files)} manifests written to ./k8s/")

    # ── Walk through the Deployment manifest ──
    dep = make_deployment()
    spec = dep["spec"]
    container = spec["template"]["spec"]["containers"][0]

    print(f"""
DEPLOYMENT WALKTHROUGH
{'='*55}
Name:        {dep['metadata']['name']}
Namespace:   {dep['metadata']['namespace']}
Replicas:    {spec['replicas']}  (always 3 pods running for HA)
Strategy:    {spec['strategy']['type']}
  maxSurge:        {spec['strategy']['rollingUpdate']['maxSurge']}   (add 1 new pod first)
  maxUnavailable:  {spec['strategy']['rollingUpdate']['maxUnavailable']}   (never go below 3 during update)

Container:   {container['name']}
  Image:     {container['image']}
  CPU:       {container['resources']['requests']['cpu']} req → {container['resources']['limits']['cpu']} limit
  Memory:    {container['resources']['requests']['memory']} req → {container['resources']['limits']['memory']} limit

  livenessProbe:
    GET /health every {container['livenessProbe']['periodSeconds']}s
    initialDelay: {container['livenessProbe']['initialDelaySeconds']}s (wait for model loading)
    failThreshold: {container['livenessProbe']['failureThreshold']} failures → pod RESTARTED

  readinessProbe:
    GET /health every {container['readinessProbe']['periodSeconds']}s
    initialDelay: {container['readinessProbe']['initialDelaySeconds']}s
    failThreshold: {container['readinessProbe']['failureThreshold']} failures → pod REMOVED from load balancer
""")

    # ── HPA explanation ──
    hpa = make_hpa()
    hpa_spec = hpa["spec"]
    print(f"""HPA WALKTHROUGH
{'='*55}
Target:      {hpa_spec['scaleTargetRef']['name']}
Min pods:    {hpa_spec['minReplicas']}  (never go below this — always HA)
Max pods:    {hpa_spec['maxReplicas']}  (cost cap)

Scale UP  when: CPU > 70% OR agent_runs_active > 10/pod
Scale DOWN when: above thresholds are satisfied for 5 minutes

Scale-up   stabilization: {hpa_spec['behavior']['scaleUp']['stabilizationWindowSeconds']}s (fast response to load)
Scale-down stabilization: {hpa_spec['behavior']['scaleDown']['stabilizationWindowSeconds']}s (prevent thrashing)
""")

    print_deployment_commands()

    # ─── CHALLENGES ───────────────────────────────────────────────────────────
    print("CHALLENGES:")
    print("  TODO: Deploy to minikube: minikube start && kubectl apply -f k8s/")
    print("  TODO: Add an Ingress resource for HTTPS termination (with cert-manager)")
    print("  TODO: Add a ResourceQuota to the namespace to limit total cluster usage")
    print("  TODO: Add a NetworkPolicy to restrict pod-to-pod communication")
    print("  TODO: Extend HPA to scale Celery workers based on Redis queue depth")
