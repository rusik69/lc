package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1728,
			Title:       "Kubernetes Networking Deep Dive",
			Description: "Master Kubernetes networking: CNI plugins, Service types, DNS, CoreDNS, iptables/IPVS, and network troubleshooting.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "CNI and Pod Networking",
					Content: `Kubernetes networking model requires that every pod gets its own IP address and can communicate with any other pod without NAT.

**Networking Requirements:**
` + "```" + `
Kubernetes networking rules:
  1. Every pod gets a unique IP address
  2. Pods on a node can communicate with all pods on all nodes without NAT
  3. Agents on a node can communicate with all pods on that node
  4. Pods in the host network can communicate with all pods on all nodes

Network planes:
  Pod network:     pod-to-pod communication
  Service network: virtual IPs (ClusterIP) for service discovery
  Node network:    physical/cloud networking between nodes
  External:        traffic entering/leaving the cluster
` + "```" + `

**CNI (Container Network Interface):**
` + "```" + `
CNI plugins implement pod networking:

Calico:
  ✓ BGP-based routing (no overlay for on-prem)
  ✓ VXLAN/IPIP overlay option
  ✓ Full NetworkPolicy support
  ✓ eBPF dataplane option (high performance)
  ✓ Most widely used in production
  
Cilium:
  ✓ eBPF-based (kernel-level, very fast)
  ✓ L3-L7 NetworkPolicy (HTTP, gRPC, Kafka)
  ✓ Transparent encryption (WireGuard/IPsec)
  ✓ Service mesh without sidecars
  ✓ Hubble for observability
  ✓ Growing fast, default in many distros

Flannel:
  ✓ Simple VXLAN overlay
  ✓ Easy to set up
  ✗ No NetworkPolicy support
  ✗ Not recommended for production

Weave Net:
  ✓ Mesh overlay with encryption
  ✓ NetworkPolicy support
  ✓ Good for multi-cloud
  ✗ Less performant than Cilium/Calico

AWS VPC CNI:
  ✓ Pods get real VPC IPs
  ✓ No overlay (native performance)
  ✓ Security groups for pods
  ✗ Limited by ENI/IP limits per instance type
` + "```" + `

**Pod IP Assignment Flow:**
` + "```" + `
1. kubelet calls CNI plugin to set up networking
2. CNI plugin:
   a. Allocates IP from IPAM (IP Address Management)
   b. Creates veth pair (virtual ethernet)
   c. One end in pod network namespace (eth0)
   d. Other end on host (attached to bridge or directly routed)
   e. Configures routes

IPAM strategies:
  host-local:  pool of IPs per node (e.g., 10.244.1.0/24 for node 1)
  calico-ipam: cluster-wide IPAM with blocks per node
  aws-cni:     ENI IP allocation from VPC subnets

Network namespace:
  Each pod gets its own network namespace:
    - Own IP address
    - Own routing table
    - Own iptables rules
    - Own loopback interface (127.0.0.1)
  
  Containers in same pod share the network namespace:
    - Same IP, can reach each other via localhost
    - Different ports (port conflict possible)
` + "```" + `

**Overlay vs Underlay Networking:**
` + "```" + `
Overlay (VXLAN, IPIP, Geneve):
  - Encapsulates pod traffic in node-to-node packets
  - Works anywhere (cloud, on-prem, mixed)
  - Slight overhead (encapsulation/decapsulation)
  - Easy to set up
  
  Node A (10.0.1.1)           Node B (10.0.2.1)
  Pod (10.244.1.5) → VXLAN encap → VXLAN decap → Pod (10.244.2.3)
  
Underlay / Direct routing:
  - Pod IPs are routable on the physical network
  - No encapsulation overhead
  - Requires network infrastructure support (BGP)
  - Best performance
  
  Node A (10.0.1.1)           Node B (10.0.2.1)
  Pod (10.244.1.5) → BGP route → direct → Pod (10.244.2.3)

eBPF dataplane:
  - Replaces iptables with eBPF programs
  - Much better performance at scale
  - Lower latency, higher throughput
  - Used by Cilium, Calico eBPF mode
` + "```" + ``,
					CodeExamples: `# CNI and Networking Configuration

# 1. Calico Installation (Tigera operator)
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  calicoNetwork:
    bgp: Enabled
    ipPools:
    - blockSize: 26
      cidr: 192.168.0.0/16
      encapsulation: VXLANCrossSubnet
      natOutgoing: Enabled
      nodeSelector: all()
    nodeAddressAutodetectionV4:
      firstFound: true
  controlPlaneReplicas: 2
  typhaDeployment:
    spec:
      minReadySeconds: 30
      template:
        spec:
          containers:
          - name: calico-typha
            resources:
              limits:
                cpu: "1"
                memory: 512Mi
              requests:
                cpu: 200m
                memory: 256Mi

---
# 2. Cilium Helm values
# helm install cilium cilium/cilium \
#   --namespace kube-system \
#   --set kubeProxyReplacement=true \
#   --set k8sServiceHost=<API_SERVER_IP> \
#   --set k8sServicePort=6443 \
#   --set hubble.enabled=true \
#   --set hubble.relay.enabled=true \
#   --set hubble.ui.enabled=true \
#   --set encryption.enabled=true \
#   --set encryption.type=wireguard

# Cilium BGP Peering Policy (for direct routing)
apiVersion: cilium.io/v2alpha1
kind: CiliumBGPPeeringPolicy
metadata:
  name: rack-peer
spec:
  nodeSelector:
    matchLabels:
      rack: rack-1
  virtualRouters:
  - localASN: 65001
    exportPodCIDR: true
    neighbors:
    - peerAddress: 10.0.0.1/32
      peerASN: 65000
      connectRetryTimeSeconds: 120
      holdTimeSeconds: 90
      keepAliveTimeSeconds: 30

---
# 3. IP Pool management (Calico)
apiVersion: projectcalico.org/v3
kind: IPPool
metadata:
  name: production-pool
spec:
  cidr: 10.244.0.0/16
  ipipMode: CrossSubnet
  vxlanMode: Never
  natOutgoing: true
  nodeSelector: "!all()"
  blockSize: 26
---
apiVersion: projectcalico.org/v3
kind: IPPool
metadata:
  name: services-pool
spec:
  cidr: 10.96.0.0/12
  ipipMode: Never
  vxlanMode: Never
  natOutgoing: false

---
# 4. Multi-NIC pods (Multus)
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: fast-network
  namespace: production
spec:
  config: |
    {
      "cniVersion": "0.3.1",
      "type": "macvlan",
      "master": "eth1",
      "mode": "bridge",
      "ipam": {
        "type": "host-local",
        "subnet": "192.168.1.0/24",
        "rangeStart": "192.168.1.200",
        "rangeEnd": "192.168.1.250"
      }
    }
---
# Pod with additional network interface
apiVersion: v1
kind: Pod
metadata:
  name: multi-nic-pod
  namespace: production
  annotations:
    k8s.v1.cni.cncf.io/networks: fast-network
spec:
  containers:
  - name: app
    image: myapp:v1
    # eth0: default pod network (10.244.x.x)
    # net1: fast-network (192.168.1.x)

---
# 5. Network troubleshooting tools
apiVersion: v1
kind: Pod
metadata:
  name: netshoot
  namespace: production
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot:latest
    command: ["sleep", "infinity"]
    # Tools available: tcpdump, nslookup, dig, curl, wget,
    # traceroute, mtr, ip, iptables, ss, netstat, ping`,
				},
				{
					Title: "Services, DNS, and CoreDNS",
					Content: `Services provide stable endpoints for pod communication. CoreDNS handles service discovery within the cluster.

**Service Types:**
` + "```" + `
ClusterIP (default):
  - Internal-only virtual IP
  - Accessible only within the cluster
  - DNS: service-name.namespace.svc.cluster.local
  spec:
    type: ClusterIP
    ports:
    - port: 80
      targetPort: 8080

NodePort:
  - Exposes on each node's IP at a static port (30000-32767)
  - ClusterIP is also created
  - Access: <NodeIP>:<NodePort>
  spec:
    type: NodePort
    ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080

LoadBalancer:
  - Creates external load balancer (cloud provider)
  - NodePort and ClusterIP also created
  - Access: external IP
  spec:
    type: LoadBalancer
    ports:
    - port: 443
      targetPort: 8443

ExternalName:
  - CNAME record to external service
  - No proxy, just DNS alias
  spec:
    type: ExternalName
    externalName: db.example.com

Headless (ClusterIP: None):
  - No load balancing, no cluster IP
  - DNS returns individual pod IPs
  - Used with StatefulSets
  spec:
    type: ClusterIP
    clusterIP: None
` + "```" + `

**Service Implementation (kube-proxy modes):**
` + "```" + `
iptables mode (default):
  - Creates iptables rules for each Service/endpoint
  - Random pod selection (no true load balancing)
  - O(n) rule chains (slow with many services)
  - Good for small-medium clusters

IPVS mode:
  - Linux Virtual Server in kernel
  - Hash table lookup (O(1))
  - Multiple LB algorithms: rr, lc, dh, sh, sed, nq
  - Better for large clusters (>1000 services)
  - Lower latency, higher throughput
  
  # Enable: kube-proxy --proxy-mode=ipvs
  # Or via configmap:
  # mode: "ipvs"
  # ipvs:
  #   scheduler: "lc"  # least connections

eBPF (Cilium kube-proxy replacement):
  - Replaces kube-proxy entirely
  - eBPF programs in kernel
  - Fastest option
  - Socket-level load balancing
  - DSR (Direct Server Return) support
` + "```" + `

**CoreDNS:**
` + "```" + `
CoreDNS is the cluster DNS server. It resolves:

Service DNS:
  <service>.<namespace>.svc.cluster.local
  → ClusterIP of the service
  
  Example: redis.production.svc.cluster.local → 10.96.45.12

Pod DNS:
  <pod-ip-dashed>.<namespace>.pod.cluster.local
  → Pod IP
  
  Example: 10-244-1-5.production.pod.cluster.local → 10.244.1.5

Headless Service DNS:
  <service>.<namespace>.svc.cluster.local
  → Returns ALL pod IPs (A records)
  
  <pod-name>.<service>.<namespace>.svc.cluster.local
  → Specific pod IP (StatefulSet)
  
  Example: postgres-0.postgres-headless.production.svc.cluster.local

SRV records:
  _<port-name>._<protocol>.<service>.<namespace>.svc.cluster.local
  → Port and hostname

DNS search domains (in pod):
  /etc/resolv.conf:
    nameserver 10.96.0.10
    search production.svc.cluster.local svc.cluster.local cluster.local
    options ndots:5
  
  Short names: just "redis" → tries redis.production.svc.cluster.local first

ndots:5 means:
  - If hostname has < 5 dots, try search domains first
  - "redis" → tries search domains (4 DNS queries!)
  - "api.example.com" → tries search domains first (wasteful)
  
  Optimization:
    Use FQDN with trailing dot: "redis.production.svc.cluster.local."
    Or reduce ndots: spec.dnsConfig.options: [{name: ndots, value: "2"}]
` + "```" + `

**CoreDNS Configuration:**
` + "```" + `yaml
# CoreDNS ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf {
            max_concurrent 1000
        }
        cache 30
        loop
        reload
        loadbalance
    }
    
    # Custom DNS for internal zones
    example.com:53 {
        errors
        cache 30
        forward . 10.0.0.53 10.0.0.54
    }
    
    # Block certain domains
    block.example.com:53 {
        template IN A block.example.com {
            rcode NXDOMAIN
        }
    }
` + "```" + `

**DNS Debugging:**
` + "```" + `
# Test DNS resolution from a pod
kubectl run dnsutils --image=gcr.io/kubernetes-e2e-test-images/dnsutils:1.3 --command sleep infinity
kubectl exec -it dnsutils -- nslookup kubernetes
kubectl exec -it dnsutils -- nslookup myservice.production.svc.cluster.local
kubectl exec -it dnsutils -- dig +short myservice.production.svc.cluster.local
kubectl exec -it dnsutils -- cat /etc/resolv.conf

# Check CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns

# CoreDNS metrics
# dns_requests_total — total queries
# dns_responses_total — responses by rcode
# dns_request_duration_seconds — query latency
# coredns_cache_hits_total — cache hit rate
` + "```" + ``,
					CodeExamples: `# Service and DNS Configuration

# 1. Complete service setup for microservice
apiVersion: v1
kind: Service
metadata:
  name: user-service
  namespace: production
  labels:
    app: user-service
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
spec:
  type: ClusterIP
  selector:
    app: user-service
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: grpc
    port: 9090
    targetPort: 9090
    protocol: TCP
  - name: metrics
    port: 9100
    targetPort: 9100
    protocol: TCP
  sessionAffinity: None

---
# 2. Internal load balancer (cloud)
apiVersion: v1
kind: Service
metadata:
  name: internal-api
  namespace: production
  annotations:
    # AWS
    service.beta.kubernetes.io/aws-load-balancer-internal: "true"
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-scheme: internal
    # GCP
    # networking.gke.io/load-balancer-type: Internal
    # Azure
    # service.beta.kubernetes.io/azure-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  selector:
    app: internal-api
  ports:
  - port: 443
    targetPort: 8443

---
# 3. ExternalName for database migration
apiVersion: v1
kind: Service
metadata:
  name: orders-db
  namespace: production
spec:
  type: ExternalName
  externalName: orders-db.us-east-1.rds.amazonaws.com
  # Pods can connect to "orders-db" and reach RDS

---
# 4. Headless Service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: kafka-headless
  namespace: production
spec:
  clusterIP: None
  selector:
    app: kafka
  ports:
  - name: client
    port: 9092
  - name: internal
    port: 9093
  publishNotReadyAddresses: true  # Include not-ready pods in DNS

---
# 5. Service with topology keys (prefer local)
apiVersion: v1
kind: Service
metadata:
  name: cache-service
  namespace: production
spec:
  selector:
    app: cache
  ports:
  - port: 6379
  # Traffic policy: prefer same-zone pods
  internalTrafficPolicy: Local
  # External traffic: only route to local node
  externalTrafficPolicy: Local

---
# 6. Endpoints for external service (manual)
apiVersion: v1
kind: Service
metadata:
  name: legacy-api
  namespace: production
spec:
  ports:
  - port: 443
    targetPort: 443
# No selector — manual endpoints
---
apiVersion: v1
kind: Endpoints
metadata:
  name: legacy-api
  namespace: production
subsets:
- addresses:
  - ip: 203.0.113.10
  - ip: 203.0.113.11
  ports:
  - port: 443

---
# 7. EndpointSlice (modern replacement for Endpoints)
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: legacy-api-1
  namespace: production
  labels:
    kubernetes.io/service-name: legacy-api
addressType: IPv4
ports:
- name: https
  port: 443
  protocol: TCP
endpoints:
- addresses:
  - "203.0.113.10"
  conditions:
    ready: true
  zone: us-east-1a
- addresses:
  - "203.0.113.11"
  conditions:
    ready: true
  zone: us-east-1b

---
# 8. DNS customization per pod
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns-pod
  namespace: production
spec:
  dnsPolicy: "None"  # Fully custom DNS
  dnsConfig:
    nameservers:
    - 10.96.0.10       # CoreDNS
    - 8.8.8.8          # Fallback
    searches:
    - production.svc.cluster.local
    - svc.cluster.local
    options:
    - name: ndots
      value: "2"       # Reduce unnecessary DNS lookups
    - name: timeout
      value: "3"
    - name: attempts
      value: "2"
  containers:
  - name: app
    image: myapp:v1`,
				},
			},
		},
	})
}
