package kubernetes

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterKubernetesModules([]problems.CourseModule{
		{
			ID:          1729,
			Title:       "Kubernetes Troubleshooting",
			Description: "Master Kubernetes troubleshooting: debugging pods, nodes, networking, storage, and performance issues.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Debugging Pods and Workloads",
					Content: `Systematic troubleshooting is essential for managing Kubernetes clusters. Most issues fall into categories: scheduling, runtime, networking, or configuration.

**Pod Lifecycle and Failure States:**
` + "```" + `
Pod phases:
  Pending     → not yet scheduled or pulling images
  Running     → at least one container is running
  Succeeded   → all containers exited with 0
  Failed      → at least one container exited non-zero
  Unknown     → node communication lost

Common pod conditions:
  PodScheduled       → accepted by scheduler
  Initialized        → init containers completed
  ContainersReady    → all containers ready
  Ready              → pod serving traffic (readiness probe)

Container states:
  Waiting      → pulling image, creating container
  Running      → container process is executing
  Terminated   → container exited (check exit code)
` + "```" + `

**Debugging Commands:**
` + "```" + `
# Pod overview
kubectl get pods -n production -o wide
kubectl get pods --field-selector=status.phase!=Running -A

# Detailed pod info
kubectl describe pod <name> -n production
# Look at:
#   Events (scheduling, pulling, starting)
#   Conditions
#   Container statuses
#   Restart count and last state

# Container logs
kubectl logs <pod> -n production
kubectl logs <pod> -c <container>       # specific container
kubectl logs <pod> --previous            # previous instance (after restart)
kubectl logs <pod> -f                    # follow (tail)
kubectl logs <pod> --since=1h            # last hour
kubectl logs <pod> --tail=100            # last 100 lines

# Execute command in container
kubectl exec -it <pod> -n production -- /bin/sh
kubectl exec <pod> -- cat /etc/config/app.yaml
kubectl exec <pod> -- env | sort

# Ephemeral debug container (when no shell in image)
kubectl debug -it <pod> --image=busybox:1.36 --target=<container>
kubectl debug <pod> --copy-to=debug-pod --container=debug --image=nicolaka/netshoot

# Copy files from/to pod
kubectl cp <pod>:/path/to/file ./local-file
kubectl cp ./local-file <pod>:/path/to/file

# Resource usage
kubectl top pods -n production
kubectl top pods --sort-by=cpu
kubectl top pods --sort-by=memory
kubectl top nodes
` + "```" + `

**Common Issues and Solutions:**
` + "```" + `
1. ImagePullBackOff:
  Causes:
    - Wrong image name/tag
    - Private registry, missing imagePullSecrets
    - Image doesn't exist
    - Rate limited (Docker Hub)
  Debug:
    kubectl describe pod <name> | grep -A5 Events
    kubectl get events --field-selector reason=Failed
  Fix:
    - Verify image: docker pull <image>
    - Add imagePullSecrets to ServiceAccount
    - Check registry credentials

2. CrashLoopBackOff:
  Causes:
    - Application error on startup
    - Missing config/secrets/env vars
    - Wrong command/args
    - Insufficient resources (OOMKill)
    - Failed health checks
  Debug:
    kubectl logs <pod> --previous
    kubectl describe pod <pod> | grep -A10 "Last State"
  Fix:
    - Check logs for startup errors
    - Verify env vars and mounted configs
    - Check resource limits (OOMKilled reason)
    - Adjust health check timings

3. Pending (not scheduling):
  Causes:
    - Insufficient resources on any node
    - Node affinity/selector doesn't match
    - Taints without tolerations
    - PVC binding pending
    - ResourceQuota exceeded
  Debug:
    kubectl describe pod <pod> | grep -A5 Events
    kubectl get events --field-selector reason=FailedScheduling
    kubectl describe nodes | grep -A5 "Allocated resources"
  Fix:
    - Scale up nodes or adjust requests
    - Fix node selector/affinity
    - Add tolerations
    - Check PVC status

4. OOMKilled:
  Causes:
    - Memory limit too low
    - Memory leak in application
  Debug:
    kubectl describe pod <pod> | grep OOMKilled
    kubectl get events --field-selector reason=OOMKilling
    # Check memory usage over time in Grafana
  Fix:
    - Increase memory limit
    - Profile application for memory leaks
    - Set memory request = limit (Guaranteed QoS)

5. Evicted:
  Causes:
    - Node under disk/memory pressure
    - Ephemeral storage limit exceeded
  Debug:
    kubectl describe node <node> | grep Conditions
    kubectl get events --field-selector reason=Evicted
  Fix:
    - Clean up disk on nodes
    - Set ephemeral-storage limits
    - Add more nodes
` + "```" + ``,
					CodeExamples: `# Troubleshooting Scripts and Patterns

# 1. Quick cluster health check script
# #!/bin/bash
# echo "=== Node Status ==="
# kubectl get nodes -o wide
# echo ""
# echo "=== Not-Running Pods ==="
# kubectl get pods -A --field-selector=status.phase!=Running,status.phase!=Succeeded
# echo ""
# echo "=== Recent Events (Warnings) ==="
# kubectl get events -A --sort-by='.lastTimestamp' --field-selector type=Warning | tail -20
# echo ""
# echo "=== Resource Usage ==="
# kubectl top nodes
# echo ""
# echo "=== PVC Status ==="
# kubectl get pvc -A --field-selector=status.phase!=Bound

---
# 2. Debug pod template
apiVersion: v1
kind: Pod
metadata:
  name: debug
  namespace: production
spec:
  containers:
  - name: debug
    image: nicolaka/netshoot:latest
    command: ["sleep", "infinity"]
    securityContext:
      runAsUser: 0
  # Useful tools: tcpdump, dig, nslookup, curl, wget,
  # traceroute, mtr, ip, iptables, ss, netstat, ping,
  # strace, htop, iftop, nmap

---
# 3. Log aggregation for troubleshooting
# kubectl logs -l app=myapp --all-containers --prefix
# kubectl logs -l app=myapp --since=30m | grep -i error
# kubectl logs -l app=myapp -c init-container --prefix

---
# 4. Pod with node debugging capabilities
apiVersion: v1
kind: Pod
metadata:
  name: node-debug
  namespace: kube-system
spec:
  nodeName: worker-node-1  # Target specific node
  hostNetwork: true
  hostPID: true
  containers:
  - name: debug
    image: busybox:1.36
    command: ["sleep", "infinity"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: hostroot
      mountPath: /host
  volumes:
  - name: hostroot
    hostPath:
      path: /

---
# 5. Prometheus alerts for common issues
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: troubleshooting-alerts
  namespace: monitoring
spec:
  groups:
  - name: troubleshooting.rules
    rules:
    - alert: PodCrashLooping
      expr: |
        increase(kube_pod_container_status_restarts_total[1h]) > 5
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
        runbook: "Check logs: kubectl logs {{ $labels.pod }} -n {{ $labels.namespace }} --previous"
    
    - alert: PodNotReady
      expr: |
        kube_pod_status_ready{condition="false"} == 1
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} not ready for 15m"
    
    - alert: PodPending
      expr: |
        kube_pod_status_phase{phase="Pending"} == 1
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} stuck in Pending"
    
    - alert: NodeNotReady
      expr: |
        kube_node_status_condition{condition="Ready",status="true"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.node }} not ready"
    
    - alert: NodeDiskPressure
      expr: |
        kube_node_status_condition{condition="DiskPressure",status="true"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.node }} has disk pressure"
    
    - alert: NodeMemoryPressure
      expr: |
        kube_node_status_condition{condition="MemoryPressure",status="true"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.node }} has memory pressure"
    
    - alert: PVCAlmostFull
      expr: |
        (kubelet_volume_stats_available_bytes / kubelet_volume_stats_capacity_bytes) < 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "PVC {{ $labels.persistentvolumeclaim }} is >90% full"
    
    - alert: DeploymentReplicasMismatch
      expr: |
        kube_deployment_spec_replicas != kube_deployment_status_ready_replicas
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Deployment {{ $labels.namespace }}/{{ $labels.deployment }} has replica mismatch"
    
    - alert: HPAMaxedOut
      expr: |
        kube_horizontalpodautoscaler_status_current_replicas == kube_horizontalpodautoscaler_spec_max_replicas
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "HPA {{ $labels.namespace }}/{{ $labels.horizontalpodautoscaler }} at max replicas"`,
				},
				{
					Title: "Node and Network Troubleshooting",
					Content: `Node and network issues are harder to diagnose because they often affect multiple pods simultaneously.

**Node Troubleshooting:**
` + "```" + `
# Check node status and conditions
kubectl get nodes -o wide
kubectl describe node <node-name>
# Key conditions:
#   Ready            → kubelet healthy
#   MemoryPressure   → low memory
#   DiskPressure     → low disk
#   PIDPressure      → too many processes
#   NetworkUnavailable → CNI not configured

# Check node resource allocation
kubectl describe node <node> | grep -A20 "Allocated resources"
# Shows: requests/limits vs capacity

# Check system pods on node
kubectl get pods -A --field-selector spec.nodeName=<node>

# Node events
kubectl get events --field-selector involvedObject.kind=Node,involvedObject.name=<node>

# SSH into node (if accessible)
# Check kubelet: systemctl status kubelet
# Check kubelet logs: journalctl -u kubelet -f
# Check container runtime: crictl ps
# Check disk: df -h
# Check memory: free -h
# Check processes: top / htop
# Check network: ip addr, ip route, iptables -L
` + "```" + `

**Common Node Issues:**
` + "```" + `
1. Node NotReady:
  Causes:
    - kubelet crashed/stopped
    - Container runtime (containerd) crashed
    - Network connectivity lost
    - Resource exhaustion (disk, memory, PIDs)
    - Certificate expired
  Debug:
    ssh node → systemctl status kubelet
    ssh node → journalctl -u kubelet --since "10 min ago"
    kubectl get events --field-selector involvedObject.name=<node>

2. DiskPressure:
  Causes:
    - Container images filling disk
    - Logs not rotated
    - Too many unused images
    - Large PVCs or emptyDir volumes
  Fix:
    - crictl rmi --prune    # Remove unused images
    - Clean old logs
    - Increase disk size
    - Set imagefs eviction thresholds

3. kubelet certificate issues:
  Causes:
    - Auto-rotation failed
    - Clock skew between nodes
  Debug:
    openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -text -noout
  Fix:
    - Ensure time sync (NTP/chrony)
    - Delete kubelet certs and restart (auto-renew)
` + "```" + `

**Network Troubleshooting:**
` + "```" + `
Common network issues:
  1. Pod can't reach another pod
  2. Pod can't reach a Service (ClusterIP)
  3. Pod can't reach external internet
  4. DNS resolution failures
  5. Intermittent connectivity

Step-by-step debugging:

1. DNS check:
   kubectl exec <pod> -- nslookup <service>.namespace.svc.cluster.local
   kubectl exec <pod> -- nslookup kubernetes.default
   kubectl exec <pod> -- cat /etc/resolv.conf
   
2. Service endpoint check:
   kubectl get endpoints <service> -n <namespace>
   kubectl get endpointslices -l kubernetes.io/service-name=<service>
   # If no endpoints → pods not matching selector or not ready

3. Pod-to-pod connectivity:
   kubectl exec <pod-a> -- ping <pod-b-ip>
   kubectl exec <pod-a> -- curl -v http://<pod-b-ip>:8080/healthz
   
4. Pod-to-service connectivity:
   kubectl exec <pod> -- curl -v http://<service>.<namespace>:80
   kubectl exec <pod> -- curl -v http://<clusterIP>:80
   
5. kube-proxy / iptables:
   # On node:
   iptables -t nat -L KUBE-SERVICES | grep <service>
   ipvsadm -L -n | grep <clusterIP>
   
6. CNI plugin:
   # On node:
   ls /etc/cni/net.d/       # CNI config
   ls /opt/cni/bin/         # CNI binaries
   journalctl -u kubelet | grep cni
   
7. Network policy:
   kubectl get networkpolicy -n <namespace>
   # Try temporarily deleting NetworkPolicy to isolate issue
   
8. tcpdump:
   kubectl exec <netshoot-pod> -- tcpdump -i eth0 port 8080 -nn
   kubectl exec <netshoot-pod> -- tcpdump -i any host <target-ip> -nn
` + "```" + `

**Performance Troubleshooting:**
` + "```" + `
CPU throttling:
  Symptom: high latency, low CPU utilization
  Check: rate(container_cpu_cfs_throttled_periods_total[5m])
  Fix: increase or remove CPU limits

Memory issues:
  Symptom: OOMKills, high restart count
  Check: container_memory_working_set_bytes vs limits
  Fix: increase memory limits, fix memory leaks

Network latency:
  Symptom: slow service-to-service calls
  Check: 
    - DNS latency (coredns metrics)
    - iptables rule count: iptables -t nat -L | wc -l
    - MTU issues: tracepath <destination>
  Fix:
    - Switch to IPVS or eBPF mode
    - Reduce ndots in /etc/resolv.conf
    - Fix MTU mismatches

Disk I/O:
  Symptom: slow pod startup, slow writes
  Check: 
    - kubectl describe node | grep disk
    - iostat on node
  Fix:
    - Use faster StorageClass (SSD)
    - Increase IOPS allocation
    - Optimize application disk access patterns

etcd performance:
  Symptom: slow API responses, leader elections
  Check:
    - etcd_disk_wal_fsync_duration_seconds
    - etcd_server_leader_changes_seen_total
    - etcd_disk_backend_commit_duration_seconds
  Fix:
    - Use SSD for etcd storage
    - Defragment etcd: etcdctl defrag
    - Increase etcd resources
    - Reduce object count (clean up old resources)
` + "```" + ``,
					CodeExamples: `# Network and Performance Troubleshooting

# 1. Comprehensive network test pod
apiVersion: v1
kind: Pod
metadata:
  name: net-troubleshoot
  namespace: production
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot:latest
    command:
    - sh
    - -c
    - |
      echo "=== DNS Test ==="
      nslookup kubernetes.default.svc.cluster.local
      echo ""
      echo "=== DNS Resolve Time ==="
      time nslookup kubernetes.default.svc.cluster.local
      echo ""
      echo "=== Cluster API connectivity ==="
      curl -sk https://kubernetes.default/version
      echo ""
      echo "=== Network interfaces ==="
      ip addr show
      echo ""
      echo "=== Routes ==="
      ip route
      echo ""
      echo "=== DNS Config ==="
      cat /etc/resolv.conf
      echo ""
      echo "=== Done, sleeping ==="
      sleep infinity
  restartPolicy: Never

---
# 2. Service connectivity checker (CronJob)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: connectivity-check
  namespace: monitoring
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: checker
            image: curlimages/curl:8.4.0
            command:
            - sh
            - -c
            - |
              echo "Testing service endpoints..."
              for svc in user-service order-service payment-service; do
                if curl -sf -o /dev/null -w "%{http_code}" "http://$svc.production.svc.cluster.local/healthz" --connect-timeout 5; then
                  echo "$svc: OK"
                else
                  echo "$svc: FAILED"
                fi
              done
          restartPolicy: OnFailure
      backoffLimit: 1

---
# 3. kubectl useful diagnostic commands reference
# 
# === Cluster overview ===
# kubectl cluster-info
# kubectl get componentstatuses  # deprecated but sometimes useful
# kubectl get --raw /healthz
# kubectl get --raw /readyz
# 
# === Events (most useful for troubleshooting) ===
# kubectl get events -A --sort-by='.lastTimestamp' | tail -30
# kubectl get events -A --field-selector type=Warning
# kubectl get events -A --field-selector reason=FailedScheduling
# kubectl get events -A --field-selector reason=Unhealthy
# kubectl get events -A --field-selector reason=OOMKilling
# 
# === Resource usage ===
# kubectl top nodes --sort-by=cpu
# kubectl top pods -A --sort-by=memory | head -20
# kubectl resource-capacity  # (kubectl-resource-capacity plugin)
# 
# === API resources ===
# kubectl api-resources --verbs=list --namespaced -o name
# kubectl get all -n production
# kubectl get deploy,sts,ds,job,cronjob -n production
# 
# === Diff (before apply) ===
# kubectl diff -f manifests/

---
# 4. PrometheusRule for SLI monitoring
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: sli-monitoring
  namespace: monitoring
spec:
  groups:
  - name: sli.recording
    rules:
    # Availability SLI (% of successful requests)
    - record: sli:availability:ratio_rate5m
      expr: |
        sum(rate(http_requests_total{status!~"5.."}[5m])) by (service, namespace)
        /
        sum(rate(http_requests_total[5m])) by (service, namespace)
    
    # Latency SLI (% of requests under threshold)
    - record: sli:latency_good:ratio_rate5m
      expr: |
        sum(rate(http_request_duration_seconds_bucket{le="0.25"}[5m])) by (service, namespace)
        /
        sum(rate(http_request_duration_seconds_count[5m])) by (service, namespace)
    
    # Error budget remaining
    - record: sli:error_budget:remaining
      expr: |
        1 - (
          (1 - sli:availability:ratio_rate5m) / (1 - 0.999)
        )
    
  - name: sli.alerts
    rules:
    - alert: ErrorBudgetBurning
      expr: sli:error_budget:remaining < 0.5
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "{{ $labels.service }} error budget <50% remaining"

---
# 5. Useful kubectl plugins for troubleshooting
# Install via krew:
# kubectl krew install tree       # Resource hierarchy
# kubectl krew install neat       # Clean YAML output
# kubectl krew install images     # Show images used
# kubectl krew install who-can    # RBAC checking
# kubectl krew install sniff      # Packet capture
# kubectl krew install node-shell # SSH to node
# kubectl krew install resource-capacity  # Node capacity view
# kubectl krew install stern      # Multi-pod log tailing
#
# Usage:
# kubectl tree deploy myapp -n production
# kubectl neat get pod myapp-xxx -n production
# kubectl images -n production
# kubectl who-can create pods -n production
# kubectl sniff myapp-xxx -n production -f "port 8080"
# kubectl node-shell worker-1
# stern "myapp-*" -n production --since=5m`,
				},
			},
		},
	})
}
