package aws

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterAWSModules([]problems.CourseModule{
		{
			ID:          2121,
			Title:       "AWS Storage Services Deep Dive",
			Description: "Master S3 storage classes, lifecycle policies, EBS volumes, EFS, FSx, Storage Gateway, and data transfer strategies.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "S3 EBS EFS and AWS Storage Architecture",
					Content: `AWS storage services range from object storage (S3) to block storage (EBS) to file storage (EFS/FSx), each optimized for different workloads.

**Amazon S3 (Simple Storage Service):**

Storage Classes:
  S3 Standard:
    99.99% availability, 11 nines durability
    Frequently accessed data
    No retrieval fee
    
  S3 Intelligent-Tiering:
    Automatic tiering based on access patterns
    Frequent Access tier -> Infrequent Access (30 days)
    -> Archive Instant Access (90 days)
    -> Archive Access (90 days, optional)
    -> Deep Archive Access (180 days, optional)
    Monitoring fee per object
    
  S3 Standard-IA (Infrequent Access):
    99.9% availability
    Lower storage cost, retrieval fee
    Minimum 30-day charge, 128KB minimum
    
  S3 One Zone-IA:
    Single AZ (99.5% availability)
    20% cheaper than Standard-IA
    Non-critical, reproducible data
    
  S3 Glacier Instant Retrieval:
    Millisecond retrieval
    Minimum 90-day charge
    Rarely accessed data needing fast access
    
  S3 Glacier Flexible Retrieval:
    Expedited: 1-5 minutes
    Standard: 3-5 hours
    Bulk: 5-12 hours
    Minimum 90-day charge
    
  S3 Glacier Deep Archive:
    Standard: 12 hours
    Bulk: 48 hours
    Minimum 180-day charge
    Lowest storage cost

Lifecycle Policies:
  Transition Actions:
    Move to cheaper storage class
    Example: Standard -> Standard-IA after 30 days -> Glacier after 90 days
  Expiration Actions:
    Delete objects after N days
    Delete noncurrent versions
    Delete expired delete markers
    Abort incomplete multipart uploads

S3 Features:
  Versioning: Protect against accidental deletion
  Object Lock: WORM (Write Once Read Many)
    Governance Mode: Admin override
    Compliance Mode: No one can delete
    Retention Period: Fixed duration
    Legal Hold: Indefinite until removed
  
  Replication:
    Same-Region Replication (SRR)
    Cross-Region Replication (CRR)
    Bidirectional replication
    S3 Replication Time Control (15 minutes SLA)
  
  Event Notifications:
    Lambda, SQS, SNS, EventBridge
    Object created, removed, restored, replication
  
  S3 Select / Glacier Select:
    Query objects with SQL
    Process subset of data
    CSV, JSON, Parquet formats
    
  Transfer Acceleration:
    CloudFront edge locations for upload
    Faster long-distance transfers
    
  Multipart Upload:
    Required for > 5 GB objects
    Recommended for > 100 MB
    Up to 10,000 parts
    5 MB to 5 GB per part
    
  S3 Access Points:
    Named network endpoints
    Per-access-point policies
    VPC-restricted access points

S3 Security:
  Bucket Policies: Resource-based JSON policies
  ACLs: Legacy, use bucket policies instead
  Block Public Access: Account and bucket level
  Object Ownership: Disable ACLs
  Encryption:
    SSE-S3: S3-managed keys (default)
    SSE-KMS: KMS customer managed key
    SSE-C: Customer-provided key
    Client-side: Encrypt before upload
  Access Logging: Log all bucket requests
  Macie: Discover sensitive data

S3 Performance:
  3,500 PUT/s per prefix
  5,500 GET/s per prefix
  Parallelize across prefixes
  Byte-range fetches for large objects
  
  S3 Express One Zone:
    Single-digit millisecond latency
    150K+ requests/s per prefix
    Co-located with compute (same AZ)

**Amazon EBS (Elastic Block Store):**

Volume Types:
  gp3 (General Purpose SSD):
    3,000 IOPS baseline (free)
    Up to 16,000 IOPS, 1,000 MB/s
    1 GiB - 16 TiB
    
  gp2 (General Purpose SSD):
    3 IOPS/GiB (100 IOPS min)
    Up to 16,000 IOPS
    Burst to 3,000 IOPS (< 1 TiB)
    
  io2 Block Express:
    Up to 256,000 IOPS
    4,000 MB/s throughput
    Sub-millisecond latency
    99.999% durability
    Nitro instances only
    
  io1 (Provisioned IOPS):
    Up to 64,000 IOPS
    50 IOPS/GiB max ratio
    
  st1 (Throughput Optimized HDD):
    500 MB/s throughput
    500 IOPS max
    Big data, data warehouses
    
  sc1 (Cold HDD):
    250 MB/s throughput
    250 IOPS max
    Infrequently accessed data

EBS Features:
  Snapshots: Point-in-time backup to S3
    Incremental (only changed blocks)
    Share cross-account and cross-region
    Fast Snapshot Restore (FSR)
    Archive tier (75% cheaper, 24-72h restore)
    
  Multi-Attach (io1/io2):
    Same volume on up to 16 instances
    Same AZ only
    Cluster-aware applications
    
  Encryption:
    AES-256 at rest
    KMS customer managed keys
    Encrypt volumes, snapshots, data in transit
    Encrypted snapshots produce encrypted volumes

**Amazon EFS (Elastic File System):**

  NFS v4.1 file system
  Multi-AZ by default
  Auto-scales (petabyte scale)
  
  Performance Modes:
    General Purpose: Low latency, IOPS limit
    Max I/O: Higher latency, higher throughput
    
  Throughput Modes:
    Bursting: Based on storage amount
    Provisioned: Specify throughput (MiB/s)
    Elastic: Auto-scales (recommended)
    
  Storage Classes:
    Standard: Frequently accessed
    Infrequent Access (IA): Lower storage cost
    Archive: Very rarely accessed
    One Zone: Single AZ, 47% cheaper
    One Zone-IA: Combine for cost savings
    
  Lifecycle Management:
    Auto-move to IA/Archive based on access
    Configurable transition periods
    
  Access:
    Security groups on mount targets
    IAM policies for mount
    Access points: Application-specific entry
    NFS client (amazon-efs-utils)
    ECS/EKS integration

**Amazon FSx:**

FSx for Lustre:
  High-performance file system
  100+ GB/s throughput, millions of IOPS
  S3 integration (lazy loading)
  Use: Machine learning, HPC, media processing
  
FSx for Windows File Server:
  Windows-native (SMB protocol)
  Active Directory integration
  DFSR replication
  VSS snapshots
  Deduplication, compression
  
FSx for NetApp ONTAP:
  NFS, SMB, iSCSI
  SnapMirror replication
  FlexClone (instant copies)
  Multi-protocol access
  
FSx for OpenZFS:
  ZFS file system
  NFS protocol
  Snapshots, clones
  Data compression (up to 4x)

**AWS Storage Gateway:**

S3 File Gateway:
  NFS/SMB -> S3
  Local cache for low-latency access
  
FSx File Gateway:
  SMB -> FSx for Windows
  Local cache on-premises
  
Volume Gateway:
  iSCSI block storage
  Cached: Primary data in S3, cache local
  Stored: Primary data local, async backup to S3
  
Tape Gateway:
  iSCSI virtual tape library
  Backup to S3 Glacier`,
					CodeExamples: `// AWS storage service implementations

package main

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "sort"
    "strings"
    "sync"
    "time"
)

// S3 bucket simulator
type S3Bucket struct {
    Name           string
    Region         string
    Versioning     bool
    Encryption     string
    PublicBlocked  bool
    LifecycleRules []S3LifecycleRule
    Objects        map[string][]*S3Object // key -> versions
    mu             sync.RWMutex
}

type S3Object struct {
    Key           string
    Size          int64
    StorageClass  string
    ContentType   string
    ETag          string
    LastModified  time.Time
    VersionID     string
    IsLatest      bool
    DeleteMarker  bool
    Data          []byte
    Metadata      map[string]string
    Encryption    string
    Tags          map[string]string
}

type S3LifecycleRule struct {
    ID          string
    Prefix      string
    Enabled     bool
    Transitions []S3Transition
    Expiration  *S3Expiration
}

type S3Transition struct {
    Days         int
    StorageClass string
}

type S3Expiration struct {
    Days                    int
    ExpiredDeleteMarker     bool
    NoncurrentDays          int
    NoncurrentVersions      int
}

func NewS3Bucket(name, region string) *S3Bucket {
    return &S3Bucket{
        Name:          name,
        Region:        region,
        Encryption:    "AES256",
        PublicBlocked: true,
        Objects:       make(map[string][]*S3Object),
    }
}

func (b *S3Bucket) PutObject(key string, data []byte, contentType string) (*S3Object, error) {
    b.mu.Lock()
    defer b.mu.Unlock()
    
    hash := sha256.Sum256(data)
    etag := hex.EncodeToString(hash[:])
    
    obj := &S3Object{
        Key:          key,
        Size:         int64(len(data)),
        StorageClass: "STANDARD",
        ContentType:  contentType,
        ETag:         etag,
        LastModified: time.Now(),
        IsLatest:     true,
        Data:         data,
        Metadata:     make(map[string]string),
        Tags:         make(map[string]string),
        Encryption:   b.Encryption,
    }
    
    if b.Versioning {
        obj.VersionID = fmt.Sprintf("v%d", time.Now().UnixNano())
        // Mark previous as not latest
        for _, v := range b.Objects[key] {
            v.IsLatest = false
        }
        b.Objects[key] = append(b.Objects[key], obj)
    } else {
        b.Objects[key] = []*S3Object{obj}
    }
    
    return obj, nil
}

func (b *S3Bucket) GetObject(key string) (*S3Object, error) {
    b.mu.RLock()
    defer b.mu.RUnlock()
    
    versions, exists := b.Objects[key]
    if !exists || len(versions) == 0 {
        return nil, fmt.Errorf("NoSuchKey: %s", key)
    }
    
    // Get latest non-delete-marker
    for i := len(versions) - 1; i >= 0; i-- {
        if !versions[i].DeleteMarker {
            return versions[i], nil
        }
    }
    
    return nil, fmt.Errorf("NoSuchKey: %s (delete marker)", key)
}

func (b *S3Bucket) DeleteObject(key string) error {
    b.mu.Lock()
    defer b.mu.Unlock()
    
    if b.Versioning {
        // Add delete marker
        dm := &S3Object{
            Key:          key,
            DeleteMarker: true,
            IsLatest:     true,
            LastModified: time.Now(),
            VersionID:    fmt.Sprintf("dm%d", time.Now().UnixNano()),
        }
        for _, v := range b.Objects[key] {
            v.IsLatest = false
        }
        b.Objects[key] = append(b.Objects[key], dm)
    } else {
        delete(b.Objects, key)
    }
    
    return nil
}

func (b *S3Bucket) ListObjects(prefix string, maxKeys int) []*S3Object {
    b.mu.RLock()
    defer b.mu.RUnlock()
    
    var results []*S3Object
    
    for key, versions := range b.Objects {
        if prefix != "" && !strings.HasPrefix(key, prefix) {
            continue
        }
        
        // Get latest version
        for i := len(versions) - 1; i >= 0; i-- {
            if !versions[i].DeleteMarker {
                results = append(results, versions[i])
                break
            }
        }
    }
    
    sort.Slice(results, func(i, j int) bool {
        return results[i].Key < results[j].Key
    })
    
    if maxKeys > 0 && len(results) > maxKeys {
        results = results[:maxKeys]
    }
    
    return results
}

func (b *S3Bucket) ApplyLifecycleRules() map[string][]string {
    b.mu.Lock()
    defer b.mu.Unlock()
    
    actions := make(map[string][]string)
    now := time.Now()
    
    for _, rule := range b.LifecycleRules {
        if !rule.Enabled {
            continue
        }
        
        for key, versions := range b.Objects {
            if rule.Prefix != "" && !strings.HasPrefix(key, rule.Prefix) {
                continue
            }
            
            for _, obj := range versions {
                age := int(now.Sub(obj.LastModified).Hours() / 24)
                
                // Transitions
                for _, transition := range rule.Transitions {
                    if age >= transition.Days && obj.StorageClass != transition.StorageClass {
                        oldClass := obj.StorageClass
                        obj.StorageClass = transition.StorageClass
                        actions["transition"] = append(actions["transition"],
                            fmt.Sprintf("%s: %s -> %s", key, oldClass, transition.StorageClass))
                    }
                }
                
                // Expiration
                if rule.Expiration != nil && rule.Expiration.Days > 0 {
                    if age >= rule.Expiration.Days && obj.IsLatest {
                        actions["expire"] = append(actions["expire"], key)
                    }
                }
            }
        }
    }
    
    return actions
}

// Calculate storage cost
func (b *S3Bucket) EstimateMonthlyCost() float64 {
    b.mu.RLock()
    defer b.mu.RUnlock()
    
    var totalCost float64
    
    classPricing := map[string]float64{
        "STANDARD":            0.023,
        "STANDARD_IA":         0.0125,
        "ONEZONE_IA":          0.01,
        "INTELLIGENT_TIERING": 0.023,
        "GLACIER":             0.004,
        "GLACIER_IR":          0.004,
        "DEEP_ARCHIVE":        0.00099,
    }
    
    classSizes := make(map[string]int64)
    for _, versions := range b.Objects {
        for _, obj := range versions {
            if !obj.DeleteMarker {
                classSizes[obj.StorageClass] += obj.Size
            }
        }
    }
    
    for class, size := range classSizes {
        pricePerGB := classPricing[class]
        sizeGB := float64(size) / (1024 * 1024 * 1024)
        totalCost += sizeGB * pricePerGB
    }
    
    return totalCost
}

// EBS volume manager
type EBSManager struct {
    volumes   map[string]*EBSVolume
    snapshots map[string]*EBSSnapshot
    mu        sync.RWMutex
}

type EBSVolume struct {
    ID         string
    Type       string // gp3, gp2, io2, io1, st1, sc1
    Size       int    // GiB
    IOPS       int
    Throughput int    // MiB/s
    AZ         string
    State      string
    Encrypted  bool
    KMSKeyID   string
    AttachedTo string
    MultiAttach bool
    Tags       map[string]string
    CreatedAt  time.Time
}

type EBSSnapshot struct {
    ID         string
    VolumeID   string
    Size       int
    State      string
    Encrypted  bool
    StartTime  time.Time
    Description string
    Tags       map[string]string
}

func NewEBSManager() *EBSManager {
    return &EBSManager{
        volumes:   make(map[string]*EBSVolume),
        snapshots: make(map[string]*EBSSnapshot),
    }
}

func (m *EBSManager) CreateVolume(volType string, size, iops int, az string, encrypted bool) (*EBSVolume, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    // Validate
    switch volType {
    case "gp3":
        if iops == 0 {
            iops = 3000
        }
        if iops > 16000 {
            return nil, fmt.Errorf("gp3 max IOPS is 16000")
        }
    case "io2":
        if iops > 256000 {
            return nil, fmt.Errorf("io2 Block Express max IOPS is 256000")
        }
    case "st1":
        if size < 125 {
            return nil, fmt.Errorf("st1 minimum size is 125 GiB")
        }
    }
    
    vol := &EBSVolume{
        ID:        fmt.Sprintf("vol-%s", generateStorageID()),
        Type:      volType,
        Size:      size,
        IOPS:      iops,
        AZ:        az,
        State:     "available",
        Encrypted: encrypted,
        Tags:      make(map[string]string),
        CreatedAt: time.Now(),
    }
    
    m.volumes[vol.ID] = vol
    return vol, nil
}

func (m *EBSManager) CreateSnapshot(volumeID, description string) (*EBSSnapshot, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    vol, exists := m.volumes[volumeID]
    if !exists {
        return nil, fmt.Errorf("volume %s not found", volumeID)
    }
    
    snap := &EBSSnapshot{
        ID:          fmt.Sprintf("snap-%s", generateStorageID()),
        VolumeID:    volumeID,
        Size:        vol.Size,
        State:       "completed",
        Encrypted:   vol.Encrypted,
        StartTime:   time.Now(),
        Description: description,
        Tags:        make(map[string]string),
    }
    
    m.snapshots[snap.ID] = snap
    return snap, nil
}

func (m *EBSManager) AttachVolume(volumeID, instanceID string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    vol, exists := m.volumes[volumeID]
    if !exists {
        return fmt.Errorf("volume %s not found", volumeID)
    }
    
    if vol.AttachedTo != "" && !vol.MultiAttach {
        return fmt.Errorf("volume already attached to %s", vol.AttachedTo)
    }
    
    vol.AttachedTo = instanceID
    vol.State = "in-use"
    return nil
}

// EFS file system
type EFSFileSystem struct {
    ID             string
    PerformanceMode string
    ThroughputMode string
    ProvisionedThroughput float64
    Encrypted      bool
    LifecyclePolicy string
    SizeBytes      int64
    MountTargets   []EFSMountTarget
    AccessPoints   []EFSAccessPoint
}

type EFSMountTarget struct {
    ID        string
    SubnetID  string
    AZ        string
    IPAddress string
    SecurityGroups []string
}

type EFSAccessPoint struct {
    ID       string
    Path     string
    PosixUser *PosixUser
    RootDir  *RootDirectory
}

type PosixUser struct {
    UID int
    GID int
}

type RootDirectory struct {
    Path        string
    Permissions string
    OwnerUID    int
    OwnerGID    int
}

func generateStorageID() string {
    return fmt.Sprintf("%012x", time.Now().UnixNano()%0xFFFFFFFFFFFF)
}`,
				},
			},
		},
	})
}
