package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          208,
			Title:       "System Interconnects",
			Description: "Learn about system interconnect technologies: PCIe architecture, USB standards, Thunderbolt, Network-on-Chip, and bus arbitration protocols.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "PCIe Architecture and Protocol",
					Content: `PCIe (Peripheral Component Interconnect Express) is the standard high-speed interconnect for connecting peripherals to the CPU.

**PCIe Architecture:**

**Layered Architecture:**
- **Transaction Layer**: Handles transactions, TLP (Transaction Layer Packet)
- **Data Link Layer**: Error detection/correction, flow control
- **Physical Layer**: Electrical signaling, encoding

**Lane Configuration:**
- **x1**: 1 lane (250 MB/s per direction, Gen 1)
- **x2**: 2 lanes
- **x4**: 4 lanes
- **x8**: 8 lanes
- **x16**: 16 lanes (common for GPUs)

**PCIe Generations:**

**Gen 1**: 2.5 GT/s per lane (250 MB/s)
**Gen 2**: 5 GT/s per lane (500 MB/s)
**Gen 3**: 8 GT/s per lane (~985 MB/s)
**Gen 4**: 16 GT/s per lane (~1.97 GB/s)
**Gen 5**: 32 GT/s per lane (~3.94 GB/s)
**Gen 6**: 64 GT/s per lane (planned)

**Topology:**
- **Root Complex**: Connects CPU to PCIe
- **Switches**: Route traffic between devices
- **Endpoints**: PCIe devices (GPU, NIC, SSD, etc.)

**Transaction Types:**
- **Memory Read/Write**: Access device memory
- **I/O Read/Write**: Legacy I/O space
- **Configuration**: Device configuration
- **Message**: Interrupts, power management`,
					CodeExamples: `// Example: PCIe configuration space access
// Each device has 256-byte configuration space
// First 64 bytes standardized

struct pci_config_header {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t command;
    uint16_t status;
    uint8_t revision_id;
    uint8_t class_code[3];
    uint8_t cache_line_size;
    uint8_t latency_timer;
    uint8_t header_type;
    uint8_t bist;
    uint32_t bar[6];  // Base Address Registers
    uint32_t cardbus_cis;
    uint16_t subsystem_vendor_id;
    uint16_t subsystem_id;
    uint32_t expansion_rom_base;
    uint8_t capabilities_ptr;
    uint8_t reserved[7];
    uint8_t interrupt_line;
    uint8_t interrupt_pin;
    uint8_t min_gnt;
    uint8_t max_lat;
};

// Read configuration space
uint32_t pci_read_config(uint8_t bus, uint8_t device, 
                          uint8_t function, uint8_t offset) {
    uint32_t address = (1 << 31) |           // Enable bit
                       (bus << 16) |         // Bus number
                       (device << 11) |      // Device number
                       (function << 8) |     // Function number
                       (offset & 0xFC);      // Register offset
    
    outl(0xCF8, address);  // Configuration address port
    return inl(0xCFC);      // Configuration data port
}

// Example: PCIe memory mapping
void map_pcie_device(uint8_t bus, uint8_t device) {
    // Read BAR (Base Address Register)
    uint32_t bar = pci_read_config(bus, device, 0, 0x10);
    
    // Determine BAR type
    if (bar & 1) {
        // I/O space (legacy)
        uint16_t io_base = bar & 0xFFFC;
        // Map I/O port
    } else {
        // Memory space
        uint64_t mem_base;
        if ((bar >> 1) & 1) {
            // 64-bit BAR
            uint32_t bar_high = pci_read_config(bus, device, 0, 0x14);
            mem_base = ((uint64_t)bar_high << 32) | (bar & 0xFFFFFFF0);
        } else {
            // 32-bit BAR
            mem_base = bar & 0xFFFFFFF0;
        }
        // Map to virtual address space
        void *vaddr = ioremap(mem_base, device_size);
    }
}

// Example: PCIe transaction (simplified)
struct pcie_tlp {
    uint32_t header[4];
    uint8_t data[];
};

// Memory read request
void pcie_memory_read(uint64_t address, uint32_t length) {
    struct pcie_tlp tlp = {0};
    
    // TLP header
    tlp.header[0] = (3 << 0) |        // Format: Memory Read
                    (0 << 4) |        // Type: Memory Read
                    (0 << 8) |        // T9-T10: Reserved
                    (length << 20) |  // Length in DW
                    (0 << 29) |       // Attr
                    (0 << 32);        // Requester ID
    
    tlp.header[1] = address & 0xFFFFFFFF;  // Address low
    tlp.header[2] = (address >> 32) & 0xFFFFFFFF;  // Address high
    
    // Send TLP
    send_tlp(&tlp);
}`,
				},
				{
					Title: "USB Standards and Architecture",
					Content: `USB (Universal Serial Bus) is the ubiquitous interface for connecting peripherals. Understanding USB architecture is essential for system design.

**USB Evolution:**

**USB 1.0/1.1:**
- Low Speed: 1.5 Mbps
- Full Speed: 12 Mbps
- Legacy standard

**USB 2.0:**
- High Speed: 480 Mbps
- Backward compatible
- Still widely used

**USB 3.x:**
- **USB 3.0 (Gen 1)**: 5 Gbps
- **USB 3.1 (Gen 2)**: 10 Gbps
- **USB 3.2 (Gen 2x2)**: 20 Gbps (dual lane)

**USB4:**
- Based on Thunderbolt 3
- Up to 40 Gbps
- DisplayPort, PCIe tunneling

**USB Architecture:**

**Host-Device Model:**
- **Host**: Computer (initiates communication)
- **Device**: Peripheral (responds)
- **Hub**: Expands connectivity

**USB Topology:**
- Tiered star topology
- Up to 5 tiers (host + 4 hub levels)
- Up to 127 devices per host

**USB Communication:**

**Endpoints:**
- Unidirectional communication channel
- **Control**: Device configuration
- **Interrupt**: Time-sensitive data
- **Bulk**: Large data transfers
- **Isochronous**: Real-time streaming

**Pipes:**
- Logical connection to endpoint
- **Control Pipe**: Always endpoint 0
- **Data Pipes**: Other endpoints`,
					CodeExamples: `// Example: USB device descriptor
struct usb_device_descriptor {
    uint8_t bLength;           // Descriptor length
    uint8_t bDescriptorType;   // Descriptor type (1 = device)
    uint16_t bcdUSB;           // USB spec version
    uint8_t bDeviceClass;      // Device class
    uint8_t bDeviceSubClass;   // Device subclass
    uint8_t bDeviceProtocol;   // Device protocol
    uint8_t bMaxPacketSize0;   // Max packet size for EP0
    uint16_t idVendor;         // Vendor ID
    uint16_t idProduct;        // Product ID
    uint16_t bcdDevice;        // Device version
    uint8_t iManufacturer;     // Manufacturer string index
    uint8_t iProduct;          // Product string index
    uint8_t iSerialNumber;     // Serial number string index
    uint8_t bNumConfigurations;// Number of configurations
};

// Example: USB endpoint descriptor
struct usb_endpoint_descriptor {
    uint8_t bLength;
    uint8_t bDescriptorType;   // Descriptor type (5 = endpoint)
    uint8_t bEndpointAddress;  // Endpoint address + direction
    uint8_t bmAttributes;      // Transfer type + synchronization
    uint16_t wMaxPacketSize;   // Max packet size
    uint8_t bInterval;         // Polling interval
};

// Example: USB control transfer
int usb_control_transfer(usb_device_t *dev, uint8_t request_type,
                          uint8_t request, uint16_t value,
                          uint16_t index, void *data, uint16_t length) {
    struct usb_control_request ctrl = {0};
    
    ctrl.bmRequestType = request_type;
    ctrl.bRequest = request;
    ctrl.wValue = value;
    ctrl.wIndex = index;
    ctrl.wLength = length;
    
    // Setup stage
    send_setup_packet(dev, &ctrl);
    
    // Data stage (if needed)
    if (length > 0) {
        if (request_type & 0x80) {
            // Device to host
            receive_data(dev, data, length);
        } else {
            // Host to device
            send_data(dev, data, length);
        }
    }
    
    // Status stage
    if (request_type & 0x80) {
        send_status(dev, 0);  // ACK
    } else {
        receive_status(dev);  // ACK
    }
    
    return 0;
}

// Example: USB bulk transfer
int usb_bulk_transfer(usb_device_t *dev, uint8_t endpoint,
                      void *data, uint32_t length) {
    uint8_t ep_addr = endpoint | 0x80;  // IN endpoint
    
    // Split into packets
    uint32_t offset = 0;
    while (offset < length) {
        uint16_t packet_size = min(length - offset, dev->max_packet_size);
        
        // Send/receive packet
        if (endpoint & 0x80) {
            receive_packet(dev, ep_addr, data + offset, packet_size);
        } else {
            send_packet(dev, endpoint, data + offset, packet_size);
        }
        
        offset += packet_size;
    }
    
    return 0;
}

// Example: USB enumeration
void usb_enumerate_device(usb_device_t *dev) {
    // Get device descriptor
    struct usb_device_descriptor desc;
    usb_control_transfer(dev, 0x80, 0x06, 0x0100, 0,
                        &desc, sizeof(desc));
    
    // Set address
    usb_control_transfer(dev, 0x00, 0x05, dev->address, 0,
                        NULL, 0);
    
    // Get configuration descriptor
    struct usb_config_descriptor config;
    usb_control_transfer(dev, 0x80, 0x06, 0x0200, 0,
                        &config, sizeof(config));
    
    // Set configuration
    usb_control_transfer(dev, 0x00, 0x09, config.bConfigurationValue, 0,
                        NULL, 0);
}`,
				},
				{
					Title: "Thunderbolt and High-Speed Interconnects",
					Content: `Thunderbolt combines PCIe and DisplayPort into a single high-speed interface, enabling powerful connectivity options.

**Thunderbolt Evolution:**

**Thunderbolt 1/2:**
- 10 Gbps per channel (20 Gbps total)
- Mini DisplayPort connector
- PCIe 2.0 ×4 + DisplayPort

**Thunderbolt 3:**
- 40 Gbps total bandwidth
- USB-C connector
- PCIe 3.0 ×4 + DisplayPort 1.2
- USB 3.1 Gen 2 support
- Power delivery (up to 100W)

**Thunderbolt 4:**
- 40 Gbps (same as TB3)
- Stricter requirements
- PCIe 3.0 ×4 (32 Gbps guaranteed)
- USB4 compatible
- Longer cables supported

**Thunderbolt Architecture:**

**Tunneling:**
- PCIe traffic tunneled over Thunderbolt
- DisplayPort traffic tunneled
- USB traffic tunneled
- All share same physical link

**Topology:**
- Daisy-chain capable
- Up to 6 devices per port
- Hot-pluggable
- Power delivery

**Security:**
- IOMMU for DMA protection
- Authentication for devices
- Secure connection establishment

**Use Cases:**
- External GPUs (eGPU)
- High-speed storage
- Docking stations
- Multiple displays
- Network adapters`,
					CodeExamples: `// Example: Thunderbolt device discovery
// Thunderbolt uses PCIe configuration space extensions

struct thunderbolt_capability {
    uint16_t vendor_id;      // 0x8086 (Intel)
    uint16_t device_id;
    uint8_t cap_id;          // 0x0A (Thunderbolt)
    uint8_t next_cap;
    uint8_t cap_version;
    uint8_t cap_length;
    // Thunderbolt-specific registers
};

// Example: Thunderbolt PCIe tunneling
// PCIe transactions encapsulated in Thunderbolt packets

struct thunderbolt_packet {
    uint32_t header;
    uint8_t pcie_tlp[];  // Encapsulated PCIe TLP
};

// PCIe device appears as if directly connected
// Software sees normal PCIe device
// Hardware handles tunneling transparently

// Example: External GPU via Thunderbolt
// 1. Connect eGPU enclosure via Thunderbolt
// 2. System enumerates GPU as PCIe device
// 3. GPU driver loads normally
// 4. PCIe transactions tunneled over Thunderbolt

void enumerate_thunderbolt_gpu(void) {
    // Discover Thunderbolt controller
    struct pci_device *tb_controller = find_thunderbolt_controller();
    
    // Enumerate devices on Thunderbolt bus
    for (int i = 0; i < 6; i++) {
        struct pci_device *device = 
            pci_scan_thunderbolt_bus(tb_controller, i);
        
        if (device && device->vendor_id == GPU_VENDOR_ID) {
            // Found GPU
            initialize_gpu(device);
        }
    }
}

// Example: Thunderbolt security (IOMMU)
// Protect against malicious DMA

void setup_thunderbolt_iommu(struct thunderbolt_device *dev) {
    // Map device to IOMMU domain
    iommu_domain_t *domain = iommu_create_domain();
    
    // Set up DMA remapping
    iommu_map(domain, dev->dma_address, 
              physical_address, size, IOMMU_READ | IOMMU_WRITE);
    
    // Attach device to domain
    iommu_attach_device(domain, dev);
    
    // Device can only access mapped memory
}

// Example: USB4 (based on Thunderbolt 3)
// USB4 uses Thunderbolt 3 protocol
// Backward compatible with USB 3.x and Thunderbolt 3

struct usb4_capability {
    uint8_t version;         // USB4 version
    uint8_t route_string[6]; // Route string
    uint32_t adapter_id;     // Adapter ID
};

// USB4 supports:
// - PCIe tunneling (like Thunderbolt)
// - DisplayPort tunneling
// - USB 3.x tunneling
// - USB 2.0 (always available)`,
				},
				{
					Title: "Network-on-Chip (NoC)",
					Content: `Network-on-Chip is an interconnect architecture for System-on-Chip (SoC) designs, replacing traditional buses with packet-switched networks.

**NoC Architecture:**

**Why NoC?**
- Traditional buses don't scale
- Multiple masters need parallel access
- Lower latency than buses
- Better power efficiency

**NoC Components:**

**Routers:**
- Route packets between nodes
- Store-and-forward or wormhole routing
- Multiple ports (typically 5: N, S, E, W, local)

**Links:**
- Physical connections between routers
- Bidirectional or unidirectional
- Multiple parallel links possible

**Network Interfaces:**
- Connect IP cores to network
- Convert transactions to packets
- Handle flow control

**Topologies:**

**Mesh:**
- 2D grid of routers
- Simple, scalable
- Common in many-core processors

**Torus:**
- Mesh with wrap-around links
- Lower average hop count
- More complex routing

**Ring:**
- Simple topology
- Limited scalability
- Low cost

**Tree:**
- Hierarchical structure
- Good for cache coherence
- Potential bottlenecks at root`,
					CodeExamples: `// Example: NoC packet structure
struct noc_packet {
    uint32_t header;
    uint32_t address;
    uint32_t data[];
    uint32_t tail;
};

// Header fields:
// - Source ID
// - Destination ID
// - Packet type (read, write, response)
// - Packet length
// - Virtual channel ID

// Example: NoC router (simplified)
struct noc_router {
    struct noc_packet input_buffer[5][BUFFER_SIZE];  // 5 ports
    struct noc_router *neighbors[4];  // N, S, E, W
    uint8_t router_id;
};

void noc_router_route(struct noc_router *router) {
    // Process packets from each input port
    for (int port = 0; port < 5; port++) {
        struct noc_packet *packet = 
            get_packet(&router->input_buffer[port]);
        
        if (packet == NULL) continue;
        
        // Determine output port
        int output_port = route_packet(router, packet);
        
        // Forward packet
        if (output_port == LOCAL_PORT) {
            // Deliver to local IP core
            deliver_to_ip_core(packet);
        } else {
            // Forward to neighbor router
            forward_to_neighbor(router->neighbors[output_port], 
                               packet);
        }
    }
}

// Example: XY routing (mesh topology)
int route_xy(struct noc_router *router, struct noc_packet *packet) {
    uint8_t src_x = router->router_id % MESH_WIDTH;
    uint8_t src_y = router->router_id / MESH_WIDTH;
    uint8_t dst_x = packet->dest_id % MESH_WIDTH;
    uint8_t dst_y = packet->dest_id / MESH_WIDTH;
    
    // Route X first, then Y
    if (dst_x > src_x) {
        return EAST_PORT;
    } else if (dst_x < src_x) {
        return WEST_PORT;
    } else if (dst_y > src_y) {
        return SOUTH_PORT;
    } else if (dst_y < src_y) {
        return NORTH_PORT;
    } else {
        return LOCAL_PORT;  // Destination reached
    }
}

// Example: NoC transaction (memory read)
void noc_memory_read(uint32_t address, uint32_t *data) {
    // Create read request packet
    struct noc_packet request = {0};
    request.header = PACKET_TYPE_READ | 
                     (my_core_id << SOURCE_SHIFT) |
                     (memory_controller_id << DEST_SHIFT);
    request.address = address;
    
    // Send packet
    send_packet(&request);
    
    // Wait for response
    struct noc_packet *response = wait_for_response();
    *data = response->data[0];
}

// Example: Virtual channels (avoid deadlock)
struct virtual_channel {
    struct noc_packet buffer[VC_DEPTH];
    int head, tail;
    int credits;  // Flow control
};

// Multiple virtual channels per physical link
// Different message types use different VCs
// Prevents deadlock`,
				},
				{
					Title: "Bus Arbitration and Protocols",
					Content: `Bus arbitration determines which device can use the bus when multiple devices request access simultaneously.

**Arbitration Methods:**

**Daisy Chain (Priority):**
- Fixed priority based on position
- Simple but unfair
- Lower priority devices may starve

**Round Robin:**
- Equal time slots for each device
- Fair but may waste bandwidth
- Good for equal-priority devices

**Time Division Multiple Access (TDMA):**
- Fixed time slots
- Predictable latency
- May waste bandwidth

**Token Passing:**
- Token circulates among devices
- Device with token can transmit
- Fair, no collisions

**Arbitration Protocols:**

**Centralized:**
- Single arbiter decides
- Simple, fast
- Single point of failure

**Distributed:**
- Each device participates
- More complex
- No single point of failure

**Bus Protocols:**

**Synchronous:**
- Clocked protocol
- All signals aligned to clock
- Simpler timing

**Asynchronous:**
- Handshake-based
- No global clock
- More complex but flexible`,
					CodeExamples: `// Example: Centralized bus arbiter
struct bus_arbiter {
    uint32_t request_lines;  // One bit per device
    uint32_t grant_lines;    // One bit per device
    int current_master;
    enum {IDLE, ARBITRATING} state;
};

void bus_arbiter_tick(struct bus_arbiter *arbiter) {
    // Check for requests
    if (arbiter->request_lines == 0) {
        arbiter->state = IDLE;
        arbiter->grant_lines = 0;
        return;
    }
    
    // Grant to highest priority requester
    if (arbiter->state == IDLE) {
        int highest_priority = find_highest_priority(arbiter->request_lines);
        arbiter->grant_lines = (1 << highest_priority);
        arbiter->current_master = highest_priority;
        arbiter->state = ARBITRATING;
    }
    
    // Release grant when done
    if (!(arbiter->request_lines & (1 << arbiter->current_master))) {
        arbiter->grant_lines = 0;
        arbiter->state = IDLE;
    }
}

// Example: Round-robin arbitration
void round_robin_arbitrate(struct bus_arbiter *arbiter) {
    static int current_priority = 0;
    
    // Start from last granted device + 1
    for (int i = 0; i < NUM_DEVICES; i++) {
        int device = (current_priority + i) % NUM_DEVICES;
        
        if (arbiter->request_lines & (1 << device)) {
            arbiter->grant_lines = (1 << device);
            current_priority = (device + 1) % NUM_DEVICES;
            return;
        }
    }
    
    arbiter->grant_lines = 0;
}

// Example: Token passing
struct token_ring {
    int token_holder;
    int num_devices;
};

void token_passing_tick(struct token_ring *ring) {
    // Current holder can transmit
    if (ring->token_holder >= 0) {
        // Device transmits if it has data
        if (device_has_data(ring->token_holder)) {
            transmit_data(ring->token_holder);
        }
        
        // Pass token to next device
        ring->token_holder = 
            (ring->token_holder + 1) % ring->num_devices;
    }
}

// Example: Bus transaction (simplified)
void bus_transaction(struct bus_device *master, 
                    uint32_t address, uint32_t *data, 
                    enum {READ, WRITE} type) {
    // Request bus
    set_request_line(master->device_id);
    
    // Wait for grant
    while (!(get_grant_lines() & (1 << master->device_id))) {
        // Wait...
    }
    
    // Drive address and control signals
    set_address(address);
    set_read_write(type == READ ? 1 : 0);
    
    if (type == WRITE) {
        set_data(*data);
    }
    
    // Assert transaction start
    assert_transaction_start();
    
    // Wait for acknowledgment
    while (!get_acknowledge()) {
        // Wait...
    }
    
    if (type == READ) {
        *data = get_data();
    }
    
    // Release bus
    clear_request_line(master->device_id);
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
