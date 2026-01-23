package computer_architecture

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterComputerArchitectureModules([]problems.CourseModule{
		{
			ID:          204,
			Title:       "Embedded Systems and Microcontrollers",
			Description: "Learn about microcontroller architecture, real-time systems, interrupt handling, power management, and peripheral interfaces.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Microcontroller Architecture",
					Content: `Microcontrollers are complete computer systems on a single chip, designed for embedded applications with constraints on power, cost, and size.

**Microcontroller Components:**

**1. CPU Core:**
- Typically 8-bit, 16-bit, or 32-bit
- RISC architecture common (ARM Cortex-M, AVR, RISC-V)
- Lower clock speeds (MHz range)
- Optimized for low power

**2. Memory:**
- **Flash Memory**: Program storage (non-volatile)
- **SRAM**: Data storage (volatile)
- **EEPROM**: Small non-volatile data storage
- Limited capacity (KB to MB range)

**3. Peripherals:**
- **GPIO**: General Purpose Input/Output pins
- **Timers/Counters**: For timing and PWM
- **ADC/DAC**: Analog-to-Digital and Digital-to-Analog converters
- **Communication**: UART, SPI, I2C, CAN, USB
- **Interrupt Controller**: Manage interrupts

**4. Power Management:**
- Sleep modes for power saving
- Clock gating
- Voltage scaling

**Common Microcontroller Families:**

**ARM Cortex-M:**
- 32-bit RISC architecture
- Wide range (M0 to M7)
- Used in many embedded applications
- Examples: STM32, NXP LPC, Nordic nRF

**AVR:**
- 8-bit RISC architecture
- Popular in hobbyist projects
- Examples: ATmega328 (Arduino Uno), ATtiny

**RISC-V:**
- Open-source instruction set
- Growing adoption
- Examples: SiFive, ESP32-C3`,
					CodeExamples: `// Example: ARM Cortex-M register model
// Core registers: R0-R15
// R13: Stack Pointer (SP)
// R14: Link Register (LR)
// R15: Program Counter (PC)
// 
// Special registers:
// - Program Status Register (PSR)
// - Control Register
// - Interrupt Mask Registers

// Example: AVR register model
// 32 general-purpose registers (R0-R31)
// Special registers:
// - Status Register (SREG)
// - Stack Pointer (SP)
// - Program Counter (PC)

// Example: GPIO configuration (ARM Cortex-M style)
// Configure pin as output
GPIOA->MODER |= (1 << (PIN * 2));  // Set mode to output
GPIOA->OTYPER &= ~(1 << PIN);       // Push-pull output
GPIOA->OSPEEDR |= (3 << (PIN * 2)); // High speed
GPIOA->PUPDR &= ~(3 << (PIN * 2));  // No pull-up/pull-down

// Set pin high
GPIOA->BSRR = (1 << PIN);

// Set pin low
GPIOA->BSRR = (1 << (PIN + 16));

// Example: GPIO input with pull-up
GPIOA->MODER &= ~(3 << (PIN * 2));  // Input mode
GPIOA->PUPDR |= (1 << (PIN * 2));    // Pull-up enabled

// Read pin state
uint32_t pin_state = (GPIOA->IDR >> PIN) & 1;`,
				},
				{
					Title: "Real-Time Systems",
					Content: `Real-time systems must respond to events within guaranteed time constraints. Understanding real-time requirements is crucial for embedded systems.

**Real-Time System Types:**

**Hard Real-Time:**
- Missing deadline causes system failure
- Safety-critical applications
- Examples: Medical devices, automotive brakes, flight control

**Soft Real-Time:**
- Missing deadline degrades performance
- Acceptable occasional delays
- Examples: Multimedia streaming, user interfaces

**Firm Real-Time:**
- Occasional deadline misses acceptable
- But must be rare
- Examples: Network protocols, some control systems

**Real-Time Constraints:**

**Response Time:**
- Time from event to response
- Must be bounded and predictable

**Jitter:**
- Variation in response time
- Should be minimized

**Determinism:**
- Predictable execution time
- No unbounded operations

**Real-Time Scheduling:**

**Rate Monotonic (RM):**
- Higher frequency = higher priority
- Preemptive
- Optimal for fixed priorities

**Earliest Deadline First (EDF):**
- Earlier deadline = higher priority
- Dynamic priority
- Optimal scheduling algorithm

**Priority Inversion:**
- Low-priority task blocks high-priority task
- Solved with priority inheritance or ceiling protocol`,
					CodeExamples: `// Example: Real-time task structure
typedef struct {
    uint32_t period;        // Task period (ms)
    uint32_t deadline;      // Task deadline (ms)
    uint32_t wcet;          // Worst-case execution time (ms)
    uint32_t priority;      // Task priority
    void (*task_func)(void); // Task function
} rt_task_t;

// Example: Rate Monotonic scheduling
// Task 1: period = 10ms, wcet = 3ms
// Task 2: period = 20ms, wcet = 5ms
// Task 3: period = 40ms, wcet = 8ms
//
// Priority assignment (higher frequency = higher priority):
// Task 1: priority = 3 (highest)
// Task 2: priority = 2
// Task 3: priority = 1 (lowest)

// Example: Task execution
void task1_handler(void) {
    // Critical section
    // Must complete within 3ms
    read_sensor();
    process_data();
    update_output();
}

void task2_handler(void) {
    // Must complete within 5ms
    update_display();
}

// Example: Scheduler (simplified)
void rt_scheduler(void) {
    while (1) {
        // Check for ready tasks
        for (int i = 0; i < num_tasks; i++) {
            if (is_task_ready(&tasks[i])) {
                if (should_preempt(current_task, &tasks[i])) {
                    context_switch(&tasks[i]);
                }
            }
        }
        // Execute current task
        current_task->task_func();
    }
}`,
				},
				{
					Title: "Interrupt Handling in Embedded Systems",
					Content: `Interrupts are essential for embedded systems to respond to external events efficiently. Understanding interrupt handling is crucial.

**Interrupt Types:**

**External Interrupts:**
- Triggered by external hardware
- GPIO pin changes
- Communication interfaces
- Timers

**Internal Interrupts:**
- Timer overflow
- ADC conversion complete
- Communication events
- System exceptions

**Interrupt Handling Process:**

1. **Interrupt Occurs**: Hardware sets interrupt flag
2. **CPU Saves Context**: Push registers onto stack
3. **Jump to ISR**: Load interrupt vector address
4. **Execute ISR**: Handle the interrupt
5. **Restore Context**: Pop registers from stack
6. **Return**: Resume interrupted code

**Interrupt Priorities:**
- Higher priority interrupts can preempt lower priority
- Nested interrupts possible
- Critical sections need interrupt disabling

**Interrupt Service Routine (ISR) Best Practices:**
- Keep ISR short and fast
- Avoid blocking operations
- Use flags for deferred processing
- Disable interrupts only when necessary
- Clear interrupt flags`,
					CodeExamples: `// Example: GPIO interrupt (ARM Cortex-M)
// Configure pin for interrupt
GPIOA->MODER &= ~(3 << (PIN * 2));  // Input mode
GPIOA->PUPDR |= (1 << (PIN * 2));    // Pull-up
EXTI->IMR |= (1 << PIN);             // Enable interrupt
EXTI->RTSR |= (1 << PIN);            // Rising edge trigger
NVIC_EnableIRQ(EXTI0_IRQn);          // Enable in NVIC

// Interrupt Service Routine
void EXTI0_IRQHandler(void) {
    if (EXTI->PR & (1 << PIN)) {
        // Handle interrupt
        button_pressed_flag = 1;
        
        // Clear pending bit
        EXTI->PR = (1 << PIN);
    }
}

// Example: Timer interrupt
// Configure timer for 1ms interrupt
TIM2->PSC = 7999;        // Prescaler: 8MHz / 8000 = 1kHz
TIM2->ARR = 999;         // Auto-reload: 1000 counts = 1ms
TIM2->DIER |= TIM_DIER_UIE; // Enable update interrupt
NVIC_EnableIRQ(TIM2_IRQn);

// Timer ISR
void TIM2_IRQHandler(void) {
    if (TIM2->SR & TIM_SR_UIF) {
        // 1ms tick
        system_tick++;
        
        // Update tasks
        update_tasks();
        
        // Clear interrupt flag
        TIM2->SR &= ~TIM_SR_UIF;
    }
}

// Example: Critical section
void critical_function(void) {
    __disable_irq();  // Disable interrupts
    
    // Critical code
    shared_variable++;
    // Must be atomic
    
    __enable_irq();   // Re-enable interrupts
}

// Example: Deferred processing (better approach)
volatile uint8_t adc_ready_flag = 0;

void ADC1_IRQHandler(void) {
    // Short ISR
    adc_ready_flag = 1;
    // Clear interrupt...
}

void main_loop(void) {
    while (1) {
        if (adc_ready_flag) {
            adc_ready_flag = 0;
            // Process ADC data (can take time)
            process_adc_data();
        }
        // Other tasks...
    }
}`,
				},
				{
					Title: "Power Management",
					Content: `Power management is critical for battery-powered embedded systems. Understanding power modes and optimization techniques is essential.

**Power Consumption Sources:**

**Dynamic Power:**
- P = C × V² × f
- Proportional to switching frequency
- Dominant at high frequencies
- Reduced by: Lower voltage, lower frequency, clock gating

**Static Power:**
- Leakage current
- Proportional to voltage
- Dominant at low frequencies
- Reduced by: Power gating, lower voltage

**Power Modes:**

**Active Mode:**
- Full operation
- Highest power consumption
- All peripherals active

**Sleep Mode:**
- CPU halted
- Peripherals can remain active
- Wake on interrupt
- Moderate power savings

**Deep Sleep Mode:**
- CPU and most peripherals off
- Only wake-up sources active
- Significant power savings
- Longer wake-up time

**Shutdown Mode:**
- Minimal functionality
- Only RTC or wake-up pins active
- Lowest power consumption
- Very long wake-up time

**Power Optimization Techniques:**

**Clock Gating:**
- Disable clocks to unused peripherals
- Reduces dynamic power

**Power Gating:**
- Turn off power to unused blocks
- Reduces static power

**Voltage Scaling:**
- Lower voltage for lower performance
- Quadratic power reduction

**Duty Cycling:**
- Periodic wake-up and sleep
- Average power reduction`,
					CodeExamples: `// Example: Enter sleep mode (ARM Cortex-M)
void enter_sleep_mode(void) {
    // Configure wake-up source
    EXTI->IMR |= (1 << WAKE_PIN);
    
    // Enter sleep mode
    SCB->SCR |= SCB_SCR_SLEEPONEXIT_Msk;  // Sleep on exit from ISR
    // Or
    __WFI();  // Wait for interrupt
    
    // CPU enters sleep, wakes on interrupt
}

// Example: Enter deep sleep
void enter_deep_sleep(void) {
    // Configure wake-up source
    PWR->CR |= PWR_CR_LPDS;  // Low-power deep sleep
    
    // Disable unnecessary peripherals
    RCC->AHBENR &= ~(RCC_AHBENR_GPIOAEN | ...);
    
    // Enter deep sleep
    SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
    __WFI();
}

// Example: Clock gating
void disable_unused_peripherals(void) {
    // Disable unused peripheral clocks
    RCC->APB1ENR &= ~RCC_APB1ENR_TIM2EN;  // Disable Timer 2
    RCC->APB2ENR &= ~RCC_APB2ENR_USART1EN; // Disable UART 1
    // Saves power
}

// Example: Duty cycling
void duty_cycle_task(void) {
    while (1) {
        // Active period
        read_sensors();
        process_data();
        transmit_data();
        
        // Sleep period
        enter_sleep_mode();
        // Wake after 1 second (timer interrupt)
    }
}

// Example: Voltage scaling
void set_low_power_mode(void) {
    // Reduce clock frequency
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_HPRE) | RCC_CFGR_HPRE_DIV8;
    // 8MHz / 8 = 1MHz
    
    // Reduce voltage (if supported)
    PWR->CR |= PWR_CR_VOS;  // Voltage scaling
    
    // Power scales with V² × f
    // 1/8 frequency × 1/2 voltage = 1/32 power`,
				},
				{
					Title: "Peripheral Interfaces",
					Content: `Embedded systems communicate with external devices through various peripheral interfaces. Understanding these protocols is essential.

**UART (Universal Asynchronous Receiver/Transmitter):**
- Asynchronous serial communication
- Two wires: TX (transmit), RX (receive)
- Common baud rates: 9600, 115200
- Simple, widely used
- No clock signal needed

**SPI (Serial Peripheral Interface):**
- Synchronous serial communication
- Four wires: MOSI, MISO, SCK, CS
- Master-slave architecture
- High speed (MHz range)
- Full duplex
- Used for sensors, displays, flash memory

**I2C (Inter-Integrated Circuit):**
- Synchronous serial communication
- Two wires: SDA (data), SCL (clock)
- Multi-master, multi-slave
- Addressing scheme (7-bit or 10-bit)
- Lower speed than SPI
- Used for sensors, EEPROMs, RTCs

**CAN (Controller Area Network):**
- Multi-master bus protocol
- Differential signaling (noise resistant)
- Message-based protocol
- Used in automotive, industrial applications
- Error detection and correction

**USB:**
- Universal Serial Bus
- Host-device architecture
- Multiple speed modes (Low, Full, High, SuperSpeed)
- Power delivery
- Complex protocol`,
					CodeExamples: `// Example: UART configuration (ARM Cortex-M)
void uart_init(uint32_t baudrate) {
    // Enable UART clock
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    
    // Configure GPIO for UART
    GPIOA->AFR[1] |= (7 << 4);  // PA2: AF7 (USART2_TX)
    GPIOA->AFR[1] |= (7 << 8);  // PA3: AF7 (USART2_RX)
    GPIOA->MODER |= (2 << 4) | (2 << 6);  // Alternate function
    
    // Configure UART
    USART2->BRR = 8000000 / baudrate;  // Baud rate
    USART2->CR1 |= USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

// UART transmit
void uart_send(char c) {
    while (!(USART2->ISR & USART_ISR_TXE));
    USART2->TDR = c;
}

// UART receive
char uart_receive(void) {
    while (!(USART2->ISR & USART_ISR_RXNE));
    return USART2->RDR;
}

// Example: SPI configuration
void spi_init(void) {
    // Enable SPI clock
    RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
    
    // Configure SPI
    SPI1->CR1 |= SPI_CR1_MSTR;        // Master mode
    SPI1->CR1 |= SPI_CR1_SSM | SPI_CR1_SSI; // Software NSS
    SPI1->CR1 |= SPI_CR1_BR_0;        // Baud rate prescaler
    SPI1->CR1 |= SPI_CR1_SPE;         // Enable SPI
}

// SPI transfer
uint8_t spi_transfer(uint8_t data) {
    SPI1->DR = data;
    while (!(SPI1->SR & SPI_SR_RXNE));
    return SPI1->DR;
}

// Example: I2C configuration
void i2c_init(void) {
    // Enable I2C clock
    RCC->APB1ENR |= RCC_APB1ENR_I2C1EN;
    
    // Configure I2C
    I2C1->CR2 |= 8;  // Peripheral clock: 8MHz
    I2C1->CCR = 40;  // Clock control: 100kHz
    I2C1->TRISE = 9; // Rise time
    I2C1->CR1 |= I2C_CR1_PE;  // Enable I2C
}

// I2C write
void i2c_write(uint8_t addr, uint8_t reg, uint8_t data) {
    // Start condition
    I2C1->CR1 |= I2C_CR1_START;
    while (!(I2C1->SR1 & I2C_SR1_SB));
    
    // Send address
    I2C1->DR = (addr << 1) | 0;  // Write
    while (!(I2C1->SR1 & I2C_SR1_ADDR));
    I2C1->SR2;  // Clear ADDR flag
    
    // Send register address
    I2C1->DR = reg;
    while (!(I2C1->SR1 & I2C_SR1_TXE));
    
    // Send data
    I2C1->DR = data;
    while (!(I2C1->SR1 & I2C_SR1_TXE));
    
    // Stop condition
    I2C1->CR1 |= I2C_CR1_STOP;
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
