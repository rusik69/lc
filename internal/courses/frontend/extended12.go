package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1570,
			Title:       "Frontend Monitoring and Observability",
			Description: "Implement comprehensive frontend monitoring with error tracking, performance metrics, Real User Monitoring (RUM), logging, and debugging strategies.",
			Order:       70,
			Lessons: []problems.Lesson{
				{
					Title: "Error Tracking and Performance Monitoring",
					Content: `Frontend observability encompasses error tracking, performance monitoring, user behavior analytics, and debugging tools.

**Error Boundary and Tracking:**
` + "```" + `javascript
// Error Boundary with reporting
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Report to monitoring service
    this.reportError(error, errorInfo);
    this.setState({ errorInfo });
  }

  reportError(error, errorInfo) {
    const errorData = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo?.componentStack,
      url: window.location.href,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      // Add user context if available
      userId: this.props.userId,
      sessionId: this.props.sessionId,
    };

    // Send to error tracking service
    fetch('/api/errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorData),
      keepalive: true, // Ensure delivery even on page unload
    }).catch(() => {
      // Queue for retry
      queueErrorForRetry(errorData);
    });
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div role="alert">
          <h2>Something went wrong</h2>
          <button onClick={() => this.setState({ hasError: false })}>
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Global error handlers
window.addEventListener('error', (event) => {
  // Uncaught JavaScript errors
  reportError({
    type: 'uncaught_error',
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    stack: event.error?.stack,
  });
});

window.addEventListener('unhandledrejection', (event) => {
  // Unhandled promise rejections
  reportError({
    type: 'unhandled_rejection',
    message: event.reason?.message || String(event.reason),
    stack: event.reason?.stack,
  });
});

// Resource loading errors
window.addEventListener('error', (event) => {
  if (event.target !== window) {
    reportError({
      type: 'resource_error',
      tagName: event.target.tagName,
      src: event.target.src || event.target.href,
    });
  }
}, true); // Capture phase to catch resource errors

// Console error interception
const originalError = console.error;
console.error = function(...args) {
  originalError.apply(console, args);
  reportError({
    type: 'console_error',
    message: args.map(String).join(' '),
  });
};
` + "```" + `

**Web Vitals and Performance Monitoring:**
` + "```" + `javascript
// Web Vitals monitoring
import { onCLS, onFID, onFCP, onINP, onLCP, onTTFB } from 'web-vitals';

function sendMetric(metric) {
  const data = {
    name: metric.name,
    value: metric.value,
    delta: metric.delta,
    id: metric.id,
    rating: metric.rating, // 'good', 'needs-improvement', 'poor'
    navigationType: metric.navigationType,
    url: window.location.href,
    timestamp: Date.now(),
  };

  // Use Beacon API for reliable delivery
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/metrics', JSON.stringify(data));
  } else {
    fetch('/api/metrics', {
      method: 'POST',
      body: JSON.stringify(data),
      keepalive: true,
    });
  }
}

// Register all Web Vital metrics
onCLS(sendMetric);   // Cumulative Layout Shift
onFID(sendMetric);   // First Input Delay (deprecated, use INP)
onFCP(sendMetric);   // First Contentful Paint
onINP(sendMetric);   // Interaction to Next Paint
onLCP(sendMetric);   // Largest Contentful Paint
onTTFB(sendMetric);  // Time to First Byte

// Web Vitals thresholds:
// LCP:  Good < 2.5s,  Poor > 4.0s
// INP:  Good < 200ms, Poor > 500ms
// CLS:  Good < 0.1,   Poor > 0.25
// FCP:  Good < 1.8s,  Poor > 3.0s
// TTFB: Good < 800ms, Poor > 1800ms

// Custom performance marks and measures
function measureOperation(name, fn) {
  performance.mark(name + '-start');
  const result = fn();
  
  if (result instanceof Promise) {
    return result.finally(() => {
      performance.mark(name + '-end');
      performance.measure(name, name + '-start', name + '-end');
      
      const measure = performance.getEntriesByName(name)[0];
      sendMetric({
        name: 'custom.' + name,
        value: measure.duration,
        rating: measure.duration < 100 ? 'good' : 
                measure.duration < 500 ? 'needs-improvement' : 'poor',
      });
    });
  }
  
  performance.mark(name + '-end');
  performance.measure(name, name + '-start', name + '-end');
  return result;
}

// Usage
await measureOperation('fetchUserData', () => fetch('/api/user'));

// Performance Observer for long tasks
const longTaskObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    sendMetric({
      name: 'long-task',
      value: entry.duration,
      rating: entry.duration > 100 ? 'poor' : 'needs-improvement',
    });
  }
});

longTaskObserver.observe({ type: 'longtask', buffered: true });

// Layout shift observer
const layoutShiftObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.log('Layout shift:', {
        value: entry.value,
        sources: entry.sources?.map(s => ({
          node: s.node?.nodeName,
          previousRect: s.previousRect,
          currentRect: s.currentRect,
        })),
      });
    }
  }
});

layoutShiftObserver.observe({ type: 'layout-shift', buffered: true });

// Navigation timing
window.addEventListener('load', () => {
  setTimeout(() => {
    const timing = performance.getEntriesByType('navigation')[0];
    sendMetric({
      name: 'page-load',
      value: timing.loadEventEnd - timing.fetchStart,
      details: {
        dns: timing.domainLookupEnd - timing.domainLookupStart,
        tcp: timing.connectEnd - timing.connectStart,
        ttfb: timing.responseStart - timing.fetchStart,
        download: timing.responseEnd - timing.responseStart,
        domParse: timing.domInteractive - timing.responseEnd,
        domReady: timing.domContentLoadedEventEnd - timing.fetchStart,
        transfer: timing.transferSize,
        encoded: timing.encodedBodySize,
        decoded: timing.decodedBodySize,
      },
    });
  }, 0);
});

// Resource timing
const resourceObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.duration > 1000) { // Slow resources
      sendMetric({
        name: 'slow-resource',
        value: entry.duration,
        details: {
          name: entry.name,
          type: entry.initiatorType,
          size: entry.transferSize,
        },
      });
    }
  }
});

resourceObserver.observe({ type: 'resource', buffered: true });
` + "```" + ``,
					CodeExamples: `// Real User Monitoring (RUM) and session management

// 1. Session replay data collection
class SessionRecorder {
  constructor(options = {}) {
    this.events = [];
    this.sessionId = crypto.randomUUID();
    this.maxEvents = options.maxEvents || 10000;
    this.flushInterval = options.flushInterval || 10000;
    this.url = options.url || '/api/session-events';
    
    this.startRecording();
    this.flushTimer = setInterval(() => this.flush(), this.flushInterval);
  }

  startRecording() {
    // DOM mutations
    this.mutationObserver = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        this.addEvent('dom_mutation', {
          type: mutation.type,
          target: this.getSelector(mutation.target),
          addedNodes: mutation.addedNodes.length,
          removedNodes: mutation.removedNodes.length,
        });
      }
    });
    this.mutationObserver.observe(document.body, {
      childList: true, subtree: true, attributes: true,
      characterData: true, attributeOldValue: true,
    });

    // User interactions
    document.addEventListener('click', (e) => {
      this.addEvent('click', {
        target: this.getSelector(e.target),
        x: e.clientX,
        y: e.clientY,
      });
    }, { passive: true });

    document.addEventListener('input', (e) => {
      this.addEvent('input', {
        target: this.getSelector(e.target),
        // Don't record sensitive field values
        hasValue: !!e.target.value,
        type: e.target.type,
      });
    }, { passive: true });

    // Scroll tracking
    let scrollTimeout;
    document.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        this.addEvent('scroll', {
          scrollY: window.scrollY,
          scrollX: window.scrollX,
          scrollHeight: document.documentElement.scrollHeight,
        });
      }, 150);
    }, { passive: true });

    // Page visibility
    document.addEventListener('visibilitychange', () => {
      this.addEvent('visibility', {
        state: document.visibilityState,
      });
    });

    // Navigation events
    window.addEventListener('popstate', () => {
      this.addEvent('navigation', { url: window.location.href });
    });
  }

  addEvent(type, data) {
    if (this.events.length >= this.maxEvents) {
      this.flush();
    }
    this.events.push({
      type,
      data,
      timestamp: Date.now(),
    });
  }

  getSelector(element) {
    if (!element || element === document.body) return 'body';
    const parts = [];
    while (element && element !== document.body) {
      let selector = element.tagName.toLowerCase();
      if (element.id) {
        selector += '#' + element.id;
        parts.unshift(selector);
        break;
      }
      if (element.className && typeof element.className === 'string') {
        selector += '.' + element.className.trim().split(/\s+/).slice(0, 2).join('.');
      }
      parts.unshift(selector);
      element = element.parentElement;
    }
    return parts.join(' > ');
  }

  async flush() {
    if (this.events.length === 0) return;
    const batch = this.events.splice(0);
    
    try {
      await fetch(this.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: this.sessionId,
          events: batch,
        }),
        keepalive: true,
      });
    } catch {
      // Re-queue failed events (limit to prevent memory leak)
      this.events = batch.slice(-100).concat(this.events);
    }
  }

  stop() {
    this.mutationObserver?.disconnect();
    clearInterval(this.flushTimer);
    this.flush();
  }
}

// 2. Structured logging for frontend
const LogLevel = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };

class Logger {
  constructor(options = {}) {
    this.level = options.level || LogLevel.INFO;
    this.context = options.context || {};
    this.handlers = options.handlers || [consoleHandler, remoteHandler];
  }

  debug(message, data) { this.log(LogLevel.DEBUG, message, data); }
  info(message, data) { this.log(LogLevel.INFO, message, data); }
  warn(message, data) { this.log(LogLevel.WARN, message, data); }
  error(message, data) { this.log(LogLevel.ERROR, message, data); }

  log(level, message, data = {}) {
    if (level < this.level) return;
    
    const entry = {
      level: ['DEBUG', 'INFO', 'WARN', 'ERROR'][level],
      message,
      data,
      context: this.context,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      sessionId: getSessionId(),
    };

    this.handlers.forEach(handler => handler(entry));
  }

  child(context) {
    return new Logger({
      ...this,
      context: { ...this.context, ...context },
    });
  }
}

function consoleHandler(entry) {
  const fn = entry.level === 'ERROR' ? console.error :
             entry.level === 'WARN' ? console.warn :
             entry.level === 'DEBUG' ? console.debug :
             console.log;
  fn('[' + entry.level + ']', entry.message, entry.data);
}

const buffer = [];
function remoteHandler(entry) {
  buffer.push(entry);
  if (buffer.length >= 10 || entry.level === 'ERROR') {
    const batch = buffer.splice(0);
    navigator.sendBeacon?.('/api/logs', JSON.stringify(batch));
  }
}

// Usage
const logger = new Logger({ level: LogLevel.DEBUG });
const apiLogger = logger.child({ component: 'api-client' });
apiLogger.info('Request started', { method: 'GET', url: '/api/users' });
apiLogger.error('Request failed', { status: 500, message: 'Server error' });`,
				},
				{
					Title: "Debugging and DevTools Techniques",
					Content: `Advanced debugging strategies and browser DevTools features for efficient frontend development.

**Advanced Console Methods:**
` + "```" + `javascript
// Styled console output
console.log(
  '%cWarning!%c Something is slow',
  'color: red; font-size: 20px; font-weight: bold;',
  'color: orange; font-size: 14px;'
);

// Console groups
console.group('User Authentication');
console.log('Checking credentials...');
console.log('Token valid:', true);
console.groupEnd();

// Collapsed group
console.groupCollapsed('API Responses (5 items)');
responses.forEach(r => console.log(r));
console.groupEnd();

// Console table for structured data
console.table([
  { name: 'React', version: '18.2', size: '44kb' },
  { name: 'Vue', version: '3.4', size: '33kb' },
  { name: 'Svelte', version: '4.2', size: '2kb' },
]);

// Timing
console.time('render');
renderComponent();
console.timeEnd('render'); // render: 23.4ms

// Counting
function processItem(item) {
  console.count('processItem called'); // processItem called: 1, 2, 3...
}

// Assert (only logs when condition is false)
console.assert(items.length > 0, 'Items array should not be empty');

// Trace (show call stack)
function deepFunction() {
  console.trace('How did we get here?');
}

// Profile (CPU profiling)
console.profile('Heavy Operation');
heavyOperation();
console.profileEnd('Heavy Operation');

// Memory info
console.log(performance.memory); // Chrome only
// { usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit }
` + "```" + `

**Debugging Strategies:**
` + "```" + `javascript
// 1. Conditional breakpoints
// In DevTools: Right-click line number -> "Add conditional breakpoint"
// Only breaks when condition is true:
// e.g., user.id === 'abc123'

// 2. Logpoints (non-breaking breakpoints)
// Right-click line number -> "Add logpoint"
// Logs to console without stopping execution

// 3. DOM breakpoints
// Elements panel -> Right-click element -> "Break on"
// - Subtree modifications
// - Attribute modifications
// - Node removal

// 4. XHR/Fetch breakpoints
// Sources panel -> XHR/fetch Breakpoints
// Break when URL contains specific string

// 5. Event listener breakpoints
// Sources panel -> Event Listener Breakpoints
// Break on specific DOM events

// 6. Debug utility functions
function debugProxy(obj, name = 'Object') {
  return new Proxy(obj, {
    get(target, prop) {
      console.log(name + '.get(' + String(prop) + '):', target[prop]);
      return target[prop];
    },
    set(target, prop, value) {
      console.log(name + '.set(' + String(prop) + '):', value);
      target[prop] = value;
      return true;
    },
  });
}

// Watch object changes
const state = debugProxy({ count: 0, name: 'test' }, 'state');
state.count = 1; // Logs: state.set(count): 1

// 7. Performance debugging
function findPerformanceBottleneck() {
  // Check for unnecessary re-renders (React)
  // React DevTools Profiler -> Record -> Interact -> Stop
  // Look for components that render without prop changes
  
  // Check for layout thrashing
  // Performance panel -> Record -> Look for forced reflows
  
  // Memory leak detection
  // Memory panel -> Take heap snapshot
  // Perform action -> Take another snapshot
  // Compare snapshots -> Look for growing allocations
}

// 8. Network debugging
// Network panel features:
// - Throttle connection speed
// - Block specific requests
// - Replay requests with modifications
// - Copy as cURL/fetch
// - Filter by type, status, domain

// 9. Source map configuration
// vite.config.js
export default defineConfig({
  build: {
    sourcemap: true, // Generate source maps for production debugging
  },
});

// Conditional source maps (production)
// Upload source maps to error tracking service
// but don't serve them publicly
// sourcemap: 'hidden' in Vite

// 10. Debug React component
function useDebugValue(label, value) {
  React.useDebugValue(label + ': ' + JSON.stringify(value));
}

function useDebugRender(componentName) {
  const renderCount = useRef(0);
  renderCount.current++;
  
  useEffect(() => {
    console.log(componentName + ' rendered ' + renderCount.current + ' times');
  });
}

// React DevTools component highlights
// Settings -> General -> Highlight updates
// Shows which components re-render on state changes

// why-did-you-render library
// Patches React.createElement to log unnecessary re-renders
// import whyDidYouRender from '@welldone-software/why-did-you-render';
// whyDidYouRender(React, { trackAllPureComponents: true });

// 11. Network request intercepting for debugging
const originalFetch = window.fetch;
window.fetch = async function(...args) {
  const start = performance.now();
  const url = typeof args[0] === 'string' ? args[0] : args[0].url;
  
  try {
    const response = await originalFetch.apply(this, args);
    const duration = performance.now() - start;
    
    if (duration > 1000) {
      console.warn('Slow request:', url, duration.toFixed(0) + 'ms');
    }
    
    if (!response.ok) {
      console.error('Failed request:', response.status, url);
    }
    
    return response;
  } catch (error) {
    console.error('Network error:', url, error.message);
    throw error;
  }
};
` + "```" + `

**React DevTools and Testing Utilities:**
` + "```" + `javascript
// React Profiler API
function ProfiledApp() {
  return (
    <React.Profiler id="App" onRender={onRender}>
      <App />
    </React.Profiler>
  );
}

function onRender(
  id,           // Component tree id
  phase,        // "mount" or "update"
  actualDuration, // Time spent rendering
  baseDuration,   // Estimated time without memoization
  startTime,      // When React began rendering
  commitTime,     // When React committed
  interactions    // Set of interactions that triggered this
) {
  if (actualDuration > 16) { // Longer than one frame
    console.warn(
      'Slow render: ' + id + ' (' + phase + ')',
      actualDuration.toFixed(2) + 'ms'
    );
  }
}

// Custom React DevTools hook for debugging
function useRenderTracker(name, props) {
  const prevProps = useRef(props);
  
  useEffect(() => {
    const changes = {};
    for (const key of Object.keys(props)) {
      if (prevProps.current[key] !== props[key]) {
        changes[key] = {
          prev: prevProps.current[key],
          next: props[key],
        };
      }
    }
    
    if (Object.keys(changes).length > 0) {
      console.log(name + ' re-rendered due to:', changes);
    }
    
    prevProps.current = props;
  });
}

// Usage in component
function ExpensiveComponent(props) {
  useRenderTracker('ExpensiveComponent', props);
  // ... component logic
}

// Memory leak prevention checklist:
// 1. Clean up event listeners in useEffect cleanup
// 2. Cancel fetch requests with AbortController
// 3. Clear timers (setTimeout, setInterval)
// 4. Unsubscribe from observables
// 5. Remove DOM references
// 6. Close WebSocket connections
// 7. Revoke object URLs (URL.revokeObjectURL)

function useCleanupExample() {
  useEffect(() => {
    const controller = new AbortController();
    const timer = setInterval(poll, 5000);
    const ws = new WebSocket('wss://api.example.com/ws');
    
    document.addEventListener('keydown', handler);
    
    return () => {
      controller.abort();
      clearInterval(timer);
      ws.close();
      document.removeEventListener('keydown', handler);
    };
  }, []);
}
` + "```" + ``,
					CodeExamples: `// Feature flags and A/B testing

// 1. Feature flag client
class FeatureFlags {
  constructor(config = {}) {
    this.flags = {};
    this.overrides = {};
    this.userId = config.userId;
    this.environment = config.environment || 'production';
    
    this.loadOverrides();
  }

  async initialize() {
    try {
      const response = await fetch('/api/feature-flags', {
        headers: { 'X-User-Id': this.userId },
      });
      this.flags = await response.json();
    } catch {
      // Use cached flags
      const cached = localStorage.getItem('feature-flags');
      if (cached) this.flags = JSON.parse(cached);
    }
    
    // Cache for offline use
    localStorage.setItem('feature-flags', JSON.stringify(this.flags));
  }

  isEnabled(flagName) {
    // DevTools overrides take precedence
    if (flagName in this.overrides) return this.overrides[flagName];
    
    const flag = this.flags[flagName];
    if (!flag) return false;
    
    // Environment check
    if (flag.environments && !flag.environments.includes(this.environment)) {
      return false;
    }
    
    // Percentage rollout
    if (flag.percentage !== undefined) {
      const hash = this.hashString(this.userId + flagName);
      return (hash % 100) < flag.percentage;
    }
    
    // User targeting
    if (flag.users && flag.users.includes(this.userId)) {
      return true;
    }
    
    return flag.enabled || false;
  }

  getVariant(experimentName) {
    const experiment = this.flags[experimentName];
    if (!experiment || !experiment.variants) return 'control';
    
    const hash = this.hashString(this.userId + experimentName);
    let cumulative = 0;
    
    for (const variant of experiment.variants) {
      cumulative += variant.weight;
      if ((hash % 100) < cumulative) return variant.name;
    }
    
    return 'control';
  }

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  loadOverrides() {
    const saved = localStorage.getItem('feature-flag-overrides');
    if (saved) this.overrides = JSON.parse(saved);
  }

  setOverride(flagName, value) {
    this.overrides[flagName] = value;
    localStorage.setItem('feature-flag-overrides', JSON.stringify(this.overrides));
  }

  clearOverrides() {
    this.overrides = {};
    localStorage.removeItem('feature-flag-overrides');
  }
}

// React hook for feature flags
const FeatureFlagContext = createContext(null);

function useFeatureFlag(flagName) {
  const flags = useContext(FeatureFlagContext);
  return flags.isEnabled(flagName);
}

function useExperiment(experimentName) {
  const flags = useContext(FeatureFlagContext);
  const variant = flags.getVariant(experimentName);
  
  useEffect(() => {
    // Track experiment exposure
    analytics.track('experiment_exposure', {
      experiment: experimentName,
      variant,
    });
  }, [experimentName, variant]);
  
  return variant;
}

// Usage
function CheckoutPage() {
  const showNewCheckout = useFeatureFlag('new-checkout-flow');
  const pricingVariant = useExperiment('pricing-display');
  
  return showNewCheckout ? <NewCheckout pricing={pricingVariant} /> : <OldCheckout />;
}

// 2. Analytics tracking
class Analytics {
  constructor({ apiKey, endpoint }) {
    this.apiKey = apiKey;
    this.endpoint = endpoint;
    this.queue = [];
    this.flushInterval = setInterval(() => this.flush(), 5000);
  }

  track(event, properties = {}) {
    this.queue.push({
      type: 'track',
      event,
      properties,
      timestamp: Date.now(),
      context: {
        page: { url: location.href, title: document.title },
        screen: { width: screen.width, height: screen.height },
        locale: navigator.language,
      },
    });

    if (this.queue.length >= 20) this.flush();
  }

  page(name, properties = {}) {
    this.track('page_view', { page_name: name, ...properties });
  }

  identify(userId, traits = {}) {
    this.queue.push({
      type: 'identify',
      userId,
      traits,
      timestamp: Date.now(),
    });
  }

  async flush() {
    if (this.queue.length === 0) return;
    const batch = this.queue.splice(0);
    
    try {
      navigator.sendBeacon?.(this.endpoint, JSON.stringify({
        apiKey: this.apiKey,
        batch,
      }));
    } catch {
      this.queue = batch.concat(this.queue);
    }
  }

  destroy() {
    clearInterval(this.flushInterval);
    this.flush();
  }
}

const analytics = new Analytics({
  apiKey: 'ak_prod_xxx',
  endpoint: '/api/analytics',
});

// React hook
function usePageTracking() {
  const location = useLocation();
  useEffect(() => {
    analytics.page(location.pathname);
  }, [location.pathname]);
}`,
				},
			},
		},
	})
}
