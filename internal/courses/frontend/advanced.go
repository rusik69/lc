package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          220,
			Title:       "React Advanced Patterns",
			Description: "Advanced React patterns: render props, HOCs, compound components, and performance optimization.",
			Order:       20,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced React Patterns",
					Content: `Advanced patterns solve complex problems in React applications by providing reusable, composable abstractions that go beyond basic component composition. As your React codebase grows, you will inevitably encounter scenarios where simple props and state are not enough to keep your code DRY and maintainable. These patterns have emerged from the community over years of building large-scale applications, and understanding them is essential for any senior React developer.

**1. Render Props**

The render props pattern involves passing a function as a prop to a component, which that component then calls to determine what to render. Think of it like a contract: the parent component says "I will tell you how to display your data," while the child component says "I will provide the data and behavior." This pattern is incredibly powerful for sharing cross-cutting logic — such as mouse tracking, data fetching, or form handling — between components that need to render that logic differently. The child component manages the state and behavior, but delegates the rendering decision entirely to the parent via the function prop. Before hooks, this was one of the primary ways to share stateful logic across components. While hooks have largely replaced render props for simple cases, the pattern remains valuable when you need to share behavior that involves rendering decisions.

**2. Higher-Order Components (HOCs)**

A Higher-Order Component is a function that takes a component as an argument and returns a new, enhanced component. It is essentially the decorator pattern applied to React components. For example, a withAuth HOC might wrap any component with authentication logic, redirecting unauthenticated users to a login page. HOCs are powerful for cross-cutting concerns like authentication, logging, theming, and data fetching. However, they come with trade-offs: they can lead to "wrapper hell" (deeply nested component trees), make it harder to trace where props come from, and create naming collisions. With the introduction of hooks, many use cases that previously required HOCs can now be handled more cleanly with custom hooks, but HOCs remain relevant in codebases that rely on class components or need to wrap third-party components.

**3. Compound Components**

Compound components are a set of components that work together to form a cohesive unit, sharing implicit state through React context. Think of the HTML select and option elements — they are meaningless on their own but work beautifully together. In React, you might build a Tabs component that provides shared state (like the active tab index) to its child Tab and TabPanel components via context. The consumer of the API gets a clean, declarative interface while the internal state management is completely hidden. This pattern gives users maximum flexibility in how they arrange and customize the sub-components while ensuring consistent behavior.

**4. Performance Optimization**

Performance optimization in React revolves around minimizing unnecessary re-renders and reducing the amount of work the browser must do on each frame. React.memo is a higher-order component that prevents a functional component from re-rendering when its props have not changed — think of it as shouldComponentUpdate for function components. useMemo allows you to memoize the result of an expensive calculation so it is only recomputed when its dependencies change, which is critical for heavy data transformations or complex derived state. useCallback memoizes function references so that child components receiving those functions as props do not re-render unnecessarily. Finally, code splitting with React.lazy and dynamic imports allows you to break your bundle into smaller chunks that load on demand, dramatically improving initial page load times for large applications.`,
					CodeExamples: `// Render props
function Mouse({ render }) {
    const [position, setPosition] = useState({ x: 0, y: 0 });
    
    useEffect(() => {
        const handleMove = (e) => {
            setPosition({ x: e.clientX, y: e.clientY });
        };
        window.addEventListener('mousemove', handleMove);
        return () => window.removeEventListener('mousemove', handleMove);
    }, []);
    
    return render(position);
}

<Mouse render={({ x, y }) => <p>Mouse at {x}, {y}</p>} />

// Higher-order component
function withAuth(Component) {
    return function AuthenticatedComponent(props) {
        const { user } = useAuth();
        if (!user) return <Login />;
        return <Component {...props} user={user} />;
    };
}

const ProtectedPage = withAuth(Page);

// Compound components
function Tabs({ children }) {
    const [activeIndex, setActiveIndex] = useState(0);
    return (
        <TabsContext.Provider value={{ activeIndex, setActiveIndex }}>
            {children}
        </TabsContext.Provider>
    );
}

// Performance optimization
const ExpensiveComponent = React.memo(function ExpensiveComponent({ data }) {
    const processed = useMemo(() => {
        return expensiveCalculation(data);
    }, [data]);
    
    const handleClick = useCallback(() => {
        doSomething();
    }, []);
    
    return <div>{processed}</div>;
});

// Code splitting
const LazyComponent = React.lazy(() => import('./LazyComponent'));`,
				},
				{
					Title: "React Suspense and Concurrent Features",
					Content: `React Suspense and concurrent features represent a fundamental shift in how React handles asynchronous operations and rendering priorities. Rather than forcing developers to manually manage loading states with boolean flags and conditional rendering, these features provide a declarative, framework-level approach to handling the inherently asynchronous nature of modern web applications.

**1. Suspense**

Suspense is React's built-in mechanism for declaratively specifying loading states. When a component inside a Suspense boundary is waiting for something — whether it is a lazily-loaded code chunk, data from an API, or any other asynchronous resource — React automatically shows the fallback UI you specified. This is a profound improvement over the traditional pattern of tracking isLoading booleans in every component. Suspense works seamlessly with React.lazy for code splitting, and with compatible data-fetching libraries like Relay, React Query, and SWR. You can nest multiple Suspense boundaries to create granular loading experiences: for instance, a page-level Suspense for the route's code bundle and component-level Suspense boundaries for individual data-fetching sections. When combined with Error Boundaries, you get a complete declarative error and loading handling strategy that keeps your component code focused on the happy path.

**2. Concurrent Features**

Concurrent rendering is React's ability to prepare multiple versions of the UI simultaneously and choose when to commit them to the screen. This means React can start rendering an update, pause it if something more urgent comes in (like a user typing), and resume it later — making your app feel responsive even during heavy computations. The startTransition function lets you mark certain state updates as non-urgent transitions, telling React it is okay to interrupt them. For example, when a user types in a search box, the keystroke update is urgent (the input must reflect immediately), but filtering a large list of results can be deferred. useDeferredValue provides a similar capability by giving you a "lagged" version of a value that React can update at lower priority, perfect for expensive derived computations. useTransition gives you both the startTransition function and an isPending boolean, so you can show subtle loading indicators while the transition is in progress.

**3. Benefits and Real-World Impact**

The primary benefit of these features is better perceived performance. Users experience a snappier interface because urgent interactions like typing, clicking, and scrolling are never blocked by expensive rendering work. The interruptible rendering model means React can abandon stale work when new input arrives, avoiding the common problem where the UI freezes during heavy updates. Priority-based updates ensure that what matters most to the user is always processed first, leading to smoother interactions across the entire application.

**4. Practical Use Cases**

These features shine in several common scenarios. Code splitting benefits enormously from Suspense, which provides elegant loading states without any manual state management. Data fetching with compatible libraries becomes declarative and composable. Heavy computations — such as filtering large datasets, rendering complex charts, or processing images — can be wrapped in transitions so they never block user input. Route transitions become silky smooth because React can keep showing the current page while preparing the new one in the background, only switching once the new page is ready to display.`,
					CodeExamples: `// Suspense with lazy loading
import { Suspense, lazy } from 'react';

const LazyComponent = lazy(() => import('./LazyComponent'));

function App() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <LazyComponent />
        </Suspense>
    );
}

// startTransition
import { startTransition } from 'react';

function handleInput(e) {
    setInputValue(e.target.value); // Urgent update
    
    startTransition(() => {
        setSearchResults(search(e.target.value)); // Non-urgent
    });
}

// useTransition
import { useTransition } from 'react';

function TabContainer() {
    const [isPending, startTransition] = useTransition();
    const [tab, setTab] = useState('about');
    
    function selectTab(nextTab) {
        startTransition(() => {
            setTab(nextTab);
        });
    }
    
    return (
        <>
            {isPending && <Spinner />}
            <TabButton onClick={() => selectTab('about')}>About</TabButton>
        </>
    );
}

// useDeferredValue
import { useDeferredValue } from 'react';

function SearchResults({ query }) {
    const deferredQuery = useDeferredValue(query);
    const results = useMemo(() => search(deferredQuery), [deferredQuery]);
    
    return <div>{results}</div>;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          221,
			Title:       "Vue.js Advanced",
			Description: "Advanced Vue.js: composition API, reactivity system, plugins, and advanced patterns.",
			Order:       21,
			Lessons: []problems.Lesson{
				{
					Title: "Vue Composition API",
					Content: `The Composition API is Vue 3's answer to the challenge of organizing complex component logic. In the Options API (Vue 2's primary approach), related logic for a single feature — such as a search feature's reactive data, computed properties, watchers, and lifecycle hooks — gets scattered across different options objects (data, computed, watch, methods, mounted, etc.). As components grow, this fragmentation makes it increasingly difficult to understand and maintain the code. The Composition API solves this by letting you group related logic together using plain JavaScript functions and Vue's reactivity primitives.

**1. Why the Composition API Matters**

The Composition API provides four key advantages over the Options API. First, logic reuse becomes dramatically easier because you can extract related reactive state and behavior into standalone "composable" functions — Vue's equivalent of React's custom hooks. Second, TypeScript support is naturally excellent because everything is plain functions and variables with clear types, rather than the magic this context of the Options API. Third, you gain complete flexibility in how you organize your code within a component, grouping related concerns together instead of splitting them by option type. Fourth, testing becomes simpler because composables are just functions you can call in isolation.

**2. Core Reactivity Functions**

The ref function creates a reactive reference that wraps a single value. When the value changes, any component or computed property that reads it automatically updates. For primitive values like numbers and strings, ref is the go-to choice — you access and modify the value through the .value property. The reactive function creates a reactive proxy around an entire object, making all its properties reactive without needing .value access. The computed function creates a derived value that automatically recalculates when its dependencies change, just like computed properties in the Options API. The watch function lets you observe reactive sources and run side effects when they change, with access to both the new and old values. Finally, provide and inject implement dependency injection, allowing ancestor components to make values available to all their descendants without explicit prop drilling through every intermediate component.`,
					CodeExamples: `// Composition API
<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const count = ref(0)
const name = ref('John')

const doubled = computed(() => count.value * 2)

watch(count, (newVal, oldVal) => {
    console.log('Count changed from ' + oldVal + ' to ' + newVal)
})

function increment() {
    count.value++
}

onMounted(() => {
    console.log('Component mounted')
})
</script>

// Composables (custom hooks)
function useCounter(initialValue = 0) {
    const count = ref(initialValue)
    
    const increment = () => count.value++
    const decrement = () => count.value--
    const reset = () => count.value = initialValue
    
    return { count, increment, decrement, reset }
}

// Usage
const { count, increment } = useCounter(10)`,
				},
				{
					Title: "Vue Teleport and Advanced Patterns",
					Content: `Vue provides several advanced patterns and built-in features that solve common but tricky problems in component-based architectures. These tools become essential when building production-grade applications with complex UI requirements like modals, deeply nested component trees, and dynamic component loading.

**1. Teleport**

Teleport is Vue's built-in mechanism for rendering a component's content at a different location in the DOM tree, outside of the component's parent hierarchy. This solves a fundamental CSS and DOM problem: when you build a modal dialog inside a deeply nested component, the modal's positioning and z-index behavior can be affected by ancestor elements with overflow: hidden, transform, or their own stacking contexts. Teleport lets you say "render this content as a child of the body element (or any other target)" while keeping the component's reactive state, event handlers, and lifecycle hooks fully intact within the original component tree. This is conceptually similar to React's portals, and it is indispensable for building modals, tooltips, dropdown menus, toast notifications, and any overlay that needs to visually "break out" of its parent's layout constraints.

**2. Provide and Inject**

The provide/inject system is Vue's dependency injection mechanism, designed to solve the "prop drilling" problem. In deeply nested component trees, passing data through every intermediate component via props becomes tedious and error-prone. With provide, an ancestor component can make values available to all its descendants, no matter how deeply nested. Any descendant can then use inject to access those values directly, skipping all the components in between. This is particularly useful for theming (providing color schemes or design tokens), configuration (providing API base URLs or feature flags), and shared services (providing a store or event bus). When combined with reactive refs or reactive objects, provided values remain reactive throughout the tree — changes at the provider automatically update all injecting components.

**3. Advanced Component Patterns**

Vue supports several advanced patterns for specialized use cases. Render functions let you programmatically create VNodes using JavaScript instead of templates, giving you the full power of the language for highly dynamic rendering scenarios. Functional components are stateless, instance-less components that are lighter weight and render faster because they skip the component instance creation. Async components let you define components that are loaded on demand — perfect for code splitting and lazy loading routes or heavy UI sections. Dynamic components with the built-in component element and the :is prop let you switch between components at runtime, which is ideal for tab interfaces, form builders, and plugin architectures.

**4. Best Practices**

Use Teleport for any overlay UI (modals, tooltips, popovers) to avoid CSS stacking context issues. Rely on provide/inject for configuration and theming that needs to flow deep into the component tree, but prefer props for direct parent-child communication. Keep components focused on a single responsibility and extract reusable logic into composables. Monitor re-renders using Vue DevTools and apply v-memo or computed properties to optimize expensive rendering paths.`,
					CodeExamples: `// Teleport
<template>
    <button @click="showModal = true">Open Modal</button>
    <Teleport to="body">
        <Modal v-if="showModal" @close="showModal = false">
            <h2>Modal Content</h2>
        </Modal>
    </Teleport>
</template>

// provide/inject
// Parent component
<script setup>
import { provide } from 'vue';

const theme = {
    primary: '#3b82f6',
    secondary: '#8b5cf6'
};

provide('theme', theme);
</script>

// Child component
<script setup>
import { inject } from 'vue';

const theme = inject('theme');
</script>

<template>
    <div :style="{ color: theme.primary }">Themed content</div>
</template>

// Dynamic components
<component :is="currentComponent" />

// Async components
const AsyncComponent = defineAsyncComponent(() => import('./HeavyComponent.vue'));`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          222,
			Title:       "Performance Optimization",
			Description: "Optimize frontend performance: lazy loading, code splitting, caching, and rendering optimization.",
			Order:       22,
			Lessons: []problems.Lesson{
				{
					Title: "Performance Techniques",
					Content: `Performance optimization is one of the most impactful areas of frontend development because it directly affects user experience, conversion rates, and search engine rankings. Studies consistently show that even a one-second delay in page load time can reduce conversions by 7% and increase bounce rates significantly. Optimizing performance is not a one-time task but an ongoing discipline that touches every aspect of your application — from how code is bundled and delivered to how components render and respond to user input.

**1. Code Splitting and Lazy Loading**

Code splitting breaks your JavaScript bundle into smaller chunks that are loaded on demand rather than all at once. Instead of shipping a single massive bundle that includes every page, component, and library, you split it so that users only download the code they actually need for the page they are viewing. React.lazy and dynamic imports make this straightforward at the route level, but you can also split at the component level for heavy widgets like charts, editors, or maps. Lazy loading extends this concept to all resources: images can use the native loading="lazy" attribute to defer loading until they enter the viewport, and fonts can use font-display: swap to show text immediately with a fallback font.

**2. Memoization and Virtual Scrolling**

Memoization is the technique of caching the results of expensive computations so they are not repeated unnecessarily. In React, useMemo caches computed values and React.memo prevents re-renders of components whose props have not changed. Virtual scrolling (or windowing) is a critical optimization for long lists and tables. Instead of rendering thousands of DOM nodes for a large dataset, virtual scrolling libraries like react-window or react-virtuoso only render the items currently visible in the viewport, plus a small buffer. This can reduce the DOM node count from tens of thousands to just a few dozen, dramatically improving scroll performance and memory usage.

**3. Debouncing, Throttling, and Image Optimization**

Debouncing and throttling limit how frequently expensive functions execute in response to rapid events like scrolling, resizing, or typing. Debouncing waits until the user stops triggering the event for a specified delay, then fires once — perfect for search-as-you-type. Throttling ensures a function runs at most once per specified interval — ideal for scroll handlers. Image optimization is another huge win: converting images to modern formats like WebP or AVIF can reduce file sizes by 30-50%, responsive images with srcset serve appropriately sized images for each device, and lazy loading defers off-screen images. Bundle optimization through tree shaking (eliminating unused exports) and minification (reducing code size) further reduces what users must download.

**4. Core Web Vitals Metrics**

Google's Core Web Vitals are the standard metrics for measuring user experience. First Contentful Paint (FCP) measures when the first content appears on screen. Largest Contentful Paint (LCP) measures when the main content finishes loading — Google recommends under 2.5 seconds. Time to Interactive (TTI) measures when the page becomes fully interactive and responsive to user input. Cumulative Layout Shift (CLS) measures visual stability — how much the page layout shifts unexpectedly during loading. These metrics directly impact SEO rankings and should be monitored continuously using tools like Lighthouse, Chrome DevTools, and web-vitals library.`,
					CodeExamples: `// Code splitting
const LazyComponent = React.lazy(() => import('./Component'));

<Suspense fallback={<Loading />}>
    <LazyComponent />
</Suspense>

// Virtual scrolling
import { FixedSizeList } from 'react-window';

function VirtualList({ items }) {
    return (
        <FixedSizeList
            height={600}
            itemCount={items.length}
            itemSize={50}
        >
            {({ index, style }) => (
                <div style={style}>{items[index]}</div>
            )}
        </FixedSizeList>
    );
}

// Debouncing
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);
        
        return () => clearTimeout(handler);
    }, [value, delay]);
    
    return debouncedValue;
}

// Image optimization
<img 
    src="image.webp" 
    loading="lazy"
    srcset="image-small.webp 480w, image-large.webp 1200w"
    sizes="(max-width: 600px) 480px, 1200px"
    alt="Description"
/>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          223,
			Title:       "Progressive Web Apps (PWA)",
			Description: "Build Progressive Web Apps: service workers, offline support, and app-like experiences.",
			Order:       23,
			Lessons: []problems.Lesson{
				{
					Title: "PWA Fundamentals",
					Content: `Progressive Web Apps (PWAs) represent the convergence of web and native mobile application experiences. They use modern web technologies to deliver app-like capabilities — offline access, push notifications, home screen installation, and fast performance — while retaining the advantages of the web: no app store required, instant updates, linkability, and cross-platform compatibility. Companies like Twitter, Pinterest, and Starbucks have seen dramatic improvements in engagement and conversion rates after launching PWAs.

**1. Core PWA Technologies**

The three pillars of a PWA are Service Workers, the Web App Manifest, and HTTPS. Service Workers are JavaScript files that run in a separate thread from your main application, acting as a programmable network proxy between your app and the server. They can intercept every network request your app makes and decide how to handle it — serve from cache, fetch from network, or return a custom offline page. The Web App Manifest is a JSON file that tells the browser about your application's metadata: its name, icons, start URL, display mode (standalone, fullscreen, or minimal-ui), theme colors, and orientation. This metadata is what allows users to "install" your PWA on their home screen and have it launch like a native app. HTTPS is required because Service Workers have powerful capabilities (intercepting network requests), so browsers only allow them on secure origins.

**2. Service Worker Lifecycle**

Service Workers follow a specific lifecycle that is important to understand. During the install phase, the Service Worker downloads and caches the essential resources your app needs to work offline — HTML, CSS, JavaScript, images, and fonts. This is your app's "shell." During the activate phase, the Service Worker takes control of all pages within its scope and cleans up old caches from previous versions. Once active, the Service Worker intercepts fetch events for every network request, giving you full control over the caching strategy. The Service Worker continues running in the background even when the user closes the tab, enabling push notifications and background sync.

**3. Web App Manifest Configuration**

The manifest file controls how your PWA appears when installed. The name and short_name fields define the full and abbreviated app names. Icons should be provided in multiple sizes (at least 192x192 and 512x512 pixels) for different device contexts. The start_url defines which page loads when the user launches the installed PWA. The display field controls the UI chrome — "standalone" removes the browser address bar for a native-app feel, while "fullscreen" removes all browser UI. Theme colors and background colors ensure your app looks polished during the splash screen that appears while the PWA loads.`,
					CodeExamples: `// Service Worker registration
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(reg => console.log('SW registered'))
        .catch(err => console.log('SW registration failed'));
}

// Service Worker (sw.js)
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('v1').then((cache) => {
            return cache.addAll([
                '/',
                '/index.html',
                '/styles.css',
                '/app.js'
            ]);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});

// Manifest (manifest.json)
{
    "name": "My PWA",
    "short_name": "PWA",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": "#000000",
    "icons": [
        {
            "src": "/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        }
    ]
}

<!-- Link manifest -->
<link rel="manifest" href="/manifest.json">`,
				},
				{
					Title: "Background Sync and Push Notifications",
					Content: `Background sync and push notifications are advanced PWA capabilities that bridge the gap between web and native applications. Background sync ensures that user actions taken while offline are reliably completed when connectivity returns, while push notifications keep users engaged even when they are not actively using your application. Together, they create a resilient and engaging user experience that rivals native apps.

**1. Background Sync**

Background sync solves one of the most frustrating problems in web applications: what happens when a user performs an action (submitting a form, sending a message, saving data) while their network connection is unreliable or absent. Without background sync, the request simply fails and the user must remember to retry later. With background sync, your application queues the failed request in IndexedDB or another local storage mechanism, and the Service Worker automatically retries the request when the device regains connectivity. The browser fires a sync event on the Service Worker, which processes the queued requests in order. This happens even if the user has closed the tab, because the Service Worker runs independently. The result is a seamless experience where users can interact with your app confidently regardless of network conditions — think of it like how a native email app queues messages in your outbox until you are back online.

**2. Push Notifications**

Push notifications allow your server to send messages directly to a user's device, which are displayed by the Service Worker even when your application is not open. The flow works like this: first, your app requests notification permission from the user. If granted, the browser generates a push subscription containing an endpoint URL and encryption keys. Your app sends this subscription to your server, which can then send encrypted push messages to the browser's push service. The browser delivers these to your Service Worker, which handles the push event and displays a notification using the Notification API. When the user clicks the notification, your Service Worker receives a notificationclick event and can open or focus the appropriate page. This is incredibly powerful for re-engagement — order status updates, new message alerts, breaking news, and price drop notifications all benefit from push.

**3. Caching Strategies**

Service Workers support several caching strategies, each suited to different types of resources. Cache-first (also called "cache falling back to network") serves cached content immediately and only hits the network if the cache misses — perfect for static assets like images, fonts, and CSS that rarely change. Network-first tries the network and falls back to the cache on failure — ideal for API responses and dynamic content where freshness matters. Stale-while-revalidate serves the cached version immediately for speed, then fetches a fresh version from the network and updates the cache for next time — a great balance of speed and freshness for content that changes periodically. Offline fallback provides a custom offline page when both cache and network fail, ensuring users always see something meaningful rather than the browser's default offline error.

**4. Best Practices**

Request notification permission at a contextually appropriate moment — not immediately on page load, but after the user has engaged with your app and understands the value of notifications. Always provide clear value with each notification: a generic "Come back to our app!" message will get you blocked, while "Your order has shipped — tap to track" is genuinely useful. Handle notification clicks by navigating the user to the relevant content. Respect user preferences by providing in-app controls to customize notification types and frequency, and always honor the system-level notification permissions.`,
					CodeExamples: `// Background Sync
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-data') {
        event.waitUntil(syncData());
    }
});

async function syncData() {
    const requests = await getQueuedRequests();
    for (const request of requests) {
        try {
            await fetch(request.url, request.options);
            await removeFromQueue(request.id);
        } catch (error) {
            console.error('Sync failed:', error);
        }
    }
}

// Push Notifications
self.addEventListener('push', (event) => {
    const data = event.data.json();
    const options = {
        body: data.body,
        icon: '/icon.png',
        badge: '/badge.png',
        data: data.url
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

// Notification click
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    event.waitUntil(
        clients.openWindow(event.notification.data)
    );
});

// Cache strategies
// Network first
self.addEventListener('fetch', (event) => {
    event.respondWith(
        fetch(event.request)
            .then(response => {
                const clone = response.clone();
                caches.open('v1').then(cache => {
                    cache.put(event.request, clone);
                });
                return response;
            })
            .catch(() => caches.match(event.request))
    );
});`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          224,
			Title:       "Server-Side Rendering",
			Description: "SSR with Next.js and Nuxt.js: SSR benefits, hydration, and SEO optimization.",
			Order:       24,
			Lessons: []problems.Lesson{
				{
					Title: "SSR with Next.js",
					Content: `Server-Side Rendering (SSR) addresses a fundamental limitation of Single-Page Applications: when a user or search engine crawler requests a page, the server sends a fully-rendered HTML document instead of an empty shell that waits for JavaScript to download and execute. This dramatically improves both SEO (search engines can immediately index the content) and perceived performance (users see meaningful content within milliseconds instead of staring at a blank screen). Next.js is the most popular React framework for SSR, providing a powerful and flexible set of rendering strategies out of the box.

**1. Next.js Core Features**

Next.js offers a comprehensive toolkit for building production React applications. Its file-based routing system maps your file structure directly to URL paths — no manual route configuration needed. Server-side rendering runs your React components on the server for each request, generating full HTML that includes the fetched data. Static site generation pre-renders pages at build time, producing HTML files that can be served from a CDN for maximum speed. API routes let you build backend endpoints directly within your Next.js project, making it a true full-stack framework. The built-in Image component automatically optimizes images with resizing, format conversion (to WebP), lazy loading, and blur-up placeholders, often reducing image sizes by 50% or more.

**2. Rendering Modes Explained**

Next.js gives you four rendering strategies, and you can mix and match them on a per-page basis. Server-Side Rendering (SSR) with getServerSideProps runs on every request, fetching fresh data and rendering the page on the server — ideal for personalized content, dashboards, and pages where data changes frequently. Static Site Generation (SSG) with getStaticProps renders pages once at build time, producing static HTML that loads instantly from a CDN — perfect for blogs, documentation, and marketing pages. Incremental Static Regeneration (ISR) combines the speed of SSG with the freshness of SSR by serving the cached static page but regenerating it in the background at a configurable interval — for example, updating a product page every 60 seconds without rebuilding the entire site. Client-Side Rendering (CSR) remains available for highly interactive sections that do not need SEO, using standard React patterns with useEffect and data-fetching libraries.

**3. Benefits and Trade-offs**

The benefits of SSR are substantial. SEO improves because search engine crawlers receive complete HTML with all content immediately, rather than needing to execute JavaScript. The initial load is faster because users see rendered content while JavaScript hydrates in the background. Social sharing works correctly because platforms like Twitter and Facebook read meta tags from the initial HTML response. Pages even work with JavaScript disabled, providing graceful degradation. The trade-offs include increased server costs (every SSR request requires server computation), more complex deployment compared to a static SPA, and the need to understand the hydration process where the client-side React attaches event listeners to the server-rendered HTML.`,
					CodeExamples: `// Next.js page (SSR)
export async function getServerSideProps(context) {
    const res = await fetch('https://api.example.com/data');
    const data = await res.json();
    
    return {
        props: { data }
    };
}

function Page({ data }) {
    return <div>{data.title}</div>;
}

// Static generation (SSG)
export async function getStaticProps() {
    const res = await fetch('https://api.example.com/data');
    const data = await res.json();
    
    return {
        props: { data },
        revalidate: 60 // ISR: revalidate every 60 seconds
    };
}

// API route
export default function handler(req, res) {
    res.status(200).json({ message: 'Hello' });
}

// Dynamic routes
// pages/posts/[id].js
export async function getStaticPaths() {
    return {
        paths: [{ params: { id: '1' } }],
        fallback: false
    };
}`,
				},
				{
					Title: "SSR Frameworks: Nuxt.js, Remix, Astro",
					Content: `The SSR framework landscape has expanded well beyond Next.js, with several compelling alternatives that each bring unique philosophies and architectural approaches to server-side rendering. Understanding these options helps you choose the right tool for each project's specific requirements, whether you prioritize developer experience, performance, or flexibility.

**1. Nuxt.js — The Vue.js Ecosystem's Answer to Next.js**

Nuxt.js is the leading SSR framework for the Vue.js ecosystem, providing an opinionated but highly productive development experience. Like Next.js, it offers file-based routing, but it goes further with auto-imports — you do not need to manually import Vue's composition API functions, components, or composables, as Nuxt automatically resolves them. Nuxt 3 supports hybrid rendering, letting you choose SSR, SSG, or client-side rendering on a per-route basis. Its server engine (Nitro) can deploy to virtually any hosting platform, from traditional servers to edge functions. The module ecosystem is rich and mature, covering everything from authentication to content management.

**2. Remix — Web Standards First**

Remix takes a refreshingly different approach by building on top of web standards rather than abstracting them away. Instead of inventing its own data-fetching patterns, Remix leans heavily on native browser features like HTML forms, HTTP caching headers, and the Fetch API. Its nested routing system mirrors your UI hierarchy, with each route segment capable of loading its own data in parallel — eliminating the waterfall requests common in other frameworks. Remix's loader functions run on the server and provide data to components, while action functions handle form submissions. Error boundaries at every route level ensure that errors in one section of the page do not crash the entire application. The philosophy is that by embracing the platform, your app will work better for users on slow connections, with spotty JavaScript support, or using assistive technologies.

**3. Astro — The Islands Architecture Pioneer**

Astro introduced the islands architecture to the mainstream, fundamentally rethinking how much JavaScript is actually needed in most websites. Astro renders everything to static HTML by default and ships zero JavaScript to the client. Interactive components (islands) can be selectively hydrated using directives like client:load, client:visible, or client:idle, meaning JavaScript is only shipped for the specific components that need interactivity. The revolutionary aspect is that Astro is framework-agnostic — you can use React, Vue, Svelte, Solid, or even mix them in the same project. This makes Astro ideal for content-heavy sites like blogs, documentation, portfolios, and marketing pages where most content is static and only a few widgets need interactivity.

**4. Hydration Strategies**

Hydration is the process of making server-rendered HTML interactive by attaching JavaScript event listeners and state. Full hydration (used by Next.js and Nuxt by default) hydrates the entire page, which can be slow for large applications. Partial hydration selectively hydrates only the interactive parts of the page, reducing the JavaScript that needs to execute. The islands architecture takes this further by treating each interactive component as an independent island that hydrates on its own schedule. Progressive enhancement starts with fully-functional HTML and CSS, then layers on JavaScript enhancements for users whose browsers support them — ensuring your application works for everyone, everywhere.`,
					CodeExamples: `// Nuxt.js
// pages/users/[id].vue
<template>
    <div>
        <h1>{{ user.name }}</h1>
    </div>
</template>

<script setup>
const route = useRoute();
const { data: user } = await useFetch('/api/users/' + route.params.id);
</script>

// Remix
// app/routes/users/$id.tsx
export async function loader({ params }) {
    const user = await getUser(params.id);
    return json({ user });
}

export default function User() {
    const { user } = useLoaderData();
    return <h1>{user.name}</h1>;
}

// Astro
---
const { user } = Astro.props;
---
<div>
    <h1>{user.name}</h1>
</div>

// Hydration
<div data-island="UserProfile" data-props={JSON.stringify(user)}>
    <!-- Server-rendered content -->
</div>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          225,
			Title:       "Micro Frontends",
			Description: "Architect micro frontends: module federation, independent deployment, and team autonomy.",
			Order:       25,
			Lessons: []problems.Lesson{
				{
					Title: "Micro Frontends Architecture",
					Content: `Micro frontends apply the microservices philosophy to frontend development, breaking a monolithic frontend application into smaller, independently developed, tested, and deployed pieces. Just as backend microservices allow different teams to own different services with their own release cycles, micro frontends let different teams own different sections of the user interface. This architecture is particularly valuable for large organizations with multiple teams contributing to the same user-facing product — think of how a large e-commerce site might have separate teams for search, product pages, shopping cart, and checkout.

**1. Core Concepts**

The fundamental principle of micro frontends is team autonomy through technical boundaries. Each micro frontend is an independent application that owns a specific business domain or section of the UI. Teams can develop, test, and deploy their micro frontend without coordinating with other teams, dramatically reducing the communication overhead that slows large organizations. Technology diversity is a natural consequence — since each micro frontend is independent, teams can choose the framework and tools best suited to their needs. One team might use React, another Vue, and another Angular, all within the same product. This also means teams can incrementally modernize their stack without a risky big-bang rewrite.

**2. Implementation Approaches**

Module Federation, introduced in Webpack 5, is currently the most popular approach for runtime integration of micro frontends. It allows separate Webpack builds to share code at runtime, enabling one application to dynamically load components from another application hosted on a different server. Single-SPA is a JavaScript framework specifically designed for orchestrating multiple micro frontends, managing their lifecycle (mounting, unmounting, and routing). Iframe-based integration is the simplest and most isolated approach — each micro frontend runs in its own iframe, with complete CSS and JavaScript isolation — but it comes with limitations around communication, accessibility, and responsive design. Build-time integration composes micro frontends during the build process, producing a single deployable artifact, which is simpler but sacrifices independent deployment.

**3. Benefits in Practice**

The benefits become most apparent in large organizations. Independent teams can move at their own pace without being blocked by other teams' release schedules. Technology freedom means teams can adopt new frameworks or upgrade existing ones incrementally rather than committing the entire organization to a synchronized migration. Deployment becomes less risky because changes are scoped to a single micro frontend — if the checkout team deploys a bug, it does not affect the search or product browsing experience. Scaling both the application and the organization becomes easier because new teams can be spun up to own new features without expanding the existing teams.

**4. Challenges to Address**

Micro frontends introduce real complexity that must be carefully managed. Shared dependencies like React or a design system must be coordinated to avoid shipping duplicate copies of large libraries to the browser — Module Federation's shared configuration helps here. Styling conflicts arise when different micro frontends use CSS that leaks across boundaries — CSS modules, Shadow DOM, or strict naming conventions are essential. Cross-micro-frontend state management requires a well-defined communication protocol, typically using custom events, a shared event bus, or URL-based state. Testing becomes more complex because you need to verify not just individual micro frontends but also their integration — contract testing and end-to-end tests across the composed application are critical.`,
					CodeExamples: `// Module Federation (Webpack 5)
// Host app (webpack.config.js)
module.exports = {
    plugins: [
        new ModuleFederationPlugin({
            name: 'host',
            remotes: {
                remoteApp: 'remote@http://localhost:3001/remoteEntry.js'
            }
        })
    ]
};

// Remote app
module.exports = {
    plugins: [
        new ModuleFederationPlugin({
            name: 'remote',
            filename: 'remoteEntry.js',
            exposes: {
                './Button': './src/Button'
            }
        })
    ]
};

// Usage in host
const RemoteButton = React.lazy(() => import('remoteApp/Button'));

function App() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <RemoteButton />
        </Suspense>
    );
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          226,
			Title:       "WebAssembly Basics",
			Description: "Introduction to WebAssembly: performance-critical code, WASM modules, and integration.",
			Order:       26,
			Lessons: []problems.Lesson{
				{
					Title: "WebAssembly Introduction",
					Content: `WebAssembly (WASM) is a binary instruction format that runs in the browser at near-native speed, opening the door to a class of applications that were previously impossible or impractical on the web. While JavaScript has become remarkably fast thanks to JIT compilation, there are computationally intensive tasks where you need predictable, consistent, high-throughput performance — and that is exactly where WebAssembly shines.

**1. What is WebAssembly?**

WebAssembly is a low-level, assembly-like language with a compact binary format that runs in a sandboxed execution environment inside the browser. Unlike JavaScript, which must be parsed, compiled, and optimized at runtime, WASM modules arrive pre-compiled in a binary format that the browser can decode and execute almost instantly. This means startup time is fast and execution speed approaches that of native code — typically within 10-20% of native performance. WebAssembly is language-agnostic: it is not a language you write directly, but rather a compilation target. You write code in a higher-level language like C, C++, Rust, Go, or AssemblyScript, and compile it to WASM. The result is a .wasm file that the browser can load and execute alongside your JavaScript code.

**2. Practical Use Cases**

WebAssembly excels in scenarios where raw computational performance matters. Image and video processing — filters, resizing, format conversion, and real-time effects — runs significantly faster in WASM than in JavaScript. Browser-based games can use WASM for physics simulations, pathfinding, and rendering logic, which is why game engines like Unity and Unreal can now export to the web. Scientific computing applications like molecular modeling, data analysis, and simulation benefit from WASM's predictable performance. Cryptographic operations — encryption, hashing, and signature verification — are both faster and more secure when compiled from audited C/Rust libraries to WASM. Other real-world examples include PDF rendering (pdf.js uses WASM), font rasterization, audio processing, and CAD tools running entirely in the browser.

**3. Language Ecosystem**

C and C++ were the original WASM compilation targets via the Emscripten toolchain, and they remain popular for porting existing native codebases to the web. Rust has emerged as the most popular language for writing new WASM code, thanks to its excellent tooling (wasm-pack, wasm-bindgen), zero-cost abstractions, and lack of a garbage collector (which would add bloat). Go can compile to WASM, though the output is larger because it includes the Go runtime. AssemblyScript offers a TypeScript-like syntax that compiles to WASM, providing a gentle learning curve for JavaScript developers who want WASM performance without learning a systems language.

**4. JavaScript Integration**

WebAssembly modules are loaded and instantiated from JavaScript using the WebAssembly API. The instantiateStreaming method is the most efficient, as it compiles the WASM module while it is still downloading. Once loaded, you can call exported WASM functions directly from JavaScript as if they were regular functions. Data passing works through a shared linear memory buffer — a flat array of bytes that both JavaScript and WASM can read from and write to. For simple types like numbers, data passes directly. For complex types like strings and arrays, you need to serialize them into the shared memory. Libraries like wasm-bindgen (for Rust) and Emscripten (for C++) automate this marshaling, providing high-level bindings that make WASM functions feel like native JavaScript imports.`,
					CodeExamples: `// Load WASM module
async function loadWasm() {
    const wasmModule = await WebAssembly.instantiateStreaming(
        fetch('module.wasm')
    );
    
    return wasmModule.instance.exports;
}

// Use WASM function
const wasm = await loadWasm();
const result = wasm.add(5, 3); // 8

// Rust example (compiled to WASM)
// src/lib.rs
#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

// JavaScript integration
import init, { add } from './pkg/wasm_module.js';

async function run() {
    await init();
    console.log(add(5, 3)); // 8
}

run();`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          227,
			Title:       "Advanced TypeScript",
			Description: "Advanced TypeScript: conditional types, mapped types, template literals, and type manipulation.",
			Order:       27,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced TypeScript Types",
					Content: `Advanced TypeScript takes you beyond basic type annotations into the realm of type-level programming, where the type system itself becomes a powerful tool for enforcing correctness, generating types automatically, and catching entire categories of bugs at compile time. These features are what make TypeScript capable of modeling complex APIs, library interfaces, and domain-specific constraints that simpler type systems cannot express.

**1. Conditional Types**

Conditional types are the if/else of the type system. They follow the syntax T extends U ? X : Y, meaning "if type T is assignable to type U, the result is type X, otherwise type Y." This lets you create types that adapt based on their input. For example, you can create a type that extracts the return type of a function, unwraps a Promise, or converts a union of types into a different shape. A crucial feature is type inference within conditional types using the infer keyword — you can "capture" a type from within a pattern match. For instance, T extends Promise<infer R> ? R : T extracts the resolved type from a Promise. Distributive conditional types automatically distribute over union types: when you pass string | number to a conditional type, TypeScript evaluates the condition separately for each member of the union and combines the results. This is the foundation for powerful utility types like Exclude and Extract.

**2. Mapped Types**

Mapped types let you create new object types by transforming every property of an existing type. The syntax [K in keyof T] iterates over all keys of type T, and for each key you can modify the value type, add or remove modifiers like readonly or optional, or even rename the key. This is how TypeScript's built-in utility types like Readonly<T>, Partial<T>, Required<T>, and Pick<T, K> are implemented. Key remapping with the as clause lets you transform property names during mapping — for example, creating getter methods for every property by mapping each key K to a new key prefixed with "get" and capitalized. Template literal types combine with mapped types to generate string-based types dynamically, such as creating event handler names from property names.

**3. Type Manipulation Utilities**

TypeScript provides a rich set of built-in utility types for common type transformations. Extract<T, U> pulls out members of a union T that are assignable to U — useful for narrowing unions. Exclude<T, U> does the opposite, removing members that are assignable to U. NonNullable<T> removes null and undefined from a type. Branded types (also called nominal types or opaque types) are a pattern where you intersect a primitive type with a unique tag to create types that are structurally identical but semantically distinct — for example, making UserId and ProductId both strings but incompatible with each other. This prevents a common class of bugs where you accidentally pass a user ID where a product ID is expected, something structural typing alone cannot catch.`,
					CodeExamples: `// Conditional types
type IsArray<T> = T extends any[] ? true : false;
type A = IsArray<number[]>; // true
type B = IsArray<string>; // false

// Mapped types
type Readonly<T> = {
    readonly [P in keyof T]: T[P];
};

type Partial<T> = {
    [P in keyof T]?: T[P];
};

// Template literal types
// Note: Using string concatenation syntax to avoid Go parsing issues
// In TypeScript: type EventName<T extends string> = 'on' + Capitalize<T>;
type EventName<T extends string> = 'on' + Capitalize<T>;
type ClickEvent = EventName<'click'>; // "onClick"

// Type extraction
type Extract<T, U> = T extends U ? T : never;
type A = Extract<string | number, string>; // string

// Branded types
type UserId = string & { __brand: 'UserId' };
function createUserId(id: string): UserId {
    return id as UserId;
}`,
				},
				{
					Title: "Template Literal Types and Mapped Types",
					Content: `Template literal types and mapped types are two of TypeScript's most powerful features for metaprogramming at the type level. When combined, they allow you to generate complex type structures from simple inputs, create type-safe APIs that model string patterns, and build utility types that automate tedious type transformations across your codebase.

**1. Template Literal Types**

Template literal types bring the familiar string interpolation syntax to the type system. Just as JavaScript template literals let you embed expressions in strings at runtime, TypeScript template literal types let you embed types within string types at compile time. You can combine string literal types to create new string patterns: given type Prefix = "get" and type Name = "User", the template literal type produces "getUser". When you use unions in template literals, TypeScript distributes across all combinations — given type Method = "GET" | "POST" and type Path = "/users" | "/posts", a template literal type produces all four combinations. Built-in string manipulation types like Capitalize, Uppercase, Lowercase, and Uncapitalize transform string literal types, enabling patterns like automatically generating "onClick" from "click". Pattern matching with template literals lets you parse and extract parts of string types — for example, extracting "click" from "onClick" using conditional types with infer.

**2. Mapped Types in Depth**

Mapped types iterate over the keys of a type and produce a new type with transformed properties. The basic form [K in keyof T]: T[K] copies a type exactly, but you can modify it in powerful ways. Adding the readonly modifier makes all properties read-only. Adding the ? modifier makes all properties optional. The as clause enables key remapping, letting you rename, filter, or transform keys during the mapping. For example, you can create a Getters<T> type that takes an interface with properties like name and age and produces a new interface with methods getName() and getAge() — all generated automatically from the property names using template literal types in the as clause. You can also filter keys by using never in the as clause to exclude properties that do not match a condition.

**3. Building Advanced Type Utilities**

The real power emerges when you combine these features to build sophisticated type utilities. You can create type-safe event emitter interfaces where the method names and parameter types are derived from a schema type. You can build form validation types that automatically generate error message types matching each field. You can create API client types where the method signatures are derived from route definitions. Dynamic type generation means you can define your types once in a central schema and have TypeScript automatically derive all the related types — request types, response types, error types, and validation types — eliminating the manual synchronization that causes bugs in large codebases. These patterns are the foundation of how libraries like tRPC, Zod, and Prisma provide their remarkable type safety.`,
					CodeExamples: `// Template literal types
// Note: Using string concatenation syntax to avoid Go parsing issues  
// In TypeScript: type EventName<T extends string> = 'on' + Capitalize<T>;
type EventName<T extends string> = 'on' + Capitalize<T>;
type ClickEvent = EventName<'click'>; // "onClick"

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';
// In TypeScript: type ApiEndpoint = '/api/' + string;
type ApiEndpoint = '/api/' + string;
// In TypeScript: type ApiCall = HttpMethod + ' ' + ApiEndpoint;
type ApiCall = HttpMethod + ' ' + ApiEndpoint;

// Mapped types
type Readonly<T> = {
    readonly [P in keyof T]: T[P];
};

type Partial<T> = {
    [P in keyof T]?: T[P];
};

type Pick<T, K extends keyof T> = {
    [P in K]: T[P];
};

// Key remapping
// In TypeScript: type Getters<T> = { [K in keyof T as 'get' + Capitalize<string & K>]: () => T[K]; };
type Getters<T> = {
    [K in keyof T as 'get' + Capitalize<string & K>]: () => T[K];
};

interface User {
    name: string;
    age: number;
}

type UserGetters = Getters<User>;
// { getName: () => string; getAge: () => number; }`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          228,
			Title:       "Design Systems",
			Description: "Build design systems: component libraries, tokens, documentation, and consistency.",
			Order:       28,
			Lessons: []problems.Lesson{
				{
					Title: "Design System Fundamentals",
					Content: `A design system is a comprehensive collection of reusable components, design tokens, guidelines, and documentation that together form a single source of truth for how a product's user interface looks and behaves. Think of it as the shared language between designers and developers — it ensures that every button, form field, modal, and layout across your entire product family looks and works consistently, regardless of which team built it or when it was created.

**1. Design Tokens — The Foundation**

Design tokens are the atomic values that define your visual language: colors, spacing scales, typography settings, border radii, shadows, breakpoints, and animation durations. Rather than hardcoding hex values and pixel sizes throughout your codebase, you define them once as tokens and reference them everywhere. This provides two critical benefits: consistency (every team uses exactly the same shade of blue) and changeability (updating a single token value propagates the change everywhere). Tokens are typically organized in a hierarchy — global tokens define the raw values (blue-500: #3b82f6), semantic tokens map them to purposes (color-primary: blue-500), and component tokens specialize further (button-background: color-primary). Tools like Style Dictionary can generate tokens for multiple platforms (CSS custom properties, JavaScript objects, iOS/Android values) from a single source definition.

**2. Component Library — The Building Blocks**

The component library is the collection of reusable UI components that implement your design tokens and interaction patterns. These range from primitive components (Button, Input, Text, Icon) to composite components (Card, Modal, Table, Form) to layout components (Container, Stack, Grid). Each component should be well-typed with TypeScript, thoroughly accessible (proper ARIA attributes, keyboard navigation, screen reader support), and flexible enough to handle the range of use cases across your product. A good component library dramatically accelerates development because teams spend their time assembling pre-built, tested components rather than building UI from scratch.

**3. Documentation and Governance**

Documentation is what transforms a collection of components into a true design system. Every component needs clear usage guidelines: when to use it, when not to use it, the available variants and props, accessibility considerations, and visual examples of correct and incorrect usage. Pattern documentation goes beyond individual components to show how they compose together for common scenarios — form layouts, data tables with pagination, navigation patterns, and empty states. Governance defines how the design system evolves: how new components are proposed, reviewed, and added; how breaking changes are handled; and how teams contribute back to the system. Without governance, the design system either stagnates or fragments.

**4. Tooling Ecosystem**

Storybook is the de facto standard for developing, testing, and documenting components in isolation. It provides a sandbox environment where developers can build components without the overhead of the full application, test edge cases and different states, and create living documentation. Style Dictionary transforms design tokens from a platform-agnostic format into platform-specific outputs. Chromatic (built by the Storybook team) provides visual regression testing by capturing screenshots of every component story and detecting visual changes. Figma plugins can sync design tokens bidirectionally, ensuring designers and developers stay aligned.`,
					CodeExamples: `// Design tokens
const tokens = {
    colors: {
        primary: '#3b82f6',
        secondary: '#8b5cf6',
        success: '#10b981',
        error: '#ef4444'
    },
    spacing: {
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
        xl: '32px'
    },
    typography: {
        fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['Fira Code', 'monospace']
        },
        fontSize: {
            sm: '14px',
            base: '16px',
            lg: '18px',
            xl: '24px'
        }
    }
};

// Component using tokens
function Button({ variant = 'primary', children }) {
    return (
        <button 
            style={{
                backgroundColor: tokens.colors[variant],
                padding: tokens.spacing.md,
                fontFamily: tokens.typography.fontFamily.sans[0]
            }}
        >
            {children}
        </button>
    );
}

// Storybook story
export default {
    title: 'Components/Button',
    component: Button
};

export const Primary = () => <Button variant="primary">Click me</Button>;`,
				},
				{
					Title: "Storybook and Component Documentation",
					Content: `Storybook is the industry-standard tool for developing, testing, and documenting UI components in isolation from your application. Instead of navigating through your app to reach a specific component state, Storybook lets you render any component in any state instantly, making it dramatically faster to develop, debug, and verify UI components. Major companies like Airbnb, GitHub, and Microsoft use Storybook as the centerpiece of their design system workflow.

**1. Component Stories — The Core Concept**

A "story" in Storybook represents a single visual state of a component. Each story captures a specific combination of props, state, and context that produces a meaningful UI variation. For a Button component, you might have stories for Primary, Secondary, Disabled, Loading, With Icon, and Small/Large sizes. Stories are written as simple JavaScript objects or functions that return rendered components with specific props. The Component Story Format (CSF) is the standard way to write stories — each file exports a default metadata object (title, component, and configuration) and named exports for each story. This format is framework-agnostic and works with React, Vue, Angular, Svelte, and more.

**2. Interactive Features**

Storybook's Controls addon automatically generates a UI panel where you can dynamically adjust component props in real time — changing colors, toggling booleans, entering text — and immediately see the result. The Actions addon logs events fired by the component (onClick, onChange, onSubmit), letting you verify that event handlers work correctly without writing test code. Viewport testing lets you see how components render at different screen sizes, which is essential for responsive design. The accessibility addon runs axe-core checks against each story, flagging WCAG violations like missing alt text, insufficient color contrast, or missing ARIA attributes. Together, these tools create a comprehensive development environment where most component issues can be caught before they reach your application.

**3. Visual Regression Testing**

One of Storybook's most powerful capabilities is visual regression testing — automatically detecting unintended visual changes to your components. Tools like Chromatic capture screenshots of every story on every commit and compare them pixel-by-pixel against the baseline. When a visual change is detected, it is flagged for human review. This catches CSS regressions, layout shifts, and unintended side effects from seemingly unrelated code changes that automated unit tests would never detect. Combined with interaction testing (simulating user actions like clicks and keyboard input within stories), you can achieve comprehensive component testing without the flakiness and slowness of full end-to-end tests.

**4. Best Practices for Effective Stories**

Write stories for every component in your design system, covering all meaningful states: default, hover, active, disabled, error, loading, empty, and overflow. Document props with descriptions and default values using argTypes so that consumers understand the component's API. Organize stories in a logical hierarchy that mirrors your component library structure. Use decorators to provide consistent context (theme providers, layout wrappers, mock data) across related stories. Keep stories simple and focused — each story should demonstrate one thing clearly. Use the Docs addon to generate rich documentation pages that combine live examples, prop tables, and usage guidelines in a format that both designers and developers can reference.`,
					CodeExamples: `// Storybook story
export default {
    title: 'Components/Button',
    component: Button,
    parameters: {
        docs: {
            description: {
                component: 'A reusable button component'
            }
        }
    },
    argTypes: {
        variant: {
            control: 'select',
            options: ['primary', 'secondary', 'danger']
        }
    }
};

export const Primary = {
    args: {
        variant: 'primary',
        children: 'Click me'
    }
};

export const Secondary = {
    args: {
        variant: 'secondary',
        children: 'Secondary'
    }
};

// With interactions
import { userEvent, within } from '@storybook/testing-library';

export const Clicked = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const button = canvas.getByRole('button');
        await userEvent.click(button);
    }
};`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          229,
			Title:       "Frontend Architecture",
			Description: "Design scalable frontend architectures: folder structure, state management, and patterns.",
			Order:       29,
			Lessons: []problems.Lesson{
				{
					Title: "Architecture Patterns",
					Content: `Frontend architecture is the art of organizing your codebase so that it remains maintainable, scalable, and understandable as it grows from a simple prototype to a complex production application with dozens of contributors. Poor architecture leads to spaghetti code, circular dependencies, and the dreaded "big ball of mud" where every change requires touching dozens of files and praying nothing breaks. Good architecture creates clear boundaries, predictable patterns, and a codebase that new team members can navigate confidently within their first week.

**1. Architecture Principles**

Separation of concerns means each module, component, or function should handle one distinct aspect of the application. Business logic should not live inside UI components, API calls should not be scattered throughout the render tree, and styling concerns should not be tangled with data transformation. The Single Responsibility Principle (SRP) states that each module should have only one reason to change — if a component handles both user display and data fetching, changes to the API contract force modifications to a UI file. DRY (Do Not Repeat Yourself) eliminates duplication by extracting shared logic into reusable utilities, hooks, or components. However, premature abstraction is worse than duplication — wait until you see the same pattern three times before extracting it. The SOLID principles, originally from object-oriented design, translate well to frontend architecture: dependency inversion (depend on abstractions, not concretions) is particularly relevant for services and API layers.

**2. Folder Structure Strategies**

Feature-based organization groups all files related to a feature (components, hooks, services, types, tests) together in a single directory. This makes features self-contained and easy to find, move, or delete — when you remove a feature, you delete one folder. Layer-based organization groups files by their technical role (all components in one folder, all hooks in another, all services in another). This works well for small applications but breaks down as the codebase grows because related files are spread across many directories. Domain-driven design (DDD) organization groups files by business domain (auth, products, orders, payments), with each domain containing its own components, logic, and types. The best approach often combines these: feature-based at the top level with layer-based organization within each feature folder.

**3. State Management Categories**

Modern frontend applications deal with four distinct categories of state, each requiring different management strategies. Local state (component state) is managed with useState or useReducer and should be the default for data that only one component needs. Global state (shared across many components) uses Context API, Redux, Zustand, or Jotai for data like user authentication, theme preferences, and shopping cart contents. Server state (data fetched from APIs) is best managed with dedicated libraries like React Query, SWR, or Apollo Client that handle caching, revalidation, background refetching, and optimistic updates. URL state (encoded in the browser's address bar) includes route parameters, query strings, and hash values — this state should be the source of truth for anything that should be shareable via link, such as filters, pagination, and search queries.

**4. Component Architecture Patterns**

The Container/Presentational pattern separates components into containers (which handle data fetching and state management) and presentational components (which receive data via props and focus purely on rendering). The hooks pattern extracts stateful logic into custom hooks, keeping components thin and logic reusable. Render props provide maximum flexibility by letting parent components control how data is rendered. Compound components create cohesive component families that share state implicitly, offering clean APIs for consumers while encapsulating complexity internally. Choosing the right pattern depends on the specific situation — hooks for most reusable logic, compound components for flexible multi-part UI elements, and container/presentational when you need clear separation between data orchestration and visual rendering.`,
					CodeExamples: `// Feature-based structure
src/
  features/
    auth/
      components/
      hooks/
      services/
      types/
    dashboard/
      components/
      hooks/
      services/
  shared/
    components/
    hooks/
    utils/
  app/
    routes/
    store/

// Container/Presentational pattern
// Container
function UserListContainer() {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUsers().then(data => {
            setUsers(data);
            setLoading(false);
        });
    }, []);
    
    return <UserList users={users} loading={loading} />;
}

// Presentational
function UserList({ users, loading }) {
    if (loading) return <Spinner />;
    return (
        <ul>
            {users.map(user => (
                <li key={user.id}>{user.name}</li>
            ))}
        </ul>
    );
}

// Custom hooks for logic
function useUsers() {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUsers()
            .then(setUsers)
            .finally(() => setLoading(false));
    }, []);
    
    return { users, loading };
}`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
