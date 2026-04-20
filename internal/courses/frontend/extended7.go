package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1560,
			Title:       "Accessibility and Inclusive Design",
			Description: "Build accessible web applications following WCAG guidelines with ARIA patterns, semantic HTML, keyboard navigation, and screen reader support.",
			Order:       60,
			Lessons: []problems.Lesson{
				{
					Title: "WCAG Guidelines and Semantic HTML",
					Content: `Web accessibility ensures applications are usable by everyone, including people with disabilities.

**WCAG 2.2 Principles:**
` + "```" + `
POUR principles:
  Perceivable:
    Provide text alternatives for non-text content
    Provide captions and alternatives for multimedia
    Content adaptable to different presentations
    Distinguishable (color, contrast, resize text)
  
  Operable:
    Keyboard accessible (all functionality)
    Enough time to read and use content
    No content that causes seizures
    Navigable (help users find content)
    Input modalities beyond keyboard
  
  Understandable:
    Readable text content
    Predictable web page behavior
    Input assistance (help users avoid errors)
  
  Robust:
    Compatible with current and future tools
    Valid HTML
    Status messages for assistive technology

Conformance levels:
  Level A: Minimum (must fix)
    Images have alt text
    Keyboard accessible
    No auto-playing audio
  
  Level AA: Standard (target for most sites)
    Color contrast 4.5:1 (text), 3:1 (large text)
    Text resizable to 200%
    Focus visible
    Consistent navigation
  
  Level AAA: Enhanced (aspirational)
    Color contrast 7:1
    Sign language for media
    No timing constraints

Color contrast:
  Normal text (< 18pt): 4.5:1 ratio minimum
  Large text (>= 18pt or >= 14pt bold): 3:1 ratio minimum
  UI components and graphical objects: 3:1 ratio minimum
  
  Tools:
    Chrome DevTools contrast checker
    axe DevTools extension
    WebAIM contrast checker
    Polypane contrast checker
` + "```" + `

**Semantic HTML:**
` + "```" + `html
<!-- BAD: Div soup -->
<div class="header">
  <div class="nav">
    <div class="nav-item" onclick="goTo('/')">Home</div>
  </div>
</div>
<div class="content">
  <div class="article">
    <div class="title">My Post</div>
    <div class="text">Content here...</div>
  </div>
  <div class="sidebar">
    <div class="widget">Related Links</div>
  </div>
</div>
<div class="footer">Copyright 2024</div>

<!-- GOOD: Semantic HTML -->
<header>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
      <li><a href="/contact">Contact</a></li>
    </ul>
  </nav>
</header>

<main>
  <article>
    <h1>My Post</h1>
    <time datetime="2024-01-15">January 15, 2024</time>
    <p>Content here...</p>
    
    <section aria-labelledby="comments-heading">
      <h2 id="comments-heading">Comments</h2>
      <!-- comments -->
    </section>
  </article>
  
  <aside aria-label="Related content">
    <h2>Related Links</h2>
    <nav aria-label="Related articles">
      <ul>
        <li><a href="/post-2">Related Post</a></li>
      </ul>
    </nav>
  </aside>
</main>

<footer>
  <p>&copy; 2024 My Site</p>
</footer>

<!-- Form accessibility -->
<form aria-labelledby="signup-heading">
  <h2 id="signup-heading">Sign Up</h2>
  
  <div>
    <label for="email">Email <span aria-hidden="true">*</span></label>
    <input 
      id="email" 
      type="email" 
      required
      aria-required="true"
      aria-describedby="email-help email-error"
      aria-invalid="false"
    />
    <p id="email-help">We'll never share your email.</p>
    <p id="email-error" role="alert" hidden>Please enter a valid email.</p>
  </div>
  
  <div>
    <label for="password">Password <span aria-hidden="true">*</span></label>
    <input 
      id="password" 
      type="password" 
      required
      aria-required="true"
      aria-describedby="password-requirements"
      minlength="8"
    />
    <div id="password-requirements">
      <p>Password must contain:</p>
      <ul>
        <li>At least 8 characters</li>
        <li>One uppercase letter</li>
        <li>One number</li>
      </ul>
    </div>
  </div>
  
  <fieldset>
    <legend>Notification preferences</legend>
    <label>
      <input type="checkbox" name="notifications" value="email" />
      Email notifications
    </label>
    <label>
      <input type="checkbox" name="notifications" value="sms" />
      SMS notifications
    </label>
  </fieldset>
  
  <button type="submit">Create Account</button>
</form>

<!-- Table accessibility -->
<table>
  <caption>Quarterly Sales Report</caption>
  <thead>
    <tr>
      <th scope="col">Product</th>
      <th scope="col">Q1</th>
      <th scope="col">Q2</th>
      <th scope="col">Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Widget A</th>
      <td>$10,000</td>
      <td>$12,000</td>
      <td>$22,000</td>
    </tr>
  </tbody>
</table>
` + "```" + ``,
					CodeExamples: `// Accessible React components

// 1. Accessible Modal/Dialog
function Modal({ isOpen, onClose, title, children }) {
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement;
      modalRef.current?.focus();
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
      previousFocusRef.current?.focus();
    }
    
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // Trap focus inside modal
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') { onClose(); return; }
    if (e.key !== 'Tab') return;

    const focusable = modalRef.current.querySelectorAll(
      'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === first) { last.focus(); e.preventDefault(); }
    } else {
      if (document.activeElement === last) { first.focus(); e.preventDefault(); }
    }
  };

  if (!isOpen) return null;

  return createPortal(
    <div
      className="modal-overlay"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      aria-hidden="true"
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        tabIndex={-1}
        onKeyDown={handleKeyDown}
        className="modal"
      >
        <h2 id="modal-title">{title}</h2>
        {children}
        <button onClick={onClose} aria-label="Close dialog">
          &times;
        </button>
      </div>
    </div>,
    document.body
  );
}

// 2. Accessible Tabs
function Tabs({ tabs }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const tabRefs = useRef([]);

  const handleKeyDown = (e, index) => {
    let newIndex;
    switch (e.key) {
      case 'ArrowRight':
        newIndex = (index + 1) % tabs.length;
        break;
      case 'ArrowLeft':
        newIndex = (index - 1 + tabs.length) % tabs.length;
        break;
      case 'Home':
        newIndex = 0;
        break;
      case 'End':
        newIndex = tabs.length - 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    setActiveIndex(newIndex);
    tabRefs.current[newIndex]?.focus();
  };

  return (
    <div>
      <div role="tablist" aria-label="Content tabs">
        {tabs.map((tab, i) => (
          <button
            key={tab.id}
            ref={el => tabRefs.current[i] = el}
            role="tab"
            id={'tab-' + tab.id}
            aria-controls={'panel-' + tab.id}
            aria-selected={activeIndex === i}
            tabIndex={activeIndex === i ? 0 : -1}
            onClick={() => setActiveIndex(i)}
            onKeyDown={(e) => handleKeyDown(e, i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab, i) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={'panel-' + tab.id}
          aria-labelledby={'tab-' + tab.id}
          hidden={activeIndex !== i}
          tabIndex={0}
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}

// 3. Screen reader announcements
function LiveRegion() {
  const [message, setMessage] = useState('');

  const announce = (text, priority = 'polite') => {
    setMessage('');
    requestAnimationFrame(() => setMessage(text));
  };

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="sr-only"
    >
      {message}
    </div>
  );
}

// CSS for screen readers only
// .sr-only {
//   position: absolute;
//   width: 1px;
//   height: 1px;
//   padding: 0;
//   margin: -1px;
//   overflow: hidden;
//   clip: rect(0, 0, 0, 0);
//   white-space: nowrap;
//   border-width: 0;
// }`,
				},
				{
					Title: "ARIA Patterns and Keyboard Navigation",
					Content: `ARIA (Accessible Rich Internet Applications) provides attributes to make dynamic web content accessible to assistive technologies.

**ARIA Roles and Attributes:**
` + "```" + `html
<!-- ARIA Landmark Roles (prefer semantic HTML) -->
<div role="banner">         <!-- prefer <header> -->
<div role="navigation">      <!-- prefer <nav> -->
<div role="main">            <!-- prefer <main> -->
<div role="complementary">   <!-- prefer <aside> -->
<div role="contentinfo">     <!-- prefer <footer> -->
<div role="search">          <!-- <search> in HTML5.2 -->
<div role="region" aria-labelledby="section-title">

<!-- Live Regions -->
<!-- Announce changes to assistive technology -->
<div aria-live="polite">  <!-- Non-urgent updates -->
  Items added to cart: 3
</div>

<div aria-live="assertive"> <!-- Urgent, interrupt -->
  Error: Connection lost
</div>

<div role="status">  <!-- Implicit aria-live="polite" -->
  Search results: 42 items found
</div>

<div role="alert">  <!-- Implicit aria-live="assertive" -->
  Session expires in 2 minutes
</div>

<!-- Disclosure pattern -->
<button 
  aria-expanded="false" 
  aria-controls="menu-content"
>
  Menu
</button>
<div id="menu-content" hidden>
  <a href="/profile">Profile</a>
  <a href="/settings">Settings</a>
</div>

<!-- Combobox (Autocomplete) -->
<label for="search">Search</label>
<div role="combobox" aria-expanded="true" aria-haspopup="listbox">
  <input
    id="search"
    type="text"
    aria-autocomplete="list"
    aria-controls="search-listbox"
    aria-activedescendant="option-1"
  />
  <ul id="search-listbox" role="listbox">
    <li id="option-1" role="option" aria-selected="true">
      React
    </li>
    <li id="option-2" role="option" aria-selected="false">
      Vue
    </li>
  </ul>
</div>

<!-- Toast notifications -->
<div 
  role="region" 
  aria-label="Notifications"
  aria-live="polite"
>
  <div role="status" class="toast">
    File saved successfully
  </div>
</div>

<!-- Progress indicator -->
<div 
  role="progressbar" 
  aria-valuenow="65" 
  aria-valuemin="0" 
  aria-valuemax="100"
  aria-label="Upload progress"
>
  65%
</div>

<!-- Breadcrumb -->
<nav aria-label="Breadcrumb">
  <ol>
    <li><a href="/">Home</a></li>
    <li><a href="/products">Products</a></li>
    <li><a href="/products/electronics" aria-current="page">Electronics</a></li>
  </ol>
</nav>
` + "```" + `

**Keyboard Navigation Patterns:**
` + "```" + `javascript
// Roving tabindex pattern
// Only one item in group is tabbable, arrows move focus
function Menu({ items, onSelect }) {
  const [focusedIndex, setFocusedIndex] = useState(0);
  const itemRefs = useRef([]);

  const handleKeyDown = (e) => {
    let newIndex = focusedIndex;
    
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        e.preventDefault();
        newIndex = (focusedIndex + 1) % items.length;
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        e.preventDefault();
        newIndex = (focusedIndex - 1 + items.length) % items.length;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = items.length - 1;
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        onSelect(items[focusedIndex]);
        return;
      default:
        // Type-ahead: focus item starting with typed character
        const char = e.key.toLowerCase();
        const matchIndex = items.findIndex(
          (item, i) => i > focusedIndex && 
            item.label.toLowerCase().startsWith(char)
        );
        if (matchIndex >= 0) newIndex = matchIndex;
        break;
    }
    
    setFocusedIndex(newIndex);
    itemRefs.current[newIndex]?.focus();
  };

  return (
    <ul role="menu" onKeyDown={handleKeyDown}>
      {items.map((item, i) => (
        <li
          key={item.id}
          ref={el => itemRefs.current[i] = el}
          role="menuitem"
          tabIndex={i === focusedIndex ? 0 : -1}
          onClick={() => onSelect(item)}
        >
          {item.label}
        </li>
      ))}
    </ul>
  );
}

// Skip navigation link
function SkipNav() {
  return (
    <a 
      href="#main-content" 
      className="skip-nav"
      // CSS: position absolute, off-screen until focused
    >
      Skip to main content
    </a>
  );
}

// Focus management after route changes (SPA)
function useFocusOnRouteChange() {
  const location = useLocation();
  const mainRef = useRef(null);

  useEffect(() => {
    // Focus main content after navigation
    mainRef.current?.focus();
    
    // Announce page change
    const title = document.title;
    const announcer = document.getElementById('route-announcer');
    if (announcer) announcer.textContent = 'Navigated to ' + title;
  }, [location.pathname]);

  return mainRef;
}

// Accessible drag and drop (keyboard alternative)
function DraggableList({ items, onReorder }) {
  const [activeId, setActiveId] = useState(null);

  const handleKeyDown = (e, index) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      if (activeId === null) {
        setActiveId(items[index].id);
        // Announce: "Grabbed item. Use arrow keys to move."
      } else {
        setActiveId(null);
        // Announce: "Dropped item at position X"
      }
    }
    
    if (activeId !== null) {
      if (e.key === 'ArrowUp' && index > 0) {
        e.preventDefault();
        const newItems = [...items];
        [newItems[index], newItems[index - 1]] = [newItems[index - 1], newItems[index]];
        onReorder(newItems);
      }
      if (e.key === 'ArrowDown' && index < items.length - 1) {
        e.preventDefault();
        const newItems = [...items];
        [newItems[index], newItems[index + 1]] = [newItems[index + 1], newItems[index]];
        onReorder(newItems);
      }
      if (e.key === 'Escape') {
        setActiveId(null);
      }
    }
  };

  return (
    <ul role="listbox" aria-label="Reorderable list">
      {items.map((item, i) => (
        <li
          key={item.id}
          role="option"
          aria-selected={activeId === item.id}
          aria-grabbed={activeId === item.id}
          tabIndex={0}
          onKeyDown={(e) => handleKeyDown(e, i)}
        >
          {item.label}
          {activeId === item.id && ' (grabbed)'}
        </li>
      ))}
    </ul>
  );
}
` + "```" + ``,
					CodeExamples: `// Accessibility testing and patterns

// 1. Accessible form validation
function AccessibleForm() {
  const [errors, setErrors] = useState({});
  const errorSummaryRef = useRef(null);

  const validate = (values) => {
    const errs = {};
    if (!values.name) errs.name = 'Name is required';
    if (!values.email) errs.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(values.email))
      errs.email = 'Email format is invalid';
    if (!values.password) errs.password = 'Password is required';
    else if (values.password.length < 8)
      errs.password = 'Password must be at least 8 characters';
    return errs;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const values = Object.fromEntries(formData);
    const errs = validate(values);
    setErrors(errs);
    
    if (Object.keys(errs).length > 0) {
      // Focus error summary for screen readers
      errorSummaryRef.current?.focus();
    }
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      {Object.keys(errors).length > 0 && (
        <div
          ref={errorSummaryRef}
          role="alert"
          tabIndex={-1}
          className="error-summary"
        >
          <h3>Please fix the following errors:</h3>
          <ul>
            {Object.entries(errors).map(([field, msg]) => (
              <li key={field}>
                <a href={'#' + field}>{msg}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      <div>
        <label htmlFor="name">
          Name <span aria-hidden="true">*</span>
        </label>
        <input
          id="name"
          name="name"
          aria-required="true"
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
        />
        {errors.name && (
          <p id="name-error" className="error" role="alert">
            {errors.name}
          </p>
        )}
      </div>
      
      <div>
        <label htmlFor="email">
          Email <span aria-hidden="true">*</span>
        </label>
        <input
          id="email"
          name="email"
          type="email"
          aria-required="true"
          aria-invalid={!!errors.email}
          aria-describedby={errors.email ? 'email-error' : undefined}
        />
        {errors.email && (
          <p id="email-error" className="error" role="alert">
            {errors.email}
          </p>
        )}
      </div>
      
      <button type="submit">Submit</button>
    </form>
  );
}

// 2. useReducedMotion hook
function useReducedMotion() {
  const [prefersReduced, setPrefersReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e) => setPrefersReduced(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  return prefersReduced;
}

// Usage
function AnimatedComponent({ children }) {
  const prefersReduced = useReducedMotion();
  
  return (
    <motion.div
      initial={{ opacity: 0, y: prefersReduced ? 0 : 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: prefersReduced ? 0 : 0.3 }}
    >
      {children}
    </motion.div>
  );
}

// 3. Color scheme support
function useColorScheme() {
  const [scheme, setScheme] = useState(() => {
    const stored = localStorage.getItem('color-scheme');
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', scheme);
    localStorage.setItem('color-scheme', scheme);
  }, [scheme]);

  return { scheme, setScheme, toggle: () => 
    setScheme(s => s === 'dark' ? 'light' : 'dark')
  };
}`,
				},
			},
		},
		{
			ID:          1561,
			Title:       "Build Tools and Module Systems",
			Description: "Modern JavaScript build tooling including Vite, esbuild, webpack, module federation, and monorepo management with Turborepo.",
			Order:       61,
			Lessons: []problems.Lesson{
				{
					Title: "Vite and Modern Build Tools",
					Content: `Modern build tools prioritize speed and developer experience with native ES modules and efficient bundling.

**Vite:**
` + "```" + `javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({ open: true, gzipSize: true }),
  ],
  
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@utils': '/src/utils',
    },
  },
  
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  
  build: {
    target: 'es2020',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          utils: ['lodash-es', 'date-fns'],
        },
      },
    },
    chunkSizeWarningLimit: 500,
  },
  
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
    preprocessorOptions: {
      scss: {
        additionalData: '@import "@/styles/variables";',
      },
    },
  },
  
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});

// Environment variables
// .env.development
VITE_API_URL=http://localhost:8080
VITE_APP_TITLE=My App (Dev)

// .env.production
VITE_API_URL=https://api.example.com
VITE_APP_TITLE=My App

// Usage
const apiUrl = import.meta.env.VITE_API_URL;
const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;
` + "```" + `

**Build Optimization:**
` + "```" + `javascript
// Code splitting strategies
// 1. Route-based splitting (most common)
const Home = lazy(() => import('./pages/Home'));
const Dashboard = lazy(() => import('./pages/Dashboard'));

// 2. Component-based splitting
const HeavyChart = lazy(() => import('./components/HeavyChart'));
const MarkdownEditor = lazy(() => import('./components/MarkdownEditor'));

// 3. Library splitting
// Dynamic import of heavy library
async function processImage(file) {
  const sharp = await import('sharp');
  return sharp(file).resize(800).toBuffer();
}

// Tree shaking
// Good: Named imports allow tree shaking
import { debounce } from 'lodash-es';

// Bad: Default import pulls entire library
// import _ from 'lodash';

// Barrel file optimization
// Instead of: export * from './ComponentA';
// Use direct imports: import { ComponentA } from './ComponentA';

// Package.json sideEffects
{
  "sideEffects": false,
  // Or specify files with side effects:
  "sideEffects": ["*.css", "*.scss", "./src/polyfills.js"]
}

// Bundle analysis
// vite: rollup-plugin-visualizer
// webpack: webpack-bundle-analyzer
// generic: source-map-explorer

// Asset optimization
// Images: Use WebP/AVIF with fallbacks
<picture>
  <source srcset="image.avif" type="image/avif" />
  <source srcset="image.webp" type="image/webp" />
  <img src="image.jpg" alt="Description" width="800" height="600" loading="lazy" />
</picture>

// Font optimization
// 1. Subset fonts to used characters
// 2. Use font-display: swap
// 3. Preload critical fonts
<link rel="preload" href="/fonts/Inter.woff2" as="font" type="font/woff2" crossorigin />

@font-face {
  font-family: 'Inter';
  src: url('/fonts/Inter.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
  unicode-range: U+0000-00FF; /* Latin subset */
}

// Critical CSS
// Inline above-the-fold CSS in <head>
// Load rest asynchronously
<link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'" />
<noscript><link rel="stylesheet" href="styles.css" /></noscript>
` + "```" + ``,
					CodeExamples: `// Build tool configurations

// 1. Monorepo with Turborepo
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "test": {
      "dependsOn": ["build"]
    },
    "lint": {},
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}

// Package structure
// packages/
//   ui/          → shared component library
//   config/      → shared configs (eslint, tsconfig)
//   utils/       → shared utilities
// apps/
//   web/         → Next.js app
//   mobile/      → React Native app
//   docs/        → documentation site

// packages/ui/package.json
{
  "name": "@acme/ui",
  "version": "0.1.0",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": "./dist/index.js",
    "./button": "./dist/button.js",
    "./input": "./dist/input.js"
  },
  "scripts": {
    "build": "tsup src/index.ts --dts --format esm,cjs",
    "dev": "tsup src/index.ts --dts --format esm,cjs --watch"
  }
}

// 2. Module Federation (Micro-frontends)
// webpack.config.js for shell app
const ModuleFederationPlugin = require('webpack/lib/container/ModuleFederationPlugin');

module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'shell',
      remotes: {
        dashboard: 'dashboard@http://localhost:3001/remoteEntry.js',
        settings: 'settings@http://localhost:3002/remoteEntry.js',
      },
      shared: {
        react: { singleton: true, requiredVersion: '^18.0.0' },
        'react-dom': { singleton: true, requiredVersion: '^18.0.0' },
      },
    }),
  ],
};

// Remote app (dashboard)
module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'dashboard',
      filename: 'remoteEntry.js',
      exposes: {
        './DashboardApp': './src/App',
        './DashboardWidget': './src/components/Widget',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
      },
    }),
  ],
};

// Loading remote in shell
const DashboardApp = React.lazy(() => import('dashboard/DashboardApp'));

function App() {
  return (
    <Suspense fallback="Loading dashboard...">
      <DashboardApp />
    </Suspense>
  );
}

// 3. tsup configuration (library bundler)
// tsup.config.ts
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  minify: true,
  external: ['react', 'react-dom'],
});`,
				},
			},
		},
	})
}
