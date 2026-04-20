package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1568,
			Title:       "Web Components and Micro-frontends",
			Description: "Build framework-agnostic web components with Shadow DOM, custom elements, and implement micro-frontend architectures for large-scale applications.",
			Order:       68,
			Lessons: []problems.Lesson{
				{
					Title: "Web Components and Shadow DOM",
					Content: `Web Components are a set of web platform APIs that allow creating reusable custom elements with encapsulated functionality.

**Custom Elements:**
` + "```" + `javascript
// Define a custom element
class UserCard extends HTMLElement {
  // Observed attributes trigger attributeChangedCallback
  static get observedAttributes() {
    return ['name', 'email', 'avatar', 'role'];
  }

  constructor() {
    super();
    // Attach shadow DOM for style encapsulation
    this.attachShadow({ mode: 'open' });
  }

  // Called when element is added to the DOM
  connectedCallback() {
    this.render();
    this.setupEventListeners();
  }

  // Called when element is removed from the DOM
  disconnectedCallback() {
    this.cleanup();
  }

  // Called when an observed attribute changes
  attributeChangedCallback(name, oldValue, newValue) {
    if (oldValue !== newValue) {
      this.render();
    }
  }

  // Called when element is moved to a new document
  adoptedCallback() {}

  setupEventListeners() {
    this.shadowRoot.querySelector('.follow-btn')
      ?.addEventListener('click', () => {
        this.dispatchEvent(new CustomEvent('follow', {
          bubbles: true,
          composed: true, // Cross shadow DOM boundary
          detail: { name: this.getAttribute('name') },
        }));
      });
  }

  cleanup() {
    // Remove event listeners, cancel requests, etc.
  }

  render() {
    const name = this.getAttribute('name') || 'Unknown';
    const email = this.getAttribute('email') || '';
    const avatar = this.getAttribute('avatar') || '/default-avatar.png';
    const role = this.getAttribute('role') || 'Member';

    this.shadowRoot.innerHTML = '';

    const style = document.createElement('style');
    style.textContent = '.card { ' +
      'display: flex; align-items: center; gap: 1rem; ' +
      'padding: 1rem; border: 1px solid #e5e7eb; ' +
      'border-radius: 0.75rem; background: white; ' +
      'font-family: system-ui, sans-serif; ' +
    '} ' +
    '.avatar { ' +
      'width: 48px; height: 48px; border-radius: 50%; object-fit: cover; ' +
    '} ' +
    '.info { flex: 1; } ' +
    '.name { font-weight: 600; font-size: 1rem; margin: 0; } ' +
    '.email { color: #6b7280; font-size: 0.875rem; margin: 0.25rem 0 0; } ' +
    '.role { ' +
      'display: inline-block; padding: 0.125rem 0.5rem; ' +
      'background: #eff6ff; color: #2563eb; ' +
      'border-radius: 9999px; font-size: 0.75rem; margin-top: 0.25rem; ' +
    '} ' +
    '.follow-btn { ' +
      'padding: 0.5rem 1rem; background: #3b82f6; color: white; ' +
      'border: none; border-radius: 0.5rem; cursor: pointer; ' +
      'font-size: 0.875rem; ' +
    '} ' +
    '.follow-btn:hover { background: #2563eb; } ' +
    ':host { display: block; } ' +
    ':host([hidden]) { display: none; } ' +
    ':host-context(.dark-theme) .card { background: #1f2937; border-color: #374151; } ' +
    ':host-context(.dark-theme) .name { color: #f9fafb; } ' +
    '::slotted(p) { margin: 0.5rem 0; color: #4b5563; }';

    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML =
      '<img class="avatar" src="' + avatar + '" alt="' + name + ' avatar" />' +
      '<div class="info">' +
        '<p class="name">' + name + '</p>' +
        '<p class="email">' + email + '</p>' +
        '<span class="role">' + role + '</span>' +
        '<slot name="bio"></slot>' +
      '</div>' +
      '<button class="follow-btn">Follow</button>';

    this.shadowRoot.appendChild(style);
    this.shadowRoot.appendChild(card);
    this.setupEventListeners();
  }
}

// Register the element
customElements.define('user-card', UserCard);

// Usage in HTML:
// <user-card 
//   name="Jane Doe" 
//   email="jane@example.com" 
//   avatar="/jane.jpg"
//   role="Admin"
// >
//   <p slot="bio">Software engineer passionate about web standards.</p>
// </user-card>

// Form-associated custom element
class CustomInput extends HTMLElement {
  static formAssociated = true;
  static get observedAttributes() { return ['value', 'required']; }

  constructor() {
    super();
    this.internals = this.attachInternals();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.shadowRoot.innerHTML =
      '<style> input { padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 0.375rem; } </style>' +
      '<input type="text" />';

    this.input = this.shadowRoot.querySelector('input');
    this.input.addEventListener('input', () => {
      this.internals.setFormValue(this.input.value);
      this.validate();
    });
  }

  validate() {
    if (this.hasAttribute('required') && !this.input.value) {
      this.internals.setValidity(
        { valueMissing: true },
        'This field is required',
        this.input
      );
    } else {
      this.internals.setValidity({});
    }
  }

  get value() { return this.input?.value || ''; }
  set value(v) { if (this.input) this.input.value = v; }
}

customElements.define('custom-input', CustomInput);
` + "```" + `

**Lit Framework:**
` + "```" + `javascript
import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

@customElement('todo-list')
class TodoList extends LitElement {
  static styles = css'
    :host {
      display: block;
      font-family: system-ui, sans-serif;
    }
    ul {
      list-style: none;
      padding: 0;
    }
    li {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem;
      border-bottom: 1px solid #e5e7eb;
    }
    .completed {
      text-decoration: line-through;
      color: #9ca3af;
    }
    input[type="text"] {
      flex: 1;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
    }
    button {
      padding: 0.5rem 1rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 0.375rem;
      cursor: pointer;
    }
  ';

  @property({ type: String }) title = 'Todo List';
  @state() todos = [];
  @state() newTodo = '';

  render() {
    return html'
      <h2>${this.title}</h2>
      <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
        <input
          type="text"
          .value=${this.newTodo}
          @input=${(e) => this.newTodo = e.target.value}
          @keydown=${(e) => e.key === 'Enter' && this.addTodo()}
          placeholder="Add a todo..."
        />
        <button @click=${this.addTodo}>Add</button>
      </div>
      <ul>
        ${this.todos.map((todo) => html'
          <li class=${todo.completed ? 'completed' : ''}>
            <input
              type="checkbox"
              .checked=${todo.completed}
              @change=${() => this.toggleTodo(todo.id)}
            />
            <span>${todo.text}</span>
            <button @click=${() => this.removeTodo(todo.id)}>×</button>
          </li>
        ')}
      </ul>
      <p>${this.todos.filter(t => !t.completed).length} items remaining</p>
    ';
  }

  addTodo() {
    if (!this.newTodo.trim()) return;
    this.todos = [...this.todos, {
      id: Date.now(),
      text: this.newTodo.trim(),
      completed: false,
    }];
    this.newTodo = '';
  }

  toggleTodo(id) {
    this.todos = this.todos.map(t =>
      t.id === id ? { ...t, completed: !t.completed } : t
    );
  }

  removeTodo(id) {
    this.todos = this.todos.filter(t => t.id !== id);
  }
}
` + "```" + ``,
					CodeExamples: `// Micro-frontend architecture

// 1. Module Federation approach (webpack 5)
// Shell app loads remote micro-frontends at runtime

// shell/src/App.jsx
const DashboardApp = React.lazy(() => import('dashboard/DashboardApp'));
const SettingsApp = React.lazy(() => import('settings/SettingsApp'));

function Shell() {
  return (
    <div>
      <header>
        <nav>
          <Link to="/">Home</Link>
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/settings">Settings</Link>
        </nav>
      </header>
      
      <ErrorBoundary fallback={<div>Failed to load module</div>}>
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard/*" element={<DashboardApp />} />
            <Route path="/settings/*" element={<SettingsApp />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </div>
  );
}

// 2. Web Components as micro-frontend containers
// Framework-agnostic wrapper
class MicroFrontend extends HTMLElement {
  static get observedAttributes() { return ['src', 'name']; }

  connectedCallback() {
    const name = this.getAttribute('name');
    const src = this.getAttribute('src');
    
    // Create isolated container
    this.attachShadow({ mode: 'open' });
    const container = document.createElement('div');
    container.id = name + '-root';
    this.shadowRoot.appendChild(container);
    
    // Load and mount micro-frontend
    this.loadApp(src, container);
  }

  async loadApp(src, container) {
    try {
      const module = await import(/* webpackIgnore: true */ src);
      this.cleanup = module.mount(container, {
        basePath: this.getAttribute('base-path') || '/',
        onNavigate: (path) => {
          this.dispatchEvent(new CustomEvent('navigate', {
            bubbles: true,
            composed: true,
            detail: { path },
          }));
        },
      });
    } catch (err) {
      container.innerHTML = '<p>Failed to load application</p>';
      console.error('Micro-frontend load error:', err);
    }
  }

  disconnectedCallback() {
    this.cleanup?.();
  }
}

customElements.define('micro-frontend', MicroFrontend);

// Each micro-frontend exports mount/unmount
// dashboard/src/index.js
export function mount(container, props) {
  const root = ReactDOM.createRoot(container);
  root.render(<DashboardApp {...props} />);
  return () => root.unmount();
}

// 3. Event bus for micro-frontend communication
class EventBus {
  constructor() {
    this.events = new Map();
  }

  on(event, callback) {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }
    this.events.get(event).add(callback);
    return () => this.events.get(event).delete(callback);
  }

  emit(event, data) {
    this.events.get(event)?.forEach((cb) => cb(data));
  }

  once(event, callback) {
    const unsub = this.on(event, (data) => {
      callback(data);
      unsub();
    });
    return unsub;
  }
}

// Shared instance across micro-frontends
const eventBus = window.__EVENT_BUS__ = window.__EVENT_BUS__ || new EventBus();

// Dashboard emits
eventBus.emit('user:updated', { id: 1, name: 'Jane' });

// Settings listens
eventBus.on('user:updated', (user) => {
  console.log('User updated:', user);
});

// 4. Shared state with custom store
class SharedStore {
  constructor(initialState = {}) {
    this.state = initialState;
    this.subscribers = new Set();
  }

  getState() { return this.state; }

  setState(updater) {
    const newState = typeof updater === 'function'
      ? updater(this.state)
      : { ...this.state, ...updater };
    this.state = newState;
    this.subscribers.forEach((cb) => cb(this.state));
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }
}

// React hook for shared store
function useSharedStore(store, selector) {
  const [value, setValue] = useState(() => selector(store.getState()));
  
  useEffect(() => {
    return store.subscribe((state) => {
      const newValue = selector(state);
      setValue(newValue);
    });
  }, [store, selector]);
  
  return value;
}

// Usage
const sharedStore = window.__SHARED_STORE__ = 
  window.__SHARED_STORE__ || new SharedStore({ user: null, theme: 'light' });

function UserName() {
  const user = useSharedStore(sharedStore, (s) => s.user);
  return <span>{user?.name}</span>;
}`,
				},
			},
		},
		{
			ID:          1569,
			Title:       "Advanced CSS Techniques",
			Description: "Master modern CSS features including Container Queries, Cascade Layers, CSS Grid subgrid, custom properties, and advanced layout patterns.",
			Order:       69,
			Lessons: []problems.Lesson{
				{
					Title: "Container Queries and Modern CSS Layout",
					Content: `Container queries allow styling elements based on their container's size rather than the viewport, enabling truly responsive components.

**Container Queries:**
` + "```" + `css
/* Define a containment context */
.card-container {
  container-type: inline-size;
  container-name: card;
}

/* Style based on container width */
@container card (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 1rem;
  }
}

@container card (min-width: 600px) {
  .card {
    grid-template-columns: 250px 1fr auto;
  }
  .card .actions {
    flex-direction: column;
  }
}

/* Container query units */
.card-title {
  font-size: clamp(1rem, 3cqi, 1.5rem); /* cqi = 1% of container inline size */
}

/* Container style queries (experimental) */
@container card style(--variant: featured) {
  .card {
    border: 2px solid gold;
    background: #fffdf0;
  }
}

/* Container queries for components */
.sidebar {
  container-type: inline-size;
  container-name: sidebar;
}

@container sidebar (max-width: 250px) {
  .nav-item {
    padding: 0.5rem;
  }
  .nav-label {
    display: none; /* Icon only in narrow sidebar */
  }
}

@container sidebar (min-width: 251px) {
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
  }
}
` + "```" + `

**CSS Grid Subgrid:**
` + "```" + `css
/* Subgrid allows children to participate in parent's grid */
.page-layout {
  display: grid;
  grid-template-columns: [full-start] 1fr [content-start] minmax(0, 60ch) [content-end] 1fr [full-end];
  gap: 2rem;
}

.page-layout > * {
  grid-column: content;
}

/* Full-bleed elements */
.page-layout > .full-bleed {
  grid-column: full;
}

/* Card grid with aligned content */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.card {
  display: grid;
  grid-template-rows: auto 1fr auto; /* Header, body, footer */
  /* With subgrid, cards in same row align their internal rows */
  grid-row: span 3;
  grid-template-rows: subgrid;
  gap: 0;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
}

.card-header {
  padding: 1rem;
  border-bottom: 1px solid #f3f4f6;
}

.card-body {
  padding: 1rem;
}

.card-footer {
  padding: 1rem;
  border-top: 1px solid #f3f4f6;
  margin-top: auto;
}

/* Named grid lines with subgrid */
.form {
  display: grid;
  grid-template-columns: [label] auto [input] 1fr [end];
  gap: 1rem;
  align-items: center;
}

.form-row {
  display: grid;
  grid-column: label / end;
  grid-template-columns: subgrid;
}
` + "```" + `

**Cascade Layers:**
` + "```" + `css
/* Define layer order (lowest to highest priority) */
@layer reset, base, components, utilities;

/* Reset layer */
@layer reset {
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  img, picture, video, canvas, svg {
    display: block;
    max-width: 100%;
  }
}

/* Base layer */
@layer base {
  body {
    font-family: var(--font-sans);
    color: var(--color-text-primary);
    background: var(--color-bg-primary);
    line-height: 1.5;
  }
  
  h1, h2, h3, h4, h5, h6 {
    line-height: 1.2;
    font-weight: 600;
  }
  
  a {
    color: var(--color-interactive-primary);
    text-decoration: none;
  }
}

/* Components layer */
@layer components {
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: 500;
    transition: all 0.15s ease;
  }
  
  .btn-primary {
    background: var(--color-interactive-primary);
    color: white;
  }
  
  .card {
    background: var(--color-bg-primary);
    border: 1px solid var(--color-border-primary);
    border-radius: 0.75rem;
    padding: 1.5rem;
  }
}

/* Utilities layer (highest priority) */
@layer utilities {
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
  
  .text-center { text-align: center; }
  .hidden { display: none; }
  .flex { display: flex; }
  .grid { display: grid; }
}

/* Import third-party CSS into a specific layer */
@import url("normalize.css") layer(reset);
@import url("component-lib.css") layer(components);
` + "```" + `

**Advanced Custom Properties:**
` + "```" + `css
/* Dynamic spacing with custom properties */
:root {
  --space-unit: 0.25rem;
  --space-1: calc(var(--space-unit) * 1);
  --space-2: calc(var(--space-unit) * 2);
  --space-4: calc(var(--space-unit) * 4);
  --space-8: calc(var(--space-unit) * 8);
}

/* Component-scoped defaults */
.button {
  --btn-padding-x: var(--space-4);
  --btn-padding-y: var(--space-2);
  --btn-bg: var(--color-interactive-primary);
  --btn-color: white;
  --btn-radius: 0.375rem;
  
  padding: var(--btn-padding-y) var(--btn-padding-x);
  background: var(--btn-bg);
  color: var(--btn-color);
  border-radius: var(--btn-radius);
}

.button--sm {
  --btn-padding-x: var(--space-2);
  --btn-padding-y: var(--space-1);
}

.button--lg {
  --btn-padding-x: var(--space-8);
  --btn-padding-y: var(--space-4);
}

/* Scroll-driven animations */
@keyframes reveal {
  from { opacity: 0; transform: translateY(30px); }
  to { opacity: 1; transform: translateY(0); }
}

.scroll-reveal {
  animation: reveal linear;
  animation-timeline: view();
  animation-range: entry 0% entry 100%;
}

/* Scroll progress indicator */
.progress-bar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: var(--color-interactive-primary);
  transform-origin: left;
  animation: grow-width linear;
  animation-timeline: scroll();
}

@keyframes grow-width {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

/* CSS nesting (native) */
.card {
  background: white;
  border-radius: 0.75rem;
  
  & .header {
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
    
    & h2 {
      font-size: 1.25rem;
      font-weight: 600;
    }
  }
  
  & .body {
    padding: 1rem;
  }
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  @media (prefers-color-scheme: dark) {
    & {
      background: #1f2937;
    }
  }
}

/* :has() selector (parent selector) */
.form-group:has(input:invalid) {
  .label { color: red; }
  .help-text { display: none; }
  .error-text { display: block; }
}

/* Style parent based on child state */
.card:has(img) {
  grid-template-rows: auto 1fr;
}

.card:has(.badge) {
  padding-top: 2rem;
}

/* Select previous sibling */
label:has(+ input:focus) {
  color: var(--color-interactive-primary);
  font-weight: 600;
}
` + "```" + ``,
					CodeExamples: `// CSS-in-JS modern patterns

// 1. Vanilla Extract (zero-runtime CSS-in-JS)
// styles.css.ts
import { style, globalStyle, createTheme, createVar } from '@vanilla-extract/css';

const accentColor = createVar();

export const [themeClass, vars] = createTheme({
  color: {
    brand: '#3b82f6',
    text: '#111827',
    background: '#ffffff',
  },
  space: {
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
  },
});

export const container = style({
  maxWidth: 1200,
  margin: '0 auto',
  padding: vars.space.md,
});

export const card = style({
  background: vars.color.background,
  borderRadius: '0.75rem',
  padding: vars.space.lg,
  border: '1px solid #e5e7eb',
  
  ':hover': {
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  },
  
  selectors: {
    '&:first-child': { marginTop: 0 },
    ['.dark &']: { background: '#1f2937' },
  },
  
  '@media': {
    '(max-width: 768px)': { padding: vars.space.sm },
  },
  
  '@container sidebar (max-width: 250px)': {
    padding: vars.space.sm,
  },
});

// 2. CSS Modules with TypeScript
// Button.module.css
// .button { ... }
// .primary { ... }
// .large { ... }

// Button.module.css.d.ts (generated)
// declare const styles: {
//   button: string;
//   primary: string;
//   large: string;
// };
// export default styles;

// 3. Tailwind CSS patterns
// tailwind.config.ts
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { transform: 'translateY(10px)', opacity: '0' }, to: { transform: 'translateY(0)', opacity: '1' } },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
};

// Custom Tailwind plugin
const plugin = require('tailwindcss/plugin');

module.exports = plugin(function({ addUtilities, addComponents, theme }) {
  addUtilities({
    '.text-balance': { 'text-wrap': 'balance' },
    '.text-pretty': { 'text-wrap': 'pretty' },
  });
  
  addComponents({
    '.btn': {
      padding: theme('spacing.2') + ' ' + theme('spacing.4'),
      borderRadius: theme('borderRadius.md'),
      fontWeight: theme('fontWeight.medium'),
    },
  });
});`,
				},
			},
		},
	})
}
