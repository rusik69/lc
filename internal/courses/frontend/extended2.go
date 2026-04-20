package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1550,
			Title:       "Advanced React Patterns",
			Description: "Master advanced React patterns including compound components, render props, higher-order components, custom hooks, and state machines for building scalable UIs.",
			Order:       50,
			Lessons: []problems.Lesson{
				{
					Title: "Compound Components and Render Props",
					Content: `Advanced React patterns solve common UI composition problems by providing flexible, reusable component APIs.

**Compound Components:**
` + "```" + `javascript
// Compound components share implicit state through context
import { createContext, useContext, useState } from 'react';

// 1. Create shared context
const AccordionContext = createContext(null);

function Accordion({ children, allowMultiple = false }) {
  const [openItems, setOpenItems] = useState(new Set());

  const toggle = (id) => {
    setOpenItems(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (!allowMultiple) next.clear();
        next.add(id);
      }
      return next;
    });
  };

  return (
    <AccordionContext.Provider value={{ openItems, toggle }}>
      <div className="accordion">{children}</div>
    </AccordionContext.Provider>
  );
}

function AccordionItem({ id, children }) {
  const { openItems } = useContext(AccordionContext);
  const isOpen = openItems.has(id);

  return (
    <div className={` + "`" + `accordion-item ${isOpen ? 'open' : ''}` + "`" + `}>
      {children}
    </div>
  );
}

function AccordionTrigger({ id, children }) {
  const { toggle } = useContext(AccordionContext);
  return (
    <button onClick={() => toggle(id)} className="accordion-trigger">
      {children}
    </button>
  );
}

function AccordionContent({ id, children }) {
  const { openItems } = useContext(AccordionContext);
  if (!openItems.has(id)) return null;
  return <div className="accordion-content">{children}</div>;
}

// Attach sub-components
Accordion.Item = AccordionItem;
Accordion.Trigger = AccordionTrigger;
Accordion.Content = AccordionContent;

// Usage - clean, declarative API
function App() {
  return (
    <Accordion allowMultiple>
      <Accordion.Item id="1">
        <Accordion.Trigger id="1">Section 1</Accordion.Trigger>
        <Accordion.Content id="1">Content 1</Accordion.Content>
      </Accordion.Item>
      <Accordion.Item id="2">
        <Accordion.Trigger id="2">Section 2</Accordion.Trigger>
        <Accordion.Content id="2">Content 2</Accordion.Content>
      </Accordion.Item>
    </Accordion>
  );
}

// 2. Flexible Select compound component
const SelectContext = createContext(null);

function Select({ value, onChange, children }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <SelectContext.Provider value={{ value, onChange, isOpen, setIsOpen }}>
      <div className="select-container">{children}</div>
    </SelectContext.Provider>
  );
}

function SelectTrigger({ children, placeholder }) {
  const { value, isOpen, setIsOpen } = useContext(SelectContext);
  return (
    <button
      onClick={() => setIsOpen(!isOpen)}
      className="select-trigger"
      aria-expanded={isOpen}
    >
      {value || placeholder || 'Select...'}
    </button>
  );
}

function SelectOptions({ children }) {
  const { isOpen } = useContext(SelectContext);
  if (!isOpen) return null;
  return (
    <ul role="listbox" className="select-options">{children}</ul>
  );
}

function SelectOption({ value, children }) {
  const ctx = useContext(SelectContext);
  const isSelected = ctx.value === value;
  return (
    <li
      role="option"
      aria-selected={isSelected}
      onClick={() => { ctx.onChange(value); ctx.setIsOpen(false); }}
      className={` + "`" + `select-option ${isSelected ? 'selected' : ''}` + "`" + `}
    >
      {children}
    </li>
  );
}

Select.Trigger = SelectTrigger;
Select.Options = SelectOptions;
Select.Option = SelectOption;
` + "```" + `

**Render Props Pattern:**
` + "```" + `javascript
// Render props - share logic via a function prop
function MouseTracker({ render }) {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div onMouseMove={handleMouseMove} style={{ height: '100vh' }}>
      {render(position)}
    </div>
  );
}

// Usage
<MouseTracker
  render={({ x, y }) => (
    <div>Mouse: ({x}, {y})</div>
  )}
/>

// Children as render prop (more common)
function DataFetcher({ url, children }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(res.statusText);
        return res.json();
      })
      .then(data => { if (!cancelled) { setData(data); setLoading(false); } })
      .catch(err => { if (!cancelled) { setError(err); setLoading(false); } });
    
    return () => { cancelled = true; };
  }, [url]);

  return children({ data, loading, error });
}

// Usage
<DataFetcher url="/api/users">
  {({ data, loading, error }) => {
    if (loading) return <Spinner />;
    if (error) return <Error message={error.message} />;
    return <UserList users={data} />;
  }}
</DataFetcher>

// Controlled component pattern with render props
function Toggle({ initialOn = false, children }) {
  const [on, setOn] = useState(initialOn);
  const toggle = () => setOn(prev => !prev);
  const reset = () => setOn(initialOn);

  return children({ on, toggle, reset });
}

<Toggle>
  {({ on, toggle, reset }) => (
    <div>
      <button onClick={toggle}>{on ? 'ON' : 'OFF'}</button>
      <button onClick={reset}>Reset</button>
    </div>
  )}
</Toggle>
` + "```" + ``,
					CodeExamples: `// Advanced React pattern examples

// 1. Higher-Order Component (HOC)
function withAuth(WrappedComponent) {
  return function AuthenticatedComponent(props) {
    const { user, loading } = useAuth();
    
    if (loading) return <LoadingSpinner />;
    if (!user) return <Redirect to="/login" />;
    
    return <WrappedComponent {...props} user={user} />;
  };
}

const ProtectedDashboard = withAuth(Dashboard);

// 2. Custom hook replacing HOC
function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = authService.onAuthStateChanged((user) => {
      setUser(user);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const login = async (email, password) => {
    return authService.signIn(email, password);
  };

  const logout = async () => {
    return authService.signOut();
  };

  return { user, loading, login, logout };
}

// 3. Polymorphic component with "as" prop
function Button({ as: Component = 'button', children, ...props }) {
  return <Component {...props}>{children}</Component>;
}

// Usage
<Button onClick={handleClick}>Click me</Button>
<Button as="a" href="/about">About</Button>
<Button as={Link} to="/dashboard">Dashboard</Button>

// 4. Slot pattern
function Card({ children }) {
  const header = Children.toArray(children).find(
    child => child.type === CardHeader
  );
  const body = Children.toArray(children).find(
    child => child.type === CardBody
  );
  const footer = Children.toArray(children).find(
    child => child.type === CardFooter
  );

  return (
    <div className="card">
      {header && <div className="card-header">{header}</div>}
      <div className="card-body">{body}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

function CardHeader({ children }) { return children; }
function CardBody({ children }) { return children; }
function CardFooter({ children }) { return children; }

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;`,
				},
				{
					Title: "Custom Hooks and State Machines",
					Content: `Custom hooks extract reusable stateful logic. State machines provide predictable complex state management.

**Advanced Custom Hooks:**
` + "```" + `javascript
// 1. useLocalStorage - persist state
function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value) => {
    const valueToStore = value instanceof Function ? value(storedValue) : value;
    setStoredValue(valueToStore);
    window.localStorage.setItem(key, JSON.stringify(valueToStore));
  };

  return [storedValue, setValue];
}

// 2. useDebounce
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Usage
function SearchInput() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery) searchAPI(debouncedQuery);
  }, [debouncedQuery]);

  return <input value={query} onChange={e => setQuery(e.target.value)} />;
}

// 3. useMediaQuery
function useMediaQuery(query) {
  const [matches, setMatches] = useState(
    () => window.matchMedia(query).matches
  );

  useEffect(() => {
    const mql = window.matchMedia(query);
    const handler = (e) => setMatches(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, [query]);

  return matches;
}

// 4. useIntersectionObserver
function useIntersectionObserver(ref, options = {}) {
  const [entry, setEntry] = useState(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => setEntry(entry),
      { threshold: 0.1, ...options }
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, options.threshold, options.rootMargin]);

  return entry;
}

// Usage - lazy loading
function LazyImage({ src, alt }) {
  const ref = useRef();
  const entry = useIntersectionObserver(ref);
  const isVisible = entry?.isIntersecting;

  return (
    <img
      ref={ref}
      src={isVisible ? src : undefined}
      alt={alt}
      loading="lazy"
    />
  );
}

// 5. usePrevious
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => { ref.current = value; });
  return ref.current;
}

// 6. useOnClickOutside
function useOnClickOutside(ref, handler) {
  useEffect(() => {
    const listener = (event) => {
      if (!ref.current || ref.current.contains(event.target)) return;
      handler(event);
    };
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);
    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}
` + "```" + `

**State Machines with useReducer:**
` + "```" + `javascript
// Finite state machine for async operations
const fetchMachine = {
  idle: { FETCH: 'loading' },
  loading: { RESOLVE: 'success', REJECT: 'error' },
  success: { FETCH: 'loading', RESET: 'idle' },
  error: { FETCH: 'loading', RESET: 'idle' },
};

function fetchReducer(state, event) {
  const nextState = fetchMachine[state.status]?.[event.type];
  if (!nextState) return state;

  switch (nextState) {
    case 'loading':
      return { status: 'loading', data: null, error: null };
    case 'success':
      return { status: 'success', data: event.data, error: null };
    case 'error':
      return { status: 'error', data: null, error: event.error };
    case 'idle':
      return { status: 'idle', data: null, error: null };
    default:
      return state;
  }
}

function useFetchMachine(fetchFn) {
  const [state, dispatch] = useReducer(fetchReducer, {
    status: 'idle', data: null, error: null,
  });

  const execute = useCallback(async (...args) => {
    dispatch({ type: 'FETCH' });
    try {
      const data = await fetchFn(...args);
      dispatch({ type: 'RESOLVE', data });
      return data;
    } catch (error) {
      dispatch({ type: 'REJECT', error });
      throw error;
    }
  }, [fetchFn]);

  const reset = () => dispatch({ type: 'RESET' });

  return { ...state, execute, reset };
}

// Form state machine
const formMachine = {
  editing: { SUBMIT: 'validating', RESET: 'pristine' },
  pristine: { CHANGE: 'editing' },
  validating: { VALID: 'submitting', INVALID: 'editing' },
  submitting: { SUCCESS: 'submitted', ERROR: 'editing' },
  submitted: { RESET: 'pristine', EDIT: 'editing' },
};

function formReducer(state, event) {
  const nextStatus = formMachine[state.status]?.[event.type];
  if (!nextStatus) return state;

  switch (event.type) {
    case 'CHANGE':
      return { ...state, status: nextStatus, 
        values: { ...state.values, [event.field]: event.value },
        errors: { ...state.errors, [event.field]: null } };
    case 'INVALID':
      return { ...state, status: nextStatus, errors: event.errors };
    case 'SUCCESS':
      return { ...state, status: nextStatus, result: event.result };
    case 'ERROR':
      return { ...state, status: nextStatus, submitError: event.error };
    case 'RESET':
      return { status: 'pristine', values: {}, errors: {}, result: null };
    default:
      return { ...state, status: nextStatus };
  }
}
` + "```" + ``,
					CodeExamples: `// Custom hook and state machine examples

// 1. useForm hook with validation
function useForm({ initialValues, validate, onSubmit }) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setValues(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleBlur = (e) => {
    const { name } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    if (validate) {
      const fieldErrors = validate(values);
      setErrors(fieldErrors);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const allTouched = Object.keys(values).reduce(
      (acc, key) => ({ ...acc, [key]: true }), {}
    );
    setTouched(allTouched);
    
    if (validate) {
      const fieldErrors = validate(values);
      setErrors(fieldErrors);
      if (Object.keys(fieldErrors).length > 0) return;
    }
    
    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getFieldProps = (name) => ({
    name,
    value: values[name] || '',
    onChange: handleChange,
    onBlur: handleBlur,
  });

  return {
    values, errors, touched, isSubmitting,
    handleChange, handleBlur, handleSubmit,
    getFieldProps, setValues, setErrors,
  };
}

// 2. useAsync hook
function useAsync(asyncFn, immediate = true) {
  const [state, setState] = useState({
    status: 'idle', data: null, error: null,
  });

  const execute = useCallback(async (...args) => {
    setState({ status: 'pending', data: null, error: null });
    try {
      const data = await asyncFn(...args);
      setState({ status: 'success', data, error: null });
      return data;
    } catch (error) {
      setState({ status: 'error', data: null, error });
      throw error;
    }
  }, [asyncFn]);

  useEffect(() => {
    if (immediate) execute();
  }, [execute, immediate]);

  return { ...state, execute };
}

// 3. useEventSource hook for SSE
function useEventSource(url) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const source = new EventSource(url);
    source.onmessage = (e) => setData(JSON.parse(e.data));
    source.onerror = (e) => setError(e);
    return () => source.close();
  }, [url]);

  return { data, error };
}`,
				},
			},
		},
		{
			ID:          1551,
			Title:       "Modern CSS Architecture",
			Description: "Master modern CSS techniques including CSS Grid advanced layouts, container queries, CSS custom properties systems, and modern CSS architecture patterns.",
			Order:       51,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced CSS Grid and Container Queries",
					Content: `Modern CSS provides powerful layout and responsive design capabilities without JavaScript.

**Advanced CSS Grid:**
` + "```" + `css
/* Subgrid - inherit parent grid tracks */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
}

.card {
  display: grid;
  grid-template-rows: subgrid;
  grid-row: span 3; /* header, content, footer */
}

/* Named grid areas for complex layouts */
.dashboard {
  display: grid;
  grid-template-columns: 250px 1fr 300px;
  grid-template-rows: 60px 1fr 40px;
  grid-template-areas:
    "header  header  header"
    "sidebar content aside"
    "footer  footer  footer";
  min-height: 100vh;
}

.header  { grid-area: header; }
.sidebar { grid-area: sidebar; }
.content { grid-area: content; }
.aside   { grid-area: aside; }
.footer  { grid-area: footer; }

@media (max-width: 768px) {
  .dashboard {
    grid-template-columns: 1fr;
    grid-template-rows: auto;
    grid-template-areas:
      "header"
      "content"
      "sidebar"
      "aside"
      "footer";
  }
}

/* Auto-placement with dense packing */
.masonry {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  grid-auto-rows: 20px;
  grid-auto-flow: dense;
  gap: 1rem;
}

.masonry-item.tall { grid-row: span 10; }
.masonry-item.medium { grid-row: span 6; }
.masonry-item.short { grid-row: span 3; }
.masonry-item.wide { grid-column: span 2; }

/* Responsive grid without media queries */
.auto-grid {
  display: grid;
  grid-template-columns: repeat(
    auto-fit,
    minmax(min(100%, 300px), 1fr)
  );
  gap: clamp(1rem, 3vw, 2rem);
}
` + "```" + `

**Container Queries:**
` + "```" + `css
/* Container queries - respond to parent size, not viewport */
.card-container {
  container-type: inline-size;
  container-name: card;
}

/* Respond to container width */
@container card (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 1rem;
  }
}

@container card (min-width: 600px) {
  .card {
    grid-template-columns: 250px 1fr 150px;
  }
  .card-actions {
    flex-direction: column;
  }
}

/* Container query units */
.card-title {
  font-size: clamp(1rem, 3cqi, 1.5rem); /* cqi = container query inline */
}

/* Style queries (check custom property values) */
@container style(--theme: dark) {
  .card {
    background: #1a1a1a;
    color: #e0e0e0;
  }
}

/* Nested containers */
.sidebar {
  container-type: inline-size;
  container-name: sidebar;
}

@container sidebar (max-width: 200px) {
  .nav-item span { display: none; } /* Icon only */
}

@container sidebar (min-width: 201px) {
  .nav-item { display: flex; gap: 0.5rem; } /* Icon + text */
}

/* Modern responsive patterns */
/* Fluid typography */
:root {
  --fluid-min-width: 320;
  --fluid-max-width: 1200;
  --fluid-min-size: 16;
  --fluid-max-size: 20;
  
  font-size: clamp(
    calc(var(--fluid-min-size) / 16 * 1rem),
    calc(var(--fluid-min-size) / 16 * 1rem +
      (var(--fluid-max-size) - var(--fluid-min-size)) *
      (100vw - var(--fluid-min-width) * 1px) /
      (var(--fluid-max-width) - var(--fluid-min-width))),
    calc(var(--fluid-max-size) / 16 * 1rem)
  );
}

/* Modern spacing with clamp */
.section {
  padding: clamp(1rem, 5vw, 3rem);
  margin-bottom: clamp(2rem, 8vw, 6rem);
}

/* Logical properties */
.card {
  margin-inline: auto;
  padding-block: 1rem;
  padding-inline: 1.5rem;
  border-inline-start: 3px solid var(--accent);
}
` + "```" + ``,
					CodeExamples: `/* Modern CSS architecture examples */

/* 1. CSS Custom Properties Design System */
:root {
  /* Colors */
  --color-primary-50: oklch(0.97 0.01 250);
  --color-primary-100: oklch(0.93 0.03 250);
  --color-primary-500: oklch(0.55 0.15 250);
  --color-primary-600: oklch(0.48 0.15 250);
  --color-primary-900: oklch(0.25 0.08 250);
  
  /* Spacing scale */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;
  --space-16: 4rem;
  
  /* Typography */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  
  --text-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.8rem);
  --text-sm: clamp(0.875rem, 0.8rem + 0.3vw, 0.95rem);
  --text-base: clamp(1rem, 0.9rem + 0.4vw, 1.125rem);
  --text-lg: clamp(1.125rem, 1rem + 0.5vw, 1.25rem);
  --text-xl: clamp(1.25rem, 1.1rem + 0.6vw, 1.5rem);
  --text-2xl: clamp(1.5rem, 1.2rem + 1vw, 2rem);
  --text-3xl: clamp(1.875rem, 1.4rem + 1.5vw, 2.5rem);
  
  /* Shadows */
  --shadow-sm: 0 1px 2px oklch(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px oklch(0 0 0 / 0.07), 0 2px 4px oklch(0 0 0 / 0.06);
  --shadow-lg: 0 10px 15px oklch(0 0 0 / 0.1), 0 4px 6px oklch(0 0 0 / 0.05);
  
  /* Radii */
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 1rem;
  --radius-full: 9999px;
  
  /* Transitions */
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --duration-slow: 400ms;
}

/* 2. Dark mode with custom properties */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: oklch(0.15 0.01 250);
    --bg-secondary: oklch(0.2 0.01 250);
    --text-primary: oklch(0.95 0 0);
    --text-secondary: oklch(0.75 0 0);
    --border-color: oklch(0.3 0.01 250);
  }
}

[data-theme="light"] {
  --bg-primary: oklch(1 0 0);
  --bg-secondary: oklch(0.97 0 0);
  --text-primary: oklch(0.15 0 0);
  --text-secondary: oklch(0.4 0 0);
  --border-color: oklch(0.88 0 0);
}

/* 3. View Transitions API */
@view-transition {
  navigation: auto;
}

.card-image {
  view-transition-name: card-image;
}

::view-transition-old(card-image) {
  animation: 300ms ease-out fade-out;
}

::view-transition-new(card-image) {
  animation: 300ms ease-in fade-in;
}`,
				},
			},
		},
	})
}
