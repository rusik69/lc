package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          210,
			Title:       "Advanced CSS",
			Description: "Master advanced CSS: animations, transitions, custom properties, and modern techniques.",
			Order:       10,
			Lessons: []problems.Lesson{
				{
					Title: "CSS Animations and Transitions",
					Content: `CSS animations and transitions create smooth, engaging user experiences.

**Transitions:**
- Smooth property changes
- transition-property: Which properties to animate
- transition-duration: How long
- transition-timing-function: Easing curve
- transition-delay: When to start

**Animations:**
- @keyframes: Define animation steps
- animation-name: Keyframe name
- animation-duration: How long
- animation-timing-function: Easing
- animation-delay: Start delay
- animation-iteration-count: How many times
- animation-direction: Normal/reverse/alternate
- animation-fill-mode: Before/after states
- animation-play-state: Running/paused

**Performance:**
- Use transform and opacity (GPU accelerated)
- Avoid animating layout properties
- Use will-change for optimization`,
					CodeExamples: `/* Transitions */
.button {
    background-color: blue;
    transition: background-color 0.3s ease;
}

.button:hover {
    background-color: red;
}

/* Multiple properties */
.card {
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Animations */
@keyframes slideIn {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}

/* Complex animation */
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
}

.bounce {
    animation: bounce 1s infinite;
}

/* Performance optimization */
.optimized {
    will-change: transform;
    transform: translateZ(0); /* Force GPU */
}`,
				},
				{
					Title: "CSS Custom Properties (Variables)",
					Content: `CSS custom properties (CSS variables) enable dynamic theming and reusable values.

**Custom Properties:**
- Declared with -- prefix
- Scoped to selector
- Inherited by children
- Can be changed with JavaScript
- Fallback values supported

**Use Cases:**
- Theming (light/dark mode)
- Responsive design
- Component styling
- Dynamic values

**Best Practices:**
- Use meaningful names
- Define in :root for global
- Use fallbacks
- Consider browser support`,
					CodeExamples: `/* Define variables */
:root {
    --primary-color: #3b82f6;
    --secondary-color: #8b5cf6;
    --font-size-base: 16px;
    --spacing-unit: 8px;
    --border-radius: 4px;
}

/* Use variables */
.button {
    background-color: var(--primary-color);
    font-size: var(--font-size-base);
    padding: calc(var(--spacing-unit) * 2);
    border-radius: var(--border-radius);
}

/* Fallback values */
.text {
    color: var(--text-color, #333);
}

/* Scoped variables */
.card {
    --card-padding: 20px;
    padding: var(--card-padding);
}

/* Theming */
[data-theme="dark"] {
    --bg-color: #1a1a1a;
    --text-color: #fff;
}

[data-theme="light"] {
    --bg-color: #fff;
    --text-color: #000;
}

/* JavaScript manipulation */
document.documentElement.style.setProperty('--primary-color', '#ff0000');`,
				},
				{
					Title: "CSS Preprocessors (SASS/SCSS)",
					Content: `CSS preprocessors extend CSS with variables, nesting, mixins, and more.

**SASS/SCSS:**
- Syntactically Awesome Style Sheets
- SCSS: Sassy CSS (CSS-like syntax)
- Variables: Reusable values
- Nesting: Nested selectors
- Mixins: Reusable code blocks
- Functions: Custom functions
- Partials: Modular CSS files
- Imports: Combine files

**Key Features:**
- Variables: $variable-name
- Nesting: Nested selectors
- Mixins: @mixin and @include
- Functions: @function
- Operators: Math operations
- Control flow: @if, @for, @each

**Compilation:**
- SASS compiles to CSS
- Use build tools (Webpack, Vite)
- Watch mode for development
- Production minification`,
					CodeExamples: `// Variables
$primary-color: #3b82f6;
$font-size-base: 16px;
$spacing-unit: 8px;

.button {
    background-color: $primary-color;
    font-size: $font-size-base;
    padding: $spacing-unit * 2;
}

// Nesting
.nav {
    ul {
        margin: 0;
        padding: 0;
        list-style: none;
    }
    
    li {
        display: inline-block;
    }
    
    a {
        display: block;
        padding: 6px 12px;
        text-decoration: none;
        
        &:hover {
            background-color: $primary-color;
        }
    }
}

// Mixins
@mixin flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

@mixin button-variant($bg-color, $text-color) {
    background-color: $bg-color;
    color: $text-color;
    
    &:hover {
        background-color: darken($bg-color, 10%);
    }
}

.card {
    @include flex-center;
}

.btn-primary {
    @include button-variant($primary-color, white);
}

// Functions
@function calculate-rem($pixels) {
    @return $pixels / $font-size-base * 1rem;
}

.text {
    font-size: calculate-rem(24px); // 1.5rem
}

// Control flow
@for $i from 1 through 5 {
    .col-#{$i} {
        width: percentage($i / 5);
    }
}

// Each loop
$colors: red, blue, green;
@each $color in $colors {
    .text-#{$color} {
        color: $color;
    }
}

// Conditionals
@mixin theme($theme) {
    @if $theme == dark {
        background-color: black;
        color: white;
    } @else {
        background-color: white;
        color: black;
    }
}`,
				},
				{
					Title: "CSS-in-JS Libraries",
					Content: `CSS-in-JS allows writing CSS directly in JavaScript components.

**CSS-in-JS Benefits:**
- Scoped styles (no conflicts)
- Dynamic styles based on props
- Better component encapsulation
- TypeScript support
- Dead code elimination

**Popular Libraries:**
- styled-components: Most popular
- Emotion: High performance
- Styled-jsx: Next.js default
- JSS: JavaScript Style Sheets
- Linaria: Zero-runtime

**Key Concepts:**
- Tagged template literals
- Styled components
- Theme providers
- Dynamic props
- Animations

**When to Use:**
- Component-based architecture
- Need dynamic styles
- Want scoped styles
- Using React/Vue`,
					CodeExamples: `// styled-components
import styled from 'styled-components';

const Button = styled.button\`
    background-color: \${props => props.primary ? 'blue' : 'gray'};
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    
    &:hover {
        opacity: 0.8;
    }
\`;

// Usage
<Button primary>Primary Button</Button>
<Button>Secondary Button</Button>

// With theme
const ThemedButton = styled.button\`
    background-color: \${props => props.theme.primary};
    color: \${props => props.theme.text};
\`;

// Theme provider
const theme = {
    primary: '#3b82f6',
    text: '#333'
};

<ThemeProvider theme={theme}>
    <ThemedButton>Button</ThemedButton>
</ThemeProvider>

// Emotion
import { css } from '@emotion/react';

const buttonStyle = css\`
    background-color: blue;
    color: white;
    padding: 10px 20px;
\`;

<button css={buttonStyle}>Button</button>

// Styled-jsx (Next.js)
<style jsx>{\`
    .button {
        background-color: blue;
        color: white;
    }
\`}</style>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          211,
			Title:       "Advanced JavaScript",
			Description: "Deep dive into JavaScript: closures, prototypes, async/await, and advanced patterns.",
			Order:       11,
			Lessons: []problems.Lesson{
				{
					Title: "Closures and Scope",
					Content: `Closures are functions that have access to variables in their outer scope.

**Closures:**
- Function remembers its lexical environment
- Access outer variables even after outer function returns
- Common in callbacks and event handlers
- Enable data privacy and function factories

**Scope Chain:**
- Local scope
- Outer function scope
- Global scope
- Variable lookup follows chain

**Use Cases:**
- Data privacy
- Function factories
- Event handlers
- Module pattern
- Memoization`,
					CodeExamples: `// Basic closure
function outer() {
    const outerVar = "I'm outside";
    
    function inner() {
        console.log(outerVar); // Access outer variable
    }
    
    return inner;
}

const innerFunc = outer();
innerFunc(); // "I'm outside"

// Data privacy
function createCounter() {
    let count = 0;
    
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count
    };
}

const counter = createCounter();
counter.increment(); // 1
counter.increment(); // 2

// Function factory
function createMultiplier(multiplier) {
    return function(number) {
        return number * multiplier;
    };
}

const double = createMultiplier(2);
const triple = createMultiplier(3);

double(5); // 10
triple(5); // 15

// Module pattern
const module = (function() {
    let privateVar = 0;
    
    return {
        getPrivate: () => privateVar,
        setPrivate: (val) => { privateVar = val; }
    };
})();`,
				},
				{
					Title: "Prototypes and Inheritance",
					Content: `JavaScript uses prototypal inheritance - objects inherit from other objects.

**Prototypes:**
- Every object has a prototype
- Prototype chain for property lookup
- Object.prototype is root
- __proto__ vs prototype

**Constructor Functions:**
- Create objects with new
- prototype property
- Methods shared across instances

**Classes (ES6):**
- Syntactic sugar over prototypes
- extends for inheritance
- super for parent access
- Static methods`,
					CodeExamples: `// Constructor function
function Person(name, age) {
    this.name = name;
    this.age = age;
}

Person.prototype.greet = function() {
    return `Hello, I'm ${this.name}`;
};

const john = new Person("John", 30);
john.greet(); // "Hello, I'm John"

// ES6 Classes
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    greet() {
        return `Hello, I'm ${this.name}`;
    }
    
    static create(name) {
        return new Person(name, 0);
    }
}

class Student extends Person {
    constructor(name, age, school) {
        super(name, age);
        this.school = school;
    }
    
    study() {
        return `${this.name} is studying`;
    }
}

// Prototype chain
const obj = {};
console.log(obj.toString); // From Object.prototype`,
				},
				{
					Title: "Async/Await and Promises",
					Content: `Modern JavaScript handles asynchronous operations with Promises and async/await.

**Promises:**
- Represent eventual completion
- States: pending, fulfilled, rejected
- .then() for success
- .catch() for errors
- .finally() for cleanup

**Async/Await:**
- Syntactic sugar over Promises
- async functions return Promises
- await pauses execution
- try/catch for errors

**Promise Methods:**
- Promise.all(): All succeed
- Promise.race(): First to settle
- Promise.allSettled(): All complete
- Promise.any(): First to succeed`,
					CodeExamples: `// Promises
const promise = new Promise((resolve, reject) => {
    setTimeout(() => {
        Math.random() > 0.5 ? resolve("Success") : reject("Error");
    }, 1000);
});

promise
    .then(result => console.log(result))
    .catch(error => console.error(error))
    .finally(() => console.log("Done"));

// Async/await
async function fetchData() {
    try {
        const response = await fetch("/api/data");
        if (!response.ok) throw new Error("HTTP error");
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
        throw error;
    }
}

// Promise.all
const promises = [
    fetch("/api/users"),
    fetch("/api/posts"),
    fetch("/api/comments")
];

const [users, posts, comments] = await Promise.all(promises);

// Promise.race
const fastest = await Promise.race([
    fetch("/api/slow"),
    fetch("/api/fast")
]);`,
				},
				{
					Title: "Error Handling and Debugging",
					Content: `Effective error handling and debugging are essential for robust applications.

**Error Handling Patterns:**
- Try-catch-finally blocks
- Error boundaries (React)
- Global error handlers
- Promise error handling
- Async error handling

**Debugging Tools:**
- console methods (log, error, warn, table)
- debugger statement
- Browser DevTools
- Source maps
- Breakpoints

**Error Types:**
- SyntaxError: Code syntax issues
- ReferenceError: Undefined variables
- TypeError: Wrong type operations
- RangeError: Out of range values
- Custom errors: User-defined

**Best Practices:**
- Always handle errors
- Provide meaningful messages
- Log errors for debugging
- Use error boundaries
- Validate input`,
					CodeExamples: `// Error handling patterns
try {
    riskyOperation();
} catch (error) {
    console.error("Error:", error);
    // Handle error gracefully
} finally {
    // Cleanup code
}

// Async error handling
async function fetchData() {
    try {
        const response = await fetch("/api/data");
        if (!response.ok) throw new Error("HTTP error");
        return await response.json();
    } catch (error) {
        console.error("Fetch failed:", error);
        return null; // Fallback
    }
}

// Global error handler
window.addEventListener("error", (event) => {
    console.error("Global error:", event.error);
    // Send to error tracking service
});

// Promise error handling
fetch("/api/data")
    .then(response => response.json())
    .catch(error => {
        console.error("Error:", error);
        return defaultData;
    });

// Debugging
console.log("Debug info:", data);
console.table(arrayData);
console.group("Group");
console.log("Item 1");
console.log("Item 2");
console.groupEnd();

// Debugger statement
function complexFunction() {
    debugger; // Pause execution here
    // Code to debug
}

// Error tracking
function trackError(error, context) {
    // Send to error tracking service
    console.error("Error:", error, "Context:", context);
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          212,
			Title:       "TypeScript Fundamentals",
			Description: "Learn TypeScript: types, interfaces, generics, and type safety.",
			Order:       12,
			Lessons: []problems.Lesson{
				{
					Title: "TypeScript Basics",
					Content: `TypeScript is JavaScript with static type checking.

**What is TypeScript?**
- Superset of JavaScript
- Adds static types
- Compiles to JavaScript
- Catches errors at compile time
- Better IDE support

**Basic Types:**
- string, number, boolean
- array, tuple
- enum, any, void
- null, undefined
- never, unknown

**Type Annotations:**
- Explicit type declarations
- Type inference
- Optional types
- Union types`,
					CodeExamples: `// Basic types
let name: string = "John";
let age: number = 30;
let isActive: boolean = true;

// Arrays
let numbers: number[] = [1, 2, 3];
let names: Array<string> = ["John", "Jane"];

// Tuples
let tuple: [string, number] = ["John", 30];

// Enums
enum Color {
    Red,
    Green,
    Blue
}

let color: Color = Color.Red;

// Any (avoid when possible)
let value: any = "anything";

// Union types
let id: string | number;
id = "123";
id = 123;

// Optional
function greet(name?: string) {
    return name ? `Hello, ${name}` : "Hello";
}

// Type inference
let inferred = "Hello"; // Type: string

// Interfaces
interface User {
    name: string;
    age: number;
    email?: string;
}

const user: User = {
    name: "John",
    age: 30
};`,
				},
				{
					Title: "Advanced TypeScript",
					Content: `Advanced TypeScript features for complex applications.

**Generics:**
- Type parameters
- Reusable code
- Type constraints
- Default types

**Utility Types:**
- Partial<T>: All optional
- Required<T>: All required
- Pick<T, K>: Select properties
- Omit<T, K>: Exclude properties
- Readonly<T>: Immutable

**Type Guards:**
- typeof checks
- instanceof checks
- Custom type guards
- Discriminated unions`,
					CodeExamples: `// Generics
function identity<T>(arg: T): T {
    return arg;
}

const num = identity<number>(42);
const str = identity<string>("hello");

// Generic constraints
interface Lengthwise {
    length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
}

// Utility types
interface User {
    name: string;
    age: number;
    email: string;
}

type PartialUser = Partial<User>; // All optional
type RequiredUser = Required<User>; // All required
type UserName = Pick<User, "name">; // { name: string }
type UserWithoutEmail = Omit<User, "email">; // { name, age }

// Type guards
function isString(value: unknown): value is string {
    return typeof value === "string";
}

if (isString(value)) {
    // TypeScript knows value is string
    console.log(value.toUpperCase());
}`,
				},
				{
					Title: "TypeScript Decorators and Namespaces",
					Content: `Decorators and namespaces provide advanced TypeScript features for organizing code.

**Decorators:**
- Experimental feature
- Modify classes, methods, properties
- Used in frameworks (Angular, NestJS)
- Enable metadata and AOP patterns

**Namespaces:**
- Organize code into logical groups
- Avoid global namespace pollution
- Can be nested
- Alternative to modules

**Use Cases:**
- Framework development
- Code organization
- Metadata annotations
- Cross-file organization`,
					CodeExamples: `// Decorators (experimental)
function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args: any[]) {
        console.log(\`Calling \${propertyKey}\`);
        return originalMethod.apply(this, args);
    };
}

class Calculator {
    @log
    add(a: number, b: number): number {
        return a + b;
    }
}

// Namespaces
namespace MathUtils {
    export function add(a: number, b: number): number {
        return a + b;
    }
    
    export function multiply(a: number, b: number): number {
        return a * b;
    }
}

MathUtils.add(1, 2);

// Nested namespaces
namespace Geometry {
    export namespace Circle {
        export function area(radius: number): number {
            return Math.PI * radius * radius;
        }
    }
}

Geometry.Circle.area(5);`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          213,
			Title:       "React Basics",
			Description: "Introduction to React: components, JSX, props, state, and React fundamentals.",
			Order:       13,
			Lessons: []problems.Lesson{
				{
					Title: "React Introduction",
					Content: `React is a JavaScript library for building user interfaces.

**What is React?**
- Component-based
- Declarative
- Virtual DOM
- One-way data flow
- Ecosystem (React Router, Redux, etc.)

**Key Concepts:**
- Components: Reusable UI pieces
- JSX: JavaScript XML syntax
- Props: Component inputs
- State: Component data
- Virtual DOM: Efficient updates

**React Setup:**
- Create React App
- Vite
- Next.js
- Manual setup`,
					CodeExamples: `// Functional component
import React from 'react';

function Welcome(props) {
    return <h1>Hello, {props.name}</h1>;
}

// JSX
const element = <h1>Hello, World!</h1>;

// JSX with expressions
const name = "John";
const element = <h1>Hello, {name}</h1>;

// JSX attributes
const element = <img src="image.jpg" alt="Photo" />;

// Component with props
function Button({ label, onClick }) {
    return <button onClick={onClick}>{label}</button>;
}

// Using component
<Button label="Click me" onClick={() => alert("Clicked")} />

// State with hooks
import { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    
    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
}`,
				},
				{
					Title: "React Components and Props",
					Content: `Components are the building blocks of React applications.

**Component Types:**
- Functional components (modern)
- Class components (legacy)
- Higher-order components
- Render props

**Props:**
- Pass data to components
- Read-only
- Type checking with PropTypes/TypeScript
- Default props
- Children prop

**Component Composition:**
- Nesting components
- Component reuse
- Props drilling
- Context API for deep nesting`,
					CodeExamples: `// Functional component
function Card({ title, content, children }) {
    return (
        <div className="card">
            <h2>{title}</h2>
            <p>{content}</p>
            {children}
        </div>
    );
}

// Using component
<Card title="Hello" content="World">
    <button>Action</button>
</Card>

// Props with default values
function Button({ label = "Click", onClick }) {
    return <button onClick={onClick}>{label}</button>;
}

// PropTypes (runtime checking)
import PropTypes from 'prop-types';

Button.propTypes = {
    label: PropTypes.string.isRequired,
    onClick: PropTypes.func
};

// TypeScript props
interface ButtonProps {
    label: string;
    onClick: () => void;
}

function Button({ label, onClick }: ButtonProps) {
    return <button onClick={onClick}>{label}</button>;
}

// Component composition
function App() {
    return (
        <Layout>
            <Header />
            <Main>
                <Sidebar />
                <Content />
            </Main>
            <Footer />
        </Layout>
    );
}`,
				},
				{
					Title: "React Router and Navigation",
					Content: `React Router enables client-side routing and navigation in React applications.

**React Router:**
- Declarative routing
- BrowserRouter: HTML5 history API
- Routes: Define route paths
- Link: Navigation links
- useNavigate: Programmatic navigation
- useParams: Access route parameters

**Key Concepts:**
- Route matching
- Nested routes
- Route parameters
- Query strings
- Protected routes
- Code splitting with routes

**Best Practices:**
- Use Link instead of <a>
- Organize routes logically
- Protect sensitive routes
- Lazy load route components`,
					CodeExamples: `// Basic routing setup
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';

function App() {
    return (
        <BrowserRouter>
            <nav>
                <Link to="/">Home</Link>
                <Link to="/about">About</Link>
                <Link to="/users">Users</Link>
            </nav>
            
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/users" element={<Users />} />
                <Route path="/users/:id" element={<UserDetail />} />
            </Routes>
        </BrowserRouter>
    );
}

// Accessing route parameters
import { useParams } from 'react-router-dom';

function UserDetail() {
    const { id } = useParams();
    return <div>User ID: {id}</div>;
}

// Programmatic navigation
import { useNavigate } from 'react-router-dom';

function LoginButton() {
    const navigate = useNavigate();
    
    const handleLogin = () => {
        // Login logic
        navigate('/dashboard');
    };
    
    return <button onClick={handleLogin}>Login</button>;
}

// Protected routes
function ProtectedRoute({ children }) {
    const { user } = useAuth();
    return user ? children : <Navigate to="/login" />;
}`,
				},
				{
					Title: "Error Boundaries and Forms in React",
					Content: `Error boundaries catch errors and forms handle user input in React.

**Error Boundaries:**
- Catch errors in component tree
- Display fallback UI
- Class components only (or libraries)
- Prevent app crashes

**Forms in React:**
- Controlled components: Value controlled by state
- Uncontrolled components: DOM handles value
- Form validation
- Form libraries (Formik, React Hook Form)

**Best Practices:**
- Use error boundaries strategically
- Validate form input
- Provide user feedback
- Handle form submission`,
					CodeExamples: `// Error Boundary
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }
    
    static getDerivedStateFromError(error) {
        return { hasError: true };
    }
    
    componentDidCatch(error, errorInfo) {
        console.error("Error:", error, errorInfo);
    }
    
    render() {
        if (this.state.hasError) {
            return <h1>Something went wrong.</h1>;
        }
        return this.props.children;
    }
}

// Controlled form
function LoginForm() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    
    const handleSubmit = (e) => {
        e.preventDefault();
        console.log({ email, password });
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
            />
            <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
            />
            <button type="submit">Login</button>
        </form>
    );
}

// Form validation
function ValidatedForm() {
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [errors, setErrors] = useState({});
    
    const validate = () => {
        const newErrors = {};
        if (!formData.email) newErrors.email = "Email required";
        if (formData.password.length < 8) {
            newErrors.password = "Password must be 8+ characters";
        }
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if (validate()) {
            // Submit form
        }
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
            />
            {errors.email && <span>{errors.email}</span>}
            {/* ... */}
        </form>
    );
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          214,
			Title:       "React Hooks",
			Description: "Master React Hooks: useState, useEffect, useContext, and custom hooks.",
			Order:       14,
			Lessons: []problems.Lesson{
				{
					Title: "Essential Hooks",
					Content: `Hooks allow functional components to use state and lifecycle features.

**useState:**
- Add state to functional components
- Returns [value, setter]
- Can initialize with function
- State updates are asynchronous

**useEffect:**
- Side effects in functional components
- Runs after render
- Dependency array controls when
- Cleanup function for unmount

**useContext:**
- Access context values
- Avoids props drilling
- Provider/Consumer pattern

**useRef:**
- Mutable value that persists
- Doesn't trigger re-renders
- Access DOM elements`,
					CodeExamples: `import { useState, useEffect, useContext, useRef } from 'react';

// useState
function Counter() {
    const [count, setCount] = useState(0);
    const [name, setName] = useState("");
    
    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>+</button>
            <input value={name} onChange={(e) => setName(e.target.value)} />
        </div>
    );
}

// useEffect
function DataFetcher({ userId }) {
    const [data, setData] = useState(null);
    
    useEffect(() => {
        fetch(`/api/users/${userId}`)
            .then(res => res.json())
            .then(setData);
    }, [userId]); // Run when userId changes
    
    return <div>{data?.name}</div>;
}

// Cleanup
useEffect(() => {
    const timer = setInterval(() => {
        console.log("Tick");
    }, 1000);
    
    return () => clearInterval(timer); // Cleanup
}, []);

// useContext
const ThemeContext = createContext("light");

function ThemedButton() {
    const theme = useContext(ThemeContext);
    return <button className={theme}>Themed</button>;
}

// useRef
function TextInput() {
    const inputRef = useRef(null);
    
    const focusInput = () => {
        inputRef.current?.focus();
    };
    
    return (
        <>
            <input ref={inputRef} />
            <button onClick={focusInput}>Focus</button>
        </>
    );
}`,
				},
				{
					Title: "Custom Hooks",
					Content: `Custom hooks extract component logic into reusable functions.

**Custom Hook Rules:**
- Start with "use"
- Can call other hooks
- Share logic between components
- Return values/functions

**Common Patterns:**
- Data fetching
- Form handling
- Local storage
- Debouncing/throttling
- Authentication`,
					CodeExamples: `// Custom hook for data fetching
function useFetch(url) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        fetch(url)
            .then(res => res.json())
            .then(setData)
            .catch(setError)
            .finally(() => setLoading(false));
    }, [url]);
    
    return { data, loading, error };
}

// Usage
function UserProfile({ userId }) {
    const { data, loading, error } = useFetch(`/api/users/${userId}`);
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error.message}</div>;
    return <div>{data.name}</div>;
}

// Custom hook for local storage
function useLocalStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            return initialValue;
        }
    });
    
    const setValue = (value) => {
        try {
            setStoredValue(value);
            window.localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error(error);
        }
    };
    
    return [storedValue, setValue];
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          215,
			Title:       "State Management",
			Description: "Manage application state: Redux, Context API, Zustand, and state management patterns.",
			Order:       15,
			Lessons: []problems.Lesson{
				{
					Title: "Context API",
					Content: `React Context provides a way to share data without props drilling.

**Context API:**
- createContext: Create context
- Provider: Supply value
- useContext: Consume value
- Multiple contexts
- Performance considerations

**When to Use:**
- Global state
- Theme
- Authentication
- Language/locale
- Avoiding props drilling

**Best Practices:**
- Split contexts by concern
- Use with useReducer for complex state
- Memoize context values
- Avoid overuse`,
					CodeExamples: `import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext();

// Provider component
function ThemeProvider({ children }) {
    const [theme, setTheme] = useState("light");
    
    const toggleTheme = () => {
        setTheme(prev => prev === "light" ? "dark" : "light");
    };
    
    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

// Custom hook
function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error("useTheme must be used within ThemeProvider");
    }
    return context;
}

// Usage
function App() {
    return (
        <ThemeProvider>
            <ThemedButton />
        </ThemeProvider>
    );
}

function ThemedButton() {
    const { theme, toggleTheme } = useTheme();
    return (
        <button onClick={toggleTheme}>
            Current theme: {theme}
        </button>
    );
}

// Multiple contexts
const AuthContext = createContext();
const UserContext = createContext();`,
				},
				{
					Title: "Redux Basics",
					Content: `Redux is a predictable state container for JavaScript apps.

**Redux Concepts:**
- Store: Single source of truth
- Actions: Describe what happened
- Reducers: Specify how state updates
- Dispatch: Send actions
- Selectors: Extract data

**Redux Toolkit:**
- Modern Redux (recommended)
- configureStore
- createSlice
- createAsyncThunk
- RTK Query`,
					CodeExamples: `// Redux Toolkit setup
import { configureStore, createSlice } from '@reduxjs/toolkit';

// Slice
const counterSlice = createSlice({
    name: 'counter',
    initialState: { value: 0 },
    reducers: {
        increment: (state) => {
            state.value += 1;
        },
        decrement: (state) => {
            state.value -= 1;
        },
        incrementByAmount: (state, action) => {
            state.value += action.payload;
        },
    },
});

// Store
const store = configureStore({
    reducer: {
        counter: counterSlice.reducer,
    },
});

// React Redux
import { useSelector, useDispatch } from 'react-redux';

function Counter() {
    const count = useSelector((state) => state.counter.value);
    const dispatch = useDispatch();
    
    return (
        <div>
            <span>{count}</span>
            <button onClick={() => dispatch(counterSlice.actions.increment())}>
                +
            </button>
        </div>
    );
}`,
				},
				{
					Title: "Modern State Management Libraries",
					Content: `Modern state management libraries provide simpler alternatives to Redux.

**Zustand:**
- Lightweight state management
- No boilerplate
- Simple API
- Good TypeScript support

**Jotai:**
- Atomic state management
- Bottom-up approach
- Fine-grained reactivity
- Great for complex state

**React Query / SWR:**
- Server state management
- Caching and synchronization
- Automatic refetching
- Optimistic updates

**When to Use:**
- Zustand: Simple global state
- Jotai: Complex atomic state
- React Query: Server data
- Redux: Large applications`,
					CodeExamples: `// Zustand
import create from 'zustand';

const useStore = create((set) => ({
    count: 0,
    increment: () => set((state) => ({ count: state.count + 1 })),
}));

function Counter() {
    const { count, increment } = useStore();
    return <button onClick={increment}>{count}</button>;
}

// React Query
import { useQuery } from 'react-query';

function Users() {
    const { data, isLoading } = useQuery('users', fetchUsers);
    if (isLoading) return <div>Loading...</div>;
    return <div>{data.map(u => <div key={u.id}>{u.name}</div>)}</div>;
}

// SWR
import useSWR from 'swr';

function Profile() {
    const { data } = useSWR('/api/user', fetcher);
    return <div>{data?.name}</div>;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          216,
			Title:       "Vue.js Basics",
			Description: "Introduction to Vue.js: components, directives, reactivity, and Vue fundamentals.",
			Order:       16,
			Lessons: []problems.Lesson{
				{
					Title: "Vue.js Introduction",
					Content: `Vue.js is a progressive JavaScript framework for building user interfaces.

**What is Vue?**
- Progressive framework
- Component-based
- Reactive data binding
- Template syntax
- Easy to learn

**Key Features:**
- Declarative rendering
- Component system
- Reactivity system
- Directives
- Single File Components

**Vue Setup:**
- CDN
- Vue CLI
- Vite
- Nuxt.js`,
					CodeExamples: `<!-- Vue 3 Composition API -->
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
const message = ref('Hello Vue!')

const doubled = computed(() => count.value * 2)

function increment() {
    count.value++
}
</script>

<template>
    <div>
        <h1>{{ message }}</h1>
        <p>Count: {{ count }}</p>
        <p>Doubled: {{ doubled }}</p>
        <button @click="increment">Increment</button>
    </div>
</template>

<!-- Options API -->
<script>
export default {
    data() {
        return {
            count: 0,
            message: 'Hello Vue!'
        }
    },
    computed: {
        doubled() {
            return this.count * 2
        }
    },
    methods: {
        increment() {
            this.count++
        }
    }
}
</script>

<template>
    <div>
        <h1>{{ message }}</h1>
        <p>Count: {{ count }}</p>
        <button @click="increment">Increment</button>
    </div>
</template>`,
				},
				{
					Title: "Vue Directives and Components",
					Content: `Vue directives are special attributes that provide reactive behavior.

**Common Directives:**
- v-if/v-else/v-show: Conditional rendering
- v-for: List rendering
- v-bind: Attribute binding
- v-on: Event handling
- v-model: Two-way binding
- v-text/v-html: Text rendering

**Components:**
- Single File Components (.vue)
- Props: Parent to child
- Events: Child to parent
- Slots: Content projection
- Provide/Inject: Dependency injection`,
					CodeExamples: `<!-- Directives -->
<div v-if="isVisible">Visible</div>
<div v-else>Hidden</div>

<ul>
    <li v-for="item in items" :key="item.id">
        {{ item.name }}
    </li>
</ul>

<img :src="imageUrl" :alt="imageAlt">
<button @click="handleClick">Click</button>

<input v-model="username" />

<!-- Component -->
<script setup>
import { defineProps, defineEmits } from 'vue'

const props = defineProps({
    title: String,
    count: {
        type: Number,
        default: 0
    }
})

const emit = defineEmits(['update', 'delete'])

function handleUpdate() {
    emit('update', props.count + 1)
}
</script>

<template>
    <div>
        <h2>{{ title }}</h2>
        <p>Count: {{ count }}</p>
        <button @click="handleUpdate">Update</button>
    </div>
</template>`,
				},
				{
					Title: "Vue Router and Pinia",
					Content: `Vue Router handles navigation and Pinia manages state in Vue applications.

**Vue Router:**
- Client-side routing
- Route configuration
- Navigation guards
- Dynamic routes
- Nested routes

**Pinia:**
- Vue's official state management
- Simpler than Vuex
- TypeScript support
- DevTools integration
- Modular stores

**Key Features:**
- Route parameters
- Query strings
- Programmatic navigation
- Route guards
- Store composition`,
					CodeExamples: `// Vue Router setup
import { createRouter, createWebHistory } from 'vue-router';

const routes = [
    { path: '/', component: Home },
    { path: '/about', component: About },
    { path: '/users/:id', component: UserDetail }
];

const router = createRouter({
    history: createWebHistory(),
    routes
});

// Using router
import { useRoute, useRouter } from 'vue-router';

export default {
    setup() {
        const route = useRoute();
        const router = useRouter();
        
        const userId = route.params.id;
        const goToAbout = () => router.push('/about');
        
        return { userId, goToAbout };
    }
};

// Pinia store
import { defineStore } from 'pinia';

export const useUserStore = defineStore('user', {
    state: () => ({
        user: null,
        isAuthenticated: false
    }),
    actions: {
        login(user) {
            this.user = user;
            this.isAuthenticated = true;
        },
        logout() {
            this.user = null;
            this.isAuthenticated = false;
        }
    },
    getters: {
        userName: (state) => state.user?.name
    }
});

// Using store
import { useUserStore } from './stores/user';

export default {
    setup() {
        const userStore = useUserStore();
        return { userStore };
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          217,
			Title:       "Component Architecture",
			Description: "Design scalable component architectures: composition, patterns, and best practices.",
			Order:       17,
			Lessons: []problems.Lesson{
				{
					Title: "Component Design Principles",
					Content: `Good component architecture makes applications maintainable and scalable.

**Principles:**
- Single Responsibility
- Composition over inheritance
- Props down, events up
- Separation of concerns
- Reusability

**Component Types:**
- Presentational: UI only
- Container: Data/logic
- Layout: Structure
- Higher-order: Enhance components

**Patterns:**
- Compound components
- Render props
- Controlled vs uncontrolled
- Lifting state up`,
					CodeExamples: `// Presentational component
function Button({ label, onClick, variant = "primary" }) {
    return (
        <button 
            className={`btn btn-${variant}`}
            onClick={onClick}
        >
            {label}
        </button>
    );
}

// Container component
function UserList() {
    const [users, setUsers] = useState([]);
    
    useEffect(() => {
        fetchUsers().then(setUsers);
    }, []);
    
    return <UserListView users={users} />;
}

// Compound components
function Tabs({ children }) {
    const [activeTab, setActiveTab] = useState(0);
    return (
        <div>
            {React.Children.map(children, (child, index) =>
                React.cloneElement(child, {
                    isActive: index === activeTab,
                    onClick: () => setActiveTab(index)
                })
            )}
        </div>
    );
}

// Controlled component
function Input({ value, onChange }) {
    return (
        <input 
            value={value}
            onChange={(e) => onChange(e.target.value)}
        />
    );
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          218,
			Title:       "Build Tools",
			Description: "Modern build tools: Webpack, Vite, bundling, and optimization.",
			Order:       18,
			Lessons: []problems.Lesson{
				{
					Title: "Webpack and Vite",
					Content: `Build tools bundle, transform, and optimize code for production.

**Webpack:**
- Module bundler
- Loaders for transformations
- Plugins for enhancements
- Code splitting
- Tree shaking

**Vite:**
- Fast dev server
- ES modules
- Hot Module Replacement
- Optimized builds
- Modern tooling

**Key Features:**
- Module bundling
- Code transformation
- Asset optimization
- Development server
- Production builds`,
					CodeExamples: `// webpack.config.js
module.exports = {
    entry: './src/index.js',
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader']
            }
        ]
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './index.html'
        })
    ]
};

// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: 'dist',
        sourcemap: true
    },
    server: {
        port: 3000,
        open: true
    }
})`,
				},
				{
					Title: "Modern Build Tools: Parcel, Rollup, esbuild",
					Content: `Modern build tools offer faster builds and better developer experience.

**Parcel:**
- Zero configuration
- Fast builds
- Built-in optimizations
- Automatic code splitting

**Rollup:**
- Tree shaking
- ES modules output
- Library bundling
- Small bundle sizes

**esbuild:**
- Extremely fast
- Written in Go
- TypeScript support
- Used by Vite

**Turbopack:**
- Next.js build tool
- Rust-based
- Incremental compilation
- Very fast

**Choosing a Tool:**
- Vite: Modern apps, fast dev server
- Webpack: Complex needs, mature ecosystem
- Rollup: Libraries
- Parcel: Simple setup`,
					CodeExamples: `// Parcel (zero config)
// Just run: parcel index.html
// Automatically handles everything

// Rollup config
import { rollup } from 'rollup';
import resolve from '@rollup/plugin-node-resolve';

export default {
    input: 'src/index.js',
    output: {
        file: 'dist/bundle.js',
        format: 'es'
    },
    plugins: [resolve()]
};

// esbuild
import * as esbuild from 'esbuild';

await esbuild.build({
    entryPoints: ['src/index.js'],
    bundle: true,
    outfile: 'dist/bundle.js',
    minify: true,
    target: 'es2020'
});

// Vite (uses esbuild)
// vite.config.js
export default {
    build: {
        target: 'es2020',
        minify: 'esbuild'
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          219,
			Title:       "Testing Frontend Code",
			Description: "Test frontend applications: unit tests, integration tests, and testing tools.",
			Order:       19,
			Lessons: []problems.Lesson{
				{
					Title: "Testing Fundamentals",
					Content: `Testing ensures code works correctly and prevents regressions.

**Testing Types:**
- Unit tests: Test individual functions
- Integration tests: Test component interactions
- E2E tests: Test full user flows
- Snapshot tests: Test UI consistency

**Testing Tools:**
- Jest: Test runner
- React Testing Library: Component testing
- Cypress: E2E testing
- Vitest: Fast unit testing

**Best Practices:**
- Test behavior, not implementation
- Write tests first (TDD)
- Keep tests simple
- Mock external dependencies`,
					CodeExamples: `// Jest test
import { render, screen, fireEvent } from '@testing-library/react';
import Counter from './Counter';

test('increments count on button click', () => {
    render(<Counter />);
    const button = screen.getByText('Increment');
    const count = screen.getByText(/count: 0/i);
    
    fireEvent.click(button);
    
    expect(count).toHaveTextContent('Count: 1');
});

// Component test
test('renders user name', () => {
    const user = { name: 'John', age: 30 };
    render(<UserProfile user={user} />);
    
    expect(screen.getByText('John')).toBeInTheDocument();
});

// Async test
test('fetches and displays data', async () => {
    render(<DataFetcher />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    
    await waitFor(() => {
        expect(screen.getByText('Data loaded')).toBeInTheDocument();
    });
});`,
				},
				{
					Title: "E2E Testing with Cypress and Playwright",
					Content: `End-to-end testing tools test complete user workflows.

**Cypress:**
- Browser-based testing
- Time travel debugging
- Real browser testing
- Great developer experience

**Playwright:**
- Cross-browser testing
- Multiple browsers
- Fast and reliable
- Modern API

**Key Features:**
- Test user interactions
- Assert DOM state
- Network interception
- Screenshot comparison
- Video recording

**Best Practices:**
- Test critical user flows
- Keep tests independent
- Use data-testid attributes
- Avoid testing implementation details`,
					CodeExamples: `// Cypress
describe('Login Flow', () => {
    it('should login successfully', () => {
        cy.visit('/login');
        cy.get('[data-testid="email"]').type('user@example.com');
        cy.get('[data-testid="password"]').type('password');
        cy.get('[data-testid="submit"]').click();
        cy.url().should('include', '/dashboard');
        cy.get('[data-testid="welcome"]').should('contain', 'Welcome');
    });
});

// Playwright
import { test, expect } from '@playwright/test';

test('login flow', async ({ page }) => {
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'user@example.com');
    await page.fill('[data-testid="password"]', 'password');
    await page.click('[data-testid="submit"]');
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator('[data-testid="welcome"]')).toContainText('Welcome');
});`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
