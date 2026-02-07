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
					Content: `CSS animations and transitions are the cornerstone of creating smooth, engaging, and visually polished user experiences on the modern web. Rather than abrupt changes between states, these CSS features allow elements to gradually shift from one appearance to another, giving your interface a professional and responsive feel that users have come to expect.

**1. CSS Transitions**

Transitions are the simplest way to animate changes to CSS properties. When a property value changes — for example, on hover or when a class is toggled — a transition smoothly interpolates between the old and new values over a specified duration. There are four key sub-properties that control how a transition behaves. The transition-property specifies which CSS properties should be animated (you can target a single property like background-color, or use "all" to animate every changing property). The transition-duration determines how long the animation takes, typically expressed in seconds or milliseconds. The transition-timing-function defines the acceleration curve of the transition — common values include "ease" (starts slow, speeds up, then slows down), "linear" (constant speed), "ease-in" (starts slow), "ease-out" (ends slow), and "ease-in-out" (slow at both ends). You can also use cubic-bezier() for fully custom easing. Finally, transition-delay specifies a waiting period before the transition begins, which is useful for staggering multiple animations or creating sequential effects.

**2. CSS Animations with @keyframes**

While transitions handle simple A-to-B changes, CSS animations provide full control over multi-step sequences. You define an animation using @keyframes, which lets you specify the exact state of an element at various points during the animation timeline. The animation-name property links an element to a defined @keyframes rule. The animation-duration controls the total time for one cycle, and animation-timing-function applies easing to each step. The animation-delay sets when the animation starts after being applied. The animation-iteration-count determines how many times the animation plays — use "infinite" for continuous loops. The animation-direction controls playback order: "normal" plays forward, "reverse" plays backward, and "alternate" bounces back and forth. The animation-fill-mode is particularly important — it determines the element's style before and after the animation runs. A value of "forwards" keeps the final keyframe styles, "backwards" applies the first keyframe styles during the delay, and "both" does both. Lastly, animation-play-state lets you pause and resume an animation by toggling between "running" and "paused".

**3. Performance Considerations**

Not all CSS properties animate with the same efficiency. Properties like transform and opacity are handled by the GPU compositor thread, meaning they can animate at 60 frames per second without triggering expensive layout recalculations or repaints. In contrast, animating properties like width, height, margin, or top forces the browser to recalculate the layout of potentially many elements on the page, which can cause visible stuttering — especially on lower-powered devices. To optimize animations, prefer transform (for movement, scaling, and rotation) and opacity (for fade effects). The will-change property can hint to the browser that an element will be animated soon, allowing it to prepare optimizations in advance — but use it sparingly, as overuse can actually degrade performance by consuming extra memory.`,
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
					Content: `CSS custom properties, commonly known as CSS variables, represent one of the most powerful additions to the CSS specification in recent years. They enable developers to define reusable values that can be referenced throughout a stylesheet, dynamically updated with JavaScript, and scoped to specific parts of the document tree. Unlike preprocessor variables (which are compiled away before the browser sees them), CSS custom properties are live in the browser and participate in the cascade and inheritance just like any other CSS property.

**1. Understanding CSS Custom Properties**

A custom property is declared by prefixing a name with two hyphens (--), such as --primary-color or --spacing-unit. These properties are scoped to the selector in which they are declared, meaning a variable defined on a specific element is available to that element and all of its descendants through CSS inheritance. This scoping behavior is what makes custom properties so flexible — you can define global variables on the :root pseudo-class (which targets the html element), making them available everywhere, or you can define them on a specific component to create locally scoped design tokens. To use a custom property, you reference it with the var() function, which also accepts an optional fallback value as its second argument. If the variable is not defined or is invalid, the fallback value will be used instead, providing a safety net that makes your stylesheets more resilient.

**2. Practical Use Cases**

The most common and impactful use case for CSS custom properties is theming. By defining your color palette, typography scale, spacing units, and border radii as custom properties on :root, you can switch between a light mode and dark mode simply by redefining those variables under a different selector, such as [data-theme="dark"]. This approach is far cleaner than maintaining separate stylesheets or using complex class toggling. Custom properties also shine in responsive design — you can redefine spacing or font sizes at different breakpoints without duplicating entire rule sets. For component-based architectures, custom properties allow parent components to influence the styling of children without tight coupling, enabling truly flexible and composable design systems. Perhaps most powerfully, because custom properties are live in the DOM, JavaScript can read and modify them at runtime using element.style.setProperty(), enabling dynamic styling that responds to user input, application state, or real-time data.

**3. Best Practices for Custom Properties**

When working with CSS custom properties, use descriptive and consistent naming conventions — for example, prefix semantic tokens with their purpose (--color-primary, --spacing-lg) rather than their value (--blue, --8px). Always define your global design tokens on :root so they are universally accessible, but do not hesitate to override them locally when a component needs different values. Always provide sensible fallback values in your var() calls, especially in shared component libraries where a variable might not always be defined by the consuming application. Finally, while custom properties enjoy broad browser support today, be mindful if you need to support very old browsers — in such cases, always provide a static fallback declaration before the var() usage.`,
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
					Content: `CSS preprocessors are tools that extend the native CSS language with powerful programming features such as variables, nesting, mixins, functions, and control flow. They allow you to write more maintainable, modular, and DRY (Don't Repeat Yourself) stylesheets that are then compiled into standard CSS that browsers can understand. Among the various preprocessors available, SASS (Syntactically Awesome Style Sheets) and its more popular syntax variant SCSS (Sassy CSS) are by far the most widely adopted in the industry.

**1. SASS and SCSS: The Foundation**

SASS comes in two syntax flavors. The original SASS syntax uses indentation-based formatting (similar to Python or YAML) and omits braces and semicolons. SCSS, on the other hand, uses a syntax that is a strict superset of regular CSS — any valid CSS file is also valid SCSS. This makes SCSS much easier to adopt because developers can gradually introduce preprocessor features into existing CSS codebases without rewriting anything. SCSS files use the .scss extension, while the indented syntax uses .sass. The SCSS syntax has become the de facto standard and is what you will encounter in most modern projects and frameworks.

**2. Key Language Features**

Variables in SCSS are declared with a dollar sign prefix ($primary-color: #3b82f6) and allow you to store values like colors, font sizes, and spacing units in a single place, making global design changes trivial. Nesting lets you write selectors inside other selectors, mirroring the HTML structure and making your stylesheets much more readable — though you should avoid nesting more than three or four levels deep to prevent overly specific selectors. Mixins are reusable blocks of CSS declarations defined with @mixin and included with @include. They can accept arguments, making them incredibly powerful for generating repetitive patterns like vendor prefixes, responsive breakpoints, or button variants. Functions, defined with @function and returning values with @return, allow you to perform calculations and transformations — for example, converting pixel values to rems or generating color variations. SCSS also supports mathematical operators (+, -, *, /, %) for computing values directly in your styles. Control flow directives like @if/@else for conditionals, @for for numeric loops, and @each for iterating over lists and maps give SCSS genuine programming capabilities that enable sophisticated style generation from compact source code.

**3. Modular Architecture and Compilation**

SCSS supports partials — files prefixed with an underscore (like _variables.scss) that are meant to be imported into other files rather than compiled on their own. Combined with the @use and @forward directives (which replaced the older @import), partials enable you to organize your styles into a clean modular architecture with separate files for variables, mixins, components, layouts, and utilities. The compilation step transforms your SCSS source files into standard CSS. This is typically handled by build tools like Webpack (via sass-loader), Vite (which has built-in SASS support), or dedicated CLI tools. During development, you use watch mode to automatically recompile on file changes. For production, the compiled CSS is minified and optimized to reduce file size and improve loading performance.`,
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
					Content: `CSS-in-JS is an approach to styling that allows developers to write CSS directly within JavaScript files, co-locating styles with the components they belong to. This paradigm shift from traditional separate stylesheet files has gained enormous popularity in the React ecosystem and beyond, because it solves several long-standing problems with CSS at scale — namely, global namespace collisions, dead code accumulation, and the difficulty of making styles truly dynamic based on application state.

**1. Benefits of CSS-in-JS**

The most compelling advantage of CSS-in-JS is automatic style scoping. Every styled component generates unique class names at runtime (or build time), ensuring that styles never leak out and accidentally affect other parts of the application. This eliminates the need for naming conventions like BEM or complex selector strategies. Dynamic styling is another major benefit — because your styles live in JavaScript, they can directly reference component props, state, or theme values, making it trivial to create variants, responsive behaviors, and interactive visual effects without maintaining separate CSS classes. CSS-in-JS also provides excellent TypeScript support, with full type checking for your style definitions and theme objects. Additionally, because styles are tied to specific components, bundlers and tree-shaking tools can automatically remove unused styles when components are not imported, resulting in smaller production bundles through dead code elimination.

**2. Popular Libraries in the Ecosystem**

The CSS-in-JS ecosystem offers several mature libraries, each with different trade-offs. styled-components is the most widely adopted, using tagged template literals to create React components with attached styles. It supports theming through a ThemeProvider, global styles, and even CSS animations. Emotion is another popular choice that offers both a styled API (similar to styled-components) and a css prop for inline style objects, with a focus on high performance and small bundle size. Styled-jsx is the default CSS-in-JS solution in Next.js, scoping styles to components using a style jsx tag inside JSX. JSS (JavaScript Style Sheets) takes a different approach by using JavaScript objects to describe styles rather than template literal strings. Finally, Linaria and vanilla-extract represent the "zero-runtime" category — they extract CSS at build time, giving you the authoring benefits of CSS-in-JS without the runtime cost of generating styles in the browser.

**3. Key Concepts and When to Use CSS-in-JS**

The core concepts you will encounter across most CSS-in-JS libraries include tagged template literals (which allow you to write CSS syntax inside JavaScript template strings), styled components (functions that return React components with styles attached), theme providers (React context providers that make a theme object available to all styled components), and dynamic props (the ability to change styles based on component props). CSS-in-JS is an excellent fit when you are working in a component-based architecture (particularly React or Vue), when you need styles that respond to runtime data or user interactions, when you want guaranteed style isolation without manual naming conventions, and when your team values co-location of concerns. However, it is worth noting that CSS-in-JS does introduce a runtime cost — styles must be parsed and injected into the DOM by JavaScript — so for performance-critical applications or static sites, traditional CSS, CSS Modules, or zero-runtime solutions like Linaria may be more appropriate.`,
					CodeExamples: `// styled-components
import styled from 'styled-components';

// Note: styled-components uses template literals (backticks) in JavaScript
// const Button = styled.button(template literal)
//     background-color: props => props.primary ? 'blue' : 'gray';
//     color: white;
//     padding: 10px 20px;
//     border: none;
//     border-radius: 4px;
//     cursor: pointer;
//     &:hover { opacity: 0.8; }

// Usage
<Button primary>Primary Button</Button>
<Button>Secondary Button</Button>

// With theme
// const ThemedButton = styled.button(template literal)
//     background-color: props => props.theme.primary;
//     color: props => props.theme.text;

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

// const buttonStyle = css(template literal)
//     background-color: blue;
//     color: white;
//     padding: 10px 20px;

<button css={buttonStyle}>Button</button>

// Styled-jsx (Next.js)
// <style jsx>{template literal}
//     .button {
//         background-color: blue;
//         color: white;
//     }
// }</style>`,
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
					Content: `Closures are one of the most fundamental and powerful concepts in JavaScript, and understanding them deeply is essential for writing effective, idiomatic code. At its core, a closure is a function that retains access to variables from its surrounding lexical scope, even after that outer scope has finished executing. This behavior arises naturally from how JavaScript handles variable scoping and function creation, and it underpins many common patterns you encounter every day — from callbacks and event handlers to module systems and memoization.

**1. How Closures Work**

When a function is defined in JavaScript, it does not merely capture its own local variables — it also captures a reference to the entire lexical environment in which it was created. This means that if an inner function references a variable declared in an outer function, that variable remains alive in memory even after the outer function has returned. The inner function "closes over" those variables, forming a closure. This is not a special syntax or an opt-in feature; it happens automatically any time a function accesses variables from an enclosing scope. The JavaScript engine keeps the referenced variables on the heap rather than discarding them when the outer function's execution context is popped off the call stack, ensuring they remain available whenever the inner function is invoked.

**2. The Scope Chain**

JavaScript resolves variable names by walking up the scope chain. When your code references a variable, the engine first looks in the local (function or block) scope. If the variable is not found there, it moves to the next outer scope, and continues outward through each enclosing function scope until it reaches the global scope. If the variable is still not found at the global level, a ReferenceError is thrown. This hierarchical lookup mechanism is what gives closures their power — an inner function can access variables from any level of the scope chain, not just its immediate parent. Understanding this chain is critical for avoiding subtle bugs, such as accidentally capturing loop variables by reference rather than by value, which is one of the classic closure pitfalls in JavaScript.

**3. Practical Use Cases**

Closures enable several essential programming patterns. Data privacy is perhaps the most important — by declaring variables inside a function and only exposing controlled access through returned methods, you can create truly private state that cannot be modified from outside, mimicking private fields before JavaScript had native class private syntax. Function factories use closures to generate specialized functions from a template — for example, a createMultiplier function can return tailored multiply-by-2 or multiply-by-3 functions. Event handlers and callbacks naturally form closures over the variables in scope when they are registered, allowing them to reference component state or configuration without passing everything as explicit parameters. The classic module pattern (using an IIFE that returns an object of public methods) is built entirely on closures, and it was the standard way to create encapsulated modules before ES6 introduced native import/export. Finally, memoization — caching expensive computation results — relies on closures to maintain the cache object across multiple function invocations.`,
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
					Content: `JavaScript uses a prototypal inheritance model, which is fundamentally different from the classical inheritance found in languages like Java or C++. Rather than classes serving as blueprints that stamp out instances, JavaScript objects inherit directly from other objects through a mechanism called the prototype chain. Understanding this system is crucial because it underlies everything in JavaScript — from how built-in methods like toString() and hasOwnProperty() work, to how modern ES6 classes function under the hood.

**1. The Prototype Chain**

Every object in JavaScript has an internal link to another object called its prototype (accessible via the __proto__ property or the Object.getPrototypeOf() method). When you attempt to access a property or method on an object and it does not exist directly on that object, the JavaScript engine follows the prototype link to the prototype object and looks for the property there. If it still is not found, the engine continues up the chain to the prototype's prototype, and so on, until it either finds the property or reaches Object.prototype — the root of all prototype chains — whose own prototype is null. This chain-based lookup is what allows objects to "inherit" behavior from their prototypes without duplicating code. It is also what makes JavaScript extremely flexible, since you can modify prototypes at runtime to add or change behavior for all objects that inherit from them.

**2. Constructor Functions and the prototype Property**

Before ES6 classes, the standard way to create objects with shared behavior was through constructor functions. A constructor function is a regular function that is called with the new keyword. When you invoke new Person("John", 30), JavaScript creates a fresh empty object, sets that object's internal prototype link to Person.prototype, executes the constructor function with "this" bound to the new object, and returns the new object (unless the constructor explicitly returns a different object). Methods defined on Person.prototype are shared across all instances created by new Person(), meaning they exist in memory only once rather than being duplicated on each instance. This prototype property on constructor functions is not the same as __proto__ — prototype is a property that exists on functions and determines what the __proto__ of newly created instances will be, while __proto__ is the actual link on every object that points to its prototype in the chain.

**3. ES6 Classes: Syntactic Sugar**

ES6 introduced the class syntax as a cleaner and more familiar way to define constructor functions and their prototypes. Despite appearances, classes in JavaScript are not a new inheritance mechanism — they are syntactic sugar over the existing prototypal system. A class declaration creates a constructor function, and methods defined in the class body are placed on the constructor's prototype object. The extends keyword sets up prototype chain inheritance between two classes, and super() calls the parent class's constructor. Classes also support static methods (defined on the constructor function itself rather than on the prototype) and getter/setter accessors. While the class syntax makes JavaScript code more readable and approachable for developers coming from classical OOP languages, it is important to remember that the underlying prototype mechanism has not changed — understanding prototypes helps you debug inheritance issues and leverage JavaScript's dynamic nature effectively.`,
					CodeExamples: `// Constructor function
function Person(name, age) {
    this.name = name;
    this.age = age;
}

Person.prototype.greet = function() {
    return 'Hello, I\'m ' + this.name;
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
        return 'Hello, I\'m ' + this.name;
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
        return this.name + ' is studying';
    }
}

// Prototype chain
const obj = {};
console.log(obj.toString); // From Object.prototype`,
				},
				{
					Title: "Async/Await and Promises",
					Content: `Asynchronous programming is at the heart of JavaScript, which runs on a single-threaded event loop. Because JavaScript cannot perform blocking operations without freezing the entire user interface, it relies on asynchronous patterns to handle time-consuming tasks like network requests, file I/O, and timers. Promises and the async/await syntax are the modern, standardized tools for managing this asynchronous flow, replacing the older callback-based approach that often led to deeply nested "callback hell."

**1. Promises: Representing Future Values**

A Promise is an object that represents the eventual completion (or failure) of an asynchronous operation and its resulting value. Every Promise exists in one of three states: pending (the initial state, meaning the operation has not yet completed), fulfilled (the operation completed successfully and the Promise holds a result value), or rejected (the operation failed and the Promise holds a reason for the failure, typically an Error object). Once a Promise transitions from pending to either fulfilled or rejected, it is said to be "settled" and its state can never change again — Promises are immutable once settled. You interact with a Promise's result using the .then() method, which accepts a callback that receives the fulfilled value. Errors are handled with .catch(), which catches rejections from anywhere earlier in the Promise chain. The .finally() method runs a callback regardless of whether the Promise was fulfilled or rejected, making it ideal for cleanup operations like hiding loading spinners or closing connections. Promises can be chained by returning new values (or new Promises) from .then() callbacks, creating a clean linear flow of asynchronous steps.

**2. Async/Await: Synchronous-Looking Asynchronous Code**

The async/await syntax, introduced in ES2017, is syntactic sugar built on top of Promises that makes asynchronous code read almost like synchronous code. When you declare a function with the async keyword, it automatically returns a Promise — any value you explicitly return is wrapped in Promise.resolve(), and any thrown error is wrapped in Promise.reject(). Inside an async function, the await keyword pauses execution of that function until the awaited Promise settles. If the Promise fulfills, await returns the result value; if it rejects, await throws the rejection reason, which you can catch with a standard try/catch block. This try/catch pattern for error handling is much more intuitive than chaining .catch() on Promises, especially when you have multiple sequential asynchronous operations that each need different error handling logic. It is important to note that await only pauses the async function itself — the event loop continues running other code while the function is suspended.

**3. Promise Combinators for Concurrent Operations**

JavaScript provides several static methods on the Promise constructor for coordinating multiple concurrent asynchronous operations. Promise.all() takes an array of Promises and returns a single Promise that fulfills with an array of all results when every input Promise has fulfilled — but if any single Promise rejects, the entire Promise.all() rejects immediately with that error. This is perfect for fetching multiple independent resources in parallel. Promise.race() returns a Promise that settles as soon as the first input Promise settles (whether it fulfills or rejects), useful for implementing timeouts. Promise.allSettled() waits for all input Promises to settle regardless of outcome, returning an array of objects describing each result — ideal when you want all results even if some fail. Promise.any() resolves with the first Promise that fulfills successfully, ignoring rejections unless all Promises reject. Choosing the right combinator depends on whether you need all results, just the fastest, or a best-effort collection.`,
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
					Content: `Effective error handling and debugging are not afterthoughts — they are essential disciplines that separate fragile, hard-to-maintain applications from robust, production-ready software. JavaScript applications run in unpredictable environments (varying browsers, network conditions, and user behaviors), so anticipating and gracefully handling errors is critical for delivering a reliable user experience. Equally important is the ability to efficiently debug issues when they arise, using the powerful tools that modern browsers and development environments provide.

**1. Error Handling Patterns**

The try-catch-finally block is the foundation of synchronous error handling in JavaScript. Code that might throw an error is placed inside the try block, and if an error occurs, execution immediately jumps to the catch block, where you receive the error object and can respond appropriately — logging it, displaying a user-friendly message, or attempting a recovery strategy. The finally block runs regardless of whether an error occurred, making it the right place for cleanup operations like closing connections or restoring UI state. For asynchronous code using Promises, errors are caught with .catch() at the end of a Promise chain. With async/await, you wrap your await calls in try/catch blocks, which provides a more natural and readable error handling flow. In React applications, Error Boundaries are class components that catch JavaScript errors anywhere in their child component tree, log those errors, and display a fallback UI instead of crashing the entire application. For truly unexpected errors that escape all other handlers, you can register global error handlers using window.addEventListener("error") for synchronous errors and window.addEventListener("unhandledrejection") for unhandled Promise rejections — these act as a safety net for logging errors to monitoring services.

**2. Debugging Tools and Techniques**

The console object offers far more than just console.log(). The console.error() method outputs errors with a stack trace, console.warn() highlights potential issues, console.table() displays arrays and objects in a readable table format, console.group()/console.groupEnd() organize related log messages into collapsible groups, and console.time()/console.timeEnd() measure execution durations. The debugger statement, when placed in your code, causes the browser to pause execution at that exact point (if DevTools are open), allowing you to inspect the call stack, variable values, and scope chain in real time. Browser DevTools provide a rich suite of features including setting breakpoints (both on specific lines and conditionally based on expressions), stepping through code line by line, watching specific variable values, profiling performance, and inspecting network requests. Source maps are critical for debugging production code — they map minified/bundled code back to your original source files, so breakpoints and stack traces reference readable code even in optimized builds.

**3. Error Types and Best Practices**

JavaScript defines several built-in error types that help identify the nature of a problem. SyntaxError occurs when the JavaScript engine encounters code that violates the language grammar — these are caught at parse time before any code executes. ReferenceError is thrown when you try to access a variable that does not exist in any accessible scope. TypeError occurs when an operation is performed on a value of the wrong type, such as calling undefined as a function or accessing a property on null. RangeError signals that a numeric value is outside its allowed range, such as creating an array with a negative length. You can also create custom error classes by extending the built-in Error class, adding domain-specific information like error codes and context data. Best practices for error handling include always handling errors rather than silently swallowing them, providing meaningful error messages that help diagnose the problem, logging errors with sufficient context for debugging, using error boundaries to prevent cascading failures in component trees, and validating all external input (user input, API responses, configuration) at system boundaries to catch problems early.`,
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
					Content: `TypeScript is a statically typed superset of JavaScript developed by Microsoft that has rapidly become the industry standard for building large-scale frontend and backend applications. It adds an optional type system on top of JavaScript, enabling developers to catch a wide range of errors at compile time rather than discovering them at runtime in production. Because TypeScript is a superset, every valid JavaScript program is already a valid TypeScript program — you can adopt TypeScript gradually, adding type annotations where they provide the most value.

**1. What is TypeScript and Why Use It?**

At its core, TypeScript is a compiler (tsc) that takes TypeScript source code and produces plain JavaScript that runs in any browser or Node.js environment. The key addition is the type system, which allows you to annotate variables, function parameters, return values, and object shapes with explicit types. The TypeScript compiler uses these annotations to verify that your code is internally consistent — for example, that you are not passing a string where a number is expected, or accessing a property that might not exist. This catches entire categories of bugs before your code ever runs. Beyond error detection, the type system dramatically improves the developer experience: IDEs like VS Code use TypeScript's type information to provide intelligent autocompletion, inline documentation, safe refactoring tools, and real-time error highlighting. For teams working on large codebases, TypeScript serves as living documentation — interfaces and type definitions make the shape of data flowing through the application explicit and self-describing.

**2. The Basic Type System**

TypeScript provides several primitive types that correspond to JavaScript values: string for text, number for all numeric values (integers and floating point), and boolean for true/false values. Arrays can be typed with either the bracket syntax (number[]) or the generic syntax (Array<number>). Tuples are fixed-length arrays where each position has a specific type — for example, [string, number] describes a two-element array where the first element is a string and the second is a number. Enums define a set of named constants, which can be numeric (auto-incrementing from 0) or string-valued. The any type effectively opts out of type checking for a value and should be avoided whenever possible, as it defeats the purpose of using TypeScript. The void type represents the absence of a return value (used for functions that do not return anything). The null and undefined types correspond to their JavaScript counterparts. The never type represents values that never occur — for example, a function that always throws an error has return type never. Finally, unknown is the type-safe counterpart to any: you can assign anything to it, but you must perform type narrowing (like typeof checks) before you can use the value, which forces you to handle all possibilities safely.

**3. Type Annotations, Inference, and Union Types**

TypeScript supports both explicit type annotations and automatic type inference. When you write let name: string = "John", you are explicitly annotating the type. But TypeScript is often smart enough to figure out the type on its own — writing let name = "John" will automatically infer the type as string, so you do not need to be redundant. As a general rule, annotate function parameters and return types explicitly (for clarity and documentation), and let TypeScript infer local variable types. Optional types are marked with a question mark (name?: string), indicating that a value may be either the specified type or undefined. Union types, written with the pipe operator (string | number), allow a value to be one of several types, which is invaluable for modeling real-world data that can take different forms — such as an API response that returns either a data object or an error string.`,
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
    return name ? 'Hello, ' + name : "Hello";
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
					Content: `Advanced TypeScript features enable you to build complex, highly type-safe applications by expressing sophisticated relationships between types. These features — generics, utility types, and type guards — allow you to write code that is both maximally reusable and minimally error-prone, catching subtle bugs at compile time that would otherwise surface as runtime failures in production.

**1. Generics: Parameterized Types for Reusable Code**

Generics allow you to write functions, classes, and interfaces that work with a variety of types while preserving full type safety. Rather than using any (which discards type information) or writing separate implementations for each type, you define a type parameter (conventionally named T, U, K, V, etc.) that acts as a placeholder, filled in when the generic is actually used. For example, a generic identity function identity<T>(arg: T): T declares that it accepts an argument of some type T and returns a value of the same type — the compiler ensures the input and output types always match, regardless of what T turns out to be. Generics become even more powerful with constraints. Using the extends keyword, you can restrict what types are acceptable for a type parameter — for instance, <T extends { length: number }> ensures that T must have a length property, allowing you to safely access that property inside the function. You can also provide default type parameters (<T = string>) that are used when no explicit type argument is supplied. Generics are pervasive in TypeScript — they power built-in types like Array<T>, Promise<T>, and Map<K, V>, and they are essential for building reusable libraries, data structures, and API client code.

**2. Utility Types: Transforming Existing Types**

TypeScript ships with a rich set of built-in utility types that let you derive new types from existing ones without rewriting type definitions. Partial<T> makes all properties of T optional, which is perfect for functions that accept partial updates to an object. Required<T> does the opposite, making all properties mandatory. Pick<T, K> creates a new type by selecting only the specified properties from T — for example, Pick<User, "name" | "email"> creates a type with just those two fields. Omit<T, K> excludes specified properties, which is useful for creating types that represent a subset of a larger type. Readonly<T> makes all properties immutable, preventing accidental mutations. Record<K, T> constructs an object type whose keys are of type K and values are of type T, useful for dictionaries and maps. ReturnType<T> extracts the return type of a function type, and Parameters<T> extracts the parameter types as a tuple. These utility types compose together beautifully — you can write types like Partial<Pick<User, "name" | "email">> to express exactly the shape you need, keeping your type definitions DRY and your code precise.

**3. Type Guards: Runtime Type Narrowing**

Type guards are expressions that perform runtime checks and narrow the type of a variable within a conditional block, allowing TypeScript to understand what type a value is at a specific point in your code. The simplest type guards use typeof (for primitives like string, number, boolean) and instanceof (for class instances). TypeScript automatically narrows the type within the corresponding if/else branches, so after if (typeof value === "string"), TypeScript knows that value is a string and provides string-specific autocompletion and error checking. For more complex scenarios, you can write custom type guard functions that return a boolean and use the "value is Type" return type annotation — for example, function isUser(obj: unknown): obj is User. Discriminated unions combine union types with a common literal property (a "discriminant") to enable exhaustive type narrowing via switch statements — this pattern is especially powerful for modeling state machines, Redux actions, or API response variants, where each variant has a type or kind field that uniquely identifies it.`,
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
					Content: `Decorators and namespaces are advanced TypeScript features that address two important aspects of building large applications: cross-cutting concerns (like logging, validation, and access control) and code organization at scale. While decorators provide a powerful mechanism for modifying and annotating classes and their members, namespaces offer a way to group related code under a single umbrella to avoid naming collisions and improve discoverability.

**1. Decorators: Metadata and Aspect-Oriented Programming**

Decorators are special declarations that can be attached to classes, methods, properties, accessors, and parameters using the @ syntax. When a decorator is applied, it receives information about the thing it decorates and can modify its behavior, add metadata, or even replace it entirely. For example, a method decorator receives the target object (the class prototype), the method name, and the property descriptor — by wrapping the original method in the descriptor, you can add logging, caching, access control, or input validation without changing the method's own code. This is a form of Aspect-Oriented Programming (AOP), where cross-cutting concerns are separated from business logic and applied declaratively. Decorators are central to major frameworks: Angular uses them extensively for defining components (@Component), services (@Injectable), and modules (@NgModule), while NestJS uses decorators for controllers (@Controller), routes (@Get, @Post), and dependency injection (@Inject). Although decorators have been an "experimental" feature in TypeScript for years (enabled via the experimentalDecorators compiler option), the TC39 proposal for native JavaScript decorators has reached Stage 3 and TypeScript 5.0+ supports the standardized decorator syntax alongside the legacy experimental syntax.

**2. Namespaces: Logical Code Grouping**

Namespaces (formerly called "internal modules" in early TypeScript) provide a way to organize related code under a named scope, preventing pollution of the global namespace. A namespace is declared with the namespace keyword, and any functions, classes, interfaces, or variables that need to be accessible outside the namespace must be explicitly marked with export. Namespaces can be nested to create hierarchical structures — for example, Geometry.Circle.area() — providing clear, dot-notation access paths that make it immediately obvious where a function belongs. Namespaces can also be split across multiple files using reference directives, allowing large codebases to maintain logical grouping while keeping files manageable. However, it is important to note that in modern TypeScript development, ES modules (import/export) have largely replaced namespaces for code organization. Namespaces are still valuable in specific scenarios: when writing type declaration files for JavaScript libraries that use global namespaces, when working with legacy codebases that predate ES modules, or when you need to merge declarations across multiple files using TypeScript's declaration merging feature. For new projects, the general recommendation is to prefer ES module imports over namespaces.

**3. Practical Considerations**

When deciding whether to use decorators, consider your project's framework and tooling. If you are using Angular or NestJS, decorators are a first-class citizen of the architecture and you will use them extensively. For other projects, decorators are most valuable when you have cross-cutting concerns that apply to many classes or methods — logging, performance monitoring, validation, serialization, and authorization are classic examples. Keep decorator logic simple and composable, and be aware that decorators execute at class definition time, not at call time. For namespaces versus modules, the choice is straightforward in most cases: use ES modules for new code and only use namespaces when you have a specific technical reason, such as augmenting global type declarations or working within the constraints of a legacy build system.`,
					CodeExamples: `// Decorators (experimental)
function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function(...args: any[]) {
        console.log('Calling ' + propertyKey);
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
					Content: `React is an open-source JavaScript library created by Meta (formerly Facebook) for building user interfaces, and it has become the most widely used frontend library in the world. React fundamentally changed how developers think about building web applications by introducing a component-based, declarative programming model that makes complex UIs easier to reason about, build, and maintain. Rather than imperatively manipulating the DOM to reflect state changes, you describe what the UI should look like for any given state, and React efficiently figures out how to update the actual DOM to match.

**1. Core Philosophy and Architecture**

React is built around several key architectural principles. It is component-based, meaning the entire UI is composed of small, self-contained, reusable pieces called components — each responsible for rendering a portion of the interface. It is declarative, meaning you write code that describes the desired end state rather than the step-by-step instructions to get there. React uses a Virtual DOM — a lightweight in-memory representation of the actual DOM — to optimize updates. When state changes, React creates a new Virtual DOM tree, compares it with the previous one (a process called "reconciliation" or "diffing"), and calculates the minimal set of actual DOM operations needed to bring the UI up to date. This approach makes updates efficient even in complex, frequently changing interfaces. React enforces a one-way data flow (also called unidirectional data flow), where data flows from parent components down to children through props. This makes the application's data flow predictable and easier to debug, since you always know where data comes from and how it gets to each component.

**2. Key Concepts: Components, JSX, Props, and State**

Components are the building blocks of every React application. Modern React primarily uses functional components — plain JavaScript functions that accept props as an argument and return JSX describing what should appear on screen. JSX (JavaScript XML) is a syntax extension that lets you write HTML-like markup directly in your JavaScript code. Under the hood, JSX is compiled to React.createElement() calls, but the HTML-like syntax makes your component templates much more readable and intuitive. Props (short for "properties") are the mechanism for passing data from a parent component to a child component — they are read-only from the child's perspective, enforcing the unidirectional data flow principle. State is data that belongs to a component and can change over time in response to user interactions, network responses, or other events. When state changes, React re-renders the component (and its children) to reflect the new data. In functional components, state is managed using the useState hook, which returns the current value and a setter function.

**3. Setting Up a React Project**

There are several ways to start a new React project, each suited to different needs. Vite has become the recommended tool for new React projects — it provides an extremely fast development server with hot module replacement, near-instant startup times, and optimized production builds, all with minimal configuration. Next.js is a full-featured React framework that adds server-side rendering, static site generation, API routes, file-based routing, and many other production-ready features out of the box — it is the best choice for production applications that need SEO, performance optimization, or server-side capabilities. Create React App (CRA) was historically the standard bootstrapping tool but has been largely superseded by Vite and Next.js for new projects. Manual setup is always an option for developers who want full control over their build configuration, using tools like Webpack or Rollup directly, though this requires significantly more configuration effort.`,
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
					Content: `Components are the fundamental building blocks of every React application. Each component encapsulates a piece of the user interface — its structure, behavior, and potentially its styling — into a self-contained, reusable unit. Understanding the different types of components, how data flows between them through props, and how to compose them together is essential for building well-structured, maintainable React applications.

**1. Component Types and Their Evolution**

Modern React development overwhelmingly uses functional components — plain JavaScript functions that accept a props object and return JSX. With the introduction of hooks in React 16.8, functional components gained the ability to manage state, perform side effects, and access context, eliminating the need for class components in almost all scenarios. Class components, which extend React.Component and define a render() method, are now considered legacy — you will encounter them in older codebases and in error boundaries (which still require class syntax), but new code should be written as functional components. Higher-order components (HOCs) are a pattern where a function takes a component and returns an enhanced version of it, adding additional props, behavior, or rendering logic — for example, a withAuth HOC might wrap a component and redirect unauthenticated users to a login page. Render props is another pattern where a component receives a function as a prop (or as its children) and calls that function to determine what to render, allowing flexible sharing of behavior between components. While HOCs and render props were essential patterns before hooks, custom hooks have largely replaced them as the preferred way to share logic between components.

**2. Props: The Component Communication System**

Props are the primary mechanism for passing data from parent components to their children, forming the backbone of React's unidirectional data flow. Props are read-only — a child component must never modify the props it receives, which ensures that the source of truth for any piece of data remains clear and predictable. Props can carry any JavaScript value: strings, numbers, booleans, objects, arrays, functions (commonly used as event callbacks), and even other React elements. Default prop values can be specified using JavaScript default parameters in the function signature (e.g., function Button({ label = "Click" })). The special children prop allows components to accept nested JSX content, enabling a powerful composition pattern where wrapper components render whatever is placed between their opening and closing tags. For type safety, you can validate props at runtime using the PropTypes library (which issues console warnings in development when props do not match expected types) or at compile time using TypeScript interfaces, which provide much stronger guarantees and better developer experience through autocompletion and inline error reporting.

**3. Component Composition and Data Flow**

Composition is the art of combining simple components to build complex UIs. Rather than creating monolithic components that handle everything, you build small, focused components and nest them together — a Layout component contains a Header, Main, and Footer; the Main component contains a Sidebar and Content area; and so on. This approach makes individual components easier to understand, test, and reuse. However, when deeply nested components need access to data from a distant ancestor, passing props through every intermediate component (known as "props drilling") becomes tedious and clutters components with props they do not directly use. The Context API solves this by allowing you to create a "context" that holds data and make it directly available to any descendant component, no matter how deep in the tree it sits. Context is ideal for truly global concerns like the current user, theme preferences, or locale settings, but should not be overused for all state management, as it can lead to unnecessary re-renders when the context value changes.`,
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
					Content: `React Router is the de facto standard library for handling navigation and routing in React single-page applications (SPAs). In a traditional multi-page website, clicking a link causes the browser to make a new HTTP request to the server and load an entirely new HTML page. In a SPA, however, the application loads once, and subsequent navigation happens entirely on the client side — React Router intercepts navigation events, updates the URL in the browser's address bar, and renders the appropriate component, all without a full page reload. This results in faster, more fluid navigation that feels like a native application.

**1. Core Components and Hooks**

React Router provides a declarative API built around a few core components. BrowserRouter (typically aliased as Router) wraps your entire application and uses the HTML5 History API to keep the URL in the address bar synchronized with the currently displayed content. Inside the router, the Routes component acts as a container for your route definitions, and each Route component maps a URL path to a React component that should be rendered when the path matches. The Link component replaces standard HTML anchor tags for internal navigation — instead of triggering a full page reload, Link uses the router to perform client-side navigation, preserving application state and avoiding unnecessary re-fetching. The useNavigate hook enables programmatic navigation from within your code — for example, redirecting to a dashboard after a successful login, or sending the user back to the previous page. The useParams hook extracts dynamic parameters from the URL (like a user ID in /users/:id), and useSearchParams provides access to query string parameters.

**2. Advanced Routing Concepts**

Route matching in React Router is path-based and supports dynamic segments (prefixed with :), optional segments, splat/catch-all patterns (*), and exact matching. Nested routes allow you to define route hierarchies that mirror your UI structure — for example, a /dashboard route might have child routes for /dashboard/profile and /dashboard/settings, with the parent route providing a shared layout that includes navigation and renders child routes via an Outlet component. Route parameters enable dynamic pages — a single /users/:id route definition can handle /users/1, /users/42, or any other user ID. Protected routes (also called authenticated or guarded routes) are a pattern where you wrap route components with an authentication check — if the user is not logged in, they are redirected to a login page instead of seeing the protected content. Code splitting with routes leverages React.lazy() and Suspense to load route components only when they are needed, dramatically reducing the initial bundle size of your application by deferring the download of code for routes the user has not yet visited.

**3. Best Practices for Routing**

Always use the Link component (or NavLink for navigation elements that need active styling) instead of plain HTML anchor tags for internal navigation — this ensures client-side routing is used and prevents unnecessary full page reloads. Organize your route definitions logically, grouping related routes together and using nested routes to reflect the UI hierarchy. Protect sensitive routes by implementing a reusable ProtectedRoute wrapper component that checks authentication status and redirects unauthenticated users. Lazy load route components to keep your initial bundle small and improve Time to Interactive, especially for large applications with many routes. Finally, handle 404/not-found cases by adding a catch-all route at the end of your route definitions that renders a helpful not-found page.`,
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
					Content: `Error boundaries and forms address two critical aspects of building production-quality React applications: gracefully handling runtime errors so they do not crash the entire UI, and managing user input in a way that is reliable, validated, and provides a good user experience. Both topics are essential for any application that interacts with real users and real data.

**1. Error Boundaries: Preventing Cascading Failures**

In React, an unhandled JavaScript error thrown during rendering, in a lifecycle method, or in a constructor of any component will cause the entire component tree to unmount, leaving the user staring at a blank page. Error boundaries solve this problem by catching errors in their child component tree and displaying a graceful fallback UI instead. An error boundary is a class component that implements either (or both) of two special lifecycle methods: static getDerivedStateFromError(error), which updates state so the next render shows a fallback UI, and componentDidCatch(error, errorInfo), which is used for logging the error to an error reporting service. Error boundaries only catch errors during rendering and lifecycle methods — they do not catch errors in event handlers, asynchronous code, or server-side rendering. The strategic placement of error boundaries is important: you might wrap your entire application in a top-level error boundary to prevent complete whitescreens, while also placing more granular error boundaries around individual features or widgets so that a failure in one section does not affect the rest of the page. Libraries like react-error-boundary provide functional component wrappers and additional features like retry buttons and error recovery.

**2. Forms in React: Controlled and Uncontrolled Patterns**

Form handling in React follows two fundamental patterns. In the controlled component pattern, the form element's value is driven by React state — the input displays whatever is stored in state, and every keystroke triggers an onChange handler that updates the state. This gives React full control over the form data at all times, making it straightforward to validate input on every change, conditionally enable/disable buttons, format input values, and implement complex dependent field logic. In the uncontrolled component pattern, the DOM itself maintains the form state, and you access values using refs (typically via useRef) when you need them — usually at submission time. Uncontrolled components require less code and can be simpler for basic forms, but they provide less real-time control over the data. For anything beyond the simplest forms, controlled components are generally recommended. Form libraries like React Hook Form and Formik abstract away much of the boilerplate involved in form management. React Hook Form is particularly popular because it minimizes re-renders by using uncontrolled components internally while providing a controlled-like API, resulting in excellent performance even with large forms.

**3. Best Practices for Errors and Forms**

Place error boundaries strategically at both the application level (to catch catastrophic errors) and at the feature level (to isolate failures). Always log caught errors to a monitoring service like Sentry or LogRocket so you can discover and fix issues proactively. For forms, always validate user input — both on the client side (for immediate feedback) and on the server side (for security). Provide clear, specific error messages that tell the user exactly what is wrong and how to fix it, and display validation errors adjacent to the relevant input field. Handle form submission gracefully by disabling the submit button during processing, showing loading indicators, and handling both success and error responses. Consider using established form validation libraries like Yup or Zod for schema-based validation, which can be integrated with form libraries to provide declarative, composable validation rules.`,
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
					Content: `Hooks are special functions introduced in React 16.8 that allow functional components to use state, lifecycle behavior, context, and other React features that were previously only available in class components. Hooks fundamentally transformed how React code is written, enabling a simpler, more composable, and more reusable programming model. The four essential hooks — useState, useEffect, useContext, and useRef — form the foundation upon which virtually all React applications are built.

**1. useState: Managing Component State**

The useState hook adds state management to functional components. When you call useState(initialValue), it returns an array with exactly two elements: the current state value and a setter function to update it. The initial value can be any JavaScript value — a string, number, boolean, object, array, or null. For expensive initial computations, you can pass a function to useState (called "lazy initialization"), which will only be executed on the first render. State updates in React are asynchronous and batched — when you call the setter function, React does not immediately update the state and re-render. Instead, it schedules an update and batches multiple state changes together for performance. This means you should not rely on reading the state value immediately after calling the setter. When your new state depends on the previous state, always use the functional update form: setCount(prev => prev + 1) rather than setCount(count + 1), which avoids stale closure issues. Each call to useState manages an independent piece of state, so you can call it multiple times in a single component to manage different values.

**2. useEffect: Handling Side Effects**

The useEffect hook is React's mechanism for performing side effects in functional components — operations that reach outside the React rendering cycle, such as fetching data from an API, subscribing to events, setting up timers, or directly manipulating the DOM. useEffect accepts two arguments: a function containing the side effect logic, and an optional dependency array. The effect function runs after React has committed updates to the DOM (i.e., after rendering). The dependency array controls when the effect re-runs: if you provide an empty array [], the effect runs only once after the initial render (equivalent to componentDidMount). If you provide specific values [userId, query], the effect re-runs whenever any of those values change. If you omit the dependency array entirely, the effect runs after every render, which is rarely what you want. The effect function can optionally return a cleanup function, which React calls before re-running the effect and when the component unmounts — this is essential for canceling subscriptions, clearing timers, and preventing memory leaks.

**3. useContext: Consuming Context Values**

The useContext hook provides a clean, straightforward way to access values from a React context without the cumbersome Consumer component wrapper pattern. You pass it a context object (created by React.createContext()), and it returns the current value of that context, as determined by the nearest matching Provider higher up in the component tree. Whenever the Provider's value changes, every component that calls useContext with that context will re-render with the new value. This makes useContext ideal for accessing global or semi-global data like the current theme, authenticated user, or locale settings, effectively solving the props drilling problem where data must be passed through many intermediate components that do not use it themselves.

**4. useRef: Persistent Mutable Values and DOM Access**

The useRef hook creates a mutable reference object with a .current property that persists across renders without causing re-renders when modified. This makes it fundamentally different from state — updating a ref is synchronous and silent (no re-render triggered). The most common use of useRef is accessing DOM elements directly: you attach the ref to a JSX element via the ref attribute, and after rendering, ref.current points to the actual DOM node, allowing you to programmatically focus inputs, measure element dimensions, or integrate with non-React libraries. Beyond DOM access, useRef is also valuable for storing any mutable value that needs to persist between renders but should not trigger re-renders when it changes — for example, storing a previous state value for comparison, holding a timer ID for later cleanup, or tracking whether a component is mounted.`,
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
        fetch('/api/users/' + userId)
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
					Content: `Custom hooks are one of the most powerful features of the React hooks system, allowing you to extract component logic into reusable, testable, and shareable functions. When you find yourself writing the same stateful logic — such as data fetching, form handling, or event listener management — in multiple components, a custom hook lets you encapsulate that logic once and reuse it everywhere, keeping your components focused on rendering UI rather than managing complex behavior.

**1. Custom Hook Rules and Conventions**

A custom hook is simply a JavaScript function whose name starts with "use" — this naming convention is not just cosmetic; it signals to React's linting tools (via the eslint-plugin-react-hooks package) that the function follows the Rules of Hooks and should be checked accordingly. Inside a custom hook, you can call any built-in hook (useState, useEffect, useContext, useRef, etc.) as well as other custom hooks, composing complex behavior from simpler building blocks. The key insight is that each component that calls your custom hook gets its own independent copy of the state and effects defined within the hook — custom hooks share logic, not state. A custom hook can return anything that is useful to the consuming component: a single value, an array (like useState does), or an object containing multiple values and functions. This return value becomes the public API of your hook, so design it thoughtfully — expose only what consumers need, keeping internal implementation details private.

**2. Common Custom Hook Patterns**

The most universally useful custom hook pattern is data fetching. A useFetch hook typically manages loading state, error state, and the fetched data, returning all three to the component so it can render appropriate UI for each scenario — a loading spinner, an error message, or the actual data. This eliminates the repetitive boilerplate of writing useEffect with fetch, useState for loading/error/data, and cleanup logic in every component that loads data. Form handling hooks (like useForm) manage form state, validation, dirty tracking, and submission in a reusable way, often returning register functions, handleSubmit wrappers, and error objects. Local storage hooks (useLocalStorage) synchronize a piece of React state with the browser's localStorage, initializing from stored data if available and persisting changes automatically. Debounce and throttle hooks (useDebounce, useThrottle) wrap a value or callback to limit how frequently it updates, which is essential for performance-sensitive operations like search-as-you-type or resize handlers. Authentication hooks (useAuth) encapsulate the logic for checking login status, providing user data, and exposing login/logout functions.

**3. Best Practices for Custom Hooks**

Keep each custom hook focused on a single concern — it is better to have several small, composable hooks than one large hook that does everything. Name your hooks descriptively so their purpose is immediately clear (useWindowSize, useOnlineStatus, useDebounce). Always handle cleanup in your hooks by returning cleanup functions from useEffect to prevent memory leaks and stale subscriptions. Write your hooks to be generic and configurable through parameters rather than hardcoding specific URLs, keys, or behaviors. Test your custom hooks independently using tools like @testing-library/react-hooks (or renderHook from @testing-library/react in newer versions), which lets you test hook logic without mounting a full component. Finally, consider publishing particularly useful hooks as part of a shared internal library or even as open-source packages — the community has produced excellent collections like usehooks-ts and ahooks that demonstrate well-crafted custom hook design.`,
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
    const { data, loading, error } = useFetch('/api/users/' + userId);
    
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
					Content: `The React Context API provides a built-in mechanism for sharing data across the component tree without having to explicitly pass props through every level of nesting. It solves one of the most common pain points in React development — "props drilling" — where data needed by a deeply nested component must be threaded through multiple intermediate components that have no use for it themselves. Context creates a direct channel between a data provider and any descendant consumers, regardless of how many layers of components separate them.

**1. How the Context API Works**

The Context API revolves around three core pieces. First, you create a context object using React.createContext(defaultValue), which produces a Provider component and a default value that is used when no Provider is found above a consumer in the tree. Second, you wrap a portion of your component tree with the Context.Provider component, passing the shared data as its value prop — any component within this subtree can access the provided value. Third, consumer components call the useContext(MyContext) hook to read the current context value. When the Provider's value changes, React automatically re-renders every component that consumes that context. It is important to understand that context uses reference identity to determine when to re-render consumers — if you pass a new object literal as the value on every render, all consumers will re-render even if the actual data has not changed. You can create and use multiple contexts in a single application, each serving a different purpose, and nest their Providers to compose multiple pieces of shared data.

**2. Ideal Use Cases for Context**

Context is designed for data that is truly "global" to a section of your component tree — data that many components at various nesting levels need to access. The most classic use cases include theming (providing the current color scheme, font sizes, and design tokens to all styled components), authentication (sharing the current user object, login status, and auth methods across protected routes and navigation), and internationalization (providing the current locale and translation functions to all text-rendering components). Context is also excellent for feature flags, user preferences, and any configuration that applies broadly across the application. However, context is not a general-purpose state management solution — it lacks features like middleware, devtools integration, optimized selectors, and time-travel debugging that dedicated state management libraries provide. For complex state with frequent updates that affect only small parts of the UI, a library like Redux, Zustand, or Jotai is often a better fit.

**3. Best Practices and Performance**

Split your contexts by concern rather than putting everything into a single monolithic context. Having separate ThemeContext, AuthContext, and LocaleContext ensures that changes to one domain (like switching the theme) do not cause re-renders in components that only consume a different context (like auth state). For complex state logic with multiple related values and actions, combine useContext with useReducer — the reducer handles state transitions, and the context distributes the state and dispatch function to consumers, creating a lightweight Redux-like pattern. To prevent unnecessary re-renders, memoize the context value using useMemo so that a new object is only created when the underlying data actually changes. Create custom hooks (like useTheme() or useAuth()) that wrap useContext calls — this provides a cleaner API, enables validation (throwing an error if the hook is called outside a Provider), and makes it easy to refactor the underlying implementation later without changing consumer code. Avoid overusing context for frequently changing values that only affect a small number of components — in such cases, lifting state up or using component composition is often more performant.`,
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
					Content: `Redux is a predictable state management library for JavaScript applications that has been the dominant global state solution in the React ecosystem for many years. It implements a strict unidirectional data flow architecture inspired by Flux and functional programming principles, making state changes explicit, traceable, and debuggable. While Redux has been criticized for its boilerplate, the modern Redux Toolkit has dramatically simplified the developer experience while retaining all of Redux's powerful guarantees.

**1. Core Redux Concepts**

Redux is built around a few fundamental concepts that enforce a predictable state management pattern. The Store is a single JavaScript object that holds the entire application state — there is exactly one store, serving as the "single source of truth" for all data. This makes it straightforward to inspect the current state, persist it, or hydrate it from a server. Actions are plain JavaScript objects with a type property (a string that describes what happened) and an optional payload carrying any associated data. Actions are the only way to trigger state changes — you never modify the state directly. Reducers are pure functions that take the current state and an action, and return a new state object. Because reducers are pure (no side effects, same inputs always produce same outputs), state transitions are completely predictable and easy to test. The dispatch function is how you send actions to the store — calling store.dispatch(action) triggers the reducer, computes the new state, and notifies all subscribed components. Selectors are functions that extract specific pieces of data from the store, decoupling components from the store's internal structure and enabling memoized derivations of computed data.

**2. Redux Toolkit: Modern Redux Development**

Redux Toolkit (RTK) is the officially recommended way to write Redux logic, and it dramatically reduces the amount of code you need to write. The configureStore function replaces the manual store setup with sensible defaults — it automatically configures the Redux DevTools extension, adds middleware for catching common mistakes (like accidentally mutating state), and combines your reducers. The createSlice function is the heart of RTK — it generates action creators and action types automatically from a set of reducer functions that you define using a simple, readable syntax. Inside createSlice reducers, you can write code that appears to mutate state directly (state.value += 1) because RTK uses the Immer library internally to convert these "mutations" into proper immutable updates. For asynchronous operations like API calls, createAsyncThunk creates thunks that dispatch pending, fulfilled, and rejected actions automatically, which your slice can handle in extraReducers. RTK Query is an advanced data fetching and caching solution built into Redux Toolkit that eliminates the need to write thunks and reducers for server data entirely — you define API endpoints declaratively, and RTK Query generates hooks that handle fetching, caching, polling, invalidation, and optimistic updates automatically.`,
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
					Content: `The frontend state management landscape has evolved significantly beyond Redux, with several modern libraries offering simpler, more focused approaches to different kinds of state management challenges. Understanding the strengths of each library helps you choose the right tool for your specific needs, rather than defaulting to a one-size-fits-all solution that may be more complex than necessary.

**1. Zustand: Minimal Global State**

Zustand is a lightweight state management library that has gained massive popularity for its simplicity and minimal boilerplate. You create a store by calling a create function with a callback that defines your state and actions — there are no reducers, action types, providers, or dispatchers. Components subscribe to the store using the returned hook, and Zustand automatically re-renders only the components that access the specific pieces of state that changed. The API is intentionally small and intuitive: you define state as plain values, actions as functions that call set() to update state, and selectors as arguments to the hook for fine-grained subscriptions. Zustand has excellent TypeScript support with full type inference, works outside of React (useful for integrating with non-React code), and supports middleware for persistence, devtools integration, and more. It is an ideal choice when you need simple, shared global state without the ceremony of Redux.

**2. Jotai: Atomic State Management**

Jotai takes a fundamentally different approach to state management by using atoms — independent, minimal units of state that components can subscribe to individually. Unlike top-down stores (where all state lives in a single object), Jotai uses a bottom-up approach where state is composed from small atoms that can depend on each other. You define an atom with a default value, and components read and write atoms using the useAtom hook. Derived atoms can compute values from other atoms, creating a reactive dependency graph similar to a spreadsheet. This atomic model provides fine-grained reactivity — when an atom changes, only the components that directly subscribe to that specific atom re-render, without any selector optimization needed. Jotai is particularly well-suited for complex state with many interdependent values, for applications that need precise control over re-rendering, and for teams that prefer thinking about state as a collection of independent pieces rather than a single store.

**3. React Query and SWR: Server State Management**

React Query (now TanStack Query) and SWR (stale-while-revalidate, by Vercel) represent a paradigm shift in how we think about data from APIs. They recognize that "server state" — data fetched from a backend — has fundamentally different characteristics than "client state" — local UI state like which tab is active or what the user typed in a form. Server state is asynchronous, cached, potentially stale, and shared across multiple components. These libraries handle all of this complexity automatically: they cache fetched data and serve it instantly on subsequent requests, automatically refetch data when it becomes stale or when the window regains focus, provide loading and error states, support optimistic updates (updating the UI before the server confirms), handle pagination and infinite scrolling, and deduplicate identical requests. By using React Query or SWR for server data, you often eliminate the need for a global state manager entirely — the remaining client state can usually be handled with useState and useContext.

**4. Choosing the Right Tool**

The choice of state management library should be driven by the nature of your state. For simple global client state (theme, user preferences, UI flags), Zustand provides the simplest and most ergonomic solution. For complex client state with many interdependent values and fine-grained update requirements, Jotai's atomic model shines. For server data (API responses, database records), React Query or SWR should be your first choice, as they handle caching, synchronization, and lifecycle concerns that client state libraries are not designed for. For very large applications with complex business logic, middleware needs, and teams that benefit from strict architectural patterns, Redux Toolkit remains a solid choice. Many modern applications combine multiple approaches — for example, React Query for server data, Zustand for small amounts of global client state, and plain useState for local component state.`,
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
					Content: `Vue.js is a progressive JavaScript framework for building user interfaces, created by Evan You and maintained by a dedicated open-source community. The term "progressive" is central to Vue's philosophy — it is designed to be incrementally adoptable, meaning you can start using it as a simple library to enhance parts of an existing page (like jQuery once did) and gradually scale up to a full-featured framework for building complex single-page applications. This flexibility, combined with an exceptionally gentle learning curve and excellent documentation, has made Vue one of the three most popular frontend frameworks alongside React and Angular.

**1. Core Philosophy and Design**

Vue is fundamentally component-based — every Vue application is composed of a tree of components, each encapsulating its own template (HTML), logic (JavaScript), and styling (CSS). Vue's reactivity system is one of its most distinctive features: when you declare reactive data properties, Vue automatically tracks which components depend on which pieces of data, and when data changes, only the affected components are re-rendered. This "just works" without requiring explicit setState calls, immutability patterns, or manual optimization — you modify your data directly, and the UI updates automatically. Vue uses an HTML-based template syntax that allows you to declaratively bind the rendered DOM to the component's data using intuitive directives and mustache interpolation ({{ expression }}). The templates are compiled into optimized render functions at build time, giving you the readability of templates with the performance of compiled code. Vue also supports JSX for developers who prefer the React-style approach of writing render logic directly in JavaScript.

**2. Key Features and API Styles**

Vue 3 offers two API styles for writing component logic. The Composition API (using the <script setup> syntax) is the modern, recommended approach — it uses imported functions like ref(), reactive(), computed(), and watch() to define state and behavior in a flat, composable manner that is easy to extract into reusable composition functions (Vue's equivalent of React custom hooks). The Options API is the classic Vue style that organizes component logic into an object with predefined sections: data() for reactive state, methods for functions, computed for derived values, watch for side effects, and lifecycle hooks like mounted and unmounted. Both APIs compile to the same underlying system, so the choice is largely a matter of preference and project requirements — the Composition API tends to scale better for complex components and makes logic reuse more natural, while the Options API provides a more structured and beginner-friendly organization. Vue's Single File Component (SFC) format (.vue files) elegantly co-locates template, script, and style in a single file, with scoped styles that are automatically limited to the component's own elements.

**3. Setting Up a Vue Project**

Vue offers several starting points depending on your needs. For quick prototyping or enhancing an existing page, you can include Vue via a CDN script tag and start using it immediately without any build step. For production applications, Vite is the recommended build tool — it provides lightning-fast hot module replacement, out-of-the-box Vue support, and optimized production builds. Vue CLI is the older official scaffolding tool that provides a rich plugin ecosystem and GUI-based project management, though Vite has largely superseded it for new projects. Nuxt.js is the full-featured meta-framework for Vue (analogous to Next.js for React), providing server-side rendering, static site generation, file-based routing, auto-imports, and many other production-ready features. Nuxt is the best choice when you need SEO optimization, server-side rendering, or want a batteries-included framework experience.`,
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
					Content: `Vue directives are special HTML attributes prefixed with "v-" that provide declarative, reactive behavior to DOM elements. They are one of Vue's most distinctive features, offering an intuitive template syntax that makes it easy to express dynamic rendering logic directly in your HTML markup. Combined with Vue's component system, directives enable you to build complex, interactive user interfaces with clear and readable template code.

**1. Essential Directives for Dynamic Rendering**

Vue provides a rich set of built-in directives that cover all common rendering scenarios. The v-if, v-else-if, and v-else directives handle conditional rendering — they completely add or remove elements from the DOM based on a condition, meaning the element and its event listeners are truly destroyed and recreated. For toggling visibility without destroying elements, v-show simply sets the CSS display property, which is more efficient when you toggle frequently but less efficient for rarely shown content. The v-for directive renders a list of elements by iterating over an array, object, or range, requiring a :key attribute on each element to help Vue's diffing algorithm efficiently update the list when items are added, removed, or reordered. The v-bind directive (shorthand :) dynamically binds HTML attributes to expressions — for example, :src="imageUrl" sets the image source from a reactive variable, :class="{ active: isActive }" conditionally applies CSS classes, and :style="{ color: textColor }" sets inline styles dynamically. The v-on directive (shorthand @) attaches event listeners — @click="handleClick" responds to clicks, @input="onInput" handles input events, and modifiers like @click.prevent (calls event.preventDefault()) and @keyup.enter (only triggers on the Enter key) reduce boilerplate. The v-model directive is Vue's signature feature for two-way data binding — it synchronizes a form input's value with a reactive variable, automatically updating the variable when the user types and updating the input when the variable changes programmatically.

**2. The Vue Component System**

Vue's component system enables you to build applications from small, isolated, reusable pieces. Single File Components (.vue files) co-locate the template, script, and scoped styles of a component in a single file, providing an excellent developer experience with clear structure and IDE support. Data flows from parent to child through props — you define the props a component accepts using defineProps() (in script setup syntax), specifying types, default values, and whether each prop is required. Data flows from child to parent through custom events — the child calls emit() with an event name and optional payload, and the parent listens with @eventName="handler". Slots provide a content projection mechanism where a parent can inject arbitrary template content into designated locations within a child component — default slots receive any content placed between the component's tags, while named slots allow multiple content areas. For cases where props drilling becomes cumbersome, Vue provides the provide/inject API — an ancestor component "provides" a value, and any descendant (regardless of depth) can "inject" it, functioning similarly to React's Context API.`,
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
					Content: `Vue Router and Pinia are the two official companion libraries that handle navigation and state management in Vue applications, respectively. Together with Vue itself, they form the core of the Vue ecosystem and are designed to work seamlessly with Vue's reactivity system, component model, and developer tooling. Understanding both is essential for building any non-trivial Vue application.

**1. Vue Router: Client-Side Navigation**

Vue Router provides full-featured client-side routing for Vue single-page applications. You define your routes as an array of objects, each mapping a URL path to a Vue component. The router intercepts navigation events and renders the appropriate component without a full page reload, creating a fast, app-like user experience. Dynamic route segments (like /users/:id) allow a single route definition to handle many URLs, with the dynamic values accessible via the useRoute() composable. Nested routes let you define hierarchical layouts — for example, a /dashboard route might provide a shared navigation sidebar, with child routes like /dashboard/profile and /dashboard/settings rendering their content in a router-view outlet within the dashboard layout. Navigation guards are one of Vue Router's most powerful features — they are hooks that run before, during, or after navigation, allowing you to implement authentication checks (redirecting unauthenticated users to login), data prefetching (loading data before a route component renders), unsaved changes warnings (confirming navigation away from a form with unsaved data), and analytics tracking. The useRouter() composable provides programmatic navigation for cases where you need to navigate in response to code logic rather than user clicks — for example, redirecting to a dashboard after a successful API login call.

**2. Pinia: Modern State Management for Vue**

Pinia is Vue's officially recommended state management library, replacing the older Vuex library. It was designed from the ground up for Vue 3's Composition API and provides a dramatically simpler, more intuitive developer experience. You define a store using defineStore(), which accepts a name (for DevTools identification) and either an options object (with state, getters, and actions) or a setup function (using Composition API patterns like ref and computed). State is defined as a function returning an object of reactive values. Getters are computed properties derived from state — they are cached and only re-evaluate when their dependencies change. Actions are methods that can contain any logic, including asynchronous operations, and can directly mutate state (unlike Vuex, which required separate mutations and actions). Pinia stores are modular by design — each store is independent and can import and use other stores, enabling clean separation of concerns. Pinia provides excellent TypeScript support with full type inference, integrates seamlessly with Vue DevTools (allowing you to inspect state, track mutations, and time-travel debug), and supports hot module replacement so you can modify stores without losing state during development.

**3. Practical Integration Patterns**

In practice, Vue Router and Pinia often work together. Navigation guards frequently access Pinia stores to check authentication status or load required data before rendering a route. Stores can use the router to programmatically navigate after completing operations like login or logout. Route parameters from Vue Router are commonly used as inputs to Pinia actions for fetching route-specific data. The composition of these libraries creates a clean architecture: Vue Router handles what the user sees based on the URL, Pinia manages the application's data and business logic, and Vue components bind it all together into a reactive user interface.`,
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
					Content: `Good component architecture is the difference between a frontend codebase that scales gracefully as features are added and a codebase that becomes an unmaintainable tangle of tightly coupled, hard-to-understand code. Thoughtful component design — guided by established principles and patterns — ensures that your application remains readable, testable, and adaptable to changing requirements over time. These principles apply across all component-based frameworks (React, Vue, Angular, Svelte) and are among the most valuable skills a frontend developer can develop.

**1. Foundational Design Principles**

The Single Responsibility Principle states that each component should have one clear reason to exist — one job it does well. A component that fetches data, transforms it, handles user input, and renders a complex UI is doing too many things and should be broken into smaller, focused pieces. Composition over inheritance means building complex behavior by combining simple components together rather than creating deep inheritance hierarchies — in modern frontend development, you achieve reuse through component nesting and hooks/composables rather than class inheritance. The "props down, events up" principle (also called unidirectional data flow) establishes that data should flow from parent to child through props, and children should communicate changes back to parents through events or callbacks, never by directly modifying parent state. Separation of concerns means keeping different types of logic — data fetching, business rules, and visual presentation — in different components or layers, making each piece easier to understand, test, and replace. Reusability means designing components to be generic enough to use in multiple contexts — a well-designed Button component should work in a form, a navigation bar, or a dialog without modification.

**2. Component Classification and Roles**

Organizing components by their role creates a clear mental model for your codebase. Presentational components (also called "dumb" or "UI" components) focus purely on how things look — they receive data and callbacks through props and render UI, with no knowledge of where the data comes from or what happens when the user interacts with them. Examples include buttons, cards, lists, and form fields. Container components (also called "smart" or "connected" components) handle data fetching, state management, and business logic, then pass the results to presentational components for rendering. This separation means your UI components are highly reusable (they work with any data source) and your logic components are easy to test (they do not depend on specific UI implementations). Layout components define the structural arrangement of a page or section — grids, sidebars, headers, and page shells — without being concerned with the content they contain. Higher-order components are functions that wrap a component with additional behavior, though in modern development, custom hooks (React) or composables (Vue) have largely replaced this pattern.

**3. Advanced Composition Patterns**

Compound components work together as a cohesive group, sharing implicit state — for example, a Tabs component with Tab children, where clicking a Tab automatically updates which content panel is displayed, without the consumer needing to manage the active state manually. Render props allow a component to delegate its rendering logic to a function passed as a prop, enabling maximum flexibility in how data and behavior are presented. The controlled versus uncontrolled pattern determines whether a component manages its own internal state or receives its state entirely from its parent — controlled components are more predictable and testable, while uncontrolled components are simpler for basic use cases. Lifting state up is the practice of moving shared state to the nearest common ancestor of the components that need it, ensuring a single source of truth — when two sibling components need to coordinate, their shared state should live in their parent rather than being duplicated.`,
					CodeExamples: `// Presentational component
function Button({ label, onClick, variant = "primary" }) {
    return (
        <button 
            className={'btn btn-' + variant}
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
					Content: `Build tools are an essential part of the modern frontend development workflow, responsible for bundling, transforming, and optimizing your source code so that it can be efficiently delivered to browsers in production. As web applications have grown in complexity — with hundreds of modules, TypeScript compilation, CSS preprocessing, image optimization, and more — build tools have become indispensable for managing this complexity and ensuring fast load times for end users. Webpack and Vite are the two most important build tools in the frontend ecosystem today, each representing a different generation and philosophy of tooling.

**1. Webpack: The Veteran Module Bundler**

Webpack is a highly configurable, plugin-driven module bundler that has been the backbone of frontend build systems for nearly a decade. At its core, Webpack takes your application's entry point (typically a JavaScript file), recursively follows all import and require statements to discover every module your application depends on, and bundles everything into one or more output files that browsers can load. What makes Webpack so powerful — and so complex — is its loader and plugin system. Loaders are transformations applied to individual files as they are processed: babel-loader transpiles modern JavaScript and JSX into browser-compatible code, css-loader and style-loader handle CSS imports, ts-loader or fork-ts-checker-webpack-plugin processes TypeScript, and file-loader or asset modules handle images and fonts. Plugins operate at a higher level, hooking into the build lifecycle to perform tasks like generating HTML files (HtmlWebpackPlugin), extracting CSS into separate files (MiniCssExtractPlugin), defining environment variables (DefinePlugin), or analyzing bundle contents (BundleAnalyzerPlugin). Webpack's code splitting capability is one of its most valuable features — it allows you to break your bundle into smaller chunks that can be loaded on demand, dramatically improving initial page load time for large applications. Dynamic imports (import()) create automatic split points, and the SplitChunksPlugin optimizes how shared dependencies are grouped. Tree shaking, enabled by ES module static analysis, removes unused exports from your bundles, reducing file sizes. While Webpack's configuration can be daunting for beginners, its maturity means it has solutions for virtually every build scenario and an enormous ecosystem of loaders and plugins.

**2. Vite: The Modern, Lightning-Fast Alternative**

Vite (French for "fast") represents a fundamentally different approach to frontend tooling, designed from the ground up to leverage modern browser capabilities for a dramatically faster development experience. During development, Vite does not bundle your code at all — instead, it serves your source files directly as native ES modules, letting the browser handle module resolution and loading. This means that no matter how large your application grows, the dev server starts in milliseconds because there is no bundling step. When you edit a file, Vite uses Hot Module Replacement (HMR) to update only the changed module in the browser, and because it does not need to rebuild a bundle, HMR updates are nearly instantaneous even in massive codebases. Vite achieves its speed by using esbuild (written in Go) for dependency pre-bundling and TypeScript/JSX transformation, which is 10-100x faster than JavaScript-based tools like Babel. For production builds, Vite switches to Rollup, which produces highly optimized, tree-shaken bundles with efficient code splitting. Vite provides first-class support for TypeScript, JSX, CSS Modules, CSS preprocessors (SASS, Less, Stylus), static assets, and JSON — most features work out of the box with zero configuration. The plugin API is based on Rollup's plugin interface, making it familiar and well-documented. Vite has become the recommended build tool for React, Vue, Svelte, and many other frameworks, and its combination of blazing-fast development experience with production-quality output has made it the clear choice for new projects.

**3. Key Build Concepts Shared by Both Tools**

Regardless of which build tool you choose, several core concepts apply universally. Module bundling is the process of combining many separate source files into fewer output files, reducing the number of HTTP requests the browser needs to make. Code transformation converts modern syntax (ES2024+, TypeScript, JSX) into code that target browsers can execute, ensuring broad compatibility. Asset optimization includes minifying JavaScript and CSS (removing whitespace, shortening variable names), compressing images, and generating optimized formats. The development server provides a local environment with features like hot module replacement (updating code in the browser without a full page reload), source maps (mapping bundled code back to original source files for debugging), and proxy configuration (forwarding API requests to a backend server during development). Production builds apply aggressive optimizations — minification, tree shaking, code splitting, content hashing (for cache busting), and compression — to produce the smallest, fastest-loading bundles possible.`,
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
					Content: `Beyond Webpack and Vite, the frontend ecosystem offers several other build tools, each with its own philosophy, strengths, and ideal use cases. Understanding the landscape of modern build tools helps you make informed decisions about which tool to use for different types of projects — whether you are building a complex web application, publishing a reusable library, or prototyping a quick idea. The trend in recent years has been toward dramatically faster build times, achieved by rewriting core compilation logic in systems languages like Go and Rust rather than JavaScript.

**1. Parcel: Zero-Configuration Bundling**

Parcel is a build tool designed around the principle that you should not have to write any configuration to get started. You simply point Parcel at your HTML entry file, and it automatically discovers and bundles all of your JavaScript, CSS, images, and other assets by following the references in your code. Parcel automatically detects which transformations are needed based on your source files — if it encounters TypeScript, it compiles it; if it finds SCSS, it processes it; if it sees modern JavaScript syntax, it transpiles it for target browsers defined in your browserslist configuration. Code splitting happens automatically at dynamic import() boundaries, and Parcel applies scope hoisting (a technique that inlines modules into a single scope to reduce overhead) for smaller output. Parcel uses a multi-core compilation architecture that parallelizes work across CPU cores, resulting in fast builds even for large projects. Its caching system is also highly effective — subsequent builds after the first are significantly faster because Parcel caches the results of every transformation. Parcel is an excellent choice for projects where you want to focus on writing code rather than configuring build tools, for rapid prototyping, and for smaller applications where the overhead of Webpack configuration is not justified. However, for very large or complex projects that require fine-grained control over the build process, Webpack or Vite may offer more flexibility.

**2. Rollup: The Library Bundler**

Rollup is a module bundler that was specifically designed to produce clean, efficient ES module output, making it the gold standard for bundling JavaScript libraries and packages. While Webpack was designed primarily for applications (where you want everything in a single bundle), Rollup excels at producing library output in multiple formats — ES modules (for modern bundlers that can tree-shake), CommonJS (for Node.js), and UMD (for direct browser use via script tags). Rollup pioneered tree shaking in the JavaScript ecosystem, using static analysis of ES module import/export statements to determine which exports are actually used and removing all dead code from the final bundle. This results in remarkably small output files. Rollup's plugin interface is clean and well-documented, and it has become the de facto standard — Vite's plugin API is based on Rollup's, meaning plugins are often compatible with both tools. For application development, Rollup is less commonly used directly (Vite wraps it for production builds), but for publishing npm packages, component libraries, and utility libraries, Rollup remains the recommended choice because of its superior output quality and multi-format support.

**3. esbuild and Turbopack: The Speed Revolution**

esbuild is an extremely fast JavaScript bundler and minifier written in Go, capable of processing code 10 to 100 times faster than traditional JavaScript-based tools like Webpack, Rollup, or Babel. It achieves this speed through several technical advantages: it is written in a compiled language (Go) that produces native machine code, it uses heavy parallelism across CPU cores, it performs minimal AST transformations (avoiding the overhead of full Babel-style plugin chains), and it uses efficient memory management. esbuild supports TypeScript and JSX natively, handles tree shaking and minification, and can output ES modules, CommonJS, or IIFE formats. While esbuild is used directly in some projects, its most significant impact has been as the engine behind other tools — Vite uses esbuild for dependency pre-bundling and TypeScript compilation during development, and many CI/CD pipelines use esbuild for fast production builds. Turbopack is a newer entrant developed by Vercel (the company behind Next.js), written in Rust for maximum performance. It is designed as an incremental computation engine — rather than rebuilding everything when a file changes, it tracks fine-grained dependencies and recomputes only the minimal necessary work. Turbopack is integrated into Next.js as an alternative to Webpack for development, promising startup times and HMR speeds that scale well even in very large applications. As of now, Turbopack is focused on the Next.js ecosystem, but its Rust-based architecture represents the direction the industry is moving.

**4. Choosing the Right Build Tool for Your Project**

The choice of build tool depends on what you are building and what you value most. For modern web applications with a focus on developer experience and fast iteration, Vite is the clear winner — its instant dev server, excellent framework support, and production-quality output make it the best all-around choice for new projects. For complex enterprise applications with highly specific build requirements, advanced code-splitting strategies, or a large existing plugin ecosystem to leverage, Webpack remains a solid and battle-tested option. For publishing libraries and npm packages, Rollup produces the cleanest and most compatible output across module formats. For projects that prioritize simplicity and want zero configuration overhead, Parcel gets you from zero to a running application faster than any other tool. If raw build speed is your primary concern — for example, in large monorepos or CI/CD pipelines where build time directly impacts developer productivity — esbuild or tools built on top of it offer unmatched performance.`,
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
					Content: `Testing is a critical discipline in frontend development that ensures your code works correctly, prevents regressions when changes are made, and gives developers the confidence to refactor and improve the codebase without fear of breaking existing functionality. In the frontend world, testing is especially important because user interfaces are inherently complex — they must handle a wide variety of user interactions, screen sizes, browser quirks, network conditions, and asynchronous data flows. A comprehensive testing strategy combines multiple types of tests at different levels of abstraction, each serving a distinct purpose in verifying that your application behaves as expected.

**1. The Testing Pyramid: Types of Frontend Tests**

The testing pyramid is a widely adopted model that describes the ideal distribution of tests across three main levels. At the base are unit tests, which verify that individual functions, utilities, and small pieces of logic work correctly in isolation. Unit tests are the fastest to write and execute, and they provide precise feedback about what broke and where. For example, a unit test might verify that a formatDate() utility correctly converts a timestamp to a human-readable string, or that a calculateTotal() function accurately sums line items with tax. In the middle of the pyramid are integration tests (also called component tests in the frontend context), which verify that multiple pieces work correctly together. In React, an integration test might render a component with its children, simulate user interactions (clicking buttons, typing in inputs), and assert that the rendered output changes as expected. Integration tests are particularly valuable in frontend development because they test the component as users actually experience it — including how it handles props, state changes, side effects, and event handling. At the top of the pyramid are end-to-end (E2E) tests, which simulate real user workflows by driving an actual browser through complete scenarios — logging in, navigating between pages, filling out forms, and verifying that the correct data appears on screen. E2E tests are the most realistic but also the slowest and most brittle, so you should have fewer of them, focused on the most critical user journeys. Snapshot tests are a supplementary technique where you capture the rendered output of a component (as a serialized HTML or JSON structure) and compare it against a previously saved "snapshot" — any differences cause the test to fail, making it easy to detect unintended UI changes.

**2. Essential Testing Tools and Frameworks**

The frontend testing ecosystem offers specialized tools for each level of the testing pyramid. Jest is the most popular JavaScript testing framework, providing a complete testing solution with a test runner, assertion library, mocking capabilities, code coverage reporting, and snapshot testing — all with zero configuration for most projects. Vitest is a newer alternative built specifically for Vite-powered projects, offering a Jest-compatible API with dramatically faster execution thanks to Vite's native ES module support and esbuild-powered transformation. React Testing Library (part of the broader Testing Library family, which also supports Vue, Angular, and Svelte) has become the standard for component testing — it provides utilities for rendering components, querying the DOM using accessible selectors (like getByRole, getByText, and getByLabelText), and simulating user events. Its philosophy of testing components the way users interact with them (rather than testing internal implementation details) leads to more resilient tests that do not break when you refactor component internals. For mocking, Jest and Vitest both provide built-in capabilities to mock modules, functions, and timers, allowing you to isolate the code under test from external dependencies like API calls, browser APIs, or third-party libraries.

**3. Best Practices for Writing Effective Frontend Tests**

The most important principle in frontend testing is to test behavior, not implementation. This means your tests should assert what the user sees and experiences — text content, element visibility, navigation outcomes — rather than internal details like state values, method calls, or component structure. Tests that focus on behavior remain valid even when you refactor the underlying code, while implementation-focused tests break on every refactor, creating maintenance burden without adding safety. Keep each test focused on a single behavior or scenario, with a clear arrange-act-assert structure: set up the necessary state (arrange), perform the action being tested (act), and verify the expected outcome (assert). Mock external dependencies like API calls, timers, and browser APIs to keep tests fast, deterministic, and independent of external systems. Use descriptive test names that read like specifications — "should display an error message when the email field is empty" is far more useful than "test validation." Run your tests in continuous integration (CI) to catch regressions automatically on every code change, and maintain a fast test suite by keeping the majority of your tests at the unit and integration levels where execution is quick.`,
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
					Content: `End-to-end (E2E) testing tools automate a real browser to simulate complete user workflows — clicking buttons, filling out forms, navigating between pages, and verifying that the application behaves correctly from the user's perspective. Unlike unit tests or component tests, which test isolated pieces of code, E2E tests exercise the full application stack including the frontend, backend APIs, database, and any third-party integrations. This makes them the most realistic form of testing, but also the most complex to set up, maintain, and debug. Cypress and Playwright are the two dominant E2E testing tools in the modern frontend ecosystem, and understanding their strengths helps you choose the right one for your project.

**1. Cypress: Developer-Focused Browser Testing**

Cypress is an E2E testing framework that was designed from the ground up with developer experience as a top priority. Unlike older tools like Selenium that control the browser from outside, Cypress runs directly inside the browser alongside your application code, giving it direct access to the DOM, network layer, and application state. This architecture enables several unique features. Time travel debugging is one of Cypress's most loved features — as your test runs, Cypress takes a snapshot of the DOM at every command step, and you can hover over any step in the test log to see exactly what the application looked like at that point. This makes debugging failing tests dramatically easier than reading through logs or adding screenshots manually. Cypress automatically waits for elements to appear, animations to complete, and network requests to finish before proceeding with the next command, eliminating the flaky "sleep" statements that plague other testing tools. The built-in test runner provides a visual interface where you can watch your tests execute in real time, inspect element states, and re-run specific tests instantly. Cypress supports network interception (cy.intercept()), allowing you to stub API responses to test specific scenarios without depending on a live backend — you can simulate error responses, slow networks, or specific data payloads. One important limitation to understand is that Cypress historically only supported Chromium-based browsers, though it has since added Firefox and WebKit (Safari) support. Cypress runs tests within a single browser tab, which means testing multi-tab workflows or cross-origin navigation requires workarounds.

**2. Playwright: Cross-Browser Testing with Modern Architecture**

Playwright, developed by Microsoft, is a newer E2E testing framework that was built by members of the team that previously created Puppeteer (Google's browser automation library). Playwright's standout feature is true cross-browser support — it can run tests in Chromium (Chrome, Edge), Firefox, and WebKit (Safari) using a single API, ensuring your application works correctly across all major browser engines. This is achieved through direct communication with browser engines via the Chrome DevTools Protocol (for Chromium) and custom protocols (for Firefox and WebKit), bypassing the limitations of older driver-based approaches. Playwright provides powerful auto-waiting behavior, where every action automatically waits for the target element to be visible, enabled, and stable before interacting with it, significantly reducing test flakiness. The locator API encourages writing resilient selectors based on user-facing attributes (like text content, roles, and labels) rather than brittle CSS selectors or XPaths. Playwright supports testing across multiple browser contexts (isolated browser sessions) and even multiple pages simultaneously, enabling scenarios like testing real-time collaboration features, multi-user workflows, or popup windows. The built-in tracing feature records a detailed log of every action, screenshot, network request, and console message during test execution, producing a trace file that you can open in Playwright's Trace Viewer for post-mortem debugging. Playwright also provides built-in support for visual regression testing through screenshot comparison, API testing for verifying backend endpoints, and component testing for rendering individual components in isolation.

**3. Best Practices for E2E Testing**

Effective E2E testing requires a disciplined approach because these tests are inherently more expensive to write, run, and maintain than unit or integration tests. Focus your E2E tests on critical user journeys — the workflows that are most important to your business and most likely to break. Examples include authentication flows (signup, login, password reset), core feature paths (creating a resource, completing a purchase, submitting a form), and key navigation patterns (accessing protected routes, handling deep links). Keep each test independent and self-contained — tests should not depend on the state left behind by other tests, because this coupling makes tests order-dependent and extremely difficult to debug when they fail. Use stable, semantic selectors for locating elements — data-testid attributes provide reliable anchors that are immune to styling changes and refactoring, while role-based and text-based selectors (preferred by Testing Library and Playwright's locator API) match how users identify elements. Implement network interception to control API responses, enabling you to test error states, loading states, and edge cases without depending on specific backend data. Use both screenshot and video recording capabilities (supported by both Cypress and Playwright) to capture visual evidence of test execution, which is invaluable for debugging failures in CI environments where you cannot watch the test run interactively. Finally, run your E2E tests in your CI/CD pipeline but keep the suite focused and fast — a slow E2E suite that developers skip or ignore provides no value.`,
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
