package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1530,
			Title:       "Angular Basics",
			Description: "Introduction to Angular: components, services, dependency injection, and Angular fundamentals.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Angular Introduction",
					Content: `Angular is a comprehensive, opinionated platform and framework for building enterprise-grade single-page applications. Developed and maintained by Google, Angular provides everything you need out of the box — routing, forms, HTTP communication, dependency injection, and a powerful CLI — making it the go-to choice for large organizations that value structure, consistency, and long-term maintainability over maximum flexibility. Unlike React (which is a library focused on the view layer), Angular is a full framework with strong opinions about how applications should be structured.

**1. What is Angular?**

Angular is built entirely on TypeScript, which means you get type safety, excellent IDE support, and compile-time error catching from day one. Its component-based architecture breaks the UI into reusable, self-contained pieces, each with its own template, styles, and logic. The dependency injection (DI) system is one of Angular's most distinctive features — it provides a built-in mechanism for supplying components with the services and dependencies they need, making code more modular, testable, and maintainable. Angular includes a full-featured router for navigation between views, a powerful forms module with both template-driven and reactive approaches, and an HTTP client for communicating with backend APIs. This "batteries included" approach means teams spend less time evaluating and integrating third-party libraries and more time building features.

**2. Key Concepts**

Components are the fundamental building blocks of Angular applications. Each component consists of a TypeScript class (the logic), an HTML template (the view), and optional CSS (the styles). Services encapsulate business logic, data access, and other functionality that does not belong in components — they are injectable classes that follow the single responsibility principle. Modules (NgModules) organize related components, services, directives, and pipes into cohesive functional units, making large applications easier to manage and enabling lazy loading of feature modules. Directives extend HTML with custom behavior — structural directives like *ngIf and *ngFor modify the DOM structure, while attribute directives like [ngClass] modify the appearance or behavior of existing elements. Pipes transform data for display in templates — formatting dates, currencies, percentages, or applying custom transformations without modifying the underlying data.

**3. Angular CLI — Your Development Companion**

The Angular CLI is a powerful command-line tool that automates the development workflow. Running ng new creates a complete, properly configured project with TypeScript compilation, testing frameworks, linting, and a development server. The ng generate command scaffolds components, services, modules, pipes, guards, and more — all following Angular's conventions and best practices, complete with boilerplate code and test files. The ng serve command launches a development server with hot module replacement, so changes appear instantly in the browser. The ng build command produces optimized production bundles with ahead-of-time (AOT) compilation, tree shaking, and minification. The CLI also handles updates (ng update), testing (ng test, ng e2e), and linting (ng lint), providing a unified toolchain that keeps the entire team on the same page.`,
					CodeExamples: `// Component
import { Component } from '@angular/core';

@Component({
    selector: 'app-user',
    template: '<h1>{{ name }}</h1>',
    styleUrls: ['./user.component.css']
})
export class UserComponent {
    name = 'John Doe';
}

// Service
import { Injectable } from '@angular/core';

@Injectable({
    providedIn: 'root'
})
export class UserService {
    getUsers() {
        return ['John', 'Jane'];
    }
}

// Using service
constructor(private userService: UserService) {}

ngOnInit() {
    const users = this.userService.getUsers();
}

// Module
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

@NgModule({
    declarations: [UserComponent],
    imports: [BrowserModule],
    providers: [UserService],
    bootstrap: [AppComponent]
})
export class AppModule { }`,
				},
				{
					Title: "Angular Components and Data Binding",
					Content: `Angular's data binding system is one of its most powerful features, creating a seamless connection between the component's TypeScript class and its HTML template. Understanding the different types of data binding and when to use each one is fundamental to building responsive, interactive Angular applications. The framework also provides a rich component lifecycle and a set of built-in directives that handle the most common rendering patterns.

**1. Data Binding — The Four Types**

Angular provides four distinct types of data binding, each serving a different purpose. Interpolation ({{ value }}) is the simplest — it evaluates a TypeScript expression and inserts the result as text into the template. Think of it as a one-way street from the component class to the view. Property binding ([property]="value") sets an HTML element property or an Angular directive/component input to a value from the component — for example, binding the disabled attribute of a button to a boolean property. Event binding ((event)="handler()") listens for DOM events and calls component methods when they fire — clicks, key presses, form submissions, and custom events. Two-way binding ([(ngModel)]="value") combines property and event binding, keeping the component property and the form input in perfect sync — when the user types in an input field, the property updates immediately, and when the property changes programmatically, the input reflects the new value. This syntactic sugar, sometimes called "banana in a box" for its bracket-parenthesis syntax, is essential for building forms.

**2. Component Lifecycle Hooks**

Angular components go through a well-defined lifecycle from creation to destruction, and lifecycle hooks let you tap into key moments in that lifecycle. ngOnInit fires after Angular has set all input properties for the first time — this is where you should perform initialization logic like fetching data, not in the constructor (which should be reserved for simple dependency injection). ngOnChanges fires every time an input property changes, providing both the current and previous values — useful for reacting to configuration changes from parent components. ngAfterViewInit fires after Angular has fully initialized the component's view and child views — this is where you can safely access view children and perform DOM measurements. ngOnDestroy fires just before Angular destroys the component — this is where you must clean up subscriptions, timers, and event listeners to prevent memory leaks. Understanding this lifecycle is crucial for writing performant, bug-free components.

**3. Built-in Directives**

Angular's structural directives modify the DOM structure based on conditions. The *ngIf directive conditionally renders an element and its children — when the expression is falsy, Angular removes the element entirely from the DOM (not just hides it), which is important for performance. The *ngFor directive iterates over a collection and stamps out a template for each item, supporting trackBy functions for efficient list updates. Attribute directives modify the appearance or behavior of existing elements. The [ngClass] directive dynamically adds and removes CSS classes based on an expression — you can pass it a string, an array, or an object where keys are class names and values are booleans. The [ngStyle] directive works similarly for inline styles. These directives handle the vast majority of dynamic rendering needs, and when they are not sufficient, you can create custom directives for specialized DOM manipulation.`,
					CodeExamples: `// Component with data binding
@Component({
    selector: 'app-user',
    template: '<h1>{{ user.name }}</h1>' +
        '<input [value]="user.email" (input)="onEmailChange($event)">' +
        '<button [disabled]="!isValid" (click)="save()">Save</button>' +
        '<div *ngIf="showDetails"><p>{{ user.bio }}</p></div>' +
        '<ul><li *ngFor="let item of items">{{ item }}</li></ul>'
})
export class UserComponent {
    user = { name: 'John', email: 'john@example.com' };
    isValid = true;
    showDetails = false;
    items = ['Item 1', 'Item 2'];
    
    onEmailChange(event: any) {
        this.user.email = event.target.value;
    }
    
    save() {
        // Save logic
    }
}

// Two-way binding
<input [(ngModel)]="user.name">

// Lifecycle hooks
export class UserComponent implements OnInit, OnDestroy {
    ngOnInit() {
        console.log('Component initialized');
    }
    
    ngOnDestroy() {
        console.log('Component destroyed');
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1531,
			Title:       "Svelte and SvelteKit",
			Description: "Learn Svelte: reactive framework, SvelteKit, and modern web development.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Svelte Introduction",
					Content: `Svelte takes a fundamentally different approach to building user interfaces compared to React, Vue, and Angular. While those frameworks do the bulk of their work in the browser using a runtime (virtual DOM diffing, change detection, or reactivity tracking), Svelte shifts that work to compile time. When you build a Svelte application, the compiler transforms your declarative component code into highly optimized imperative JavaScript that surgically updates the DOM — no virtual DOM, no runtime overhead, no framework code shipped to the browser. The result is remarkably small bundle sizes and blazing-fast performance.

**1. What Makes Svelte Different?**

The key insight behind Svelte is that frameworks can be a compile step rather than a runtime dependency. When you write a Svelte component, the compiler analyzes your code and generates the minimal JavaScript needed to keep the DOM in sync with your state. Instead of diffing a virtual DOM tree on every state change (as React does), Svelte generates code that knows exactly which DOM nodes need to update when a specific variable changes. This means there is no overhead from a runtime reconciliation algorithm, no virtual DOM memory allocation, and no unnecessary re-renders. The resulting bundles are typically 30-50% smaller than equivalent React applications, and runtime performance is consistently among the fastest of any framework. Reactivity is built into the language itself — simply assigning a value to a variable triggers an update, with no useState, no this.setState, and no special API to learn.

**2. Key Features**

Reactive declarations using the $: label syntax (borrowed from JavaScript's rarely-used label syntax) let you declare derived values and side effects that automatically re-run when their dependencies change — similar to computed properties and watchers in Vue, but with a more concise syntax. Svelte stores provide a simple, built-in state management solution for sharing state across components without prop drilling. Stores are just objects with a subscribe method, and Svelte provides writable, readable, and derived store types. The $ prefix syntax automatically subscribes and unsubscribes components to stores. Built-in transitions and animations make it trivial to add entrance, exit, and layout animations with just a directive — no third-party library needed. Svelte also includes built-in accessibility warnings that alert you during compilation if your templates have common a11y issues.

**3. SvelteKit — The Full-Stack Framework**

SvelteKit is to Svelte what Next.js is to React — a full-stack framework built on top of the core UI library. It provides file-based routing where your file structure defines your URL paths, server-side rendering for improved SEO and initial load performance, API routes for building backend endpoints, and a flexible adapter system that lets you deploy to virtually any hosting platform (Vercel, Netlify, Cloudflare Workers, Node.js, static hosting). SvelteKit supports multiple rendering strategies per route — SSR, SSG, or client-side rendering — and its load functions provide a clean pattern for fetching data on both the server and client. The build output is highly optimized with automatic code splitting, asset hashing, and preloading.`,
					CodeExamples: `<!-- Svelte component -->
<script>
    let count = 0;
    let doubled = $derived(count * 2);
    
    function increment() {
        count++;
    }
</script>

<button on:click={increment}>
    Count: {count}, Doubled: {doubled}
</button>

<!-- Reactive statements -->
<script>
    let name = 'world';
    $: greeting = 'Hello, ' + name + '!';
    $: console.log('Name changed:', name);
</script>

<h1>{greeting}</h1>

<!-- Stores -->
<script>
    import { writable } from 'svelte/store';
    
    const count = writable(0);
    $: doubled = $count * 2;
</script>

<button on:click={() => count.update(n => n + 1)}>
    Count: {$count}
</button>

<!-- SvelteKit page -->
<script>
    export let data;
</script>

<h1>{data.title}</h1>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1532,
			Title:       "GraphQL and Apollo",
			Description: "Master GraphQL: queries, mutations, Apollo Client, and modern data fetching.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "GraphQL Fundamentals",
					Content: `GraphQL is a query language for APIs and a runtime for executing those queries, developed by Facebook and open-sourced in 2015. It fundamentally changes the relationship between frontend and backend by letting the client specify exactly what data it needs, in the shape it needs it. Instead of the traditional REST approach where the server dictates the response structure and the client must make multiple requests to assemble a view, GraphQL lets the frontend developer write a single query that describes the exact data requirements for a component or page.

**1. How GraphQL Works**

Unlike REST APIs that expose multiple endpoints (GET /users, GET /users/123/posts, GET /posts/456/comments), GraphQL exposes a single endpoint that accepts queries describing the desired data shape. The client sends a query that mirrors the structure of the response it wants to receive — if you need a user's name, email, and the titles of their last five posts, you write a query that specifies exactly those fields. The server resolves the query against its schema and returns a JSON response that matches the query structure precisely. This eliminates two common REST problems: over-fetching (getting back 50 fields when you only need 3) and under-fetching (needing to make 5 separate requests to assemble one view). The schema is strongly typed, defining every type, field, and relationship in the API, which enables powerful tooling like autocomplete, validation, and automatic documentation.

**2. Key Concepts**

Queries are read operations that fetch data — they are the GraphQL equivalent of GET requests. Mutations are write operations that create, update, or delete data — equivalent to POST, PUT, and DELETE. Subscriptions enable real-time functionality by maintaining a persistent connection (typically over WebSocket) and pushing updates to the client when relevant data changes — perfect for chat messages, live notifications, or collaborative editing. The schema defines the complete type system of your API: what types exist, what fields they have, what types those fields return, and what arguments queries and mutations accept. Resolvers are the functions on the server that actually fetch the data for each field in the schema — they are the bridge between the GraphQL type system and your data sources (databases, REST APIs, microservices).

**3. Benefits for Frontend Development**

The benefits for frontend teams are transformative. Fetching only the needed data means mobile clients can request minimal payloads while desktop clients request richer data — all from the same endpoint, with no backend changes. Getting multiple related resources in a single request eliminates the waterfall of sequential REST calls that slow down page loads. The strong type system catches errors at development time and enables incredible tooling: code generation tools can automatically create TypeScript types and React hooks from your GraphQL queries. Introspection means the API is self-documenting — tools like GraphQL Playground and GraphiQL let developers explore the entire API interactively. Perhaps most importantly, GraphQL decouples frontend and backend evolution: the backend can add new fields and types without breaking existing clients, and the frontend can start using new fields without waiting for a new API version.`,
					CodeExamples: `// GraphQL Query
// Note: GraphQL queries use $ prefix for variables (e.g., $id: ID!)
// Example:
// query GetUser($id: ID!) {
//     user(id: $id) {
//         id
//         name
//         email
//         posts {
//             title
//             content
//         }
//     }
// }
//
// GraphQL Mutation
// mutation CreateUser($input: UserInput!) {
//     createUser(input: $input) {
//         id
//         name
//         email
//     }
// }

// Apollo Client
import { ApolloClient, InMemoryCache, gql } from '@apollo/client';

const client = new ApolloClient({
    uri: 'https://api.example.com/graphql',
    cache: new InMemoryCache()
});

// Query with Apollo
// Note: GraphQL queries use template literals in JavaScript
// Example (using template literal syntax):
// const GET_USERS = gql(template literal)query GetUsers { users { id name } };
// const { data, loading, error } = useQuery(GET_USERS);
//
// Mutation with Apollo
// Example (using template literal syntax):
// const CREATE_USER = gql(template literal)mutation CreateUser(nameParam: String!) { createUser(name: nameParam) { id name } };
// const [createUser] = useMutation(CREATE_USER);
// Note: GraphQL variable syntax uses $ prefix for variables (e.g., $name: String!)`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1533,
			Title:       "CSS Preprocessors (SASS/SCSS)",
			Description: "Master SASS/SCSS: variables, mixins, functions, and advanced CSS preprocessing.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "SASS Advanced Features",
					Content: `Advanced SASS features transform CSS authoring from a manual, repetitive process into a powerful, programmable system. While basic SASS gives you variables, nesting, and mixins, the advanced features — custom functions, control flow, interpolation, and modular architecture — let you build sophisticated stylesheet systems that generate CSS dynamically, enforce consistency automatically, and scale gracefully across large projects.

**1. Custom Functions and Control Flow**

SASS functions let you encapsulate calculations and transformations into reusable units that return values. Unlike mixins (which output CSS declarations), functions compute and return a single value — making them perfect for unit conversions (pixels to rems), color manipulations (darken, lighten, mix), spacing calculations, and responsive scaling. Control flow directives bring programming logic to your stylesheets. The @if directive enables conditional output — generate different styles based on variables or parameters. The @for directive generates numbered sequences — perfect for creating grid column classes (.col-1 through .col-12) from a single loop. The @each directive iterates over lists and maps — ideal for generating utility classes from a list of colors, sizes, or breakpoints. The @while directive handles more complex iterative patterns. Together, these features let you write a few lines of SASS that generate hundreds of lines of CSS, ensuring consistency and eliminating manual repetition.

**2. Interpolation, Parent Selectors, and Extend**

Interpolation (#{$variable}) lets you insert SASS values into selectors, property names, and string values — making your stylesheets truly dynamic. Combined with loops, interpolation enables patterns like generating .bg-red, .bg-blue, .bg-green classes from a list of color names. The parent selector (&) is deceptively powerful — beyond basic nesting, it enables BEM-style class naming (.block, .block__element, .block--modifier), state-based modifiers (.button.is-active), and context-dependent styles (.dark-theme & to style a component when it is inside a dark theme container). The @extend directive lets one selector inherit the styles of another, generating comma-separated selectors in the output CSS. Placeholder selectors (%placeholder) are designed specifically for @extend — they define reusable style sets that only appear in the output when extended, keeping your CSS clean.

**3. Best Practices for Scalable SASS**

Organize your stylesheets into partials — small, focused files prefixed with an underscore (_variables.scss, _mixins.scss, _buttons.scss) — and compose them with @use (the modern replacement for @import). Create a core set of reusable mixins for common patterns like responsive breakpoints, flexbox centering, truncated text, and accessible focus styles. Use functions for any calculation that appears in more than one place. Avoid nesting beyond three levels deep, as deep nesting generates overly specific selectors that are hard to override and maintain. Keep specificity as low as possible by preferring class selectors over ID selectors and avoiding unnecessary element qualifiers. Following the ITCSS (Inverted Triangle CSS) or 7-1 pattern gives your SASS architecture a clear, scalable structure that new team members can navigate easily.`,
					CodeExamples: `// Functions
@function calculate-rem($pixels, $base: 16px) {
    @return ($pixels / $base) * 1rem;
}

.text {
    font-size: calculate-rem(24px); // 1.5rem
}

// Control flow
@for $i from 1 through 12 {
    .col-#{$i} {
        width: percentage($i / 12);
    }
}

$colors: red, blue, green;
@each $color in $colors {
    .bg-#{$color} {
        background-color: $color;
    }
}

// Extend
%button-base {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
}

.button-primary {
    @extend %button-base;
    background-color: blue;
}

// Advanced mixins
@mixin respond-to($breakpoint) {
    @if $breakpoint == mobile {
        @media (max-width: 767px) { @content; }
    } @else if $breakpoint == tablet {
        @media (min-width: 768px) { @content; }
    }
}

.container {
    padding: 10px;
    @include respond-to(tablet) {
        padding: 20px;
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1534,
			Title:       "CSS-in-JS Libraries",
			Description: "Explore CSS-in-JS: styled-components, Emotion, and component-scoped styling.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "CSS-in-JS Patterns and Best Practices",
					Content: `CSS-in-JS is an approach to styling where CSS is authored directly within JavaScript files, co-located with the components it styles. Libraries like styled-components, Emotion, and Stitches popularized this paradigm by solving several long-standing CSS problems: global namespace collisions, dead code elimination, dynamic styling based on component state, and the disconnect between component logic and its visual presentation. Understanding the patterns, performance implications, and best practices of CSS-in-JS is essential for modern frontend development.

**1. Core Patterns**

Styled components create new React components with styles baked in — you define a Button component with specific padding, colors, and hover states, and use it just like any other React component. This provides true component-scoped styles with automatically generated unique class names, eliminating any risk of CSS conflicts across your application. Theme providers use React's context system to inject a theme object (containing design tokens like colors, spacing, and typography) into all styled components, enabling consistent theming and easy theme switching (like dark mode). Dynamic styling is perhaps the most compelling feature — styled components can read their React props and adjust styles accordingly. A Button component can change its background color based on a variant prop, its size based on a size prop, and its opacity based on a disabled prop — all within the same style definition. Animations use the keyframes helper to define CSS animations that are scoped and reusable. Media queries work naturally within styled component definitions, keeping responsive styles co-located with the component they affect.

**2. Performance Considerations**

CSS-in-JS libraries broadly fall into two categories with very different performance profiles. Runtime libraries (styled-components, Emotion) generate and inject CSS at runtime — they parse your style definitions, resolve dynamic values, and insert style tags into the DOM as components render. This adds overhead, especially during initial render and when many components mount simultaneously. Compile-time libraries (vanilla-extract, Linaria, Panda CSS) extract styles at build time into static CSS files, achieving zero runtime overhead while still providing the authoring benefits of CSS-in-JS. When using runtime libraries, avoid creating styled components inside render functions (which forces style recalculation on every render), memoize styled components that receive complex prop-based styles, and consider using CSS custom properties (CSS variables) for frequently changing values like theme colors, since updating a CSS variable is much cheaper than regenerating a stylesheet.

**3. Best Practices**

Organize styled components in a consistent manner — either co-locate them in the same file as the component they style, or extract them into adjacent .styles.ts files for larger components. Use the theme object consistently for all color, spacing, and typography values rather than hardcoding values, even when it seems faster. Extract common style patterns into shared styled components or style mixins to maintain DRY code — for example, a shared flex-center mixin or a standard card shadow. Test styled components by verifying that the correct styles are applied based on different prop combinations, using tools like jest-styled-components for snapshot testing or visual regression testing with Storybook and Chromatic.`,
					CodeExamples: `// styled-components patterns
// Note: Template literal syntax shown in comments to avoid Go parsing issues
// In JavaScript, styled-components uses template literals (backticks):
// const Button = styled.button(template literal)
//     background: props => props.variant === 'primary' ? 'blue' : 'gray';
//     padding: props => props.size === 'large' ? '15px' : '10px';
//     &:hover { opacity: 0.8; }
//     @media (max-width: 768px) { width: 100%; }

// Theme with TypeScript
interface Theme {
    colors: {
        primary: string;
        secondary: string;
    };
}

// const ThemedButton = styled.button<{ theme: Theme }>(template literal)
//     background: props => props.theme.colors.primary;

// Animations
// const fadeIn = keyframes(template literal)
//     from { opacity: 0; }
//     to { opacity: 1; }

// const FadeInBox = styled.div(template literal)
//     animation: fadeIn 0.5s;`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1535,
			Title:       "React Router and Navigation",
			Description: "Master React Router: routing, navigation, protected routes, and advanced patterns.",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced React Router",
					Content: `React Router is the standard routing library for React applications, and its advanced features enable you to build sophisticated navigation patterns that match the complexity of real-world applications. Beyond simple path-to-component mapping, modern React Router (v6+) provides nested routing, layout management, data-driven routing, and powerful URL-based state management that are essential for building production applications.

**1. Nested Routes and Layout Routes**

Nested routes are one of React Router's most powerful features, allowing you to mirror the hierarchical structure of your UI in your route configuration. When you define a parent route with child routes, the parent component renders an Outlet component that acts as a placeholder for whichever child route matches the current URL. This creates a natural layout system — the parent route renders shared UI (navigation, sidebars, breadcrumbs) while child routes fill in the content area. For example, a /users route renders a UsersLayout with navigation, the /users index route renders a user list, and /users/:id renders user details — all sharing the same layout wrapper without re-mounting. Layout routes (routes without a path that only provide UI structure) take this further, letting you wrap groups of routes with shared layouts, error boundaries, or loading states without affecting the URL structure.

**2. Route Parameters, Query Strings, and Navigation**

Route parameters (:id in /users/:id) capture dynamic segments of the URL, accessible via the useParams hook. Query strings store filter, sort, and pagination state in the URL (e.g., /products?category=shoes&sort=price), managed through the useSearchParams hook. This is critically important because it makes the application state shareable — users can bookmark, share, or navigate back to a specific filtered view. The useLocation hook provides access to the full location object including pathname, search, hash, and state. The Navigate component enables declarative redirects — rendering <Navigate to="/login" /> when authentication fails, for instance. Programmatic navigation via the useNavigate hook handles imperative navigation needs after form submissions, API calls, or other side effects.

**3. Route Guards, Lazy Loading, and Best Practices**

Route guards (also called protected routes) wrap route elements with authentication and authorization checks, redirecting unauthorized users to login or error pages. The pattern typically involves a wrapper component that checks authentication state and either renders its children or navigates to a login route. Lazy loading routes with React.lazy and Suspense is essential for large applications — instead of including every page's code in the initial bundle, each route's component is loaded on demand when the user navigates to it, dramatically reducing initial bundle size. Organize your routes in a centralized configuration file for visibility, implement a catch-all route for 404 pages, use relative navigation (navigate("edit") instead of navigate("/users/123/edit")) for more maintainable code, and preserve scroll position across navigations using the ScrollRestoration component.`,
					CodeExamples: `// Nested routes
<Routes>
    <Route path="/users" element={<UsersLayout />}>
        <Route index element={<UserList />} />
        <Route path=":id" element={<UserDetail />} />
        <Route path=":id/edit" element={<UserEdit />} />
    </Route>
</Routes>

// Layout with Outlet
function UsersLayout() {
    return (
        <div>
            <nav>Users Navigation</nav>
            <Outlet />
        </div>
    );
}

// Protected route
function ProtectedRoute({ children }) {
    const { user } = useAuth();
    return user ? children : <Navigate to="/login" />;
}

// Query parameters
function SearchResults() {
    const [searchParams, setSearchParams] = useSearchParams();
    const query = searchParams.get('q');
    
    return (
        <div>
            <input
                value={query || ''}
                onChange={(e) => setSearchParams({ q: e.target.value })}
            />
        </div>
    );
}

// Lazy loading
const UserDetail = lazy(() => import('./UserDetail'));`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1536,
			Title:       "Vue Router and Navigation",
			Description: "Master Vue Router: routing, navigation guards, and advanced Vue navigation patterns.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Vue Router",
					Content: `Vue Router is the official routing library for Vue.js, providing deep integration with Vue's reactivity system and component lifecycle. Its advanced features — navigation guards, route meta fields, dynamic matching, scroll behavior control, and route transitions — give you fine-grained control over the navigation experience, making it possible to build complex, secure, and polished single-page applications.

**1. Navigation Guards — Controlling the Flow**

Navigation guards are hooks that let you intercept and control navigation events, running logic before, during, or after route transitions. The global beforeEach guard runs before every navigation and is the most commonly used — it is the ideal place to check authentication, verify permissions, and redirect unauthorized users. The beforeResolve guard runs after all in-component guards and async route components have been resolved but before the navigation is confirmed — useful for data-fetching logic that must complete before showing a page. The afterEach guard runs after navigation has completed and cannot prevent it — perfect for analytics tracking, page title updates, and scroll position management. Route-specific guards (beforeEnter on individual routes) let you apply guard logic to specific routes without affecting the global navigation flow. Guards receive three arguments: the target route (to), the current route (from), and a next function that controls whether the navigation proceeds, is cancelled, or is redirected.

**2. Route Meta Fields and Dynamic Matching**

Route meta fields are custom properties you can attach to any route definition, providing a clean way to associate arbitrary data with routes. The most common use is marking routes that require authentication (meta: { requiresAuth: true }) or specific roles (meta: { role: "admin" }), which guards then check during navigation. You can also use meta for page titles, layout selection, breadcrumb labels, and transition animations. Dynamic route matching with regular expression constraints lets you enforce URL parameter formats — for example, ensuring a user ID parameter contains only digits with /user/:id(\\d+). Optional parameters, repeatable parameters, and custom regex patterns give you precise control over which URLs match which routes.

**3. Scroll Behavior, Transitions, and Best Practices**

The scrollBehavior option on the router instance lets you control scroll position during navigation. You can return to the top of the page on every navigation, restore the saved scroll position when the user clicks the back button, or scroll to a specific element identified by a hash in the URL. Route transitions leverage Vue's built-in transition component to animate between route views — fade effects, slide animations, or any custom CSS/JavaScript animation. Best practices include using guards consistently for authentication and authorization (never rely solely on hiding UI elements), handling route-level errors with dedicated error pages, providing loading states during async route component resolution, using route meta to keep route-related configuration centralized, and organizing routes in a modular structure where each feature module defines its own routes.`,
					CodeExamples: `// Navigation guards
router.beforeEach((to, from, next) => {
    if (to.meta.requiresAuth && !isAuthenticated()) {
        next('/login');
    } else {
        next();
    }
});

// Route meta
const routes = [
    {
        path: '/admin',
        component: Admin,
        meta: { requiresAuth: true, role: 'admin' }
    }
];

// Dynamic route matching
const routes = [
    {
        path: '/user/:id(\\d+)', // Only numbers
        component: UserDetail
    }
];

// Scroll behavior
const router = createRouter({
    scrollBehavior(to, from, savedPosition) {
        if (savedPosition) {
            return savedPosition;
        } else {
            return { top: 0 };
        }
    }
});`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1537,
			Title:       "WebSockets and Real-time",
			Description: "Build real-time applications: WebSockets, Socket.io, and real-time communication.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "WebSockets and Real-time Communication",
					Content: `WebSockets enable persistent, bidirectional communication between a client and server, fundamentally changing what is possible in web applications. Unlike the traditional HTTP request-response model where the client must initiate every interaction, WebSockets establish an open connection through which both the client and server can send messages at any time. This makes real-time features — live chat, collaborative editing, multiplayer games, stock tickers, and live dashboards — both possible and efficient.

**1. The WebSocket Protocol**

The WebSocket protocol starts as a regular HTTP request with an "upgrade" header, and once the server agrees, the connection is "upgraded" to a persistent WebSocket connection. From that point on, both sides can send messages freely without the overhead of establishing new connections or sending HTTP headers with each message. This full-duplex communication means data can flow in both directions simultaneously, unlike HTTP where the server can only respond to client requests. The latency is dramatically lower than polling or long-polling alternatives because there is no connection setup overhead for each message. WebSockets support both text (JSON strings being the most common) and binary data (ArrayBuffer, Blob), making them suitable for everything from chat messages to streaming audio and video data. The native WebSocket API is event-driven, with handlers for open (connection established), message (data received), error (something went wrong), and close (connection terminated).

**2. Socket.io — WebSockets Made Practical**

While the native WebSocket API provides the low-level foundation, Socket.io adds the practical features needed for production applications. It provides automatic fallback to alternative transports (HTTP long-polling, Server-Sent Events) when WebSockets are not available, ensuring your application works behind corporate firewalls and proxies that block WebSocket connections. Automatic reconnection handles the inevitable network interruptions — when a connection drops, Socket.io automatically attempts to reconnect with configurable backoff strategies. Rooms and namespaces provide built-in multiplexing — you can group connected clients into rooms (e.g., a chat room) and send messages to all clients in a room with a single call. Namespaces let you run separate communication channels over a single connection, useful for separating concerns like chat, notifications, and real-time data updates. The event-based API lets you define custom events (socket.emit("chat", data) and socket.on("chat", handler)) that are more semantic and maintainable than raw message passing.

**3. Use Cases and Best Practices**

Real-time applications span a wide range of use cases. Chat applications are the classic example, where messages must appear instantly for all participants. Live update dashboards display real-time metrics, order statuses, or IoT sensor readings. Collaborative tools like Google Docs use WebSockets to sync document changes across all editors in real time. Multiplayer games use them for low-latency state synchronization. Real-time notifications keep users informed without polling. Best practices include implementing robust reconnection logic with exponential backoff and state recovery, managing connection state in a centralized service or hook to avoid multiple connections, always cleaning up WebSocket connections when components unmount to prevent memory leaks and ghost connections, implementing heartbeat/ping mechanisms to detect dead connections, and using message queuing on the client side to handle messages that arrive while the connection is temporarily down.`,
					CodeExamples: `// Native WebSocket
const socket = new WebSocket('ws://localhost:8080');

socket.onopen = () => {
    console.log('Connected');
    socket.send('Hello Server');
};

socket.onmessage = (event) => {
    console.log('Message:', event.data);
};

socket.onerror = (error) => {
    console.error('Error:', error);
};

socket.onclose = () => {
    console.log('Disconnected');
};

// Socket.io client
import io from 'socket.io-client';

const socket = io('http://localhost:3000');

socket.on('connect', () => {
    console.log('Connected');
});

socket.on('message', (data) => {
    console.log('Message:', data);
});

socket.emit('chat', { message: 'Hello' });

// React hook
function useSocket(url) {
    const [socket, setSocket] = useState(null);
    
    useEffect(() => {
        const ws = io(url);
        setSocket(ws);
        
        return () => ws.close();
    }, [url]);
    
    return socket;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1538,
			Title:       "Internationalization (i18n)",
			Description: "Implement i18n: multi-language support, locale management, and translation systems.",
			Order:       38,
			Lessons: []problems.Lesson{
				{
					Title: "i18n Implementation",
					Content: `Internationalization (i18n) is the process of designing and building your application so it can be adapted to different languages, regions, and cultures without requiring code changes. In a global market, supporting multiple languages is not optional for many products — it is a business requirement that can open your product to billions of additional users. However, i18n goes far beyond simply translating strings: it encompasses text direction, date and number formatting, pluralization rules, cultural conventions, and accessibility for diverse audiences.

**1. Core i18n Concepts**

A locale identifies a specific language and region combination (e.g., en-US for American English, pt-BR for Brazilian Portuguese). Translation files map abstract keys to locale-specific strings — your code references t("welcome_message") and the i18n library resolves it to "Welcome back!" in English or "Bienvenue!" in French. Pluralization is surprisingly complex across languages: English has two forms (1 item, 2 items), but Arabic has six plural forms, and some languages have no plural distinction at all. A robust i18n solution must handle these rules correctly using CLDR (Unicode Common Locale Data Repository) rules. Date and number formatting varies dramatically by locale — the date 3/4/2024 means March 4th in the US but April 3rd in Europe, the number 1,234.56 in English is 1.234,56 in German. Right-to-left (RTL) support for languages like Arabic, Hebrew, and Persian requires mirroring the entire UI layout — navigation, text alignment, icons, and even directional metaphors like "forward" and "back."

**2. Library Ecosystem**

i18next is the most popular framework-agnostic i18n library, providing a comprehensive feature set including interpolation, pluralization, nesting, context-based translations, and plugin support for backends, caching, and language detection. react-i18next wraps i18next for React with hooks (useTranslation), components (Trans for inline markup), and Suspense support for lazy-loading translations. vue-i18n provides similar deep integration for Vue applications with a composable API, directive-based translations, and SFC (Single File Component) support. Format.js (formerly React Intl) implements the ICU Message Format standard, which is particularly powerful for complex messages involving plurals, gender, select, and nested formatting.

**3. Best Practices**

Extract every user-visible string from your codebase — even error messages, button labels, and placeholder text. Use descriptive translation keys (user.profile.edit_button rather than btn1) that provide context for translators. Never concatenate translated strings (t("hello") + " " + name) because word order varies by language — use interpolation instead (t("hello", { name })). Handle pluralization properly using your library's plural features rather than conditional logic. Format all dates, numbers, and currencies using the Intl API or a locale-aware library — never hardcode formats. Test your application in multiple locales, including RTL languages and languages with long words (German) or complex scripts (Chinese, Arabic). Use pseudo-localization during development to visually identify untranslated strings and detect layout issues with longer text.`,
					CodeExamples: `// react-i18next
import { useTranslation } from 'react-i18next';

function Component() {
    const { t, i18n } = useTranslation();
    
    return (
        <div>
            <h1>{t('welcome')}</h1>
            <button onClick={() => i18n.changeLanguage('es')}>
                Spanish
            </button>
        </div>
    );
}

// Translation files
// en.json
{
    "welcome": "Welcome",
    "items": "{{count}} item",
    "items_plural": "{{count}} items"
}

// Usage with pluralization
{t('items', { count: 1 })} // "1 item"
{t('items', { count: 5 })} // "5 items"

// Date formatting
import { format } from 'date-fns';
import { enUS, es } from 'date-fns/locale';

format(new Date(), 'PPP', { locale: enUS }); // English
format(new Date(), 'PPP', { locale: es }); // Spanish`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1539,
			Title:       "Animation Libraries",
			Description: "Master animation: Framer Motion, GSAP, and creating smooth animations.",
			Order:       39,
			Lessons: []problems.Lesson{
				{
					Title: "Animation Libraries and Techniques",
					Content: `Animation libraries provide powerful, high-level tools for creating engaging, polished user interfaces that delight users and communicate meaning through motion. While CSS transitions and keyframes handle simple hover effects and loading spinners, complex animations — orchestrated sequences, physics-based gestures, scroll-driven effects, and layout transitions — require specialized libraries that abstract away the mathematical and performance complexities of smooth animation.

**1. Framer Motion — The React Animation Powerhouse**

Framer Motion is the most popular animation library in the React ecosystem, providing a declarative, component-based API that feels natural to React developers. Instead of imperatively telling elements where to move frame by frame, you declare the desired initial state, the target animate state, and the transition parameters — Framer Motion handles all the interpolation, timing, and performance optimization. The motion component wrapper (motion.div, motion.button, etc.) transforms any HTML element into an animatable entity. AnimatePresence enables exit animations, solving the notoriously difficult problem of animating components as they unmount from the DOM. Layout animations automatically animate between different CSS layouts — when a list reorders, items smoothly slide to their new positions rather than jumping. Gesture support (whileHover, whileTap, drag) makes interactive animations trivial, and the spring-based physics engine produces natural-feeling motion without manually tuning cubic-bezier curves.

**2. GSAP — The Professional Animation Toolkit**

GSAP (GreenSock Animation Platform) is the industry-standard animation library used by major studios, agencies, and product teams worldwide. Unlike Framer Motion, which is React-specific, GSAP is framework-agnostic and works with vanilla JavaScript, React, Vue, Angular, or any other framework. Its Timeline API lets you orchestrate complex multi-step animation sequences with precise control over timing, staggering, and overlapping. The ScrollTrigger plugin enables scroll-driven animations — elements can fade in as they enter the viewport, parallax layers can move at different speeds, and entire sections can pin and animate based on scroll position. GSAP consistently delivers the best raw animation performance of any JavaScript library because it uses its own optimized rendering pipeline that batches DOM reads and writes, avoiding layout thrashing. It is the go-to choice for marketing sites, interactive storytelling, and any project where animation quality and performance are paramount.

**3. CSS Animations — The Zero-Dependency Foundation**

CSS animations using @keyframes and transitions remain the most performant option for simple, predictable animations because they can be offloaded entirely to the browser's compositor thread, running at a smooth 60fps even when the main JavaScript thread is busy. Transitions animate between two states (e.g., hover effects, color changes, size adjustments) and are triggered by property changes. Keyframe animations define multi-step sequences that can loop, alternate direction, and control timing with granular precision. The transform property (translate, scale, rotate) and opacity are the only CSS properties that can be animated without triggering expensive layout recalculations — sticking to these "compositor-only" properties is the single most important performance optimization for CSS animations.

**4. Best Practices for Production Animations**

Always use GPU-accelerated properties (transform and opacity) instead of animating layout-triggering properties like width, height, top, or left, which force the browser to recalculate the layout of the entire page on every frame. Respect the prefers-reduced-motion media query — users who have enabled reduced motion in their operating system settings may experience motion sickness or seizures from animations, so provide alternative static transitions or disable animations entirely for these users. Keep animations meaningful and purposeful: every animation should either communicate a state change (an item being added or removed), provide spatial context (where something came from or went to), or draw attention to important information. Avoid gratuitous animation that slows down task completion or distracts from content. Test animations on low-powered devices to ensure they remain smooth, and use the browser's Performance panel to identify animation-related jank.`,
					CodeExamples: `// Framer Motion
import { motion } from 'framer-motion';

<motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.5 }}
>
    Content
</motion.div>

// Gestures
<motion.div
    drag
    dragConstraints={{ left: 0, right: 300 }}
    whileHover={{ scale: 1.1 }}
    whileTap={{ scale: 0.9 }}
>
    Draggable
</motion.div>

// GSAP
import gsap from 'gsap';

gsap.to('.element', {
    x: 100,
    duration: 1,
    ease: 'power2.out'
});

// Timeline
const tl = gsap.timeline();
tl.to('.element1', { x: 100 })
  .to('.element2', { y: 50 }, '-=0.5')
  .to('.element3', { rotation: 360 });`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1540,
			Title:       "Canvas API and Graphics",
			Description: "Create graphics with Canvas API: drawing, animations, and interactive graphics.",
			Order:       40,
			Lessons: []problems.Lesson{
				{
					Title: "Canvas API Fundamentals",
					Content: `The Canvas API is a powerful, low-level drawing interface built into every modern browser that lets you programmatically render 2D graphics, create animations, manipulate images pixel by pixel, and build interactive visual experiences directly in a web page. Unlike SVG, which uses a retained-mode model where each shape is a DOM element that can be individually styled and animated, Canvas uses an immediate-mode model where you issue drawing commands to a bitmap surface — once a shape is drawn, it becomes pixels and the canvas has no memory of it as a distinct object. This makes Canvas exceptionally fast for scenarios involving thousands of moving elements, complex visualizations, or real-time image processing.

**1. Canvas Basics and the 2D Context**

Every Canvas interaction starts by getting a reference to the canvas HTML element and obtaining its 2D rendering context via getContext('2d'). The context object is your drawing toolkit — it provides all the methods and properties for rendering shapes, text, images, and paths. The canvas coordinate system starts at the top-left corner (0, 0) with x increasing to the right and y increasing downward. The canvas has an internal resolution (its width and height attributes) that is separate from its CSS display size — for crisp rendering on high-DPI (Retina) displays, you should set the canvas resolution to twice its CSS size and scale the context accordingly. Everything you draw is rasterized immediately into the pixel buffer, which means you cannot click on or select individual drawn elements — if you need interactivity, you must implement hit detection yourself by tracking the positions and bounds of drawn objects.

**2. Drawing Operations — Shapes, Paths, and Text**

The Canvas API provides a rich set of drawing primitives. fillRect draws a filled rectangle and strokeRect draws an outlined one — these are the simplest operations and do not require a path. For more complex shapes, you use the path API: call beginPath to start a new path, then use moveTo, lineTo, arc, quadraticCurveTo, and bezierCurveTo to define the path's geometry, and finally call fill or stroke to render it. The arc method draws circles and circular arcs by specifying a center point, radius, start angle, and end angle in radians. Text rendering uses fillText and strokeText, with font, textAlign, and textBaseline properties controlling the appearance. The drawImage method renders images (from img elements, other canvases, or video frames) onto the canvas, with optional cropping and scaling — this is the foundation for image processing, sprite-based games, and video effects.

**3. Transformations and State Management**

Canvas transformations modify the coordinate system rather than individual shapes, which is a powerful concept for building complex scenes. The translate method moves the origin point, rotate spins the coordinate system around the current origin, and scale stretches or compresses it. These transformations are cumulative — each one applies on top of the current transformation matrix. The save and restore methods are essential for managing this: save pushes the current state (transformations, styles, clipping region) onto a stack, and restore pops it back. This pattern lets you apply transformations for drawing one element, restore to the previous state, and draw the next element without the transformations interfering. For example, to draw a spinning wheel, you save the state, translate to the wheel's center, rotate by the current angle, draw the wheel centered at the origin, then restore — the rest of your drawing code is unaffected.

**4. Animations and Performance**

Canvas animations work on a simple principle: clear the entire canvas, recalculate positions, redraw everything, and repeat. The requestAnimationFrame function is the correct way to drive this loop — it synchronizes your drawing with the browser's display refresh rate (typically 60fps), pauses when the tab is not visible (saving CPU and battery), and provides a high-resolution timestamp for smooth, time-based animation. For performance, minimize the area you clear and redraw when possible, use offscreen canvases to pre-render complex static elements, batch similar drawing operations together, and avoid reading pixel data (getImageData) during animation frames as it forces a GPU-to-CPU transfer. For very complex scenes with many thousands of elements, consider using Web Workers to compute positions off the main thread, or switch to WebGL for GPU-accelerated rendering.`,
					CodeExamples: `// Basic canvas setup
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

// Drawing shapes
ctx.fillStyle = 'blue';
ctx.fillRect(10, 10, 100, 100);

ctx.strokeStyle = 'red';
ctx.lineWidth = 2;
ctx.strokeRect(120, 10, 100, 100);

// Circles
ctx.beginPath();
ctx.arc(200, 200, 50, 0, Math.PI * 2);
ctx.fill();

// Paths
ctx.beginPath();
ctx.moveTo(10, 10);
ctx.lineTo(100, 100);
ctx.lineTo(200, 50);
ctx.closePath();
ctx.stroke();

// Text
ctx.font = '30px Arial';
ctx.fillText('Hello Canvas', 10, 50);

// Animation
function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw frame
    ctx.fillRect(x, y, 50, 50);
    
    x += 1;
    requestAnimationFrame(animate);
}

animate();`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1541,
			Title:       "WebGL and Three.js",
			Description: "3D graphics with WebGL and Three.js: 3D rendering, scenes, and interactive 3D applications.",
			Order:       41,
			Lessons: []problems.Lesson{
				{
					Title: "WebGL and Three.js Basics",
					Content: `WebGL and Three.js bring the power of GPU-accelerated 3D graphics to the browser, enabling experiences that were once limited to native desktop applications — interactive product configurators, immersive data visualizations, browser-based games, virtual showrooms, and architectural walkthroughs. WebGL provides the low-level interface to the GPU, while Three.js wraps it in a developer-friendly abstraction that makes 3D programming accessible to web developers without requiring deep knowledge of computer graphics or shader programming.

**1. WebGL — The Low-Level Graphics Pipeline**

WebGL (Web Graphics Library) is a JavaScript API that provides direct access to the GPU through an interface based on OpenGL ES. It operates at a very low level: you define vertices (points in 3D space), write shader programs in GLSL (a C-like language) that run on the GPU to determine how vertices are positioned and how pixels are colored, and manage buffers, textures, and rendering state manually. The vertex shader processes each vertex of your 3D geometry, transforming it from object space through world space and camera space to screen coordinates. The fragment shader (also called pixel shader) determines the color of each pixel by computing lighting, applying textures, and blending colors. This low-level access gives you maximum control and performance — you can implement any rendering technique the GPU supports — but it requires significant boilerplate code and a deep understanding of the graphics pipeline. Most developers use WebGL indirectly through higher-level libraries.

**2. Three.js — Making 3D Accessible**

Three.js is the most popular 3D library for the web, providing a high-level, object-oriented API that abstracts away the complexities of WebGL while retaining the ability to drop down to raw WebGL when needed. It introduces a scene graph architecture where you build your 3D world by adding objects to a hierarchical scene. Three.js ships with an extensive library of built-in geometries (boxes, spheres, planes, tori, custom shapes), materials (basic flat shading, Phong for shiny surfaces, physically-based rendering for photorealistic results), and light types (ambient, directional, point, spot, hemisphere). Camera types include perspective (mimicking human vision with foreshortening) and orthographic (no perspective distortion, used for 2D-style games and technical views). Orbit controls, fly controls, and other interaction handlers let users navigate the 3D scene with mouse, touch, or keyboard input.

**3. Core Concepts — Scene, Camera, Renderer, and Mesh**

The Scene is the container for your 3D world — it holds all objects, lights, and cameras. Think of it as a stage where everything is placed. The Camera defines the viewpoint from which the scene is rendered — a PerspectiveCamera mimics how human eyes see the world (objects farther away appear smaller), while an OrthographicCamera renders without perspective distortion. The Renderer takes the scene and camera and produces the actual image — the WebGLRenderer is the standard choice, drawing to an HTML canvas element using the GPU. A Mesh is the fundamental visible object, composed of a Geometry (the shape — vertices, faces, and UV coordinates) and a Material (the appearance — color, texture, shininess, transparency). Lights illuminate the scene and interact with materials to produce realistic shading — without lights, physically-based materials appear black because there is no light for them to reflect.

**4. Use Cases and the Modern 3D Web**

3D on the web has matured into a mainstream technology used across industries. E-commerce sites use 3D product viewers that let customers rotate, zoom, and customize products before purchasing — Nike, Apple, and IKEA all use this approach. Data visualization in three dimensions can reveal patterns invisible in 2D charts, such as clustering in high-dimensional datasets. Browser-based games from simple casual experiences to complex multiplayer worlds leverage Three.js and WebGL. Architectural and real estate companies offer virtual walkthroughs of properties. Educational platforms create interactive 3D models of molecules, anatomy, and engineering systems. The emerging ecosystem around React Three Fiber brings Three.js into the React component model, letting you build 3D scenes with JSX and manage them with React state and lifecycle — dramatically lowering the barrier to entry for React developers.`,
					CodeExamples: `// Three.js setup
import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create cube
const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

camera.position.z = 5;

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    renderer.render(scene, camera);
}

animate();`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1542,
			Title:       "Web Audio API",
			Description: "Create audio applications: Web Audio API, sound synthesis, and audio processing.",
			Order:       42,
			Lessons: []problems.Lesson{
				{
					Title: "Web Audio API Fundamentals",
					Content: `The Web Audio API is a powerful, low-level system for generating, processing, and analyzing audio directly in the browser. It goes far beyond what the simple HTML audio element can do — while the audio element handles basic playback of audio files, the Web Audio API gives you a complete audio processing graph where you can synthesize sounds from scratch, apply real-time effects like reverb and distortion, perform frequency analysis for visualizations, and mix multiple audio sources with precise timing control. This API is the foundation for browser-based music applications, interactive sound effects in games, audio visualizations, podcast editors, and accessibility tools.

**1. The AudioContext and Audio Graph Architecture**

Everything in the Web Audio API begins with creating an AudioContext, which represents the audio processing environment and serves as the factory for creating all audio nodes. The API is built around a modular graph architecture: you create individual audio nodes (each performing a specific function), connect them together in a chain, and route the final output to the speakers (the context's destination node). Think of it like a physical audio studio — you have sound sources (microphones, synthesizers), processing equipment (equalizers, compressors, reverb units), and outputs (speakers, recorders). Each node has inputs and outputs, and you wire them together by calling source.connect(processor).connect(destination). This modular design means you can build arbitrarily complex audio processing pipelines by combining simple building blocks, and you can dynamically add, remove, or reconfigure nodes while audio is playing.

**2. Sound Sources — Oscillators, Buffers, and Media Streams**

The Web Audio API provides several types of sound sources. The OscillatorNode generates periodic waveforms (sine, square, sawtooth, triangle) at a specified frequency — this is the foundation of sound synthesis, and by combining multiple oscillators with different frequencies and waveforms, you can create complex tones and musical instruments. The AudioBufferSourceNode plays pre-recorded audio loaded into memory, with precise control over playback rate, looping, and start/stop timing — perfect for sound effects and music playback. The MediaElementAudioSourceNode wraps an HTML audio or video element, letting you apply Web Audio processing to media playback. The MediaStreamAudioSourceNode captures live audio from a microphone via getUserMedia, enabling real-time voice processing, pitch detection, and audio recording.

**3. Processing Nodes — Effects and Analysis**

Processing nodes transform audio as it flows through the graph. The GainNode controls volume — you can fade audio in and out by scheduling gain value changes over time using the ramp methods. The BiquadFilterNode implements common audio filters (lowpass, highpass, bandpass, notch) for equalizer effects and sound shaping. The DelayNode adds a time delay to the audio signal, which is the building block for echo and chorus effects. The ConvolverNode applies convolution reverb using an impulse response, allowing you to simulate the acoustics of any real space — a cathedral, a concert hall, or a small room. The AnalyserNode does not modify the audio but provides real-time frequency and time-domain data that you can use to create audio visualizations — spectrum analyzers, waveform displays, and beat-reactive graphics rendered on a Canvas element.

**4. Use Cases and Practical Applications**

The Web Audio API enables a wide range of applications. Browser-based music production tools like BandLab and Soundtrap use it for multi-track recording, mixing, and effects processing. Games use it for spatial audio (positioning sounds in 3D space relative to the player), dynamic soundtracks that respond to gameplay, and low-latency sound effects triggered by user actions. Audio visualization creates compelling visual experiences synchronized to music — bars that pulse with the bass, particles that react to treble frequencies, and waveforms that dance with the audio. Educational music applications use oscillators and the MIDI API for interactive piano keyboards and synthesizers. Accessibility tools use the AnalyserNode for speech recognition preprocessing and the OscillatorNode for screen reader sonification.`,
					CodeExamples: `// Audio context
const audioContext = new AudioContext();

// Create oscillator
const oscillator = audioContext.createOscillator();
const gainNode = audioContext.createGain();

oscillator.connect(gainNode);
gainNode.connect(audioContext.destination);

oscillator.frequency.value = 440; // A4 note
gainNode.gain.value = 0.5;

oscillator.start();
oscillator.stop(audioContext.currentTime + 1); // Stop after 1 second

// Load and play audio
async function playAudio(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1543,
			Title:       "WebRTC",
			Description: "Real-time communication with WebRTC: peer-to-peer connections, video/audio streaming.",
			Order:       43,
			Lessons: []problems.Lesson{
				{
					Title: "WebRTC Fundamentals",
					Content: `WebRTC (Web Real-Time Communication) is a set of browser APIs that enable direct peer-to-peer communication between browsers without requiring data to pass through a central server. This technology powers video conferencing, voice calling, screen sharing, and real-time data exchange — all running natively in the browser with no plugins or installations required. Applications like Google Meet, Discord (in the browser), and peer-to-peer file sharing tools are built on WebRTC, and understanding its architecture is essential for building any real-time communication feature on the web.

**1. The Three Pillars of WebRTC**

WebRTC is built around three core APIs that work together. RTCPeerConnection is the heart of the system — it manages the entire peer-to-peer connection lifecycle, including codec negotiation, encryption, bandwidth management, and NAT traversal. Every WebRTC connection is encrypted by default using DTLS (for data channels) and SRTP (for media streams), so communication is secure even though it travels directly between browsers. MediaStream (accessed via getUserMedia and getDisplayMedia) captures audio and video from the user's microphone, camera, or screen, providing the media tracks that are sent over the peer connection. RTCDataChannel provides a generic, bidirectional channel for sending arbitrary data between peers — text messages, file chunks, game state updates, or any other data — with configurable reliability and ordering guarantees, making it useful for far more than just audio and video.

**2. The Signaling Process — How Peers Find Each Other**

Before two browsers can communicate directly, they need to exchange connection metadata through a process called signaling. WebRTC deliberately does not specify how signaling should work — you can use WebSockets, HTTP polling, carrier pigeons, or any other communication channel. The signaling process exchanges two types of information. First, SDP (Session Description Protocol) offers and answers describe each peer's media capabilities — what codecs they support, what media tracks they want to send and receive, and connection parameters. The initiating peer creates an SDP offer, sends it to the remote peer via your signaling server, and the remote peer responds with an SDP answer. Second, ICE (Interactive Connectivity Establishment) candidates are potential network paths the peers can use to reach each other. ICE discovers candidates by querying STUN servers (which tell the browser its public IP address and port as seen from outside its local network) and optionally TURN servers (which act as relays when direct connections are impossible due to restrictive firewalls or symmetric NATs). The peers exchange these candidates via signaling, and ICE tests all candidate pairs to find the optimal connection path.

**3. Media Handling and Screen Sharing**

getUserMedia prompts the user for permission to access their camera and microphone and returns a MediaStream containing audio and video tracks. You can specify constraints for resolution, frame rate, and which devices to use. The tracks from this stream are added to the RTCPeerConnection, which handles encoding, packetization, and transmission to the remote peer. On the receiving end, the ontrack event fires when remote tracks arrive, and you attach the received stream to a video element for display. Screen sharing uses getDisplayMedia, which lets the user choose to share their entire screen, a specific application window, or a browser tab. The resulting MediaStream works the same way — you add its tracks to the peer connection. You can combine camera video and screen sharing in the same session by managing multiple tracks.

**4. Use Cases and Practical Considerations**

Video conferencing is the most visible use case, but WebRTC's capabilities extend much further. Voice-over-IP calling, telehealth consultations, live customer support with video, collaborative whiteboarding, peer-to-peer file transfer (using data channels to send file chunks directly between browsers), multiplayer gaming with low-latency state synchronization, and IoT device control all leverage WebRTC. Key practical considerations include handling network changes gracefully (users switching from WiFi to cellular), implementing quality adaptation (reducing resolution or frame rate when bandwidth drops), managing multiple participants (which typically requires a Selective Forwarding Unit server rather than full mesh peer connections), and providing fallback strategies when peer-to-peer connections fail (routing through TURN relay servers). Always test across different network conditions and browser implementations, as WebRTC behavior can vary significantly.`,
					CodeExamples: `// Get user media
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        videoElement.srcObject = stream;
    });

// Create peer connection
const pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
});

// Add stream
stream.getTracks().forEach(track => {
    pc.addTrack(track, stream);
});

// Handle ICE candidates
pc.onicecandidate = (event) => {
    if (event.candidate) {
        // Send to peer via signaling
        sendToPeer({ type: 'ice-candidate', candidate: event.candidate });
    }
};

// Handle remote stream
pc.ontrack = (event) => {
    remoteVideo.srcObject = event.streams[0];
};

// Create offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
// Send offer to peer`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1544,
			Title:       "CI/CD for Frontend",
			Description: "Implement CI/CD pipelines: GitHub Actions, automated testing, and deployment.",
			Order:       44,
			Lessons: []problems.Lesson{
				{
					Title: "CI/CD Pipeline Setup",
					Content: `Continuous Integration and Continuous Deployment (CI/CD) automate the process of testing, building, and deploying frontend applications, transforming what was once a manual, error-prone release process into a reliable, repeatable pipeline that runs on every code change. In modern frontend development, where teams push code multiple times per day and applications must work across dozens of browser and device combinations, CI/CD is not a luxury — it is essential infrastructure that catches bugs early, enforces code quality standards, and delivers updates to users safely and rapidly.

**1. Understanding CI/CD — The Two Halves**

Continuous Integration (CI) is the practice of automatically building and testing every code change as soon as it is pushed to the repository. When a developer opens a pull request, the CI pipeline kicks in: it installs dependencies, runs linters to enforce code style, executes unit and integration tests, and builds the application to verify that compilation succeeds. If any step fails, the pull request is blocked from merging, preventing broken code from reaching the main branch. Continuous Deployment (CD) extends this automation to the release process: once code is merged to the main branch and passes all checks, it is automatically deployed to staging environments for final verification and then to production. Some teams use Continuous Delivery instead, where the pipeline prepares a deployable artifact but requires a manual approval step before releasing to production — this provides a safety net for high-stakes applications. The combination of CI and CD means that code changes flow from a developer's laptop to production in minutes rather than days or weeks.

**2. GitHub Actions — The Modern CI/CD Platform**

GitHub Actions is the most popular CI/CD platform for frontend projects, deeply integrated with GitHub's repository and pull request workflow. Workflows are defined in YAML files stored in the .github/workflows directory of your repository. Each workflow is triggered by events — pushes, pull requests, scheduled cron jobs, or manual dispatches. Jobs within a workflow run on virtual machines (runners) and consist of sequential steps that execute commands. Matrix builds let you test across multiple configurations simultaneously — for example, running your test suite against Node.js 16, 18, and 20 in parallel to ensure compatibility. Dependency caching dramatically speeds up pipelines by preserving the node_modules directory between runs, avoiding redundant npm install operations. Secrets management securely stores sensitive values like API keys, deployment tokens, and environment-specific configurations, making them available to workflow steps without exposing them in code.

**3. Pipeline Stages — From Code to Production**

A well-designed frontend CI/CD pipeline typically includes five stages executed in order. First, dependency installation (npm ci for reproducible installs from the lockfile). Second, linting and formatting checks (ESLint, Prettier, stylelint) that enforce code quality standards and catch common errors. Third, automated tests — unit tests with Jest or Vitest, component tests with Testing Library, and optionally end-to-end tests with Playwright or Cypress. Fourth, the production build step that compiles TypeScript, bundles with Webpack or Vite, optimizes assets, and generates the deployable artifact. Fifth, deployment to the target environment — staging for verification or production for release. Each stage acts as a quality gate: if linting fails, tests do not run; if tests fail, the build does not happen; if the build fails, deployment is skipped.

**4. Best Practices for Reliable Pipelines**

Run the full CI pipeline on every commit and pull request, not just on merges to main — catching issues early is dramatically cheaper than catching them after merge. Use environment variables for all configuration that differs between environments (API URLs, feature flags, analytics keys) rather than hardcoding values. Cache dependencies aggressively — npm ci with a cached node_modules can reduce pipeline time from minutes to seconds. Always deploy to a staging environment first and run smoke tests or end-to-end tests against it before promoting to production. Implement rollback strategies (keeping previous deployment artifacts, using blue-green deployments, or leveraging platform-specific instant rollback features) so you can recover quickly from bad deployments. Monitor your deployments with health checks and alerting so you know immediately if a release causes errors or performance regressions.`,
					CodeExamples: `# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run lint
      - run: npm test
      - run: npm run build
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1545,
			Title:       "Docker for Frontend",
			Description: "Containerize frontend applications: Docker, multi-stage builds, and containerization.",
			Order:       45,
			Lessons: []problems.Lesson{
				{
					Title: "Dockerizing Frontend Applications",
					Content: `Docker enables consistent, reproducible deployment environments for frontend applications by packaging your built application and its serving infrastructure into a lightweight, portable container that runs identically on any machine — your laptop, a CI server, a staging environment, or production. The classic "it works on my machine" problem disappears because the container includes everything needed to serve your application, and it behaves the same everywhere. For frontend applications specifically, Docker is most valuable for ensuring consistent build environments, simplifying deployment to container orchestration platforms like Kubernetes, and standardizing how your application is served in production.

**1. Docker Core Concepts for Frontend Developers**

A Docker image is a read-only template that contains everything needed to run your application — the operating system, runtime dependencies, your built application code, and a web server to serve it. Think of it as a snapshot of a perfectly configured server. A container is a running instance of an image — you can start, stop, and delete containers without affecting the underlying image. The Dockerfile is a script of instructions that tells Docker how to build an image, step by step: start from a base image, copy files, run commands, configure the server, and specify what happens when the container starts. Docker Compose is a tool for defining and running multi-container applications — for example, your frontend container alongside an API backend container and a database container, all networked together and started with a single command.

**2. Multi-Stage Builds — The Key to Lean Frontend Images**

Multi-stage builds are the single most important Docker technique for frontend applications. The problem they solve is size: a Node.js image with your source code, node_modules, and build tools can easily be 1-2 GB, but your actual production artifact (the built HTML, CSS, and JavaScript files) is typically only 5-50 MB. A multi-stage Dockerfile uses two separate stages. The first stage (the build stage) starts from a Node.js image, installs all dependencies including dev dependencies, copies your source code, and runs the build command — this produces optimized static files in a dist or build directory. The second stage (the production stage) starts from a tiny nginx:alpine image (only ~25 MB), copies just the built static files from the first stage, and configures nginx to serve them. The build stage's Node.js runtime, node_modules, source code, and build tools are completely discarded, resulting in a final production image that is typically under 50 MB — fast to push, pull, and deploy.

**3. Nginx Configuration for Single-Page Applications**

When serving a single-page application from nginx inside a Docker container, you need a custom nginx configuration to handle client-side routing correctly. By default, nginx looks for a file matching the URL path — so navigating to /users/123 would look for a file at /users/123, which does not exist because the SPA has only a single index.html. The solution is the try_files directive, which tells nginx to try the requested path first, then fall back to index.html for any path that does not match a real file. You should also configure caching headers: long-lived cache (one year) for hashed asset files (main.abc123.js, styles.def456.css) that change names when their content changes, and no-cache for index.html so that browsers always fetch the latest version that references the current asset filenames. Gzip or Brotli compression should be enabled for text-based assets to reduce transfer sizes.

**4. Best Practices for Dockerized Frontend Applications**

Always use multi-stage builds to keep your production images small and free of build tools and source code. Use a .dockerignore file to prevent node_modules, .git, and other large or sensitive directories from being copied into the build context, which speeds up builds significantly. Pin your base images to specific versions (node:18.17-alpine rather than node:latest) to ensure reproducible builds. Leverage Docker's layer caching by copying package.json and package-lock.json before copying the rest of the source code — this way, the npm install layer is cached and only re-runs when dependencies actually change. Use environment variables (via Docker's ENV instruction or runtime environment injection) for configuration that differs between environments. Run your container as a non-root user for security, and scan your images for known vulnerabilities using tools like Docker Scout or Trivy.`,
					CodeExamples: `# Multi-stage Dockerfile
# Stage 1: Build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Production
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: .
    ports:
      - "3000:80"
    environment:
      - API_URL=http://api:8080`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1546,
			Title:       "Monitoring and Error Tracking",
			Description: "Monitor applications: error tracking, performance monitoring, and observability.",
			Order:       46,
			Lessons: []problems.Lesson{
				{
					Title: "Error Tracking and Monitoring",
					Content: `Monitoring and error tracking are the eyes and ears of your production frontend application, providing visibility into what is actually happening when real users interact with your code across thousands of different devices, browsers, network conditions, and usage patterns. Without monitoring, bugs and performance regressions go undetected until users complain (or silently leave), and you have no data to diagnose issues when they are reported. A comprehensive monitoring strategy combines error tracking, performance monitoring, and logging to give you a complete picture of your application's health and user experience.

**1. Error Tracking — Catching Production Bugs**

Error tracking tools like Sentry, Bugsnag, and LogRocket automatically capture JavaScript errors, unhandled promise rejections, and failed network requests in your production application and report them to a centralized dashboard. When an error occurs, these tools capture not just the error message and stack trace, but also the user's browser, operating system, device, the sequence of user actions leading to the error, and any custom context you attach (user ID, feature flags, application state). Source maps are critical here — your production code is minified and bundled, making raw stack traces unreadable. By uploading source maps to your error tracking service, stack traces are automatically decompiled back to your original source code with correct file names and line numbers, making debugging straightforward. LogRocket goes a step further with session replay, recording the user's entire session (DOM changes, network requests, console logs) as a video that you can replay to see exactly what the user experienced when the error occurred.

**2. Performance Monitoring — Measuring Real User Experience**

Performance monitoring comes in two forms. Real User Monitoring (RUM) collects performance data from actual user sessions — real page load times, interaction responsiveness, and resource loading speeds across the full diversity of your user base. This is invaluable because lab measurements on a developer's powerful MacBook with a fast connection do not reflect the experience of a user on a mid-range Android phone on a 3G connection. Core Web Vitals (LCP, FID/INP, CLS) are the standard metrics that Google uses for search ranking, and RUM tools track these continuously so you can spot regressions immediately. Synthetic monitoring complements RUM by running automated tests from consistent environments at regular intervals — similar to a health check that pings your application every minute from multiple geographic locations to verify it is responding correctly and within acceptable time thresholds. API performance monitoring tracks response times, error rates, and payload sizes for your backend calls, helping you identify whether slow page loads are caused by frontend rendering issues or backend bottlenecks.

**3. Logging — Structured Observability**

Frontend logging in production requires a different approach than the console.log statements you use during development. Structured logging means emitting log entries as structured data (JSON objects with consistent fields like timestamp, level, message, userId, sessionId, and context) rather than arbitrary strings. This structured format makes logs searchable, filterable, and aggregatable in centralized logging platforms like Datadog, Splunk, or the ELK stack. Log levels (debug, info, warn, error) let you control the verbosity of logging in different environments — verbose debug logs during development, minimal info and error logs in production. Centralized logging aggregates logs from your frontend, backend, and infrastructure into a single searchable system, enabling you to trace a user's journey across the entire stack when investigating issues.

**4. Best Practices for Production Monitoring**

Track all unhandled errors automatically by integrating your error tracking SDK early in your application's initialization. Add custom context to error reports — the user's ID, the current route, recent user actions, and relevant application state — to make every error report actionable without needing to reproduce the issue. Set up alerting thresholds so your team is notified immediately when error rates spike or performance degrades beyond acceptable levels, rather than discovering problems days later during a manual review. Monitor your Core Web Vitals continuously and set up alerts when they cross Google's "good" thresholds. Review your error and performance dashboards regularly — weekly triage of top errors and performance bottlenecks should be a standing team practice. Use feature flags to gradually roll out changes to a subset of users, monitoring error rates and performance metrics during the rollout to catch issues before they affect everyone.`,
					CodeExamples: `// Sentry
import * as Sentry from "@sentry/react";

Sentry.init({
    dsn: "your-dsn",
    environment: "production",
    tracesSampleRate: 1.0,
});

// Error boundary
<Sentry.ErrorBoundary fallback={<ErrorFallback />}>
    <App />
</Sentry.ErrorBoundary>

// Custom error tracking
window.addEventListener('error', (event) => {
    trackError({
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
    });
});

// Performance monitoring
const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        if (entry.entryType === 'navigation') {
            trackMetric('pageLoad', entry.loadEventEnd - entry.fetchStart);
        }
    }
});
observer.observe({ entryTypes: ['navigation'] });`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1547,
			Title:       "Security Best Practices",
			Description: "Secure frontend applications: XSS prevention, CSRF protection, and security patterns.",
			Order:       47,
			Lessons: []problems.Lesson{
				{
					Title: "Frontend Security",
					Content: `Security in frontend development is a critical responsibility because your client-side code is the first line of defense between malicious actors and your users' data. Unlike backend code that runs in a controlled environment, frontend code executes in the user's browser — an environment you do not control, where every line of code is visible and modifiable. Attackers exploit frontend vulnerabilities to steal user credentials, hijack sessions, redirect users to phishing sites, inject malicious content, and exfiltrate sensitive data. Understanding common attack vectors and their defenses is not optional — it is a fundamental skill for every frontend developer building applications that handle user data.

**1. Cross-Site Scripting (XSS) — The Most Common Frontend Vulnerability**

XSS attacks occur when an attacker manages to inject malicious JavaScript into your application that then executes in other users' browsers. There are three main types. Stored XSS happens when malicious script is permanently stored on the server (in a database, comment system, or forum post) and served to every user who views that content. Reflected XSS embeds malicious script in a URL parameter that the server includes in the response — the attacker tricks a user into clicking a crafted link. DOM-based XSS manipulates the client-side DOM directly, exploiting JavaScript that reads from untrusted sources like window.location or document.referrer and inserts it into the page without sanitization. The consequences are severe: attackers can steal session cookies, capture keystrokes (including passwords), redirect users to phishing sites, or modify the page to display fake content. Defense requires multiple layers: never use innerHTML or dangerouslySetInnerHTML with untrusted input, always use textContent or the framework's built-in escaping (React, Vue, and Angular all escape by default), and sanitize any HTML that must be rendered using a library like DOMPurify.

**2. CSRF, Clickjacking, and Other Attack Vectors**

Cross-Site Request Forgery (CSRF) tricks a user's browser into making unwanted requests to a site where the user is authenticated. For example, if a user is logged into their bank, a malicious page could include an image tag or form that submits a transfer request — the browser automatically includes the bank's session cookies with the request. Defense involves using CSRF tokens (unique, unpredictable values included in each form submission and verified on the server), setting the SameSite attribute on cookies to Strict or Lax, and requiring custom headers on API requests (which cannot be set by cross-origin forms). Clickjacking (UI redressing) renders your application in a transparent iframe overlaid on a malicious page, tricking users into clicking buttons they cannot see — the defense is the X-Frame-Options header or Content-Security-Policy frame-ancestors directive, which prevent your pages from being embedded in iframes. Man-in-the-middle attacks intercept communication between the browser and server — the defense is HTTPS everywhere, HSTS headers to prevent protocol downgrade attacks, and certificate pinning for sensitive applications.

**3. Content Security Policy and Security Headers**

Content Security Policy (CSP) is one of the most powerful defenses against XSS and data injection attacks. CSP is an HTTP header that tells the browser exactly which sources of content (scripts, styles, images, fonts, frames) are allowed to load on your page. A strict CSP that only allows scripts from your own domain (script-src 'self') prevents injected inline scripts from executing, even if an attacker finds a way to inject HTML into your page. Other critical security headers include X-Content-Type-Options: nosniff (prevents MIME type sniffing), X-Frame-Options: DENY (prevents clickjacking), Strict-Transport-Security (forces HTTPS), and Referrer-Policy (controls what information is sent in the Referer header when navigating away from your site). These headers are your application's immune system — they provide broad protection against entire categories of attacks with minimal implementation effort.

**4. Dependency Security and Best Practices**

Third-party npm packages are a major attack surface — a single compromised dependency can affect thousands of applications (as seen in the event-stream and ua-parser-js incidents). Run npm audit regularly to check for known vulnerabilities in your dependency tree, and keep dependencies updated with tools like Dependabot or Renovate that automatically create pull requests when security patches are available. Pin dependency versions in your lockfile to prevent unexpected updates, and review the permissions and reputation of new packages before adding them. Validate all user input on both the client (for user experience) and the server (for security — client-side validation can always be bypassed). Use HttpOnly and Secure flags on cookies containing session tokens to prevent JavaScript access and ensure transmission only over HTTPS. Store authentication tokens in HttpOnly cookies rather than localStorage, which is accessible to any JavaScript running on the page, including injected XSS scripts.`,
					CodeExamples: `// XSS Prevention
// Don't use innerHTML with user input
const userInput = "<script>alert('XSS')</script>";
element.textContent = userInput; // Safe

// Use DOMPurify for HTML
import DOMPurify from 'dompurify';
const clean = DOMPurify.sanitize(userInput);

// Content Security Policy
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline';">

// Secure cookies
document.cookie = "session=abc123; Secure; HttpOnly; SameSite=Strict";

// CSRF token
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
fetch('/api/data', {
    method: 'POST',
    headers: {
        'X-CSRF-Token': csrfToken
    }
});`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1548,
			Title:       "SEO for SPAs",
			Description: "Optimize SPAs for search engines: SSR, meta tags, structured data, and SEO strategies.",
			Order:       48,
			Lessons: []problems.Lesson{
				{
					Title: "SEO Optimization for Single Page Applications",
					Content: `Search Engine Optimization (SEO) for Single-Page Applications presents unique challenges because SPAs fundamentally change how content is delivered to browsers and search engine crawlers. In a traditional multi-page application, each URL returns a complete HTML document with all content ready for indexing. In an SPA, the server returns a mostly empty HTML shell, and JavaScript builds the page content after loading — which means search engine crawlers that do not execute JavaScript see a blank page instead of your carefully crafted content. While Google's crawler can execute JavaScript, it does so in a deferred "second wave" of indexing that is slower and less reliable, and other search engines (Bing, DuckDuckGo, Baidu) have varying levels of JavaScript rendering support. For any application where organic search traffic matters, SPA SEO requires deliberate architectural decisions.

**1. The Core SEO Challenges of SPAs**

The most fundamental challenge is JavaScript rendering dependency. When Googlebot or any crawler fetches your SPA's URL, it receives minimal HTML with a JavaScript bundle reference. The crawler must download, parse, and execute that JavaScript to see the actual content — a process that takes additional time and resources, and may not always succeed completely. Client-side routing creates another challenge: all your pages share the same server-side URL (the SPA entry point), so you must ensure the server can respond appropriately to direct requests for any route. Dynamic content loaded from APIs may not be available when the crawler renders the page, especially if data fetching depends on user interactions or authentication. Meta tags (title, description, Open Graph tags for social sharing) must be unique per page and present in the initial HTML response — dynamically injecting them with JavaScript after page load is unreliable for crawlers and social media preview bots that only read the initial HTML.

**2. Solutions — SSR, Pre-Rendering, and Hybrid Approaches**

Server-Side Rendering (SSR) with frameworks like Next.js or Nuxt.js is the gold standard solution. SSR generates complete HTML on the server for each request, ensuring crawlers receive fully-rendered content with correct meta tags on the first request. This also improves perceived performance because users see content immediately rather than waiting for JavaScript to execute. Pre-rendering is an alternative that generates static HTML for each route at build time — tools like react-snap crawl your SPA and save the rendered HTML for each route. This works well for sites with a manageable number of routes and content that does not change frequently. Dynamic rendering is a middle-ground approach where you detect crawler requests (by user-agent) and serve them pre-rendered HTML while serving the regular SPA to human users — services like Prerender.io automate this. For the best results, most production applications use a hybrid approach: SSR or SSG for content-heavy, SEO-critical pages (marketing pages, blog posts, product pages) and client-side rendering for authenticated, interactive sections (dashboards, settings, admin panels).

**3. Dynamic Meta Tags and Structured Data**

Each page in your application needs unique, descriptive meta tags that accurately represent its content. The title tag should be concise (50-60 characters), include relevant keywords, and follow a consistent pattern (Page Title - Site Name). The meta description should summarize the page's content in 150-160 characters and encourage clicks from search results. Open Graph tags (og:title, og:description, og:image) control how your pages appear when shared on social media platforms like Facebook, Twitter, and LinkedIn — without these, shared links show generic previews that discourage clicks. Libraries like React Helmet, Next.js Head component, or Vue Meta make it easy to set these tags dynamically on a per-page basis. Structured data using JSON-LD (JavaScript Object Notation for Linked Data) tells search engines about the semantic meaning of your content — marking up articles, products, reviews, events, recipes, and FAQs with Schema.org vocabulary enables rich snippets in search results (star ratings, pricing, FAQ accordions, breadcrumbs) that dramatically improve click-through rates.

**4. Technical SEO Best Practices for SPAs**

Generate and maintain an XML sitemap that lists all indexable URLs in your application, and submit it to Google Search Console. Implement canonical tags to prevent duplicate content issues when the same content is accessible via multiple URLs. Use semantic HTML elements (header, nav, main, article, section, footer, h1-h6) to help search engines understand your page structure and content hierarchy. Ensure fast page load speeds — Google considers Core Web Vitals (LCP, INP, CLS) as ranking factors, so optimize your JavaScript bundle size, image loading, and server response times. Implement proper 404 handling for non-existent routes. Use the robots.txt file to guide crawlers and prevent indexing of pages that should not appear in search results (admin panels, user settings, authentication pages). Monitor your search performance in Google Search Console, watching for crawl errors, indexing issues, and Core Web Vitals problems that could hurt your rankings.`,
					CodeExamples: `// Dynamic meta tags (React)
import { Helmet } from 'react-helmet';

function Page({ title, description }) {
    return (
        <>
            <Helmet>
                <title>{title}</title>
                <meta name="description" content={description} />
                <meta property="og:title" content={title} />
            </Helmet>
            <div>Content</div>
        </>
    );
}

// Structured data (JSON-LD)
const structuredData = {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": "Article Title",
    "author": {
        "@type": "Person",
        "name": "Author Name"
    }
};

<script type="application/ld+json">
    {JSON.stringify(structuredData)}
</script>

// Pre-rendering with react-snap
// After build, generates static HTML for each route`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          1549,
			Title:       "Advanced Build Optimization",
			Description: "Advanced optimization: tree shaking, code splitting, bundle analysis, and performance tuning.",
			Order:       49,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Build Optimization Techniques",
					Content: `Advanced build optimization is the discipline of reducing the size of your JavaScript bundles, eliminating unused code, and structuring your application's delivery to minimize the time between a user requesting your page and being able to interact with it. In modern frontend applications that can easily accumulate hundreds of npm dependencies and thousands of modules, the difference between an optimized and unoptimized build can be the difference between a sub-second load time and a five-second blank screen. These techniques are not premature optimization — they are essential practices that should be baked into your build pipeline from the start.

**1. Tree Shaking — Eliminating Dead Code**

Tree shaking is the process of analyzing your code's import and export graph to identify and remove modules, functions, and variables that are imported but never actually used. The name comes from the mental image of shaking a tree and letting the dead leaves fall off. This works because ES modules have a static structure — import and export statements must be at the top level and cannot be conditional, so the bundler can determine at build time exactly which exports are used. For tree shaking to work effectively, you must use ES module syntax (import/export) rather than CommonJS (require/module.exports), because CommonJS's dynamic nature makes it impossible to statically analyze which exports are used. This is why importing specific functions (import { debounce } from 'lodash-es') results in a dramatically smaller bundle than importing the entire library (import _ from 'lodash'). Many popular libraries now ship ES module builds specifically to enable tree shaking. The sideEffects field in package.json tells the bundler which files are safe to eliminate if their exports are unused, further improving dead code elimination.

**2. Code Splitting and Dynamic Imports**

Code splitting breaks your application into multiple smaller bundles (chunks) that are loaded on demand rather than all at once in a single massive file. Route-based splitting is the most common approach — each route's components and dependencies are bundled into their own chunk, loaded only when the user navigates to that route. This means a user visiting your home page only downloads the code for the home page, not the code for the admin dashboard, settings panel, or every other page. Dynamic imports (import('./module')) create split points that the bundler uses to determine chunk boundaries. Component-level splitting goes further by lazy-loading heavy widgets — a rich text editor, a chart library, or a date picker — only when they are actually rendered on screen. Vendor splitting separates third-party library code into its own chunk, which can be cached independently from your application code — since libraries change less frequently than your code, users who revisit your site after an update only need to re-download the changed application chunk, not the unchanged vendor chunk.

**3. Bundle Analysis — Understanding What You Ship**

You cannot optimize what you do not measure. Bundle analysis tools like webpack-bundle-analyzer, source-map-explorer, and the Vite build --report flag generate visual treemaps that show exactly what is in each bundle chunk, how large each module is, and where the bloat is coming from. Common findings include surprise large dependencies (a date formatting library that includes locale data for every language, a utility library imported in its entirety for one function), duplicate copies of the same library included at different versions, and polyfills for browser features your target browsers already support. After analysis, the most impactful optimizations are typically replacing heavy dependencies with lighter alternatives (date-fns instead of moment.js, just-debounce instead of lodash), switching to named imports for tree-shakeable libraries, and using dynamic imports to defer loading of dependencies only needed for specific features.

**4. Compression, CDN, and Caching Strategies**

Minification removes whitespace, shortens variable names, and eliminates dead code branches to reduce file sizes — Terser for JavaScript and cssnano for CSS are the standard tools, both typically included in your bundler's production configuration. Compression (Gzip or the more efficient Brotli) further reduces transfer sizes by 60-80% for text-based assets — configure your server or CDN to serve compressed responses. A Content Delivery Network (CDN) serves your static assets from edge servers geographically close to your users, reducing latency from hundreds of milliseconds to single-digit milliseconds. Implement aggressive caching with content-hashed filenames (main.abc123.js) so browsers cache assets indefinitely, only downloading new versions when the hash changes — this means returning users load your site almost instantly from cache. The combination of these techniques — tree shaking to eliminate dead code, code splitting to defer unnecessary code, bundle analysis to find optimization opportunities, minification and compression to reduce transfer sizes, and CDN caching to eliminate redundant downloads — can easily reduce your effective load time by 80% or more.`,
					CodeExamples: `// Tree shaking (ES modules)
// Only imports used code
import { add } from './math'; // Only 'add' included

// Code splitting
const LazyComponent = React.lazy(() => import('./LazyComponent'));

// Dynamic imports
async function loadModule() {
    const module = await import('./heavy-module');
    module.doSomething();
}

// Bundle analysis
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = {
    plugins: [
        new BundleAnalyzerPlugin({
            analyzerMode: 'static',
            openAnalyzer: false
        })
    ]
};

// Optimize imports
// Bad: import * as _ from 'lodash';
// Good: import debounce from 'lodash/debounce';`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
