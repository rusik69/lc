package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          230,
			Title:       "Angular Basics",
			Description: "Introduction to Angular: components, services, dependency injection, and Angular fundamentals.",
			Order:       30,
			Lessons: []problems.Lesson{
				{
					Title: "Angular Introduction",
					Content: `Angular is a platform and framework for building single-page applications.

**What is Angular?**
- TypeScript-based framework
- Component-based architecture
- Dependency injection
- Built-in routing
- Forms handling
- HTTP client

**Key Concepts:**
- Components: UI building blocks
- Services: Business logic
- Modules: Feature organization
- Directives: DOM manipulation
- Pipes: Data transformation

**Angular CLI:**
- ng new: Create project
- ng generate: Generate code
- ng serve: Development server
- ng build: Production build`,
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
					Content: `Angular components and data binding enable reactive user interfaces.

**Data Binding:**
- Interpolation: {{ value }}
- Property binding: [property]="value"
- Event binding: (event)="handler()"
- Two-way binding: [(ngModel)]="value"

**Component Lifecycle:**
- ngOnInit: After first change detection
- ngOnChanges: When inputs change
- ngAfterViewInit: After view initialized
- ngOnDestroy: Before component destroyed

**Directives:**
- *ngIf: Conditional rendering
- *ngFor: List rendering
- [ngClass]: Dynamic classes
- [ngStyle]: Dynamic styles`,
					CodeExamples: `// Component with data binding
@Component({
    selector: 'app-user',
    template: \`
        <h1>{{ user.name }}</h1>
        <input [value]="user.email" (input)="onEmailChange($event)">
        <button [disabled]="!isValid" (click)="save()">Save</button>
        <div *ngIf="showDetails">
            <p>{{ user.bio }}</p>
        </div>
        <ul>
            <li *ngFor="let item of items">{{ item }}</li>
        </ul>
    \`
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
			ID:          231,
			Title:       "Svelte and SvelteKit",
			Description: "Learn Svelte: reactive framework, SvelteKit, and modern web development.",
			Order:       31,
			Lessons: []problems.Lesson{
				{
					Title: "Svelte Introduction",
					Content: `Svelte is a compiler that generates optimized JavaScript.

**What is Svelte?**
- Compile-time framework
- No virtual DOM
- Small bundle sizes
- Reactive by default
- Great performance

**Key Features:**
- Reactive declarations
- Stores for state
- Transitions and animations
- Built-in accessibility
- Component composition

**SvelteKit:**
- Full-stack framework
- File-based routing
- Server-side rendering
- API routes
- Optimized builds`,
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
    $: greeting = \`Hello, \${name}!\`;
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
			ID:          232,
			Title:       "GraphQL and Apollo",
			Description: "Master GraphQL: queries, mutations, Apollo Client, and modern data fetching.",
			Order:       32,
			Lessons: []problems.Lesson{
				{
					Title: "GraphQL Fundamentals",
					Content: `GraphQL is a query language and runtime for APIs.

**What is GraphQL?**
- Query language for APIs
- Single endpoint
- Client specifies data needs
- Strongly typed
- Introspection

**Key Concepts:**
- Queries: Read data
- Mutations: Modify data
- Subscriptions: Real-time updates
- Schema: Type definitions
- Resolvers: Data fetching logic

**Benefits:**
- Fetch only needed data
- Single request for multiple resources
- Strong typing
- Great tooling
- Versioning flexibility`,
					CodeExamples: `// GraphQL Query
query GetUser($id: ID!) {
    user(id: $id) {
        id
        name
        email
        posts {
            title
            content
        }
    }
}

// GraphQL Mutation
mutation CreateUser($input: UserInput!) {
    createUser(input: $input) {
        id
        name
        email
    }
}

// Apollo Client
import { ApolloClient, InMemoryCache, gql } from '@apollo/client';

const client = new ApolloClient({
    uri: 'https://api.example.com/graphql',
    cache: new InMemoryCache()
});

// Query with Apollo
const GET_USERS = gql\`
    query GetUsers {
        users {
            id
            name
        }
    }
\`;

const { data, loading, error } = useQuery(GET_USERS);

// Mutation with Apollo
const CREATE_USER = gql\`
    mutation CreateUser($name: String!) {
        createUser(name: $name) {
            id
            name
        }
    }
\`;

const [createUser] = useMutation(CREATE_USER);`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          233,
			Title:       "CSS Preprocessors (SASS/SCSS)",
			Description: "Master SASS/SCSS: variables, mixins, functions, and advanced CSS preprocessing.",
			Order:       33,
			Lessons: []problems.Lesson{
				{
					Title: "SASS Advanced Features",
					Content: `Advanced SASS features for powerful stylesheet development.

**Advanced Features:**
- Functions: Custom functions
- Control flow: @if, @for, @each, @while
- Interpolation: Dynamic values
- Parent selector: &
- Extend: @extend directive
- Partials: Modular files

**Best Practices:**
- Use partials for organization
- Create reusable mixins
- Use functions for calculations
- Avoid deep nesting
- Keep specificity low`,
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
			ID:          234,
			Title:       "CSS-in-JS Libraries",
			Description: "Explore CSS-in-JS: styled-components, Emotion, and component-scoped styling.",
			Order:       34,
			Lessons: []problems.Lesson{
				{
					Title: "CSS-in-JS Patterns and Best Practices",
					Content: `CSS-in-JS patterns and best practices for maintainable styles.

**Patterns:**
- Styled components: Component-scoped styles
- Theme providers: Global theming
- Dynamic styles: Props-based styling
- Animations: Keyframe animations
- Media queries: Responsive styles

**Performance:**
- Avoid inline styles in loops
- Use CSS variables when possible
- Memoize styled components
- Consider runtime vs compile-time

**Best Practices:**
- Organize styled components
- Use theme consistently
- Extract common patterns
- Test styled components`,
					CodeExamples: `// styled-components patterns
const Button = styled.button\`
    background: \${props => props.variant === 'primary' ? 'blue' : 'gray'};
    padding: \${props => props.size === 'large' ? '15px' : '10px'};
    
    &:hover {
        opacity: 0.8;
    }
    
    @media (max-width: 768px) {
        width: 100%;
    }
\`;

// Theme with TypeScript
interface Theme {
    colors: {
        primary: string;
        secondary: string;
    };
}

const ThemedButton = styled.button<{ theme: Theme }>\`
    background: \${props => props.theme.colors.primary};
\`;

// Animations
const fadeIn = keyframes\`
    from { opacity: 0; }
    to { opacity: 1; }
\`;

const FadeInBox = styled.div\`
    animation: \${fadeIn} 0.5s;
\`;`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          235,
			Title:       "React Router and Navigation",
			Description: "Master React Router: routing, navigation, protected routes, and advanced patterns.",
			Order:       35,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced React Router",
					Content: `Advanced React Router patterns for complex applications.

**Advanced Features:**
- Nested routes
- Route parameters
- Query strings
- Route guards
- Lazy loading routes
- Route transitions

**Patterns:**
- Layout routes
- Outlet for nested routes
- useSearchParams for queries
- useLocation for location data
- Navigate component

**Best Practices:**
- Organize routes logically
- Use route guards
- Lazy load routes
- Handle 404 pages
- Use relative navigation`,
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
			ID:          236,
			Title:       "Vue Router and Navigation",
			Description: "Master Vue Router: routing, navigation guards, and advanced Vue navigation patterns.",
			Order:       36,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Vue Router",
					Content: `Advanced Vue Router features for complex navigation.

**Advanced Features:**
- Navigation guards
- Route meta fields
- Dynamic route matching
- Programmatic navigation
- Route transitions
- Scroll behavior

**Guards:**
- beforeEach: Before navigation
- beforeResolve: Before resolve
- afterEach: After navigation
- beforeEnter: Route-specific

**Best Practices:**
- Use guards for auth
- Handle route errors
- Provide loading states
- Use route meta for permissions`,
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
			ID:          237,
			Title:       "WebSockets and Real-time",
			Description: "Build real-time applications: WebSockets, Socket.io, and real-time communication.",
			Order:       37,
			Lessons: []problems.Lesson{
				{
					Title: "WebSockets and Real-time Communication",
					Content: `WebSockets enable bidirectional real-time communication.

**WebSocket API:**
- Full-duplex communication
- Persistent connection
- Low latency
- Event-driven
- Binary and text data

**Socket.io:**
- WebSocket abstraction
- Fallback transports
- Rooms and namespaces
- Automatic reconnection
- Event-based API

**Use Cases:**
- Chat applications
- Live updates
- Gaming
- Collaboration tools
- Real-time dashboards

**Best Practices:**
- Handle reconnection
- Manage connection state
- Clean up on unmount
- Handle errors gracefully`,
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
			ID:          238,
			Title:       "Internationalization (i18n)",
			Description: "Implement i18n: multi-language support, locale management, and translation systems.",
			Order:       38,
			Lessons: []problems.Lesson{
				{
					Title: "i18n Implementation",
					Content: `Internationalization enables applications to support multiple languages.

**i18n Concepts:**
- Locale: Language and region
- Translation files: Key-value pairs
- Pluralization: Handle plurals
- Date/number formatting: Locale-specific
- RTL support: Right-to-left languages

**Libraries:**
- react-i18next: React i18n
- vue-i18n: Vue i18n
- Format.js: ICU message format
- i18next: Framework agnostic

**Best Practices:**
- Extract all strings
- Use translation keys
- Handle pluralization
- Format dates/numbers
- Test all locales`,
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
			ID:          239,
			Title:       "Animation Libraries",
			Description: "Master animation: Framer Motion, GSAP, and creating smooth animations.",
			Order:       39,
			Lessons: []problems.Lesson{
				{
					Title: "Animation Libraries and Techniques",
					Content: `Animation libraries provide powerful tools for creating engaging interfaces.

**Framer Motion:**
- React animation library
- Declarative animations
- Gesture support
- Layout animations
- Simple API

**GSAP:**
- Professional animation library
- Timeline control
- ScrollTrigger
- Performance optimized
- Framework agnostic

**CSS Animations:**
- @keyframes
- Transitions
- Transform
- Performance considerations

**Best Practices:**
- Use GPU-accelerated properties
- Respect prefers-reduced-motion
- Optimize for performance
- Provide meaningful animations`,
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
			ID:          240,
			Title:       "Canvas API and Graphics",
			Description: "Create graphics with Canvas API: drawing, animations, and interactive graphics.",
			Order:       40,
			Lessons: []problems.Lesson{
				{
					Title: "Canvas API Fundamentals",
					Content: `Canvas API enables programmatic drawing and graphics.

**Canvas Basics:**
- Get 2D context
- Drawing shapes
- Paths and curves
- Text rendering
- Image manipulation

**Drawing Operations:**
- fillRect: Filled rectangles
- strokeRect: Outlined rectangles
- arc: Circles and arcs
- lineTo: Lines
- fillText: Text

**Transformations:**
- translate: Move origin
- rotate: Rotate context
- scale: Scale context
- save/restore: State management

**Animations:**
- requestAnimationFrame
- Clear and redraw
- Smooth animations
- Performance optimization`,
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
			ID:          241,
			Title:       "WebGL and Three.js",
			Description: "3D graphics with WebGL and Three.js: 3D rendering, scenes, and interactive 3D applications.",
			Order:       41,
			Lessons: []problems.Lesson{
				{
					Title: "WebGL and Three.js Basics",
					Content: `WebGL and Three.js enable 3D graphics in the browser.

**WebGL:**
- Low-level 3D graphics API
- GPU-accelerated
- Shader programming
- Complex but powerful

**Three.js:**
- High-level 3D library
- Scene graph
- Built-in geometries
- Materials and lights
- Camera controls

**Key Concepts:**
- Scene: 3D world
- Camera: Viewpoint
- Renderer: Drawing engine
- Mesh: Geometry + Material
- Lights: Illumination

**Use Cases:**
- 3D visualizations
- Games
- Product showcases
- Data visualization
- Interactive experiences`,
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
			ID:          242,
			Title:       "Web Audio API",
			Description: "Create audio applications: Web Audio API, sound synthesis, and audio processing.",
			Order:       42,
			Lessons: []problems.Lesson{
				{
					Title: "Web Audio API Fundamentals",
					Content: `Web Audio API enables audio processing and synthesis.

**Web Audio API:**
- Audio context
- Audio nodes
- Audio routing
- Real-time processing
- Sound synthesis

**Key Concepts:**
- AudioContext: Audio processing context
- AudioNode: Processing units
- Oscillator: Sound generation
- GainNode: Volume control
- AnalyserNode: Audio analysis

**Use Cases:**
- Music players
- Sound effects
- Audio visualization
- Synthesizers
- Audio games`,
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
			ID:          243,
			Title:       "WebRTC",
			Description: "Real-time communication with WebRTC: peer-to-peer connections, video/audio streaming.",
			Order:       43,
			Lessons: []problems.Lesson{
				{
					Title: "WebRTC Fundamentals",
					Content: `WebRTC enables peer-to-peer communication in browsers.

**WebRTC Features:**
- Peer-to-peer connections
- Video/audio streaming
- Data channels
- NAT traversal
- Encryption

**Key Components:**
- RTCPeerConnection: Peer connection
- MediaStream: Audio/video stream
- RTCDataChannel: Data channel
- Signaling: Connection setup

**Use Cases:**
- Video conferencing
- Voice calls
- Screen sharing
- File transfer
- Gaming

**Signaling:**
- Exchange SDP offers/answers
- ICE candidate exchange
- Usually via WebSocket or HTTP`,
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
			ID:          244,
			Title:       "CI/CD for Frontend",
			Description: "Implement CI/CD pipelines: GitHub Actions, automated testing, and deployment.",
			Order:       44,
			Lessons: []problems.Lesson{
				{
					Title: "CI/CD Pipeline Setup",
					Content: `CI/CD automates testing and deployment for frontend applications.

**CI/CD Concepts:**
- Continuous Integration: Automated testing
- Continuous Deployment: Automated deployment
- Pipeline stages: Build, test, deploy
- Environment management
- Rollback strategies

**GitHub Actions:**
- Workflow files
- Matrix builds
- Caching
- Secrets management
- Deployment actions

**Stages:**
- Install dependencies
- Lint code
- Run tests
- Build application
- Deploy to staging/production

**Best Practices:**
- Run tests on every commit
- Use environment variables
- Cache dependencies
- Deploy to staging first
- Monitor deployments`,
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
			ID:          245,
			Title:       "Docker for Frontend",
			Description: "Containerize frontend applications: Docker, multi-stage builds, and containerization.",
			Order:       45,
			Lessons: []problems.Lesson{
				{
					Title: "Dockerizing Frontend Applications",
					Content: `Docker enables consistent deployment environments for frontend apps.

**Docker Concepts:**
- Images: Application snapshots
- Containers: Running instances
- Dockerfile: Build instructions
- Multi-stage builds: Optimize images
- Docker Compose: Multi-container apps

**Frontend Dockerfile:**
- Base image (Node.js)
- Install dependencies
- Build application
- Serve with nginx
- Multi-stage for optimization

**Best Practices:**
- Use multi-stage builds
- Minimize image size
- Use .dockerignore
- Cache dependencies
- Use specific base images`,
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
			ID:          246,
			Title:       "Monitoring and Error Tracking",
			Description: "Monitor applications: error tracking, performance monitoring, and observability.",
			Order:       46,
			Lessons: []problems.Lesson{
				{
					Title: "Error Tracking and Monitoring",
					Content: `Monitoring and error tracking ensure application reliability.

**Error Tracking:**
- Sentry: Error tracking
- LogRocket: Session replay
- Bugsnag: Error monitoring
- Track errors in production
- Source maps for debugging

**Performance Monitoring:**
- Real User Monitoring (RUM)
- Synthetic monitoring
- Core Web Vitals
- API performance
- Resource loading

**Logging:**
- Structured logging
- Log levels
- Centralized logging
- Log aggregation

**Best Practices:**
- Track all errors
- Include context
- Monitor performance
- Set up alerts
- Review regularly`,
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
			ID:          247,
			Title:       "Security Best Practices",
			Description: "Secure frontend applications: XSS prevention, CSRF protection, and security patterns.",
			Order:       47,
			Lessons: []problems.Lesson{
				{
					Title: "Frontend Security",
					Content: `Security is crucial for protecting users and applications.

**Common Vulnerabilities:**
- XSS: Cross-Site Scripting
- CSRF: Cross-Site Request Forgery
- Clickjacking: UI redressing
- Man-in-the-middle: Interception
- Dependency vulnerabilities

**Protection Strategies:**
- Input sanitization
- Content Security Policy
- HTTPS only
- Secure cookies
- Dependency updates

**Best Practices:**
- Sanitize user input
- Use CSP headers
- Validate on client and server
- Keep dependencies updated
- Use security headers`,
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
			ID:          248,
			Title:       "SEO for SPAs",
			Description: "Optimize SPAs for search engines: SSR, meta tags, structured data, and SEO strategies.",
			Order:       48,
			Lessons: []problems.Lesson{
				{
					Title: "SEO Optimization for Single Page Applications",
					Content: `SEO is challenging for SPAs but essential for discoverability.

**SPA SEO Challenges:**
- JavaScript rendering
- Dynamic content
- Client-side routing
- Meta tag management

**Solutions:**
- Server-side rendering
- Pre-rendering
- Dynamic meta tags
- Structured data
- Sitemap generation

**Best Practices:**
- Use SSR or pre-rendering
- Set meta tags dynamically
- Provide fallback content
- Use semantic HTML
- Optimize page speed`,
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
			ID:          249,
			Title:       "Advanced Build Optimization",
			Description: "Advanced optimization: tree shaking, code splitting, bundle analysis, and performance tuning.",
			Order:       49,
			Lessons: []problems.Lesson{
				{
					Title: "Advanced Build Optimization Techniques",
					Content: `Advanced techniques for optimizing frontend builds.

**Optimization Techniques:**
- Tree shaking: Remove unused code
- Code splitting: Split bundles
- Minification: Reduce size
- Compression: Gzip/Brotli
- Asset optimization: Images, fonts

**Bundle Analysis:**
- Analyze bundle size
- Identify large dependencies
- Find duplicate code
- Optimize imports
- Use dynamic imports

**Performance:**
- Reduce initial bundle
- Lazy load routes
- Optimize images
- Use CDN
- Cache strategies`,
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
