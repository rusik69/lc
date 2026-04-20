package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1562,
			Title:       "Web Animations and Motion Design",
			Description: "Implement performant animations with CSS transitions, Web Animations API, Framer Motion, and GSAP for interactive user experiences.",
			Order:       62,
			Lessons: []problems.Lesson{
				{
					Title: "CSS Animations and Transitions",
					Content: `CSS provides powerful animation capabilities without JavaScript for smooth, hardware-accelerated motion.

**CSS Transitions:**
` + "```" + `css
/* Basic transitions */
.button {
  background: #3b82f6;
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  transition: all 0.2s ease-in-out;
}

.button:hover {
  background: #2563eb;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
}

/* Individual property transitions */
.card {
  transition-property: transform, box-shadow, opacity;
  transition-duration: 0.3s, 0.3s, 0.2s;
  transition-timing-function: ease-out, ease-out, ease-in;
}

/* Custom timing functions */
.bounce-in {
  transition: transform 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

.smooth-out {
  transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Prefers-reduced-motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
` + "```" + `

**CSS Keyframe Animations:**
` + "```" + `css
/* Fade in up */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s ease-out forwards;
}

/* Staggered children */
.stagger-container > * {
  opacity: 0;
  animation: fadeInUp 0.4s ease-out forwards;
}

.stagger-container > *:nth-child(1) { animation-delay: 0ms; }
.stagger-container > *:nth-child(2) { animation-delay: 100ms; }
.stagger-container > *:nth-child(3) { animation-delay: 200ms; }
.stagger-container > *:nth-child(4) { animation-delay: 300ms; }
.stagger-container > *:nth-child(5) { animation-delay: 400ms; }

/* Skeleton loading */
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.skeleton {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

.skeleton-text {
  height: 1em;
  margin-bottom: 0.5em;
}

.skeleton-text:last-child {
  width: 60%;
}

/* Spinning loader */
@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

/* Pulse */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* Shake (error feedback) */
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
  20%, 40%, 60%, 80% { transform: translateX(4px); }
}

.shake {
  animation: shake 0.6s ease-in-out;
}

/* Scroll-triggered animations */
.scroll-reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.6s, transform 0.6s;
}

.scroll-reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* View Transitions API */
::view-transition-old(page) {
  animation: fade-out 0.3s ease-out;
}

::view-transition-new(page) {
  animation: fade-in 0.3s ease-in;
}

.hero-image {
  view-transition-name: hero;
}

/* Container transform animation */
@keyframes expand {
  from {
    clip-path: circle(0% at var(--origin-x) var(--origin-y));
  }
  to {
    clip-path: circle(150% at var(--origin-x) var(--origin-y));
  }
}
` + "```" + ``,
					CodeExamples: `// JavaScript animation patterns

// 1. Intersection Observer for scroll animations
function useScrollAnimation(options = {}) {
  const ref = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (options.once !== false) {
            observer.unobserve(entry.target);
          }
        } else if (options.once === false) {
          setIsVisible(false);
        }
      },
      {
        threshold: options.threshold || 0.1,
        rootMargin: options.rootMargin || '0px',
      }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return { ref, isVisible };
}

// Usage
function AnimatedCard({ children }) {
  const { ref, isVisible } = useScrollAnimation({ threshold: 0.2 });

  return (
    <div
      ref={ref}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'translateY(0)' : 'translateY(30px)',
        transition: 'opacity 0.6s ease-out, transform 0.6s ease-out',
      }}
    >
      {children}
    </div>
  );
}

// 2. Web Animations API
function animateElement(element, keyframes, options) {
  const animation = element.animate(keyframes, {
    duration: 300,
    easing: 'ease-out',
    fill: 'forwards',
    ...options,
  });
  return animation.finished;
}

// Flip animation technique
function flipAnimate(element, first, last) {
  const deltaX = first.left - last.left;
  const deltaY = first.top - last.top;
  const deltaW = first.width / last.width;
  const deltaH = first.height / last.height;

  element.animate([
    {
      transform: 'translate(' + deltaX + 'px, ' + deltaY + 'px) scale(' + deltaW + ', ' + deltaH + ')',
    },
    { transform: 'none' },
  ], {
    duration: 300,
    easing: 'ease-out',
  });
}

// 3. Spring animation (physics-based)
function spring({ from, to, stiffness = 100, damping = 10, mass = 1, onUpdate }) {
  let position = from;
  let velocity = 0;
  let animationFrame;

  function step() {
    const force = -stiffness * (position - to);
    const dampingForce = -damping * velocity;
    const acceleration = (force + dampingForce) / mass;
    
    velocity += acceleration * (1 / 60);
    position += velocity * (1 / 60);

    onUpdate(position);

    if (Math.abs(velocity) > 0.01 || Math.abs(position - to) > 0.01) {
      animationFrame = requestAnimationFrame(step);
    } else {
      onUpdate(to);
    }
  }

  animationFrame = requestAnimationFrame(step);
  return () => cancelAnimationFrame(animationFrame);
}`,
				},
				{
					Title: "Framer Motion and Advanced Animation Libraries",
					Content: `Framer Motion provides a declarative animation API for React with gesture support and layout animations.

**Framer Motion Basics:**
` + "```" + `javascript
import { motion, AnimatePresence } from 'framer-motion';

// Basic animation
function FadeIn({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  );
}

// Variants for orchestrated animations
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
  exit: {
    opacity: 0,
    transition: { staggerChildren: 0.05, staggerDirection: -1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

function StaggeredList({ items }) {
  return (
    <motion.ul
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
    >
      {items.map((item) => (
        <motion.li key={item.id} variants={itemVariants}>
          {item.label}
        </motion.li>
      ))}
    </motion.ul>
  );
}

// Layout animations
function ExpandableCard({ title, content }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      layout
      onClick={() => setIsExpanded(!isExpanded)}
      style={{ borderRadius: '12px', overflow: 'hidden' }}
    >
      <motion.h3 layout="position">{title}</motion.h3>
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {content}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Gesture animations
function DraggableCard() {
  return (
    <motion.div
      drag
      dragConstraints={{ left: -100, right: 100, top: -50, bottom: 50 }}
      dragElastic={0.2}
      whileDrag={{ scale: 1.1, cursor: 'grabbing' }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      Drag me!
    </motion.div>
  );
}

// Scroll-linked animations
function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  return (
    <motion.div
      style={{
        scaleX,
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '4px',
        background: '#3b82f6',
        transformOrigin: '0%',
      }}
    />
  );
}

// Shared layout animations (page transitions)
function Gallery({ items }) {
  const [selectedId, setSelectedId] = useState(null);

  return (
    <>
      <div className="grid">
        {items.map((item) => (
          <motion.div
            key={item.id}
            layoutId={'card-' + item.id}
            onClick={() => setSelectedId(item.id)}
          >
            <motion.img layoutId={'img-' + item.id} src={item.image} />
            <motion.h3 layoutId={'title-' + item.id}>{item.title}</motion.h3>
          </motion.div>
        ))}
      </div>
      
      <AnimatePresence>
        {selectedId && (
          <motion.div
            layoutId={'card-' + selectedId}
            className="overlay"
            onClick={() => setSelectedId(null)}
          >
            <motion.img
              layoutId={'img-' + selectedId}
              src={items.find(i => i.id === selectedId).image}
            />
            <motion.h3 layoutId={'title-' + selectedId}>
              {items.find(i => i.id === selectedId).title}
            </motion.h3>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {items.find(i => i.id === selectedId).description}
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// Path drawing animation
function DrawSVG() {
  return (
    <motion.svg width="200" height="200" viewBox="0 0 200 200">
      <motion.circle
        cx="100"
        cy="100"
        r="80"
        stroke="#3b82f6"
        strokeWidth="4"
        fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 2, ease: "easeInOut" }}
      />
      <motion.path
        d="M 60 100 L 90 130 L 140 70"
        stroke="#10b981"
        strokeWidth="4"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.5, delay: 1.5 }}
      />
    </motion.svg>
  );
}
` + "```" + ``,
					CodeExamples: `// Advanced animation patterns

// 1. Page transition wrapper
const pageTransition = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 20 },
  transition: { duration: 0.3, ease: 'easeInOut' },
};

function PageWrapper({ children }) {
  return (
    <motion.div {...pageTransition}>
      {children}
    </motion.div>
  );
}

// In router
function AnimatedRoutes() {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
        <Route path="/about" element={<PageWrapper><About /></PageWrapper>} />
      </Routes>
    </AnimatePresence>
  );
}

// 2. Animated counter
function AnimatedCounter({ value, duration = 1 }) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, Math.round);
  const display = useMotionTemplate(rounded);

  useEffect(() => {
    const animation = animate(count, value, { duration });
    return animation.stop;
  }, [value]);

  return <motion.span>{display}</motion.span>;
}

// 3. Parallax effect
function useParallax(distance = 100) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });
  const y = useTransform(scrollYProgress, [0, 1], [-distance, distance]);
  return { ref, y };
}

function ParallaxSection({ children, distance }) {
  const { ref, y } = useParallax(distance);
  
  return (
    <div ref={ref} style={{ overflow: 'hidden' }}>
      <motion.div style={{ y }}>
        {children}
      </motion.div>
    </div>
  );
}

// 4. Morphing shapes
function MorphingShape() {
  const [isCircle, setIsCircle] = useState(true);

  return (
    <motion.div
      animate={{
        borderRadius: isCircle ? '50%' : '16px',
        rotate: isCircle ? 0 : 45,
        scale: isCircle ? 1 : 1.2,
        backgroundColor: isCircle ? '#3b82f6' : '#ef4444',
      }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
      onClick={() => setIsCircle(!isCircle)}
      style={{ width: 100, height: 100, cursor: 'pointer' }}
    />
  );
}

// 5. Gesture-based card stack (Tinder-like)
function CardStack({ cards, onSwipe }) {
  return (
    <div style={{ position: 'relative', width: 300, height: 400 }}>
      {cards.map((card, i) => (
        <motion.div
          key={card.id}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            zIndex: cards.length - i,
          }}
          drag={i === 0 ? 'x' : false}
          dragConstraints={{ left: 0, right: 0 }}
          onDragEnd={(e, { offset, velocity }) => {
            const swipe = Math.abs(offset.x) * velocity.x;
            if (swipe < -10000) {
              onSwipe(card, 'left');
            } else if (swipe > 10000) {
              onSwipe(card, 'right');
            }
          }}
          animate={{
            scale: 1 - i * 0.05,
            y: i * 10,
          }}
        >
          {card.content}
        </motion.div>
      ))}
    </div>
  );
}

// 6. GSAP integration
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

function useGSAP(animationFn, deps = []) {
  const ref = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      animationFn(ref.current);
    }, ref);

    return () => ctx.revert();
  }, deps);

  return ref;
}

// Usage
function HeroSection() {
  const ref = useGSAP((el) => {
    gsap.from(el.querySelectorAll('.animate'), {
      y: 60,
      opacity: 0,
      duration: 1,
      stagger: 0.2,
      ease: 'power3.out',
      scrollTrigger: {
        trigger: el,
        start: 'top 80%',
        end: 'bottom 20%',
        toggleActions: 'play none none reverse',
      },
    });
  });

  return (
    <section ref={ref}>
      <h1 className="animate">Welcome</h1>
      <p className="animate">Description</p>
      <button className="animate">Get Started</button>
    </section>
  );
}`,
				},
			},
		},
		{
			ID:          1563,
			Title:       "GraphQL and Data Layer Patterns",
			Description: "Build type-safe data layers with GraphQL, Apollo Client, urql, tRPC, and React Query for efficient data fetching and caching.",
			Order:       63,
			Lessons: []problems.Lesson{
				{
					Title: "GraphQL Client Architecture",
					Content: `GraphQL provides a flexible query language for APIs, allowing clients to request exactly the data they need.

**Apollo Client Setup:**
` + "```" + `javascript
// apollo-client.js
import {
  ApolloClient,
  InMemoryCache,
  createHttpLink,
  from,
} from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
import { onError } from '@apollo/client/link/error';
import { RetryLink } from '@apollo/client/link/retry';

// Auth link
const authLink = setContext((_, { headers }) => {
  const token = localStorage.getItem('token');
  return {
    headers: {
      ...headers,
      authorization: token ? 'Bearer ' + token : '',
    },
  };
});

// Error handling link
const errorLink = onError(({ graphQLErrors, networkError, operation }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path, extensions }) => {
      console.error(
        '[GraphQL error]:',
        { message, locations, path, code: extensions?.code }
      );
      
      if (extensions?.code === 'UNAUTHENTICATED') {
        // Redirect to login or refresh token
        window.location.href = '/login';
      }
    });
  }
  if (networkError) {
    console.error('[Network error]:', networkError);
  }
});

// Retry link for network failures
const retryLink = new RetryLink({
  delay: { initial: 300, max: 5000, jitter: true },
  attempts: { max: 3, retryIf: (error) => !!error },
});

// HTTP link
const httpLink = createHttpLink({
  uri: '/graphql',
});

// Create client
const client = new ApolloClient({
  link: from([errorLink, retryLink, authLink, httpLink]),
  cache: new InMemoryCache({
    typePolicies: {
      Query: {
        fields: {
          // Cursor-based pagination
          posts: {
            keyArgs: ['filter'],
            merge(existing, incoming, { args }) {
              if (!args?.after) return incoming;
              return {
                ...incoming,
                edges: [...(existing?.edges || []), ...incoming.edges],
              };
            },
          },
          // Offset pagination
          comments: {
            keyArgs: ['postId'],
            merge(existing = [], incoming, { args }) {
              const merged = existing.slice(0);
              const offset = args?.offset || 0;
              for (let i = 0; i < incoming.length; i++) {
                merged[offset + i] = incoming[i];
              }
              return merged;
            },
          },
        },
      },
      User: {
        fields: {
          fullName: {
            read(_, { readField }) {
              const first = readField('firstName');
              const last = readField('lastName');
              return first + ' ' + last;
            },
          },
        },
      },
    },
  }),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
      nextFetchPolicy: 'cache-first',
    },
  },
});

// Queries and Mutations
import { gql, useQuery, useMutation } from '@apollo/client';

const GET_POSTS = gql'
  query GetPosts($filter: PostFilter, $after: String, $first: Int) {
    posts(filter: $filter, after: $after, first: $first) {
      edges {
        node {
          id
          title
          content
          author {
            id
            name
            avatar
          }
          createdAt
          commentsCount
        }
        cursor
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
';

const CREATE_POST = gql'
  mutation CreatePost($input: CreatePostInput!) {
    createPost(input: $input) {
      id
      title
      content
      author {
        id
        name
      }
    }
  }
';

// Component usage
function PostList({ filter }) {
  const { data, loading, error, fetchMore } = useQuery(GET_POSTS, {
    variables: { filter, first: 20 },
    notifyOnNetworkStatusChange: true,
  });

  const [createPost] = useMutation(CREATE_POST, {
    update(cache, { data: { createPost } }) {
      cache.modify({
        fields: {
          posts(existingPosts = { edges: [] }) {
            const newPostRef = cache.writeFragment({
              data: createPost,
              fragment: gql'
                fragment NewPost on Post {
                  id title content author { id name }
                }
              ',
            });
            return {
              ...existingPosts,
              edges: [
                { node: newPostRef, cursor: createPost.id },
                ...existingPosts.edges,
              ],
            };
          },
        },
      });
    },
    optimisticResponse: {
      createPost: {
        __typename: 'Post',
        id: 'temp-' + Date.now(),
        title: 'New Post',
        content: '',
        author: { __typename: 'User', id: 'me', name: 'Me' },
      },
    },
  });

  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      {data?.posts.edges.map(({ node }) => (
        <PostCard key={node.id} post={node} />
      ))}
      {data?.posts.pageInfo.hasNextPage && (
        <button
          onClick={() => fetchMore({
            variables: { after: data.posts.pageInfo.endCursor },
          })}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Load More'}
        </button>
      )}
    </div>
  );
}
` + "```" + ``,
					CodeExamples: `// Advanced GraphQL patterns

// 1. GraphQL Code Generator setup
// codegen.ts
import type { CodegenConfig } from '@graphql-codegen/cli';

const config: CodegenConfig = {
  schema: 'http://localhost:4000/graphql',
  documents: ['src/**/*.{ts,tsx}'],
  generates: {
    './src/generated/graphql.ts': {
      plugins: [
        'typescript',
        'typescript-operations',
        'typescript-react-apollo',
      ],
      config: {
        withHooks: true,
        withComponent: false,
      },
    },
  },
};

export default config;

// Generated hooks usage
import { useGetPostsQuery, useCreatePostMutation } from './generated/graphql';

function Posts() {
  const { data, loading } = useGetPostsQuery({
    variables: { first: 20, filter: { published: true } },
  });
  
  const [createPost] = useCreatePostMutation();
  // Fully typed!
}

// 2. Fragment colocation
const USER_FIELDS = gql(
  'fragment UserFields on User { id name email avatar }'
);

const POST_WITH_AUTHOR = gql(
  'fragment PostWithAuthor on Post { id title content author { ...UserFields } }' +
  USER_FIELDS
);

// 3. Subscriptions for real-time data
const COMMENT_ADDED = gql(
  'subscription OnCommentAdded($postId: ID!) { commentAdded(postId: $postId) { id content author { id name } createdAt } }'
);

function PostComments({ postId }) {
  const { data: commentsData } = useQuery(GET_COMMENTS, {
    variables: { postId },
  });

  useSubscription(COMMENT_ADDED, {
    variables: { postId },
    onData: ({ client, data }) => {
      client.cache.modify({
        id: client.cache.identify({ __typename: 'Post', id: postId }),
        fields: {
          comments(existing = []) {
            const newRef = client.cache.writeFragment({
              data: data.data.commentAdded,
              fragment: gql('fragment NewComment on Comment { id content author { id name } createdAt }'),
            });
            return [...existing, newRef];
          },
        },
      });
    },
  });

  return (
    <ul>
      {commentsData?.comments.map((comment) => (
        <li key={comment.id}>
          <strong>{comment.author.name}</strong>: {comment.content}
        </li>
      ))}
    </ul>
  );
}

// 4. tRPC - End-to-end type safety without GraphQL
// server/trpc.ts
import { initTRPC, TRPCError } from '@trpc/server';
import { z } from 'zod';

const t = initTRPC.context().create();

const isAuthed = t.middleware(({ ctx, next }) => {
  if (!ctx.user) throw new TRPCError({ code: 'UNAUTHORIZED' });
  return next({ ctx: { ...ctx, user: ctx.user } });
});

const protectedProcedure = t.procedure.use(isAuthed);

export const appRouter = t.router({
  post: t.router({
    list: t.procedure
      .input(z.object({
        cursor: z.string().optional(),
        limit: z.number().min(1).max(100).default(20),
      }))
      .query(async ({ input }) => {
        const posts = await db.post.findMany({
          take: input.limit + 1,
          cursor: input.cursor ? { id: input.cursor } : undefined,
          orderBy: { createdAt: 'desc' },
        });
        
        let nextCursor;
        if (posts.length > input.limit) {
          const nextItem = posts.pop();
          nextCursor = nextItem.id;
        }
        
        return { posts, nextCursor };
      }),
    
    create: protectedProcedure
      .input(z.object({
        title: z.string().min(1).max(200),
        content: z.string().min(1),
      }))
      .mutation(async ({ input, ctx }) => {
        return db.post.create({
          data: { ...input, authorId: ctx.user.id },
        });
      }),
  }),
});

// Client usage - fully type-safe
import { trpc } from './utils/trpc';

function PostList() {
  const postsQuery = trpc.post.list.useInfiniteQuery(
    { limit: 20 },
    { getNextPageParam: (lastPage) => lastPage.nextCursor }
  );

  const createPost = trpc.post.create.useMutation({
    onSuccess: () => {
      postsQuery.refetch();
    },
  });

  return <div>{/* render posts */}</div>;
}`,
				},
			},
		},
	})
}
