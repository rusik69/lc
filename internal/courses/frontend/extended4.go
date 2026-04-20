package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1554,
			Title:       "State Management Architecture",
			Description: "Compare and implement modern state management solutions including Zustand, Jotai, Redux Toolkit, and server state with TanStack Query.",
			Order:       54,
			Lessons: []problems.Lesson{
				{
					Title: "Client State Management",
					Content: `Modern frontend state management has evolved beyond Redux to simpler, more focused solutions.

**Zustand:**
` + "```" + `javascript
// Zustand - minimalist state management
import { create } from 'zustand';
import { devtools, persist, immer } from 'zustand/middleware';

// Basic store
const useStore = create((set, get) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  reset: () => set({ count: 0 }),
  getDoubled: () => get().count * 2,
}));

// With middleware (devtools + persist + immer)
const useAppStore = create(
  devtools(
    persist(
      immer((set, get) => ({
        user: null,
        theme: 'light',
        notifications: [],
        
        setUser: (user) => set((state) => { state.user = user; }),
        
        toggleTheme: () => set((state) => {
          state.theme = state.theme === 'light' ? 'dark' : 'light';
        }),
        
        addNotification: (notification) => set((state) => {
          state.notifications.push({
            id: Date.now(),
            ...notification,
            read: false,
          });
        }),
        
        markRead: (id) => set((state) => {
          const notif = state.notifications.find(n => n.id === id);
          if (notif) notif.read = true;
        }),
        
        unreadCount: () => get().notifications.filter(n => !n.read).length,
      })),
      { name: 'app-store' }
    )
  )
);

// Sliced stores for better organization
const createUserSlice = (set) => ({
  user: null,
  setUser: (user) => set({ user }),
  logout: () => set({ user: null }),
});

const createCartSlice = (set) => ({
  items: [],
  addItem: (item) => set((state) => ({
    items: [...state.items, item],
  })),
  removeItem: (id) => set((state) => ({
    items: state.items.filter(i => i.id !== id),
  })),
  total: () => 0, // compute in selector
});

const useBoundStore = create((...a) => ({
  ...createUserSlice(...a),
  ...createCartSlice(...a),
}));

// Selectors for performance
function CartTotal() {
  // Only re-renders when items change
  const total = useBoundStore(
    (state) => state.items.reduce((sum, item) => sum + item.price, 0)
  );
  return <span>${total.toFixed(2)}</span>;
}

// Shallow comparison for object selectors
import { shallow } from 'zustand/shallow';

function UserInfo() {
  const { name, email } = useBoundStore(
    (state) => ({ name: state.user?.name, email: state.user?.email }),
    shallow
  );
  return <div>{name} ({email})</div>;
}
` + "```" + `

**Jotai (Atomic State):**
` + "```" + `javascript
// Jotai - primitive and flexible atomic state
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';
import { atomWithStorage, atomWithDefault } from 'jotai/utils';

// Primitive atoms
const countAtom = atom(0);
const nameAtom = atom('');
const darkModeAtom = atomWithStorage('darkMode', false);

// Derived atoms (read-only)
const doubleCountAtom = atom((get) => get(countAtom) * 2);

const fullNameAtom = atom((get) => {
  const first = get(firstNameAtom);
  const last = get(lastNameAtom);
  return first && last ? ` + "`" + `${first} ${last}` + "`" + ` : '';
});

// Writable derived atoms
const uppercaseNameAtom = atom(
  (get) => get(nameAtom).toUpperCase(),
  (get, set, newName) => set(nameAtom, newName)
);

// Async atoms
const userAtom = atom(async (get) => {
  const id = get(userIdAtom);
  const response = await fetch(` + "`" + `/api/users/${id}` + "`" + `);
  return response.json();
});

// Atom with reducer
const todosAtom = atom([]);
const todosReducerAtom = atom(
  (get) => get(todosAtom),
  (get, set, action) => {
    const todos = get(todosAtom);
    switch (action.type) {
      case 'ADD':
        set(todosAtom, [...todos, { id: Date.now(), text: action.text, done: false }]);
        break;
      case 'TOGGLE':
        set(todosAtom, todos.map(t =>
          t.id === action.id ? { ...t, done: !t.done } : t
        ));
        break;
      case 'REMOVE':
        set(todosAtom, todos.filter(t => t.id !== action.id));
        break;
    }
  }
);

// Atom families (parameterized atoms)
import { atomFamily } from 'jotai/utils';

const userAtomFamily = atomFamily((userId) =>
  atom(async () => {
    const res = await fetch(` + "`" + `/api/users/${userId}` + "`" + `);
    return res.json();
  })
);

// Usage
function UserProfile({ userId }) {
  const [user] = useAtom(userAtomFamily(userId));
  return <div>{user.name}</div>;
}

// Component usage
function Counter() {
  const [count, setCount] = useAtom(countAtom);
  const doubled = useAtomValue(doubleCountAtom);
  
  return (
    <div>
      <p>Count: {count} (Double: {doubled})</p>
      <button onClick={() => setCount(c => c + 1)}>+</button>
    </div>
  );
}
` + "```" + ``,
					CodeExamples: `// State management examples

// 1. Redux Toolkit modern patterns
import { configureStore, createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

// Slice with async thunks
const usersSlice = createSlice({
  name: 'users',
  initialState: { list: [], loading: false, error: null },
  reducers: {
    clearUsers: (state) => { state.list = []; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.loading = false;
        state.list = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

const fetchUsers = createAsyncThunk('users/fetch', async () => {
  const res = await fetch('/api/users');
  if (!res.ok) throw new Error('Failed to fetch');
  return res.json();
});

// RTK Query - server state
const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['Users', 'Posts'],
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: () => '/users',
      providesTags: ['Users'],
    }),
    getUser: builder.query({
      query: (id) => '/users/' + id,
      providesTags: (result, error, id) => [{ type: 'Users', id }],
    }),
    createUser: builder.mutation({
      query: (body) => ({ url: '/users', method: 'POST', body }),
      invalidatesTags: ['Users'],
    }),
    updateUser: builder.mutation({
      query: ({ id, ...body }) => ({ url: '/users/' + id, method: 'PUT', body }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Users', id }],
    }),
  }),
});

export const { useGetUsersQuery, useGetUserQuery, useCreateUserMutation } = apiSlice;

// 2. TanStack Query (React Query)
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => fetch('/api/users').then(r => r.json()),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000,    // 10 minutes (was cacheTime)
  });
}

function useUser(id) {
  return useQuery({
    queryKey: ['users', id],
    queryFn: () => fetch('/api/users/' + id).then(r => r.json()),
    enabled: !!id,
  });
}

function useCreateUser() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (newUser) => 
      fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newUser),
      }).then(r => r.json()),
    
    // Optimistic update
    onMutate: async (newUser) => {
      await queryClient.cancelQueries({ queryKey: ['users'] });
      const previous = queryClient.getQueryData(['users']);
      queryClient.setQueryData(['users'], (old) => [
        ...old,
        { ...newUser, id: Date.now() },
      ]);
      return { previous };
    },
    onError: (err, newUser, context) => {
      queryClient.setQueryData(['users'], context.previous);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

// 3. Component using TanStack Query
function UsersList() {
  const { data: users, isLoading, error } = useUsers();
  const createUser = useCreateUser();

  if (isLoading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return (
    <div>
      <button
        onClick={() => createUser.mutate({ name: 'New User' })}
        disabled={createUser.isPending}
      >
        Add User
      </button>
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}`,
				},
				{
					Title: "Server State and Data Synchronization",
					Content: `Server state management handles data fetching, caching, synchronization, and optimistic updates across client and server.

**TanStack Query Advanced Patterns:**
` + "```" + `javascript
// Infinite queries (pagination)
function useInfiniteUsers() {
  return useInfiniteQuery({
    queryKey: ['users', 'infinite'],
    queryFn: ({ pageParam = 0 }) =>
      fetch(` + "`" + `/api/users?offset=${pageParam}&limit=20` + "`" + `).then(r => r.json()),
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage.length < 20) return undefined;
      return allPages.length * 20;
    },
    initialPageParam: 0,
  });
}

function InfiniteUserList() {
  const {
    data, fetchNextPage, hasNextPage,
    isFetchingNextPage, isLoading,
  } = useInfiniteUsers();

  // Intersection observer for infinite scroll
  const loadMoreRef = useRef();
  const entry = useIntersectionObserver(loadMoreRef, {});
  
  useEffect(() => {
    if (entry?.isIntersecting && hasNextPage) {
      fetchNextPage();
    }
  }, [entry?.isIntersecting, hasNextPage, fetchNextPage]);

  return (
    <div>
      {data?.pages.map((page, i) => (
        <div key={i}>
          {page.map(user => <UserCard key={user.id} user={user} />)}
        </div>
      ))}
      <div ref={loadMoreRef}>
        {isFetchingNextPage ? <Spinner /> : null}
      </div>
    </div>
  );
}

// Dependent queries
function useUserPosts(userId) {
  return useQuery({
    queryKey: ['users', userId, 'posts'],
    queryFn: () => fetch(` + "`" + `/api/users/${userId}/posts` + "`" + `).then(r => r.json()),
    enabled: !!userId, // Only fetch when userId is available
  });
}

// Parallel queries
function Dashboard() {
  const usersQuery = useQuery({ queryKey: ['users'], queryFn: fetchUsers });
  const postsQuery = useQuery({ queryKey: ['posts'], queryFn: fetchPosts });
  const statsQuery = useQuery({ queryKey: ['stats'], queryFn: fetchStats });

  if ([usersQuery, postsQuery, statsQuery].some(q => q.isLoading)) {
    return <Spinner />;
  }

  return (
    <div>
      <UserCount count={usersQuery.data?.length} />
      <PostCount count={postsQuery.data?.length} />
      <StatsOverview stats={statsQuery.data} />
    </div>
  );
}

// Prefetching
function UserList() {
  const queryClient = useQueryClient();

  const prefetchUser = (userId) => {
    queryClient.prefetchQuery({
      queryKey: ['users', userId],
      queryFn: () => fetchUser(userId),
      staleTime: 5 * 60 * 1000,
    });
  };

  return (
    <ul>
      {users.map(user => (
        <li
          key={user.id}
          onMouseEnter={() => prefetchUser(user.id)}
        >
          <Link to={` + "`" + `/users/${user.id}` + "`" + `}>{user.name}</Link>
        </li>
      ))}
    </ul>
  );
}
` + "```" + `

**SWR and Data Patterns:**
` + "```" + `javascript
// SWR (stale-while-revalidate)
import useSWR from 'swr';

const fetcher = (url) => fetch(url).then(r => r.json());

function Profile() {
  const { data, error, isLoading, mutate } = useSWR('/api/user', fetcher, {
    revalidateOnFocus: true,
    revalidateOnReconnect: true,
    refreshInterval: 30000, // Poll every 30s
    dedupingInterval: 5000,
  });

  const updateProfile = async (updates) => {
    // Optimistic update
    mutate(
      fetch('/api/user', {
        method: 'PATCH',
        body: JSON.stringify(updates),
      }).then(r => r.json()),
      {
        optimisticData: { ...data, ...updates },
        rollbackOnError: true,
        revalidate: true,
      }
    );
  };

  if (isLoading) return <Spinner />;
  if (error) return <Error />;
  return <ProfileView data={data} onUpdate={updateProfile} />;
}

// Real-time data synchronization patterns
// 1. WebSocket + Query invalidation
function useRealtimeSync() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const ws = new WebSocket('wss://api.example.com/ws');
    
    ws.onmessage = (event) => {
      const { type, entity, id } = JSON.parse(event.data);
      
      switch (type) {
        case 'CREATED':
        case 'UPDATED':
          queryClient.invalidateQueries({ queryKey: [entity] });
          queryClient.invalidateQueries({ queryKey: [entity, id] });
          break;
        case 'DELETED':
          queryClient.invalidateQueries({ queryKey: [entity] });
          queryClient.removeQueries({ queryKey: [entity, id] });
          break;
      }
    };

    return () => ws.close();
  }, [queryClient]);
}

// 2. Server-Sent Events
function useSSE(url) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const source = new EventSource(url);
    
    source.addEventListener('update', (e) => {
      const data = JSON.parse(e.data);
      queryClient.setQueryData(
        ['items', data.id],
        (old) => ({ ...old, ...data })
      );
    });

    source.addEventListener('invalidate', (e) => {
      const { queryKey } = JSON.parse(e.data);
      queryClient.invalidateQueries({ queryKey });
    });

    return () => source.close();
  }, [url, queryClient]);
}

// 3. Polling with backoff
function usePolling(queryKey, queryFn, { interval = 5000 } = {}) {
  const [failCount, setFailCount] = useState(0);
  
  return useQuery({
    queryKey,
    queryFn: async () => {
      try {
        const data = await queryFn();
        setFailCount(0);
        return data;
      } catch (e) {
        setFailCount(c => c + 1);
        throw e;
      }
    },
    refetchInterval: Math.min(interval * Math.pow(2, failCount), 60000),
  });
}
` + "```" + ``,
					CodeExamples: `// Data synchronization patterns

// 1. Offline-first data manager
class OfflineDataManager {
  constructor(queryClient) {
    this.queryClient = queryClient;
    this.pendingMutations = [];
    this.loadPending();
    
    window.addEventListener('online', () => this.syncPending());
  }

  loadPending() {
    try {
      const stored = localStorage.getItem('pendingMutations');
      this.pendingMutations = stored ? JSON.parse(stored) : [];
    } catch {
      this.pendingMutations = [];
    }
  }

  savePending() {
    localStorage.setItem(
      'pendingMutations',
      JSON.stringify(this.pendingMutations)
    );
  }

  async mutate(mutation) {
    const id = Date.now().toString(36);
    const entry = { id, ...mutation, timestamp: Date.now() };

    // Apply optimistic update
    this.queryClient.setQueryData(
      mutation.queryKey,
      mutation.optimisticUpdate
    );

    if (navigator.onLine) {
      try {
        const result = await mutation.mutationFn();
        this.queryClient.invalidateQueries({ queryKey: mutation.queryKey });
        return result;
      } catch (error) {
        // Queue for retry
        this.pendingMutations.push(entry);
        this.savePending();
        throw error;
      }
    } else {
      this.pendingMutations.push(entry);
      this.savePending();
    }
  }

  async syncPending() {
    const pending = [...this.pendingMutations];
    this.pendingMutations = [];
    this.savePending();

    for (const mutation of pending) {
      try {
        await mutation.mutationFn();
      } catch {
        this.pendingMutations.push(mutation);
      }
    }
    
    this.savePending();
    this.queryClient.invalidateQueries();
  }
}

// 2. Cursor-based pagination hook
function useCursorPagination(queryKey, fetchFn) {
  const [cursor, setCursor] = useState(null);
  const [allItems, setAllItems] = useState([]);

  const query = useQuery({
    queryKey: [...queryKey, cursor],
    queryFn: () => fetchFn(cursor),
    keepPreviousData: true,
  });

  useEffect(() => {
    if (query.data?.items) {
      if (cursor === null) {
        setAllItems(query.data.items);
      } else {
        setAllItems(prev => [...prev, ...query.data.items]);
      }
    }
  }, [query.data, cursor]);

  return {
    items: allItems,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    hasMore: query.data?.nextCursor != null,
    loadMore: () => setCursor(query.data?.nextCursor),
  };
}

// 3. Debounced search with caching
function useSearchQuery(initialQuery = '') {
  const [query, setQuery] = useState(initialQuery);
  const debouncedQuery = useDebounce(query, 300);

  const searchQuery = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () =>
      fetch(` + "`" + `/api/search?q=${encodeURIComponent(debouncedQuery)}` + "`" + `)
        .then(r => r.json()),
    enabled: debouncedQuery.length >= 2,
    staleTime: 2 * 60 * 1000,
    placeholderData: (previousData) => previousData,
  });

  return {
    query,
    setQuery,
    results: searchQuery.data ?? [],
    isSearching: searchQuery.isFetching,
  };
}`,
				},
			},
		},
		{
			ID:          1555,
			Title:       "Web Performance Optimization",
			Description: "Optimize web application performance with code splitting, lazy loading, rendering strategies, Core Web Vitals, and resource optimization.",
			Order:       55,
			Lessons: []problems.Lesson{
				{
					Title: "Core Web Vitals and Rendering Performance",
					Content: `Web performance directly impacts user experience, conversion rates, and SEO rankings.

**Core Web Vitals:**
` + "```" + `
Metrics:
  LCP (Largest Contentful Paint):
    Goal: < 2.5 seconds
    Measures: Loading performance
    What: Time until largest visible content renders
    Optimize:
      Preload critical resources
      Optimize images (WebP, AVIF, srcset)
      Remove render-blocking resources
      Use CDN for assets
      Server-side rendering for above-the-fold
  
  INP (Interaction to Next Paint):
    Goal: < 200ms
    Measures: Responsiveness (replaced FID)
    What: Latency of all user interactions
    Optimize:
      Break long tasks (> 50ms)
      Use requestIdleCallback for non-urgent work
      Web Workers for heavy computation
      Minimize main thread blocking
      Debounce/throttle event handlers
  
  CLS (Cumulative Layout Shift):
    Goal: < 0.1
    Measures: Visual stability
    What: Unexpected layout shifts during page load
    Optimize:
      Set explicit dimensions on images/videos
      Reserve space for dynamic content
      Avoid inserting content above existing content
      Use CSS contain for isolated components
      Preload fonts with font-display: swap

Measurement tools:
  Lighthouse: Chrome DevTools > Lighthouse
  PageSpeed Insights: web.dev/measure
  Chrome UX Report: Real user metrics
  Web Vitals library:
    import { onLCP, onINP, onCLS } from 'web-vitals';
    
    onLCP(console.log);
    onINP(console.log);
    onCLS(console.log);
    
    // Report to analytics
    function sendToAnalytics(metric) {
      const body = JSON.stringify({
        name: metric.name,
        value: metric.value,
        id: metric.id,
        navigationType: metric.navigationType,
      });
      
      if (navigator.sendBeacon) {
        navigator.sendBeacon('/analytics', body);
      } else {
        fetch('/analytics', { body, method: 'POST', keepalive: true });
      }
    }
    
    onLCP(sendToAnalytics);
    onINP(sendToAnalytics);
    onCLS(sendToAnalytics);
` + "```" + `

**Rendering Optimization:**
` + "```" + `javascript
// React rendering optimization

// 1. React.memo - prevent unnecessary re-renders
const ExpensiveList = React.memo(function ExpensiveList({ items, onSelect }) {
  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={() => onSelect(item.id)}>
          {item.name}
        </li>
      ))}
    </ul>
  );
}, (prevProps, nextProps) => {
  // Custom comparison
  return prevProps.items === nextProps.items;
});

// 2. useMemo and useCallback
function Dashboard({ userId }) {
  const [filter, setFilter] = useState('');
  
  // Memoize expensive computation
  const filteredData = useMemo(() => {
    return expensiveFilter(data, filter);
  }, [data, filter]);
  
  // Stable callback reference
  const handleSelect = useCallback((id) => {
    setSelectedId(id);
  }, []);
  
  return <ExpensiveList items={filteredData} onSelect={handleSelect} />;
}

// 3. Virtualization for long lists
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualList({ items }) {
  const parentRef = useRef(null);
  
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,
  });

  return (
    <div ref={parentRef} style={{ height: '500px', overflow: 'auto' }}>
      <div style={{ height: virtualizer.getTotalSize() + 'px', position: 'relative' }}>
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              transform: 'translateY(' + virtualItem.start + 'px)',
              height: virtualItem.size + 'px',
            }}
          >
            {items[virtualItem.index].name}
          </div>
        ))}
      </div>
    </div>
  );
}

// 4. Concurrent features (React 18+)
import { useTransition, useDeferredValue, startTransition } from 'react';

function SearchResults() {
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;
  
  return (
    <div>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      <div style={{ opacity: isStale ? 0.5 : 1 }}>
        <Results query={deferredQuery} />
      </div>
    </div>
  );
}

function TabSwitcher() {
  const [isPending, startTransition] = useTransition();
  const [tab, setTab] = useState('home');
  
  function selectTab(nextTab) {
    startTransition(() => {
      setTab(nextTab); // Low priority update
    });
  }
  
  return (
    <div>
      <TabBar activeTab={tab} onSelect={selectTab} />
      {isPending && <Spinner />}
      <TabContent tab={tab} />
    </div>
  );
}
` + "```" + ``,
					CodeExamples: `// Performance optimization examples

// 1. Code splitting with React.lazy
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));
const Analytics = lazy(() =>
  import('./pages/Analytics').then(module => ({
    default: module.AnalyticsPage,
  }))
);

function App() {
  return (
    <Router>
      <Suspense fallback={<PageSkeleton />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

// 2. Image optimization component
function OptimizedImage({ src, alt, width, height, priority = false }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const imgRef = useRef();
  const entry = useIntersectionObserver(imgRef, { rootMargin: '200px' });
  const shouldLoad = priority || entry?.isIntersecting;

  // Generate responsive srcset
  const widths = [320, 640, 960, 1280, 1920];
  const srcSet = widths
    .map(w => src.replace(/\.(jpg|png)/, '') + '-' + w + 'w.webp ' + w + 'w')
    .join(', ');

  return (
    <div
      ref={imgRef}
      style={{ aspectRatio: width + '/' + height, background: '#f0f0f0' }}
    >
      {shouldLoad && (
        <picture>
          <source srcSet={srcSet} type="image/webp" />
          <img
            src={src}
            alt={alt}
            width={width}
            height={height}
            loading={priority ? 'eager' : 'lazy'}
            decoding="async"
            onLoad={() => setIsLoaded(true)}
            style={{
              opacity: isLoaded ? 1 : 0,
              transition: 'opacity 0.3s',
            }}
          />
        </picture>
      )}
    </div>
  );
}

// 3. Web Worker for heavy computation
// worker.js
self.onmessage = function(e) {
  const { type, data } = e.data;
  
  switch (type) {
    case 'SORT_LARGE_DATASET':
      const sorted = data.sort((a, b) => a.value - b.value);
      self.postMessage({ type: 'SORTED', data: sorted });
      break;
    case 'FILTER_DATASET':
      const filtered = data.items.filter(item =>
        item.name.toLowerCase().includes(data.query.toLowerCase())
      );
      self.postMessage({ type: 'FILTERED', data: filtered });
      break;
  }
};

// useWorker hook
function useWorker(workerPath) {
  const workerRef = useRef(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    workerRef.current = new Worker(workerPath);
    workerRef.current.onmessage = (e) => {
      setResult(e.data);
      setIsProcessing(false);
    };
    return () => workerRef.current?.terminate();
  }, [workerPath]);

  const postMessage = useCallback((message) => {
    setIsProcessing(true);
    workerRef.current?.postMessage(message);
  }, []);

  return { result, isProcessing, postMessage };
}`,
				},
			},
		},
	})
}
