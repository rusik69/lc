package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1572,
			Title:       "Server-Side Rendering and Streaming",
			Description: "Implement server-side rendering with React Server Components, streaming SSR, selective hydration, and island architecture patterns.",
			Order:       72,
			Lessons: []problems.Lesson{
				{
					Title: "React Server Components and Streaming SSR",
					Content: `React Server Components (RSC) run exclusively on the server, reducing client bundle size and enabling direct database access.

**Server Components vs Client Components:**
` + "```" + `javascript
// app/page.tsx - Server Component (default)
// Can use async/await, access database, read files
// Cannot use useState, useEffect, event handlers, browser APIs

import { db } from '@/lib/database';

export default async function PostsPage() {
  // Direct database access (no API needed)
  const posts = await db.post.findMany({
    orderBy: { createdAt: 'desc' },
    take: 20,
    include: { author: true },
  });

  return (
    <main>
      <h1>Latest Posts</h1>
      {/* Server component can render client components */}
      <PostFilter /> {/* Client component for interactivity */}
      
      {/* This data never leaves the server as JS */}
      <PostList posts={posts} />
      
      {/* Streaming with Suspense */}
      <Suspense fallback={<CommentsSkeleton />}>
        <Comments />
      </Suspense>
    </main>
  );
}

// Async server component with streaming
async function Comments() {
  // This can be slow - it streams in when ready
  const comments = await fetchComments();
  return (
    <ul>
      {comments.map(c => (
        <li key={c.id}>
          <strong>{c.author.name}</strong>: {c.text}
        </li>
      ))}
    </ul>
  );
}

// components/PostFilter.tsx - Client Component
'use client'; // Opt into client rendering

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';

export default function PostFilter() {
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  function handleSearch(value) {
    setQuery(value);
    startTransition(() => {
      router.push('/posts?search=' + encodeURIComponent(value));
    });
  }

  return (
    <div>
      <input
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search posts..."
        aria-label="Search posts"
      />
      {isPending && <span>Searching...</span>}
    </div>
  );
}

// Server Actions (form mutations)
// app/actions.ts
'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { z } from 'zod';

const CreatePostSchema = z.object({
  title: z.string().min(1).max(200),
  content: z.string().min(1),
});

export async function createPost(formData) {
  const parsed = CreatePostSchema.safeParse({
    title: formData.get('title'),
    content: formData.get('content'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  const session = await getSession();
  if (!session) redirect('/login');

  await db.post.create({
    data: {
      ...parsed.data,
      authorId: session.user.id,
    },
  });

  revalidatePath('/posts');
  redirect('/posts');
}

// Using server actions in client component
'use client';
import { createPost } from '@/app/actions';
import { useFormStatus } from 'react-dom';

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button type="submit" disabled={pending}>
      {pending ? 'Creating...' : 'Create Post'}
    </button>
  );
}

export default function CreatePostForm() {
  return (
    <form action={createPost}>
      <label htmlFor="title">Title</label>
      <input id="title" name="title" required />
      
      <label htmlFor="content">Content</label>
      <textarea id="content" name="content" required rows={10} />
      
      <SubmitButton />
    </form>
  );
}
` + "```" + `

**Streaming SSR with React 18:**
` + "```" + `javascript
// server.js - Node.js streaming SSR
import { renderToPipeableStream } from 'react-dom/server';
import express from 'express';
import App from './App';

const app = express();

app.get('*', (req, res) => {
  let didError = false;

  const { pipe, abort } = renderToPipeableStream(
    <App url={req.url} />,
    {
      bootstrapScripts: ['/client.js'],
      
      onShellReady() {
        // The shell (content outside Suspense) is ready
        res.statusCode = didError ? 500 : 200;
        res.setHeader('Content-Type', 'text/html');
        pipe(res);
      },
      
      onShellError(error) {
        // Shell failed to render - send fallback HTML
        res.statusCode = 500;
        res.send('<!DOCTYPE html><html><body><h1>Error</h1></body></html>');
      },
      
      onAllReady() {
        // All Suspense boundaries resolved
        // Called after streaming is complete
      },
      
      onError(error) {
        didError = true;
        console.error('SSR error:', error);
      },
    }
  );

  // Abort if response takes too long
  setTimeout(abort, 10000);
});

// Client-side selective hydration
// client.js
import { hydrateRoot } from 'react-dom/client';
import App from './App';

hydrateRoot(document.getElementById('root'), <App />);

// Selective hydration with Suspense
// React prioritizes hydrating components the user interacts with
function App() {
  return (
    <html>
      <body>
        <header>
          <Nav /> {/* Hydrates first (part of shell) */}
        </header>
        <main>
          <Hero /> {/* Hydrates as part of shell */}
          
          {/* These stream in and hydrate independently */}
          <Suspense fallback={<ProductsSkeleton />}>
            <ProductGrid /> {/* Hydrates when ready */}
          </Suspense>
          
          <Suspense fallback={<ReviewsSkeleton />}>
            <Reviews /> {/* Hydrates when user scrolls/clicks */}
          </Suspense>
        </main>
      </body>
    </html>
  );
}
` + "```" + `

**Next.js App Router Patterns:**
` + "```" + `javascript
// app/layout.tsx - Root layout
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Header />
        {children}
        <Footer />
      </body>
    </html>
  );
}

// Parallel routes
// app/@modal/(.)photo/[id]/page.tsx - Intercepted route
export default function PhotoModal({ params }) {
  return (
    <Modal>
      <PhotoDetail id={params.id} />
    </Modal>
  );
}

// app/layout.tsx with parallel routes
export default function Layout({ children, modal }) {
  return (
    <>
      {children}
      {modal} {/* Parallel route renders alongside */}
    </>
  );
}

// Loading and error states
// app/posts/loading.tsx
export default function PostsLoading() {
  return <PostsSkeleton />;
}

// app/posts/error.tsx
'use client';
export default function PostsError({ error, reset }) {
  return (
    <div role="alert">
      <h2>Failed to load posts</h2>
      <p>{error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  );
}

// Data caching and revalidation
// Time-based revalidation
export const revalidate = 3600; // Revalidate every hour

// On-demand revalidation
import { revalidateTag } from 'next/cache';

async function getPost(id) {
  const res = await fetch('/api/posts/' + id, {
    next: { tags: ['post-' + id], revalidate: 3600 },
  });
  return res.json();
}

// Webhooks or server actions trigger revalidation
async function handlePostUpdate(postId) {
  revalidateTag('post-' + postId);
}

// Metadata API
export async function generateMetadata({ params }) {
  const post = await getPost(params.id);
  return {
    title: post.title,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      images: [{ url: post.coverImage }],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
    },
  };
}

// Static generation with dynamic params
export async function generateStaticParams() {
  const posts = await db.post.findMany({ select: { slug: true } });
  return posts.map((post) => ({ slug: post.slug }));
}

// Route segment config
export const dynamic = 'force-dynamic'; // or 'auto', 'error', 'force-static'
export const runtime = 'edge'; // or 'nodejs'
export const preferredRegion = 'iad1';
export const maxDuration = 30;
` + "```" + ``,
					CodeExamples: `// Island Architecture and partial hydration

// 1. Astro-style island architecture in React
// Only interactive components are hydrated

// Static shell (no JavaScript)
function ProductPage({ product }) {
  return (
    <html>
      <head><title>{product.name}</title></head>
      <body>
        {/* Static content - zero JS */}
        <Header />
        <ProductImages images={product.images} />
        <ProductDescription text={product.description} />
        
        {/* Interactive islands */}
        <Island id="add-to-cart" component="AddToCart" props={{ productId: product.id }}>
          <AddToCart productId={product.id} />
        </Island>
        
        <Island id="reviews" component="Reviews" props={{ productId: product.id }} client="visible">
          <Reviews productId={product.id} />
        </Island>
        
        {/* Static footer - no JS */}
        <Footer />
      </body>
    </html>
  );
}

// Island wrapper with hydration strategies
function Island({ id, component, props, client = 'load', children }) {
  const scriptContent = JSON.stringify({ id, component, props, client });
  
  return (
    <div id={'island-' + id} data-island={scriptContent}>
      {children} {/* SSR content shown immediately */}
    </div>
  );
}

// Client-side island hydrator
function hydrateIslands() {
  const islands = document.querySelectorAll('[data-island]');
  
  islands.forEach(async (el) => {
    const config = JSON.parse(el.getAttribute('data-island'));
    
    switch (config.client) {
      case 'load':
        await hydrateIsland(el, config);
        break;
      case 'idle':
        requestIdleCallback(() => hydrateIsland(el, config));
        break;
      case 'visible':
        const observer = new IntersectionObserver(async (entries) => {
          if (entries[0].isIntersecting) {
            observer.disconnect();
            await hydrateIsland(el, config);
          }
        });
        observer.observe(el);
        break;
      case 'media':
        const mql = window.matchMedia(config.mediaQuery);
        if (mql.matches) await hydrateIsland(el, config);
        else mql.addEventListener('change', () => hydrateIsland(el, config), { once: true });
        break;
    }
  });
}

async function hydrateIsland(el, config) {
  const module = await import('./components/' + config.component + '.js');
  const Component = module.default;
  hydrateRoot(el, <Component {...config.props} />);
}

// 2. Streaming with progressive enhancement
// Works without JavaScript, enhanced with JavaScript
function SearchPage() {
  return (
    <form action="/search" method="GET">
      <input name="q" type="search" aria-label="Search" />
      <button type="submit">Search</button>
    </form>
  );
}

// Enhanced version with client-side search
'use client';
function EnhancedSearch() {
  const [results, setResults] = useState([]);
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (!debouncedQuery) { setResults([]); return; }
    
    const controller = new AbortController();
    fetch('/api/search?q=' + encodeURIComponent(debouncedQuery), {
      signal: controller.signal,
    })
      .then(r => r.json())
      .then(setResults)
      .catch(() => {});
    
    return () => controller.abort();
  }, [debouncedQuery]);

  return (
    <div>
      <form action="/search" method="GET">
        <input
          name="q"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          type="search"
          role="combobox"
          aria-expanded={results.length > 0}
          aria-autocomplete="list"
          aria-controls="search-results"
          aria-label="Search"
        />
        <button type="submit">Search</button>
      </form>
      {results.length > 0 && (
        <ul id="search-results" role="listbox">
          {results.map(r => (
            <li key={r.id} role="option">
              <a href={r.url}>{r.title}</a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// 3. Edge rendering with streaming
// Cloudflare Workers / Vercel Edge
export default {
  async fetch(request) {
    const url = new URL(request.url);
    
    // Edge-side personalization
    const country = request.headers.get('cf-ipcountry');
    const language = request.headers.get('accept-language')?.split(',')[0] || 'en';
    
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    
    // Start streaming HTML shell immediately
    writer.write(encoder.encode(
      '<!DOCTYPE html><html lang="' + language + '">' +
      '<head><title>My App</title></head>' +
      '<body><div id="root">'
    ));
    
    // Fetch main content (can be slow)
    const contentPromise = fetch('https://api.example.com/page' + url.pathname);
    
    // Stream initial shell while waiting
    writer.write(encoder.encode('<header>Personalized for ' + country + '</header><main>'));
    
    const response = await contentPromise;
    const content = await response.text();
    writer.write(encoder.encode(content));
    
    writer.write(encoder.encode('</main></div></body></html>'));
    writer.close();
    
    return new Response(readable, {
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 's-maxage=60, stale-while-revalidate=600',
      },
    });
  },
};`,
				},
			},
		},
	})
}
