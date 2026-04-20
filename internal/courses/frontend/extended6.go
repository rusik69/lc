package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1558,
			Title:       "Next.js and Full-Stack React",
			Description: "Build full-stack applications with Next.js App Router, Server Components, Server Actions, and modern rendering strategies.",
			Order:       58,
			Lessons: []problems.Lesson{
				{
					Title: "App Router and Server Components",
					Content: `Next.js App Router introduces React Server Components and a new file-based routing paradigm.

**App Router Fundamentals:**
` + "```" + `javascript
// File-based routing
// app/
// ├── layout.js          → Root layout (wraps all)
// ├── page.js            → / route
// ├── loading.js         → Loading UI
// ├── error.js           → Error boundary
// ├── not-found.js       → 404 page
// ├── about/
// │   └── page.js        → /about
// ├── blog/
// │   ├── page.js        → /blog
// │   ├── [slug]/
// │   │   └── page.js    → /blog/hello-world
// │   └── [...catchAll]/
// │       └── page.js    → /blog/a/b/c
// ├── (marketing)/       → Route group (no URL segment)
// │   ├── layout.js
// │   └── pricing/
// │       └── page.js    → /pricing
// └── @modal/            → Parallel route (slot)
//     └── login/
//         └── page.js

// Root layout (required)
// app/layout.js
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <nav>{'...'}</nav>
        <main>{children}</main>
      </body>
    </html>
  );
}

export const metadata = {
  title: { template: '%s | My App', default: 'My App' },
  description: 'My amazing application',
  openGraph: { images: ['/og-image.png'] },
};

// Server Components (default in App Router)
// These run ONLY on the server
// app/blog/[slug]/page.js
async function BlogPost({ params }) {
  const { slug } = await params;
  
  // Direct database access - no API needed!
  const post = await db.post.findUnique({ where: { slug } });
  
  if (!post) notFound();
  
  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
      <CommentSection postId={post.id} />
    </article>
  );
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const post = await db.post.findUnique({ where: { slug } });
  return { title: post?.title, description: post?.excerpt };
}

export async function generateStaticParams() {
  const posts = await db.post.findMany({ select: { slug: true } });
  return posts.map(post => ({ slug: post.slug }));
}

export default BlogPost;

// Loading UI (Suspense boundary)
// app/blog/[slug]/loading.js
export default function Loading() {
  return (
    <div className="animate-pulse">
      <div className="h-8 bg-gray-200 rounded w-3/4 mb-4" />
      <div className="h-4 bg-gray-200 rounded w-full mb-2" />
      <div className="h-4 bg-gray-200 rounded w-full mb-2" />
      <div className="h-4 bg-gray-200 rounded w-2/3" />
    </div>
  );
}

// Error boundary
// app/blog/[slug]/error.js
'use client'; // Error components must be client

export default function Error({ error, reset }) {
  return (
    <div role="alert">
      <h2>Something went wrong</h2>
      <p>{error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  );
}

// Client Components
'use client';
import { useState } from 'react';

function LikeButton({ postId, initialLikes }) {
  const [likes, setLikes] = useState(initialLikes);
  const [isLiked, setIsLiked] = useState(false);

  async function handleLike() {
    setIsLiked(!isLiked);
    setLikes(prev => isLiked ? prev - 1 : prev + 1);
    
    await fetch(` + "`" + `/api/posts/${postId}/like` + "`" + `, {
      method: isLiked ? 'DELETE' : 'POST',
    });
  }

  return (
    <button onClick={handleLike}>
      {isLiked ? '❤️' : '🤍'} {likes}
    </button>
  );
}
` + "```" + ``,
					CodeExamples: `// Next.js patterns

// 1. Server Actions
// app/actions.js
'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';

export async function createPost(formData) {
  const title = formData.get('title');
  const content = formData.get('content');
  
  // Validate
  if (!title || title.length < 3) {
    return { error: 'Title must be at least 3 characters' };
  }
  
  // Create in database
  const post = await db.post.create({
    data: { title, content, authorId: await getCurrentUserId() },
  });
  
  revalidatePath('/blog');
  redirect('/blog/' + post.slug);
}

export async function deletePost(postId) {
  await db.post.delete({ where: { id: postId } });
  revalidatePath('/blog');
}

// Using Server Actions in forms
// app/blog/new/page.js
import { createPost } from '../actions';

export default function NewPostPage() {
  return (
    <form action={createPost}>
      <label htmlFor="title">Title</label>
      <input id="title" name="title" required />
      
      <label htmlFor="content">Content</label>
      <textarea id="content" name="content" rows={10} />
      
      <SubmitButton />
    </form>
  );
}

// Client component for form state
'use client';
import { useFormStatus } from 'react-dom';

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button type="submit" disabled={pending}>
      {pending ? 'Creating...' : 'Create Post'}
    </button>
  );
}

// 2. Parallel routes and intercepting routes
// app/layout.js - parallel route slots
export default function Layout({ children, modal }) {
  return (
    <>
      {children}
      {modal}
    </>
  );
}

// app/@modal/login/page.js
export default function LoginModal() {
  return (
    <dialog open>
      <h2>Sign In</h2>
      <LoginForm />
    </dialog>
  );
}

// app/@modal/default.js
export default function Default() {
  return null; // No modal by default
}

// 3. Data fetching patterns
// Streaming with Suspense
import { Suspense } from 'react';

async function UserPosts({ userId }) {
  const posts = await db.post.findMany({ where: { authorId: userId } });
  return (
    <ul>
      {posts.map(post => <li key={post.id}>{post.title}</li>)}
    </ul>
  );
}

async function UserStats({ userId }) {
  const stats = await getStats(userId); // Slow query
  return <StatsPanel stats={stats} />;
}

export default async function ProfilePage({ params }) {
  const { id } = await params;
  const user = await db.user.findUnique({ where: { id } });
  
  return (
    <div>
      <h1>{user.name}</h1>
      
      {/* These stream independently */}
      <Suspense fallback={<PostsSkeleton />}>
        <UserPosts userId={id} />
      </Suspense>
      
      <Suspense fallback={<StatsSkeleton />}>
        <UserStats userId={id} />
      </Suspense>
    </div>
  );
}

// 4. Middleware
// middleware.js (at project root)
import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;
  
  // Authentication check
  const token = request.cookies.get('session');
  if (pathname.startsWith('/dashboard') && !token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  // Add headers
  const response = NextResponse.next();
  response.headers.set('x-pathname', pathname);
  
  return response;
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
};`,
				},
				{
					Title: "Rendering Strategies and Caching",
					Content: `Next.js provides multiple rendering strategies optimized for different use cases.

**Rendering Modes:**
` + "```" + `javascript
// 1. Static Rendering (default for Server Components)
// Pages are built at build time and cached.
// Best for: Marketing pages, blog posts, documentation

export default async function AboutPage() {
  // This runs at build time
  const content = await getAboutContent();
  return <div>{content}</div>;
}

// 2. Dynamic Rendering
// Pages are rendered per-request.
// Triggered by: cookies(), headers(), searchParams, uncached fetch

export default async function DashboardPage() {
  const session = await cookies(); // Makes it dynamic
  const user = await getUser(session.get('token'));
  return <Dashboard user={user} />;
}

// 3. Streaming
// Progressive rendering with Suspense boundaries.
export default async function Page() {
  return (
    <div>
      <StaticHeader /> {/* Renders immediately */}
      <Suspense fallback={<Skeleton />}>
        <SlowComponent /> {/* Streams when ready */}
      </Suspense>
    </div>
  );
}

// 4. ISR (Incremental Static Regeneration)
// Static pages that revalidate after a time period.
export const revalidate = 3600; // Revalidate every hour

export default async function ProductsPage() {
  const products = await getProducts();
  return <ProductList products={products} />;
}

// Or per-fetch revalidation
async function getProducts() {
  const res = await fetch('https://api.example.com/products', {
    next: { revalidate: 3600 },
  });
  return res.json();
}

// On-demand revalidation
// app/api/revalidate/route.js
import { revalidatePath, revalidateTag } from 'next/cache';

export async function POST(request) {
  const { path, tag, secret } = await request.json();
  
  if (secret !== process.env.REVALIDATION_SECRET) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  if (tag) revalidateTag(tag);
  if (path) revalidatePath(path);
  
  return Response.json({ revalidated: true });
}

// Fetch with tags
async function getPost(slug) {
  const res = await fetch(` + "`" + `https://api.example.com/posts/${slug}` + "`" + `, {
    next: { tags: ['posts', ` + "`" + `post-${slug}` + "`" + `] },
  });
  return res.json();
}

// Cache function results
import { unstable_cache } from 'next/cache';

const getCachedUser = unstable_cache(
  async (userId) => {
    return db.user.findUnique({ where: { id: userId } });
  },
  ['user-cache'], // cache key prefix
  { revalidate: 3600, tags: ['users'] }
);
` + "```" + `

**Data Patterns:**
` + "```" + `javascript
// Parallel data fetching
export default async function Dashboard() {
  // Fetch in parallel, not waterfall
  const [user, posts, analytics] = await Promise.all([
    getUser(),
    getPosts(),
    getAnalytics(),
  ]);

  return (
    <div>
      <UserProfile user={user} />
      <PostList posts={posts} />
      <AnalyticsChart data={analytics} />
    </div>
  );
}

// Preload pattern
import { preload } from 'react-dom';

async function getUser(id) {
  // Start fetching early
  const res = await fetch(` + "`" + `/api/users/${id}` + "`" + `);
  return res.json();
}

// Preload in layout, use in page
// layout.js
export default function UserLayout({ params }) {
  // Start fetching before rendering children
  preload(getUser, params.id);
  return <div>{children}</div>;
}

// Route handlers (API routes)
// app/api/posts/route.js
import { NextResponse } from 'next/server';

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get('page') || '1');
  const limit = parseInt(searchParams.get('limit') || '10');
  
  const posts = await db.post.findMany({
    skip: (page - 1) * limit,
    take: limit,
    orderBy: { createdAt: 'desc' },
  });

  return NextResponse.json(posts);
}

export async function POST(request) {
  const body = await request.json();
  const post = await db.post.create({ data: body });
  return NextResponse.json(post, { status: 201 });
}

// Dynamic route handler
// app/api/posts/[id]/route.js
export async function GET(request, { params }) {
  const { id } = await params;
  const post = await db.post.findUnique({ where: { id } });
  
  if (!post) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }
  
  return NextResponse.json(post);
}

export async function PUT(request, { params }) {
  const { id } = await params;
  const body = await request.json();
  const post = await db.post.update({ where: { id }, data: body });
  return NextResponse.json(post);
}

export async function DELETE(request, { params }) {
  const { id } = await params;
  await db.post.delete({ where: { id } });
  return new Response(null, { status: 204 });
}
` + "```" + ``,
					CodeExamples: `// Next.js advanced patterns

// 1. Authentication with NextAuth.js
// app/api/auth/[...nextauth]/route.js
import NextAuth from 'next-auth';
import GitHub from 'next-auth/providers/github';
import Google from 'next-auth/providers/google';
import Credentials from 'next-auth/providers/credentials';

const handler = NextAuth({
  providers: [
    GitHub({
      clientId: process.env.GITHUB_ID,
      clientSecret: process.env.GITHUB_SECRET,
    }),
    Google({
      clientId: process.env.GOOGLE_ID,
      clientSecret: process.env.GOOGLE_SECRET,
    }),
    Credentials({
      name: 'Email',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        const user = await verifyCredentials(
          credentials.email,
          credentials.password
        );
        return user || null;
      },
    }),
  ],
  callbacks: {
    async session({ session, token }) {
      session.user.id = token.sub;
      session.user.role = token.role;
      return session;
    },
    async jwt({ token, user }) {
      if (user) { token.role = user.role; }
      return token;
    },
  },
  pages: { signIn: '/login', error: '/auth/error' },
});

export { handler as GET, handler as POST };

// 2. Internationalization
// middleware.js
import { match } from '@formatjs/intl-localematcher';
import Negotiator from 'negotiator';

const locales = ['en', 'es', 'fr', 'de'];
const defaultLocale = 'en';

function getLocale(request) {
  const negotiator = new Negotiator({
    headers: Object.fromEntries(request.headers),
  });
  return match(negotiator.languages(), locales, defaultLocale);
}

export function middleware(request) {
  const { pathname } = request.nextUrl;
  const hasLocale = locales.some(
    locale => pathname.startsWith('/' + locale + '/') || pathname === '/' + locale
  );
  
  if (!hasLocale) {
    const locale = getLocale(request);
    return NextResponse.redirect(
      new URL('/' + locale + pathname, request.url)
    );
  }
}

// app/[lang]/page.js
import { getDictionary } from './dictionaries';

export default async function Home({ params }) {
  const { lang } = await params;
  const dict = await getDictionary(lang);
  
  return (
    <div>
      <h1>{dict.welcome}</h1>
      <p>{dict.description}</p>
    </div>
  );
}

// dictionaries.js
const dictionaries = {
  en: () => import('./dictionaries/en.json').then(m => m.default),
  es: () => import('./dictionaries/es.json').then(m => m.default),
  fr: () => import('./dictionaries/fr.json').then(m => m.default),
};

export function getDictionary(locale) {
  return dictionaries[locale]();
}

// 3. Edge runtime
// app/api/geo/route.js
export const runtime = 'edge';

export async function GET(request) {
  const country = request.headers.get('x-vercel-ip-country') || 'US';
  const city = request.headers.get('x-vercel-ip-city') || 'Unknown';
  
  return Response.json({ country, city });
}`,
				},
			},
		},
	})
}
