package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1566,
			Title:       "Internationalization and Localization",
			Description: "Implement multi-language support with i18n libraries, RTL layouts, date/number formatting, pluralization, and dynamic language switching.",
			Order:       66,
			Lessons: []problems.Lesson{
				{
					Title: "i18n Architecture and Implementation",
					Content: `Internationalization (i18n) prepares applications to support multiple languages and cultural conventions.

**React i18n with react-intl:**
` + "```" + `javascript
// i18n/messages/en.json
{
  "app.greeting": "Hello, {name}!",
  "app.welcome": "Welcome to our platform",
  "nav.home": "Home",
  "nav.about": "About",
  "nav.settings": "Settings",
  "items.count": "{count, plural, =0 {No items} one {# item} other {# items}}",
  "cart.total": "Total: {total, number, ::currency/USD}",
  "date.posted": "Posted on {date, date, long}",
  "time.relative": "{value, relative, short}",
  "form.email": "Email address",
  "form.password": "Password",
  "form.submit": "Sign In",
  "form.error.required": "{field} is required",
  "form.error.email": "Please enter a valid email",
  "error.notFound": "Page not found",
  "error.serverError": "Something went wrong. Please try again later.",
  "auth.signOut": "Sign Out",
  "auth.signIn": "Sign In",
  "settings.language": "Language",
  "settings.theme": "Theme",
  "settings.theme.light": "Light",
  "settings.theme.dark": "Dark",
  "settings.theme.system": "System"
}

// i18n/messages/ja.json (Japanese)
{
  "app.greeting": "こんにちは、{name}さん！",
  "app.welcome": "プラットフォームへようこそ",
  "nav.home": "ホーム",
  "nav.about": "概要",
  "nav.settings": "設定",
  "items.count": "{count, plural, other {#個のアイテム}}",
  "cart.total": "合計：{total, number, ::currency/JPY}",
  "date.posted": "{date, date, long}に投稿",
  "form.email": "メールアドレス",
  "form.password": "パスワード",
  "form.submit": "ログイン"
}

// i18n/messages/ar.json (Arabic - RTL)
{
  "app.greeting": "مرحبًا، {name}!",
  "app.welcome": "مرحبًا بك في منصتنا",
  "nav.home": "الرئيسية",
  "nav.about": "حول",
  "nav.settings": "الإعدادات"
}

// i18n setup
import { IntlProvider, FormattedMessage, useIntl } from 'react-intl';

// Lazy load messages
async function loadMessages(locale) {
  switch (locale) {
    case 'ja': return (await import('./messages/ja.json')).default;
    case 'ar': return (await import('./messages/ar.json')).default;
    case 'de': return (await import('./messages/de.json')).default;
    case 'es': return (await import('./messages/es.json')).default;
    default: return (await import('./messages/en.json')).default;
  }
}

function I18nProvider({ children }) {
  const [locale, setLocale] = useState(
    navigator.language.split('-')[0] || 'en'
  );
  const [messages, setMessages] = useState(null);

  useEffect(() => {
    loadMessages(locale).then(setMessages);
    document.documentElement.lang = locale;
    document.documentElement.dir = ['ar', 'he', 'fa'].includes(locale) 
      ? 'rtl' : 'ltr';
  }, [locale]);

  if (!messages) return <LoadingSpinner />;

  return (
    <IntlProvider
      locale={locale}
      messages={messages}
      defaultLocale="en"
      onError={(err) => {
        if (err.code === 'MISSING_TRANSLATION') {
          console.warn('Missing translation:', err.message);
          return;
        }
        throw err;
      }}
    >
      <LocaleContext.Provider value={{ locale, setLocale }}>
        {children}
      </LocaleContext.Provider>
    </IntlProvider>
  );
}

// Using translation messages
function Greeting({ user }) {
  return (
    <h1>
      <FormattedMessage
        id="app.greeting"
        values={{ name: user.name }}
        defaultMessage="Hello, {name}!"
      />
    </h1>
  );
}

// Imperative API with hook
function SearchResults({ count }) {
  const intl = useIntl();

  return (
    <div>
      <p>{intl.formatMessage({ id: 'items.count' }, { count })}</p>
      <p>
        {intl.formatNumber(1234567.89, {
          style: 'currency',
          currency: 'USD',
        })}
      </p>
      <p>
        {intl.formatDate(new Date(), {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
        })}
      </p>
      <p>
        {intl.formatRelativeTime(-3, 'day', { style: 'long' })}
      </p>
    </div>
  );
}
` + "```" + `

**next-intl for Next.js:**
` + "```" + `javascript
// i18n.ts (next-intl v3+)
import { getRequestConfig } from 'next-intl/server';

export default getRequestConfig(async ({ locale }) => ({
  messages: (await import('./messages/' + locale + '.json')).default,
}));

// middleware.ts
import createMiddleware from 'next-intl/middleware';

export default createMiddleware({
  locales: ['en', 'de', 'ja', 'ar'],
  defaultLocale: 'en',
  localePrefix: 'as-needed',
});

export const config = {
  matcher: ['/', '/(de|ja|ar)/:path*'],
};

// app/[locale]/layout.tsx
import { NextIntlClientProvider, useMessages } from 'next-intl';

export default function LocaleLayout({ children, params: { locale } }) {
  const messages = useMessages();
  return (
    <html lang={locale} dir={locale === 'ar' ? 'rtl' : 'ltr'}>
      <body>
        <NextIntlClientProvider locale={locale} messages={messages}>
          {children}
        </NextIntlClientProvider>
      </body>
    </html>
  );
}

// Server component
import { useTranslations } from 'next-intl';

export default function HomePage() {
  const t = useTranslations('app');
  return <h1>{t('welcome')}</h1>;
}
` + "```" + ``,
					CodeExamples: `// Advanced i18n patterns

// 1. RTL-aware styling
// Use logical properties instead of physical
// margin-left → margin-inline-start
// padding-right → padding-inline-end
// left → inset-inline-start

const styles = {
  container: {
    paddingInlineStart: '1rem',
    paddingInlineEnd: '1rem',
    marginBlockStart: '2rem',
    borderInlineStart: '3px solid blue',
    textAlign: 'start', // Instead of 'left'
  },
  
  // Float replacement
  floatedImage: {
    float: 'inline-start', // Instead of 'left'
  },
  
  // Flexbox is naturally RTL-aware
  // row direction automatically reverses in RTL
  nav: {
    display: 'flex',
    flexDirection: 'row',
    gap: '1rem',
  },
};

// CSS logical properties reference:
// margin-left → margin-inline-start
// margin-right → margin-inline-end
// margin-top → margin-block-start
// margin-bottom → margin-block-end
// padding-left → padding-inline-start
// width → inline-size
// height → block-size
// border-left → border-inline-start
// left → inset-inline-start
// text-align: left → text-align: start

// 2. Number and date formatting
function FormattedData({ value, type, locale }) {
  const intl = useIntl();
  
  switch (type) {
    case 'currency':
      return intl.formatNumber(value, { 
        style: 'currency', 
        currency: getCurrencyForLocale(locale),
        minimumFractionDigits: 2,
      });
    
    case 'percent':
      return intl.formatNumber(value / 100, { 
        style: 'percent',
        minimumFractionDigits: 1,
      });
    
    case 'compact':
      return intl.formatNumber(value, { 
        notation: 'compact',
        compactDisplay: 'short',
      });
    
    case 'date':
      return intl.formatDate(value, {
        dateStyle: 'medium',
      });
    
    case 'datetime':
      return intl.formatDate(value, {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    
    case 'relative':
      return intl.formatRelativeTime(
        Math.round((value - Date.now()) / 86400000),
        'day'
      );
    
    default:
      return intl.formatNumber(value);
  }
}

// Locale-specific formatting examples:
// Number 1234567.89:
//   en-US: 1,234,567.89
//   de-DE: 1.234.567,89
//   ja-JP: 1,234,567.89
//   ar-SA: ١٬٢٣٤٬٥٦٧٫٨٩

// Date Dec 31, 2024:
//   en-US: December 31, 2024
//   de-DE: 31. Dezember 2024
//   ja-JP: 2024年12月31日
//   ar-SA: ٣١ ديسمبر ٢٠٢٤

// 3. Pluralization rules
// ICU Message Format handles complex plural rules
const messages = {
  // English: one, other
  'en.items': '{count, plural, =0 {No items} one {# item} other {# items}}',
  
  // Russian: one, few, many, other
  'ru.items': '{count, plural, one {# элемент} few {# элемента} many {# элементов} other {# элемента}}',
  
  // Arabic: zero, one, two, few, many, other
  'ar.items': '{count, plural, =0 {لا عناصر} one {عنصر واحد} two {عنصران} few {# عناصر} many {# عنصرًا} other {# عنصر}}',
  
  // Select for gender
  'invite': '{gender, select, male {He invited} female {She invited} other {They invited}} {count, plural, one {# person} other {# people}}.',
};

// 4. Translation extraction automation
// babel-plugin-react-intl extracts messages at build time
// Format: { id, defaultMessage, description }
// Generates JSON for translators

// 5. Language switcher component
function LanguageSwitcher() {
  const { locale, setLocale } = useContext(LocaleContext);
  const intl = useIntl();
  
  const languages = [
    { code: 'en', name: 'English', nativeName: 'English' },
    { code: 'ja', name: 'Japanese', nativeName: '日本語' },
    { code: 'de', name: 'German', nativeName: 'Deutsch' },
    { code: 'ar', name: 'Arabic', nativeName: 'العربية' },
    { code: 'es', name: 'Spanish', nativeName: 'Español' },
  ];

  return (
    <select
      value={locale}
      onChange={(e) => setLocale(e.target.value)}
      aria-label={intl.formatMessage({ id: 'settings.language' })}
    >
      {languages.map((lang) => (
        <option key={lang.code} value={lang.code}>
          {lang.nativeName}
        </option>
      ))}
    </select>
  );
}`,
				},
			},
		},
		{
			ID:          1567,
			Title:       "Progressive Web Apps and Web APIs",
			Description: "Build progressive web applications with service workers, offline support, push notifications, and modern Web APIs for native-like experiences.",
			Order:       67,
			Lessons: []problems.Lesson{
				{
					Title: "Service Workers and Offline Support",
					Content: `Service workers are scripts that run in the background, enabling offline functionality, caching strategies, and push notifications.

**Service Worker Lifecycle:**
` + "```" + `javascript
// service-worker.js

const CACHE_NAME = 'app-cache-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles.css',
  '/app.js',
  '/manifest.json',
  '/offline.html',
];

// Install: Cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting()) // Activate immediately
  );
});

// Activate: Clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      ))
      .then(() => self.clients.claim()) // Take control immediately
  );
});

// Fetch: Intercept network requests
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API requests: Network first, fallback to cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Static assets: Cache first, fallback to network
  if (request.destination === 'image' || 
      request.destination === 'script' ||
      request.destination === 'style') {
    event.respondWith(cacheFirst(request));
    return;
  }

  // HTML pages: Stale while revalidate
  if (request.mode === 'navigate') {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  event.respondWith(cacheFirst(request));
});

// Caching strategies
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503 });
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached || new Response(
      JSON.stringify({ error: 'Offline' }),
      { headers: { 'Content-Type': 'application/json' }, status: 503 }
    );
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  
  const networkPromise = fetch(request).then((response) => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  }).catch(() => cached || caches.match('/offline.html'));
  
  return cached || networkPromise;
}

// Background Sync
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-posts') {
    event.waitUntil(syncPendingPosts());
  }
});

async function syncPendingPosts() {
  const db = await openDB('pending-posts');
  const posts = await db.getAll('posts');
  
  for (const post of posts) {
    try {
      await fetch('/api/posts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(post),
      });
      await db.delete('posts', post.id);
    } catch {
      // Will retry on next sync event
      break;
    }
  }
}

// Push Notifications
self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'Notification', {
      body: data.body,
      icon: '/icon-192.png',
      badge: '/badge-72.png',
      image: data.image,
      tag: data.tag || 'default',
      data: { url: data.url || '/' },
      actions: data.actions || [
        { action: 'open', title: 'Open' },
        { action: 'dismiss', title: 'Dismiss' },
      ],
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'dismiss') return;
  
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((windowClients) => {
      // Focus existing window or open new one
      for (const client of windowClients) {
        if (client.url === event.notification.data.url && 'focus' in client) {
          return client.focus();
        }
      }
      return clients.openWindow(event.notification.data.url);
    })
  );
});
` + "```" + `

**Workbox Integration:**
` + "```" + `javascript
// workbox-config.js
module.exports = {
  globDirectory: 'dist/',
  globPatterns: ['**/*.{html,js,css,png,jpg,svg,woff2}'],
  swDest: 'dist/sw.js',
  runtimeCaching: [
    {
      urlPattern: /^https:\/\/api\.example\.com\/.*$/,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 60 * 24, // 24 hours
        },
        networkTimeoutSeconds: 5,
      },
    },
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'images',
        expiration: {
          maxEntries: 200,
          maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
        },
      },
    },
    {
      urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/,
      handler: 'StaleWhileRevalidate',
      options: {
        cacheName: 'google-fonts',
      },
    },
  ],
};

// Using Workbox in service worker
import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';

// Precache static assets (injected by build)
precacheAndRoute(self.__WB_MANIFEST);

// API routes - Network first
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new NetworkFirst({
    cacheName: 'api-cache',
    networkTimeoutSeconds: 5,
    plugins: [
      new ExpirationPlugin({ maxEntries: 100, maxAgeSeconds: 86400 }),
      new CacheableResponsePlugin({ statuses: [0, 200] }),
    ],
  })
);

// Images - Cache first
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'images',
    plugins: [
      new ExpirationPlugin({ maxEntries: 200, maxAgeSeconds: 2592000 }),
    ],
  })
);
` + "```" + ``,
					CodeExamples: `// Modern Web APIs for PWA features

// 1. Web App Manifest
// manifest.json
{
  "name": "My App",
  "short_name": "App",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#3b82f6",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icon-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "screenshots": [
    { "src": "/screenshot-wide.png", "sizes": "1280x720", "type": "image/png", "form_factor": "wide" },
    { "src": "/screenshot-narrow.png", "sizes": "750x1334", "type": "image/png", "form_factor": "narrow" }
  ],
  "shortcuts": [
    { "name": "New Post", "url": "/new", "icons": [{ "src": "/icons/new.png", "sizes": "96x96" }] }
  ],
  "share_target": {
    "action": "/share",
    "method": "POST",
    "enctype": "multipart/form-data",
    "params": {
      "title": "title",
      "text": "text",
      "url": "url",
      "files": [{ "name": "media", "accept": ["image/*", "video/*"] }]
    }
  }
}

// 2. Service worker registration
async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  
  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });
    
    // Check for updates periodically
    setInterval(() => registration.update(), 60 * 60 * 1000);
    
    // Prompt user when new version is available
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          // New version available
          showUpdatePrompt();
        }
      });
    });
  } catch (err) {
    console.error('SW registration failed:', err);
  }
}

// 3. Push notification subscription
async function subscribeToPush() {
  const registration = await navigator.serviceWorker.ready;
  
  // Check permission
  const permission = await Notification.requestPermission();
  if (permission !== 'granted') return null;
  
  // Subscribe
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY),
  });
  
  // Send subscription to server
  await fetch('/api/push/subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(subscription),
  });
  
  return subscription;
}

function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const rawData = atob(base64);
  return new Uint8Array([...rawData].map(c => c.charCodeAt(0)));
}

// 4. Install prompt handling
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  showInstallButton();
});

async function installApp() {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  console.log('Install outcome:', outcome);
  deferredPrompt = null;
  hideInstallButton();
}

// 5. File System Access API
async function openFile() {
  const [fileHandle] = await window.showOpenFilePicker({
    types: [{ description: 'Text files', accept: { 'text/plain': ['.txt'] } }],
  });
  const file = await fileHandle.getFile();
  return await file.text();
}

async function saveFile(content) {
  const handle = await window.showSaveFilePicker({
    suggestedName: 'document.txt',
    types: [{ description: 'Text files', accept: { 'text/plain': ['.txt'] } }],
  });
  const writable = await handle.createWritable();
  await writable.write(content);
  await writable.close();
}`,
				},
			},
		},
	})
}
