package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1556,
			Title:       "Testing Frontend Applications",
			Description: "Comprehensive frontend testing strategies including unit tests, component tests, integration tests, and end-to-end tests with modern tools.",
			Order:       56,
			Lessons: []problems.Lesson{
				{
					Title: "Component and Integration Testing",
					Content: `Modern frontend testing focuses on testing user behavior rather than implementation details.

**Testing Library Philosophy:**
` + "```" + `javascript
// React Testing Library - test from user perspective
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Queries priority (most to least preferred):
// 1. getByRole - accessible queries (buttons, inputs, headings)
// 2. getByLabelText - form elements
// 3. getByPlaceholderText - when no label
// 4. getByText - non-interactive elements
// 5. getByDisplayValue - filled form elements
// 6. getByAltText - images
// 7. getByTitle - title attribute
// 8. getByTestId - last resort

// Basic component test
test('renders login form', () => {
  render(<LoginForm />);
  
  expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
  expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
});

// User interaction test
test('submits login form', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  
  render(<LoginForm onSubmit={onSubmit} />);
  
  await user.type(screen.getByLabelText(/email/i), 'user@test.com');
  await user.type(screen.getByLabelText(/password/i), 'password123');
  await user.click(screen.getByRole('button', { name: /sign in/i }));
  
  expect(onSubmit).toHaveBeenCalledWith({
    email: 'user@test.com',
    password: 'password123',
  });
});

// Async content test
test('loads and displays users', async () => {
  render(<UserList />);
  
  // Loading state
  expect(screen.getByText(/loading/i)).toBeInTheDocument();
  
  // Wait for data
  const users = await screen.findAllByRole('listitem');
  expect(users).toHaveLength(3);
  expect(screen.getByText('Alice')).toBeInTheDocument();
});

// Testing with providers
function renderWithProviders(ui, { initialState, ...options } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  
  function Wrapper({ children }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <Router>{children}</Router>
        </ThemeProvider>
      </QueryClientProvider>
    );
  }
  
  return render(ui, { wrapper: Wrapper, ...options });
}

// MSW (Mock Service Worker) for API mocking
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

const handlers = [
  http.get('/api/users', () => {
    return HttpResponse.json([
      { id: 1, name: 'Alice' },
      { id: 2, name: 'Bob' },
    ]);
  }),
  
  http.post('/api/users', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json(
      { id: 3, ...body },
      { status: 201 }
    );
  }),
  
  http.get('/api/users/:id', ({ params }) => {
    return HttpResponse.json({ id: params.id, name: 'Alice' });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('handles API error', async () => {
  server.use(
    http.get('/api/users', () => {
      return new HttpResponse(null, { status: 500 });
    })
  );
  
  render(<UserList />);
  
  await waitFor(() => {
    expect(screen.getByText(/error/i)).toBeInTheDocument();
  });
});
` + "```" + ``,
					CodeExamples: `// Frontend testing examples

// 1. Complex component test
import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('TodoApp', () => {
  test('full workflow: add, complete, filter, delete', async () => {
    const user = userEvent.setup();
    render(<TodoApp />);
    
    // Add todos
    const input = screen.getByPlaceholderText(/add a todo/i);
    const addBtn = screen.getByRole('button', { name: /add/i });
    
    await user.type(input, 'Buy groceries');
    await user.click(addBtn);
    
    await user.type(input, 'Read book');
    await user.click(addBtn);
    
    await user.type(input, 'Exercise');
    await user.click(addBtn);
    
    expect(screen.getAllByRole('listitem')).toHaveLength(3);
    
    // Complete a todo
    const groceryItem = screen.getByText('Buy groceries').closest('li');
    const checkbox = within(groceryItem).getByRole('checkbox');
    await user.click(checkbox);
    
    expect(checkbox).toBeChecked();
    
    // Filter completed
    await user.click(screen.getByRole('button', { name: /completed/i }));
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
    expect(screen.getByText('Buy groceries')).toBeInTheDocument();
    
    // Filter active
    await user.click(screen.getByRole('button', { name: /active/i }));
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
    
    // Show all
    await user.click(screen.getByRole('button', { name: /all/i }));
    expect(screen.getAllByRole('listitem')).toHaveLength(3);
    
    // Delete a todo
    const bookItem = screen.getByText('Read book').closest('li');
    const deleteBtn = within(bookItem).getByRole('button', { name: /delete/i });
    await user.click(deleteBtn);
    
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
    expect(screen.queryByText('Read book')).not.toBeInTheDocument();
  });
});

// 2. Custom hook test
import { renderHook, act } from '@testing-library/react';

describe('useCounter', () => {
  test('increments and decrements', () => {
    const { result } = renderHook(() => useCounter(10));
    
    expect(result.current.count).toBe(10);
    
    act(() => result.current.increment());
    expect(result.current.count).toBe(11);
    
    act(() => result.current.decrement());
    expect(result.current.count).toBe(10);
    
    act(() => result.current.reset());
    expect(result.current.count).toBe(10);
  });
});

describe('useDebounce', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });
  
  test('debounces value changes', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'initial', delay: 300 } }
    );
    
    expect(result.current).toBe('initial');
    
    rerender({ value: 'updated', delay: 300 });
    expect(result.current).toBe('initial'); // Not yet
    
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toBe('updated'); // Now updated
  });
});

// 3. Accessibility test
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);

test('form has no accessibility violations', async () => {
  const { container } = render(<RegistrationForm />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});

// 4. Snapshot test (use sparingly)
test('renders correctly', () => {
  const { asFragment } = render(
    <Badge variant="success">Active</Badge>
  );
  expect(asFragment()).toMatchSnapshot();
});`,
				},
				{
					Title: "End-to-End Testing with Playwright",
					Content: `E2E tests verify complete user flows across the full application stack.

**Playwright:**
` + "```" + `javascript
// playwright.config.js
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [['html'], ['junit', { outputFile: 'results.xml' }]],
  
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile', use: { ...devices['iPhone 14'] } },
  ],
  
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});

// Basic test
import { test, expect } from '@playwright/test';

test('homepage has title', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/My App/);
  await expect(page.getByRole('heading', { name: /welcome/i })).toBeVisible();
});

// Authentication flow
test('user can sign in', async ({ page }) => {
  await page.goto('/login');
  
  await page.getByLabel('Email').fill('user@example.com');
  await page.getByLabel('Password').fill('password123');
  await page.getByRole('button', { name: 'Sign In' }).click();
  
  // Should redirect to dashboard
  await expect(page).toHaveURL('/dashboard');
  await expect(page.getByText('Welcome, User')).toBeVisible();
});

// Page Object Model
class LoginPage {
  constructor(page) {
    this.page = page;
    this.emailInput = page.getByLabel('Email');
    this.passwordInput = page.getByLabel('Password');
    this.submitButton = page.getByRole('button', { name: 'Sign In' });
    this.errorMessage = page.getByRole('alert');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email, password) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}

class DashboardPage {
  constructor(page) {
    this.page = page;
    this.welcomeMessage = page.getByTestId('welcome-message');
    this.userMenu = page.getByRole('button', { name: /user menu/i });
    this.logoutButton = page.getByRole('menuitem', { name: /logout/i });
  }

  async logout() {
    await this.userMenu.click();
    await this.logoutButton.click();
  }
}

test('login and logout flow', async ({ page }) => {
  const loginPage = new LoginPage(page);
  const dashboard = new DashboardPage(page);
  
  await loginPage.goto();
  await loginPage.login('user@test.com', 'password');
  
  await expect(dashboard.welcomeMessage).toBeVisible();
  
  await dashboard.logout();
  await expect(page).toHaveURL('/login');
});

// API mocking in Playwright
test('shows error on API failure', async ({ page }) => {
  await page.route('/api/users', (route) => {
    route.fulfill({
      status: 500,
      body: JSON.stringify({ error: 'Internal Server Error' }),
    });
  });
  
  await page.goto('/users');
  await expect(page.getByText(/something went wrong/i)).toBeVisible();
});

// Visual regression testing
test('dashboard layout matches', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveScreenshot('dashboard.png', {
    maxDiffPixels: 100,
  });
});

// Network interception
test('tracks analytics events', async ({ page }) => {
  const analyticsRequests = [];
  
  page.on('request', (request) => {
    if (request.url().includes('/analytics')) {
      analyticsRequests.push(request.postDataJSON());
    }
  });
  
  await page.goto('/');
  await page.getByRole('button', { name: 'Get Started' }).click();
  
  expect(analyticsRequests).toContainEqual(
    expect.objectContaining({ event: 'cta_click' })
  );
});
` + "```" + ``,
					CodeExamples: `// E2E testing patterns

// 1. Authentication fixtures
import { test as base, expect } from '@playwright/test';

// Extend base test with authenticated page
const test = base.extend({
  authenticatedPage: async ({ page }, use) => {
    // Login once
    await page.goto('/login');
    await page.getByLabel('Email').fill('test@example.com');
    await page.getByLabel('Password').fill('password');
    await page.getByRole('button', { name: 'Sign In' }).click();
    await page.waitForURL('/dashboard');
    
    // Save auth state
    await page.context().storageState({ path: '.auth/user.json' });
    
    await use(page);
  },
});

// Reuse auth state
const authenticatedTest = base.extend({
  storageState: '.auth/user.json',
});

authenticatedTest('can access profile', async ({ page }) => {
  await page.goto('/profile');
  await expect(page.getByRole('heading', { name: 'Profile' })).toBeVisible();
});

// 2. Data-driven tests
const testCases = [
  { input: '', error: 'Email is required' },
  { input: 'invalid', error: 'Invalid email format' },
  { input: 'a@b', error: 'Invalid email format' },
  { input: 'valid@example.com', error: null },
];

for (const { input, error } of testCases) {
  test('email validation: ' + (input || 'empty'), async ({ page }) => {
    await page.goto('/register');
    
    if (input) {
      await page.getByLabel('Email').fill(input);
    }
    await page.getByLabel('Email').blur();
    
    if (error) {
      await expect(page.getByText(error)).toBeVisible();
    } else {
      await expect(page.getByRole('alert')).not.toBeVisible();
    }
  });
}

// 3. Full user journey
test.describe('E-commerce checkout', () => {
  test('complete purchase flow', async ({ page }) => {
    // Browse products
    await page.goto('/products');
    await expect(page.getByRole('heading', { name: 'Products' })).toBeVisible();
    
    // Search and filter
    await page.getByPlaceholder('Search...').fill('laptop');
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('product-card')).toHaveCount(5);
    
    // Add to cart
    const firstProduct = page.getByTestId('product-card').first();
    await firstProduct.getByRole('button', { name: 'Add to Cart' }).click();
    await expect(page.getByTestId('cart-count')).toHaveText('1');
    
    // Go to cart
    await page.getByTestId('cart-icon').click();
    await expect(page).toHaveURL('/cart');
    await expect(page.getByTestId('cart-item')).toHaveCount(1);
    
    // Update quantity
    await page.getByRole('spinbutton').fill('2');
    await expect(page.getByTestId('cart-total')).toContainText('$');
    
    // Checkout
    await page.getByRole('button', { name: 'Checkout' }).click();
    await expect(page).toHaveURL('/checkout');
    
    // Fill shipping
    await page.getByLabel('Full Name').fill('John Doe');
    await page.getByLabel('Address').fill('123 Main St');
    await page.getByLabel('City').fill('Springfield');
    await page.getByLabel('Zip').fill('12345');
    
    // Submit order
    await page.getByRole('button', { name: 'Place Order' }).click();
    
    // Confirmation
    await expect(page).toHaveURL(/\/orders\/\w+/);
    await expect(page.getByText('Order Confirmed')).toBeVisible();
  });
});

// 4. Accessibility E2E
test('keyboard navigation works', async ({ page }) => {
  await page.goto('/');
  
  // Tab through navigation
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Home' })).toBeFocused();
  
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'About' })).toBeFocused();
  
  // Enter activates link
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL('/about');
  
  // Escape closes modal
  await page.getByRole('button', { name: 'Open Menu' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
});`,
				},
			},
		},
	})
}
