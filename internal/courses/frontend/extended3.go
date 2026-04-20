package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1552,
			Title:       "TypeScript Advanced Types",
			Description: "Master TypeScript advanced type system features including conditional types, mapped types, template literal types, and type-level programming.",
			Order:       52,
			Lessons: []problems.Lesson{
				{
					Title: "Conditional and Mapped Types",
					Content: `TypeScript's type system enables powerful compile-time type manipulation for safer, more expressive APIs.

**Conditional Types:**
` + "```" + `typescript
// Basic conditional type
type IsString<T> = T extends string ? true : false;
type A = IsString<"hello">; // true
type B = IsString<42>;      // false

// Distributive conditional types
type NonNullable<T> = T extends null | undefined ? never : T;
type C = NonNullable<string | null | undefined>; // string

// infer keyword - extract types
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;
type ElementType<T> = T extends (infer E)[] ? E : never;
type PromiseType<T> = T extends Promise<infer R> ? R : T;

type D = ReturnType<() => string>;           // string
type E = ElementType<number[]>;              // number
type F = PromiseType<Promise<{ id: number }>>; // { id: number }

// Nested infer
type Unpacked<T> =
  T extends (infer U)[] ? U :
  T extends Promise<infer U> ? U :
  T extends (...args: any[]) => infer U ? U :
  T;

// Extract/Exclude utility types
type Extract<T, U> = T extends U ? T : never;
type Exclude<T, U> = T extends U ? never : T;

type Events = 'click' | 'scroll' | 'mousemove' | 'keydown' | 'keyup';
type MouseEvents = Extract<Events, 'click' | 'scroll' | 'mousemove'>;
type KeyEvents = Exclude<Events, MouseEvents>;

// Recursive conditional types
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object
    ? T[P] extends Function
      ? T[P]
      : DeepReadonly<T[P]>
    : T[P];
};

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object
    ? DeepPartial<T[P]>
    : T[P];
};

// Flatten nested types
type Flatten<T> = T extends Array<infer Item>
  ? Flatten<Item>
  : T;

type Nested = number[][][];
type Flat = Flatten<Nested>; // number
` + "```" + `

**Mapped Types:**
` + "```" + `typescript
// Basic mapped types
type Readonly<T> = { readonly [P in keyof T]: T[P] };
type Partial<T> = { [P in keyof T]?: T[P] };
type Required<T> = { [P in keyof T]-?: T[P] };
type Mutable<T> = { -readonly [P in keyof T]: T[P] };

// Key remapping with 'as'
type Getters<T> = {
  [K in keyof T as ` + "`" + `get${Capitalize<string & K>}` + "`" + `]: () => T[K];
};

type Setters<T> = {
  [K in keyof T as ` + "`" + `set${Capitalize<string & K>}` + "`" + `]: (value: T[K]) => void;
};

interface User { name: string; age: number; }
type UserGetters = Getters<User>;
// { getName: () => string; getAge: () => number; }

// Filter keys
type FilterByType<T, U> = {
  [K in keyof T as T[K] extends U ? K : never]: T[K];
};

interface Mixed {
  id: number;
  name: string;
  active: boolean;
  email: string;
}
type StringProps = FilterByType<Mixed, string>;
// { name: string; email: string; }

// Create event handler types
type EventHandlers<T> = {
  [K in keyof T as ` + "`" + `on${Capitalize<string & K>}Change` + "`" + `]: (
    newValue: T[K],
    oldValue: T[K]
  ) => void;
};

type UserHandlers = EventHandlers<User>;
// { onNameChange: (n: string, o: string) => void; onAgeChange: ... }

// Pick and Omit
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// Record
type Record<K extends keyof any, T> = { [P in K]: T };
type StatusMap = Record<'active' | 'inactive' | 'pending', User[]>;

// Template literal types
type HTTPMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
type APIRoute = ` + "`" + `/api/${string}` + "`" + `;
type TypedRoute = ` + "`" + `${HTTPMethod} ${APIRoute}` + "`" + `;

type CSSProperty = ` + "`" + `--${string}` + "`" + `;
type EventName = ` + "`" + `on${Capitalize<string>}` + "`" + `;

// Split string type
type Split<S extends string, D extends string> =
  S extends ` + "`" + `${infer T}${D}${infer U}` + "`" + `
    ? [T, ...Split<U, D>]
    : [S];

type Path = Split<'a.b.c', '.'>; // ['a', 'b', 'c']
` + "```" + ``,
					CodeExamples: `// TypeScript advanced type examples

// 1. Type-safe API client
type APIEndpoints = {
  '/users': { GET: User[]; POST: CreateUserDTO };
  '/users/:id': { GET: User; PUT: UpdateUserDTO; DELETE: void };
  '/posts': { GET: Post[]; POST: CreatePostDTO };
};

type ExtractParams<T extends string> =
  T extends ` + "`" + `${infer _}:${infer Param}/${infer Rest}` + "`" + `
    ? { [K in Param | keyof ExtractParams<Rest>]: string }
    : T extends ` + "`" + `${infer _}:${infer Param}` + "`" + `
    ? { [K in Param]: string }
    : {};

async function apiClient<
  Path extends keyof APIEndpoints,
  Method extends keyof APIEndpoints[Path]
>(
  path: Path,
  method: Method,
  params?: ExtractParams<Path & string>,
  body?: APIEndpoints[Path][Method] extends { POST: infer B } ? B : never
): Promise<APIEndpoints[Path][Method]> {
  let url: string = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url = url.replace(` + "`" + `:${key}` + "`" + `, value as string);
    });
  }
  const res = await fetch(url, {
    method: method as string,
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

// 2. Type-safe event emitter
type EventMap = {
  userLogin: { userId: string; timestamp: Date };
  userLogout: { userId: string };
  pageView: { path: string; referrer?: string };
  error: { message: string; code: number };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners = new Map<keyof T, Set<Function>>();

  on<K extends keyof T>(event: K, callback: (data: T[K]) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
    return () => this.listeners.get(event)!.delete(callback);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(cb => cb(data));
  }
}

const emitter = new TypedEventEmitter<EventMap>();
emitter.on('userLogin', (data) => {
  // data is typed as { userId: string; timestamp: Date }
  console.log(data.userId);
});

// 3. Builder pattern with types
type BuilderState<Required extends string, Optional extends string> = {
  required: Required;
  optional: Optional;
};

class QueryBuilder<
  Selected extends string = never,
  Filtered extends string = never
> {
  private query: any = {};

  select<F extends string>(...fields: F[]): QueryBuilder<F, Filtered> {
    this.query.select = fields;
    return this as any;
  }

  where<F extends string>(
    field: F,
    op: '=' | '!=' | '>' | '<',
    value: any
  ): QueryBuilder<Selected, Filtered | F> {
    this.query.where = { field, op, value };
    return this as any;
  }

  build(): { select: Selected[]; where: any } {
    return this.query;
  }
}

// 4. Discriminated union helpers
type Action =
  | { type: 'INCREMENT'; amount: number }
  | { type: 'DECREMENT'; amount: number }
  | { type: 'SET'; value: number }
  | { type: 'RESET' };

type ActionHandler<A extends Action, T extends A['type']> =
  (action: Extract<A, { type: T }>) => void;

function createReducer<S>(
  handlers: { [T in Action['type']]: (state: S, action: Extract<Action, { type: T }>) => S }
): (state: S, action: Action) => S {
  return (state, action) => {
    const handler = handlers[action.type] as any;
    return handler ? handler(state, action) : state;
  };
}`,
				},
				{
					Title: "Generic Utilities and Type-Level Programming",
					Content: `Advanced generic patterns enable building reusable, type-safe libraries and frameworks.

**Advanced Generics:**
` + "```" + `typescript
// Constrained generics
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

// Generic with default
function createStore<T extends Record<string, any> = Record<string, unknown>>(
  initialState: T
) {
  let state = { ...initialState };
  return {
    get<K extends keyof T>(key: K): T[K] { return state[key]; },
    set<K extends keyof T>(key: K, value: T[K]) { state[key] = value; },
    getState(): Readonly<T> { return state; },
  };
}

// Variadic tuple types
type Concat<A extends any[], B extends any[]> = [...A, ...B];
type Push<T extends any[], V> = [...T, V];
type Unshift<T extends any[], V> = [V, ...T];

// Typed pipe function
type PipeFn<A, B> = (a: A) => B;

function pipe<A, B>(fn1: PipeFn<A, B>): PipeFn<A, B>;
function pipe<A, B, C>(fn1: PipeFn<A, B>, fn2: PipeFn<B, C>): PipeFn<A, C>;
function pipe<A, B, C, D>(
  fn1: PipeFn<A, B>, fn2: PipeFn<B, C>, fn3: PipeFn<C, D>
): PipeFn<A, D>;
function pipe(...fns: Function[]) {
  return (input: any) => fns.reduce((acc, fn) => fn(acc), input);
}

const processUser = pipe(
  (id: number) => fetchUser(id),           // number => Promise<User>
  (userP: Promise<User>) => userP,
);

// Branded types for type safety
type Brand<T, B> = T & { __brand: B };
type USD = Brand<number, 'USD'>;
type EUR = Brand<number, 'EUR'>;
type UserId = Brand<string, 'UserId'>;
type Email = Brand<string, 'Email'>;

function usd(amount: number): USD { return amount as USD; }
function eur(amount: number): EUR { return amount as EUR; }

function addUSD(a: USD, b: USD): USD {
  return (a + b) as USD;
}
// addUSD(usd(10), eur(20)); // Type error!
addUSD(usd(10), usd(20)); // OK

// Validation with branded types
function validateEmail(input: string): Email | null {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(input) ? (input as Email) : null;
}

function sendEmail(to: Email, subject: string, body: string) {
  // to is guaranteed to be a valid email
}

// Path access types
type PathValue<T, P extends string> =
  P extends ` + "`" + `${infer K}.${infer Rest}` + "`" + `
    ? K extends keyof T
      ? PathValue<T[K], Rest>
      : never
    : P extends keyof T
      ? T[P]
      : never;

type NestedObj = {
  user: {
    profile: {
      name: string;
      address: {
        city: string;
        zip: string;
      };
    };
    settings: { theme: 'light' | 'dark' };
  };
};

type City = PathValue<NestedObj, 'user.profile.address.city'>; // string
type Theme = PathValue<NestedObj, 'user.settings.theme'>; // 'light' | 'dark'

function get<T, P extends string>(obj: T, path: P): PathValue<T, P> {
  return path.split('.').reduce((o: any, k) => o?.[k], obj) as any;
}
` + "```" + `

**Type-safe Patterns:**
` + "```" + `typescript
// Exhaustive switch helper
function assertNever(x: never): never {
  throw new Error('Unexpected value: ' + x);
}

type Shape =
  | { kind: 'circle'; radius: number }
  | { kind: 'rectangle'; width: number; height: number }
  | { kind: 'triangle'; base: number; height: number };

function area(shape: Shape): number {
  switch (shape.kind) {
    case 'circle': return Math.PI * shape.radius ** 2;
    case 'rectangle': return shape.width * shape.height;
    case 'triangle': return (shape.base * shape.height) / 2;
    default: return assertNever(shape); // compile error if case missing
  }
}

// Type guards
function isString(value: unknown): value is string {
  return typeof value === 'string';
}

function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' && value !== null &&
    'id' in value && 'name' in value
  );
}

// Assertion functions
function assertDefined<T>(
  value: T | null | undefined,
  message?: string
): asserts value is T {
  if (value == null) {
    throw new Error(message || 'Value is null or undefined');
  }
}

// Const assertions for literal types
const ROUTES = {
  HOME: '/',
  ABOUT: '/about',
  USERS: '/users',
  USER_DETAIL: '/users/:id',
} as const;

type Route = typeof ROUTES[keyof typeof ROUTES];
// '/' | '/about' | '/users' | '/users/:id'

// Satisfies operator (TS 5.0+)
const palette = {
  red: [255, 0, 0],
  green: '#00ff00',
  blue: [0, 0, 255],
} satisfies Record<string, string | number[]>;

// palette.red is still number[] (not string | number[])
const redComponent = palette.red[0]; // number, not error

// Using NoInfer to prevent inference from specific positions
function createFSM<S extends string>(config: {
  initial: NoInfer<S>;
  states: Record<S, { on?: Record<string, NoInfer<S>> }>;
}) {
  // S is inferred only from states keys
}
` + "```" + ``,
					CodeExamples: `// Type-level programming examples

// 1. Type-safe router
type RouteDefinition = {
  path: string;
  params?: Record<string, string>;
};

type ExtractRouteParams<T extends string> =
  T extends ` + "`" + `${string}:${infer Param}/${infer Rest}` + "`" + `
    ? { [K in Param]: string } & ExtractRouteParams<Rest>
    : T extends ` + "`" + `${string}:${infer Param}` + "`" + `
    ? { [K in Param]: string }
    : {};

function createRoute<P extends string>(path: P) {
  return {
    path,
    build(params: ExtractRouteParams<P>): string {
      let result: string = path;
      for (const [key, value] of Object.entries(params)) {
        result = result.replace(':' + key, value as string);
      }
      return result;
    },
  };
}

const userRoute = createRoute('/users/:userId/posts/:postId');
userRoute.build({ userId: '123', postId: '456' }); // '/users/123/posts/456'
// userRoute.build({ userId: '123' }); // Error: missing postId

// 2. Type-safe SQL query builder
type SQLType = 'string' | 'number' | 'boolean' | 'date';
type TSType<T extends SQLType> =
  T extends 'string' ? string :
  T extends 'number' ? number :
  T extends 'boolean' ? boolean :
  T extends 'date' ? Date :
  never;

type Schema = Record<string, Record<string, SQLType>>;

type TableRow<S extends Schema, T extends keyof S> = {
  [K in keyof S[T]]: TSType<S[T][K]>;
};

function defineSchema<S extends Schema>(schema: S) {
  return {
    from<T extends keyof S & string>(table: T) {
      return {
        select<C extends (keyof S[T] & string)[]>(...columns: C) {
          return {
            where(condition: Partial<TableRow<S, T>>) {
              return [] as Pick<TableRow<S, T>, C[number]>[];
            },
          };
        },
      };
    },
  };
}

const db = defineSchema({
  users: { id: 'number', name: 'string', active: 'boolean' },
  posts: { id: 'number', title: 'string', authorId: 'number' },
});

// Fully typed!
const users = db.from('users').select('name', 'active').where({ active: true });
// type: { name: string; active: boolean }[]

// 3. Middleware chain types
type Middleware<Ctx, NextCtx> = (
  ctx: Ctx,
  next: (ctx: NextCtx) => Promise<void>
) => Promise<void>;

type InferContext<M> = M extends Middleware<any, infer C> ? C : never;

class MiddlewareChain<Ctx = {}> {
  private middlewares: Function[] = [];

  use<NextCtx>(
    middleware: Middleware<Ctx, Ctx & NextCtx>
  ): MiddlewareChain<Ctx & NextCtx> {
    this.middlewares.push(middleware);
    return this as any;
  }

  async run(ctx: Ctx): Promise<void> {
    let index = 0;
    const next = async (ctx: any) => {
      const mw = this.middlewares[index++];
      if (mw) await mw(ctx, next);
    };
    await next(ctx);
  }
}

// Type accumulates through middleware chain
const chain = new MiddlewareChain()
  .use(async (ctx, next) => {
    await next({ ...ctx, userId: '123' });
  })
  .use(async (ctx, next) => {
    // ctx now has userId: string
    await next({ ...ctx, isAdmin: true });
  });`,
				},
			},
		},
	})
}
