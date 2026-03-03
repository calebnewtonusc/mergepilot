# TypeScript / JavaScript Code Review Guide

Reference guide for MergePilot's TypeScript-specific review patterns.

## Type Safety

### Avoiding `any`
```typescript
// WRONG: disables type checking
function process(data: any): any {
  return data.value;
}

// RIGHT: use unknown with type guards
function process(data: unknown): string {
  if (typeof data === "object" && data !== null && "value" in data) {
    return String((data as { value: unknown }).value);
  }
  throw new Error("Invalid data shape");
}
```

### Non-null assertions without checks
```typescript
// WRONG: will throw if user is null
const name = user!.name;

// RIGHT: explicit null check
if (!user) throw new Error("User not found");
const name = user.name;

// or: optional chaining with fallback
const name = user?.name ?? "Anonymous";
```

### Type assertions instead of guards
```typescript
// WRONG: unsafe cast
const response = await fetch(url);
const data = await response.json() as UserResponse;

// RIGHT: validate at runtime
const raw = await response.json();
const data = UserResponseSchema.parse(raw); // zod, io-ts, etc.
```

---

## React / Next.js

### Missing dependency arrays (useEffect)
```tsx
// WRONG: runs on every render
useEffect(() => {
  fetchUser(userId);
});

// WRONG: stale closure — fetchUser reads old userId
useEffect(() => {
  fetchUser(userId);
}, []); // ESLint: react-hooks/exhaustive-deps

// RIGHT
useEffect(() => {
  fetchUser(userId);
}, [userId]);
```

### Inline object/function props cause re-renders
```tsx
// WRONG: new object reference on every render
<Component style={{ margin: 0 }} onClick={() => handleClick(id)} />

// RIGHT: stable references
const style = useMemo(() => ({ margin: 0 }), []);
const handleClick = useCallback(() => doClick(id), [id]);
<Component style={style} onClick={handleClick} />
```

### XSS via `dangerouslySetInnerHTML`
```tsx
// WRONG: executes arbitrary user HTML
<div dangerouslySetInnerHTML={{ __html: userContent }} />

// RIGHT: sanitize first
import DOMPurify from "dompurify";
<div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(userContent) }} />
```

### Keys using array index
```tsx
// WRONG: breaks reconciliation when list reorders
{items.map((item, index) => <Row key={index} item={item} />)}

// RIGHT: use stable unique ID
{items.map((item) => <Row key={item.id} item={item} />)}
```

---

## Async / Promises

### Unhandled promise rejections
```typescript
// WRONG: rejection silently swallowed
someAsyncOp().then(result => {
  process(result);
});

// RIGHT: always handle rejection
someAsyncOp()
  .then(result => process(result))
  .catch(err => logger.error("Failed:", err));

// or: use async/await with try/catch
try {
  const result = await someAsyncOp();
  process(result);
} catch (err) {
  logger.error("Failed:", err);
}
```

### Floating promises (no await)
```typescript
// WRONG: async function not awaited
async function saveUser(user: User) {
  db.insert(user);  // not awaited — errors lost
}

// RIGHT
async function saveUser(user: User) {
  await db.insert(user);
}
```

### Sequential awaits that could be parallel
```typescript
// WRONG: 3 sequential network calls
const user = await getUser(id);
const posts = await getPosts(id);
const comments = await getComments(id);

// RIGHT: parallel
const [user, posts, comments] = await Promise.all([
  getUser(id),
  getPosts(id),
  getComments(id),
]);
```

---

## Security

### Prototype pollution
```typescript
// WRONG: user controls property name
function merge(target: any, source: any) {
  for (const key of Object.keys(source)) {
    target[key] = source[key];  // allows __proto__ pollution
  }
}

// RIGHT: check for dangerous keys
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);
for (const key of Object.keys(source)) {
  if (!FORBIDDEN_KEYS.has(key)) {
    target[key] = source[key];
  }
}
```

### Open redirects
```typescript
// WRONG: redirect to attacker-controlled URL
const next = req.query.next as string;
res.redirect(next);

// RIGHT: validate against allowlist
const ALLOWED_HOSTS = new Set(["app.example.com"]);
const url = new URL(next, "https://app.example.com");
if (!ALLOWED_HOSTS.has(url.hostname)) {
  return res.redirect("/");
}
res.redirect(url.toString());
```

---

## Node.js / Server

### Synchronous FS in request handlers
```typescript
// WRONG: blocks event loop
app.get("/config", (req, res) => {
  const config = fs.readFileSync("./config.json", "utf8");
  res.json(JSON.parse(config));
});

// RIGHT: async
app.get("/config", async (req, res) => {
  const config = await fs.promises.readFile("./config.json", "utf8");
  res.json(JSON.parse(config));
});
```

### Not validating request input
```typescript
// WRONG: trusting user input directly
app.post("/users", async (req, res) => {
  await db.createUser(req.body);
});

// RIGHT: validate with schema
import { z } from "zod";
const CreateUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
});

app.post("/users", async (req, res) => {
  const body = CreateUserSchema.parse(req.body);
  await db.createUser(body);
});
```

---

## Performance

### Missing memoization for expensive computations
```typescript
// WRONG: recomputed on every render
const sortedItems = items.sort((a, b) => b.score - a.score);

// RIGHT
const sortedItems = useMemo(
  () => [...items].sort((a, b) => b.score - a.score),
  [items]
);
```

### Large bundle — missing dynamic imports
```typescript
// WRONG: ships heavy library to all users
import Chart from "chart.js";

// RIGHT: load on demand
const Chart = dynamic(() => import("chart.js"), { ssr: false });
```
