# Python Code Review Guide

Reference guide for MergePilot's Python-specific review patterns.

## Correctness

### Off-by-one errors
```python
# WRONG: misses last element
for i in range(len(items) - 1):

# RIGHT
for i in range(len(items)):
# or better
for item in items:
```

### Mutable default arguments
```python
# WRONG: list shared across all calls
def process(data, result=[]):
    result.append(data)
    return result

# RIGHT
def process(data, result=None):
    if result is None:
        result = []
    result.append(data)
    return result
```

### Late binding in closures
```python
# WRONG: all lambdas capture i=9 at call time
funcs = [lambda x: x + i for i in range(10)]

# RIGHT: bind i at definition time
funcs = [lambda x, i=i: x + i for i in range(10)]
```

---

## Security

### SQL injection
```python
# WRONG: f-string in query
cursor.execute(f"SELECT * FROM users WHERE id={user_id}")

# RIGHT: parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### Insecure deserialization
```python
# WRONG: pickle from untrusted source
import pickle
data = pickle.loads(user_input)

# RIGHT: use json for untrusted data
import json
data = json.loads(user_input)
```

### Hardcoded secrets
```python
# WRONG
API_KEY = "sk-abc123secrettoken"

# RIGHT
import os
API_KEY = os.environ["API_KEY"]
```

### Path traversal
```python
# WRONG: allows ../../etc/passwd
path = Path(base_dir) / user_filename

# RIGHT: validate and resolve
path = (Path(base_dir) / user_filename).resolve()
if not path.is_relative_to(base_dir):
    raise ValueError("Invalid path")
```

---

## Performance

### N+1 queries (Django ORM)
```python
# WRONG: N+1 queries
for post in Post.objects.all():
    print(post.author.name)  # query per post

# RIGHT: eager loading
for post in Post.objects.select_related("author").all():
    print(post.author.name)
```

### List comprehension vs generator
```python
# WRONG: materializes full list in memory
total = sum([x**2 for x in range(1_000_000)])

# RIGHT: generator expression — O(1) memory
total = sum(x**2 for x in range(1_000_000))
```

### String concatenation in loops
```python
# WRONG: O(n²) — new string object each iteration
result = ""
for item in items:
    result += str(item)

# RIGHT: O(n)
result = "".join(str(item) for item in items)
```

### Using sets for membership testing
```python
# WRONG: O(n) lookup
if item in large_list:

# RIGHT: O(1) lookup
large_set = set(large_list)
if item in large_set:
```

---

## Error Handling

### Bare except
```python
# WRONG: catches SystemExit, KeyboardInterrupt, etc.
try:
    risky_operation()
except:
    pass

# RIGHT: catch specific exceptions
try:
    risky_operation()
except (ValueError, IOError) as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Silent exception swallowing
```python
# WRONG
try:
    result = compute()
except Exception:
    result = None  # caller can't tell if it failed

# RIGHT
try:
    result = compute()
except Exception as e:
    logger.warning(f"compute() failed: {e}")
    raise RuntimeError("Computation failed") from e
```

---

## Type Safety

### Missing type annotations on public API
```python
# WRONG: no annotations
def process_user(user, config):
    ...

# RIGHT: full annotations
from typing import Optional
def process_user(user: User, config: dict[str, str]) -> Optional[ProcessResult]:
    ...
```

### Using `Any` unnecessarily
```python
# WRONG: defeats type checking
from typing import Any
def get_value(key: str) -> Any:
    return self._data[key]

# RIGHT: use Union or TypeVar
from typing import Union
def get_value(key: str) -> Union[str, int, None]:
    return self._data.get(key)
```

---

## Code Structure

### Long functions (> 50 lines)
Extract helper functions. Each function should do one thing.

### Deep nesting (> 3 levels)
```python
# WRONG: arrow anti-pattern
def process(data):
    if data:
        if data.valid:
            if data.items:
                for item in data.items:
                    ...

# RIGHT: early returns
def process(data):
    if not data:
        return
    if not data.valid:
        return
    if not data.items:
        return
    for item in data.items:
        ...
```

### Magic numbers
```python
# WRONG
if retry_count > 3:
    time.sleep(0.5)

# RIGHT
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 0.5
if retry_count > MAX_RETRIES:
    time.sleep(RETRY_DELAY_SECONDS)
```

---

## Testing

### Weak assertions
```python
# WRONG: passes even if result is wrong type
assert result is not None

# RIGHT: assert exact expected value
assert result == {"status": "ok", "count": 5}
```

### Not testing edge cases
Always test:
- Empty input (`[]`, `""`, `None`)
- Single element
- Maximum valid input
- Invalid/boundary input
- Error paths
