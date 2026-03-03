# Security Review Guide

Cross-language security vulnerabilities that MergePilot must identify and flag.

## OWASP Top 10 Coverage

### A01 — Broken Access Control

**Missing authorization checks:**
```python
# WRONG: any logged-in user can delete any post
@login_required
def delete_post(request, post_id):
    post = Post.objects.get(id=post_id)
    post.delete()

# RIGHT: verify ownership
@login_required
def delete_post(request, post_id):
    post = get_object_or_404(Post, id=post_id, author=request.user)
    post.delete()
```

**Horizontal privilege escalation:**
- Accessing another user's data using their ID
- IDOR (Insecure Direct Object Reference): `GET /api/invoices/12345` — does the server verify 12345 belongs to the authenticated user?

**Signs to flag:**
- `get_object_or_404(Model, id=id)` without user filter
- Admin routes without admin check
- JWT claims not verified server-side

---

### A02 — Cryptographic Failures

**Weak hashing:**
```python
# WRONG: MD5 for passwords
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()

# WRONG: SHA-256 without salt
password_hash = hashlib.sha256(password.encode()).hexdigest()

# RIGHT: bcrypt / argon2 / scrypt
from argon2 import PasswordHasher
ph = PasswordHasher()
password_hash = ph.hash(password)
```

**Hardcoded keys/IVs:**
```python
# WRONG
KEY = b"mysecretkey12345"
IV = b"0000000000000000"

# RIGHT: generate randomly
import os
KEY = os.urandom(32)
IV = os.urandom(16)
```

**Transmitting secrets in URL:**
```
WRONG: GET /api/data?api_key=sk-abc123
RIGHT: Authorization: Bearer sk-abc123  (header)
```

---

### A03 — Injection

**SQL Injection:**
```python
# WRONG
f"SELECT * FROM users WHERE name = '{name}'"

# RIGHT: parameterized
"SELECT * FROM users WHERE name = %s", (name,)
```

**Command Injection:**
```python
# WRONG
import os
os.system(f"ffmpeg -i {user_filename} output.mp4")

# WRONG
import subprocess
subprocess.run(f"convert {user_input}", shell=True)

# RIGHT
subprocess.run(["ffmpeg", "-i", user_filename, "output.mp4"])
```

**SSTI (Server-Side Template Injection):**
```python
# WRONG: Jinja2 with user-controlled template
from jinja2 import Template
template = Template(user_provided_template)
result = template.render()

# RIGHT: render with sandboxed environment or use fixed templates
from jinja2 import Environment, select_autoescape
env = Environment(autoescape=select_autoescape())
template = env.from_string(FIXED_TEMPLATE)
result = template.render(user_var=sanitized_value)
```

**LDAP Injection, XPath Injection, NoSQL Injection:**
- Same principle: never interpolate user input into query strings
- Use parameterized APIs or ORMs

---

### A05 — Security Misconfiguration

**Debug mode in production:**
```python
# WRONG in production
DEBUG = True
```

**Missing security headers:**
```nginx
# Should have:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# Strict-Transport-Security: max-age=31536000
# Content-Security-Policy: default-src 'self'
```

**Verbose error messages:**
```python
# WRONG: leaks stack trace to user
return {"error": traceback.format_exc()}

# RIGHT: log internally, return generic message
logger.error("Internal error", exc_info=True)
return {"error": "Internal server error"}
```

---

### A07 — Identification and Authentication Failures

**Weak session tokens:**
```python
# WRONG: predictable token
session_id = str(user_id) + str(int(time.time()))

# RIGHT: cryptographically random
import secrets
session_id = secrets.token_urlsafe(32)
```

**No rate limiting on login:**
- Flag any auth endpoint without rate limiting or account lockout

**JWT algorithm confusion:**
```python
# WRONG: accepts 'none' algorithm
jwt.decode(token, options={"verify_signature": False})

# RIGHT: specify allowed algorithms
jwt.decode(token, SECRET, algorithms=["HS256"])
```

---

### A09 — Security Logging and Monitoring Failures

**Logging sensitive data:**
```python
# WRONG: password in log
logger.info(f"User login: {username} with password {password}")

# WRONG: full card number logged
logger.debug(f"Processing card: {card_number}")

# RIGHT: log event without sensitive fields
logger.info(f"User login attempt: {username}")
```

**Not logging security events:**
Flag when there's no logging for:
- Failed authentication attempts
- Access control violations
- Admin actions
- Data exports

---

## Additional Patterns

### Race Conditions / TOCTOU

```python
# WRONG: check-then-use race
if not User.objects.filter(email=email).exists():
    User.objects.create(email=email)  # another request could create between check and create

# RIGHT: unique constraint + handle IntegrityError
try:
    User.objects.create(email=email)
except IntegrityError:
    raise ValueError("Email already registered")
```

### Insecure Deserialization

```python
# WRONG: never deserialize untrusted pickle/yaml.load
import pickle
data = pickle.loads(request.body)

import yaml
config = yaml.load(user_input)  # executes arbitrary code

# RIGHT
import json
data = json.loads(request.body)

import yaml
config = yaml.safe_load(user_input)
```

### Open Redirects

```python
# WRONG
next_url = request.GET.get("next", "/")
return redirect(next_url)  # attacker sends ?next=https://evil.com

# RIGHT: validate scheme and host
from urllib.parse import urlparse
parsed = urlparse(next_url)
if parsed.scheme in ("http", "https") and parsed.netloc != "":
    next_url = "/"
return redirect(next_url)
```

### File Upload Security

```python
# WRONG: trust user-supplied filename
filename = request.FILES["upload"].name
save_path = os.path.join(UPLOAD_DIR, filename)

# WRONG: no file type validation
# RIGHT:
import uuid
from pathlib import Path

upload = request.FILES["upload"]
ext = Path(upload.name).suffix.lower()
ALLOWED_EXTS = {".jpg", ".png", ".pdf", ".csv"}
if ext not in ALLOWED_EXTS:
    raise ValidationError("File type not allowed")

# Generate safe filename
safe_name = f"{uuid.uuid4()}{ext}"
save_path = Path(UPLOAD_DIR) / safe_name
```

---

## Severity Classification

| Vulnerability | Severity |
|---|---|
| SQL/Command/SSTI injection | BLOCKING |
| Hardcoded secrets | BLOCKING |
| Missing auth check | BLOCKING |
| XSS | BLOCKING |
| Path traversal | BLOCKING |
| Weak cryptography | BLOCKING |
| Open redirect | SUGGESTION |
| Missing rate limiting | SUGGESTION |
| Sensitive data in logs | SUGGESTION |
| Missing security headers | OPTIONAL |
| Verbose errors | SUGGESTION |
