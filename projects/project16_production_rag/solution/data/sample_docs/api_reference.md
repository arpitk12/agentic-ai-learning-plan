# TechFlow API Reference — v1

## Base URL

```
https://api.techflow.io/v1
```

All requests must be made over HTTPS.  Plain HTTP is rejected with **301 Moved Permanently**.

---

## Authentication

TechFlow uses **Bearer token** authentication.  Include your API token in every request:

```http
Authorization: Bearer <your_api_token>
```

You can generate a token from **Settings → API → Generate Token** in the TechFlow dashboard.
Tokens do not expire but can be revoked at any time.

> **Note**: Tokens carry the permissions of the user who created them.  
> For CI/CD pipelines, create a dedicated service-account user.

---

## Rate Limits

| Plan         | Requests per Hour |
| ------------ | ----------------- |
| Starter      | 100               |
| Professional | 1 000             |
| Enterprise   | 10 000            |

Rate limit headers returned on every response:

| Header                  | Description                                |
| ----------------------- | ------------------------------------------ |
| `X-RateLimit-Limit`     | Maximum allowed requests per hour          |
| `X-RateLimit-Remaining` | Requests remaining in the current window   |
| `X-RateLimit-Reset`     | UTC epoch seconds when the window resets   |

When the limit is exceeded the API returns **429 Too Many Requests**.

---

## Endpoints

### Projects

#### List all projects

```http
GET /v1/projects
```

Returns a **paginated** list of all projects visible to the authenticated user.

**Query parameters**

| Parameter  | Type    | Default | Description                       |
| ---------- | ------- | ------- | --------------------------------- |
| `page`     | integer | 1       | Page number (1-indexed)           |
| `per_page` | integer | 20      | Items per page (max 100)          |
| `sort`     | string  | `created_at` | Field to sort by (`name`, `updated_at`) |
| `order`    | string  | `desc`  | `asc` or `desc`                   |

**Response (200 OK)**

```json
{
  "data": [
    {
      "id": "proj_01ABC",
      "name": "Backend Redesign",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-03-22T14:05:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 42
  }
}
```

#### Get a project

```http
GET /v1/projects/{project_id}
```

#### Create a project

```http
POST /v1/projects
Content-Type: application/json

{
  "name": "My New Project",
  "description": "Optional description",
  "visibility": "private"
}
```

#### Delete a project

```http
DELETE /v1/projects/{project_id}
```

Returns **204 No Content** on success.

---

### Tasks

#### List tasks in a project

```http
GET /v1/projects/{project_id}/tasks
```

Supports the same pagination parameters as `/v1/projects`.

#### Create a task

```http
POST /v1/projects/{project_id}/tasks
Content-Type: application/json

{
  "title": "Fix login bug",
  "description": "Users cannot log in with SSO on Safari",
  "priority": "high",
  "assignee_id": "usr_XYZ"
}
```

**Priority values**: `low` | `medium` | `high` | `critical`

---

## Error Format

All errors follow a consistent JSON shape:

```json
{
  "error": {
    "code": "not_found",
    "message": "Project proj_INVALID does not exist or you do not have access.",
    "request_id": "req_abc123"
  }
}
```

Common HTTP status codes:

| Code | Meaning                              |
| ---- | ------------------------------------ |
| 400  | Bad request — invalid payload        |
| 401  | Unauthorised — missing / bad token   |
| 403  | Forbidden — insufficient permissions |
| 404  | Not found                            |
| 429  | Rate limit exceeded                  |
| 500  | Internal server error                |

---

## Pagination

All list endpoints return a `meta` object.  Use `page` and `per_page` to navigate:

```http
GET /v1/projects?page=2&per_page=50
```

The API does **not** use cursor-based pagination on the v1 surface.

---

## Webhooks

TechFlow can push events to your server. Configure webhook URLs under
**Settings → Webhooks**. Each payload includes an HMAC-SHA256 signature in the
`X-TechFlow-Signature` header so you can verify authenticity.

Supported events: `task.created`, `task.updated`, `task.deleted`,
`project.created`, `project.archived`, `member.invited`.
