# TechFlow API Reference

## Authentication

All API requests require an API key passed as a Bearer token in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

API keys are generated from the TechFlow dashboard under Settings → API Keys.
Each key is scoped to a specific set of permissions (read, write, admin).

## Rate Limits

The API enforces the following rate limits:

- **Standard plan**: 100 requests per minute per API key
- **Pro plan**: 1,000 requests per minute per API key
- **Enterprise plan**: 10,000 requests per minute per API key

Rate limit headers are included in every response:
- `X-RateLimit-Limit`: your plan's limit
- `X-RateLimit-Remaining`: requests remaining in the current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

When you exceed the rate limit, the API returns a `429 Too Many Requests` response.
Implement exponential backoff starting at 1 second, doubling up to 60 seconds maximum.

## Payload Size Limits

- Maximum request body: **10 MB**
- Maximum response body: **50 MB**
- Maximum number of items in a batch request: **500**

## Webhooks

TechFlow supports webhook callbacks for asynchronous events (e.g., document processing complete).

### Webhook Retry Policy

If your endpoint returns a non-2xx status code, TechFlow retries the webhook:
- Retry 1: after 5 minutes
- Retry 2: after 30 minutes
- Retry 3: after 2 hours
- Retry 4: after 8 hours
- Retry 5: after 24 hours

After 5 failed retries, the webhook delivery is marked as failed.
Configure your endpoint to respond within 30 seconds to avoid timeout.
Use idempotency keys (`X-TechFlow-Event-ID` header) to handle duplicate deliveries.

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request — invalid parameters |
| 401 | Unauthorized — invalid or missing API key |
| 403 | Forbidden — insufficient permissions |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Temporarily Unavailable |
