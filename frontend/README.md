# OCR_LOCAL Operator Console (Plan D Wave 1)

Next.js 15 management UI for the OCR_LOCAL pipeline. The console now includes
API-key auth, dashboard health with EDC_TRANSLATION readiness, jobs list/detail
with WebSocket progress, batch submit/status, output artifact downloads,
DocumentBundle/evidence links, custody audit views, worker fleet/queue views
with queue threshold editing, review flows, tenant settings, and feature-flag
operations.

- **D2** -- Dashboard summary cards and pipeline gauges
- **D3** -- Jobs list + detail with WebSocket progress and output browser
- **D4** -- Custody timeline with client-side hash-chain verification
- **D5** -- Worker fleet + queue depth views

## Stack

- Next.js 15 (App Router)
- React 18 + TypeScript (strict mode)
- Tailwind CSS 3 with hand-rolled shadcn-style primitives
  (`Button`, `Input`, `Card`) -- no `shadcn` CLI install
- Vitest + Testing Library (jsdom)

## Environment contract

Copy `.env.example` to `.env.local` and adjust as needed:

| Variable                    | Default                  | Description                                                      |
|-----------------------------|--------------------------|------------------------------------------------------------------|
| `OCR_FRONTEND_API_BASE_URL` | `/` in production | Runtime backend origin or root-relative prefix. `/api/v1/*` paths are appended. Helm: `frontend.apiBaseUrl`. |
| `OCR_FRONTEND_HTTPONLY_SESSION` | `false` | Runtime cookie mode. Helm: `frontend.httpOnlySession`. |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` in development | Legacy/build fallback when the runtime variable is unset. |
| `NEXT_PUBLIC_OCR_HTTPONLY_SESSION` | `false` | Legacy/build cookie-mode fallback when the runtime variable is unset. |

The server places only the API base and cookie-mode flag in each HTML response.
Changing the runtime variables requires a server/pod restart and a page reload,
but no image rebuild. Browser Settings can override the API base. Absolute bases
must use HTTP(S), without user credentials, query strings or fragments.

In the default API-key mode, operators paste an OCR_LOCAL API key on `/login`.
The browser stores it in `sessionStorage` under `ocr_local_api_key` and sends
it as `X-API-Key`; API requests explicitly omit ambient cookies. A legacy
`localStorage` value is migrated to `sessionStorage` and removed. When
`OCR_FRONTEND_HTTPONLY_SESSION=true`, login exchanges the candidate key for
an HttpOnly session cookie and removes browser-readable copies. The browser
sends that cookie only in this mode. API-key mode also offers **Clear prior
session** to revoke a cookie left by a previous cookie-mode build.

A `401` means the credential was rejected: the console clears local key state
and returns to `/login`. A `403` is a policy or capability denial and preserves
the current credential so the operator can address the policy without
re-authenticating. The session cookie is `HttpOnly`, `SameSite=Strict`, and
`Secure` in production deployments.

## Commands

```bash
npm install      # install dependencies
npm run dev      # start the dev server on http://localhost:3000
npm run build    # production build
npm run start    # serve the production build
npm run test     # vitest run (single pass)
npm run test:watch
npm run lint     # next lint (eslint-config-next)
npm run format   # prettier --write src/**
```

## Container and Helm

The repository root includes `Dockerfile.frontend` for the operator console.
The Helm chart deploys it when `frontend.enabled=true` and routes same-origin
Ingress traffic to the frontend while preserving `/api/*` for the coordinator.

## Routes

| Path         | Status         | Lands in |
|--------------|----------------|----------|
| `/`          | Redirects to `/dashboard` | D1 |
| `/login`     | API-key entry  | D1 |
| `/dashboard` | Health, EDC_TRANSLATION readiness, and dashboard summary | D2 |
| `/jobs`      | Jobs list, filters, detail, progress, outputs | D3 |
| `/batches`   | Batch submit and status list | EDC_OCR |
| `/batches/[batchId]` | Batch child job status | EDC_OCR |
| `/audit`     | Custody timeline and verification | D4 |
| `/fleet`     | Worker fleet, queue depth, and queue threshold controls | D5 |
| `/review`    | Review queue and certification flow | Later wave |
| `/admin/tenants` | Tenant config and glossary controls | Later wave |
| `/admin/features` | Feature flag operations | Later wave |

## File layout

```
frontend/
├── src/
│   ├── app/                 # App Router routes
│   │   ├── layout.tsx       # Sidebar + Topbar shell
│   │   ├── page.tsx         # Redirects to /dashboard
│   │   ├── login/page.tsx   # API key entry
│   │   ├── dashboard/page.tsx
│   │   ├── jobs/page.tsx
│   │   ├── audit/page.tsx
│   │   ├── fleet/page.tsx
│   │   ├── review/
│   │   └── admin/
│   ├── components/
│   │   ├── sidebar.tsx
│   │   ├── topbar.tsx
│   │   ├── JobOutputs.tsx
│   │   └── ui/              # Button, Input, Card primitives
│   └── lib/
│       ├── api-client.ts    # fetch wrapper, X-API-Key injection
│       ├── outputs-api.ts   # output, DocumentBundle, evidence links
│       ├── auth.ts          # session/API-key helpers + useRequireAuth
│       └── cn.ts            # className join helper
└── __tests__/               # vitest suite
```

## Auth flow

1. Operator visits any protected route.
2. `useRequireAuth()` redirects to `/login` if no key is cached.
3. Operator submits the key; the login probe validates it before storage.
4. API-key mode stores it in `sessionStorage` and sends `X-API-Key` with
   cookies omitted. HttpOnly mode exchanges it for a server-issued cookie.
5. A `401` clears rejected auth and redirects to `/login`; a `403` preserves
   auth and reports the denial. Sign out deletes the server cookie and clears
   local key state. API-key mode exposes cookie cleanup for a prior session.

Server-side enforcement remains the FastAPI layer's responsibility; this UI
is a defense-in-depth client gate, not the trust boundary.
