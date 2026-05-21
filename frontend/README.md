# EDCOCR Operator Console

Next.js 15 management UI for the EDCOCR pipeline. The console now includes
API-key auth, dashboard health with EXTERNAL_TRANSLATION readiness, jobs list/detail
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
| `NEXT_PUBLIC_API_BASE_URL`  | `http://localhost:8000`  | Base URL of the FastAPI backend. `/api/v1/*` paths are appended. |

Authentication is API-key only for . Operators paste an EDCOCR
API key on `/login`; it is stored in `localStorage` under
`ocr_local_api_key` and forwarded as the `X-API-Key` header on every
request. A 401 or 403 response clears the cached key.

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
| `/dashboard` | Health, EXTERNAL_TRANSLATION readiness, and dashboard summary | D2 |
| `/jobs`      | Jobs list, filters, detail, progress, outputs | D3 |
| `/batches`   | Batch submit and status list | EDCOCR |
| `/batches/[batchId]` | Batch child job status | EDCOCR |
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
│       ├── auth.ts          # localStorage helpers + useRequireAuth
│       └── cn.ts            # className join helper
└── __tests__/               # vitest suite
```

## Auth flow

1. Operator visits any protected route.
2. `useRequireAuth` redirects to `/login` if no key is cached.
3. Operator submits the key; `setApiKey` validates non-empty and persists.
4. Subsequent requests use `lib/api-client` which injects `X-API-Key`.
5. On `401` or `403`, the client clears the cached key via `clearApiKey`.

Server-side enforcement remains the FastAPI layer's responsibility; this UI
is a defense-in-depth client gate, not the trust boundary.
