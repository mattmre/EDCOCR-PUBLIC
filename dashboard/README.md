# EDCOCR Operations Dashboard

Production deployment configuration for the EDCOCR operations dashboard.
Serves a React SPA via nginx with API proxying to the coordinator backend.

## Quick Start

### Development

```bash
cd dashboard
npm install
npm start
```

The development server runs on `http://localhost:3000` and proxies API
requests to `http://localhost:8000`.

### Production Build

```bash
cd dashboard
npm run build
```

Built assets are output to `dashboard/build/`.

### Docker Deployment

```bash
# Build and run standalone
docker-compose -f dashboard/docker-compose.dashboard.yml up -d

# Build with custom API URL
docker-compose -f dashboard/docker-compose.dashboard.yml up -d \
  --build --build-arg NEXT_PUBLIC_API_URL=https://api.example.com

# Connect to existing coordinator stack
docker-compose -f coordinator/docker-compose.coordinator.yml \
               -f dashboard/docker-compose.dashboard.yml up -d
```

## Environment Variables

### Build-Time

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base URL injected at build time |

### Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://coordinator:8000` | Backend API URL for nginx proxy |
| `API_KEY` | (none) | API key for authenticated requests |
| `DASHBOARD_PORT` | `3000` | Host port for dashboard |
| `OCR_NETWORK` | `coordinator_default` | Docker network to join |

## Architecture

```
Browser --> nginx (port 3000)
              |
              |--> /           --> SPA (React build)
              |--> /api/*      --> proxy to coordinator:8000
              |--> /ws/*       --> WebSocket proxy to coordinator:8000
              |--> /health     --> nginx health response
              |--> /static/*   --> cached static assets (1y)
```

## nginx Configuration

The `nginx.conf` provides:

- **SPA Routing**: `try_files $uri /index.html` for client-side routing
- **API Proxy**: `/api/` requests forwarded to coordinator backend
- **WebSocket Proxy**: `/ws/` WebSocket upgrade for real-time job updates
- **Security Headers**: X-Frame-Options, CSP, HSTS, XSS-Protection
- **Gzip Compression**: Enabled for text, CSS, JS, JSON, SVG
- **Static Caching**: 1-year cache for hashed build assets
- **Health Check**: `/health` endpoint for Docker/Kubernetes probes

## Security

- Runs as non-root nginx user inside container
- Content-Security-Policy restricts script/style sources
- X-Frame-Options prevents clickjacking
- Hidden files (dotfiles) are denied
- API proxy preserves original client IP via X-Forwarded-For

## Integration with Coordinator

The dashboard connects to these coordinator API endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/metrics/` | Pipeline throughput metrics |
| `GET /api/v1/jobs/` | Job listing and status |
| `GET /api/v1/dashboard/throughput/` | Dashboard throughput data |
| `GET /api/v1/dashboard/fleet/` | Worker fleet status |
| `GET /api/v1/dashboard/alerts/` | Active alerts |
| `GET /api/v1/dashboard/analytics/` | Historical analytics |
| `WS /ws/jobs/{id}` | Real-time job progress |

## Kubernetes Deployment

For Kubernetes deployments, use the Helm chart values:

```yaml
dashboard:
  enabled: true
  replicas: 2
  image:
    repository: ocr-local-dashboard
    tag: latest
  ingress:
    enabled: true
    host: dashboard.example.com
```
