# @edcocr/sdk

TypeScript SDK for the [EDCOCR](https://github.com/mattmre/EDCOCR-PUBLIC) forensic-grade OCR pipeline.

- Zero runtime dependencies
- Works in Node.js 18+ and modern browsers
- Full TypeScript type definitions
- Automatic retry with exponential backoff
- WebSocket streaming for real-time progress

## Installation

```bash
npm install @edcocr/sdk
```

## Quick Start

```typescript
import { EDCOCRClient } from '@edcocr/sdk';

const client = new EDCOCRClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key',
});

// Check API health
const health = await client.healthCheck;
console.log(health.status); // "healthy"

// Submit a document and wait for completion
const status = await client.submitAndWait({
  fileBuffer: myPdfBytes,
  filename: 'document.pdf',
});

if (status.status === 'completed') {
  // Download the searchable PDF
  const pdf = await client.downloadArtifact(status.job_id, 'pdf');
  // Download extracted text
  const text = await client.downloadArtifact(status.job_id, 'text');
}

client.close;
```

## Usage

### Client Configuration

```typescript
import { EDCOCRClient } from '@edcocr/sdk';

const client = new EDCOCRClient({
  baseUrl: 'http://localhost:8000', // Required
  apiKey: 'your-api-key',          // Optional (X-API-Key header)
  timeoutMs: 30000,                // Request timeout (default: 30s)
  maxRetries: 3,                   // Retry count for transient errors
});
```

### Submit a Job

```typescript
// From a file path (Node.js only)
const job = await client.submitJob({
  filePath: '/path/to/document.pdf',
  priority: 'normal',
  enableDocintel: true,
  docintelMode: 'full',
});

// From a buffer (Node.js and browser)
const job = await client.submitJob({
  fileBuffer: new Uint8Array([...]),
  filename: 'document.pdf',
  webhookUrl: 'https://example.com/webhook',
});

console.log(job.job_id);  // "job_a1b2c3d4e5f6"
console.log(job.status);  // "queued"
```

### Check Job Status

```typescript
const status = await client.getStatus('job_a1b2c3d4e5f6');
console.log(status.status);                    // "processing"
console.log(status.progress?.percent_complete); // 45.0
```

### Wait for Completion

```typescript
const result = await client.waitForCompletion('job_a1b2c3d4e5f6', {
  pollIntervalMs: 2000,  // Poll every 2 seconds
  timeoutMs: 600000,     // Max 10 minutes
  onProgress: (status) => {
    console.log(`${status.progress?.percent_complete}% complete`);
  },
});
```

### Get Result Metadata

```typescript
const result = await client.getResult('job_a1b2c3d4e5f6');
console.log(result.artifacts);            // { pdf: "...", text: "..." }
console.log(result.processing_time_seconds);
```

### Download Artifacts

```typescript
const pdf = await client.downloadArtifact('job_a1b2c3d4e5f6', 'pdf');
const text = await client.downloadArtifact('job_a1b2c3d4e5f6', 'text');
const structure = await client.downloadArtifact('job_a1b2c3d4e5f6', 'structure');
```

### Cancel a Job

```typescript
const cancelled = await client.cancelJob('job_a1b2c3d4e5f6');
console.log(cancelled.status); // "cancelled"
```

### Retry a Failed Job

```typescript
const newJob = await client.retryJob('job_a1b2c3d4e5f6');
console.log(newJob.job_id); // New job ID
```

### List Jobs

```typescript
const list = await client.listJobs({
  status: 'completed',
  page: 1,
  perPage: 20,
});
console.log(list.total);     // Total matching jobs
console.log(list.jobs);      // Array of JobStatusResponse
```

### WebSocket Progress Streaming

```typescript
const ws = client.streamProgress('job_a1b2c3d4e5f6');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'connected':
      console.log('Connected, current status:', msg.status);
      break;
    case 'progress':
      console.log('Progress update:', msg.status);
      break;
    case 'completed':
      console.log('Job completed!');
      ws.close;
      break;
    case 'failed':
      console.error('Job failed:', msg.error);
      ws.close;
      break;
  }
};
```

## Error Handling

All errors extend `OCRLocalError`:

```typescript
import {
  OCRLocalError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ConflictError,
  ClientClosedError,
} from '@edcocr/sdk';

try {
  await client.getStatus('job_nonexistent0');
} catch (err) {
  if (err instanceof NotFoundError) {
    console.log('Job not found');
  } else if (err instanceof AuthenticationError) {
    console.log('Invalid API key');
  } else if (err instanceof RateLimitError) {
    console.log(`Rate limited, retry after ${err.retryAfterSeconds}s`);
  } else if (err instanceof ServerError) {
    console.log(`Server error: HTTP ${err.statusCode}`);
  } else if (err instanceof TimeoutError) {
    console.log('Request timed out');
  } else if (err instanceof ConflictError) {
    console.log('Job not in valid state for this operation');
  }
}
```

## API Reference

### `EDCOCRClient`

| Method | Description |
|--------|-------------|
| `healthCheck` | Check API health status |
| `submitJob(options)` | Submit a document for OCR |
| `getStatus(jobId)` | Get current job status |
| `getResult(jobId)` | Get result metadata for completed job |
| `downloadArtifact(jobId, type)` | Download a result artifact |
| `cancelJob(jobId)` | Cancel a job |
| `retryJob(jobId)` | Retry a failed/cancelled job |
| `listJobs(options)` | List jobs with filtering |
| `waitForCompletion(jobId, options)` | Poll until job completes |
| `submitAndWait(options, waitOptions)` | Submit and poll in one call |
| `streamProgress(jobId)` | Open WebSocket for live progress |
| `close` | Release resources |

### Types

See `src/models.ts` for the complete list of TypeScript interfaces and enums.

## Browser Usage

The SDK works in modern browsers with native `fetch` and `FormData`:

```html
<script type="module">
  import { EDCOCRClient } from '@edcocr/sdk';

  const client = new EDCOCRClient({
    baseUrl: 'https://your-ocr-api.example.com',
    apiKey: 'your-api-key',
  });

  // Use fileBuffer instead of filePath in browsers
  const fileInput = document.getElementById('fileInput');
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const buffer = new Uint8Array(await file.arrayBuffer);

    const job = await client.submitJob({
      fileBuffer: buffer,
      filename: file.name,
    });

    const result = await client.waitForCompletion(job.job_id);
    console.log('Done:', result.status);
  });
</script>
```

## License

Apache-2.0
