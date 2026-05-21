/**
 * @edcocr/sdk - TypeScript SDK for EDCOCR forensic-grade OCR pipeline.
 *
 * Zero runtime dependencies. Requires Node.js 18+ or a modern browser
 * with native `fetch` and `FormData` support.
 *
 * @example
 * ```typescript
 * import { EDCOCRClient } from '@edcocr/sdk';
 *
 * const client = new EDCOCRClient({
 *   baseUrl: 'http://localhost:8000',
 *   apiKey: 'my-api-key',
 * });
 *
 * // Submit and wait for completion
 * const status = await client.submitAndWait({
 *   fileBuffer: myPdfBytes,
 *   filename: 'document.pdf',
 * });
 *
 * // Download the searchable PDF
 * const pdf = await client.downloadArtifact(status.job_id, 'pdf');
 *
 * client.close();
 * ```
 *
 * @packageDocumentation
 */

// Client
export { EDCOCRClient, SDK_VERSION } from './client.js';

// Errors
export {
  OCRLocalError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ClientClosedError,
  ConflictError,
} from './errors.js';

// Models & types
export {
  JobStatus,
  type JobPriority,
  type DocintelMode,
  type JobProgress,
  type JobLinks,
  type JobSubmitResponse,
  type JobStatusResponse,
  type JobListResponse,
  type JobResultResponse,
  type HealthResponse,
  type ErrorResponse,
  type ClientConfig,
  type SubmitOptions,
  type ListJobsOptions,
  type WaitOptions,
  type WebSocketMessage,
} from './models.js';
