/**
 * TypeScript interfaces for the EDCOCR API.
 *
 * These types mirror the Pydantic models defined in `api/models.py`
 * and provide compile-time safety for SDK consumers.
 *
 * @packageDocumentation
 */

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/**
 * Possible states of an OCR job.
 */
export enum JobStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Job priority levels accepted by the API.
 */
export type JobPriority = 'urgent' | 'normal' | 'low';

/**
 * Document intelligence analysis modes.
 */
export type DocintelMode = 'layout_only' | 'tables_only' | 'full';

// ---------------------------------------------------------------------------
// API response shapes
// ---------------------------------------------------------------------------

/**
 * Progress information for an in-flight job.
 */
export interface JobProgress {
  /** Total number of pages in the document. */
  total_pages: number;
  /** Number of pages processed so far. */
  pages_completed: number;
  /** Processing progress as a percentage (0-100). */
  percent_complete: number;
  /** Current processing stage name. */
  current_stage: string;
}

/**
 * HATEOAS links returned alongside job responses.
 */
export interface JobLinks {
  /** URL to the job status endpoint. */
  self: string;
  /** URL to the job result endpoint. */
  result: string;
}

/**
 * Response from the job submission endpoint.
 */
export interface JobSubmitResponse {
  /** Unique job identifier (e.g. `job_a1b2c3d4e5f6`). */
  job_id: string;
  /** Initial job status (typically `queued`). */
  status: string;
  /** ISO-8601 timestamp when the job was created. */
  created_at: string;
  /** Priority level assigned to the job. */
  priority: string;
  /** Name of the source file. */
  source_file: string;
  /** Estimated page count (may be null if unknown). */
  estimated_pages: number | null;
  /** HATEOAS navigation links. */
  links: JobLinks;
}

/**
 * Full status response for a single job.
 */
export interface JobStatusResponse {
  /** Unique job identifier. */
  job_id: string;
  /** Current job status. */
  status: string;
  /** ISO-8601 timestamp when the job was created. */
  created_at: string;
  /** ISO-8601 timestamp when processing started (null if not started). */
  started_at: string | null;
  /** ISO-8601 timestamp when the job completed (null if still running). */
  completed_at: string | null;
  /** Priority level. */
  priority: string;
  /** Name of the source file. */
  source_file: string;
  /** Processing progress details. */
  progress: JobProgress | null;
  /** Job settings/configuration. */
  settings: Record<string, unknown>;
  /** Webhook delivery status (null if no webhook configured). */
  webhook_status: string | null;
}

/**
 * Paginated list of jobs.
 */
export interface JobListResponse {
  /** Array of job status objects. */
  jobs: JobStatusResponse[];
  /** Total number of matching jobs. */
  total: number;
  /** Maximum number of results returned. */
  limit: number;
  /** Number of results skipped. */
  offset: number;
  /** Current page number (deprecated -- use offset). */
  page?: number;
  /** Number of jobs per page (deprecated -- use limit). */
  per_page?: number;
}

/**
 * Result metadata for a completed job.
 */
export interface JobResultResponse {
  /** Unique job identifier. */
  job_id: string;
  /** Final job status. */
  status: string;
  /** ISO-8601 timestamp when the job completed. */
  completed_at: string | null;
  /** Total processing time in seconds. */
  processing_time_seconds: number | null;
  /** Map of artifact type to download URL. */
  artifacts: Record<string, string>;
  /** Additional metadata (e.g. pages_processed). */
  metadata: Record<string, unknown>;
}

/**
 * Health check response.
 */
export interface HealthResponse {
  /** Overall service status (e.g. `healthy`). */
  status: string;
  /** API version string. */
  version: string;
  /** Uptime in seconds. */
  uptime_seconds: number;
  /** Job counts by status. */
  jobs: Record<string, number>;
}

/**
 * API error response body.
 */
export interface ErrorResponse {
  /** Error code/type. */
  error: string;
  /** Human-readable error message. */
  message: string;
  /** Additional error details. */
  details: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/**
 * Configuration options for the EDCOCR SDK client.
 */
export interface ClientConfig {
  /** Root URL of the OCR API (e.g. `http://localhost:8000`). */
  baseUrl: string;
  /** API key for authentication via the `X-API-Key` header. */
  apiKey?: string;
  /** Default HTTP timeout in milliseconds. Defaults to 30000. */
  timeoutMs?: number;
  /** Number of retry attempts on transient errors. Defaults to 3. */
  maxRetries?: number;
  /**
   * Custom fetch implementation.
   * Defaults to the global `fetch`. Useful for testing or polyfills.
   */
  fetch?: typeof globalThis.fetch;
}

/**
 * Options for submitting a document for OCR processing.
 */
export interface SubmitOptions {
  /** Path to the file on the local filesystem (Node.js only). */
  filePath?: string;
  /** File content as a `Uint8Array` (works in both Node.js and browsers). */
  fileBuffer?: Uint8Array;
  /** Filename to use when uploading a buffer. */
  filename?: string;
  /** Enable document intelligence analysis. */
  enableDocintel?: boolean;
  /** Document intelligence mode. */
  docintelMode?: DocintelMode;
  /** Job priority level. */
  priority?: JobPriority;
  /** Skip the primary OCR engine step. */
  skipOcr?: boolean;
  /** Per-job processing timeout in minutes. */
  processingTimeoutMinutes?: number;
  /** HTTPS URL for completion webhook notification. */
  webhookUrl?: string;
  /** HMAC secret for signing webhook payloads. */
  webhookSecret?: string;
}

/**
 * Options for listing jobs.
 */
export interface ListJobsOptions {
  /** Filter by job status. */
  status?: string;
  /** Maximum number of results to return (1-200). */
  limit?: number;
  /** Number of results to skip. */
  offset?: number;
  /** Page number (deprecated -- use offset). */
  page?: number;
  /** Number of results per page (deprecated -- use limit). */
  perPage?: number;
  /** Filter by batch ID. */
  batchId?: string;
}

/**
 * Options for polling until a job completes.
 */
export interface WaitOptions {
  /** Milliseconds between status polls. Defaults to 2000. */
  pollIntervalMs?: number;
  /** Maximum milliseconds to wait. Defaults to 600000 (10 minutes). */
  timeoutMs?: number;
  /** Optional callback invoked on each status poll. */
  onProgress?: (status: JobStatusResponse) => void;
}

// ---------------------------------------------------------------------------
// WebSocket message types
// ---------------------------------------------------------------------------

/**
 * Union of all WebSocket message types sent by the server.
 */
export type WebSocketMessage =
  | { type: 'connected'; job_id: string; status: string }
  | { type: 'progress'; job_id: string; status: string }
  | { type: 'completed'; job_id: string; status: 'completed'; output_path?: string }
  | { type: 'failed'; job_id: string; status: 'failed'; error?: string }
  | { type: 'cancelled'; job_id: string; status: 'cancelled' }
  | { type: 'error'; message: string }
  | { type: 'pong' };
