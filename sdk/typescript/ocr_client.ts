/**
 * TypeScript SDK client for EDCOCR API.
 *
 * Provides a typed, ergonomic interface for submitting OCR jobs,
 * polling for results, and downloading processed documents.
 *
 * Requires Node.js 18+ (uses native `fetch` API).
 * No external dependencies.
 *
 * @example
 * ```typescript
 * import { OcrClient } from './ocr_client';
 *
 * const client = new OcrClient({
 *   baseUrl: 'http://localhost:8000',
 *   apiKey: 'my-key',
 * });
 *
 * const job = await client.submit({ filePath: 'document.pdf' });
 * const result = await client.waitForResult(job.jobId);
 * const data = await client.downloadResult(job.jobId);
 * client.close();
 * ```
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

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

/**
 * OCR job information returned by the API.
 */
export interface JobInfo {
  /** Unique job identifier. */
  jobId: string;
  /** Current job status. */
  status: string;
  /** Original filename of the submitted document. */
  filename: string;
  /** Total number of pages in the document. */
  pages: number;
  /** ISO-8601 timestamp when the job was created. */
  createdAt: string;
  /** ISO-8601 timestamp when the job completed (empty if still running). */
  completedAt: string;
  /** Error message if the job failed. */
  error: string;
  /** Processing progress as a percentage (0-100). */
  progress: number;
}

/**
 * API health status information.
 */
export interface HealthInfo {
  /** Overall service status (e.g. "healthy"). */
  status: string;
  /** API version string. */
  version: string;
  /** Number of seconds the service has been running. */
  uptimeSeconds: number;
}

/**
 * Options for submitting an OCR job.
 */
export interface SubmitOptions {
  /** Path to the file on the local filesystem. */
  filePath?: string;
  /** File content as a Buffer or Uint8Array (alternative to filePath). */
  fileBuffer?: Uint8Array;
  /** Filename to use when uploading a buffer. */
  filename?: string;
  /** Enable document intelligence analysis. */
  enableDocintel?: boolean;
  /** URL for completion webhook notification. */
  webhookUrl?: string;
  /** Job priority level. */
  priority?: 'low' | 'normal' | 'high';
}

/**
 * Configuration options for the OcrClient.
 */
export interface ClientOptions {
  /** Root URL of the OCR API (e.g. "http://localhost:8000"). */
  baseUrl: string;
  /** API key for authentication via the X-API-Key header. */
  apiKey?: string;
  /** Default HTTP timeout in milliseconds. Defaults to 30000. */
  timeoutMs?: number;
  /** Number of retry attempts on transient errors. Defaults to 3. */
  maxRetries?: number;
}

/**
 * Options for listing jobs.
 */
export interface ListOptions {
  /** Filter by job status. */
  status?: string;
  /** Maximum number of jobs to return. */
  limit?: number;
  /** Offset for pagination. */
  offset?: number;
}

/**
 * A single output artifact in a job's output manifest.
 */
export interface OutputArtifact {
  /** Type of output (e.g. "ocr_text", "searchable_pdf", "ner"). */
  output_type: string;
  /** Filename of the artifact. */
  filename: string;
  /** Path relative to the job output directory. */
  relative_path: string;
  /** Size of the artifact in bytes. */
  size_bytes: number;
  /** Schema version for this output type. */
  schema_version: string;
  /** MIME type of the artifact (optional). */
  mime_type?: string;
}

/**
 * Output manifest for a completed job.
 */
export interface OutputManifest {
  /** Job identifier. */
  job_id: string;
  /** List of available output artifacts. */
  artifacts: OutputArtifact[];
  /** Map of output_type to schema_version. */
  schema_versions: Record<string, string>;
}

/**
 * Schema list entry returned by the schemas endpoint.
 */
export interface SchemaListItem {
  /** Output type identifier. */
  output_type: string;
  /** Schema version string. */
  schema_version: string;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Base error class for all SDK errors.
 */
export class OcrClientError extends Error {
  /** HTTP status code, or 0 if not an HTTP error. */
  public readonly statusCode: number;
  /** Raw response body text. */
  public readonly responseBody: string;

  constructor(message: string, statusCode: number = 0, responseBody: string = '') {
    super(message);
    this.name = 'OcrClientError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

/**
 * API key is invalid or missing (HTTP 401/403).
 */
export class AuthenticationError extends OcrClientError {
  constructor(message: string, statusCode: number = 401, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'AuthenticationError';
  }
}

/**
 * Requested resource was not found (HTTP 404).
 */
export class NotFoundError extends OcrClientError {
  constructor(message: string, statusCode: number = 404, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'NotFoundError';
  }
}

/**
 * Operation timed out waiting for a result.
 */
export class TimeoutError extends OcrClientError {
  constructor(message: string, statusCode: number = 0, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'TimeoutError';
  }
}

/**
 * Server returned a 5xx error.
 */
export class ServerError extends OcrClientError {
  constructor(message: string, statusCode: number = 500, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'ServerError';
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Parse a raw API response object into a {@link JobInfo}.
 *
 * Handles both canonical and alternate key names so the SDK works even
 * when the server uses slightly different field names.
 */
function parseJobInfo(data: Record<string, unknown>): JobInfo {
  return {
    jobId: (data.job_id ?? data.id ?? '') as string,
    status: (data.status ?? '') as string,
    filename: (data.filename ?? data.original_filename ?? '') as string,
    pages: (data.pages ?? data.total_pages ?? 0) as number,
    createdAt: (data.created_at ?? '') as string,
    completedAt: (data.completed_at ?? '') as string,
    error: (data.error ?? data.error_message ?? '') as string,
    progress: (data.progress ?? 0) as number,
  };
}

/**
 * Parse a raw API response into a {@link HealthInfo}.
 */
function parseHealthInfo(data: Record<string, unknown>): HealthInfo {
  return {
    status: (data.status ?? '') as string,
    version: (data.version ?? '') as string,
    uptimeSeconds: (data.uptime_seconds ?? data.uptime ?? 0) as number,
  };
}

/**
 * Sleep for the given number of milliseconds.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check whether a job status represents a terminal state.
 */
function isTerminalStatus(status: string): boolean {
  return (
    status === JobStatus.COMPLETED ||
    status === JobStatus.FAILED ||
    status === JobStatus.CANCELLED
  );
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * TypeScript SDK client for the EDCOCR API.
 *
 * Uses the native `fetch` API available in Node.js 18+ and modern browsers.
 * All methods are async and return typed results.
 *
 * @example
 * ```typescript
 * const client = new OcrClient({ baseUrl: 'http://localhost:8000', apiKey: 'key' });
 * const health = await client.health();
 * console.log(health.status); // "healthy"
 * ```
 */
export class OcrClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private closed: boolean = false;

  /**
   * Create a new OcrClient instance.
   *
   * @param options - Client configuration options.
   */
  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.apiKey = options.apiKey ?? '';
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.maxRetries = options.maxRetries ?? 3;
  }

  // ---------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------

  /**
   * Check API health status.
   *
   * @returns Health information including status, version, and uptime.
   */
  async health(): Promise<HealthInfo> {
    const data = await this.request<Record<string, unknown>>('GET', '/api/v1/health');
    return parseHealthInfo(data);
  }

  /**
   * Submit a document for OCR processing.
   *
   * Uploads the file as a multipart form and returns job tracking information.
   *
   * @param options - Submission options including file path or buffer.
   * @returns Job information with a jobId for tracking.
   * @throws {OcrClientError} If neither filePath nor fileBuffer is provided.
   */
  async submit(options: SubmitOptions): Promise<JobInfo> {
    if (!options.filePath && !options.fileBuffer) {
      throw new OcrClientError('Either filePath or fileBuffer must be provided');
    }

    const formData = await this.buildSubmitForm(options);
    const data = await this.request<Record<string, unknown>>('POST', '/api/v1/jobs/', {
      body: formData,
      isMultipart: true,
    });
    return parseJobInfo(data);
  }

  /**
   * Get current status and details of a job.
   *
   * @param jobId - The unique job identifier.
   * @returns Current job information.
   * @throws {NotFoundError} If the job does not exist.
   */
  async getJob(jobId: string): Promise<JobInfo> {
    const data = await this.request<Record<string, unknown>>('GET', `/api/v1/jobs/${jobId}`);
    return parseJobInfo(data);
  }

  /**
   * List jobs with optional filtering and pagination.
   *
   * @param options - Filtering and pagination options.
   * @returns Array of job information objects.
   */
  async listJobs(options: ListOptions = {}): Promise<JobInfo[]> {
    const params = new URLSearchParams();
    if (options.status) params.set('status', options.status);
    if (options.limit !== undefined) params.set('limit', String(options.limit));
    if (options.offset !== undefined) params.set('offset', String(options.offset));

    const query = params.toString();
    const path = query ? `/api/v1/jobs/?${query}` : '/api/v1/jobs/';

    const data = await this.request<Record<string, unknown> | unknown[]>('GET', path);

    if (Array.isArray(data)) {
      return data.map((item) => parseJobInfo(item as Record<string, unknown>));
    }

    const jobs =
      (data as Record<string, unknown>).jobs ??
      (data as Record<string, unknown>).items ??
      [];
    return (jobs as unknown[]).map((item) => parseJobInfo(item as Record<string, unknown>));
  }

  /**
   * Cancel a queued or in-progress job.
   *
   * @param jobId - The unique job identifier.
   * @returns `true` if the job was cancelled, `false` if it was not found.
   */
  async cancelJob(jobId: string): Promise<boolean> {
    try {
      await this.request('DELETE', `/api/v1/jobs/${jobId}`);
      return true;
    } catch (err) {
      if (err instanceof NotFoundError) {
        return false;
      }
      throw err;
    }
  }

  /**
   * Download the result of a completed job.
   *
   * @param jobId - The unique job identifier.
   * @returns Raw result data as a Uint8Array.
   * @throws {NotFoundError} If the job or result does not exist.
   */
  async downloadResult(jobId: string): Promise<Uint8Array> {
    const url = `${this.baseUrl}/api/v1/jobs/${jobId}/result`;
    const headers = this.buildHeaders();

    const response = await this.fetchWithTimeout(url, {
      method: 'GET',
      headers,
    });

    await this.checkResponse(response);

    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  }

  /**
   * Get the output manifest for a completed job.
   *
   * @param jobId - The unique job identifier.
   * @returns Output manifest with artifact list and schema versions.
   * @throws {NotFoundError} If the job does not exist.
   */
  async getOutputs(jobId: string): Promise<OutputManifest> {
    const data = await this.request<OutputManifest>('GET', `/api/v1/jobs/${jobId}/outputs`);
    return data;
  }

  /**
   * Download a specific output artifact as raw bytes.
   *
   * @param jobId - The unique job identifier.
   * @param outputType - Output type (e.g. "ocr_text", "searchable_pdf", "ner").
   * @returns Raw artifact data as an ArrayBuffer.
   * @throws {NotFoundError} If the job or output does not exist.
   */
  async getOutput(jobId: string, outputType: string): Promise<ArrayBuffer> {
    const url = `${this.baseUrl}/api/v1/jobs/${jobId}/outputs/${outputType}`;
    const headers = this.buildHeaders();

    const response = await this.fetchWithTimeout(url, {
      method: 'GET',
      headers,
    });

    await this.checkResponse(response);

    return response.arrayBuffer();
  }

  /**
   * Download a JSON output artifact and parse it.
   *
   * For JSON sidecar outputs (entities, ner, extraction, classification,
   * validation, handwriting, signature, vertical, structure).
   *
   * @param jobId - The unique job identifier.
   * @param outputType - Output type that produces JSON content.
   * @returns Parsed JSON object.
   */
  async getOutputJson(jobId: string, outputType: string): Promise<Record<string, unknown>> {
    const buffer = await this.getOutput(jobId, outputType);
    const text = new TextDecoder().decode(buffer);
    return JSON.parse(text) as Record<string, unknown>;
  }

  /**
   * List available output schemas.
   *
   * @returns Array of schema list items with output_type and schema_version.
   */
  async listSchemas(): Promise<SchemaListItem[]> {
    const data = await this.request<Record<string, unknown>>('GET', '/api/v1/schemas');
    return (data.schemas ?? []) as SchemaListItem[];
  }

  /**
   * Get the JSON Schema definition for a specific output type.
   *
   * @param outputType - Output type identifier.
   * @returns JSON Schema definition object.
   */
  async getSchema(outputType: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>('GET', `/api/v1/schemas/${outputType}`);
  }

  /**
   * Poll until a job reaches a terminal state (completed, failed, or cancelled).
   *
   * @param jobId - The unique job identifier.
   * @param pollIntervalMs - Milliseconds between status polls. Defaults to 2000.
   * @param timeoutMs - Maximum milliseconds to wait. Defaults to 600000 (10 minutes).
   * @returns Final job information.
   * @throws {TimeoutError} If the job does not complete within the timeout.
   */
  async waitForResult(
    jobId: string,
    pollIntervalMs: number = 2_000,
    timeoutMs: number = 600_000): Promise<JobInfo> {
    const start = Date.now();

    while (true) {
      const job = await this.getJob(jobId);

      if (isTerminalStatus(job.status)) {
        return job;
      }

      const elapsed = Date.now() - start;
      if (elapsed >= timeoutMs) {
        throw new TimeoutError(
          `Job ${jobId} did not complete within ${timeoutMs}ms (status: ${job.status})`);
      }

      await sleep(pollIntervalMs);
    }
  }

  /**
   * Submit a document and wait for processing to complete.
   *
   * Convenience method combining {@link submit} and {@link waitForResult}.
   *
   * @param options - Submission options.
   * @param pollIntervalMs - Milliseconds between status polls. Defaults to 2000.
   * @param timeoutMs - Maximum milliseconds to wait. Defaults to 600000.
   * @returns Final job information after processing completes.
   */
  async submitAndWait(
    options: SubmitOptions,
    pollIntervalMs: number = 2_000,
    timeoutMs: number = 600_000): Promise<JobInfo> {
    const job = await this.submit(options);
    return this.waitForResult(job.jobId, pollIntervalMs, timeoutMs);
  }

  /**
   * Close the client and release any resources.
   *
   * After calling close(), the client should not be used for further requests.
   */
  close(): void {
    this.closed = true;
  }

  // ---------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------

  /**
   * Build the standard headers for API requests.
   */
  private buildHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      'User-Agent': 'ocr-local-typescript-sdk/4.1.0',
      ...extra,
    };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  /**
   * Build a FormData object for the submit endpoint.
   */
  private async buildSubmitForm(options: SubmitOptions): Promise<FormData> {
    const form = new FormData();

    if (options.filePath) {
      // Node.js 20+ supports File from a path; for broader compat use fs.
      // We dynamically import `fs` so the SDK can also work in browsers
      // if a fileBuffer is provided instead.
      const fs = await import('fs');
      const path = await import('path');
      const content = fs.readFileSync(options.filePath);
      const fname = options.filename ?? path.basename(options.filePath);
      const blob = new Blob([content]);
      form.append('file', blob, fname);
    } else if (options.fileBuffer) {
      const fname = options.filename ?? 'upload';
      const blob = new Blob([options.fileBuffer]);
      form.append('file', blob, fname);
    }

    if (options.enableDocintel) {
      form.append('enable_docintel', 'true');
    }
    if (options.webhookUrl) {
      form.append('webhook_url', options.webhookUrl);
    }
    if (options.priority) {
      form.append('priority', options.priority);
    }

    return form;
  }

  /**
   * Execute a fetch request with an AbortController-based timeout.
   */
  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    timeoutMs?: number): Promise<Response> {
    const ms = timeoutMs ?? this.timeoutMs;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ms);

    try {
      const response = await fetch(url, {
        ...init,
        signal: controller.signal,
      });
      return response;
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new TimeoutError(`Request timed out after ${ms}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Make an HTTP request with automatic retries on transient errors.
   *
   * @param method - HTTP method.
   * @param path - URL path relative to baseUrl.
   * @param options - Optional request body and flags.
   * @returns Parsed JSON response body.
   */
  private async request<T = Record<string, unknown>>(
    method: string,
    path: string,
    options: { body?: BodyInit; isMultipart?: boolean } = {}): Promise<T> {
    if (this.closed) {
      throw new OcrClientError('Client has been closed');
    }

    const url = `${this.baseUrl}${path}`;
    const headers = this.buildHeaders(
      options.isMultipart ? {} : { 'Content-Type': 'application/json' });

    let lastError: unknown;

    for (let attempt = 0; attempt < Math.max(1, this.maxRetries); attempt++) {
      try {
        const response = await this.fetchWithTimeout(url, {
          method,
          headers,
          body: options.body,
        });

        await this.checkResponse(response);

        if (response.status === 204) {
          return {} as T;
        }

        const contentType = response.headers.get('content-type') ?? '';
        if (contentType.includes('application/json')) {
          return (await response.json()) as T;
        }

        return { raw: await response.text() } as unknown as T;
      } catch (err: unknown) {
        lastError = err;

        // Don't retry on client errors (4xx) or known SDK errors
        if (err instanceof AuthenticationError || err instanceof NotFoundError) {
          throw err;
        }

        // Retry on network/transient errors
        const isRetryable =
          err instanceof TypeError || // fetch network error
          err instanceof ServerError;

        if (isRetryable && attempt < this.maxRetries - 1) {
          const backoffMs = Math.min(2 ** attempt * 1000, 10_000);
          await sleep(backoffMs);
          continue;
        }

        // Non-retryable SDK errors should be re-thrown immediately
        if (err instanceof OcrClientError) {
          throw err;
        }
      }
    }

    throw new OcrClientError(
      `Request failed after ${this.maxRetries} retries: ${lastError}`);
  }

  /**
   * Check an HTTP response and throw the appropriate typed error.
   */
  private async checkResponse(response: Response): Promise<void> {
    if (response.status < 400) {
      return;
    }

    let body = '';
    try {
      body = await response.text();
    } catch {
      // ignore read errors
    }

    const status = response.status;

    if (status === 401 || status === 403) {
      throw new AuthenticationError(
        `Authentication failed (HTTP ${status})`,
        status,
        body);
    }

    if (status === 404) {
      throw new NotFoundError('Resource not found (HTTP 404)', 404, body);
    }

    if (status >= 500) {
      throw new ServerError(`Server error (HTTP ${status})`, status, body);
    }

    throw new OcrClientError(`Request failed (HTTP ${status})`, status, body);
  }
}
