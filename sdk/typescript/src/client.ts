/**
 * EDCOCR SDK client for TypeScript / JavaScript.
 *
 * Provides a typed, ergonomic interface for submitting OCR jobs,
 * polling for results, streaming progress via WebSocket, and
 * downloading processed documents.
 *
 * Uses the native `fetch` API available in Node.js 18+ and modern
 * browsers. Zero runtime dependencies.
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
 * const job = await client.submitJob({ filePath: 'document.pdf' });
 * const result = await client.waitForCompletion(job.job_id);
 * const pdf = await client.downloadArtifact(job.job_id, 'pdf');
 * client.close();
 * ```
 *
 * @packageDocumentation
 */

import {
  AuthenticationError,
  ClientClosedError,
  ConflictError,
  NotFoundError,
  OCRLocalError,
  RateLimitError,
  ServerError,
  TimeoutError,
} from './errors.js';

import type {
  ClientConfig,
  HealthResponse,
  JobListResponse,
  JobResultResponse,
  JobStatusResponse,
  JobSubmitResponse,
  ListJobsOptions,
  SubmitOptions,
  WaitOptions,
  WebSocketMessage,
} from './models.js';

/** SDK version string. */
export const SDK_VERSION = '4.1.0';

/** Default User-Agent header value. */
const USER_AGENT = `ocr-local-typescript-sdk/${SDK_VERSION}`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTerminalStatus(status: string): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * Main SDK client for the EDCOCR REST API.
 *
 * All methods are async and return strongly-typed results.
 * The client uses the native `fetch` API and has zero runtime dependencies.
 */
export class EDCOCRClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly _fetch: typeof globalThis.fetch;
  private closed = false;

  /**
   * Create a new client instance.
   *
   * @param config - Client configuration options.
   */
  constructor(config: ClientConfig) {
    if (!config.baseUrl) {
      throw new OCRLocalError('baseUrl is required');
    }
    this.baseUrl = config.baseUrl.replace(/\/+$/, '');
    this.apiKey = config.apiKey ?? '';
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.maxRetries = config.maxRetries ?? 3;
    this._fetch = config.fetch ?? globalThis.fetch;
  }

  // ---------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------

  /**
   * Check API health status.
   *
   * @returns Health information including status, version, and uptime.
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/api/v1/health');
  }

  /**
   * Submit a document for OCR processing.
   *
   * Uploads the file as a multipart form and returns job tracking information.
   *
   * @param options - Submission options including file path or buffer.
   * @returns Job submission response with a `job_id` for tracking.
   * @throws {@link OCRLocalError} If neither `filePath` nor `fileBuffer` is provided.
   */
  async submitJob(options: SubmitOptions): Promise<JobSubmitResponse> {
    if (!options.filePath && !options.fileBuffer) {
      throw new OCRLocalError('Either filePath or fileBuffer must be provided');
    }

    const formData = await this.buildSubmitForm(options);
    return this.request<JobSubmitResponse>('POST', '/api/v1/jobs', {
      body: formData,
      isMultipart: true,
    });
  }

  /**
   * Get current status and details of a job.
   *
   * @param jobId - The unique job identifier.
   * @returns Current job status information.
   * @throws {@link NotFoundError} If the job does not exist.
   */
  async getStatus(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>('GET', `/api/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  /**
   * Get result metadata and artifact links for a completed job.
   *
   * @param jobId - The unique job identifier.
   * @returns Result metadata with artifact download URLs.
   * @throws {@link NotFoundError} If the job does not exist.
   * @throws {@link ConflictError} If the job has not completed yet.
   */
  async getResult(jobId: string): Promise<JobResultResponse> {
    return this.request<JobResultResponse>(
      'GET',
      `/api/v1/jobs/${encodeURIComponent(jobId)}/result`);
  }

  /**
   * Download a specific artifact from a completed job.
   *
   * @param jobId - The unique job identifier.
   * @param artifactType - Type of artifact to download (`pdf`, `text`, `structure`).
   * @returns Raw artifact content as a `Uint8Array`.
   * @throws {@link NotFoundError} If the job or artifact does not exist.
   */
  async downloadArtifact(jobId: string, artifactType: string = 'pdf'): Promise<Uint8Array> {
    const url = `${this.baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/result/download?type=${encodeURIComponent(artifactType)}`;
    const headers = this.buildHeaders();

    const response = await this.fetchWithTimeout(url, { method: 'GET', headers });
    await this.checkResponse(response);

    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  }

  /**
   * Cancel a queued or in-progress job.
   *
   * @param jobId - The unique job identifier.
   * @returns Updated job status after cancellation.
   * @throws {@link NotFoundError} If the job does not exist.
   */
  async cancelJob(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(
      'DELETE',
      `/api/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  /**
   * Retry a failed or cancelled job.
   *
   * @param jobId - The unique job identifier of the original job.
   * @returns New job submission response.
   * @throws {@link NotFoundError} If the original job does not exist.
   * @throws {@link ConflictError} If the job is not in a retryable state.
   */
  async retryJob(jobId: string): Promise<JobSubmitResponse> {
    return this.request<JobSubmitResponse>(
      'POST',
      `/api/v1/jobs/${encodeURIComponent(jobId)}/retry`);
  }

  /**
   * List jobs with optional filtering and pagination.
   *
   * @param options - Filtering and pagination options.
   * @returns Paginated list of jobs.
   */
  async listJobs(options: ListJobsOptions = {}): Promise<JobListResponse> {
    const params = new URLSearchParams();
    if (options.status) params.set('status', options.status);
    if (options.limit !== undefined) params.set('limit', String(options.limit));
    if (options.offset !== undefined) params.set('offset', String(options.offset));
    // Backward-compatible aliases (deprecated)
    if (options.page !== undefined) params.set('page', String(options.page));
    if (options.perPage !== undefined) params.set('per_page', String(options.perPage));
    if (options.batchId) params.set('batch_id', options.batchId);

    const query = params.toString();
    const path = query ? `/api/v1/jobs?${query}` : '/api/v1/jobs';

    return this.request<JobListResponse>('GET', path);
  }

  /**
   * Poll until a job reaches a terminal state (completed, failed, or cancelled).
   *
   * @param jobId - The unique job identifier.
   * @param options - Polling configuration.
   * @returns Final job status.
   * @throws {@link TimeoutError} If the job does not complete within the timeout.
   */
  async waitForCompletion(
    jobId: string,
    options: WaitOptions = {}): Promise<JobStatusResponse> {
    const pollInterval = options.pollIntervalMs ?? 2_000;
    const timeout = options.timeoutMs ?? 600_000;
    const start = Date.now();

    while (true) {
      const status = await this.getStatus(jobId);

      if (options.onProgress) {
        options.onProgress(status);
      }

      if (isTerminalStatus(status.status)) {
        return status;
      }

      const elapsed = Date.now() - start;
      if (elapsed >= timeout) {
        throw new TimeoutError(
          `Job ${jobId} did not complete within ${timeout}ms (status: ${status.status})`);
      }

      await sleep(pollInterval);
    }
  }

  /**
   * Submit a document and wait for processing to complete.
   *
   * Convenience method combining {@link submitJob} and {@link waitForCompletion}.
   *
   * @param submitOptions - Submission options.
   * @param waitOptions - Polling configuration.
   * @returns Final job status after processing completes.
   */
  async submitAndWait(
    submitOptions: SubmitOptions,
    waitOptions: WaitOptions = {}): Promise<JobStatusResponse> {
    const job = await this.submitJob(submitOptions);
    return this.waitForCompletion(job.job_id, waitOptions);
  }

  /**
   * Open a WebSocket connection to stream real-time progress for a job.
   *
   * Returns a standard `WebSocket` instance. Callers are responsible for
   * closing the connection when done.
   *
   * @param jobId - The unique job identifier.
   * @returns A WebSocket connected to the job progress endpoint.
   *
   * @example
   * ```typescript
   * const ws = client.streamProgress('job_abc123');
   * ws.onmessage = (event) => {
   *   const msg: WebSocketMessage = JSON.parse(event.data);
   *   console.log(msg.type, msg);
   * };
   * ```
   */
  streamProgress(jobId: string): WebSocket {
    if (this.closed) {
      throw new ClientClosedError();
    }

    const wsUrl = this.baseUrl.replace(/^http/, 'ws');
    let url = `${wsUrl}/ws/jobs/${encodeURIComponent(jobId)}`;

    // WebSocket does not support custom headers in browsers, so pass token
    // as query parameter if an API key is configured.
    if (this.apiKey) {
      url += `?token=${encodeURIComponent(this.apiKey)}`;
    }

    return new WebSocket(url);
  }

  /**
   * Close the client and release any resources.
   *
   * After calling `close()`, the client will reject all further requests.
   */
  close(): void {
    this.closed = true;
  }

  // ---------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------

  /**
   * Build standard HTTP headers for API requests.
   */
  private buildHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      'User-Agent': USER_AGENT,
      ...extra,
    };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  /**
   * Build a `FormData` object for the submit endpoint.
   */
  private async buildSubmitForm(options: SubmitOptions): Promise<FormData> {
    const form = new FormData();

    if (options.filePath) {
      // Node.js only: dynamically import `fs` and `path` so the SDK
      // also works in browsers when fileBuffer is used instead.
      const fs = await import('fs');
      const path = await import('path');
      const content = fs.readFileSync(options.filePath);
      const fname = options.filename ?? path.basename(options.filePath);
      const blob = new Blob([content as BlobPart]);
      form.append('file', blob, fname);
    } else if (options.fileBuffer) {
      const fname = options.filename ?? 'upload';
      const blob = new Blob([options.fileBuffer as BlobPart]);
      form.append('file', blob, fname);
    }

    if (options.enableDocintel) {
      form.append('enable_docintel', 'true');
    }
    if (options.docintelMode) {
      form.append('docintel_mode', options.docintelMode);
    }
    if (options.priority) {
      form.append('priority', options.priority);
    }
    if (options.skipOcr) {
      form.append('skip_ocr', 'true');
    }
    if (options.processingTimeoutMinutes !== undefined) {
      form.append('processing_timeout_minutes', String(options.processingTimeoutMinutes));
    }
    if (options.webhookUrl) {
      form.append('webhook_url', options.webhookUrl);
    }
    if (options.webhookSecret) {
      form.append('webhook_secret', options.webhookSecret);
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
      return await this._fetch(url, {
        ...init,
        signal: controller.signal,
      });
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
   */
  private async request<T>(
    method: string,
    path: string,
    options: { body?: BodyInit; isMultipart?: boolean } = {}): Promise<T> {
    if (this.closed) {
      throw new ClientClosedError();
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

        // Non-retryable errors: re-throw immediately
        if (
          err instanceof AuthenticationError ||
          err instanceof NotFoundError ||
          err instanceof ConflictError ||
          err instanceof ClientClosedError
        ) {
          throw err;
        }

        // Rate limit: respect Retry-After if present
        if (err instanceof RateLimitError && err.retryAfterSeconds && attempt < this.maxRetries - 1) {
          await sleep(err.retryAfterSeconds * 1000);
          continue;
        }

        // Retry on network/server errors
        const isRetryable =
          err instanceof TypeError || // fetch network error
          err instanceof ServerError;

        if (isRetryable && attempt < this.maxRetries - 1) {
          const backoffMs = Math.min(2 ** attempt * 1000, 10_000);
          await sleep(backoffMs);
          continue;
        }

        // Non-retryable SDK errors
        if (err instanceof OCRLocalError) {
          throw err;
        }
      }
    }

    throw new OCRLocalError(
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
      // Ignore read errors
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

    if (status === 409) {
      throw new ConflictError(
        `Conflict (HTTP 409)`,
        409,
        body);
    }

    if (status === 429) {
      const retryAfter = response.headers.get('retry-after');
      const retrySeconds = retryAfter ? parseInt(retryAfter, 10) : null;
      throw new RateLimitError(
        `Rate limit exceeded (HTTP 429)`,
        429,
        body,
        Number.isNaN(retrySeconds) ? null : retrySeconds);
    }

    if (status >= 500) {
      throw new ServerError(`Server error (HTTP ${status})`, status, body);
    }

    throw new OCRLocalError(`Request failed (HTTP ${status})`, status, body);
  }
}
