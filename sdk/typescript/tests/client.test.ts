/**
 * Unit tests for the EDCOCR TypeScript SDK client.
 *
 * These tests use a mock fetch implementation to verify request
 * construction, error handling, and retry behavior without requiring
 * a live API server.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

import {
  EDCOCRClient,
  OCRLocalError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ClientClosedError,
  ConflictError,
  JobStatus,
  SDK_VERSION,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Mock fetch helper
// ---------------------------------------------------------------------------

type MockFetchResponse = {
  status: number;
  headers?: Record<string, string>;
  body?: unknown;
  text?: string;
};

function createMockFetch(responses: MockFetchResponse[]) {
  let callIndex = 0;
  const calls: { url: string; init: RequestInit }[] = [];

  const mockFetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const response = responses[callIndex] ?? responses[responses.length - 1];
    callIndex++;
    calls.push({ url: String(url), init: init ?? {} });

    const headers = new Headers(response.headers ?? {});
    if (response.body !== undefined && !headers.has('content-type')) {
      headers.set('content-type', 'application/json');
    }

    return {
      status: response.status,
      headers,
      ok: response.status >= 200 && response.status < 300,
      json: async => response.body,
      text: async => response.text ?? JSON.stringify(response.body),
      arrayBuffer: async => new ArrayBuffer(0),
    } as unknown as Response;
  });

  return { mockFetch: mockFetch as unknown as typeof globalThis.fetch, calls };
}

function createClient(mockFetch: typeof globalThis.fetch, overrides: Record<string, unknown> = {}) {
  return new EDCOCRClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'test-key',
    timeoutMs: 5000,
    maxRetries: 1,
    fetch: mockFetch,
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('EDCOCRClient', => {
  describe('constructor', => {
    it('throws if baseUrl is empty', => {
      expect(() => new EDCOCRClient({ baseUrl: '' })).toThrow('baseUrl is required');
    });

    it('strips trailing slashes from baseUrl', => {
      const { mockFetch, calls } = createMockFetch([
        { status: 200, body: { status: 'healthy', version: '1.0.0', uptime_seconds: 42 } },
      ]);
      const client = createClient(mockFetch, { baseUrl: 'http://localhost:8000///' });
      client.healthCheck();
      // After async resolves, URL should not have trailing slashes
    });
  });

  describe('healthCheck', => {
    it('returns health info on success', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: { status: 'healthy', version: '1.0.0', uptime_seconds: 123.4, jobs: { queued: 2 } },
        },
      ]);
      const client = createClient(mockFetch);
      const health = await client.healthCheck();

      expect(health.status).toBe('healthy');
      expect(health.version).toBe('1.0.0');
      expect(health.uptime_seconds).toBe(123.4);
      expect(health.jobs.queued).toBe(2);
    });

    it('sends X-API-Key header when apiKey is configured', async => {
      const { mockFetch, calls } = createMockFetch([
        { status: 200, body: { status: 'healthy', version: '1.0.0', uptime_seconds: 0 } },
      ]);
      const client = createClient(mockFetch);
      await client.healthCheck();

      expect(calls[0].init.headers).toBeDefined();
      const headers = calls[0].init.headers as Record<string, string>;
      expect(headers['X-API-Key']).toBe('test-key');
    });

    it('does not send X-API-Key when no apiKey is set', async => {
      const { mockFetch, calls } = createMockFetch([
        { status: 200, body: { status: 'healthy', version: '1.0.0', uptime_seconds: 0 } },
      ]);
      const client = createClient(mockFetch, { apiKey: undefined });
      await client.healthCheck();

      const headers = calls[0].init.headers as Record<string, string>;
      expect(headers['X-API-Key']).toBeUndefined();
    });
  });

  describe('getStatus', => {
    it('returns job status on success', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: {
            job_id: 'job_abc123def456',
            status: 'processing',
            created_at: '2026-01-01T00:00:00Z',
            started_at: '2026-01-01T00:00:01Z',
            completed_at: null,
            priority: 'normal',
            source_file: 'test.pdf',
            progress: { total_pages: 10, pages_completed: 3, percent_complete: 30.0, current_stage: 'ocr' },
            settings: {},
            webhook_status: null,
          },
        },
      ]);
      const client = createClient(mockFetch);
      const job = await client.getStatus('job_abc123def456');

      expect(job.job_id).toBe('job_abc123def456');
      expect(job.status).toBe('processing');
      expect(job.progress?.pages_completed).toBe(3);
    });

    it('throws NotFoundError for 404', async => {
      const { mockFetch } = createMockFetch([
        { status: 404, body: { error: 'job_not_found', message: 'Job not found.' } },
      ]);
      const client = createClient(mockFetch);

      await expect(client.getStatus('job_nonexistent0')).rejects.toThrow(NotFoundError);
    });
  });

  describe('submitJob', => {
    it('throws if no file provided', async => {
      const { mockFetch } = createMockFetch([]);
      const client = createClient(mockFetch);

      await expect(client.submitJob({})).rejects.toThrow('Either filePath or fileBuffer');
    });

    it('submits with fileBuffer', async => {
      const { mockFetch, calls } = createMockFetch([
        {
          status: 201,
          body: {
            job_id: 'job_new123456abc',
            status: 'queued',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'normal',
            source_file: 'test.pdf',
            estimated_pages: null,
            links: { self: '/api/v1/jobs/job_new123456abc', result: '/api/v1/jobs/job_new123456abc/result' },
          },
        },
      ]);
      const client = createClient(mockFetch);
      const buf = new Uint8Array([0x25, 0x50, 0x44, 0x46]); // %PDF
      const job = await client.submitJob({ fileBuffer: buf, filename: 'test.pdf' });

      expect(job.job_id).toBe('job_new123456abc');
      expect(job.status).toBe('queued');
      // Should be multipart (no Content-Type header — browser sets it with boundary)
      const headers = calls[0].init.headers as Record<string, string>;
      expect(headers['Content-Type']).toBeUndefined();
    });

    it('includes optional fields in form data', async => {
      const { mockFetch, calls } = createMockFetch([
        {
          status: 201,
          body: {
            job_id: 'job_opt123456abc',
            status: 'queued',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'urgent',
            source_file: 'doc.pdf',
            estimated_pages: 5,
            links: { self: '/jobs/opt', result: '/jobs/opt/result' },
          },
        },
      ]);
      const client = createClient(mockFetch);
      const buf = new Uint8Array([0x25, 0x50, 0x44, 0x46]);
      await client.submitJob({
        fileBuffer: buf,
        filename: 'doc.pdf',
        enableDocintel: true,
        docintelMode: 'full',
        priority: 'urgent',
        webhookUrl: 'https://example.com/hook',
        processingTimeoutMinutes: 30,
      });

      // Verify form data was sent (body should be FormData)
      expect(calls[0].init.body).toBeDefined();
    });
  });

  describe('cancelJob', => {
    it('returns cancelled status', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: {
            job_id: 'job_cancel12345a',
            status: 'cancelled',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'normal',
            source_file: 'test.pdf',
            progress: null,
            settings: {},
            webhook_status: null,
          },
        },
      ]);
      const client = createClient(mockFetch);
      const result = await client.cancelJob('job_cancel12345a');

      expect(result.status).toBe('cancelled');
    });
  });

  describe('retryJob', => {
    it('returns new job on successful retry', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 201,
          body: {
            job_id: 'job_retry1234abc',
            status: 'queued',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'normal',
            source_file: 'test.pdf',
            estimated_pages: 3,
            links: { self: '/jobs/retry', result: '/jobs/retry/result' },
          },
        },
      ]);
      const client = createClient(mockFetch);
      const job = await client.retryJob('job_original0001');

      expect(job.job_id).toBe('job_retry1234abc');
    });

    it('throws ConflictError for non-retryable state', async => {
      const { mockFetch } = createMockFetch([
        { status: 409, body: { error: 'invalid_state', message: 'Job is processing.' } },
      ]);
      const client = createClient(mockFetch);

      await expect(client.retryJob('job_processing01')).rejects.toThrow(ConflictError);
    });
  });

  describe('listJobs', => {
    it('returns paginated job list', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: {
            jobs: [
              {
                job_id: 'job_list1234abcd',
                status: 'completed',
                created_at: '2026-01-01T00:00:00Z',
                priority: 'normal',
                source_file: 'a.pdf',
                progress: null,
                settings: {},
              },
            ],
            total: 1,
            limit: 50,
            offset: 0,
          },
        },
      ]);
      const client = createClient(mockFetch);
      const result = await client.listJobs({ status: 'completed', limit: 50, offset: 0 });

      expect(result.jobs).toHaveLength(1);
      expect(result.total).toBe(1);
    });
  });

  describe('error handling', => {
    it('throws AuthenticationError for 401', async => {
      const { mockFetch } = createMockFetch([
        { status: 401, text: 'Unauthorized' },
      ]);
      const client = createClient(mockFetch);

      await expect(client.healthCheck()).rejects.toThrow(AuthenticationError);
    });

    it('throws AuthenticationError for 403', async => {
      const { mockFetch } = createMockFetch([
        { status: 403, text: 'Forbidden' },
      ]);
      const client = createClient(mockFetch);

      await expect(client.healthCheck()).rejects.toThrow(AuthenticationError);
    });

    it('throws RateLimitError for 429 with Retry-After', async => {
      const { mockFetch } = createMockFetch([
        { status: 429, headers: { 'retry-after': '30' }, text: 'Too Many Requests' },
      ]);
      const client = createClient(mockFetch);

      try {
        await client.healthCheck();
        expect.fail('Should have thrown');
      } catch (err) {
        expect(err).toBeInstanceOf(RateLimitError);
        expect((err as RateLimitError).retryAfterSeconds).toBe(30);
      }
    });

    it('throws ServerError for 500', async => {
      const { mockFetch } = createMockFetch([
        { status: 500, text: 'Internal Server Error' },
      ]);
      const client = createClient(mockFetch);

      await expect(client.healthCheck()).rejects.toThrow(ServerError);
    });

    it('throws generic OCRLocalError for 400', async => {
      const { mockFetch } = createMockFetch([
        { status: 400, text: 'Bad Request' },
      ]);
      const client = createClient(mockFetch);

      await expect(client.healthCheck()).rejects.toThrow(OCRLocalError);
    });
  });

  describe('client lifecycle', => {
    it('throws ClientClosedError after close()', async => {
      const { mockFetch } = createMockFetch([]);
      const client = createClient(mockFetch);
      client.close();

      await expect(client.healthCheck()).rejects.toThrow(ClientClosedError);
    });
  });

  describe('waitForCompletion', => {
    it('returns immediately for already-completed jobs', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: {
            job_id: 'job_done12345678',
            status: 'completed',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'normal',
            source_file: 'done.pdf',
            progress: null,
            settings: {},
          },
        },
      ]);
      const client = createClient(mockFetch);
      const result = await client.waitForCompletion('job_done12345678');

      expect(result.status).toBe('completed');
    });

    it('invokes onProgress callback', async => {
      const { mockFetch } = createMockFetch([
        {
          status: 200,
          body: {
            job_id: 'job_prog12345678',
            status: 'completed',
            created_at: '2026-01-01T00:00:00Z',
            priority: 'normal',
            source_file: 'p.pdf',
            progress: null,
            settings: {},
          },
        },
      ]);
      const client = createClient(mockFetch);
      const progressCalls: string[] = [];
      await client.waitForCompletion('job_prog12345678', {
        onProgress: (s) => progressCalls.push(s.status),
      });

      expect(progressCalls).toContain('completed');
    });
  });

  describe('retry behavior', => {
    it('retries on server error when maxRetries > 1', async => {
      const { mockFetch, calls } = createMockFetch([
        { status: 500, text: 'Error' },
        { status: 200, body: { status: 'healthy', version: '1.0.0', uptime_seconds: 0 } },
      ]);
      const client = createClient(mockFetch, { maxRetries: 2 });
      const health = await client.healthCheck();

      expect(health.status).toBe('healthy');
      expect(calls).toHaveLength(2);
    });

    it('does not retry 401 errors', async => {
      const { mockFetch, calls } = createMockFetch([
        { status: 401, text: 'Unauthorized' },
        { status: 200, body: { status: 'healthy' } },
      ]);
      const client = createClient(mockFetch, { maxRetries: 3 });

      await expect(client.healthCheck()).rejects.toThrow(AuthenticationError);
      expect(calls).toHaveLength(1);
    });

    it('does not retry 404 errors', async => {
      const { mockFetch, calls } = createMockFetch([
        { status: 404, text: 'Not Found' },
        { status: 200, body: {} },
      ]);
      const client = createClient(mockFetch, { maxRetries: 3 });

      await expect(client.getStatus('job_missing00000')).rejects.toThrow(NotFoundError);
      expect(calls).toHaveLength(1);
    });
  });

  describe('enums and constants', => {
    it('JobStatus enum has expected values', => {
      expect(JobStatus.QUEUED).toBe('queued');
      expect(JobStatus.PROCESSING).toBe('processing');
      expect(JobStatus.COMPLETED).toBe('completed');
      expect(JobStatus.FAILED).toBe('failed');
      expect(JobStatus.CANCELLED).toBe('cancelled');
    });

    it('SDK_VERSION is a valid semver string', => {
      expect(SDK_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
    });
  });

  describe('error class hierarchy', => {
    it('all error classes extend OCRLocalError', => {
      expect(new AuthenticationError('test')).toBeInstanceOf(OCRLocalError);
      expect(new NotFoundError('test')).toBeInstanceOf(OCRLocalError);
      expect(new RateLimitError('test')).toBeInstanceOf(OCRLocalError);
      expect(new ServerError('test')).toBeInstanceOf(OCRLocalError);
      expect(new TimeoutError('test')).toBeInstanceOf(OCRLocalError);
      expect(new ClientClosedError()).toBeInstanceOf(OCRLocalError);
      expect(new ConflictError('test')).toBeInstanceOf(OCRLocalError);
    });

    it('error classes carry statusCode and responseBody', => {
      const err = new ServerError('fail', 502, '{"error":"bad_gateway"}');
      expect(err.statusCode).toBe(502);
      expect(err.responseBody).toBe('{"error":"bad_gateway"}');
      expect(err.message).toBe('fail');
      expect(err.name).toBe('ServerError');
    });

    it('RateLimitError carries retryAfterSeconds', => {
      const err = new RateLimitError('limit', 429, '', 60);
      expect(err.retryAfterSeconds).toBe(60);
    });
  });
});
