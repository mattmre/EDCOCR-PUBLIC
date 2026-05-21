/**
 * Custom error classes for the EDCOCR SDK.
 *
 * All SDK errors extend {@link OCRLocalError} so consumers can catch
 * the entire hierarchy with a single `catch` clause if desired.
 *
 * @packageDocumentation
 */

/**
 * Base error for all EDCOCR SDK errors.
 *
 * Carries the HTTP status code and raw response body when the error
 * originates from an API response.
 */
export class OCRLocalError extends Error {
  /** HTTP status code, or 0 for non-HTTP errors. */
  public readonly statusCode: number;
  /** Raw response body text, if available. */
  public readonly responseBody: string;

  constructor(message: string, statusCode: number = 0, responseBody: string = '') {
    super(message);
    this.name = 'OCRLocalError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
    // Maintain correct prototype chain for instanceof checks.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * API key is invalid, missing, or expired (HTTP 401 or 403).
 */
export class AuthenticationError extends OCRLocalError {
  constructor(message: string, statusCode: number = 401, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'AuthenticationError';
  }
}

/**
 * Requested resource was not found (HTTP 404).
 */
export class NotFoundError extends OCRLocalError {
  constructor(message: string, statusCode: number = 404, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'NotFoundError';
  }
}

/**
 * Client has exceeded the rate limit (HTTP 429).
 */
export class RateLimitError extends OCRLocalError {
  /** Seconds until the rate limit resets, parsed from `Retry-After` header. */
  public readonly retryAfterSeconds: number | null;

  constructor(
    message: string,
    statusCode: number = 429,
    responseBody: string = '',
    retryAfterSeconds: number | null = null) {
    super(message, statusCode, responseBody);
    this.name = 'RateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * Server returned a 5xx error.
 */
export class ServerError extends OCRLocalError {
  constructor(message: string, statusCode: number = 500, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'ServerError';
  }
}

/**
 * Operation timed out (either HTTP request or polling timeout).
 */
export class TimeoutError extends OCRLocalError {
  constructor(message: string) {
    super(message, 0, '');
    this.name = 'TimeoutError';
  }
}

/**
 * Client has been closed and cannot make further requests.
 */
export class ClientClosedError extends OCRLocalError {
  constructor() {
    super('Client has been closed', 0, '');
    this.name = 'ClientClosedError';
  }
}

/**
 * Job is not in a valid state for the requested operation (HTTP 409).
 */
export class ConflictError extends OCRLocalError {
  constructor(message: string, statusCode: number = 409, responseBody: string = '') {
    super(message, statusCode, responseBody);
    this.name = 'ConflictError';
  }
}
