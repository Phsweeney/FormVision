/**
 * Typed API client.
 *
 * Every network call in the app goes through this module. Centralising it means
 * the base URL, the error envelope, and the media URL construction each have
 * exactly one definition.
 */

import type {
  Analysis,
  AnalyzeResponse,
  ApiErrorBody,
  UploadResponse,
} from "./types";

/**
 * Backend origin.
 *
 * Read from the environment so the same build can point at a local backend or a
 * deployed one. `NEXT_PUBLIC_` is required for the value to reach the browser.
 */
export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * An error carrying the backend's structured code alongside its message.
 *
 * The code lets the UI react to specific failures (an unsupported file type
 * deserves different treatment from a server fault) without matching on
 * message text, which is prose and will change.
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail: Record<string, unknown>;

  constructor(
    message: string,
    code = "UNKNOWN",
    status = 0,
    detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

/** Turn any failed response into an `ApiError`, whatever shape it arrived in. */
async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (body?.error?.message) {
      return new ApiError(
        body.error.message,
        body.error.code,
        response.status,
        body.error.detail ?? {},
      );
    }
  } catch {
    // A proxy timeout or a crash before the handler can return HTML or
    // nothing at all. Fall through to a generic message rather than letting a
    // JSON parse error mask the real status.
  }
  return new ApiError(
    `Request failed with status ${response.status}.`,
    "HTTP_ERROR",
    response.status,
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    // fetch only rejects on network-level failure, which almost always means
    // the backend is not running. Say so, rather than "Failed to fetch".
    throw new ApiError(
      "Could not reach the FormVision API. Check that the backend is running.",
      "NETWORK_ERROR",
    );
  }

  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as T;
}

/** Absolute URL for a media path returned by the API. */
export function mediaUrl(path: string | null): string | null {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
}

/** `GET /analysis/{id}` — status and, once complete, full results. */
export function fetchAnalysis(id: string): Promise<Analysis> {
  // Analysis results change while processing, so a cached response would stall
  // the poll on a stale status.
  return request<Analysis>(`/analysis/${id}`, { cache: "no-store" });
}

/** `POST /analyze` — queues background analysis and returns immediately. */
export function startAnalysis(id: string): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: id }),
  });
}

/**
 * `POST /upload`, reporting progress.
 *
 * Uses `XMLHttpRequest` rather than `fetch` deliberately: `fetch` has no
 * upload-progress event, and a video upload is slow enough that a progress bar
 * is the difference between "working" and "broken" to the person watching.
 */
export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.open("POST", `${API_BASE_URL}/upload`);
    xhr.responseType = "json";

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) {
        // The bytes are transferred, but the server still has to probe the
        // file. Only report 100% once it has actually accepted the upload.
        onProgress?.(100);
        resolve(body as UploadResponse);
        return;
      }
      const error = (body as ApiErrorBody | undefined)?.error;
      reject(
        new ApiError(
          error?.message ?? `Upload failed with status ${xhr.status}.`,
          error?.code ?? "UPLOAD_FAILED",
          xhr.status,
          error?.detail ?? {},
        ),
      );
    };

    xhr.onerror = () =>
      reject(
        new ApiError(
          "Could not reach the FormVision API. Check that the backend is running.",
          "NETWORK_ERROR",
        ),
      );

    xhr.onabort = () =>
      reject(new ApiError("Upload cancelled.", "UPLOAD_CANCELLED"));

    signal?.addEventListener("abort", () => xhr.abort());

    xhr.send(form);
  });
}
