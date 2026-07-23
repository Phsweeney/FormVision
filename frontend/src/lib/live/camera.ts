/**
 * Webcam access.
 *
 * A thin wrapper over `getUserMedia` so the rest of the live code never touches
 * the raw media API, and so the two failure modes that actually matter —
 * permission denied and no camera present — surface as clear, typed errors
 * rather than a raw DOMException.
 */

export class CameraError extends Error {
  constructor(
    message: string,
    readonly kind: "denied" | "not-found" | "unavailable",
  ) {
    super(message);
    this.name = "CameraError";
  }
}

/** Default constraints: a front-facing 720p feed, no audio. */
const DEFAULT_CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: "user",
  },
  audio: false,
};

/**
 * Request the webcam and attach it to a `<video>` element.
 *
 * Resolves once the video has real dimensions (so callers can size a canvas to
 * it immediately). The returned stream must be passed to `stopCamera` when done.
 */
export async function startCamera(
  video: HTMLVideoElement,
  constraints: MediaStreamConstraints = DEFAULT_CONSTRAINTS,
): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraError(
      "This browser does not support webcam capture.",
      "unavailable",
    );
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    throw toCameraError(error);
  }

  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  await waitForDimensions(video);
  return stream;
}

/** Stop every track and detach the stream. Safe to call with null. */
export function stopCamera(
  stream: MediaStream | null,
  video?: HTMLVideoElement | null,
): void {
  stream?.getTracks().forEach((track) => track.stop());
  if (video) video.srcObject = null;
}

function toCameraError(error: unknown): CameraError {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return new CameraError(
      "Camera permission was denied. Allow camera access and try again.",
      "denied",
    );
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return new CameraError("No camera was found on this device.", "not-found");
  }
  return new CameraError(
    "The camera could not be started. It may be in use by another app.",
    "unavailable",
  );
}

/** Resolve once the element reports non-zero intrinsic dimensions. */
function waitForDimensions(video: HTMLVideoElement): Promise<void> {
  if (video.videoWidth > 0 && video.videoHeight > 0) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("loadedmetadata", done);
      resolve();
    };
    video.addEventListener("loadedmetadata", done);
  });
}
