"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { FramePose } from "@/lib/analysis/types";
import { CameraError, startCamera, stopCamera } from "@/lib/live/camera";
import { PoseRunner } from "@/lib/live/pose-runner";
import { drawSkeleton, type SkeletonThresholds } from "@/lib/live/skeleton";
import { Button } from "@/components/ui/button";

/**
 * The camera stage: a mirrored webcam feed with a live skeleton drawn on a
 * canvas above it.
 *
 * This owns the frame loop — camera in, pose out, skeleton drawn — and nothing
 * else. Analysis (rep counting, metrics, coaching) attaches through the
 * `onFrame` callback, which fires once per processed frame with a
 * framework-free `FramePose`. Keeping the loop here and the analysis elsewhere
 * is what lets later milestones grow without touching pose plumbing.
 */

// Drawing thresholds. `confidentVisibility` mirrors the backend's
// `landmark_visibility_threshold`; a later milestone feeds it the value fetched
// from GET /config so the skeleton and the analysis agree on "confident".
const SKELETON_THRESHOLDS: SkeletonThresholds = {
  confidentVisibility: 0.5,
  drawVisibility: 0.1,
};

type Status = "idle" | "starting" | "running" | "error";

export interface LiveStageProps {
  /** Fires once per processed frame. The analysis engine plugs in here. */
  onFrame?: (frame: FramePose) => void;
  /** Fires when the session starts or stops, so a parent can reset state. */
  onRunningChange?: (running: boolean) => void;
}

export function LiveStage({ onFrame, onRunningChange }: LiveStageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const runnerRef = useRef<PoseRunner | null>(null);
  const runningRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  // Which scheduler produced the current handle, so stop() cancels correctly.
  const usingRvfcRef = useRef(false);
  const lastFrameMsRef = useRef<number>(0);
  const onFrameRef = useRef(onFrame);
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [tracking, setTracking] = useState<boolean>(false);

  const stop = useCallback(() => {
    runningRef.current = false;
    const video = videoRef.current;
    if (rafRef.current !== null) {
      if (usingRvfcRef.current && video?.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(rafRef.current);
      } else {
        cancelAnimationFrame(rafRef.current);
      }
    }
    rafRef.current = null;
    runnerRef.current?.close();
    runnerRef.current = null;
    stopCamera(streamRef.current, videoRef.current);
    streamRef.current = null;

    // Wipe the overlay. Stopping the loop leaves the last frame's skeleton
    // painted on the canvas, and the "camera is off" panel is only partly
    // opaque, so it stayed floating over the blank stage after the session
    // ended. The canvas is a separate element from the video and nothing else
    // clears it.
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

    setStatus("idle");
    setTracking(false);
    setFps(0);
    onRunningChange?.(false);
  }, [onRunningChange]);

  // Always release the camera when the component unmounts.
  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    setStatus("starting");
    setError(null);
    try {
      streamRef.current = await startCamera(video);
      // Size the canvas to the true frame so normalised landmarks map 1:1.
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const runner = new PoseRunner();
      await runner.load();
      runnerRef.current = runner;

      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas 2D context unavailable.");

      runningRef.current = true;
      setStatus("running");
      onRunningChange?.(true);

      // The frame loop lives inline so it closes over this session's video,
      // canvas, context, and runner without threading them through more hooks.
      const step = () => {
        if (!runningRef.current) return;
        const frame = runner.detect(video, performance.now());
        drawSkeleton(ctx, frame, canvas.width, canvas.height, SKELETON_THRESHOLDS);
        setTracking(frame.detected);
        onFrameRef.current?.(frame);

        const now = performance.now();
        const delta = now - lastFrameMsRef.current;
        lastFrameMsRef.current = now;
        if (delta > 0) {
          // Exponential moving average keeps the readout from flickering.
          setFps((prev) =>
            prev === 0 ? 1000 / delta : prev * 0.85 + (1000 / delta) * 0.15,
          );
        }
        schedule();
      };
      const supportsRvfc = typeof video.requestVideoFrameCallback === "function";
      usingRvfcRef.current = supportsRvfc;
      const schedule = () => {
        rafRef.current = supportsRvfc
          ? video.requestVideoFrameCallback(step)
          : requestAnimationFrame(step);
      };
      schedule();
    } catch (err) {
      const message =
        err instanceof CameraError
          ? err.message
          : "Live mode failed to start. See the console for details.";
      setError(message);
      setStatus("error");
      stop();
    }
  }, [onRunningChange, stop]);

  const running = status === "running";

  return (
    <div className="space-y-3">
      <div
        className="border-border/60 bg-card/40 relative w-full overflow-hidden rounded-xl border"
        style={{ aspectRatio: "16 / 9" }}
      >
        {/* Mirrored so it reads as a selfie view; the canvas is mirrored the
            same way, so the skeleton stays aligned with the video. */}
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover"
        />

        {status !== "running" && (
          <div className="absolute inset-0 grid place-items-center bg-black/40">
            {status === "starting" ? (
              <p className="text-sm text-white/80">Starting camera…</p>
            ) : status === "error" ? (
              <p className="max-w-sm px-6 text-center text-sm text-red-300">
                {error}
              </p>
            ) : (
              <p className="text-muted-foreground text-sm">
                Camera is off.
              </p>
            )}
          </div>
        )}

        {running && (
          <div className="absolute left-3 top-3 flex items-center gap-2">
            <StatusPill
              tone={tracking ? "good" : "warn"}
              label={tracking ? "Tracking" : "Searching for you"}
            />
            <StatusPill tone="muted" label={`${Math.round(fps)} FPS`} />
          </div>
        )}
      </div>

      <div className="flex justify-center">
        {running ? (
          <Button variant="outline" size="lg" onClick={stop}>
            Stop session
          </Button>
        ) : (
          <Button size="lg" onClick={start} disabled={status === "starting"}>
            {status === "starting" ? "Starting…" : "Start camera"}
          </Button>
        )}
      </div>
    </div>
  );
}

function StatusPill({
  tone,
  label,
}: {
  tone: "good" | "warn" | "muted";
  label: string;
}) {
  const toneClass =
    tone === "good"
      ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
      : tone === "warn"
        ? "border-amber-500/40 bg-amber-500/15 text-amber-300"
        : "border-white/20 bg-black/40 text-white/80";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium backdrop-blur-sm ${toneClass}`}
    >
      {label}
    </span>
  );
}
