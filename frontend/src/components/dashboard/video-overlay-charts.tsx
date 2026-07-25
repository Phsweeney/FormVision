"use client";

import { useCallback, useEffect, useRef } from "react";

import {
  OVERLAY_TRACKS,
  indexForTime,
  projectTrack,
  resolveDomain,
  stackPanels,
  type Box,
  type Point,
} from "@/lib/charts/overlay-series";
import type { Series } from "@/lib/types";

/**
 * The three movement traces, drawn over the video as it plays.
 *
 * A companion to the rep counter and depth bar burned into the overlay video by
 * `backend/app/analysis/overlay.py`, and styled to match them: the same
 * semi-transparent dark panel so a trace stays readable over arbitrary footage.
 * The difference is that this one is drawn in the browser, which is why it works
 * on clips analysed before it existed and on the "Original" tab as well.
 *
 * The full "Movement over time" section below the video is untouched and remains
 * the place to actually read these signals. This is for glancing at while
 * watching the movement that produced them.
 *
 * Everything here is canvas and effects; the arithmetic lives in
 * `lib/charts/overlay-series.ts` where it can be tested.
 */
export function VideoOverlayCharts({
  video,
  series,
}: {
  /** The element to follow. Null while the player is remounting. */
  video: HTMLVideoElement | null;
  series: Series;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<number | null>(null);
  /** Which scheduler produced the current handle, so cancelling picks right. */
  const usingRvfcRef = useRef(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !video) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;

    // Backing store at device resolution, drawing in CSS pixels. Without this
    // a 2px trace is visibly soft on any high-DPI screen.
    const ratio = window.devicePixelRatio || 1;
    const targetW = Math.round(width * ratio);
    const targetH = Math.round(height * ratio);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const boxes = stackPanels(width, height, OVERLAY_TRACKS.length);
    const upTo = indexForTime(series.time_s, video.currentTime);

    OVERLAY_TRACKS.forEach((track, trackIndex) => {
      const box = boxes[trackIndex];
      const signals = track.keys.map((key) => series[key] as (number | null)[]);
      const domain = resolveDomain(signals, track.domain);

      drawPanel(ctx, box);

      signals.forEach((values, signalIndex) => {
        const points = projectTrack(values, series.time_s, domain, box);
        drawTrace(ctx, points, upTo, track.colors[signalIndex]);
      });

      drawLabels(ctx, box, track.label, currentValue(signals, upTo), track);
    });
  }, [series, video]);

  // The frame loop. `requestVideoFrameCallback` fires once per decoded video
  // frame, which is exactly the cadence wanted and cheaper than rAF on a paused
  // or slow video; rAF is the fallback where it does not exist.
  useEffect(() => {
    if (!video) return;

    const supportsRvfc = typeof video.requestVideoFrameCallback === "function";
    usingRvfcRef.current = supportsRvfc;
    let cancelled = false;

    const step = () => {
      if (cancelled) return;
      draw();
      schedule();
    };
    const schedule = () => {
      frameRef.current = supportsRvfc
        ? video.requestVideoFrameCallback(step)
        : requestAnimationFrame(step);
    };
    schedule();

    // Neither callback fires reliably while paused, so scrubbing a stopped
    // video would leave the traces frozen at wherever playback last was.
    // `loadedmetadata` covers the first paint before anything has played.
    const redraw = () => draw();
    for (const event of ["seeking", "seeked", "loadedmetadata"] as const) {
      video.addEventListener(event, redraw);
    }

    return () => {
      cancelled = true;
      if (frameRef.current !== null) {
        if (usingRvfcRef.current && video.cancelVideoFrameCallback) {
          video.cancelVideoFrameCallback(frameRef.current);
        } else {
          cancelAnimationFrame(frameRef.current);
        }
        frameRef.current = null;
      }
      for (const event of ["seeking", "seeked", "loadedmetadata"] as const) {
        video.removeEventListener(event, redraw);
      }
    };
  }, [draw, video]);

  // The player is a fixed-height box, so the canvas resizes with the viewport
  // rather than only on mount.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}

/** Panel colours, matched to the burned-in HUD this sits opposite. */
const PANEL_FILL = "rgba(24, 22, 20, 0.55)";
const PANEL_STROKE = "rgba(255, 255, 255, 0.10)";
const LABEL_INK = "rgba(255, 255, 255, 0.65)";
const VALUE_INK = "rgba(255, 255, 255, 0.92)";

function drawPanel(ctx: CanvasRenderingContext2D, box: Box): void {
  ctx.beginPath();
  // `roundRect` is recent enough to be worth guarding. This runs once per panel
  // per frame, so an unsupported call would not fail once, it would throw
  // thirty times a second and take the whole overlay down with it.
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(box.x, box.y, box.width, box.height, Math.min(8, box.height / 4));
  } else {
    ctx.rect(box.x, box.y, box.width, box.height);
  }
  ctx.fillStyle = PANEL_FILL;
  ctx.fill();
  ctx.strokeStyle = PANEL_STROKE;
  ctx.lineWidth = 1;
  ctx.stroke();
}

/**
 * Draw one signal up to `upTo`, then mark the head.
 *
 * Nulls break the path rather than joining across, matching the full charts.
 * The head dot is what makes it read as "now" rather than as a static thumbnail.
 */
function drawTrace(
  ctx: CanvasRenderingContext2D,
  points: (Point | null)[],
  upTo: number,
  color: string,
): void {
  if (upTo < 0) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  ctx.beginPath();
  let penDown = false;
  for (let i = 0; i <= upTo && i < points.length; i++) {
    const point = points[i];
    if (point === null) {
      penDown = false;
      continue;
    }
    if (penDown) ctx.lineTo(point.x, point.y);
    else ctx.moveTo(point.x, point.y);
    penDown = true;
  }
  ctx.stroke();

  const head = lastPresent(points, upTo);
  if (head) {
    ctx.beginPath();
    ctx.arc(head.x, head.y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

function drawLabels(
  ctx: CanvasRenderingContext2D,
  box: Box,
  label: string,
  value: number | null,
  track: { unit: string; digits: number },
): void {
  const size = Math.max(8, Math.min(11, box.height * 0.22));
  ctx.save();
  ctx.font = `600 ${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.textBaseline = "top";

  ctx.fillStyle = LABEL_INK;
  ctx.fillText(label, box.x + 6, box.y + 5);

  if (value !== null) {
    const text = `${value.toFixed(track.digits)}${track.unit}`;
    ctx.fillStyle = VALUE_INK;
    ctx.textAlign = "right";
    ctx.fillText(text, box.x + box.width - 6, box.y + 5);
  }
  ctx.restore();
}

/** The most recent measured value at or before `upTo`, across a track's signals. */
function currentValue(signals: (number | null)[][], upTo: number): number | null {
  if (upTo < 0) return null;
  for (const values of signals) {
    for (let i = Math.min(upTo, values.length - 1); i >= 0; i--) {
      const value = values[i];
      if (value !== null && value !== undefined) return value;
    }
  }
  return null;
}

function lastPresent(points: (Point | null)[], upTo: number): Point | null {
  for (let i = Math.min(upTo, points.length - 1); i >= 0; i--) {
    const point = points[i];
    if (point) return point;
  }
  return null;
}
