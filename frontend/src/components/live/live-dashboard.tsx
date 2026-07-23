"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { DEFAULT_CONFIG, fetchConfig, type AnalysisConfig } from "@/lib/analysis/config";
import type { FramePose } from "@/lib/analysis/types";
import { LiveAnalyzer, type LiveState } from "@/lib/live/live-analyzer";

import { LiveMetrics } from "./live-metrics";
import { LiveStage } from "./live-stage";

/** Push UI state at most this often; pose still runs every frame underneath. */
const UI_UPDATE_INTERVAL_MS = 100;

/**
 * The live coaching experience: the camera stage plus a live-updating metrics
 * panel, driven by the analysis engine.
 *
 * The engine runs on every frame (via `LiveStage`'s `onFrame`), but React state
 * is flushed at ~10 Hz so re-renders never compete with the 30 FPS pose loop.
 * The latest state is always kept in a ref and flushed when the session stops.
 */
export function LiveDashboard() {
  // Config in state (read during render for depth thresholds) plus a ref mirror
  // (read inside callbacks, where reading state directly would go stale).
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const configRef = useRef<AnalysisConfig>(DEFAULT_CONFIG);
  const analyzerRef = useRef<LiveAnalyzer | null>(null);
  const latestStateRef = useRef<LiveState | null>(null);
  const lastUiFlushRef = useRef(0);

  const [state, setState] = useState<LiveState | null>(null);
  const [running, setRunning] = useState(false);

  // Fetch the shared thresholds once; until then the defaults stand in.
  useEffect(() => {
    const controller = new AbortController();
    fetchConfig(controller.signal).then((fetched) => {
      configRef.current = fetched;
      setConfig(fetched);
    });
    return () => controller.abort();
  }, []);

  const handleFrame = useCallback((frame: FramePose) => {
    const analyzer = analyzerRef.current;
    if (!analyzer) return;
    const next = analyzer.push(frame);
    latestStateRef.current = next;

    const now = performance.now();
    if (now - lastUiFlushRef.current >= UI_UPDATE_INTERVAL_MS) {
      lastUiFlushRef.current = now;
      setState(next);
    }
  }, []);

  const handleRunningChange = useCallback((isRunning: boolean) => {
    setRunning(isRunning);
    if (isRunning) {
      // Fresh analyzer per session so calibration and reps start clean.
      analyzerRef.current = new LiveAnalyzer(configRef.current);
      latestStateRef.current = null;
      setState(null);
    } else {
      // Flush the final frame's state so the last numbers stay on screen.
      if (latestStateRef.current) setState(latestStateRef.current);
      analyzerRef.current = null;
    }
  }, []);

  const calibrating = running && state !== null && state.phase === "calibrating";

  return (
    <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
      <div className="space-y-3">
        <LiveStage onFrame={handleFrame} onRunningChange={handleRunningChange} />
        {calibrating && state && (
          <div className="border-amber-500/40 bg-amber-500/10 rounded-xl border p-4">
            <p className="text-sm font-medium text-amber-200">
              Calibrating — stand still
            </p>
            <p className="text-muted-foreground mt-1 text-xs">
              Measuring your standing height and body scale so reps and depth are
              accurate.
            </p>
            <div className="bg-muted/40 mt-3 h-1.5 overflow-hidden rounded-full">
              <div
                className="h-full rounded-full bg-amber-400 transition-all"
                style={{ width: `${Math.round(state.calibrationProgress * 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div>
        {state ? (
          <LiveMetrics state={state} config={config} />
        ) : (
          <div className="border-border/60 bg-card/40 text-muted-foreground grid h-full min-h-48 place-items-center rounded-xl border p-6 text-center text-sm">
            Start the camera to see live reps, depth, and tempo.
          </div>
        )}
      </div>
    </div>
  );
}
