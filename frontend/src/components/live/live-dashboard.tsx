"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { DEFAULT_CONFIG, fetchConfig, type AnalysisConfig } from "@/lib/analysis/config";
import type { FramePose, Severity } from "@/lib/analysis/types";
import { Coach, type Cue } from "@/lib/live/coach";
import { LiveAnalyzer, type LiveState } from "@/lib/live/live-analyzer";
import { summarizeSession, type SessionSummary } from "@/lib/live/session";
import { createVoiceCoach, type VoiceCoach } from "@/lib/live/voice";
import { fetchFaultModels } from "@/lib/ml/classify";
import type { FaultModelBundle } from "@/lib/ml/model";

import { LiveMetrics } from "./live-metrics";
import { LiveStage } from "./live-stage";
import { MlStatusBox } from "./ml-status-box";
import { WorkoutSummary } from "./workout-summary";

/** Push UI state at most this often; pose still runs every frame underneath. */
const UI_UPDATE_INTERVAL_MS = 100;

const CUE_TONE: Record<Severity, string> = {
  good: "border-emerald-500/40 bg-emerald-500/15 text-emerald-200",
  info: "border-sky-500/40 bg-sky-500/15 text-sky-200",
  warning: "border-amber-500/40 bg-amber-500/15 text-amber-200",
  critical: "border-red-500/40 bg-red-500/15 text-red-200",
};

/**
 * The live coaching experience: the camera stage plus a live-updating metrics
 * panel, a coaching cue, and spoken feedback.
 *
 * The analysis engine and coach run on every frame (via `LiveStage`'s
 * `onFrame`); React state is flushed at ~10 Hz so re-renders never compete with
 * the 30 FPS pose loop. Cues are rare, so they update immediately.
 */
export function LiveDashboard() {
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const configRef = useRef<AnalysisConfig>(DEFAULT_CONFIG);
  const analyzerRef = useRef<LiveAnalyzer | null>(null);
  const coachRef = useRef<Coach | null>(null);
  const voiceRef = useRef<VoiceCoach | null>(null);
  const latestStateRef = useRef<LiveState | null>(null);
  const lastUiFlushRef = useRef(0);
  const sessionStartMsRef = useRef(0);
  const modelsRef = useRef<FaultModelBundle | null>(null);

  const [state, setState] = useState<LiveState | null>(null);
  const [running, setRunning] = useState(false);
  const [cue, setCue] = useState<Cue | null>(null);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [speechRate, setSpeechRate] = useState(1);
  // null until the fetch resolves: the first paint must not claim the model is
  // unavailable before it has been asked for.
  const [modelsLoaded, setModelsLoaded] = useState<boolean | null>(null);

  // Fetch the shared thresholds once; until then the defaults stand in.
  useEffect(() => {
    const controller = new AbortController();
    fetchConfig(controller.signal).then((fetched) => {
      configRef.current = fetched;
      setConfig(fetched);
    });
    return () => controller.abort();
  }, []);

  // Fetch the fault detectors once. Like `fetchConfig` above this never
  // rejects: a missing or corrupt bundle resolves to null, the model box says
  // so, and every other part of live coaching is unaffected.
  useEffect(() => {
    const controller = new AbortController();
    fetchFaultModels(controller.signal).then((bundle) => {
      modelsRef.current = bundle;
      setModelsLoaded(bundle !== null);
    });
    return () => controller.abort();
  }, []);

  // The voice coach needs the browser, so create it after mount.
  useEffect(() => {
    voiceRef.current = createVoiceCoach();
    return () => voiceRef.current?.cancel();
  }, []);

  const handleFrame = useCallback((frame: FramePose) => {
    const analyzer = analyzerRef.current;
    if (!analyzer) return;
    const next = analyzer.push(frame);
    latestStateRef.current = next;

    const nowS = performance.now() / 1000;
    const nextCue = coachRef.current?.update(next, nowS) ?? null;
    if (nextCue) {
      setCue(nextCue);
      voiceRef.current?.speak(nextCue.text);
    }

    const nowMs = performance.now();
    if (nowMs - lastUiFlushRef.current >= UI_UPDATE_INTERVAL_MS) {
      lastUiFlushRef.current = nowMs;
      setState(next);
    }
  }, []);

  const handleRunningChange = useCallback((isRunning: boolean) => {
    setRunning(isRunning);
    if (isRunning) {
      analyzerRef.current = new LiveAnalyzer(configRef.current, modelsRef.current);
      coachRef.current = new Coach(configRef.current);
      sessionStartMsRef.current = performance.now();
      latestStateRef.current = null;
      setState(null);
      setCue(null);
      setSummary(null);
    } else {
      const analyzer = analyzerRef.current;
      if (analyzer && analyzer.reps.length > 0) {
        const durationS = (performance.now() - sessionStartMsRef.current) / 1000;
        setSummary(summarizeSession(analyzer.reps, durationS));
      }
      if (latestStateRef.current) setState(latestStateRef.current);
      analyzerRef.current = null;
      voiceRef.current?.cancel();
    }
  }, []);

  const toggleVoice = useCallback(() => {
    setVoiceEnabled((prev) => {
      const next = !prev;
      voiceRef.current?.setEnabled(next);
      return next;
    });
  }, []);

  const changeRate = useCallback((rate: number) => {
    setSpeechRate(rate);
    voiceRef.current?.setRate(rate);
  }, []);

  const settingUp =
    running &&
    state !== null &&
    (state.phase === "waiting" || state.phase === "calibrating");

  return (
    <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
      <div className="space-y-3">
        <LiveStage onFrame={handleFrame} onRunningChange={handleRunningChange} />
        {settingUp && state && (
          <div className="border-amber-500/40 bg-amber-500/10 rounded-xl border p-4">
            <p className="text-sm font-medium text-amber-200">
              {state.phase === "waiting"
                ? "Get into position and hold still"
                : "Calibrating — keep still"}
            </p>
            <p className="text-muted-foreground mt-1 text-xs">
              {state.phase === "waiting"
                ? "Stand where your whole body is in frame. Calibration starts once you settle, so you can start the camera before walking into position."
                : "Measuring your standing height and body scale so reps and depth are accurate."}
            </p>
            <div className="bg-muted/40 mt-3 h-1.5 overflow-hidden rounded-full">
              <div
                className="h-full rounded-full bg-amber-400 transition-all"
                style={{ width: `${Math.round(state.calibrationProgress * 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Under the video rather than in the right-hand column. The right side
            was carrying the cue, the metrics grid and the summary already, and
            this box belongs next to the thing it is describing. */}
        <MlStatusBox
          verdict={state?.mlVerdict ?? null}
          available={modelsLoaded}
          running={running}
          // The model does not judge depth, so the box has to say so out loud
          // when the rules have just measured a rep as short. Otherwise "no
          // fault detected" reads as approval of a rep that was clearly poor.
          shallowDepthPercent={
            state?.lastRep?.halfRep ? (state.lastRep.depthPercent ?? null) : null
          }
          repInProgress={state !== null && state.phase !== "standing" && state.ready}
        />
      </div>

      <div className="space-y-4">
        {/* Coaching cue, while a session is live. */}
        {running && cue && (
          <div
            className={`rounded-xl border px-4 py-3 text-lg font-semibold ${CUE_TONE[cue.severity]}`}
          >
            {cue.text}
          </div>
        )}

        {!running && summary ? (
          <WorkoutSummary summary={summary} />
        ) : state ? (
          <LiveMetrics state={state} config={config} />
        ) : (
          <div className="border-border/60 bg-card/40 text-muted-foreground grid min-h-48 place-items-center rounded-xl border p-6 text-center text-sm">
            Start the camera to see live reps, depth, and tempo.
          </div>
        )}

        {/* Voice controls. */}
        <div className="border-border/60 bg-card/40 flex flex-wrap items-center gap-4 rounded-xl border p-4">
          <Button variant={voiceEnabled ? "secondary" : "outline"} size="sm" onClick={toggleVoice}>
            {voiceEnabled ? "Voice on" : "Voice off"}
          </Button>
          <label className="text-muted-foreground flex flex-1 items-center gap-2 text-xs">
            Rate
            <input
              type="range"
              min={0.6}
              max={1.6}
              step={0.1}
              value={speechRate}
              onChange={(event) => changeRate(Number(event.target.value))}
              className="flex-1 accent-emerald-500"
              aria-label="Speech rate"
            />
            <span className="font-mono tabular-nums">{speechRate.toFixed(1)}x</span>
          </label>
        </div>
      </div>
    </div>
  );
}
