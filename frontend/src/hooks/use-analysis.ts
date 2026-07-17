"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchAnalysis } from "@/lib/api";
import { type Analysis, isPending } from "@/lib/types";

/** How often to re-check while analysis is running. */
const POLL_INTERVAL_MS = 1500;

/**
 * Give up after this long.
 *
 * A safety net for the case where the backend dies mid-analysis and the record
 * is left on `processing` forever. Without it the page would poll indefinitely
 * and never tell the user anything.
 */
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Consecutive network failures tolerated before surfacing an error.
 *
 * One failed poll is usually a blip; tearing the dashboard down over a single
 * dropped request is worse than waiting for the next tick.
 */
const MAX_CONSECUTIVE_ERRORS = 4;

export interface UseAnalysisResult {
  analysis: Analysis | null;
  error: string | null;
  isLoading: boolean;
  isPolling: boolean;
  refresh: () => void;
}

/**
 * Fetches an analysis and polls until it reaches a terminal state.
 *
 * Polling rather than websockets: analysis takes seconds to a couple of
 * minutes, exactly one client cares about the result, and polling needs no
 * connection management or reconnection logic. A websocket would be more
 * machinery for no benefit at this scale.
 */
export function useAnalysis(id: string): UseAnalysisResult {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPolling, setIsPolling] = useState(false);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Initialised to 0 rather than `Date.now()`: a ref initialiser runs during
  // render, and calling an impure function there can produce a different value
  // on every re-render. The real start time is stamped in the effect below.
  const startedAtRef = useRef(0);
  const failureCountRef = useRef(0);
  // Guards against a response arriving after unmount, which would otherwise set
  // state on a dead component.
  const activeRef = useRef(true);
  // Holds the latest `poll` so the scheduled callback can invoke it without
  // `poll` referring to itself before it is declared.
  const pollRef = useRef<() => void>(() => {});

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const scheduleNext = useCallback(() => {
    timerRef.current = setTimeout(() => pollRef.current(), POLL_INTERVAL_MS);
  }, []);

  const poll = useCallback(async () => {
    if (!activeRef.current) return;

    try {
      const next = await fetchAnalysis(id);
      if (!activeRef.current) return;

      failureCountRef.current = 0;
      setAnalysis(next);
      setError(null);
      setIsLoading(false);

      if (!isPending(next)) {
        setIsPolling(false);
        return;
      }

      if (Date.now() - startedAtRef.current > POLL_TIMEOUT_MS) {
        setIsPolling(false);
        setError(
          "Analysis is taking much longer than expected. It may have stalled — "
            + "try uploading the video again.",
        );
        return;
      }

      setIsPolling(true);
      // Scheduled after each response rather than on a fixed interval, so a
      // slow response cannot cause requests to pile up behind it.
      scheduleNext();
    } catch (caught) {
      if (!activeRef.current) return;

      const isNotFound = caught instanceof ApiError && caught.status === 404;
      const message
        = caught instanceof ApiError
          ? caught.message
          : "Could not load this analysis.";

      // A 404 is final — retrying will not conjure the record into existence.
      if (isNotFound) {
        setError(message);
        setIsLoading(false);
        setIsPolling(false);
        return;
      }

      failureCountRef.current += 1;
      if (failureCountRef.current >= MAX_CONSECUTIVE_ERRORS) {
        setError(message);
        setIsLoading(false);
        setIsPolling(false);
        return;
      }

      scheduleNext();
    }
  }, [id, scheduleNext]);

  // Keep the scheduled callback pointing at the current `poll`.
  useEffect(() => {
    pollRef.current = () => void poll();
  }, [poll]);

  const refresh = useCallback(() => {
    clearTimer();
    failureCountRef.current = 0;
    startedAtRef.current = Date.now();
    setError(null);
    setIsLoading(true);
    void poll();
  }, [clearTimer, poll]);

  useEffect(() => {
    activeRef.current = true;
    startedAtRef.current = Date.now();
    failureCountRef.current = 0;

    // Scheduled rather than invoked directly. Starting the fetch inside the
    // effect body puts its state updates in the commit phase and triggers a
    // cascading render; a zero-delay timer moves the whole request out of it.
    // It also reuses `timerRef`, so unmounting cancels a first poll that has
    // not fired yet.
    timerRef.current = setTimeout(() => pollRef.current(), 0);

    return () => {
      activeRef.current = false;
      clearTimer();
    };
  }, [clearTimer]);

  return { analysis, error, isLoading, isPolling, refresh };
}
