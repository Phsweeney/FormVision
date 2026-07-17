"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, startAnalysis, uploadVideo } from "@/lib/api";

/** Extensions the backend accepts. Kept in step with `allowed_video_extensions`. */
const ACCEPTED_EXTENSIONS = [".mp4", ".mov"] as const;

/**
 * Client-side size ceiling, mirroring the backend's 200 MB cap.
 *
 * Duplicated deliberately. The server check is the one that enforces the rule;
 * this one exists so a user who picks a 2 GB file learns immediately instead of
 * after a long upload that was always going to be rejected.
 */
const MAX_BYTES = 200 * 1024 * 1024;

export type UploadPhase =
  | "idle"
  | "uploading"
  | "starting"
  | "redirecting"
  | "error";

export interface UseUploadResult {
  phase: UploadPhase;
  progress: number;
  error: string | null;
  file: File | null;
  previewUrl: string | null;
  selectFile: (file: File) => void;
  clear: () => void;
  submit: () => Promise<void>;
  isBusy: boolean;
}

/** Reject an obviously wrong file before any bytes go over the network. */
function validate(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    return `Unsupported file type. Please choose an MP4 or MOV video.`;
  }
  if (file.size === 0) {
    return "That file is empty.";
  }
  if (file.size > MAX_BYTES) {
    return `That file is ${(file.size / 1024 / 1024).toFixed(0)} MB. The limit is 200 MB.`;
  }
  return null;
}

/**
 * Owns the upload flow: select, validate, preview, upload, start analysis,
 * navigate to the dashboard.
 */
export function useUpload(): UseUploadResult {
  const router = useRouter();

  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Object URLs pin the file in memory until revoked, so the previous one is
  // always released before a new one replaces it.
  const previewRef = useRef<string | null>(null);

  const releasePreview = useCallback(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = null;
    }
  }, []);

  const selectFile = useCallback(
    (candidate: File) => {
      const problem = validate(candidate);

      releasePreview();

      if (problem) {
        setError(problem);
        setPhase("error");
        setFile(null);
        setPreviewUrl(null);
        return;
      }

      const url = URL.createObjectURL(candidate);
      previewRef.current = url;

      setFile(candidate);
      setPreviewUrl(url);
      setError(null);
      setProgress(0);
      setPhase("idle");
    },
    [releasePreview],
  );

  const clear = useCallback(() => {
    releasePreview();
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    setProgress(0);
    setPhase("idle");
  }, [releasePreview]);

  const submit = useCallback(async () => {
    if (!file) return;

    setPhase("uploading");
    setProgress(0);
    setError(null);

    try {
      const uploaded = await uploadVideo(file, setProgress);

      // Two distinct phases, because after the bytes land the server still has
      // to probe the file and queue the work. Showing "uploading 100%" through
      // that gap looks like a stall.
      setPhase("starting");
      await startAnalysis(uploaded.id);

      setPhase("redirecting");
      // The preview is not revoked here: the object URL must outlive the
      // navigation or the browser cancels the in-flight video element.
      router.push(`/analysis/${uploaded.id}`);
    } catch (caught) {
      const message
        = caught instanceof ApiError
          ? caught.message
          : "Something went wrong while uploading. Please try again.";
      setError(message);
      setPhase("error");
    }
  }, [file, router]);

  return {
    phase,
    progress,
    error,
    file,
    previewUrl,
    selectFile,
    clear,
    submit,
    isBusy:
      phase === "uploading" || phase === "starting" || phase === "redirecting",
  };
}

export { ACCEPTED_EXTENSIONS, MAX_BYTES };
