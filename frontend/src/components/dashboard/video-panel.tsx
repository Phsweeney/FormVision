"use client";

import { useState } from "react";

import { VideoOverlayCharts } from "@/components/dashboard/video-overlay-charts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mediaUrl } from "@/lib/api";
import type { Series } from "@/lib/types";

interface VideoPanelProps {
  videoUrl: string | null;
  overlayUrl: string | null;
  /** Null while an analysis is still processing. */
  series?: Series | null;
}

/**
 * The original video and the skeleton overlay.
 *
 * Tabbed rather than side by side. Squat footage is almost always shot in
 * portrait, so two vertical videos next to each other would each be too small
 * to read a skeleton in. Tabs also let the two share screen position, which
 * makes switching between them a direct comparison.
 *
 * The overlay tab is first because it is the reason someone came here.
 */
export function VideoPanel({ videoUrl, overlayUrl, series }: VideoPanelProps) {
  const original = mediaUrl(videoUrl);
  const overlay = mediaUrl(overlayUrl);
  const [showCharts, setShowCharts] = useState(true);

  if (!original && !overlay) return null;

  const defaultTab = overlay ? "overlay" : "original";
  const charts = series && series.time_s.length > 0 ? series : null;

  return (
    <Tabs defaultValue={defaultTab} className="w-full">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TabsList>
          <TabsTrigger value="overlay" disabled={!overlay}>
            Skeleton overlay
          </TabsTrigger>
          <TabsTrigger value="original" disabled={!original}>
            Original
          </TabsTrigger>
        </TabsList>

        {/* Only offered when there is something to draw, so a still-processing
            analysis does not show a control that would do nothing. */}
        {charts && (
          <button
            type="button"
            onClick={() => setShowCharts((shown) => !shown)}
            aria-pressed={showCharts}
            className="border-border/60 text-muted-foreground hover:text-foreground rounded-full border px-3 py-1 text-xs transition-colors"
          >
            {showCharts ? "Hide traces" : "Show traces"}
          </button>
        )}
      </div>

      <TabsContent value="overlay" className="mt-3">
        {overlay ? (
          <VideoFrame
            src={overlay}
            label="Skeleton overlay video"
            series={showCharts ? charts : null}
          />
        ) : (
          <p className="text-muted-foreground border-border/60 rounded-xl border border-dashed p-8 text-center text-sm">
            The overlay could not be rendered for this video. Your measurements
            below are unaffected.
          </p>
        )}
      </TabsContent>

      <TabsContent value="original" className="mt-3">
        {original && (
          <VideoFrame
            src={original}
            label="Original video"
            series={showCharts ? charts : null}
          />
        )}
      </TabsContent>
    </Tabs>
  );
}

function VideoFrame({
  src,
  label,
  series,
}: {
  src: string;
  label: string;
  series: Series | null;
}) {
  // A state setter as the ref, not `useRef`, because `key={src}` below recreates
  // the element whenever the tab or source changes. This re-renders the overlay
  // with the new node instead of leaving it holding a detached one.
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);

  return (
    // The height lives on the container, not the video. A <video> whose
    // metadata has not loaded reports a default 2:1 intrinsic size, so a
    // width-driven element would size itself from that and then jump when the
    // real dimensions arrive. Fixing the box means the layout never depends on
    // network timing, and `object-contain` letterboxes whatever shape arrives.
    //
    // Capped absolutely as well as by viewport fraction: squat footage is
    // portrait, and a bare `70vh` lets the player swallow a tall screen and
    // push every metric below the fold.
    //
    // `relative` anchors the trace overlay. Note the overlay is pinned to this
    // container rather than to the painted video rect: on portrait footage,
    // which is most of it, that puts the traces in the letterbox bar where they
    // obscure nothing at all.
    <div className="border-border/60 bg-black/40 relative flex h-[min(70vh,540px)] w-full items-center justify-center overflow-hidden rounded-xl border">
      <video
        // `key` forces a fresh element when the source changes, so the browser
        // stops buffering the previous file.
        key={src}
        ref={setVideo}
        src={src}
        controls
        playsInline
        preload="metadata"
        aria-label={label}
        className="h-full w-full object-contain"
      />
      {series && <VideoOverlayCharts video={video} series={series} />}
    </div>
  );
}
