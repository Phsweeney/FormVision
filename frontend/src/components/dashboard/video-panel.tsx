"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mediaUrl } from "@/lib/api";

interface VideoPanelProps {
  videoUrl: string | null;
  overlayUrl: string | null;
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
export function VideoPanel({ videoUrl, overlayUrl }: VideoPanelProps) {
  const original = mediaUrl(videoUrl);
  const overlay = mediaUrl(overlayUrl);

  if (!original && !overlay) return null;

  const defaultTab = overlay ? "overlay" : "original";

  return (
    <Tabs defaultValue={defaultTab} className="w-full">
      <TabsList>
        <TabsTrigger value="overlay" disabled={!overlay}>
          Skeleton overlay
        </TabsTrigger>
        <TabsTrigger value="original" disabled={!original}>
          Original
        </TabsTrigger>
      </TabsList>

      <TabsContent value="overlay" className="mt-3">
        {overlay ? (
          <VideoFrame src={overlay} label="Skeleton overlay video" />
        ) : (
          <p className="text-muted-foreground border-border/60 rounded-xl border border-dashed p-8 text-center text-sm">
            The overlay could not be rendered for this video. Your measurements
            below are unaffected.
          </p>
        )}
      </TabsContent>

      <TabsContent value="original" className="mt-3">
        {original && <VideoFrame src={original} label="Original video" />}
      </TabsContent>
    </Tabs>
  );
}

function VideoFrame({ src, label }: { src: string; label: string }) {
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
    <div className="border-border/60 bg-black/40 flex h-[min(70vh,540px)] w-full items-center justify-center overflow-hidden rounded-xl border">
      <video
        // `key` forces a fresh element when the source changes, so the browser
        // stops buffering the previous file.
        key={src}
        src={src}
        controls
        playsInline
        preload="metadata"
        aria-label={label}
        className="h-full w-full object-contain"
      />
    </div>
  );
}
