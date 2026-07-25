import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "@/lib/analysis/config";

import { Coach } from "./coach";
import type { LiveState } from "./live-analyzer";
import type { LiveRep } from "./rep-analysis";

function makeRep(index: number, over: Partial<LiveRep> = {}): LiveRep {
  return {
    index,
    startFrame: 0,
    bottomFrame: 30,
    endFrame: 60,
    startTimeS: 0,
    bottomTimeS: 1,
    endTimeS: 2,
    minKneeAngleDeg: 95,
    minLeftKneeDeg: 95,
    minRightKneeDeg: 95,
    minHipAngleDeg: 40,
    maxTorsoLeanDeg: null,
    kneeAsymmetryDeg: null,
    depthPercent: 95,
    hipBelowKnee: true,
    eccentricS: 1,
    pauseS: 0,
    concentricS: 1,
    tempo: "1-0-1",
    pauseKind: "none",
    halfRep: false,
    ...over,
  };
}

function stateWith(rep: LiveRep): LiveState {
  return {
    phase: "standing",
    calibrationProgress: 1,
    ready: true,
    view: "side",
    repCount: rep.index,
    currentKneeAngleDeg: null,
    currentDepthPercent: null,
    maxDepthPercent: null,
    currentTorsoLeanDeg: null,
    currentRepElapsedS: null,
    lastRep: rep,
    // The coach is rule-based and must stay that way. Pinned as null here so
    // that if a model verdict ever starts influencing a cue, these tests fail.
    mlVerdict: null,
  };
}

describe("Coach", () => {
  it("cues 'Go deeper' for a half rep and praises a good one", () => {
    const coach = new Coach(DEFAULT_CONFIG);
    expect(coach.update(stateWith(makeRep(1, { halfRep: true, depthPercent: 55 })), 0)?.text).toBe(
      "Go deeper",
    );
    expect(coach.update(stateWith(makeRep(2, { depthPercent: 95 })), 10)?.text).toBe(
      "Nice rep",
    );
  });

  it("prioritises a problem over praise within one rep", () => {
    const coach = new Coach(DEFAULT_CONFIG);
    // Deep enough to praise, but also over the lean limit: the problem wins.
    const rep = makeRep(1, { depthPercent: 95, maxTorsoLeanDeg: 60 });
    expect(coach.update(stateWith(rep), 0)?.text).toBe("Chest up");
  });

  it("respects the per-cue cooldown, then fires again", () => {
    const coach = new Coach(DEFAULT_CONFIG);
    const cooldown = DEFAULT_CONFIG.coaching_cooldown_s;

    expect(coach.update(stateWith(makeRep(1, { halfRep: true })), 0)?.text).toBe(
      "Go deeper",
    );
    // A second half rep within the cooldown stays silent.
    expect(
      coach.update(stateWith(makeRep(2, { halfRep: true })), cooldown / 2),
    ).toBeNull();
    // After the cooldown it may speak again.
    expect(
      coach.update(stateWith(makeRep(3, { halfRep: true })), cooldown + 1)?.text,
    ).toBe("Go deeper");
  });

  it("only reacts to a newly completed rep", () => {
    const coach = new Coach(DEFAULT_CONFIG);
    const rep = makeRep(1, { halfRep: true });
    expect(coach.update(stateWith(rep), 0)).not.toBeNull();
    // Same rep index on the next frame: nothing new to say.
    expect(coach.update(stateWith(rep), 0.1)).toBeNull();
  });
});
