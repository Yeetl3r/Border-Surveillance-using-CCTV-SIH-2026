"""Typed configuration.

Every timing, distance and confidence threshold in the system lives here and is
loaded from YAML. No magic numbers in code -- demo and operational values
differ (the unattended-baggage dwell is 15 s on stage and 120 s in the field)
and both must be selectable without editing source.

Loading is: ``system.yaml`` <- ``profiles/<profile>.yaml`` overlay <- explicit
overrides. The profile only ever overrides; it never introduces new keys.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class RuntimeCfg(BaseModel):
    device: Literal["auto", "cuda", "cpu"] = "auto"
    profile: str = "demo"
    # F15: when true, no component may reach the network. Model loaders set the
    # relevant offline env vars and refuse to download.
    offline: bool = True
    log_level: str = "INFO"
    target_fps: float = 25.0


# ---------------------------------------------------------------------------
# Tier 1 -- perception
# ---------------------------------------------------------------------------
class SahiCfg(BaseModel):
    """Sliced inference for distant targets.

    A person at 150-200 m on wide-angle CCTV is 15-25 px tall; full-frame
    inference at 640 px loses them. Tiling costs roughly (tiles + 1) forward
    passes, so it is off by default and enabled per camera where range demands.
    """

    enabled: bool = False
    tile: int = 640
    overlap: float = 0.2
    merge_iou: float = 0.55
    # Only tile when the frame is meaningfully larger than one tile.
    min_frame_px: int = 900


class DetectorCfg(BaseModel):
    model: str = "yolo26s.pt"
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.7
    half: bool = True
    max_det: int = 300
    # COCO ids grouped into the four operational classes F1 asks for. Data, not
    # an enum -- Phase 2 drone classes are a config change, not a refactor.
    class_groups: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "person": [0],
            "vehicle": [1, 2, 3, 5, 7],
            "animal": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            "bag": [24, 26, 28],
        }
    )
    sahi: SahiCfg = Field(default_factory=SahiCfg)


class TrackerCfg(BaseModel):
    """F2 -- both trackers wired, config-selectable.

    BoT-SORT is the default: published comparisons give it the best IDF1 and
    the lowest ID-switch count, and identity stability across fence and tree
    occlusion is what this application actually needs. ByteTrack remains the
    throughput pick and is one config key away.
    """

    kind: Literal["botsort", "bytetrack"] = "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 60
    match_thresh: float = 0.8
    fuse_score: bool = True
    # BoT-SORT only
    gmc_method: Literal["sparseOptFlow", "orb", "ecc", "none"] = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False


class PoseCfg(BaseModel):
    """F16 -- infiltration stance.

    Off by default. The stance signal is inherently high-false-alarm: crouching
    versus standing is a continuum at CCTV resolution, and a wrong CRAWLING
    alert is precisely the alert that teaches an operator to ignore the system.
    Enable deliberately, not incidentally.
    """

    enabled: bool = False
    model: str = "yolo26s-pose.pt"
    imgsz: int = 320
    conf: float = 0.35
    min_keypoints: int = 5
    keypoint_conf: float = 0.35
    # A box touching a frame border has unreliable geometry -- the classic
    # false CRAWLING trigger is a person walking out of shot. Suppress it.
    suppress_edge_clipped: bool = True
    edge_margin_px: int = 4
    max_crops_per_frame: int = 6
    # Torso-vertical-extent ratios. Below prone_ratio the body is horizontal.
    prone_ratio: float = 0.65
    crouch_ratio: float = 0.85

    # Crawling human vs quadruped. A box wider than this is low-slung enough
    # that class alone cannot be trusted: a person on hands and knees and a
    # dog are the same shape, and the detector confuses them in both
    # directions. Above this aspect ratio the pose model is asked whether a
    # coherent HUMAN skeleton resolves, and that answer outranks the label.
    quadruped_aspect: float = 0.95
    # Measured: people p5 0.208 / median 0.826 (n=121) against animals
    # p95 0.633 / median 0.000 (n=1533). At 0.37 the split keeps 90% of
    # people and admits 11% of animals. See configs/system.yaml.
    human_skeleton_min: float = 0.37


# ---------------------------------------------------------------------------
# Tier 0 -- health and operator stream
# ---------------------------------------------------------------------------
class HealthCfg(BaseModel):
    """F9 -- anti-tamper.

    ``check_within_s`` is the only hard latency requirement in the brief: the
    health indicator must trip within a second of the lens being covered.
    """

    enabled: bool = True
    blur_min: float = 55.0          # absolute floor: a true blackout, any camera
    ssim_min: float = 0.55          # structural drift below this = repositioned/covered
    ssim_size: int = 160            # downscale for SSIM; full-res is wasteful
    freeze_max_frames: int = 25     # identical frames before declaring a frozen feed
    ref_update_alpha: float = 0.02  # rolling reference EMA; slow enough to ignore people
    consecutive_bad: int = 5        # frames of fault before state changes (anti-flicker)
    check_within_s: float = 1.0

    # An ABSOLUTE sharpness floor does not transfer between cameras. A dark IR
    # scene sits at a Laplacian variance around 190; a bright indoor camera
    # sits near 970. A floor tuned for one is meaningless on the other, and a
    # hand held over a well-lit lens stays sharp (skin texture) while being
    # completely wrong -- it never approaches an absolute floor.
    #
    # So each camera also learns its own healthy baseline and reports a
    # RELATIVE collapse against it.
    blur_baseline_alpha: float = 0.01   # EMA of the healthy baseline
    blur_warmup_frames: int = 30        # frames before the baseline is trusted
    blur_drop_ratio: float = 0.55       # severe defocus alone: blur < ratio x baseline

    # Obstruction — something held over the lens — is the case neither signal
    # catches alone. A palm is still sharp in absolute terms and the scene it
    # replaces is still structured, so blur and SSIM each overlap the healthy
    # range. Measured on assets/tamper_test.mp4 against a hand-labelled event:
    #
    #   relative blur   healthy min 0.796   tampered max 0.845
    #   relative SSIM   healthy min 0.644   tampered max 0.929
    #   combined        99% recall, 1 false positive in 760 healthy frames
    #
    # So the rule is the conjunction, and the ratios below are the measured
    # operating point rather than a guess.
    obstruct_blur_ratio: float = 0.85
    obstruct_ssim_ratio: float = 0.90
    ssim_baseline_alpha: float = 0.01


class EnhanceCfg(BaseModel):
    """Operator-viewable stream only. NEVER in the detection path.

    Enhancement optimises perceptual quality, not the detection objective, and
    introduces artefacts that damage small-object detection specifically. The
    detector sees raw frames; the human sees these.
    """

    enabled: bool = True
    mode: Literal["none", "clahe", "dcp", "clahe+dcp"] = "clahe"
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    dcp_omega: float = 0.85
    dcp_patch: int = 11
    dcp_t0: float = 0.1


# ---------------------------------------------------------------------------
# Tier 2/3 -- spatial and temporal
# ---------------------------------------------------------------------------
class ConfirmCfg(BaseModel):
    """F7 -- temporal confirmation before any alert fires.

    Specified in SECONDS plus a hit ratio, not in raw frame count. N frames at
    30 fps and N frames at 8 fps are different guarantees, and degraded backhaul
    -- the stated target condition -- is exactly what produces 8 fps. A
    frame-count threshold silently changes meaning on the footage we are
    designing for.
    """

    window_s: float = 2.0
    min_ratio: float = 0.6
    min_hits: int = 3
    # Per-kind overrides; health faults must trip fast (see HealthCfg).
    overrides: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "HEALTH_FAULT": {"window_s": 0.8, "min_ratio": 0.7, "min_hits": 2},
            "BAG_UNATTENDED": {"window_s": 1.5, "min_ratio": 0.6, "min_hits": 2},
            "STANCE_ANOMALY": {"window_s": 3.0, "min_ratio": 0.7, "min_hits": 5},
        }
    )


class SpatialCfg(BaseModel):
    """Tier 2 tuning."""

    # A tripwire crossing is INSTANTANEOUS -- it happens in exactly one frame.
    # An N-of-M confirmation rule designed for persistent states can therefore
    # never confirm it. So after a crossing we keep emitting corroborating
    # observations while the subject REMAINS on the side it crossed to, for
    # this long. A real incursion corroborates; a detector flicker across the
    # line does not, because the track does not persist on the far side.
    tripwire_corroborate_s: float = 2.5
    default_groups: list[str] = Field(
        default_factory=lambda: ["person", "vehicle", "animal"])


class BaggageCfg(BaseModel):
    """F6 -- unattended baggage.

    Distances are in METRES via the homography, not pixels, so the thresholds
    are depth-invariant. A bag 4 m from its owner is 4 m away whether it is
    near the camera or at the far end of the scene.
    """

    enabled: bool = True
    assoc_radius_m: float = 2.5     # bag within this of a person -> ACCOMPANIED
    detach_distance_m: float = 4.0  # owner further than this -> DETACHED, timer runs
    dwell_s: float = 15.0           # DEMO value. Operational default is 120 s.
    reclaim_grace_s: float = 2.0    # owner must be back this long before clearing
    min_bag_conf: float = 0.30
    # Pixel fallback when the camera is uncalibrated. Used only then, and the
    # resulting observation is marked degraded so the UI can say so.
    assoc_radius_px: float = 120.0
    detach_distance_px: float = 220.0


class AlertCfg(BaseModel):
    clear_after_s: float = 6.0      # no supporting observation for this long -> CLEARED
    expire_after_s: float = 600.0   # hard ceiling on an unresolved alert
    max_live: int = 500
    conf_high: float = 0.75
    conf_medium: float = 0.50
    # The threat matrix. Which observation, against which identity verdict,
    # produces which colour. This is the core semantic of the whole system and
    # it belongs in config where it can be argued about, not buried in code.
    threat_matrix: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "TRIPWIRE_CROSS": {
                "FRIENDLY": "FRIENDLY",
                "UNIDENTIFIED": "CRITICAL",
                "INSUFFICIENT_QUALITY": "WARNING",
                "NONE": "WARNING",
            },
            "ZONE_PRESENCE": {
                "FRIENDLY": "FRIENDLY",
                "UNIDENTIFIED": "WARNING",
                "INSUFFICIENT_QUALITY": "WATCH",
                "NONE": "WATCH",
            },
            "PROXIMITY": {
                "FRIENDLY": "INFO",
                "UNIDENTIFIED": "WARNING",
                "INSUFFICIENT_QUALITY": "WATCH",
                "NONE": "WATCH",
            },
            "BAG_UNATTENDED": {"NONE": "WARNING"},
            "STANCE_ANOMALY": {
                "FRIENDLY": "WATCH",
                "UNIDENTIFIED": "WARNING",
                "INSUFFICIENT_QUALITY": "WATCH",
                "NONE": "WATCH",
            },
            "IDENTITY_UNKNOWN": {"NONE": "WATCH"},
            "HEALTH_FAULT": {"NONE": "WARNING"},
            "SENSOR_CUE": {"NONE": "WATCH"},
        }
    )


# ---------------------------------------------------------------------------
# Tier 5 -- identity
# ---------------------------------------------------------------------------
class IdentityCfg(BaseModel):
    """F8 -- open-set friendly matching.

    Single-image enrolment false-matches more than is acceptable, so the
    gallery expects several images per identity across pose and lighting, and
    the threshold is tuned against measured precision/recall rather than by
    feel. It is config for exactly that reason.
    """

    enabled: bool = True
    weights: str = "osnet_x0_25_msmt17.pth"
    gallery_dir: str = "data/gallery"
    sim_threshold: float = 0.62
    # Quality gate. A crop below these dimensions cannot support an identity
    # claim, and the honest answer is INSUFFICIENT_QUALITY, not a guess.
    min_crop_h: int = 96
    min_crop_w: int = 32
    min_blur: float = 25.0
    min_det_conf: float = 0.45
    embed_every_n_frames: int = 8
    max_crops_per_frame: int = 4


# ---------------------------------------------------------------------------
# Tier 6/9 -- VLM and query
# ---------------------------------------------------------------------------
class VlmCfg(BaseModel):
    """F11 -- forensic visual grounding on alert keyframes, on demand.

    LocateAnything-3B is NVIDIA-licensed for non-commercial academic and
    non-profit research use only, and its base LM (Qwen2.5-3B-Instruct) carries
    the Qwen Research licence. That is recorded here, and surfaced in the UI,
    because a government-facing evaluator asking "can this be fielded?" deserves
    an accurate answer.
    """

    enabled: bool = True
    model_id: str = "nvidia/LocateAnything-3B"
    local_dir: str = "weights/LocateAnything-3B"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_new_tokens: int = 2048
    max_image_side: int = 1024
    timeout_s: float = 60.0
    # Queue depth 1 with rejection. A 3B-parameter job is never queued -- if
    # one is already running, the next request is refused with 503. Refusing is
    # the feature; buffering is how a live demo runs out of memory.
    queue_depth: int = 1
    licence: str = "NVIDIA Open Model License -- non-commercial research only"


class QueryCfg(BaseModel):
    max_rows: int = 200
    default_window_h: int = 24


# ---------------------------------------------------------------------------
# Tier 8 -- relay and storage
# ---------------------------------------------------------------------------
class RelayCfg(BaseModel):
    """F14 -- sub-3 KB tactical relay for a constrained backhaul link."""

    enabled: bool = True
    max_bytes: int = 3072
    crop_max_px: int = 160
    jpeg_quality_start: int = 60
    jpeg_quality_min: int = 15
    out_dir: str = "data/relay"


class StorageCfg(BaseModel):
    db_path: str = "data/db/sentinel.db"
    evidence_dir: str = "data/evidence"
    clip_dir: str = "data/clips"
    clip_pre_s: float = 3.0
    clip_post_s: float = 3.0
    keep_clip_frames: int = 300


class PipelineCfg(BaseModel):
    """Bounded queues with an explicit, logged drop policy.

    Tier 1 runs at video rate; the VLM takes seconds. Every queue crossing that
    boundary is bounded and every drop is counted, because an invisible drop
    policy is the same as no drop policy.
    """

    q_decode: int = 4          # drop-oldest: latency beats completeness
    q_crops: int = 16          # drop-newest: re-ID is opportunistic, retried next frame
    q_alerts: int = 256        # drop lowest severity, never CRITICAL
    q_evidence: int = 64       # block: evidence loss is unacceptable
    jpeg_quality: int = 72
    stream_max_width: int = 1280


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class HomographyCfg(BaseModel):
    """F3 -- four operator-marked ground points -> metric coordinates.

    Assumes a FIXED camera. A PTZ camera invalidates the homography on every
    pan and tilt, and supporting one needs a preset-linked calibration table.
    That is not built; ``fixed`` is asserted rather than assumed silently.
    """

    enabled: bool = False
    fixed_camera: bool = True
    image_points: list[list[float]] = Field(default_factory=list)
    world_points: list[list[float]] = Field(default_factory=list)
    homography_id: str = ""
    # Optional: a known real-world length used to sanity-check the scale.
    scale_check_m: float | None = None

    @field_validator("image_points", "world_points")
    @classmethod
    def _four_points(cls, v: list[list[float]]) -> list[list[float]]:
        if v and len(v) != 4:
            raise ValueError("homography needs exactly 4 point pairs")
        return v


class TripwireCfg(BaseModel):
    """F4 -- a directional line, in ground metres when calibrated."""

    id: str
    points: list[list[float]]   # exactly 2 points: [[x1,y1],[x2,y2]]
    space: Literal["metric", "pixel"] = "metric"
    # Which side counts as inbound. The sign of the 2D cross product of the
    # line direction with (point - line_start).
    inbound_sign: Literal[1, -1] = 1
    label: str = ""
    # Which object groups this geometry tests. Empty = the analyzer default.
    # Wildlife crossing a perimeter is the canonical false alarm in this
    # domain, so excluding `animal` per-zone is a first-class control.
    groups: list[str] = Field(default_factory=list)


class ZonePolygonCfg(BaseModel):
    id: str
    points: list[list[float]]
    space: Literal["metric", "pixel"] = "metric"
    label: str = ""
    kind: Literal["restricted", "mask"] = "restricted"
    groups: list[str] = Field(default_factory=list)


class ProximityCfg(BaseModel):
    """F5 -- metric inner ring around an asset."""

    id: str
    geometry: Literal["point", "line", "polygon"] = "line"
    points: list[list[float]]
    space: Literal["metric", "pixel"] = "metric"
    threshold_m: float = 3.0
    label: str = ""
    groups: list[str] = Field(default_factory=list)


class CameraCfg(BaseModel):
    camera_id: str
    source: str
    loop: bool = True
    name: str = ""
    enabled: bool = True
    # `enabled` means the pipeline is BUILT; `autostart` means it also runs at
    # boot. The demo cameras exist from startup so the demo page can retarget
    # them instantly, but they must not burn GPU decoding clips nobody is
    # looking at -- three pipelines each running the detector would leave every
    # one of them at a third of the frame rate.
    autostart: bool = True
    read_timeout_s: float = 5.0
    reconnect_delay_s: float = 2.0
    # Cap ingest rate for file sources so a 12-second clip does not replay at
    # 900 fps and flood every downstream queue.
    pace_to_source_fps: bool = True
    homography: HomographyCfg = Field(default_factory=HomographyCfg)
    tripwires: list[TripwireCfg] = Field(default_factory=list)
    zones: list[ZonePolygonCfg] = Field(default_factory=list)
    proximity: list[ProximityCfg] = Field(default_factory=list)
    detector_overrides: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class WeaponCfg(BaseModel):
    """Specialised weapon detector, run on person crops.

    Cropped rather than full-frame on purpose: a rifle at 80 m is a handful of
    pixels in a 1920-wide frame, and the only scale at which it is legible is
    the person box that has already been detected and tracked. Running the
    small model on a few crops is also cheaper than running it on every frame.
    """

    enabled: bool = False
    model: str = "weapons_yolo26n.pt"
    imgsz: int = 320
    # Higher than the perimeter detector's. A weapon claim is a claim that
    # escalates immediately, so it is one place to buy precision with recall.
    conf: float = 0.45
    max_crops_per_frame: int = 4
    min_crop_h: int = 64
    # Measured: 0.30 raises end-to-end weapon recall from 45.1% to 52.3%
    # at an unchanged 5.9% false rate. See configs/system.yaml.
    crop_pad: float = 0.30


class AerialCfg(BaseModel):
    """Specialised drone detector, run on the full frame.

    Full-frame rather than cropped, because unlike a weapon there is no parent
    object to crop to -- a drone is not attached to anything the perimeter
    model tracks.
    """

    enabled: bool = False
    model: str = "drones_yolo26n.pt"
    imgsz: int = 960          # drones are small; do not downscale them away
    # Measured on the shipped weights: 0.75 gives 62.2% recall at zero false
    # frames across 442 frames of drone-free footage, whose highest false
    # confidence is 0.730. See configs/system.yaml.
    conf: float = 0.75
    every_n_frames: int = 2   # sky changes slowly; halve the cost
    # A drone filling a third of a surveillance frame is not a perimeter
    # intrusion -- it is a close-up, and the training set is full of them
    # (44% of its boxes cover more than 2.25% of the image). This cap encodes
    # the deployment context rather than the training distribution, and it is
    # what stops the model calling an entire dark scene a drone.
    max_area_frac: float = 0.35
    # Bird rejection. The stock COCO detector carries bird as class 14 and was
    # trained on far more bird imagery than any drone set contains, so the
    # discrimination is made BETWEEN models rather than by a thin bird class
    # trained alongside the drone. A drone box that a confident stock bird
    # detection overlaps is withheld, and the reason is recorded.
    bird_class_id: int = 14
    # Measured: 0.45/1.0 withheld only 14% of bird-clip candidates; 0.30/0.5
    # withholds 70%. Two models box the same bird differently, and the drone
    # model is the over-confident one. See configs/system.yaml.
    bird_iou: float = 0.30
    bird_margin: float = 0.25  # bird_conf must exceed drone_conf x this to win


class ThreatCfg(BaseModel):
    """Specialised detectors that answer questions the perimeter model cannot."""

    weapons: WeaponCfg = Field(default_factory=WeaponCfg)
    aerial: AerialCfg = Field(default_factory=AerialCfg)


class DemoPanelCfg(BaseModel):
    """One feed inside a demo: a camera bound to a clip.

    `detector` carries per-panel overrides, applied to the camera when the demo
    is selected. Sensitivity is a property of the footage: a crawling subject
    in a wide shot needs a lower confidence floor than a perimeter approach,
    and the same floor on the perimeter would flood the tracker.
    """

    camera: str
    label: str = ""
    source: str
    detector: dict[str, Any] = Field(default_factory=dict)
    #: Per-panel overrides for the specialised detectors, same shape as the
    #: `threat.aerial` / `threat.weapons` blocks in system.yaml. Aerial size and
    #: confidence in particular are properties of the footage: a drone filmed
    #: close fills far more of the frame than one crossing a perimeter.
    aerial: dict[str, Any] = Field(default_factory=dict)
    weapons: dict[str, Any] = Field(default_factory=dict)
    #: Detectors this panel does not run. A demo isolates one capability, and
    #: the others are noise against it -- the tamper clip is a hand filling the
    #: lens, which is not a scene any weapon claim should be made about.
    suppress: list[str] = Field(default_factory=list)
    #: How this panel presents what it found. Rendering only.
    present: dict[str, Any] = Field(default_factory=dict)


class DemoCfg(BaseModel):
    """One scripted demonstration.

    `claim` is what the demo asserts in one line; `note` is the paragraph worth
    saying out loud while it runs. Both live in config rather than in the
    template so they can be reworded an hour before a presentation.
    """

    id: int
    title: str
    claim: str = ""
    note: str = ""
    panels: list[DemoPanelCfg] = Field(default_factory=list)
    #: Extra surfaces this demo shows beneath its panels: "ask" for the natural
    #: language query box, "evidence" for the keyframe / subject / face tiles.
    #: Most demos isolate one detector and want neither.
    extras: list[str] = Field(default_factory=list)
    #: How this demo names the alert kinds it raises. HEALTH_FAULT is accurate
    #: and tells an audience nothing; on the tamper demo it is the headline.
    alert_labels: dict[str, str] = Field(default_factory=dict)


class SystemCfg(BaseModel):
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    detector: DetectorCfg = Field(default_factory=DetectorCfg)
    tracker: TrackerCfg = Field(default_factory=TrackerCfg)
    pose: PoseCfg = Field(default_factory=PoseCfg)
    threat: ThreatCfg = Field(default_factory=ThreatCfg)
    health: HealthCfg = Field(default_factory=HealthCfg)
    enhance: EnhanceCfg = Field(default_factory=EnhanceCfg)
    spatial: SpatialCfg = Field(default_factory=SpatialCfg)
    confirm: ConfirmCfg = Field(default_factory=ConfirmCfg)
    baggage: BaggageCfg = Field(default_factory=BaggageCfg)
    alerts: AlertCfg = Field(default_factory=AlertCfg)
    identity: IdentityCfg = Field(default_factory=IdentityCfg)
    vlm: VlmCfg = Field(default_factory=VlmCfg)
    query: QueryCfg = Field(default_factory=QueryCfg)
    relay: RelayCfg = Field(default_factory=RelayCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    cameras: list[CameraCfg] = Field(default_factory=list)
    demos: list[DemoCfg] = Field(default_factory=list)

    # Populated at load time; recorded into every evidence record so an alert
    # can always name the thresholds that produced it.
    profile_name: str = "demo"
    root_dir: str = "."

    def snapshot(self) -> dict[str, Any]:
        """The subset of config worth freezing into an evidence record."""
        return {
            "profile": self.profile_name,
            "detector": {"model": self.detector.model, "conf": self.detector.conf,
                         "imgsz": self.detector.imgsz, "sahi": self.detector.sahi.enabled},
            "tracker": {"kind": self.tracker.kind},
            "confirm": self.confirm.model_dump(exclude={"overrides"}),
            "baggage": {"dwell_s": self.baggage.dwell_s,
                        "detach_distance_m": self.baggage.detach_distance_m},
            "identity": {"sim_threshold": self.identity.sim_threshold,
                         "enabled": self.identity.enabled},
            "alerts": {"clear_after_s": self.alerts.clear_after_s},
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    root: str | Path = ".",
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> SystemCfg:
    """Load system.yaml, apply the profile overlay, then explicit overrides.

    Cameras are read from ``configs/cameras/*.yaml`` -- one file per camera, so
    adding a camera never means editing shared config.
    """
    root = Path(root).resolve()
    cfg_dir = root / "configs"

    data = _read_yaml(cfg_dir / "system.yaml")

    chosen = profile or data.get("runtime", {}).get("profile", "demo")
    overlay = _read_yaml(cfg_dir / "profiles" / f"{chosen}.yaml")
    data = _deep_merge(data, overlay)

    if overrides:
        data = _deep_merge(data, overrides)

    cameras = data.pop("cameras", []) or []
    if not cameras:
        for cam_file in sorted((cfg_dir / "cameras").glob("*.yaml")):
            cam = _read_yaml(cam_file)
            if cam:
                cameras.append(cam)
    data["cameras"] = cameras

    # The scripted demos are their own file for the same reason cameras are:
    # which clip a demo plays is a presentation decision, not a system setting.
    if "demos" not in data:
        data["demos"] = (_read_yaml(cfg_dir / "demos.yaml") or {}).get("demos", [])

    data["profile_name"] = chosen
    data["root_dir"] = str(root)

    cfg = SystemCfg.model_validate(data)
    return cfg


def resolve(cfg: SystemCfg, relative: str) -> Path:
    """Resolve a config path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (Path(cfg.root_dir) / p)
