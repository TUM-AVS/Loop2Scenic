#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Recorder for Scenic-Carla

Good at Scenic 3-view BEV, FPV, TPV simulation recording

Fixes included:
- FPV position corrected (driver-seat-like)
- TPV distance/height tuned
- 180° camera flip fixed (angle wrap smoothing)
- Stutter reduced (async ffmpeg writer: queue + worker thread)
- Main loop tick-driven (world.wait_for_tick)
- BEV wider coverage via --fov and --bev_height
"""

import os
import time
import math
import argparse
import subprocess
import threading
import queue
from datetime import datetime

import carla


# ============================================================
# Async FFmpeg writer (non-blocking callback)
# ============================================================
class AsyncFFmpegWriter:
    def __init__(self, out_path, w, h, fps, crf=23, qsize=96):
        self.out_path = out_path
        self.q = queue.Queue(maxsize=qsize)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgra",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                out_path
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        self._running = True
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def write(self, raw):
        # NEVER block sensor callback; drop frames if queue is full
        try:
            self.q.put_nowait(raw)
        except queue.Full:
            pass

    def _worker(self):
        while self._running:
            item = self.q.get()
            if item is None:
                break
            try:
                if self.proc and self.proc.stdin:
                    self.proc.stdin.write(item)
            except Exception:
                # If ffmpeg dies, just stop consuming
                break

    def close(self):
        self._running = False
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
            if self.proc:
                self.proc.wait(timeout=5)
        except Exception:
            pass


# ============================================================
# Ego identification
# - role_name (ego/hero) first
# - fallback to scoring if allowed
# ============================================================
def identify_ego(world, allow_scoring=True, observe=0.4, dt=0.05, debug=False):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if not vehicles:
        raise RuntimeError("No vehicles")

    role_hits = []
    for v in vehicles:
        rn = v.attributes.get("role_name", "")
        if debug:
            print(f"[DEBUG identify_ego] id={v.id} role_name='{rn}'", flush=True)
        if rn.lower() in ("ego", "hero"):
            role_hits.append(v)

    if role_hits:
        if debug:
            print(f"[DEBUG identify_ego] role_name match -> id={role_hits[0].id}", flush=True)
        return role_hits[0]

    if debug:
        print("[DEBUG identify_ego] no role_name ego found", flush=True)

    if not allow_scoring:
        raise RuntimeError("No role_name ego found")

    # ---- behavior scoring fallback ----
    hist = {v.id: {"loc": [], "spd": [], "thr": [], "brk": []} for v in vehicles}
    steps = max(1, int(observe / dt))
    for _ in range(steps):
        for v in vehicles:
            try:
                loc = v.get_location()
                vel = v.get_velocity()
                ctrl = v.get_control()
                spd = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                hist[v.id]["loc"].append((loc.x, loc.y))
                hist[v.id]["spd"].append(spd)
                hist[v.id]["thr"].append(ctrl.throttle)
                hist[v.id]["brk"].append(ctrl.brake)
            except Exception:
                pass
        time.sleep(dt)

    def score(h):
        s = 0.0
        if h["thr"] and (max(h["thr"]) - min(h["thr"]) > 0.2): s += 2.0
        if h["brk"] and (max(h["brk"]) > 0.1): s += 1.0
        if h["spd"] and (max(h["spd"]) - min(h["spd"]) > 1.0): s += 1.0
        if len(h["loc"]) >= 2:
            dx = h["loc"][-1][0] - h["loc"][0][0]
            dy = h["loc"][-1][1] - h["loc"][0][1]
            if dx*dx + dy*dy > 1.0: s += 1.0
        return s

    best_id = max(hist.items(), key=lambda kv: score(kv[1]))[0]
    return next(v for v in vehicles if v.id == best_id)


# ============================================================
# Camera smoothing (FIX: shortest-angle wrap to avoid 180° flip)
# ============================================================
def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

class SmoothPose:
    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.init = False
        self.loc = carla.Location()
        self.rot = carla.Rotation()

    def update(self, loc, rot):
        if not self.init:
            self.loc = carla.Location(loc.x, loc.y, loc.z)
            self.rot = carla.Rotation(rot.pitch, rot.yaw, rot.roll)
            self.init = True
            return self.loc, self.rot

        # position smoothing
        self.loc.x += (loc.x - self.loc.x) * self.alpha
        self.loc.y += (loc.y - self.loc.y) * self.alpha
        self.loc.z += (loc.z - self.loc.z) * self.alpha

        # rotation smoothing (shortest path!)
        dp = _wrap_deg(rot.pitch - self.rot.pitch)
        dy = _wrap_deg(rot.yaw   - self.rot.yaw)
        dr = _wrap_deg(rot.roll  - self.rot.roll)

        self.rot.pitch = _wrap_deg(self.rot.pitch + dp * self.alpha)
        self.rot.yaw   = _wrap_deg(self.rot.yaw   + dy * self.alpha)
        self.rot.roll  = _wrap_deg(self.rot.roll  + dr * self.alpha)

        return self.loc, self.rot


def compose(base_tf: carla.Transform, rel_tf: carla.Transform):
    loc = carla.Location(rel_tf.location.x, rel_tf.location.y, rel_tf.location.z)
    base_tf.transform(loc)
    rot = carla.Rotation(
        pitch=base_tf.rotation.pitch + rel_tf.rotation.pitch,
        yaw=base_tf.rotation.yaw + rel_tf.rotation.yaw,
        roll=base_tf.rotation.roll + rel_tf.rotation.roll,
    )
    return carla.Transform(loc, rot)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default="./recordings")
    ap.add_argument("--duration", type=float, default=30.0)

    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=23)

    # view tuning
    ap.add_argument("--fov", type=float, default=110.0)          # wider coverage
    ap.add_argument("--bev_height", type=float, default=50.0)    # meters
    ap.add_argument("--smooth_alpha", type=float, default=0.12)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ego-alive-threshold", type=float, default=0.5)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    if args.debug:
        print("[DEBUG] Connected to CARLA", flush=True)
        print("[DEBUG] Map:", world.get_map().name, flush=True)

    # ensure world ready
    #world.wait_for_tick()
    #snapshot = world.wait_for_tick()
    #frame = snapshot.frame

    map_name = world.get_map().name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]

    scene_dir = os.path.join(args.outdir, f"{args.prefix}__{map_name}__{timestamp}")
    os.makedirs(scene_dir, exist_ok=True)

    print("[RECORDER] waiting for first ego (no stability required)...", flush=True)
    ego_first_time = None

    # Wait until ego appears
    while True:
        try:
            ego = identify_ego(world, allow_scoring=True, debug=args.debug)
            print(f"[RECORDER] first ego detected id={ego.id} role_name={ego.attributes.get('role_name','')}", flush=True)
            ego_first_time = time.time()
            break
        except Exception:
            time.sleep(0.2)

    # camera blueprint
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(args.width))
    bp.set_attribute("image_size_y", str(args.height))
    bp.set_attribute("sensor_tick", str(1.0 / args.fps))
    bp.set_attribute("fov", str(args.fov))

    # =======================================================
    # Camera relative transforms (FIXED values for MKZ-like sedan)
    # - FPV: driver-seat-ish (avoid interior occlusion)
    # - TPV: slightly higher & farther
    # =======================================================
    rel_fpv = carla.Transform(
        carla.Location(x=1.6, y=-0.25, z=1.35),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
    )
    rel_tpv = carla.Transform(
        carla.Location(x=-8.0, y=0.0, z=3.2),
        carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0)
    )

    # BEV: fixed world rotation, smooth only location
    bev_h = float(args.bev_height)

    # writers + cams + smoothers
    writers = {}
    cams = {}
    smooth = {
        "FPV": SmoothPose(alpha=args.smooth_alpha),
        "TPV": SmoothPose(alpha=args.smooth_alpha),
        "BEV": SmoothPose(alpha=args.smooth_alpha),
    }

    def outpath(view):
        return os.path.join(scene_dir, f"{args.prefix}__{map_name}__{view}__{timestamp}.mp4")

    def spawn_cam(view, init_tf):
        writers[view] = AsyncFFmpegWriter(outpath(view), args.width, args.height, args.fps, crf=args.crf)
        cam = world.spawn_actor(bp, init_tf)
        cam.listen(lambda img, v=view: writers[v].write(img.raw_data))
        cams[view] = cam

    # initial placement
    ego_tf = ego.get_transform()
    spawn_cam("FPV", compose(ego_tf, rel_fpv))
    spawn_cam("TPV", compose(ego_tf, rel_tpv))
    spawn_cam("BEV", carla.Transform(
        carla.Location(ego_tf.location.x, ego_tf.location.y, ego_tf.location.z + bev_h),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    ))

    print("[RECORDER] recording started", flush=True)

    missing_ego_since = None
    MAX_MISSING_EGO_TIME = 1.0   # 秒，工程里 0.5–2s 都合理

    last_frame = None
    stale_frame_since = None
    MAX_STALE_TIME = 1.0         # 世界停滞阈值

    start = time.time()
    last_debug = 0.0

    try:
        while True:
            now = time.time()

            # =================================================
            # A) ego 是否存在（在 tick 前判断）
            # =================================================
            try:
                ego = identify_ego(world, allow_scoring=False, debug=False)
                missing_ego_since = None
            except Exception:
                if missing_ego_since is None:
                    missing_ego_since = now
                elif now - missing_ego_since > MAX_MISSING_EGO_TIME:
                    print("[RECORDER] ego lost → scene finished, stopping recorder", flush=True)
                    break
                # ego 暂时没了，不推进画面
                world.wait_for_tick()
                continue

            tf = ego.get_transform()

            # =================================================
            # B) FPV: HARD LOCK（必须在 tick 之前）
            # =================================================
            desired_fpv = compose(tf, rel_fpv)
            cams["FPV"].set_transform(desired_fpv)
            # cams["FPV"].set_transform(compose(tf, rel_fpv))

            # =================================================
            # C) 推进世界（唯一一次 tick）
            # =================================================
            snapshot = world.wait_for_tick()
            frame = snapshot.frame

            # =================================================
            # D) 世界是否停滞（tick 之后判断）
            # =================================================
            if last_frame is None or frame != last_frame:
                last_frame = frame
                stale_frame_since = None
            else:
                if stale_frame_since is None:
                    stale_frame_since = now
                elif now - stale_frame_since > MAX_STALE_TIME:
                    print("[RECORDER] world stalled → scene finished, stopping recorder", flush=True)
                    break

            # =================================================
            # E) TPV: smoothing（tick 之后）
            # =================================================
            desired_tpv = compose(tf, rel_tpv)
            loc_tpv, rot_tpv = smooth["TPV"].update(
                desired_tpv.location,
                desired_tpv.rotation
            )
            cams["TPV"].set_transform(carla.Transform(loc_tpv, rot_tpv))

            # =================================================
            # F) BEV: smoothing location only（tick 之后）
            # =================================================
            desired_bev_loc = carla.Location(
                tf.location.x,
                tf.location.y,
                tf.location.z + bev_h
            )
            loc_bev, _ = smooth["BEV"].update(
                desired_bev_loc,
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            )
            cams["BEV"].set_transform(
                carla.Transform(
                    loc_bev,
                    carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
                )
            )

            # =================================================
            # G) debug
            # =================================================
            if args.debug and (now - last_debug) > 2.0:
                last_debug = now
                print(
                    f"[DEBUG] ego id={ego.id} "
                    f"loc=({tf.location.x:.1f},{tf.location.y:.1f}) "
                    f"yaw={tf.rotation.yaw:.1f}",
                    flush=True
                )

            # =================================================
            # H) 可选 duration（不想要可删）
            # =================================================
            if now - start >= args.duration:
                break

        if ego_first_time is not None:
            ego_alive_seconds = time.time() - ego_first_time
        else:
            ego_alive_seconds = 0.0
        print(f"[RECORDER] ego_alive_seconds: {ego_alive_seconds:.3f}", flush=True)
        print(f"[RECORDER] ego_alive_threshold: {args.ego_alive_threshold:.3f}", flush=True)

    # try:
    #     while True:
    #         if time.time() - start >= args.duration:
    #             break


    #         # ego may respawn; always lock by role_name first
    #         try:
    #             ego = identify_ego(world, allow_scoring=False, debug=False)
    #         except Exception:
    #             # if temporarily missing, just skip this tick
    #             continue

    #         tf = ego.get_transform()
            
    #         # === FPV: NO SMOOTHING, HARD LOCK ===
    #         desired_fpv = compose(tf, rel_fpv)
    #         cams["FPV"].set_transform(desired_fpv)
            
    #         # tick-driven loop (smoother than sleep)
    #         world.wait_for_tick()
            
    #         # === TPV: keep smoothing ===
    #         desired_tpv = compose(tf, rel_tpv)
    #         loc_tpv, rot_tpv = smooth["TPV"].update(desired_tpv.location, desired_tpv.rotation)
    #         cams["TPV"].set_transform(carla.Transform(loc_tpv, rot_tpv))


    #         # FPV / TPV follow ego frame (smoothed)
    #         #for view, rel in (("FPV", rel_fpv), ("TPV", rel_tpv)):
    #         #    desired = compose(tf, rel)
    #         #    loc, rot = smooth[view].update(desired.location, desired.rotation)
    #         #    cams[view].set_transform(carla.Transform(loc, rot))

    #         # BEV: smooth location only, keep fixed rotation (no yaw wobble)
    #         desired_bev_loc = carla.Location(tf.location.x, tf.location.y, tf.location.z + bev_h)
    #         loc_bev, _ = smooth["BEV"].update(desired_bev_loc, carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
    #         cams["BEV"].set_transform(carla.Transform(loc_bev, carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

    #         if args.debug and (time.time() - last_debug) > 2.0:
    #             last_debug = time.time()
    #             print(f"[DEBUG] ego id={ego.id} loc=({tf.location.x:.1f},{tf.location.y:.1f}) yaw={tf.rotation.yaw:.1f}", flush=True)

    finally:
        for cam in cams.values():
            try:
                cam.stop()
            except Exception:
                pass
            try:
                cam.destroy()
            except Exception:
                pass
        for w in writers.values():
            w.close()

    print(f"[RECORDER] saved to: {scene_dir}", flush=True)


if __name__ == "__main__":
    main()
