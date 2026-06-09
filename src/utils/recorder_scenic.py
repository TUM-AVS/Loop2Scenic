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


def build_arg_parser():
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
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--bev_height", type=float, default=30.0)
    ap.add_argument("--smooth_alpha", type=float, default=0.12)
    ap.add_argument(
        "--views",
        nargs="+",
        default=["bev"],
        choices=["bev", "fpv", "tpv"],
        help="Views to record. Default: bev. Example: --views bev fpv tpv",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ego-alive-threshold", type=float, default=0.5)
    return ap


def connect_world(host, port, debug=False):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    if debug:
        print("[DEBUG] Connected to CARLA", flush=True)
        print("[DEBUG] Map:", world.get_map().name, flush=True)
    return world


def create_camera_blueprint(world, width, height, fps, fov):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("sensor_tick", str(1.0 / fps))
    bp.set_attribute("fov", str(fov))
    return bp


def wait_for_first_ego(world, debug=False, poll_interval=0.2):
    print("[RECORDER] waiting for first ego (no stability required)...", flush=True)
    while True:
        try:
            ego = identify_ego(world, allow_scoring=True, debug=debug)
            print(
                f"[RECORDER] first ego detected id={ego.id} "
                f"role_name={ego.attributes.get('role_name', '')}",
                flush=True,
            )
            return ego, time.time()
        except Exception:
            time.sleep(poll_interval)


def spawn_camera_with_writer(world, blueprint, init_tf, out_path, width, height, fps, crf):
    writer = AsyncFFmpegWriter(out_path, width, height, fps, crf=crf)
    cam = world.spawn_actor(blueprint, init_tf)
    cam.listen(lambda img: writer.write(img.raw_data))
    return cam, writer


def update_fpv_camera(cam, ego_tf, rel_fpv):
    # FPV is hard-locked to ego transform (no smoothing).
    cam.set_transform(compose(ego_tf, rel_fpv))


def update_tpv_camera(cam, smooth_pose, ego_tf, rel_tpv):
    desired_tpv = compose(ego_tf, rel_tpv)
    loc_tpv, rot_tpv = smooth_pose.update(desired_tpv.location, desired_tpv.rotation)
    cam.set_transform(carla.Transform(loc_tpv, rot_tpv))


def update_bev_camera(cam, smooth_pose, ego_tf, bev_height):
    desired_bev_loc = carla.Location(
        ego_tf.location.x,
        ego_tf.location.y,
        ego_tf.location.z + bev_height,
    )
    bev_rot = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    loc_bev, _ = smooth_pose.update(desired_bev_loc, bev_rot)
    cam.set_transform(carla.Transform(loc_bev, bev_rot))


def cleanup(cams, writers):
    for cam in cams.values():
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cam.destroy()
        except Exception:
            pass
    for writer in writers.values():
        writer.close()


def run_recording_loop(
    world,
    cams,
    smoothers,
    selected_views,
    rel_fpv,
    rel_tpv,
    bev_height,
    debug=False,
):
    missing_ego_since = None
    max_missing_ego_time = 1.0
    last_debug = 0.0

    # Create a threading Event to handle clean exiting when ego is lost
    stop_event = threading.Event()

    def on_tick_callback(snapshot):
        now = time.time()
        nonlocal missing_ego_since
        nonlocal last_debug
        
        # If we already told the loop to stop, do nothing
        if stop_event.is_set():
            return

        try:
            ego = identify_ego(world, allow_scoring=False, debug=False)
            missing_ego_since = None
        except Exception:
            if missing_ego_since is None:
                missing_ego_since = now
            elif now - missing_ego_since > max_missing_ego_time:
                print("[RECORDER] ego lost → scene finished, stopping recorder", flush=True)
                stop_event.set()
            return

        tf = ego.get_transform()
        if "FPV" in selected_views:
            update_fpv_camera(cams["FPV"], tf, rel_fpv)
        if "TPV" in selected_views:
            update_tpv_camera(cams["TPV"], smoothers["TPV"], tf, rel_tpv)
        if "BEV" in selected_views:
            update_bev_camera(cams["BEV"], smoothers["BEV"], tf, bev_height)

        if debug and (now - last_debug) > 2.0:
            last_debug = now
            print(
                f"[DEBUG] ego id={ego.id} "
                f"loc=({tf.location.x:.1f},{tf.location.y:.1f}) "
                f"yaw={tf.rotation.yaw:.1f}",
                flush=True,
            )

    # Attach the callback to the world. 
    # This ensures the camera moves ONLY when Scenic tells the server to step forward.
    tick_id = world.on_tick(on_tick_callback)

    try:
        # Keep the main thread alive while the async callback does the work
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        # Crucial cleanup: remove the callback before exiting
        world.remove_on_tick(tick_id)

# ============================================================
# Main
# ============================================================
def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    world = connect_world(args.host, args.port, debug=args.debug)
    scene_dir = args.outdir
    os.makedirs(scene_dir, exist_ok=True)
    ego, ego_first_time = wait_for_first_ego(world, debug=args.debug)
    bp = create_camera_blueprint(world, args.width, args.height, args.fps, args.fov)
    bev_height = float(args.bev_height)
    selected_views = {view.upper() for view in args.views}
    cams = {}
    writers = {}
    smoothers = {}
    if "TPV" in selected_views:
        smoothers["TPV"] = SmoothPose(alpha=args.smooth_alpha)
    if "BEV" in selected_views:
        smoothers["BEV"] = SmoothPose(alpha=args.smooth_alpha)

    rel_fpv = carla.Transform(
        carla.Location(x=1.6, y=-0.25, z=1.35),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )
    rel_tpv = carla.Transform(
        carla.Location(x=-8.0, y=0.0, z=3.2),
        carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0),
    )

    ego_tf = ego.get_transform()
    init_fpv_tf = compose(ego_tf, rel_fpv)
    init_tpv_tf = compose(ego_tf, rel_tpv)
    init_bev_tf = carla.Transform(
        carla.Location(ego_tf.location.x, ego_tf.location.y, ego_tf.location.z + bev_height),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    fpv_out_path = os.path.join(scene_dir, "FPV.mp4")
    tpv_out_path = os.path.join(scene_dir, "TPV.mp4")
    bev_out_path = os.path.join(scene_dir, "BEV.mp4")
    if "FPV" in selected_views:
        cams["FPV"], writers["FPV"] = spawn_camera_with_writer(
            world=world,
            blueprint=bp,
            init_tf=init_fpv_tf,
            out_path=fpv_out_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
        )
    if "TPV" in selected_views:
        cams["TPV"], writers["TPV"] = spawn_camera_with_writer(
            world=world,
            blueprint=bp,
            init_tf=init_tpv_tf,
            out_path=tpv_out_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
        )
    if "BEV" in selected_views:
        cams["BEV"], writers["BEV"] = spawn_camera_with_writer(
            world=world,
            blueprint=bp,
            init_tf=init_bev_tf,
            out_path=bev_out_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
        )

    print(f"[RECORDER] selected views: {sorted(selected_views)}", flush=True)

    print("[RECORDER] recording started", flush=True)
    try:
        run_recording_loop(
            world=world,
            cams=cams,
            smoothers=smoothers,
            selected_views=selected_views,
            rel_fpv=rel_fpv,
            rel_tpv=rel_tpv,
            bev_height=bev_height,
            debug=args.debug,
        )

        if ego_first_time is not None:
            ego_alive_seconds = time.time() - ego_first_time
        else:
            ego_alive_seconds = 0.0
        print(f"[RECORDER] ego_alive_seconds: {ego_alive_seconds:.3f}", flush=True)
        print(f"[RECORDER] ego_alive_threshold: {args.ego_alive_threshold:.3f}", flush=True)
    finally:
        cleanup(cams, writers)

    print(f"[RECORDER] saved to: {scene_dir}", flush=True)


if __name__ == "__main__":
    main()
