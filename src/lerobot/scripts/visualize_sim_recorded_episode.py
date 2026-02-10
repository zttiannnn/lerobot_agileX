#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path
import numpy as np
import rerun as rr
from PIL import Image
import tqdm

def to_hwc_uint8_numpy(img):
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)
            return (img * 255).astype(np.uint8)
    return np.array(img)

def visualize_sim_recorded_episode(
    episode_dir,
    batch_size=32,
    mode="local",
    web_port=9090,
    ws_port=9087,
    save=False,
    output_dir=None,
    show_depth=True,
):
    episode_dir = Path(episode_dir)
    # manager_node / je_to_lerobot 使用的文件名是 meta.jsonl
    meta_path = episode_dir / "meta.jsonl"
    images_root = episode_dir / "images"
    # 自动检测所有相机目录（兼容多种命名），并处理可能的多层嵌套目录结构
    if show_depth:
        camera_names = [d.name for d in images_root.iterdir() if d.is_dir()]
    else:
        camera_names = [d.name for d in images_root.iterdir() if d.is_dir() and "depth" not in d.name.lower()]
    cam_dirs = {}
    for cam in camera_names:
        cam_root = images_root / cam
        first_frame = cam_root / f"frame_{0:06d}.png"
        if first_frame.exists():
            cam_dirs[cam] = cam_root
            continue
        # 在子目录中查找包含帧文件的目录
        found = None
        for p in cam_root.iterdir():
            if p.is_dir() and any(p.glob('frame_*.png')):
                found = p
                break
        if found is None:
            matches = list(cam_root.rglob('frame_*.png'))
            if matches:
                found = matches[0].parent
        cam_dirs[cam] = found if found is not None else cam_root
    print(f"[INFO] 检测到相机: {list(cam_dirs.keys())}")
    # 读取所有元数据（支持 meta.jsonl，每行一帧）
    with open(meta_path, "r") as f:
        meta_list = [json.loads(line) for line in f]
    num_frames = len(meta_list)
    print(f"[INFO] 读取到 {num_frames} 帧元数据")
    # rerun 初始化
    repo_id = episode_dir.name
    spawn_local_viewer = mode == "local" and not save
    # 使用带时间戳的 scene 名，保证每次运行都创建新的 viewer scene，避免复用旧数据流
    rr.init(f"sim_recorded/{repo_id}_{int(time.time())}", spawn=spawn_local_viewer)
    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)
    # 可视化主循环
    for frame_idx in tqdm.tqdm(range(num_frames)):
        meta = meta_list[frame_idx]
        rr.set_time("frame_index", sequence=frame_idx)
        rr.set_time("timestamp", timestamp=meta.get("timestamp", 0))
        # 相机图片（使用已解析的 cam_dirs，以兼容嵌套）
        for cam, cam_dir in cam_dirs.items():
            img_path = cam_dir / f"frame_{frame_idx:06d}.png"
            if img_path.exists():
                img = Image.open(img_path)
                img_np = np.array(img)
                rr.log(cam, rr.Image(to_hwc_uint8_numpy(img_np)))

        # joints: 如果存在 joints 列表，按 joint 单独记录 position/velocity/effort（每个 joint 为独立 namespace）
        joints = meta.get("joints")
        if joints:
            try:
                # 逐 joint 记录 position/velocity/effort，分成三个顶层命名空间，分别为 position/、velocity/、effort/，以便在 rerun 中显示为不同的图
                for j_idx, j in enumerate(joints):
                    pos = j.get("position", []) or []
                    vel = j.get("velocity", []) or []
                    eff = j.get("effort", []) or []
                    # 将属性放在顶层命名空间，joint 作为子命名空间
                    for d_idx, v in enumerate(pos):
                        rr.log(f"position/joint_{j_idx}/dim_{d_idx}", rr.Scalars(float(v)))
                    for d_idx, v in enumerate(vel):
                        rr.log(f"velocity/joint_{j_idx}/dim_{d_idx}", rr.Scalars(float(v)))
                    for d_idx, v in enumerate(eff):
                        rr.log(f"effort/joint_{j_idx}/dim_{d_idx}", rr.Scalars(float(v)))
            except Exception:
                pass

        # 不再记录 action 或 observation（确保没有其它地方写入这些流）

    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / f"sim_recorded_{repo_id}.rrd"
        rr.save(rrd_path)
        print(f"[INFO] 已保存 rrd 文件到 {rrd_path}")
        return rrd_path
    elif mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=str, required=True, help="episode_000000 目录路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="local", help="local 或 distant")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9087)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show-depth", type=int, default=0, help="是否显示名称中包含 'depth' 的相机（1 显示，0 不显示）")
    args = parser.parse_args()
    visualize_sim_recorded_episode(
        episode_dir=args.episode_dir,
        batch_size=args.batch_size,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=bool(args.save),
        output_dir=args.output_dir,
        show_depth=bool(args.show_depth),
    )
