import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_expand_to_lerobot_batch(
    episodes_root,
    lerobot_root,
    task="test_task",
    repo_id="dragonbobo-no3/lerobot_dataset",
    fps=30,
    use_videos=True,
    batch_encode_num=5,  # 新增参数，控制每次并发编码多少个episode
    resume = False,
    target_meta_file_name = "meta.jsonl"
):
    t0 = time.time()
    episodes_root = Path(episodes_root)
    # 如果不是 resume 模式但目标数据目录已存在，立刻报错以避免覆盖已有数据
    if (not resume) and Path(lerobot_root).exists():
        raise FileExistsError(
            f"目标数据目录已存在: {lerobot_root}. 若要在已有数据集后追加，请设置 resume=True；若确实要覆盖请先删除该目录。"
        )
    # 自动查找所有 episode 目录
    episode_dirs = sorted([d for d in episodes_root.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"[INFO] 共检测到 {len(episode_dirs)} 个episode: {[d.name for d in episode_dirs]}")
    dataset = None
    # 用于 resume 时从现有数据集的下一集开始编号
    next_episode_index = None
    # 记录新 episode_index 与源 episode 目录的映射，后续编码时使用
    episode_index_map = {}
    first_ep_idx = None
    last_ep_idx = None
    gripper_dim = None
    for idx, episode_dir in enumerate(episode_dirs):
        if resume and dataset is None:
            # 仅在第一次循环时加载已有数据集
            dataset = LeRobotDataset(
                repo_id,
                root=lerobot_root,
            )
            next_episode_index = dataset.meta.total_episodes
            gripper_feature = dataset.meta.features.get("gripper") if dataset is not None else None
            if gripper_feature is not None:
                shape = gripper_feature.get("shape", []) if isinstance(gripper_feature, dict) else getattr(gripper_feature, "shape", [])
                if shape:
                    gripper_dim = shape[0]
        # 读取 meta.jsonl
        features_path = episode_dir / target_meta_file_name
        with open(features_path, "r") as f:
            features_list = [json.loads(line) for line in f]
        batch_size = len(features_list)
        print(f"[INFO] [{episode_dir.name}] {target_meta_file_name} 读取完成，帧数: {batch_size}")
        # 相机列表由 meta.jsonl 中的键决定：以 "cam" 开头的字段被视为相机属性（如 camera_01_color_image_raw）
        # 不再直接从 images/ 目录枚举相机。保持排序以稳定输出顺序。
        camera_names = sorted([k for k in (features_list[0].keys() if len(features_list)>0 else []) if str(k).startswith('cam')])
        print(f"[INFO] 从 meta 字段检测到相机: {camera_names}")
        # 预备一个可选的回退查找（如果 meta 中没有有效的第一帧路径），使用旧的 images/ 目录结构查找
        images_root = episode_dir / "images"
        cam_dirs = {}
        for cam in camera_names:
            # 尝试使用 meta 中第一帧的路径（若提供）作为相机目录/文件基准
            first_meta_path = None
            if len(features_list) > 0 and cam in features_list[0]:
                p = Path(features_list[0][cam])
                if not p.is_absolute():
                    p = episode_dir / p
                if p.exists():
                    # 如果 meta 给的是具体文件，则使用其父目录作为 cam_dir
                    first_meta_path = p
                    cam_dirs[cam] = p.parent
                    continue
            # 回退：在 images_root 中查找相应目录（与以前行为一致）
            cam_root = images_root / cam
            if cam_root.exists() and cam_root.is_dir():
                first_frame = cam_root / f"frame_{0:06d}.png"
                if first_frame.exists():
                    cam_dirs[cam] = cam_root
                    continue
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
            else:
                # images_root 中无该目录且 meta 也未提供有效路径，标记为 None，后续 per-frame 将使用 meta 的 frame 路径或 None
                cam_dirs[cam] = None
        # 只读取每个相机的第一帧图片用于推断 shape，无需全部加载
        img_shapes = {}
        for cam in camera_names:
            cam_dir = cam_dirs.get(cam)
            # 优先尝试使用 meta 中第一帧指定的路径（如果存在），否则使用 cam_dir 下的 frame_000000.png
            img_path = None
            if len(features_list) > 0 and cam in features_list[0]:
                p = Path(features_list[0][cam])
                if not p.is_absolute():
                    p = episode_dir / p
                if p.exists():
                    img_path = p
            if img_path is None and cam_dir is not None:
                img_path = cam_dir / f"frame_{0:06d}.png"
            if img_path is None or not img_path.exists():
                raise FileNotFoundError(f"无法为相机 {cam} 找到用于推断 shape 的第一帧图片: {img_path}")
            # 使用 PIL 获取尺寸和通道数，但不要把图片读成 numpy.ndarray
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
            if mode == 'L':
                channels = 1
            elif mode == 'RGBA':
                channels = 4
            else:
                channels = 3
            img_shapes[cam] = (h, w, channels)
        # 构造 features，自动处理 joints -> observation.state & action
        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

            """
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=lerobot_root,
                use_videos=use_videos,
            )
            Raises:
                FileNotFoundError: _description_
            """
        if dataset is None:
            features = {}
            # 如果 frames 中包含 joints，则构造 observation.state 与 action
            if len(features_list) > 0 and "joints" in features_list[0] and features_list[0]["joints"]:
                # 计算总关节数（所有 joint entries 的 position 长度之和）
                first_joints = features_list[0]["joints"]
                total_joints = sum(len(j.get("position", [])) for j in first_joints)
                # observation.state 包含 positions, velocities, efforts 串联
                features["observation.state"] = {"dtype": "float32", "shape": [total_joints * 3]}
                # action 使用所有 joint 的 position 串联
                features["action"] = {"dtype": "float32", "shape": [total_joints]}
                # 加一个feature: effort，暂时不去除state里的effort
                features["effort"] = {"dtype": "float32", "shape": [total_joints]}
                # gripper（如有），按与 position 并行的维度保存
                first_grippers = []
                for j in first_joints:
                    g_val = j.get("gripper")
                    if g_val is None:
                        continue
                    g_arr = np.array(g_val, dtype=np.float32).reshape(-1)
                    first_grippers.extend(g_arr.tolist())
                if first_grippers:
                    gripper_dim = len(first_grippers)
                    features["gripper"] = {"dtype": "float32", "shape": [gripper_dim]}
            # 如果存在 observation dict，兼容展开其他字段
            if len(features_list) > 0 and "observation" in features_list[0]:
                obs_flat = flatten_dict(features_list[0]["observation"], parent_key="observation")
                for k, v in obs_flat.items():
                    arr = np.array(v)
                    dtype = str(arr.dtype)
                    if dtype.startswith("float"):
                        dtype = "float32"
                    shape = list(arr.shape)
                    if dtype in ("int64", "float32") and (shape == [] or shape is None):
                        shape = [1]
                    features[k] = {"dtype": dtype, "shape": shape}
            # 相机字段
            for cam in camera_names:
                img_shape = img_shapes[cam]
                features[cam] = {
                    "dtype": "video" if use_videos else "image",
                    "shape": list(img_shape),
                    "names": ["height", "width", "channels"],
                }
            print(f"[INFO] 自动构造 features: {features}")
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=lerobot_root,
                use_videos=use_videos,
            )
        # 确定 episode_index：resume 时追加在已有集数之后，否则沿用目录编号
        if resume:
            if next_episode_index is None:
                next_episode_index = dataset.meta.total_episodes
            episode_index = next_episode_index
            next_episode_index += 1
        else:
            episode_index = int(episode_dir.name.split('_')[-1])
        episode_index_map[episode_index] = episode_dir
        if first_ep_idx is None:
            first_ep_idx = episode_index
        last_ep_idx = episode_index
        # 构造 frames 列表，每帧一个 dict，包含所有必需字段
        frames = []
        for i in range(batch_size):
            meta = {
                "episode_index": episode_index,
                "frame_index": i,
                "index": i,  # 通常 index == frame_index
                "timestamp": float(i) / fps,
                "task_index": 0,  # 默认 0，可根据需要调整
                "task": task,
            }
            # 相机图片字段（如 camera_01_color_image_raw）
            # 按要求：每帧直接使用 meta（features_list）中对应相机属性的值作为图片路径，不再假设按 frame 连续
            for cam in camera_names:
                img_path = None
                # 如果 meta 明确提供了路径，则必须存在，否则立即报错
                if i < len(features_list) and cam in features_list[i] and features_list[i][cam]:
                    p = Path(features_list[i][cam])
                    if not p.is_absolute():
                        p = episode_dir / p
                    if not p.exists():
                        raise FileNotFoundError(f"meta 中为相机 {cam} 指定的图片不存在: episode={episode_index} frame={i} path={p}")
                    img_path = p
                # 如果 meta 未提供该字段，保留为 None（或根据需要改为抛错）
                meta[cam] = str(img_path) if img_path is not None else None
            # joints/tactiles（如有）——将 joints 的 position/velocity/effort 拼接到 observation.state；action 为 position
            joints_list = features_list[i].get("joints", []) if i < len(features_list) else []
            if joints_list:
                positions = []
                velocities = []
                efforts = []
                grippers = []
                for j in joints_list:
                    pos = list(j.get("position", []))
                    vel = list(j.get("velocity", []))
                    eff = list(j.get("effort", []))
                    g_val = j.get("gripper")
                    if g_val is not None:
                        g_arr = np.array(g_val, dtype=np.float32).reshape(-1)
                        grippers.extend(g_arr.tolist())
                    positions.extend(pos)
                    velocities.extend(vel)
                    efforts.extend(eff)
                # observation.state = [positions..., velocities..., efforts...]
                obs_state = np.array(positions + velocities + efforts, dtype=np.float32)
                act = np.array(positions, dtype=np.float32)
                meta["observation.state"] = obs_state
                meta["action"] = act
                meta["effort"] = np.array(efforts, dtype=np.float32)
                if grippers:
                    if gripper_dim is not None and len(grippers) != gripper_dim:
                        raise ValueError(f"episode {episode_index} frame {i} gripper dim mismatch: expected {gripper_dim}, got {len(grippers)}")
                    meta["gripper"] = np.array(grippers, dtype=np.float32)
            # tactiles（如有）
            if "tactiles" in features_list[i]:
                meta["tactiles"] = features_list[i]["tactiles"]
            # gripper（顶层字段，若未在 joints 中解析）
            if "gripper" in features_list[i] and "gripper" not in meta:
                g_arr = np.array(features_list[i]["gripper"], dtype=np.float32).reshape(-1)
                if gripper_dim is not None and len(g_arr) != gripper_dim:
                    raise ValueError(f"episode {episode_index} frame {i} gripper dim mismatch: expected {gripper_dim}, got {len(g_arr)}")
                meta["gripper"] = g_arr
            # 兼容原有 observation/action 字段
            if "action" in features_list[i] and "action" not in meta:
                meta["action"] = features_list[i]["action"]
            if "observation" in features_list[i]:
                obs_flat = flatten_dict(features_list[i]["observation"], parent_key="observation")
                meta.update(obs_flat)
            frames.append(meta)
        # 为 save_episode 构造 episode_data（使用 frames 列表包装并包含 size/task/episode_index）
        # frames 已构造好，接下来把 frames 转换为 save_episode 接受的 episode_data（按 feature 聚合）
        # 保持相机字段为图片路径字符串，交由 dataset / compute_stats 在需要时读取。
        # 不要在这里把路径读成 numpy.ndarray（会导致后续 PIL.open 报错）。
        # 使用 dataset.meta.features 来决定需要哪些 keys
        feat_keys = list(dataset.meta.features.keys())
        episode_data = {
            "size": batch_size,
            "task": [task] * batch_size,
            "episode_index": episode_index,
        }
        # 为每个 feature 生成按帧的列表
        for key in feat_keys:
            # episode_index 是 scalar（dataset 要求），不要覆盖
            if key == "episode_index":
                continue
            vals = []
            for f in frames:
                if key in f:
                    vals.append(f[key])
                else:
                    # 补默认值
                    if key == "timestamp":
                        vals.append(f.get("timestamp", float(0)))
                    elif key == "frame_index":
                        vals.append(f.get("frame_index", None))
                    elif key == "index":
                        vals.append(f.get("index", None))
                    elif key == "task_index":
                        vals.append(0)
                    else:
                        vals.append(None)
            episode_data[key] = vals

        # Ensure task_index field exists as a per-frame list
        if "task_index" not in episode_data:
            episode_data["task_index"] = [0] * batch_size

        # 存储 episode，不编码视频
        t4 = time.time()
        dataset.save_episode(episode_data=episode_data, encode_videos=False)
        t5 = time.time()
        print(f"[INFO] Saved episode {episode_index} to {lerobot_root}，耗时: {t5-t4:.3f}s")
    # exit(1)
    # 分批顺序编码视频
    print(f"[INFO] 开始分批编码视频，每批 {batch_encode_num} 个episode")
    for start in range(first_ep_idx, last_ep_idx + 1, batch_encode_num):
        end = min(start + batch_encode_num, last_ep_idx + 1)
        print(f"[INFO] 批量编码: episodes {start} ~ {end - 1}")
        batch_t0 = time.time()
        # 只为本批次 episode 建软链接
        for ep_idx in range(start, end):
            ep_dir = episode_index_map[ep_idx]
            # 使用 meta.jsonl 中每帧的相机字段路径来创建目标目录下按帧命名的软链接，确保编码器看到连续的 frame_000000.png.. 文件
            meta_path = ep_dir / target_meta_file_name
            with open(meta_path, "r") as f:
                features_list_ep = [json.loads(line) for line in f]
            # 从 meta 中读取相机字段名（以 'cam' 开头）以保持一致
            camera_names_ep = sorted([k for k in (features_list_ep[0].keys() if len(features_list_ep)>0 else []) if str(k).startswith('cam')])
            for cam in camera_names_ep:
                target_dir = Path(lerobot_root) / "images" / cam / f"episode_{ep_idx:06d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                # 对每一帧，根据 meta 中该相机字段给出的路径创建名为 frame_{i:06d}.png 的软链接
                for i, frame_meta in enumerate(features_list_ep):
                    if cam not in frame_meta or not frame_meta[cam]:
                        # meta 未给出该帧的相机路径，跳过并记录警告
                        print(f"[WARN] episode {ep_idx} frame {i} missing camera field {cam}, skip symlink")
                        continue
                    src_p = Path(frame_meta[cam])
                    if not src_p.is_absolute():
                        src_p = ep_dir / src_p
                    if not src_p.exists():
                        print(f"[WARN] episode {ep_idx} frame {i} camera {cam} path does not exist: {src_p}")
                        continue
                    dst_img = target_dir / f"frame_{i:06d}.png"
                    if dst_img.exists():
                        continue
                    try:
                        os.symlink(src_p, dst_img)
                    except FileExistsError:
                        pass
        # 编码
        dataset.batch_encode_videos(start, end)
        batch_t1 = time.time()
        print(f"[INFO] 本批次编码耗时: {batch_t1-batch_t0:.2f}s")
    print(f"[INFO] 批量编码完成，总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    # 示例用法
    lerobot_root = "/home/hyc/JE_robot_data_lerobot/"
    episodes_root = "/home/hyc/JE_robot_data/"  # 传入包含多个episode_xxxxxx的根目录
    task = "Put the purple carton of milk into the cardboard box."
    resume = False
    target_meta_file_name = "meta.jsonl"
    convert_expand_to_lerobot_batch(episodes_root, lerobot_root, task, resume=resume,target_meta_file_name = target_meta_file_name)
