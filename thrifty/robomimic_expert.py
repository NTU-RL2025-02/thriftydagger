import json

import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils


class RobomimicExpert:
    """
    載入 robomimic checkpoint，提供一個 expert_policy(obs) 介面，
    可以直接丟給 thriftydagger 使用。
    """

    def __init__(self, ckpt_path, device="cuda"):
        # 從 checkpoint 載入 policy 和 config dict
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=device, verbose=True
        )

        # ckpt_dict["config"] 在新版 robomimic 通常是一個 JSON 字串
        raw_cfg = self.ckpt_dict["config"]
        algo_name = self.ckpt_dict.get("algo_name", None)

        if isinstance(raw_cfg, str):
            cfg_dict = json.loads(raw_cfg)
        else:
            # 可能已經是 dict
            cfg_dict = raw_cfg

        if algo_name is None and isinstance(cfg_dict, dict):
            algo_name = cfg_dict.get("algo_name", "bc")

        # 建成 robomimic 的 Config 物件
        config = config_factory(algo_name, dic=cfg_dict)

        # 用這個 config 初始化 ObsUtils（告訴 robomimic 哪些 key 是 low_dim / rgb 等）
        ObsUtils.initialize_obs_utils_with_config(config)

    def __call__(self, obs):
        import torch

        # 如果 env 回傳的是 (obs, reward, done, info) 這種 tuple，就只拿 obs
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        # ---------- 情況一：obs 是 dict ----------
        if isinstance(obs, dict):
            # 只保留 policy 比較有可能真的用到、而且不會炸掉的 key
            # 注意：刻意把 "object" 排除掉，避免 KeyError('object')
            allowed_keys = {
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            }

            filtered = {}
            for k, v in obs.items():
                if k not in allowed_keys:
                    continue
                arr = np.asarray(v)
                if arr.ndim == 1:
                    arr = arr[None, :]      # 加 batch 維度 -> (1, dim)
                filtered[k] = arr

            # 萬一上面一句一個都沒抓到，就退而求其次：全部 flatten 成一個向量
            if len(filtered) == 0:
                flat = np.concatenate(
                    [np.asarray(v).ravel() for v in obs.values()]
                ).astype(np.float32)[None, :]   # (1, D)
                filtered = {"obs": flat}

            input_obs = filtered

        # ---------- 情況二：obs 不是 dict（例如是一個純向量） ----------
        else:
            arr = np.asarray(obs)
            if arr.ndim == 1:
                arr = arr[None, :]   # (1, dim)
            input_obs = {"obs": arr}

        # 丟進 robomimic policy
        with torch.no_grad():
            act = self.policy(ob=input_obs)

        # 轉 numpy
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        else:
            act = np.asarray(act, dtype=np.float32)

        # 如果是 (1, act_dim)，把 batch 維度去掉 -> (act_dim,)
        if act.ndim > 1 and act.shape[0] == 1:
            act = act[0]

        return act
