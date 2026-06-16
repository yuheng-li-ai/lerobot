#!/usr/bin/env python
"""ACT training harness with CALVIN language-conditioning support."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision
import yaml
from dataclasses import dataclass
from itertools import chain

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.configs.types import PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

try:
    from train_act import (  # noqa: E402
        ImageBatchAugmenter,
        _append_metrics,
        _apply_imagenet_stats,
        _format_progress,
        _load_config,
        _make_loader,
        _move_uint8_images_to_float,
        _next_batch,
        _parse_episode_spec,
        _save_checkpoint,
        _set_seed,
        _validate,
        _write_metrics_header,
    )
except ModuleNotFoundError:
    from project.scripts.train_act import (  # noqa: E402
        ImageBatchAugmenter,
        _append_metrics,
        _apply_imagenet_stats,
        _format_progress,
        _load_config,
        _make_loader,
        _move_uint8_images_to_float,
        _next_batch,
        _parse_episode_spec,
        _save_checkpoint,
        _set_seed,
        _validate,
        _write_metrics_header,
    )


@PreTrainedConfig.register_subclass("act_lang")
@dataclass
class ACTLangConfig(ACTConfig):
    """ACT config extended with one language-conditioning token."""

    language_embedding_key: str = "observation.language_embedding"
    language_embedding_dim: int = 384
    language_fusion: str = "conditioning_token"
    language_projection_dim: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.language_fusion != "conditioning_token":
            raise ValueError(f"Unsupported language_fusion={self.language_fusion!r}")
        if self.language_projection_dim != self.dim_model:
            raise ValueError(
                "This harness keeps the language token in ACT dim_model space; "
                f"got language_projection_dim={self.language_projection_dim}, dim_model={self.dim_model}."
            )
        self.normalization_mapping.setdefault("LANGUAGE", NormalizationMode.IDENTITY)

    def validate_features(self) -> None:
        super().validate_features()
        if self.language_embedding_key not in self.input_features:
            raise ValueError(f"Missing language feature {self.language_embedding_key!r} in input_features")
        language_feature = self.input_features[self.language_embedding_key]
        if language_feature.type != FeatureType.LANGUAGE:
            raise ValueError(
                f"{self.language_embedding_key!r} must be FeatureType.LANGUAGE, got {language_feature.type}"
            )
        if tuple(language_feature.shape) != (self.language_embedding_dim,):
            raise ValueError(
                f"{self.language_embedding_key!r} shape must be {(self.language_embedding_dim,)}, "
                f"got {language_feature.shape}"
            )


class ACTLangPolicy(PreTrainedPolicy):
    """ACT policy variant that conditions the decoder on a CALVIN language embedding."""

    config_class = ACTLangConfig
    name = "act_lang"

    def __init__(self, config: ACTLangConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTLang(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self) -> None:
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTLang(nn.Module):
    """ACT network with one projected language token in the transformer encoder."""

    def __init__(self, config: ACTLangConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_language_input_proj = nn.Linear(config.language_embedding_dim, config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)

        n_1d_tokens = 2  # latent + language
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.encoder_language_input_proj.weight)
        nn.init.zeros_(self.encoder_language_input_proj.bias)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."
        if self.config.language_embedding_key not in batch:
            raise KeyError(f"Batch is missing {self.config.language_embedding_key!r}")

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = batch[OBS_STATE].device if OBS_STATE in batch else batch[self.config.language_embedding_key].device

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, dim=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        language_embedding = batch[self.config.language_embedding_key]
        if language_embedding.ndim != 2 or language_embedding.shape[-1] != self.config.language_embedding_dim:
            raise ValueError(
                f"Expected language tensor shape (B, {self.config.language_embedding_dim}), "
                f"got {tuple(language_embedding.shape)}"
            )
        encoder_in_tokens.append(self.encoder_language_input_proj(language_embedding.float()))

        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        actions = self.action_head(decoder_out.transpose(0, 1))
        return actions, (mu, log_sigma_x2)


def _build_act_lang_config(policy_cfg: dict[str, Any]) -> ACTLangConfig:
    kwargs = dict(policy_cfg)
    policy_type = kwargs.pop("type", "act_lang")
    if policy_type != "act_lang":
        raise ValueError(f"Only act_lang is supported by this harness, got {policy_type!r}")
    return ACTLangConfig(**kwargs)


def _attach_dataset_features(policy_cfg: ACTLangConfig, ds_meta: LeRobotDatasetMetadata) -> None:
    if policy_cfg.language_embedding_key not in ds_meta.features:
        raise ValueError(
            f"Dataset {ds_meta.repo_id} does not contain {policy_cfg.language_embedding_key!r}; "
            "it cannot support official language-conditioned CALVIN evaluation."
        )

    policy_features = dataset_to_policy_features(ds_meta.features)
    raw_language_shape = tuple(ds_meta.features[policy_cfg.language_embedding_key]["shape"])
    policy_features[policy_cfg.language_embedding_key] = PolicyFeature(
        type=FeatureType.LANGUAGE,
        shape=raw_language_shape,
    )
    if ACTION not in policy_features:
        raise ValueError("Dataset is missing action feature")

    policy_cfg.input_features = {key: ft for key, ft in policy_features.items() if key != ACTION}
    policy_cfg.output_features = {ACTION: policy_features[ACTION]}
    policy_cfg.normalization_mapping["LANGUAGE"] = NormalizationMode.IDENTITY
    policy_cfg.validate_features()


def _make_dataset(
    repo_id: str,
    root: str,
    episodes: list[int],
    policy_cfg: ACTLangConfig,
    *,
    return_uint8: bool,
    video_backend: str | None,
) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=return_uint8,
        video_backend=video_backend,
    )


@torch.no_grad()
def _language_minibatch_check(
    policy: ACTLangPolicy,
    preprocessor,
    train_loader: torch.utils.data.DataLoader,
    camera_keys: list[str],
) -> dict[str, Any]:
    raw_batch = next(iter(train_loader))
    language_key = policy.config.language_embedding_key
    checks: dict[str, Any] = {
        "language_key": language_key,
        "raw_language_key_exists": language_key in raw_batch,
    }
    if language_key not in raw_batch:
        checks["passed"] = False
        return checks

    checks["raw_language_shape"] = list(raw_batch[language_key].shape)
    checks["raw_action_shape"] = list(raw_batch[ACTION].shape)
    checks["raw_language_finite"] = bool(torch.isfinite(raw_batch[language_key]).all().item())

    batch = _move_uint8_images_to_float(raw_batch, camera_keys)
    batch = preprocessor(batch)
    checks["processed_language_shape"] = list(batch[language_key].shape)
    checks["processed_language_device"] = str(batch[language_key].device)
    checks["processed_action_shape"] = list(batch[ACTION].shape)

    was_training = policy.training
    policy.eval()
    original_actions = policy.predict_action_chunk(batch)
    zero_language_batch = dict(batch)
    zero_language_batch[language_key] = torch.zeros_like(batch[language_key])
    zero_language_actions = policy.predict_action_chunk(zero_language_batch)
    language_sensitivity_l1 = (original_actions - zero_language_actions).abs().mean().item()
    policy.train(was_training)

    checks["predicted_action_chunk_shape"] = list(original_actions.shape)
    checks["language_sensitivity_l1"] = language_sensitivity_l1
    checks["expected_language_dim"] = policy.config.language_embedding_dim
    checks["language_shape_ok"] = (
        batch[language_key].ndim == 2 and batch[language_key].shape[-1] == policy.config.language_embedding_dim
    )
    checks["action_chunk_shape_ok"] = (
        original_actions.ndim == 3
        and original_actions.shape[1] == policy.config.chunk_size
        and original_actions.shape[2] == policy.config.action_feature.shape[0]
    )
    checks["language_connected"] = bool(language_sensitivity_l1 > 1e-8)
    checks["passed"] = bool(
        checks["raw_language_key_exists"]
        and checks["raw_language_finite"]
        and checks["language_shape_ok"]
        and checks["action_chunk_shape_ok"]
        and checks["language_connected"]
    )
    return checks


def _self_check(
    output_dir: Path,
    metrics_path: Path,
    checkpoint_dir: Path,
    minibatch_checks: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {"minibatch": minibatch_checks}
    checks["metrics_csv_exists"] = metrics_path.is_file()
    rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    checks["metrics_rows"] = len(rows)
    if rows:
        train_values = [float(r["train_action_l1"]) for r in rows if r.get("train_action_l1")]
        val_values = [float(r["val_action_l1"]) for r in rows if r.get("val_action_l1")]
        finite_val_values = [v for v in val_values if math.isfinite(v)]
        checks["finite_train_action_l1"] = bool(train_values and all(math.isfinite(v) for v in train_values))
        checks["finite_val_action_l1"] = bool(finite_val_values and math.isfinite(val_values[-1]))
    else:
        checks["finite_train_action_l1"] = False
        checks["finite_val_action_l1"] = False

    checks["checkpoint_dir_exists"] = checkpoint_dir.is_dir()
    checks["policy_config_exists"] = (checkpoint_dir / "config.json").is_file()
    checks["model_weights_exists"] = any(checkpoint_dir.glob("*.safetensors")) or any(checkpoint_dir.glob("*.bin"))
    checks["preprocessor_exists"] = any(checkpoint_dir.glob("*preprocessor*.json"))
    checks["postprocessor_exists"] = any(checkpoint_dir.glob("*postprocessor*.json"))
    checks["training_state_exists"] = (checkpoint_dir / "training_state.pt").is_file()
    checks["passed"] = all(
        bool(checks[k])
        for k in [
            "metrics_csv_exists",
            "finite_train_action_l1",
            "finite_val_action_l1",
            "checkpoint_dir_exists",
            "policy_config_exists",
            "model_weights_exists",
            "preprocessor_exists",
            "postprocessor_exists",
            "training_state_exists",
        ]
    ) and bool(minibatch_checks.get("passed"))

    with (output_dir / "self_check.json").open("w") as f:
        json.dump(checks, f, indent=2)
    return checks


def _resolve_resume_checkpoint(resume_from: str | None) -> Path | None:
    if not resume_from:
        return None

    resume_path = Path(resume_from).expanduser()
    if (resume_path / "checkpoint" / "training_state.pt").is_file():
        resume_path = resume_path / "checkpoint"

    required = [
        resume_path / "config.json",
        resume_path / "model.safetensors",
        resume_path / "training_state.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Resume checkpoint is incomplete. Missing:\n  " + "\n  ".join(missing)
        )
    return resume_path


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(target_device)


def _load_resume_state(
    checkpoint_dir: Path,
    policy_cfg: ACTLangConfig,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(checkpoint_dir / "training_state.pt", map_location=policy_cfg.device)
    if "step" not in state or "optimizer" not in state:
        raise KeyError(f"{checkpoint_dir / 'training_state.pt'} must contain 'step' and 'optimizer'")

    optimizer.load_state_dict(state["optimizer"])
    _move_optimizer_state_to_device(optimizer, policy_cfg.device)
    return int(state["step"])


def _episode_summary(episodes: list[int]) -> dict[str, int | None]:
    return {
        "count": len(episodes),
        "first": episodes[0] if episodes else None,
        "last": episodes[-1] if episodes else None,
    }


def _format_resume_progress(step: int, total_steps: int, resume_step: int, elapsed_s: float) -> str:
    if resume_step <= 0:
        return _format_progress(step, total_steps, elapsed_s)

    continuation_step = step - resume_step
    continuation_total = total_steps - resume_step
    base = _format_progress(continuation_step, continuation_total, elapsed_s)
    return f"{base} total_step={step}/{total_steps} resumed_from={resume_step}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment.output_dir.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint directory, or run directory containing checkpoint/, to continue from.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device_override = os.environ.get("ACT_DEVICE_OVERRIDE")
    if device_override:
        cfg.setdefault("policy", {})["device"] = device_override
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    policy_cfg = _build_act_lang_config(cfg["policy"])
    resume_checkpoint = _resolve_resume_checkpoint(
        str(args.resume_from) if args.resume_from is not None else train_cfg.get("resume_from")
    )

    output_dir = args.output_dir if args.output_dir is not None else Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    _set_seed(int(exp_cfg["seed"]))

    repo_id = data_cfg["repo_id"]
    root = data_cfg["root"]
    video_backend = data_cfg.get("video_backend")
    return_uint8 = bool(data_cfg.get("return_uint8", True))
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    _attach_dataset_features(policy_cfg, ds_meta)
    train_episodes = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
    val_episodes = _parse_episode_spec(data_cfg["val_episodes"], ds_meta.total_episodes)

    train_dataset = _make_dataset(
        repo_id, root, train_episodes, policy_cfg, return_uint8=return_uint8, video_backend=video_backend
    )
    val_dataset = _make_dataset(
        repo_id, root, val_episodes, policy_cfg, return_uint8=return_uint8, video_backend=video_backend
    )
    if data_cfg.get("use_imagenet_stats", True):
        _apply_imagenet_stats(train_dataset)
        _apply_imagenet_stats(val_dataset)

    if resume_checkpoint is None:
        policy = ACTLangPolicy(policy_cfg)
        policy.to(policy.config.device)
    else:
        policy = ACTLangPolicy.from_pretrained(
            resume_checkpoint,
            config=policy_cfg,
            local_files_only=True,
            strict=True,
        )
    preprocessor, postprocessor = make_act_pre_post_processors(
        config=policy.config,
        dataset_stats=train_dataset.meta.stats,
    )

    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    resume_step = 0
    if resume_checkpoint is not None:
        resume_step = _load_resume_state(resume_checkpoint, policy.config, optimizer)

    train_loader = _make_loader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=False,
    )
    camera_keys = list(train_dataset.meta.camera_keys)
    augmenter = ImageBatchAugmenter(aug_cfg)
    metrics_path = output_dir / "metrics.csv"
    _write_metrics_header(metrics_path)

    steps = int(train_cfg["steps"])
    if resume_step >= steps:
        raise ValueError(
            f"Resume checkpoint step {resume_step} is not smaller than target training.steps={steps}. "
            "Set training.steps to the desired total step count."
        )
    if resume_checkpoint is not None:
        resume_info = {
            "resume_from": str(resume_checkpoint),
            "resume_step": resume_step,
            "target_total_steps": steps,
            "additional_steps": steps - resume_step,
        }
        with (output_dir / "resume_info.json").open("w") as f:
            json.dump(resume_info, f, indent=2)

    minibatch_checks = _language_minibatch_check(policy, preprocessor, train_loader, camera_keys)
    with (output_dir / "minibatch_check.json").open("w") as f:
        json.dump(minibatch_checks, f, indent=2)
    if not minibatch_checks["passed"]:
        print(json.dumps({"event": "minibatch_check_failed", **minibatch_checks}, indent=2), flush=True)
        return 2

    print(
        json.dumps(
            {
                "event": "start",
                "output_dir": str(output_dir),
                "device": policy.config.device,
                "train_episodes": _episode_summary(train_episodes),
                "val_episodes": _episode_summary(val_episodes),
                "train_frames": train_dataset.num_frames,
                "val_frames": val_dataset.num_frames,
                "camera_keys": camera_keys,
                "chunk_size": policy.config.chunk_size,
                "n_action_steps": policy.config.n_action_steps,
                "language_embedding_key": policy.config.language_embedding_key,
                "language_embedding_dim": policy.config.language_embedding_dim,
                "resume_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
                "resume_step": resume_step,
                "target_total_steps": steps,
                "additional_steps": steps - resume_step,
                "minibatch_check": minibatch_checks,
                "augmentation": augmenter.state_dict(),
            },
            indent=2,
        ),
        flush=True,
    )

    policy.train()
    val_freq = int(train_cfg["val_freq"])
    log_freq = int(train_cfg["log_freq"])
    save_freq = int(train_cfg.get("save_freq", 0))
    progress_bar = bool(train_cfg.get("progress_bar", True))
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    max_val_batches = int(train_cfg["max_val_batches"])
    last_val = {"val_loss": math.nan, "val_action_l1": math.nan}
    train_iter = iter(train_loader)
    run_start = time.perf_counter()

    for step in range(resume_step + 1, steps + 1):
        start = time.perf_counter()
        batch, train_iter = _next_batch(train_iter, train_loader)
        batch = _move_uint8_images_to_float(batch, camera_keys)
        batch = augmenter(batch, camera_keys)
        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % val_freq == 0 or step == steps:
            last_val = _validate(policy, preprocessor, val_loader, camera_keys, max_val_batches)

        row = {
            "step": step,
            "train_loss": float(loss.item()),
            "train_action_l1": float(loss_dict["l1_loss"]),
            "train_kld": float(loss_dict.get("kld_loss", math.nan)),
            "val_loss": float(last_val["val_loss"]),
            "val_action_l1": float(last_val["val_action_l1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "step_s": float(time.perf_counter() - start),
        }
        _append_metrics(metrics_path, row)
        if step % log_freq == 0:
            log_row = dict(row)
            log_row["grad_norm"] = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            print(json.dumps({"event": "step", **log_row}), flush=True)
            if progress_bar:
                print(
                    f"PROGRESS {exp_cfg['name']} "
                    f"{_format_resume_progress(step, steps, resume_step, time.perf_counter() - run_start)}",
                    flush=True,
                )

        if bool(train_cfg.get("save_checkpoint", True)) and save_freq > 0 and step % save_freq == 0 and step != steps:
            checkpoint_name = f"checkpoints/step_{step:08d}"
            _save_checkpoint(output_dir, policy, preprocessor, postprocessor, optimizer, step, checkpoint_name=checkpoint_name)

    checkpoint_dir = _save_checkpoint(output_dir, policy, preprocessor, postprocessor, optimizer, steps)
    checks = _self_check(output_dir, metrics_path, checkpoint_dir, minibatch_checks)
    print(json.dumps({"event": "self_check", **checks}, indent=2), flush=True)
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
