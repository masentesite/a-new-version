# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        use_multimodal_fusion=True,
        multimodal_fusion_type="reliability_gated",
        fusion_modalities=("infrared", "depth"),
        multimodal_fusion_dropout=0.0,
        modality_dropout_prob=0.2,
        aux_modality_loss_weight=0.3,
        alignment_loss_weight=0.05,
        fusion_attention_reduction=4,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.use_multimodal_fusion = use_multimodal_fusion
        self.multimodal_fusion_type = multimodal_fusion_type
        self.fusion_modalities = tuple(fusion_modalities)
        self.modality_dropout_prob = modality_dropout_prob
        self.aux_modality_loss_weight = aux_modality_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        if not use_multimodal_fusion:
            self.multimodal_fusion = None
        elif multimodal_fusion_type == "reliability_gated":
            self.multimodal_fusion = ReliabilityGatedFusion(
                backbone.num_channels,
                modality_names=self.fusion_modalities,
                num_heads=nheads,
                dropout=multimodal_fusion_dropout,
                attention_reduction=fusion_attention_reduction,
            )
        elif multimodal_fusion_type == "legacy_cross_attention":
            self.multimodal_fusion = MultiModalFeatureFusion(
                backbone.num_channels,
                num_heads=nheads,
                dropout=multimodal_fusion_dropout,
            )
        else:
            raise ValueError(f"unknown multimodal_fusion_type: {multimodal_fusion_type}")
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def _normalize_samples(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        return samples

    def _split_multimodal_samples(self, samples, kw):
        auxiliary_samples = {}
        if isinstance(samples, dict):
            rgb_samples = None
            for rgb_key in ("rgb", "visible", "image", "samples"):
                if rgb_key in samples:
                    rgb_samples = samples[rgb_key]
                    break
            if rgb_samples is None:
                raise KeyError("multimodal samples must include one of: rgb, visible, image, samples")

            seen_ids = set()
            for modality_key, canonical_name in (
                ("infrared", "infrared"),
                ("ir", "infrared"),
                ("thermal", "infrared"),
                ("depth", "depth"),
            ):
                modality_samples = samples.get(modality_key)
                if modality_samples is not None and id(modality_samples) not in seen_ids:
                    auxiliary_samples[canonical_name] = modality_samples
                    seen_ids.add(id(modality_samples))
            samples = rgb_samples

        explicit_auxiliary_samples = kw.get("auxiliary_samples")
        if explicit_auxiliary_samples is not None:
            if isinstance(explicit_auxiliary_samples, dict):
                for modality_name, modality_samples in explicit_auxiliary_samples.items():
                    auxiliary_samples[modality_name] = modality_samples
            elif isinstance(explicit_auxiliary_samples, (list, tuple)):
                for index, modality_samples in enumerate(explicit_auxiliary_samples):
                    modality_name = (
                        self.fusion_modalities[index]
                        if index < len(self.fusion_modalities)
                        else f"auxiliary_{index}"
                    )
                    auxiliary_samples[modality_name] = modality_samples
            else:
                auxiliary_samples["auxiliary"] = explicit_auxiliary_samples

        for modality_key, canonical_name in (
            ("infrared", "infrared"),
            ("ir", "infrared"),
            ("thermal", "infrared"),
            ("depth", "depth"),
        ):
            modality_samples = kw.get(modality_key)
            if modality_samples is not None:
                auxiliary_samples[canonical_name] = modality_samples

        samples = self._normalize_samples(samples)
        auxiliary_samples = {
            modality_name: self._normalize_samples(item)
            for modality_name, item in auxiliary_samples.items()
        }
        return samples, auxiliary_samples

    def _set_multimodal_image_tensor(self, samples, auxiliary_samples):
        rgb_features, rgb_poss = self.backbone(samples)
        self.features, self.poss = rgb_features, rgb_poss
        modality_features = {"rgb": (rgb_features, rgb_poss, samples)}
        if not self.use_multimodal_fusion or not auxiliary_samples:
            return {"modalities": modality_features, "diagnostics": {}}

        auxiliary_feature_groups = {}
        for modality_name, modality_samples in auxiliary_samples.items():
            features, poss = self.backbone(modality_samples)
            auxiliary_feature_groups[modality_name] = features
            modality_features[modality_name] = (features, poss, modality_samples)

        fusion_result = self.multimodal_fusion(
            rgb_features,
            auxiliary_feature_groups,
            training=self.training,
            modality_dropout_prob=self.modality_dropout_prob,
        )
        if isinstance(fusion_result, tuple):
            self.features, diagnostics = fusion_result
        else:
            self.features, diagnostics = fusion_result, {}
        return {"modalities": modality_features, "diagnostics": diagnostics}

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    @staticmethod
    def _copy_text_dict(text_dict):
        return {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in text_dict.items()
        }

    def _run_detection_head(self, features, poss, samples, text_dict):
        srcs = []
        masks = []
        poss = list(poss)
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for level in range(len_srcs, self.num_feature_levels):
                if level == len_srcs:
                    src = self.input_proj[level](features[-1].tensors)
                else:
                    src = self.input_proj[level](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # Keep the original optional outputs available for training pipelines that use them.
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)

        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out["interm_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

        return out

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        samples, auxiliary_samples = self._split_multimodal_samples(samples, kw)

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        multimodal_state = {"modalities": {}, "diagnostics": {}}
        if auxiliary_samples:
            multimodal_state = self._set_multimodal_image_tensor(samples, auxiliary_samples)
        elif not hasattr(self, 'features') or not hasattr(self, 'poss'):
            self.set_image_tensor(samples)
            multimodal_state["modalities"]["rgb"] = (self.features, self.poss, samples)

        out = self._run_detection_head(
            self.features, self.poss, samples, self._copy_text_dict(text_dict)
        )

        if self.training and multimodal_state["modalities"]:
            aux_outputs = {}
            active_modalities = multimodal_state["diagnostics"].get(
                "active_modalities",
                [name for name in multimodal_state["modalities"] if name != "rgb"],
            )
            for modality_name, (features, poss, modality_samples) in multimodal_state[
                "modalities"
            ].items():
                if modality_name == "rgb" or modality_name not in active_modalities:
                    continue
                aux_outputs[modality_name] = self._run_detection_head(
                    features, poss, modality_samples, self._copy_text_dict(text_dict)
                )
            out["aux_modality_outputs"] = aux_outputs
            out["fusion_diagnostics"] = multimodal_state["diagnostics"]
            out["aux_modality_loss_weight"] = self.aux_modality_loss_weight
            out["alignment_loss_weight"] = self.alignment_loss_weight
            alignment_loss = self._fusion_alignment_loss(
                self.features,
                multimodal_state["modalities"],
                active_modalities,
            )
            if alignment_loss is not None:
                out["fusion_alignment_loss"] = alignment_loss

        unset_image_tensor = kw.get('unset_image_tensor', True)
        if unset_image_tensor:
            self.unset_image_tensor() ## If necessary
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    @staticmethod
    def _fusion_alignment_loss(fused_features, modality_features, active_modalities):
        losses = []
        for modality_name in active_modalities:
            if modality_name not in modality_features:
                continue
            features, _, _ = modality_features[modality_name]
            for fused_feature, modality_feature in zip(fused_features, features):
                modality_tensor = modality_feature.tensors
                if modality_tensor.shape[-2:] != fused_feature.tensors.shape[-2:]:
                    modality_tensor = F.interpolate(
                        modality_tensor,
                        size=fused_feature.tensors.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                losses.append(
                    F.smooth_l1_loss(
                        fused_feature.tensors.mean(dim=1),
                        modality_tensor.mean(dim=1),
                        reduction="mean",
                    )
                )
        if not losses:
            return None
        return torch.stack(losses).mean()


class MultiModalFeatureFusion(nn.Module):
    """Fuse auxiliary visual modalities into RGB features with residual cross-attention."""

    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(channel),
                        "norm_kv": nn.LayerNorm(channel),
                        "attn": nn.MultiheadAttention(
                            embed_dim=channel,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                    }
                )
                for channel in channels
            ]
        )
        self.gates = nn.Parameter(torch.zeros(len(channels)))

    def forward(self, rgb_features, auxiliary_feature_groups, **kwargs):
        if not auxiliary_feature_groups:
            return rgb_features
        if isinstance(auxiliary_feature_groups, dict):
            auxiliary_feature_groups = list(auxiliary_feature_groups.values())

        fused_features = []
        for level, rgb_feature in enumerate(rgb_features):
            rgb_tokens = self._to_tokens(rgb_feature.tensors)
            kv_tokens = [
                self._to_tokens(features[level].tensors)
                for features in auxiliary_feature_groups
                if level < len(features)
            ]
            kv_padding_masks = [
                features[level].mask.flatten(1)
                for features in auxiliary_feature_groups
                if level < len(features)
            ]
            if not kv_tokens:
                fused_features.append(rgb_feature)
                continue

            kv_tokens = torch.cat(kv_tokens, dim=1)
            kv_padding_mask = torch.cat(kv_padding_masks, dim=1)
            layer = self.layers[level]
            delta, _ = layer["attn"](
                layer["norm_q"](rgb_tokens),
                layer["norm_kv"](kv_tokens),
                layer["norm_kv"](kv_tokens),
                key_padding_mask=kv_padding_mask,
                need_weights=False,
            )
            delta = self._to_feature_map(delta, rgb_feature.tensors.shape)
            fused = rgb_feature.tensors + self.gates[level].tanh() * delta
            fused_features.append(NestedTensor(fused, rgb_feature.mask))

        return fused_features

    @staticmethod
    def _to_tokens(feature_map):
        return feature_map.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_feature_map(tokens, feature_shape):
        batch_size, channels, height, width = feature_shape
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class ReliabilityGatedFusion(nn.Module):
    """Fuse RGB with auxiliary modalities using spatial reliability gates."""

    def __init__(
        self,
        channels,
        modality_names=("infrared", "depth"),
        num_heads=8,
        dropout=0.0,
        attention_reduction=4,
    ):
        super().__init__()
        self.modality_names = tuple(modality_names)
        self.attention_reduction = max(1, attention_reduction)
        self.layers = nn.ModuleList()

        for channel in channels:
            attention_heads = self._valid_num_heads(channel, num_heads)
            hidden_channels = max(channel // 4, 32)
            self.layers.append(
                nn.ModuleDict(
                    {
                        "rgb_gate": self._gate_block(channel, hidden_channels),
                        "aux_gate": self._gate_block(channel, hidden_channels),
                        "norm_q": nn.LayerNorm(channel),
                        "norm_kv": nn.LayerNorm(channel),
                        "attn": nn.MultiheadAttention(
                            embed_dim=channel,
                            num_heads=attention_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "delta": nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=1),
                            nn.GELU(),
                            nn.Conv2d(channel, channel, kernel_size=1),
                        ),
                    }
                )
            )

    @staticmethod
    def _valid_num_heads(channel, requested_heads):
        for heads in range(min(channel, requested_heads), 0, -1):
            if channel % heads == 0:
                return heads
        return 1

    @staticmethod
    def _gate_block(channel, hidden_channels):
        return nn.Sequential(
            nn.Conv2d(channel * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        rgb_features,
        auxiliary_feature_groups: Dict[str, List[NestedTensor]],
        training=False,
        modality_dropout_prob=0.0,
    ):
        active_modalities, dropped_modalities = self._select_modalities(
            auxiliary_feature_groups, training, modality_dropout_prob
        )
        diagnostics = {
            "active_modalities": active_modalities,
            "dropped_modalities": dropped_modalities,
            "gate_means": {},
            "rgb_gate_means": [],
            "aux_gate_means": {},
        }
        if not active_modalities:
            return rgb_features, diagnostics

        fused_features = []
        for level, rgb_feature in enumerate(rgb_features):
            layer = self.layers[level]
            rgb_tensor = rgb_feature.tensors
            aux_candidates = [
                (name, auxiliary_feature_groups[name][level].tensors)
                for name in active_modalities
                if level < len(auxiliary_feature_groups[name])
            ]
            if not aux_candidates:
                fused_features.append(rgb_feature)
                continue

            calibrated_aux = []
            aux_gate_logits = []
            for modality_name, aux_tensor in aux_candidates:
                if aux_tensor.shape[-2:] != rgb_tensor.shape[-2:]:
                    aux_tensor = F.interpolate(
                        aux_tensor,
                        size=rgb_tensor.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                calibrated = self._cross_calibrate(layer, rgb_feature, aux_tensor)
                calibrated_aux.append((modality_name, calibrated))
                aux_gate_logits.append(layer["aux_gate"](torch.cat([rgb_tensor, calibrated], dim=1)))

            rgb_gate_logit = layer["rgb_gate"](
                torch.cat([rgb_tensor, torch.stack([x for _, x in calibrated_aux]).mean(0)], dim=1)
            )
            gate_logits = torch.cat([rgb_gate_logit, *aux_gate_logits], dim=1)
            gates = gate_logits.softmax(dim=1)
            rgb_gate = gates[:, 0:1]
            aux_gates = gates[:, 1:]

            fused_delta = torch.zeros_like(rgb_tensor)
            for index, (modality_name, calibrated) in enumerate(calibrated_aux):
                gate = aux_gates[:, index : index + 1]
                delta = layer["delta"](calibrated - rgb_tensor)
                fused_delta = fused_delta + gate * delta

                diagnostics["gate_means"].setdefault(modality_name, []).append(
                    gate.detach().mean()
                )
                diagnostics["aux_gate_means"].setdefault(modality_name, []).append(
                    gate.detach().mean()
                )

            diagnostics["rgb_gate_means"].append(rgb_gate.detach().mean())
            fused = rgb_tensor + fused_delta
            fused_features.append(NestedTensor(fused, rgb_feature.mask))

        diagnostics["rgb_gate_means"] = self._stack_diagnostic(diagnostics["rgb_gate_means"])
        diagnostics["gate_means"] = {
            modality_name: self._stack_diagnostic(values)
            for modality_name, values in diagnostics["gate_means"].items()
        }
        diagnostics["aux_gate_means"] = {
            modality_name: self._stack_diagnostic(values)
            for modality_name, values in diagnostics["aux_gate_means"].items()
        }
        return fused_features, diagnostics

    def _cross_calibrate(self, layer, rgb_feature, aux_tensor):
        rgb_tensor = rgb_feature.tensors
        rgb_pooled = self._pool_feature(rgb_tensor)
        aux_pooled = self._pool_feature(aux_tensor)
        query = self._to_tokens(rgb_pooled)
        key_value = self._to_tokens(aux_pooled)
        calibrated_tokens, _ = layer["attn"](
            layer["norm_q"](query),
            layer["norm_kv"](key_value),
            layer["norm_kv"](key_value),
            need_weights=False,
        )
        calibrated = self._to_feature_map(calibrated_tokens, rgb_pooled.shape)
        if calibrated.shape[-2:] != rgb_tensor.shape[-2:]:
            calibrated = F.interpolate(
                calibrated,
                size=rgb_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return aux_tensor + calibrated

    def _pool_feature(self, feature_map):
        if self.attention_reduction == 1:
            return feature_map
        return F.avg_pool2d(
            feature_map,
            kernel_size=self.attention_reduction,
            stride=self.attention_reduction,
            ceil_mode=True,
        )

    def _select_modalities(self, auxiliary_feature_groups, training, modality_dropout_prob):
        available_modalities = [
            modality_name
            for modality_name in self.modality_names
            if modality_name in auxiliary_feature_groups
        ]
        available_modalities.extend(
            modality_name
            for modality_name in auxiliary_feature_groups
            if modality_name not in available_modalities
        )
        if not training or modality_dropout_prob <= 0 or len(available_modalities) <= 1:
            return available_modalities, []

        active_modalities = []
        dropped_modalities = []
        for modality_name in available_modalities:
            drop = torch.rand((), device=auxiliary_feature_groups[modality_name][0].tensors.device)
            if drop.item() < modality_dropout_prob:
                dropped_modalities.append(modality_name)
            else:
                active_modalities.append(modality_name)

        if not active_modalities:
            recovered = dropped_modalities.pop(0)
            active_modalities.append(recovered)

        return active_modalities, dropped_modalities

    @staticmethod
    def _to_tokens(feature_map):
        return feature_map.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_feature_map(tokens, feature_shape):
        batch_size, channels, height, width = feature_shape
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

    @staticmethod
    def _stack_diagnostic(values):
        if not values:
            return torch.empty(0)
        return torch.stack(values)


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        use_multimodal_fusion=getattr(args, "use_multimodal_fusion", True),
        multimodal_fusion_type=getattr(args, "multimodal_fusion_type", "reliability_gated"),
        fusion_modalities=getattr(args, "fusion_modalities", ("infrared", "depth")),
        multimodal_fusion_dropout=getattr(args, "multimodal_fusion_dropout", 0.0),
        modality_dropout_prob=getattr(args, "modality_dropout_prob", 0.2),
        aux_modality_loss_weight=getattr(args, "aux_modality_loss_weight", 0.3),
        alignment_loss_weight=getattr(args, "alignment_loss_weight", 0.05),
        fusion_attention_reduction=getattr(args, "fusion_attention_reduction", 4),
    )

    return model
