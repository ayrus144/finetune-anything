# @copyright ziqi-jin

import torch.nn as nn
import torch
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
from .segment_anything_ori.modeling.mask_decoder import MaskDecoder
from typing import List, Tuple
from torch.nn import functional as F
from .mask_decoder_heads import SemSegHead
from .mask_decoder_neck import MaskDecoderNeck


class BaseMaskDecoderAdapter(MaskDecoder):
    '''
      multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    '''

    # is fix and load params
    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseMaskDecoderAdapter, self).__init__(transformer_dim=ori_sam.mask_decoder.transformer_dim,
                                                     transformer=ori_sam.mask_decoder.transformer)
        self.sam_mask_decoder = ori_sam.mask_decoder
        if fix:
            fix_params(self.sam_mask_decoder)  # move to runner to implement

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(image_embeddings=image_embeddings,
                                                                   image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                   sparse_prompt_embeddings=sparse_embeddings,
                                                                   dense_prompt_embeddings=dense_embeddings,
                                                                   multimask_output=multimask_output, )
        return low_res_masks, iou_predictions


class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False, class_num=20):
        self.class_num = class_num
        super(SemMaskDecoderAdapter, self).__init__(ori_sam, fix)
        self.decoder_neck = MaskDecoderNeck(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                            transformer=self.sam_mask_decoder.transformer,
                                            num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs)
        self.decoder_head = SemSegHead(transformer_dim=self.sam_mask_decoder.transformer_dim,
                                       num_multimask_outputs=self.sam_mask_decoder.num_multimask_outputs,
                                       iou_head_depth=self.sam_mask_decoder.iou_head_depth,
                                       iou_head_hidden_dim=self.sam_mask_decoder.iou_head_hidden_dim,
                                       class_num=class_num)
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck)
        self.pair_params(self.decoder_head)

    def forward(self, image_embeddings, prompt_adapter, sparse_embeddings, dense_embeddings, multimask_output=True,
                scale=1):
        src, iou_token_out, mask_tokens_out, src_shape = self.decoder_neck(image_embeddings=image_embeddings,
                                                                           image_pe=prompt_adapter.sam_prompt_encoder.get_dense_pe(),
                                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                                           dense_prompt_embeddings=dense_embeddings,
                                                                           multimask_output=multimask_output, )
        masks, iou_pred = self.decoder_head(src, iou_token_out, mask_tokens_out, src_shape, mask_scale=scale)
        return masks, iou_pred

    def pair_params(self, target_model: nn.Module):
        src_dict = self.sam_mask_decoder.state_dict()
        found_groups = {
            # these group architectures are different from original SAM
            # hence, requires appropriate initializations
            'output_hypernetworks_mlps' : 0,
            'iou_prediction_head' : 0,
        }
        # Either copy state or update found_groups counter
        for name, value in target_model.named_parameters():
            if name in src_dict.keys():
                group = name.split('.')[0]
                if group in found_groups.keys():
                    found_groups[group] += 1
                else:
                    value.data.copy_(src_dict[name].data)
        # Keep only found groups
        del_keys = []
        for group, counter in found_groups.items():
            if counter==0:
                del_keys.append(group)
        for key in del_keys:
            found_groups.pop(key)
        # copy state differently for found groups
        for name, value in target_model.named_parameters():
            if name.startswith('output_hypernetworks_mlps.') and 'output_hypernetworks_mlps' in found_groups:
                # idea - initialize all mpls with same weight/bias as mlp for mask_scale = 0
                # example name: output_hypernetworks_mlps.0.layers.2.bias
                group, mpl_num, layer_detail = name.split('.', 2)
                src_name = group + '.' + '0' + '.' + layer_detail # corresponds to mask_scale 0
                value.data.copy_(src_dict[src_name].data)
            if name.startswith('iou_prediction_head.layers.2') and 'iou_prediction_head' in found_groups:
                # idea - initialize by expanding original (4,...) tensor to (num_classes,...) shaped tensor
                # Similar to mlp, but initialize only the last_layer weight(matrix)/bias(vector) [for mask_scale 0-3]
                # with same weight(vector)/bias(scalar) values corresponding to iou_prediction output for mask_scale = 0
                # expanded_src_data = src_dict[name].data.repeat_interleave(3, dim=0)
                src_data = src_dict[name].data[0:1] # corresponds to mask_scale 0
                expanded_src_data = src_data.repeat_interleave(self.class_num, dim=0)
                value.data.copy_(expanded_src_data)

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
