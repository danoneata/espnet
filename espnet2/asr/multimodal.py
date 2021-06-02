from abc import ABC
from abc import abstractmethod

from typing import Dict, Tuple

import torch
import torchvision.models as models  # type: ignore

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from g_mlp_pytorch import Residual, PreNorm, gMLPBlock


# Encoders for the visual channel.
class AbsEncoderVisual(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Resnet18(AbsEncoderVisual):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self._output_size = 512

    def forward(self, image):
        enc = self.feature_extractor(image)
        enc = enc.squeeze(-1).squeeze(-1)
        return enc

    def output_size(self):
        return self._output_size


class Resnet50(AbsEncoderVisual):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self._output_size = 2048

    def forward(self, image):
        enc = self.feature_extractor(image)
        enc = enc.squeeze(-1).squeeze(-1)
        return enc

    def output_size(self):
        return self._output_size


class ResnetGMLP(AbsEncoderVisual):
    def __init__(self, num_gmlp_layers=1, num_resnet_layers=18, use_pos_emb=False):
        super().__init__()
        if num_resnet_layers == 18:
            resnet_model = models.resnet18(pretrained=True)
            self._output_size = 512
        elif resnet_type == 50:
            resnet_model = models.resnet50(pretrained=True)
            self._output_size = 2048

        dim = self._output_size
        dim_ff = dim // 2
        self.seq_len = 7 * 7

        self.feature_extractor = torch.nn.Sequential(
            *(list(resnet_model.children())[:-2])
        )

        if num_gmlp_layers > 0:
            self.gmlp = torch.nn.Sequential(
                *[
                    Residual(
                        PreNorm(
                            dim,
                            gMLPBlock(
                                dim=dim,
                                dim_ff=dim_ff,
                                seq_len=self.seq_len,
                            ),
                        )
                    )
                    for i in range(num_gmlp_layers)
                ]
            )
        else:
            self.gmlp = None

    def forward(self, image):
        enc = self.feature_extractor(image)
        enc = enc.view(enc.shape[0], enc.shape[1], -1)
        enc = enc.permute(0, 2, 1)
        if self.gmlp is not None:
            enc = self.gmlp(enc)
        # enc = enc.permute(0, 2, 1)
        return enc

    def output_size(self):
        return self._output_size


# Methods of fusing the speech and visual encodings.
class AbsFeatureFuser(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_temporal_concat = False

    @abstractmethod
    def forward(
        self, enc_speech: torch.Tensor, enc_visual: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class SelectSpeech(AbsFeatureFuser):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, speech, visual):
        return speech


class ConcatProjFuser(AbsFeatureFuser):
    def __init__(self, dim_speech, dim_visual, use_residual=True):
        super().__init__()
        d_in = dim_speech + dim_visual
        d_out = dim_speech
        self.proj = torch.nn.Conv1d(d_in, d_out, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, speech, visual):
        _, T, _ = speech.shape

        # concatenate
        visual = visual.unsqueeze(1).repeat(1, T, 1)
        out = torch.cat((speech, visual), dim=2)

        # project
        out = out.permute(0, 2, 1)
        out = self.proj(out)
        out = out.permute(0, 2, 1)

        # (optional) residual connection
        if self.use_residual:
            out = out + speech

        return out


class ProjConcatFuser(AbsFeatureFuser):
    def __init__(self, dim_speech, dim_visual, p_speech=0.5, use_residual=True):
        """The argument `p_speech` indicates the proportion of speech in the
        output dimensionality.

        """
        super().__init__()
        assert 0 < p_speech < 1
        dim_speech_out = max(1, min(int(dim_speech * p_speech), dim_speech - 1))
        dim_visual_out = dim_speech - dim_speech_out
        self.proj_speech = torch.nn.Conv1d(dim_speech, dim_speech_out, kernel_size=1)
        self.proj_visual = torch.nn.Conv1d(dim_visual, dim_visual_out, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, speech, visual):
        _, T, _ = speech.shape

        # project speech
        speech_out = speech.permute(0, 2, 1)
        speech_out = self.proj_speech(speech_out)
        speech_out = speech_out.permute(0, 2, 1)

        # project visual
        visual_out = visual.unsqueeze(2)
        visual_out = self.proj_visual(visual_out)
        visual_out = visual_out.permute(0, 2, 1)
        visual_out = visual_out.repeat(1, T, 1)

        # concatenate
        out = torch.cat((speech_out, visual_out), dim=2)

        # (optional) residual connection
        if self.use_residual:
            out = out + speech

        return out


class ProjConcatProjFuser(AbsFeatureFuser):
    def __init__(
        self,
        dim_speech,
        dim_visual,
        dim_speech_inter=128,
        dim_visual_inter=128,
        use_layer_norm=True,
    ):
        super().__init__()
        if use_layer_norm:
            self.norm_speech = torch.nn.LayerNorm(dim_speech)
            self.norm_visual = torch.nn.LayerNorm(dim_visual)
        else:
            self.norm_speech = torch.nn.Identity()
            self.norm_visual = torch.nn.Identity()
        self.proj_speech = torch.nn.Conv1d(dim_speech, dim_speech_inter, kernel_size=1)
        self.proj_visual = torch.nn.Conv1d(dim_visual, dim_visual_inter, kernel_size=1)
        self.proj_back = torch.nn.Conv1d(
            dim_speech_inter + dim_visual_inter, dim_speech, kernel_size=1
        )
        self.activation = torch.nn.GELU()

    def forward(self, speech, visual):
        _, T, _ = speech.shape

        # project speech
        speech_out = self.norm_speech(speech)
        speech_out = speech_out.permute(0, 2, 1)
        speech_out = self.proj_speech(speech_out)

        # project visual
        visual_out = self.norm_visual(visual)
        visual_out = visual_out.unsqueeze(2)
        visual_out = self.proj_visual(visual_out)
        visual_out = visual_out.repeat(1, 1, T)

        # concatenate
        out = torch.cat((speech_out, visual_out), dim=1)
        out = self.activation(out)
        out = self.proj_back(out)
        out = out.permute(0, 2, 1)

        out = out + speech

        return out


class ConcatTemp(AbsFeatureFuser):
    def __init__(self, dim_speech, dim_visual):
        super().__init__()
        self.is_temporal_concat = True
        self.norm_visual = torch.nn.LayerNorm(dim_visual)
        self.proj_visual = torch.nn.Conv1d(dim_visual, dim_speech, kernel_size=1)
        assert dim_speech == dim_visual

    def forward(self, speech, visual):
        visual_out = self.norm_visual(visual)
        visual_out = visual_out.permute(0, 2, 1)
        visual_out = self.proj_visual(visual_out)
        visual_out = visual_out.permute(0, 2, 1)
        out = torch.cat((visual_out, speech), dim=1)
        return out


# The multimodal model
class ESPnetASRMultimodalModel(AbsESPnetModel):
    def __init__(
        self,
        asr: ESPnetASRModel,
        encoder_visual,
        feature_fuser,
    ):
        super().__init__()
        self.asr = asr
        self.encoder_visual = encoder_visual
        self.feature_fuser = feature_fuser

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == visual.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            visual.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, visual)

        # 2a. Attention-decoder branch
        if self.asr.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self.asr._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2b. CTC branch
        if self.asr.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self.asr._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.asr.rnnt_decoder is not None:
            _ = self.asr._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        if self.asr.ctc_weight == 0.0:
            loss = loss_att
        elif self.asr.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.asr.ctc_weight * loss_ctc + (1 - self.asr.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        visual: torch.Tensor,
    ):
        encoder_speech_out, encoder_out_lens = self.asr.encode(speech, speech_lengths)
        encoder_visual_out = self.encoder_visual(visual)
        encoder_out = self.feature_fuser(encoder_speech_out, encoder_visual_out)
        if self.feature_fuser.is_temporal_concat:
            encoder_out_lens += self.encoder_visual.seq_len
        return encoder_out, encoder_out_lens

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.asr.collect_feats(speech, speech_lengths, text, text_lengths)
