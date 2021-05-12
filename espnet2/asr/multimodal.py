from abc import ABC
from abc import abstractmethod

from typing import Dict, Tuple

import torch
import torchvision.models as models  # type: ignore

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


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


# Methods of fusing the speech and visual encodings.
class AbsFeatureFuser(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, enc_speech: torch.Tensor, enc_visual: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ConcatProjFuser(AbsFeatureFuser):
    def __init__(self, dim_speech, dim_visual):
        super().__init__()
        d_in = dim_speech + dim_visual
        d_out = dim_speech
        self.proj = torch.nn.Conv1d(d_in, d_out, kernel_size=1)

    def forward(self, speech, visual):
        _, T, _ = speech.shape
        visual = visual.unsqueeze(1).repeat(1, T, 1)
        out = torch.cat((speech, visual), dim=2)
        out = out.permute(0, 2, 1)
        out = self.proj(out)
        out = out.permute(0, 2, 1)
        return out


# The multimodal model
class ESPnetASRMultimodalModel(AbsESPnetModel):
    def __init__(
        self, asr: ESPnetASRModel, encoder_visual, feature_fuser,
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
        encoder_speech_out, encoder_out_lens = self.asr.encode(speech, speech_lengths)
        encoder_visual_out = self.encoder_visual(visual)
        encoder_out = self.feature_fuser(encoder_speech_out, encoder_visual_out)

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
        import pdb; pdb.set_trace()
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.asr.collect_feats(speech, speech_lengths, text, text_lengths)
