import pdb

from contextlib import contextmanager
from distutils.version import LooseVersion

from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.audio_to_lip.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetAudioToLipModel(AbsESPnetModel):
    def __init__(
        self,
        frontend: AbsFrontend,
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
    ):
        assert check_argument_types()
        super().__init__()

        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = torch.nn.MSELoss()

    def forward(
        self, 
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lips: torch.Tensor,
        lips_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        encoder_mask = self._source_mask(encoder_out_lens)
        pred = self.decoder(encoder_out, encoder_mask, lips, lips_lengths)
        loss = self.criterion(lips, pred)
        stats = {"loss": loss.item()}
        batch_size = speech.size(0)
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        assert len(speech) == 1 # batch of one?
        encoder_out, _ = self.encode(speech, speech_lengths)
        return self.decoder.inference(encoder_out, speech_lengths)

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        lips: torch.Tensor,
        lips_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        # Frontend
        #  e.g. STFT and Feature extract
        #       data_loader may send time-domain signal in this case
        # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
        feats, feats_lengths = self.frontend(speech, speech_lengths)
        return feats, feats_lengths

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
