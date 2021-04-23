import torch

from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet2.audio_to_lip.decoder.abs_decoder import AbsDecoder


class TransformerDecoder(AbsDecoder):
    def __init__(self,
        output_dim: int,
        adim: int = 256,
        aheads: int = 4,
        dlayers: int = 6,
        dunits: int = 1024,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        transformer_enc_dec_attn_dropout_rate: float = 0.1,
        decoder_normalize_before: bool = True,
        decoder_concat_after: bool = False,
        dprenet_layers: int = 0,
        dprenet_units: int = 20,
        dprenet_dropout_rate: float = 0.5,
        use_scaled_pos_enc: bool = True,
    ):
        pos_enc_class = (
            ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
        )
        self.output_dim = output_dim
        # define transformer decoder
        if dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=self.output_dim,
                    n_layers=dprenet_layers,
                    n_units=dprenet_units,
                    dropout_rate=dprenet_dropout_rate,
                ),
                torch.nn.Linear(dprenet_units, adim),
            )
        else:
            decoder_input_layer = "linear"
        super().__init__()
        self.decoder = Decoder(
            odim=self.output_dim,  # odim is needed when no prenet is used
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
        )
        self.feat_out = torch.nn.Linear(adim, self.output_dim)

    def forward(self, hs, h_masks, ys, olens):
        ys_in = self._add_first_frame_and_remove_last_frame(ys)
        y_masks = self._target_mask(olens)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.output_dim)
        return before_outs

    def inference(self, hs, speech_lengths, frames_ratio):
        assert hs.shape[0] == 1 and speech_lengths.shape[0] == 1

        num_out = speech_lengths[0] * frames_ratio
        num_out = int(num_out.round().cpu().item())

        # initialize
        ys = hs.new_zeros(1, 1, self.output_dim).to(hs.device)
        outs = torch.zeros(num_out, self.output_dim).to(hs.device)

        z_cache = self.decoder.init_state(hs)
        for idx in range(1, num_out + 1):
            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(hs.device)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache
            )  # (B, adim)
            outs[idx - 1] = self.feat_out(z)

            # update next inputs
            ys = torch.cat(
                (ys, outs[idx - 1].view(1, 1, self.output_dim)), dim=1
            )  # (1, idx + 1, output_dim)

        return outs.unsqueeze(0)

    def _add_first_frame_and_remove_last_frame(self, ys: torch.Tensor) -> torch.Tensor:
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

    def _target_mask(self, olens: torch.Tensor) -> torch.Tensor:
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks
