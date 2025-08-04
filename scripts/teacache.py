from typing import Optional

import gradio as gr
import torch
from numpy.polynomial import Polynomial
from sgm.modules.diffusionmodules.openaimodel import timestep_embedding

from modules import processing, script_callbacks, scripts
from modules.sd_hijack_unet import th
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui_components import InputAccordion

_cache = None


def relative_l1_distance(prev: torch.Tensor, curr: torch.Tensor):
    return ((prev - curr).abs().mean() / prev.abs().mean()).item()


class TeaCacheSession:
    def __init__(self, threshold: float, max_consecutive: int, start: float, end: float, steps: int, initial_step: int = 1):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self.start = start
        self.end = end
        self.steps = steps

        self.current_step = initial_step
        self.call_index = 0
        self.residuals: dict[torch.Tensor] = {}
        self.previous_fb: Optional[torch.Tensor] = None
        self.distance = 0.0
        self.consecutive_hits = 0
        self.use_cache = True

    def update_condition(self, first_block_residual: torch.Tensor):
        # check step range
        progress = self.current_step / self.steps
        if not (self.start < progress <= self.end):
            self.use_cache = False
        # check max consecutive cache hits
        if self.max_consecutive > 0 and self.consecutive_hits >= self.max_consecutive:
            self.use_cache = False
        # check cached value exists
        if self.previous_fb is None or self.call_index not in self.residuals:
            self.use_cache = False

        if self.use_cache:
            p = Polynomial([ 4.72656327e-03,  1.09937816e+00,  4.82785530e+00, -2.93749209e+01, 4.22227031e+01])  # NoobAI XL vpred v1.0
            # p = Polynomial([-4.46619183e-02,  2.04088614e+00, -1.30308644e+01,  1.01387815e+02, -2.48935677e+02])  # NoobAI XL v1.1
            self.distance += p(relative_l1_distance(self.previous_fb, first_block_residual)).item()
            if self.distance >= self.threshold:
                self.use_cache = False
                self.distance = 0.0
            else:
                self.consecutive_hits += 1

    def next_step(self):
        self.current_step += 1
        self.call_index = 0
        self.use_cache = True


class TeaCacheScript(scripts.Script):
    def __init__(self):
        self.original_forward = None

    def title(self):
        return "TeaCache"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                threshold = gr.Slider(
                    label="Cache threshold",
                    info="Higher caches more aggressively.",
                    minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                )
            with gr.Row():
                max_consecutive = gr.Number(
                    label="Max consecutive cached steps",
                    minimum=0, maximum=150, value=0, step=1,
                )
                start = gr.Slider(
                    label="Start",
                    minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                )
                end = gr.Slider(
                    label="End",
                    minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                )

        infotext_keys = ["TeaCache threshold", "TeaCache max consecutive", "TeaCache start", "TeaCache end"]
        self.infotext_fields = [
            (enabled, lambda d: any(key in d for key in infotext_keys)),
            (threshold, "TeaCache threshold"),
            (max_consecutive, "TeaCache max consecutive"),
            (start, "TeaCache start"),
            (end, "TeaCache end"),
        ]

        components = [enabled, threshold, max_consecutive, start, end]

        return components
    
    def process(self, p: processing.StableDiffusionProcessing, *args):
        # patch model forward method
        enabled = args[0]
        if not enabled:
            # fix model if patch was not reverted (due to exception, oom)
            unet = p.sd_model.model.diffusion_model
            if getattr(unet, "_teacache_patched", False):
                self.postprocess(p)
            return
        unet = p.sd_model.model.diffusion_model
        if not self.original_forward:
            self.original_forward = unet.forward
        unet.forward = patched_forward.__get__(unet)
        unet._teacache_patched = True

    def process_before_every_sampling(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        # initialize and configure cache
        global _cache
        enabled, threshold, max_consecutive, start, end = args
        if not enabled:
            return
        # initial step based on denoise strength
        total_steps = p.steps
        initial_step = 1
        if self.is_img2img:
            total_steps, steps = setup_img2img_steps(p)
            initial_step = total_steps - steps
        elif p.is_hr_pass:
            total_steps = getattr(p, "hr_second_pass_steps", 0) or p.steps
            total_steps, steps = setup_img2img_steps(p, total_steps)  # hires fix doesn't reduce steps
            initial_step = total_steps - steps
        _cache = TeaCacheSession(threshold, max_consecutive, start, end, total_steps, initial_step)

        # set infotext
        p.extra_generation_params["TeaCache threshold"] = threshold
        if max_consecutive > 0:
            p.extra_generation_params["TeaCache max consecutive"] = max_consecutive
        if start > 0.0:
            p.extra_generation_params["TeaCache start"] = start
        if end < 1.0:
            p.extra_generation_params["TeaCache end"] = end

    def postprocess(self, p: processing.StableDiffusionProcessing, *args):
        # restore model, clear cache
        global _cache
        unet = p.sd_model.model.diffusion_model
        if not getattr(unet, "_teacache_patched", False):
            return
        unet.forward = self.original_forward
        unet._teacache_patched = False
        self.original_forward = None
        _cache = None


def patched_forward(
    self,
    x: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """

    global _cache

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x

    # call first two blocks
    h = self.input_blocks[0](h, emb, context)
    hs.append(h)
    original_h = h
    h = self.input_blocks[1](h, emb, context)
    hs.append(h)

    # check cache condition
    if _cache.call_index == 0:
        first_block_residual = h - original_h
        _cache.update_condition(first_block_residual)
        _cache.previous_fb = first_block_residual

    # use cache or call full model
    if _cache.use_cache:
        h += _cache.residuals[_cache.call_index]
    else:
        original_h = h
        for module in self.input_blocks[2:]:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        _cache.consecutive_hits = 0
        _cache.residuals[_cache.call_index] = h - original_h

    _cache.call_index += 1

    h = h.type(x.dtype)

    return self.out(h)


def next_step(*args):
    global _cache
    if _cache is not None:
        _cache.next_step()


script_callbacks.on_cfg_after_cfg(next_step)
