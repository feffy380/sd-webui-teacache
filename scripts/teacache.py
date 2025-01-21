from typing import Optional

import gradio as gr
import torch as th
from numpy.polynomial import Polynomial
from sgm.modules.diffusionmodules.openaimodel import timestep_embedding

from modules import processing, script_callbacks, scripts
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui_components import InputAccordion

_cache = None


class TeaCacheSession:
    def __init__(self, threshold: float, max_consecutive: int, start: float, end: float, steps: int, initial_step: int = 1):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self.start = start
        self.end = end
        self.steps = steps

        self.current_step = initial_step
        self.call_index = 0
        self.residual = [None]
        self.previous = [None]
        self.distance = [0.0]
        self.consecutive_hits = [0]

    def next_step(self):
        self.current_step += 1
        self.call_index = 0


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
                    minimum=0.0, maximum=1.0, value=0.2, step=0.01,
                )
            with gr.Row():
                max_consecutive = gr.Number(
                    label="Max consecutive cached steps",
                    minimum=-1, maximum=150, value=-1, step=1,
                )
                start = gr.Slider(
                    label="Start",
                    minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                )
                end = gr.Slider(
                    label="End",
                    minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                )

        return enabled, threshold, max_consecutive, start, end
    
    def process(self, p: processing.StableDiffusionProcessing, *args):
        # patch model forward method
        enabled = args[0]
        if not enabled:
            return
        unet = p.sd_model.model.diffusion_model
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
            # print(f"DEBUG: img2img initial_step: {initial_step}")
        elif p.is_hr_pass:
            total_steps = getattr(p, "hr_second_pass_steps", 0) or p.steps
            total_steps, steps = setup_img2img_steps(p, total_steps)  # hires fix doesn't reduce steps
            initial_step = total_steps - steps
            # print(f"DEBUG: hr_pass initial_step: {initial_step}")
        # else:
            # print(f"DEBUG: txt2img initial_step: {initial_step}")
        _cache = TeaCacheSession(threshold, max_consecutive, start, end, total_steps, initial_step)

    def postprocess(self, p: processing.StableDiffusionProcessing, processed, *args):
        # restore model, clear cache
        global _cache
        unet = p.sd_model.model.diffusion_model
        if not getattr(unet, "_teacache_patched", False):
            return
        unet.forward = self.original_forward
        unet._teacache_patched = False
        self.original_forward = None
        _cache = None


def relative_l1_distance(prev: th.Tensor, curr: th.Tensor):
    return ((prev - curr).abs().mean() / prev.abs().mean()).item()


def patched_forward(
    self,
    x: th.Tensor,
    timesteps: Optional[th.Tensor] = None,
    context: Optional[th.Tensor] = None,
    y: Optional[th.Tensor] = None,
    **kwargs,
) -> th.Tensor:
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """

    global _cache
    index = _cache.call_index
    # cache multiple model calls in one step
    if index == len(_cache.residual):
        _cache.residual.append(None)
        _cache.previous.append(None)
        _cache.distance.append(0.0)
        _cache.consecutive_hits.append(0)

    previous = _cache.previous[index]
    residual = _cache.residual[index]
    distance = _cache.distance[index]
    consecutive_hits = _cache.consecutive_hits[index]

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    use_cache = True
    first_block_residual = None
    h = x
    for id, module in enumerate(self.input_blocks[:2]):
        if id == 1:
            original_h = h
        h = module(h, emb, context)
        hs.append(h)

        # check cache
        if id == 1:  # id=0 is only conv
            first_block_residual = h - original_h

            # cache conditions
            # check step range
            progress = _cache.current_step / _cache.steps
            if not (_cache.start < progress <= _cache.end):
                use_cache = False
            # check max consecutive cache hits
            if _cache.max_consecutive >= 0 and consecutive_hits >= _cache.max_consecutive:
                use_cache = False
            # check cached value exists
            if previous is None or residual is None:
                use_cache = False

    if use_cache:
        p = Polynomial([-7.33293101e-02, 5.57968864e+00, -6.85753651e+01, 4.04814722e+02, -8.11904600e+02])  # NoobAI XL vpred v1.0
        # p = Polynomial([-4.46619183e-02,  2.04088614e+00, -1.30308644e+01,  1.01387815e+02, -2.48935677e+02])  # NoobAI XL v1.1
        distance += p(relative_l1_distance(previous, first_block_residual)).item()
        # print(f"DEBUG: distance: {distance}")
        if distance >= _cache.threshold:
            use_cache = False
    previous = first_block_residual

    if use_cache:
        h += residual
        consecutive_hits += 1
    else:
        original_h = h
        for module in self.input_blocks[2:]:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        residual = h - original_h
        distance = 0.0
        consecutive_hits = 0

    _cache.residual[index] = residual
    _cache.previous[index] = previous
    _cache.distance[index] = distance
    _cache.consecutive_hits[index] = consecutive_hits
    _cache.call_index += 1

    h = h.type(x.dtype)

    return self.out(h)


def next_step(*args):
    global _cache
    if _cache is not None:
        _cache.next_step()


script_callbacks.on_cfg_after_cfg(next_step)