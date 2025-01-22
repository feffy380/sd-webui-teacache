# sd-webui-teacache
SDXL TeaCache for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

TeaCache uses the difference between model inputs across timesteps to determine when to reuse previous outputs. The difference between inputs is used to estimate the difference between model outputs, which is then refined through a polynomial fitted on input-output samples from the target model.

The official TeaCache repository doesn't support SDXL, so I've computed the polynomial coefficients myself. Without them, this implementation is equivalent to FBCache.

## Credits
- Official TeaCache repo - https://github.com/ali-vilab/TeaCache
- TeaCache paper - https://arxiv.org/abs/2411.19108
- SDXL compatibility based on FBCache - https://github.com/chengzeyi/Comfy-WaveSpeed
