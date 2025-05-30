# forgeChroma

Add Chroma architecture to Forge

## Installation

0) you may want to force Forge to be in the same version than mine `git checkout d557aef9d889556e5765e5497a6b8187100dbeb5`
1) inside your forge root directory, apply the patch: `git apply forge.patch` it will modify `backend/loader.py`, `backend/condition.py` and `backend/text_processing/t5_engine.py`
2) add `huggingface/Chroma` directory inside `backend/huggingface`
3) add `diffusion_engine/chroma.py` file inside `backend/diffusion_engine`
4) add `nn/chroma.py` file inside `backend/nn`

## Architecture

Chroma is quite similar to Flux-Schnell, but the guidance part has been removed:
- `double_block`, don't have `txt_mod` nor `img_mod` anymore
- `single_blocks`, don't have `modulation` anymore
- `final_layer`, don't have `adaLN_modulation` anymore
- `guidance_in`, `time_in`, `vector_in` are removed

instead there is an independent multilayered network called `distilled_guidance_layer` which handle the same function without being deeply intertwined to the different layers

another change is on the text encoder, Flux is relying on both CLIP_L and T5, but Chroma only need T5.

last change, the original Flux model is padding the result of the T5 Tokenizer to force it to at least 256 token. Chroma is trained without the padding.

## Result

- in Forge, on the top left select `all` and not `flux`
- use a resolution around 1024x1024
- I'm using `Euler` sampler with `Simple` scheduler, be carreful some combination don't work at all
- set the distilled config scale to 1, and the normal config scale to something like 3.5
- use a negative prompt, example: `Low quality, deformed, out of focus, restricted palette, flat colors`
- forge doesn't seem to work with all quantized model, `Q4_K_S` fail, but `Q4_1` work
- vae: `flux-vae-dev.safetensors`
- text: `t5xxl_fp8_e4m3fn.safetensors`

![sailor](sailor.png)
