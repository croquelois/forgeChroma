# forgeChroma

Add Chroma architecture to Forge

## Installation

1) inside your forge root directory, apply the patch to loader.py `git apply loader.patch`
2) apply the patch to condition.py `git apply condition.patch`
3) add `huggingface/Chroma` directory inside `backend/huggingface`
4) add `diffusion_engine/chroma.py` file inside `backend/diffusion_engine`
5) add `nn/chroma.py` file inside `backend/nn`

## Architecture

Chroma is quite similar to Flux-Schnell, but the guidance part has been removed:
- `double_block`, don't have `txt_mod` nor `img_mod` anymore
- `single_blocks`, don't have `modulation` anymore
- `final_layer`, don't have `adaLN_modulation` anymore
- `guidance_in`, `time_in`, `vector_in` are removed

instead there is an independent multilayered network called `distilled_guidance_layer` which handle the same function without being deeply intertwined to the different layers

another change is on the text encoder, Flux is relying on both CLIP_L and T5, but Chroma only need T5.

## Result

![sailor](sailor.png)
