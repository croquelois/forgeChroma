# forgeChroma

Add Chroma architecture to Forge

## Installation

1) inside your forge root directory, apply the patch to loader.py `git apply loader.patch`
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

## Result

![sailor](sailor.png)
