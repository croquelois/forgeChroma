# forgeChroma

Forge now support Chroma, so this repository will now only contains optional "good to have" patches

## Architecture

Chroma is quite similar to Flux-Schnell, but the guidance part has been removed:
- `double_block`, don't have `txt_mod` nor `img_mod` anymore
- `single_blocks`, don't have `modulation` anymore
- `final_layer`, don't have `adaLN_modulation` anymore
- `guidance_in`, `time_in`, `vector_in` are removed

instead there is an independent multilayered network called `distilled_guidance_layer` which handle the same function without being deeply intertwined to the different layers

another change is on the text encoder, Flux is relying on both CLIP_L and T5, but Chroma only need T5.

last change, the original Flux model is padding the result of the T5 Tokenizer to force it to at least 256 token. Chroma is trained without the padding.

the model is trained with `T5 XXL Flan` as encoder, while the classic `T5 XXL` work fine, the `Flan` version give better result.

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

## FP8 Scaled support

FP8 scaled safetensors are not yet supported by Forge

to add it: `git apply fp8.patch`

you may need to do `git config apply.whitespace nowarn` to avoid `git apply` complaining about mixed Windows/Linux end of lines.

## sigmoid scheduler

The sigmoid scheduler is a a quite powerful alternative to the one already available in Forge.

to add it: `git apply sigmoidScheduler.patch`

## Mag/Tea cache

Chroma is slow, it is twice slower than Flux. But caching can help. Both Mag Cache and Tea Cache are implemented.

I observe more than 50% speed improvement without much loss of quality

to add it, use what is in the `cache` directory.

![image](https://github.com/user-attachments/assets/51157cdb-3545-412e-8976-4fbd658fe828)

prompt: `An image of a squirrel in Picasso style`

the performance and parameters are in the title above the pictures (all 30 step of euler simple):

- original: 67s
- magcache(0.25): 36s
- teacache(0.25): 46s
- teacache(0.40): 28s

