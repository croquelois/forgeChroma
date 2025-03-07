import torch

from backend import memory_management
from backend.diffusion_engine.flux import Flux

class Chroma(Flux):
    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        print("Chroma __init__")

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond_l, pooled_l = self.text_processing_engine_l(prompt)
        cond_t5 = self.text_processing_engine_t5(prompt)
        cond = dict(crossattn=cond_t5, vector=pooled_l)
        cond['guidance'] = torch.FloatTensor([0] * len(prompt))
        return cond