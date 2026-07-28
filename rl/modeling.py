from transformers import PretrainedConfig, PreTrainedModel


class PointerCADTRLConfig(PretrainedConfig):
    model_type = "pointercad"

    def __init__(self, hidden_size: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self._attn_implementation = "flash_attention_2"


class PointerCADTRLModel(PreTrainedModel):
    """Small Transformers-compatible shell around the existing PointerCAD model."""

    config_class = PointerCADTRLConfig
    base_model_prefix = "pointercad"

    def __init__(self, pointercad):
        config = PointerCADTRLConfig(
            hidden_size=pointercad.model.config.hidden_size,
            pad_token_id=pointercad.model.config.pad_token_id,
        )
        super().__init__(config)
        self.pointercad = pointercad
        self.warnings_issued = {}

    def forward(self, *args, **kwargs):
        return self.pointercad(*args, **kwargs)

    def get_input_embeddings(self):
        return self.pointercad.model.get_input_embeddings()

    def enable_input_require_grads(self):
        return self.pointercad.model.enable_input_require_grads()

    def clamp_parameters(self):
        self.pointercad.clamp_parameters()

    def add_model_tags(self, tags: list[str]):
        # PointerCAD checkpoints are local torch checkpoints, not Hub model cards.
        return None
