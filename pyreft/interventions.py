import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyvene.models.layers import LowRankRotateLayer, RotateLayer
from pyvene.models.basic_utils import sigmoid_boundary
from transformers.activations import ACT2FN


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return




class LoreftInterventionNoBias(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=False).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return


class DistributedRotationIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False
        )
        self.learned_source = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=False).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=True
        )
        
    def forward(self, base, source=None, subspaces=None):
        batch_size = base.shape[0]
        prompt_max_len = base.shape[1]
        rotated_base = self.rotate_layer(base)

        # "learn" the rotation
        pseudo_rotated_source = self.learned_source(base)
        pseudo_rotated_source = self.act_fn(pseudo_rotated_source)

        # get boundary
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, prompt_max_len, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        boundary_mask = (
            torch.ones((batch_size, prompt_max_len), device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * pseudo_rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return self.dropout(output.to(base.dtype))

    def __str__(self):
        return f"DistributedRotationIntervention()"


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))

