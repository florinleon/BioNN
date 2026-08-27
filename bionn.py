import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


InitializationSeedModulus = 2 ** 63 - 1
LayerSeedStride = 1000003
StreamSeedStride = 10007

MechanismNames = [
    "UseInputGate",
    "UseGainModulation",
    "UseThresholdModulation",
    "UseLateralInhibition",
    "UseHomeostasis",
    "UseStructuralPlasticity",
    "UseActivationDecorrelation",
]


DefaultBioNNParameters = {
    # Model shape. Override these values only when the input, hidden, or output tensor dimensions must change.
    # HiddenDims defines the widths of the hidden layers and therefore the model capacity.
    "InputDim": 1,
    "HiddenDims": [1000],
    "OutputDim": 1,
    "ModulatorSize": 32,
    "UseHiddenBias": True,
    "UseOutputBias": True,

    # Mechanism switches. These seven booleans control which regulatory mechanisms are enabled.
    # Numerical strengths are configured separately from the mechanism switches.
    "UseInputGate": True,
    "UseGainModulation": True,
    "UseThresholdModulation": True,
    "UseLateralInhibition": True,
    "UseHomeostasis": True,
    "UseStructuralPlasticity": False,
    "UseActivationDecorrelation": False,

    # Input modulation. The gate is multiplicative and centered at one, so this value sets the maximum relative
    # coordinate scaling when input gating is active.
    "InputGateStrength": 0.60,

    # Hidden modulation. Gain is multiplicative, while threshold shift is additive. ThresholdStrength should match
    # the scale of hidden preactivations for the input distribution being modeled.
    "GainStrength": 0.70,
    "ThresholdStrength": 0.60,

    # Lateral inhibition. This coefficient scales the mean activity of competing units before the final ReLU.
    "InhibitionStrength": 0.18,

    # Homeostasis. These values define the target hidden-unit activity, the threshold adaptation rate, the threshold
    # safety bound, the activity-detection cutoff, and the penalty on large slow thresholds.
    "TargetActivity": 0.12,
    "HomeostasisRate": 0.01,
    "ThresholdLimit": 2.0,
    "ActivityEpsilon": 1e-4,
    "HomeostasisRegularization": 1e-4,

    # Structural plasticity. The mask is initialized from this density for every ablation, but the mask affects the
    # forward pass only when UseStructuralPlasticity is true.
    "StructuralDensity": 0.35,

    # Activation decorrelation. This weight controls the auxiliary penalty on correlated hidden-unit activity.
    "DecorrelateStrength": 2e-3,

    # Initialization. A None seed uses PyTorch's current seed; otherwise local streams make initialization independent
    # of which mechanisms are enabled.
    "InitializationSeed": None,
}

AllowedConfigNames = set(DefaultBioNNParameters)


def DefaultBioNNConfig():
    # Return a fresh copy so callers can safely edit list-valued entries such as HiddenDims without mutating the
    # canonical defaults used by later experiments.
    return copy.deepcopy(DefaultBioNNParameters)


def MakeBioNNConfig(overrides=None):
    # Configuration overrides are intentionally shallow and explicit. Unknown keys fail early because a misspelled
    # parameter would otherwise be silently ignored and produce an experiment that looks reproducible but is not.
    config = DefaultBioNNConfig()
    if overrides is not None:
        unknown = sorted(set(overrides) - AllowedConfigNames)
        if len(unknown) > 0:
            raise ValueError("Unknown BioNN configuration key(s): {}".format(", ".join(unknown)))
        config.update(overrides)
    return config


def MakeInitializationGenerator(config, layer_index, stream_index):
    # Each parameter family receives a deterministic random-number stream derived from the experiment seed, layer index,
    # and stream index. This keeps a gate head, a gain head, and the main weight tensor from stealing randomness from
    # one another and makes ablation switches change only computation, not unrelated initialization draws.
    seed = config.get("InitializationSeed", None)
    if seed is None:
        seed = torch.initial_seed()
    seed = int(seed) + LayerSeedStride * int(layer_index + 1) + StreamSeedStride * int(stream_index + 1)
    generator = torch.Generator()
    generator.manual_seed(seed % InitializationSeedModulus)
    return generator


def SaveTorchRngState():
    # PyTorch Linear layers initialize their parameters during construction. This helper captures the ambient RNG state
    # so temporary module construction does not advance unrelated random streams.
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def RestoreTorchRngState(state):
    # Restore both CPU and CUDA RNG streams after a temporary module construction. CUDA states are restored only when
    # CUDA exists, which keeps the same code usable on CPU-only machines.
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def MakeLinear(in_features, out_features, bias=True):
    # This creates a Linear layer while preserving the ambient RNG state. The layer is reset later with an explicit
    # generator, so the construction-time initialization is only a temporary allocation side effect.
    state = SaveTorchRngState()
    try:
        layer = nn.Linear(in_features, out_features, bias=bias)
    finally:
        RestoreTorchRngState(state)
    return layer


def ResetLinearDefault(linear, generator):
    # This matches the default nn.Linear initialization, but uses a local generator so ablation switches cannot change
    # the initialization of any component that exists in the fixed architecture.
    fan_in = linear.weight.size(1)
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound, generator=generator)


def ResetKaimingUniform(weight, generator):
    # Kaiming uniform with a=sqrt(5) matches the affine weight initialization used by PyTorch Linear. The explicit
    # implementation keeps the hidden-layer reset independent of any temporary modules that were constructed earlier.
    fan_in = weight.size(1)
    gain = math.sqrt(2.0 / (1.0 + 5.0))
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    with torch.no_grad():
        weight.uniform_(-bound, bound, generator=generator)


class BioNNLayer(nn.Module):
    def __init__(self, input_size, output_size, config, layer_index=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = copy.deepcopy(config)
        self.layer_index = int(layer_index)
        self.modulator_size = int(self.config.get("ModulatorSize", 32))
        self.validate_layer_config()

        # The affine weight and bias form the ordinary hidden-layer drive. All regulatory mechanisms act around this
        # same parameter tensor, so a mechanism ablation does not replace the underlying hidden layer.
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if self.config.get("UseHiddenBias", True):
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)

        # Homeostatic thresholds and structural masks are state variables. They move with the module across devices and
        # checkpoints, but optimizers do not update them directly as ordinary learned parameters.
        self.register_buffer("homeostatic_threshold", torch.zeros(output_size))
        self.register_buffer("structural_mask", torch.ones(output_size, input_size))

        self.make_modulator()
        self.make_modulation_heads()
        self.reset_parameters()
        self.initialize_sparse_mask()


    def validate_layer_config(self):
        # These checks catch malformed configurations before the first minibatch. The structural density is allowed to
        # be one, which gives a dense mask, and must be positive so every layer keeps at least one possible connection.
        if self.modulator_size < 1:
            raise ValueError("ModulatorSize must be at least 1.")
        structural_density = self.config.get("StructuralDensity", 0.35)
        if structural_density <= 0.0 or structural_density > 1.0:
            raise ValueError("StructuralDensity must be in the interval (0, 1].")


    def make_modulator(self):
        # The context pathway computes c = tanh(Cx + b_c) from the current layer input. It is always present so the
        # model has the same parameters and initialization across ablations; switches decide whether downstream
        # mechanisms use the context or replace their effect with a neutral value.
        linear = MakeLinear(self.input_size, self.modulator_size)
        ResetLinearDefault(linear, MakeInitializationGenerator(self.config, self.layer_index, 10))
        self.modulator = nn.Sequential(linear, nn.Tanh())


    def make_modulation_heads(self):
        # The three context heads project the shared context into input-coordinate gates, hidden-unit gains, and
        # hidden-unit threshold shifts. Keeping all heads allocated prevents parameter count, optimizer state, and
        # initialization order from depending on which mechanisms are enabled in a particular ablation.
        self.input_gate = MakeLinear(self.modulator_size, self.input_size)
        self.output_gain = MakeLinear(self.modulator_size, self.output_size)
        self.threshold_shift = MakeLinear(self.modulator_size, self.output_size)


    def reset_parameters(self):
        # The hidden weight uses a fixed Kaiming-style initialization, and the bias follows the corresponding fan-in
        # scale. This keeps the unregulated layer well behaved before the regulatory heads begin to learn nonzero maps.
        generator = MakeInitializationGenerator(self.config, self.layer_index, 20)
        ResetKaimingUniform(self.weight, generator)

        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound, generator=generator)

        # Zero-initialized modulation heads make the initial gate equal to one, the initial gain equal to one, and the
        # initial fast threshold shift equal to zero. The initial model therefore starts as a standard affine-ReLU layer
        # plus any nonzero homeostatic or structural state, and learns regulatory effects from data.
        for head in [self.input_gate, self.output_gain, self.threshold_shift]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)


    def initialize_sparse_mask(self):
        # The structural mask is initialized once from the configured density. When structural plasticity is disabled,
        # the mask is stored but ignored by the forward pass; when it is enabled, the same mask becomes the effective
        # connectivity pattern. This makes the switch control mask use, not mask creation.
        structural_density = self.config.get("StructuralDensity", 0.35)
        if structural_density >= 1.0:
            return

        with torch.no_grad():
            # The initial sparse pattern is random rather than weight-based. Weight magnitudes are still untrained at
            # construction time, so a random mask avoids treating initialization noise as an early structural signal.
            total_connections = self.output_size * self.input_size
            active_connections = max(1, int(total_connections * structural_density))
            flat_mask = torch.zeros(total_connections, device=self.structural_mask.device)
            generator = MakeInitializationGenerator(self.config, self.layer_index, 30)
            order = torch.randperm(total_connections, generator=generator, device=self.structural_mask.device)
            flat_mask[order[:active_connections]] = 1.0
            self.structural_mask.copy_(flat_mask.view(self.output_size, self.input_size))


    def modulation_context(self, x, ablate=None):
        # The context vector is the shared sample-dependent signal used by input gating, gain modulation, and threshold
        # modulation. A runtime context ablation replaces it with zeros, which also makes zero-initialized or neutralized
        # context heads produce no sample-specific regulatory signal.
        ablate = ablate or {}
        if ablate.get("context", False):
            return torch.zeros(x.size(0), self.modulator_size, device=x.device, dtype=x.dtype)
        return self.modulator(x)


    def compute_input_gate(self, context, x, ablate):
        # Input gating computes one multiplicative factor per input coordinate. When disabled, the neutral gate is all
        # ones, so x_g = x and the affine drive sees the unmodified layer input.
        if not self.config.get("UseInputGate", True) or ablate.get("input_gate", False) or ablate.get("context", False):
            return torch.ones_like(x)

        # The tanh bounds the learned gate signal, and InputGateStrength controls how far the factor may move away from
        # one. With the default strength 0.60, gates lie in the interval [0.40, 1.60].
        return 1.0 + self.config.get("InputGateStrength", 0.60) * torch.tanh(self.input_gate(context))


    def compute_gain(self, context, x, ablate):
        # Gain modulation produces one multiplicative response scale per hidden unit. Disabling it returns one, which
        # leaves the preactivation amplitude unchanged before thresholding and inhibition.
        if not self.config.get("UseGainModulation", True) or ablate.get("gain", False) or ablate.get("context", False):
            return torch.ones(x.size(0), self.output_size, device=x.device, dtype=x.dtype)

        # The gain is centered at one, so the mechanism changes unit responsiveness without introducing an additive
        # offset. Larger gain makes a given preactivation more likely to survive the threshold and final ReLU.
        return 1.0 + self.config.get("GainStrength", 0.70) * torch.tanh(self.output_gain(context))


    def compute_threshold(self, context, x, ablate):
        # The total threshold has two parts: a slow homeostatic state and a fast context-dependent shift. Either part can
        # be neutralized independently, which lets ablations isolate long-timescale activity control from sample-specific
        # threshold modulation.
        if self.config.get("UseHomeostasis", True) and not ablate.get("homeostasis_state", False):
            threshold = self.homeostatic_threshold.unsqueeze(0).to(dtype=x.dtype)
        else:
            threshold = torch.zeros(1, self.output_size, device=x.device, dtype=x.dtype)

        use_threshold_shift = self.config.get("UseThresholdModulation", True) and not ablate.get("threshold_shift", False)
        if use_threshold_shift and not ablate.get("context", False):
            # The fast shift is additive rather than multiplicative, so its scale should be interpreted relative to the
            # hidden preactivation scale. It can raise or lower the current threshold for each sample and unit.
            shift = self.config.get("ThresholdStrength", 0.60) * torch.tanh(self.threshold_shift(context))
            threshold = threshold + shift
        return threshold


    def compute_lateral_pressure(self, primary_activity, ablate):
        # Lateral inhibition is computed from primary activity, before the final inhibited ReLU. Each unit receives the
        # mean primary activity of all other units, so the pressure reflects competition from the rest of the layer.
        if not self.config.get("UseLateralInhibition", True) or ablate.get("lateral_inhibition", False):
            return torch.zeros_like(primary_activity)
        if self.output_size > 1:
            total_activity = primary_activity.sum(dim=1, keepdim=True)
            return (total_activity - primary_activity) / (self.output_size - 1)
        return torch.zeros_like(primary_activity)


    def forward(self, x, update_homeostasis=False, return_details=False, ablate=None):
        # The computation order is deliberate: context is computed first, the input gate modifies the layer input, the
        # optional structural mask modifies the weight tensor, gain and thresholds regulate the affine drive, primary
        # activity defines inhibitory pressure, and the final ReLU produces the hidden activity used downstream.
        ablate = ablate or {}
        context = self.modulation_context(x, ablate)
        input_gate = self.compute_input_gate(context, x, ablate)
        gated_x = x * input_gate

        # Structural plasticity masks connections in the effective computation while leaving the underlying weight tensor
        # trainable. When the switch is off, the dense weight is used even though the mask state still exists.
        if self.config.get("UseStructuralPlasticity", False) and not ablate.get("structural_mask", False):
            effective_weight = self.weight * self.structural_mask
        else:
            effective_weight = self.weight

        preactivation = F.linear(gated_x, effective_weight, self.bias)
        gain = self.compute_gain(context, x, ablate)
        threshold = self.compute_threshold(context, x, ablate)

        # Primary activity is the regulated activity before lateral inhibition. It is kept separate because inhibition
        # should depend on the current competitive drive, not on activity after inhibition has already suppressed units.
        primary_activity = F.relu(gain * preactivation - threshold)
        lateral_pressure = self.compute_lateral_pressure(primary_activity, ablate)
        activity = F.relu(gain * preactivation - threshold - self.config.get("InhibitionStrength", 0.18) * lateral_pressure)

        # Homeostasis is an explicit state update based on detached activity statistics. It is not a gradient update, so
        # it should not retain computation graphs or backpropagate through the threshold adaptation rule.
        if update_homeostasis and self.config.get("UseHomeostasis", True) and not ablate.get("homeostasis_update", False):
            self.update_homeostasis(activity.detach())

        if not return_details:
            return activity

        # Diagnostic tensors are detached unless they are needed for an auxiliary loss. The regularization tensor keeps
        # its graph so activation decorrelation can contribute gradients to the hidden representation.
        details = {
            "context": context.detach(),
            "input_gate": input_gate.detach(),
            "gated_input": gated_x.detach(),
            "preactivation": preactivation.detach(),
            "gain": gain.detach(),
            "threshold": threshold.detach(),
            "primary_activity": primary_activity.detach(),
            "lateral_pressure": lateral_pressure.detach(),
            "activity": activity.detach(),
            "activity_for_regularization": activity,
            "structural_mask": self.structural_mask.detach(),
        }
        return activity, details


    def update_homeostasis(self, activity):
        # Homeostasis measures how often each unit is active in the current minibatch. Units above the target receive a
        # higher threshold and become harder to activate; units below the target receive a lower threshold and become
        # easier to activate.
        with torch.no_grad():
            batch_frequency = (activity > self.config.get("ActivityEpsilon", 1e-4)).float().mean(dim=0)
            update = self.config.get("HomeostasisRate", 0.01) * (batch_frequency - self.config.get("TargetActivity", 0.12))
            self.homeostatic_threshold.add_(update)
            limit = self.config.get("ThresholdLimit", 2.0)
            self.homeostatic_threshold.clamp_(-limit, limit)


    def structural_update(self):
        # Structural plasticity refreshes the binary mask from the current learned weight magnitudes. The weight tensor
        # remains trainable even for masked connections; the mask only decides which connections participate in the
        # effective forward computation.
        if not self.config.get("UseStructuralPlasticity", False):
            return
        if self.config.get("StructuralDensity", 0.35) >= 1.0:
            return

        with torch.no_grad():
            # The top-k rule keeps the configured fraction of connections with largest absolute weights. This treats
            # large magnitude as the current proxy for connection utility, while preserving the configured mask density.
            total_connections = self.output_size * self.input_size
            active_connections = max(1, int(total_connections * self.config.get("StructuralDensity", 0.35)))
            utility = self.weight.abs().view(-1)
            threshold = torch.topk(utility, active_connections, sorted=False).values.min()
            self.structural_mask.copy_((self.weight.abs() >= threshold).float())


    def effective_weight(self, ignore_mask=False):
        # This helper exposes the actual hidden-layer weight used by the forward pass. It is useful for diagnostics that
        # should count only active structural connections when structural plasticity is enabled.
        if self.config.get("UseStructuralPlasticity", False) and not ignore_mask:
            return self.weight.detach() * self.structural_mask.detach()
        return self.weight.detach()


    def effective_weight_norms(self):
        # Row norms summarize the effective incoming connection strength for each hidden unit. Masked-out coordinates do
        # not contribute when structural plasticity is enabled.
        return torch.linalg.norm(self.effective_weight(), dim=1)


    def active_connection_fraction(self):
        # When structural plasticity is disabled, the forward pass uses the dense weight matrix, so the effective active
        # fraction is one even if a sparse mask state has been initialized and stored.
        if not self.config.get("UseStructuralPlasticity", False):
            return 1.0
        return float(self.structural_mask.float().mean().item())


    def homeostasis_regularizer(self):
        # The homeostatic regularizer keeps slow thresholds from drifting far from zero unless activity statistics make
        # that useful. The loss vanishes when homeostasis is disabled, while the configured coefficient remains fixed.
        if not self.config.get("UseHomeostasis", True):
            return torch.tensor(0.0, device=self.weight.device)
        return self.config.get("HomeostasisRegularization", 1e-4) * self.homeostatic_threshold.pow(2).mean()


    def activation_decorrelation_regularizer(self, activity):
        # Decorrelation operates on final hidden activities across a minibatch. It centers each unit, normalizes each
        # activity profile, forms the unit-by-unit correlation matrix, and penalizes only off-diagonal correlations.
        if not self.config.get("UseActivationDecorrelation", False):
            return torch.tensor(0.0, device=activity.device)

        activations = activity - activity.mean(dim=0, keepdim=True)
        normalized = F.normalize(activations, dim=0, eps=1e-8)
        correlation = normalized.t() @ normalized
        off_diagonal = correlation - torch.diag(torch.diag(correlation))
        return self.config.get("DecorrelateStrength", 2e-3) * off_diagonal.pow(2).mean()


class BioNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = MakeBioNNConfig(config)
        self.validate_network_config()

        # Hidden layers share one configuration so every layer uses the same mechanism switches and parameter values.
        # Different HiddenDims entries change only the layer widths.
        layer_sizes = [self.config["InputDim"]] + list(self.config["HiddenDims"])
        self.layers = nn.ModuleList()
        for index in range(len(layer_sizes) - 1):
            self.layers.append(BioNNLayer(layer_sizes[index], layer_sizes[index + 1], self.config, layer_index=index))

        # The readout is an affine linear layer with an independently configurable bias. Keeping this separate prevents
        # ablations in hidden-layer regulation from also changing the output parameterization.
        final_dim = layer_sizes[-1]
        self.readout = MakeLinear(final_dim, self.config.get("OutputDim", 1), bias=self.config.get("UseOutputBias", True))
        ResetLinearDefault(self.readout, MakeInitializationGenerator(self.config, len(self.layers), 40))


    def validate_network_config(self):
        # Network-level validation checks dimensions that depend on the full stack rather than a single layer. Failures
        # here indicate an invalid problem definition rather than a mechanism-specific issue.
        if self.config.get("InputDim", 0) < 1:
            raise ValueError("InputDim must be at least 1.")
        if len(self.config.get("HiddenDims", [])) == 0:
            raise ValueError("HiddenDims must contain at least one hidden layer size.")
        for hidden_dim in self.config.get("HiddenDims", []):
            if hidden_dim < 1:
                raise ValueError("All hidden dimensions must be at least 1.")
        if self.config.get("OutputDim", 1) < 1:
            raise ValueError("OutputDim must be at least 1.")


    def forward(self, x, update_homeostasis=False, return_details=False, ablate=None, unit_masks=None):
        # The network applies each BioNN layer in sequence, then applies the fixed affine readout. Optional unit masks
        # are diagnostic interventions applied after a selected hidden layer and do not alter model parameters.
        details = {"layers": []}
        h = x
        for index, layer in enumerate(self.layers):
            if return_details:
                h, layer_details = layer(h, update_homeostasis=update_homeostasis, return_details=True, ablate=ablate)
                details["layers"].append(layer_details)
            else:
                h = layer(h, update_homeostasis=update_homeostasis, return_details=False, ablate=ablate)

            if unit_masks is not None and index in unit_masks:
                mask = unit_masks[index].to(device=h.device, dtype=h.dtype)
                h = h * mask.unsqueeze(0)

        details["hidden"] = h.detach()
        logits = self.apply_readout(h)
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        if return_details:
            details["logits"] = logits.detach()
            return logits, details
        return logits


    def apply_readout(self, h):
        # The readout maps the final hidden activity to task logits or predictions. It is intentionally simple so the
        # regulatory mechanisms remain localized in the hidden representation.
        return self.readout(h)


    def regularization_loss(self, details=None):
        # Regularization is the sum of enabled BioNN auxiliary losses across layers. Homeostasis regularization depends
        # only on threshold state, while decorrelation needs the current forward-pass activity with gradients attached.
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for layer in self.layers:
            loss = loss + layer.homeostasis_regularizer()

        uses_decorrelation = any(layer.config.get("UseActivationDecorrelation", False) for layer in self.layers)
        if uses_decorrelation and details is None:
            raise ValueError("Activation decorrelation requires forward(..., return_details=True).")

        if details is not None:
            for index, layer in enumerate(self.layers):
                activity = details["layers"][index].get("activity_for_regularization", details["layers"][index]["activity"])
                loss = loss + layer.activation_decorrelation_regularizer(activity)
        return loss


    def structural_update(self):
        # Structural masks are refreshed layer by layer. The training loop controls when this method is called, which
        # keeps the structural update schedule explicit and outside the ordinary forward pass.
        for layer in self.layers:
            layer.structural_update()


    def effective_weight_norms(self, layer_index=0):
        # Expose hidden-unit effective weight norms for analysis routines that rank units by their active incoming
        # connection strength.
        return self.layers[layer_index].effective_weight_norms()


    def active_connection_fraction(self, layer_index=0):
        # Report the effective active connection fraction for a selected layer. This reflects dense connectivity when
        # structural plasticity is disabled and the stored mask fraction when structural plasticity is enabled.
        return self.layers[layer_index].active_connection_fraction()


    def hidden_activity(self, x, layer_index=-1, ablate=None):
        # Return the hidden activity at a selected layer without updating homeostasis. This is used by diagnostics and
        # evaluation code that should inspect representations without changing model state.
        first_parameter = next(self.parameters(), None)
        if first_parameter is not None:
            x = x.to(device=first_parameter.device)

        h = x
        target_index = layer_index if layer_index >= 0 else len(self.layers) + layer_index
        for index, layer in enumerate(self.layers):
            h = layer(h, update_homeostasis=False, return_details=False, ablate=ablate)
            if index == target_index:
                return h
        return h


    def forward_with_unit_mask(self, x, selected_indices, layer_index=-1, ablate=None):
        # Run a forward pass while keeping only selected hidden units active at one layer. This supports diagnostic
        # subnetwork tests without modifying learned weights, masks, thresholds, or readout parameters.
        target_index = layer_index if layer_index >= 0 else len(self.layers) + layer_index
        mask = torch.zeros(self.layers[target_index].output_size, device=x.device)
        mask[torch.as_tensor(selected_indices, device=x.device, dtype=torch.long)] = 1.0
        unit_masks = {target_index: mask}
        return self.forward(x, update_homeostasis=False, return_details=False, ablate=ablate, unit_masks=unit_masks)
