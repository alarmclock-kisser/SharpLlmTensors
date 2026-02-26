using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json;

namespace SharpLlmTensors.Runtime.Modules
{
    public class SwiGLUMLP : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> gate_proj;
        private readonly Module<Tensor, Tensor> up_proj;
        private readonly Module<Tensor, Tensor> down_proj;

        public SwiGLUMLP(JsonElement config) : base("SwiGLUMLP")
        {
            long hidden = config.GetProperty("hidden_size").GetInt64();
            long intermediate = config.GetProperty("intermediate_size").GetInt64();

            TorchService.LogVerbose($"[SwiGLUMLP] Initializing. hidden: {hidden}, intermediate: {intermediate}");

            this.gate_proj = Linear(hidden, intermediate, hasBias: false);
            this.up_proj = Linear(hidden, intermediate, hasBias: false);
            this.down_proj = Linear(intermediate, hidden, hasBias: false);

            // Diese Registrierungen sind mandatory für Safetensors
            this.register_module("gate_proj", this.gate_proj);
            this.register_module("up_proj", this.up_proj);
            this.register_module("down_proj", this.down_proj);
            TorchService.LogVerbose("[SwiGLUMLP] Projections registered");
        }

        public override Tensor forward(Tensor x)
        {
            TorchService.LogVerbose($"[SwiGLUMLP] Forward called. input shape: {string.Join(',', x.shape)}");
            using var g = this.gate_proj.forward(x);
            TorchService.LogVerbose($"[SwiGLUMLP] gate_proj shape: {string.Join(',', g.shape)}");
            using var u = this.up_proj.forward(x);
            TorchService.LogVerbose($"[SwiGLUMLP] up_proj shape: {string.Join(',', u.shape)}");
            using var activated = functional.silu(g) * u;
            TorchService.LogVerbose($"[SwiGLUMLP] activated shape: {string.Join(',', activated.shape)}");
            return this.down_proj.forward(activated);
        }
    }
}