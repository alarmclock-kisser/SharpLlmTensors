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
        private readonly string hidden_activation;

        public SwiGLUMLP(JsonElement config) : base("SwiGLUMLP")
        {
            long hidden = config.GetProperty("hidden_size").GetInt64();
            long intermediate = config.GetProperty("intermediate_size").GetInt64();

            if (config.TryGetProperty("hidden_activation", out var actProp) && actProp.ValueKind != JsonValueKind.Null)
            {
                this.hidden_activation = actProp.GetString()?.ToLower() ?? "silu";
            }
            else
            {
                this.hidden_activation = "silu";
            }

            this.gate_proj = Linear(hidden, intermediate, hasBias: false);
            this.up_proj = Linear(hidden, intermediate, hasBias: false);
            this.down_proj = Linear(intermediate, hidden, hasBias: false);

            this.register_module("gate_proj", this.gate_proj);
            this.register_module("up_proj", this.up_proj);
            this.register_module("down_proj", this.down_proj);
        }

        public override Tensor forward(Tensor x)
        {
            using var g = this.gate_proj.forward(x);
            using var u = this.up_proj.forward(x);

            Tensor activated;

            if (this.hidden_activation.Contains("gelu"))
            {
                // ANTI-NAN FIX: Pow(3) verursacht in Float16 extrem schnell Infinitys. Wir machen das in F32.
                using var g_f32 = g.to(ScalarType.Float32);

                var coeff = (float) System.Math.Sqrt(2.0 / System.Math.PI);
                using var x3 = torch.pow(g_f32, 3);
                using var inner1 = g_f32 + 0.044715f * x3;
                using var inner2 = inner1 * coeff;
                using var t = torch.tanh(inner2);
                using var one_plus_t = 1.0f + t;
                using var act_f32 = 0.5f * g_f32 * one_plus_t;

                // Cast zurück auf F16 nach sicherer Berechnung
                activated = act_f32.to(g.dtype);
            }
            else
            {
                activated = torch.nn.functional.silu(g);
            }

            using var multiplied = activated * u;

            activated.Dispose();
            return this.down_proj.forward(multiplied);
        }
    }
}