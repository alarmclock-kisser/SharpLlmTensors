using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Modules
{
    public class RMSNorm : Module<Tensor, Tensor>
    {
        private readonly Parameter weight;
        private readonly double eps;
        private readonly bool addUnitOffset;

        public RMSNorm(long[] shape, double eps = 1e-6, bool addUnitOffset = false) : base("RMSNorm")
        {
            this.eps = eps;
            this.addUnitOffset = addUnitOffset;

            // Gemma initialisiert die Gewichte mit 0 und addiert später 1.0 dazu!
            this.weight = Parameter(addUnitOffset ? zeros(shape) : ones(shape));
            this.register_parameter("weight", this.weight);
        }

        public override Tensor forward(Tensor x)
        {
            using var x_f32 = x.to(ScalarType.Float32);

            using var x_squared = x_f32.pow(2);
            using var variance = x_squared.mean(new long[] { -1 }, keepdim: true);
            using var rsqrt = torch.rsqrt(variance + this.eps);

            using var normed_f32 = x_f32 * rsqrt;
            using var weight_f32 = this.weight.to(ScalarType.Float32);

            Tensor result_f32;

            if (this.addUnitOffset)
            {
                // WICHTIG FÜR GEMMA: weight + 1.0!
                using var w_plus_1 = weight_f32 + 1.0f;
                result_f32 = normed_f32 * w_plus_1;
            }
            else
            {
                result_f32 = normed_f32 * weight_f32;
            }

            var result = result_f32.to(x.dtype);
            result_f32.Dispose();

            return result;
        }
    }
}