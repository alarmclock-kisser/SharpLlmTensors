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

        public RMSNorm(long[] shape, double eps = 1e-6) : base("RMSNorm")
        {
            this.eps = eps;
            this.weight = Parameter(ones(shape));
            this.register_parameter("weight", this.weight);
            TorchService.LogVerbose($"[RMSNorm] Created with shape: {string.Join(',', shape)} eps: {eps}");
        }

        public override Tensor forward(Tensor x)
        {
            TorchService.LogVerbose($"[RMSNorm] Forward called. input shape: {string.Join(',', x.shape)}");
            // WICHTIG: Cast auf Float32 vor der Quadrierung, um Overflow/NaN zu verhindern!
            using var x_f32 = x.to(ScalarType.Float32);

            using var x_squared = x_f32.pow(2);
            using var variance = x_squared.mean(new long[] { -1 }, keepdim: true);
            using var rsqrt = torch.rsqrt(variance + this.eps);

            using var normed_f32 = x_f32 * rsqrt;

            // Auch das Gewicht kurz als Float32 anwenden
            using var weight_f32 = this.weight.to(ScalarType.Float32);
            using var result_f32 = normed_f32 * weight_f32;

            // Das sichere, normalisierte Ergebnis zurück auf Float16 casten
            TorchService.LogVerbose($"[RMSNorm] Forward completed. output shape will be same as input: {string.Join(',', x.shape)}");
            return result_f32.to(x.dtype);
        }
    }
}