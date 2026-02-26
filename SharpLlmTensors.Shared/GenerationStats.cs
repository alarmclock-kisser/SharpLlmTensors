using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLlmTensors.Shared
{
    public class GenerationStats
    {
        public DateTime GenerationStarted { get; set; } = DateTime.UtcNow;
        public DateTime? GenerationFinished { get; set; } = null;

        public int TotalTokensGenerated { get; set; }

        public TimeSpan TotalGenerationTime => this.GenerationFinished.HasValue ? this.GenerationFinished.Value - this.GenerationStarted : DateTime.UtcNow - this.GenerationStarted;

        public double AverageTimePerToken => this.TotalTokensGenerated > 0 ? this.TotalGenerationTime.TotalSeconds / this.TotalTokensGenerated : 0.0;
        public double TokensPerSecond => this.TotalGenerationTime.TotalSeconds > 0 ? this.TotalTokensGenerated / this.TotalGenerationTime.TotalSeconds : 0.0;



    }
}
