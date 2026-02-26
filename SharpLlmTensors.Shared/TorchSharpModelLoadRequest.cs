using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace SharpLlmTensors.Shared
{
    public class TorchSharpModelLoadRequest
    {
        public TorchSharpModel Model { get; set; }

        public bool ForceCpu { get; set; } = false;

        public string ScalarT { get; set; } = "Float16";

        public bool StrictLoadingMode { get; set; } = false;


        public TorchSharpModelLoadRequest(TorchSharpModel model, bool forceCpu = false, string scalarT = "Float16", bool strictLoading = false)
        {
            this.Model = model ?? throw new ArgumentNullException(nameof(model));
            this.ForceCpu = forceCpu;
            this.ScalarT = scalarT;
            this.StrictLoadingMode = strictLoading;
        }
    }


    public class TorchSharpModelLoadResponse
    {
        public bool Success { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public int LoadingElapsedMs { get; set; } = 0;
        public TorchSharpModel? LoadedModel { get; set; }

        public HardwareStatistics? HardwareStatsBeforeLoad { get; set; }
        public HardwareStatistics? HardwareStatsAfterLoad { get; set; }

        public double? CpuMemoryOffloadSizeMb => (this.HardwareStatsBeforeLoad != null && this.HardwareStatsAfterLoad != null) ? this.HardwareStatsAfterLoad.RamStats.UsedMemoryMb - this.HardwareStatsBeforeLoad.RamStats.UsedMemoryMb : null;
        public double? GpuMemoryOffloadSizeMb => (this.HardwareStatsBeforeLoad != null && this.HardwareStatsAfterLoad != null) ? this.HardwareStatsAfterLoad.RamStats.UsedMemoryMb - this.HardwareStatsBeforeLoad.RamStats.UsedMemoryMb : null;

        public List<string> ErrorMessages { get; set; } = [];

    }
}
