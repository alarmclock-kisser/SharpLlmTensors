using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLlmTensors.Shared
{
    public class HardwareStatistics
    {
        public CpuStatistics CpuStats { get; set; }
        public MemoryStatistics RamStats { get; set; }
        public GpuStatistics GpuStats {  get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public int FetchingDurationMs { get; set; } = 0;


        public HardwareStatistics(IEnumerable<double>? cpuCoreLoads = null, long totalRamBytes = 0, long usedRamBytes = 0, double gpuCoreLoadPercentage = 0, double wattsUsage = 0, double totalVramBytes = 0, double usedVramBytes = 0)
        {
            this.CpuStats = new CpuStatistics(cpuCoreLoads);
            this.RamStats = new MemoryStatistics(totalRamBytes, usedRamBytes);
            this.GpuStats = new GpuStatistics(gpuCoreLoadPercentage, wattsUsage, totalVramBytes, usedVramBytes);
        }
    }



    public class CpuStatistics
    {
        public string Name { get; set; } = "CPU";

        public List<double> CpuCoreLoads { get; set; } = [];
        public int CpuCoreCount => this.CpuCoreLoads.Count;
        public double AverageLoadPercentage => this.CpuCoreLoads.Count > 0 ? this.CpuCoreLoads.Average() : 0;

        public CpuStatistics(IEnumerable<double>? cpuCoreLoads = null)
        {
            this.CpuCoreLoads = cpuCoreLoads?.ToList() ?? [];
        }
    }

    public class MemoryStatistics
    {
        public string Name { get; set; } = "RAM";

        public double TotalMemoryMb { get; set; }
        public double UsedMemoryMb { get; set; }
        public double FreeMemoryMb => this.TotalMemoryMb - this.UsedMemoryMb;
        public double MemoryUsagePercentage => this.TotalMemoryMb > 0 ? (this.UsedMemoryMb / this.TotalMemoryMb) * 100 : 0;

        public MemoryStatistics(double totalMemoryBytes = 0, double usedMemoryBytes = 0)
        {
            this.TotalMemoryMb = totalMemoryBytes / (1024 * 1024);
            this.UsedMemoryMb = usedMemoryBytes / (1024 * 1024);
        }
    }

    public class GpuStatistics
    {
        public string Name { get; set; } = "GPU";

        public double CoreLoadPercentage { get; set; }
        public double WattsUsage { get; set; }
        public MemoryStatistics VramStats { get; set; }


        public GpuStatistics(double gpuCoreLoadPercentage = 0, double wattsUsage = 0, double totalVramBytes = 0, double usedVramBytes = 0)
        {
            this.CoreLoadPercentage = gpuCoreLoadPercentage;
            this.WattsUsage = wattsUsage;
            this.VramStats = new MemoryStatistics(totalVramBytes, usedVramBytes);
        }
    }

}
