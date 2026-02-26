using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLlmTensors.Shared
{
    public class AppSettings
    {
        public bool SilentLog { get; set; } = false;
        public bool VerboseLog { get; set; } = false;
        public bool HardwareMonitoring { get; set; } = false;
        public bool ShowGenerationStats { get; set; } = false;

        public string[] ModelDirectories { get; set; } = [];
        public string? DefaultModelName { get; set; } = null;

        public bool ForceCpu { get; set; } = false;


    }
}
