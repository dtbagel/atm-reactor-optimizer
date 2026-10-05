template<class T> struct CompactResult {
    T power, fuel, efficiency, fuel_heat, reactor_heat, fertility;
    int rods, ticks;
};
#ifdef ER2_GPU_DOUBLE
using GpuScalar=double;
#else
using GpuScalar=float;
#endif
static_assert(sizeof(CompactResult<float>)==32);
static_assert(sizeof(CompactResult<double>)==56);
extern "C" __global__ void evaluate_layouts(const er2::Layout* layouts, CompactResult<GpuScalar>* output,
                                            int count, er2::Settings settings, er2::SimConfig config) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=count) return;
#ifdef ER2_SPECIALIZE_SETTINGS
    // Compile the selected reactor settings into this kernel without changing
    // the generic API. The host rejects calls with different settings.
    settings.width=ER2_GPU_WIDTH; settings.depth=ER2_GPU_DEPTH; settings.height=ER2_GPU_HEIGHT;
    settings.insertion=ER2_GPU_INSERTION; settings.fill=ER2_GPU_FILL; settings.variant=ER2_GPU_VARIANT;
    settings.moderator.absorption=ER2_GPU_ABSORPTION;
    settings.moderator.heat_efficiency=ER2_GPU_HEAT_EFFICIENCY;
    settings.moderator.moderation=ER2_GPU_MODERATION;
    settings.moderator.conductivity=ER2_GPU_CONDUCTIVITY;
#endif
    auto result=er2::evaluate_fixed<GpuScalar>(layouts[index],settings,config);
    output[index]={GpuScalar(result.power),GpuScalar(result.fuel),GpuScalar(result.efficiency),
                   GpuScalar(result.fuel_heat),GpuScalar(result.reactor_heat),GpuScalar(result.fertility),
                   result.rods,result.ticks};
}
