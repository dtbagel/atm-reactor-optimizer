extern "C" __global__ void evaluate_layouts(const er2::Layout* layouts, er2::Result* output,
                                            int count, er2::Settings settings, er2::SimConfig config) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=count) return;
#ifdef ER2_GPU_DOUBLE
    output[index]=er2::evaluate_fixed<double>(layouts[index],settings,config);
#else
    output[index]=er2::evaluate_fixed<float>(layouts[index],settings,config);
#endif
}
