// Controlled full-tick GPU benchmark. Candidate generation and compilation are
// outside each timed evaluation; both binaries use identical masks and settings.
#include "gpu.hpp"
#include "moderators.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

using Clock=std::chrono::steady_clock;
template<class G> std::unique_ptr<G> make_gpu(int batch,const er2::Settings& settings) {
    if constexpr(std::is_constructible_v<G,bool,int,int,int,const er2::Settings*>)
        return std::make_unique<G>(false,0,batch,settings.cells(),&settings);
    else return std::make_unique<G>(false,0,batch,settings.cells());
}
template<class G> void configure(G& gpu) {
    if constexpr(requires { gpu.diagnostics(); gpu.timings(); }) {
        gpu.set_profiling(std::getenv("ER2_GPU_PROFILE")!=nullptr);
        const auto& d=gpu.diagnostics();
        std::cerr<<"GPU: input="<<d.input_bytes_per_layout<<" output="<<d.output_bytes_per_layout
                 <<" block="<<d.block_threads<<" registers="<<d.registers_per_thread<<" local="<<d.local_bytes_per_thread
                 <<" fma="<<d.fma<<" packed="<<d.packed_rays<<" unroll="<<d.unroll_rays<<" reciprocals="<<d.reciprocal_divisors<<'\n';
    }
}
template<class G> void details(G& gpu,const std::string& label) {
    if constexpr(requires { gpu.timings(); }) {
        if(std::getenv("ER2_GPU_PROFILE")) {
            const auto& t=gpu.timings();
            std::cerr<<label<<" ms: pack="<<t.packing_ms<<" upload="<<t.upload_ms<<" kernel="<<t.kernel_ms
                     <<" download="<<t.download_ms<<" unpack="<<t.unpacking_ms<<" total="<<t.total_ms<<'\n';
        }
    }
}
int main(int argc,char** argv) {
    try {
        int batch=argc>1?std::atoi(argv[1]):65536;
        int repeats=argc>2?std::atoi(argv[2]):7;
        std::string output=argc>3?argv[3]:"";
        int width=argc>4?std::atoi(argv[4]):7;
        int depth=argc>5?std::atoi(argv[5]):7;
        int height=argc>6?std::atoi(argv[6]):7;
        std::string material=argc>7?argv[7]:"unobtainium";
        double insertion=argc>8?std::atof(argv[8]):0;
        double fill=argc>9?std::atof(argv[9]):1;
        if(batch<1||batch>131072||repeats<1||width<1||width>32||depth<1||depth>32||height<1||height>64)throw std::runtime_error("Invalid benchmark arguments");
        if(!std::isfinite(insertion)||insertion<0||insertion>1||!std::isfinite(fill)||fill<=0||fill>1)
            throw std::runtime_error("Benchmark insertion must be 0..1 and fill must be greater than 0 and at most 1");
        er2::Settings settings;settings.width=width;settings.depth=depth;settings.height=height;
        settings.insertion=insertion;settings.fill=fill;
        bool found=false;for(const auto& preset:er2::moderators)if(material==preset.key){settings.moderator=preset.material;found=true;break;}
        if(!found)throw std::runtime_error("Unknown moderator");
        auto pointer=make_gpu<er2::Gpu>(batch,settings);auto& gpu=*pointer;configure(gpu);
        er2::SimConfig config;
        std::vector<er2::Layout> masks(batch);std::vector<er2::Result> results(batch);
        std::ofstream saved;if(!output.empty())saved.open(output,std::ios::binary);
        if(!output.empty()&&!saved)throw std::runtime_error("Cannot create result fixture");
        std::cout<<std::setprecision(10)<<"case,batch,ticks,median_seconds,layouts_per_second,min_seconds,max_seconds\n";
        for(int rods:std::vector<int>{1,std::min(8,settings.cells()),std::min(25,settings.cells()),settings.cells(),0}) {
            std::mt19937_64 rng(1234567);
            std::vector<int> cells(settings.cells());std::iota(cells.begin(),cells.end(),0);
            for(auto& mask:masks){mask={};std::shuffle(cells.begin(),cells.end(),rng);int count=rods?rods:1+int(rng()%settings.cells());for(int i=0;i<count;++i)mask.set(cells[i]);}
            gpu.evaluate(masks,results,settings,config); // warm the exact workload
            std::vector<double> seconds;
            for(int r=0;r<repeats;++r){auto begin=Clock::now();gpu.evaluate(masks,results,settings,config);seconds.push_back(std::chrono::duration<double>(Clock::now()-begin).count());}
            std::sort(seconds.begin(),seconds.end());double median=seconds[seconds.size()/2];
            std::cout<<(rods?std::to_string(rods):"uniform")<<','<<batch<<','<<config.max_ticks<<','<<median<<','<<batch/median<<','<<seconds.front()<<','<<seconds.back()<<'\n';
            if(saved) saved.write(reinterpret_cast<const char*>(results.data()),std::streamsize(results.size()*sizeof(er2::Result)));
            details(gpu,rods?std::to_string(rods):"uniform");
        }
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
