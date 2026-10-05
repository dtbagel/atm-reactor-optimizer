#include "gpu.hpp"
#include "moderators.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double float_tolerance=0.002;
struct Summary {
    int configurations=0,comparisons=0,floor_changes=0,ranked_fixtures=0;
    double max_power=0,max_fuel=0,max_efficiency=0,average_power=0,average_fuel=0,average_efficiency=0;
    double minimum_overlap=1;
};
struct ExactEnvironment {
    struct Entry {const char* name;std::string previous;bool existed;};
    std::vector<Entry> entries;
    ExactEnvironment() {
        // Exact-GPU verification explicitly retains its original arithmetic.
        for(const char* name:{"ER2_GPU_FMAD","ER2_GPU_RECIPROCAL_DIVISORS"}) {
            const char* previous=std::getenv(name);
            entries.push_back({name,previous?previous:"",previous!=nullptr});
#ifdef _WIN32
            _putenv_s(name,"0");
#else
            setenv(name,"0",1);
#endif
        }
    }
    ~ExactEnvironment() {
        for(const auto& entry:entries) {
#ifdef _WIN32
            _putenv_s(entry.name,entry.existed?entry.previous.c_str():"");
#else
            if(entry.existed) setenv(entry.name,entry.previous.c_str(),1);else unsetenv(entry.name);
#endif
        }
    }
};
double relative(double actual,double expected) {
    if(!std::isfinite(actual)||!std::isfinite(expected)) throw std::runtime_error("Non-finite evaluation result");
    return std::abs(actual-expected)/std::max(std::abs(expected),1e-12);
}
std::vector<er2::Layout> layouts(er2::Settings settings,int count,std::mt19937_64& random) {
    std::vector<er2::Layout> masks(count);
    masks[1].set(0);masks[2].set(settings.cells()/2);masks[3]=er2::full_mask(settings.cells());
    for(int cell=0;cell<settings.cells();++cell)
        if((cell%settings.width+cell/settings.width)%2==0) masks[4].set(cell);
    if(settings.width==7&&settings.depth==7) masks[5]=0x2c28b0000ULL;
    else masks[5].set(settings.cells()-1);
    std::vector<int> cells(settings.cells());std::iota(cells.begin(),cells.end(),0);
    for(int i=6;i<count;++i) {
        std::shuffle(cells.begin(),cells.end(),random);
        const int rods=(i%4==0)?1+int(random()%settings.cells()):1+int(random()%std::min(16,settings.cells()));
        for(int j=0;j<rods;++j) masks[i].set(cells[j]);
    }
    return masks;
}
double overlap(const std::vector<er2::Result>& expected,const std::vector<er2::Result>& actual) {
    std::vector<int> reference(expected.size()),fast(expected.size());std::iota(reference.begin(),reference.end(),0);fast=reference;
    auto rank=[&](auto& indexes,const auto& results) {
        std::sort(indexes.begin(),indexes.end(),[&](int a,int b) {
            if(results[a].efficiency!=results[b].efficiency) return results[a].efficiency>results[b].efficiency;
            if(results[a].power!=results[b].power) return results[a].power>results[b].power;
            return a<b;
        });
    };
    rank(reference,expected);rank(fast,actual);
    const int finalists=std::min(32,int(reference.size()));int shared=0;
    for(int i=0;i<finalists;++i) shared+=std::find(fast.begin(),fast.begin()+finalists,reference[i])!=fast.begin()+finalists;
    return double(shared)/finalists;
}
void compare(er2::Gpu& gpu,const std::vector<er2::Layout>& masks,er2::Settings settings,er2::SimConfig config,
             bool exact,Summary& summary,const std::string& label) {
    std::vector<er2::Result> expected(masks.size()),actual(masks.size());
    for(std::size_t i=0;i<masks.size();++i) expected[i]=er2::evaluate_fixed<double>(masks[i],settings,config);
    gpu.evaluate(masks,actual,settings,config);
    const double tolerance=exact?1e-8:float_tolerance;
    double fixture_power=0,fixture_fuel=0,fixture_efficiency=0;int changed=0;
    for(std::size_t i=0;i<masks.size();++i) {
        if(actual[i].mask!=masks[i] || actual[i].rods!=expected[i].rods || actual[i].ticks!=expected[i].ticks)
            throw std::runtime_error(label+": grouped/compact output did not preserve identity");
        const double power=relative(actual[i].power,expected[i].power),fuel=relative(actual[i].fuel,expected[i].fuel);
        const double efficiency=relative(actual[i].efficiency,expected[i].efficiency);
        fixture_power=std::max(fixture_power,power);fixture_fuel=std::max(fixture_fuel,fuel);fixture_efficiency=std::max(fixture_efficiency,efficiency);
        summary.max_power=std::max(summary.max_power,power);summary.max_fuel=std::max(summary.max_fuel,fuel);
        summary.max_efficiency=std::max(summary.max_efficiency,efficiency);
        summary.average_power+=power;summary.average_fuel+=fuel;summary.average_efficiency+=efficiency;++summary.comparisons;
        if(std::max({power,fuel,efficiency})>tolerance)
            throw std::runtime_error(label+": mask "+std::to_string(i)+" exceeded numerical tolerance (power="+
                                     std::to_string(power)+", fuel="+std::to_string(fuel)+", efficiency="+std::to_string(efficiency)+")");
        constexpr double floor=350000;
        if((actual[i].power>=floor)!=(expected[i].power>=floor)) {
            ++changed;
            if(std::abs(expected[i].power-floor)>floor*tolerance)
                throw std::runtime_error(label+": floor classification changed outside the power-error envelope");
        }
    }
    const double shared=overlap(expected,actual);summary.minimum_overlap=std::min(summary.minimum_overlap,shared);
    ++summary.ranked_fixtures;summary.floor_changes+=changed;++summary.configurations;
    std::cout<<label<<" ticks="<<config.max_ticks<<" samples="<<config.sample_ticks
             <<" max errors % power/fuel/efficiency="<<fixture_power*100<<'/'<<fixture_fuel*100<<'/'<<fixture_efficiency*100
             <<" top32 overlap="<<shared*100<<"% floor changes="<<changed<<'\n';
}
void report(const char* name,const Summary& summary) {
    if(!summary.comparisons) return;
    std::cout<<name<<" PASS: "<<summary.configurations<<" configurations, "<<summary.comparisons<<" layouts; max errors % power/fuel/efficiency="
             <<summary.max_power*100<<'/'<<summary.max_fuel*100<<'/'<<summary.max_efficiency*100
             <<"; average errors %="<<summary.average_power/summary.comparisons*100<<'/'<<summary.average_fuel/summary.comparisons*100
             <<'/'<<summary.average_efficiency/summary.comparisons*100<<"; minimum top32 overlap="<<summary.minimum_overlap*100
             <<"%; floor changes="<<summary.floor_changes<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        int limit=68,stride=1,count=128,double_stride=0;
        for(int i=1;i<argc;++i) {
            const std::string option=argv[i];
            if(i+1>=argc) throw std::runtime_error("Options require a value");
            const int value=std::stoi(argv[++i]);
            if(option=="--limit") limit=value;else if(option=="--stride") stride=value;
            else if(option=="--layouts") count=value;else if(option=="--double-stride") double_stride=value;
            else throw std::runtime_error("Unknown option: "+option);
        }
        if(limit<1||limit>68||stride<1||count<8||count>8192||double_stride<0) throw std::runtime_error("Invalid fixture bounds");
        std::cout<<std::fixed<<std::setprecision(8);
        constexpr int shapes[][3]={{7,7,7},{3,5,4},{1,9,6},{9,1,8},{9,8,5},{16,12,32},{32,32,64},{1,1,1}};
        std::mt19937_64 random(919);Summary fast_summary,exact_summary;
        for(int index=0;index<limit;index+=stride) {
            const auto& preset=er2::moderators[index];const auto& shape=shapes[index%8];
            er2::Settings settings;settings.width=shape[0];settings.depth=shape[1];settings.height=shape[2];settings.moderator=preset.material;
            settings.insertion=index%3==0?0.25:0;settings.fill=index%4==0?0.6:1;settings.variant=index%5==0?1.2:1;
            auto masks=layouts(settings,count,random);
            er2::Gpu fast(false,0,count,settings.cells(),&settings);
            compare(fast,masks,settings,{4500,1500,500},false,fast_summary,preset.key);
            // The entire trajectory is a sample, then a one-tick sample after warmup.
            compare(fast,masks,settings,{7,1,500},false,fast_summary,std::string(preset.key)+"/short");
            compare(fast,masks,settings,{17,1,1},false,fast_summary,std::string(preset.key)+"/tail");
            if(double_stride>0&&index%double_stride==0) {
                ExactEnvironment original_arithmetic;
                er2::Gpu exact(true,0,count,settings.cells(),&settings);
                compare(exact,masks,settings,{4500,1500,500},true,exact_summary,std::string(preset.key)+"/double");
                compare(exact,masks,settings,{7,1,500},true,exact_summary,std::string(preset.key)+"/double-short");
            }
        }
        // The user's target, a deliberately ambiguous floor, and vanishing rays
        // complement the material/shape matrix without changing sample lengths.
        for(int boundary=0;boundary<4;++boundary) {
            er2::Settings settings;
            std::string label="target-unobtainium";
            if(boundary==1) {
                const auto reference=er2::evaluate_fixed<double>(0x2c28b0000ULL,settings,{4500,1500,500});
                settings.variant=350000/reference.power;label+="/near-floor";
            }
            if(boundary==2) {settings.insertion=1;label+="/fully-inserted";}
            if(boundary==3) {settings.fill=1e-8;label+="/tiny-fill";}
            const auto masks=layouts(settings,count,random);
            er2::Gpu fast(false,0,count,settings.cells(),&settings);
            compare(fast,masks,settings,{4500,1500,500},false,fast_summary,label);
        }
        report("GPU float",fast_summary);report("GPU double (original arithmetic)",exact_summary);
    } catch(const std::exception& error) {std::cerr<<"GPU accuracy failure: "<<error.what()<<'\n';return 1;}
}
