#include "simulator.hpp"
#include "gpu.hpp"
#include "engine.hpp"
#include "moderators.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {
thread_local er2::RunControl* run_control=nullptr;
void stage(const std::string& text) { if(run_control&&run_control->stage)run_control->stage(text); }
bool finishing() { return run_control&&run_control->finish.load(); }
using namespace er2;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
struct Options {
    Settings settings;
    SearchTuning search;
    SimConfig sim;
    int threads=std::max(1U,std::thread::hardware_concurrency()), min_rods=1,max_rods=0;
    int batch=65536,device=0,final_ticks=20000,progress_ms=1000,validation_layouts=128;
    double seconds=60,min_power=0;
    unsigned long long seed=1337,evaluations=0;
    std::string backend="auto",math="fast",objective="efficiency",output="best_reactor",moderator="unobtainium";
    bool benchmark_only=false,list_moderators=false;
    bool calibrate=false,benchmark=false,self_test=false,quiet=false,fixed=false,has_mask=false;
    Layout mask=0;
};
void help() {
    std::cout<<"ATM10 ER2 C++20 / CUDA reactor optimizer\n"
        "  --backend auto|cuda|cpu   auto prefers the NVIDIA GPU\n"
        "  --seconds 60             search wall-clock budget (startup/final verification extra)\n"
        "  --threads N              CPU evaluation/verification workers\n"
        "  --min-power 350000       exact verified minimum FE/t\n"
        "  --min-rods 1             all interior positions available by default\n"
        "  --max-rods N             optional CLI constraint, otherwise full footprint\n"
        "  --width 7 --depth 7 --height 7  interior; width/depth 1..32, height 1..64\n"
        "  --moderator unobtainium   preset key; --list-moderators to see all 68\n"
        "  --insertion 0            percent, 0..100\n"
        "  --fill 1 --variant-efficiency 1\n"
        "  --math fast|exact        float discovery or double calculations\n"
        "  --batch 65536 --device 0 GPU batch size and device index\n"
        "  --search-max-ticks 4500 --search-min-ticks 1500\n"
        "  --sample-ticks 500 --final-ticks 20000\n"
        "  --objective efficiency|power --seed 1337\n"
        "  --agents 4 --power-agents 1 --elites 64   independent populations / parents\n"
        "  --mutation-percent 90 --max-flips 5 --move-percent 45\n"
        "  --crossover-percent 20 --random-percent 20\n"
        "  --migration-generations 25 --restart-generations 100 (0 disables)\n"
        "  --evaluations N          fixed candidate budget, overrides seconds (deterministic)\n"
        "  --progress-ms 1000 --output best_reactor --quiet\n"
        "  --calibrate --benchmark --self-test --validation-layouts 128\n"
        "  --benchmark-only         benchmark and exit without a search\n"
        "  --evaluate 0x1555555555555 [--fixed-ticks]  inspect one mask\n"
        "One evaluation = one complete layout simulation; ticks/s is reported separately.\n";
}
Options parse(int argc,char** argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        auto value=[&]() -> std::string { if (i+1>=argc) throw std::runtime_error("Missing value for "+key); return argv[++i]; };
        auto integer=[&]() { auto s=value();size_t n;long v=std::stol(s,&n);if(n!=s.size()||v<0||v>1000000000)throw std::runtime_error("Invalid integer for "+key);return int(v); };
        auto real=[&]() { auto s=value();size_t n;double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v))throw std::runtime_error("Invalid number for "+key);return v; };
        auto wide=[&]() { auto s=value();size_t n;if(s.starts_with('-'))throw std::runtime_error("Invalid value for "+key);auto v=std::stoull(s,&n,0);if(n!=s.size())throw std::runtime_error("Invalid value for "+key);return v; };
        if (key=="--help"||key=="-h") { help();std::exit(0); }
        else if(key=="--threads")o.threads=integer();
        else if(key=="--agents")o.search.agents=integer();
        else if(key=="--power-agents")o.search.power_agents=integer();
        else if(key=="--elites")o.search.elites=integer();
        else if(key=="--mutation-percent")o.search.mutation_percent=integer();
        else if(key=="--max-flips")o.search.max_flips=integer();
        else if(key=="--move-percent")o.search.move_percent=integer();
        else if(key=="--crossover-percent")o.search.crossover_percent=integer();
        else if(key=="--random-percent")o.search.random_percent=integer();
        else if(key=="--migration-generations")o.search.migration_generations=integer();
        else if(key=="--restart-generations")o.search.restart_generations=integer();
        else if(key=="--seconds")o.seconds=real();
        else if(key=="--min-rods")o.min_rods=integer();
        else if(key=="--max-rods")o.max_rods=integer();
        else if(key=="--width")o.settings.width=integer();
        else if(key=="--depth")o.settings.depth=integer();
        else if(key=="--height")o.settings.height=integer();
        else if(key=="--moderator")o.moderator=value();
        else if(key=="--list-moderators")o.list_moderators=true;
        else if(key=="--min-power")o.min_power=real();
        else if(key=="--insertion")o.settings.insertion=real()/100;
        else if(key=="--fill")o.settings.fill=real();
        else if(key=="--variant-efficiency")o.settings.variant=real();
        else if(key=="--search-max-ticks")o.sim.max_ticks=integer();
        else if(key=="--search-min-ticks")o.sim.min_ticks=integer();
        else if(key=="--sample-ticks")o.sim.sample_ticks=integer();
        else if(key=="--final-ticks")o.final_ticks=integer();
        else if(key=="--batch")o.batch=integer();
        else if(key=="--device")o.device=integer();
        else if(key=="--progress-ms")o.progress_ms=integer();
        else if(key=="--validation-layouts")o.validation_layouts=integer();
        else if(key=="--seed")o.seed=wide();
        else if(key=="--evaluations")o.evaluations=wide();
        else if(key=="--backend")o.backend=value();
        else if(key=="--math")o.math=value();
        else if(key=="--objective")o.objective=value();
        else if(key=="--output")o.output=value();
        else if(key=="--calibrate")o.calibrate=true;
        else if(key=="--benchmark")o.benchmark=true;
        else if(key=="--benchmark-only")o.benchmark=o.benchmark_only=true;
        else if(key=="--self-test")o.self_test=true;
        else if(key=="--quiet")o.quiet=true;
        else if(key=="--fixed-ticks")o.fixed=true;
        else if(key=="--evaluate") {
            auto s=value();o.has_mask=true;
            if(s.starts_with("0x")||s.starts_with("0X")) {
                s.erase(0,2); if(s.empty()||s.size()>256)throw std::runtime_error("Invalid hex mask");
                int bit=0;for(auto it=s.rbegin();it!=s.rend();++it,bit+=4) {
                    int digit=(*it>='0'&&*it<='9')?*it-'0':(*it>='a'&&*it<='f')?*it-'a'+10:(*it>='A'&&*it<='F')?*it-'A'+10:-1;
                    if(digit<0)throw std::runtime_error("Invalid hex mask");
                    for(int j=0;j<4;++j)if(digit&(1<<j))o.mask.set(bit+j);
                }
            } else {size_t n;auto v=std::stoull(s,&n);if(n!=s.size()||s.starts_with('-'))throw std::runtime_error("Invalid mask");o.mask=Layout(v);}
        }
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(o.threads<1||o.threads>1024)throw std::runtime_error("threads must be 1..1024");
    if(!valid_tuning(o.search))
        throw std::runtime_error("agents 1..32, elites 1..256, max-flips 1..1024, percentages 0..100, generation intervals 0..1000000");
    if(o.seconds<=0||o.min_power<0)throw std::runtime_error("seconds must be positive; min-power must be nonnegative");
    if(o.settings.width<1||o.settings.width>32||o.settings.depth<1||o.settings.depth>32||o.settings.height<1||o.settings.height>64)throw std::runtime_error("interior width/depth must be 1..32; height 1..64");
    bool explicit_max=false;for(int i=1;i<argc;++i)if(std::string(argv[i])=="--max-rods")explicit_max=true;
    if(!explicit_max)o.max_rods=o.settings.cells();
    if(o.min_rods<1||o.max_rods>o.settings.cells()||o.min_rods>o.max_rods)throw std::runtime_error("rod range must fit the interior footprint");
    auto material=find_moderator(o.moderator);if(!material)throw std::runtime_error("Unknown moderator: "+o.moderator);
    o.settings.moderator=material->material;
    if(o.settings.insertion<0||o.settings.insertion>1||o.settings.fill<=0||o.settings.fill>1||o.settings.variant<=0)throw std::runtime_error("invalid insertion, fill, or variant efficiency");
    if(o.sim.min_ticks<1||o.sim.max_ticks<o.sim.min_ticks||o.sim.sample_ticks<1||o.sim.sample_ticks>8192||o.sim.sample_ticks>o.sim.max_ticks||o.final_ticks<o.sim.sample_ticks)throw std::runtime_error("invalid simulation tick limits (sample-ticks 1..8192)");
    if(o.batch<1||o.batch>131072||o.progress_ms<1||o.validation_layouts<4||o.validation_layouts>4096)throw std::runtime_error("invalid batch, progress, or validation count");
    if(o.backend!="auto"&&o.backend!="cpu"&&o.backend!="cuda")throw std::runtime_error("backend must be auto, cpu, or cuda");
    if(o.math!="exact"&&o.math!="fast")throw std::runtime_error("math must be exact or fast");
    if(o.objective!="efficiency"&&o.objective!="power")throw std::runtime_error("objective must be efficiency or power");
    if(o.has_mask&&(!o.mask||(o.mask&~full_mask(o.settings.cells()))))throw std::runtime_error("mask must contain fuel within the interior footprint");
    return o;
}
// Persistent workers, with no thread creation or synchronization inside a simulation.
class Pool {
    std::mutex mutex;
    std::condition_variable ready,done;
    std::vector<std::thread> workers;
    std::function<void(size_t)> job;
    std::atomic<size_t> next{0};
    size_t count=0;
    unsigned generation=0;
    int active=0;
    bool stopping=false;
public:
    explicit Pool(int threads) {
        for(int i=0;i<threads;++i)workers.emplace_back([this] {
            unsigned seen=0;
            for(;;) {
                std::unique_lock lock(mutex);
                ready.wait(lock,[&]{return stopping||generation!=seen;});
                if(stopping)return;
                seen=generation;lock.unlock();
                for(;;) {size_t index=next.fetch_add(1,std::memory_order_relaxed);if(index>=count)break;job(index);}
                lock.lock();if(--active==0)done.notify_one();
            }
        });
    }
    ~Pool() { {std::lock_guard lock(mutex);stopping=true;}ready.notify_all();for(auto& t:workers)t.join(); }
    void run(size_t size,std::function<void(size_t)> function) {
        if(!size)return;
        std::unique_lock lock(mutex);job=std::move(function);count=size;next=0;active=int(workers.size());++generation;
        ready.notify_all();done.wait(lock,[&]{return active==0;});job={};
    }
};
Layout lattice(int spacing,Settings settings={}) {
    Layout mask;for(int z=0;z<settings.depth;z+=spacing)for(int x=0;x<settings.width;x+=spacing)mask.set(z*settings.width+x);return mask;
}
Layout checker(Settings settings) {
    Layout m;for(int z=0;z<settings.depth;++z)for(int x=0;x<settings.width;++x)if((x+z)%2==0)m.set(z*settings.width+x);return m;
}
Layout cross(Settings settings) {
    Layout m;for(int cell=0;cell<settings.cells();++cell)if(cell%settings.width==settings.width/2||cell/settings.width==settings.depth/2)m.set(cell);return m;
}
Layout random_layout(std::mt19937_64& rng,int lo,int hi,int cells,bool uniform=false) {
    int n=lo+int(rng()%unsigned(hi-lo+1));if(!uniform)n=std::min(n,lo+int(rng()%unsigned(hi-lo+1)));
    Layout mask;int count=0;while(count<n) {int cell=int(rng()%cells);if(!mask.test(cell)){mask.set(cell);++count;}}return mask;
}
Layout mutate(Layout mask,std::mt19937_64& rng,int lo,int hi,int cells,const SearchTuning& tuning) {
    int count=rod_count(mask);
    if(rng()%100<unsigned(tuning.move_percent) && count>0 && count<cells) {
        int on,off;do{on=int(rng()%cells);}while(!mask.test(on));do{off=int(rng()%cells);}while(mask.test(off));
        mask.toggle(on);mask.toggle(off);return mask;
    }
    // Sample distinct positions, so a wider mutation cannot undo its own flips.
    int flips=1+int(rng()%unsigned(std::min(cells,tuning.max_flips)));Layout touched;
    for(int i=0;i<flips;++i) {int cell;do{cell=int(rng()%cells);}while(touched.test(cell));touched.set(cell);int next=count+(mask.test(cell)?-1:1);if(next>=lo&&next<=hi){mask.toggle(cell);count=next;}}
    return mask;
}
bool valid(const Result& r) {return r.rods>0&&r.fuel>0&&std::isfinite(r.efficiency)&&std::isfinite(r.power);}
double score(const Result& r,const Options& o) {
    if(!valid(r)||r.power<o.min_power)return -std::numeric_limits<double>::infinity();
    return o.objective=="power"?r.power:r.efficiency;
}
bool better(const Result& a,const Result& b,const Options& o) {
    double sa=score(a,o),sb=score(b,o);
    if(sa!=sb)return sa>sb;
    // When the entire population misses the floor, breed toward greater power.
    if(!std::isfinite(sa)&&a.power!=b.power)return a.power>b.power;
    return a.mask<b.mask;
}
std::string quote(const std::string& value) {
    std::string out="\"";for(unsigned char c:value) {if(c=='\"'||c=='\\'){out+='\\';out+=char(c);}else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c<32){char buf[7];std::snprintf(buf,7,"\\u%04x",c);out+=buf;}else out+=char(c);}return out+'\"';
}
void print_result(const Result& r,const Options& o) {
    std::cout<<layout_text(r.mask,o.settings)<<std::fixed<<std::setprecision(3)
        <<"Power: "<<r.power<<" FE/t\nFuel: "<<std::setprecision(9)<<r.fuel
        <<" mB/t\nEfficiency: "<<std::setprecision(3)<<r.efficiency<<" FE/mB\nRod columns: "<<r.rods
        <<"\nFuel heat: "<<r.fuel_heat<<" C\nReactor heat: "<<r.reactor_heat<<" C\nFertility: "<<r.fertility
        <<"\nMask: "<<mask_hex(r.mask)<<"\nTicks: "<<r.ticks<<'\n';
}
void save(const Result& r,const Options& o,const std::string& backend,const std::string& device,
          double runtime,unsigned long long candidates,unsigned long long ticks,bool verified,const Progress* stats=nullptr,bool live_changed=false) {
    auto path_utf8=[](const std::string& s){return std::filesystem::path(std::u8string(s.begin(),s.end()));};
    std::filesystem::path prefix=path_utf8(o.output);
    if(!prefix.parent_path().empty())std::filesystem::create_directories(prefix.parent_path());
    std::ofstream json(path_utf8(o.output+".json")),txt(path_utf8(o.output+".txt"));
    if(!json||!txt)throw std::runtime_error("Cannot write result output");
    json<<std::setprecision(17)<<"{\n  \"settings\": {\n"
        <<"    \"threads\": "<<o.threads<<", \"seconds\": "<<o.seconds<<", \"seed\": "<<o.seed
        <<", \"evaluation_budget\": "<<o.evaluations<<",\n    \"min_rods\": "<<o.min_rods<<", \"max_rods\": "<<o.max_rods
        <<",\n    \"search_tuning\": {\"agents\": "<<o.search.agents<<", \"power_agents\": "<<o.search.power_agents<<", \"elites\": "<<o.search.elites
        <<", \"mutation_percent\": "<<o.search.mutation_percent<<", \"max_flips\": "<<o.search.max_flips<<", \"move_percent\": "<<o.search.move_percent
        <<", \"crossover_percent\": "<<o.search.crossover_percent<<", \"random_percent\": "<<o.search.random_percent
        <<", \"migration_generations\": "<<o.search.migration_generations<<", \"restart_generations\": "<<o.search.restart_generations<<"}"
        <<", \"min_power\": "<<o.min_power<<",\n    \"insertion_percent\": "<<o.settings.insertion*100
        <<", \"fuel_fill\": "<<o.settings.fill<<", \"variant_efficiency\": "<<o.settings.variant
        <<",\n    \"search_max_ticks\": "<<o.sim.max_ticks<<", \"search_min_ticks\": "<<o.sim.min_ticks
        <<", \"sample_ticks\": "<<o.sim.sample_ticks<<", \"final_ticks\": "<<o.final_ticks
        <<", \"final_min_ticks\": "<<std::min(8000,std::max(2000,o.final_ticks/3))
        <<", \"final_sample_ticks\": "<<std::max(o.sim.sample_ticks,std::min(1000,o.final_ticks))
        <<",\n    \"backend_requested\": "<<quote(o.backend)<<", \"backend\": "<<quote(backend)
        <<", \"math\": "<<quote(o.math)<<", \"objective\": "<<quote(o.objective)<<", \"fixed_ticks\": "<<(o.fixed?"true":"false")
        <<",\n    \"batch\": "<<o.batch<<", \"device_index\": "<<o.device<<", \"device_name\": "<<quote(device)
        <<",\n    \"geometry\": ["<<o.settings.width<<","<<o.settings.depth<<","<<o.settings.height<<"], \"atm10_fuel_multiplier\": 0.8, \"atm10_power_multiplier\": 12,\n"
        <<"    \"moderator\": {\"key\": "<<quote(o.moderator)<<", \"name\": "<<quote(find_moderator(o.moderator)->name)
        <<", \"absorption\": "<<o.settings.moderator.absorption<<", \"heat_efficiency\": "<<o.settings.moderator.heat_efficiency
        <<", \"moderation\": "<<o.settings.moderator.moderation<<", \"conductivity\": "<<o.settings.moderator.conductivity<<"},\n"
        <<"    \"source_schedule\": \"one event per tick, columns cycled as in ato.py\"\n  },\n"
        <<"  \"run\": {\"search_seconds\": "<<runtime<<", \"candidate_evaluations\": "<<candidates
        <<", \"reactor_ticks\": "<<ticks<<", \"layouts_per_second\": "<<(runtime>0?candidates/runtime:0)
        <<", \"generations\": "<<(stats?stats->generation:0)<<", \"population_restarts\": "<<(stats?stats->restarts:0)
        <<", \"champion_improvements\": "<<(stats?stats->improvements:0)<<", \"live_tuning_changed\": "<<(live_changed?"true":"false")
        <<", \"exact_cpu_verified\": "<<(verified?"true":"false")<<"},\n  \"result\": {\n"
        <<"    \"mask\": "<<(o.settings.cells()<=64?std::to_string(r.mask.words[0]):quote(mask_hex(r.mask)))<<", \"layout\": "<<quote(layout_text(r.mask,o.settings))<<", \"rod_columns\": "<<r.rods
        <<", \"rod_blocks\": "<<r.rods*o.settings.height<<",\n    \"score\": "<<(o.objective=="power"?r.power:r.efficiency)
        <<", \"power_fe_t\": "<<r.power<<", \"fuel_mb_t\": "<<r.fuel
        <<", \"efficiency_fe_per_mb\": "<<r.efficiency<<",\n    \"fuel_heat_c\": "<<r.fuel_heat
        <<", \"reactor_heat_c\": "<<r.reactor_heat<<", \"fertility\": "<<r.fertility<<", \"ticks\": "<<r.ticks<<"\n  }\n}\n";
    txt<<"R = fuel column; "<<(o.moderator=="unobtainium"?"U":"M")<<" = "<<find_moderator(o.moderator)->name<<"; each column is "<<o.settings.height<<" blocks tall\nInterior: "<<o.settings.width<<" x "<<o.settings.depth<<" x "<<o.settings.height<<" (width x depth x height)\n\n"<<layout_text(r.mask,o.settings)
        <<std::setprecision(12)<<"\nPower: "<<r.power<<" FE/t\nFuel: "<<r.fuel<<" mB/t\nEfficiency: "<<r.efficiency
        <<" FE/mB\nRod columns: "<<r.rods<<"\nFuel heat: "<<r.fuel_heat<<" C\nReactor heat: "<<r.reactor_heat
        <<" C\nFertility: "<<r.fertility<<"\nSearch time: "<<runtime<<" s\nCandidates: "<<candidates<<"\nBackend: "<<backend
        <<"\nExact CPU verified: "<<(verified?"yes":"no")<<'\n';
    json.flush();txt.flush();if(!json||!txt)throw std::runtime_error("Error writing result output");
}
void evaluate_cpu(Pool& pool,const std::vector<Layout>& masks,std::vector<Result>& results,const Options& o,bool final=false) {
    SimConfig config=o.sim;
    if(final)config={o.final_ticks,std::min(8000,std::max(2000,o.final_ticks/3)),std::max(o.sim.sample_ticks,std::min(1000,o.final_ticks))};
    pool.run(masks.size(),[&](size_t i){results[i]=(final||o.math=="exact")?evaluate_exact(masks[i],o.settings,config,!o.fixed):evaluate_fixed<float>(masks[i],o.settings,config);});
}
void verify_baseline() {
    Topology top(checkerboard());
    if(top.rods!=25||std::abs(top.heat_transfer-3007.2)>1e-8)throw std::runtime_error("Checkerboard topology failed");
    if(Topology(1).rods!=1||std::abs(Topology(1).heat_transfer-78.4)>1e-8||
       std::abs(Topology(1ULL<<24).heat_transfer-140)>1e-8||
       std::abs(Topology(all_mask).heat_transfer-117.6)>1e-8)
        throw std::runtime_error("Boundary heat-transfer topology failed");
    auto r=evaluate_exact(checkerboard(),{}, {20000,6000,1000});
    // ato.py's heat-transfer loop oscillates; tiny libm rounding differences
    // change its final phase. Compare window averages within 0.2%, not bitwise.
    if(std::abs(r.power/529608.8550159342-1)>0.002||std::abs(r.fuel/0.8735003729730746-1)>0.002)
        throw std::runtime_error("Checkerboard calibration failed: power="+std::to_string(r.power)+", fuel="+std::to_string(r.fuel));
}
// Same masks, tick count, averaging phase and settings for both numerical paths.
void validate(Gpu* gpu,Pool& pool,const Options& o) {
    int n=o.validation_layouts;
    std::mt19937_64 rng(o.seed);
    std::vector<Layout> masks(n);masks[0]=checker(o.settings);masks[1]=lattice(3,o.settings);masks[2]=lattice(2,o.settings);masks[3]=full_mask(o.settings.cells());
    for(int i=4;i<n;++i)masks[i]=random_layout(rng,1,o.settings.cells(),o.settings.cells());
    std::vector<Result> exact(n),fast(n);
    pool.run(n,[&](size_t i){exact[i]=evaluate_fixed<double>(masks[i],o.settings,o.sim);});
    if(gpu) {
        for(int i=0;i<n;i+=o.batch) {int count=std::min(o.batch,n-i);gpu->evaluate(std::span(masks).subspan(i,count),std::span(fast).subspan(i,count),o.settings,o.sim);}
    } else pool.run(n,[&](size_t i){fast[i]=evaluate_fixed<float>(masks[i],o.settings,o.sim);});
    double power_error=0,fuel_error=0,score_error=0;
    for(int i=0;i<n;++i) {
        auto relative=[](double a,double b){return std::abs(a-b)/std::max(std::abs(a),1e-12);};
        if(!std::isfinite(fast[i].power)||!std::isfinite(fast[i].fuel)||!std::isfinite(fast[i].efficiency))throw std::runtime_error("Nonfinite accelerated evaluation");
        power_error=std::max(power_error,relative(exact[i].power,fast[i].power));
        fuel_error=std::max(fuel_error,relative(exact[i].fuel,fast[i].fuel));
        score_error=std::max(score_error,relative(exact[i].efficiency,fast[i].efficiency));
    }
    std::vector<int> a(n),b(n);std::iota(a.begin(),a.end(),0);b=a;
    std::sort(a.begin(),a.end(),[&](int i,int j){return exact[i].efficiency>exact[j].efficiency;});
    std::sort(b.begin(),b.end(),[&](int i,int j){return fast[i].efficiency>fast[j].efficiency;});
    std::vector<int> ranks(n);for(int i=0;i<n;++i)ranks[b[i]]=i;
    double squares=0;for(int i=0;i<n;++i){double d=i-ranks[a[i]];squares+=d*d;}
    int k=std::min(16,n),overlap=0;for(int i=0;i<k;++i)if(std::find(b.begin(),b.begin()+k,a[i])!=b.begin()+k)++overlap;
    double rho=1-6*squares/(double(n)*(double(n)*n-1));
    std::cout<<std::setprecision(8)<<"Numerical validation ("<<n<<" layouts): max power error "<<power_error*100
        <<"%, fuel "<<fuel_error*100<<"%, efficiency "<<score_error*100<<"%; rank correlation "<<rho
        <<"; top-"<<k<<" overlap "<<overlap<<'/'<<k<<'\n';
    if(power_error>0.02||fuel_error>0.02||score_error>0.02||rho<0.99||overlap<k-1)throw std::runtime_error("Accelerated numerical validation failed");
}
void benchmark(Gpu* gpu,Pool& pool,const Options& o) {
    std::mt19937_64 rng(o.seed);
    int n=gpu?std::min(o.batch,8192):256;
    std::vector<Layout> masks(n);for(auto& m:masks)m=random_layout(rng,o.min_rods,o.max_rods,o.settings.cells());
    std::vector<Result> results(n);
    auto run=[&](const char* label,auto function) {
        auto start=Clock::now();function();double time=elapsed(start);
        unsigned long long ticks=0;for(auto& r:results)ticks+=r.ticks;
        std::cout<<label<<": "<<n/time<<" layouts/s, "<<ticks/time<<" reactor ticks/s ("<<n<<" layouts, "<<time<<" s)\n";
    };
    run("CPU double, 1 worker, fixed ticks",[&]{for(int i=0;i<n;++i)results[i]=evaluate_fixed<double>(masks[i],o.settings,o.sim);});
    run("CPU double, all requested workers, fixed ticks",[&]{pool.run(n,[&](size_t i){results[i]=evaluate_fixed<double>(masks[i],o.settings,o.sim);});});
    run("CPU float, all requested workers, fixed ticks",[&]{pool.run(n,[&](size_t i){results[i]=evaluate_fixed<float>(masks[i],o.settings,o.sim);});});
    if(gpu)run("CUDA, fixed ticks, including transfers",[&]{gpu->evaluate(masks,results,o.settings,o.sim);});
    std::fill(masks.begin(),masks.end(),checker(o.settings));
    run("Checkerboard CPU double, all requested workers",[&]{pool.run(n,[&](size_t i){results[i]=evaluate_exact(masks[i],o.settings,o.sim);});});
}
struct MaskHash {
    size_t operator()(const Layout& mask) const {
        size_t h=0;for(auto word:mask.words)h^=std::hash<unsigned long long>{}(word)+0x9e3779b9+(h<<6)+(h>>2);return h;
    }
};
std::vector<Result> retain(std::vector<Result>& population,size_t limit,const Options& o) {
    std::vector<Result> kept;kept.reserve(limit);
    if(population.empty())return kept;
    std::unordered_set<Layout,MaskHash> seen;seen.reserve(limit*2);
    size_t shortlist=std::min(std::max<size_t>(limit*8,256),population.size());
    for(;;) {
        auto compare=[&](const Result& a,const Result& b){return better(a,b,o);};
        if(shortlist<population.size())std::nth_element(population.begin(),population.begin()+shortlist,population.end(),compare);
        std::sort(population.begin(),population.begin()+shortlist,compare);
        kept.clear();seen.clear();
        for(size_t i=0;i<shortlist;++i) {
            const auto& r=population[i];
            if(valid(r)&&seen.insert(r.mask).second)kept.push_back(r);
            if(kept.size()==limit)break;
        }
        if(kept.size()==limit||shortlist==population.size())return kept;
        shortlist=std::min(population.size(),shortlist*4);
    }
}
int search(Gpu* gpu,Pool& pool,const Options& options) {
    Options o=options;
    Options power_selection=o;power_selection.objective="power";
    const std::string backend=gpu?"cuda":"cpu",device=gpu?gpu->name():"CPU";
    struct Agent {
        std::mt19937_64 rng;
        std::vector<Result> elites;
        unsigned long long stagnant=0;
        bool fresh=true;
        explicit Agent(unsigned long long seed):rng(seed){}
    };
    std::vector<Agent> agents;
    for(int i=0;i<o.search.agents;++i)agents.emplace_back(o.seed+1000003ULL*i);
    std::vector<Result> archive;
    std::vector<Layout> masks;std::vector<Result> results;
    std::vector<Layout> seeds={checker(o.settings),lattice(3,o.settings),lattice(2,o.settings),cross(o.settings),full_mask(o.settings.cells())};
    seeds.erase(std::remove_if(seeds.begin(),seeds.end(),[&](Layout m){int n=rod_count(m);return n<o.min_rods||n>o.max_rods;}),seeds.end());
    unsigned long long candidates=0,ticks=0,accepted=0,generation=0,restarts=0,last_improvement_candidate=0;
    double evaluation_time=0,generation_time=0,last_improvement_time=0;
    auto start=Clock::now(),next_progress=start;
    auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.seconds));
    int initial_gpu_batch=std::max(128,65536*49/o.settings.cells());
    if(o.math=="exact")initial_gpu_batch=std::max(128,initial_gpu_batch/16);
    int adaptive_batch=gpu?std::min(o.batch,initial_gpu_batch):std::min(o.batch,std::max(32,o.threads*8));
    bool live_changed=false;Progress latest;
    auto phase=[&](const char* name){if(run_control&&run_control->phase)run_control->phase(name);};
    auto report=[&](double time) {
        Progress p{time,o.seconds,candidates,ticks,archive.empty()?Result{}:archive.front(),!archive.empty()&&std::isfinite(score(archive.front(),o))};
        p.generation=generation;p.improvements=accepted;p.restarts=restarts;p.since_improvement=candidates-last_improvement_candidate;
        p.last_improvement_seconds=last_improvement_time;p.agents=o.search.agents;
        std::unordered_set<Layout,MaskHash> parents;for(const auto& agent:agents)for(const auto& r:agent.elites)parents.insert(r.mask);p.distinct_parents=int(parents.size());
        latest=p;
        if(run_control&&run_control->progress)run_control->progress(p);
        if(!o.quiet) {
            std::cout<<std::fixed<<std::setprecision(2)<<"Time "<<time<<" s | "<<backend<<" | "<<candidates<<" evaluations | "
                <<candidates/std::max(time,1e-9)<<" layouts/s | "<<ticks/std::max(time,1e-9)<<" reactor ticks/s | "
                <<o.search.agents<<" agents | "<<restarts<<" restarts | "<<time-last_improvement_time<<" s since improvement\n";
            if(p.feasible)std::cout<<"Search best: "<<p.best.power<<" FE/t, "<<std::setprecision(6)<<p.best.fuel<<" mB/t, "<<p.best.efficiency<<" FE/mB, "<<p.best.rods<<" rods\n";
        }
    };
    stage("Searching");
    while(!finishing()&&(o.evaluations?candidates<o.evaluations:Clock::now()<deadline)) {
        if(run_control) {
            SearchTuning next=run_control->get_tuning(o.search);
            bool changed=next.agents!=o.search.agents||next.power_agents!=o.search.power_agents||next.elites!=o.search.elites||next.mutation_percent!=o.search.mutation_percent||next.max_flips!=o.search.max_flips||next.move_percent!=o.search.move_percent||next.crossover_percent!=o.search.crossover_percent||next.random_percent!=o.search.random_percent||next.migration_generations!=o.search.migration_generations||next.restart_generations!=o.search.restart_generations;
            if(changed) {
                live_changed=true;
                while(int(agents.size())<next.agents)agents.emplace_back(o.seed+1000003ULL*agents.size());
                if(int(agents.size())>next.agents)agents.erase(agents.begin()+next.agents,agents.end());
                o.search=next;
                std::cout<<"Applied live search settings at generation "<<generation<<": "<<next.agents<<" agents, "<<next.power_agents<<" power scouts, mutation "<<next.mutation_percent<<"%, max flips "<<next.max_flips<<", random "<<next.random_percent<<"%\n";
            }
            if(run_control->reseed.exchange(false)) {
                for(auto& agent:agents){agent.elites.clear();agent.stagnant=0;agent.fresh=true;++restarts;}
                std::cout<<"Reseeded search agents at generation "<<generation<<"; archived best layouts preserved.\n";
            }
        }
        int count=adaptive_batch;
        if(o.evaluations)count=int(std::min<unsigned long long>(count,o.evaluations-candidates));
        masks.resize(count);results.resize(count);
        // A small last batch rotates its active agents instead of always feeding agent 0.
        int active_agents=std::min(o.search.agents,count);
        int rotation=int(generation%unsigned(o.search.agents));
        auto agent_index=[&](int lane){return (rotation+lane)%o.search.agents;};
        auto began=Clock::now();phase("Generating candidates");
        pool.run(active_agents,[&](size_t lane) {
            auto& agent=agents[agent_index(int(lane))];auto& rng=agent.rng;
            int begin=int(lane)*count/active_agents,end=(int(lane)+1)*count/active_agents;
            for(int i=begin;i<end;++i) {
                int local=i-begin;
                if(agent.fresh&&!seeds.empty()&&local<int(seeds.size()))masks[i]=seeds[(local+agent_index(int(lane)))%seeds.size()];
                else if(agent.elites.empty()||rng()%100<unsigned(o.search.random_percent)) {
                    bool power_scout=agent_index(int(lane))<o.search.power_agents;
                    bool dense=power_scout&&rng()%4==0;
                    masks[i]=random_layout(rng,dense?o.max_rods:o.min_rods,o.max_rods,o.settings.cells(),power_scout);
                }
                else {
                    Layout parent=agent.elites[rng()%agent.elites.size()].mask;
                    if(rng()%100<unsigned(o.search.crossover_percent)) {
                        Layout other=agent.elites[rng()%agent.elites.size()].mask,select;for(auto& word:select.words)word=rng();
                        Layout child=(parent&select)|(other&~select);int n=rod_count(child);
                        if(n>=o.min_rods&&n<=o.max_rods)parent=child;
                    }
                    masks[i]=rng()%100<unsigned(o.search.mutation_percent)?mutate(parent,rng,o.min_rods,o.max_rods,o.settings.cells(),o.search):parent;
                }
            }
        });
        generation_time+=elapsed(began);began=Clock::now();phase(gpu?"Evaluating GPU batch":"Evaluating CPU batch");
        if(gpu)gpu->evaluate(masks,results,o.settings,o.sim);else evaluate_cpu(pool,masks,results,o);
        double batch_time=elapsed(began);evaluation_time+=batch_time;
        candidates+=count;for(const auto& r:results)ticks+=r.ticks;
        phase("Selecting parents");
        Result old=archive.empty()?Result{}:archive.front();
        for(int lane=0;lane<active_agents;++lane) {
            auto& agent=agents[agent_index(lane)];
            const auto& selection=agent_index(lane)<o.search.power_agents?power_selection:o;
            Result previous=agent.elites.empty()?Result{}:agent.elites.front();
            int begin=lane*count/active_agents,end=(lane+1)*count/active_agents;
            agent.elites.insert(agent.elites.end(),results.begin()+begin,results.begin()+end);
            agent.elites=retain(agent.elites,o.search.elites,selection);
            bool improved=!agent.elites.empty()&&(!previous.rods||score(agent.elites.front(),selection)>score(previous,selection)||(!std::isfinite(score(previous,selection))&&agent.elites.front().power>previous.power));
            agent.stagnant=improved?0:agent.stagnant+1;agent.fresh=false;
            archive.insert(archive.end(),agent.elites.begin(),agent.elites.end());
        }
        archive=retain(archive,128,o);
        ++generation;
        if(!archive.empty()&&(!old.rods||score(archive.front(),o)>score(old,o)||(!std::isfinite(score(old,o))&&archive.front().power>old.power))) {
            ++accepted;last_improvement_candidate=candidates;last_improvement_time=elapsed(start);
        }
        // Ring migration uses a snapshot, so one champion doesn't flood every agent in a single exchange.
        if(o.search.migration_generations&&generation%unsigned(o.search.migration_generations)==0&&agents.size()>1) {
            std::vector<Result> migrants;for(const auto& agent:agents)migrants.push_back(agent.elites.empty()?Result{}:agent.elites.front());
            for(size_t a=0;a<agents.size();++a)if(valid(migrants[a])) {
                size_t receiver_id=(a+1)%agents.size();auto& receiver=agents[receiver_id];receiver.elites.push_back(migrants[a]);receiver.elites=retain(receiver.elites,o.search.elites,receiver_id<size_t(o.search.power_agents)?power_selection:o);
            }
        }
        // Record best results in the global archive before clearing a stagnant population.
        for(int lane=0;lane<active_agents;++lane) {
            auto& agent=agents[agent_index(lane)];
            if(o.search.restart_generations&&agent.stagnant>=unsigned(o.search.restart_generations)) {
                agent.elites.clear();agent.stagnant=0;agent.fresh=true;++restarts;
            }
        }
        // Fixed budgets keep their batch grouping reproducible regardless of GPU load.
        if(gpu&&!o.evaluations&&batch_time>0.4&&adaptive_batch>128)adaptive_batch=std::max(128,int(adaptive_batch*0.3/batch_time));
        if(!o.evaluations) {
            double remaining=std::chrono::duration<double>(deadline-Clock::now()).count();
            if(remaining>0&&batch_time>0)adaptive_batch=std::max(1,std::min(adaptive_batch,int(count*remaining/batch_time)));
        }
        if(Clock::now()>=next_progress) {report(elapsed(start));next_progress=Clock::now()+std::chrono::milliseconds(o.progress_ms);}
    }
    double runtime=elapsed(start);report(runtime);stage("Verifying finalists");phase("CPU final verification");
    std::cout<<"Search completed: "<<candidates<<" evaluations in "<<runtime<<" s; "<<candidates/std::max(runtime,1e-9)<<" layouts/s; "
        <<ticks/std::max(runtime,1e-9)<<" reactor ticks/s\nTiming: generation "<<generation_time<<" s, simulation/transfers "<<evaluation_time<<" s\n";
    masks.clear();for(const auto& r:archive)masks.push_back(r.mask);results.resize(masks.size());
    evaluate_cpu(pool,masks,results,o,true);
    std::sort(results.begin(),results.end(),[&](const auto& a,const auto& b){return better(a,b,o);});
    if(results.empty()||!std::isfinite(score(results.front(),o))) {
        if(!archive.empty())std::cout<<"Best search estimate: "<<archive.front().power<<" FE/t; requested minimum: "<<o.min_power<<" FE/t.\n";
        std::cout<<"No exact-verified layout meets the requested power floor and rod limits. No result was saved.\n";return 3;
    }
    std::cout<<"\nBEST LAYOUT (exact CPU verified)\n";print_result(results.front(),o);
    save(results.front(),o,backend,device,runtime,candidates,ticks,true,&latest,live_changed);
    if(run_control&&run_control->verified)run_control->verified(results.front());
    std::cout<<"Saved "<<o.output<<".txt and "<<o.output<<".json\n";return 0;
}
}
int er2::optimizer_main(int argc,char** argv,RunControl* control) {
    run_control=control;
    struct LogBuffer:std::streambuf {
        RunControl* control; std::string line;
        explicit LogBuffer(RunControl* c):control(c){}
        int overflow(int c) override {if(c==traits_type::eof())return traits_type::not_eof(c);if(c=='\n'){if(control->log)control->log(line);line.clear();}else line+=char(c);return c;}
        int sync() override {if(!line.empty()&&control->log)control->log(line);line.clear();return 0;}
    } buffer(control);
    struct Restore {std::streambuf *out=nullptr,*err=nullptr;~Restore(){if(out){std::cout.flush();std::cerr.flush();std::cout.rdbuf(out);std::cerr.rdbuf(err);}run_control=nullptr;}} restore;
    if(control){restore.out=std::cout.rdbuf(&buffer);restore.err=std::cerr.rdbuf(&buffer);}
    try {
        stage("Preparing engine");
        auto o=parse(argc,argv);
        if(o.list_moderators){for(const auto& m:moderators)std::cout<<m.key<<" | "<<m.name<<(m.placeable?"":" (flowing state)")<<'\n';return 0;}
        verify_baseline();
        Pool pool(o.threads);
        std::unique_ptr<er2::Gpu> gpu;
        if(o.backend!="cpu") {
            try {stage("Preparing GPU");gpu=std::make_unique<er2::Gpu>(o.math=="exact",o.device,o.batch,o.settings.cells());}
            catch(const std::exception& error) {if(o.backend=="cuda")throw;std::cerr<<"GPU unavailable: "<<error.what()<<"\nUsing CPU backend.\n";}
        }
        std::cout<<"ATM10 ER2 optimizer | "<<(gpu?gpu->name():"CPU")<<" | "<<o.threads<<" CPU workers | "<<o.math<<" search math\n";
        if(o.self_test||o.benchmark)validate(gpu.get(),pool,o);
        if(o.self_test) {std::cout<<"PASS: topology, baseline, finite results, numerical error, and ranking\n";return 0;}
        if(o.benchmark) {stage("Benchmarking");benchmark(gpu.get(),pool,o);return o.benchmark_only?0:search(gpu.get(),pool,o);}
        if(o.calibrate||o.has_mask) {
            er2::Layout mask=o.has_mask?o.mask:checker(o.settings);
            er2::SimConfig config=o.has_mask?o.sim:er2::SimConfig{std::max(o.final_ticks,12000),6000,std::max(o.sim.sample_ticks,1000)};
            auto result=er2::evaluate_exact(mask,o.settings,config,!o.fixed);
            std::cout<<"EXACT CPU REFERENCE\n";print_result(result,o);
            if(gpu) {
                er2::Result accelerated;gpu->evaluate(std::span(&mask,1),std::span(&accelerated,1),o.settings,config);
                std::cout<<"CUDA (fixed ticks)\n";print_result(accelerated,o);
            }
            save(result,o,"cpu","CPU",0,1,result.ticks,true);return 0;
        }
        if(gpu||o.math=="fast") {
            // Always check the accelerated baseline before trusting search results.
            er2::Layout mask=checker(o.settings);er2::Result result;
            auto reference=er2::evaluate_fixed<double>(mask,o.settings,o.sim);
            if(gpu)gpu->evaluate(std::span(&mask,1),std::span(&result,1),o.settings,o.sim);
            else result=er2::evaluate_fixed<float>(mask,o.settings,o.sim);
            if(reference.fuel>0&&(std::abs(result.power/reference.power-1)>0.02||std::abs(result.fuel/reference.fuel-1)>0.02))
                throw std::runtime_error("Accelerated checkerboard differs by more than 2%");
        }
        return search(gpu.get(),pool,o);
    } catch(const std::exception& error) {std::cerr<<"Error: "<<error.what()<<'\n';return 1;}
}

namespace er2 {
std::string mask_hex(Layout mask) {
    std::ostringstream out;out<<"0x";int last=15;while(last>0&&!mask.words[last])--last;
    out<<std::hex<<mask.words[last];for(int i=last-1;i>=0;--i)out<<std::setw(16)<<std::setfill('0')<<mask.words[i];return out.str();
}
std::string layout_text(Layout mask,Settings settings) {
    std::string s;bool unob=settings.moderator.absorption==0.972&&settings.moderator.conductivity==5;
    for(int cell=0;cell<settings.cells();++cell){s+=mask.test(cell)?'R':(unob?'U':'M');s+=cell%settings.width==settings.width-1?'\n':' ';}return s;
}
}
#ifndef ER2_NO_MAIN
int main(int argc,char** argv) { return er2::optimizer_main(argc,argv); }
#endif
