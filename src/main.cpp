#include "simulator.hpp"
#include "gpu.hpp"
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
#include <vector>

namespace {
using namespace er2;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
struct Options {
    Settings settings;
    SimConfig sim;
    int threads=std::max(1U,std::thread::hardware_concurrency()), min_rods=1,max_rods=35;
    int batch=65536,device=0,final_ticks=20000,progress_ms=1000,validation_layouts=128;
    double seconds=60,min_power=0;
    unsigned long long seed=1337,evaluations=0;
    std::string backend="auto",math="fast",objective="efficiency",output="best_reactor";
    bool calibrate=false,benchmark=false,self_test=false,quiet=false,fixed=false,has_mask=false;
    Layout mask=0;
};
void help() {
    std::cout<<"ATM10 ER2 C++20 / CUDA reactor optimizer\n"
        "  --backend auto|cuda|cpu   auto prefers the NVIDIA GPU\n"
        "  --seconds 60             search wall-clock budget (startup/final verification extra)\n"
        "  --threads N              CPU evaluation/verification workers\n"
        "  --min-power 350000       exact verified minimum FE/t\n"
        "  --min-rods 1 --max-rods 35\n"
        "  --insertion 0            percent, 0..100\n"
        "  --fill 1 --variant-efficiency 1\n"
        "  --math fast|exact        float discovery or double calculations\n"
        "  --batch 65536 --device 0 GPU batch size and device index\n"
        "  --search-max-ticks 4500 --search-min-ticks 1500\n"
        "  --sample-ticks 500 --final-ticks 20000\n"
        "  --objective efficiency|power --seed 1337\n"
        "  --evaluations N          fixed candidate budget, overrides seconds (deterministic)\n"
        "  --progress-ms 1000 --output best_reactor --quiet\n"
        "  --calibrate --benchmark --self-test --validation-layouts 128\n"
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
        else if(key=="--seconds")o.seconds=real();
        else if(key=="--min-rods")o.min_rods=integer();
        else if(key=="--max-rods")o.max_rods=integer();
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
        else if(key=="--self-test")o.self_test=true;
        else if(key=="--quiet")o.quiet=true;
        else if(key=="--fixed-ticks")o.fixed=true;
        else if(key=="--evaluate") {o.mask=wide();o.has_mask=true;}
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(o.threads<1||o.threads>1024)throw std::runtime_error("threads must be 1..1024");
    if(o.seconds<=0||o.min_power<0)throw std::runtime_error("seconds must be positive; min-power must be nonnegative");
    if(o.min_rods<1||o.max_rods>49||o.min_rods>o.max_rods)throw std::runtime_error("rod range must be within 1..49");
    if(o.settings.insertion<0||o.settings.insertion>1||o.settings.fill<=0||o.settings.fill>1||o.settings.variant<=0)throw std::runtime_error("invalid insertion, fill, or variant efficiency");
    if(o.sim.min_ticks<1||o.sim.max_ticks<o.sim.min_ticks||o.sim.sample_ticks<1||o.sim.sample_ticks>8192||o.sim.sample_ticks>o.sim.max_ticks||o.final_ticks<o.sim.sample_ticks)throw std::runtime_error("invalid simulation tick limits (sample-ticks 1..8192)");
    if(o.batch<1||o.batch>131072||o.progress_ms<1||o.validation_layouts<4||o.validation_layouts>4096)throw std::runtime_error("invalid batch, progress, or validation count");
    if(o.backend!="auto"&&o.backend!="cpu"&&o.backend!="cuda")throw std::runtime_error("backend must be auto, cpu, or cuda");
    if(o.math!="exact"&&o.math!="fast")throw std::runtime_error("math must be exact or fast");
    if(o.objective!="efficiency"&&o.objective!="power")throw std::runtime_error("objective must be efficiency or power");
    if(o.has_mask&&(o.mask==0||(o.mask&~all_mask)))throw std::runtime_error("mask must have 1..49 valid cells");
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
Layout lattice(int spacing) {Layout mask=0;for(int z=0;z<7;z+=spacing)for(int x=0;x<7;x+=spacing)mask|=1ULL<<(z*7+x);return mask;}
Layout cross() {Layout m=0;for(int cell=0;cell<49;++cell)if(cell%7==3||cell/7==3)m|=1ULL<<cell;return m;}
Layout random_layout(std::mt19937_64& rng,int lo,int hi) {
    int n=lo+int(rng()%unsigned(hi-lo+1));n=std::min(n,lo+int(rng()%unsigned(hi-lo+1)));
    Layout mask=0;while(std::popcount(mask)<n)mask|=1ULL<<(rng()%49);return mask;
}
Layout mutate(Layout mask,std::mt19937_64& rng,int lo,int hi) {
    if(rng()%100<45 && mask && mask!=all_mask) {
        unsigned on,off;do{on=unsigned(rng()%49);}while(!(mask&(1ULL<<on)));
        do{off=unsigned(rng()%49);}while(mask&(1ULL<<off));
        return mask^(1ULL<<on)^(1ULL<<off);
    }
    int flips=1+int(rng()%3);
    for(int i=0;i<flips;++i) {Layout trial=mask^(1ULL<<(rng()%49));int n=std::popcount(trial);if(n>=lo&&n<=hi)mask=trial;}
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
std::string layout(Layout mask) {
    std::string s;for(int cell=0;cell<49;++cell){s+=(mask&(1ULL<<cell))?'R':'U';s+=(cell%7==6)?'\n':' ';}return s;
}
std::string quote(const std::string& value) {
    std::string out="\"";for(unsigned char c:value) {if(c=='\"'||c=='\\'){out+='\\';out+=char(c);}else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c<32){char buf[7];std::snprintf(buf,7,"\\u%04x",c);out+=buf;}else out+=char(c);}return out+'\"';
}
void print_result(const Result& r) {
    std::cout<<layout(r.mask)<<std::fixed<<std::setprecision(3)
        <<"Power: "<<r.power<<" FE/t\nFuel: "<<std::setprecision(9)<<r.fuel
        <<" mB/t\nEfficiency: "<<std::setprecision(3)<<r.efficiency<<" FE/mB\nRod columns: "<<r.rods
        <<"\nFuel heat: "<<r.fuel_heat<<" C\nReactor heat: "<<r.reactor_heat<<" C\nFertility: "<<r.fertility
        <<"\nMask: 0x"<<std::hex<<r.mask<<std::dec<<"\nTicks: "<<r.ticks<<'\n';
}
void save(const Result& r,const Options& o,const std::string& backend,const std::string& device,
          double runtime,unsigned long long candidates,unsigned long long ticks,bool verified) {
    std::filesystem::path prefix=o.output;
    if(!prefix.parent_path().empty())std::filesystem::create_directories(prefix.parent_path());
    std::ofstream json(o.output+".json"),txt(o.output+".txt");
    if(!json||!txt)throw std::runtime_error("Cannot write result output");
    json<<std::setprecision(17)<<"{\n  \"settings\": {\n"
        <<"    \"threads\": "<<o.threads<<", \"seconds\": "<<o.seconds<<", \"seed\": "<<o.seed
        <<", \"evaluation_budget\": "<<o.evaluations<<",\n    \"min_rods\": "<<o.min_rods<<", \"max_rods\": "<<o.max_rods
        <<", \"min_power\": "<<o.min_power<<",\n    \"insertion_percent\": "<<o.settings.insertion*100
        <<", \"fuel_fill\": "<<o.settings.fill<<", \"variant_efficiency\": "<<o.settings.variant
        <<",\n    \"search_max_ticks\": "<<o.sim.max_ticks<<", \"search_min_ticks\": "<<o.sim.min_ticks
        <<", \"sample_ticks\": "<<o.sim.sample_ticks<<", \"final_ticks\": "<<o.final_ticks
        <<", \"final_min_ticks\": "<<std::min(8000,std::max(2000,o.final_ticks/3))
        <<", \"final_sample_ticks\": "<<std::max(o.sim.sample_ticks,std::min(1000,o.final_ticks))
        <<",\n    \"backend_requested\": "<<quote(o.backend)<<", \"backend\": "<<quote(backend)
        <<", \"math\": "<<quote(o.math)<<", \"objective\": "<<quote(o.objective)<<", \"fixed_ticks\": "<<(o.fixed?"true":"false")
        <<",\n    \"batch\": "<<o.batch<<", \"device_index\": "<<o.device<<", \"device_name\": "<<quote(device)
        <<",\n    \"geometry\": [7,7,7], \"atm10_fuel_multiplier\": 0.8, \"atm10_power_multiplier\": 12,\n"
        <<"    \"unobtainium\": {\"absorption\": 0.972, \"heat_efficiency\": 0.91, \"moderation\": 3.074, \"conductivity\": 5.0},\n"
        <<"    \"source_schedule\": \"one event per tick, columns cycled as in ato.py\"\n  },\n"
        <<"  \"run\": {\"search_seconds\": "<<runtime<<", \"candidate_evaluations\": "<<candidates
        <<", \"reactor_ticks\": "<<ticks<<", \"layouts_per_second\": "<<(runtime>0?candidates/runtime:0)
        <<", \"exact_cpu_verified\": "<<(verified?"true":"false")<<"},\n  \"result\": {\n"
        <<"    \"mask\": "<<r.mask<<", \"layout\": "<<quote(layout(r.mask))<<", \"rod_columns\": "<<r.rods
        <<", \"rod_blocks\": "<<r.rods*7<<",\n    \"score\": "<<(o.objective=="power"?r.power:r.efficiency)
        <<", \"power_fe_t\": "<<r.power<<", \"fuel_mb_t\": "<<r.fuel
        <<", \"efficiency_fe_per_mb\": "<<r.efficiency<<",\n    \"fuel_heat_c\": "<<r.fuel_heat
        <<", \"reactor_heat_c\": "<<r.reactor_heat<<", \"fertility\": "<<r.fertility<<", \"ticks\": "<<r.ticks<<"\n  }\n}\n";
    txt<<"R = fuel column; U = Unobtainium; each column is 7 blocks tall\n\n"<<layout(r.mask)
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
    std::vector<Layout> masks(n);masks[0]=checkerboard();masks[1]=lattice(3);masks[2]=lattice(2);masks[3]=all_mask;
    for(int i=4;i<n;++i)masks[i]=random_layout(rng,1,49);
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
    std::vector<Layout> masks(n);for(auto& m:masks)m=random_layout(rng,o.min_rods,o.max_rods);
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
    std::fill(masks.begin(),masks.end(),checkerboard());
    run("Checkerboard CPU double, all requested workers",[&]{pool.run(n,[&](size_t i){results[i]=evaluate_exact(masks[i],o.settings,o.sim);});});
}
int search(Gpu* gpu,Pool& pool,const Options& o) {
    const std::string backend=gpu?"cuda":"cpu",device=gpu?gpu->name():"CPU";
    std::vector<std::mt19937_64> generators;
    for(int i=0;i<o.threads;++i)generators.emplace_back(o.seed+1000003ULL*i);
    std::vector<Result> elites;
    std::vector<Layout> masks;std::vector<Result> results;
    std::vector<Layout> seeds={checkerboard(),lattice(3),lattice(2),cross()};
    seeds.erase(std::remove_if(seeds.begin(),seeds.end(),[&](Layout m){int n=std::popcount(m);return n<o.min_rods||n>o.max_rods;}),seeds.end());
    unsigned long long candidates=0,ticks=0,accepted=0;
    double evaluation_time=0,generation_time=0;
    auto start=Clock::now(),next_progress=start;
    auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.seconds));
    bool first=true;
    int adaptive_batch=gpu?o.batch:std::min(o.batch,std::max(32,o.threads*8));
    while(o.evaluations?candidates<o.evaluations:Clock::now()<deadline) {
        int count=adaptive_batch;
        if(o.evaluations)count=int(std::min<unsigned long long>(count,o.evaluations-candidates));
        masks.resize(count);results.resize(count);
        auto phase=Clock::now();
        // Fixed chunks have independent RNG streams: parallel generation remains
        // deterministic for a fixed budget and settings, regardless of scheduling.
        pool.run(std::min(o.threads,count),[&](size_t island) {
            auto& rng=generators[island];
            int islands=std::min(o.threads,count);
            int begin=int(island)*count/islands,end=(int(island)+1)*count/islands;
            for(int i=begin;i<end;++i) {
                if(first&&i<int(seeds.size()))masks[i]=seeds[i];
                else if(elites.empty()||rng()%100<15)masks[i]=random_layout(rng,o.min_rods,o.max_rods);
                else {
                    Layout parent=elites[rng()%elites.size()].mask;
                    if(rng()%100<20) {
                        Layout other=elites[rng()%elites.size()].mask,select=rng()&all_mask;
                        Layout child=(parent&select)|(other&~select);int n=std::popcount(child);
                        if(n>=o.min_rods&&n<=o.max_rods)parent=child;
                    }
                    masks[i]=mutate(parent,rng,o.min_rods,o.max_rods);
                }
            }
        });
        generation_time+=elapsed(phase);phase=Clock::now();
        if(gpu)gpu->evaluate(masks,results,o.settings,o.sim);else evaluate_cpu(pool,masks,results,o);
        double batch_time=elapsed(phase);evaluation_time+=batch_time;
        candidates+=count;for(auto& r:results)ticks+=r.ticks;
        Result old=elites.empty()?Result{}:elites.front();
        elites.insert(elites.end(),results.begin(),results.end());
        // Select a bounded shortlist before sorting; expand only if duplicates
        // leave fewer than 128 distinct parents. All candidates remain eligible.
        std::vector<Result> retained;retained.reserve(128);
        size_t shortlist=std::min<size_t>(4096,elites.size());
        for(;;) {
            auto compare=[&](auto& a,auto& b){return better(a,b,o);};
            if(shortlist<elites.size())std::nth_element(elites.begin(),elites.begin()+shortlist,elites.end(),compare);
            std::sort(elites.begin(),elites.begin()+shortlist,compare);
            retained.clear();
            for(size_t i=0;i<shortlist;++i) {
                auto& r=elites[i];
                if(!valid(r))continue;
                if(std::none_of(retained.begin(),retained.end(),[&](auto& q){return q.mask==r.mask;}))retained.push_back(r);
                if(retained.size()==128)break;
            }
            if(retained.size()==128||shortlist==elites.size())break;
            shortlist=std::min(elites.size(),shortlist*4);
        }
        elites=std::move(retained);
        if(!elites.empty()&&score(elites.front(),o)>score(old,o))++accepted;
        first=false;
        // Keep subsequent launches brief for Windows' display-driver watchdog and progress.
        if(gpu&&!o.evaluations&&batch_time>0.4&&adaptive_batch>128)adaptive_batch=std::max(128,int(adaptive_batch*0.3/batch_time));
        if(!o.evaluations) {
            double remaining=std::chrono::duration<double>(deadline-Clock::now()).count();
            if(remaining>0&&batch_time>0)adaptive_batch=std::max(1,std::min(adaptive_batch,int(count*remaining/batch_time)));
        }
        if(!o.quiet&&Clock::now()>=next_progress) {
            double time=elapsed(start);
            std::cout<<std::fixed<<std::setprecision(2)<<"Time "<<time<<" s | "<<backend<<" | "<<candidates<<" evaluations | "
                <<candidates/time<<" layouts/s | "<<ticks/time<<" reactor ticks/s | "<<accepted/time<<" champion improvements/s\n";
            if(!elites.empty()&&std::isfinite(score(elites.front(),o)))std::cout<<"Search best: "<<elites.front().power<<" FE/t, "
                <<std::setprecision(6)<<elites.front().fuel<<" mB/t, "<<elites.front().efficiency<<" FE/mB, "<<elites.front().rods<<" rods\n";
            next_progress=Clock::now()+std::chrono::milliseconds(o.progress_ms);
        }
    }
    double runtime=elapsed(start);
    std::cout<<"Search completed: "<<candidates<<" evaluations in "<<runtime<<" s; "<<candidates/runtime<<" layouts/s; "
        <<ticks/runtime<<" reactor ticks/s\nTiming: generation "<<generation_time<<" s, simulation/transfers "<<evaluation_time<<" s\n";
    masks.clear();for(auto& r:elites)masks.push_back(r.mask);results.resize(masks.size());
    evaluate_cpu(pool,masks,results,o,true);
    std::sort(results.begin(),results.end(),[&](auto& a,auto& b){return better(a,b,o);});
    if(results.empty()||!std::isfinite(score(results.front(),o))) {
        std::cout<<"No exact-verified layout meets the requested power floor and rod limits. No result was saved.\n";return 3;
    }
    std::cout<<"\nBEST LAYOUT (exact CPU verified)\n";print_result(results.front());
    save(results.front(),o,backend,device,runtime,candidates,ticks,true);
    std::cout<<"Saved "<<o.output<<".txt and "<<o.output<<".json\n";
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        auto o=parse(argc,argv);
        verify_baseline();
        Pool pool(o.threads);
        std::unique_ptr<er2::Gpu> gpu;
        if(o.backend!="cpu") {
            try {gpu=std::make_unique<er2::Gpu>(o.math=="exact",o.device,o.batch);}
            catch(const std::exception& error) {if(o.backend=="cuda")throw;std::cerr<<"GPU unavailable: "<<error.what()<<"\nUsing CPU backend.\n";}
        }
        std::cout<<"ATM10 ER2 optimizer | "<<(gpu?gpu->name():"CPU")<<" | "<<o.threads<<" CPU workers | "<<o.math<<" search math\n";
        if(o.self_test||o.benchmark)validate(gpu.get(),pool,o);
        if(o.self_test) {std::cout<<"PASS: topology, baseline, finite results, numerical error, and ranking\n";return 0;}
        if(o.benchmark) {benchmark(gpu.get(),pool,o);return search(gpu.get(),pool,o);}
        if(o.calibrate||o.has_mask) {
            er2::Layout mask=o.has_mask?o.mask:er2::checkerboard();
            er2::SimConfig config=o.has_mask?o.sim:er2::SimConfig{std::max(o.final_ticks,12000),6000,std::max(o.sim.sample_ticks,1000)};
            auto result=er2::evaluate_exact(mask,o.settings,config,!o.fixed);
            std::cout<<"EXACT CPU REFERENCE\n";print_result(result);
            if(gpu) {
                er2::Result accelerated;gpu->evaluate(std::span(&mask,1),std::span(&accelerated,1),o.settings,config);
                std::cout<<"CUDA (fixed ticks)\n";print_result(accelerated);
            }
            save(result,o,"cpu","CPU",0,1,result.ticks,true);return 0;
        }
        if(gpu||o.math=="fast") {
            // Always check the accelerated baseline before trusting search results.
            er2::Layout mask=er2::checkerboard();er2::Result result;
            auto reference=er2::evaluate_fixed<double>(mask,o.settings,o.sim);
            if(gpu)gpu->evaluate(std::span(&mask,1),std::span(&result,1),o.settings,o.sim);
            else result=er2::evaluate_fixed<float>(mask,o.settings,o.sim);
            if(reference.fuel>0&&(std::abs(result.power/reference.power-1)>0.02||std::abs(result.fuel/reference.fuel-1)>0.02))
                throw std::runtime_error("Accelerated checkerboard differs by more than 2%");
        }
        return search(gpu.get(),pool,o);
    } catch(const std::exception& error) {std::cerr<<"Error: "<<error.what()<<'\n';return 1;}
}
