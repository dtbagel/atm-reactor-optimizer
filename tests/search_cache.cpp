#include "engine.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Run {
    int code=0;
    er2::Progress progress;
    er2::Result verified;
    std::string saved;
};
Run search(const std::string& prefix,int cache_mb,bool dedup,int depth=2,bool exact=false,int minimum_rods=0,int width=2,bool reseed_each_batch=false) {
    Run result;er2::RunControl control;
    control.log=[](const std::string&){};
    control.progress=[&](const er2::Progress& p){result.progress=p;};
    control.verified=[&](const er2::Result& r){result.verified=r;};
    if(reseed_each_batch)control.phase=[&](const std::string& phase){if(phase=="Selecting parents")control.reseed=true;};
    std::vector<std::string> args={"test","--backend","cpu","--threads","2","--width",std::to_string(width),"--depth",std::to_string(depth),"--height","4","--moderator","water","--agents","4","--power-agents","1","--objective","power","--evaluations","256","--batch","32","--search-max-ticks","64","--search-min-ticks","32","--sample-ticks","16","--final-ticks","128","--min-rods",std::to_string(minimum_rods?minimum_rods:width*depth),"--cache-mb",std::to_string(cache_mb),"--mutation-percent",minimum_rods?"90":"0","--crossover-percent",minimum_rods?"20":"0","--random-percent",minimum_rods?"20":"0","--restart-generations","0","--math",exact?"exact":"fast","--quiet","--output",prefix};
    if(!dedup)args.push_back("--no-dedup");
    std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());
    result.code=er2::optimizer_main(int(pointers.size()),pointers.data(),&control);
    std::ifstream input(prefix+".json");std::ostringstream contents;contents<<input.rdbuf();result.saved=contents.str();
    return result;
}
bool equal(const er2::Result& a,const er2::Result& b) {
    return a.mask==b.mask&&a.power==b.power&&a.fuel==b.fuel&&a.efficiency==b.efficiency&&a.fuel_heat==b.fuel_heat&&a.reactor_heat==b.reactor_heat&&a.fertility==b.fertility&&a.rods==b.rods&&a.ticks==b.ticks;
}
int main(int argc,char** argv) {
    if(argc!=2)return 1;
    std::string prefix=argv[1];
    auto plain=search(prefix+"_plain",0,false);
    auto batch=search(prefix+"_batch",0,true);
    auto cached=search(prefix+"_cached",1,true);
    if(plain.code||batch.code||cached.code||plain.progress.candidates!=256||batch.progress.candidates!=256||cached.progress.candidates!=256||
       plain.progress.simulation_evaluations!=256||plain.progress.cache_hits!=0||plain.progress.ticks!=256*64||
       batch.progress.simulation_evaluations!=8||batch.progress.cache_hits!=248||batch.progress.batch_duplicates!=248||batch.progress.ticks!=8*64||
       cached.progress.simulation_evaluations!=1||cached.progress.cache_hits!=255||cached.progress.batch_duplicates!=31||cached.progress.ticks!=64||
       !equal(plain.verified,batch.verified)||!equal(plain.verified,cached.verified)||!equal(plain.progress.best,cached.progress.best)||
       cached.saved.find("\"candidate_evaluations\": 256")==std::string::npos||cached.saved.find("\"simulation_evaluations\": 1")==std::string::npos||
       cached.saved.find("\"cache_hits\": 255")==std::string::npos)return 2;
    // Separate invocations must not reuse results across geometry or discovery math.
    auto changed=search(prefix+"_changed",1,true,3,true);
    auto changed_plain=search(prefix+"_changed_plain",0,false,3,true);
    if(changed.code||changed_plain.code||changed.progress.simulation_evaluations!=1||changed.verified.rods!=6||
       !equal(changed.verified,changed_plain.verified)||!equal(changed.progress.best,changed_plain.progress.best))return 3;
    // Mixed layouts and independent populations exercise reconstruction order.
    auto mixed=search(prefix+"_mixed",1,true,3,true,1);
    auto mixed_plain=search(prefix+"_mixed_plain",0,false,3,true,1);
    if(mixed.code||mixed_plain.code||mixed.progress.candidates!=256||mixed.progress.simulation_evaluations<=1||mixed.progress.simulation_evaluations>63||
       mixed.progress.simulation_evaluations+mixed.progress.cache_hits!=256||mixed.progress.ticks!=mixed.progress.simulation_evaluations*64||
       !equal(mixed.verified,mixed_plain.verified)||!equal(mixed.progress.best,mixed_plain.progress.best))return 4;
    // Distinct candidates beyond bit 63 must compare every active key word.
    auto extended=search(prefix+"_extended",1,true,32,false,95,3);
    auto extended_plain=search(prefix+"_extended_plain",0,false,32,false,95,3);
    if(extended.code||extended_plain.code||extended.progress.simulation_evaluations<=1||extended.progress.simulation_evaluations>97||
       extended.progress.simulation_evaluations+extended.progress.cache_hits!=256||extended.progress.best.mask.words[1]==0||
       !equal(extended.verified,extended_plain.verified)||!equal(extended.progress.best,extended_plain.progress.best))return 5;
    // With 1 MiB / 5 active key words this shape gets 8192 cache slots.
    // Its checkerboard and spacing-2 seed both map to slot 2212. Repeated
    // reseeding forces those distinct exact keys to evict one another.
    auto collisions=search(prefix+"_collisions",1,true,16,false,1,18,true);
    auto collisions_plain=search(prefix+"_collisions_plain",0,false,16,false,1,18,true);
    if(collisions.code||collisions_plain.code||collisions.progress.simulation_evaluations>=256||collisions.progress.simulation_evaluations<16||
       collisions.progress.simulation_evaluations+collisions.progress.cache_hits!=256||collisions.progress.ticks!=collisions.progress.simulation_evaluations*64||
       !equal(collisions.verified,collisions_plain.verified)||!equal(collisions.progress.best,collisions_plain.progress.best))return 6;
    std::cout<<"PASS: exact batch dedup, same-run reuse, preserved proposal budgets/order, honest simulation/tick counters, and run isolation\n";
    return 0;
}
