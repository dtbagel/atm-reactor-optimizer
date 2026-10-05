#include "engine.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=2)return 1;
    er2::RunControl control;
    int batches=0;bool verified=false,monotonic=true;
    double last_power=0;er2::Progress final;
    std::string logs;
    control.log=[&](const std::string& line){logs+=line+'\n';};
    control.phase=[&](const std::string& phase){
        if(phase!="Selecting parents")return;
        ++batches;
        if(batches==1){er2::SearchTuning t;t.agents=3;t.power_agents=1;t.max_flips=7;t.random_percent=0;t.restart_generations=1;control.set_tuning(t);control.reseed=true;}
        if(batches==4){er2::SearchTuning t;t.agents=1;t.power_agents=0;t.mutation_percent=0;t.crossover_percent=0;t.random_percent=0;t.restart_generations=1;control.set_tuning(t);}
    };
    control.progress=[&](const er2::Progress& p){if(p.best.power<last_power)monotonic=false;last_power=p.best.power;final=p;};
    control.verified=[&](const er2::Result& r){verified=r.rods>0&&r.rods<=15&&r.power>0;};
    std::vector<std::string> args={"test","--backend","cpu","--threads","2","--width","3","--depth","5","--height","4","--moderator","water","--agents","2","--power-agents","0","--objective","power","--evaluations","144","--batch","6","--progress-ms","1","--output",argv[1]};
    std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());
    int code=er2::optimizer_main(int(pointers.size()),pointers.data(),&control);
    std::ifstream input(std::string(argv[1])+".json");std::ostringstream contents;contents<<input.rdbuf();auto saved=contents.str();
    if(code||!verified||!monotonic||final.candidates!=144||final.generation!=24||final.agents!=1||final.restarts<3||
       logs.find("Applied live search settings at generation 1: 3 agents")==std::string::npos||
       logs.find("Applied live search settings at generation 4: 1 agents")==std::string::npos||
       logs.find("Reseeded search agents")==std::string::npos||
       saved.find("\"max_rods\": 15")==std::string::npos||saved.find("\"live_tuning_changed\": true")==std::string::npos)return 2;
    std::cout<<"PASS: live tuning, agent resize, reseeding, automatic restarts, preserved archive, exact budget, and full footprint\n";
    return 0;
}
