#include "engine.hpp"
#include <filesystem>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2)return 1;
    er2::RunControl control;
    bool verified=false;unsigned long long count=0;
    control.progress=[&](const er2::Progress& p){count=p.candidates;if(count)control.finish=true;};
    control.verified=[&](const er2::Result& r){verified=r.rods>0&&r.rods<=15&&r.power>0;};
    std::vector<std::string> args={"test","--backend","cpu","--threads","4","--seconds","60","--width","3","--depth","5","--height","4","--max-rods","15","--moderator","water","--output",argv[1]};
    std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());
    int code=er2::optimizer_main(int(pointers.size()),pointers.data(),&control);
    if(code||!verified||count==0||count>32||!std::filesystem::exists(std::string(argv[1])+".json"))return 2;
    std::cout<<"PASS: early finish verifies and saves the available winner\n";return 0;
}
