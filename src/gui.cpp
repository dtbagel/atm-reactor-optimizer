#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dwmapi.h>
#include <shobjidl.h>
#include <shellapi.h>
#include <wincodec.h>
#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"
#include "engine.hpp"
#include "moderators.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cfloat>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND,UINT,WPARAM,LPARAM);
namespace {
using namespace er2;
ID3D11Device* device=nullptr;
ID3D11DeviceContext* context=nullptr;
IDXGISwapChain* swap_chain=nullptr;
ID3D11RenderTargetView* target=nullptr;
HWND window=nullptr;
UINT resize_width=0,resize_height=0;
ImFont *body=nullptr,*small=nullptr,*heading=nullptr,*big=nullptr,*bold=nullptr;
float scale=1;
constexpr ImU32 white=IM_COL32(236,243,255,255),muted=IM_COL32(142,165,198,255);
constexpr ImU32 cyan=IM_COL32(91,214,255,255),purple=IM_COL32(168,142,255,255);
template<class T> void release(T*& p) {if(p){p->Release();p=nullptr;}}
std::filesystem::path utfpath(const std::string& s) {return std::filesystem::path(std::u8string(s.begin(),s.end()));}
std::string utf8(const std::filesystem::path& p) {auto s=p.u8string();return {s.begin(),s.end()};}
std::filesystem::path exe_folder() {wchar_t p[32768]{};GetModuleFileNameW(nullptr,p,32768);return std::filesystem::path(p).parent_path();}
std::string compact(double n) {char s[64];if(n>=1e9)std::snprintf(s,sizeof(s),"%.2fB",n/1e9);else if(n>=1e6)std::snprintf(s,sizeof(s),"%.2fM",n/1e6);else if(n>=1e3)std::snprintf(s,sizeof(s),"%.1fk",n/1e3);else std::snprintf(s,sizeof(s),"%.0f",n);return s;}
double now_seconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
std::string counted(unsigned long long n){auto s=std::to_string(n);for(int i=int(s.size())-3;i>0;i-=3)s.insert(size_t(i),",");return s;}
struct Configuration {
    int backend=0,threads=std::max(1U,std::thread::hardware_concurrency()),seconds=60;
    int width=7,depth=7,height=7,minimum_rods=1,objective=0,math=0;
    SearchTuning search;
    int search_preset=0;
    int batch=65536,gpu_index=0,seed=1337;
    float minimum_power=0,insertion=0,fill=1;
    int moderator=0;
    char output[4096]{},name[128]="best_reactor",filter[128]{};
    Configuration() {for(size_t i=0;i<std::size(moderators);++i)if(std::string(moderators[i].key)=="unobtainium")moderator=int(i);auto path=utf8(exe_folder()/"results");std::snprintf(output,sizeof(output),"%s",path.c_str());}
    Settings settings() const {Settings s;s.width=width;s.depth=depth;s.height=height;s.moderator=moderators[moderator].material;s.insertion=insertion/100.;s.fill=fill;return s;}
};
struct Snapshot {
    std::string stage="Ready",error,device_name;
    std::string phase;
    double progress_at=0;
    Progress progress;
    Result result;
    bool verified=false,benchmark=false,done=false;
    int code=0;
    std::deque<std::string> log;
};
class Job {
    std::mutex mutex;
    Snapshot data;
    std::thread worker;
public:
    RunControl control;
    std::atomic<bool> busy{false};
    Configuration used;
    ~Job(){control.finish=true;if(worker.joinable())worker.join();}
    Snapshot snapshot(){std::lock_guard lock(mutex);return data;}
    void start(const Configuration& c,bool benchmark,unsigned long long test_budget=0) {
        if(busy)return;if(worker.joinable())worker.join();
        used=c;control.finish=false;control.reseed=false;control.set_tuning(c.search);{std::lock_guard lock(mutex);data={};data.stage="Preparing engine";data.benchmark=benchmark;}
        control.log=[this](const std::string& line){std::lock_guard lock(mutex);if(data.log.size()==400)data.log.pop_front();data.log.push_back(line);
            if(line.starts_with("Error:"))data.error=line.substr(7);
            if(line.starts_with("GPU unavailable:"))data.device_name="CPU fallback";
            if(line.starts_with("ATM10 ER2 optimizer |")){auto a=line.find('|')+2,b=line.find('|',a);data.device_name=line.substr(a,b-a-1);}
        };
        control.stage=[this](const std::string& s){std::lock_guard lock(mutex);data.stage=s;};
        control.phase=[this](const std::string& s){std::lock_guard lock(mutex);data.phase=s;};
        control.progress=[this](const Progress& p){std::lock_guard lock(mutex);data.progress=p;data.progress_at=now_seconds();};
        control.verified=[this](const Result& r){std::lock_guard lock(mutex);data.result=r;data.verified=true;};
        busy=true;
        worker=std::thread([this,c,benchmark,test_budget]{
            std::vector<std::string> args={"optimizer","--backend",std::array{"auto","cuda","cpu"}[c.backend],"--threads",std::to_string(c.threads),
                "--seconds",std::to_string(c.seconds),"--min-power",std::to_string(c.minimum_power),
                "--width",std::to_string(c.width),"--depth",std::to_string(c.depth),"--height",std::to_string(c.height),
                "--min-rods",std::to_string(c.minimum_rods),
                "--agents",std::to_string(c.search.agents),"--power-agents",std::to_string(c.search.power_agents),"--elites",std::to_string(c.search.elites),
                "--mutation-percent",std::to_string(c.search.mutation_percent),"--max-flips",std::to_string(c.search.max_flips),"--move-percent",std::to_string(c.search.move_percent),
                "--crossover-percent",std::to_string(c.search.crossover_percent),"--random-percent",std::to_string(c.search.random_percent),
                "--migration-generations",std::to_string(c.search.migration_generations),"--restart-generations",std::to_string(c.search.restart_generations),
                "--moderator",moderators[c.moderator].key,"--insertion",std::to_string(c.insertion),"--fill",std::to_string(c.fill),
                "--batch",std::to_string(c.batch),"--device",std::to_string(c.gpu_index),"--math",c.math?"exact":"fast",
                "--objective",c.objective?"power":"efficiency","--seed",std::to_string(c.seed),"--progress-ms","100",
                "--output",utf8(utfpath(c.output)/utfpath(c.name))};
            if(benchmark){args.push_back("--benchmark-only");args.push_back("--validation-layouts");args.push_back("32");}
            if(test_budget){args.push_back("--evaluations");args.push_back(std::to_string(test_budget));}
            std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
            int code=optimizer_main(int(argv.size()),argv.data(),&control);
            {std::lock_guard lock(mutex);data.code=code;data.done=true;
                data.stage=code==0?(benchmark?"Benchmark complete":control.finish?"Finished early":"Complete"):code==3?"No feasible layout":"Error";
                if(code==3)data.error="No verified layout met the minimum power. Try more exploration, a longer run, or a larger reactor. The best search estimate is shown for comparison.";
                if(code==1&&data.error.empty())data.error="The run could not finish. Open the run log for details.";
            }busy=false;
        });
    }
};
Configuration configuration;
Job job;
bool close_when_done=false;
unsigned long long developer_budget=0;
bool developer_advanced_preview=false;

void text(ImDrawList* draw,ImVec2 p,const char* s,ImFont* font=nullptr,ImU32 color=white) {font=font?font:body;draw->AddText(font,font->LegacySize,p,color,s);}
void glow(ImDrawList* d,ImVec2 center,float radius,ImU32 color) {
    ImVec4 c=ImGui::ColorConvertU32ToFloat4(color);
    for(int i=32;i>0;--i){float f=float(i)/32;c.w=0.023f*(1-f);d->AddCircleFilled(center,radius*f,ImGui::ColorConvertFloat4ToU32(c),64);}
}
void glass(ImDrawList* d,ImVec2 p,ImVec2 size,float radius=20) {
    d->AddRectFilled({p.x,p.y+8*scale},{p.x+size.x,p.y+size.y+8*scale},IM_COL32(0,4,16,60),radius*scale);
    d->AddRectFilled(p,{p.x+size.x,p.y+size.y},IM_COL32(21,34,57,172),radius*scale);
    d->AddRectFilledMultiColor({p.x+radius*scale,p.y+2*scale},{p.x+size.x-radius*scale,p.y+std::min(size.y*.3f,100*scale)},IM_COL32(188,215,255,14),IM_COL32(220,232,255,5),IM_COL32(220,232,255,0),IM_COL32(188,215,255,0));
    d->AddRect(p,{p.x+size.x,p.y+size.y},IM_COL32(159,192,236,55),radius*scale,0,scale);
    d->AddLine({p.x+radius*scale,p.y+scale},{p.x+size.x-radius*scale,p.y+scale},IM_COL32(209,233,255,65),scale);
    d->AddBezierCubic({p.x+radius*scale,p.y+scale},{p.x+size.x*.3f,p.y+scale},{p.x+size.x*.6f,p.y+scale},{p.x+size.x-radius*scale,p.y+scale},IM_COL32(173,213,255,30),3*scale);
}
void cube(ImDrawList* d,ImVec2 p,float r) {
    ImVec2 a{p.x,p.y-r},b{p.x+r,p.y-r/2},c{p.x+r,p.y+r/2},e{p.x,p.y+r},f{p.x-r,p.y+r/2},g{p.x-r,p.y-r/2};
    glow(d,p,r*2.5f,cyan);d->AddQuadFilled(a,b,p,g,IM_COL32(41,170,238,45));d->AddQuadFilled(p,b,c,e,IM_COL32(134,100,235,60));d->AddQuadFilled(g,p,e,f,IM_COL32(46,135,240,60));
    for(auto pair:std::array<std::pair<ImVec2,ImVec2>,9>{{{a,b},{b,c},{c,e},{e,f},{f,g},{g,a},{g,p},{p,b},{p,e}}})d->AddLine(pair.first,pair.second,cyan,1.5f*scale);
}
void field_at(float x,float y,float width) {ImGui::SetCursorPos({x*scale,y*scale});ImGui::SetNextItemWidth(width*scale);}
void label(ImDrawList* d,float x,float y,const char* s){text(d,{x*scale,(y+10)*scale},s,body,muted);}
bool action(const char* id,const char* label,ImVec2 size,bool primary=false) {
    ImVec2 p=ImGui::GetCursorScreenPos();ImGui::InvisibleButton(id,size);bool hovered=ImGui::IsItemHovered(),pressed=ImGui::IsItemClicked();
    auto* d=ImGui::GetWindowDrawList();float r=12*scale;
    if(primary){glow(d,{p.x+size.x/2,p.y+size.y/2},size.x*.55f,cyan);d->AddRectFilled(p,{p.x+size.x,p.y+size.y},IM_COL32(40,104,210,255),r);
        d->PushClipRect(p,{p.x+size.x,p.y+size.y},true);d->AddRectFilledMultiColor({p.x+size.x*.35f,p.y+2*scale},{p.x+size.x-10*scale,p.y+size.y-2*scale},IM_COL32(65,140,251,0),IM_COL32(148,82,241,180),IM_COL32(107,73,220,180),IM_COL32(48,135,240,0));d->PopClipRect();
    }else d->AddRectFilled(p,{p.x+size.x,p.y+size.y},hovered?IM_COL32(57,83,117,240):IM_COL32(38,56,82,210),r);
    d->AddRect(p,{p.x+size.x,p.y+size.y},primary?cyan:IM_COL32(152,185,224,90),r,0,scale);
    if(hovered)d->AddRectFilled(p,{p.x+size.x,p.y+size.y},IM_COL32(230,245,255,12),r);
    ImVec2 t=bold->CalcTextSizeA(bold->LegacySize,FLT_MAX,0,label);text(d,{p.x+(size.x-t.x)/2,p.y+(size.y-t.y)/2},label,bold);return pressed;
}
void choose_folder() {
    IFileDialog* dialog=nullptr;if(SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog)))){
        DWORD options;dialog->GetOptions(&options);dialog->SetOptions(options|FOS_PICKFOLDERS|FOS_FORCEFILESYSTEM);dialog->SetTitle(L"Choose where to save reactor layouts");
        if(SUCCEEDED(dialog->Show(window))){IShellItem* item=nullptr;if(SUCCEEDED(dialog->GetResult(&item))){PWSTR path=nullptr;if(SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH,&path))){auto s=utf8(path);std::snprintf(configuration.output,sizeof(configuration.output),"%s",s.c_str());CoTaskMemFree(path);}item->Release();}}dialog->Release();}
}
bool config_valid() {
    auto& c=configuration;return c.width>=1&&c.width<=32&&c.depth>=1&&c.depth<=32&&c.height>=1&&c.height<=64&&c.seconds>=1&&c.threads>=1&&c.threads<=1024&&c.minimum_power>=0&&std::isfinite(c.minimum_power)&&c.minimum_rods>=1&&c.minimum_rods<=c.width*c.depth&&valid_tuning(c.search)&&c.batch>=1&&c.batch<=131072&&c.gpu_index>=0&&c.insertion>=0&&c.insertion<=100&&c.fill>0&&c.fill<=1&&c.output[0]&&c.name[0]&&std::string(c.name).find_first_of("<>:\"/\\|?*")==std::string::npos;
}
void advanced_dialog() {
    ImGui::SetNextWindowSize({620*scale,680*scale},ImGuiCond_Appearing);
    if(ImGui::BeginPopupModal("Advanced settings",nullptr,ImGuiWindowFlags_NoResize)) {
        auto& c=configuration;bool changed=false;
        ImGui::BeginChild("settings_scroll",{0,-58*scale});
        if(ImGui::BeginTabBar("settings_tabs")) {
            if(ImGui::BeginTabItem("Search agents & mutations")) {
                ImGui::TextWrapped("Separate populations explore different layouts. Power scouts seek higher output; the other agents pursue your selected objective.");
                ImGui::Spacing();ImGui::SetNextItemWidth(-1);
                if(ImGui::Combo("##search_preset",&c.search_preset,"Balanced\0Explore broadly\0Refine layouts\0Find power\0Custom\0")) {
                    if(c.search_preset<4) {
                        c.search=SearchTuning{};
                        if(c.search_preset==1){c.search.agents=8;c.search.power_agents=2;c.search.elites=48;c.search.mutation_percent=100;c.search.max_flips=12;c.search.move_percent=25;c.search.crossover_percent=35;c.search.random_percent=35;c.search.migration_generations=40;c.search.restart_generations=60;}
                        if(c.search_preset==2){c.search.max_flips=2;c.search.move_percent=70;c.search.crossover_percent=10;c.search.random_percent=10;c.search.restart_generations=200;}
                        if(c.search_preset==3){c.search.agents=8;c.search.power_agents=4;c.search.mutation_percent=100;c.search.max_flips=8;c.search.move_percent=50;c.search.random_percent=35;c.search.migration_generations=30;c.search.restart_generations=80;}
                        changed=true;
                    }
                }
                ImGui::SetNextItemWidth(280*scale);bool manual=ImGui::SliderInt("Search agents",&c.search.agents,1,32);
                c.search.power_agents=std::min(c.search.power_agents,c.search.agents);
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::SliderInt("Power scouts",&c.search.power_agents,0,c.search.agents);
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::SliderInt("Mutation chance",&c.search.mutation_percent,0,100,"%d%%");
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::InputInt("Max flips per mutation",&c.search.max_flips);
                ImGui::SetItemTooltip("Flip 1 to this many distinct cells. Moving a rod changes two cells and keeps the fuel count constant.");
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::SliderInt("Move an existing rod",&c.search.move_percent,0,100,"%d%%");
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::SliderInt("Crossover chance",&c.search.crossover_percent,0,100,"%d%%");
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::SliderInt("Fresh random layouts",&c.search.random_percent,0,100,"%d%%");
                ImGui::SetItemTooltip("This fraction starts from scratch. Mutation and crossover chances apply to the remaining offspring.");
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::InputInt("Parents per agent",&c.search.elites);
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::InputInt("Share every N generations",&c.search.migration_generations);
                ImGui::SetNextItemWidth(280*scale);manual|=ImGui::InputInt("Restart after N stale generations",&c.search.restart_generations);
                if(manual){c.search_preset=4;changed=true;}
                ImGui::TextWrapped("Set either generation interval to 0 to disable it. Restarts preserve the best layouts already found. More agents share the same total search budget.");
                if(job.busy){ImGui::TextColored({.36f,.84f,1,1},"Valid changes apply after the current batch.");if(ImGui::Button("Reseed agents now",{0,36*scale}))job.control.reseed=true;}
                if(!valid_tuning(c.search))ImGui::TextColored({1,.7f,.45f,1},"Check parents (1-256), changed cells (1-1024), and nonnegative intervals.");
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Reactor & runtime")) {
                ImGui::BeginDisabled(job.busy);
                ImGui::InputInt("Minimum fuel columns",&c.minimum_rods);
                ImGui::TextWrapped("Every interior position is available for fuel. There is no additional upper fuel-column limit.");
                ImGui::SliderFloat("Rod insertion",&c.insertion,0,100,"%.0f%%");ImGui::SliderFloat("Fuel fill",&c.fill,.01f,1,"%.2f");
                ImGui::Combo("Search precision",&c.math,"Fast (float)\0Double\0");
                ImGui::InputInt("GPU batch size",&c.batch,1024,8192);ImGui::InputInt("GPU device",&c.gpu_index);
                ImGui::InputInt("Random seed",&c.seed);ImGui::InputText("Result filename",c.name,sizeof(c.name));
                ImGui::EndDisabled();ImGui::Spacing();ImGui::TextWrapped("Finalists are always verified using CPU double precision. Runtime settings are locked during a run; mutation and agent settings remain adjustable.");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();
        if(changed&&job.busy&&valid_tuning(c.search))job.control.set_tuning(c.search);
        if(ImGui::Button("Done",{ImGui::GetContentRegionAvail().x,40*scale}))ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}
void moderator_select(float x,float y,float width) {
    field_at(x,y,width);auto& c=configuration;ImGui::SetNextWindowSizeConstraints({width*scale,0},{width*scale,360*scale});
    if(ImGui::BeginCombo("##moderator",moderators[c.moderator].name)){
        ImGui::SetNextItemWidth(-1);ImGui::InputTextWithHint("##filter","Search materials...",c.filter,sizeof(c.filter));
        std::string needle=c.filter;std::transform(needle.begin(),needle.end(),needle.begin(),[](unsigned char ch){return char(std::tolower(ch));});int count=0;
        for(size_t i=0;i<std::size(moderators);++i){const auto& m=moderators[i];if(!m.placeable)continue;std::string name=m.name;std::transform(name.begin(),name.end(),name.begin(),[](unsigned char ch){return char(std::tolower(ch));});
            if(name.find(needle)==std::string::npos)continue;++count;if(ImGui::Selectable(m.name,c.moderator==int(i)))c.moderator=int(i);}
        if(!count)ImGui::TextDisabled("No matching materials");ImGui::EndCombo();
    }
}
void metric(ImDrawList* d,float x,float y,float width,const char* title,const std::string& value,const char* unit,ImU32 accent) {
    glass(d,{x*scale,y*scale},{width*scale,91*scale},13);text(d,{(x+16)*scale,(y+12)*scale},title,small,muted);
    text(d,{(x+16)*scale,(y+34)*scale},value.c_str(),big,accent);text(d,{(x+16)*scale,(y+69)*scale},unit,small,muted);
}
void draw_ui() {
    ImVec2 display=ImGui::GetIO().DisplaySize;float w=display.x/scale,h=display.y/scale;
    ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize(display);
    ImGui::Begin("ATM10",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoBackground);
    auto* d=ImGui::GetWindowDrawList();d->AddRectFilledMultiColor({0,0},display,IM_COL32(22,39,69,255),IM_COL32(8,15,30,255),IM_COL32(16,24,48,255),IM_COL32(11,21,40,255));
    glow(d,{display.x*.16f,display.y*.04f},420*scale,IM_COL32(77,149,246,255));glow(d,{display.x*.86f,display.y*.55f},340*scale,purple);glow(d,{display.x*.6f,display.y},360*scale,cyan);
    for(int i=0;i<5;++i)d->AddBezierCubic({0,(100+i*18.f)*scale},{display.x*.35f,(-90+i*25.f)*scale},{display.x*.48f,(230+i*12.f)*scale},{display.x,(20+i*15.f)*scale},IM_COL32(74,165,235,8),9*scale,50);
    auto s=job.snapshot();bool busy=job.busy;const auto& used=busy||s.done?job.used:configuration;Settings settings=used.settings();
    cube(d,{55*scale,54*scale},24*scale);text(d,{97*scale,26*scale},"ATM10 ER2 Optimizer",heading);text(d,{98*scale,66*scale},"Build a better reactor. Use less fuel.",body,muted);
    float chip_width=std::max(110.f,float(s.stage.size())*8.f+42);ImVec2 chip{(w-28-chip_width)*scale,35*scale};glass(d,chip,{chip_width*scale,38*scale},19);
    d->AddCircleFilled({chip.x+18*scale,chip.y+19*scale},4*scale,s.code?IM_COL32(255,174,106,255):busy?cyan:IM_COL32(98,230,168,255));text(d,{chip.x+31*scale,chip.y+9*scale},s.stage.c_str(),small);
    float left=std::clamp(w*.52f,470.f,640.f),right=w-left-72;
    glass(d,{24*scale,111*scale},{left*scale,(h-199)*scale});glass(d,{(left+48)*scale,111*scale},{right*scale,(h-199)*scale});
    text(d,{48*scale,134*scale},"SEARCH SETTINGS",small,cyan);text(d,{(left+72)*scale,134*scale},s.verified?"VERIFIED REACTOR":s.progress.best.rods>0?"SEARCH LEADER":"REACTOR PREVIEW",small,cyan);
    float x=48,field=x+158,fw=left-206,y=177,row_gap=h<800?46.f:52.f;auto& c=configuration;
    ImGui::BeginDisabled(busy);
    label(d,x,y,"Compute backend");field_at(field,y,fw);ImGui::Combo("##backend",&c.backend,"Auto (GPU preferred)\0NVIDIA GPU\0CPU\0");
    y+=row_gap;label(d,x,y,"CPU workers");field_at(field,y,fw);ImGui::InputInt("##threads",&c.threads);
    y+=row_gap;label(d,x,y,"Search duration");field_at(field,y,fw);ImGui::InputInt("##seconds",&c.seconds);if(ImGui::IsItemHovered())ImGui::SetTooltip("Seconds of search. Startup and final verification take additional time.");
    y+=row_gap;label(d,x,y,"Minimum power");field_at(field,y,fw);ImGui::InputFloat("##power",&c.minimum_power,10000,100000,"%.0f FE/t");
    y+=row_gap;label(d,x,y,"Interior size");float each=(fw-16)/3;field_at(field,y,each);bool changed=ImGui::InputInt("##width",&c.width,0,0);if(ImGui::IsItemHovered())ImGui::SetTooltip("Width: 1 to 32 blocks, excluding casing");field_at(field+each+8,y,each);changed|=ImGui::InputInt("##depth",&c.depth,0,0);if(ImGui::IsItemHovered())ImGui::SetTooltip("Depth: 1 to 32 blocks, excluding casing");field_at(field+2*(each+8),y,each);changed|=ImGui::InputInt("##height",&c.height,0,0);if(ImGui::IsItemHovered())ImGui::SetTooltip("Height: 1 to 64 blocks, excluding casing");
    text(d,{field*scale,(y+37)*scale},"WIDTH",small,muted);text(d,{(field+each+8)*scale,(y+37)*scale},"DEPTH",small,muted);text(d,{(field+2*(each+8))*scale,(y+37)*scale},"HEIGHT",small,muted);if(changed&&c.width>0&&c.width<=32&&c.depth>0&&c.depth<=32){c.minimum_rods=std::min(c.minimum_rods,c.width*c.depth);}
    y+=row_gap+20;label(d,x,y,"Moderator");moderator_select(field,y,fw);
    if(ImGui::IsItemHovered()){const auto& m=moderators[c.moderator].material;ImGui::SetTooltip("Absorption: %.3f\nHeat efficiency: %.3f\nModeration: %.3f\nConductivity: %.3f",m.absorption,m.heat_efficiency,m.moderation,m.conductivity);}
    y+=row_gap;label(d,x,y,"Optimize for");field_at(field,y,fw);ImGui::Combo("##objective",&c.objective,"Fuel efficiency\0Maximum power\0");
    y+=row_gap;label(d,x,y,"Save results to");field_at(field,y,fw-48);ImGui::InputText("##output",c.output,sizeof(c.output));ImGui::SetCursorPos({(field+fw-40)*scale,y*scale});bool browse=action("folder","",{40*scale,38*scale});
    ImVec2 fp{(field+fw-31)*scale,(y+13)*scale};d->AddRect(fp,{fp.x+22*scale,fp.y+14*scale},cyan,2*scale,0,1.5f*scale);d->AddLine({fp.x,fp.y},{fp.x+8*scale,fp.y-4*scale},cyan,1.5f*scale);d->AddLine({fp.x+8*scale,fp.y-4*scale},{fp.x+14*scale,fp.y},cyan,1.5f*scale);if(ImGui::IsItemHovered())ImGui::SetTooltip("Choose output folder");if(browse)choose_folder();
    ImGui::EndDisabled();
    y+=row_gap+2;ImGui::SetCursorPos({x*scale,y*scale});if(action("advanced","Advanced settings",{180*scale,36*scale}))ImGui::OpenPopup("Advanced settings");text(d,{(x+198)*scale,(y+10)*scale},"56 moderator presets",small,muted);
    if(developer_advanced_preview&&!ImGui::IsPopupOpen("Advanced settings"))ImGui::OpenPopup("Advanced settings");
    advanced_dialog();
    if(!config_valid())text(d,{48*scale,(h-118)*scale},"Check dimensions, search settings, and positive values.",small,IM_COL32(255,180,121,255));
    // Preview follows the run's immutable settings until the next run starts.
    float rx=left+72,rw=right-48;float grid_top=176,grid_height=std::max(140.f,h-539);
    float cell=std::min(rw/std::max(1,settings.width),grid_height/std::max(1,settings.depth));cell=std::min(cell,36.f);
    float grid_x=rx+(rw-cell*settings.width)/2,grid_y=grid_top+(grid_height-cell*settings.depth)/2;
    bool have=s.verified||s.progress.best.rods>0;Result result=s.verified?s.result:s.progress.best;
    for(int z=0;z<std::clamp(settings.depth,1,32);++z)for(int a=0;a<std::clamp(settings.width,1,32);++a){bool fuel=have&&result.mask.test(z*settings.width+a);ImVec2 p{(grid_x+a*cell)*scale,(grid_y+z*cell)*scale};float inset=std::max(1.f,cell*.10f)*scale;
        d->AddRectFilled({p.x+inset,p.y+inset},{p.x+(cell*scale)-inset,p.y+(cell*scale)-inset},fuel?IM_COL32(96,217,255,210):IM_COL32(95,92,157,50),std::min(6.f,cell*.16f)*scale);
        d->AddRect({p.x+inset,p.y+inset},{p.x+cell*scale-inset,p.y+cell*scale-inset},fuel?cyan:IM_COL32(147,135,207,90),std::min(6.f,cell*.16f)*scale,0,scale);
        if(fuel&&cell>=20)text(d,{p.x+(cell*.34f)*scale,p.y+(cell*.22f)*scale},"R",small,IM_COL32(14,47,73,255));
        if(ImGui::IsMouseHoveringRect(p,{p.x+cell*scale,p.y+cell*scale}))ImGui::SetTooltip("Column %d, %d: %s\n%d blocks tall",a+1,z+1,fuel?"Fuel rod":moderators[used.moderator].name,settings.height);
    }
    char dims[100];std::snprintf(dims,sizeof(dims),"%d x %d x %d interior  /  %d fuel columns",settings.width,settings.depth,settings.height,have?result.rods:0);
    text(d,{rx*scale,(grid_top+grid_height+10)*scale},dims,small,muted);
    float my=h-335;metric(d,rx,my,(rw-12)/2,"POWER",have?compact(result.power):"--","FE / tick",cyan);metric(d,rx+(rw+12)/2,my,(rw-12)/2,"FUEL USE",have?([&]{char n[32];std::snprintf(n,sizeof(n),"%.4f",result.fuel);return std::string(n);}()):"--","mB / tick",purple);
    text(d,{rx*scale,(my+107)*scale},"FUEL EFFICIENCY",small,muted);text(d,{rx*scale,(my+131)*scale},have?(compact(result.efficiency)+" FE / mB").c_str():"Awaiting first layout",bold);
    text(d,{rx*scale,(my+152)*scale},s.verified?"Verified with CPU double precision":have?"Search estimate - verification pending":"Fuel rods repeat through the reactor height",small,s.verified?IM_COL32(111,230,183,255):muted);
    float progress_y=h-163;
    double batch_age=s.progress_at?std::max(0.,now_seconds()-s.progress_at):0;
    double duration=s.progress.seconds+(busy&&s.stage=="Searching"?batch_age:0);
    float fraction=busy?float(duration/std::max(1,used.seconds)):s.done?1.f:0.f;
    if(s.stage=="Verifying finalists")fraction=1;
    ImGui::SetCursorPos({rx*scale,progress_y*scale});ImGui::SetNextItemWidth(rw*scale);ImGui::ProgressBar(std::clamp(fraction,0.f,1.f),{rw*scale,5*scale},"");
    bool meets=have&&result.power>=used.minimum_power;
    std::string summary=busy?s.stage:s.error.empty()?(s.verified?"Saved - verified power target met":"Ready when you are"):s.stage;
    if(busy&&s.stage=="Searching"&&have)summary=meets?"Power target met in search - verifying at finish":"Below target: "+compact(std::max(0.,used.minimum_power-result.power))+" FE/t more needed";
    text(d,{rx*scale,(progress_y+14)*scale},summary.c_str(),small,!s.error.empty()||(have&&!meets)?IM_COL32(255,180,121,255):muted);
    if(s.progress.candidates){
        std::string rate=counted(s.progress.candidates)+" evaluations  /  "+compact(double(s.progress.candidates)/std::max(.0001,s.progress.seconds))+" per sec";
        text(d,{rx*scale,(progress_y+35)*scale},rate.c_str(),small,muted);
        char activity[180];
        if(busy&&s.stage=="Searching"&&batch_age>2)std::snprintf(activity,sizeof(activity),"Last batch %.1fs ago: %s",batch_age,s.phase.c_str());
        else std::snprintf(activity,sizeof(activity),"%d agents  /  %llu restarts  /  last improvement %.1fs ago",s.progress.agents,s.progress.restarts,std::max(0.,duration-s.progress.last_improvement_seconds));
        text(d,{rx*scale,(progress_y+55)*scale},activity,small,muted);
    }
    ImGui::SetCursorPos({24*scale,(h-68)*scale});
    if(busy){if(action("finish",job.control.finish?"Finishing current work...":"Finish early & verify",{(left*.65f)*scale,46*scale},true))job.control.finish=true;}
    else {ImGui::BeginDisabled(!config_valid());if(action("run","Run optimization",{(left*.65f)*scale,46*scale},true))job.start(c,false,developer_budget);ImGui::EndDisabled();}
    ImGui::SetCursorPos({(24+left*.65f+12)*scale,(h-68)*scale});
    if(busy){ImGui::BeginDisabled(s.stage!="Searching");if(action("reseed",job.control.reseed?"Reseeding...":"Reseed agents",{(left*.35f-12)*scale,46*scale}))job.control.reseed=true;ImGui::EndDisabled();}
    else {ImGui::BeginDisabled(!config_valid());if(action("bench","Benchmark",{(left*.35f-12)*scale,46*scale}))job.start(c,true);ImGui::EndDisabled();}
    ImGui::SetCursorPos({(left+48)*scale,(h-68)*scale});if(action("log","Run log",{110*scale,46*scale}))ImGui::OpenPopup("Run details");
    ImGui::SetCursorPos({(left+170)*scale,(h-68)*scale});ImGui::BeginDisabled(!s.verified);if(action("copy","Copy layout",{126*scale,46*scale}))ImGui::SetClipboardText(layout_text(s.result.mask,job.used.settings()).c_str());ImGui::EndDisabled();
    ImGui::SetCursorPos({(left+308)*scale,(h-68)*scale});ImGui::BeginDisabled(!s.verified);if(action("open","Open results",{(right-260)*scale,46*scale})){auto path=utfpath(job.used.output);ShellExecuteW(window,L"open",path.c_str(),nullptr,nullptr,SW_SHOWNORMAL);}ImGui::EndDisabled();
    ImGui::SetNextWindowSize({780*scale,540*scale},ImGuiCond_Appearing);
    static bool was_done=false;if(s.done&&!was_done&&(s.benchmark||s.code))ImGui::OpenPopup("Run details");was_done=s.done;
    if(ImGui::BeginPopupModal("Run details",nullptr,ImGuiWindowFlags_NoResize)){
        if(!s.error.empty()){ImGui::PushTextWrapPos(0);ImGui::TextColored({1,.7f,.45f,1},"%s",s.error.c_str());ImGui::PopTextWrapPos();ImGui::Separator();}
        if(!s.device_name.empty())ImGui::Text("Compute device: %s",s.device_name.c_str());
        ImGui::BeginChild("loglines",{0,-50*scale});for(auto& line:s.log)ImGui::TextUnformatted(line.c_str());ImGui::EndChild();
        if(ImGui::Button("Close",{120*scale,36*scale}))ImGui::CloseCurrentPopup();ImGui::SameLine();if(ImGui::Button("Copy log",{120*scale,36*scale})){std::string all;for(auto& line:s.log)all+=line+'\n';ImGui::SetClipboardText(all.c_str());}ImGui::EndPopup();
    }
    ImGui::End();
}

void create_target() {ID3D11Texture2D* buffer=nullptr;if(SUCCEEDED(swap_chain->GetBuffer(0,IID_PPV_ARGS(&buffer)))){device->CreateRenderTargetView(buffer,nullptr,&target);buffer->Release();}}
bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC desc{};desc.BufferCount=2;desc.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.OutputWindow=hwnd;desc.SampleDesc.Count=1;desc.Windowed=TRUE;desc.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_0};
    HRESULT hr=D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,levels,2,D3D11_SDK_VERSION,&desc,&swap_chain,&device,nullptr,&context);
    if(FAILED(hr))hr=D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,levels,2,D3D11_SDK_VERSION,&desc,&swap_chain,&device,nullptr,&context);
    if(FAILED(hr))return false;create_target();return target!=nullptr;
}
bool screenshot(const std::filesystem::path& path) {
    ID3D11Texture2D *buffer=nullptr,*staging=nullptr;if(FAILED(swap_chain->GetBuffer(0,IID_PPV_ARGS(&buffer))))return false;
    D3D11_TEXTURE2D_DESC desc;buffer->GetDesc(&desc);desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
    bool ok=false;if(SUCCEEDED(device->CreateTexture2D(&desc,nullptr,&staging))){context->CopyResource(staging,buffer);D3D11_MAPPED_SUBRESOURCE mapped{};
        if(SUCCEEDED(context->Map(staging,0,D3D11_MAP_READ,0,&mapped))){
            IWICImagingFactory* factory=nullptr;IWICStream* stream=nullptr;IWICBitmapEncoder* encoder=nullptr;IWICBitmapFrameEncode* frame=nullptr;
            if(SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)))&&SUCCEEDED(factory->CreateStream(&stream))&&SUCCEEDED(stream->InitializeFromFilename(path.c_str(),GENERIC_WRITE))&&SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder))&&SUCCEEDED(encoder->Initialize(stream,WICBitmapEncoderNoCache))&&SUCCEEDED(encoder->CreateNewFrame(&frame,nullptr))&&SUCCEEDED(frame->Initialize(nullptr))&&SUCCEEDED(frame->SetSize(desc.Width,desc.Height))){
                WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;
                std::vector<BYTE> pixels(size_t(desc.Width)*desc.Height*4);
                for(UINT y=0;y<desc.Height;++y)for(UINT x=0;x<desc.Width;++x){const BYTE* in=(BYTE*)mapped.pData+y*mapped.RowPitch+x*4;BYTE* out=pixels.data()+(size_t(y)*desc.Width+x)*4;out[0]=in[2];out[1]=in[1];out[2]=in[0];out[3]=in[3];}
                ok=SUCCEEDED(frame->SetPixelFormat(&format))&&format==GUID_WICPixelFormat32bppBGRA&&SUCCEEDED(frame->WritePixels(desc.Height,desc.Width*4,UINT(pixels.size()),pixels.data()))&&SUCCEEDED(frame->Commit())&&SUCCEEDED(encoder->Commit());}
            release(frame);release(encoder);release(stream);release(factory);context->Unmap(staging,0);
        }}release(staging);release(buffer);return ok;
}
LRESULT WINAPI window_proc(HWND hwnd,UINT message,WPARAM wp,LPARAM lp) {
    if(ImGui_ImplWin32_WndProcHandler(hwnd,message,wp,lp))return true;
    switch(message){
        case WM_SIZE:if(wp!=SIZE_MINIMIZED){resize_width=LOWORD(lp);resize_height=HIWORD(lp);}return 0;
        case WM_GETMINMAXINFO:{auto* m=(MINMAXINFO*)lp;m->ptMinTrackSize={LONG(1080*scale),LONG(790*scale)};return 0;}
        case WM_DPICHANGED:{auto* r=(RECT*)lp;SetWindowPos(hwnd,nullptr,r->left,r->top,r->right-r->left,r->bottom-r->top,SWP_NOZORDER|SWP_NOACTIVATE);return 0;}
        case WM_SYSCOMMAND:if((wp&0xfff0)==SC_KEYMENU)return 0;break;
        case WM_CLOSE:if(job.busy){job.control.finish=true;close_when_done=true;return 0;}DestroyWindow(hwnd);return 0;
        case WM_DESTROY:PostQuitMessage(0);return 0;
    }return DefWindowProcW(hwnd,message,wp,lp);
}
void fonts() {
    auto& io=ImGui::GetIO();io.Fonts->Clear();wchar_t windows_dir[MAX_PATH]{};GetWindowsDirectoryW(windows_dir,MAX_PATH);auto dir=std::filesystem::path(windows_dir)/"Fonts";
    auto add=[&](const wchar_t* name,float size){auto p=utf8(dir/name);auto* f=io.Fonts->AddFontFromFileTTF(p.c_str(),size*scale);return f?f:io.Fonts->AddFontDefault();};
    body=add(L"segoeui.ttf",18);small=add(L"segoeui.ttf",14);bold=add(L"segoeuib.ttf",17);heading=add(L"segoeuisl.ttf",31);big=add(L"segoeuisl.ttf",28);io.FontDefault=body;
    io.Fonts->Build();
    for(auto* font:{body,small,bold,heading,big}){auto* baked=font->GetFontBaked(font->LegacySize);for(ImWchar c=32;c<127;++c)baked->FindGlyph(c);}
}
void style() {
    auto& s=ImGui::GetStyle();s=ImGuiStyle{};s.WindowPadding={20,18};s.FramePadding={14,9};s.ItemSpacing={10,10};s.WindowRounding=16;s.FrameRounding=10;s.PopupRounding=12;s.ScrollbarRounding=8;s.FrameBorderSize=1;s.WindowBorderSize=1;s.ScaleAllSizes(scale);
    auto* c=s.Colors;c[ImGuiCol_Text]={.92f,.95f,1,1};c[ImGuiCol_TextDisabled]={.54f,.65f,.78f,1};c[ImGuiCol_WindowBg]={.065f,.10f,.17f,.99f};c[ImGuiCol_PopupBg]={.08f,.13f,.21f,.99f};c[ImGuiCol_Border]={.51f,.64f,.81f,.30f};c[ImGuiCol_FrameBg]={.10f,.16f,.25f,.82f};c[ImGuiCol_FrameBgHovered]={.16f,.24f,.36f,.9f};c[ImGuiCol_FrameBgActive]={.18f,.30f,.43f,1};c[ImGuiCol_Button]={.14f,.24f,.37f,1};c[ImGuiCol_ButtonHovered]={.20f,.38f,.55f,1};c[ImGuiCol_ButtonActive]={.24f,.48f,.65f,1};c[ImGuiCol_Header]={.18f,.34f,.53f,1};c[ImGuiCol_HeaderHovered]={.23f,.43f,.66f,1};c[ImGuiCol_HeaderActive]={.28f,.49f,.71f,1};c[ImGuiCol_CheckMark]={.36f,.83f,1,1};c[ImGuiCol_SliderGrab]={.36f,.83f,1,1};c[ImGuiCol_PlotHistogram]={.36f,.83f,1,1};
}
}
int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR,int show) {
    ImGui_ImplWin32_EnableDpiAwareness();CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    int argc=0;LPWSTR* argv=CommandLineToArgvW(GetCommandLineW(),&argc);std::filesystem::path capture,smoke_report;bool demo=false,small_window=false,target_test=false;
    for(int i=1;i<argc;++i){if(std::wstring(argv[i])==L"--screenshot"&&i+1<argc)capture=argv[++i];else if(std::wstring(argv[i])==L"--smoke-test"&&i+1<argc)smoke_report=argv[++i];else if(std::wstring(argv[i])==L"--target-test"&&i+1<argc){smoke_report=argv[++i];target_test=true;}else if(std::wstring(argv[i])==L"--demo")demo=true;else if(std::wstring(argv[i])==L"--advanced-preview")developer_advanced_preview=true;else if(std::wstring(argv[i])==L"--small-window")small_window=true;}LocalFree(argv);
    bool hidden=!capture.empty()||!smoke_report.empty();
    scale=ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint({0,0},MONITOR_DEFAULTTOPRIMARY));
    if(!hidden){MONITORINFO info{};info.cbSize=sizeof(info);if(GetMonitorInfoW(MonitorFromPoint({0,0},MONITOR_DEFAULTTOPRIMARY),&info)){float fit=std::min((info.rcWork.right-info.rcWork.left-40)/1200.f,(info.rcWork.bottom-info.rcWork.top-60)/850.f);scale=std::min(scale,std::max(.65f,fit));}}
    WNDCLASSEXW wc{sizeof(wc),CS_CLASSDC,window_proc,0,0,instance,nullptr,LoadCursor(nullptr,IDC_ARROW),nullptr,nullptr,L"ATM10Glass",nullptr};RegisterClassExW(&wc);
    RECT rect{0,0,LONG((small_window?1080:1200)*scale),LONG((small_window?750:850)*scale)};AdjustWindowRect(&rect,WS_OVERLAPPEDWINDOW,FALSE);
    window=CreateWindowW(wc.lpszClassName,L"ATM10 ER2 Optimizer",WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,rect.right-rect.left,rect.bottom-rect.top,nullptr,nullptr,instance,nullptr);
    BOOL dark=TRUE;DwmSetWindowAttribute(window,20,&dark,sizeof(dark));int corners=2;DwmSetWindowAttribute(window,33,&corners,sizeof(corners));
    if(!create_device(window)){MessageBoxW(nullptr,L"Could not initialize DirectX 11.",L"ATM10 Optimizer",MB_OK|MB_ICONERROR);DestroyWindow(window);CoUninitialize();return 1;}
    if(!hidden){ShowWindow(window,show);UpdateWindow(window);}IMGUI_CHECKVERSION();ImGui::CreateContext();ImGui::GetIO().IniFilename=nullptr;ImGui::GetIO().ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;
    fonts();style();ImGui_ImplWin32_Init(window);ImGui_ImplDX11_Init(device,context);
    if(demo){configuration.seconds=5;configuration.minimum_power=350000;job.start(configuration,false);}
    if(!smoke_report.empty()){
        if(target_test){configuration.backend=1;configuration.seconds=5;configuration.minimum_power=350000;}
        else {configuration.backend=2;configuration.threads=4;configuration.width=3;configuration.depth=5;configuration.height=4;configuration.minimum_power=0;
            for(size_t i=0;i<std::size(moderators);++i)if(std::string(moderators[i].key)=="water")configuration.moderator=int(i);developer_budget=512;}
        auto folder=utf8(smoke_report.parent_path()/"gui_results");std::snprintf(configuration.output,sizeof(configuration.output),"%s",folder.c_str());}
    bool quit=false,captured=false;int frames=0,result_code=0,settled_frames=0;auto start=std::chrono::steady_clock::now();
    while(!quit){MSG msg;while(PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)){TranslateMessage(&msg);DispatchMessageW(&msg);if(msg.message==WM_QUIT)quit=true;}if(quit)break;
        if(close_when_done&&!job.busy){DestroyWindow(window);continue;}
        if(IsIconic(window)){Sleep(20);continue;}
        if(resize_width&&resize_height){release(target);swap_chain->ResizeBuffers(0,resize_width,resize_height,DXGI_FORMAT_UNKNOWN,0);resize_width=resize_height=0;create_target();}
        RECT client{};GetClientRect(window,&client);float new_scale=std::min(ImGui_ImplWin32_GetDpiScaleForHwnd(window),std::min(client.right/1080.f,client.bottom/750.f));if(std::abs(new_scale-scale)>.001f){scale=new_scale;ImGui_ImplDX11_InvalidateDeviceObjects();fonts();style();}
        ImGui_ImplDX11_NewFrame();ImGui_ImplWin32_NewFrame();
        if(!smoke_report.empty()&&frames<3){auto& io=ImGui::GetIO();float logical_width=io.DisplaySize.x/scale,logical_height=io.DisplaySize.y/scale,left=std::clamp(logical_width*.52f,470.f,640.f);
            io.AddMousePosEvent((24+left*.3f)*scale,(logical_height-45)*scale);if(frames==1)io.AddMouseButtonEvent(0,true);if(frames==2)io.AddMouseButtonEvent(0,false);}
        ImGui::NewFrame();draw_ui();ImGui::Render();
        float clear[4]={.04f,.07f,.12f,1};context->OMSetRenderTargets(1,&target,nullptr);context->ClearRenderTargetView(target,clear);ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());++frames;
        if(!demo||!job.busy)++settled_frames;else settled_frames=0;
        if(!capture.empty()&&settled_frames>8){captured=screenshot(capture);result_code=captured?0:2;quit=true;}
        if(!smoke_report.empty()&&frames>3&&!job.busy){auto s=job.snapshot();std::ofstream report(smoke_report);bool pass=s.code==0&&s.verified&&s.result.rods>0&&s.result.rods<=configuration.width*configuration.depth&&std::filesystem::exists(utfpath(configuration.output)/"best_reactor.json")&&(target_test?s.result.power>=350000&&s.progress.candidates>0:s.progress.candidates==512);
            report<<"GUI button / worker test: "<<(pass?"PASS":"FAIL")<<"\nRendered frames: "<<frames<<"\nRun button activated with ImGui mouse events\nGeometry: "<<configuration.width<<" x "<<configuration.depth<<" x "<<configuration.height<<"\nModerator: "<<moderators[configuration.moderator].key<<"\nExact CPU verified: "<<s.verified<<"\nPower: "<<std::setprecision(17)<<s.result.power<<" FE/t\nMinimum power: "<<configuration.minimum_power<<" FE/t\nEvaluations: "<<s.progress.candidates<<"\n";result_code=pass?0:3;quit=true;}
        if(hidden&&std::chrono::steady_clock::now()-start>std::chrono::seconds(60)){job.control.finish=true;result_code=4;quit=true;}
        swap_chain->Present(hidden?0:1,0);if(hidden)Sleep(8);
    }
    job.control.finish=true;while(job.busy)Sleep(10);
    ImGui_ImplDX11_Shutdown();ImGui_ImplWin32_Shutdown();ImGui::DestroyContext();release(target);release(swap_chain);release(context);release(device);if(IsWindow(window))DestroyWindow(window);UnregisterClassW(wc.lpszClassName,instance);CoUninitialize();return result_code;
}
