#include "infer/sam3infer.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <chrono>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    if(argc != 6) { std::cerr << "usage: sam3_export ENGINE_DIR IMAGE TOKENS PROMPT OUT_DIR\n"; return 2; }
    std::array<int64_t,32> ids{}, attention{};
    std::ifstream tokens(argv[3]);
    for(auto &x:ids) if(!(tokens>>x)) return 2;
    for(auto &x:attention) if(!(tokens>>x)) return 2;
    std::string dir(argv[1]), prompt(argv[4]);
    auto engine=load(dir+"/vision_encoder.engine",dir+"/text_encoder.engine","",dir+"/decoder.engine",0);
    if(!engine) return 1;
    engine->setup_text_inputs(prompt,ids,attention);
    auto run = [&](const std::string &image_path, const std::string &output_path) -> int {
    std::filesystem::path out(output_path);
    if(std::filesystem::exists(out) && !std::filesystem::is_empty(out)) return 2;
    std::filesystem::create_directories(out);
    cv::Mat img=cv::imread(image_path); if(img.empty()) return 2;
    std::vector<Sam3Input> inputs{Sam3Input(img,{Sam3PromptUnit(prompt)},0.3f)};
    cudaEvent_t begin,end;
    if(cudaEventCreate(&begin)!=cudaSuccess || cudaEventCreate(&end)!=cudaSuccess) return 1;
    cudaDeviceSynchronize();
    auto wall_start=std::chrono::steady_clock::now();
    cudaEventRecord(begin,0);
    auto results=engine->forwards(inputs,true);
    cudaEventRecord(end,0); cudaEventSynchronize(end);
    float gpu_ms=0;cudaEventElapsedTime(&gpu_ms,begin,end);
    double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count();
    cudaEventDestroy(begin);cudaEventDestroy(end);
    if(results.size()!=1) return 1;
    cv::FileStorage meta((out/"detections.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    meta<<"cuda_event_span_ms"<<gpu_ms<<"inference_wall_ms"<<wall_ms;
    meta<<"prompt"<<prompt<<"detections"<<"[";
    int i=0;
    for(const auto &r:results[0]) {
        if(!r.segmentation || r.segmentation->mask.empty()) continue;
        int x=std::max(0,(int)r.box.left), y=std::max(0,(int)r.box.top);
        auto m=r.segmentation->mask;
        if(x+m.cols>img.cols || y+m.rows>img.rows) return 1;
        cv::Mat full=cv::Mat::zeros(img.size(),CV_8U);
        m.copyTo(full(cv::Rect(x,y,m.cols,m.rows)));
        std::string name="mask_"+std::to_string(i++)+".png";
        if(!cv::imwrite((out/name).string(),full)) return 1;
        meta<<"{"<<"mask"<<name<<"score"<<r.score<<"}";
    }
    meta<<"]"; meta.release(); return 0;
    };
    if(std::string(argv[2])=="--serve") {
        std::cout << "READY" << std::endl;
        std::string line;
        while(std::getline(std::cin,line)) {
            auto pos=line.find('\t');
            if(pos==std::string::npos) { std::cout<<"DONE 2"<<std::endl; continue; }
            int code=run(line.substr(0,pos),line.substr(pos+1));
            std::cout<<"DONE "<<code<<std::endl;
        }
        return 0;
    }
    return run(argv[2],argv[5]);
}
