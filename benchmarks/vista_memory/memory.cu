#include "common.cuh"
__global__ void read_all(const double*input,double*out,size_t n){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=input[i]*2.0+1.0;
}
int main(int argc,char**argv)try{
    Env env;
    Stream stream;
    Events events;
    int reps=argc>2?positive(argv[2]):30;
    CSV csv(argc>1?argv[1]:"memory.csv",env,"strategy,bytes,warmups,repetitions,setup_ms,movement_event_ms,movement_host_ms,first_ms,repeat_ms,correct,status");
    std::cout<<"strategy bytes setup_ms movement_ms first_ms repeat_ms correct\n";
    for(size_t bytes:{
        size_t(4096),size_t(1)<<20,size_t(16)<<20,size_t(64)<<20,size_t(256)<<20
    })for(std::string mode:{
        "hbm","managed","host"
    }){
        std::string reason=mode=="managed"&&(!env.managed||!env.concurrent)?"managed prefetch requires managedMemory and concurrentManagedAccess":mode=="host"&&!env.mapped?"mapped host memory unsupported":"";
        if(!reason.empty()){
            csv.row(mode+","+std::to_string(bytes)+",5,"+std::to_string(reps)+",,,,,,,skip,"+quote(reason));
            std::cout<<mode<<": skip "<<reason<<'\n';
            continue;
        }
        size_t n=bytes/sizeof(double);
        std::vector<double> src(n);
        for(size_t i=0;i<n;++i)src[i]=double(i%251)/256;
        Buffer output(bytes,"hbm");
        auto t=Clock::now();
        Buffer input(bytes,mode);
        if(mode!="hbm")input.init(src.data());
        double setup=elapsed(t);
        t=Clock::now();
        double movement=events.time(stream,[&](){
            if(mode=="hbm")CU(cudaMemcpyAsync(input.device,src.data(),bytes,cudaMemcpyHostToDevice,stream));else input.prefetch(stream);
        });
        double movement_host=elapsed(t);
        auto run=[&](){
            read_all<<<(n+255)/256,256,0,stream>>>((double*)input.device,(double*)output.device,n);
            CU(cudaGetLastError());
        };
        double first=events.time(stream,run);
        for(int i=0;i<5;++i)run();
        CU(cudaStreamSynchronize(stream));
        double repeated=events.time(stream,[&](){
            for(int i=0;i<reps;++i)run();
        })/reps;
        std::vector<double> result(n);
        CU(cudaMemcpy(result.data(),output.device,bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<n;++i)if(result[i]!=src[i]*2+1)throw std::runtime_error("memory validation failed");
        csv.row(mode+","+std::to_string(bytes)+",5,"+std::to_string(reps)+","+number(setup)+","+number(movement)+","+number(movement_host)+","+number(first)+","+number(repeated)+",pass,ok");
        std::cout<<mode<<' '<<bytes<<' '<<setup<<' '<<movement<<' '<<first<<' '<<repeated<<" pass\n";
    }
    return 0;
}
catch(const std::exception&e){
    std::cerr<<e.what()<<'\n';
    return 1;
}
