#include "common.cuh"
int main(int argc,char**argv)try{
    int reps=argc>2?positive(argv[2]):30;
    Env env;
    Stream stream;
    Events events;
    CSV csv(argc>1?argv[1]:"transfer.csv",env,"strategy,direction,bytes,warmups,repetitions,event_ms,host_ms,GB_s,correct");
    std::cout<<"strategy direction bytes event_ms host_ms GB/s correct\n";
    for(size_t n:{
        size_t(4096),size_t(1)<<20,size_t(16)<<20,size_t(64)<<20,size_t(256)<<20
    })for(std::string mode:{
        "pageable","pinned","pinned_async"
    }){
        std::vector<unsigned char> expected(n),page;
        for(size_t i=0;i<n;++i)expected[i]=(i*13+7)%251;
        unsigned char*h=nullptr;
        if(mode=="pageable"){
            page.resize(n);
            h=page.data();
        }
        else CU(cudaMallocHost(&h,n));
        // Host allocation ownership covers all subsequent CUDA failures.
        struct HostGuard {
            void*p;
            ~HostGuard(){
                if(p)cudaFreeHost(p);
            }
        }
        guard{
            mode=="pageable"?nullptr:h
        };
        Buffer device(n,"hbm");
        std::memcpy(h,expected.data(),n);
        CU(cudaMemcpy(device.device,h,n,cudaMemcpyHostToDevice));
        for(bool h2d:{
            true,false
        }){
            auto copy=[&](){
                auto kind=h2d?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost;
                void*dst=h2d?device.device:h;
                const void*src=h2d?h:device.device;
                if(mode=="pinned_async")CU(cudaMemcpyAsync(dst,src,n,kind,stream));
                else CU(cudaMemcpy(dst,src,n,kind));
            };
            // Synchronous copies use the default stream; async copies use the private stream.
            cudaStream_t s=mode=="pinned_async"?static_cast<cudaStream_t>(stream):nullptr;
            for(int i=0;i<5;++i)copy();
            CU(cudaDeviceSynchronize());
            double gpu=0,host=0;
            for(int i=0;i<reps;++i){
                auto t=Clock::now();
                gpu+=events.time(s,copy);
                host+=elapsed(t);
            }
            gpu/=reps;
            host/=reps;
            // Validate a fresh copy into poisoned storage outside the timing loop.
            if(h2d) {
                CU(cudaMemset(device.device, 0, n));
            } else {
                std::memset(h, 0, n);
            }
            copy();
            CU(cudaStreamSynchronize(s));
            if(h2d){
                std::vector<unsigned char> got(n);
                CU(cudaMemcpy(got.data(),device.device,n,cudaMemcpyDeviceToHost));
                if(got!=expected)throw std::runtime_error("H2D validation failed");
            }
            else if(std::memcmp(h,expected.data(),n))throw std::runtime_error("D2H validation failed");
            double bw=n/(gpu*1e6);
            std::string dir=h2d?"H2D":"D2H";
            csv.row(mode+","+dir+","+std::to_string(n)+",5,"+std::to_string(reps)+","+number(gpu)+","+number(host)+","+number(bw)+",pass");
            std::cout<<mode<<' '<<dir<<' '<<n<<' '<<gpu<<' '<<host<<' '<<bw<<" pass\n";
        }
    }
    return 0;
}
catch(const std::exception&e){
    std::cerr<<e.what()<<'\n';
    return 1;
}
