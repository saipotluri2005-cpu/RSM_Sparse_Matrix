#include "common.cuh"
#include "matrix.hpp"
#include <cusparse.h>
inline void sparse(cusparseStatus_t s){
    if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(std::string("cuSPARSE: ")+cusparseGetErrorString(s));
}
struct SparseObjects {
    cusparseHandle_t h=nullptr;
    cusparseSpMatDescr_t a=nullptr;
    cusparseDnVecDescr_t x=nullptr,y=nullptr;
    SparseObjects(){
        sparse(cusparseCreate(&h));
    }
    ~SparseObjects(){
        if(y)cusparseDestroyDnVec(y);
        if(x)cusparseDestroyDnVec(x);
        if(a)cusparseDestroySpMat(a);
        if(h)cusparseDestroy(h);
    }
};
int main(int argc,char**argv)try{
    if(argc<3)throw std::runtime_error("Usage: spmv OUTPUT_DIRECTORY MATRIX [MATRIX ...] (set VISTA_REPS, default 50)");
    int reps=std::getenv("VISTA_REPS")?positive(std::getenv("VISTA_REPS")):50;
    Env env;
    Stream stream;
    Events events;
    CSV csv(std::string(argv[1])+"/spmv.csv",env,"matrix,strategy,rows,cols,nnz,csr_bytes,warmups,repetitions,setup_ms,movement_event_ms,movement_host_ms,first_ms,repeat_ms,max_scaled_error,correct,status");
    CSV amort(std::string(argv[1])+"/amortization.csv",env,"matrix,strategy,rows,cols,nnz,csr_bytes,iterations,trials,movement_host_ms,total_host_ms,spmv_event_ms,correct,status");
    for(int arg=2;arg<argc;++arg){
        Matrix m=load_matrix(argv[arg]);
        std::vector<double>x(m.cols),reference(m.rows,0),scale(m.rows,0);
        for(int j=0;j<m.cols;++j)x[j]=1.0+(j%17)/32.0;
        for(int i=0;i<m.rows;++i)for(int k=m.row[i];k<m.row[i+1];++k){
            reference[i]+=m.val[k]*x[m.col[k]];
            scale[i]+=std::abs(m.val[k]*x[m.col[k]]);
        }
        std::cout<<argv[arg]<<" rows="<<m.rows<<" cols="<<m.cols<<" nnz="<<m.val.size()<<" CSR_bytes="<<m.bytes()<<'\n';
        for(std::string mode:{
            "hbm","managed","host"
        }){
            std::string prefix=quote(argv[arg])+","+mode+","+std::to_string(m.rows)+","+std::to_string(m.cols)+","+std::to_string(m.val.size())+","+std::to_string(m.bytes());
            auto skip=[&](std::string why){
                csv.row(prefix+",5,"+std::to_string(reps)+",,,,,,,,skip,"+quote(why));
                for(int n:{
                    1,10,100,1000
                })amort.row(prefix+","+std::to_string(n)+",3,,,,skip,"+quote(why));
                std::cout<<mode<<" skip: "<<why<<'\n';
            };
            if(mode=="managed"&&(!env.managed||!env.concurrent)){
                skip("managed prefetch unsupported");
                continue;
            }
            if(mode=="host"&&(!env.mapped||!env.p.unifiedAddressing)){
                skip("mapped host access / unified addressing unsupported");
                continue;
            }
            try{
                Buffer dx(x.size()*8,"hbm"),dy(m.rows*size_t(8),"hbm");
                dx.init(x.data());
                auto t=Clock::now();
                Buffer row(m.row.size()*4,mode),col(m.col.size()*4,mode),val(m.val.size()*8,mode);
                if(mode!="hbm"){
                    row.init(m.row.data());
                    col.init(m.col.data());
                    val.init(m.val.data());
                }
                double setup=elapsed(t);
                auto move=[&](){
                    if(mode=="hbm"){
                        CU(cudaMemcpyAsync(row.device,m.row.data(),row.bytes,cudaMemcpyHostToDevice,stream));
                        CU(cudaMemcpyAsync(col.device,m.col.data(),col.bytes,cudaMemcpyHostToDevice,stream));
                        CU(cudaMemcpyAsync(val.device,m.val.data(),val.bytes,cudaMemcpyHostToDevice,stream));
                    }
                    else {
                        row.prefetch(stream);
                        col.prefetch(stream);
                        val.prefetch(stream);
                    }
                };
                t=Clock::now();
                double movement=events.time(stream,move);
                double movement_host=elapsed(t);
                SparseObjects sp;
                sparse(cusparseSetStream(sp.h,stream));
                sparse(cusparseCreateCsr(&sp.a,m.rows,m.cols,m.val.size(),row.device,col.device,val.device,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
                sparse(cusparseCreateDnVec(&sp.x,m.cols,dx.device,CUDA_R_64F));
                sparse(cusparseCreateDnVec(&sp.y,m.rows,dy.device,CUDA_R_64F));
                double alpha=1,beta=0;
                size_t workbytes=0;
                sparse(cusparseSpMV_bufferSize(sp.h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,sp.a,sp.x,&beta,sp.y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&workbytes));
                Buffer workspace(workbytes,"hbm");
                auto run=[&](){
                    sparse(cusparseSpMV(sp.h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,sp.a,sp.x,&beta,sp.y,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,workspace.device));
                };
                auto validate=[&](){
                    std::vector<double>y(m.rows);
                    CU(cudaMemcpy(y.data(),dy.device,y.size()*8,cudaMemcpyDeviceToHost));
                    double err=0;
                    for(int i=0;i<m.rows;++i){
                        if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite SpMV result");
                        err=std::max(err,std::abs(y[i]-reference[i])/(1+scale[i]));
                    }
                    if(err>1e-10)throw std::runtime_error("SpMV validation failed");
                    return err;
                };
                double first=events.time(stream,run);
                validate();
                for(int i=0;i<5;++i)run();
                CU(cudaStreamSynchronize(stream));
                double repeat=events.time(stream,[&](){
                    for(int i=0;i<reps;++i)run();
                })/reps;
                double err=validate();
                csv.row(prefix+",5,"+std::to_string(reps)+","+number(setup)+","+number(movement)+","+number(movement_host)+","+number(first)+","+number(repeat)+","+number(err)+",pass,ok");
                std::cout<<mode<<" move_ms="<<movement_host<<" first_ms="<<first<<" repeat_ms="<<repeat<<" pass\n";
                for(int count:{
                    1,10,100,1000
                }){
                    double total=0,gpu=0,moving=0;
                    for(int trial=0;trial<3;++trial){
                        // Restore CPU placement outside timing; every trial includes its own initial movement.
                        if(mode=="managed"){
                            CU(cudaMemcpy(row.host,m.row.data(),row.bytes,cudaMemcpyHostToHost));
                            CU(cudaMemcpy(col.host,m.col.data(),col.bytes,cudaMemcpyHostToHost));
                            CU(cudaMemcpy(val.host,m.val.data(),val.bytes,cudaMemcpyHostToHost));
#if CUDART_VERSION >= 13000
                            cudaMemLocation cpu{
                            };
                            cpu.type=cudaMemLocationTypeHost;
                            CU(cudaMemPrefetchAsync(row.device,row.bytes,cpu,0,stream));
                            CU(cudaMemPrefetchAsync(col.device,col.bytes,cpu,0,stream));
                            CU(cudaMemPrefetchAsync(val.device,val.bytes,cpu,0,stream));
#else
                            CU(cudaMemPrefetchAsync(row.device,row.bytes,cudaCpuDeviceId,stream));
                            CU(cudaMemPrefetchAsync(col.device,col.bytes,cudaCpuDeviceId,stream));
                            CU(cudaMemPrefetchAsync(val.device,val.bytes,cudaCpuDeviceId,stream));
#endif
                            CU(cudaStreamSynchronize(stream));
                        }
                        auto start=Clock::now();
                        auto mt=Clock::now();
                        events.time(stream,move);
                        moving+=elapsed(mt);
                        gpu+=events.time(stream,[&](){
                            for(int i=0;i<count;++i)run();
                        });
                        total+=elapsed(start);
                        validate();
                    }
                    amort.row(prefix+","+std::to_string(count)+",3,"+number(moving/3)+","+number(total/3)+","+number(gpu/3)+",pass,ok");
                }
            }
            catch(const std::exception&e){
                if(mode=="host" && std::string(e.what()).rfind("cuSPARSE:",0)==0){
                    CU(cudaGetLastError());
                    skip(std::string("CPU-resident cuSPARSE attempt failed: ")+e.what());
                }
                else throw;
            }
        }
    }
    return 0;
}
catch(const std::exception&e){
    std::cerr<<e.what()<<'\n';
    return 1;
}
