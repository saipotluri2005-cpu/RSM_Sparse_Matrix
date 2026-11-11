#include "matrix.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <unistd.h>
void require(bool ok){
    if(!ok)throw std::runtime_error("reader test failed");
}
int main(int argc,char**argv)try{
    std::string path="/tmp/vista_reader_"+std::to_string(getpid())+".mtx";
    auto write=[&](std::string s){
        std::ofstream f(path);
        f<<s;
    };
    write("%%MatrixMarket matrix coordinate real symmetric\n% test\n3 3 4\n1 1 2\n1 2 3\n1 2 -1\n3 3 5\n");
    auto m=load_matrix(path);
    require(m.row==std::vector<int>({
        0,2,3,4
    }));
    require(m.col==std::vector<int>({
        0,1,0,2
    }));
    require(m.val==std::vector<double>({
        2,2,2,5
    }));
    write("%%MatrixMarket matrix coordinate pattern general\n2 3 1\n2 3\n");
    m=load_matrix(path);
    require(m.val==std::vector<double>({
        1
    }));
    require(m.row==std::vector<int>({
        0,0,1
    }));
    for(std::string bad:{
        "%%MatrixMarket matrix array real general\n2 2\n", "%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1 0\n", "%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 2\n", "%%MatrixMarket matrix coordinate real general\n2 2 1\n", "%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n"
    }){
        write(bad);
        bool rejected=false;
        try{
            load_matrix(path);
        }
        catch(const std::exception&){
            rejected=true;
        }
        require(rejected);
    }
    std::remove(path.c_str());
    for(int i=1;i<argc;++i){
        m=load_matrix(argv[i]);
        require(m.row.back()==int(m.val.size()));
        std::cout<<argv[i]<<" "<<m.rows<<"x"<<m.cols<<" nnz="<<m.val.size()<<'\n';
    }
    std::cout<<"Reader tests passed\n";
}
catch(const std::exception&e){
    std::cerr<<e.what()<<'\n';
    return 1;
}
