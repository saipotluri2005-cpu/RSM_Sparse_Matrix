#pragma once
#include <cstdint>
#include <zlib.h>
#include <sstream>
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <string>
struct Matrix {
    int rows,cols;
    std::vector<int> row,col;
    std::vector<double> val;
    size_t bytes()const{
        return row.size()*sizeof(int)+col.size()*sizeof(int)+val.size()*sizeof(double);
    }
};
// gzopen transparently reads plain files as well as gzip streams.
struct Lines {
    gzFile f;
    explicit Lines(const std::string&p):f(gzopen(p.c_str(),"rb")){
        if(!f)throw std::runtime_error("Cannot open "+p);
    }
    ~Lines(){
        gzclose(f);
    }
    bool get(std::string&s){
        s.clear();
        char buf[4096];
        while(gzgets(f,buf,sizeof(buf))){
            s+=buf;
            if(!s.empty()&&s.back()=='\n')return true;
        }
        int err;
        const char*msg=gzerror(f,&err);
        if(err!=Z_OK&&err!=Z_STREAM_END)throw std::runtime_error(msg);
        return !s.empty();
    }
    bool data(std::string&s){
        while(get(s)){
            auto p=s.find_first_not_of(" \t\r\n");
            if(p!=std::string::npos&&s[p]!='%')return true;
        }
        return false;
    }
};
inline Matrix load_matrix(const std::string&path){
    Lines in(path);
    std::string s,magic,obj,format,field,sym,extra;
    if(!in.get(s))throw std::runtime_error("Empty matrix");
    std::istringstream header(s);
    header>>magic>>obj>>format>>field>>sym;
    if(magic!="%%MatrixMarket"||obj!="matrix"||format!="coordinate"||(field!="real"&&field!="integer"&&field!="pattern")||(sym!="general"&&sym!="symmetric"))throw std::runtime_error("Supported: coordinate real/integer/pattern, general/symmetric only");
    long long r,c,n;
    if(!in.data(s))throw std::runtime_error("Missing dimensions");
    std::istringstream dims(s);
    if(!(dims>>r>>c>>n)||r<1||c<1||n<0||r>=INT32_MAX||c>INT32_MAX||n>INT32_MAX||(dims>>extra))throw std::runtime_error("Invalid or unsupported 32-bit dimensions");
    if(sym=="symmetric"&&r!=c)throw std::runtime_error("Symmetric matrix must be square");
    std::vector<std::tuple<int,int,double>> entries;
    for(long long k=0;k<n;++k){
        if(!in.data(s))throw std::runtime_error("Truncated matrix");
        std::istringstream line(s);
        long long i,j;
        double v=1;
        if(!(line>>i>>j)||(field!="pattern"&&!(line>>v))||(line>>extra)||i<1||i>r||j<1||j>c||!std::isfinite(v))throw std::runtime_error("Invalid coordinate");
        entries.emplace_back(i-1,j-1,v);
        if(sym=="symmetric"&&i!=j)entries.emplace_back(j-1,i-1,v);
        if(entries.size()>INT32_MAX)throw std::runtime_error("Expanded matrix exceeds 32-bit indices");
    }
    if(in.data(s))throw std::runtime_error("Extra coordinate data");
    std::sort(entries.begin(),entries.end(),[](auto&a,auto&b){
        return std::tie(std::get<0>(a),std::get<1>(a))<std::tie(std::get<0>(b),std::get<1>(b));
    });
    Matrix m{
        int(r),int(c),std::vector<int>(r+1,0),{
        }, {
        }
    };
    for(size_t k=0;k<entries.size();){
        int i=std::get<0>(entries[k]),j=std::get<1>(entries[k]);
        double v=0;
        do{
            v+=std::get<2>(entries[k++]);
        }
        while(k<entries.size()&&std::get<0>(entries[k])==i&&std::get<1>(entries[k])==j);
        if(!std::isfinite(v))throw std::runtime_error("Duplicate sum overflow");
        m.col.push_back(j);
        m.val.push_back(v);
        ++m.row[i+1];
    }
    for(int i=0;i<m.rows;++i)m.row[i+1]+=m.row[i];
    return m;
}
