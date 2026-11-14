// Host-only executable: no CUDA API calls or kernel launches.
#define EIGEN_NO_CUDA
#include "chimera/config.cuh"
#include "chimera/fp16_input.hpp"
#include "chimera/index_file_io.hpp"
#include "chimera/spilled_full_codes.hpp"
#include "chimera/quantization.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "mapped_fp16_npy.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <omp.h>
namespace fs=std::filesystem;
struct Header { size_t n,d,clusters,ex_bits,padded; };
static_assert(sizeof(Header)==40,"This tool requires the existing 64-bit native index ABI");
void require(bool value,const char*msg){if(!value)throw std::runtime_error(msg);}
size_t multiply(size_t a,size_t b){require(b==0||a<=std::numeric_limits<size_t>::max()/b,"Size overflow");return a*b;}
int main(int argc,char**argv) {try {
 require(argc==4,"Usage: requantize SOURCE_INDEX FP16_NPY EMPTY_STAGING_DIRECTORY");
 const fs::path src=fs::canonical(argv[1]),out=fs::canonical(argv[3]);
 require(fs::is_directory(out)&&fs::is_empty(out),"Staging directory must exist and be empty");
 std::ifstream input(src/"doc_1bit.bin",std::ios::binary);Header h{};input.read(reinterpret_cast<char*>(&h),sizeof h);
 require(bool(input)&&h.n>0&&h.d==128&&h.padded==128&&h.ex_bits==3&&h.clusters>0&&h.clusters<=h.n,"Expected valid 128-dimensional 4-bit index");
 require(h.n<=size_t(std::numeric_limits<int64_t>::max()),"Rows exceed loop range");
 const size_t rowbytes=PADDED_DIM/8,full4stride=PADDED_DIM*4/8,full8stride=PADDED_DIM;
 auto rotator=std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(h.d,rabitqlib::RotatorType::FhtKacRotator,h.padded));
 rotator->load(input);require(bool(input),"Truncated source rotator");
 const auto payload_offset=static_cast<size_t>(input.tellg());
 const size_t fixedbytes=payload_offset+multiply(h.n,rowbytes+sizeof(float));
 const size_t docbytes=fixedbytes+multiply(h.n,sizeof(float));
 Chimera::ReadOnlyMappedFile docmap((src/"doc_1bit.bin").string(),docbytes);
 Chimera::ReadOnlyMappedFile full4((src/"doc_full.bin").string(),multiply(h.n,full4stride));
 std::ifstream lensfile(src/"doclens.bin",std::ios::binary);int ndocs=0;lensfile.read(reinterpret_cast<char*>(&ndocs),sizeof ndocs);
 require(bool(lensfile)&&ndocs>0,"Invalid document count");size_t tokens=0;
 for(int i=0;i<ndocs;++i){int len=0;lensfile.read(reinterpret_cast<char*>(&len),sizeof len);require(bool(lensfile)&&len>0&&size_t(len)<=h.n-tokens,"Invalid document lengths");tokens+=size_t(len);}
 require(tokens==h.n&&lensfile.peek()==EOF,"Document lengths do not exactly cover token rows");
 MappedFp16Npy raw(argv[2],h.n);require(raw.dimension()==h.d,"Raw FP16 dimension mismatch");
 Chimera::SpilledFullCodes destdoc((out/".doc_1bit.spill").string(),docbytes);
 Chimera::SpilledFullCodes destfull((out/".doc_full.spill").string(),multiply(h.n,full8stride));
 constexpr size_t chunk=64ull*1024*1024;
 for(size_t begin=0;begin<fixedbytes;){size_t count=std::min(chunk,fixedbytes-begin);std::memcpy(destdoc.data()+begin,docmap.data()+begin,count);destdoc.sync_and_drop(begin,count);begin+=count;}
 Header newh=h;newh.ex_bits=7;std::memcpy(destdoc.data(),&newh,sizeof newh);destdoc.sync_and_drop(0,sizeof newh);
 const char* source_codes=docmap.data()+payload_offset;
 const char* source_onefactors=source_codes+multiply(h.n,rowbytes);
 const char* source_fullfactors=source_onefactors+multiply(h.n,sizeof(float));
 std::atomic<size_t> first_bad(h.n);
 const size_t rows=destfull.rows_per_flush(full8stride);
 for(size_t begin=0;begin<h.n;){const size_t end=begin+std::min(rows,h.n-begin);
#pragma omp parallel
  {
   std::array<float,PADDED_DIM> expanded{},rotated{};std::array<uint64_t,PADDED_DIM/64> codes{};std::array<uint8_t,PADDED_DIM/2> codes4{};
#pragma omp for schedule(static)
   for(int64_t row=static_cast<int64_t>(begin);row<static_cast<int64_t>(end);++row){
    if(first_bad.load(std::memory_order_relaxed)!=h.n) continue;
    const size_t i=static_cast<size_t>(row);float onefactor=0,oldfactor=0,newfactor=0;
    bool finite=Chimera::expand_fp16_to_fp32(raw.data()+i*h.d,expanded.data(),h.d);
    if(finite){rotator->rotate(expanded.data(),rotated.data());Chimera::encode_one_bit(rotated.data(),PADDED_DIM,codes.data(),&onefactor);Chimera::encode_full_code(rotated.data(),PADDED_DIM,3,codes4.data(),&oldfactor);}
    if(!finite||std::memcmp(codes.data(),source_codes+i*rowbytes,rowbytes)||std::memcmp(&onefactor,source_onefactors+i*4,4)||std::memcmp(codes4.data(),full4.data()+i*full4stride,full4stride)||std::memcmp(&oldfactor,source_fullfactors+i*4,4)){
     size_t previous=first_bad.load();while(i<previous&&!first_bad.compare_exchange_weak(previous,i)){}continue;
    }
    Chimera::encode_full_code(rotated.data(),PADDED_DIM,7,reinterpret_cast<uint8_t*>(destfull.data()+i*full8stride),&newfactor);
    std::memcpy(destdoc.data()+fixedbytes+i*4,&newfactor,4);
   }
  }
  require(first_bad.load()==h.n,("Source 4-bit invariant mismatch at row "+std::to_string(first_bad.load())).c_str());
  destfull.sync_and_drop(begin*full8stride,(end-begin)*full8stride);
  destdoc.sync_and_drop(fixedbytes+begin*4,(end-begin)*4);
  begin=end;
 }
 // Source payloads remain untouched; staging is published only by verified wrapper.
 for(const char*name:{"doclens.bin","ivf.bin","cluster_1bit.bin","centroids.carga"})fs::create_hard_link(src/name,out/name);
 destdoc.publish_hardlink((out/"doc_1bit.bin").string());destfull.publish_hardlink((out/"doc_full.bin").string());
 std::cout<<"{\"tokens\":"<<h.n<<",\"dimension\":"<<h.d<<",\"source_bits\":4,\"destination_bits\":8,\"all_rows_bitwise_source_verified\":true,\"gpu_used\":false}"<<std::endl;
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
