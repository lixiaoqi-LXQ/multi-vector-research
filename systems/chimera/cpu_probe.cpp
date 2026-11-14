#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
int main() {float x[128]{}; rabitqlib::rotator_impl::FhtKacRotator r(128,128,42); r.rotate(x,x);}
