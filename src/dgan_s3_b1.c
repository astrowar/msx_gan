 
#include "msxgl.h"

#define USE_W_l0_w
 #include "weigths_export.h"

 

#include "layer_i8.c"

void process_layer_1(const i8* in_buf, i8* tmp_buf , i8* out_buf) __banked {
     conv3x3_i8(in_buf,  G_NW,  2,  2, W_l0,  SCALEW_L0_Q15,  B_l0_f16 ,  G_C2,  tmp_buf,   1);
     saturation(tmp_buf, G_C2 * 2 * 2);
  // up 2->6
    upsample_nearest_i8(tmp_buf, G_C2,  2,  2, out_buf,  6,  6);
}
