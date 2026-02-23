#include "msxgl.h"

#define USE_W_l1_w
#include "weigths_export.h"

#include "layer_i8.c"

void process_layer_2(  const i8* in_buf, i8* tmp_buf , i8* out_buf) __banked {
     conv3x3_i8(in_buf,  G_C2,  6,  6, W_l1,  SCALEW_L1_Q15,  B_l1_f16 ,  G_C6,  tmp_buf,   1);
     saturation(tmp_buf, G_C6 * 6 * 6);
     // up 6->12
    upsample_nearest_i8(tmp_buf, G_C6,  6,  6, out_buf, 12, 12);
 }  
 