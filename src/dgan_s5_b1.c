 
#include "msxgl.h"

#define USE_W_l2_w
#include "weigths_export.h"
    
#include "layer_i8.c"

void process_layer_3(const i8* in_buf, i8* tmp_buf , i8* out_buf) __banked {
    conv3x3_i8(in_buf, G_C6, 12, 12, W_l2,  SCALEW_L2_Q15,  B_l2_f16 ,  G_C12, tmp_buf,  1);
    saturation(tmp_buf, G_C12 * 12 * 12);
    // up 12->24
    upsample_nearest_i8(tmp_buf, G_C12, 12, 12, out_buf, 24, 24);
}
