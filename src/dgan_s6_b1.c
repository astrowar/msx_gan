  
#include "msxgl.h"

#define USE_W_l3_w
#include "weigths_export.h"


 
#include "layer_i8.c"

void process_layer_4(const i8* in_buf,   i8* out_buf) __banked {
   conv3x3_i8(in_buf, G_C12, 24, 24, W_l3,  SCALEW_L3_Q15,  B_l3_f16 ,  G_C24, out_buf, 1);
   saturation(out_buf, G_C24 * 24 * 24);
}
