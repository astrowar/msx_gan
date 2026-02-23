  
 #include "msxgl.h"

 #define USE_W_l4_w
#define USE_W_out_w

#include "weigths_export.h"

#include "layer_i8.c"

void process_layer_5(const i8* in_buf,   i8* out_buf) __banked {
   conv3x3_i8(in_buf, G_C24, 24, 24, W_l4,  SCALEW_L4_Q15,  B_l4_f16 ,  G_CH,  out_buf, 1);
   saturation(out_buf, G_CH* 24*24);
}

void process_layer_6(const i8* in_buf,   i8* out_buf) __banked {
   conv3x3_i8(in_buf, G_CH, 24, 24, W_out, SCALEW_OUT_Q15, B_out_f16 , 1, out_buf, 0);
}