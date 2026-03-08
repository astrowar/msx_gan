#include "msxgl.h" 

typedef i16 fx16_t;
typedef i16 q15_t;  // Q0.15
 
static const i8 TANH_LUT_I8[256] = {
    -62, -62, -62, -61, -61, -61, -61, -61, -61, -61, -61, -61, -61, -61, -60, -60,
    -60, -60, -60, -60, -60, -60, -60, -59, -59, -59, -59, -59, -59, -58, -58, -58,
    -58, -58, -58, -57, -57, -57, -57, -57, -56, -56, -56, -56, -55, -55, -55, -55,
    -54, -54, -54, -53, -53, -53, -52, -52, -52, -51, -51, -51, -50, -50, -50, -49,
    -49, -48, -48, -47, -47, -47, -46, -46, -45, -45, -44, -43, -43, -42, -42, -41,
    -41, -40, -39, -39, -38, -38, -37, -36, -35, -35, -34, -33, -33, -32, -31, -30,
    -30, -29, -28, -27, -26, -26, -25, -24, -23, -22, -21, -20, -19, -18, -18, -17,
    -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27, 28, 29,
    30, 30, 31, 32, 33, 33, 34, 35, 35, 36, 37, 38, 38, 39, 39, 40,
    41, 41, 42, 42, 43, 43, 44, 45, 45, 46, 46, 47, 47, 47, 48, 48,
    49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54,
    54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58,
    58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62
};

static  inline void   saturation(i8* x , i16 N) {

    for (i16 i = 0; i < N; ++i) {
        u8 idx = (u8)(x[i] + 128); // shift to [0,255]
        x[i] = TANH_LUT_I8[idx];
    }
}

static inline i8 clamp_i8_s(i32 v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return (i8)v;
}


 
 

 


// Q8.8 acumulado (int32) -> int8 com rounding
static inline i8 acc_q88_to_i8_sat(i32 acc_q88) {
//safe saturation 
  if (acc_q88 > 8388607)  return 127;
  if (acc_q88 < -8388608) return -128; 

  // arredondar para inteiro
  if (acc_q88 >= 0) acc_q88 += 128;
  else              acc_q88 -= 128;

  i32 vi = acc_q88 >> 8;
  return clamp_i8_s(vi);
}

 
 
  
 


static void upsample_nearest_i8(
    const i8* in, u8 C, u8 Hin, u8 Win,
    i8* out, u8 Hout, u8 Wout)
{
  for (u8 c = 0; c < C; ++c) {
    const i8* pin = in + c * Hin * Win;
    i8* pout = out + c * Hout * Wout;

    for (u8 y = 0; y < Hout; ++y) {
      u8 sy = (y * Hin) / Hout;
      for (u8 x = 0; x < Wout; ++x) {
        u8 sx = (x * Win) / Wout;
        pout[y * Wout + x] = pin[sy * Win + sx];
      }
    }
  }
}
typedef union {
    u16 full;
    struct { u8 lo; u8 hi; } b;   // a: bytes unsigned
} u16_bytes;

typedef union {
    i16 full;
    struct { u8 lo; i8 hi; } b;   // b: lo unsigned, hi signed
} i16_bytes;

static inline i32 mul_u16_i16_to_i32(u16 a, i16 b)
{
    const u16_bytes a16 = { .full = a };
    const u8 a0 = a16.b.lo;
    const u8 a1 = a16.b.hi;

    const i16_bytes b16 = { .full = b };
    const u8 b0u = b16.b.lo;
    const i8 b1  = b16.b.hi;

    // produto = a0*b0 + ((a0*b1 + a1*b0)<<8) + (a1*b1<<16)
    i32 p  = (i32)a0 * (i32)b0u;
    p     += (((i32)a0 * (i32)b1) + ((i32)a1 * (i32)b0u)) << 8;
    p     += ((i32)a1 * (i32)b1) << 16;

    return p;
}

 

typedef union {
    i32 full;
    struct { u8 b0,b1,b2,b3; } b; // ordem em memória
} i32_bytes;
 

  static inline u16 make_u16(u8 lo, u8 hi) {
    u16_bytes x;
    x.b.lo = lo;
    x.b.hi = hi;
    return x.full;
}

 
static inline i16 make_i16(u8 lo, u8 hi) { return (i16)make_u16(lo, hi); }

static inline i32 mul_i32_q15_to_q88(i32 sum, q15_t scaleW_q15)
{
    i32_bytes S; S.full = sum;

    // LITTLE ENDIAN: b0=LSB ... b3=MSB
    const u16 lo = make_u16(S.b.b0, S.b.b1);
    const i16 hi = make_i16(S.b.b2, S.b.b3);

    i16 s = (i16)scaleW_q15;

    i32 prod = ((i32)hi * (i32)s) << 16;
    prod += mul_u16_i16_to_i32(lo, s);

    prod += (prod >= 0) ? 64 : -64;
    return (prod >> 7);
}

// ------------------------------------------------------------
// Conv3x3 int8 -> int8 (internamente acumula em Q8.8)
// Entrada/saída int8 representam inteiros "puros" (equiv. Q8.8 com frac=0)
// ------------------------------------------------------------


//  static void conv3x3_i8_nounroll(
//     const i8* in_i8, u8 Cin, u8 H, u8 W,
//     const i8* wq, q15_t scaleW_q15,
//     const fx16_t* b,      // pode ser NULL (bias já em Q8.8)
//     int Cout,
//     i8* out_i8,
//     int apply_relu)
// {
//   const int HW = H * W;
//   const int W_OC_STRIDE = Cin * 9;

//   for (u8 oc = 0; oc < Cout; ++oc) {
//     const i32 bias_q88 = b ? (i32)b[oc] : 0;

//     const i8* w_oc = wq + oc * W_OC_STRIDE;
//     i8* out_oc = out_i8 + oc * HW;

//     for (u8 y = 0; y < H; ++y) {
//       i8* out_row = out_oc + y * W;

//       for (u8 x = 0; x < W; ++x) {

//         // acumula só a*w (sem scale) em inteiro
//         i32 sum_aw = 0;

//         for (u8 ic = 0; ic < Cin; ++ic) {
//           const i8* pin = in_i8 + ic * HW;
//           const i8* pw  = w_oc  + ic * 9;

//           for (i8 ky = -1; ky <= 1; ++ky) {

//             u8 yy =  y +  ky;
//             if ( yy >= H) continue; 

//             const int yy_mul_w = yy * W;
//             const int kbase = (ky + 1) * 3;

//             for (i8 kx = -1; kx <= 1; ++kx) {
//               u8 xx = x + kx;  
//               if (xx >= W) continue;


//               const i8 a  = pin[yy_mul_w + xx];
//               const i8 wi = pw[kbase + (kx + 1)];

//               sum_aw += (i16)a * (i16)wi;              
//             }
//           }
//         }

//         // aplica scale UMA vez por pixel -> Q8.8
//         i32 acc = bias_q88 + mul_i32_q15_to_q88(sum_aw, scaleW_q15);

//         // ReLU em Q8.8
//         if (apply_relu && acc < 0) acc = 0;

//         out_row[x] = acc_q88_to_i8_sat(acc);
//       }
//     }
//   }
// }

static void conv3x3_i8(
    const i8* in_i8, u8 Cin, u8 H, u8 W,
    const i8* wq, q15_t scaleW_q15,
    const fx16_t* b,      // pode ser NULL (bias já em Q8.8)
    u8 Cout,
    i8* out_i8,
    u8 apply_relu)
{
    const u16 HW = (u16)H * (u16)W;
    const u16 W_OC_STRIDE = (u16)Cin * 9;

    const i8* w_oc = wq;
    i8* out_oc = out_i8;

    for (u8 oc = 0; oc < Cout; ++oc) {
        const i32 bias_q88 = b ? (i32)b[oc] : 0;

        for (u8 y = 0; y < H; ++y) {
            i8* out_row = out_oc + (u16)y * W;

            for (u8 x = 0; x < W; ++x) {
                i32 sum_aw = 0;

                const i8* pin = in_i8;
                const i8* pw  = w_oc;

                for (u8 ic = 0; ic < Cin; ++ic) {
                    // ky = -1
                    {
                        u8 yy = y - 1;
                        if (yy < H) {
                            const i8* row = pin + (u16)yy * W;

                            u8 xx = x - 1;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[0];

                            xx = x;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[1];

                            xx = x + 1;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[2];
                        }
                    }

                    // ky = 0
                    {
                        u8 yy = y;
                        const i8* row = pin + (u16)yy * W;

                        u8 xx = x - 1;
                        if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[3];

                        xx = x;
                        sum_aw += (i16)row[xx] * (i16)pw[4];

                        xx = x + 1;
                        if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[5];
                    }

                    // ky = +1
                    {
                        u8 yy = y + 1;
                        if (yy < H) {
                            const i8* row = pin + (u16)yy * W;

                            u8 xx = x - 1;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[6];

                            xx = x;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[7];

                            xx = x + 1;
                            if (xx < W) sum_aw += (i16)row[xx] * (i16)pw[8];
                        }
                    }

                    pin += HW;
                    pw  += 9;
                }

                {
                    i32 acc = bias_q88 + mul_i32_q15_to_q88(sum_aw, scaleW_q15);
                    if (apply_relu && acc < 0)
                        acc = 0;
                    out_row[x] = acc_q88_to_i8_sat(acc);
                }
            }
        }

        w_oc  += W_OC_STRIDE;
        out_oc += HW;
    }
}
