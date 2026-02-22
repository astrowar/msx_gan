#define USE_W_l0_w
#define USE_W_l1_w
#define USE_W_l2_w
#define USE_W_l3_w
#define USE_W_l4_w
#define USE_W_out_w

#include "gen_weights.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>


typedef int16_t fx16_t;

// Q8.8
#ifndef FX_FRAC_BITS
#define FX_FRAC_BITS 8
#endif
#define FX_SCALE (1 << FX_FRAC_BITS)

static inline fx16_t relu_fixed(fx16_t x) { return (x < 0) ? (fx16_t)0 : x; }

static inline fx16_t fx16_sat_i32(int32_t x) {
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return (fx16_t)x;
}

// float -> fixed(Q8.8)
static inline fx16_t fx16_from_float(float f) {
  int ipart =  (int)f;
  int frac = (f - (float)ipart) * 256.0f; // parte fracionária em 8 bits
  int32_t v = (ipart << FX_FRAC_BITS) +frac;
  return fx16_sat_i32(v);
}

// fixed(Q8.8) -> float
static inline float fx16_to_float(fx16_t x) {
  return (float)x * (1.0f / (float)FX_SCALE);
}

// fixed(Q8.8) * fixed(Q8.8) -> float  (evita erro de escala)
static inline float fx16_mul_to_float(fx16_t a, fx16_t b) {
  // a*b está em Q(2*FRAC). Convertendo direto para float:
  // (a*b) / (2^(2*FRAC))
  return (float)((int32_t)a * (int32_t)b) * (1.0f / (float)(FX_SCALE * FX_SCALE));
}



static inline fx16_t fx16_mul_16_16(fx16_t a, fx16_t b) {
    int16_t ah = (int16_t)(a >> 8);     // signed
    int16_t bh = (int16_t)(b >> 8);     // signed
    uint16_t al = (uint8_t)(a & 0xFF);  // unsigned
    uint16_t bl = (uint8_t)(b & 0xFF);  // unsigned

    int32_t p =
        ((int32_t)ah * (int32_t)bh << 8) +   // ah*bh * 256
        ((int32_t)ah * (int32_t)bl) +        // ah*bl
        ((int32_t)bh * (int32_t)al) +        // bh*al
        (((int32_t)al * (int32_t)bl + 128) >> 8); // (al*bl)/256 com rounding

    if (p >  32767) p =  32767;
    if (p < -32768) p = -32768;
    return (fx16_t)p;
}



typedef int16_t q15_t;  // Q0.15




// float -> Q0.15 (para scaleW)
static inline q15_t q15_from_float(float x) {
  int32_t q = (int32_t)(x * 32768.0f + (x >= 0.0f ? 0.5f : -0.5f));
  if (q >  32767) q =  32767;
  if (q < -32768) q = -32768;
  return (q15_t)q;
}

// Arredondamento simétrico para shift à direita por 7
static inline int16_t rshift7_round_i16(int16_t x) {
  if (x >= 0) return (int16_t)((x + 64) >> 7);
  return (int16_t)(-(((-x) + 64) >> 7));
}

// wi * scale_q15 -> Q8.8 (caminho rápido: só byte low)
static inline fx16_t w_i8_scale_q15_to_q88_lo(int8_t wi, uint8_t slo) {
  int16_t t = (int16_t)wi * (int16_t)slo;
  return (fx16_t)rshift7_round_i16(t);
}

// wi * scale_q15 -> Q8.8 (geral, sem int32)
static inline fx16_t w_i8_scale_q15_to_q88_hi_lo(int8_t wi, uint8_t shi, uint8_t slo) {
  int16_t t_hi = (int16_t)wi * (int16_t)shi;
  int16_t t_lo = (int16_t)wi * (int16_t)slo;
  int16_t q = (int16_t)(t_hi << 1);               // 2*(wi*shi)
  q = (int16_t)(q + rshift7_round_i16(t_lo));     // + round((wi*slo)/128)
  return (fx16_t)q;
}


static inline fx16_t fx16_from_acc_q88_i32(int32_t acc_q88) {
  if (acc_q88 >  32767) acc_q88 =  32767;
  if (acc_q88 < -32768) acc_q88 = -32768;
  return (fx16_t)acc_q88;
}

static inline int32_t div8_toward_zero_i32(int32_t x) {
  if (x >= 0) return x >> 3;
  return -(((-x) >> 3));
}


// ------------------------------------------------------------
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

static inline float leakrelu_f(float x) { return x > 0.0f ? x :  x/8.0f; }






static inline int16_t clamp_i16(int32_t x) {
  if (x < -32768) return -32768;
  if (x >  32767) return  32767;
  return (int16_t)x;
}

    static inline uint8_t clamp_i8(int32_t x) {
  if (x < 0) return 0;
  if (x >  255) return  255;
  return (uint8_t)x;
}


// Tensor layout: CHW contiguous: idx = (c*H + y)*W + x

// ------------------------------------------------------------
// Nearest upsample float (replicação)
// ------------------------------------------------------------
static void upsample_nearest_f32(const float* in, int C, int Hin, int Win,
                                 float* out, int Hout, int Wout)
{
  int ry = Hout / Hin;
  int rx = Wout / Win;

  for (int c = 0; c < C; c++) {
    const float* pin = in + c * Hin * Win;
    float* pout = out + c * Hout * Wout;

    for (int y = 0; y < Hin; y++) {
      for (int x = 0; x < Win; x++) {
        float v = pin[y * Win + x];
        int oy0 = y * ry;
        int ox0 = x * rx;
        for (int yy = 0; yy < ry; yy++) {
          float* row = pout + (oy0 + yy) * Wout + ox0;
          for (int xx = 0; xx < rx; xx++) row[xx] = v;
        }
      }
    }
  }
}


// ------------------------------------------------------------
// Nearest upsample fixed (replicação)
// ------------------------------------------------------------
static void upsample_nearest_fixed(const fx16_t* in, int C, int Hin, int Win,
                                 fx16_t* out, int Hout, int Wout)
{
  int ry = Hout / Hin;
  int rx = Wout / Win;

  for (int c = 0; c < C; c++) {
    const fx16_t* pin = in + c * Hin * Win;
    fx16_t* pout = out + c * Hout * Wout;

    for (int y = 0; y < Hin; y++) {
      for (int x = 0; x < Win; x++) {
        fx16_t v = pin[y * Win + x];
        int oy0 = y * ry;
        int ox0 = x * rx;
        for (int yy = 0; yy < ry; yy++) {
          fx16_t* row = pout + (oy0 + yy) * Wout + ox0;
          for (int xx = 0; xx < rx; xx++) row[xx] = v;
        }
      }
    }
  }
}

// ------------------------------------------------------------
// Conv3x3 float, padding=1, stride=1
// Pesos: int8 + scaleW  => w_float = (float)wq * scaleW
// Bias: float (original do PyTorch)
// apply_relu: 1 aplica ReLU, 0 não
// ------------------------------------------------------------
static void conv3x3_f32(
    const float* in, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,            // bias float (pode ser NULL)
    int Cout,
    float* out,
    int apply_relu)
{
  // Inputs are 8.8 fixed-point (fixed), weights are for float, so apply /255 correction
  // Output remains in 8.8 fixed-point
  for (int oc = 0; oc < Cout; oc++) {
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        float acc = b ? b[oc] : 0.0f;

        for (int ic = 0; ic < Cin; ic++) {
          const float* pin = in + ic * H * W;
          const int8_t* pw = wq + (((oc * Cin + ic) * 3) * 3);

          for (int ky = -1; ky <= 1; ky++) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;
            for (int kx = -1; kx <= 1; kx++) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              float a = pin[yy * W + xx];
              float w = (float)pw[(ky + 1) * 3 + (kx + 1)] * scaleW;
              acc += a * w;
            }
          }
        }

        if (apply_relu) acc = relu_f(acc);
        out[(oc * H + y) * W + x] = acc;
      }
    }
  }
}






// Conv3x3: in/out Q8.8, pesos int8 + scaleW, bias float opcional
static void conv3x3_fxio_v1(
    const fx16_t* in_q88, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,      // pode ser NULL
    int Cout,
    fx16_t* out_q88,
    int apply_relu)
{
  const int HW = H * W;

  for (int oc = 0; oc < Cout; ++oc) {
    const float bias = b ? b[oc] : 0.0f;

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {

        float acc = bias;

        for (int ic = 0; ic < Cin; ++ic) {
          const fx16_t* pin = in_q88 + ic * HW;
          const int8_t* pw  = wq + (oc * Cin + ic) * 9; // 3*3

          for (int ky = -1; ky <= 1; ++ky) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            for (int kx = -1; kx <= 1; ++kx) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              float a = fx16_to_float(pin[yy * W + xx]);              // Q8.8 -> float
              float w = (float)pw[(ky + 1) * 3 + (kx + 1)] * scaleW;    // int8 -> float
              acc += a * w;
            }
          }
        }

        if (apply_relu && acc < 0.0f) acc = 0.0f;

        out_q88[oc * HW + y * W + x] = fx16_from_float(acc); // float -> Q8.8
      }
    }
  }
}



// ------------------------------------------------------------
// Conv3x3 fixed, padding=1, stride=1
// Pesos: int8 + scaleW  => w_float = (float)wq * scaleW
// Bias: float (original do PyTorch)
// apply_relu: 1 aplica ReLU, 0 não
// ------------------------------------------------------------
static void conv3x3_fxio_v2(
    const fx16_t* in_q88, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,      // pode ser NULL
    int Cout,
    fx16_t* out_q88,
    int apply_relu)
{
  const int HW = H * W;

  // pré-cálculo por chamada
  const q15_t scaleW_q15 = q15_from_float(scaleW);
  const uint16_t su = (uint16_t)scaleW_q15; // scale positivo
  const uint8_t slo = (uint8_t)(su & 0xFF);
  const uint8_t shi = (uint8_t)(su >> 8);
  const uint8_t use_lo_only = (shi == 0);

  for (int oc = 0; oc < Cout; ++oc) {
    const float bias = b ? b[oc] : 0.0f;

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {

        float acc = bias;

        for (int ic = 0; ic < Cin; ++ic) {
          const fx16_t* pin = in_q88 + ic * HW;
          const int8_t* pw  = wq + (oc * Cin + ic) * 9; // 3x3

          for (int ky = -1; ky <= 1; ++ky) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            for (int kx = -1; kx <= 1; ++kx) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              fx16_t a_q = pin[yy * W + xx];
              int8_t wi = pw[(ky + 1) * 3 + (kx + 1)];

              fx16_t w_q;
              if (use_lo_only) {
                w_q = w_i8_scale_q15_to_q88_lo(wi, slo);
              } else {
                w_q = w_i8_scale_q15_to_q88_hi_lo(wi, shi, slo);
              }

              fx16_t prod_q = fx16_mul_16_16(a_q, w_q);
              acc += fx16_to_float(prod_q);
            }
          }
        }

        if (apply_relu && acc < 0.0f)acc = acc / 8.0f;

        fx16_t acc_q = fx16_from_float(acc);

        acc_q = acc_q & 0xff00; // opcional: limpa parte fracionária para reduzir ruído de quantização
        out_q88[oc * HW + y * W + x] = acc_q;

      }
    }
  }
}


static void conv3x3_fxio_v3(
    const fx16_t* in, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,              // bias float (pode ser NULL)
    int Cout,
    fx16_t* out,
    int apply_relu)
{
  // scaleW em fixed(Q8.8) uma vez
  const fx16_t scaleW_fx = fx16_from_float(scaleW);

  for (int oc = 0; oc < Cout; oc++) {

    // bias -> fixed(Q8.8) uma vez por oc (se existir)
    const int32_t bias_fx = b ? (int32_t)fx16_from_float(b[oc]) : 0;

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {

        // acc em fixed(Q8.8), mas em int32 para não estourar fácil
        int32_t acc_fx = bias_fx;

        for (int ic = 0; ic < Cin; ic++) {
          const fx16_t* pin = in + ic * H * W;
          const int8_t* pw  = wq + (((oc * Cin + ic) * 3) * 3);

          for (int ky = -1; ky <= 1; ky++) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            for (int kx = -1; kx <= 1; kx++) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              // a em fixed(Q8.8)
              fx16_t a_fx = pin[yy * W + xx];

              // w em fixed(Q8.8) usando scaleW_fx:
              // w_fx = int8 * scaleW_fx  (resultado Q8.8 em int32)
              int32_t w_fx32 = (int32_t)pw[(ky + 1) * 3 + (kx + 1)] * (int32_t)scaleW_fx;
              fx16_t  w_fx   = fx16_sat_i32(w_fx32);

              // produto: Q8.8 * Q8.8 = Q16.16 (int32)
              int32_t p = (int32_t)a_fx * (int32_t)w_fx;

              // arredondamento e volta para Q8.8
              if (p >= 0) p += (1 << (FX_FRAC_BITS - 1));
              else        p -= (1 << (FX_FRAC_BITS - 1));
              p >>= FX_FRAC_BITS; // agora p está em Q8.8

              acc_fx += p;
            }
          }
        }

        // saturar para fx16_t
        fx16_t q16 = fx16_sat_i32(acc_fx);

        if (apply_relu) q16 = relu_fixed(q16);

        out[(oc * H + y) * W + x] = q16;
      }
    }
  }
}

static void conv3x3_fxio_v4(
    const fx16_t* in_q88, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,      // pode ser NULL
    int Cout,
    fx16_t* out_q88,
    int apply_relu)
{
  const int HW = H * W;

  // pré-cálculo por chamada
  const q15_t scaleW_q15 = q15_from_float(scaleW);
  const uint16_t su = (uint16_t)scaleW_q15; // scale positivo
  const uint8_t slo = (uint8_t)(su & 0xFF);
  const uint8_t shi = (uint8_t)(su >> 8);
  const uint8_t use_lo_only = (shi == 0);

  for (int oc = 0; oc < Cout; ++oc) {
    // bias em Q8.8 (1x por canal)
    const fx16_t bias_q88 = b ? fx16_from_float(b[oc]) : (fx16_t)0;

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {

        int32_t acc = (int32_t)bias_q88; // acumulador em Q8.8

        for (int ic = 0; ic < Cin; ++ic) {
          const fx16_t* pin = in_q88 + ic * HW;
          const int8_t* pw  = wq + (oc * Cin + ic) * 9; // 3x3

          for (int ky = -1; ky <= 1; ++ky) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            for (int kx = -1; kx <= 1; ++kx) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              fx16_t a_q = pin[yy * W + xx];
              int8_t wi  = pw[(ky + 1) * 3 + (kx + 1)];

              fx16_t w_q;
              if (use_lo_only) {
                w_q = w_i8_scale_q15_to_q88_lo(wi, slo);
              } else {
                w_q = w_i8_scale_q15_to_q88_hi_lo(wi, shi, slo);
              }

              fx16_t prod_q = fx16_mul_16_16(a_q, w_q); // Q8.8
              acc += (int32_t)prod_q;
            }
          }
        }

        // LeakyReLU(1/8)
        if (apply_relu && acc < 0) {
          acc = 0;
        }

        fx16_t acc_q = fx16_from_acc_q88_i32(acc);

        // opcional: zerar fração (reduz ruído)
        // acc_q = (fx16_t)((acc_q >> 8) << 8);

        out_q88[oc * HW + y * W + x] = acc_q;
      }
    }
  }
}




// ---------- Helpers esperados (os mesmos que você já tem) ----------
// q15_from_float
// fx16_from_float
// w_i8_scale_q15_to_q88_lo
// w_i8_scale_q15_to_q88_hi_lo
// fx16_mul_16_16
// div8_toward_zero_i32

static inline int8_t clamp_i8_s(int32_t v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return (int8_t)v;
}

static inline uint8_t clamp_u8_s(int32_t v) {
  if (v > 255) return 255;
  if (v < 0) return 0;
  return (uint8_t)v;
}


// Q8.8 acumulado (int32) -> int8 com rounding
static inline int8_t acc_q88_to_i8_sat(int32_t acc_q88) {
  // arredondar para inteiro
  if (acc_q88 >= 0) acc_q88 += 128;
  else              acc_q88 -= 128;

  int32_t vi = acc_q88 >> 8;
  return clamp_i8_s(vi);
}

// bias float -> int32 Q8.8 (1x por canal)
static inline int32_t bias_float_to_q88_i32(float x) {
  // Se quiser evitar float aqui também, pré-converta no export.
  float s = x * 256.0f;
  int32_t q = (int32_t)(s + (s >= 0.0f ? 0.5f : -0.5f));
  return q;
}




// ------------------------------------------------------------
// Conv3x3 int8 -> int8 (internamente acumula em Q8.8)
// Entrada/saída int8 representam inteiros "puros" (equiv. Q8.8 com frac=0)
// ------------------------------------------------------------
static void conv3x3_i8(
    const int8_t* in_i8, int Cin, int H, int W,
    const int8_t* wq, q15_t scaleW_q15,
    const fx16_t* b,      // pode ser NULL
    int Cout,
    int8_t* out_i8,
    int apply_relu)
{
  const int HW = H * W;

  // pré-cálculo por chamada
  //const q15_t scaleW_q15 = q15_from_float(scaleW);
  const uint16_t su = (uint16_t)scaleW_q15; // scale positivo
  const uint8_t slo = (uint8_t)(su & 0xFF);
  const uint8_t shi = (uint8_t)(su >> 8);
  const uint8_t use_lo_only = (shi == 0);

  for (int oc = 0; oc < Cout; ++oc) {
    // bias em Q8.8 (1x por canal)
    //const int32_t bias_q88 = b ? bias_float_to_q88_i32(b[oc]) : 0;
    //const int32_t bias_q88 = b ? bias_float_to_q88_i32(  fx16_to_float( b[oc]  )) : 0;
    const int32_t bias_q88 = b ?  b[oc]   : 0;

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {

        int32_t acc = bias_q88; // acumulador em Q8.8

        for (int ic = 0; ic < Cin; ++ic) {
          const int8_t* pin = in_i8 + ic * HW;
          const int8_t* pw  = wq + (oc * Cin + ic) * 9; // 3x3

          for (int ky = -1; ky <= 1; ++ky) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            for (int kx = -1; kx <= 1; ++kx) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              // int8 ativação -> Q8.8 (fração zero)
              fx16_t a_q = (fx16_t)((int16_t)pin[yy * W + xx] << 8);
              int8_t wi  = pw[(ky + 1) * 3 + (kx + 1)];

              fx16_t w_q;
              if (use_lo_only) {
                w_q = w_i8_scale_q15_to_q88_lo(wi, slo);
              } else {
                w_q = w_i8_scale_q15_to_q88_hi_lo(wi, shi, slo);
              }

              fx16_t prod_q = fx16_mul_16_16(a_q, w_q); // Q8.8
              acc += (int32_t)prod_q;
            }
          }
        }

        // LeakyReLU(1/8)
        if (apply_relu && acc < 0) {
          acc = 0;
        }

        out_i8[(oc * H + y) * W + x] = acc_q88_to_i8_sat(acc);
      }
    }
  }
}



 

static void conv3x3_int(
    const fx16_t* in, int Cin, int H, int W,
    const int8_t* wq, float scaleW,
    const float* b,              // bias float (pode ser NULL)
    int Cout,
    fx16_t* out,
    int apply_relu){
      conv3x3_fxio_v4(in, Cin, H, W, wq, scaleW, b, Cout, out, apply_relu);
    }
// ------------------------------------------------------------
// API: forward do gerador (float interno)
// Entrada: z_s16 (G_NW*2*2)
// Saída: out24_s8 (24*24) com tanh aplicado
//
// IMPORTANTE: aqui eu trato z_s16 como "valor real" direto.
// Se no PyTorch você normaliza z (ex.: divide por 1024),
// faça o mesmo abaixo na conversão.
// ------------------------------------------------------------

void saturation_float(float* data, int N) {
 
    for (int i = 0; i < N; i++) {
        float x = data[i];        
        x = tanh(x /64.0) * 64.0;        
        data[i] = x;
    }
}

void generator_forward_float(const int8_t* z_s8, uint8_t* out24_s8)
{
  const int ZN = G_NW * 2 * 2;

  // Aloca buffers float
  float* x2   = (float*)malloc((size_t)G_NW  * 2  * 2  * sizeof(float));
  float* y2   = (float*)malloc((size_t)G_C2  * 2  * 2  * sizeof(float));

  float* up6  = (float*)malloc((size_t)G_C2  * 6  * 6  * sizeof(float));
  float* y6   = (float*)malloc((size_t)G_C6  * 6  * 6  * sizeof(float));

  float* up12 = (float*)malloc((size_t)G_C6  * 12 * 12 * sizeof(float));
  float* y12  = (float*)malloc((size_t)G_C12 * 12 * 12 * sizeof(float));

  float* up24 = (float*)malloc((size_t)G_C12 * 24 * 24 * sizeof(float));
  float* y24a = (float*)malloc((size_t)G_C24 * 24 * 24 * sizeof(float));
  float* y24b = (float*)malloc((size_t)G_CH  * 24 * 24 * sizeof(float));
  float* yout = (float*)malloc((size_t)1     * 24 * 24 * sizeof(float));

  if (!x2||!y2||!up6||!y6||!up12||!y12||!up24||!y24a||!y24b||!yout) {
    fprintf(stderr, "generator_forward_float: malloc failed\n");
    goto cleanup;
  }

  // 1) z int8 -> float (C=G_NW, H=W=2)
  // Se quiser normalizar: x2[i] = (float)z_s8[i] / 1024.0f;
  for (int i = 0; i < ZN; i++) x2[i] = (float)z_s8[i]    ;

  // l0: (G_NW,2,2) -> (G_C2,2,2) + ReLU
  conv3x3_f32(x2, G_NW, 2, 2, W_l0, SCALEW_L0, B_l0, G_C2, y2, 1);
 saturation_float(  y2 , G_C2 * 2 * 2);

  float out_l1_min = y2[0];
  float out_l1_max = y2[0];
  for (int i = 1; i < G_C2 * 2 * 2; i++) {
      if (y2[i] < out_l1_min) out_l1_min = y2[i];
      if (y2[i] > out_l1_max) out_l1_max = y2[i];
  }
  printf("l1 out min: %f, max: %f\n", out_l1_min, out_l1_max);


  // up 2->6
  upsample_nearest_f32(y2, G_C2, 2, 2, up6, 6, 6);

  // l1: (G_C2,6,6) -> (G_C6,6,6) + ReLU
  conv3x3_f32(up6, G_C2, 6, 6, W_l1, SCALEW_L1, B_l1, G_C6, y6, 1);
  saturation_float( y6 , G_C6 * 6 * 6);

float out_l2_min = y6[0];
float out_l2_max = y6[0];
for (int i = 1; i < G_C6 * 6 * 6; i++) {
    if (y6[i] < out_l2_min) out_l2_min = y6[i];
    if (y6[i] > out_l2_max) out_l2_max = y6[i];
}
printf("y6 out min: %f, max: %f\n", out_l2_min, out_l2_max);

  // up 6->12
  upsample_nearest_f32(y6, G_C6, 6, 6, up12, 12, 12);

  // l2: (G_C6,12,12) -> (G_C12,12,12) + ReLU
  conv3x3_f32(up12, G_C6, 12, 12, W_l2, SCALEW_L2, B_l2, G_C12, y12, 1);
  saturation_float( y12 , G_C12 * 12 * 12);

   float out_l3_min = y12[0];
  float out_l3_max = y12[0];
  for (int i = 1; i < G_C12 * 12 * 12; i++) {
      if (y12[i] < out_l3_min) out_l3_min = y12[i];
      if (y12[i] > out_l3_max) out_l3_max = y12[i];
  }
  printf("y12 out min: %f, max: %f\n", out_l3_min, out_l3_max);

  // up 12->24
  upsample_nearest_f32(y12, G_C12, 12, 12, up24, 24, 24);

  float out_up24_min = up24[0];
  float out_up24_max = up24[0];
  for (int i = 1; i < G_C12 * 24 * 24; i++) {
      if (up24[i] < out_up24_min) out_up24_min = up24[i];
      if (up24[i] > out_up24_max) out_up24_max = up24[i];
  }
  printf("up24 out min: %f, max: %f\n", out_up24_min, out_up24_max);

  // l3: (G_C12,24,24) -> (G_C24,24,24) + ReLU
  conv3x3_f32(up24, G_C12, 24, 24, W_l3, SCALEW_L3, B_l3, G_C24, y24a, 1);
  saturation_float( y24a , G_C24 * 24 * 24);

  float out_l4_min = y24a[0];
  float out_l4_max = y24a[0];
  for (int i = 1; i < G_C24 * 24 * 24; i++) {
      if (y24a[i] < out_l4_min) out_l4_min = y24a[i];
      if (y24a[i] > out_l4_max) out_l4_max = y24a[i];
  }
  printf("l4 out min: %f, max: %f\n", out_l4_min, out_l4_max);

  // l4: (G_C24,24,24) -> (G_CH,24,24) + ReLU
  conv3x3_f32(y24a, G_C24, 24, 24, W_l4, SCALEW_L4, B_l4, G_CH, y24b, 1);
  saturation_float( y24b , G_CH * 24 * 24);

   float out_l5_min = y24b[0];

  // out: (G_CH,24,24) -> (1,24,24) sem ReLU
  conv3x3_f32(y24b, G_CH, 24, 24, W_out, SCALEW_OUT, B_out, 1, yout, 0);

//compute max and min of yout for validação
float max_yout = yout[0];
float min_yout = yout[0];
for (int i = 1; i < 24 * 24; i++) {
    if (yout[i] > max_yout) max_yout = yout[i];
    if (yout[i] < min_yout) min_yout = yout[i];
}
printf("yout min: %f, max: %f\n", min_yout, max_yout);

float out_th_min = 1.0f;
float out_th_max = -1.0f;

  // tanh e export int8: [-1,1] -> [0, 255]
  for (int i = 0; i < 24 * 24; i++) {
    float x =  yout[i]/16;
    float t = tanhf((x)); // validação -1 to +1
    if (t > out_th_max)  out_th_max = t;
    if (t < out_th_min) out_th_min = t;
    int32_t v = (int32_t)lrintf( 128*t + 128  ); // [-1,1] -> [0,255]
    out24_s8[i] = clamp_i8(v);
  }

printf("Output (tanh) min: %f, max: %f\n", out_th_min, out_th_max);


cleanup:
  free(x2);
  free(y2);
  free(up6);
  free(y6);
  free(up12);
  free(y12);
  free(up24);
  free(y24a);
  free(y24b);
  free(yout);
}


void generator_forward_fx16(const int8_t* z_s16, uint8_t* out24_s8)
{
  const int ZN = G_NW * 2 * 2;

  fx16_t* x2   = (fx16_t*)malloc((size_t)G_NW  * 2  * 2  * sizeof(fx16_t));
  fx16_t* y2   = (fx16_t*)malloc((size_t)G_C2  * 2  * 2  * sizeof(fx16_t));
  fx16_t* up6  = (fx16_t*)malloc((size_t)G_C2  * 6  * 6  * sizeof(fx16_t));
  fx16_t* y6   = (fx16_t*)malloc((size_t)G_C6  * 6  * 6  * sizeof(fx16_t));
  fx16_t* up12 = (fx16_t*)malloc((size_t)G_C6  * 12 * 12 * sizeof(fx16_t));
  fx16_t* y12  = (fx16_t*)malloc((size_t)G_C12 * 12 * 12 * sizeof(fx16_t));
  fx16_t* up24 = (fx16_t*)malloc((size_t)G_C12 * 24 * 24 * sizeof(fx16_t));
  fx16_t* y24a = (fx16_t*)malloc((size_t)G_C24 * 24 * 24 * sizeof(fx16_t));
  fx16_t* y24b = (fx16_t*)malloc((size_t)G_CH  * 24 * 24 * sizeof(fx16_t));
  fx16_t* yout = (fx16_t*)malloc((size_t)1     * 24 * 24 * sizeof(fx16_t));

  if (!x2||!y2||!up6||!y6||!up12||!y12||!up24||!y24a||!y24b||!yout) {
    fprintf(stderr, "generator_forward_fixed: malloc failed\n");
    goto cleanup;
  }

  // 1) z int16 -> fixed (C=G_NW, H=W=2)
  for (int i = 0; i < ZN; i++) x2[i] = z_s16[i] << 8; // example quantization

    //entrada
     fx16_t x2_min = x2[0];
    fx16_t x2_max = x2[0];
    for (int i = 1; i < G_NW * 2 * 2; i++) {
      if (x2[i] < x2_min) x2_min = x2[i];
      if (x2[i] > x2_max) x2_max = x2[i];
    }
    printf("x2 input min: %f, max: %f\n", fx16_to_float(x2_min), fx16_to_float(x2_max));

    // l0: (G_NW,2,2) -> (G_C2,2,2) + ReLU
    conv3x3_int(x2, G_NW, 2, 2, W_l0, SCALEW_L0, B_l0, G_C2, y2, 1);
    //show min max
    fx16_t y2_min = y2[0];
    fx16_t y2_max = y2[0];
    for (int i = 1; i < G_C2 * 2 * 2; i++) {
      if (y2[i] < y2_min) y2_min = y2[i];
      if (y2[i] > y2_max) y2_max = y2[i];
    }
    printf("l1 out min: %f, max: %f\n", fx16_to_float(y2_min), fx16_to_float(y2_max));
    size_t mem_l0 = (G_NW*2*2 + G_C2*2*2) * sizeof(fx16_t);
    printf("[mem] l0: %.1f KB\n", mem_l0/1024.0);

    // up 2->6
    upsample_nearest_fixed(y2, G_C2, 2, 2, up6, 6, 6);
    size_t mem_up6 = (G_C2*2*2 + G_C2*6*6) * sizeof(fx16_t);
    printf("[mem] up6: %.1f KB\n", mem_up6/1024.0);

    // l1: (G_C2,6,6) -> (G_C6,6,6) + ReLU
    conv3x3_int(up6, G_C2, 6, 6, W_l1, SCALEW_L1, B_l1, G_C6, y6, 1);
    fx16_t y6_min = y6[0];
    fx16_t y6_max = y6[0];
    for (int i = 1; i < G_C6 * 6 * 6; i++) {
      if (y6[i] < y6_min) y6_min = y6[i];
      if (y6[i] > y6_max) y6_max = y6[i];
    }
    printf("y6 out min: %f, max: %f\n", fx16_to_float(y6_min), fx16_to_float(y6_max));
    size_t mem_l1 = (G_C2*6*6 + G_C6*6*6) * sizeof(fx16_t);
    printf("[mem] l1: %.1f KB\n", mem_l1/1024.0);

    // up 6->12
    upsample_nearest_fixed(y6, G_C6, 6, 6, up12, 12, 12);
    size_t mem_up12 = (G_C6*6*6 + G_C6*12*12) * sizeof(fx16_t);
    printf("[mem] up12: %.1f KB\n", mem_up12/1024.0);

    // l2: (G_C6,12,12) -> (G_C12,12,12) + ReLU
    conv3x3_int(up12, G_C6, 12, 12, W_l2, SCALEW_L2, B_l2, G_C12, y12, 1);
    fx16_t y12_min = y12[0];
    fx16_t y12_max = y12[0];
    for (int i = 1; i < G_C12 * 12 * 12; i++) {
      if (y12[i] < y12_min) y12_min = y12[i];
      if (y12[i] > y12_max) y12_max = y12[i];
    }
    printf("y12 out min: %f, max: %f\n", fx16_to_float(y12_min), fx16_to_float(y12_max));
    size_t mem_l2 = (G_C6*12*12 + G_C12*12*12) * sizeof(fx16_t);
    printf("[mem] l2: %.1f KB\n", mem_l2/1024.0);


    // up 12->24
    upsample_nearest_fixed(y12, G_C12, 12, 12, up24, 24, 24);
    size_t mem_up24 = (G_C12*12*12 + G_C12*24*24) * sizeof(fx16_t);
    printf("[mem] up24: %.1f KB\n", mem_up24/1024.0);

    // l3: (G_C12,24,24) -> (G_C24,24,24) + ReLU
    conv3x3_int(up24, G_C12, 24, 24, W_l3, SCALEW_L3, B_l3, G_C24, y24a, 1);
    fx16_t y24a_min = y24a[0];
    fx16_t y24a_max = y24a[0];
    for (int i = 1; i < G_C24 * 24 * 24; i++) {
      if (y24a[i] < y24a_min) y24a_min = y24a[i];
      if (y24a[i] > y24a_max) y24a_max = y24a[i];
    }
    printf("l3/y24a out min: %f, max: %f\n", fx16_to_float(y24a_min), fx16_to_float(y24a_max));
    size_t mem_l3 = (G_C12*24*24 + G_C24*24*24) * sizeof(fx16_t);
    printf("[mem] l3: %.1f KB\n", mem_l3/1024.0);

  // l4: (G_C24,24,24) -> (G_CH,24,24) + ReLU
  conv3x3_int(y24a, G_C24, 24, 24, W_l4, SCALEW_L4, B_l4, G_CH, y24b, 1);
  fx16_t y24b_min = y24b[0];
  fx16_t y24b_max = y24b[0];
  for (int i = 1; i < G_CH * 24 * 24; i++) {
      if (y24b[i] < y24b_min) y24b_min = y24b[i];
      if (y24b[i] > y24b_max) y24b_max = y24b[i];
  }
  printf(" l4/y24b out min: %f, max: %f\n", fx16_to_float(y24b_min), fx16_to_float(y24b_max));
  size_t mem_l4 = (G_C24*24*24 + G_CH*24*24) * sizeof(fx16_t);
  printf("[mem] l4: %.1f KB\n", mem_l4/1024.0);

  // out: (G_CH,24,24) -> (1,24,24) sem ReLU
  conv3x3_int(y24b, G_CH, 24, 24, W_out, SCALEW_OUT, B_out, 1, yout, 0);
  size_t mem_out = (G_CH*24*24 + 24*24) * sizeof(fx16_t);
  printf("[mem] out: %.1f KB\n", mem_out/1024.0);


  fx16_t yout_min = yout[0];
  fx16_t yout_max = yout[0];
  for (int i = 1; i < 24 * 24; i++) {
      if (yout[i] < yout_min) yout_min = yout[i];
      if (yout[i] > yout_max) yout_max = yout[i];
  }
  printf("yout min: %f, max: %f\n", fx16_to_float(yout_min), fx16_to_float(yout_max));

    // tanh e export int8: [-1,1] -> [0, 255]
  for (int i = 0; i < 24 * 24; i++) {
    float x =  fx16_to_float(yout[i]); // fixed -> float para tanh
    float t = tanhf((x)); // validação -1 to +1
    int32_t v = (int32_t)lrintf(  t * 127.0f + 128.0f); // [-1,1] -> [0,255]
    out24_s8[i] = clamp_i8(v);
  }
  //normalize imageout
  uint8_t omin = out24_s8[0];
  uint8_t omax = out24_s8[0];
  for (int i = 1; i < 24 * 24; i++) {
      if (out24_s8[i] < omin) omin = out24_s8[i];
      if (out24_s8[i] > omax) omax = out24_s8[i];
  }

    for (int i = 0; i < 24 * 24; i++) {
        uint8_t v = out24_s8[i];
        uint8_t vn = (uint8_t)(((int)v - omin) * 255 / (omax - omin)); // normalize to [0,255]
        out24_s8[i] = vn;
    }



cleanup:
  free(x2);
  free(y2);
  free(up6);
  free(y6);
  free(up12);
  free(y12);
  free(up24);
  free(y24a);
  free(y24b);
  free(yout);
}


// Q8.8 -> int8 (com saturação)
static inline int8_t fx16_to_i8_sat(fx16_t x)
{
  // arredonda para inteiro antes de remover fração
  int16_t t = x;
  if (t >= 0) t += 0x0080;
  else        t -= 0x0080;

  int16_t v = (int16_t)(t >> 8); // inteiro com sinal

  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return (int8_t)v;
}

// int8 -> Q8.8
static inline fx16_t i8_to_fx16(int8_t x)
{
  return (fx16_t)((int16_t)x << 8);
}

// buffer inteiro: Q8.8 -> int8
static void quantize_fx16_to_i8(
    const fx16_t* in_q88, int8_t* out_i8, int n)
{
  for (int i = 0; i < n; ++i) {
    out_i8[i] = fx16_to_i8_sat(in_q88[i]);
  }
}

// buffer inteiro: int8 -> Q8.8
static void dequantize_i8_to_fx16(
    const int8_t* in_i8, fx16_t* out_q88, int n)
{
  for (int i = 0; i < n; ++i) {
    out_q88[i] = i8_to_fx16(in_i8[i]);
  }
}


static void upsample_nearest_i8(
    const int8_t* in, int C, int Hin, int Win,
    int8_t* out, int Hout, int Wout)
{
  for (int c = 0; c < C; ++c) {
    const int8_t* pin = in + c * Hin * Win;
    int8_t* pout = out + c * Hout * Wout;

    for (int y = 0; y < Hout; ++y) {
      int sy = (y * Hin) / Hout;
      for (int x = 0; x < Wout; ++x) {
        int sx = (x * Win) / Wout;
        pout[y * Wout + x] = pin[sy * Win + sx];
      }
    }
  }
}

static const int8_t TANH_LUT_I8[256] = {
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

void saturation(int8_t* x , int N) {
    return ;
    for (int i = 0; i < N; ++i) {
        uint8_t idx = (uint8_t)(x[i] + 128); // shift to [0,255]
        x[i] = TANH_LUT_I8[idx];
    }
}

static int8_t tanh_lut[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4,
    5, 5, 6, 7, 8, 8, 10, 11, 12, 14, 15, 17, 19, 22, 24, 27,
    31, 34, 38, 42, 47, 52, 57, 63, 69, 75, 82, 89, 97, 104, 112, 120,
    128, 136, 144, 152, 159, 167, 174, 181, 187, 193, 199, 204, 209, 214, 218, 222,
    225, 229, 232, 234, 237, 239, 241, 242, 244, 245, 246, 248, 248, 249, 250, 251,
    251, 252, 252, 253, 253, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};


void generator_forward_int8(const int8_t* z_s16, uint8_t* out24_u8)
{

  const int ZN = G_NW * 2 * 2;

  // ---- Ativadores INT8 ----
  int8_t* x2   = (int8_t*)malloc((size_t)G_NW  * 2  * 2);
  int8_t* y2   = (int8_t*)malloc((size_t)G_C2  * 2  * 2);
  int8_t* up6  = (int8_t*)malloc((size_t)G_C2  * 6  * 6);
  int8_t* y6   = (int8_t*)malloc((size_t)G_C6  * 6  * 6);
  int8_t* up12 = (int8_t*)malloc((size_t)G_C6  * 12 * 12);
  int8_t* y12  = (int8_t*)malloc((size_t)G_C12 * 12 * 12);
  int8_t* up24 = (int8_t*)malloc((size_t)G_C12 * 24 * 24);
  int8_t* y24a = (int8_t*)malloc((size_t)G_C24 * 24 * 24);
  int8_t* y24b = (int8_t*)malloc((size_t)G_CH  * 24 * 24);

  // saída pré-tanh (pode ser int8 ou fx16; aqui vou deixar fx16 para tanh mais fiel)
  int8_t* yout = (int8_t*)malloc((size_t)24 * 24 * sizeof(int8_t));

  if (!x2 || !y2 || !up6 || !y6 || !up12 || !y12 || !up24 || !y24a || !y24b || !yout) {
    fprintf(stderr, "generator_forward_int8: malloc failed\n");
    goto cleanup;
  }

  // 1) z int16 -> int8 (ajuste a escala conforme seu z real)
  for (int i = 0; i < ZN; ++i) {
    int32_t v = z_s16[i]  ;   // exemplo: int16 bruto -> int8

    x2[i] = z_s16[i] ;
  }

  //write x2
  for (int i = 0; i < ZN; i++) {
    printf("%d  ", x2[i]); if ( i < ZN-1 ) printf(", ");
  }
  printf("\n");

  // l0: (G_NW,2,2) -> (G_C2,2,2)
  conv3x3_i8(x2,  G_NW,  2,  2, W_l0,  SCALEW_L0_Q15,  B_l0_f16 ,  G_C2,  y2,   1);
  saturation(y2, G_C2 * 2 * 2); // ReLU + saturação para int8
  // up 2->6
  upsample_nearest_i8(y2, G_C2,  2,  2, up6,  6,  6);
  // l1
  conv3x3_i8(up6, G_C2,  6,  6, W_l1,  SCALEW_L1_Q15,  B_l1_f16 ,  G_C6,  y6,   1);
  saturation(y6, G_C6 * 6 * 6); // ReLU + saturação para int8
  // up 6->12
  upsample_nearest_i8(y6, G_C6,  6,  6, up12, 12, 12);
  // l2
  conv3x3_i8(up12, G_C6, 12, 12, W_l2,  SCALEW_L2_Q15,  B_l2_f16 ,  G_C12, y12,  1);
  saturation(y12, G_C12 * 12 * 12); // ReLU + saturação para int8
  // up 12->24
  upsample_nearest_i8(y12, G_C12, 12, 12, up24, 24, 24);
  // l3
  conv3x3_i8(up24, G_C12, 24, 24, W_l3,  SCALEW_L3_Q15,  B_l3_f16 ,  G_C24, y24a, 1);
  saturation(y24a, G_C24 * 24 * 24); // ReLU + saturação para int8
  // l4
  conv3x3_i8(y24a, G_C24, 24, 24, W_l4,  SCALEW_L4_Q15,  B_l4_f16 ,  G_CH,  y24b, 1);
  saturation(y24b, G_CH * 24 * 24); // ReLU + saturação para int8
  // out: aqui idealmente usar conv de saída com maior precisão (fx16), porque vai para tanh
  // Se você tiver conv3x3_i8_to_fx16, use ela:
  conv3x3_i8(y24b, G_CH, 24, 24, W_out, SCALEW_OUT_Q15, B_out_f16 , 1, yout, 0);

int8_t yout_min = yout[0];
int8_t yout_max = yout[0];
for (int i = 1; i < 24 * 24; i++) {
    if (yout[i] < yout_min) yout_min = yout[i];
    if (yout[i] > yout_max) yout_max = yout[i];
}
printf("yout (pre-tanh) min: %d, max: %d\n", yout_min, yout_max);

 
  // tanh + export [0..255]
  for (int i = 0; i < 24 * 24; ++i) {    
    uint8_t idx = (uint8_t)clamp_u8_s(yout[i]/2 + 127); // shift para índice positivo
    out24_u8[i] = (uint8_t) tanh_lut[ idx ]; // usando LUT para tanh
  }

  // //write LUT
  // for(int i = -128; i <= 127; i++) {
  //   float x = i / 8.0f;
  //   float t = tanhf(x) ;
  //   int32_t v = (int32_t)lrintf(t * 127.0f + 128.0f);
  //   if (v < 0) v = 0;
  //   if (v > 255) v = 255;
  //   printf("%d ,", v);
  // }
  // printf("\n");

cleanup:
  free(x2);
  free(y2);
  free(up6);
  free(y6);
  free(up12);
  free(y12);
  free(up24);
  free(y24a);
  free(y24b);
  free(yout);
}
void generator_forward(const int8_t* z_s8, uint8_t* out24_s8)
{

   generator_forward_int8 (z_s8, out24_s8);
   //generator_forward_float(z_s8, out24_s8);
   //generator_forward_fx16(z_s8, out24_s8);

}
