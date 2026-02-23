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


typedef union {
    i16 full;
    struct {
        u8 l;  // Parte fracionária (Low byte)
        i8 h;   // Parte inteira (High byte com sinal)
    } bytes;
} fx16_hl;

static inline fx16_t fx16_mul_16_16_z80(fx16_t a, fx16_t b) {
    fx16_hl ua = { .full = a };
    fx16_hl ub = { .full = b };

    // 2. Forçamos multiplicações de 16 bits (muito mais rápidas que 32 bits).
    // O Z80 fará isso usando as rotinas de 8x8->16 ou 16x16->16 do compilador.
    i16 ah_bh = (i16)ua.bytes.h * ub.bytes.h; 
    
    // 3. Early Exit (Saída Antecipada)
    // Se a parte inteira já estourou o limite do Q8.8, pulamos o resto da matemática!
    // O valor máximo da parte inteira que cabe em um Q8.8 é 127 a -128.
    if (ah_bh > 127) return 32767;
    if (ah_bh < -128) return -32768;

    // Calculamos as partes cruzadas e fracionárias em 16 bits
    i16 cross1 = (i16)ua.bytes.h * ub.bytes.l;
    i16 cross2 = (i16)ub.bytes.h * ua.bytes.l;
    u16 al_bl = (u16)ua.bytes.l * ub.bytes.l;

    // Para evitar o shift >> 8 da parte fracionária, lemos o High Byte direto
    fx16_hl frac_round = { .full = (i16)(al_bl + 128) };

    // 4. Somamos usando um acumulador de 32 bits APENAS no final.
    // Somar em 32 bits no Z80 é muito rápido (são apenas rápidas instruções 
    // ADD/ADC em cascata). O proibido era MULTIPLICAR em 32 bits.
    i32 p = ((i32)ah_bh << 8) + cross1 + cross2 + frac_round.bytes.h;

    // 5. Saturação final fina (caso os termos cruzados tenham gerado overflow)
    if (p > 32767) return 32767;
    if (p < -32768) return -32768;

    return (fx16_t)p;
}


static inline fx16_t fx16_mul_8_16_z80(i8 a, fx16_t b) {
    fx16_hl ua = { .full = 0 };
    ua.bytes.h = a; // i8 para parte inteira, parte fracionária zero
    fx16_hl ub = { .full = b };

    // 2. Forçamos multiplicações de 16 bits (muito mais rápidas que 32 bits).
    // O Z80 fará isso usando as rotinas de 8x8->16 ou 16x16->16 do compilador.
    i16 ah_bh = (i16)ua.bytes.h * ub.bytes.h; 
    
    // 3. Early Exit (Saída Antecipada)
    // Se a parte inteira já estourou o limite do Q8.8, pulamos o resto da matemática!
    // O valor máximo da parte inteira que cabe em um Q8.8 é 127 a -128.
    if (ah_bh > 127) return 32767;
    if (ah_bh < -128) return -32768;

    // Calculamos as partes cruzadas e fracionárias em 16 bits
    i16 cross1 = (i16)ua.bytes.h * ub.bytes.l;
    //i16 cross2 = (i16)ub.bytes.h * ua.bytes.l;
    //u16 al_bl = (u16)ua.bytes.l * ub.bytes.l;

    // Para evitar o shift >> 8 da parte fracionária, lemos o High Byte direto
    //fx16_hl frac_round = { .full = (i16)(al_bl + 128) };

    // 4. Somamos usando um acumulador de 32 bits APENAS no final.
    // Somar em 32 bits no Z80 é muito rápido (são apenas rápidas instruções 
    // ADD/ADC em cascata). O proibido era MULTIPLICAR em 32 bits.
    i32 p = ((i32)ah_bh << 8) + cross1; // + frac_round.bytes.h;

    // 5. Saturação final fina (caso os termos cruzados tenham gerado overflow)
    if (p > 32767) return 32767;
    if (p < -32768) return -32768;

    return (fx16_t)p;
}



static inline fx16_t fx16_mul_16_16(fx16_t a, fx16_t b) {
    i16 ah = (i16)(a >> 8);     // signed
    i16 bh = (i16)(b >> 8);     // signed
    u16 al = (u8)(a & 0xFF);  // unsigned
    u16 bl = (u8)(b & 0xFF);  // unsigned

    i32 p =
        ((i32)ah * (i32)bh << 8) +   // ah*bh * 256
        ((i32)ah * (i32)bl) +        // ah*bl
        ((i32)bh * (i32)al) +        // bh*al
        (((i32)al * (i32)bl + 128) >> 8); // (al*bl)/256 com rounding

    if (p >  32767) p =  32767;
    if (p < -32768) p = -32768;
    return (fx16_t)p;
}



// Q8.8 acumulado (int32) -> int8 com rounding
static inline i8 acc_q88_to_i8_sat(i32 acc_q88) {
  // arredondar para inteiro
  if (acc_q88 >= 0) acc_q88 += 128;
  else              acc_q88 -= 128;

  i32 vi = acc_q88 >> 8;
  return clamp_i8_s(vi);
}

// bias float -> int32 Q8.8 (1x por canal)
static inline i32 bias_float_to_q88_i32(float x) {
  // Se quiser evitar float aqui também, pré-converta no export.
  float s = x * 256.0f;
  i32 q = (i32)(s + (s >= 0.0f ? 0.5f : -0.5f));
  return q;
}

// Arredondamento simétrico para shift à direita por 7
static inline i16 rshift7_round_i16(i16 x) {
  if (x >= 0) return (i16)((x + 64) >> 7);
  return (i16)(-(((-x) + 64) >> 7));
}

typedef union {
    i16 full;
    struct {
        u8 l;  // Parte fracionária (Low byte)
        i8 h;   // Parte inteira (High byte com sinal)
    } bytes;
} i16_hl;

 

// wi * scale_q15 -> Q8.8 (caminho rápido: só byte low)
static inline fx16_t w_i8_scale_q15_to_q88_lo(i8 wi, u8 slo) {
  i16 t = (i16)wi *  slo;
  return (fx16_t)rshift7_round_i16(t);
}

// wi * scale_q15 -> Q8.8 (geral, sem int32)
static inline fx16_t w_i8_scale_q15_to_q88_hi_lo(i8 wi, u8 shi, u8 slo) {
  i16 t_hi = (i16)wi * (u8)shi;
  i16 t_lo = (i16)wi * (u8)slo;
  i16 q = (i16)(t_hi << 1);               // 2*(wi*shi)
  q = (i16)(q + rshift7_round_i16(t_lo));     // + round((wi*slo)/128)
  return (fx16_t)q;
}

static inline i32 div8_toward_zero_i32(i32 x) {
  if (x >= 0) return x >> 3;
  return -(((-x) >> 3));
}


static void upsample_nearest_i8(
    const i8* in, int C, int Hin, int Win,
    i8* out, int Hout, int Wout)
{
  for (int c = 0; c < C; ++c) {
    const i8* pin = in + c * Hin * Win;
    i8* pout = out + c * Hout * Wout;

    for (int y = 0; y < Hout; ++y) {
      int sy = (y * Hin) / Hout;
      for (int x = 0; x < Wout; ++x) {
        int sx = (x * Win) / Wout;
        pout[y * Wout + x] = pin[sy * Win + sx];
      }
    }
  }
}
// ------------------------------------------------------------
// Conv3x3 int8 -> int8 (internamente acumula em Q8.8)
// Entrada/saída int8 representam inteiros "puros" (equiv. Q8.8 com frac=0)
// ------------------------------------------------------------
static void conv3x3_i8(
    const i8* in_i8, int Cin, int H, int W,
    const i8* wq, q15_t scaleW_q15,
    const fx16_t* b,      // pode ser NULL
    int Cout,
    i8* out_i8,
    int apply_relu)
{
  const int HW = H * W;
  const int W_OC_STRIDE = Cin * 9;   // pesos por canal de saída

  // pré-cálculo por chamada
  const u16 su = (u16)scaleW_q15; // scale positivo
  const u8 slo = (u8)(su & 0xFF);
  const u8 shi = (u8)(su >> 8);
  const u8 use_lo_only = (shi == 0);

  for (int oc = 0; oc < Cout; ++oc) {
    const i32 bias_q88 = b ? (i32)b[oc] : 0;

    // base dos pesos e da saída para este canal de saída
    const i8* w_oc = wq + oc * W_OC_STRIDE;
    i8* out_oc = out_i8 + oc * HW;

    for (int y = 0; y < H; ++y) {
      const int yW = y * W;
      i8* out_row = out_oc + yW;

      for (int x = 0; x < W; ++x) {

        i32 acc = bias_q88; // acumulador em Q8.8

        // ponteiros incrementais por canal de entrada
        const i8* pin = in_i8;
        const i8* pw  = w_oc;

        for (int ic = 0; ic < Cin; ++ic) {

          for (i8 ky = -1; ky <= 1; ++ky) {
            int yy = y + ky;
            if ((unsigned)yy >= (unsigned)H) continue;

            const int yy_mul_w = yy * W;
            const int ky_plus_one_mul_3 = (ky + 1) * 3;

            for (i8 kx = -1; kx <= 1; ++kx) {
              int xx = x + kx;
              if ((unsigned)xx >= (unsigned)W) continue;

              // int8 ativação -> Q8.8 (fração zero)
              //fx16_t a_q = (fx16_t)((i16)pin[yy_mul_w + xx] << 8);                            
              i8 a_q = (i8)pin[yy_mul_w + xx];                            
              i8 wi = pw[ky_plus_one_mul_3 + (kx + 1)];

              fx16_t w_q;
              if (use_lo_only) {
                w_q = w_i8_scale_q15_to_q88_lo(wi, slo);
              } else {
                w_q = w_i8_scale_q15_to_q88_hi_lo(wi, shi, slo);
              }

              //fx16_t prod_q = fx16_mul_16_16_z80((fx16_t)a_q, w_q); // Q8.8
              fx16_t prod_q = fx16_mul_8_16_z80((fx16_t)a_q, w_q); // Q8.8
          
              

              acc += (i32)prod_q;
            }
          }

          // próximo canal de entrada / pesos (soma em vez de multiplicação)
          pin += HW;
          pw  += 9;
        }

        // ReLU
        if (apply_relu && acc < 0) {
          acc = 0;
        }

        // saída via ponteiro de linha (evita multiplicações no índice final)
        out_row[x] = acc_q88_to_i8_sat(acc);
      }
    }
  }
}