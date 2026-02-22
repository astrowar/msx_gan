#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gen_weights.h"   // defines G_NW etc.

#include "random_sequences.h"

void generator_forward(const int8_t* z_s16, uint8_t* out24_s8); // do gen_runtime.c



// ------------------------------------------------------------
// PRNG (LCG) simples e rápido
// ------------------------------------------------------------
static uint32_t g_rng = 1;

static void rng_seed(uint32_t seed) {
  if (seed == 0) seed = 1;
  g_rng = seed;
}

static uint16_t rng_u16(void) {
  // LCG clássico (32-bit)
  g_rng = g_rng * 1664525u + 1013904223u;
  return (uint16_t)(g_rng >> 16);
}



// Z int16 "pseudo-gaussiano" barato (soma de uniformes, tipo CLT)
static int16_t rand_i16_centered(void) {
  // soma 4 uniformes pra aproximar gauss (bem simples)
  // cada u em [0..65535], soma em [0..262140], centro ~131070
  uint32_t s = (uint32_t)rng_u16() + (uint32_t)rng_u16() + (uint32_t)rng_u16() + (uint32_t)rng_u16();
  int32_t v = (int32_t)s - 131070;         // ~centrado em 0
  // reduz amplitude pra caber bem no int16 e evitar saturar demais
  v >>= 2;                                 // divide por 4
  if (v < -32768) v = -32768;
  if (v >  32767) v =  32767;
  return (int16_t)v;
}

// Z int16 "pseudo-gaussiano" barato (soma de uniformes, tipo CLT)
static int8_t zpy[] = {
  -28,    0,  -81,   40,  -13,  -63,  -33,  -43,    6,   85,   73,
        -68,   48,   11,  -18,   53,   83,   81,   56,   28,   44,  -12,
        -14,   -6,   44,   51,  -18,   34,  -61,  -68,   27,  -78,    4,
         -3,   24,   80,   92,   14,  -99,   82,  -27,  -16,   18,  103,
         32,   40,   75,   43,   69, -113,  -16,  -19,  128,  -17,   -7,
         93,   -1,  158,   11,   58,   63,  -53,   20
};
static int8_t rand_i8_centered_16(void) {
    //centro em zero e valores entre -63 a 63

    int32_t acc = 0;
    for(int k =0 ;k < 8; k++) {
      int8_t v1 = (int8_t)(rng_u16() & 0x1F) - 16; // gera valor entre -16 e 15
      acc += v1;
    }

    return (int8_t)(acc)  ;


}


static void make_z(int8_t* z, int n) {
  for (int i = 0; i < n; i++) {
    //z[i] = rand_i16_centered() >> 4;
    z[i] = (int8_t)rand_i8_centered_16() ;
    //z[i] = zpy[i];
  }
}

// ------------------------------------------------------------
// Salvar PGM (P5) 8-bit
// ------------------------------------------------------------
static int write_pgm_u8(const char* path, const uint8_t* img, int w, int h) {
  FILE* f = fopen(path, "wb");
  if (!f) return 0;

  // header
  fprintf(f, "P5\n%d %d\n255\n", w, h);

  // pixels
  size_t n = (size_t)w * (size_t)h;
  if (fwrite(img, 1, n, f) != n) {
    fclose(f);
    return 0;
  }
  fclose(f);
  return 1;
}


static inline float tanh_fast(float x)
{
  // clamp pra evitar exageros
  if (x >  5.0f) return  1.0f;
  if (x < -5.0f) return -1.0f;
  float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static void out16_to_u8_tanh(const int16_t* in, uint8_t* out, int n )
{
  float scale = 1024.0f;

  for (int i = 0; i < n; i++) {
    float x = (float)in[i] / scale;
    float t = x;             // [-1, 1]
    float u = (t * 0.5f + 0.5f) * 255.0f; // [0, 255]
    int ui = (int)(u + 0.5f);
    if (ui < 0) ui = 0;
    if (ui > 255) ui = 255;
    out[i] = (uint8_t)ui;
  }
}

// ------------------------------------------------------------
// Converter out int16 -> uint8 (normalização min/max por frame)
// ------------------------------------------------------------
static void out16_to_u8_minmax(const int16_t* in, uint8_t* out, int n) {
  int16_t mn = in[0], mx = in[0];
  for (int i = 1; i < n; i++) {
    if (in[i] < mn) mn = in[i];
    if (in[i] > mx) mx = in[i];
  }

  int32_t range = (int32_t)mx - (int32_t)mn;
  if (range < 1) range = 1;

  for (int i = 0; i < n; i++) {
    int32_t v = (int32_t)in[i] - (int32_t)mn;     // 0..range
    // escala para 0..255
    int32_t u = (v * 255 + (range / 2)) / range;  // arredondamento
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    out[i] = (uint8_t)u;
  }
}

/*
// Alternativa "fixa" (sem min/max):
// mapeia int16 [-32768..32767] para uint8 [0..255]
static void out16_to_u8_fixed(const int16_t* in, uint8_t* out, int n) {
  for (int i = 0; i < n; i++) {
    int32_t v = (int32_t)in[i] + 32768;   // 0..65535
    out[i] = (uint8_t)(v >> 8);
  }
}
*/

float z_fix[]  = {
  -1.9738,  0.0618, -1.5948,  0.6032,  0.0683,  0.3670, -0.5749,  0.5121,
         -0.5839, -0.8883,  0.1993, -0.7714, -0.0397,  0.0900,  0.9670,  2.8870,
          0.1531,  0.7619, -0.6656, -1.2862,  0.1225, -1.3954, -0.1894,  0.4509,
          0.2842,  0.7681,  1.2077, -2.4052, -0.5683, -1.0605, -1.0879,  1.2864,
         -0.8662, -0.1339, -0.2337,  0.4907,  0.3531, -0.0322, -0.2392, -0.1784,
         -0.6338, -0.8406, -0.6204, -0.2964, -0.9887, -0.4281,  0.1734,  0.6690,
          0.6348,  0.4976,  0.1969, -0.7030, -0.1391,  2.3322, -0.7836,  1.2866,
         -0.9534, -1.2717, -0.6996, -0.5482, -0.9145, -0.1284,  0.2567, -0.0656
};

int main(int argc, char** argv) {
  const char* out_path = (argc >= 2) ? argv[1] : "out.pgm";
  uint32_t seed = (argc >= 3) ? (uint32_t)strtoul(argv[2], NULL, 10) : 12345u;

  // Z tem tamanho G_NW*2*2
  enum { ZN = G_NW * 2 * 2 };
  int8_t z[ZN];

  // saída 24x24 mono

  uint8_t out8[24 * 24];

  rng_seed(seed);
  make_z(z, ZN);


  generator_forward(sequences[seed % 8192], out8);
  if (0) {
  //generator_forward_float(z, out8); // teste versão float (deve ser similar à fixed)
  for (int i = 0; i < 24 * 24; i++) {
    // opcional: aplica tanh aproximada (saturação) antes de converter
    printf("%6d ", out8[i]);
    if ((i % 24) == 23) printf("\n");

  }
}



  if (!write_pgm_u8(out_path, out8, 24, 24)) {
    fprintf(stderr, "Erro ao escrever %s\n", out_path);
    return 1;
  }

  //printf("OK: wrote %s (seed=%u)\n", out_path, seed);
  return 0;
}
