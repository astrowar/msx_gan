// ____________________________
// ██▀▀█▀▀██▀▀▀▀▀▀▀█▀▀█        │   ▄▄▄                ▄▄
// ██  ▀  █▄  ▀██▄ ▀ ▄█ ▄▀▀ █  │  ▀█▄  ▄▀██ ▄█▄█ ██▀▄ ██  ▄███
// █  █ █  ▀▀  ▄█  █  █ ▀▄█ █▄ │  ▄▄█▀ ▀▄██ ██ █ ██▀  ▀█▄ ▀█▄▄
// ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀────────┘                 ▀▀
//  Program template
// ─────────────────────────────────────────────────────────────────────────────

//=============================================================================
//=============================================================================
// ENUMS PARA CATEGORIAS
//=============================================================================

typedef enum {
  GENDER_MAN = 0,
  GENDER_WOMAN = 1
} Gender;

typedef enum {
  HAIR_STYLE_LONG = 0,
  HAIR_STYLE_MEDIUM = 1,
  HAIR_STYLE_SHORT = 2
} HairStyle;

typedef enum {
  HAIR_TONE_BLACK = 0,
  HAIR_TONE_BROWN_DARK = 1,
  HAIR_TONE_BROWN_LIGHT = 2,
  HAIR_TONE_LIGHT = 3
} HairTone;

typedef enum {
  HAIR_TYPE_BALD = 0,
  HAIR_TYPE_CURLY = 1,
  HAIR_TYPE_STRAIGHT = 2,
  HAIR_TYPE_WAVY = 3
} HairType;

typedef enum {
  SKIN_TONE_DARK = 0,
  SKIN_TONE_MEDIUM = 1
} SkinTone;
// INCLUDES
//=============================================================================
#include "msxgl.h"
#include "vdp.h"
//=============================================================================
// DEFINES
//=============================================================================

// Library's logo
#define MSX_GL "\x01\x02\x03\x04\x05\x06"

//=============================================================================
// READ-ONLY DATA
//=============================================================================

// Fonts data
#include "font_gray.h"

#define GEN_OC_BLK 8

 

// base generator 32x32
#define G_NW   16
#define G_NZ   64
#define G_C2   22
#define G_C6   22
#define G_C12  6
#define G_C24  6
#define G_CH   4

#include "layers.h"

#define LOGLINE (13)
  
 

static u8  tanh_lut[] = {
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 
  ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0
   ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,3 ,4 ,5 ,7 ,9 ,12 ,15 ,19 ,24 ,30 ,37 ,46 ,57 ,68 ,82 ,96 ,112 ,128
    ,143 ,159 ,173 ,187 ,198 ,209 ,218 ,225 ,231 ,236 ,240 ,243 ,246 ,248 ,250 ,251 ,252 ,253 ,253 ,254 ,254 ,254 ,255 ,255 ,255 ,255
     ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255
      ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255
       ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255
        ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255
};

 

void generator_forward_int8(const i8* z_s8, u8* out24_u8)
{
    const int ZN = G_NW * 2 * 2;

    // 1. Identifica todos os tamanhos necessários em bytes
    u16 sizes[] = {
        (u16)G_NW  * 2 *2 ,      // x2
        (u16)G_C2  * 2 * 2 ,      // y2
        (u16)G_C2  * 6 * 6 ,     // up6
        (u16)G_C6  * 6 * 6 ,     // y6
        (u16)G_C6  * 12 * 12 ,    // up12
        (u16)G_C12 * 12 * 12 ,    // y12
        (u16)G_C12 * 24 * 24 ,    // up24
        (u16)G_C24 * 24 * 24 ,    // y24a
        (u16)G_CH  * 24 * 24,    // y24b
        (u16)24 * 24             // yout (24*24)
    };

    // 2. Encontra o maior tamanho de buffer exigido por qualquer camada
    u16 max_size = 0;
    for (int i = 0; i < 10; ++i) {
        if (sizes[i] > max_size) {
            max_size = sizes[i];
        }
    }

    // Alinhamento de memória (Muito importante!): 
    // Garante que o tamanho seja múltiplo de 16/32 bytes caso suas 
    // funções conv3x3_i8 usem instruções vetorizadas (SIMD / NEON / AVX)
    max_size = (max_size + 31) & ~31; 

    // 3. Aloca um ÚNICO bloco capaz de suportar as duas maiores camadas simultâneas
    //i8* arena = (i8*)malloc(max_size * 2);
     u8 *arena = Mem_HeapAlloc(max_size * 2); // aloca arena dinamicamente
    if (!arena) {
       // fprintf(stderr, "generator_forward_int8: malloc failed\n");
        return;
    }

    // 4. Divide a arena em dois buffers (Ping-Pong)
    i8* bufA = arena;
    i8* bufB = arena + max_size;

    // 5. Mapeia os ponteiros originais alternando entre A e B
    // Dessa forma você não precisa mudar nada nas chamadas das suas funções!
    i8* x2   = bufA;
    i8* y2   = bufB;
    i8* up6  = bufA;
    i8* y6   = bufB;
    i8* up12 = bufA;
    i8* y12  = bufB;
    i8* up24 = bufA;
    i8* y24a = bufB;
    i8* y24b = bufA;
    i8* yout = bufB;

    // ---- Início do seu código de processamento (sem alterações) ----
    
    // 1) z int16 -> int8 (ajuste a escala conforme seu z real)
    for (i8 i = 0; i < ZN; ++i) {
   
        x2[i] = z_s8[i];
    }
    Print_DrawChar('.'); 
    // l0: (G_NW,2,2) -> (G_C2,2,2)
    //conv3x3_i8(x2,  G_NW,  2,  2, W_l0,  SCALEW_L0_Q15,  B_l0_f16 ,  G_C2,  y2,   1);
    // up 2->6
    //upsample_nearest_i8(y2, G_C2,  2,  2, up6,  6,  6);
    process_layer_1(x2, y2, up6);
    Print_DrawChar('.'); 
    // l1
    //conv3x3_i8(up6, G_C2,  6,  6, W_l1,  SCALEW_L1_Q15,  B_l1_f16 ,  G_C6,  y6,   1);
    // up 6->12
    //upsample_nearest_i8(y6, G_C6,  6,  6, up12, 12, 12);
    process_layer_2(up6, y6, up12);
    Print_DrawChar('.'); 
    // l2
    //conv3x3_i8(up12, G_C6, 12, 12, W_l2,  SCALEW_L2_Q15,  B_l2_f16 ,  G_C12, y12,  1);
    // up 12->24
    //upsample_nearest_i8(y12, G_C12, 12, 12, up24, 24, 24);
    process_layer_3(up12, y12, up24);
    Print_DrawChar('.'); 
    // l3
    //conv3x3_i8(up24, G_C12, 24, 24, W_l3,  SCALEW_L3_Q15,  B_l3_f16 ,  G_C24, y24a, 1);
    process_layer_4(up24, y24a);
    Print_DrawChar('.'); 
    // l4
    //conv3x3_i8(y24a, G_C24, 24, 24, W_l4,  SCALEW_L4_Q15,  B_l4_f16 ,  G_CH,  y24b, 1);
    process_layer_5(y24a, y24b);
    Print_DrawChar('.'); 
    // out
    //conv3x3_i8(y24b, G_CH, 24, 24, W_out, SCALEW_OUT_Q15, B_out_f16 , 1, yout, 0);
    process_layer_6(y24b, yout);
    Print_DrawChar('.'); 
    
    // tanh + export
    for (i16 i = 0; i < 24 * 24; ++i) {
        i16 index = (i16)yout[i] + 128; // converte de [-128,127] para [0,255]
        if (index < 0) index = 0;
        if (index > 255) index = 255;
        out24_u8[i] = (u8) tanh_lut[ (u8)index]  ; // converte de [-128,127] para [0,255]
    }

    // ---- Fim do processamento ----

    // Desaloca a arena inteira de uma só vez
   // free(arena);
    Mem_HeapFree((u16)arena);
}

 
#define CC_WHITE ((u8)15)
#define CC_GRAY ((u8)14)
#define CC_BLACK ((u8)1)

// Seus chars de dither
#define CH_D0 ((u8)156)
#define CH_D100 ((u8)163)

// v: 0=branco, 255=preto
// bg FIXO: você passa e ele NÃO muda.
void Print_DrawGray(u8 v)
{

  // Quantiza em 8 níveis: 0,15,30,45,60,75,90,100
  // Índice 0..7 -> char 156..163
  u8 idx = (u8)(((u16)v * 7u + 127u) / 255u); // arredondado
  u8 ch = (u8)(CH_D0 + idx);
  Print_DrawChar(ch);
}


 
 __at(0xFC9E) u16 time_var;

void waitForNoKey()
{
 
 // for (u16 j = 0; j < 2747; ++j)
 /// {                 // 1 frame de delay (aprox 1/60s a 3.58MHz)
  //  __asm__("nop"); // delay curto
 // }
  //while (Bios_HasCharacter() != 0)
  //{
   //Bios_GetCharacter();
  //}
 
  u16 next_time = time_var +8 ; //30 frames de espera (aprox 0.5s a 60fps)
   
  while (time_var < next_time)
  {
     __asm__("nop"); // delay curto
  }
}

 
static const i8 urand[] =  {-4, 9, 0, -14, -4, -5, 26, -43, -12, 24, -12, -33, -22, -31, -17, 10, -12, -12, 29, 5, -16, -25, 12, -8, -6, 0, -15, -19, 11, -8, 12, 0, 26, 5, 27, 4, 0, -7, -20, 0, 8, 5, 3, -8, -1, 0, 19, -11, 26, 0, -5, -4, 24, -7, 8, -51, -18, -9, 9, 7, -8, -2, 17, 6, -4, 20, 5, -13, 0, 5, 5, 17, 0, -1, -6, -13, -8, -21, 1, 8, 31, 30, -1, 29, 7, 6, -10, 2, -1, -1, 8, 30, 11, -6, 3, 12, -18, 36, 11, 17, 11, 6, -8, -11, 3, 18, 6, 2, -6, -9, -23, -7, 26, 32, -9, 2, 13, -3, -2, -12, -8, -10, 7, -25, -12, -15, -17, 0, -2, -13, -2, 25, 11, 7, 7, 19, 5, 18, -14, 16, -18, 35, -46, -17, 0, -14, -4, -11, -12, -36, 4, 13, -12, -5, 23, 4, -1, -14, -37, 14, 13, 9, -10, 2, 8, 15, 11, 12, 33, 16, -22, 7, 15, 2, -2, 1, 11, 10, -15, -14, -10, -14, 31, -22, -28, -30, -27, -7, 0, 16, 15, -1, 0, 8, 0, 22, 21, -3, 2, -20, 3, -25, 18, -1, 2, 0, -1, 25, 14, 27, 6, -23, 25, -19, 24, 31, -7, 6, -22, -40, -4, -13, -21, 9, 3, -20, 9, -22, -15, 33, -3, 15, -4, -11, 12, -7, 3, 14, -15, 5, -11, 2, 22, 14, 1, 29, -17, -1, -14, -24, 5, -19, 5, -10, 3, 13};

 

void  getUserSeed( i8 z_s8[64])
{
   u16 offset = time_var ;
   for(u8 i = 0; i < 64; ++i)
   {
     z_s8[i] = urand[(offset + i) & 255];
   }
}

#include "zmeans.h"

void generate_z_from_user_input(i8 z_s8[64], Gender gender_in , HairStyle hair_style_in , HairTone hair_tone_in , HairType hair_type_in , SkinTone skin_tone_in )
{  
   u16 offset = time_var ;
  for (u8 i = 0; i < 64; ++i)
  {
    i8 zi =  (gender_in == GENDER_MAN) ? z_gender_man[i] : z_gender_woman[i];

    //Hair Style
    if (hair_style_in == HAIR_STYLE_LONG) {
      zi += 2* z_hair_style_long[i];
    } else if (hair_style_in == HAIR_STYLE_MEDIUM) {
      zi += 2*z_hair_style_medium[i];
    } else {
      zi += 2*z_hair_style_short[i];
    }
    //Hair Tone
    if (hair_tone_in == HAIR_TONE_BLACK) {
      zi += z_hair_tone_black[i];
    } else if (hair_tone_in == HAIR_TONE_BROWN_DARK) {
      zi += z_hair_tone_brown_dark[i];
    } else if (hair_tone_in == HAIR_TONE_BROWN_LIGHT) {
      zi += z_hair_tone_brown_light[i];
    } else {
      zi += z_hair_tone_light[i];
    }

    //Hair Type
    if (hair_type_in == HAIR_TYPE_BALD) {
      zi += 2*z_hair_type_bald[i];
    } else if (hair_type_in == HAIR_TYPE_CURLY) {
      zi += z_hair_type_curly[i];
    } else if (hair_type_in == HAIR_TYPE_STRAIGHT) {
      zi += z_hair_type_straight[i];
    } else {
      zi += z_hair_type_wavy[i];
    }

    
    //Skin Tone
    if (skin_tone_in == SKIN_TONE_DARK) {
      zi += 1.4*z_skin_tone_dark[i];
    } else {
      zi += z_skin_tone_medium[i] ; // ajuste para evitar que o tom médio fique muito claro
    }

    zi += urand[(offset + i) & 255]/2; // adiciona um pouco de aleatoriedade para diversidade dentro da mesma categoria

    z_s8[i] = zi >> 2 ;
  }
}
 
 
//=============================================================================

 

//=============================================================================
// MAIN LOOP
//=============================================================================
 

//===========================================================================
//-----------------------------------------------------------------------------
/// Program entry point
void main()
{
  VDP_SetMode(VDP_MODE_SCREEN1);
  VDP_EnableVBlank(TRUE);
  VDP_ClearVRAM();

  Print_SetTextFont(g_Font_MGL_Sample8, 1);
  Print_SetColor(COLOR_WHITE, COLOR_BLACK);
  Print_SetPosition(0, 0);
  Print_DrawText(MSX_GL " GAN");


 

  i8 *in_z;           // buffer de output para a imagem 32x32 final
  u8 *out_u8_nc_24_24; // buffer de output para a imagem 32x32 final

  in_z = Mem_HeapAlloc(64  );
  out_u8_nc_24_24 = Mem_HeapAlloc(24 * 24);
 

  Print_SetPosition(0, 4);
  Print_DrawText("Select:");

  for (u8 x = 0; x < 24; ++x)
  {
    for (u8 y = 0; y < 24; ++y)
    {
      // fill an diagonal shade in the output buffer for testing
      out_u8_nc_24_24[y * 24 + x] = (x + y) * 255 / (24 + 24 - 2); // valor de cinza baseado na posição (0 a 255)
    }
  }

  u8 has_gen = 0 ;
  
  Gender gender = GENDER_WOMAN;
  HairStyle hair_style = HAIR_STYLE_LONG;
  HairTone hair_tone = HAIR_TONE_BROWN_DARK;
  HairType hair_type = HAIR_TYPE_STRAIGHT;
  SkinTone skin_tone = SKIN_TONE_MEDIUM;
  int category_selection = 0; // 0: gender, 1: hair_style, 2: hair_tone, 3: hair_type, 4: skin_tone, 5: random


  
  while(has_gen == 0) {
    int hasChange = 1;
    if (Keyboard_IsKeyPressed(KEY_UP)) {
      category_selection = (category_selection - 1);
      if (category_selection < 0) category_selection = 5;
      hasChange = 1;
    } else if (Keyboard_IsKeyPressed(KEY_DOWN)) {
      category_selection = (category_selection + 1);
      if (category_selection > 5) category_selection = 0;
      hasChange = 1;
    }
    else  if (Keyboard_IsKeyPressed(KEY_LEFT)) {
      if (category_selection == 0) { gender = 1 - gender; }
      else if (category_selection == 1) { hair_style = (hair_style + 2) % 3; }
      else if (category_selection == 2) { hair_tone = (hair_tone + 3) % 4; }
      else if (category_selection == 3) { hair_type = (hair_type + 3) % 4; }
      else if (category_selection == 4) { skin_tone = 1 - skin_tone; }
      // random não altera nada
      hasChange = 1;
    }
    else if (Keyboard_IsKeyPressed(KEY_RIGHT)) {
      if (category_selection == 0) { gender = 1 - gender; }
      else if (category_selection == 1) { hair_style = (hair_style + 1) % 3; }
      else if (category_selection == 2) { hair_tone = (hair_tone + 1) % 4; }
      else if (category_selection == 3) { hair_type = (hair_type + 1) % 4; }
      else if (category_selection == 4) { skin_tone = 1 - skin_tone; }
      // random não altera nada
      hasChange = 1;
    }
    else if (Keyboard_IsKeyPressed(KEY_ENTER)) {
      Print_SetPosition(0, 4);
      Print_DrawText("            ");
      Print_SetPosition(0, 4);
      Print_DrawText("PROCESSING");
      if (category_selection == 5) {
        getUserSeed(in_z);
      } else {
        generate_z_from_user_input(in_z, gender, hair_style, hair_tone, hair_type, skin_tone);
      }
      generator_forward_int8(in_z, out_u8_nc_24_24);
      has_gen = 1;
    }
    if (hasChange == 1) {
      int ROWSTART = 13;
      Print_SetPosition(0, ROWSTART);
      if (category_selection == 0) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Gender: ");
      if (gender == GENDER_MAN) Print_DrawText("Man   "); else Print_DrawText("Woman ");
      Print_SetPosition(0, ROWSTART + 1);
      if (category_selection == 1) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Hair Style: ");
      if (hair_style == HAIR_STYLE_LONG) Print_DrawText("Long   ");
      else if (hair_style == HAIR_STYLE_MEDIUM) Print_DrawText("Medium ");
      else Print_DrawText("Short  ");
      Print_SetPosition(0, ROWSTART + 2);
      if (category_selection == 2) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Hair Tone: ");
      if (hair_tone == HAIR_TONE_BLACK) Print_DrawText("Black      ");
      else if (hair_tone == HAIR_TONE_BROWN_DARK) Print_DrawText("Brown Dark ");
      else if (hair_tone == HAIR_TONE_BROWN_LIGHT) Print_DrawText("Brown Light");
      else Print_DrawText("Light      ");
      Print_SetPosition(0, ROWSTART + 3);
      if (category_selection == 3) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Hair Type: ");
      if (hair_type == HAIR_TYPE_BALD) Print_DrawText("Bald    ");
      else if (hair_type == HAIR_TYPE_CURLY) Print_DrawText("Curly   ");
      else if (hair_type == HAIR_TYPE_STRAIGHT) Print_DrawText("Straight");
      else Print_DrawText("Wavy    ");
      Print_SetPosition(0, ROWSTART + 4);
      if (category_selection == 4) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Skin Tone: ");
      if (skin_tone == SKIN_TONE_DARK) Print_DrawText("Dark  "); else Print_DrawText("Medium");
      Print_SetPosition(0, ROWSTART + 7);
      if (category_selection == 5) Print_DrawChar('*'); else Print_DrawChar(' ');
      Print_DrawText("Random");
      waitForNoKey();
      hasChange = 0;
    }
  }

  for (u8 y = 0; y < 24; ++y)
  {
    for (u8 x = 0; x < 24; ++x)
    {
      Print_SetPosition(x, y);
      u8 z = out_u8_nc_24_24[((y) * 24) + x];
      Print_DrawGray(z);
    }
  }

  while (!Keyboard_IsKeyPressed(KEY_SPACE))
  {
    Print_SetPosition(39, 0);
    Print_DrawChar('/');
    Halt();
    Print_SetPosition(39, 0);
    Print_DrawChar('\\');
    Halt();
  }

  Bios_Exit(0);
}
