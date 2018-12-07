// 32 x 4 x 32K bit bitmap memory
#define VMAX (16 * 1024 * 1024)
#define BW 32
#define BATCH 4
#define BITMAP_DEPTH (VMAX / (BW * BATCH)) 
#define BUFFER_SIZE 16
#define BANK_BW   2 // 2 when pad = 4, 3 when pad == 8, 4 when pad == 16
#define BITSEL_BW 5 // 3 when data width is 8, 5 when data width is 32

#define BITMASK0  0x1
#define BITMASK1  0x2
#define BITMASK2  0x4
#define BITMASK3  0x8
#define BITMASK4  0x10
#define BITMASK5  0x20
#define BITMASK6  0x40
#define BITMASK7  0x80
#define BITMASK8  0x100
#define BITMASK9  0x200
#define BITMASKa  0x400
#define BITMASKb  0x800
#define BITMASKc  0x1000
#define BITMASKd  0x2000
#define BITMASKe  0x4000
#define BITMASKf  0x8000
#define BITMASK10 0x10000
#define BITMASK11 0x20000
#define BITMASK12 0x40000
#define BITMASK13 0x80000
#define BITMASK14 0x100000
#define BITMASK15 0x200000
#define BITMASK16 0x400000
#define BITMASK17 0x800000
#define BITMASK18 0x1000000
#define BITMASK19 0x2000000
#define BITMASK1a 0x4000000
#define BITMASK1b 0x8000000
#define BITMASK1c 0x10000000
#define BITMASK1d 0x20000000
#define BITMASK1e 0x40000000
#define BITMASK1f 0x80000000

//#define SW_EMU
#define BANK_MASK   0x3      //0000_0000_0000_0000_0000_0011
#define SEL_MASK    0x7C     //0000_0000_0000_0000_0111_1100
#define OFFSET_MASK 0xFFFF80 //1111_1111_1111_1111_1000_0000
#define CH_DEPTH    128
typedef uint        bmap_dt;

typedef int4 cia_dt;

inline int bitop(int bitsel){
	switch(bitsel){
		case 0:
			return BITMASK0;
		case 1:
			return BITMASK1;
		case 2:
			return BITMASK2;
		case 3:
			return BITMASK3;
		case 4:
			return BITMASK4;
		case 5:
			return BITMASK5;
		case 6:
			return BITMASK6;
		case 7:
			return BITMASK7;
		case 8:
			return BITMASK8;
		case 9:
			return BITMASK9;
		case 10:
			return BITMASKa;
		case 11:
			return BITMASKb;
		case 12:
			return BITMASKc;
		case 13: 
			return BITMASKd;
		case 14:
			return BITMASKe;
		case 15:
			return BITMASKf;
		case 16:
			return BITMASK10;
		case 17:
			return BITMASK11;
		case 18:
			return BITMASK12;
		case 19:
			return BITMASK13;
		case 20:
			return BITMASK14;
		case 21:
			return BITMASK15;
		case 22:
			return BITMASK16;
		case 23:
			return BITMASK17;
		case 24:
			return BITMASK18;
		case 25:
			return BITMASK19;
		case 26:
			return BITMASK1a;
		case 27:
			return BITMASK1b;
		case 28:
			return BITMASK1c;
		case 29:
			return BITMASK1d;
		case 30:
			return BITMASK1e;
		case 31:
			return BITMASK1f;
		default:
			return -1;
	}
}

channel int2 rpa_channel                     __attribute__ ((depth(CH_DEPTH)));   
channel cia_dt cia_channel                   __attribute__ ((depth(CH_DEPTH)));
channel int  cia_end_channel                 __attribute__ ((depth(4)));
channel int next_frontier_channel[BATCH]     __attribute__ ((depth(CH_DEPTH)));
channel int next_frontier_end_channel[BATCH] __attribute__ ((depth(4)));

__kernel void __attribute__((task)) read_rpa(
		__global int2* restrict rpa,
		const int frontier_size
		){
	for(int i = 0; i < frontier_size; i++){
		write_channel_altera(rpa_channel, rpa[i]);
	}

}
	
// Read cia of the frontier 
__kernel void __attribute__((task)) read_cia(
		__global cia_dt* restrict cia,
		const int               frontier_size
		)
{
	for(int i = 0; i < frontier_size; i++){
		int2 word  = read_channel_altera(rpa_channel);
		int  num   = (word.s1) >> BANK_BW;
		int  start = (word.s0) >> BANK_BW;

		for(int j = 0; j < num; j++){
			write_channel_altera(cia_channel, cia[start + j]);
		}
	}

	write_channel_altera(cia_end_channel, 1);
}

// Traverse cia 
__kernel void __attribute__((task)) traverse_cia(	
		__global int* restrict next_frontier_size,
		const int              root_vidx,
		const char             level
//#ifdef SW_EMU
//	  , __global bmap_dt* restrict bmap0,
//		__global bmap_dt* restrict bmap1,
//		__global bmap_dt* restrict bmap2,
//		__global bmap_dt* restrict bmap3
//#endif
		)
{
	 bmap_dt bitmap0[BITMAP_DEPTH];
	 bmap_dt bitmap1[BITMAP_DEPTH];
	 bmap_dt bitmap2[BITMAP_DEPTH];
	 bmap_dt bitmap3[BITMAP_DEPTH];

	 if(level == 0){
		 for (int i = 0; i < BITMAP_DEPTH; i++){
			 bitmap0[i] = 0;
			 bitmap1[i] = 0;
			 bitmap2[i] = 0;
			 bitmap3[i] = 0;
		 }

		 int bank_idx = root_vidx & BANK_MASK;
		 int bitsel   = (root_vidx & SEL_MASK) >> BANK_BW;
		 int offset   = (root_vidx & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);

		 if(bank_idx == 0) bitmap0[offset] = bitop(bitsel);
		 if(bank_idx == 1) bitmap1[offset] = bitop(bitsel);
		 if(bank_idx == 2) bitmap2[offset] = bitop(bitsel);
		 if(bank_idx == 3) bitmap3[offset] = bitop(bitsel);
	 }
//#ifdef SW_EMU
//	 else{
//		 for(int i = 0; i < BITMAP_DEPTH; i++){
//			 bitmap0[i] = bmap0[i];
//			 bitmap1[i] = bmap1[i];
//			 bitmap2[i] = bmap2[i];
//			 bitmap3[i] = bmap3[i];
//		 }
//	 }
//#endif
	 
	 int mem_addr0 = 0;
	 int mem_addr1 = 0;
	 int mem_addr2 = 0;
	 int mem_addr3 = 0;

	 bool data_valid = false;
	 bool flag_valid = false;
	 int  flag = 0;
	 while(true){

		 cia_dt word = read_channel_nb_altera(cia_channel, &data_valid);
		 if(data_valid){
			 int bitsel0 = (word.s0 & SEL_MASK) >> BANK_BW;
			 int bitsel1 = (word.s1 & SEL_MASK) >> BANK_BW;
			 int bitsel2 = (word.s2 & SEL_MASK) >> BANK_BW;
			 int bitsel3 = (word.s3 & SEL_MASK) >> BANK_BW;

			 int offset0 = (word.s0 & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			 int offset1 = (word.s1 & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			 int offset2 = (word.s2 & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			 int offset3 = (word.s3 & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);

			 if(word.s0 != -1){
				 bmap_dt tmp = bitmap0[offset0]; 
				 bmap_dt mask = bitop(bitsel0); 
				 if((tmp & mask) == 0){
					 write_channel_altera(next_frontier_channel[0], word.s0);
					 bitmap0[offset0] = tmp | mask;
					 mem_addr0++;
				 }
			 }

			 if(word.s1 != -1){
				 bmap_dt tmp = bitmap1[offset1]; 
				 bmap_dt mask = bitop(bitsel1); 
				 if((tmp & mask) == 0){
					 write_channel_altera(next_frontier_channel[1], word.s1);
					 bitmap1[offset1] = tmp | mask;
					 mem_addr1++;
				 }
			 }

			 if(word.s2 != -1){
				 bmap_dt tmp = bitmap2[offset2]; 
				 bmap_dt mask = bitop(bitsel2); 
				 if((tmp & mask) == 0){
					 write_channel_altera(next_frontier_channel[2], word.s2);
					 bitmap2[offset2] = tmp | mask;
					 mem_addr2++;
				 }
			 }

			 if(word.s3 != -1){
				 bmap_dt tmp = bitmap3[offset3]; 
				 bmap_dt mask = bitop(bitsel3); 
				 if((tmp & mask) == 0){
					 write_channel_altera(next_frontier_channel[3], word.s3);
					 bitmap3[offset3] = tmp | mask;
					 mem_addr3++;
				 }
			 }
		 }

		 // Read flag channel
		 int flag_tmp = read_channel_nb_altera(cia_end_channel, &flag_valid);
		 if(flag_valid) flag = flag_tmp;
		 if(flag == 1 && !data_valid && !flag_valid){
			 break;
		 }
	 }
//#ifdef SW_EMU
//	 for(int i = 0; i < BITMAP_DEPTH; i++){
//		 bmap0[i] = bitmap0[i];
//		 bmap1[i] = bitmap1[i];
//		 bmap2[i] = bitmap2[i];
//		 bmap3[i] = bitmap3[i];
//	 }
//#endif

	 write_channel_altera(next_frontier_end_channel[0], 1);
	 write_channel_altera(next_frontier_end_channel[1], 1);
	 write_channel_altera(next_frontier_end_channel[2], 1);
	 write_channel_altera(next_frontier_end_channel[3], 1);

	 *(next_frontier_size  + 0)  = mem_addr0;
	 *(next_frontier_size  + 1)  = mem_addr1;
	 *(next_frontier_size  + 2)  = mem_addr2;
	 *(next_frontier_size  + 3)  = mem_addr3;

}

// write channel0
__kernel void __attribute__ ((task)) write_frontier0(
		__global int* restrict next_frontier0
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int  global_idx = 0;
	int  flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[0], &data_valid);
		if(data_valid){
			next_frontier0[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[0], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}

}

// write channel 1
__kernel void __attribute__ ((task)) write_frontier1(
		__global int* restrict next_frontier1
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int  global_idx = 0;
	int  flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[1], &data_valid);
		if(data_valid){
			next_frontier1[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[1], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 2
__kernel void __attribute__ ((task)) write_frontier2(
		__global int* restrict next_frontier2
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int  global_idx = 0;
	int  flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[2], &data_valid);
		if(data_valid){
			next_frontier2[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[2], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 3
__kernel void __attribute__ ((task)) write_frontier3(
		__global int* restrict next_frontier3
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int  global_idx = 0;
	int  flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[3], &data_valid);
		if(data_valid){
			next_frontier3[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[3], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

