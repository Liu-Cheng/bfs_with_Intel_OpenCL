//#define SW_EMU
#define VMAX (4 * 1024 * 1024)
#define BW 16
#define BATCH 16
#define BITMAP_DEPTH (VMAX / (BW * BATCH)) 
#define BUFFER_SIZE 16
#define BANK_BW 4   // 3 when pad == 8, 4 when batch == 16
#define BITSEL_BW 4 // 4 when data with of the buffer is 16

#define BITMASK0 0x1
#define BITMASK1 0x2
#define BITMASK2 0x4
#define BITMASK3 0x8
#define BITMASK4 0x10
#define BITMASK5 0x20
#define BITMASK6 0x40
#define BITMASK7 0x80
#define BITMASK8 0x100
#define BITMASK9 0x200
#define BITMASKa 0x400
#define BITMASKb 0x800
#define BITMASKc 0x1000
#define BITMASKd 0x2000
#define BITMASKe 0x4000
#define BITMASKf 0x8000

#define BANK_MASK 0xf        //0000_0000_0000_0000_0000_1111
#define SEL_MASK  0xf0       //0000_0000_0000_0000_1111_0000
#define OFFSET_MASK 0x7FFF00 //0111_1111_1111_1111_0000_0000
#define CH_DEPTH 64
typedef ushort bmap_dt;
typedef int16 cia_dt;

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
		default:
			return -1;
	}
}

channel int    frontier_channel                 __attribute__ ((depth(32)));
channel int2   rpa_channel                      __attribute__ ((depth(CH_DEPTH)));   
channel cia_dt cia_channel                      __attribute__ ((depth(32)));
channel int    cia_end_channel                  __attribute__ ((depth(4)));
channel int    next_frontier_channel[BATCH]     __attribute__ ((depth(CH_DEPTH)));
channel int    next_frontier_end_channel[BATCH] __attribute__ ((depth(4)));

__kernel void __attribute__((task)) read_frontier(
		__global const int* restrict frontier, 
		__global int* restrict frontierSize
		)
{
	for(int i = 0; i < frontierSize[0]; i++){
		int vertexIdx = frontier[i];
		write_channel_altera(frontier_channel, vertexIdx);
	}

}

__kernel void __attribute__((task)) read_rpa(
		__global int2* restrict rpaInfo,
		__global int* restrict frontierSize
		)
{
	for(int i = 0; i < frontierSize[0]; i++){
		int vertexIdx = read_channel_altera(frontier_channel);
		int2 data = rpaInfo[vertexIdx];
		write_channel_altera(rpa_channel, data);
	}
}


// Read cia of the frontier 
__kernel void __attribute__((task)) read_cia(
		__global cia_dt* restrict cia,
		__global int*    restrict frontier_size
		)
{
	for(int i = 0; i < frontier_size[0]; i++){
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
		__global int* restrict  next_frontier_size,
		const int               root_vidx,
		__global char* restrict level
//#ifdef SW_EMU
//	   ,__global bmap_dt* restrict bmap0,
//		__global bmap_dt* restrict bmap1,
//		__global bmap_dt* restrict bmap2,
//		__global bmap_dt* restrict bmap3,
//		__global bmap_dt* restrict bmap4,
//		__global bmap_dt* restrict bmap5,
//		__global bmap_dt* restrict bmap6,
//		__global bmap_dt* restrict bmap7,
//		__global bmap_dt* restrict bmap8,
//		__global bmap_dt* restrict bmap9,
//		__global bmap_dt* restrict bmap10,
//		__global bmap_dt* restrict bmap11,
//		__global bmap_dt* restrict bmap12,
//		__global bmap_dt* restrict bmap13,
//		__global bmap_dt* restrict bmap14,
//		__global bmap_dt* restrict bmap15
//#endif
		)
{
	bmap_dt bitmap0[BITMAP_DEPTH];
	bmap_dt bitmap1[BITMAP_DEPTH];
	bmap_dt bitmap2[BITMAP_DEPTH];
	bmap_dt bitmap3[BITMAP_DEPTH];
	bmap_dt bitmap4[BITMAP_DEPTH];
	bmap_dt bitmap5[BITMAP_DEPTH];
	bmap_dt bitmap6[BITMAP_DEPTH];
	bmap_dt bitmap7[BITMAP_DEPTH];

	bmap_dt bitmap8[BITMAP_DEPTH];
	bmap_dt bitmap9[BITMAP_DEPTH];
	bmap_dt bitmap10[BITMAP_DEPTH];
	bmap_dt bitmap11[BITMAP_DEPTH];
	bmap_dt bitmap12[BITMAP_DEPTH];
	bmap_dt bitmap13[BITMAP_DEPTH];
	bmap_dt bitmap14[BITMAP_DEPTH];
	bmap_dt bitmap15[BITMAP_DEPTH];

	if(level[0] == 0){
		for (int i = 0; i < BITMAP_DEPTH; i++){
			bitmap0[i] = 0;
			bitmap1[i] = 0;
			bitmap2[i] = 0;
			bitmap3[i] = 0;
			bitmap4[i] = 0;
			bitmap5[i] = 0;
			bitmap6[i] = 0;
			bitmap7[i] = 0;

			bitmap8[i] = 0;
			bitmap9[i] = 0;
			bitmap10[i] = 0;
			bitmap11[i] = 0;
			bitmap12[i] = 0;
			bitmap13[i] = 0;
			bitmap14[i] = 0;
			bitmap15[i] = 0;
		}

		int bank_idx = root_vidx & BANK_MASK;
		int bitsel   = (root_vidx & SEL_MASK) >> BANK_BW;
		int offset   = (root_vidx & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);

		if(bank_idx == 0)  bitmap0[offset]  = bitop(bitsel);
		if(bank_idx == 1)  bitmap1[offset]  = bitop(bitsel);
		if(bank_idx == 2)  bitmap2[offset]  = bitop(bitsel);
		if(bank_idx == 3)  bitmap3[offset]  = bitop(bitsel);
		if(bank_idx == 4)  bitmap4[offset]  = bitop(bitsel);
		if(bank_idx == 5)  bitmap5[offset]  = bitop(bitsel);
		if(bank_idx == 6)  bitmap6[offset]  = bitop(bitsel);
		if(bank_idx == 7)  bitmap7[offset]  = bitop(bitsel);
		if(bank_idx == 8)  bitmap8[offset]  = bitop(bitsel);
		if(bank_idx == 9)  bitmap9[offset]  = bitop(bitsel);
		if(bank_idx == 10) bitmap10[offset] = bitop(bitsel);
		if(bank_idx == 11) bitmap11[offset] = bitop(bitsel);
		if(bank_idx == 12) bitmap12[offset] = bitop(bitsel);
		if(bank_idx == 13) bitmap13[offset] = bitop(bitsel);
		if(bank_idx == 14) bitmap14[offset] = bitop(bitsel);
		if(bank_idx == 15) bitmap15[offset] = bitop(bitsel);
	}
//#ifdef SW_EMU
//	else{
//		for(int i = 0; i < BITMAP_DEPTH; i++){
//			bitmap0[i] = bmap0[i];
//			bitmap1[i] = bmap1[i];
//			bitmap2[i] = bmap2[i];
//			bitmap3[i] = bmap3[i];
//			bitmap4[i] = bmap4[i];
//			bitmap5[i] = bmap5[i];
//			bitmap6[i] = bmap6[i];
//			bitmap7[i] = bmap7[i];
//
//			bitmap8[i] = bmap8[i];
//			bitmap9[i] = bmap9[i];
//			bitmap10[i] = bmap10[i];
//			bitmap11[i] = bmap11[i];
//			bitmap12[i] = bmap12[i];
//			bitmap13[i] = bmap13[i];
//			bitmap14[i] = bmap14[i];
//			bitmap15[i] = bmap15[i];
//		}
//	}
//#endif

	int mem_addr0 = 0; int mem_addr8 = 0;
	int mem_addr1 = 0; int mem_addr9 = 0;
	int mem_addr2 = 0; int mem_addr10 = 0;
	int mem_addr3 = 0; int mem_addr11 = 0;
	int mem_addr4 = 0; int mem_addr12 = 0;
	int mem_addr5 = 0; int mem_addr13 = 0;
	int mem_addr6 = 0; int mem_addr14 = 0;
	int mem_addr7 = 0; int mem_addr15 = 0;

	bool data_valid = false;
	bool flag_valid = false;
	int  flag = 0;
	while(true){

		cia_dt word = read_channel_nb_altera(cia_channel, &data_valid);
		if(data_valid){
			int bitsel0  = (word.s0 & SEL_MASK) >> BANK_BW;
			int bitsel1  = (word.s1 & SEL_MASK) >> BANK_BW;
			int bitsel2  = (word.s2 & SEL_MASK) >> BANK_BW;
			int bitsel3  = (word.s3 & SEL_MASK) >> BANK_BW;
			int bitsel4  = (word.s4 & SEL_MASK) >> BANK_BW;
			int bitsel5  = (word.s5 & SEL_MASK) >> BANK_BW;
			int bitsel6  = (word.s6 & SEL_MASK) >> BANK_BW;
			int bitsel7  = (word.s7 & SEL_MASK) >> BANK_BW;
			int bitsel8  = (word.s8 & SEL_MASK) >> BANK_BW;
			int bitsel9  = (word.s9 & SEL_MASK) >> BANK_BW;
			int bitsel10 = (word.sa & SEL_MASK) >> BANK_BW;
			int bitsel11 = (word.sb & SEL_MASK) >> BANK_BW;
			int bitsel12 = (word.sc & SEL_MASK) >> BANK_BW;
			int bitsel13 = (word.sd & SEL_MASK) >> BANK_BW;
			int bitsel14 = (word.se & SEL_MASK) >> BANK_BW;
			int bitsel15 = (word.sf & SEL_MASK) >> BANK_BW;

			int offset0  = (word.s0  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset1  = (word.s1  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset2  = (word.s2  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset3  = (word.s3  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset4  = (word.s4  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset5  = (word.s5  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset6  = (word.s6  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset7  = (word.s7  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset8  = (word.s8  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset9  = (word.s9  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset10 = (word.sa  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset11 = (word.sb  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset12 = (word.sc  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset13 = (word.sd  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset14 = (word.se  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);
			int offset15 = (word.sf  & OFFSET_MASK) >> (BANK_BW + BITSEL_BW);

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

			if(word.s4 != -1){
				bmap_dt tmp = bitmap4[offset4]; 
				bmap_dt mask = bitop(bitsel4); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[4], word.s4);
					bitmap4[offset4] = tmp | mask;
					mem_addr4++;
				}
			}

			if(word.s5 != -1){
				bmap_dt tmp = bitmap5[offset5]; 
				bmap_dt mask = bitop(bitsel5);
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[5], word.s5);
					bitmap5[offset5] = tmp | mask;
					mem_addr5++;
				}
			}

			if(word.s6 != -1){
				bmap_dt tmp = bitmap6[offset6]; 
				bmap_dt mask = bitop(bitsel6); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[6], word.s6);
					bitmap6[offset6] = tmp | mask;
					mem_addr6++; 
				}
			}

			if(word.s7 != -1){
				bmap_dt tmp = bitmap7[offset7]; 
				bmap_dt mask = bitop(bitsel7); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[7], word.s7);
					bitmap7[offset7] = tmp | mask;
					mem_addr7++;
				}
			}

			if(word.s8 != -1){
				bmap_dt tmp = bitmap8[offset8]; 
				bmap_dt mask = bitop(bitsel8); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[8], word.s8);
					bitmap8[offset8] = tmp | mask;
					mem_addr8++;
				}
			}

			if(word.s9 != -1){
				bmap_dt tmp = bitmap9[offset9]; 
				bmap_dt mask = bitop(bitsel9); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[9], word.s9);
					bitmap9[offset9] = tmp | mask;
					mem_addr9++;
				}
			}

			if(word.sa != -1){
				bmap_dt tmp = bitmap10[offset10]; 
				bmap_dt mask = bitop(bitsel10); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[10], word.sa);
					bitmap10[offset10] = tmp | mask;
					mem_addr10++;
				}
			}

			if(word.sb != -1){
				bmap_dt tmp = bitmap11[offset11]; 
				bmap_dt mask = bitop(bitsel11); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[11], word.sb);
					bitmap11[offset11] = tmp | mask;
					mem_addr11++;
				}
			}

			if(word.sc != -1){
				bmap_dt tmp = bitmap12[offset12]; 
				bmap_dt mask = bitop(bitsel12); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[12], word.sc);
					bitmap12[offset12] = tmp | mask;
					mem_addr12++;
				}
			}

			if(word.sd != -1){
				bmap_dt tmp = bitmap13[offset13]; 
				bmap_dt mask = bitop(bitsel13);
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[13], word.sd);
					bitmap13[offset13] = tmp | mask;
					mem_addr13++;
				}
			}

			if(word.se != -1){
				bmap_dt tmp = bitmap14[offset14]; 
				bmap_dt mask = bitop(bitsel14); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[14], word.se);
					bitmap14[offset14] = tmp | mask;
					mem_addr14++; 
				}
			}

			if(word.sf != -1){
				bmap_dt tmp = bitmap15[offset15]; 
				bmap_dt mask = bitop(bitsel15); 
				if((tmp & mask) == 0){
					write_channel_altera(next_frontier_channel[15], word.sf);
					bitmap15[offset15] = tmp | mask;
					mem_addr15++;
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
//	for(int i = 0; i < BITMAP_DEPTH; i++){
//		bmap0[i]  = bitmap0[i];
//		bmap1[i]  = bitmap1[i];
//		bmap2[i]  = bitmap2[i];
//		bmap3[i]  = bitmap3[i];
//		bmap4[i]  = bitmap4[i];
//		bmap5[i]  = bitmap5[i];
//		bmap6[i]  = bitmap6[i];
//		bmap7[i]  = bitmap7[i];
//
//		bmap8[i]  = bitmap8[i];
//		bmap9[i]  = bitmap9[i];
//		bmap10[i] = bitmap10[i];
//		bmap11[i] = bitmap11[i];
//		bmap12[i] = bitmap12[i];
//		bmap13[i] = bitmap13[i];
//		bmap14[i] = bitmap14[i];
//		bmap15[i] = bitmap15[i];
//	}
//#endif

	write_channel_altera(next_frontier_end_channel[0], 1);
	write_channel_altera(next_frontier_end_channel[1], 1);
	write_channel_altera(next_frontier_end_channel[2], 1);
	write_channel_altera(next_frontier_end_channel[3], 1);
	write_channel_altera(next_frontier_end_channel[4], 1);
	write_channel_altera(next_frontier_end_channel[5], 1);
	write_channel_altera(next_frontier_end_channel[6], 1);
	write_channel_altera(next_frontier_end_channel[7], 1);
	write_channel_altera(next_frontier_end_channel[8], 1);
	write_channel_altera(next_frontier_end_channel[9], 1);
	write_channel_altera(next_frontier_end_channel[10], 1);
	write_channel_altera(next_frontier_end_channel[11], 1);
	write_channel_altera(next_frontier_end_channel[12], 1);
	write_channel_altera(next_frontier_end_channel[13], 1);
	write_channel_altera(next_frontier_end_channel[14], 1);
	write_channel_altera(next_frontier_end_channel[15], 1);
	
	*(next_frontier_size  + 0)  = mem_addr0;
	*(next_frontier_size  + 1)  = mem_addr1;
	*(next_frontier_size  + 2)  = mem_addr2;
	*(next_frontier_size  + 3)  = mem_addr3;
	*(next_frontier_size  + 4)  = mem_addr4;
	*(next_frontier_size  + 5)  = mem_addr5;
	*(next_frontier_size  + 6)  = mem_addr6;
	*(next_frontier_size  + 7)  = mem_addr7;
	*(next_frontier_size  + 8)  = mem_addr8;
	*(next_frontier_size  + 9)  = mem_addr9;
	*(next_frontier_size  + 10) = mem_addr10;
	*(next_frontier_size  + 11) = mem_addr11;
	*(next_frontier_size  + 12) = mem_addr12;
	*(next_frontier_size  + 13) = mem_addr13;
	*(next_frontier_size  + 14) = mem_addr14;
	*(next_frontier_size  + 15) = mem_addr15;

}
 
// write channel0
__kernel void __attribute__ ((task)) write_frontier0(
		__global int* restrict next_frontier0
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
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
	int global_idx = 0;
	int flag = 0;
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
	int global_idx = 0;
	int local_idx  = 0;
	int flag = 0;
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
	int global_idx = 0;
	int flag = 0;
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

// write channel 4
__kernel void __attribute__ ((task)) write_frontier4(
	__global int* restrict next_frontier4
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[4], &data_valid);
		if(data_valid){
			next_frontier4[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[4], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 5
__kernel void __attribute__ ((task)) write_frontier5(
	__global int* restrict next_frontier5
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[5], &data_valid);
		if(data_valid){
			next_frontier5[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[5], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 6
__kernel void __attribute__ ((task)) write_frontier6(
	__global int* restrict next_frontier6
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[6], &data_valid);
		if(data_valid){
			next_frontier6[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[6], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

//write channel 7
__kernel void __attribute__ ((task)) write_frontier7(
	__global int* restrict next_frontier7
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[7], &data_valid);
		if(data_valid){
			next_frontier7[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[7], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel8
__kernel void __attribute__ ((task)) write_frontier8(
		__global int* restrict next_frontier8
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[8], &data_valid);
		if(data_valid){
			next_frontier8[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[8], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 9
__kernel void __attribute__ ((task)) write_frontier9(
		__global int* restrict next_frontier9
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[9], &data_valid);
		if(data_valid){
			next_frontier9[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[9], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 2
__kernel void __attribute__ ((task)) write_frontier10(
	__global int* restrict next_frontier10
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[10], &data_valid);
		if(data_valid){
			next_frontier10[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[10], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 3
__kernel void __attribute__ ((task)) write_frontier11(
	__global int* restrict next_frontier11
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[11], &data_valid);
		if(data_valid){
			next_frontier11[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[11], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 4
__kernel void __attribute__ ((task)) write_frontier12(
	__global int* restrict next_frontier12
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[12], &data_valid);
		if(data_valid){
			next_frontier12[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[12], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 5
__kernel void __attribute__ ((task)) write_frontier13(
	__global int* restrict next_frontier13
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[13], &data_valid);
		if(data_valid){
			next_frontier13[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[13], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

// write channel 6
__kernel void __attribute__ ((task)) write_frontier14(
	__global int* restrict next_frontier14
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[14], &data_valid);
		if(data_valid){
			next_frontier14[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[14], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

//write channel 7
__kernel void __attribute__ ((task)) write_frontier15(
	__global int* restrict next_frontier15
		)
{

	bool data_valid = false;
	bool flag_valid = false;
	int global_idx = 0;
	int flag = 0;
	while(true){
		int vidx = read_channel_nb_altera(next_frontier_channel[15], &data_valid);
		if(data_valid){
			next_frontier15[global_idx++] = vidx;
		}

		int flag_tmp = read_channel_nb_altera(next_frontier_end_channel[15], &flag_valid);
		if(flag_valid) flag = flag_tmp;
		if(flag == 1 && !data_valid && !flag_valid){
			break;
		}
	}
}

